//! Request encoding for the Messages API.

use crate::client::route::ResolvedRoute;
use crate::transport::HttpRequest;
use crate::RequestOptions;
use lingxi_agent_api::protocol::{
    CompletionRequest, ContentBlock, ConversationMessage, DocumentSource, ImageSource, LlmError,
    MessageRole, ProviderProfile, ToolChoice, ToolSpec, VideoSource,
};
use serde_json::{json, Map, Value};

/// This provider requires `max_tokens`; a request that does not set one still
/// has to carry something.
const DEFAULT_MAX_TOKENS: u64 = 4096;

pub fn request(
    req: &CompletionRequest,
    profile: &ProviderProfile,
    route: &ResolvedRoute,
    opts: &RequestOptions,
) -> Result<HttpRequest, LlmError> {
    crate::codecs::reject_responses_continuation(
        req,
        lingxi_agent_api::protocol::ProtocolFamily::AnthropicMessages,
    )?;
    if req.file_search.is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "hosted file search is supported only on Qwen Responses profiles".into(),
        });
    }
    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        Value::String(route.request_model.clone()),
    );
    body.insert(
        "max_tokens".to_owned(),
        Value::from(req.max_tokens.map_or(DEFAULT_MAX_TOKENS, u64::from)),
    );

    // The system prompt is a top-level array here, not a message, and the
    // cacheable/non-cacheable split survives as `cache_control` on the blocks
    // that carry it — which is the whole point of partitioning it (§13 step 2).
    if !req.system.is_empty() {
        body.insert(
            "system".to_owned(),
            Value::Array(
                req.system
                    .iter()
                    .map(|b| {
                        let mut block = json!({"type": "text", "text": b.text});
                        if b.cacheable {
                            block["cache_control"] = json!({"type": "ephemeral"});
                        }
                        block
                    })
                    .collect(),
            ),
        );
    }

    let mut messages = Vec::new();
    let unsigned_thinking = profile
        .extra
        .get("supports_unsigned_thinking")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    for m in &req.messages {
        messages.push(encode_message(
            m,
            unsigned_thinking,
            &route.request_model,
            profile,
            opts,
        )?);
    }
    body.insert("messages".to_owned(), Value::Array(messages));

    if let Some(t) = req.temperature {
        body.insert("temperature".to_owned(), Value::from(t));
    }
    if !req.stop_sequences.is_empty() {
        body.insert(
            "stop_sequences".to_owned(),
            Value::Array(
                req.stop_sequences
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
    }
    if let Some(thinking) = &req.thinking {
        if let Some(budget) = thinking.budget_tokens {
            body.insert(
                "thinking".to_owned(),
                json!({"type": "enabled", "budget_tokens": budget}),
            );
        }
    }
    if opts.stream {
        body.insert("stream".to_owned(), Value::Bool(true));
    }
    if !req.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(req.tools.iter().map(encode_tool).collect()),
        );
        body.insert(
            "tool_choice".to_owned(),
            encode_tool_choice(&req.tool_choice),
        );
    }

    let mut headers = vec![
        ("content-type".to_owned(), "application/json".to_owned()),
        (
            "anthropic-version".to_owned(),
            profile
                .extra
                .get("api_version")
                .and_then(Value::as_str)
                .unwrap_or(super::DEFAULT_API_VERSION)
                .to_owned(),
        ),
    ];
    // Beta headers accumulate, comma-joined. Overwriting would silently drop
    // one when a request needs two.
    let betas: Vec<String> = profile
        .extra
        .get("betas")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if !betas.is_empty() {
        headers.push(("anthropic-beta".to_owned(), betas.join(",")));
    }

    crate::codecs::web_search::apply(req, profile, &mut body)?;
    crate::codecs::extras::merge_body(profile, &mut body);
    crate::codecs::extras::merge_headers(profile, &mut headers);

    Ok(HttpRequest {
        method: "POST".to_owned(),
        url: format!("{}/v1/messages", profile.base_url.trim_end_matches('/')),
        headers,
        body: serde_json::to_vec(&Value::Object(body))
            .map_err(|e| LlmError::InvalidRequest {
                message: format!("request body is not serializable: {e}"),
            })?
            .into(),
        timeout: None,
    })
}

fn encode_message(
    m: &ConversationMessage,
    unsigned_thinking: bool,
    model: &str,
    profile: &ProviderProfile,
    opts: &RequestOptions,
) -> Result<Value, LlmError> {
    let role = match m.role {
        MessageRole::Assistant => "assistant",
        // This wire has no system role in `messages`; a system block that got
        // this far is the caller's mistake, and sending it as a user turn would
        // change what the model was told.
        MessageRole::System => {
            return Err(LlmError::InvalidRequest {
                message: "the system prompt belongs in `system`, not in `messages`".to_owned(),
            });
        }
        MessageRole::User => "user",
    };
    let mut blocks = Vec::new();
    for b in &m.content {
        blocks.push(encode_block(b, unsigned_thinking, model, profile, opts)?);
    }
    Ok(json!({"role": role, "content": blocks}))
}

fn encode_block(
    b: &ContentBlock,
    unsigned_thinking: bool,
    model: &str,
    profile: &ProviderProfile,
    opts: &RequestOptions,
) -> Result<Value, LlmError> {
    Ok(match b {
        ContentBlock::ProviderContent { protocol, value } => {
            if *protocol != lingxi_agent_api::protocol::ProtocolFamily::AnthropicMessages {
                return Err(LlmError::UnsupportedCapability {
                    message: "native content cannot be replayed on a different protocol".to_owned(),
                });
            }
            value.clone()
        }
        ContentBlock::Text { text, .. } => json!({"type": "text", "text": text}),
        ContentBlock::Thinking { text, signature } => {
            if signature.is_none() && unsigned_thinking {
                return Ok(json!({"type": "thinking", "thinking": text}));
            }
            // Refusing beats sending: the provider rejects an unsigned replay,
            // and a codec that dropped the signature would surface the failure
            // on a later turn with nothing to point at (gate 18).
            let Some(signature) = signature else {
                return Err(LlmError::InvalidRequest {
                    message: "a thinking block needs its signature to round-trip".to_owned(),
                });
            };
            json!({"type": "thinking", "thinking": text, "signature": signature})
        }
        ContentBlock::RedactedThinking { data } => {
            json!({"type": "redacted_thinking", "data": data})
        }
        ContentBlock::ToolUse {
            id, name, input, ..
        } => {
            json!({"type": "tool_use", "id": id, "name": name, "input": input})
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            blocks,
        } => {
            let mut v = json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": blocks.clone().map_or_else(
                    || Value::String(content.clone()),
                    Value::Array,
                ),
            });
            if *is_error {
                v["is_error"] = Value::Bool(true);
            }
            v
        }
        ContentBlock::Image { source } => match source {
            ImageSource::Base64 { media_type, data } => json!({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }),
            ImageSource::Url { url } => json!({
                "type": "image",
                "source": {"type": "url", "url": url},
            }),
            ImageSource::Attachment { .. } => {
                return Err(crate::codecs::unresolved_attachment_error());
            }
            ImageSource::ProviderFile { file } => {
                let file = crate::codecs::validate_provider_file(file, profile, opts)?;
                if file.protocol != lingxi_agent_api::protocol::ProtocolFamily::AnthropicMessages {
                    return Err(crate::codecs::provider_file_protocol_error());
                }
                json!({
                    "type": "image",
                    "source": {"type": "file", "file_id": file.file_id},
                })
            }
        },
        ContentBlock::Document { source, title } => {
            let media_type = match source {
                DocumentSource::Base64 { media_type, .. }
                | DocumentSource::Text { media_type, .. } => Some(media_type.as_str()),
                DocumentSource::Attachment { attachment } => Some(attachment.media_type.as_str()),
                DocumentSource::ProviderFile { file } => file.media_type.as_deref(),
                DocumentSource::Url { .. } => None,
            };
            if media_type.is_some_and(|media_type| {
                media_type.trim().to_ascii_lowercase().starts_with("image/")
            }) {
                return Err(LlmError::InvalidRequest {
                    message: "Anthropic document blocks cannot use an image media type".into(),
                });
            }
            let mut v = match source {
                DocumentSource::Base64 { media_type, data } => json!({
                    "type": "document",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                }),
                DocumentSource::Text { media_type, data } => json!({
                    "type": "document",
                    "source": {"type": "text", "media_type": media_type, "data": data},
                }),
                DocumentSource::Url { url } => json!({
                    "type": "document",
                    "source": {"type": "url", "url": url},
                }),
                DocumentSource::Attachment { .. } => {
                    return Err(crate::codecs::unresolved_attachment_error());
                }
                DocumentSource::ProviderFile { file } => {
                    let file = crate::codecs::validate_provider_file(file, profile, opts)?;
                    if file.protocol
                        != lingxi_agent_api::protocol::ProtocolFamily::AnthropicMessages
                    {
                        return Err(crate::codecs::provider_file_protocol_error());
                    }
                    json!({
                        "type": "document",
                        "source": {"type": "file", "file_id": file.file_id},
                    })
                }
            };
            if let Some(title) = title {
                v["title"] = Value::String(title.clone());
            }
            v
        }
        ContentBlock::Video { source } => {
            let VideoSource::ProviderFile { file } = source else {
                return Err(LlmError::UnsupportedCapability {
                    message: "MiniMax video input requires a provider file uploaded for video_understanding".into(),
                });
            };
            let file = crate::codecs::validate_provider_file(file, profile, opts)?;
            if profile.provider_id.as_str() != "minimax"
                || !model.eq_ignore_ascii_case("minimax-m3")
                || file.protocol != lingxi_agent_api::protocol::ProtocolFamily::AnthropicMessages
                || file.purpose.as_deref() != Some("video_understanding")
            {
                return Err(LlmError::UnsupportedCapability {
                    message: "video file references are supported only by MiniMax M3".into(),
                });
            }
            json!({
                "type": "video",
                "source": {"type": "url", "url": format!("mm_file://{}", file.file_id)},
            })
        }
    })
}

fn encode_tool(t: &ToolSpec) -> Value {
    json!({
        "name": t.name,
        "description": t.description,
        "input_schema": t.input_schema,
    })
}

fn encode_tool_choice(c: &ToolChoice) -> Value {
    match c {
        ToolChoice::Auto => json!({"type": "auto"}),
        ToolChoice::Any => json!({"type": "any"}),
        ToolChoice::None => json!({"type": "none"}),
        ToolChoice::Tool { name } => json!({"type": "tool", "name": name}),
    }
}
