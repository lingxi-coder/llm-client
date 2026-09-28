//! Request encoding for the Messages API.

use crate::codecs::json::{WireRequest, WireValue};
use crate::codecs::{CodecContext, EncodeRequest};
use crate::protocol::{
    AnthropicClearAt, AnthropicMessageEffort, AnthropicToolCaller, AnthropicToolSearchConfig,
    AnthropicToolSearchStrategy, ContentBlock, ConversationMessage, DocumentSource, ImageSource,
    LlmError, MessageRole, ProviderProfile, ToolChoice, ToolSpec, VideoSource,
};

use serde_json::{json, Map, Value};

/// This provider requires `max_tokens`; a request that does not set one still
/// has to carry something.
const DEFAULT_MAX_TOKENS: u64 = 4096;

pub fn request<'a>(
    wire: EncodeRequest<'a>,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<WireRequest<'a>, LlmError> {
    let req = wire.request();
    crate::codecs::anthropic_conversation::validate(req, opts)?;
    crate::codecs::anthropic_client_toolsets::validate(req, opts)?;
    crate::codecs::anthropic_tool_search::validate(req, opts)?;
    crate::codecs::anthropic_mcp::validate(req, opts)?;
    crate::codecs::anthropic_web_fetch::validate(req, opts)?;
    crate::codecs::anthropic_code_execution::validate(req, opts)?;
    req.validate_hosted_tools()?;
    crate::codecs::reject_code_interpreter(req, profile.protocol)?;
    let has_openrouter_server_tools =
        crate::codecs::openrouter_server_tools::validate(req, profile, None, false)?;
    let has_anthropic_mcp = crate::codecs::anthropic_mcp::has_top_level_toolsets(req);
    let has_openrouter_tool_search = req.hosted_openrouter_tool_search().is_some();
    let tool_search = req.hosted_anthropic_tool_search();
    let web_fetch = req.hosted_anthropic_web_fetch();
    if req.hosted_openai_tool_search().is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI Responses tool search requires the OpenAI Responses codec".into(),
        });
    }
    if req.tools.iter().any(|tool| tool.defer_loading)
        && tool_search.is_none()
        && !has_openrouter_tool_search
        && !crate::codecs::anthropic_conversation::has_tool_additions(req)
    {
        return Err(LlmError::InvalidRequest {
            message: "defer_loading requires a supported hosted tool-search server tool on the same request".into(),
        });
    }
    if req.tools.iter().filter(|tool| tool.defer_loading).count() > 10_000 {
        return Err(LlmError::InvalidRequest {
            message:
                "Anthropic hosted tool search accepts at most 10,000 deferred tools per request"
                    .into(),
        });
    }
    if let Some(config) = tool_search {
        let server_tool_name = match config.strategy {
            AnthropicToolSearchStrategy::Regex => "tool_search_tool_regex",
            AnthropicToolSearchStrategy::Bm25 => "tool_search_tool_bm25",
        };
        if req.tools.iter().any(|tool| tool.name == server_tool_name) {
            return Err(LlmError::InvalidRequest {
                message: format!(
                    "client tool name {server_tool_name:?} conflicts with hosted tool search"
                ),
            });
        }
    }
    crate::files::validate_direct_provider_file_inputs_at(
        wire.blocks(),
        profile,
        &opts.request_model,
        opts.file_scope(),
        opts.file_validation_time(),
    )?;
    crate::codecs::reject_responses_continuation(
        req,
        crate::protocol::ProtocolFamily::AnthropicMessages,
    )?;
    if req.hosted_file_search().is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "hosted file search is supported only on Qwen Responses profiles".into(),
        });
    }
    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        Value::String(opts.request_model.clone()),
    );
    body.insert(
        "max_tokens".to_owned(),
        Value::from(req.max_tokens.map_or(DEFAULT_MAX_TOKENS, u64::from)),
    );

    // The system prompt is a top-level array here, not a message, and the
    // explicit cache breakpoints are applied after tools and messages are encoded.
    if !req.system.is_empty() {
        body.insert(
            "system".to_owned(),
            Value::Array(
                req.system
                    .iter()
                    .map(|_| json!({"type": "text", "text": Value::Null}))
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
            &opts.request_model,
            profile,
            opts,
            wire,
        )?);
    }
    body.insert("messages".to_owned(), Value::Null);

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
    if opts.stream {
        body.insert("stream".to_owned(), Value::Bool(true));
    }
    if !req.tools.is_empty()
        || tool_search.is_some()
        || web_fetch.is_some()
        || has_openrouter_server_tools
        || has_anthropic_mcp
        || req.hosted_anthropic_code_execution().is_some()
        || !req.anthropic_client_toolsets.is_empty()
    {
        body.insert(
            "tools".to_owned(),
            Value::Array(
                req.tools
                    .iter()
                    .map(encode_tool)
                    .chain(tool_search.map(encode_tool_search))
                    .chain(crate::codecs::openrouter_server_tools::response_tool_values(req))
                    .collect(),
            ),
        );
        body.insert(
            "tool_choice".to_owned(),
            encode_tool_choice(&req.tool_choice),
        );
    }
    crate::codecs::anthropic_mcp::apply(req, &mut body)?;
    crate::codecs::anthropic_web_fetch::apply(req, &mut body)?;
    crate::codecs::anthropic_client_toolsets::apply(req, &mut body);
    if crate::codecs::anthropic_conversation::has_tool_additions(req) {
        body.entry("tool_choice")
            .or_insert_with(|| encode_tool_choice(&req.tool_choice));
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
    let mut betas: Vec<String> = profile
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
    for tool in &req.tools {
        let beta = match tool.tool_type.as_deref() {
            Some("computer_use_20250124" | "computer_20250124") => Some("computer-use-2025-01-24"),
            Some("computer_20241022") => Some("computer-use-2024-10-22"),
            _ => None,
        };
        if let Some(beta) = beta {
            if !betas.iter().any(|b| b == beta) {
                betas.push(beta.into());
            }
        }
    }
    if !betas.is_empty() {
        headers.push(("anthropic-beta".to_owned(), betas.join(",")));
    }
    crate::codecs::anthropic_code_execution::apply(req, opts, &mut body)?;
    if let Some(config) = req.hosted_anthropic_code_execution() {
        if !config.files.is_empty() {
            // Validation established a final user message. Preserve its text,
            // attachments and borrowed payloads while adding these native inputs.
            let content = messages
                .last_mut()
                .and_then(|message| message.array_field_mut("content"))
                .expect("validated final Anthropic user message has content");
            content.extend(config.files.iter().map(|file| {
                WireValue::from(json!({"type":"container_upload", "file_id": file.file_id}))
            }));
        }
    }
    crate::codecs::web_search::apply(req, profile, &mut body)?;
    crate::codecs::inference::apply(req, opts, &mut body, &mut headers)?;
    crate::codecs::request_controls::apply(req, profile.protocol, &mut body)?;
    crate::codecs::structured::apply(req, opts, &mut body)?;
    crate::codecs::cache::apply(req, opts, &mut body, &mut messages)?;
    crate::wire_options::merge_body(profile, &mut body);
    crate::wire_options::merge_headers(profile, &mut headers);
    // Canonicalize after profile headers and codec-specific controls are
    // merged so the current MCP beta cannot be shadowed by a legacy value.
    crate::codecs::anthropic_mcp::apply_beta_header(req, profile, &mut headers);
    crate::codecs::anthropic_conversation::apply_beta_header(req, profile, &mut headers);

    let mut body =
        WireValue::from(Value::Object(body)).with("messages", WireValue::array(messages));
    if !req.tools.is_empty() {
        body = body.map_array_field("tools", |index, tool| {
            let has_schema = tool.get("input_schema").is_some();
            let tool = WireValue::from(tool);
            match req.tools.get(index) {
                Some(spec) if has_schema => {
                    tool.with("input_schema", WireValue::borrowed(&spec.input_schema))
                }
                _ => tool,
            }
        });
    }
    if !req.system.is_empty() {
        body = body.map_array_field("system", |index, block| {
            WireValue::from(block).with("text", WireValue::text(&req.system[index].text))
        });
    }
    let endpoint = if profile.provider_id.as_str() == "openrouter"
        && profile.base_url.trim_end_matches('/') == "https://openrouter.ai/api/v1"
    {
        "https://openrouter.ai/api/v1/messages".to_owned()
    } else {
        format!("{}/v1/messages", profile.base_url.trim_end_matches('/'))
    };
    let mut encoded = WireRequest::new(endpoint, headers, body);
    if opts.mode() == crate::RequestMode::CountTokens {
        encoded.http.url.push_str("/count_tokens");
        for key in [
            "max_tokens",
            "temperature",
            "top_p",
            "stream",
            "stop_sequences",
            "output_config",
            "context_hint",
        ] {
            encoded.body.remove(key);
        }
    }
    Ok(encoded)
}

fn encode_message<'a>(
    m: &'a ConversationMessage,
    unsigned_thinking: bool,
    model: &str,
    profile: &ProviderProfile,
    opts: &CodecContext,
    wire: EncodeRequest<'a>,
) -> Result<WireValue<'a>, LlmError> {
    let role = match m.role {
        MessageRole::Assistant => "assistant",
        // Model, route, content and grouped placement were validated before
        // encoding; preserve the operator role and its history position.
        MessageRole::System => "system",
        MessageRole::User => "user",
    };
    let mut blocks = Vec::new();
    for b in &m.content {
        let b = wire.block(b);
        blocks.push(match inline(wire, b, opts)? {
            Some(media) => media,
            None => encode_block(b, unsigned_thinking, model, profile, opts)?,
        });
    }
    let mut message =
        WireValue::from(json!({"role":role})).with("content", WireValue::array(blocks));
    if let Some(options) = &m.anthropic {
        if let Some(clear_at) = options.clear_at {
            let value = match clear_at {
                AnthropicClearAt::Never => "never",
                AnthropicClearAt::NextUserMessage => "next_user_message",
            };
            message = message.with("clear_at", WireValue::text(value));
        }
        if let Some(effort) = options.effort {
            let value = match effort {
                AnthropicMessageEffort::Low => "low",
                AnthropicMessageEffort::Medium => "medium",
                AnthropicMessageEffort::High => "high",
                AnthropicMessageEffort::XHigh => "xhigh",
                AnthropicMessageEffort::Max => "max",
            };
            message = message.with("output_config", WireValue::from(json!({"effort": value})));
        }
    }
    Ok(message)
}

fn encode_block<'a>(
    b: &'a ContentBlock,
    unsigned_thinking: bool,
    model: &str,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<WireValue<'a>, LlmError> {
    let value = match b {
        ContentBlock::ProviderContent { protocol, value } => {
            if *protocol != crate::protocol::ProtocolFamily::AnthropicMessages {
                return Err(LlmError::UnsupportedCapability {
                    message: "native content cannot be replayed on a different protocol".to_owned(),
                });
            }
            return Ok(WireValue::borrowed(value));
        }
        ContentBlock::Text { text, .. } => {
            return Ok(WireValue::from(json!({"type": "text"})).with("text", WireValue::text(text)));
        }
        ContentBlock::Thinking { text, signature } => {
            if signature.is_none() && unsigned_thinking {
                return Ok(WireValue::from(json!({"type": "thinking"}))
                    .with("thinking", WireValue::text(text)));
            }
            // Refusing beats sending: the provider rejects an unsigned replay,
            // and a codec that dropped the signature would surface the failure
            // on a later turn with nothing to point at (gate 18).
            let Some(signature) = signature else {
                return Err(LlmError::InvalidRequest {
                    message: "a thinking block needs its signature to round-trip".to_owned(),
                });
            };
            return Ok(WireValue::from(
                json!({"type": "thinking", "thinking": Value::Null, "signature": signature}),
            )
            .with("thinking", WireValue::text(text)));
        }
        ContentBlock::RedactedThinking { data } => {
            json!({"type": "redacted_thinking", "data": data})
        }
        ContentBlock::ToolUse {
            id,
            name,
            input,
            caller,
            toolset_name,
            ..
        } => {
            let mut block = WireValue::from(json!({"type": "tool_use", "id": id, "name": name}))
                .with("input", WireValue::borrowed(input));
            if let Some(caller) = caller {
                block = block.with("caller", WireValue::from(caller.clone()));
            }
            if let Some(toolset_name) = toolset_name {
                block = block.with("toolset_name", WireValue::from(json!(toolset_name)));
            }
            return Ok(block);
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            blocks,
            toolset_name,
        } => {
            let content = blocks.as_ref().map_or_else(
                || WireValue::text(content),
                |blocks| WireValue::array(blocks.iter().map(WireValue::borrowed).collect()),
            );
            let mut v = WireValue::from(json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
            }))
            .with("content", content);
            if *is_error {
                v["is_error"] = Value::Bool(true);
            }
            if let Some(toolset_name) = toolset_name {
                v["toolset_name"] = json!(toolset_name);
            }
            return Ok(v);
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
                if file.protocol != crate::protocol::ProtocolFamily::AnthropicMessages {
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
                    if file.protocol != crate::protocol::ProtocolFamily::AnthropicMessages {
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
        ContentBlock::Audio { .. } => {
            return Err(LlmError::UnsupportedCapability {
                message: "Anthropic Messages codec does not support Chat audio blocks".into(),
            });
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
                || file.protocol != crate::protocol::ProtocolFamily::AnthropicMessages
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
    };
    Ok(value.into())
}

fn encode_tool(t: &ToolSpec) -> Value {
    let mut tool = json!({
        "name": t.name,
        "description": t.description,
        "input_schema": Value::Null,
    });
    if t.strict {
        tool["strict"] = json!(true);
    }
    if t.defer_loading {
        tool["defer_loading"] = json!(true);
    }
    if !t.allowed_callers.is_empty() {
        tool["allowed_callers"] = Value::Array(
            t.allowed_callers
                .iter()
                .map(|caller| {
                    Value::String(
                        match caller {
                            AnthropicToolCaller::Direct => "direct",
                            AnthropicToolCaller::CodeExecution20260120 => "code_execution_20260120",
                            AnthropicToolCaller::CodeExecution20260521 => "code_execution_20260521",
                        }
                        .to_owned(),
                    )
                })
                .collect(),
        );
    }
    crate::codecs::request_controls::tool_extensions(t, tool)
}

fn encode_tool_search(config: &AnthropicToolSearchConfig) -> Value {
    let (kind, name) = match config.strategy {
        AnthropicToolSearchStrategy::Regex => {
            ("tool_search_tool_regex_20251119", "tool_search_tool_regex")
        }
        AnthropicToolSearchStrategy::Bm25 => {
            ("tool_search_tool_bm25_20251119", "tool_search_tool_bm25")
        }
    };
    json!({"type": kind, "name": name})
}

fn encode_tool_choice(c: &ToolChoice) -> Value {
    match c {
        ToolChoice::Auto => json!({"type": "auto"}),
        ToolChoice::Any => json!({"type": "any"}),
        ToolChoice::None => json!({"type": "none"}),
        ToolChoice::Tool { name } => json!({"type": "tool", "name": name}),
    }
}

fn inline<'a>(
    wire: EncodeRequest<'a>,
    block: &ContentBlock,
    _context: &CodecContext,
) -> Result<Option<WireValue<'a>>, LlmError> {
    let Some(media) = wire.inline_media(block)? else {
        return Ok(None);
    };
    let attachment = media.attachment;
    let (kind, title) = match block {
        ContentBlock::Image { .. } => ("image", None),
        ContentBlock::Document { title, .. } => ("document", title.as_deref()),
        ContentBlock::Video { .. } => ("video", None),
        _ => unreachable!("inline_media accepts only attachment blocks"),
    };
    let data = || WireValue::base64(media.bytes, String::new());
    let _uri = || {
        WireValue::base64(
            media.bytes,
            format!("data:{};base64,", attachment.media_type),
        )
    };
    Ok(Some({
        if kind == "video" {
            return Err(LlmError::UnsupportedCapability {
                message: "video input requires a supported provider file".into(),
            });
        }
        if kind == "document" && attachment.media_type.starts_with("image/") {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic document blocks cannot use an image media type".into(),
            });
        }
        let mut value = WireValue::from(json!({"type":kind})).with(
            "source",
            WireValue::from(json!({"type":"base64","media_type":attachment.media_type}))
                .with("data", data()),
        );
        if kind == "document" {
            value["title"] = json!(title.unwrap_or(&attachment.filename));
        }
        value
    }))
}
