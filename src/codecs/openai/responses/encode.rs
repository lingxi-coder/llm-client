//! Request encoding for the Responses API.

use crate::client::route::ResolvedRoute;
use crate::transport::HttpRequest;
use crate::RequestOptions;
use base64::Engine;
use lingxi_agent_api::protocol::{
    CompletionRequest, ContentBlock, ConversationMessage, DocumentSource, ImageSource, LlmError,
    MessageRole, ProviderProfile, ToolChoice, ToolSpec,
};
use serde_json::{json, Map, Value};

pub fn request(
    req: &CompletionRequest,
    profile: &ProviderProfile,
    route: &ResolvedRoute,
    opts: &RequestOptions,
) -> Result<HttpRequest, LlmError> {
    if req.previous_response_id.is_some()
        && !profile
            .extra
            .get("supports_previous_response_id")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        return Err(LlmError::UnsupportedCapability {
            message: format!(
                "profile {:?} does not declare previous_response_id support",
                profile.profile_name
            ),
        });
    }
    if req.file_search.is_some()
        && profile.extra.get("file_search").and_then(Value::as_str) != Some("qwen")
    {
        return Err(LlmError::UnsupportedCapability {
            message: format!(
                "profile {:?} does not declare Qwen file search",
                profile.profile_name
            ),
        });
    }
    if !req.stop_sequences.is_empty() {
        // This wire has no stop parameter. Dropping them would run the model
        // without a limit the caller asked for and never say so.
        return Err(LlmError::InvalidRequest {
            message: "the Responses API has no stop-sequence parameter".to_owned(),
        });
    }

    let mut input = Vec::new();
    for m in &req.messages {
        encode_message(m, &mut input, profile, opts)?;
    }

    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        Value::String(route.request_model.clone()),
    );
    if !req.system.is_empty() {
        body.insert(
            "instructions".to_owned(),
            Value::String(
                req.system
                    .iter()
                    .map(|b| b.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            ),
        );
    }
    body.insert("input".to_owned(), Value::Array(input));
    if let Some(id) = &req.previous_response_id {
        body.insert(
            "previous_response_id".to_owned(),
            Value::String(id.as_str().to_owned()),
        );
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
    // Add hosted web search before the Qwen knowledge-base tool so tool order
    // stays stable alongside the caller's function tools.
    crate::codecs::web_search::apply(req, profile, &mut body)?;
    if let Some(search) = &req.file_search {
        if search.knowledge_base_id.trim().is_empty() {
            return Err(LlmError::InvalidRequest {
                message: "Qwen file search requires a nonempty knowledge_base_id".into(),
            });
        }
        if !search
            .workspace_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            || search.workspace_id.is_empty()
        {
            return Err(LlmError::InvalidRequest {
                message:
                    "Qwen file search workspace_id must contain only letters, digits, or hyphens"
                        .into(),
            });
        }
        if profile.provider_id.as_str() != "qwen"
            || !matches!(
                route.request_model.as_str(),
                "qwen3.8-max" | "qwen3.8-flash"
            )
        {
            return Err(LlmError::UnsupportedCapability {
                message: "Qwen file search is available only for the supported Max and Flash Responses models".into(),
            });
        }
        body.entry("tools")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Qwen file search requires tools to be an array".into(),
            })?
            .push(json!({"type":"file_search","vector_store_ids":[search.knowledge_base_id]}));
    }
    if let Some(max) = req.max_tokens {
        body.insert("max_output_tokens".to_owned(), Value::from(max));
    }
    if let Some(t) = req.temperature {
        body.insert("temperature".to_owned(), Value::from(t));
    }
    if opts.stream {
        body.insert("stream".to_owned(), Value::Bool(true));
    }

    crate::codecs::extras::merge_body(profile, &mut body);
    let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
    crate::codecs::extras::merge_headers(profile, &mut headers);

    let url = if let Some(search) = &req.file_search {
        qwen_file_search_url(profile, &search.workspace_id)?
    } else {
        format!("{}/responses", profile.base_url.trim_end_matches('/'))
    };
    Ok(HttpRequest {
        method: "POST".to_owned(),
        url,
        headers,
        body: serde_json::to_vec(&Value::Object(body))
            .map_err(|e| LlmError::InvalidRequest {
                message: format!("request body is not serializable: {e}"),
            })?
            .into(),
        timeout: None,
    })
}

fn qwen_file_search_url(profile: &ProviderProfile, workspace_id: &str) -> Result<String, LlmError> {
    let host = reqwest::Url::parse(&profile.base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned));
    let region_domain = match (profile.provider_id.as_str(), host.as_deref()) {
        ("qwen", Some("dashscope.aliyuncs.com")) => "cn-beijing.maas.aliyuncs.com",
        ("qwen", Some("dashscope-intl.aliyuncs.com")) => "ap-southeast-1.maas.aliyuncs.com",
        ("qwen", Some("dashscope-us.aliyuncs.com")) => "us-east-1.maas.aliyuncs.com",
        ("qwen", Some("cn-hongkong.dashscope.aliyuncs.com")) => "cn-hongkong.maas.aliyuncs.com",
        _ => {
            return Err(LlmError::UnsupportedCapability {
                message: "Qwen file search requires a supported regional Model Studio profile"
                    .into(),
            });
        }
    };
    Ok(format!(
        "https://{workspace_id}.{region_domain}/compatible-mode/v1/responses"
    ))
}

/// A message becomes one item; a tool call or result becomes its own sibling
/// item, so any text collected so far is flushed first to keep the order.
fn encode_message(
    m: &ConversationMessage,
    input: &mut Vec<Value>,
    profile: &ProviderProfile,
    opts: &RequestOptions,
) -> Result<(), LlmError> {
    let role = match m.role {
        MessageRole::Assistant => "assistant",
        MessageRole::User => "user",
        MessageRole::System => "system",
    };
    // The assistant's text is an output part; everyone else's is input.
    let text_part = if m.role == MessageRole::Assistant {
        "output_text"
    } else {
        "input_text"
    };
    let mut parts: Vec<Value> = Vec::new();

    for b in &m.content {
        match b {
            ContentBlock::ProviderContent { .. } => {
                return Err(LlmError::UnsupportedCapability {
                    message: "native content cannot be replayed on Responses".to_owned(),
                })
            }
            ContentBlock::Text { text, .. } => parts.push(json!({"type": text_part, "text": text})),
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
            ContentBlock::Image { source } => {
                let mut part = json!({"type": "input_image"});
                match source {
                    ImageSource::Base64 { media_type, data } => {
                        part["image_url"] = json!(format!("data:{media_type};base64,{data}"));
                    }
                    ImageSource::Url { url } => part["image_url"] = json!(url),
                    ImageSource::Attachment { .. } => {
                        return Err(crate::codecs::unresolved_attachment_error())
                    }
                    ImageSource::ProviderFile { file } => {
                        let file = crate::codecs::validate_provider_file(file, profile, opts)?;
                        if file.protocol
                            != lingxi_agent_api::protocol::ProtocolFamily::OpenAiResponses
                        {
                            return Err(crate::codecs::provider_file_protocol_error());
                        }
                        part["file_id"] = json!(file.file_id);
                    }
                }
                parts.push(part);
            }
            ContentBlock::Document { source, title } => {
                let mut part = json!({"type": "input_file"});
                match source {
                    DocumentSource::Url { url } => part["file_url"] = json!(url),
                    DocumentSource::Base64 { media_type, data } => {
                        part["file_data"] = json!(format!("data:{media_type};base64,{data}"));
                    }
                    DocumentSource::Text { media_type, data } => {
                        let encoded =
                            base64::engine::general_purpose::STANDARD.encode(data.as_bytes());
                        part["file_data"] = json!(format!("data:{media_type};base64,{encoded}"));
                    }
                    DocumentSource::Attachment { .. } => {
                        return Err(crate::codecs::unresolved_attachment_error())
                    }
                    DocumentSource::ProviderFile { file } => {
                        let file = crate::codecs::validate_provider_file(file, profile, opts)?;
                        if file.protocol
                            != lingxi_agent_api::protocol::ProtocolFamily::OpenAiResponses
                        {
                            return Err(crate::codecs::provider_file_protocol_error());
                        }
                        part["file_id"] = json!(file.file_id);
                    }
                }
                if part.get("file_data").is_some() {
                    part["filename"] = json!(title.as_deref().unwrap_or("document"));
                }
                parts.push(part);
            }
            ContentBlock::Video { .. } => {
                return Err(LlmError::UnsupportedCapability {
                    message: "Responses input does not support video content blocks".into(),
                });
            }
            ContentBlock::ToolUse {
                id,
                name,
                input: args,
                ..
            } => {
                flush(role, &mut parts, input);
                input.push(json!({
                    "type": "function_call",
                    "call_id": id,
                    "name": name,
                    // Arguments go on the wire as a JSON string, as on the
                    // Chat wire.
                    "arguments": args.to_string(),
                }));
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                flush(role, &mut parts, input);
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": tool_use_id,
                    "output": content,
                }));
            }
        }
    }
    flush(role, &mut parts, input);
    Ok(())
}

fn flush(role: &str, parts: &mut Vec<Value>, input: &mut Vec<Value>) {
    if parts.is_empty() {
        return;
    }
    input.push(json!({
        "type": "message",
        "role": role,
        "content": std::mem::take(parts),
    }));
}

fn encode_tool(t: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": t.name,
        "description": t.description,
        "parameters": t.input_schema,
        "strict": t.strict,
    })
}

fn encode_tool_choice(c: &ToolChoice) -> Value {
    match c {
        ToolChoice::Auto => Value::String("auto".to_owned()),
        ToolChoice::Any => Value::String("required".to_owned()),
        ToolChoice::None => Value::String("none".to_owned()),
        ToolChoice::Tool { name } => json!({"type": "function", "name": name}),
    }
}
