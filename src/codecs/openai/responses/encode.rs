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
    if !req.stop_sequences.is_empty() {
        // This wire has no stop parameter. Dropping them would run the model
        // without a limit the caller asked for and never say so.
        return Err(LlmError::InvalidRequest {
            message: "the Responses API has no stop-sequence parameter".to_owned(),
        });
    }

    let mut input = Vec::new();
    for m in &req.messages {
        encode_message(m, &mut input)?;
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
    if let Some(max) = req.max_tokens {
        body.insert("max_output_tokens".to_owned(), Value::from(max));
    }
    if let Some(t) = req.temperature {
        body.insert("temperature".to_owned(), Value::from(t));
    }
    if opts.stream {
        body.insert("stream".to_owned(), Value::Bool(true));
    }

    crate::codecs::web_search::apply(req, profile, &mut body)?;
    crate::codecs::extras::merge_body(profile, &mut body);
    let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
    crate::codecs::extras::merge_headers(profile, &mut headers);

    Ok(HttpRequest {
        method: "POST".to_owned(),
        url: format!("{}/responses", profile.base_url.trim_end_matches('/')),
        headers,
        body: serde_json::to_vec(&Value::Object(body))
            .map_err(|e| LlmError::InvalidRequest {
                message: format!("request body is not serializable: {e}"),
            })?
            .into(),
        timeout: None,
    })
}

/// A message becomes one item; a tool call or result becomes its own sibling
/// item, so any text collected so far is flushed first to keep the order.
fn encode_message(m: &ConversationMessage, input: &mut Vec<Value>) -> Result<(), LlmError> {
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
            ContentBlock::ProviderContent { .. } => return Err(LlmError::UnsupportedCapability {
                message: "native content cannot be replayed on Responses".to_owned(),
            }),
            ContentBlock::Text { text } => parts.push(json!({"type": text_part, "text": text})),
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
            ContentBlock::Image { source } => parts.push(json!({
                "type": "input_image",
                "image_url": match source {
                    ImageSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
                    ImageSource::Url { url } => url.clone(),
                },
            })),
            ContentBlock::Document { source, title } => {
                let mut part = json!({"type": "input_file"});
                match source {
                    DocumentSource::Url { url } => part["file_url"] = json!(url),
                    DocumentSource::Base64 { media_type, data } => {
                        part["file_data"] = json!(format!("data:{media_type};base64,{data}"));
                    }
                    DocumentSource::Text { media_type, data } => {
                        let encoded = base64::engine::general_purpose::STANDARD.encode(data.as_bytes());
                        part["file_data"] = json!(format!("data:{media_type};base64,{encoded}"));
                    }
                }
                if part.get("file_data").is_some() {
                    part["filename"] = json!(title.as_deref().unwrap_or("document"));
                }
                parts.push(part);
            }
            ContentBlock::ToolUse { id, name, input: args } => {
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
