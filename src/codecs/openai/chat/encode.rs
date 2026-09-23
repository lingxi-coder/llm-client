//! Request encoding.
//!
//! Provider quirks are keyed off the profile, never off a provider name in
//! code: `extra` on a `ProviderProfile` is where a settings entry says "this
//! endpoint needs the usage object requested explicitly" or "this endpoint
//! rejects tool_choice in thinking mode". That is what keeps a new
//! OpenAI-compatible provider a settings change (gate 30).

use crate::client::route::ResolvedRoute;
use crate::transport::HttpRequest;
use crate::RequestOptions;
use base64::Engine;
use lingxi_agent_api::protocol::{
    CompletionRequest, ContentBlock, ConversationMessage, DocumentSource, ImageSource, LlmError,
    MessageRole, ProviderProfile, ToolChoice,
};
use serde_json::{json, Map, Value};

/// A boolean knob a profile may set in `extra`.
fn flag(profile: &ProviderProfile, key: &str) -> bool {
    profile
        .extra
        .get(key)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub fn request(
    req: &CompletionRequest,
    profile: &ProviderProfile,
    route: &ResolvedRoute,
    opts: &RequestOptions,
) -> Result<HttpRequest, LlmError> {
    crate::codecs::reject_responses_continuation(
        req,
        lingxi_agent_api::protocol::ProtocolFamily::OpenAiChat,
    )?;
    let mut messages = Vec::new();

    // Several system blocks become one system message: the wire has one slot,
    // and the cacheable/non-cacheable split is a prefix-caching concern that
    // this protocol cannot express.
    if !req.system.is_empty() {
        let text = req
            .system
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        messages.push(json!({"role": "system", "content": text}));
    }

    let keep_reasoning = flag(profile, "preserve_reasoning_content");
    for m in &req.messages {
        messages.extend(encode_message(m, keep_reasoning)?);
    }

    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        Value::String(route.request_model.clone()),
    );
    body.insert("messages".to_owned(), Value::Array(messages));

    if opts.stream {
        body.insert("stream".to_owned(), Value::Bool(true));
        if flag(profile, "stream_usage_opt_in") {
            // Some endpoints omit the terminal usage object from SSE unless it
            // is explicitly requested. Without this the transcript, the cost
            // tracker and every token counter downstream all see zero.
            body.insert("stream_options".to_owned(), json!({"include_usage": true}));
        }
    }
    if let Some(max) = req.max_tokens {
        body.insert("max_tokens".to_owned(), Value::from(max));
    }
    if let Some(t) = req.temperature {
        body.insert("temperature".to_owned(), Value::from(t));
    }
    if !req.stop_sequences.is_empty() {
        body.insert(
            "stop".to_owned(),
            Value::Array(
                req.stop_sequences
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
    }
    if !req.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(req.tools.iter().map(encode_tool).collect()),
        );
        // Some endpoints refuse a *forced* tool choice while thinking is on —
        // `required` and a named function each come back 400 — while `none`,
        // `auto` and the tools themselves are all accepted. So the tools stay,
        // `none` and `auto` pass through, and only a forced choice relaxes to
        // `auto`: the model then follows its prompt, and the caller's own
        // validation and retry path still apply.
        //
        // Dropping the key outright would be wrong. `none` means "do not call a
        // tool", and losing it lets the model call one.
        let choice = if flag(profile, "thinking_rejects_forced_tool_choice") {
            relax_forced_choice(&req.tool_choice)
        } else {
            req.tool_choice.clone()
        };
        body.insert("tool_choice".to_owned(), encode_tool_choice(&choice));
    }

    // Whatever else this particular endpoint understands. Additive only, and
    // never a credential — see `wire_extras`.
    crate::codecs::web_search::apply(req, profile, &mut body)?;
    crate::codecs::extras::merge_body(profile, &mut body);
    let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
    crate::codecs::extras::merge_headers(profile, &mut headers);

    let url = format!(
        "{}/chat/completions",
        profile.base_url.trim_end_matches('/')
    );
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

/// One conversation message becomes one or more wire messages: a tool result
/// cannot share a message with text, so any pending text is flushed first and
/// the result becomes its own `role: "tool"` entry.
fn encode_message(m: &ConversationMessage, keep_reasoning: bool) -> Result<Vec<Value>, LlmError> {
    let role = match m.role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::System => "system",
    };
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut media: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut out: Vec<Value> = Vec::new();

    for block in &m.content {
        match block {
            ContentBlock::ProviderContent { .. } => {
                return Err(LlmError::UnsupportedCapability {
                    message: "native content cannot be replayed on Chat Completions".to_owned(),
                })
            }
            ContentBlock::Text { text: t, .. } => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(t);
            }
            ContentBlock::Thinking { text: t, .. } => {
                if keep_reasoning {
                    reasoning.push_str(t);
                }
            }
            ContentBlock::Image { source } => {
                media.push(json!({
                    "type": "image_url",
                    "image_url": { "url": image_url(source) },
                }));
            }
            ContentBlock::Document { source, .. } => {
                media.push(json!({
                    "type": "file",
                    "file": { "file_data": document_url(source) },
                }));
            }
            // Signed reasoning round-trips only on providers that sign it
            // (gate 18); this wire has no slot, so a replayed block is dropped
            // rather than sent somewhere it would be rejected.
            ContentBlock::RedactedThinking { .. } => {}
            ContentBlock::ToolUse {
                id, name, input, ..
            } => tool_calls.push(json!({
                "id": id,
                "type": "function",
                "function": { "name": name, "arguments": input.to_string() },
            })),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                if !tool_calls.is_empty() {
                    out.push(assistant_tool_calls(&text, &reasoning, &tool_calls));
                    text.clear();
                    tool_calls.clear();
                }
                if !text.is_empty() || !media.is_empty() {
                    out.push(user_message(role, &text, &media));
                    text.clear();
                    media.clear();
                }
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content,
                }));
            }
        }
    }

    if !tool_calls.is_empty() {
        out.push(assistant_tool_calls(&text, &reasoning, &tool_calls));
    } else if !text.is_empty() || !media.is_empty() {
        if role == "assistant" {
            let mut msg = json!({"role": "assistant", "content": text});
            // An endpoint whose thinking mode is on by default rejects a later
            // turn that omits that turn's reasoning: "the reasoning_content in
            // the thinking mode must be passed back". It is not limited to
            // tool-call messages.
            if keep_reasoning && !reasoning.is_empty() {
                msg["reasoning_content"] = Value::String(reasoning);
            }
            out.push(msg);
        } else {
            out.push(user_message(role, &text, &media));
        }
    }
    Ok(out)
}

/// A hosted URL is sent as-is; base64 becomes a data URI, which is the only
/// inline form this wire accepts.
fn image_url(source: &ImageSource) -> String {
    match source {
        ImageSource::Url { url } => url.clone(),
        ImageSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
    }
}

fn document_url(source: &DocumentSource) -> String {
    match source {
        DocumentSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
        // Plain text is still sent as a data URI: the field is a file payload,
        // not a content part.
        DocumentSource::Text { media_type, data } => {
            format!("data:{media_type};base64,{}", base64_of(data.as_bytes()))
        }
        DocumentSource::Url { url } => url.clone(),
    }
}

fn user_message(role: &str, text: &str, media: &[Value]) -> Value {
    if media.is_empty() {
        return json!({"role": role, "content": text});
    }
    let mut parts = vec![json!({"type": "text", "text": text})];
    parts.extend(media.iter().cloned());
    json!({"role": role, "content": parts})
}

fn assistant_tool_calls(text: &str, reasoning: &str, calls: &[Value]) -> Value {
    let mut msg = json!({
        "role": "assistant",
        "content": if text.is_empty() { Value::Null } else { Value::String(text.to_owned()) },
        "tool_calls": calls,
    });
    if !reasoning.is_empty() {
        msg["reasoning_content"] = Value::String(reasoning.to_owned());
    }
    msg
}

fn encode_tool(t: &lingxi_agent_api::protocol::ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        }
    })
}

/// `required` and a named tool become `auto`; `none` and `auto` are untouched.
fn relax_forced_choice(c: &ToolChoice) -> ToolChoice {
    match c {
        ToolChoice::Any | ToolChoice::Tool { .. } => ToolChoice::Auto,
        other => other.clone(),
    }
}

fn encode_tool_choice(c: &ToolChoice) -> Value {
    match c {
        ToolChoice::Auto => Value::String("auto".to_owned()),
        ToolChoice::None => Value::String("none".to_owned()),
        ToolChoice::Any => Value::String("required".to_owned()),
        ToolChoice::Tool { name } => json!({"type": "function", "function": {"name": name}}),
    }
}

/// Base64 for an image a caller handed over as bytes.
pub fn base64_of(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}
