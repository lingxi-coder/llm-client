//! Request encoding for `generateContent`.

use crate::client::route::ResolvedRoute;
use crate::transport::HttpRequest;
use crate::RequestOptions;
use base64::Engine;
use lingxi_agent_api::protocol::{
    CompletionRequest, ContentBlock, ConversationMessage, DocumentSource, ImageSource, LlmError,
    MessageRole, ProviderProfile, ToolChoice, ToolSpec, ToolUseId,
};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

pub fn request(
    req: &CompletionRequest,
    profile: &ProviderProfile,
    route: &ResolvedRoute,
    opts: &RequestOptions,
) -> Result<HttpRequest, LlmError> {
    request_to(
        req,
        profile,
        &super::generate_content_url(&profile.base_url, &route.request_model, opts.stream),
    )
}

/// The body is identical wherever this wire is hosted, and it carries no
/// stream flag — the URL says whether to stream. So the Vertex wrapper reuses
/// this with a URL of its own and nothing else changes.
pub(crate) fn request_to(
    req: &CompletionRequest,
    profile: &ProviderProfile,
    url: &str,
) -> Result<HttpRequest, LlmError> {
    crate::codecs::reject_responses_continuation(
        req,
        lingxi_agent_api::protocol::ProtocolFamily::GeminiGenerateContent,
    )?;
    let names = tool_call_names(&req.messages);
    let mut body = Map::new();
    body.insert(
        "contents".to_owned(),
        Value::Array(contents(&req.messages, &names)?),
    );

    if !req.system.is_empty() {
        body.insert(
            "systemInstruction".to_owned(),
            json!({"parts": req.system.iter().map(|b| json!({"text": b.text})).collect::<Vec<_>>()}),
        );
    }
    if !req.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            json!([{ "functionDeclarations": req.tools.iter().map(encode_tool).collect::<Vec<_>>() }]),
        );
        body.insert(
            "toolConfig".to_owned(),
            encode_tool_choice(&req.tool_choice),
        );
    }

    let mut generation = Map::new();
    if let Some(max) = req.max_tokens {
        generation.insert("maxOutputTokens".to_owned(), Value::from(max));
    }
    if let Some(t) = req.temperature {
        generation.insert("temperature".to_owned(), Value::from(t));
    }
    if !req.stop_sequences.is_empty() {
        generation.insert(
            "stopSequences".to_owned(),
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
            generation.insert(
                "thinkingConfig".to_owned(),
                json!({"thinkingBudget": budget, "includeThoughts": true}),
            );
        }
    }
    if !generation.is_empty() {
        body.insert("generationConfig".to_owned(), Value::Object(generation));
    }

    crate::codecs::web_search::apply(req, profile, &mut body)?;
    crate::codecs::extras::merge_body(profile, &mut body);
    let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
    crate::codecs::extras::merge_headers(profile, &mut headers);

    Ok(HttpRequest {
        method: "POST".to_owned(),
        url: url.to_owned(),
        headers,
        body: serde_json::to_vec(&Value::Object(body))
            .map_err(|e| LlmError::InvalidRequest {
                message: format!("request body is not serializable: {e}"),
            })?
            .into(),
        timeout: None,
    })
}

/// A tool result on this wire names the function, and a transcript carries only
/// the call id — so the names are collected from the `ToolUse` blocks that came
/// before. First one wins: an id is issued once.
fn tool_call_names(messages: &[ConversationMessage]) -> BTreeMap<ToolUseId, String> {
    let mut names = BTreeMap::new();
    for m in messages {
        for b in &m.content {
            if let ContentBlock::ToolUse { id, name, .. } = b {
                names.entry(id.clone()).or_insert_with(|| name.clone());
            }
        }
    }
    names
}

fn contents(
    messages: &[ConversationMessage],
    names: &BTreeMap<ToolUseId, String>,
) -> Result<Vec<Value>, LlmError> {
    let mut out = Vec::new();
    for m in messages {
        let role = match m.role {
            // This wire calls the assistant "model".
            MessageRole::Assistant => "model",
            MessageRole::User => "user",
            MessageRole::System => {
                return Err(LlmError::InvalidRequest {
                    message: "the system prompt belongs in `systemInstruction`".to_owned(),
                })
            }
        };
        let mut parts = Vec::new();
        for b in &m.content {
            if let Some(part) = encode_part(b, names)? {
                parts.push(part);
            }
        }
        if !parts.is_empty() {
            out.push(json!({"role": role, "parts": parts}));
        }
    }
    Ok(out)
}

fn encode_part(
    b: &ContentBlock,
    names: &BTreeMap<ToolUseId, String>,
) -> Result<Option<Value>, LlmError> {
    Ok(match b {
        ContentBlock::ProviderContent { .. } => {
            return Err(LlmError::UnsupportedCapability {
                message: "native content cannot be replayed on Gemini".to_owned(),
            })
        }
        ContentBlock::Text { text } => Some(json!({"text": text})),
        // This wire marks reasoning with a flag on an ordinary text part; it
        // carries no signature, so nothing is lost by replaying it as one.
        ContentBlock::Thinking { text, .. } => Some(json!({"text": text, "thought": true})),
        ContentBlock::RedactedThinking { .. } => None,
        ContentBlock::ToolUse { name, input, .. } => {
            Some(json!({"functionCall": {"name": name, "args": input}}))
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } => {
            let name = names
                .get(tool_use_id)
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: format!(
                        "a tool result for {tool_use_id} has no matching call in the transcript, \
                         and this wire keys results by function name"
                    ),
                })?;
            Some(json!({
                "functionResponse": {"name": name, "response": {"result": content}},
            }))
        }
        ContentBlock::Image { source } => Some(match source {
            ImageSource::Base64 { media_type, data } => {
                json!({"inlineData": {"mimeType": media_type, "data": data}})
            }
            // The API infers the type from the server's Content-Type, so
            // `mimeType` is deliberately omitted for a URL.
            ImageSource::Url { url } => json!({"fileData": {"fileUri": url}}),
        }),
        ContentBlock::Document { source, .. } => Some(match source {
            DocumentSource::Base64 { media_type, data } => {
                json!({"inlineData": {"mimeType": media_type, "data": data}})
            }
            DocumentSource::Text { media_type, data } => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(data.as_bytes());
                json!({"inlineData": {"mimeType": media_type, "data": encoded}})
            }
            DocumentSource::Url { url } => json!({"fileData": {"fileUri": url}}),
        }),
    })
}

fn encode_tool(t: &ToolSpec) -> Value {
    json!({
        "name": t.name,
        "description": t.description,
        "parameters": t.input_schema,
    })
}

fn encode_tool_choice(c: &ToolChoice) -> Value {
    let mode = match c {
        ToolChoice::Auto => "AUTO",
        ToolChoice::Any => "ANY",
        ToolChoice::None => "NONE",
        ToolChoice::Tool { name } => {
            return json!({"functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [name],
            }})
        }
    };
    json!({"functionCallingConfig": {"mode": mode}})
}
