//! Request encoding for `generateContent`.

use crate::codecs::json::{WireRequest, WireValue};
use crate::codecs::{CodecContext, EncodeRequest};
use crate::protocol::{
    ContentBlock, ConversationMessage, DocumentSource, ImageSource, LlmError, MessageRole,
    ProviderProfile, ToolChoice, ToolSpec, ToolUseId, VideoSource,
};

use base64::Engine;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

pub fn request<'a>(
    wire: EncodeRequest<'a>,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<WireRequest<'a>, LlmError> {
    crate::files::validate_direct_provider_file_inputs(
        wire.blocks(),
        profile,
        &opts.request_model,
        opts.file_scope(),
    )?;
    request_to(
        wire,
        profile,
        &super::generate_content_url(&profile.base_url, &opts.request_model, opts.stream),
        opts,
    )
}

/// The body is identical wherever this wire is hosted, and it carries no
/// stream flag — the URL says whether to stream. So the Vertex wrapper reuses
/// this with a URL of its own and nothing else changes.
pub(crate) fn request_to<'a>(
    wire: EncodeRequest<'a>,
    profile: &ProviderProfile,
    url: &str,
    opts: &CodecContext,
) -> Result<WireRequest<'a>, LlmError> {
    let req = wire.request();
    crate::codecs::reject_responses_continuation(
        req,
        crate::protocol::ProtocolFamily::GeminiGenerateContent,
    )?;
    if req.file_search.is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "hosted file search is supported only on Qwen Responses profiles".into(),
        });
    }
    let names = tool_call_names(&req.messages);
    let mut body = Map::new();
    let contents = contents(&req.messages, &names, profile, opts, wire)?;
    body.insert("contents".to_owned(), Value::Null);

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
    if !generation.is_empty() {
        body.insert("generationConfig".to_owned(), Value::Object(generation));
    }

    crate::codecs::web_search::apply(req, profile, &mut body)?;
    let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
    crate::codecs::inference::apply(req, opts, &mut body, &mut headers)?;
    crate::wire_options::merge_body(profile, &mut body);
    // The inference adapter has validated both protobuf JSON spellings and
    // emitted the canonical key, including unrecognized native tier values.
    body.remove("service_tier");
    crate::wire_options::merge_headers(profile, &mut headers);

    Ok(WireRequest::new(
        url.to_owned(),
        headers,
        WireValue::from(Value::Object(body)).with("contents", WireValue::array(contents)),
    ))
}

/// Match each result to the earlier call, retaining a provider-issued ID only
/// when one was present. Local IDs are never sent to Gemini.
fn tool_call_names(
    messages: &[ConversationMessage],
) -> BTreeMap<ToolUseId, (String, Option<String>)> {
    let mut names = BTreeMap::new();
    for m in messages {
        for b in &m.content {
            if let ContentBlock::ToolUse {
                id,
                name,
                provider_id,
                ..
            } = b
            {
                names
                    .entry(id.clone())
                    .or_insert_with(|| (name.clone(), provider_id.clone()));
            }
        }
    }
    names
}

fn contents<'a>(
    messages: &[ConversationMessage],
    names: &BTreeMap<ToolUseId, (String, Option<String>)>,
    profile: &ProviderProfile,
    opts: &CodecContext,
    wire: EncodeRequest<'a>,
) -> Result<Vec<WireValue<'a>>, LlmError> {
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
            let b = wire.block(b);
            if let Some(part) = inline(wire, b, opts)? {
                parts.push(part);
                continue;
            }
            if let Some(part) = encode_part(b, names, profile, opts)? {
                parts.push(part.into());
            }
        }
        if !parts.is_empty() {
            out.push(WireValue::from(json!({"role":role})).with("parts", WireValue::array(parts)));
        }
    }
    Ok(out)
}

fn encode_part(
    b: &ContentBlock,
    names: &BTreeMap<ToolUseId, (String, Option<String>)>,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<Option<Value>, LlmError> {
    Ok(match b {
        ContentBlock::ProviderContent { .. } => {
            return Err(LlmError::UnsupportedCapability {
                message: "native content cannot be replayed on Gemini".to_owned(),
            })
        }
        ContentBlock::Text {
            text,
            thought_signature,
        } => {
            let mut part = json!({"text": text});
            if let Some(signature) = thought_signature {
                part["thoughtSignature"] = Value::String(signature.clone());
            }
            Some(part)
        }
        ContentBlock::Thinking { text, signature } => {
            let mut part = json!({"text": text, "thought": true});
            if let Some(signature) = signature {
                part["thoughtSignature"] = Value::String(signature.clone());
            }
            Some(part)
        }
        ContentBlock::RedactedThinking { .. } => None,
        ContentBlock::ToolUse {
            name,
            input,
            provider_id,
            thought_signature,
            ..
        } => {
            let mut call = json!({"name": name, "args": input});
            if let Some(id) = provider_id {
                call["id"] = Value::String(id.clone());
            }
            let mut part = json!({"functionCall": call});
            if let Some(signature) = thought_signature {
                part["thoughtSignature"] = Value::String(signature.clone());
            }
            Some(part)
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } => {
            let (name, provider_id) =
                names
                    .get(tool_use_id)
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: format!(
                        "a tool result for {tool_use_id} has no matching call in the transcript, \
                         and this wire keys results by function name"
                    ),
                    })?;
            let mut response = json!({"name": name, "response": {"result": content}});
            if let Some(id) = provider_id {
                response["id"] = Value::String(id.clone());
            }
            Some(json!({"functionResponse": response}))
        }
        ContentBlock::Image { source } => Some(match source {
            ImageSource::Base64 { media_type, data } => {
                json!({"inlineData": {"mimeType": media_type, "data": data}})
            }
            // The API infers the type from the server's Content-Type, so
            // `mimeType` is deliberately omitted for a URL.
            ImageSource::Url { url } => json!({"fileData": {"fileUri": url}}),
            ImageSource::Attachment { .. } => {
                return Err(crate::codecs::unresolved_attachment_error())
            }
            ImageSource::ProviderFile { file } => {
                let file = crate::codecs::validate_provider_file(file, profile, opts)?;
                if !matches!(
                    file.protocol,
                    crate::protocol::ProtocolFamily::GeminiGenerateContent
                        | crate::protocol::ProtocolFamily::VertexGemini
                ) {
                    return Err(crate::codecs::provider_file_protocol_error());
                }
                let uri = file
                    .uri
                    .as_deref()
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: "Gemini provider file reference is missing its file URI".into(),
                    })?;
                let mime_type =
                    file.media_type
                        .as_deref()
                        .ok_or_else(|| LlmError::InvalidRequest {
                            message: "Gemini provider file reference is missing its media type"
                                .into(),
                        })?;
                json!({"fileData": {"mimeType": mime_type, "fileUri": uri}})
            }
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
            DocumentSource::Attachment { .. } => {
                return Err(crate::codecs::unresolved_attachment_error())
            }
            DocumentSource::ProviderFile { file } => {
                let file = crate::codecs::validate_provider_file(file, profile, opts)?;
                if !matches!(
                    file.protocol,
                    crate::protocol::ProtocolFamily::GeminiGenerateContent
                        | crate::protocol::ProtocolFamily::VertexGemini
                ) {
                    return Err(crate::codecs::provider_file_protocol_error());
                }
                let uri = file
                    .uri
                    .as_deref()
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: "Gemini provider file reference is missing its file URI".into(),
                    })?;
                let mime_type =
                    file.media_type
                        .as_deref()
                        .ok_or_else(|| LlmError::InvalidRequest {
                            message: "Gemini provider file reference is missing its media type"
                                .into(),
                        })?;
                json!({"fileData": {"mimeType": mime_type, "fileUri": uri}})
            }
        }),
        ContentBlock::Video { source } => Some(match source {
            VideoSource::Base64 { media_type, data } => {
                json!({"inlineData": {"mimeType": media_type, "data": data}})
            }
            VideoSource::Url { url } => json!({"fileData": {"fileUri": url}}),
            VideoSource::Attachment { .. } => {
                return Err(crate::codecs::unresolved_attachment_error())
            }
            VideoSource::ProviderFile { file } => {
                let file = crate::codecs::validate_provider_file(file, profile, opts)?;
                let uri = file
                    .uri
                    .as_deref()
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: "Gemini video file reference is missing its file URI".into(),
                    })?;
                let mime_type =
                    file.media_type
                        .as_deref()
                        .ok_or_else(|| LlmError::InvalidRequest {
                            message: "Gemini video file reference is missing its media type".into(),
                        })?;
                if !mime_type.to_ascii_lowercase().starts_with("video/") {
                    return Err(LlmError::InvalidRequest {
                        message: "Gemini video file reference has a non-video media type".into(),
                    });
                }
                json!({"fileData": {"mimeType": mime_type, "fileUri": uri}})
            }
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

fn inline<'a>(
    wire: EncodeRequest<'a>,
    block: &ContentBlock,
    _context: &CodecContext,
) -> Result<Option<WireValue<'a>>, LlmError> {
    let Some(media) = wire.inline_media(block)? else {
        return Ok(None);
    };
    let attachment = media.attachment;
    let (_kind, _title) = match block {
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
    let source = WireValue::from(json!({"mimeType":attachment.media_type})).with("data", data());
    Ok(Some(WireValue::from(json!({})).with("inlineData", source)))
}
