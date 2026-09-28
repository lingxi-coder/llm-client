//! Request encoding for `generateContent`.

use crate::codecs::json::{WireRequest, WireValue};
use crate::codecs::{CodecContext, EncodeRequest};
use crate::protocol::{
    ChatRequest, ContentBlock, ConversationMessage, DocumentSource, ImageSource, LlmError,
    MessageRole, ProtocolFamily, ProviderProfile, ToolChoice, ToolSpec, ToolUseId, VideoSource,
};

use base64::Engine;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

/// Validate typed inline audio before attachments or provider dispatch.
pub(crate) fn validate_audio_input(req: &ChatRequest) -> Result<(), LlmError> {
    if req.metadata.get("openrouter_chat_audio").is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenRouter Chat audio output settings cannot be used with Gemini".into(),
        });
    }
    for message in &req.messages {
        for block in &message.content {
            if let ContentBlock::Audio { format, data } = block {
                if message.role != MessageRole::User {
                    return Err(LlmError::InvalidRequest {
                        message: "Gemini inline audio requires a user message".into(),
                    });
                }
                audio_mime(format)?;
                if data.is_empty() || data.len() > 20_000_000 {
                    return Err(LlmError::InvalidRequest {
                        message:
                            "Gemini inline audio must be non-empty and fit the 20 MB request limit"
                                .into(),
                    });
                }
                // Validate each quartet without allocating a decoded copy.
                if !data.len().is_multiple_of(4) {
                    return Err(invalid_audio_base64());
                }
                let chunks = data.as_bytes().chunks_exact(4);
                let count = chunks.len();
                for (index, chunk) in chunks.enumerate() {
                    if (index + 1 < count && chunk.contains(&b'='))
                        || base64::engine::general_purpose::STANDARD
                            .decode_slice(chunk, &mut [0u8; 3])
                            .is_err()
                    {
                        return Err(invalid_audio_base64());
                    }
                }
            }
        }
    }
    Ok(())
}

fn invalid_audio_base64() -> LlmError {
    LlmError::InvalidRequest {
        message: "Gemini inline audio requires standard base64 without a data URI".into(),
    }
}

fn audio_mime(format: &str) -> Result<&'static str, LlmError> {
    match format {
        "wav" => Ok("audio/wav"),
        "mp3" => Ok("audio/mp3"),
        "aiff" => Ok("audio/aiff"),
        "aac" => Ok("audio/aac"),
        "ogg" => Ok("audio/ogg"),
        "flac" => Ok("audio/flac"),
        "mpeg" => Ok("audio/mpeg"),
        "m4a" => Ok("audio/m4a"),
        "l16" => Ok("audio/l16"),
        "opus" => Ok("audio/opus"),
        "alaw" => Ok("audio/alaw"),
        "mulaw" => Ok("audio/mulaw"),
        "webm" => Ok("audio/webm"),
        _ => Err(LlmError::InvalidRequest {
            message: format!("unsupported Gemini inline audio format {format:?}"),
        }),
    }
}

pub fn request<'a>(
    wire: EncodeRequest<'a>,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<WireRequest<'a>, LlmError> {
    crate::files::validate_direct_provider_file_inputs_at(
        wire.blocks(),
        profile,
        &opts.request_model,
        opts.file_scope(),
        opts.file_validation_time(),
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
    validate_audio_input(req)?;
    crate::providers::anthropic::code_execution::validate(req, opts)?;
    crate::providers::openrouter::server_tools::validate(req, profile, None, false)?;
    crate::providers::google::hosted_tools::validate_hosted_tool_request(
        req,
        profile,
        &opts.request_model,
    )?;
    if req.hosted_anthropic_tool_search().is_some()
        || req.hosted_openai_tool_search().is_some()
        || req.tools.iter().any(|tool| tool.defer_loading)
    {
        return Err(LlmError::UnsupportedCapability {
            message: "Anthropic tool search and defer_loading require an Anthropic Messages codec"
                .into(),
        });
    }
    crate::codecs::reject_code_interpreter(req, profile.protocol)?;
    crate::codecs::reject_responses_continuation(
        req,
        crate::protocol::ProtocolFamily::GeminiGenerateContent,
    )?;
    if req.hosted_file_search().is_some() {
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
            json!({"parts": Value::Null}),
        );
    }
    if !req.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            json!([{ "functionDeclarations": req.tools.iter().map(encode_tool).collect::<Vec<_>>() }]),
        );
        let tool_config = if crate::providers::google::hosted_tools::has_gemini_hosted_tools(req) {
            json!({
                "functionCallingConfig": {"mode": "VALIDATED"},
                "includeServerSideToolInvocations": true
            })
        } else {
            encode_tool_choice(&req.tool_choice)
        };
        body.insert("toolConfig".to_owned(), tool_config);
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
    crate::providers::google::hosted_tools::apply_gemini_hosted_tools(
        req,
        &opts.request_model,
        &mut body,
    )?;
    let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
    crate::codecs::inference::apply(req, opts, &mut body, &mut headers)?;
    crate::codecs::request_controls::apply(req, profile.protocol, &mut body)?;
    crate::codecs::structured::apply(req, opts, &mut body)?;
    crate::wire_options::merge_body(profile, &mut body);
    // The inference adapter has validated both protobuf JSON spellings and
    // emitted the canonical key, including unrecognized native tier values.
    body.remove("service_tier");
    crate::wire_options::merge_headers(profile, &mut headers);

    let mut body =
        WireValue::from(Value::Object(body)).with("contents", WireValue::array(contents));
    if !req.tools.is_empty() {
        body = body.map_array_field("tools", |index, tool| {
            let tool = WireValue::from(tool);
            if index != 0 {
                return tool;
            }
            tool.map_array_field("functionDeclarations", |index, function| {
                WireValue::from(function).with(
                    "parameters",
                    WireValue::borrowed(&req.tools[index].input_schema),
                )
            })
        });
    }
    if !req.system.is_empty() {
        body = body.with(
            "systemInstruction",
            WireValue::from(json!({})).with(
                "parts",
                WireValue::array(
                    req.system
                        .iter()
                        .map(|block| {
                            WireValue::from(json!({})).with("text", WireValue::text(&block.text))
                        })
                        .collect(),
                ),
            ),
        );
    }
    let request = WireRequest::new(url.to_owned(), headers, body);
    if req
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .any(|block| matches!(block, ContentBlock::Audio { .. }))
        && request.body_len()? > 20_000_000
    {
        return Err(LlmError::RequestTooLarge {
            message: "Gemini inline audio request exceeds 20 MB; use the Files API".into(),
        });
    }
    Ok(request)
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
    messages: &'a [ConversationMessage],
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
                });
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
                parts.push(part);
            }
        }
        if !parts.is_empty() {
            out.push(WireValue::from(json!({"role":role})).with("parts", WireValue::array(parts)));
        }
    }
    Ok(out)
}

fn encode_part<'a>(
    b: &'a ContentBlock,
    names: &BTreeMap<ToolUseId, (String, Option<String>)>,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<Option<WireValue<'a>>, LlmError> {
    let part = match b {
        ContentBlock::ProviderContent { protocol, value } => {
            if *protocol != profile.protocol
                || !matches!(
                    profile.protocol,
                    ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini
                )
                || !super::native::is_replayable_native_part(value)
            {
                return Err(LlmError::UnsupportedCapability {
                    message: "native content is not a replayable Gemini hosted-tool Part for this protocol".to_owned(),
                });
            }
            return Ok(Some(WireValue::from(value.clone())));
        }
        ContentBlock::Text {
            text,
            thought_signature,
        } => {
            let mut part = WireValue::from(json!({})).with("text", WireValue::text(text));
            if let Some(signature) = thought_signature {
                part["thoughtSignature"] = Value::String(signature.clone());
            }
            return Ok(Some(part));
        }
        ContentBlock::Thinking { text, signature } => {
            let mut part = WireValue::from(json!({"text": Value::Null, "thought": true}))
                .with("text", WireValue::text(text));
            if let Some(signature) = signature {
                part["thoughtSignature"] = Value::String(signature.clone());
            }
            return Ok(Some(part));
        }
        ContentBlock::RedactedThinking { .. } => None,
        ContentBlock::ToolUse {
            name,
            input,
            provider_id,
            thought_signature,
            ..
        } => {
            let mut call =
                WireValue::from(json!({"name": name})).with("args", WireValue::borrowed(input));
            if let Some(id) = provider_id {
                call["id"] = Value::String(id.clone());
            }
            let mut part = WireValue::from(json!({})).with("functionCall", call);
            if let Some(signature) = thought_signature {
                part["thoughtSignature"] = Value::String(signature.clone());
            }
            return Ok(Some(part));
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
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
            let mut response = WireValue::from(json!({"name": name})).with(
                "response",
                WireValue::from(json!({})).with(
                    if *is_error { "error" } else { "result" },
                    WireValue::text(content),
                ),
            );
            if let Some(id) = provider_id {
                response["id"] = Value::String(id.clone());
            }
            return Ok(Some(
                WireValue::from(json!({})).with("functionResponse", response),
            ));
        }
        ContentBlock::Image { source } => Some(match source {
            ImageSource::Base64 { media_type, data } => {
                json!({"inlineData": {"mimeType": media_type, "data": data}})
            }
            // The API infers the type from the server's Content-Type, so
            // `mimeType` is deliberately omitted for a URL.
            ImageSource::Url { url } => json!({"fileData": {"fileUri": url}}),
            ImageSource::Attachment { .. } => {
                return Err(crate::codecs::unresolved_attachment_error());
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
                return Err(crate::codecs::unresolved_attachment_error());
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
        ContentBlock::Audio { format, data } => {
            return Ok(Some(
                WireValue::from(json!({})).with(
                    "inlineData",
                    WireValue::from(json!({ "mimeType": audio_mime(format)? }))
                        .with("data", WireValue::text(data)),
                ),
            ));
        }
        ContentBlock::Video { source } => Some(match source {
            VideoSource::Base64 { media_type, data } => {
                json!({"inlineData": {"mimeType": media_type, "data": data}})
            }
            VideoSource::Url { url } => json!({"fileData": {"fileUri": url}}),
            VideoSource::Attachment { .. } => {
                return Err(crate::codecs::unresolved_attachment_error());
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
    };
    Ok(part.map(WireValue::from))
}

fn encode_tool(t: &ToolSpec) -> Value {
    json!({
        "name": t.name,
        "description": t.description,
        "parameters": Value::Null,
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
            }});
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
