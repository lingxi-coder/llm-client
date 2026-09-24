//! Request encoding.
//!
//! Provider quirks are keyed off the profile, never off a provider name in
//! code: `extra` on a `ProviderProfile` is where a settings entry says "this
//! endpoint needs the usage object requested explicitly" or "this endpoint
//! rejects tool_choice in thinking mode". That is what keeps a new
//! OpenAI-compatible provider a settings change (gate 30).

use crate::codecs::json::{WireRequest, WireValue};
use crate::codecs::{CodecContext, EncodeRequest};
use crate::protocol::{
    ContentBlock, ConversationMessage, DocumentSource, ImageSource, LlmError, MessageRole,
    ProviderProfile, ToolChoice,
};

use base64::Engine;
use serde_json::{json, Map, Value};

/// A boolean knob a profile may set in `extra`.
fn flag(profile: &ProviderProfile, key: &str) -> bool {
    profile
        .extra
        .get(key)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub fn request<'a>(
    wire: EncodeRequest<'a>,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<WireRequest<'a>, LlmError> {
    let req = wire.request();
    crate::files::validate_direct_provider_file_inputs(
        wire.blocks(),
        profile,
        &opts.request_model,
        opts.file_scope(),
    )?;
    crate::codecs::reject_responses_continuation(req, crate::protocol::ProtocolFamily::OpenAiChat)?;
    if req.file_search.is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "hosted file search is supported only on Qwen Responses profiles".into(),
        });
    }
    let max_tokens_field = max_tokens_field(profile)?;
    let mut messages: Vec<WireValue<'a>> = Vec::new();
    let qwen_long = profile.provider_id.as_str() == "qwen"
        && crate::files::is_qwen_long_model(&opts.request_model);
    if qwen_long {
        if !crate::files::qwen_long_region_supported(profile) {
            return Err(LlmError::UnsupportedCapability {
                message: "Qwen-Long is supported only on a Beijing Qwen endpoint".into(),
            });
        }
        crate::files::validate_qwen_long_blocks(wire.blocks())?;
    }
    let keep_reasoning = flag(profile, "preserve_reasoning_content");
    let pdf_only_files = flag(profile, "chat_pdf_only")
        || url::Url::parse(&profile.base_url)
            .ok()
            .is_some_and(|url| url.host_str() == Some("api.openai.com"));
    let mut consumed_leading_system = false;

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
        if !qwen_long || !text.trim().is_empty() {
            messages.push((json!({"role": "system", "content": text})).into());
        }
    }

    if qwen_long {
        let mut file_ids = Vec::new();
        for block in req.messages.iter().flat_map(|message| &message.content) {
            let block = wire.block(block);
            let file = match block {
                ContentBlock::Document {
                    source: DocumentSource::ProviderFile { file },
                    ..
                }
                | ContentBlock::Image {
                    source: ImageSource::ProviderFile { file },
                } => file,
                _ => continue,
            };
            let file = crate::codecs::validate_provider_file(file, profile, opts)?;
            if file.protocol != crate::protocol::ProtocolFamily::OpenAiChat
                || file.purpose.as_deref() != Some("file-extract")
            {
                return Err(crate::codecs::provider_file_protocol_error());
            }
            if !crate::files::valid_qwen_file_id(&file.file_id) {
                return Err(LlmError::InvalidRequest {
                    message: "Qwen-Long file IDs must contain only letters, digits, hyphens, or underscores".into(),
                });
            }
            file_ids.push(format!("fileid://{}", file.file_id));
        }
        if !file_ids.is_empty() {
            if file_ids.len() > crate::files::QWEN_LONG_MAX_FILE_REFERENCES {
                return Err(LlmError::InvalidRequest {
                    message: format!(
                        "Qwen-Long accepts at most {} file references per request",
                        crate::files::QWEN_LONG_MAX_FILE_REFERENCES
                    ),
                });
            }
            if messages.is_empty() {
                if let Some(first) = req
                    .messages
                    .first()
                    .filter(|message| matches!(&message.role, MessageRole::System))
                {
                    let mut encoded = encode_message(
                        first,
                        keep_reasoning,
                        pdf_only_files,
                        qwen_long,
                        profile,
                        opts,
                        wire,
                    )?;
                    if encoded.len() == 1
                        && encoded[0].get("role").and_then(Value::as_str) == Some("system")
                        && encoded[0]
                            .get("content")
                            .and_then(Value::as_str)
                            .is_some_and(|content| !content.trim().is_empty())
                    {
                        messages.push(encoded.remove(0));
                        consumed_leading_system = true;
                    }
                }
                if messages.is_empty() {
                    messages.push(
                        (json!({"role": "system", "content": "You are a helpful assistant."}))
                            .into(),
                    );
                }
            } else if messages[0]
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| content.trim().is_empty())
            {
                messages[0]["content"] = Value::String("You are a helpful assistant.".into());
            }
            messages.push((json!({"role": "system", "content": file_ids.join(",")})).into());
        }
    }

    for m in req
        .messages
        .iter()
        .skip(usize::from(consumed_leading_system))
    {
        messages.extend(encode_message(
            m,
            keep_reasoning,
            pdf_only_files,
            qwen_long,
            profile,
            opts,
            wire,
        )?);
    }

    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        Value::String(opts.request_model.clone()),
    );
    body.insert("messages".to_owned(), Value::Null);

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
        body.insert(max_tokens_field.to_owned(), Value::from(max));
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
        body.insert(
            "tool_choice".to_owned(),
            encode_tool_choice(&req.tool_choice),
        );
    }

    // Whatever else this particular endpoint understands. Additive only, and
    // never a credential — see `wire_extras`.
    crate::codecs::web_search::apply(req, profile, &mut body)?;
    let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
    crate::codecs::inference::apply(req, opts, &mut body, &mut headers)?;
    crate::wire_options::merge_body(profile, &mut body);
    if req.max_tokens.is_some() {
        // The selected typed field is authoritative. A profile body extra must
        // not add the other spelling and send conflicting output limits.
        let alternate = if max_tokens_field == "max_tokens" {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        body.remove(alternate);
    }
    crate::wire_options::merge_headers(profile, &mut headers);

    let url = format!(
        "{}/chat/completions",
        profile.base_url.trim_end_matches('/')
    );
    Ok(WireRequest::new(
        url,
        headers,
        WireValue::from(Value::Object(body)).with("messages", WireValue::array(messages)),
    ))
}

/// One conversation message becomes one or more wire messages: a tool result
/// cannot share a message with text, so any pending text is flushed first and
/// the result becomes its own `role: "tool"` entry.
fn encode_message<'a>(
    m: &ConversationMessage,
    keep_reasoning: bool,
    pdf_only_files: bool,
    qwen_long: bool,
    profile: &ProviderProfile,
    opts: &CodecContext,
    wire: EncodeRequest<'a>,
) -> Result<Vec<WireValue<'a>>, LlmError> {
    let role = match m.role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::System => "system",
    };
    let mut native_reasoning = None;
    for block in &m.content {
        let block = wire.block(block);
        if let ContentBlock::ProviderContent { protocol, value } = block {
            if m.role != MessageRole::Assistant {
                return Err(LlmError::InvalidRequest {
                    message: "Chat reasoning metadata belongs to an assistant message".into(),
                });
            }
            if native_reasoning.is_some() {
                return Err(LlmError::InvalidRequest {
                    message: "an assistant message may contain only one Chat reasoning envelope"
                        .into(),
                });
            }
            native_reasoning = Some(super::reasoning::replay_fields(*protocol, value)?);
        }
    }
    let mut reasoning = String::new();
    let mut parts: Vec<WireValue<'a>> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut out: Vec<WireValue<'a>> = Vec::new();

    for block in &m.content {
        let block = wire.block(block);
        if let Some(media) = inline(wire, block, opts)? {
            parts.push(media);
            continue;
        }
        match block {
            // Pre-scanned because the envelope describes the whole message,
            // including tool calls which may precede it in the block sequence.
            ContentBlock::ProviderContent { .. } => {}
            ContentBlock::Text { text, .. } => {
                parts.push((json!({"type":"text", "text":text})).into());
            }
            ContentBlock::Thinking { text: t, .. } => {
                if keep_reasoning && native_reasoning.is_none() && m.role == MessageRole::Assistant
                {
                    reasoning.push_str(t);
                }
            }
            ContentBlock::Image { source } => match source {
                ImageSource::ProviderFile { file } => {
                    let file = crate::codecs::validate_provider_file(file, profile, opts)?;
                    if qwen_long
                        && file.protocol == crate::protocol::ProtocolFamily::OpenAiChat
                        && file.purpose.as_deref() == Some("file-extract")
                    {
                        continue;
                    }
                    return Err(LlmError::UnsupportedCapability {
                        message: "Chat Completions image input does not support provider file ids"
                            .into(),
                    });
                }
                ImageSource::Attachment { .. } => {
                    return Err(crate::codecs::unresolved_attachment_error());
                }
                _ => parts.push(
                    (json!({
                        "type": "image_url",
                        "image_url": { "url": image_url(source)? },
                    }))
                    .into(),
                ),
            },
            ContentBlock::Document { source, .. } => {
                if let DocumentSource::ProviderFile { file } = source {
                    let file = crate::codecs::validate_provider_file(file, profile, opts)?;
                    if file.protocol != crate::protocol::ProtocolFamily::OpenAiChat {
                        return Err(crate::codecs::provider_file_protocol_error());
                    }
                    if qwen_long
                        && profile.provider_id.as_str() == "qwen"
                        && file.purpose.as_deref() == Some("file-extract")
                    {
                        continue;
                    }
                    if !pdf_only_files || file.media_type.as_deref() != Some("application/pdf") {
                        return Err(LlmError::UnsupportedCapability {
                            message: "Chat Completions provider file references are supported only for PDF input".into(),
                        });
                    }
                    parts.push(
                        (json!({
                            "type": "file",
                            "file": { "file_id": file.file_id },
                        }))
                        .into(),
                    );
                    continue;
                }
                let file_data = document_url(source, pdf_only_files)?;
                parts.push(
                    (json!({
                        "type": "file",
                        "file": { "file_data": file_data },
                    }))
                    .into(),
                );
            }
            // Signed reasoning round-trips only on providers that sign it
            // (gate 18); this wire has no slot, so a replayed block is dropped
            // rather than sent somewhere it would be rejected.
            ContentBlock::RedactedThinking { .. } => {}
            ContentBlock::Video { .. } => {
                return Err(LlmError::UnsupportedCapability {
                    message: "Chat Completions does not support video content blocks".into(),
                });
            }
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
                flush_message(
                    &mut out,
                    role,
                    &mut parts,
                    &mut reasoning,
                    &mut tool_calls,
                    &mut native_reasoning,
                );
                out.push(
                    (json!({
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": content,
                    }))
                    .into(),
                );
            }
        }
    }

    flush_message(
        &mut out,
        role,
        &mut parts,
        &mut reasoning,
        &mut tool_calls,
        &mut native_reasoning,
    );
    Ok(out)
}

/// Select the request field used for the output token limit. A profile can opt
/// a direct OpenAI Chat Completions endpoint into the field required by newer
/// model families while compatible endpoints retain the historical default.
fn max_tokens_field(profile: &ProviderProfile) -> Result<&'static str, LlmError> {
    match profile.extra.get("max_tokens_field") {
        None => Ok("max_tokens"),
        Some(Value::String(field)) if field == "max_tokens" => Ok("max_tokens"),
        Some(Value::String(field)) if field == "max_completion_tokens" => {
            Ok("max_completion_tokens")
        }
        Some(_) => Err(LlmError::InvalidRequest {
            message:
                "profile extra.max_tokens_field must be \"max_tokens\" or \"max_completion_tokens\""
                    .to_owned(),
        }),
    }
}

/// Hosted images are sent as-is; base64 images become data URIs.
fn image_url(source: &ImageSource) -> Result<String, LlmError> {
    match source {
        ImageSource::Url { url } => Ok(url.clone()),
        ImageSource::Base64 { media_type, data } => Ok(format!("data:{media_type};base64,{data}")),
        ImageSource::Attachment { .. } => Err(crate::codecs::unresolved_attachment_error()),
        ImageSource::ProviderFile { .. } => Err(LlmError::UnsupportedCapability {
            message: "Chat Completions image input does not support provider file ids".into(),
        }),
    }
}

/// Base64 and text documents become data URIs. This Chat Completions form
/// expects inline file data, so a hosted URL cannot be represented safely.
fn document_url(source: &DocumentSource, pdf_only_files: bool) -> Result<String, LlmError> {
    match source {
        DocumentSource::Base64 { media_type, data } => {
            if pdf_only_files && media_type != "application/pdf" {
                return Err(LlmError::UnsupportedCapability {
                    message: "OpenAI Chat Completions accepts only PDF file input".into(),
                });
            }
            Ok(format!("data:{media_type};base64,{data}"))
        }
        // Plain text is still sent as a data URI: the field is a file payload,
        // not a content part.
        DocumentSource::Text { media_type, data } => {
            if pdf_only_files {
                return Err(LlmError::UnsupportedCapability {
                    message: "OpenAI Chat Completions accepts only PDF file input".into(),
                });
            }
            Ok(format!(
                "data:{media_type};base64,{}",
                base64_of(data.as_bytes())
            ))
        }
        DocumentSource::Url { .. } => Err(LlmError::UnsupportedCapability {
            message:
                "Chat Completions does not support document URLs; provide inline document data"
                    .to_owned(),
        }),
        DocumentSource::Attachment { .. } => Err(crate::codecs::unresolved_attachment_error()),
        DocumentSource::ProviderFile { .. } => Err(LlmError::UnsupportedCapability {
            message: "Chat Completions provider file references must be prepared as PDF input"
                .into(),
        }),
    }
}

fn user_message<'a>(role: &str, parts: &[WireValue<'a>]) -> WireValue<'a> {
    let content = if parts
        .iter()
        .all(|part| part.get("type").and_then(Value::as_str) == Some("text"))
    {
        WireValue::from(Value::String(
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
        ))
    } else {
        WireValue::array(parts.to_vec())
    };
    WireValue::from(json!({"role": role})).with("content", content)
}

fn flush_message<'a>(
    out: &mut Vec<WireValue<'a>>,
    role: &str,
    parts: &mut Vec<WireValue<'a>>,
    reasoning: &mut String,
    calls: &mut Vec<Value>,
    native_reasoning: &mut Option<Map<String, Value>>,
) {
    if parts.is_empty() && reasoning.is_empty() && calls.is_empty() && native_reasoning.is_none() {
        return;
    }
    let role = if calls.is_empty() { role } else { "assistant" };
    let mut message = user_message(role, parts);
    if role == "assistant" {
        if parts.is_empty() {
            message["content"] = Value::Null;
        }
        if let Some(native) = native_reasoning.take() {
            message.object_mut().extend(native);
        } else if !reasoning.is_empty() {
            message["reasoning_content"] = Value::String(std::mem::take(reasoning));
        }
    }
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(std::mem::take(calls));
    }
    out.push(message);
    parts.clear();
    reasoning.clear();
}

fn encode_tool(t: &crate::protocol::ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
            "strict": t.strict,
        }
    })
}

/// Preserve the caller's tool selection in Chat Completions vocabulary.
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

fn inline<'a>(
    wire: EncodeRequest<'a>,
    block: &ContentBlock,
    context: &CodecContext,
) -> Result<Option<WireValue<'a>>, LlmError> {
    let Some(media) = wire.inline_media(block)? else {
        return Ok(None);
    };
    let attachment = media.attachment;
    let (kind, _title) = match block {
        ContentBlock::Image { .. } => ("image", None),
        ContentBlock::Document { title, .. } => ("document", title.as_deref()),
        ContentBlock::Video { .. } => ("video", None),
        _ => unreachable!("inline_media accepts only attachment blocks"),
    };
    let _data = || WireValue::base64(media.bytes, String::new());
    let uri = || {
        WireValue::base64(
            media.bytes,
            format!("data:{};base64,", attachment.media_type),
        )
    };
    Ok(Some(match kind {
        "image" => WireValue::from(json!({"type":"image_url"}))
            .with("image_url", WireValue::from(json!({})).with("url", uri())),
        "document" => {
            let pdf_only = context
                .profile
                .extra
                .get("chat_pdf_only")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
                || url::Url::parse(&context.profile.base_url)
                    .ok()
                    .is_some_and(|url| url.host_str() == Some("api.openai.com"));
            if pdf_only && attachment.media_type != "application/pdf" {
                return Err(LlmError::UnsupportedCapability {
                    message: "OpenAI Chat Completions accepts only PDF file input".into(),
                });
            }
            WireValue::from(json!({"type":"file"}))
                .with("file", WireValue::from(json!({})).with("file_data", uri()))
        }
        _ => {
            return Err(LlmError::UnsupportedCapability {
                message: "Chat Completions does not support video content blocks".into(),
            })
        }
    }))
}
