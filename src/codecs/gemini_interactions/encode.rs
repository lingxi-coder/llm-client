use crate::codecs::{
    json::{WireRequest, WireValue},
    CodecContext, EncodeRequest, RequestMode,
};
use crate::protocol::*;
use crate::providers::google::{
    computer::{GeminiComputerToolConfig, CALL_FORMAT, RESULT_FORMAT},
    interactions::{InteractionInput, InteractionRequest, InteractionTool},
};
use crate::transport::HttpRequest;
use serde_json::{json, Value};
use std::collections::BTreeMap;
fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
fn unsupported(message: impl Into<String>) -> LlmError {
    LlmError::UnsupportedCapability {
        message: message.into(),
    }
}

fn text_payload(units: &[u16]) -> crate::protocol::JsonPayload {
    crate::protocol::JsonPayload::string_from_utf16_code_units(units)
}

fn without_root_field(
    source: &crate::protocol::JsonPayload,
    omitted: &str,
) -> Result<crate::protocol::JsonPayload, LlmError> {
    use crate::response_json::FlatJsonKind;

    let scanner = source.scanner()?;
    let root = scanner.root_node();
    if root.kind() != FlatJsonKind::Object {
        return Err(invalid("Interactions native step must be an object"));
    }
    let omitted = omitted.encode_utf16().collect::<Vec<_>>();
    let mut fields = Vec::new();
    for entry in root
        .object_entries()
        .ok_or_else(|| invalid("Interactions native step has no object fields"))?
    {
        if entry.key_units == omitted {
            continue;
        }
        let value = crate::exact_json::serialize_json_node(
            entry.value,
            crate::exact_json::JsonEncoding::Serde,
        )?;
        fields.push((entry.key_units.to_vec(), value));
    }
    crate::protocol::JsonPayload::object_from_fields(&fields)
        .map_err(|error| invalid(format!("Interactions native step is invalid: {error}")))
}

fn result_block(block: &crate::protocol::JsonPayload) -> Result<WireValue<'static>, LlmError> {
    match block.string_at("/type").as_deref() {
        Some("text")
            if block.kind_at("/text") == Some(crate::response_json::FlatJsonKind::String) =>
        {
            Ok(WireValue::owned_raw_json(block.clone()))
        }
        Some("image") => {
            let normalized = crate::providers::google::interactions::encode_result_block(block)?;
            Ok(WireValue::owned_raw_json(normalized))
        }
        Some("document") => {
            let normalized = crate::providers::google::interactions::encode_result_block(block)?;
            Ok(WireValue::owned_raw_json(normalized))
        }
        _ => Err(invalid(
            "unsupported or malformed Interactions function-result content",
        )),
    }
}

fn flush_parts<'a>(
    steps: &mut Vec<WireValue<'a>>,
    parts: &mut Vec<WireValue<'a>>,
    role: MessageRole,
) {
    if parts.is_empty() {
        return;
    }
    let kind = if role == MessageRole::Assistant {
        "model_output"
    } else {
        "user_input"
    };
    let content = std::mem::take(parts);
    steps.push(
        WireValue::from(json!({"type":kind,"content":[]}))
            .with("content", WireValue::array(content)),
    );
}

pub(super) fn validate(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    crate::codecs::validate_tool_wire_shapes(req)?;
    if context.mode() == RequestMode::CountTokens {
        return Err(unsupported("Interactions has no exact count endpoint"));
    }
    if context
        .account_scope()
        .is_none_or(|scope| scope.trim().is_empty())
    {
        return Err(invalid("Interactions requires a non-secret account_scope"));
    }
    if !matches!(
        context.profile().auth,
        AuthStrategy::ApiKey | AuthStrategy::None
    ) {
        return Err(unsupported(
            "Gemini Interactions requires API key authentication",
        ));
    }
    if !req.hosted_tools.is_empty()
        || !matches!(req.output_format, OutputFormat::Text)
        || req.prompt_cache != PromptCachePolicy::default()
    {
        return Err(unsupported(
            "Interactions chat currently supports text output, function tools and native desktop computer declarations",
        ));
    }
    if req.controls.responses != Default::default()
        || req.controls.anthropic != Default::default()
        || req.controls.response_format.is_some()
    {
        return Err(unsupported(
            "foreign wire controls cannot be encoded by Interactions",
        ));
    }
    if req
        .thinking
        .as_ref()
        .is_some_and(|thinking| thinking.budget.is_some() || thinking.mode.is_some())
        || req.service_tier.is_some()
    {
        return Err(unsupported(
            "Interactions thinking budgets, explicit thinking mode and service tier have no unified mapping",
        ));
    }
    if let Some(reference) = &req.continuation {
        reference.validate(
            context.profile(),
            context.request_model(),
            context.account_scope(),
            None,
        )?;
    }
    let mut native_count = 0;
    for extension in &req.native_options {
        if !extension.is::<GeminiComputerToolConfig>() {
            return Err(unsupported(format!(
                "Interactions cannot encode native options {}",
                extension.format()
            )));
        }
        let config = extension.decode::<GeminiComputerToolConfig>()?;
        config.validate()?;
        native_count += 1;
    }
    if native_count > 1 {
        return Err(invalid("duplicate Interactions desktop declaration"));
    }
    if native_count > 0
        && req.tools.iter().any(|tool| {
            tool.name == "computer"
                || crate::providers::google::computer::is_desktop_function(&tool.name)
        })
    {
        return Err(invalid(
            "desktop tool members and computer management functions must be declared in different turns",
        ));
    }
    for tool in &req.tools {
        if tool.defer_loading
            || tool
                .tool_type
                .as_ref()
                .is_some_and(|kind| kind != "function")
        {
            return Err(unsupported(
                "Interactions tools must be ordinary function definitions",
            ));
        }
    }
    Ok(())
}

pub(super) fn request(
    input: EncodeRequest<'_>,
    context: &CodecContext,
) -> Result<HttpRequest, LlmError> {
    use base64::Engine;
    let req = input.request();
    validate(req, context)?;
    let mut calls = BTreeMap::new();
    for block in input.blocks() {
        if let ContentBlock::ToolUse {
            id,
            name,
            provider_id,
            ..
        } = block
        {
            calls.insert(
                id.as_str(),
                (name.as_str(), provider_id.as_deref().unwrap_or(id.as_str())),
            );
        }
    }
    let mut steps: Vec<WireValue<'_>> = vec![];
    for message in &req.messages {
        if message.role == MessageRole::System {
            continue;
        }
        if !message.native_options.is_empty() {
            return Err(unsupported(
                "Interactions message native options are unsupported",
            ));
        }
        // Prior calls may be retained solely to identify new function receipts.
        if req.continuation.is_some() && message.role == MessageRole::Assistant {
            continue;
        }
        let mut parts: Vec<WireValue<'_>> = vec![];
        for original in &message.content {
            let block = input.block(original);
            match block {
                ContentBlock::Text {
                    text,
                    thought_signature: None,
                    citations: None,
                } => parts.push(
                    WireValue::from(json!({"type":"text","text":null}))
                        .with("text", WireValue::text(text)),
                ),
                ContentBlock::TextJsUtf16 {
                    utf16_code_units,
                    thought_signature: None,
                    citations: None,
                    ..
                } => parts.push(WireValue::from(json!({"type":"text","text":null})).with(
                    "text",
                    WireValue::owned_raw_json(text_payload(utf16_code_units)),
                )),
                ContentBlock::Image {
                    source: ImageSource::Base64 { media_type, data },
                } => parts.push(WireValue::from(
                    json!({"type":"image","mime_type":media_type,"data":data}),
                )),
                ContentBlock::Image {
                    source: ImageSource::Url { url },
                } => parts.push(WireValue::from(json!({"type":"image","uri":url}))),
                ContentBlock::Image {
                    source: ImageSource::Attachment { .. },
                } => {
                    let media = input
                        .inline_media(block)?
                        .ok_or_else(crate::codecs::unresolved_attachment_error)?;
                    parts.push(WireValue::from(json!({"type":"image","mime_type":media.attachment.media_type,"data":base64::engine::general_purpose::STANDARD.encode(media.bytes)})));
                }
                ContentBlock::Document {
                    source: DocumentSource::Base64 { media_type, data },
                    ..
                } => parts.push(WireValue::from(
                    json!({"type":"document","mime_type":media_type,"data":data}),
                )),
                ContentBlock::Document {
                    source: DocumentSource::Text { data, .. },
                    ..
                } => parts.push(WireValue::from(json!({"type":"text","text":data}))),
                ContentBlock::ToolUse {
                    id,
                    name,
                    input,
                    provider_id,
                    thought_signature,
                    ..
                } => {
                    flush_parts(&mut steps, &mut parts, message.role);
                    let mut step = WireValue::from(json!({"type":"function_call","id":provider_id.as_deref().unwrap_or(id.as_str()),"name":name,"arguments":null}))
                        .with("arguments", WireValue::raw_json(input));
                    if let Some(signature) = thought_signature {
                        step = step.with("signature", WireValue::text(signature));
                    }
                    steps.push(step);
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    blocks,
                    is_error,
                    ..
                } => {
                    flush_parts(&mut steps, &mut parts, message.role);
                    let (name,id) = calls.get(tool_use_id.as_str()).ok_or_else(|| invalid("Interactions function result requires its paired call name and provider id"))?;
                    let result = if let Some(blocks) = blocks {
                        blocks.validate_root_array("Interactions function result blocks")?;
                        blocks
                            .array_items()
                            .ok_or_else(|| {
                                invalid("Interactions function result blocks are malformed")
                            })?
                            .iter()
                            .map(result_block)
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        content.validate_root_string("Interactions function result content")?;
                        let mut units = content.root_string_code_units().ok_or_else(|| {
                            invalid("Interactions function result content must be a string")
                        })?;
                        if *is_error == Some(true) {
                            let mut prefixed = "Tool failed: ".encode_utf16().collect::<Vec<_>>();
                            prefixed.append(&mut units);
                            units = prefixed;
                        }
                        let content = text_payload(&units);
                        vec![WireValue::from(json!({"type":"text","text":null}))
                            .with("text", WireValue::owned_raw_json(content))]
                    };
                    steps.push(
                        WireValue::from(
                            json!({"type":"function_result","name":name,"call_id":id,"result":[]}),
                        )
                        .with("result", WireValue::array(result)),
                    );
                }
                ContentBlock::Native { value }
                    if matches!(value.format(), CALL_FORMAT | RESULT_FORMAT) =>
                {
                    flush_parts(&mut steps, &mut parts, message.role);
                    let expected = if value.format() == CALL_FORMAT {
                        "function_call"
                    } else {
                        "function_result"
                    };
                    if value.data().string_at("/type").as_deref() != Some(expected) {
                        return Err(invalid(
                            "Interactions native step type differs from extension format",
                        ));
                    }
                    if let Some(scoped) = value.data().payload_at("/_sdk_continuation") {
                        let reference: ContinuationRef =
                            serde_json::from_str(scoped.as_json_source()).map_err(|error| {
                                invalid(format!("invalid receipt continuation: {error}"))
                            })?;
                        if req.continuation.as_ref() != Some(&reference) {
                            return Err(invalid(
                                "Gemini native receipt does not match its originating interaction continuation",
                            ));
                        }
                        steps.push(WireValue::owned_raw_json(without_root_field(
                            value.data(),
                            "_sdk_continuation",
                        )?));
                    } else {
                        steps.push(WireValue::raw_json(value.data()));
                    }
                }
                ContentBlock::ProviderContent {
                    protocol: ProtocolFamily::GeminiInteractions,
                    value,
                } => {
                    flush_parts(&mut steps, &mut parts, message.role);
                    steps.push(WireValue::raw_json(value));
                }
                _ => {
                    return Err(unsupported(
                        "content block has no lossless Interactions representation",
                    ));
                }
            }
        }
        flush_parts(&mut steps, &mut parts, message.role);
    }
    let mut request = InteractionRequest::model(
        context.request_model(),
        InteractionInput::steps(std::iter::repeat_n(Value::Null, steps.len())),
    );
    for tool in &req.tools {
        tool.description.validate_root_string("tool description")?;
    }
    request.tools = req
        .tools
        .iter()
        // This typed helper creates only the generated request shell; final wire tools retain source.
        .map(|tool| InteractionTool::function(&tool.name, "", Value::Null))
        .collect();
    let mut wire_tools = req
        .tools
        .iter()
        .map(|tool| {
            WireValue::from(
                json!({"type":"function","name":tool.name,"description":null,"parameters":null}),
            )
            .with("description", WireValue::raw_json(&tool.description))
            .with("parameters", WireValue::raw_json(&tool.input_schema))
        })
        .collect::<Vec<_>>();
    for extension in &req.native_options {
        let native_tool = extension.decode::<GeminiComputerToolConfig>()?.tool();
        request
            .tools
            .push(InteractionTool::from_value(native_tool.clone())?);
        wire_tools.push(WireValue::from(native_tool));
    }
    let mut generation = json!({});
    if let Some(value) = req.max_tokens {
        generation["max_output_tokens"] = json!(value);
    }
    if let Some(value) = req.temperature {
        if !value.is_finite() {
            return Err(invalid("temperature must be finite"));
        }
        generation["temperature"] = json!(value);
    }
    if let Some(value) = req.controls.top_p {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(invalid("top_p must be between zero and one"));
        }
        generation["top_p"] = json!(value);
    }
    if !req.stop_sequences.is_empty() {
        generation["stop_sequences"] = json!(req.stop_sequences);
    }
    if let Some(effort) = req.thinking.as_ref().and_then(|thinking| thinking.effort) {
        generation["thinking_level"] = json!(effort);
    }
    match &req.tool_choice {
        ToolChoice::Auto => {}
        ToolChoice::None => generation["tool_choice"] = json!("none"),
        ToolChoice::Any => generation["tool_choice"] = json!("any"),
        ToolChoice::Tool { name } => {
            generation["tool_choice"] = json!({"allowed_tools":{"mode":"any","tools":[name]}})
        }
    }
    if !generation.as_object().unwrap().is_empty() {
        request.generation_config = Some(generation);
    }
    let encoded = crate::providers::google::interactions::encode_create_body(
        &request,
        context.mode() == RequestMode::Stream,
    )
    .map_err(|error| match error {
        crate::providers::google::interactions::InteractionError::Llm(error) => error,
        other => invalid(other.to_string()),
    })?;
    let mut body: Value =
        serde_json::from_slice(&encoded).map_err(|error| invalid(error.to_string()))?;
    if let Some(reference) = &req.continuation {
        body["previous_interaction_id"] = json!(reference.response_id);
    }
    let mut system_texts = req
        .system
        .iter()
        .map(|block| {
            block
                .text_payload()
                .root_string_code_units()
                .expect("SystemBlock carries a JSON string")
        })
        .collect::<Vec<_>>();
    system_texts.extend(
        req.messages
            .iter()
            .filter(|message| message.role == MessageRole::System)
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. } => Some(text.encode_utf16().collect()),
                ContentBlock::TextJsUtf16 {
                    utf16_code_units, ..
                } => Some(utf16_code_units.clone()),
                _ => None,
            }),
    );
    let system_payload = if system_texts.is_empty() {
        None
    } else {
        let mut units = Vec::new();
        for (index, text) in system_texts.into_iter().enumerate() {
            if index > 0 {
                units.push(b'\n' as u16);
            }
            units.extend(text);
        }
        if units.is_empty() {
            None
        } else {
            let display = String::from_utf16_lossy(&units);
            body["system_instruction"] = json!(display);
            Some(text_payload(&units))
        }
    };
    if let Some(extra) = context.profile().extra["body"].as_object() {
        for (key, value) in extra {
            if body.get(key).is_some_and(|old| old != value) {
                return Err(invalid(format!(
                    "extra.body conflicts with Interactions field {key}"
                )));
            }
            body[key] = value.clone();
        }
    }
    let mut wire_body = WireValue::from(body).with(
        "input",
        WireValue::from(json!({"steps":[]})).with("steps", WireValue::array(steps)),
    );
    if !wire_tools.is_empty() {
        wire_body = wire_body.with("tools", WireValue::array(std::mem::take(&mut wire_tools)));
    }
    if let Some(system_payload) = system_payload.as_ref() {
        wire_body = wire_body.with("system_instruction", WireValue::raw_json(system_payload));
    }
    let base = context.profile().base_url.trim_end_matches('/');
    let request = WireRequest::new(
        if base.ends_with("/interactions") {
            base.into()
        } else {
            format!("{base}/interactions")
        },
        vec![("content-type".into(), "application/json".into())],
        wire_body,
    );
    if request.body_len()? > 100 * 1024 * 1024 {
        return Err(LlmError::RequestTooLarge {
            message: "Interactions request exceeds 100 MiB".into(),
        });
    }
    request.encode()
}
