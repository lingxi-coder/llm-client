use crate::codecs::{CodecContext, EncodeRequest, RequestMode};
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

pub(super) fn validate(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
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
        return Err(unsupported("Interactions chat currently supports text output, function tools and native desktop computer declarations"));
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
        return Err(unsupported("Interactions thinking budgets, explicit thinking mode and service tier have no unified mapping"));
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
        return Err(invalid("desktop tool members and computer management functions must be declared in different turns"));
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
    let mut steps = vec![];
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
        let mut parts = vec![];
        let flush = |steps: &mut Vec<Value>, parts: &mut Vec<Value>| {
            if !parts.is_empty() {
                steps.push(json!({"type":if message.role == MessageRole::Assistant {"model_output"} else {"user_input"},"content":std::mem::take(parts)}));
            }
        };
        for original in &message.content {
            let block = input.block(original);
            match block {
                ContentBlock::Text {
                    text,
                    thought_signature: None,
                    citations: None,
                } => parts.push(json!({"type":"text","text":text})),
                ContentBlock::Image {
                    source: ImageSource::Base64 { media_type, data },
                } => parts.push(json!({"type":"image","mime_type":media_type,"data":data})),
                ContentBlock::Image {
                    source: ImageSource::Url { url },
                } => parts.push(json!({"type":"image","uri":url})),
                ContentBlock::Image {
                    source: ImageSource::Attachment { .. },
                } => {
                    let media = input
                        .inline_media(block)?
                        .ok_or_else(crate::codecs::unresolved_attachment_error)?;
                    parts.push(json!({"type":"image","mime_type":media.attachment.media_type,"data":base64::engine::general_purpose::STANDARD.encode(media.bytes)}));
                }
                ContentBlock::Document {
                    source: DocumentSource::Base64 { media_type, data },
                    ..
                } => parts.push(json!({"type":"document","mime_type":media_type,"data":data})),
                ContentBlock::Document {
                    source: DocumentSource::Text { data, .. },
                    ..
                } => parts.push(json!({"type":"text","text":data})),
                ContentBlock::ToolUse {
                    id,
                    name,
                    input,
                    provider_id,
                    thought_signature,
                    ..
                } => {
                    flush(&mut steps, &mut parts);
                    let mut step = json!({"type":"function_call","id":provider_id.as_deref().unwrap_or(id.as_str()),"name":name,"arguments":input});
                    if let Some(signature) = thought_signature {
                        step["signature"] = json!(signature);
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
                    flush(&mut steps, &mut parts);
                    let (name,id) = calls.get(tool_use_id.as_str()).ok_or_else(|| invalid("Interactions function result requires its paired call name and provider id"))?;
                    let result = blocks.clone().unwrap_or_else(||vec![json!({"type":"text","text":if *is_error == Some(true) {format!("Tool failed: {content}")} else {content.clone()}})]);
                    let result = result
                        .iter()
                        .map(crate::providers::google::interactions::encode_result_block)
                        .collect::<Result<Vec<_>, _>>()?;
                    steps.push(
                        json!({"type":"function_result","name":name,"call_id":id,"result":result}),
                    );
                }
                ContentBlock::Native { value }
                    if matches!(value.format(), CALL_FORMAT | RESULT_FORMAT) =>
                {
                    flush(&mut steps, &mut parts);
                    let expected = if value.format() == CALL_FORMAT {
                        "function_call"
                    } else {
                        "function_result"
                    };
                    if value.data()["type"] != expected {
                        return Err(invalid(
                            "Interactions native step type differs from extension format",
                        ));
                    }
                    let mut step = value.data().clone();
                    if let Some(scoped) = step
                        .as_object_mut()
                        .and_then(|object| object.remove("_sdk_continuation"))
                    {
                        let reference: ContinuationRef =
                            serde_json::from_value(scoped).map_err(|error| {
                                invalid(format!("invalid receipt continuation: {error}"))
                            })?;
                        if req.continuation.as_ref() != Some(&reference) {
                            return Err(invalid("Gemini native receipt does not match its originating interaction continuation"));
                        }
                    }
                    steps.push(step);
                }
                ContentBlock::ProviderContent {
                    protocol: ProtocolFamily::GeminiInteractions,
                    value,
                } => {
                    flush(&mut steps, &mut parts);
                    steps.push(value.clone());
                }
                _ => {
                    return Err(unsupported(
                        "content block has no lossless Interactions representation",
                    ))
                }
            }
        }
        flush(&mut steps, &mut parts);
    }
    let mut request =
        InteractionRequest::model(context.request_model(), InteractionInput::steps(steps));
    request.tools = req
        .tools
        .iter()
        .map(|tool| {
            InteractionTool::function(&tool.name, &tool.description, tool.input_schema.clone())
        })
        .collect();
    for extension in &req.native_options {
        request.tools.push(InteractionTool::from_value(
            extension.decode::<GeminiComputerToolConfig>()?.tool(),
        )?);
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
    let system = req
        .system
        .iter()
        .map(|block| block.text.as_str())
        .chain(
            req.messages
                .iter()
                .filter(|message| message.role == MessageRole::System)
                .flat_map(|message| message.content.iter())
                .filter_map(|block| {
                    if let ContentBlock::Text { text, .. } = block {
                        Some(text.as_str())
                    } else {
                        None
                    }
                }),
        )
        .collect::<Vec<_>>()
        .join("\n");
    if !system.is_empty() {
        body["system_instruction"] = json!(system);
    }
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
    let encoded = serde_json::to_vec(&body).map_err(|error| invalid(error.to_string()))?;
    if encoded.len() > 100 * 1024 * 1024 {
        return Err(LlmError::RequestTooLarge {
            message: "Interactions request exceeds 100 MiB".into(),
        });
    }
    let base = context.profile().base_url.trim_end_matches('/');
    Ok(HttpRequest {
        method: "POST".into(),
        url: if base.ends_with("/interactions") {
            base.into()
        } else {
            format!("{base}/interactions")
        },
        headers: vec![("content-type".into(), "application/json".into())],
        body: encoded.into(),
        timeout: None,
    })
}
