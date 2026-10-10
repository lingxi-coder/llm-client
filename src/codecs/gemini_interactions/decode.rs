use crate::codecs::CodecContext;
use crate::protocol::*;
use serde_json::Value;

pub(crate) fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::ProviderInternal {
        message: message.into(),
    }
}
fn required_source(value: &JsonPayload, key: &str) -> Result<String, LlmError> {
    value
        .string_units_at(&format!("/{key}"))
        .and_then(|units| String::from_utf16(&units).ok())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid(format!("interaction step is missing {key}")))
}

pub(crate) fn usage_report(raw: Option<&Value>) -> UsageReport {
    const SHAPE: crate::codecs::usage::ReportShape = crate::codecs::usage::ReportShape {
        input: "total_input_tokens",
        output: "total_output_tokens",
        total: &["total_tokens"],
        subsets: &[("/total_cached_tokens", true)],
        independent_counters: &["total_tool_use_tokens"],
        thoughts: Some("total_thought_tokens"),
        thoughts_are_extra: true,
        cache_creation: None,
    };
    crate::codecs::usage::report(
        raw,
        &SHAPE,
        |value| {
            let input = value["total_input_tokens"].as_u64().unwrap_or(0);
            let cached = value["total_cached_tokens"].as_u64().unwrap_or(0);
            let thoughts = value["total_thought_tokens"].as_u64().unwrap_or(0);
            Usage {
                input_tokens: input
                    .saturating_sub(cached)
                    .saturating_add(value["total_tool_use_tokens"].as_u64().unwrap_or(0)),
                cache_read_tokens: cached,
                output_tokens: value["total_output_tokens"]
                    .as_u64()
                    .unwrap_or(0)
                    .saturating_add(thoughts),
                reasoning_tokens: thoughts,
                ..Default::default()
            }
        },
        true,
    )
}

pub(crate) fn usage_report_source(raw: Option<&JsonPayload>) -> UsageReport {
    let value = raw.and_then(|payload| serde_json::from_str(payload.as_json_source()).ok());
    usage_report(value.as_ref())
}

pub(crate) fn step_blocks(
    source: &JsonPayload,
    context: &CodecContext,
) -> Result<Vec<ContentBlock>, LlmError> {
    match required_source(source, "type")?.as_str() {
        "function_call" => {
            let id = required_source(source, "id")?;
            let name = required_source(source, "name")?;
            if source.kind_at("/arguments") != Some(crate::response_json::FlatJsonKind::Object) {
                return Err(invalid("interaction function arguments must be an object"));
            }
            if context.native_options().iter().any(|option| {
                option.is::<crate::providers::google::computer::GeminiComputerToolConfig>()
            }) && crate::providers::google::computer::is_desktop_function(&name)
            {
                for option in context.native_options().iter().filter(|option| {
                    option.is::<crate::providers::google::computer::GeminiComputerToolConfig>()
                }) {
                    let config = option
                        .decode::<crate::providers::google::computer::GeminiComputerToolConfig>(
                    )?;
                    config.validate()?;
                    if config
                        .excluded_predefined_functions
                        .iter()
                        .any(|excluded| excluded == &name)
                    {
                        return Err(invalid("interaction called an excluded desktop function"));
                    }
                }
                return Ok(vec![ContentBlock::Native {
                    value: NativeExtension::new(
                        crate::providers::google::computer::CALL_FORMAT,
                        source.clone(),
                    )?,
                }]);
            }
            Ok(vec![ContentBlock::ToolUse {
                id: id.clone().into(),
                name,
                input: source
                    .payload_at("/arguments")
                    .ok_or_else(|| invalid("interaction function arguments are missing"))?,
                provider_id: Some(id),
                caller: None,
                toolset_name: None,
                thought_signature: source.string_at("/signature"),
            }])
        }
        "model_output" => {
            let content = source
                .payload_at("/content")
                .and_then(|content| content.array_items())
                .ok_or_else(|| invalid("model_output content must be an array"))?;
            // Preserve annotated and multimodal output as the native step; plain text
            // has a portable representation with identical replay bytes.
            if !source.has_only_fields_at("", &["type", "content"])
                || content.iter().any(|part| {
                    part.string_at("/type").as_deref() != Some("text")
                        || !part.has_only_fields_at("", &["type", "text"])
                        || part.kind_at("/text") != Some(crate::response_json::FlatJsonKind::String)
                })
            {
                return Ok(vec![ContentBlock::ProviderContent {
                    protocol: ProtocolFamily::GeminiInteractions,
                    value: source.clone(),
                }]);
            }
            content
                .iter()
                .map(|part| {
                    let units = part
                        .string_units_at("/text")
                        .ok_or_else(|| invalid("model output text must be a string"))?;
                    let text = part
                        .string_at("/text")
                        .ok_or_else(|| invalid("model output text must be a string"))?;
                    let display_units = text.encode_utf16().collect::<Vec<_>>();
                    Ok(if display_units == units {
                        ContentBlock::Text {
                            text,
                            thought_signature: None,
                            citations: None,
                        }
                    } else {
                        ContentBlock::TextJsUtf16 {
                            text,
                            utf16_code_units: units,
                            thought_signature: None,
                            citations: None,
                        }
                    })
                })
                .collect()
        }
        _ => Ok(vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::GeminiInteractions,
            value: source.clone(),
        }]),
    }
}

pub(crate) fn response(
    source: &JsonPayload,
    context: &CodecContext,
) -> Result<ChatResponse, LlmError> {
    let status = required_source(source, "status")?;
    if !matches!(status.as_str(), "completed" | "requires_action") {
        return Err(invalid(format!(
            "interaction is not executable at status {status}"
        )));
    }
    let id = required_source(source, "id")?;
    let mut content = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    let steps = source
        .payload_at("/steps")
        .and_then(|steps| steps.array_items())
        .ok_or_else(|| invalid("interaction response has no steps array"))?;
    for step_source in steps {
        if step_source.string_at("/type").as_deref() == Some("function_call")
            && !ids.insert(required_source(&step_source, "id")?)
        {
            return Err(invalid("interaction contains duplicate function call ids"));
        }
        content.extend(step_blocks(&step_source, context)?);
    }
    if status == "requires_action" && ids.is_empty() {
        return Err(invalid(
            "interaction requires_action without a function call",
        ));
    }
    Ok(ChatResponse {
        message: ConversationMessage {
            role: MessageRole::Assistant,
            content,
            native_options: vec![],
        },
        stop_reason: if !ids.is_empty() {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        },
        usage: usage_report_source(source.payload_at("/usage").as_ref()),
        native_resume_context_exhausted: None,
        model: source
            .string_units_at("/model")
            .and_then(|units| String::from_utf16(&units).ok())
            .filter(|model| !model.is_empty())
            .unwrap_or_else(|| context.request_model().to_owned()),
        response_id: Some(id.into()),
        continuation: None,
        executed_profile: None,
        response_cache: None,
        inference: Default::default(),
        web_search: None,
        file_search: None,
        native_metadata: vec![NativeExtension::new(
            "google.interactions.metadata.v1",
            source.clone(),
        )?],
    })
}
