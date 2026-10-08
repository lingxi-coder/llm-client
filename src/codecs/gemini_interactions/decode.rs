use crate::codecs::CodecContext;
use crate::protocol::*;
use serde_json::Value;

pub(crate) fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::ProviderInternal {
        message: message.into(),
    }
}
fn required<'a>(value: &'a Value, key: &str) -> Result<&'a str, LlmError> {
    value[key]
        .as_str()
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

pub(crate) fn step_blocks(
    step: &Value,
    context: &CodecContext,
) -> Result<Vec<ContentBlock>, LlmError> {
    match required(step, "type")? {
        "function_call" => {
            let id = required(step, "id")?;
            let name = required(step, "name")?;
            if !step["arguments"].is_object() {
                return Err(invalid("interaction function arguments must be an object"));
            }
            if context.native_options().iter().any(|option| {
                option.is::<crate::providers::google::computer::GeminiComputerToolConfig>()
            }) && crate::providers::google::computer::is_desktop_function(name)
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
                        .any(|excluded| excluded == name)
                    {
                        return Err(invalid("interaction called an excluded desktop function"));
                    }
                }
                return Ok(vec![ContentBlock::Native {
                    value: NativeExtension::new(
                        crate::providers::google::computer::CALL_FORMAT,
                        step.clone(),
                    )?,
                }]);
            }
            Ok(vec![ContentBlock::ToolUse {
                input_json: None,
                id: id.into(),
                name: name.into(),
                input: step["arguments"].clone(),
                provider_id: Some(id.into()),
                caller: None,
                toolset_name: None,
                thought_signature: step["signature"].as_str().map(str::to_owned),
            }])
        }
        "model_output" => {
            let content = step["content"]
                .as_array()
                .ok_or_else(|| invalid("model_output content must be an array"))?;
            // Preserve annotated and multimodal output as the native step; plain text
            // has a portable representation with identical replay bytes.
            if step.as_object().is_some_and(|object| {
                object
                    .keys()
                    .any(|key| !matches!(key.as_str(), "type" | "content"))
            }) || content.iter().any(|part| {
                part["type"] != "text"
                    || part
                        .as_object()
                        .is_some_and(|o| o.keys().any(|k| !matches!(k.as_str(), "type" | "text")))
            }) {
                return Ok(vec![ContentBlock::ProviderContent {
                    protocol: ProtocolFamily::GeminiInteractions,
                    value: step.clone(),
                }]);
            }
            content
                .iter()
                .map(|part| {
                    Ok(ContentBlock::Text {
                        text: part["text"]
                            .as_str()
                            .ok_or_else(|| invalid("model output text must be a string"))?
                            .into(),
                        thought_signature: None,
                        citations: None,
                    })
                })
                .collect()
        }
        _ => Ok(vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::GeminiInteractions,
            value: step.clone(),
        }]),
    }
}

pub(crate) fn response(body: &Value, context: &CodecContext) -> Result<ChatResponse, LlmError> {
    let status = required(body, "status")?;
    if !matches!(status, "completed" | "requires_action") {
        return Err(invalid(format!(
            "interaction is not executable at status {status}"
        )));
    }
    let id = required(body, "id")?;
    let mut content = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    let steps = body["steps"]
        .as_array()
        .ok_or_else(|| invalid("interaction response has no steps array"))?;
    for step in steps {
        if step["type"] == "function_call" && !ids.insert(required(step, "id")?) {
            return Err(invalid("interaction contains duplicate function call ids"));
        }
        content.extend(step_blocks(step, context)?);
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
        usage: usage_report(body.get("usage")),
        model: body["model"]
            .as_str()
            .unwrap_or(context.request_model())
            .into(),
        response_id: Some(id.into()),
        continuation: None,
        executed_profile: None,
        response_cache: None,
        inference: Default::default(),
        web_search: None,
        file_search: None,
        native_metadata: vec![NativeExtension::new(
            "google.interactions.metadata.v1",
            body.clone(),
        )?],
    })
}
