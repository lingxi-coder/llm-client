use crate::protocol::{CompletionRequest, LlmError, ProtocolFamily as P, ResponseFormat, ToolSpec};
use serde_json::{json, Map, Value};

pub(super) fn apply(
    req: &CompletionRequest,
    family: P,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    let controls = &req.controls;
    let claude = matches!(
        family,
        P::AnthropicMessages | P::VertexClaude | P::BedrockClaude | P::FoundryClaude
    );
    let gemini = matches!(family, P::GeminiGenerateContent | P::VertexGemini);
    if let Some(top_p) = controls.top_p {
        if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
            return Err(LlmError::InvalidRequest {
                message: "top_p must be between zero and one".into(),
            });
        }
        if gemini {
            object(body, "generationConfig").insert("topP".into(), json!(top_p));
        } else {
            body.insert("top_p".into(), json!(top_p));
        }
    }
    if !req.metadata.is_null() && !gemini {
        body.insert("metadata".into(), req.metadata.clone());
    }
    if let Some(hint) = &controls.anthropic.context_hint {
        if !claude {
            return Err(LlmError::UnsupportedCapability {
                message: "context_hint requires a Claude protocol".into(),
            });
        }
        body.insert("context_hint".into(), hint.clone());
    }
    let responses = serde_json::to_value(&controls.responses).expect("response controls serialize");
    if let Some(fields) = responses.as_object().filter(|fields| !fields.is_empty()) {
        if family != P::OpenAiResponses {
            return Err(LlmError::UnsupportedCapability {
                message: "Responses controls require the Responses protocol".into(),
            });
        }
        body.extend(fields.clone());
    }
    if let Some(format) = &controls.response_format {
        match (format, family) {
            (ResponseFormat::JsonObject, P::GeminiGenerateContent | P::VertexGemini) => {
                object(body, "generationConfig")
                    .insert("responseMimeType".into(), json!("application/json"));
            }
            (
                ResponseFormat::JsonSchema { schema, .. },
                P::GeminiGenerateContent | P::VertexGemini,
            ) => {
                let generation = object(body, "generationConfig");
                generation.insert("responseMimeType".into(), json!("application/json"));
                generation.insert("responseJsonSchema".into(), schema.clone());
            }
            (ResponseFormat::JsonSchema { schema, .. }, _) if claude => {
                object(body, "output_config").insert(
                    "format".into(),
                    json!({"type":"json_schema", "schema":schema}),
                );
            }
            (ResponseFormat::JsonObject, _) if claude => {
                return Err(LlmError::UnsupportedCapability {
                    message: "Claude structured output requires a JSON schema".into(),
                })
            }
            (_, P::OpenAiResponses) => {
                let format = match format {
                    ResponseFormat::JsonObject => json!({"type":"json_object"}),
                    ResponseFormat::JsonSchema {
                        name,
                        schema,
                        strict,
                    } => json!({"type":"json_schema","name":name,"schema":schema,"strict":strict}),
                };
                object(body, "text").insert("format".into(), format);
            }
            _ => {
                let format = match format {
                    ResponseFormat::JsonObject => json!({"type":"json_object"}),
                    ResponseFormat::JsonSchema {
                        name,
                        schema,
                        strict,
                    } => {
                        json!({"type":"json_schema","json_schema":{"name":name,"schema":schema,"strict":strict}})
                    }
                };
                body.insert("response_format".into(), format);
            }
        }
    }
    Ok(())
}
fn object<'a>(body: &'a mut Map<String, Value>, key: &str) -> &'a mut Map<String, Value> {
    body.entry(key)
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("codec owns this object")
}

pub(super) fn tool_extensions(tool: &ToolSpec, mut value: Value) -> Value {
    if let Some(tool_type) = &tool.tool_type {
        value["type"] = json!(tool_type);
        if tool_type != "function" && !tool_type.starts_with("custom") {
            value.as_object_mut().unwrap().remove("input_schema");
            value.as_object_mut().unwrap().remove("parameters");
        }
    }
    if let Some(defer) = tool.defer_loading {
        value["defer_loading"] = json!(defer);
    }
    if let Some(extra) = tool.extra.as_object() {
        let fields = value.as_object_mut().unwrap();
        for (key, value) in extra {
            fields.entry(key).or_insert_with(|| value.clone());
        }
    }
    value
}
