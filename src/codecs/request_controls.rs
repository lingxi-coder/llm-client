use crate::codecs::json::WireValue;
use crate::protocol::{
    ChatRequest, JsonPayload, LlmError, ProtocolFamily as P, ResponseFormat, ToolSpec,
};
use serde_json::{json, Map, Value};

pub(super) fn apply(
    req: &ChatRequest,
    family: P,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    if family != P::OpenAiResponses
        && req
            .native_options
            .iter()
            .any(|value| value.is::<crate::providers::openai::codex_search::CodexSearchRequest>())
    {
        return Err(LlmError::UnsupportedCapability {
            message: "Codex standalone search requires the Responses protocol".into(),
        });
    }
    let controls = &req.controls;
    if let (Some(previous), Some(scoped)) =
        (&controls.responses.previous_response_id, &req.continuation)
    {
        if previous != scoped.response_id.as_str() {
            return Err(LlmError::InvalidRequest {
                message: "conflicting explicit and scoped response continuation".into(),
            });
        }
    }
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
    if !req.metadata.is_root_null() && !gemini {
        body.insert("metadata".into(), Value::Null);
    }
    if controls.anthropic.context_hint.is_some() {
        if !claude {
            return Err(LlmError::UnsupportedCapability {
                message: "context_hint requires a Claude protocol".into(),
            });
        }
        body.insert("context_hint".into(), Value::Null);
    }
    let responses = serde_json::to_value(&controls.responses).expect("response controls serialize");
    if let Some(fields) = responses.as_object().filter(|fields| !fields.is_empty()) {
        if family != P::OpenAiResponses {
            return Err(LlmError::UnsupportedCapability {
                message: "Responses controls require the Responses protocol".into(),
            });
        }
        for (key, value) in fields {
            if key == "include" {
                let target = body.entry(key.clone()).or_insert_with(|| json!([]));
                let target = target
                    .as_array_mut()
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: "include must be an array".into(),
                    })?;
                for item in value.as_array().expect("typed include array") {
                    if !target.contains(item) {
                        target.push(item.clone());
                    }
                }
            } else if let Some(existing) = body.get(key) {
                if existing != value {
                    return Err(LlmError::InvalidRequest {
                        message: format!("request controls conflict with typed field {key}"),
                    });
                }
            } else {
                body.insert(key.clone(), value.clone());
            }
        }
    }
    if let Some(format) = &controls.response_format {
        match (format, family) {
            (ResponseFormat::JsonObject, P::GeminiGenerateContent | P::VertexGemini) => {
                object(body, "generationConfig")
                    .insert("responseMimeType".into(), json!("application/json"));
            }
            (ResponseFormat::JsonSchema { .. }, P::GeminiGenerateContent | P::VertexGemini) => {
                let generation = object(body, "generationConfig");
                generation.insert("responseMimeType".into(), json!("application/json"));
                generation.insert("responseJsonSchema".into(), Value::Null);
            }
            (ResponseFormat::JsonSchema { .. }, _) if claude => {
                object(body, "output_config").insert(
                    "format".into(),
                    json!({"type":"json_schema", "schema":Value::Null}),
                );
            }
            (ResponseFormat::JsonObject, _) if claude => {
                return Err(LlmError::UnsupportedCapability {
                    message: "Claude structured output requires a JSON schema".into(),
                });
            }
            (_, P::OpenAiResponses) => {
                let format = match format {
                    ResponseFormat::JsonObject => json!({"type":"json_object"}),
                    ResponseFormat::JsonSchema { name, strict, .. } => {
                        json!({"type":"json_schema","name":name,"schema":null,"strict":strict})
                    }
                };
                object(body, "text").insert("format".into(), format);
            }
            _ => {
                let format = match format {
                    ResponseFormat::JsonObject => json!({"type":"json_object"}),
                    ResponseFormat::JsonSchema { name, strict, .. } => {
                        json!({"type":"json_schema","json_schema":{"name":name,"schema":null,"strict":strict}})
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

pub(super) fn wire_body<'a>(
    req: &'a ChatRequest,
    family: P,
    body: Map<String, Value>,
) -> Result<WireValue<'a>, LlmError> {
    for tool in &req.tools {
        tool.description.validate_root_string("tool description")?;
    }
    let body = Value::Object(body);
    let mut wire = WireValue::from(body.clone());
    let gemini = matches!(family, P::GeminiGenerateContent | P::VertexGemini);
    if !gemini && !req.metadata.is_root_null() && body.get("metadata").is_some() {
        wire = wire.with("metadata", WireValue::raw_json(&req.metadata));
    }
    if body.get("context_hint").is_some() {
        if let Some(hint) = req.controls.anthropic.context_hint.as_ref() {
            wire = wire.with_json_path(&["context_hint"], hint)?;
        }
    }
    if let Some(ResponseFormat::JsonSchema { schema, .. }) = req.controls.response_format.as_ref() {
        let path = match family {
            P::AnthropicMessages | P::VertexClaude | P::BedrockClaude | P::FoundryClaude => {
                &["output_config", "format", "schema"][..]
            }
            P::GeminiGenerateContent | P::VertexGemini => {
                &["generationConfig", "responseJsonSchema"][..]
            }
            P::OpenAiResponses => &["text", "format", "schema"][..],
            _ => &["response_format", "json_schema", "schema"][..],
        };
        if has_value_path(&body, path) {
            wire = wire.with_json_path(path, schema)?;
        }
    }
    if let crate::protocol::OutputFormat::JsonSchema { schema, .. } = &req.output_format {
        let path = match family {
            P::AnthropicMessages | P::VertexClaude | P::BedrockClaude | P::FoundryClaude => {
                &["output_config", "format", "schema"][..]
            }
            P::GeminiGenerateContent | P::VertexGemini => {
                &["generationConfig", "responseFormat", "text", "schema"][..]
            }
            P::OpenAiResponses => &["text", "format", "schema"][..],
            _ => &["response_format", "json_schema", "schema"][..],
        };
        if has_value_path(&body, path) {
            wire = wire.with_json_path(path, schema)?;
        }
    }
    Ok(wire)
}

fn has_value_path(mut value: &Value, path: &[&str]) -> bool {
    for field in path {
        let Some(child) = value.get(*field) else {
            return false;
        };
        value = child;
    }
    true
}

pub(super) fn tool_extensions(
    tool: &ToolSpec,
    mut value: Value,
    encoding: crate::exact_json::JsonEncoding,
) -> Result<JsonPayload, LlmError> {
    if let Some(tool_type) = &tool.tool_type {
        value["type"] = json!(tool_type);
        if tool_type != "function" && !tool_type.starts_with("custom") {
            value.as_object_mut().unwrap().remove("input_schema");
            value.as_object_mut().unwrap().remove("parameters");
        }
    }
    if tool.defer_loading {
        value["defer_loading"] = json!(true);
    }
    let mut payload =
        JsonPayload::from_generated_value(&value).map_err(|error| LlmError::InvalidRequest {
            message: format!("tool definition is not valid JSON: {error}"),
        })?;
    payload = payload.with_canonical_fields(
        encoding,
        &[("description", tool.description.as_json_source())],
    )?;
    if tool.extra.is_root_object() {
        payload = payload.with_missing_object_fields(&tool.extra, encoding)?;
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ToolSpec;

    #[test]
    fn provider_tool_extras_keep_raw_values_and_do_not_replace_typed_fields() {
        let nested = format!("{}0{}", "[".repeat(512), "]".repeat(512));
        let tool = ToolSpec {
            name: "read".into(),
            description: crate::protocol::JsonPayload::string("Read"),
            input_schema: JsonPayload::parse(r#"{"type":"object"}"#).unwrap(),
            strict: false,
            defer_loading: false,
            native_options: Vec::new(),
            tool_type: None,
            extra: JsonPayload::parse(&format!(
                r#"{{"type":"provider","future":{nested},"lone":"\ud800"}}"#
            ))
            .unwrap(),
        };
        let payload = tool_extensions(
            &tool,
            json!({"type":"function","name":"read","parameters":null}),
            crate::exact_json::JsonEncoding::JavaScript,
        )
        .unwrap();
        let source = payload
            .canonical_json(crate::exact_json::JsonEncoding::JavaScript)
            .unwrap();
        assert!(source.starts_with(r#"{"type":"function""#));
        assert!(source.contains(&format!(r#""future":{nested}"#)));
        assert!(source.contains(r#""lone":"\ud800""#));
        assert!(source.contains(r#""parameters":null"#));
    }
}
