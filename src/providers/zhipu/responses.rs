//! Mainland metered GLM Responses; not the international Coding Plan shim.
use crate::protocol::{ChatRequest, JsonPayload, LlmError, ProviderProfile, ToolChoice};
use crate::HttpResponse;

pub(crate) fn normalize_response(
    response: &HttpResponse,
    profile: &ProviderProfile,
) -> Result<Option<HttpResponse>, LlmError> {
    if profile
        .extra
        .get("responses_contract")
        .and_then(serde_json::Value::as_str)
        != Some("glm")
        || !(200..300).contains(&response.status)
    {
        return Ok(None);
    }
    let Some(body) = std::str::from_utf8(&response.body)
        .ok()
        .and_then(|s| JsonPayload::parse(s).ok())
    else {
        return Ok(None);
    };
    let Some(mut output) = body.payload_at("/output").and_then(|v| v.array_items()) else {
        return Ok(None);
    };
    let mut changed = false;
    for item in &mut output {
        if item.string_at("/type").as_deref() != Some("message") {
            continue;
        }
        if let Some(content) = item.payload_at("/content").filter(|v| v.is_root_object()) {
            let content = JsonPayload::parse(&format!("[{}]", content.as_json_source()))
                .map_err(invalid_json)?;
            *item = with_field(item, "content", content)?;
            changed = true;
        }
    }
    if !changed {
        return Ok(None);
    }
    let output = serde_json::to_string(&output).map_err(invalid_json)?;
    let output = JsonPayload::parse(&output).map_err(invalid_json)?;
    let source = with_field(&body, "output", output)?;
    Ok(Some(HttpResponse {
        status: response.status,
        headers: response.headers.clone(),
        body: source.as_json_source().as_bytes().to_vec().into(),
    }))
}
fn with_field(
    source: &JsonPayload,
    key: &str,
    value: JsonPayload,
) -> Result<JsonPayload, LlmError> {
    let scanner = source.scanner()?;
    let key: Vec<_> = key.encode_utf16().collect();
    let mut fields = Vec::new();
    for entry in scanner.root_node().object_entries().unwrap_or_default() {
        if entry.key_units != key {
            fields.push((
                entry.key_units.to_vec(),
                scanner
                    .raw_source_for_node(entry.value)
                    .unwrap_or("null")
                    .to_owned(),
            ));
        }
    }
    fields.push((key, value.as_json_source().to_owned()));
    JsonPayload::object_from_fields(&fields).map_err(invalid_json)
}

fn invalid_json(error: impl std::fmt::Display) -> LlmError {
    LlmError::InvalidRequest {
        message: error.to_string(),
    }
}

pub(crate) fn validate(request: &ChatRequest, profile: &ProviderProfile) -> Result<(), LlmError> {
    if profile
        .extra
        .get("responses_contract")
        .and_then(serde_json::Value::as_str)
        != Some("glm")
    {
        return Ok(());
    }
    if profile.provider_id.as_str() != "zhipu"
        || profile.base_url.trim_end_matches('/') != "https://open.bigmodel.cn/api/v1"
    {
        return Err(LlmError::UnsupportedCapability {
            message: "GLM Responses contract requires the mainland metered endpoint".into(),
        });
    }
    if !matches!(request.tool_choice, ToolChoice::Auto | ToolChoice::None)
        || request
            .temperature
            .is_some_and(|v| !(0.0..=1.0).contains(&v))
        || request
            .controls
            .top_p
            .is_some_and(|v| !(0.01..=1.0).contains(&v))
        || !request.hosted_tools.is_empty()
    {
        return Err(LlmError::UnsupportedCapability { message: "GLM Responses supports auto/none function tools, temperature 0..1 and top_p 0.01..1; hosted tools require a separate adapter".into() });
    }
    if request.continuation.is_some() && request.controls.responses.store != Some(true) {
        return Err(LlmError::InvalidRequest {
            message: "GLM Responses continuation requires explicit store=true".into(),
        });
    }
    Ok(())
}
