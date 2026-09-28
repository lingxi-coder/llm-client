//! Google Developer API hosted tool model and request policy.
use super::native::GoogleHostedTool;
use crate::protocol::{
    ChatRequest, ContentBlock, HostedTool, LlmError, ProtocolFamily, ProviderProfile, ToolChoice,
};
use serde_json::{json, Map, Value};

const GEMINI_DEVELOPER_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

const CODE_EXECUTION_MODELS: &[&str] = &[
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
];
const URL_CONTEXT_MODELS: &[&str] = CODE_EXECUTION_MODELS;
const MAPS_GROUNDING_MODELS: &[&str] = CODE_EXECUTION_MODELS;

pub(crate) fn has_gemini_hosted_tools(req: &ChatRequest) -> bool {
    req.hosted_tools
        .iter()
        .any(|tool| tool.native::<GoogleHostedTool>().is_some())
}

/// Check Gemini-hosted tool scope before credential resolution, attachment
/// preparation or HTTP. Only exact Developer API routes are enabled here;
/// Vertex and compatible gateways need their own documented capability table.
pub(crate) fn validate_hosted_tool_request(
    req: &ChatRequest,
    profile: &ProviderProfile,
    model: &str,
) -> Result<(), LlmError> {
    req.validate_hosted_tools()?;
    if !has_gemini_hosted_tools(req) {
        return Ok(());
    }
    let unsupported = |message: &str| LlmError::UnsupportedCapability {
        message: format!("Gemini GenerateContent hosted tools: {message}"),
    };
    if profile.provider_id.as_str() != "google"
        || profile.protocol != ProtocolFamily::GeminiGenerateContent
        || profile.base_url.trim_end_matches('/') != GEMINI_DEVELOPER_API_BASE
    {
        return Err(unsupported(
            "code execution, URL Context and Maps grounding are enabled only for the first-party Gemini Developer API",
        ));
    }

    let mut selected = None;
    let mut count = 0;
    for tool in &req.hosted_tools {
        let candidate = match tool.native::<GoogleHostedTool>() {
            Some(GoogleHostedTool::CodeExecution) => {
                Some(("code execution", CODE_EXECUTION_MODELS))
            }
            Some(GoogleHostedTool::UrlContext) => Some(("URL Context", URL_CONTEXT_MODELS)),
            Some(GoogleHostedTool::MapsGrounding(config)) => {
                config.validate()?;
                Some(("Maps grounding", MAPS_GROUNDING_MODELS))
            }
            None if matches!(tool, HostedTool::WebSearch(_)) => None,
            _ => {
                return Err(unsupported(
                    "this request also contains a hosted tool owned by another provider",
                ));
            }
        };
        if let Some(candidate) = candidate {
            selected = Some(candidate);
            count += 1;
        }
    }
    if count != 1 {
        return Err(unsupported(
            "combining multiple Gemini built-in tools in one request is not established by the supported GenerateContent contracts",
        ));
    }
    let (name, models) = selected.expect("one Gemini hosted tool was counted");
    if !models.contains(&model) {
        return Err(unsupported(&format!(
            "{name} is not documented for model {model:?}"
        )));
    }

    let gemini_3_model = is_gemini_3_model(model);
    if (!req.tools.is_empty() || req.tool_choice != ToolChoice::Auto) && !gemini_3_model {
        return Err(unsupported(
            "combining these built-in tools with client-executed functions requires a documented Gemini 3 model",
        ));
    }
    if gemini_3_model && req.tool_choice != ToolChoice::Auto {
        return Err(unsupported(
            "tool context circulation requires validated function calling and cannot honor a forced tool mode",
        ));
    }
    let extra_body = profile.extra.get("body").and_then(Value::as_object);
    let maps_selected = req.hosted_google_maps_grounding().is_some();
    if maps_selected
        && req
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|block| {
                matches!(
                    block,
                    ContentBlock::Image { .. }
                        | ContentBlock::Audio { .. }
                        | ContentBlock::Video { .. }
                        | ContentBlock::Document { .. }
                ) || matches!(
                    block,
                    ContentBlock::ProviderContent { value, .. }
                        if crate::codecs::gemini::native::is_media_part(value)
                )
            })
    {
        return Err(unsupported(
            "Google Maps grounding currently accepts text-only inputs",
        ));
    }
    if maps_selected {
        if let Some(modalities) = extra_body
            .and_then(|body| body.get("generationConfig"))
            .and_then(Value::as_object)
            .and_then(|generation| generation.get("responseModalities"))
        {
            if !matches!(modalities.as_array(), Some(values) if values.len() == 1 && values[0].as_str() == Some("TEXT"))
            {
                return Err(unsupported(
                    "Google Maps grounding currently supports text-only outputs",
                ));
            }
        }
    }

    if req.hosted_web_search().is_some() {
        if req.hosted_google_maps_grounding().is_some()
            && !matches!(
                model,
                "gemini-3.5-flash" | "gemini-3.6-flash" | "gemini-3.7-flash" | "gemini-3.8-flash"
            )
        {
            return Err(unsupported(
                "Maps grounding with Google Search is documented only for Gemini 3.5 Flash and later models",
            ));
        }
        // Reuse the adapter's full local validation before any side effects.
        crate::codecs::web_search::apply(req, profile, &mut Map::new())?;
    }

    if extra_body.is_some_and(|body| body.contains_key("tools") || body.contains_key("toolConfig"))
    {
        return Err(unsupported(
            "profile extra.body cannot override the typed tools or toolConfig controls",
        ));
    }
    Ok(())
}

fn is_gemini_3_model(model: &str) -> bool {
    matches!(
        model,
        "gemini-3-flash-preview"
            | "gemini-3.1-flash-lite"
            | "gemini-3.1-pro-preview"
            | "gemini-3.5-flash"
            | "gemini-3.5-flash-lite"
            | "gemini-3.6-flash"
            | "gemini-3.7-flash"
            | "gemini-3.8-flash"
    )
}

pub(crate) fn apply_gemini_hosted_tools(
    req: &ChatRequest,
    model: &str,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    if !has_gemini_hosted_tools(req) {
        return Ok(());
    }
    for hosted in &req.hosted_tools {
        let tool = match hosted.native::<GoogleHostedTool>() {
            Some(GoogleHostedTool::CodeExecution) => Some(json!({"codeExecution": {}})),
            Some(GoogleHostedTool::UrlContext) => Some(json!({"urlContext": {}})),
            Some(GoogleHostedTool::MapsGrounding(config)) => {
                let mut maps = json!({});
                if let Some(enable_widget) = config.enable_widget {
                    maps["enableWidget"] = json!(enable_widget);
                }
                if let Some(location) = config.lat_lng {
                    let tool_config = body
                        .entry("toolConfig")
                        .or_insert_with(|| json!({}))
                        .as_object_mut()
                        .ok_or_else(|| LlmError::InvalidRequest {
                            message: "Gemini toolConfig must be a JSON object".into(),
                        })?;
                    tool_config.insert(
                        "retrievalConfig".into(),
                        json!({
                            "latLng": {
                                "latitude": location.latitude,
                                "longitude": location.longitude
                            }
                        }),
                    );
                }
                Some(json!({"googleMaps": maps}))
            }
            _ => None,
        };
        if let Some(tool) = tool {
            body.entry("tools")
                .or_insert_with(|| json!([]))
                .as_array_mut()
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "Gemini tools must be a JSON array".into(),
                })?
                .push(tool);
        }
    }

    if is_gemini_3_model(model) {
        let tool_config = body
            .entry("toolConfig")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Gemini toolConfig must be a JSON object".into(),
            })?;
        tool_config.insert("includeServerSideToolInvocations".into(), json!(true));
        tool_config
            .entry("functionCallingConfig")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Gemini functionCallingConfig must be a JSON object".into(),
            })?
            .insert("mode".into(), json!("VALIDATED"));
    }
    Ok(())
}
