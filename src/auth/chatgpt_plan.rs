//! The public Responses route authorized by Sign in with ChatGPT plan usage.
//! The host owns OAuth, account selection, scope checks, refresh and storage.
use super::{required, set_header, Authenticator, AUTHORIZATION};
use crate::protocol::{
    AuthStrategy, BillingMode, ChatRequest, ContentBlock, DirectoryRoute, DocumentSource,
    HostedTool, ImageSource, LlmError, ProtocolFamily, ProviderProfile, Secret, ServiceSetting,
};
use crate::transport::HttpRequest;
use async_trait::async_trait;
use serde_json::Value;

pub(crate) fn validate_profile(profile: &ProviderProfile) -> Result<(), String> {
    if profile.provider_id.as_str() != "openai"
        || profile.protocol != ProtocolFamily::OpenAiResponses
        || profile.base_url.trim_end_matches('/') != "https://api.openai.com/v1"
    {
        return Err("ChatGPT plan usage requires the public OpenAI Responses profile at https://api.openai.com/v1".into());
    }
    if profile.supports_websockets {
        return Err("ChatGPT plan usage currently requires HTTP streaming, not WebSocket".into());
    }
    if profile.pricing.billing_mode != BillingMode::Subscription
        || profile.models.iter().any(|model| {
            model
                .billing_mode
                .is_some_and(|billing| billing != BillingMode::Subscription)
        })
    {
        return Err("ChatGPT plan profiles and their models must use subscription billing".into());
    }
    if profile.model_list != DirectoryRoute::NotPublished {
        return Err("ChatGPT plan profiles must use model_list=none; use account-specific plan model discovery".into());
    }
    if [
        matches!(profile.embeddings, ServiceSetting::Enabled(_)),
        matches!(profile.retrieval, ServiceSetting::Enabled(_)),
        matches!(profile.batches, ServiceSetting::Enabled(_)),
        matches!(profile.deferred, ServiceSetting::Enabled(_)),
        matches!(profile.background, ServiceSetting::Enabled(_)),
        matches!(profile.audio, ServiceSetting::Enabled(_)),
        matches!(profile.interactions, ServiceSetting::Enabled(_)),
        matches!(profile.gemini_file_search, ServiceSetting::Enabled(_)),
        matches!(profile.glm_knowledge, ServiceSetting::Enabled(_)),
    ]
    .into_iter()
    .any(|enabled| enabled)
        || !profile.images.routes.is_empty()
        || !profile.images.models.is_empty()
    {
        return Err("ChatGPT plan profiles cannot enable independent provider services".into());
    }
    Ok(())
}

pub(crate) fn validate_request(
    request: &HttpRequest,
    profile: &ProviderProfile,
    expected_model: Option<&str>,
) -> Result<(), LlmError> {
    validate_profile(profile).map_err(|message| LlmError::InvalidRequest { message })?;
    if request.method != "POST" || request.url != "https://api.openai.com/v1/responses" {
        return Err(invalid(
            "ChatGPT plan usage permits only POST /v1/responses",
        ));
    }
    for forbidden in [
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "proxy-authorization",
        "cookie",
        "chatgpt-account-id",
        "x-openai-fedramp",
        "openai-organization",
        "openai-project",
        "x-openai-organization",
        "x-openai-project",
    ] {
        if request
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case(forbidden))
        {
            return Err(invalid(&format!(
                "ChatGPT plan usage cannot send {forbidden} alongside its OAuth token"
            )));
        }
    }
    let body: Value = serde_json::from_slice(&request.body)
        .map_err(|_| invalid("ChatGPT plan usage requires a JSON Responses body"))?;
    let fields = body
        .as_object()
        .ok_or_else(|| invalid("ChatGPT plan usage requires a JSON object"))?;
    if fields.get("store") != Some(&Value::Bool(false))
        || fields.get("stream") != Some(&Value::Bool(true))
    {
        return Err(invalid(
            "ChatGPT plan usage requires store=false and stream=true",
        ));
    }
    let model = fields.get("model").and_then(Value::as_str);
    if !model.is_some_and(|model| !model.trim().is_empty()) {
        return Err(invalid("ChatGPT plan usage requires a model"));
    }
    if expected_model.is_some_and(|expected| model != Some(expected)) {
        return Err(invalid(
            "ChatGPT plan request model differs from the selected and priced model",
        ));
    }
    let input = fields
        .get("input")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("ChatGPT plan usage requires an input array"))?;
    for item in input {
        validate_input_item(item)?;
    }
    for name in [
        "background",
        "conversation",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "moderation",
        "multi_agent",
        "prompt",
        "prompt_cache_retention",
        "safety_identifier",
        "temperature",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
        "previous_response_id",
    ] {
        if fields.contains_key(name) {
            return Err(invalid(&format!(
                "ChatGPT plan usage does not support {name}"
            )));
        }
    }
    if let Some(tools) = fields.get("tools") {
        let tools = tools
            .as_array()
            .ok_or_else(|| invalid("ChatGPT plan usage tools must be an array"))?;
        for tool in tools {
            validate_tool(tool, true)?;
        }
    }
    Ok(())
}

fn validate_input_item(item: &Value) -> Result<(), LlmError> {
    if item.get("role").and_then(Value::as_str) == Some("system") {
        return Err(invalid(
            "put system instructions in instructions or developer messages",
        ));
    }
    match item.get("type").and_then(Value::as_str) {
        Some("additional_tools") => {
            if item.get("role").and_then(Value::as_str) != Some("developer") {
                return Err(invalid(
                    "additional_tools input items require developer role",
                ));
            }
            let tools = item
                .get("tools")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("additional_tools input items require a tools array"))?;
            for tool in tools {
                validate_tool(tool, false)?;
            }
        }
        None if item.get("role").and_then(Value::as_str).is_some() => {}
        Some(
            "message"
            | "reasoning"
            | "function_call"
            | "function_call_output"
            | "custom_tool_call"
            | "custom_tool_call_output"
            | "web_search_call",
        ) => {}
        Some("input_audio" | "input_video" | "audio" | "video") => {
            return Err(invalid(
                "ChatGPT plan usage does not support audio or video input",
            ));
        }
        _ => {
            return Err(invalid(
                "this input item is unavailable for ChatGPT plan usage",
            ));
        }
    }
    if item.get("file_id").is_some() {
        return Err(invalid(
            "ChatGPT plan usage cannot reference a Files API file ID",
        ));
    }
    for field in ["content", "output"] {
        if let Some(parts) = item.get(field).and_then(Value::as_array) {
            for part in parts {
                validate_input_content(part)?;
            }
        }
    }
    Ok(())
}

fn validate_input_content(part: &Value) -> Result<(), LlmError> {
    if matches!(
        part.get("type").and_then(Value::as_str),
        Some("input_audio" | "input_video" | "audio" | "video")
    ) || part.get("input_audio").is_some()
        || part.get("video").is_some()
    {
        return Err(invalid(
            "ChatGPT plan usage does not support audio or video input",
        ));
    }
    if part.get("file_id").is_some() {
        return Err(invalid(
            "ChatGPT plan usage cannot reference a Files API file ID",
        ));
    }
    Ok(())
}

fn validate_tool(tool: &Value, top_level: bool) -> Result<(), LlmError> {
    match tool.get("type").and_then(Value::as_str) {
        Some("web_search") if top_level => Ok(()),
        Some("function" | "custom") if !top_level => Ok(()),
        Some("namespace") => {
            let tools = tool
                .get("tools")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("ChatGPT plan tool namespaces require a tools array"))?;
            for member in tools {
                validate_tool(member, false)?;
            }
            Ok(())
        }
        Some("function" | "custom") => Err(invalid(
            "ChatGPT plan usage requires function and custom tools in a namespace or additional_tools input item",
        )),
        _ => Err(invalid(
            "this hosted tool is unavailable for ChatGPT plan usage",
        )),
    }
}

/// Reject inputs that would otherwise trigger an unsupported upload before
/// the final wire request can be checked by the authenticator.
pub(crate) fn validate_chat_request(request: &ChatRequest) -> Result<(), LlmError> {
    if request.has_response_continuation() {
        return Err(invalid(
            "ChatGPT plan usage requires the full history in input over HTTP",
        ));
    }
    if !request.tools.is_empty() {
        return Err(invalid("ChatGPT plan usage requires function tools in namespaces or additional_tools input items"));
    }
    if request
        .hosted_tools
        .iter()
        .any(|tool| !matches!(tool, HostedTool::WebSearch(_)))
    {
        return Err(invalid(
            "this hosted tool is unavailable for ChatGPT plan usage",
        ));
    }
    for message in &request.messages {
        for block in &message.content {
            match block {
                ContentBlock::Image {
                    source: ImageSource::ProviderFile { .. } | ImageSource::Attachment { .. },
                }
                | ContentBlock::Document {
                    source: DocumentSource::ProviderFile { .. } | DocumentSource::Attachment { .. },
                    ..
                } => {
                    return Err(invalid(
                        "ChatGPT plan usage cannot reference a Files API upload or application attachment",
                    ));
                }
                ContentBlock::Video { .. } | ContentBlock::Audio { .. } => {
                    return Err(invalid(
                        "ChatGPT plan usage does not support audio or video input",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}

/// Validates the restricted public Responses request before adding the
/// host-supplied, already authorized and unexpired OAuth access token.
pub struct ChatGptPlanAuthenticator;

#[async_trait]
impl Authenticator for ChatGptPlanAuthenticator {
    async fn apply(
        &self,
        request: &mut HttpRequest,
        profile: &ProviderProfile,
        credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        if profile.auth != AuthStrategy::ChatGptPlan {
            return Err(invalid(
                "ChatGPT plan authenticator requires chat_gpt_plan auth",
            ));
        }
        validate_request(request, profile, None)?;
        let token = required(profile, credential)?;
        if token.expose_secret().is_empty() {
            return Err(LlmError::Authentication {
                message: "ChatGPT plan usage requires a nonempty access token".into(),
            });
        }
        set_header(
            request,
            AUTHORIZATION,
            format!("Bearer {}", token.expose_secret()),
        );
        Ok(())
    }
}
