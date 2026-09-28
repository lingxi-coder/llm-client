//! Qwen regional file-search route and model capability.
use crate::protocol::{ChatRequest, LlmError, ProviderProfile};
use serde_json::{json, Map, Value};

pub(crate) fn file_search_url(
    profile: &ProviderProfile,
    workspace_id: &str,
) -> Result<String, LlmError> {
    let host = url::Url::parse(&profile.base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned));
    let region_domain = match (profile.provider_id.as_str(), host.as_deref()) {
        ("qwen", Some("dashscope.aliyuncs.com")) => "cn-beijing.maas.aliyuncs.com",
        ("qwen", Some("dashscope-intl.aliyuncs.com")) => "ap-southeast-1.maas.aliyuncs.com",
        ("qwen", Some("dashscope-us.aliyuncs.com")) => "us-east-1.maas.aliyuncs.com",
        ("qwen", Some("cn-hongkong.dashscope.aliyuncs.com")) => "cn-hongkong.maas.aliyuncs.com",
        _ => {
            return Err(LlmError::UnsupportedCapability {
                message: "Qwen file search requires a supported regional Model Studio profile"
                    .into(),
            });
        }
    };
    Ok(format!(
        "https://{workspace_id}.{region_domain}/compatible-mode/v1/responses"
    ))
}

pub(crate) fn apply_file_search(
    req: &ChatRequest,
    profile: &ProviderProfile,
    model: &str,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    if let Some(search) = req.hosted_file_search() {
        if search.knowledge_base_id.trim().is_empty() {
            return Err(LlmError::InvalidRequest {
                message: "Qwen file search requires a nonempty knowledge_base_id".into(),
            });
        }
        if !search
            .workspace_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            || search.workspace_id.is_empty()
        {
            return Err(LlmError::InvalidRequest {
                message:
                    "Qwen file search workspace_id must contain only letters, digits, or hyphens"
                        .into(),
            });
        }
        if profile.provider_id.as_str() != "qwen"
            || !matches!(model, "qwen3.8-max" | "qwen3.8-flash")
        {
            return Err(LlmError::UnsupportedCapability {
                message: "Qwen file search is available only for the supported Max and Flash Responses models".into(),
            });
        }
        body.entry("tools")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Qwen file search requires tools to be an array".into(),
            })?
            .push(json!({"type":"file_search","vector_store_ids":[search.knowledge_base_id]}));
    }
    Ok(())
}

pub(crate) fn validate_file_search_route(
    req: &ChatRequest,
    profile: &ProviderProfile,
) -> Result<(), LlmError> {
    if req.hosted_file_search().is_some()
        && profile.extra.get("file_search").and_then(Value::as_str) != Some("qwen")
    {
        return Err(LlmError::UnsupportedCapability {
            message: format!(
                "profile {:?} does not declare Qwen file search",
                profile.profile_name
            ),
        });
    }
    Ok(())
}
