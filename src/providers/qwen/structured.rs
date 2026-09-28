//! First-party Qwen structured output constraints.
use crate::codecs::structured::{has_json_prompt_keyword, invalid, unsupported};
use crate::protocol::{ChatRequest, LlmError, OutputFormat, ProtocolFamily, ProviderProfile};
pub(crate) fn validate_output_contract(
    req: &ChatRequest,
    profile: &ProviderProfile,
    request_model: &str,
) -> Result<(), LlmError> {
    if is_qwen_openai_chat_profile(profile) {
        match &req.output_format {
            OutputFormat::JsonObject if !has_json_prompt_keyword(req) => return Err(invalid("Qwen JSON Object mode requires JSON in system or user text")),
            OutputFormat::JsonSchema { strict: true, .. } if !is_qwen_strict_schema_model(request_model) => return Err(unsupported("this client currently verifies strict Qwen JSON Schema output only for qwen3.8-flash and qwen3.8-max")),
            _ => {}
        }
    }
    Ok(())
}
pub(crate) fn is_qwen_strict_schema_model(request_model: &str) -> bool {
    matches!(request_model, "qwen3.8-flash" | "qwen3.8-max")
}

pub(crate) fn is_qwen_openai_chat_profile(profile: &crate::protocol::ProviderProfile) -> bool {
    if profile.provider_id.as_str() != "qwen" || profile.protocol != ProtocolFamily::OpenAiChat {
        return false;
    }
    let Ok(url) = url::Url::parse(&profile.base_url) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    let direct_host = matches!(
        host,
        "dashscope.aliyuncs.com"
            | "dashscope-intl.aliyuncs.com"
            | "dashscope-us.aliyuncs.com"
            | "cn-hongkong.dashscope.aliyuncs.com"
    );
    let workspace_host = [
        ".cn-beijing.maas.aliyuncs.com",
        ".ap-southeast-1.maas.aliyuncs.com",
    ]
    .iter()
    .any(|suffix| host.strip_suffix(suffix).is_some_and(valid_dns_label));
    let path = url.path();
    let path_matches = path == "/compatible-mode/v1" || path == "/compatible-mode/v1/";
    url.scheme() == "https"
        && url.username().is_empty()
        && url.password().is_none()
        && url.port().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && path_matches
        && (direct_host || workspace_host)
}

fn valid_dns_label(label: &str) -> bool {
    !label.is_empty()
        && label.len() <= 63
        && label
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        && label.as_bytes()[0].is_ascii_alphanumeric()
        && label.as_bytes()[label.len() - 1].is_ascii_alphanumeric()
}
