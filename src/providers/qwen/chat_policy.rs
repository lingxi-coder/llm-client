//! Qwen-Long file input policy used by the shared Chat wire codec.
use crate::protocol::{ContentBlock, LlmError, ProviderFileSource, ProviderProfile};
pub(crate) fn validate_long_input<'a>(
    profile: &ProviderProfile,
    model: &str,
    blocks: impl IntoIterator<Item = &'a ContentBlock>,
) -> Result<bool, LlmError> {
    let enabled = profile.provider_id.as_str() == "qwen" && crate::files::is_qwen_long_model(model);
    if enabled {
        if !crate::files::qwen_long_region_supported(profile) {
            return Err(LlmError::UnsupportedCapability {
                message: "Qwen-Long is supported only on a Beijing Qwen endpoint".into(),
            });
        }
        crate::files::validate_qwen_long_blocks(blocks)?;
    }
    Ok(enabled)
}
pub(crate) fn is_extracted_file(profile: &ProviderProfile, file: &ProviderFileSource) -> bool {
    profile.provider_id.as_str() == "qwen" && file.purpose.as_deref() == Some("file-extract")
}
/// Qwen's current Chat API documents a total (thinking + answer) budget for
/// Max 3.7+, Plus/Flash 3.5+, and DeepSeek V4 Pro / V4.1 Flash. Other model
/// families still use the endpoint's documented answer-only `max_tokens`.
pub(crate) fn supports_total_output_budget(
    profile: &crate::protocol::ProviderProfile,
    model: &str,
) -> bool {
    if profile.provider_id.as_str() != "qwen" {
        return false;
    }
    if model.starts_with("deepseek-v4-pro") || model.starts_with("deepseek-v4.1-flash") {
        return true;
    }
    let Some(version_and_family) = model.strip_prefix("qwen") else {
        return false;
    };
    let Some((version, family)) = version_and_family.split_once('-') else {
        return false;
    };
    let Some((major, minor)) = version.split_once('.') else {
        return false;
    };
    let (Ok(major), Ok(minor)) = (major.parse::<u32>(), minor.parse::<u32>()) else {
        return false;
    };
    let minimum = if family == "max" || family.starts_with("max-") {
        (3, 7)
    } else if family == "plus"
        || family.starts_with("plus-")
        || family == "flash"
        || family.starts_with("flash-")
    {
        (3, 5)
    } else {
        return false;
    };
    (major, minor) >= minimum
}
