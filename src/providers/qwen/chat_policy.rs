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
