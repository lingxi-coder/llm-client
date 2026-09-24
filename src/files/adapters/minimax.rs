//! MiniMax purpose and native status contracts.
use super::super::*;
pub(crate) fn minimax_files_root(profile: &ProviderProfile) -> String {
    let url = url::Url::parse(&profile.base_url).expect("adapter validates a URL");
    format!(
        "{}://{}/v1/files",
        url.scheme(),
        url.host_str().unwrap_or_default()
    )
}

pub(crate) fn check_minimax_base_response(value: &Value, operation: &str) -> Result<(), LlmError> {
    let base = value
        .get("base_resp")
        .ok_or_else(|| provider_shape(&format!("{operation} response omitted base_resp")))?;
    let code = base
        .get("status_code")
        .and_then(|value| value.as_i64().or_else(|| value.as_str()?.parse().ok()))
        .ok_or_else(|| {
            provider_shape(&format!(
                "{operation} response has invalid base_resp.status_code"
            ))
        })?;
    if code == 0 {
        Ok(())
    } else {
        let message = base
            .get("status_msg")
            .and_then(Value::as_str)
            .unwrap_or("provider returned an error")
            .chars()
            .take(512)
            .collect::<String>();
        Err(LlmError::ProviderInternal {
            message: format!("{operation} failed with MiniMax status {code}: {message}"),
        })
    }
}
pub(crate) fn minimax_list_purpose(purpose: FilePurpose) -> Option<&'static str> {
    match purpose {
        FilePurpose::VoiceClone => Some("voice_clone"),
        FilePurpose::PromptAudio => Some("prompt_audio"),
        FilePurpose::AsyncTtsInput => Some("t2a_async_input"),
        FilePurpose::VideoGenerationInput => Some("video_generation_input"),
        _ => None,
    }
}

pub(crate) fn minimax_delete_purpose(purpose: &str) -> Option<&'static str> {
    match purpose {
        "voice_clone" => Some("voice_clone"),
        "prompt_audio" => Some("prompt_audio"),
        "t2a_async" => Some("t2a_async"),
        "t2a_async_input" => Some("t2a_async_input"),
        "video_generation_input" | "video_generation" => Some("video_generation"),
        _ => None,
    }
}
