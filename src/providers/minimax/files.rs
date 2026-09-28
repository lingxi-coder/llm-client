//! Provider-owned file workflow and capability policy.
use crate::files::*;

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

const MINIMAX_VOICE_AUDIO_MAX_UPLOAD_BYTES: u64 = 20_000_000;

pub(crate) fn capabilities() -> FileCapabilities {
    FileCapabilities {
        upload: true,
        retrieve_metadata: true,
        list: true,
        delete: true,
        download: DownloadSupport::GeneratedFilesOnly,
        extract_text: false,
        model_input: ModelFileReference::Unsupported,
        max_upload_bytes: None,
        retention: None,
    }
}

pub(crate) fn minimax_base_response_code(value: &Value) -> Option<i64> {
    value
        .get("base_resp")?
        .get("status_code")
        .and_then(|status| status.as_i64().or_else(|| status.as_str()?.parse().ok()))
}

pub(crate) fn purpose_name(purpose: FilePurpose) -> Option<&'static str> {
    match purpose {
        FilePurpose::VoiceClone => Some("voice_clone"),
        FilePurpose::PromptAudio => Some("prompt_audio"),
        FilePurpose::AsyncTtsInput => Some("t2a_async_input"),
        FilePurpose::VideoUnderstanding => Some("video_understanding"),
        FilePurpose::VideoGenerationInput => Some("video_generation_input"),
        _ => None,
    }
}

pub(crate) fn purpose_capabilities(purpose: FilePurpose, _media_type: &str) -> FileCapabilities {
    if !matches!(
        purpose,
        FilePurpose::VoiceClone
            | FilePurpose::PromptAudio
            | FilePurpose::AsyncTtsInput
            | FilePurpose::VideoUnderstanding
            | FilePurpose::VideoGenerationInput
    ) {
        return FileCapabilities::unsupported();
    }
    let mut result = capabilities();
    if matches!(purpose, FilePurpose::VoiceClone | FilePurpose::PromptAudio) {
        result.max_upload_bytes = Some(MINIMAX_VOICE_AUDIO_MAX_UPLOAD_BYTES);
    }
    if matches!(
        purpose,
        FilePurpose::VideoUnderstanding | FilePurpose::VideoGenerationInput
    ) {
        result.retention = Some(Duration::from_secs(7 * 24 * 60 * 60));
    }
    if purpose == FilePurpose::VideoUnderstanding {
        result.list = false;
        result.delete = false;
        result.download = DownloadSupport::Unsupported;
    }
    result
}

pub(crate) fn model_reference(
    profile: &ProviderProfile,
    _model_profile: &ModelProfile,
    model: &str,
    media_type: &str,
) -> ModelFileReference {
    if profile.protocol == ProtocolFamily::AnthropicMessages
        && model.eq_ignore_ascii_case("minimax-m3")
        && media_type.starts_with("video/")
    {
        ModelFileReference::FileUri
    } else {
        ModelFileReference::Unsupported
    }
}

pub(crate) fn metadata_uri(file_id: &str, purpose: Option<&str>) -> Option<String> {
    (purpose == Some("video_understanding")).then(|| format!("mm_file://{file_id}"))
}

pub(crate) fn metadata_downloadable(value: &Value) -> Option<bool> {
    value
        .get("download_url")
        .or_else(|| value.get("downloadable"))
        .and_then(|v| v.as_bool().or_else(|| nonempty_string(v).map(|_| true)))
}

pub(crate) fn adapter(_profile: &ProviderProfile, host: &str) -> Option<Adapter> {
    (matches!(
        host,
        "api.minimaxi.com" | "api.minimax.cn" | "api.minimax.io"
    ))
    .then_some(Adapter::MiniMax)
}
