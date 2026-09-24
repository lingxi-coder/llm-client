//! Provider-specific file capabilities and validation.
use super::*;

pub(crate) fn automatic_file_cache_ttl(profile: &ProviderProfile) -> Option<Duration> {
    matches!(
        adapter(profile),
        Some(Adapter::Anthropic | Adapter::OpenAi | Adapter::Xai)
    )
    .then_some(AUTOMATIC_FILE_CACHE_TTL)
}

/// Qwen's file-extract uploads have no provider-side expiration. Keep their
/// automatic lifetime within one model attempt, even with a stable account scope.
pub(crate) fn needs_automatic_cleanup(profile: &ProviderProfile) -> bool {
    adapter(profile) == Some(Adapter::Qwen)
}

pub(crate) fn valid_qwen_file_id(file_id: &str) -> bool {
    !file_id.is_empty()
        && file_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

pub(crate) fn is_qwen_long_model(model: &str) -> bool {
    model.eq_ignore_ascii_case("qwen-long") || model.to_ascii_lowercase().starts_with("qwen-long-")
}

pub(crate) fn validate_qwen_long_inputs(
    request: &CompletionRequest,
    resolved: &[(usize, usize)],
) -> Result<(), LlmError> {
    validate_qwen_long_blocks(request.messages.iter().enumerate().flat_map(|(mi, m)| {
        m.content
            .iter()
            .enumerate()
            .filter_map(move |(bi, b)| (!resolved.contains(&(mi, bi))).then_some(b))
    }))
}
pub(crate) fn validate_qwen_long_blocks<'a>(
    blocks: impl IntoIterator<Item = &'a ContentBlock>,
) -> Result<(), LlmError> {
    for block in blocks {
        if matches!(
            block,
            ContentBlock::Image {
                source: ImageSource::Base64 { .. } | ImageSource::Url { .. }
            } | ContentBlock::Document {
                source: DocumentSource::Base64 { .. }
                    | DocumentSource::Text { .. }
                    | DocumentSource::Url { .. },
                ..
            }
        ) {
            return Err(LlmError::UnsupportedCapability{message:"Qwen-Long image and document inputs require an app attachment or a Qwen provider file reference; inline data and URLs are unsupported".into()});
        }
    }
    Ok(())
}

pub(crate) fn qwen_long_region_supported(profile: &ProviderProfile) -> bool {
    url::Url::parse(&profile.base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| {
            host == "dashscope.aliyuncs.com"
                || host
                    .strip_suffix(".cn-beijing.maas.aliyuncs.com")
                    .is_some_and(valid_workspace_id)
        })
}

/// Capability matrix keyed by provider identity and the concrete protocol.
/// OpenAI-compatible protocols do not opt providers into Files APIs by themselves.
#[must_use]
pub fn capabilities(profile: &ProviderProfile, model: &str, media_type: &str) -> FileCapabilities {
    let Some(adapter) = adapter(profile) else {
        return FileCapabilities::unsupported();
    };
    let mut result =
        capabilities_for_media_type(capabilities_for_adapter(adapter), adapter, media_type);
    result.model_input = model_reference_for(profile, model, media_type, adapter);
    result
}

/// Provider capabilities for one operation purpose plus the profile's model
/// reference format, if that purpose is a normal model input.
#[must_use]
pub fn capabilities_for_purpose(
    profile: &ProviderProfile,
    model: &str,
    media_type: &str,
    purpose: FilePurpose,
) -> FileCapabilities {
    let Some(adapter) = adapter(profile) else {
        return FileCapabilities::unsupported();
    };
    let mut result = capabilities_for_media_type(
        purpose_capabilities(adapter, purpose, media_type),
        adapter,
        media_type,
    );
    result.model_input = if matches!(
        purpose,
        FilePurpose::ModelInput | FilePurpose::VideoUnderstanding
    ) {
        model_reference_for(profile, model, media_type, adapter)
    } else {
        ModelFileReference::Unsupported
    };
    result
}

pub(crate) fn model_reference_for(
    profile: &ProviderProfile,
    model: &str,
    media_type: &str,
    adapter: Adapter,
) -> ModelFileReference {
    let model_profile = profile
        .models
        .iter()
        .find(|candidate| candidate.request_model == model);
    let Some(model_profile) = model_profile else {
        return ModelFileReference::Unsupported;
    };
    let media_type = media_type.to_ascii_lowercase();
    // OpenAI accepts non-animated GIF input; this MIME-only capability check
    // cannot inspect file bytes to distinguish animated GIFs.
    let is_openai_image = is_openai_image_type(&media_type);
    let is_anthropic_image = matches!(
        media_type.as_str(),
        "image/jpeg" | "image/png" | "image/gif" | "image/webp"
    );
    // Gemini's image-input guide documents these five MIME types.
    let is_gemini_image = matches!(
        media_type.as_str(),
        "image/jpeg" | "image/png" | "image/webp" | "image/heic" | "image/heif"
    );
    let is_pdf = media_type == "application/pdf";
    let has_file_modality = model_declares_file_input(model_profile);
    match adapter {
        Adapter::Qwen
            if qwen_long_region_supported(profile)
                && profile.protocol == ProtocolFamily::OpenAiChat
                && is_qwen_long_model(model)
                && is_qwen_file_type(&media_type) =>
        {
            ModelFileReference::FileUri
        }
        Adapter::MiniMax
            if profile.protocol == ProtocolFamily::AnthropicMessages
                && model.eq_ignore_ascii_case("minimax-m3")
                && media_type.starts_with("video/") =>
        {
            ModelFileReference::FileUri
        }
        Adapter::OpenAi => match profile.protocol {
            ProtocolFamily::OpenAiResponses
                if (is_openai_image && model_declares_image_input(model_profile))
                    || (is_openai_responses_file_type(&media_type) && has_file_modality) =>
            {
                ModelFileReference::FileId
            }
            ProtocolFamily::OpenAiChat if is_pdf && has_file_modality => ModelFileReference::FileId,
            _ => ModelFileReference::Unsupported,
        },
        Adapter::Anthropic if profile.protocol == ProtocolFamily::AnthropicMessages => {
            if (is_anthropic_image && model_declares_image_input(model_profile))
                || ((is_pdf || media_type == "text/plain") && has_file_modality)
            {
                ModelFileReference::FileId
            } else {
                ModelFileReference::Unsupported
            }
        }
        Adapter::Gemini if profile.protocol == ProtocolFamily::GeminiGenerateContent => {
            if (is_gemini_image && model_declares_image_input(model_profile))
                || (is_gemini_video_type(&media_type)
                    && model_profile
                        .metadata
                        .input_modalities
                        .iter()
                        .any(|modality| modality.eq_ignore_ascii_case("video")))
                || (is_gemini_audio_type(&media_type)
                    && model_profile
                        .metadata
                        .input_modalities
                        .iter()
                        .any(|modality| modality.eq_ignore_ascii_case("audio")))
                || (is_gemini_document_type(&media_type)
                    && model_declares_gemini_document_input(model_profile))
            {
                ModelFileReference::FileUri
            } else {
                ModelFileReference::Unsupported
            }
        }
        Adapter::Xai if profile.protocol == ProtocolFamily::OpenAiResponses => {
            if is_xai_document_type(&media_type) && has_file_modality {
                ModelFileReference::FileId
            } else {
                ModelFileReference::Unsupported
            }
        }
        // OpenRouter Files are workspace/shell storage, not a confirmed
        // ordinary-chat attachment reference. Moonshot file_id context is
        // explicitly unsupported. Z.AI uploads are auxiliary Agent API files.
        _ => ModelFileReference::Unsupported,
    }
}

pub(crate) fn is_qwen_files_host(host: &str) -> bool {
    if matches!(
        host,
        "dashscope.aliyuncs.com" | "dashscope-intl.aliyuncs.com"
    ) {
        return true;
    }
    [
        ".cn-beijing.maas.aliyuncs.com",
        ".ap-southeast-1.maas.aliyuncs.com",
    ]
    .iter()
    .find_map(|suffix| host.strip_suffix(suffix))
    .is_some_and(valid_workspace_id)
}

pub(crate) fn valid_workspace_id(workspace: &str) -> bool {
    !workspace.is_empty()
        && workspace
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

pub(crate) fn is_qwen_file_type(media_type: &str) -> bool {
    media_type.starts_with("text/")
        || matches!(
            media_type,
            "application/pdf"
                | "application/json"
                | "application/epub+zip"
                | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                | "application/vnd.oasis.opendocument.text"
                | "application/msword"
                | "application/vnd.ms-excel"
                | "image/bmp"
                | "image/png"
                | "image/jpeg"
                | "image/gif"
        )
}

pub(crate) fn purpose_name(adapter: Adapter, purpose: FilePurpose) -> Option<&'static str> {
    match (adapter, purpose) {
        (Adapter::OpenAi, FilePurpose::ModelInput) => Some("user_data"),
        (Adapter::Xai, FilePurpose::ModelInput) => Some("assistants"),
        (Adapter::Moonshot | Adapter::Qwen, FilePurpose::Extraction | FilePurpose::ModelInput) => {
            Some("file-extract")
        }
        (Adapter::Zai | Adapter::Zhipu, FilePurpose::Auxiliary) => Some("agent"),
        (Adapter::MiniMax, FilePurpose::VoiceClone) => Some("voice_clone"),
        (Adapter::MiniMax, FilePurpose::PromptAudio) => Some("prompt_audio"),
        (Adapter::MiniMax, FilePurpose::AsyncTtsInput) => Some("t2a_async_input"),
        (Adapter::MiniMax, FilePurpose::VideoUnderstanding) => Some("video_understanding"),
        (Adapter::MiniMax, FilePurpose::VideoGenerationInput) => Some("video_generation_input"),
        _ => None,
    }
}

pub(crate) fn is_xai_document_type(media_type: &str) -> bool {
    media_type.starts_with("text/")
        || matches!(
            media_type,
            "application/pdf" | "application/json" | "application/x-ndjson"
        )
}

pub(crate) fn is_gemini_video_type(media_type: &str) -> bool {
    let media_type = media_type.to_ascii_lowercase();
    matches!(
        media_type.as_str(),
        "video/mp4"
            | "video/mpeg"
            | "video/mov"
            | "video/quicktime"
            | "video/avi"
            | "video/x-flv"
            | "video/mpg"
            | "video/webm"
            | "video/wmv"
            | "video/3gpp"
    )
}

pub(crate) fn is_gemini_audio_type(media_type: &str) -> bool {
    matches!(
        media_type,
        "audio/wav"
            | "audio/mp3"
            | "audio/aiff"
            | "audio/aac"
            | "audio/ogg"
            | "audio/flac"
            | "audio/mpeg"
            | "audio/m4a"
            | "audio/l16"
            | "audio/opus"
            | "audio/alaw"
            | "audio/mulaw"
            | "audio/webm"
    )
}

pub(crate) fn is_gemini_document_type(media_type: &str) -> bool {
    media_type.starts_with("text/")
        || matches!(
            media_type,
            "application/pdf" | "application/json" | "application/rtf"
        )
}

pub(crate) fn model_declares_gemini_document_input(model: &ModelProfile) -> bool {
    if model.metadata.input_modalities.is_empty() {
        return model_declares_file_input(model);
    }
    model.metadata.input_modalities.iter().any(|modality| {
        matches!(
            modality.to_ascii_lowercase().as_str(),
            "file" | "files" | "document" | "pdf"
        )
    })
}

pub(crate) fn is_openai_image_type(media_type: &str) -> bool {
    ["image/jpeg", "image/png", "image/webp", "image/gif"]
        .iter()
        .any(|supported| media_type.eq_ignore_ascii_case(supported))
}

pub(crate) fn is_openai_responses_file_type(media_type: &str) -> bool {
    media_type.starts_with("text/")
        || matches!(
            media_type,
            "application/pdf"
                | "application/json"
                | "application/graphql"
                | "application/javascript"
                | "application/typescript"
                | "application/csv"
                | "application/x-iif"
                | "application/x-sql"
                | "application/x-scala"
                | "application/x-rust"
                | "application/x-powershell"
                | "application/x-patch"
                | "application/x-php"
                | "application/x-httpd-php"
                | "application/x-httpd-php-source"
                | "application/x-bash"
                | "application/x-awk"
                | "application/x-protobuf"
                | "application/x-terraform"
                | "application/x-graphql"
                | "application/x-ndjson"
                | "application/json5"
                | "application/x-json5"
                | "application/x-toml"
                | "application/toml"
                | "application/x-yaml"
                | "application/yaml"
                | "application/x-subrip"
                | "application/msword"
                | "application/rtf"
                | "application/vnd.ms-excel"
                | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                | "application/vnd.google-apps.spreadsheet"
                | "application/vnd.apple.pages"
                | "application/vnd.apple.iwork"
                | "application/vnd.oasis.opendocument.text"
                | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                | "application/vnd.google-apps.document"
                | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                | "application/vnd.ms-powerpoint"
                | "application/vnd.apple.keynote"
                | "application/vnd.google-apps.presentation"
                | "message/rfc822"
        )
}

pub(crate) fn model_declares_file_input(model: &ModelProfile) -> bool {
    model.metadata.input_modalities.iter().any(|modality| {
        matches!(
            modality.to_ascii_lowercase().as_str(),
            "file" | "files" | "document" | "pdf"
        )
    }) || model.capability_support_for(crate::protocol::ModelCapability::Documents)
        == crate::protocol::CapabilitySupport::Supported
        || model.metadata.attachments == Some(true)
}

pub(crate) fn model_declares_image_input(model: &ModelProfile) -> bool {
    model
        .metadata
        .input_modalities
        .iter()
        .any(|modality| modality.eq_ignore_ascii_case("image"))
        || model.capability_support_for(crate::protocol::ModelCapability::Vision)
            == crate::protocol::CapabilitySupport::Supported
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Adapter {
    OpenAi,
    Anthropic,
    Gemini,
    Xai,
    OpenRouter,
    Moonshot,
    Zai,
    Zhipu,
    Qwen,
    MiniMax,
}

pub(crate) fn adapter(profile: &ProviderProfile) -> Option<Adapter> {
    let host = url::Url::parse(&profile.base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))?;
    match (profile.provider_id.as_str(), host.as_str()) {
        ("openai", "api.openai.com") => Some(Adapter::OpenAi),
        ("anthropic", "api.anthropic.com") => Some(Adapter::Anthropic),
        ("google", "generativelanguage.googleapis.com")
            if profile.protocol == ProtocolFamily::GeminiGenerateContent =>
        {
            Some(Adapter::Gemini)
        }
        ("xai", "api.x.ai") => Some(Adapter::Xai),
        ("openrouter", "openrouter.ai") => Some(Adapter::OpenRouter),
        ("kimi" | "moonshot", "api.moonshot.cn" | "api.moonshot.ai") => Some(Adapter::Moonshot),
        ("zhipu", "api.z.ai") => Some(Adapter::Zai),
        ("zhipu", "open.bigmodel.cn") => Some(Adapter::Zhipu),
        ("qwen", host) if is_qwen_files_host(host) => Some(Adapter::Qwen),
        ("minimax", "api.minimaxi.com" | "api.minimax.io") => Some(Adapter::MiniMax),
        _ => None,
    }
}

pub(crate) fn capabilities_for_adapter(adapter: Adapter) -> FileCapabilities {
    match adapter {
        Adapter::OpenAi => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::UploadedFiles,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(512 * 1024 * 1024),
            retention: None,
        },
        Adapter::Anthropic => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::GeneratedFilesOnly,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(500 * 1024 * 1024),
            retention: None,
        },
        Adapter::Gemini => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::GeneratedFilesOnly,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(2 * 1024 * 1024 * 1024),
            retention: Some(Duration::from_secs(48 * 60 * 60)),
        },
        Adapter::Xai => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::UploadedFiles,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(512 * 1024 * 1024),
            retention: None,
        },
        Adapter::OpenRouter => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::GeneratedFilesOnly,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(100 * 1024 * 1024),
            retention: None,
        },
        Adapter::Moonshot => FileCapabilities {
            upload: false,
            retrieve_metadata: false,
            list: false,
            delete: true,
            download: DownloadSupport::Unsupported,
            extract_text: true,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(100 * 1024 * 1024),
            retention: None,
        },
        // Z.AI's documented /files API is limited to Agent API auxiliary files;
        // it is not a reference format for ordinary chat completions.
        Adapter::Zai => FileCapabilities {
            upload: false,
            retrieve_metadata: false,
            list: false,
            delete: false,
            download: DownloadSupport::Unsupported,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(100 * 1024 * 1024),
            retention: Some(Duration::from_secs(180 * 24 * 60 * 60)),
        },
        Adapter::Zhipu => FileCapabilities {
            upload: false,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::Unsupported,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(20 * 1024 * 1024),
            retention: None,
        },
        Adapter::Qwen => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::Unsupported,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: Some(150_000_000),
            retention: None,
        },
        Adapter::MiniMax => FileCapabilities {
            upload: true,
            retrieve_metadata: true,
            list: true,
            delete: true,
            download: DownloadSupport::GeneratedFilesOnly,
            extract_text: false,
            model_input: ModelFileReference::Unsupported,
            max_upload_bytes: None,
            retention: None,
        },
    }
}

pub(crate) fn purpose_capabilities(
    adapter: Adapter,
    purpose: FilePurpose,
    media_type: &str,
) -> FileCapabilities {
    match (adapter, purpose) {
        (Adapter::Moonshot, FilePurpose::Extraction) => FileCapabilities {
            upload: true,
            ..capabilities_for_adapter(adapter)
        },
        (Adapter::Zai, FilePurpose::Auxiliary) => FileCapabilities {
            upload: true,
            ..capabilities_for_adapter(adapter)
        },
        (Adapter::Zhipu, FilePurpose::Auxiliary) => FileCapabilities {
            upload: true,
            ..capabilities_for_adapter(adapter)
        },
        (Adapter::Qwen, FilePurpose::ModelInput | FilePurpose::Extraction) => FileCapabilities {
            upload: true,
            ..capabilities_for_adapter(adapter)
        },
        (Adapter::MiniMax, purpose)
            if matches!(
                purpose,
                FilePurpose::VoiceClone
                    | FilePurpose::PromptAudio
                    | FilePurpose::AsyncTtsInput
                    | FilePurpose::VideoUnderstanding
                    | FilePurpose::VideoGenerationInput
            ) =>
        {
            let mut capabilities = capabilities_for_adapter(adapter);
            capabilities.upload = true;
            capabilities.max_upload_bytes = None;
            if matches!(
                purpose,
                FilePurpose::VideoUnderstanding | FilePurpose::VideoGenerationInput
            ) {
                capabilities.retention = Some(Duration::from_secs(7 * 24 * 60 * 60));
            }
            if purpose == FilePurpose::VideoUnderstanding {
                capabilities.list = false;
                capabilities.delete = false;
                capabilities.download = DownloadSupport::Unsupported;
            }
            capabilities
        }
        (Adapter::OpenRouter, FilePurpose::Workspace) => capabilities_for_adapter(adapter),
        (Adapter::OpenAi, FilePurpose::ModelInput) => {
            let mut capabilities = capabilities_for_adapter(adapter);
            if !is_openai_image_type(media_type) {
                capabilities.max_upload_bytes = Some(OPENAI_INPUT_FILE_MAX_UPLOAD_BYTES);
            }
            capabilities
        }
        (Adapter::Anthropic | Adapter::Gemini | Adapter::Xai, FilePurpose::ModelInput) => {
            capabilities_for_adapter(adapter)
        }
        _ => FileCapabilities::unsupported(),
    }
}

pub(crate) fn capabilities_for_media_type(
    mut capabilities: FileCapabilities,
    adapter: Adapter,
    media_type: &str,
) -> FileCapabilities {
    if adapter == Adapter::Gemini
        && capabilities.upload
        && media_type.eq_ignore_ascii_case("application/pdf")
    {
        capabilities.max_upload_bytes = Some(
            capabilities
                .max_upload_bytes
                .map_or(GEMINI_PDF_MAX_UPLOAD_BYTES, |limit| {
                    limit.min(GEMINI_PDF_MAX_UPLOAD_BYTES)
                }),
        );
    }
    if adapter == Adapter::Qwen
        && capabilities.upload
        && media_type.to_ascii_lowercase().starts_with("image/")
    {
        capabilities.max_upload_bytes = Some(QWEN_IMAGE_MAX_UPLOAD_BYTES);
    }
    capabilities
}

pub(crate) fn validate_direct_provider_file_inputs<'a>(
    blocks: impl IntoIterator<Item = &'a ContentBlock>,
    profile: &ProviderProfile,
    model: &str,
    account_scope: Option<&str>,
) -> Result<(), LlmError> {
    for block in blocks {
        let (file, media_prefix) = match block {
            ContentBlock::Image {
                source: ImageSource::ProviderFile { file },
            } => (file, Some("image/")),
            ContentBlock::Document {
                source: DocumentSource::ProviderFile { file },
                ..
            } => (file, None),
            ContentBlock::Video {
                source: VideoSource::ProviderFile { file },
            } => (file, Some("video/")),
            _ => continue,
        };
        validate_provider_file(file, profile, account_scope)?;
        let media_type = file
            .media_type
            .as_deref()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "direct provider file input requires a media type".into(),
            })?;
        if media_prefix.is_some_and(|prefix| !media_type.to_ascii_lowercase().starts_with(prefix))
            || (media_prefix.is_none()
                && (media_type.to_ascii_lowercase().starts_with("image/")
                    || media_type.to_ascii_lowercase().starts_with("video/")))
        {
            return Err(LlmError::InvalidRequest {
                message: "provider file media type does not match its content block".into(),
            });
        }
        let expected_purpose = match profile.provider_id.as_str() {
            "xai" => Some("assistants"),
            "qwen" => Some("file-extract"),
            "minimax" if media_prefix == Some("video/") => Some("video_understanding"),
            _ => None,
        };
        if let Some(expected) = expected_purpose {
            if file.purpose.as_deref() != Some(expected) {
                return Err(LlmError::UnsupportedCapability {
                    message: format!("provider file input requires purpose {expected:?}"),
                });
            }
        }
        if capabilities(profile, model, media_type).model_input == ModelFileReference::Unsupported {
            return Err(LlmError::UnsupportedCapability {
                message: format!(
                    "model {model:?} does not support provider file input for {media_type:?}"
                ),
            });
        }
    }
    Ok(())
}
pub(crate) fn validate_provider_file<'a>(
    file: &'a crate::protocol::ProviderFileSource,
    profile: &ProviderProfile,
    account_scope: Option<&str>,
) -> Result<&'a crate::protocol::ProviderFileSource, LlmError> {
    let Some(active_account_scope) = account_scope.filter(|scope| !scope.trim().is_empty()) else {
        return Err(LlmError::UnsupportedCapability {
            message: "provider file inputs require an explicit account scope".into(),
        });
    };
    if file.protocol != profile.protocol
        || file.provider_id != profile.provider_id
        || file.profile_name != profile.profile_name
        || file.endpoint_fingerprint
            != crate::files::provider_file_endpoint_fingerprint(&profile.base_url)
        || file.account_scope.as_deref() != Some(active_account_scope)
    {
        return Err(LlmError::UnsupportedCapability {
            message: "provider file reference belongs to a different connection, endpoint, or account scope".into(),
        });
    }
    if file.file_id.trim().is_empty()
        || file.uri.as_deref().is_some_and(str::is_empty)
        || file.media_type.as_deref().is_some_and(str::is_empty)
    {
        return Err(LlmError::InvalidRequest {
            message: "provider file reference is missing a required identifier".into(),
        });
    }
    Ok(file)
}
/// Pure, valid placeholder used to validate the entire wire request before uploads.
pub(crate) fn projected_reference(
    profile: &ProviderProfile,
    scope: Option<&str>,
    media_type: &str,
    purpose: FilePurpose,
) -> ProviderFileSource {
    let adapter = adapter(profile);
    let file_id = if adapter == Some(Adapter::Gemini) {
        "files/file-projection"
    } else if adapter == Some(Adapter::MiniMax) {
        "1"
    } else {
        "file-projection"
    };
    let uri = match adapter {
        Some(Adapter::Gemini) => {
            Some("https://generativelanguage.googleapis.com/v1beta/files/file-projection".into())
        }
        Some(Adapter::MiniMax) if purpose == FilePurpose::VideoUnderstanding => {
            Some("mm_file://1".into())
        }
        _ => None,
    };
    ProviderFileSource {
        protocol: profile.protocol,
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&profile.base_url),
        account_scope: scope.map(str::to_owned),
        file_id: file_id.into(),
        uri,
        media_type: Some(media_type.into()),
        purpose: adapter
            .and_then(|adapter| purpose_name(adapter, purpose))
            .map(str::to_owned),
    }
}
