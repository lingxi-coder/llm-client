//! Provider-specific file capabilities and validation.
use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

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
    let Some(model_profile) = profile
        .models
        .iter()
        .find(|candidate| candidate.request_model == model)
    else {
        return ModelFileReference::Unsupported;
    };
    let media_type = media_type.to_ascii_lowercase();
    match adapter {
        Adapter::OpenAi => crate::providers::openai::files::model_reference(
            profile,
            model_profile,
            model,
            &media_type,
        ),
        Adapter::Anthropic => crate::providers::anthropic::files::model_reference(
            profile,
            model_profile,
            model,
            &media_type,
        ),
        Adapter::Gemini => crate::providers::google::files::model_reference(
            profile,
            model_profile,
            model,
            &media_type,
        ),
        Adapter::Xai => crate::providers::xai::files::model_reference(
            profile,
            model_profile,
            model,
            &media_type,
        ),
        Adapter::Qwen => crate::providers::qwen::files::model_reference(
            profile,
            model_profile,
            model,
            &media_type,
        ),
        Adapter::MiniMax => crate::providers::minimax::files::model_reference(
            profile,
            model_profile,
            model,
            &media_type,
        ),
        _ => ModelFileReference::Unsupported,
    }
}

pub(crate) fn purpose_name(adapter: Adapter, purpose: FilePurpose) -> Option<&'static str> {
    match adapter {
        Adapter::OpenAi => crate::providers::openai::files::purpose_name(purpose),
        Adapter::Xai => crate::providers::xai::files::purpose_name(purpose),
        Adapter::Moonshot => crate::providers::kimi::files::purpose_name(purpose),
        Adapter::Zai => crate::providers::zhipu::files::purpose_name(purpose),
        Adapter::Zhipu => crate::providers::zhipu::files::purpose_name(purpose),
        Adapter::Qwen => crate::providers::qwen::files::purpose_name(purpose),
        Adapter::MiniMax => crate::providers::minimax::files::purpose_name(purpose),
        _ => None,
    }
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
    match profile.provider_id.as_str() {
        "openai" => crate::providers::openai::files::adapter(profile, &host),
        "anthropic" => crate::providers::anthropic::files::adapter(profile, &host),
        "google" => crate::providers::google::files::adapter(profile, &host),
        "xai" => crate::providers::xai::files::adapter(profile, &host),
        "openrouter" => crate::providers::openrouter::files::adapter(profile, &host),
        "kimi" | "moonshot" => crate::providers::kimi::files::adapter(profile, &host),
        "qwen" => crate::providers::qwen::files::adapter(profile, &host),
        "minimax" => crate::providers::minimax::files::adapter(profile, &host),
        "zhipu" => crate::providers::zhipu::files::adapter(profile, &host),
        _ => None,
    }
}

/// Canonical endpoint identity for provider file references. Foundry's Files
/// API is resource-scoped and uses one normalized Anthropic resource base;
/// ordinary service discovery still does not infer that route from a URL.
pub(crate) fn provider_file_endpoint_identity(profile: &ProviderProfile) -> Option<String> {
    if profile.protocol == ProtocolFamily::FoundryClaude {
        crate::providers::anthropic::types::AnthropicContainerScope::normalize_foundry_endpoint(
            &profile.base_url,
        )
    } else {
        Some(profile.base_url.clone())
    }
}

pub(crate) fn capabilities_for_adapter(adapter: Adapter) -> FileCapabilities {
    match adapter {
        Adapter::OpenAi => crate::providers::openai::files::capabilities(),
        Adapter::Anthropic => crate::providers::anthropic::files::capabilities(),
        Adapter::Gemini => crate::providers::google::files::capabilities(),
        Adapter::Xai => crate::providers::xai::files::capabilities(),
        Adapter::OpenRouter => crate::providers::openrouter::files::capabilities(),
        Adapter::Moonshot => crate::providers::kimi::files::capabilities(),
        Adapter::Zai => crate::providers::zhipu::files::capabilities_global(),
        Adapter::Zhipu => crate::providers::zhipu::files::capabilities(),
        Adapter::Qwen => crate::providers::qwen::files::capabilities(),
        Adapter::MiniMax => crate::providers::minimax::files::capabilities(),
    }
}

pub(crate) fn purpose_capabilities(
    adapter: Adapter,
    purpose: FilePurpose,
    media_type: &str,
) -> FileCapabilities {
    match adapter {
        Adapter::OpenAi => {
            crate::providers::openai::files::purpose_capabilities(purpose, media_type)
        }
        Adapter::Anthropic => {
            crate::providers::anthropic::files::purpose_capabilities(purpose, media_type)
        }
        Adapter::Gemini => {
            crate::providers::google::files::purpose_capabilities(purpose, media_type)
        }
        Adapter::Xai => crate::providers::xai::files::purpose_capabilities(purpose, media_type),
        Adapter::OpenRouter => {
            crate::providers::openrouter::files::purpose_capabilities(purpose, media_type)
        }
        Adapter::Moonshot => {
            crate::providers::kimi::files::purpose_capabilities(purpose, media_type)
        }
        Adapter::Zai => {
            crate::providers::zhipu::files::purpose_capabilities(adapter, purpose, media_type)
        }
        Adapter::Zhipu => {
            crate::providers::zhipu::files::purpose_capabilities(adapter, purpose, media_type)
        }
        Adapter::Qwen => crate::providers::qwen::files::purpose_capabilities(purpose, media_type),
        Adapter::MiniMax => {
            crate::providers::minimax::files::purpose_capabilities(purpose, media_type)
        }
    }
}

pub(crate) fn capabilities_for_media_type(
    capabilities: FileCapabilities,
    adapter: Adapter,
    media_type: &str,
) -> FileCapabilities {
    match adapter {
        Adapter::Gemini => {
            crate::providers::google::files::capabilities_for_media_type(capabilities, media_type)
        }
        Adapter::Qwen => {
            crate::providers::qwen::files::capabilities_for_media_type(capabilities, media_type)
        }
        _ => capabilities,
    }
}

pub(crate) fn validate_direct_provider_file_inputs_at<'a>(
    blocks: impl IntoIterator<Item = &'a ContentBlock>,
    profile: &ProviderProfile,
    model: &str,
    account_scope: Option<&str>,
    now: SystemTime,
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
        validate_provider_file_at(file, profile, account_scope, now)?;
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
pub(crate) fn validate_provider_file_at<'a>(
    file: &'a crate::protocol::ProviderFileSource,
    profile: &ProviderProfile,
    account_scope: Option<&str>,
    now: SystemTime,
) -> Result<&'a crate::protocol::ProviderFileSource, LlmError> {
    validate_provider_file_identity(file, profile, account_scope)?;
    validate_file_expiration_at(file.expires_at.as_deref(), now)?;
    validate_provider_file_readiness(file, profile)?;
    Ok(file)
}

pub(crate) fn validate_provider_file_readiness(
    file: &crate::protocol::ProviderFileSource,
    profile: &ProviderProfile,
) -> Result<(), LlmError> {
    let Some(status) = file.processing_status.as_deref() else {
        return Ok(());
    };
    match (adapter(profile), status) {
        (Some(Adapter::Gemini), "PROCESSING")
        | (Some(Adapter::Qwen), "uploaded" | "processing") => {
            Err(LlmError::ProviderFileProcessing {
                message: format!("provider file is still processing ({status})"),
                file: Box::new(file.clone()),
            })
        }
        (Some(Adapter::Gemini), "FAILED") | (Some(Adapter::Qwen), "error") => {
            Err(LlmError::InvalidRequest {
                message: format!("provider file processing failed ({status})"),
            })
        }
        // Unknown and absent statuses remain unknown; they are not treated as
        // ready or failed based on another provider's status vocabulary.
        _ => Ok(()),
    }
}

pub(crate) fn validate_provider_file_identity<'a>(
    file: &'a crate::protocol::ProviderFileSource,
    profile: &ProviderProfile,
    account_scope: Option<&str>,
) -> Result<&'a crate::protocol::ProviderFileSource, LlmError> {
    let Some(active_account_scope) = account_scope.filter(|scope| !scope.trim().is_empty()) else {
        return Err(LlmError::UnsupportedCapability {
            message: "provider file inputs require an explicit account scope".into(),
        });
    };
    let endpoint_identity = provider_file_endpoint_identity(profile);
    if file.protocol != profile.protocol
        || file.provider_id != profile.provider_id
        || file.profile_name != profile.profile_name
        || endpoint_identity.as_deref().is_none_or(|endpoint| {
            file.endpoint_fingerprint != crate::files::provider_file_endpoint_fingerprint(endpoint)
        })
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

/// Validate a provider timestamp without changing its representation in the
/// reference. Integer values are Unix seconds; textual values are RFC 3339.
pub(crate) fn validate_file_expiration_at(
    expires_at: Option<&str>,
    now: SystemTime,
) -> Result<(), LlmError> {
    let Some(expires_at) = expires_at else {
        return Ok(());
    };
    let expiry = parse_provider_file_expiration(expires_at)?;
    if expiry <= now {
        return Err(LlmError::InvalidRequest {
            message: "provider file reference has expired".into(),
        });
    }
    Ok(())
}

fn parse_provider_file_expiration(value: &str) -> Result<SystemTime, LlmError> {
    let nanos_since_epoch = if let Ok(seconds) = value.parse::<i64>() {
        i128::from(seconds)
            .checked_mul(1_000_000_000)
            .ok_or_else(invalid_expiration)?
    } else {
        let datetime =
            chrono::DateTime::parse_from_rfc3339(value).map_err(|_| invalid_expiration())?;
        i128::from(datetime.timestamp())
            .checked_mul(1_000_000_000)
            .and_then(|nanos| nanos.checked_add(i128::from(datetime.timestamp_subsec_nanos())))
            .ok_or_else(invalid_expiration)?
    };
    system_time_from_unix_nanos(nanos_since_epoch).ok_or_else(invalid_expiration)
}

fn system_time_from_unix_nanos(nanos_since_epoch: i128) -> Option<SystemTime> {
    const NANOS_PER_SECOND: i128 = 1_000_000_000;
    let (magnitude, subtract) = if nanos_since_epoch < 0 {
        (nanos_since_epoch.checked_neg()?, true)
    } else {
        (nanos_since_epoch, false)
    };
    let seconds = u64::try_from(magnitude / NANOS_PER_SECOND).ok()?;
    let subsecond_nanos = u32::try_from(magnitude % NANOS_PER_SECOND).ok()?;
    let duration = Duration::new(seconds, subsecond_nanos);
    if subtract {
        UNIX_EPOCH.checked_sub(duration)
    } else {
        UNIX_EPOCH.checked_add(duration)
    }
}

fn invalid_expiration() -> LlmError {
    LlmError::InvalidRequest {
        message: "provider file expiration timestamp is malformed or unrepresentable".into(),
    }
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
        endpoint_fingerprint: provider_file_endpoint_fingerprint(
            &provider_file_endpoint_identity(profile).unwrap_or_else(|| profile.base_url.clone()),
        ),
        account_scope: scope.map(str::to_owned),
        file_id: file_id.into(),
        uri,
        expires_at: None,
        processing_status: None,
        media_type: Some(media_type.into()),
        purpose: adapter
            .and_then(|adapter| purpose_name(adapter, purpose))
            .map(str::to_owned),
    }
}

#[cfg(test)]
mod expiration_tests {
    use super::*;

    fn at(seconds: u64, nanos: u32) -> SystemTime {
        UNIX_EPOCH + Duration::new(seconds, nanos)
    }

    #[test]
    fn expiration_accepts_absent_and_unexpired_values_and_expires_at_equality() {
        assert!(validate_file_expiration_at(None, UNIX_EPOCH).is_ok());
        assert!(validate_file_expiration_at(Some("1"), UNIX_EPOCH).is_ok());
        assert!(validate_file_expiration_at(Some("1"), at(1, 0)).is_err());
    }

    #[test]
    fn expiration_rfc3339_fractional_seconds_are_compared_exactly() {
        let expiration = "2030-01-01T00:00:00.500Z";
        let boundary = parse_provider_file_expiration(expiration).unwrap();
        assert!(
            validate_file_expiration_at(Some(expiration), boundary - Duration::from_nanos(1))
                .is_ok()
        );
        assert!(validate_file_expiration_at(Some(expiration), boundary).is_err());
        assert!(
            validate_file_expiration_at(Some(expiration), boundary + Duration::from_nanos(1))
                .is_err()
        );
    }

    #[test]
    fn expiration_rfc3339_offsets_compare_as_the_same_instant() {
        let positive_offset = "2030-01-01T02:00:00+02:00";
        let negative_offset = "2029-12-31T19:00:00-05:00";
        assert_eq!(
            parse_provider_file_expiration(positive_offset).unwrap(),
            parse_provider_file_expiration(negative_offset).unwrap()
        );
        let boundary = parse_provider_file_expiration(positive_offset).unwrap();
        assert!(validate_file_expiration_at(Some(negative_offset), boundary).is_err());
    }

    #[test]
    fn expiration_rejects_malformed_and_unrepresentable_values() {
        for value in ["", "not-a-date", "9223372036854775808"] {
            assert!(matches!(
                validate_file_expiration_at(Some(value), UNIX_EPOCH),
                Err(LlmError::InvalidRequest { .. })
            ));
        }
    }

    #[test]
    fn pre_epoch_integer_expiration_is_already_expired_at_epoch() {
        assert!(matches!(
            validate_file_expiration_at(Some("-1"), UNIX_EPOCH),
            Err(LlmError::InvalidRequest { .. })
        ));
    }
}
