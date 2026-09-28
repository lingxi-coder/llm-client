//! Attachment policy registry. Shared preparation owns caching and scheduling.
use crate::{
    client::{AttachmentKind, RequestOptions, ResolvedAttachmentPayload},
    files::FilePurpose,
    protocol::{ChatRequest, LlmError, ProtocolFamily, ProviderProfile},
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FirstPartyEndpoint {
    Other,
    OpenAi,
    Anthropic,
    Gemini,
}

pub(crate) fn official_https_host(profile: &ProviderProfile, host: &str) -> bool {
    url::Url::parse(&profile.base_url)
        .is_ok_and(|url| url.scheme() == "https" && url.host_str() == Some(host))
}
pub(crate) fn first_party_endpoint(profile: &ProviderProfile) -> FirstPartyEndpoint {
    match profile.provider_id.as_str() {
        "openai" if super::openai::attachments::first_party(profile) => FirstPartyEndpoint::OpenAi,
        "anthropic" if super::anthropic::attachments::first_party(profile) => {
            FirstPartyEndpoint::Anthropic
        }
        "google" if super::google::attachments::first_party(profile) => FirstPartyEndpoint::Gemini,
        _ => FirstPartyEndpoint::Other,
    }
}
pub(crate) fn image_formats(
    endpoint: FirstPartyEndpoint,
) -> Option<(&'static str, &'static [&'static str])> {
    match endpoint {
        FirstPartyEndpoint::OpenAi => Some(("OpenAI", super::openai::attachments::IMAGE_FORMATS)),
        FirstPartyEndpoint::Anthropic => {
            Some(("Anthropic", super::anthropic::attachments::IMAGE_FORMATS))
        }
        FirstPartyEndpoint::Gemini => Some(("Gemini", super::google::attachments::IMAGE_FORMATS)),
        FirstPartyEndpoint::Other => None,
    }
}
pub(crate) fn validate_request(
    profile: &ProviderProfile,
    model: &str,
    request: &ChatRequest,
    attachments: &[ResolvedAttachmentPayload],
    opts: &RequestOptions,
    endpoint: FirstPartyEndpoint,
) -> Result<(), LlmError> {
    super::google::attachments::validate_gemini_video_media_types(request, profile)?;
    super::qwen::attachments::preflight_qwen_long_files(
        profile,
        model,
        request,
        attachments,
        opts,
    )?;
    if endpoint == FirstPartyEndpoint::OpenAi {
        super::openai::attachments::validate_first_party_openai_documents(
            request,
            profile.protocol,
        )?;
    }
    if profile.protocol == ProtocolFamily::AnthropicMessages {
        super::anthropic::attachments::validate_anthropic_document_media_types(request)?;
    }
    Ok(())
}
pub(crate) fn requires_file_reference(profile: &ProviderProfile, model: &str) -> bool {
    super::qwen::attachments::is_qwen_long(profile, model)
}
pub(crate) fn file_purpose(profile: &ProviderProfile, kind: AttachmentKind) -> FilePurpose {
    match profile.provider_id.as_str() {
        "minimax" => super::minimax::attachments::file_purpose(kind),
        _ => FilePurpose::ModelInput,
    }
}
