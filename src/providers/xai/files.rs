//! Provider-owned file workflow and capability policy.
use crate::files::*;

pub(crate) fn is_xai_document_type(media_type: &str) -> bool {
    media_type.starts_with("text/")
        || matches!(
            media_type,
            "application/pdf" | "application/json" | "application/x-ndjson"
        )
}

pub(crate) fn capabilities() -> FileCapabilities {
    FileCapabilities {
        upload: true,
        retrieve_metadata: true,
        list: true,
        delete: true,
        download: DownloadSupport::UploadedFiles,
        extract_text: false,
        model_input: ModelFileReference::Unsupported,
        max_upload_bytes: Some(50_000_000),
        retention: None,
    }
}

pub(crate) fn purpose_name(purpose: FilePurpose) -> Option<&'static str> {
    match purpose {
        FilePurpose::ModelInput => Some("assistants"),
        _ => None,
    }
}

pub(crate) fn purpose_capabilities(purpose: FilePurpose, _media_type: &str) -> FileCapabilities {
    if matches!(purpose, FilePurpose::Batch | FilePurpose::ModelInput) {
        capabilities()
    } else {
        FileCapabilities::unsupported()
    }
}

pub(crate) fn model_reference(
    profile: &ProviderProfile,
    model_profile: &ModelProfile,
    _model: &str,
    media_type: &str,
) -> ModelFileReference {
    if profile.protocol == ProtocolFamily::OpenAiResponses
        && is_xai_document_type(media_type)
        && model_declares_file_input(model_profile)
    {
        ModelFileReference::FileId
    } else {
        ModelFileReference::Unsupported
    }
}

pub(crate) fn adapter(_profile: &ProviderProfile, host: &str) -> Option<Adapter> {
    (host == "api.x.ai").then_some(Adapter::Xai)
}
