//! Provider-owned file workflow and capability policy.
use crate::files::*;

pub(crate) fn capabilities() -> FileCapabilities {
    FileCapabilities {
        upload: true,
        retrieve_metadata: true,
        list: true,
        delete: true,
        download: DownloadSupport::GeneratedFilesOnly,
        extract_text: false,
        model_input: ModelFileReference::Unsupported,
        max_upload_bytes: Some(100 * 1024 * 1024),
        retention: None,
    }
}

pub(crate) fn purpose_capabilities(purpose: FilePurpose, _media_type: &str) -> FileCapabilities {
    if purpose == FilePurpose::Workspace {
        capabilities()
    } else {
        FileCapabilities::unsupported()
    }
}

pub(crate) fn adapter(_profile: &ProviderProfile, host: &str) -> Option<Adapter> {
    (host == "openrouter.ai").then_some(Adapter::OpenRouter)
}
