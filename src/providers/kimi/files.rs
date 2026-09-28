//! Provider-owned file workflow and capability policy.
use crate::files::*;

pub(crate) fn capabilities() -> FileCapabilities {
    FileCapabilities {
        upload: false,
        retrieve_metadata: false,
        list: false,
        delete: true,
        download: DownloadSupport::Unsupported,
        extract_text: true,
        model_input: ModelFileReference::Unsupported,
        max_upload_bytes: Some(100 * 1024 * 1024),
        retention: None,
    }
}

pub(crate) fn purpose_name(purpose: FilePurpose) -> Option<&'static str> {
    match purpose {
        FilePurpose::Extraction | FilePurpose::ModelInput => Some("file-extract"),
        _ => None,
    }
}

pub(crate) fn purpose_capabilities(purpose: FilePurpose, _media_type: &str) -> FileCapabilities {
    if purpose == FilePurpose::Extraction {
        FileCapabilities {
            upload: true,
            ..capabilities()
        }
    } else {
        FileCapabilities::unsupported()
    }
}

pub(crate) fn adapter(_profile: &ProviderProfile, host: &str) -> Option<Adapter> {
    (matches!(host, "api.moonshot.cn" | "api.moonshot.ai")).then_some(Adapter::Moonshot)
}
