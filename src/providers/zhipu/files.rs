//! Provider-owned file workflow and capability policy.
use crate::files::*;

pub(crate) fn capabilities_global() -> FileCapabilities {
    FileCapabilities {
        upload: false,
        retrieve_metadata: false,
        list: false,
        delete: false,
        download: DownloadSupport::Unsupported,
        extract_text: false,
        model_input: ModelFileReference::Unsupported,
        max_upload_bytes: Some(100 * 1024 * 1024),
        retention: Some(Duration::from_secs(180 * 24 * 60 * 60)),
    }
}

pub(crate) fn capabilities() -> FileCapabilities {
    FileCapabilities {
        upload: false,
        retrieve_metadata: true,
        list: true,
        delete: true,
        download: DownloadSupport::Unsupported,
        extract_text: false,
        model_input: ModelFileReference::Unsupported,
        max_upload_bytes: Some(20 * 1024 * 1024),
        retention: None,
    }
}

pub(crate) fn purpose_name(purpose: FilePurpose) -> Option<&'static str> {
    match purpose {
        FilePurpose::Auxiliary => Some("agent"),
        _ => None,
    }
}

pub(crate) fn purpose_capabilities(
    adapter: Adapter,
    purpose: FilePurpose,
    _media_type: &str,
) -> FileCapabilities {
    if purpose == FilePurpose::Auxiliary {
        FileCapabilities {
            upload: true,
            ..if adapter == Adapter::Zai {
                capabilities_global()
            } else {
                capabilities()
            }
        }
    } else {
        FileCapabilities::unsupported()
    }
}

pub(crate) fn adapter(_profile: &ProviderProfile, host: &str) -> Option<Adapter> {
    match host {
        "api.z.ai" => Some(Adapter::Zai),
        "open.bigmodel.cn" => Some(Adapter::Zhipu),
        _ => None,
    }
}
