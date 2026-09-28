//! Provider-owned file workflow and capability policy.
use crate::files::*;

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

pub(crate) fn capabilities() -> FileCapabilities {
    FileCapabilities {
        upload: true,
        retrieve_metadata: true,
        list: true,
        delete: true,
        download: DownloadSupport::UploadedFiles,
        extract_text: false,
        model_input: ModelFileReference::Unsupported,
        max_upload_bytes: Some(512 * 1024 * 1024),
        retention: None,
    }
}

pub(crate) const OPENAI_INPUT_FILE_MAX_UPLOAD_BYTES: u64 = 50_000_000;

pub(crate) fn purpose_name(purpose: FilePurpose) -> Option<&'static str> {
    match purpose {
        FilePurpose::Batch => Some("batch"),
        FilePurpose::ModelInput => Some("user_data"),
        _ => None,
    }
}

pub(crate) fn purpose_capabilities(purpose: FilePurpose, media_type: &str) -> FileCapabilities {
    match purpose {
        FilePurpose::Batch => {
            let mut result = capabilities();
            result.max_upload_bytes = Some(200_000_000);
            result
        }
        FilePurpose::ModelInput => {
            let mut result = capabilities();
            if !is_openai_image_type(media_type) {
                result.max_upload_bytes = Some(OPENAI_INPUT_FILE_MAX_UPLOAD_BYTES);
            }
            result
        }
        _ => FileCapabilities::unsupported(),
    }
}

pub(crate) fn model_reference(
    profile: &ProviderProfile,
    model_profile: &ModelProfile,
    _model: &str,
    media_type: &str,
) -> ModelFileReference {
    match profile.protocol {
        ProtocolFamily::OpenAiResponses
            if (is_openai_image_type(media_type) && model_declares_image_input(model_profile))
                || (is_openai_responses_file_type(media_type)
                    && model_declares_file_input(model_profile)) =>
        {
            ModelFileReference::FileId
        }
        ProtocolFamily::OpenAiChat
            if media_type == "application/pdf" && model_declares_file_input(model_profile) =>
        {
            ModelFileReference::FileId
        }
        _ => ModelFileReference::Unsupported,
    }
}

pub(crate) fn adapter(_profile: &ProviderProfile, host: &str) -> Option<Adapter> {
    (host == "api.openai.com").then_some(Adapter::OpenAi)
}
