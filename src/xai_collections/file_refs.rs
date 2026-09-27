//! Conversion between xAI Collections upload results and the provider-wide
//! file service reference type.

use crate::{
    files::{provider_file_endpoint_fingerprint, ProviderFileRef},
    protocol::{LlmError, ProtocolFamily, ProviderProfile},
};
use serde_json::Value;

use super::XaiUploadedFile;

impl XaiUploadedFile {
    /// Convert this Collections upload into a reference accepted by
    /// [`crate::files::FileService`].
    ///
    /// The target profile and account scope must identify the same xAI
    /// connection that uploaded the file. This conversion only carries the
    /// file identity and metadata; it does not make the file usable as input
    /// for chat requests that do not support xAI file IDs.
    pub fn to_provider_file_ref(
        &self,
        profile: &ProviderProfile,
        account_scope: &str,
    ) -> Result<ProviderFileRef, LlmError> {
        if account_scope.trim().is_empty() || account_scope.chars().any(char::is_control) {
            return Err(invalid(
                "account_scope must be a non-empty, non-secret identifier",
            ));
        }
        if self.reference.file_id.trim().is_empty()
            || self.reference.file_id.len() > 1024
            || self.reference.file_id.chars().any(char::is_control)
        {
            return Err(invalid(
                "xAI uploaded file reference has an invalid file ID",
            ));
        }
        if !matches!(
            profile.protocol,
            ProtocolFamily::OpenAiChat
                | ProtocolFamily::OpenAiResponses
                | ProtocolFamily::AnthropicMessages
        ) {
            return Err(LlmError::UnsupportedCapability {
                message: "the selected xAI profile protocol is not supported by the Files API"
                    .into(),
            });
        }

        // Collections normalizes the API root before fingerprinting it. Match
        // that identity here, while retaining the profile's raw fingerprint
        // below because FileService validates references against that exact
        // profile value.
        let normalized_base = super::normalize_base_url(&profile.base_url)
            .map_err(|_| invalid("the selected profile has an invalid xAI API base URL"))?;
        let normalized_fingerprint = provider_file_endpoint_fingerprint(&normalized_base);
        let scope = &self.reference.scope;
        if scope.provider_id.as_str() != "xai"
            || profile.provider_id.as_str() != "xai"
            || profile.provider_id != scope.provider_id
            || profile.profile_name != scope.profile_name
            || normalized_fingerprint != scope.api_endpoint_fingerprint
            || account_scope != scope.account_scope
        {
            return Err(LlmError::PermissionDenied {
                message: "xAI uploaded file reference belongs to another provider profile, endpoint, or account scope".into(),
            });
        }
        if crate::files::adapter(profile) != Some(crate::files::Adapter::Xai) {
            return Err(LlmError::UnsupportedCapability {
                message: "the selected profile does not resolve to the xAI Files API".into(),
            });
        }

        let native = &self.native;
        Ok(ProviderFileRef {
            provider_id: profile.provider_id.clone(),
            profile_name: profile.profile_name.clone(),
            endpoint_fingerprint: provider_file_endpoint_fingerprint(&profile.base_url),
            account_scope: Some(account_scope.to_owned()),
            protocol: profile.protocol,
            file_id: self.reference.file_id.clone(),
            uri: None,
            filename: self
                .filename
                .clone()
                .or_else(|| native.get("filename").and_then(json_string)),
            media_type: native
                .get("mime_type")
                .and_then(json_string)
                .or_else(|| native.get("content_type").and_then(json_string)),
            size_bytes: self
                .size_bytes
                .or_else(|| native.get("bytes").and_then(json_u64)),
            expires_at: native
                .get("expires_at")
                .and_then(json_scalar_string)
                .or_else(|| {
                    self.expires_at_unix_seconds
                        .map(|seconds| seconds.to_string())
                }),
            processing_status: native
                .get("status")
                .and_then(json_string)
                .or_else(|| native.get("state").and_then(json_string)),
            downloadable: native.get("downloadable").and_then(Value::as_bool),
            purpose: native.get("purpose").and_then(json_string),
        })
    }
}

fn json_string(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned)
}

fn json_scalar_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(str::to_owned)
        .or_else(|| value.as_i64().map(|number| number.to_string()))
        .or_else(|| value.as_u64().map(|number| number.to_string()))
}

fn json_u64(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| value.as_str()?.parse().ok())
}

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
