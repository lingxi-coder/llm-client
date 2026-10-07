//! A provider response that can be continued only on its original route.

use super::{LlmError, ProtocolFamily, ProviderId, ProviderProfile, ResponseId};
use serde::{Deserialize, Serialize};

/// Opaque provider state bound to the connection, model and caller's account.
///
/// Obtain this from `ChatResponse::continuation` or `ModelStream::continuation`.
/// `account_scope` is a stable, non-secret account identifier supplied in
/// `RequestOptions`; callers must use the same value on the next request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuationRef {
    pub response_id: ResponseId,
    pub protocol: ProtocolFamily,
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub endpoint_fingerprint: String,
    pub account_scope: String,
    pub request_model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<String>,
}

impl ContinuationRef {
    pub(crate) fn scoped(
        response_id: ResponseId,
        profile: &ProviderProfile,
        request_model: &str,
        account_scope: &str,
        workspace_id: Option<&str>,
    ) -> Self {
        Self {
            response_id,
            protocol: profile.protocol,
            provider_id: profile.provider_id.clone(),
            profile_name: profile.profile_name.clone(),
            endpoint_fingerprint: crate::files::provider_file_endpoint_fingerprint(
                &profile.base_url,
            ),
            account_scope: account_scope.to_owned(),
            request_model: request_model.to_owned(),
            workspace_id: workspace_id.map(str::to_owned),
        }
    }

    pub(crate) fn validate(
        &self,
        profile: &ProviderProfile,
        request_model: &str,
        account_scope: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<(), LlmError> {
        if self.protocol != profile.protocol
            || self.response_id.as_str().trim().is_empty()
            || self.provider_id != profile.provider_id
            || self.profile_name != profile.profile_name
            || self.endpoint_fingerprint
                != crate::files::provider_file_endpoint_fingerprint(&profile.base_url)
            || self.account_scope.is_empty()
            || Some(self.account_scope.as_str()) != account_scope
            || self.request_model != request_model
            || self.workspace_id.as_deref() != workspace_id
        {
            return Err(LlmError::InvalidRequest {
                message: "continuation reference does not match the selected connection, model, workspace or account scope".into(),
            });
        }
        Ok(())
    }
}
