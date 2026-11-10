//! TypeSafe-native System One resources, independent of Chat models.
pub use crate::providers::native_decision::NativeDecisionError;
use crate::{client::RequestOptions, transport::Transport};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemOneRequest {
    pub model: String,
    pub state: Value,
    pub questions: BTreeMap<String, Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub assets: BTreeMap<String, Value>,
}
pub struct SystemOne<'a> {
    transport: &'a dyn Transport,
    endpoint: String,
}
impl<'a> SystemOne<'a> {
    pub fn new(transport: &'a dyn Transport, endpoint: impl Into<String>) -> Self {
        Self {
            transport,
            endpoint: endpoint.into(),
        }
    }
    pub async fn evaluate(
        &self,
        request: &SystemOneRequest,
        options: &RequestOptions,
    ) -> Result<Value, NativeDecisionError> {
        if request.model.trim().is_empty() || request.questions.is_empty() {
            return Err(crate::protocol::LlmError::InvalidRequest {
                message: "systemone requires a model and named questions".into(),
            }
            .into());
        }
        let response = crate::providers::native_decision::send(
            self.transport,
            &self.endpoint,
            request,
            options,
            !request.assets.is_empty(),
        )
        .await?;
        if !response["answers"].is_object() {
            return Err(NativeDecisionError::InvalidResponse);
        }
        Ok(response)
    }
}

/// Exact-profile TypeSafe binding. Native resources do not require Chat model rows.
#[derive(Clone)]
pub struct TypeSafeClient {
    binding: crate::providers::binding::ProviderBinding,
}
impl crate::providers::binding::private::Sealed for TypeSafeClient {}
impl crate::providers::ProviderClient for TypeSafeClient {
    const PROVIDER_ID: &'static str = "typesafe";
    fn from_binding(binding: crate::providers::binding::ProviderBinding) -> Self {
        Self { binding }
    }
}
impl TypeSafeClient {
    pub fn system_one(
        &self,
        endpoint: impl Into<String>,
    ) -> Result<SystemOne<'_>, crate::protocol::LlmError> {
        self.binding.pin()?;
        Ok(SystemOne::new(self.binding.transport(), endpoint))
    }
}
