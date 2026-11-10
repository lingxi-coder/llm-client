//! Native OpenAI POST /v1/decisions. No Chat or Structured Outputs emulation.
pub use crate::providers::native_decision::NativeDecisionError;
use crate::{client::RequestOptions, transport::Transport};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateRequest {
    pub model: String,
    pub input: Value,
    pub questions: Vec<Value>,
}
/// Exact endpoint and transport supplied by the host; credentials remain request-scoped.
pub struct Decisions<'a> {
    transport: &'a dyn Transport,
    endpoint: String,
    binding: Option<&'a crate::providers::binding::ProviderBinding>,
}
impl<'a> Decisions<'a> {
    pub fn new(transport: &'a dyn Transport, endpoint: impl Into<String>) -> Self {
        Self {
            transport,
            endpoint: endpoint.into(),
            binding: None,
        }
    }
    pub(crate) fn with_binding(
        mut self,
        binding: &'a crate::providers::binding::ProviderBinding,
    ) -> Self {
        self.binding = Some(binding);
        self
    }
    pub async fn create(
        &self,
        request: &CreateRequest,
        options: &RequestOptions,
    ) -> Result<Value, NativeDecisionError> {
        if let Some(binding) = self.binding {
            let snapshot = binding.pin()?;
            if snapshot
                .profile(binding.profile_name())
                .is_some_and(|profile| profile.auth == crate::protocol::AuthStrategy::ChatGptPlan)
            {
                return Err(crate::protocol::LlmError::UnsupportedCapability {
                    message: "ChatGPT plan credentials do not support native Decisions".into(),
                }
                .into());
            }
        }
        if request.model.trim().is_empty()
            || request.questions.is_empty()
            || request.questions.len() > 400
        {
            return Err(crate::protocol::LlmError::InvalidRequest {
                message: "decision requires a model and 1–400 questions".into(),
            }
            .into());
        }
        let response = crate::providers::native_decision::send(
            self.transport,
            &self.endpoint,
            request,
            options,
            false,
        )
        .await?;
        if !response["answers"].is_array() {
            return Err(NativeDecisionError::InvalidResponse);
        }
        Ok(response)
    }
}
