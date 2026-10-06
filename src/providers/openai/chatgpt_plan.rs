//! Public model discovery for an account authorized to use its ChatGPT plan.
use crate::protocol::{AuthStrategy, LlmError, Secret};
use crate::providers::binding::ProviderBinding;
use crate::runtime::Deadline;
use crate::transport::{HttpExecutor, HttpRequest};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

const MAX_MODEL_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatGptPlanModel {
    pub slug: String,
    pub display_name: String,
    /// The complete provider row for metadata that is not part of this stable view.
    pub native: Value,
}

/// Model-list failures retain OpenAI's status, request ID, and response shape
/// so the host can distinguish a revoked grant from a temporary admission or
/// usage failure. `body` may be an API error object or a direct-route detail.
#[derive(Debug, thiserror::Error)]
pub enum ChatGptPlanModelError {
    #[error(transparent)]
    Client(#[from] LlmError),
    #[error("ChatGPT plan model discovery returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
}

pub(crate) async fn list_models(
    binding: &ProviderBinding,
    access_token: &Secret<String>,
) -> Result<Vec<ChatGptPlanModel>, ChatGptPlanModelError> {
    let snapshot = binding.pin()?;
    let profile =
        snapshot
            .profile(binding.profile_name())
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "the bound OpenAI profile is unavailable".into(),
            })?;
    if profile.auth != AuthStrategy::ChatGptPlan {
        return Err(LlmError::UnsupportedCapability {
            message: "ChatGPT plan model discovery requires chat_gpt_plan authentication".into(),
        }
        .into());
    }
    crate::auth::chatgpt_plan::validate_profile(profile)
        .map_err(|message| LlmError::InvalidRequest { message })?;
    if access_token.expose_secret().is_empty() {
        return Err(LlmError::Authentication {
            message: "ChatGPT plan model discovery requires a nonempty access token".into(),
        }
        .into());
    }
    let deadline = Deadline::after(Some(Duration::from_secs(120)));
    let request = HttpRequest {
        method: "GET".into(),
        url: "https://api.openai.com/v1/models".into(),
        headers: vec![
            ("accept".into(), "application/json".into()),
            (
                "authorization".into(),
                format!("Bearer {}", access_token.expose_secret()),
            ),
        ],
        body: Vec::new().into(),
        timeout: deadline.remaining()?,
    };
    let response = HttpExecutor::new(snapshot.runtime.http.as_ref())
        .with_deadline(deadline)
        .execute_bounded(request, MAX_MODEL_RESPONSE_BYTES)
        .await?;
    if !(200..300).contains(&response.status) {
        let body = serde_json::from_slice(&response.body).unwrap_or_else(|_| {
            Value::String(String::from_utf8_lossy(&response.body).into_owned())
        });
        return Err(ChatGptPlanModelError::Provider {
            status: response.status,
            request_id: response
                .header("x-request-id")
                .or_else(|| response.header("openai-request-id"))
                .map(str::to_owned),
            body,
        });
    }
    let body: Value =
        serde_json::from_slice(&response.body).map_err(|_| LlmError::ProviderInternal {
            message: "ChatGPT plan model discovery returned invalid JSON".into(),
        })?;
    let models = body
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| LlmError::ProviderInternal {
            message: "ChatGPT plan model discovery omitted models".into(),
        })?;
    models
        .iter()
        .filter(|row| row.get("visibility").and_then(Value::as_str) == Some("list"))
        .map(|row| {
            let slug = row
                .get("slug")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| LlmError::ProviderInternal {
                    message: "ChatGPT plan model has no slug".into(),
                })?;
            let display_name = row
                .get("display_name")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| LlmError::ProviderInternal {
                    message: "ChatGPT plan model has no display_name".into(),
                })?;
            Ok(ChatGptPlanModel {
                slug: slug.into(),
                display_name: display_name.into(),
                native: row.clone(),
            })
        })
        .collect::<Result<Vec<_>, LlmError>>()
        .map_err(Into::into)
}
