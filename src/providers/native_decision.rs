//! Shared bounded HTTP execution for provider-native decision resources.
use crate::{
    client::RequestOptions,
    protocol::LlmError,
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest, Transport},
};
use serde_json::Value;
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum NativeDecisionError {
    #[error(transparent)]
    Transport(#[from] LlmError),
    #[error("decision provider returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        usage: Option<Value>,
    },
    #[error("invalid decision response")]
    InvalidResponse,
}

pub(crate) async fn send(
    transport: &dyn Transport,
    endpoint: &str,
    body: &Value,
    options: &RequestOptions,
) -> Result<Value, NativeDecisionError> {
    let url = url::Url::parse(endpoint).map_err(|_| NativeDecisionError::InvalidResponse)?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(LlmError::InvalidRequest {
            message: "invalid decision endpoint".into(),
        }
        .into());
    }
    let deadline = Deadline::after(Some(
        options.total_timeout.unwrap_or(Duration::from_secs(30)),
    ));
    let mut headers = vec![
        ("content-type".into(), "application/json".into()),
        ("accept".into(), "application/json".into()),
    ];
    if let Some(key) = &options.credential {
        if key.expose_secret().is_empty()
            || key
                .expose_secret()
                .chars()
                .any(|c| matches!(c, '\r' | '\n' | '\0'))
        {
            return Err(LlmError::InvalidRequest {
                message: "invalid decision credential".into(),
            }
            .into());
        }
        headers.push((
            "authorization".into(),
            format!("Bearer {}", key.expose_secret()),
        ));
    }
    if body
        .get("assets")
        .and_then(Value::as_object)
        .is_some_and(|assets| !assets.is_empty())
    {
        headers.push(("TypeSafe-Danger-Zone-Images".into(), "true".into()));
    }
    let bytes = serde_json::to_vec(body).map_err(|_| NativeDecisionError::InvalidResponse)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err(LlmError::InvalidRequest {
            message: "decision input exceeds 16 MiB".into(),
        }
        .into());
    }
    let response = HttpExecutor::new(transport)
        .with_deadline(deadline)
        .execute_bounded(
            HttpRequest {
                http1_header_layout: None,
                method: "POST".into(),
                url: endpoint.into(),
                headers,
                body: bytes.into(),
                timeout: deadline.remaining()?,
            },
            16 * 1024 * 1024,
        )
        .await?;
    let parsed = serde_json::from_slice::<Value>(&response.body);
    if !(200..300).contains(&response.status) {
        return Err(NativeDecisionError::Provider {
            status: response.status,
            request_id: response.header("x-request-id").map(str::to_owned),
            usage: parsed
                .as_ref()
                .ok()
                .and_then(|value| value.get("usage").cloned()),
        });
    }
    let parsed = parsed.map_err(|_| NativeDecisionError::InvalidResponse)?;
    if !parsed.is_object() {
        return Err(NativeDecisionError::InvalidResponse);
    }
    Ok(parsed)
}
