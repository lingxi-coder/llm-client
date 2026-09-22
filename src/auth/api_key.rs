//! A bare API key, in whatever header the wire reads it from.

use super::Authenticator;
use super::{key_header, required, set_header, AUTHORIZATION};
use crate::transport::HttpRequest;
use async_trait::async_trait;
use lingxi_agent_api::protocol::{LlmError, ProviderProfile, Secret};

/// A bare API key, in whatever header this wire reads it from.
pub struct ApiKeyAuthenticator;

#[async_trait]
impl Authenticator for ApiKeyAuthenticator {
    async fn apply(
        &self,
        req: &mut HttpRequest,
        profile: &ProviderProfile,
        credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        let key = required(profile, credential)?;
        let header = key_header(profile);
        let value = if header.eq_ignore_ascii_case(AUTHORIZATION) {
            format!("Bearer {}", key.expose_secret())
        } else {
            key.expose_secret().clone()
        };
        set_header(req, header, value);
        Ok(())
    }
}
