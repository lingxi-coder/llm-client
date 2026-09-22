//! A token that is already a bearer token.

use super::Authenticator;
use super::{required, set_header, AUTHORIZATION};
use crate::transport::HttpRequest;
use async_trait::async_trait;
use lingxi_agent_api::protocol::{LlmError, ProviderProfile, Secret};

/// A token that is already a bearer token — an exchanged or minted one, not a
/// key the user typed.
///
/// Identical on the wire to an API key on a wire that reads `authorization`,
/// and deliberately still a separate strategy: what a profile declares says
/// where its credential came from, which is what decides who has to refresh it.
pub struct BearerAuthenticator;

#[async_trait]
impl Authenticator for BearerAuthenticator {
    async fn apply(
        &self,
        req: &mut HttpRequest,
        profile: &ProviderProfile,
        credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        let token = required(profile, credential)?;
        set_header(
            req,
            AUTHORIZATION,
            format!("Bearer {}", token.expose_secret()),
        );
        Ok(())
    }
}
