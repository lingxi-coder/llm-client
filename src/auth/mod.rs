//! Request signing, one implementation per `AuthStrategy`.
//!
//! This crate does not hold, fetch or store credentials. It speaks to
//! providers; whoever calls it owns the secrets and hands one in per request
//! (`RequestOptions::credential`). That keeps every key and token outside a
//! crate whose job is HTTP, and keeps token storage, refresh and OAuth in the
//! host, where the platform's secure storage already lives.

mod api_key;
mod bearer;
pub mod sigv4;

pub use api_key::ApiKeyAuthenticator;
pub use bearer::BearerAuthenticator;

use crate::protocol::{LlmError, ProtocolFamily, ProviderProfile, Secret};
use crate::transport::HttpRequest;
use async_trait::async_trait;

/// Attaches a credential to a request (header, query, signature).
///
/// The credential is passed in, never looked up. An implementation that needs
/// none — or a profile whose `auth` is `None` — gets `None`.
#[async_trait]
pub trait Authenticator: Send + Sync + 'static {
    async fn apply(
        &self,
        req: &mut HttpRequest,
        profile: &ProviderProfile,
        credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError>;
}

// Gate 3.
const _: Option<&dyn Authenticator> = None;

/// Where a wire expects a bare API key to be put.
///
/// This is a property of the protocol, not of the vendor: every endpoint
/// speaking a given wire reads the key from the same header, which is what lets
/// this stay free of provider names (gate 30). A profile that needs another
/// header says so with `extra.credential_header` rather than being special-cased
/// here.
fn key_header(profile: &ProviderProfile) -> &str {
    if let Some(name) = profile
        .extra
        .get("credential_header")
        .and_then(serde_json::Value::as_str)
    {
        return name;
    }
    match profile.protocol {
        ProtocolFamily::AnthropicMessages
        | ProtocolFamily::VertexClaude
        | ProtocolFamily::BedrockClaude
        | ProtocolFamily::FoundryClaude => "x-api-key",
        ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini => "x-goog-api-key",
        ProtocolFamily::OpenAiChat
        | ProtocolFamily::OpenAiResponses
        | ProtocolFamily::AzureOpenAi => AUTHORIZATION,
    }
}

const AUTHORIZATION: &str = "authorization";

/// Replace rather than append: a request carrying two authorization headers is
/// rejected by some endpoints and silently uses the first by others, and
/// neither is a failure mode worth shipping.
fn set_header(req: &mut HttpRequest, name: &str, value: String) {
    req.headers.retain(|(k, _)| !k.eq_ignore_ascii_case(name));
    req.headers.push((name.to_owned(), value));
}

fn required<'a>(
    profile: &ProviderProfile,
    credential: Option<&'a Secret<String>>,
) -> Result<&'a Secret<String>, LlmError> {
    credential.ok_or_else(|| LlmError::Authentication {
        message: format!(
            "profile {:?} needs a credential and none was supplied",
            profile.profile_name
        ),
    })
}
