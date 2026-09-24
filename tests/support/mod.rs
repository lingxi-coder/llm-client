//! Fixtures shared by the wire tests.

use async_trait::async_trait;
use lingxi_llm_client::protocol::{LlmError, Secret};
use lingxi_llm_client::{HttpRequest, HttpResponse, StreamResponse, Transport};

/// A transport that refuses everything: these tests never leave the process.
#[allow(dead_code)]
pub struct NoHttp;

#[async_trait]
impl Transport for NoHttp {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.response(request).await.map(Into::into)
    }
}
impl NoHttp {
    async fn response(&self, _req: HttpRequest) -> Result<HttpResponse, LlmError> {
        Err(LlmError::Transport {
            message: "no transport in this test".to_owned(),
        })
    }
}

/// A stand-in for the authenticators that have not been ported yet. Attaches
/// nothing: these tests never reach a network.
///
/// Shared by several test binaries; not every one of them needs it.
#[allow(dead_code)]
pub struct NoAuth;

#[async_trait]
impl lingxi_llm_client::Authenticator for NoAuth {
    async fn apply(
        &self,
        _req: &mut HttpRequest,
        _profile: &lingxi_llm_client::protocol::ProviderProfile,
        _credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}
