//! Fixtures shared by the wire tests.

use async_trait::async_trait;
use lingxi_agent_api::protocol::{LlmError, Secret};
use lingxi_llm_client::{HttpRequest, HttpResponse, StreamResponse, Transport, WebSocketSession};

/// A transport that refuses everything: these tests never leave the process.
#[allow(dead_code)]
pub struct NoHttp;

#[async_trait]
impl Transport for NoHttp {
    async fn execute(&self, _req: HttpRequest) -> Result<HttpResponse, LlmError> {
        Err(LlmError::Transport {
            message: "no transport in this test".to_owned(),
        })
    }
    async fn open_stream(&self, _req: HttpRequest) -> Result<StreamResponse, LlmError> {
        Err(LlmError::Transport {
            message: "no transport in this test".to_owned(),
        })
    }
    async fn open_responses_websocket_session(
        &self,
        _req: HttpRequest,
    ) -> Result<Box<dyn WebSocketSession>, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "no websockets in this test".to_owned(),
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
        _profile: &lingxi_agent_api::protocol::ProviderProfile,
        _credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}
