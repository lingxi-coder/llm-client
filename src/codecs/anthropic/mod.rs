//! `WireCodec` for the Anthropic Messages protocol family.
//!
//! Ported from the previous project's `llm-client/src/providers/anthropic.rs`.
//! What distinguishes this wire from the OpenAI one, and why each matters:
//!
//! - **thinking blocks carry a signature, and replaying one without it is
//!   refused rather than sent.** The provider rejects an unsigned replay, so a
//!   codec that dropped the signature would turn a valid transcript into a
//!   400 on the *next* turn, far from the cause (gate 18).
//! - **beta headers accumulate**, comma-joined. Overwriting silently drops one
//!   when a request needs two.
//! - **`max_tokens` is required**, so it has a default here rather than being
//!   omitted.
//! - **unknown event and delta types are ignored**: this provider's streaming
//!   contract requires a client to tolerate types it has never seen.

mod decode;
mod encode;
mod stream;

pub use decode::classify_error;

use crate::client::route::ResolvedRoute;
use crate::codecs::{StreamDecoder, WireCodec};
use crate::transport::{HttpRequest, HttpResponse};
use crate::RequestOptions;
use lingxi_agent_api::protocol::{
    CompletionRequest, CompletionResponse, LlmError, ProtocolFamily, ProviderProfile, Usage,
};

/// The API version header this codec speaks. A profile may override it through
/// `extra.api_version` when a provider pins a different one.
pub const DEFAULT_API_VERSION: &str = "2023-06-01";

#[derive(Debug, Default, Clone, Copy)]
pub struct AnthropicMessagesCodec;

impl WireCodec for AnthropicMessagesCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::AnthropicMessages
    }

    fn encode_request(
        &self,
        req: &CompletionRequest,
        profile: &ProviderProfile,
        route: &ResolvedRoute,
        opts: &RequestOptions,
    ) -> Result<HttpRequest, LlmError> {
        encode::request(req, profile, route, opts)
    }

    fn decode_response(&self, resp: &HttpResponse) -> Result<CompletionResponse, LlmError> {
        decode::response(resp)
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(stream::AnthropicStreamDecoder::default())
    }

    fn response_usage(&self, resp: &HttpResponse) -> Option<Usage> {
        serde_json::from_slice::<serde_json::Value>(&resp.body)
            .ok()
            .and_then(|v| v.get("usage").cloned())
            .map(|u| decode::usage(&u))
    }
}
