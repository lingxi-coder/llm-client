//! `WireCodec` for the OpenAI Chat Completions protocol family.
//!
//! Ported from the previous project's `llm-client/src/providers/openai.rs`.
//! One codec serves every provider that speaks this wire — OpenAI itself and
//! the OpenAI-compatible endpoints — which is what makes adding a provider a
//! settings entry rather than a code change (gate 30). Anything provider-
//! specific is keyed off the `ProviderProfile` the caller passes in, never off
//! a hard-coded provider name.

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

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAiChatCodec;

impl OpenAiChatCodec {
    pub fn new() -> Self {
        Self
    }
}

impl WireCodec for OpenAiChatCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::OpenAiChat
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
        Box::new(stream::OpenAiStreamDecoder::default())
    }

    fn response_usage(&self, resp: &HttpResponse) -> Option<Usage> {
        serde_json::from_slice::<serde_json::Value>(&resp.body)
            .ok()
            .and_then(|v| v.get("usage").cloned())
            .map(|u| decode::usage(&u))
    }
}
