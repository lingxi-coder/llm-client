//! `WireCodec` for the OpenAI Responses protocol family.
//!
//! Ported from the previous project's `llm-client/src/providers/openai_responses.rs`.
//! Despite the shared vendor this is a different wire from Chat Completions,
//! and the differences are the things that break if they are assumed away:
//!
//! - the conversation is `input`, a flat list of *items* — a message is one
//!   item, and a function call is a sibling item rather than a field on one;
//! - the system prompt is `instructions`, a single string;
//! - function-call arguments are a JSON **string**, as on the Chat wire, but
//!   the call is identified by `call_id`;
//! - there is no stop-sequence parameter at all, so a request carrying one is
//!   refused rather than silently sent without it;
//! - the stream ends at `response.completed` with no `[DONE]`, though
//!   compatible gateways may append one anyway.

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
pub struct OpenAiResponsesCodec;

impl WireCodec for OpenAiResponsesCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::OpenAiResponses
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
        Box::new(stream::ResponsesStreamDecoder::default())
    }

    fn response_usage(&self, resp: &HttpResponse) -> Option<Usage> {
        serde_json::from_slice::<serde_json::Value>(&resp.body)
            .ok()
            .and_then(|v| v.get("usage").cloned())
            .map(|u| decode::usage(&u))
    }
}
