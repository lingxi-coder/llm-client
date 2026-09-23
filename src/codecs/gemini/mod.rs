//! `WireCodec` for the Gemini `generateContent` protocol family.
//!
//! Ported from the previous project's `llm-client/src/providers/gemini.rs`.
//! Three things set this wire apart, and each one bites if it is missed:
//!
//! - **a tool result names its function and echoes the provider's call ID when
//!   supplied.** The encoder finds both from earlier `ToolUse` blocks.
//! - **the assistant's role on this wire is `model`.**
//! - **a prompt over the context window arrives as HTTP 400 `INVALID_ARGUMENT`,
//!   not 413.** Left as `InvalidRequest` it would end the turn terminally
//!   instead of triggering compaction (gate 31).

mod decode;
pub(crate) mod encode;
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
pub struct GeminiCodec;

impl WireCodec for GeminiCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::GeminiGenerateContent
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
        Box::new(stream::GeminiStreamDecoder::default())
    }

    fn response_usage(&self, resp: &HttpResponse) -> Option<Usage> {
        serde_json::from_slice::<serde_json::Value>(&resp.body)
            .ok()
            .and_then(|v| v.get("usageMetadata").cloned())
            .map(|u| decode::usage(&u))
    }
}

/// `{base}/models/{model}:generateContent`, or the SSE variant when streaming.
/// Shared with the Vertex wrapper, which supplies a different base.
pub(crate) fn generate_content_url(base: &str, model: &str, stream: bool) -> String {
    let base = base.trim_end_matches('/');
    if stream {
        format!("{base}/models/{model}:streamGenerateContent?alt=sse")
    } else {
        format!("{base}/models/{model}:generateContent")
    }
}
