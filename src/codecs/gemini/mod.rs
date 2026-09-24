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

use crate::codecs::{CodecContext, EncodeRequest};
use crate::codecs::{StreamDecoder, WireCodec};
use crate::protocol::{CompletionResponse, LlmError, ProtocolFamily};
use crate::transport::{HttpRequest, HttpResponse};

#[derive(Debug, Default, Clone, Copy)]
pub struct GeminiCodec;

impl WireCodec for GeminiCodec {
    fn request_inference(
        &self,
        req: &crate::protocol::CompletionRequest,
        context: &CodecContext,
    ) -> Result<crate::protocol::InferenceReport, LlmError> {
        crate::codecs::inference::requested(req, context.profile())
    }
    fn validate_request(
        &self,
        req: &crate::protocol::CompletionRequest,
        context: &CodecContext,
    ) -> Result<(), LlmError> {
        crate::codecs::inference::validate(req, context.profile(), context.request_model())
    }
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::GeminiGenerateContent
    }
    fn encode_request(
        &self,
        req: EncodeRequest<'_>,
        context: &CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        encode::request(req, context.profile(), context)?.encode()
    }
    fn encoded_body_len(
        &self,
        req: EncodeRequest<'_>,
        context: &CodecContext,
    ) -> Result<usize, LlmError> {
        encode::request(req, context.profile(), context)?.body_len()
    }
    fn decode_response(
        &self,
        resp: &HttpResponse,
        context: &CodecContext,
    ) -> Result<CompletionResponse, LlmError> {
        let _ = context;
        decode::response(resp).map(|mut response| {
            response.inference = crate::codecs::inference::response(resp, context.profile());
            response
        })
    }
    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        let _ = context;
        Box::new(crate::codecs::stream::SseDecoder::new(
            stream::GeminiStreamDecoder::configured(context),
        ))
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
