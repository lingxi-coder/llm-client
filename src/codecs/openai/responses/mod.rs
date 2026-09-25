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

pub(crate) mod decode;
pub(crate) mod encode;
mod stream;

pub use decode::classify_error;

use crate::codecs::{CodecContext, EncodeRequest};
use crate::codecs::{StreamDecoder, WireCodec};
use crate::protocol::{CompletionResponse, LlmError, ProtocolFamily};
use crate::transport::{HttpRequest, HttpResponse};

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAiResponsesCodec;

impl WireCodec for OpenAiResponsesCodec {
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
        ProtocolFamily::OpenAiResponses
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
            stream::ResponsesStreamDecoder::configured(context),
        ))
    }
}
