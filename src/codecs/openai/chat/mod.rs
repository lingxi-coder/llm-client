//! `WireCodec` for the OpenAI Chat Completions protocol family.
//!
//! Ported from the previous project's `llm-client/src/providers/openai.rs`.
//! One codec serves every provider that speaks this wire — OpenAI itself and
//! the OpenAI-compatible endpoints — which is what makes adding a provider a
//! settings entry rather than a code change (gate 30). Anything provider-
//! specific is keyed off the `ProviderProfile` the caller passes in, never off
//! a hard-coded provider name.

mod decode;
pub(crate) mod encode;
mod reasoning;
mod stream;

pub use decode::classify_error;

use crate::codecs::{CodecContext, EncodeRequest};
use crate::codecs::{StreamDecoder, WireCodec};
use crate::protocol::{CompletionResponse, LlmError, ProtocolFamily};
use crate::transport::{HttpRequest, HttpResponse};

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAiChatCodec;

impl OpenAiChatCodec {
    pub fn new() -> Self {
        Self
    }
}

impl WireCodec for OpenAiChatCodec {
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
        ProtocolFamily::OpenAiChat
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
        decode::response_with_usage_mode(resp, separate_reasoning(&context.profile.extra)).map(
            |mut response| {
                response.inference = crate::codecs::inference::response(resp, context.profile());
                response
            },
        )
    }
    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        let _ = context;
        Box::new(crate::codecs::stream::SseDecoder::new(
            stream::OpenAiStreamDecoder::configured(context),
        ))
    }
}

fn separate_reasoning(meta: &serde_json::Value) -> bool {
    meta.get("reasoning_tokens_separate")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}
