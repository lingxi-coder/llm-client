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
pub(crate) mod encode;
pub(crate) mod stream;

pub use decode::classify_error;

use crate::codecs::{CodecContext, EncodeRequest};
use crate::codecs::{StreamDecoder, WireCodec};
use crate::protocol::{CompletionResponse, LlmError, ProtocolFamily};
use crate::transport::{HttpRequest, HttpResponse};

/// The API version header this codec speaks. A profile may override it through
/// `extra.api_version` when a provider pins a different one.
pub const DEFAULT_API_VERSION: &str = crate::wire_options::DEFAULT_ANTHROPIC_API_VERSION;

#[derive(Debug, Default, Clone, Copy)]
pub struct AnthropicMessagesCodec;

impl WireCodec for AnthropicMessagesCodec {
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
        ProtocolFamily::AnthropicMessages
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
            stream::AnthropicStreamDecoder::configured(context),
        ))
    }
}
