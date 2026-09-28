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
//! - **unknown events and deltas are retained as observational frames**:
//!   they are not request content. Native blocks become replayable only when
//!   their complete contents can be reconstructed at the block boundary.

pub(crate) mod decode;
pub(crate) mod encode;
pub(crate) mod stream;

pub use decode::classify_error;

use crate::codecs::{CodecContext, EncodeRequest};
use crate::codecs::{StreamDecoder, WireCodec};
use crate::protocol::{ChatResponse, LlmError, ProtocolFamily};
use crate::transport::{HttpRequest, HttpResponse};

/// The API version header this codec speaks. A profile may override it through
/// `extra.api_version` when a provider pins a different one.
pub const DEFAULT_API_VERSION: &str = crate::wire_options::DEFAULT_ANTHROPIC_API_VERSION;

#[derive(Debug, Default, Clone, Copy)]
pub struct AnthropicMessagesCodec;

impl WireCodec for AnthropicMessagesCodec {
    fn request_inference(
        &self,
        req: &crate::protocol::ChatRequest,
        context: &CodecContext,
    ) -> Result<crate::protocol::InferenceReport, LlmError> {
        crate::codecs::inference::requested(req, context.profile())
    }
    fn validate_request(
        &self,
        req: &crate::protocol::ChatRequest,
        context: &CodecContext,
    ) -> Result<(), LlmError> {
        crate::codecs::anthropic_client_toolsets::validate(req, context)?;
        crate::codecs::anthropic_tool_search::validate(req, context)?;
        crate::codecs::cache::validate(req, context)?;
        crate::codecs::anthropic_conversation::validate(req, context)?;
        crate::codecs::anthropic_mcp::validate(req, context)?;
        crate::codecs::anthropic_web_fetch::validate(req, context)?;
        crate::codecs::anthropic_code_execution::validate(req, context)?;
        crate::codecs::inference::validate(req, context.profile(), context.request_model())
            .and_then(|()| {
                crate::codecs::openrouter_server_tools::validate(
                    req,
                    context.profile(),
                    None,
                    false,
                )
                .map(|_| ())
            })
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
    ) -> Result<ChatResponse, LlmError> {
        let retain_openrouter_container =
            crate::codecs::openrouter_server_tools::is_official_profile(context.profile());
        decode::response(
            resp,
            retain_openrouter_container,
            crate::codecs::anthropic_code_execution::is_official_profile(context.profile())
                || crate::codecs::anthropic_code_execution::supports_execution(context),
        )
        .map(|mut response| {
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
