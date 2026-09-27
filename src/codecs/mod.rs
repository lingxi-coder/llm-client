//! The wire codecs this crate ships.
//!
//! Everything that talks to a provider lives here rather than in a capability
//! crate: a codec is not something a profile turns on, it is how this crate
//! speaks a protocol family. `LlmClientBuilder` seeds its table from these, so
//! a profile naming an OpenAI-compatible provider needs no capability entry —
//! only a `ProviderProfile` (gate 30).

pub(crate) mod code_interpreter;
pub(crate) mod inference;
mod input;
mod json;
pub(crate) mod openrouter_server_tools;
pub use input::{CodecContext, ContentBinding, EncodeRequest, PreparedMedia, RequestMode};
pub mod anthropic;
pub(crate) mod anthropic_client_toolset_history;
pub(crate) mod anthropic_client_toolsets;
pub(crate) mod anthropic_code_execution;
pub(crate) mod anthropic_conversation;
pub(crate) mod anthropic_mcp;
pub(crate) mod anthropic_tool_search;
pub(crate) mod anthropic_web_fetch;
pub(crate) mod file_search_decode;
pub(crate) mod foundry;
pub mod gemini;
pub mod hosted;
pub mod openai;
mod stream;
pub(crate) mod usage;
pub(crate) mod web_search;
pub(crate) mod web_search_decode;

use crate::protocol::{
    ChatRequest, ChatResponse, LlmError, ProtocolFamily, ProviderFileSource, ProviderProfile,
    StreamEvent,
};
use crate::transport::{HttpRequest, HttpResponse};

use std::sync::Arc;

/// A Responses continuation id is endpoint-side state, not an optional hint.
/// Refuse it on every other wire rather than silently starting a fresh turn.
pub(crate) fn reject_responses_continuation(
    req: &ChatRequest,
    family: ProtocolFamily,
) -> Result<(), LlmError> {
    if req.continuation.is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: format!("{family:?} cannot continue a Responses id"),
        });
    }
    Ok(())
}

pub(crate) fn reject_code_interpreter(
    req: &ChatRequest,
    family: ProtocolFamily,
) -> Result<(), LlmError> {
    if req.anthropic_mcp_servers().next().is_some()
        && !matches!(
            family,
            ProtocolFamily::AnthropicMessages | ProtocolFamily::FoundryClaude
        )
    {
        return Err(LlmError::UnsupportedCapability {
            message: format!("{family:?} cannot encode the Anthropic MCP connector"),
        });
    }
    if req.hosted_code_interpreter().is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: format!("{family:?} cannot encode Code Interpreter"),
        });
    }
    Ok(())
}

/// Validate that a provider-owned input reference belongs to the exact
/// connection and caller-supplied account scope used for this request.
pub(crate) fn validate_provider_file<'a>(
    file: &'a ProviderFileSource,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<&'a ProviderFileSource, LlmError> {
    crate::files::validate_provider_file_at(
        file,
        profile,
        opts.file_scope(),
        opts.file_validation_time(),
    )
}

pub(crate) fn unresolved_attachment_error() -> LlmError {
    LlmError::UnsupportedCapability {
        message: "app attachment reached a wire codec before resolution".into(),
    }
}

pub(crate) fn provider_file_protocol_error() -> LlmError {
    LlmError::UnsupportedCapability {
        message: "provider file reference does not match the active protocol".into(),
    }
}

// One `WireCodec` per protocol family. The codec owns request encoding,
// response decoding, error classification (gate 31: every family's overflow
// becomes `LlmError::ContextOverflow`) and the stream decoder.
pub trait WireCodec: Send + Sync + 'static {
    /// Validate control settings before uploads or authentication have side effects.
    fn validate_request(
        &self,
        _req: &ChatRequest,
        _context: &CodecContext,
    ) -> Result<(), LlmError> {
        Ok(())
    }
    /// Request metadata for pricing and reporting. Built-in codecs also read native body defaults.
    fn request_inference(
        &self,
        req: &ChatRequest,
        _context: &CodecContext,
    ) -> Result<crate::protocol::InferenceReport, LlmError> {
        Ok(crate::protocol::InferenceReport {
            requested_effort: req.thinking.as_ref().and_then(|thinking| thinking.effort),
            requested_service_tier: req.service_tier,
            ..Default::default()
        })
    }
    fn family(&self) -> ProtocolFamily;
    fn encode_request(
        &self,
        req: EncodeRequest<'_>,
        context: &CodecContext,
    ) -> Result<HttpRequest, LlmError>;
    fn encoded_body_len(
        &self,
        req: EncodeRequest<'_>,
        context: &CodecContext,
    ) -> Result<usize, LlmError> {
        self.encode_request(req, context)
            .map(|request| request.body.len())
    }
    fn decode_response(
        &self,
        response: &HttpResponse,
        context: &CodecContext,
    ) -> Result<ChatResponse, LlmError>;
    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder>;
}

/// Stateful decoder for arbitrary transport byte chunks. A batch may contain
/// successful events followed by one error; consumers must preserve that order.
pub trait StreamDecoder: Send {
    fn inference_report(&self) -> crate::protocol::InferenceReport {
        Default::default()
    }
    fn set_response_headers(&mut self, _headers: &[(String, String)]) {}
    fn push_bytes(&mut self, bytes: &[u8]) -> Vec<Result<StreamEvent, LlmError>>;
    fn finish(&mut self) -> Vec<Result<StreamEvent, LlmError>>;
    fn usage_report(&self) -> crate::protocol::UsageReport;
}

/// Protocol event parsing is internal; framing belongs to the codec.
pub(crate) trait EventDecoder: Send {
    fn inference_report(&self) -> crate::protocol::InferenceReport {
        Default::default()
    }
    fn set_response_headers(&mut self, _headers: &[(String, String)]) {}
    fn decode_frame(&mut self, frame: &[u8]) -> Result<Vec<StreamEvent>, LlmError>;
    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError>;
    fn usage_report(&self) -> crate::protocol::UsageReport;
}

const _: Option<&dyn WireCodec> = None;
const _: Option<&dyn StreamDecoder> = None;

/// Every codec compiled into this crate. The remaining protocol families
/// arrive with their ports; until then `build()` refuses a profile that names
/// one, and says which (gate 33).
pub(crate) fn builtin() -> Vec<Arc<dyn WireCodec>> {
    vec![
        Arc::new(crate::codecs::anthropic::AnthropicMessagesCodec),
        Arc::new(crate::codecs::gemini::GeminiCodec),
        Arc::new(crate::codecs::hosted::AzureOpenAiCodec),
        Arc::new(crate::codecs::hosted::BedrockClaudeCodec),
        Arc::new(crate::codecs::hosted::FoundryClaudeCodec),
        Arc::new(crate::codecs::hosted::VertexClaudeCodec),
        Arc::new(crate::codecs::hosted::VertexGeminiCodec),
        Arc::new(crate::codecs::openai::chat::OpenAiChatCodec),
        Arc::new(crate::codecs::openai::responses::OpenAiResponsesCodec),
    ]
}

pub(crate) mod structured;

pub(crate) mod cache;
pub(crate) mod qwen_cache;
