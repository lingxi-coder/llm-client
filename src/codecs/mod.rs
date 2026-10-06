//! The wire codecs this crate ships.
//!
//! Everything that talks to a provider lives here rather than in a capability
//! crate: a codec is not something a profile turns on, it is how this crate
//! speaks a protocol family. `LlmClientBuilder` seeds its table from these, so
//! a profile naming an OpenAI-compatible provider needs no capability entry —
//! only a `ProviderProfile` (gate 30).

pub(crate) mod inference;
pub(crate) mod input;
pub(crate) mod json;
pub(crate) mod request_controls;
pub use input::{CodecContext, ContentBinding, EncodeRequest, PreparedMedia, RequestMode};
pub mod anthropic;
pub(crate) mod file_search_decode;
pub mod gemini;
pub mod openai;
pub(crate) mod stream;
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

/// These codecs have no typed native content path. Refuse the computer
/// declaration as well, so a first request cannot silently lose its tool.
pub(crate) fn reject_typed_native(
    req: &ChatRequest,
    family: ProtocolFamily,
) -> Result<(), LlmError> {
    use crate::providers::openai::computer::OpenAiComputerToolConfig;
    if req
        .native_options
        .iter()
        .any(|value| value.is::<OpenAiComputerToolConfig>())
        || req
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|block| matches!(block, crate::protocol::ContentBlock::Native { .. }))
    {
        return Err(LlmError::UnsupportedCapability {
            message: format!(
                "{family:?} cannot encode typed native content or the OpenAI computer tool"
            ),
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
    /// Extract accounting facts independently of content or HTTP-status errors.
    fn response_usage(
        &self,
        response: &HttpResponse,
        context: &CodecContext,
    ) -> crate::protocol::UsageReport {
        usage::response_report(response, context)
    }
    /// Infer actual execution tier even when semantic decoding fails.
    fn response_inference(
        &self,
        response: &HttpResponse,
        context: &CodecContext,
    ) -> crate::protocol::InferenceReport {
        inference::response(response, context.profile())
    }

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
    fn set_response_status(&mut self, _status: u16) {}
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
    fn set_response_status(&mut self, _status: u16) {}
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
        Arc::new(crate::hosting::AzureOpenAiCodec),
        Arc::new(crate::hosting::BedrockClaudeCodec),
        Arc::new(crate::hosting::FoundryClaudeCodec),
        Arc::new(crate::hosting::VertexClaudeCodec),
        Arc::new(crate::hosting::VertexGeminiCodec),
        Arc::new(crate::codecs::openai::chat::OpenAiChatCodec),
        Arc::new(crate::codecs::openai::responses::OpenAiResponsesCodec),
    ]
}

pub(crate) mod structured;

pub(crate) mod cache;
