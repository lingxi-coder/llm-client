//! The wire codecs this crate ships.
//!
//! Everything that talks to a provider lives here rather than in a capability
//! crate: a codec is not something a profile turns on, it is how this crate
//! speaks a protocol family. `LlmClientBuilder` seeds its table from these, so
//! a profile naming an OpenAI-compatible provider needs no capability entry —
//! only a `ProviderProfile` (gate 30).

pub mod anthropic;
pub(crate) mod extras;
pub(crate) mod file_search_decode;
pub mod gemini;
pub mod hosted;
pub mod openai;
pub(crate) mod web_search;
pub(crate) mod web_search_decode;

use crate::client::route::ResolvedRoute;
use crate::transport::{HttpRequest, HttpResponse};
use crate::RequestOptions;
use async_trait::async_trait;
use bytes::Bytes;
use lingxi_agent_api::protocol::{
    CompletionRequest, CompletionResponse, LlmError, ProtocolFamily, ProviderFileSource,
    ProviderProfile, StreamEvent, Usage,
};
use serde_json::Value;
use std::sync::Arc;

/// A Responses continuation id is endpoint-side state, not an optional hint.
/// Refuse it on every other wire rather than silently starting a fresh turn.
pub(crate) fn reject_responses_continuation(
    req: &CompletionRequest,
    family: ProtocolFamily,
) -> Result<(), LlmError> {
    if req.previous_response_id.is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: format!("{family:?} cannot continue a Responses id"),
        });
    }
    Ok(())
}

/// Validate that a provider-owned input reference belongs to the exact
/// connection and caller-supplied account scope used for this request.
pub(crate) fn validate_provider_file<'a>(
    file: &'a ProviderFileSource,
    profile: &ProviderProfile,
    opts: &RequestOptions,
) -> Result<&'a ProviderFileSource, LlmError> {
    let Some(active_account_scope) = opts
        .file_account_scope
        .as_deref()
        .filter(|scope| !scope.trim().is_empty())
    else {
        return Err(LlmError::UnsupportedCapability {
            message: "provider file inputs require an explicit account scope".into(),
        });
    };
    if file.protocol != profile.protocol
        || file.provider_id != profile.provider_id
        || file.profile_name != profile.profile_name
        || file.endpoint_fingerprint
            != crate::client::files::provider_file_endpoint_fingerprint(&profile.base_url)
        || file.account_scope.as_deref() != Some(active_account_scope)
    {
        return Err(LlmError::UnsupportedCapability {
            message: "provider file reference belongs to a different connection, endpoint, or account scope".into(),
        });
    }
    if file.file_id.trim().is_empty()
        || file.uri.as_deref().is_some_and(str::is_empty)
        || file.media_type.as_deref().is_some_and(str::is_empty)
    {
        return Err(LlmError::InvalidRequest {
            message: "provider file reference is missing a required identifier".into(),
        });
    }
    Ok(file)
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
    fn family(&self) -> ProtocolFamily;

    /// The codec is registered once per protocol family and shared by every
    /// profile using it, so the connection's own configuration — base URL,
    /// Azure api-version, signing region, provider-opaque `extra` — arrives as
    /// `profile` rather than being baked into the instance.
    fn encode_request(
        &self,
        req: &CompletionRequest,
        profile: &ProviderProfile,
        route: &ResolvedRoute,
        opts: &RequestOptions,
    ) -> Result<HttpRequest, LlmError>;

    /// A non-success response must come back as the matching `LlmError`
    /// variant, never as a stringified status.
    fn decode_response(&self, resp: &HttpResponse) -> Result<CompletionResponse, LlmError>;

    fn stream_decoder(&self) -> Box<dyn StreamDecoder>;

    fn response_usage(&self, resp: &HttpResponse) -> Option<Usage>;
}

/// Turns transport frames into provider-neutral events. Stateful per response.
pub trait StreamDecoder: Send {
    fn decode_frame(&mut self, frame: &[u8]) -> Result<Vec<StreamEvent>, LlmError>;

    /// End of stream. Emits whatever is still buffered (the final `End`).
    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError>;

    fn observed_usage(&self) -> Option<Usage>;

    /// Whether the observed usage is a complete, self-consistent report — a
    /// measurement rather than a partial one.
    ///
    /// Separate from `observed_usage` because the counts are still the best
    /// thing to show a user either way; what changes is whether anything may be
    /// billed or budgeted from them. A stream cut off mid-response, or a
    /// provider whose subtotals do not reconcile, answers `false`.
    fn usage_is_complete(&self) -> bool;

    fn set_provider_metadata(&mut self, meta: Value);
}

/// A framed byte stream (SSE, NDJSON, WebSocket) the decoder pulls from.
#[async_trait]
pub trait FrameStream: Send {
    async fn next_frame(&mut self) -> Result<Option<Bytes>, LlmError>;
}

// Gate 3.
const _: Option<&dyn WireCodec> = None;
const _: Option<&dyn StreamDecoder> = None;
const _: Option<&dyn FrameStream> = None;

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
