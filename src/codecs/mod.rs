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
pub mod gemini_interactions;
pub mod openai;
pub(crate) mod stream;
pub(crate) mod tool_result_media;
pub(crate) mod usage;
pub(crate) mod web_search;
pub(crate) mod web_search_decode;

use crate::protocol::{
    ChatRequest, ChatResponse, LlmError, ProtocolFamily, ProviderFileSource, ProviderProfile,
    StreamEvent,
};
use crate::transport::{HttpRequest, HttpResponse};

use std::sync::Arc;

/// Retain the wire-field shapes that were enforced by the original typed
/// string/array carriers, before uploads or authentication can have effects.
pub(crate) fn validate_tool_wire_shapes(request: &ChatRequest) -> Result<(), LlmError> {
    if let Some(model_source) = &request.model_source {
        model_source.validate_root_string("ChatRequest.model_source")?;
    }
    for tool in &request.tools {
        if !tool.input_schema.is_root_object() {
            return Err(LlmError::InvalidRequest {
                message: "tool input schema must be a JSON object".into(),
            });
        }
    }
    for block in request.messages.iter().flat_map(|message| &message.content) {
        if let crate::protocol::ContentBlock::ToolResult {
            content, blocks, ..
        } = block
        {
            content.validate_root_string("tool result content")?;
            if let Some(blocks) = blocks {
                blocks.validate_root_array("tool result blocks")?;
            }
        }
    }
    Ok(())
}

/// A source-backed model id is meaningful only to the Anthropic Messages
/// codec family. Other providers keep their configured model strategy and
/// must reject this carrier rather than silently ignoring or coercing it.
pub(crate) fn validate_model_source_protocol(
    request: &ChatRequest,
    family: ProtocolFamily,
) -> Result<(), LlmError> {
    if let Some(source) = &request.model_source {
        if !matches!(
            family,
            ProtocolFamily::AnthropicMessages
                | ProtocolFamily::VertexClaude
                | ProtocolFamily::BedrockClaude
                | ProtocolFamily::FoundryClaude
        ) {
            return Err(LlmError::UnsupportedCapability {
                message: format!("model_source is not supported by {family:?}"),
            });
        }
        if family == ProtocolFamily::BedrockClaude {
            let normalized = crate::codecs::anthropic::encode::normalized_model_source(source)?;
            let units = normalized
                .root_string_code_units()
                .expect("normalized model source has a JSON string root");
            if String::from_utf16(&units).is_err() {
                return Err(LlmError::InvalidRequest {
                    // Native's Bedrock SDK applies encodeURIComponent while
                    // building `/model/{model}/invoke`; an isolated surrogate
                    // throws this URIError before a request reaches transport.
                    message: "URIError: URI malformed".to_owned(),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod model_source_protocol_tests {
    use super::*;

    #[test]
    fn exact_model_source_is_limited_to_claude_wire_families() {
        let mut request = ChatRequest::new("configured-route");
        request.model_source = Some(crate::protocol::JsonPayload::string_from_utf16_code_units(
            &[0xD800],
        ));
        for family in [
            ProtocolFamily::AnthropicMessages,
            ProtocolFamily::FoundryClaude,
        ] {
            assert!(validate_model_source_protocol(&request, family).is_ok());
        }
        assert!(validate_model_source_protocol(&request, ProtocolFamily::VertexClaude).is_ok());
        assert_eq!(
            validate_model_source_protocol(&request, ProtocolFamily::BedrockClaude).unwrap_err(),
            LlmError::InvalidRequest {
                message: "URIError: URI malformed".to_owned(),
            }
        );
        for family in [
            ProtocolFamily::OpenAiChat,
            ProtocolFamily::OpenAiResponses,
            ProtocolFamily::GeminiGenerateContent,
            ProtocolFamily::GeminiInteractions,
            ProtocolFamily::VertexGemini,
            ProtocolFamily::AzureOpenAi,
        ] {
            assert!(validate_model_source_protocol(&request, family).is_err());
        }

        request.model_source = Some(crate::protocol::JsonPayload::string("claude-sonnet-5-5"));
        for family in [
            ProtocolFamily::AnthropicMessages,
            ProtocolFamily::VertexClaude,
            ProtocolFamily::BedrockClaude,
            ProtocolFamily::FoundryClaude,
        ] {
            assert!(validate_model_source_protocol(&request, family).is_ok());
        }
        request.model_source = None;
        for family in ProtocolFamily::ALL {
            assert!(validate_model_source_protocol(&request, family).is_ok());
        }
    }
}

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
    if req.native_options.iter().any(|value| {
        value.is::<OpenAiComputerToolConfig>()
            || value.is::<crate::providers::google::computer::GeminiComputerToolConfig>()
    }) || req
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
    /// Preserve HTTP status classification when a bounded error body is not
    /// valid JSON. Successful responses still use the strict decoder.
    fn decode_error_response(&self, resp: &HttpResponse, context: &CodecContext) -> LlmError {
        let sanitized;
        // Native Anthropic formats an empty HTTP error body as
        // "<status> status code (no body)". Let that codec see the original
        // empty response so it can apply its route-specific source behavior.
        // Keep the bounded non-JSON fallback for every non-Native route and
        // for non-empty malformed bodies.
        let native_empty_error_body =
            resp.body.is_empty() && context.native_anthropic_error_route().is_some();
        let response = if !native_empty_error_body
            && crate::response_json::ResponseJsonScanner::parse(
                &resp.body,
                "invalid HTTP error JSON",
            )
            .is_err()
        {
            sanitized = HttpResponse {
                status: resp.status,
                headers: resp.headers.clone(),
                body: serde_json::to_vec(&serde_json::json!({"error":{"message":format!("HTTP {} returned a non-JSON or truncated error body", resp.status)}})).expect("error envelope serializes").into(),
            };
            &sanitized
        } else {
            resp
        };
        self.decode_response(response, context)
            .err()
            .unwrap_or_else(|| LlmError::ProviderInternal {
                message: format!("request failed with HTTP {}", resp.status),
            })
    }
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
    /// SDK-derived Native Anthropic resumability fact, when usage and an
    /// authoritative context source are available.
    fn native_resume_context_exhausted(&self) -> Option<bool> {
        None
    }
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
    fn native_resume_context_exhausted(&self) -> Option<bool> {
        None
    }
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
        Arc::new(crate::codecs::gemini_interactions::GeminiInteractionsCodec),
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

#[cfg(test)]
mod native_empty_error_body_tests {
    use super::*;
    use crate::protocol::LlmErrorKind;
    use serde_json::json;

    fn profile(provider_id: &str, protocol: &str) -> ProviderProfile {
        serde_json::from_value(json!({
            "provider_id": provider_id,
            "profile_name": "direct",
            "base_url": "https://api.anthropic.com",
            "protocol": protocol,
            "auth": "none",
            "models": [{
                "display_model": "claude-sonnet-4-6",
                "request_model": "claude-sonnet-4-6",
                "billing_model": "claude-sonnet-4-6"
            }]
        }))
        .expect("fixture profile")
    }

    fn empty_error() -> HttpResponse {
        HttpResponse {
            status: 400,
            headers: Vec::new(),
            body: Vec::new().into(),
        }
    }

    #[test]
    fn native_anthropic_empty_error_body_reaches_decoder_unchanged() {
        let profile = profile("anthropic", "anthropic_messages");
        let context = CodecContext::new(&profile, "claude-sonnet-4-6", RequestMode::Complete);
        let codec = anthropic::AnthropicMessagesCodec;

        let error = codec.decode_error_response(&empty_error(), &context);
        assert_eq!(error.kind(), LlmErrorKind::InvalidRequest);
        assert_eq!(
            error.provider_error_text().map(|text| text.display()),
            Some("400 status code (no body)")
        );
    }

    #[test]
    fn malformed_nonempty_native_body_still_uses_bounded_error_fallback() {
        let profile = profile("anthropic", "anthropic_messages");
        let context = CodecContext::new(&profile, "claude-sonnet-4-6", RequestMode::Complete);
        let codec = anthropic::AnthropicMessagesCodec;
        let response = HttpResponse {
            status: 400,
            headers: Vec::new(),
            body: b"{truncated".to_vec().into(),
        };

        let error = codec.decode_error_response(&response, &context);
        assert_eq!(error.kind(), LlmErrorKind::InvalidRequest);
        let display = error
            .provider_error_text()
            .map(|text| text.display())
            .expect("fallback keeps provider error text");
        assert!(display.contains("HTTP 400 returned a non-JSON or truncated error body"));
    }

    #[test]
    fn other_provider_empty_errors_keep_status_taxonomy_and_existing_fallback() {
        let profile = profile("openai", "open_ai_chat");
        let context = CodecContext::new(&profile, "claude-sonnet-4-6", RequestMode::Complete);
        let codec = openai::chat::OpenAiChatCodec;
        let response = empty_error();

        // The OpenAI decoder can classify the original empty body from its
        // status without losing the provider's 400 taxonomy.
        let raw_error = codec.decode_response(&response, &context).unwrap_err();
        assert_eq!(raw_error.kind(), LlmErrorKind::InvalidRequest);

        // The shared default wrapper remains unchanged for non-Native routes.
        let wrapped_error = codec.decode_error_response(&response, &context);
        assert_eq!(wrapped_error.kind(), LlmErrorKind::InvalidRequest);
        assert_eq!(
            wrapped_error
                .provider_error_text()
                .map(|text| text.display()),
            Some("400 HTTP 400 returned a non-JSON or truncated error body")
        );
    }
}
