//! The hosted variants: the same two wires, somewhere else.
//!
//! Four of the nine protocol families are not new wires at all. They are the
//! Anthropic Messages or Gemini `generateContent` body, reached through a
//! cloud's own URL layout and credential. Ported from the previous project's
//! four hosted-variant modules, each a thin wrapper for the same reason.
//!
//! Keeping them as wrappers rather than copies is the point: a fix to the
//! Anthropic encoder reaches Bedrock, Vertex and Foundry without being applied
//! four times, and a divergence between them can only be deliberate.
//!
//! Authentication is not here. The codec builds the request; the
//! `Authenticator` selected by `profile.auth` attaches the credential, which is
//! the only difference several of these have left.

use crate::codecs::{anthropic::AnthropicMessagesCodec, gemini, openai::chat::OpenAiChatCodec};
use crate::codecs::{CodecContext, EncodeRequest};
use crate::codecs::{StreamDecoder, WireCodec};
use crate::protocol::{ChatResponse, LlmError, ProtocolFamily};
use crate::transport::{HttpRequest, HttpResponse};

use serde_json::Value;

/// Azure OpenAI: the Chat Completions body at a deployment URL.
///
/// The deployment is the wire model, and it goes in the path — so `model` is
/// removed from the body, where Azure rejects it.
#[derive(Debug, Default, Clone, Copy)]
pub struct AzureOpenAiCodec;

impl WireCodec for AzureOpenAiCodec {
    fn request_inference(
        &self,
        req: &crate::protocol::ChatRequest,
        context: &CodecContext,
    ) -> Result<crate::protocol::InferenceReport, LlmError> {
        crate::codecs::inference::requested(req, context)
    }
    fn validate_request(
        &self,
        req: &crate::protocol::ChatRequest,
        context: &CodecContext,
    ) -> Result<(), LlmError> {
        crate::codecs::validate_tool_wire_shapes(req)?;
        crate::providers::openrouter::server_tools::validate(req, context.profile(), None, false)?;
        crate::codecs::inference::validate(req, context)?;
        crate::providers::google::hosted_tools::validate_hosted_tool_request(
            req,
            context.profile(),
            context.request_model(),
        )
    }
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::AzureOpenAi
    }

    fn encode_request(
        &self,
        req: EncodeRequest<'_>,
        opts: &CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let profile = opts.profile();
        let mut http = crate::codecs::openai::chat::encode::request(req, profile, opts)?;
        let api_version = profile
            .azure
            .as_ref()
            .and_then(|a| a.api_version.as_deref())
            .ok_or_else(|| LlmError::InvalidRequest {
                message: format!(
                    "profile {:?} is an Azure deployment but names no azure.api_version",
                    profile.profile_name
                ),
            })?;
        // An explicit deployment wins; otherwise the wire model is the
        // deployment name, which is how Azure profiles are usually written.
        let deployment = profile
            .azure
            .as_ref()
            .and_then(|a| a.deployment.as_deref())
            .unwrap_or(&opts.request_model);
        http.http.url = format!(
            "{}/openai/deployments/{deployment}/chat/completions?api-version={api_version}",
            profile.base_url.trim_end_matches('/')
        );
        strip_body_key(&mut http, "model")?;
        http.encode()
    }

    fn decode_response(
        &self,
        resp: &HttpResponse,
        context: &CodecContext,
    ) -> Result<ChatResponse, LlmError> {
        OpenAiChatCodec.decode_response(resp, context)
    }

    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        OpenAiChatCodec.stream_decoder(context)
    }
}

/// The Anthropic Messages wire under an Azure AI Foundry endpoint.
/// The body is identical; only the base URL and the credential differ.
#[derive(Debug, Default, Clone, Copy)]
pub struct FoundryClaudeCodec;

impl WireCodec for FoundryClaudeCodec {
    fn request_inference(
        &self,
        req: &crate::protocol::ChatRequest,
        context: &CodecContext,
    ) -> Result<crate::protocol::InferenceReport, LlmError> {
        crate::codecs::inference::requested(req, context)
    }
    fn validate_request(
        &self,
        req: &crate::protocol::ChatRequest,
        context: &CodecContext,
    ) -> Result<(), LlmError> {
        crate::codecs::validate_tool_wire_shapes(req)?;
        AnthropicMessagesCodec.validate_request(req, context)
    }
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::FoundryClaude
    }

    fn encode_request(
        &self,
        req: EncodeRequest<'_>,
        opts: &CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let _profile = opts.profile();
        AnthropicMessagesCodec.encode_request(req, opts)
    }

    fn decode_response(
        &self,
        resp: &HttpResponse,
        context: &CodecContext,
    ) -> Result<ChatResponse, LlmError> {
        AnthropicMessagesCodec.decode_response(resp, context)
    }

    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        AnthropicMessagesCodec.stream_decoder(context)
    }
}

/// The Anthropic Messages body at a Vertex AI publisher URL.
///
/// Two differences beyond the path. The model is in the URL, not the body. And
/// `anthropic_version` moves from a header into the body — Vertex reads it
/// there, and a request that keeps it in the header is rejected for missing it.
#[derive(Debug, Default, Clone, Copy)]
pub struct VertexClaudeCodec;

/// Vertex Claude expects this platform-specific version in the body when a
/// profile has not pinned its own `api_version`.
const VERTEX_ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

impl WireCodec for VertexClaudeCodec {
    fn request_inference(
        &self,
        req: &crate::protocol::ChatRequest,
        context: &CodecContext,
    ) -> Result<crate::protocol::InferenceReport, LlmError> {
        crate::codecs::inference::requested(req, context)
    }
    fn validate_request(
        &self,
        req: &crate::protocol::ChatRequest,
        context: &CodecContext,
    ) -> Result<(), LlmError> {
        crate::codecs::validate_tool_wire_shapes(req)?;
        AnthropicMessagesCodec.validate_request(req, context)
    }
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::VertexClaude
    }

    fn encode_request(
        &self,
        req: EncodeRequest<'_>,
        opts: &CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let source_context = req
            .request()
            .model_source
            .as_ref()
            .map(|source| opts.clone().with_model_source(Some(source), opts.profile()))
            .transpose()?;
        let opts = source_context.as_ref().unwrap_or(opts);
        let profile = opts.profile();
        crate::codecs::validate_model_source_protocol(req.request(), ProtocolFamily::VertexClaude)?;
        let action = if opts.stream {
            "streamRawPredict"
        } else {
            "rawPredict"
        };
        let source_model = req
            .request()
            .model_source
            .as_ref()
            .map(crate::codecs::anthropic::encode::vertex_url_path_model_source)
            .transpose()?;
        let path_model = source_model
            .clone()
            .unwrap_or_else(|| opts.request_model.clone());
        let mut http = crate::codecs::anthropic::encode::request(req, profile, opts)?;
        if source_model.is_some() {
            // Native assigns the complete model endpoint to URL.pathname. Do
            // the same here so source `?` and `#` remain path data instead of
            // changing the request's query or fragment. The ordinary route
            // model keeps its existing URL construction below.
            let mut url =
                url::Url::parse(&profile.base_url).map_err(|error| LlmError::InvalidRequest {
                    message: format!("Invalid Vertex base URL: {error}"),
                })?;
            let path = format!(
                "{}/publishers/anthropic/models/{}:{action}",
                url.path().trim_end_matches('/'),
                path_model
            );
            url.set_path(&path);
            http.http.url = url.into();
        } else {
            http.http.url = format!(
                "{}/publishers/anthropic/models/{}:{action}",
                profile.base_url.trim_end_matches('/'),
                path_model
            );
        }
        let version = profile
            .extra
            .get("api_version")
            .and_then(Value::as_str)
            .unwrap_or(VERTEX_ANTHROPIC_VERSION)
            .to_owned();
        edit_body(&mut http, |body| {
            body.remove("model");
            body.insert("anthropic_version".to_owned(), Value::String(version));
        })?;
        http.encode()
    }

    fn decode_response(
        &self,
        resp: &HttpResponse,
        context: &CodecContext,
    ) -> Result<ChatResponse, LlmError> {
        AnthropicMessagesCodec.decode_response(resp, context)
    }

    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        AnthropicMessagesCodec.stream_decoder(context)
    }
}

/// Vertex AI Gemini: the `generateContent` body at a Vertex publisher URL.
#[derive(Debug, Default, Clone, Copy)]
pub struct VertexGeminiCodec;

impl WireCodec for VertexGeminiCodec {
    fn request_inference(
        &self,
        req: &crate::protocol::ChatRequest,
        context: &CodecContext,
    ) -> Result<crate::protocol::InferenceReport, LlmError> {
        crate::codecs::inference::requested(req, context)
    }
    fn validate_request(
        &self,
        req: &crate::protocol::ChatRequest,
        context: &CodecContext,
    ) -> Result<(), LlmError> {
        crate::codecs::validate_tool_wire_shapes(req)?;
        crate::providers::openrouter::server_tools::validate(req, context.profile(), None, false)?;
        gemini::encode::validate_audio_input(req)?;
        crate::codecs::inference::validate(req, context)?;
        crate::providers::google::hosted_tools::validate_hosted_tool_request(
            req,
            context.profile(),
            context.request_model(),
        )
    }
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::VertexGemini
    }

    fn encode_request(
        &self,
        req: EncodeRequest<'_>,
        opts: &CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let profile = opts.profile();
        let url = gemini::generate_content_url(
            &format!(
                "{}/publishers/google",
                profile.base_url.trim_end_matches('/')
            ),
            &opts.request_model,
            opts.stream,
        );
        gemini::encode::request_to(req, profile, &url, opts)?.encode()
    }

    fn decode_response(
        &self,
        resp: &HttpResponse,
        context: &CodecContext,
    ) -> Result<ChatResponse, LlmError> {
        GeminiLike.decode_response(resp, context)
    }

    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        GeminiLike.stream_decoder(context)
    }
}

use crate::codecs::gemini::GeminiCodec as GeminiLike;

fn edit_body(
    http: &mut crate::codecs::json::WireRequest<'_>,
    f: impl FnOnce(&mut crate::codecs::json::WireValue<'_>),
) -> Result<(), LlmError> {
    f(&mut http.body);
    Ok(())
}
fn strip_body_key(
    http: &mut crate::codecs::json::WireRequest<'_>,
    key: &str,
) -> Result<(), LlmError> {
    http.body.remove(key);
    Ok(())
}

/// The Anthropic Messages body on Amazon Bedrock: signed with SigV4 and
/// streamed as AWS event-stream frames rather than SSE.
///
/// Differences from the first-party wire:
/// the model is in the URL path; `model` leaves the body; `anthropic_version`
/// is added to the body as Bedrock's own constant. The shared Messages SDK
/// version header remains, as in native Claude Code. The endpoint action chooses
/// streaming, so the first-party `stream` body field is removed too.
#[derive(Debug, Default, Clone, Copy)]
pub struct BedrockClaudeCodec;

/// The version string Bedrock expects in the body — its own, not the
/// first-party API's date.
const BEDROCK_ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

impl WireCodec for BedrockClaudeCodec {
    fn request_inference(
        &self,
        req: &crate::protocol::ChatRequest,
        context: &CodecContext,
    ) -> Result<crate::protocol::InferenceReport, LlmError> {
        crate::codecs::inference::requested(req, context)
    }
    fn validate_request(
        &self,
        req: &crate::protocol::ChatRequest,
        context: &CodecContext,
    ) -> Result<(), LlmError> {
        let source_context = req
            .model_source
            .as_ref()
            .map(|source| {
                context
                    .clone()
                    .with_model_source(Some(source), context.profile())
            })
            .transpose()?;
        let context = source_context.as_ref().unwrap_or(context);
        crate::codecs::validate_tool_wire_shapes(req)?;
        crate::providers::openrouter::server_tools::validate(req, context.profile(), None, false)?;
        crate::codecs::inference::validate(req, context)
    }
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::BedrockClaude
    }

    fn encode_request(
        &self,
        req: EncodeRequest<'_>,
        opts: &CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let source_context = req
            .request()
            .model_source
            .as_ref()
            .map(|source| opts.clone().with_model_source(Some(source), opts.profile()))
            .transpose()?;
        let opts = source_context.as_ref().unwrap_or(opts);
        let profile = opts.profile();
        crate::codecs::validate_model_source_protocol(
            req.request(),
            ProtocolFamily::BedrockClaude,
        )?;
        let mut http = crate::codecs::anthropic::encode::request(req, profile, opts)?;
        let action = if opts.stream {
            "invoke-with-response-stream"
        } else {
            "invoke"
        };
        http.http.url = bedrock_model_url(&profile.base_url, &opts.request_model, action)?;
        edit_body(&mut http, |body| {
            body.remove("model");
            // Bedrock selects streaming through the invoke-with-response-stream
            // endpoint; Anthropic's first-party `stream` field is not part of
            // the Bedrock request body.
            body.remove("stream");
            body.insert(
                "anthropic_version".to_owned(),
                Value::String(BEDROCK_ANTHROPIC_VERSION.to_owned()),
            );
        })?;
        http.encode()
    }

    fn decode_response(
        &self,
        resp: &HttpResponse,
        context: &CodecContext,
    ) -> Result<ChatResponse, LlmError> {
        AnthropicMessagesCodec.decode_response(resp, context)
    }

    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        Box::new(BedrockStreamDecoder {
            inner: Box::new(
                crate::codecs::anthropic::stream::AnthropicStreamDecoder::configured(context),
            ),
            ..BedrockStreamDecoder::default()
        })
    }
}

/// Construct Bedrock's model endpoint while keeping the model ID in one URL
/// path segment. ARNs use `/` inside their resource component, which must be
/// escaped so it cannot become another path separator; `Url` leaves the ARN's
/// colons intact.
fn bedrock_model_url(base_url: &str, model_id: &str, action: &str) -> Result<String, LlmError> {
    let mut url =
        url::Url::parse(base_url.trim_end_matches('/')).map_err(|e| LlmError::InvalidRequest {
            message: format!("Bedrock base URL is invalid: {e}"),
        })?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| LlmError::InvalidRequest {
                message: "Bedrock base URL cannot accept path segments".to_owned(),
            })?;
        segments
            .pop_if_empty()
            .push("model")
            .push(model_id)
            .push(action);
    }
    let mut url: String = url.into();
    crate::providers::anthropic::request_policy::remove_beta_query(&mut url)?;
    Ok(url)
}

/// Unwraps AWS event-stream frames and feeds what is inside to the Anthropic
/// decoder, which is the same JSON it would have received over SSE.
struct BedrockStreamDecoder {
    ended: bool,
    frames: crate::framing::eventstream::EventStreamSplitter,
    inner: Box<dyn crate::codecs::EventDecoder>,
}

impl Default for BedrockStreamDecoder {
    fn default() -> Self {
        Self {
            ended: false,
            frames: crate::framing::eventstream::EventStreamSplitter::new(),
            inner: Box::new(crate::codecs::anthropic::stream::AnthropicStreamDecoder::default()),
        }
    }
}

impl BedrockStreamDecoder {
    fn decode_message(
        &mut self,
        msg: &crate::framing::eventstream::EventStreamMessage,
    ) -> Result<Vec<StreamEvent>, LlmError> {
        let mut out = Vec::new();
        match msg.header(":message-type") {
            Some("event") => {
                // The payload is `{"bytes": "<base64 of the event JSON>"}`.
                let payload: Value = serde_json::from_slice(&msg.payload).map_err(|e| {
                    LlmError::StreamInterrupted {
                        message: format!("event-stream payload is not JSON: {e}"),
                    }
                })?;
                let b64 = payload
                    .get("bytes")
                    .and_then(Value::as_str)
                    .ok_or_else(|| LlmError::StreamInterrupted {
                        message: "event-stream payload has no `bytes` field".to_owned(),
                    })?;
                let json = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|e| LlmError::StreamInterrupted {
                        message: format!("event-stream `bytes` is not base64: {e}"),
                    })?;
                out.extend(self.inner.decode_frame(&json)?);
            }
            Some("exception" | "error") => {
                let kind = msg.header(":exception-type").unwrap_or("unknown");
                let body = String::from_utf8_lossy(&msg.payload);
                return Err(LlmError::StreamInterrupted {
                    message: format!("provider stream exception {kind}: {body}"),
                });
            }
            // Other message types carry no model output.
            _ => {}
        }
        Ok(out)
    }
    fn append(
        &mut self,
        result: Result<Vec<StreamEvent>, LlmError>,
        out: &mut Vec<Result<StreamEvent, LlmError>>,
    ) {
        match result {
            Ok(events) => {
                for event in events {
                    let terminal = matches!(event, StreamEvent::End { .. });
                    out.push(Ok(event));
                    if terminal {
                        self.ended = true;
                        break;
                    }
                }
            }
            Err(error) => {
                self.ended = true;
                out.push(Err(error));
            }
        }
    }
}

#[cfg(test)]
mod model_source_path_tests {
    use super::*;
    use crate::codecs::{CodecContext, EncodeRequest, RequestMode};
    use crate::protocol::{ChatRequest, JsonPayload, ProviderProfile};
    use serde_json::json;

    fn profile(protocol: &str, base_url: &str) -> ProviderProfile {
        serde_json::from_value(json!({
            "provider_id": "anthropic",
            "profile_name": "route",
            "base_url": base_url,
            "protocol": protocol,
            "auth": "none",
            "models": [{
                "display_model": "route-model",
                "request_model": "route-model",
                "billing_model": "route-model"
            }]
        }))
        .expect("fixture profile")
    }

    fn request(source: JsonPayload) -> ChatRequest {
        let mut request = ChatRequest::new("configured-route");
        request.max_tokens = Some(4096);
        request.model_source = Some(source);
        request
    }

    #[test]
    fn valid_source_model_replaces_bedrock_and_vertex_path_identity() {
        let source = JsonPayload::string("source-model");
        let bedrock = profile(
            "bedrock_claude",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
        );
        let bedrock_context =
            CodecContext::for_model(&bedrock, &bedrock.models[0], RequestMode::Complete);
        let bedrock_wire = BedrockClaudeCodec
            .encode_request(
                EncodeRequest::new(&request(source.clone())),
                &bedrock_context,
            )
            .expect("well-formed Bedrock model id");
        assert_eq!(
            bedrock_wire.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/source-model/invoke"
        );

        let vertex = profile(
            "vertex_claude",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1",
        );
        let vertex_context =
            CodecContext::for_model(&vertex, &vertex.models[0], RequestMode::Complete);
        let vertex_wire = VertexClaudeCodec
            .encode_request(EncodeRequest::new(&request(source)), &vertex_context)
            .expect("well-formed Vertex model id");
        assert!(vertex_wire
            .url
            .ends_with("/publishers/anthropic/models/source-model:rawPredict"));
    }

    #[test]
    fn bedrock_malformed_source_preserves_native_uri_error() {
        let source = JsonPayload::string_from_utf16_code_units(&[0xD800]);
        let profile = profile(
            "bedrock_claude",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
        );
        let context = CodecContext::for_model(&profile, &profile.models[0], RequestMode::Complete);
        assert_eq!(
            BedrockClaudeCodec
                .encode_request(EncodeRequest::new(&request(source)), &context)
                .unwrap_err(),
            LlmError::InvalidRequest {
                message: "URIError: URI malformed".to_owned(),
            }
        );
    }

    #[test]
    fn vertex_path_uses_whatwg_surrogate_replacement_without_route_fallback() {
        let source = JsonPayload::string_from_utf16_code_units(&[0xD800]);
        let profile = profile(
            "vertex_claude",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1",
        );
        let context = CodecContext::for_model(&profile, &profile.models[0], RequestMode::Complete);
        let wire = VertexClaudeCodec
            .encode_request(EncodeRequest::new(&request(source)), &context)
            .expect("Vertex URL pathname accepts a lone surrogate as U+FFFD");
        let url = url::Url::parse(&wire.url).expect("encoded Vertex request URL");
        assert!(url
            .path()
            .ends_with("/publishers/anthropic/models/%EF%BF%BD:rawPredict"));
        assert!(!url
            .path()
            .ends_with("/publishers/anthropic/models/route-model:rawPredict"));
    }

    #[test]
    fn vertex_source_model_uses_native_pathname_setter_semantics() {
        let profile = profile(
            "vertex_claude",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/?keep=1#fragment",
        );
        let context = CodecContext::for_model(&profile, &profile.models[0], RequestMode::Complete);
        let base =
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1";
        for (source, expected_path) in [
            ("x?y#z", "/publishers/anthropic/models/x%3Fy%23z:rawPredict"),
            ("x%raw", "/publishers/anthropic/models/x%raw:rawPredict"),
            ("x%2Fy", "/publishers/anthropic/models/x%2Fy:rawPredict"),
            ("x/y", "/publishers/anthropic/models/x/y:rawPredict"),
        ] {
            let wire = VertexClaudeCodec
                .encode_request(
                    EncodeRequest::new(&request(JsonPayload::string(source))),
                    &context,
                )
                .expect("valid source model is assigned as URL pathname");
            let url = url::Url::parse(&wire.url).expect("encoded Vertex request URL");
            assert!(url.path().ends_with(expected_path), "{}", url.path());
            assert_eq!(url.query(), Some("keep=1"), "{}", wire.url);
            assert_eq!(url.fragment(), Some("fragment"), "{}", wire.url);
            assert!(wire.url.starts_with(base), "{}", wire.url);
        }
    }
}
impl StreamDecoder for BedrockStreamDecoder {
    fn inference_report(&self) -> crate::protocol::InferenceReport {
        self.inner.inference_report()
    }
    fn set_response_headers(&mut self, headers: &[(String, String)]) {
        self.inner.set_response_headers(headers);
    }
    fn push_bytes(&mut self, bytes: &[u8]) -> Vec<Result<StreamEvent, LlmError>> {
        if self.ended {
            return vec![];
        }
        let mut out = vec![];
        let (frames, error) = self.frames.feed_batch(bytes);
        for frame in frames {
            let result = self.decode_message(&frame);
            self.append(result, &mut out);
            if self.ended {
                break;
            }
        }
        if !self.ended {
            if let Some(error) = error {
                self.append(Err(error), &mut out);
            }
        }
        out
    }
    fn finish(&mut self) -> Vec<Result<StreamEvent, LlmError>> {
        if self.ended {
            return vec![];
        }
        let mut out = vec![];
        let result = self.frames.finish().and_then(|()| self.inner.finish());
        self.append(result, &mut out);
        self.ended = true;
        out
    }
    fn usage_report(&self) -> crate::protocol::UsageReport {
        self.inner.usage_report()
    }
}

use crate::protocol::StreamEvent;
use base64::Engine;

pub(crate) mod foundry;

pub mod vertex;
