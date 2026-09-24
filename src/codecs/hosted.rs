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
use crate::protocol::{CompletionResponse, LlmError, ProtocolFamily};
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
        ProtocolFamily::AzureOpenAi
    }

    fn encode_request(
        &self,
        req: EncodeRequest<'_>,
        opts: &CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let profile = opts.profile();
        let mut http = super::openai::chat::encode::request(req, profile, opts)?;
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
    ) -> Result<CompletionResponse, LlmError> {
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
    ) -> Result<CompletionResponse, LlmError> {
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
        ProtocolFamily::VertexClaude
    }

    fn encode_request(
        &self,
        req: EncodeRequest<'_>,
        opts: &CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let profile = opts.profile();
        let mut http = super::anthropic::encode::request(req, profile, opts)?;
        let action = if opts.stream {
            "streamRawPredict"
        } else {
            "rawPredict"
        };
        http.http.url = format!(
            "{}/publishers/anthropic/models/{}:{action}",
            profile.base_url.trim_end_matches('/'),
            opts.request_model
        );
        let version = profile
            .extra
            .get("api_version")
            .and_then(Value::as_str)
            .unwrap_or(VERTEX_ANTHROPIC_VERSION)
            .to_owned();
        http.http.headers.retain(|(k, _)| k != "anthropic-version");
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
    ) -> Result<CompletionResponse, LlmError> {
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
    ) -> Result<CompletionResponse, LlmError> {
        GeminiLike.decode_response(resp, context)
    }

    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        GeminiLike.stream_decoder(context)
    }
}

use crate::codecs::gemini::GeminiCodec as GeminiLike;

fn edit_body(
    http: &mut crate::codecs::json::WireRequest<'_>,
    f: impl FnOnce(&mut serde_json::Map<String, Value>),
) -> Result<(), LlmError> {
    f(http.body.object_mut());
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
/// Four differences from the first-party wire, each load-bearing:
/// the model is in the URL path; `model` leaves the body; `anthropic_version`
/// moves into the body as Bedrock's own constant; and the `anthropic-version`
/// header is removed, because Bedrock rejects it. The endpoint action chooses
/// streaming, so the first-party `stream` body field is removed too.
#[derive(Debug, Default, Clone, Copy)]
pub struct BedrockClaudeCodec;

/// The version string Bedrock expects in the body — its own, not the
/// first-party API's date.
const BEDROCK_ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

impl WireCodec for BedrockClaudeCodec {
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
        ProtocolFamily::BedrockClaude
    }

    fn encode_request(
        &self,
        req: EncodeRequest<'_>,
        opts: &CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let profile = opts.profile();
        let mut http = super::anthropic::encode::request(req, profile, opts)?;
        let action = if opts.stream {
            "invoke-with-response-stream"
        } else {
            "invoke"
        };
        http.http.url = bedrock_model_url(&profile.base_url, &opts.request_model, action)?;
        http.http.headers.retain(|(k, _)| k != "anthropic-version");
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
    ) -> Result<CompletionResponse, LlmError> {
        AnthropicMessagesCodec.decode_response(resp, context)
    }

    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        Box::new(BedrockStreamDecoder {
            inner: Box::new(super::anthropic::stream::AnthropicStreamDecoder::configured(context)),
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
    Ok(url.into())
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
            inner: Box::new(super::anthropic::stream::AnthropicStreamDecoder::default()),
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
