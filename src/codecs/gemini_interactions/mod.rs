//! Unified chat and streaming wire for Gemini Interactions.
//! Pure codecs reuse the provider DTO encoder; they never call its service.
pub(crate) mod decode;
mod encode;
mod stream;
use crate::codecs::{CodecContext, EncodeRequest, StreamDecoder, WireCodec};
use crate::protocol::{ChatRequest, ChatResponse, LlmError, ProtocolFamily};
use crate::response_json::{ResponseJson, ResponseJsonScanner};
use crate::transport::{HttpRequest, HttpResponse};

#[derive(Debug, Default, Clone, Copy)]
pub struct GeminiInteractionsCodec;
impl WireCodec for GeminiInteractionsCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::GeminiInteractions
    }
    fn validate_request(
        &self,
        request: &ChatRequest,
        context: &CodecContext,
    ) -> Result<(), LlmError> {
        encode::validate(request, context)
    }
    fn encode_request(
        &self,
        request: EncodeRequest<'_>,
        context: &CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        encode::request(request, context)
    }
    fn decode_response(
        &self,
        response: &HttpResponse,
        context: &CodecContext,
    ) -> Result<ChatResponse, LlmError> {
        if !(200..300).contains(&response.status) {
            let scanner = match ResponseJsonScanner::parse(
                &response.body,
                "Gemini interaction error is not valid JSON",
            ) {
                Ok(scanner) => scanner,
                Err(_) => {
                    let (body, provider_text) =
                        ResponseJson::raw_http_error(response.status, &response.body);
                    return Err(crate::codecs::gemini::classify_error(
                        response.status,
                        &body,
                        None,
                        false,
                    )
                    .with_provider_error_text(provider_text));
                }
            };
            let selected_text = scanner.provider_error_text_at("/error/message", None)?;
            let provider_text = match &selected_text {
                Some(text) => text.prefixed(&format!("{} ", response.status)),
                None => scanner
                    .provider_error_text_or_body_at("/error/message", Some(response.status))?,
            };
            let body = scanner.provider_error_classification_body(
                "/error",
                &["message", "status"],
                selected_text.as_ref(),
            );
            return Err(
                crate::codecs::gemini::classify_error(response.status, &body, None, false)
                    .with_provider_error_text(provider_text),
            );
        }
        let source =
            std::str::from_utf8(&response.body).map_err(|error| LlmError::ProviderInternal {
                message: format!("invalid UTF-8 in Gemini interaction JSON: {error}"),
            })?;
        let source = crate::protocol::JsonPayload::parse(source).map_err(|error| {
            LlmError::ProviderInternal {
                message: format!("invalid Gemini interaction JSON source: {error}"),
            }
        })?;
        decode::response(&source, context)
    }
    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        Box::new(crate::codecs::stream::SseDecoder::new(
            stream::InteractionsDecoder::new(context),
        ))
    }
}
