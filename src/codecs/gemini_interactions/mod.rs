//! Unified chat and streaming wire for Gemini Interactions.
//! Pure codecs reuse the provider DTO encoder; they never call its service.
pub(crate) mod decode;
mod encode;
mod stream;
use crate::codecs::{CodecContext, EncodeRequest, StreamDecoder, WireCodec};
use crate::protocol::{ChatRequest, ChatResponse, LlmError, ProtocolFamily};
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
            return Err(crate::codecs::gemini::classify_error(
                response.status,
                &serde_json::from_slice(&response.body).unwrap_or(serde_json::Value::Null),
                None,
            ));
        }
        let body =
            serde_json::from_slice(&response.body).map_err(|error| LlmError::ProviderInternal {
                message: format!("invalid Gemini interaction JSON: {error}"),
            })?;
        decode::response(&body, context)
    }
    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        Box::new(crate::codecs::stream::SseDecoder::new(
            stream::InteractionsDecoder::new(context),
        ))
    }
}
