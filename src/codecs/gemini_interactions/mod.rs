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
        crate::exact_json::validate_tool_input_carriers(request)?;
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
        let mut parsed = crate::response_json::ResponseJson::parse(
            &response.body,
            "invalid Gemini interaction JSON",
        )?;
        let body = parsed.value.clone();
        let mut response = decode::response(&body, context)?;
        for (index, step) in body["steps"].as_array().into_iter().flatten().enumerate() {
            if step["type"] == "function_call" {
                let (display, raw) =
                    parsed.take_tool_input(&format!("/steps/{index}/arguments"))?;
                for block in &mut response.message.content {
                    if let crate::protocol::ContentBlock::ToolUse {
                        id,
                        input,
                        input_json,
                        ..
                    } = block
                    {
                        if step["id"].as_str() == Some(id.as_str()) {
                            *input = display.clone();
                            *input_json = raw.clone();
                        }
                    }
                }
            }
        }
        parsed.finish()?;
        Ok(response)
    }
    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        Box::new(crate::codecs::stream::SseDecoder::new(
            stream::InteractionsDecoder::new(context),
        ))
    }
}
