//! Exact UTF-16 tool descriptions must reach the real provider request JSON.

use lingxi_llm_client::{
    codecs::{CodecContext, EncodeRequest, RequestMode},
    protocol::{ChatRequest, JsonPayload, ProviderProfile},
    AnthropicMessagesCodec, GeminiCodec, GeminiInteractionsCodec, OpenAiChatCodec,
    OpenAiResponsesCodec, WireCodec,
};
use serde_json::json;

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model": "model",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "tools": [{
            "name": "read",
            "description": "placeholder",
            "input_schema": {"type": "object", "properties": {}}
        }]
    }))
    .expect("bounded provider request fixture is valid")
}

fn encoded_body(codec: &dyn WireCodec, protocol: &str) -> String {
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id": "test",
        "profile_name": "test",
        "base_url": "https://example.test/v1",
        "protocol": protocol,
        "auth": "none",
        "extra": {"account_scope": "test-account"},
        "models": [{"display_model": "model", "request_model": "model", "billing_model": "model"}]
    }))
    .expect("provider profile fixture is valid");
    let context = CodecContext::new(&profile, "model", RequestMode::Complete)
        .with_account_scope(Some("test-account"));
    let mut request = request();
    request.tools[0].description = JsonPayload::string_from_utf16_code_units(&[
        b'R' as u16,
        b'e' as u16,
        b'a' as u16,
        b'd' as u16,
        0xd800,
    ]);
    let encoded = codec
        .encode_request(EncodeRequest::new(&request), &context)
        .expect("provider request encoding succeeds");
    String::from_utf8(encoded.body.to_vec()).expect("JSON wire body is UTF-8")
}

#[test]
fn provider_codecs_emit_source_backed_tool_description_json() {
    for (codec, protocol) in [
        (&OpenAiChatCodec as &dyn WireCodec, "open_ai_chat"),
        (&OpenAiResponsesCodec as &dyn WireCodec, "open_ai_responses"),
        (
            &AnthropicMessagesCodec as &dyn WireCodec,
            "anthropic_messages",
        ),
        (&GeminiCodec as &dyn WireCodec, "gemini_generate_content"),
        (
            &GeminiInteractionsCodec as &dyn WireCodec,
            "gemini_interactions",
        ),
    ] {
        let body = encoded_body(codec, protocol);
        assert!(
            body.contains(r#""description":"Read\ud800""#),
            "{protocol} tool description must retain its exact UTF-16 source: {body}"
        );
        assert!(
            !body.contains('\u{fffd}'),
            "{protocol} replaced the lone surrogate"
        );
    }
}
