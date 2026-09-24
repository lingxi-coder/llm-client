use lingxi_llm_client::{
    codecs::{
        anthropic::AnthropicMessagesCodec,
        gemini::GeminiCodec,
        hosted::*,
        openai::{chat::OpenAiChatCodec, responses::OpenAiResponsesCodec},
    },
    protocol::*,
    *,
};
use serde_json::{json, Value};
fn codecs() -> Vec<Box<dyn WireCodec>> {
    vec![
        Box::new(OpenAiChatCodec),
        Box::new(OpenAiResponsesCodec),
        Box::new(AnthropicMessagesCodec),
        Box::new(GeminiCodec),
        Box::new(AzureOpenAiCodec),
        Box::new(BedrockClaudeCodec),
        Box::new(FoundryClaudeCodec),
        Box::new(VertexClaudeCodec),
        Box::new(VertexGeminiCodec),
    ]
}
fn context(codec: &dyn WireCodec) -> CodecContext {
    let p:ProviderProfile=serde_json::from_value(json!({"profile_name":"p","provider_id":"test","protocol":codec.family(),"base_url":"https://example.test","auth":"none","models":[]})).unwrap();
    CodecContext::new(&p, "m", RequestMode::Stream)
}
fn fixtures(family: ProtocolFamily) -> (Value, Vec<u8>) {
    let (full, events) = match family {
        ProtocolFamily::OpenAiChat | ProtocolFamily::AzureOpenAi => (
            json!({"model":"m","choices":[{"message":{"role":"assistant","content":"héllo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}),
            vec![
                json!({"model":"m","choices":[{"delta":{"content":"héllo"},"finish_reason":"stop"}]}),
                json!({"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}),
            ],
        ),
        ProtocolFamily::OpenAiResponses => {
            let response = json!({"id":"resp","model":"m","status":"completed","output":[{"id":"msg","type":"message","role":"assistant","content":[{"type":"output_text","text":"héllo"}]}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}});
            (
                response.clone(),
                vec![
                    json!({"type":"response.output_text.delta","output_index":0,"delta":"héllo"}),
                    json!({"type":"response.completed","response":response}),
                ],
            )
        }
        ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini => {
            let value = json!({"modelVersion":"m","candidates":[{"content":{"parts":[{"text":"héllo"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}});
            (value.clone(), vec![value])
        }
        _ => (
            json!({"id":"msg","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"héllo"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":3,"cache_read_tokens":0,"cache_write_tokens":0}}),
            vec![
                json!({"type":"message_start","message":{"id":"msg","model":"m","usage":{"input_tokens":2}}}),
                json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"héllo"}}),
                json!({"type":"content_block_stop","index":0}),
                json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}),
                json!({"type":"message_stop"}),
            ],
        ),
    };
    let mut bytes = Vec::new();
    for event in events {
        if family == ProtocolFamily::BedrockClaude {
            bytes.extend(bedrock_frame(&event.to_string()));
        } else {
            bytes.extend(format!("data: {event}\n\n").as_bytes());
        }
    }
    if matches!(
        family,
        ProtocolFamily::OpenAiChat | ProtocolFamily::AzureOpenAi
    ) {
        bytes.extend_from_slice(b"data: [DONE]\n\n");
    }
    (full, bytes)
}
#[test]
fn complete_and_arbitrarily_fragmented_streams_agree_for_every_codec() {
    for codec in codecs() {
        let context = context(codec.as_ref());
        let (full, bytes) = fixtures(codec.family());
        let response = codec
            .decode_response(
                &HttpResponse {
                    status: 200,
                    headers: vec![],
                    body: serde_json::to_vec(&full).unwrap().into(),
                },
                &context,
            )
            .unwrap();
        assert_eq!(
            response.usage.state,
            UsageState::Complete,
            "{:?}",
            codec.family()
        );
        for chunk_size in [1, 2, 3, 7, 31, bytes.len()] {
            let mut decoder = codec.stream_decoder(&context);
            let mut events = Vec::new();
            for chunk in bytes.chunks(chunk_size) {
                events.extend(decoder.push_bytes(chunk));
            }
            events.extend(decoder.finish());
            let events = events.into_iter().collect::<Result<Vec<_>, _>>().unwrap();
            let text = events
                .iter()
                .filter_map(|e| {
                    if let StreamEvent::TextDelta { text, .. } = e {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<String>();
            assert_eq!(text, "héllo", "{:?} chunks={chunk_size}", codec.family());
            let endings = events
                .iter()
                .filter_map(|e| {
                    if let StreamEvent::End {
                        usage, stop_reason, ..
                    } = e
                    {
                        Some((usage, stop_reason))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(endings.len(), 1);
            assert_eq!(endings[0].0, &response.usage, "{:?}", codec.family());
            assert_eq!(endings[0].1, &response.stop_reason);
            assert_eq!(decoder.usage_report(), response.usage);
            assert!(decoder.finish().is_empty());
            assert!(decoder.push_bytes(b"trailing garbage").is_empty());
        }
    }
}
#[test]
fn valid_events_precede_a_later_error_in_the_same_chunk() {
    let codec = OpenAiChatCodec;
    let context = context(&codec);
    for malformed in [
        b"data: {bad}\n\n".as_slice(),
        &vec![b'x'; 8 * 1024 * 1024 + 1],
    ] {
        let mut bytes = b"data: {\"choices\":[{\"delta\":{\"content\":\"prefix\"}}]}\n\n".to_vec();
        bytes.extend_from_slice(malformed);
        let mut decoder = codec.stream_decoder(&context);
        let events = decoder.push_bytes(&bytes);
        assert!(events
            .iter()
            .any(|e| matches!(e,Ok(StreamEvent::TextDelta{text,..})if text=="prefix")));
        assert!(events.last().unwrap().is_err());
        assert!(decoder.finish().is_empty());
    }
}
#[test]
fn truncated_streams_never_claim_complete_usage() {
    for codec in codecs() {
        let context = context(codec.as_ref());
        let (_, bytes) = fixtures(codec.family());
        let mut decoder = codec.stream_decoder(&context);
        let mut events = decoder.push_bytes(&bytes[..bytes.len() / 2]);
        events.extend(decoder.finish());
        assert!(events.iter().any(Result::is_err), "{:?}", codec.family());
        assert_ne!(decoder.usage_report().state, UsageState::Complete);
    }
}
#[test]
fn legacy_usage_is_rejected_and_malformed_partial_usage_is_invalid() {
    assert!(serde_json::from_value::<UsageReport>(
        json!({"input_tokens":2,"output_tokens":3,"cache_read_tokens":0,"cache_write_tokens":0})
    )
    .is_err());
    let codec = OpenAiChatCodec;
    let context = context(&codec);
    for usage in [
        json!({"prompt_tokens":1,"total_tokens":"bad"}),
        json!({"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":-1}}),
    ] {
        let full = json!({"model":"m","choices":[{"message":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":usage});
        let response = codec
            .decode_response(
                &HttpResponse {
                    status: 200,
                    headers: vec![],
                    body: full.to_string().into(),
                },
                &context,
            )
            .unwrap();
        assert_eq!(response.usage.state, UsageState::Invalid);
    }
}

use lingxi_llm_client::framing::eventstream::crc32;
fn bedrock_frame(event_json: &str) -> Vec<u8> {
    use base64::Engine;
    let payload = serde_json::to_vec(&json!({
        "bytes": base64::engine::general_purpose::STANDARD.encode(event_json),
    }))
    .unwrap();
    let name = ":message-type";
    let value = "event";
    let mut h = Vec::new();
    h.push(name.len() as u8);
    h.extend_from_slice(name.as_bytes());
    h.push(7);
    h.extend_from_slice(&(value.len() as u16).to_be_bytes());
    h.extend_from_slice(value.as_bytes());

    let total = (12 + h.len() + payload.len() + 4) as u32;
    let mut out = Vec::new();
    out.extend_from_slice(&total.to_be_bytes());
    out.extend_from_slice(&(h.len() as u32).to_be_bytes());
    out.extend_from_slice(&crc32(&out[0..8]).to_be_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(&payload);
    let crc = crc32(&out);
    out.extend_from_slice(&crc.to_be_bytes());
    out
}
