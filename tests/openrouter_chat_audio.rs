use lingxi_llm_client::{
    codecs::{openai::chat::OpenAiChatCodec, CodecContext, RequestMode},
    protocol::{
        message::{OpenRouterChatAudioFormat, OpenRouterChatAudioOutput},
        ChatRequest, ContentBlock, ConversationMessage, LlmError, MessageRole, ProviderProfile,
        StopReason, StreamEvent, ToolChoice,
    },
    WireCodec,
};
use serde_json::{json, Value};

fn openrouter_profile(input_modalities: &[&str], output_modalities: &[&str]) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "openrouter",
        "profile_name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "protocol": "open_ai_chat",
        "auth": "bearer",
        "models": [{
            "display_model": "GPT Audio",
            "request_model": "openai/gpt-audio",
            "billing_model": "openai/gpt-audio",
            "metadata": {
                "inputModalities": input_modalities,
                "outputModalities": output_modalities
            }
        }]
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    ChatRequest {
        prompt_cache: Default::default(),
        output_format: Default::default(),
        controls: Default::default(),
        model: "GPT Audio".into(),
        anthropic_client_toolsets: Vec::new(),
        hosted_tools: vec![],
        continuation: None,
        system: vec![],
        messages: vec![ConversationMessage {
            anthropic: None,
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: "Transcribe this clip".into(),
                thought_signature: None,
            }],
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_tokens: None,
        temperature: None,
        thinking: None,
        service_tier: None,
        stop_sequences: vec![],
        metadata: Value::Null,
    }
}

fn ctx(profile: &ProviderProfile, mode: RequestMode) -> CodecContext {
    CodecContext::new(profile, "openai/gpt-audio", mode)
}

fn body(request: &lingxi_llm_client::HttpRequest) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

#[test]
fn audio_input_and_output_use_openrouter_chat_shapes() {
    let profile = openrouter_profile(&["text", "audio"], &["text", "audio"]);
    let context = ctx(&profile, RequestMode::Stream);
    let mut request = request();
    request.messages[0].content.push(ContentBlock::Audio {
        format: "wav".into(),
        data: "AAECAw==".into(),
    });
    request.metadata = json!({"trace_id": "trace-1"});
    OpenRouterChatAudioOutput::new("alloy", OpenRouterChatAudioFormat::Pcm16)
        .write_metadata(&mut request.metadata)
        .unwrap();

    let encoded = OpenAiChatCodec
        .encode_request(
            lingxi_llm_client::codecs::EncodeRequest::new(&request),
            &context,
        )
        .unwrap();
    let body = body(&encoded);
    assert_eq!(encoded.url, "https://openrouter.ai/api/v1/chat/completions");
    assert_eq!(body["stream"], true);
    assert_eq!(body["modalities"], json!(["text", "audio"]));
    assert_eq!(body["audio"], json!({"voice":"alloy", "format":"pcm16"}));
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(
        body["messages"][0]["content"][0]["text"],
        "Transcribe this clip"
    );
    assert_eq!(
        body["messages"][0]["content"][1],
        json!({"type":"input_audio", "input_audio":{"data":"AAECAw==", "format":"wav"}})
    );
    assert_eq!(request.metadata["trace_id"], "trace-1");
}

#[test]
fn audio_output_metadata_helper_preserves_other_metadata_and_rejects_scalars() {
    let config = OpenRouterChatAudioOutput::new("nova", OpenRouterChatAudioFormat::Mp3);
    let mut metadata = json!({"trace_id":"turn-7"});
    config.write_metadata(&mut metadata).unwrap();
    assert_eq!(metadata["trace_id"], "turn-7");
    assert_eq!(
        metadata["openrouter_chat_audio"],
        json!({"voice":"nova", "format":"mp3"})
    );
    let mut scalar_metadata = json!("existing metadata");
    assert!(config.write_metadata(&mut scalar_metadata).is_err());
}

#[test]
fn input_capability_format_and_base64_are_preflighted() {
    let profile = openrouter_profile(&["text"], &["text"]);
    let context = ctx(&profile, RequestMode::Complete);
    let mut request = request();
    request.messages[0].content = vec![ContentBlock::Audio {
        format: "wav".into(),
        data: "AAECAw==".into(),
    }];
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            lingxi_llm_client::codecs::EncodeRequest::new(&request),
            &context,
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let profile = openrouter_profile(&["text", "audio"], &["text", "audio"]);
    let context = ctx(&profile, RequestMode::Complete);
    request.messages[0].content = vec![ContentBlock::Audio {
        format: "webm".into(),
        data: "AAECAw==".into(),
    }];
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            lingxi_llm_client::codecs::EncodeRequest::new(&request),
            &context,
        ),
        Err(LlmError::InvalidRequest { .. })
    ));
    request.messages[0].content = vec![ContentBlock::Audio {
        format: "wav".into(),
        data: "data:audio/wav;base64,AAECAw==".into(),
    }];
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            lingxi_llm_client::codecs::EncodeRequest::new(&request),
            &context,
        ),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn output_requires_streaming_and_an_advertised_output_modality() {
    let profile = openrouter_profile(&["text", "audio"], &["text"]);
    let mut request = request();
    OpenRouterChatAudioOutput::new("alloy", OpenRouterChatAudioFormat::Wav)
        .write_metadata(&mut request.metadata)
        .unwrap();
    let context = ctx(&profile, RequestMode::Stream);
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            lingxi_llm_client::codecs::EncodeRequest::new(&request),
            &context,
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let profile = openrouter_profile(&["text", "audio"], &["text", "audio"]);
    let context = ctx(&profile, RequestMode::Complete);
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            lingxi_llm_client::codecs::EncodeRequest::new(&request),
            &context,
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn stream_preserves_audio_chunks_response_ids_transcript_and_usage() {
    let profile = openrouter_profile(&["text", "audio"], &["text", "audio"]);
    let context = ctx(&profile, RequestMode::Stream);
    let mut decoder = OpenAiChatCodec.stream_decoder(&context);
    let start = decode_frame(
        &mut *decoder,
        br#"{"id":"gen-audio-1","model":"openai/gpt-audio","choices":[]}"#,
    )
    .unwrap();
    assert!(matches!(
        start.first(),
        Some(StreamEvent::Start { response_id: Some(id), .. }) if id.as_str() == "gen-audio-1"
    ));
    let first = decode_frame(
        &mut *decoder,
        br#"{"id":"gen-audio-1","model":"openai/gpt-audio","choices":[{"index":0,"delta":{"audio":{"id":"audio-1","data":"AQID","transcript":"Hello"}}}]}"#,
    )
    .unwrap();
    let native = first
        .iter()
        .find_map(|event| match event {
            StreamEvent::ProviderContent { block, value, .. } => Some((*block, value.clone())),
            _ => None,
        })
        .expect("audio chunk is preserved as provider-native content");
    assert_eq!(native.1["response_id"], "gen-audio-1");
    assert_eq!(native.1["audio"]["id"], "audio-1");
    assert_eq!(native.1["audio"]["data"], "AQID");
    assert_eq!(native.1["audio"]["transcript"], "Hello");

    let second = decode_frame(
        &mut *decoder,
        br#"{"id":"gen-audio-1","model":"openai/gpt-audio","choices":[{"index":0,"delta":{"audio":{"id":"audio-1","data":"BAUG","transcript":" there"}}}]}"#,
    )
    .unwrap();
    let continued = second
        .iter()
        .find_map(|event| match event {
            StreamEvent::ProviderContent { block, value, .. } => Some((*block, value)),
            _ => None,
        })
        .expect("subsequent audio chunk remains incremental");
    assert_eq!(continued.0, native.0);
    assert_eq!(continued.1["audio"]["data"], "BAUG");
    assert_eq!(continued.1["audio"]["transcript"], " there");

    let terminal = decode_frame(
        &mut *decoder,
        br#"{"id":"gen-audio-1","model":"openai/gpt-audio","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":5,"total_tokens":13,"cost":0.0002}}"#,
    )
    .unwrap();
    assert!(terminal.is_empty());
    let done = decode_frame(&mut *decoder, b"[DONE]").unwrap();
    let end = done
        .iter()
        .find_map(|event| match event {
            StreamEvent::End {
                stop_reason, usage, ..
            } => Some((stop_reason, usage)),
            _ => None,
        })
        .expect("terminal stream event");
    assert_eq!(end.0, &StopReason::EndTurn);
    let usage = end.1.usage.as_ref().expect("reported usage");
    assert_eq!(usage.input_tokens, 8);
    assert_eq!(usage.output_tokens, 5);
    assert_eq!(usage.cost.unwrap().nano_usd, 200_000);
}

#[test]
fn audio_input_is_rejected_by_non_openrouter_chat_profiles() {
    let mut profile = openrouter_profile(&["text", "audio"], &["text", "audio"]);
    profile.provider_id = "acme".into();
    let context = ctx(&profile, RequestMode::Complete);
    let mut request = request();
    request.messages[0].content = vec![ContentBlock::Audio {
        format: "wav".into(),
        data: "AAECAw==".into(),
    }];
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            lingxi_llm_client::codecs::EncodeRequest::new(&request),
            &context,
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn output_config_does_not_accept_unknown_format_or_empty_voice() {
    let profile = openrouter_profile(&["text", "audio"], &["text", "audio"]);
    let context = ctx(&profile, RequestMode::Stream);
    let mut request = request();
    request.metadata = json!({"openrouter_chat_audio":{"voice":" ","format":"wav"}});
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            lingxi_llm_client::codecs::EncodeRequest::new(&request),
            &context,
        ),
        Err(LlmError::InvalidRequest { .. })
    ));
    request.metadata = json!({"openrouter_chat_audio":{"voice":"alloy","format":"aac"}});
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            lingxi_llm_client::codecs::EncodeRequest::new(&request),
            &context,
        ),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn openrouter_output_metadata_is_rejected_for_other_chat_profiles() {
    let mut profile = openrouter_profile(&["text", "audio"], &["text", "audio"]);
    profile.provider_id = "acme".into();
    let context = ctx(&profile, RequestMode::Stream);
    let mut request = request();
    OpenRouterChatAudioOutput::new("alloy", OpenRouterChatAudioFormat::Wav)
        .write_metadata(&mut request.metadata)
        .unwrap();
    assert!(matches!(
        OpenAiChatCodec.encode_request(
            lingxi_llm_client::codecs::EncodeRequest::new(&request),
            &context,
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

fn decode_frame(
    decoder: &mut dyn lingxi_llm_client::codecs::StreamDecoder,
    frame: &[u8],
) -> Result<Vec<StreamEvent>, LlmError> {
    let mut wire = b"data: ".to_vec();
    wire.extend_from_slice(frame);
    wire.extend_from_slice(b"\n\n");
    decoder.push_bytes(&wire).into_iter().collect()
}
