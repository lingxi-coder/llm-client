# OpenRouter Chat Completions audio

OpenRouter Chat Completions accepts base64 audio input for compatible models. Audio output uses the same `/api/v1/chat/completions` route, but requires a streaming request and a model that advertises audio output.

## Audio input

Represent user-message audio as `ContentBlock::Audio`. The `data` value is standard base64 of the audio bytes, without a `data:` URL prefix. The codec accepts the common formats documented by OpenRouter (`wav`, `mp3`, `aiff`, `aac`, `ogg`, `flac`, `m4a`, `pcm16`, and `pcm24`), checks base64 syntax, and checks the selected model's published `input_modalities` when present. Audio URLs are not supported. Provider and model format support still varies, so select a model whose catalog row includes `audio` input and confirm the format for that route.

```rust,ignore
use lingxi_llm_client::protocol::{
    ContentBlock,
};
use base64::{engine::general_purpose::STANDARD, Engine};

request.messages[0].content.push(ContentBlock::Audio {
    format: "wav".into(),
    data: STANDARD.encode(audio_bytes),
});
```

The `STANDARD` value above is `base64::engine::general_purpose::STANDARD`. The actual OpenRouter content part is `{"type":"input_audio","input_audio":{"data":"...","format":"wav"}}`.

## Streaming audio output

Use `OpenRouterChatAudioOutput` to add the documented per-request configuration to `ChatRequest.metadata`; the helper preserves other object metadata. The selected model must advertise `audio` in `output_modalities`, and the request must use `.chat().stream(...)`.

```rust,ignore
use lingxi_llm_client::protocol::message::{
    OpenRouterChatAudioFormat, OpenRouterChatAudioOutput,
};
use lingxi_llm_client::protocol::{ProtocolFamily, StreamEvent};

OpenRouterChatAudioOutput::new("alloy", OpenRouterChatAudioFormat::Mp3)
    .write_metadata(&mut request.metadata).unwrap();

let mut stream = client.chat().stream(&request, &options).await?;
while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::ProviderContent {
            protocol: ProtocolFamily::OpenAiChat,
            value,
            ..
        } if value.get("audio").is_some() => {
            // Append value["audio"]["data"] base64 chunks in arrival order.
            // value also retains response_id, model, choice_index, audio.id,
            // and any transcript carried by the provider.
        }
        _ => {}
    }
}
```

The helper writes `metadata.openrouter_chat_audio = { voice, format }`. The codec sends `modalities: ["text", "audio"]` and `audio: { voice, format }`. Available voices and output formats vary by selected model; the documented format values accepted by the helper are `wav`, `mp3`, `flac`, `opus`, and `pcm16`.

Audio chunks remain `StreamEvent::ProviderContent` so native IDs, base64 data, transcript fragments, and provider fields survive without being interpreted or transcoded by the client. Their `value` is a wrapper with optional `response_id`, `model`, and `choice_index`, plus the untouched `audio` delta. Callers should consume events whose protocol is `OpenAiChat`; the audio payload is provider-native, not a normalized audio playback type. The codec requests `stream_options.include_usage: true` for audio output, and usage reported by OpenRouter remains available on the ordinary terminal `StreamEvent::End` event.

This support is specific to OpenRouter Chat profiles. Other Chat codecs reject `ContentBlock::Audio`. The separate OpenRouter STT/TTS endpoints are documented in [OpenRouter audio services](openrouter-audio.en.md).

References: OpenRouter's [audio guide](https://openrouter.ai/docs/guides/overview/multimodal/audio) documents base64 `input_audio`, model-specific modality/format support, output configuration, and streaming chunks. The [Chat Completions API reference](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion) documents the route and request schema.
