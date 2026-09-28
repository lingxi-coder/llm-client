# OpenRouter Chat Completions 音频

OpenRouter Chat Completions 为兼容模型提供 base64 音频输入。音频输出使用同一个 `/api/v1/chat/completions` 路由，但必须使用流式请求，并且所选模型需要声明支持音频输出。

## 音频输入

使用 `ContentBlock::Audio` 表示用户消息中的内联音频。`data` 是音频字节的标准 base64，不要包含 `data:` URL 前缀。编解码器接受 OpenRouter 文档列出的常见格式：`wav`、`mp3`、`aiff`、`aac`、`ogg`、`flac`、`m4a`、`pcm16` 和 `pcm24`；会检查 base64 语法，并在模型目录提供信息时检查所选模型的 `input_modalities`。音频 URL 不受支持。不同提供方和模型支持的格式不同，因此应选择目录中 `input_modalities` 包含 `audio` 的模型，并确认该路由支持对应格式。

```rust,ignore
use lingxi_llm_client::protocol::ContentBlock;
use base64::{engine::general_purpose::STANDARD, Engine};

request.messages[0].content.push(ContentBlock::Audio {
    format: "wav".into(),
    data: STANDARD.encode(audio_bytes),
});
```

上例中的 `STANDARD` 是 `base64::engine::general_purpose::STANDARD`。实际发送的 OpenRouter 内容部分为 `{"type":"input_audio","input_audio":{"data":"...","format":"wav"}}`。

## 流式音频输出

使用 `OpenRouterChatAudioOutput` 将文档规定的单次请求配置写入 `ChatRequest.metadata`；该 helper 会保留对象中的其他 metadata。所选模型必须在 `output_modalities` 中声明 `audio`，并且调用时必须使用 `.chat().stream(...)`。

```rust,ignore
use lingxi_llm_client::protocol::{OpenRouterChatAudioFormat, OpenRouterChatAudioOutput};
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
            // 按接收顺序拼接 value["audio"]["data"] 中的 base64 分片。
            // value 还保留 response_id、model、choice_index、audio.id 和 transcript。
        }
        _ => {}
    }
}
```

helper 会写入 `metadata.openrouter_chat_audio = { voice, format }`。编解码器据此发送 `modalities: ["text", "audio"]` 和 `audio: { voice, format }`。可用 voice 和输出格式取决于所选模型；helper 接受的文档格式为 `wav`、`mp3`、`flac`、`opus` 和 `pcm16`。

音频分片保留在 `StreamEvent::ProviderContent` 中，因此原生 ID、base64 数据、transcript 分片和其他提供方字段均会保留，客户端不会自行解释或转码。每个事件的 `value` 是一个包含可选 `response_id`、`model`、`choice_index` 以及未修改 `audio` delta 的包装对象。调用方应处理协议为 `OpenAiChat` 的事件；音频负载属于提供方原生格式，不是标准化的播放类型。音频输出请求会附带 `stream_options.include_usage: true`；OpenRouter 报告的用量会出现在普通的终止 `StreamEvent::End` 事件中。

此功能只适用于 OpenRouter Chat 配置。其他 Chat 编解码器会拒绝 `ContentBlock::Audio`。OpenRouter 独立的 STT/TTS 路由见 [OpenRouter 音频服务](openrouter-audio.md)。

参考资料：OpenRouter [音频指南](https://openrouter.ai/docs/guides/overview/multimodal/audio)说明 base64 `input_audio`、按模型区分的模态与格式支持、输出配置及流式分片。[Chat Completions API 参考](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion)说明请求路由和 schema。
