# MiniMax HTTP TTS

`minimax_tts` 为 MiniMax 原生 `POST /v1/t2a_v2` HTTP 接口提供独立服务，支持同步 JSON 和 SSE 流式输出。它使用 MiniMax 自己的 `voice_setting` / `audio_setting` 请求体和 `base_resp` 响应状态，不把这个服务当作 OpenAI 兼容 TTS。

```rust,no_run
# async fn example(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    minimax_tts::{
        MiniMaxTtsConfig, MiniMaxTtsCredentials, MiniMaxTtsOutput, MiniMaxTtsRegion,
        MiniMaxTtsRequest, MiniMaxTtsService,
    },
    protocol::Secret,
    HttpTransport,
};

let transport = HttpTransport::new()?;
let service = MiniMaxTtsService::new(
    &transport,
    MiniMaxTtsConfig::new("minimax-voice", "team/account-17", MiniMaxTtsRegion::International),
)?;
let credentials = MiniMaxTtsCredentials::new(Secret::new(api_key));
let request = MiniMaxTtsRequest::new("Hello from MiniMax", "English_expressive_narrator");
match service.synthesize(&request, &credentials).await? {
    MiniMaxTtsOutput::Audio(audio) => {
        let mut output_file = Vec::new();
        output_file.extend_from_slice(&audio.bytes);
    }
    MiniMaxTtsOutput::Url(audio_url) => {
        println!("caller-managed URL, valid for {:?}: {}", audio_url.valid_for, audio_url.url);
    }
}
# Ok(())
# }
```

Choose `MiniMaxTtsRegion::International` for the `.io` account route or `ChinaMainland` for the `.cn` account route. The config also accepts MiniMax's documented same-region alternate endpoint. A profile name and stable account scope are required and attached to the result with an endpoint fingerprint. Pass the matching account's API key through `MiniMaxTtsCredentials` for each call; the service does not store credentials.

已通过 `minimax_voices` 获得的 `MiniMaxVoiceRef` 可以传给 `synthesize_with_voice(&request, &voice, &credentials)`。服务会在 HTTP 发送前检查 reference 与当前 provider、profile、account scope、region 和规范化 `/v1` API root 一致，并使用 reference 中的 voice ID。内置声音 ID 仍可直接调用 `synthesize`。account scope 是调用方提供的路由标签，不证明凭证的实际归属；仍须提供对应 MiniMax 账号的 API key。

The request defaults to `speech-2.8-hd`, MP3 at 32 kHz / 128 kbps, mono, hexadecimal output, and subtitles disabled. It supports MiniMax's listed speech model IDs, custom voice IDs, speed, volume, pitch, emotion, language boost, pronunciation entries, audio format, AIGC watermark, and the HTTP `subtitle_enable` / `subtitle_type` controls. Synchronous synthesis accepts `sentence` or `word` subtitles. `synthesize_stream` supports the documented streaming-only `word_streaming` option and requires MP3 plus hex output; URL output, WAV streaming, and AIGC watermarking are non-streaming features. The input text must be nonempty and shorter than 10,000 characters. The API's documented range checks run before any HTTP request.

With `MiniMaxTtsOutputFormat::Hex`, MiniMax returns `data.audio` as a hex string; the service decodes it to raw codec bytes. With `MiniMaxTtsOutputFormat::Url`, MiniMax returns a provider-hosted URL that is valid for 24 hours. The service returns that URL and its validity window without opening it or downloading its audio. The result also keeps the `trace_id`, available `extra_info`, and full native response.

`synthesize_stream` 会逐个返回 `MiniMaxTtsStreamEvent`。每个 JSON event 都保留完整 native 对象，并在存在 `data.audio` 时将其从 hex 解码；`data.status == 2` 和 `[DONE]` 会结束流。收到音频后正常 EOF 也会结束流，与 MiniMax 已发布 CLI 的行为一致。可通过 `terminal_event_received()` 区分显式收到 status 2 / `[DONE]` 与正常 EOF；客户端不会伪造缺失的终止标记。客户端保留终止 event 中的 audio，不合并或去重 chunk，因为当前可查到的 provider contract 没有说明终止 event 的 audio 是增量还是完整聚合结果。流式字幕字段会保留在 native event 中；客户端不推测字幕 event schema，也不会下载字幕 URL。

```rust,no_run
use lingxi_llm_client::minimax_tts::{
    MiniMaxTtsCredentials, MiniMaxTtsRequest, MiniMaxTtsService,
    MiniMaxTtsStreamEvent, MiniMaxTtsSubtitleType,
};

async fn collect_audio(
    service: &MiniMaxTtsService<'_>,
    credentials: &MiniMaxTtsCredentials,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
let mut audio_output = Vec::new();

let mut request = MiniMaxTtsRequest::new("Hello from a stream", "voice-id");
request.subtitle_enable = true;
request.subtitle_type = MiniMaxTtsSubtitleType::WordStreaming;
let mut events = service.synthesize_stream(&request, credentials).await?;
while let Some(event) = events.next_event().await? {
    match event {
        MiniMaxTtsStreamEvent::Data(data) => {
            if let Some(audio_chunk) = data.audio {
                audio_output.extend_from_slice(&audio_chunk);
            }
            // 由调用方检查 status/native，再决定是否追加终止 event 的音频。
        }
        MiniMaxTtsStreamEvent::Done => break,
    }
}
Ok(audio_output)
}
```

MiniMax 的 WebSocket 与 `t2a_async_v2` task API 属于不同协议；本模块不会创建或轮询异步任务、重试请求，也不会下载返回的音频或字幕 URL。丢弃 SSE stream 会取消对应 HTTP body。每个 SSE event 上限为 8 MiB；stream error 会保留 request ID 和此前已交付的解码音频字节数。

The wire contract follows MiniMax's [Text to Speech (T2A) HTTP reference](https://platform.minimax.io/docs/api-reference/speech-t2a-http) and its [mainland China HTTP reference](https://platform.minimaxi.com/docs/api-reference/speech-t2a-http).
