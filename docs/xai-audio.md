# xAI 音频服务

`xai_audio` 为 xAI 提供原生 HTTP 语音转文字和文字转语音接口，并提供只读 TTS 音色目录。服务不复用 OpenAI 音频字段：STT 使用 multipart 上传并返回 JSON；普通 TTS 返回原始音频字节流；启用字符时间戳后，TTS 改为 JSON 包装并在 `audio` 字段中返回 Base64 音频。

```rust,no_run
# async fn example(api_key: String, audio_bytes: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    audio::AudioInput,
    HttpTransport,
    protocol::Secret,
    xai_audio::{
        XaiAudioConfig, XaiAudioCredentials, XaiAudioService, XaiSpeechOutput,
        XaiSpeechRequest, XaiTranscriptionRequest, XaiTranscriptionUrl,
    },
};

let transport = HttpTransport::new()?;
let service = XaiAudioService::new(
    &transport,
    XaiAudioConfig::new("xai-prod", "team/account-7"),
)?;
let credentials = XaiAudioCredentials::new(Secret::new(api_key));

let voices = service.list_voices(&credentials).await?;
for voice in &voices.voices {
    println!("{} ({})", voice.name, voice.voice_id);
}

let transcript = service
    .transcribe(
        AudioInput::from_bytes("meeting.wav", "audio/wav", audio_bytes),
        &XaiTranscriptionRequest::default(),
        &credentials,
    )
    .await?;
println!("{}", transcript.text);

let audio_url = XaiTranscriptionUrl::new(
    "https://media.example/meeting.wav?signature=example",
)?;
let remote_transcript = service
    .transcribe_url(&audio_url, &XaiTranscriptionRequest::default(), &credentials)
    .await?;
println!("{}", remote_transcript.text);

let mut speech_request = XaiSpeechRequest::new("欢迎使用 LingXi。", "eve", "zh");
speech_request.speed = Some(1.2);
speech_request.optimize_streaming_latency = Some(1);
speech_request.text_normalization = true;
speech_request
    .replace
    .insert("LingXi".into(), "Ling See".into());
let output = service.synthesize(&speech_request, &credentials).await?;
if let XaiSpeechOutput::Audio(mut audio) = output {
    let mut audio_file = Vec::new();
    while let Some(chunk) = audio.next_chunk().await? {
        audio_file.extend_from_slice(&chunk);
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
# }
```

`XaiAudioConfig` requires a nonempty profile name and stable account scope. It defaults to `https://api.x.ai/v1`; `with_api_base_url` selects an explicit compatible HTTPS API root. HTTP is accepted only for localhost/loopback test transports. Each call receives an `XaiAudioCredentials` value containing the caller-owned API key. Neither the service nor its result stores credentials. Returned results include the profile, account scope, and a fingerprint of the configured API root.

## Speech to text

`XaiAudioService::transcribe` calls `POST /v1/stt` with a streamed `multipart/form-data` upload. xAI requires all option fields before the final `file` field; the implementation preserves that order and does not buffer the uploaded audio. The documented upload limit is 500 MB. Supported container files include WAV, MP3, OGG, Opus, FLAC, AAC, MP4/M4A, and MKV. Raw PCM, μ-law, and A-law input requires both `audio_format` and a supported `sample_rate`.

使用服务端可读取的音频来源时，构造 `XaiTranscriptionUrl` 并调用 `XaiAudioService::transcribe_url`。客户端会按文档将来源放入 multipart 的 `url` 字段，由 xAI 服务端下载并转写；客户端本身不会下载媒体。调用方需确保 xAI 可以访问该 HTTP(S) 地址；签名查询参数会原样保留。构造 URL 时会拒绝非 HTTP(S) 地址、用户信息和片段，并在校验失败时不发送请求。`XaiTranscriptionUrl` 的 `Debug` 输出会隐藏地址。错误清理只会替换错误内容中逐字出现的完整输入 URL 或完整 `?query` 片段；供应商重排、规范化或部分回显后的地址不保证清理。调用方仍需自行决定是否记录原生错误内容。

The typed request covers the current model IDs, language formatting, multichannel transcription, diarization, keyterms, filler words, and VAD threshold. `format=true` requires a language; raw audio requires an explicit format and sample rate. The response retains transcript text, detected language, duration, word timestamps, speaker IDs, per-channel results, request ID, and the native JSON object. Missing optional provider fields remain optional/defaulted; malformed required transcript fields are reported as invalid responses.

## Text to speech

`XaiAudioService::synthesize` 调用 `POST /v1/tts`，发送 xAI 的 JSON 字段：`text`、`voice_id`、`language` 和 `output_format`。默认格式为 24 kHz / 128 kbps 的 MP3。请求类型也支持 WAV、PCM、μ-law 和 A-law；客户端会按 xAI 公布的取值校验采样率和 MP3 比特率。文本最多 60,000 个 Unicode 码点。

`XaiSpeechRequest` 还提供 `speed`（0.7–1.5，默认 1.0）、`optimize_streaming_latency`（0、1 或 2，默认 0）和 `text_normalization`（默认 false）。未设置的可选值和 false 会从 JSON 中省略，让服务端使用默认值。延迟等级 1 或 2 会缩小流式合成的首块，以降低首个音频块延迟，同时会在块边界带来音质折衷。

可选 `replace` 映射用于替换实际发音，不会改写或改变计费所用的原始输入文本。客户端会在发送前校验文档约束：最多 200 条；键最多 100 个 Unicode 码点；值最多 128 个 Unicode 码点；键仅能包含字母、数字、撇号和空格。键不能为空白，且经标准小写转换并移除空白后必须互不相同。此本地重复项检查不等同于完整的 Unicode case-fold；键匹配与最终校验仍由 xAI 负责。xAI 按不区分大小写的整词边界匹配，并在前缀重叠时优先使用最长键；无空格书写体系按字符匹配。替换后的文本上限 240,000 字符由服务端执行，客户端不会仿造服务端替换算法。若设置 `with_timestamps=true`，`graph_chars` 描述的是实际口播的替换文本，而不是原始键。

Without timestamps, the returned `XaiSpeechOutput::Audio` yields raw HTTP audio chunks and exposes the provider content type and request ID. With `with_timestamps=true`, xAI changes the response to JSON containing Base64 audio, content type, duration, and character timings; the client decodes the audio into `XaiSpeechOutput::Timestamped`. The two wire responses are not treated as interchangeable.

## TTS 音色目录

`XaiAudioService::list_voices` calls the read-only `GET /v1/tts/voices` endpoint. Each typed voice exposes `voice_id` and `name`; `language` is optional because the TTS guide does not require it in every voice-list example. `XaiVoice.native` and `XaiVoiceList.native` preserve the full provider voice objects and response envelope, including fields this client does not interpret. The returned list is bound to the service's profile, account scope, and endpoint fingerprint. This method lists voices available for TTS; it does not create or manage custom voice resources.

`XaiAudioService::get_voice(voice_id, credentials)` 调用 `GET /v1/tts/voices/{voice_id}` 并返回 `XaiVoiceDetails`，其中包含服务 scope、类型化的 `XaiVoice`、完整 native response 和 request ID。响应中的 `voice_id` 必须与请求值完全一致。客户端会把 ID 作为单独 path segment 追加；空值、首尾空白、控制字符、路径分隔符、百分号转义和 `.` / `..` segment 都会在 HTTP 请求前被拒绝。该接口为只读且只请求一次，不会自动重试。

These HTTP operations perform no live credential lookup, retry, cross-account fallback, or cache. A transport failure after sending a transcription or synthesis request may leave the provider-side outcome unknown, which is reflected by `XaiAudioError::OutcomeUnknown`.

Implementation follows xAI's [Speech to Text guide](https://docs.x.ai/developers/model-capabilities/audio/speech-to-text), [Text to Speech guide](https://docs.x.ai/developers/model-capabilities/audio/text-to-speech), and [Voice REST reference](https://docs.x.ai/developers/rest-api-reference/inference/voice).

同一服务还支持自定义音色创建、管理及参考音频流式读取，见[自定义音色](xai-custom-voices.md)。
