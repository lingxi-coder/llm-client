# Gemini Voices API

`gemini_speech::GeminiVoicesService` 封装 Gemini Developer API 的 Voices 资源，支持 `POST/GET https://generativelanguage.googleapis.com/v1beta/voices` 和 `GET/DELETE /v1beta/voices/{voice_id}`。它支持通过提示创建自定义语音、使用参考录音和同意录音创建复制语音、列出语音目录并筛选、分页，以及查询和删除已存储语音。此资源与 TTS Interactions 和 Live API 相互独立。

每个服务都绑定到一个 Google 配置档、调用方提供的账号/项目标识，以及确切的 Voices 端点。每次调用都传入 `&Secret<String>`；服务不会保留 API 密钥。返回的已存储语音引用会绑定到该作用域。只有在 Google 提供方、配置档和账号均与传入的 `GeminiSpeechScope` 匹配时，`GeminiVoice::speech_request` 才会把语音 ID 或无状态密钥用于 Interactions TTS 请求。

```rust,ignore
use bytes::Bytes;
use lingxi_llm_client::{
    gemini_speech::{
        GeminiSpeechModel, GeminiSpeechScope, GeminiVoicesScope, GeminiVoicesService,
        GeminiVoiceAudioData, GeminiVoiceCreateRequest, GeminiVoiceListOptions,
        GEMINI_SPEECH_ENDPOINT, GEMINI_VOICES_ENDPOINT,
    },
    protocol::Secret,
    transport::HttpTransport,
};

let http = HttpTransport::new()?;
let scope = GeminiVoicesScope::new(
    "gemini-tts",
    "google-project-prod",
    GEMINI_VOICES_ENDPOINT,
)?;
let voices = GeminiVoicesService::new(&http, scope)?;
let key = Secret::new(api_key);

let created = voices
    .create(&key, &GeminiVoiceCreateRequest::prompted(
        "A calm, warm narrator with a low register.",
    ).with_display_name("Warm narrator"))
    .await?;

let page = voices
    .list(&key, &GeminiVoiceListOptions::new()
        .with_page_size(100)
        .with_types([lingxi_llm_client::gemini_speech::GeminiVoiceType::Prompted]))
    .await?;
let _next_page_token = page.next_page_token;

let speech_scope = GeminiSpeechScope::new(
    "gemini-tts",
    "google-project-prod",
    GEMINI_SPEECH_ENDPOINT,
)?;
let request = created.speech_request(
    &speech_scope,
    GeminiSpeechModel::Gemini38FlashTts,
    "Hello from this custom voice.",
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

创建复制语音时，由宿主提供两段录音。以下示例请求 Google 返回无状态密钥（`store = false`）；将其转换为语音请求时，该密钥仍绑定到来源配置档和账号。

```rust,ignore
let source = GeminiVoiceAudioData::new(Bytes::from(reference_wav), "audio/wav");
let consent = GeminiVoiceAudioData::new(Bytes::from(consent_wav), "audio/wav");
let replicated = voices
    .create(&key, &GeminiVoiceCreateRequest::replicated(source, consent, false))
    .await?;
let request = replicated.speech_request(
    &speech_scope,
    GeminiSpeechModel::Gemini38FlashTts,
    "Hello from this replicated voice.",
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

通过提示创建语音时始终发送 `store: true`，因为 Google 不接受未存储的提示创建语音。创建复制语音时，将说话人参考录音和同意录音都传给 `GeminiVoiceCreateRequest::replicated(source, consent, store)`。两段录音都是必需的，使用原始字节和 IANA MIME 类型，并在同一个 JSON 请求中进行 base64 编码。服务不会保留任何录音。Google 官方指南要求使用真实成年说话人的录音，建议参考片段清晰且时长为 10–30 秒，并要求同一说话人使用支持的语言之一准确朗读同意声明。调用 `create` 前，宿主负责取得并核实该同意。

`store = true` 会返回由 Google 管理的 `voice_...` ID，可用于列出、查询和删除。复制语音支持 `store = false`，此时会返回由调用方管理的 `voicekey_...`；它不是托管资源，不能通过 Voices API 查询或删除。Google 当前文档说明，每个项目最多可存储 200 个自定义语音，存储时长为一年；无状态密钥的有效期为七天。Voices API 参考文档与指南对 `store` 的默认值说明不一致，因此此适配器始终显式发送该选项。

`GeminiVoiceListOptions` 提供文档列出的可重复筛选项（`accent`、`context`、`gender`、`language_code`、`persona`、`pitch`、`region_code` 和 `type`），以及 `search`、`page_size` 和 `page_token`。本地将每页大小限制为文档规定的最大值 1000，将搜索内容限制为文档规定的最多 2048 字节。继续获取下一页时，保持筛选值不变，并原样传入 `next_page_token`。列表结果会包含预置语音，但 Google 不支持单独查询或删除这些语音。

客户端不会自动重试、轮询或下载。传输失败、HTTP 408/5xx，以及格式错误但请求已成功的创建响应，会作为结果未知/请求已受理的状态返回。调用方可先列出已存储语音进行核对，再决定是否重试。本地将请求 JSON 限制为 32 MiB、响应限制为 64 MiB，以控制内存中的编码和解码开销；这是客户端安全限制，不是 Google API 的限制。

## 官方参考资料

- [Gemini Voices API 参考](https://ai.google.dev/api/voices)
- [语音复制指南](https://ai.google.dev/gemini-api/docs/voice-replication)
- [文本转语音生成](https://ai.google.dev/gemini-api/docs/speech-generation)
