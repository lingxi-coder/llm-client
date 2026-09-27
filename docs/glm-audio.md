# GLM-ASR 自托管语音识别

[English](glm-audio.en.md)

本次采用的智谱官方材料没有给出独立 GLM-TTS HTTP 路由的完整请求与响应契约，因此此模块实现官方 GLM-ASR-Nano 仓库示例中的自托管路径：使用 SGLang 启动 `glm-asr` 模型，再通过 OpenAI 兼容的 Chat Completions 请求输入 `audio_url` 内容片段和转写提示。SGLang 官方文档确认路径为 `POST /v1/chat/completions`。[智谱 GLM-ASR](https://github.com/zai-org/GLM-ASR)、[SGLang Quickstart](https://github.com/sgl-project/sglang/blob/main/docs/docs/get-started/quickstart.mdx)

这不是智谱托管云 ASR API。调用方必须运行 GLM-ASR SGLang 服务，并提供 `/v1` 根地址。官方模型示例使用 `http://127.0.0.1:8000/v1` 和 `EMPTY` 作为兼容客户端的占位 API key；本服务仍通过 Bearer 头发送调用方提供的凭据。非回环地址必须使用 HTTPS。

```rust,no_run
use lingxi_llm_client::{
    glm_audio::{GlmAsrRequest, GlmAsrRoute, GlmAsrScope},
    protocol::Secret,
    LlmClient,
};

async fn transcribe(
    client: &LlmClient,
    api_key: String,
) -> Result<String, Box<dyn std::error::Error>> {
    let route = GlmAsrRoute::new("http://127.0.0.1:8000/v1")?;
    let scope = GlmAsrScope::new("glm-asr-local", "local-model-host-1", &route)?;
    let service = client.glm_asr(Secret::new(api_key), route, scope)?;
    let result = service
        .transcribe(&GlmAsrRequest::new("example_zh.wav")?)
        .await?;
    Ok(result.text)
}
```

`audio_url` is forwarded exactly as provided; the inference server must be able to resolve it. The official GLM-ASR example uses a server-local relative filename. This service does not upload or download audio. Each transcription sends one request with the documented `glm-asr` model, audio and text content parts, and `max_tokens: 1024`. It does not retry; a transport failure is reported as outcome unknown because the model server may have accepted the request.

Every service is bound to a profile name, caller-defined non-secret account scope, and endpoint fingerprint. A scope created for one endpoint cannot be reused with another route. Tests use a mock transport and do not call paid services.
