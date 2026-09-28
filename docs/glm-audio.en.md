# Self-Hosted GLM-ASR Transcription

[中文](glm-audio.md)

The Zhipu official material used here does not provide a complete request/response contract for a standalone GLM-TTS HTTP route, so this module implements the self-hosted path shown in Zhipu's official GLM-ASR-Nano repository: run `glm-asr` with SGLang, then send an OpenAI-compatible Chat Completions request with an `audio_url` content part and transcription prompt. SGLang's official docs confirm the route is `POST /v1/chat/completions`. See [Zhipu GLM-ASR](https://github.com/zai-org/GLM-ASR) and the [SGLang Quickstart](https://github.com/sgl-project/sglang/blob/main/docs/docs/get-started/quickstart.mdx).

This is not a Zhipu-hosted cloud ASR API. The caller runs the GLM-ASR SGLang server and supplies its `/v1` root URL. The official model example uses `http://127.0.0.1:8000/v1` and `EMPTY` as the compatibility client's placeholder API key; this service still sends the caller-supplied credential in a Bearer header. Non-loopback endpoints must use HTTPS.

```rust,no_run
use lingxi_llm_client::{
    providers::zhipu::audio::{GlmAsrRequest, GlmAsrRoute, GlmAsrScope},
    protocol::Secret,
    LlmClient,
};

async fn transcribe(
    client: &LlmClient,
    api_key: String,
) -> Result<String, Box<dyn std::error::Error>> {
    let route = GlmAsrRoute::new("http://127.0.0.1:8000/v1")?;
    let scope = GlmAsrScope::new("glm-asr-local", "local-model-host-1", &route)?;
    let provider = client.provider::<lingxi_llm_client::providers::zhipu::ZhipuClient>(scope.profile_name())?;
    let request_options = lingxi_llm_client::RequestOptions {
        credential: Some(Secret::new(api_key)),
        ..Default::default()
    };
    let service = provider.asr(route, scope)?;
    let result = service
        .transcribe(&GlmAsrRequest::new("example_zh.wav")?, &request_options)
        .await?;
    Ok(result.text)
}
```

`audio_url` is forwarded unchanged; the inference server must be able to resolve it. The official GLM-ASR example uses a server-local relative filename. This service does not upload or download audio. Each transcription sends one request using the documented `glm-asr` model, audio and text content parts, and `max_tokens: 1024`. It never retries; a transport failure is reported as outcome unknown because the model server may have accepted the request.

Each service is bound to a profile name, caller-defined non-secret account scope, and endpoint fingerprint. A scope created for one endpoint cannot be reused with another route. Tests use a mock transport and do not call paid services.
