# Exact-profile cloud audio

`LlmClient::audio()` and `ClientSnapshot::audio()` expose cloud file transcription and speech synthesis independently of Chat. The host selects the current session's **exact** provider profile or an explicit audio override, supplies its credential, and owns session-following, microphone capture, playback, permissions, and cancellation. The SDK neither infers an audio model from the chat model nor retries/fails over an audio operation.

```rust,no_run
use lingxi_llm_client::{audio::{AudioRoute, AudioOutput, AudioOperation, SynthesisRequest}, LlmClient, RequestOptions};
async fn speak(client: &LlmClient, options: &RequestOptions) -> Result<(), Box<dyn std::error::Error>> {
    let route = AudioRoute::new("openai-production", "account-production");
    let caps = client.audio().capabilities(&route)?;
    caps.model(AudioOperation::Synthesis, None)?;
    let output = client.audio().synthesize(&route, &SynthesisRequest::new("Hello."), options).await?;
    if let AudioOutput::Stream(stream) = output {
        let audio = stream.collect(16 * 1024 * 1024).await?;
        // Host plays audio.bytes using audio.metadata; headerless PCM is not WAV.
        assert_eq!(audio.metadata.sample_rate_hz, Some(24_000));
    }
    Ok(())
}
```

The snapshot facade freezes configuration for dispatch. Profile groups, aliases, and backup credentials do not participate in audio routing. `AudioRoute.account_scope` must be nonempty, and a request-local account scope must match. The SDK stores no credential. Common adapters require `RequestOptions.credential`; request-local authenticator/finalizer hooks are rejected because most native services do not implement them. Restricted ChatGPT/Copilot authentication cannot acquire cloud audio through this facade.

`AudioCapabilities` separates file transcription, live ASR, complete-text synthesis, incremental text synthesis, and native realtime. Streaming audio output from a complete-text request is **synthesis**, not incremental text input. Each descriptor reports `common_adapter`, a provider-specific advanced entry point, explicitly declared audio model defaults, voice information, offered formats, and local/provider input bounds. `supports` means a usable common adapter with a declared model. `operation` also exposes provider-only operations. Absent bounds/voice defaults/sample geometry mean unknown; they are never filled from chat capabilities.

| Provider | Common file transcription | Common synthesis | Provider-only distinctions |
| --- | --- | --- | --- |
| OpenAI | Explicit Audio route, declared transcribe/Whisper models | Explicit Speech route; built-in voices, PCM16 LE 24 kHz mono and container formats | Scoped custom voices, translation, native realtime |
| xAI | Native STT; raw PCM/μ-law/A-law requires declared rate/channels | Native TTS has **no model parameter**; discover/select an explicit voice | Live ASR, incremental WebSocket TTS, realtime |
| Gemini Developer API | Unavailable through a dedicated file ASR adapter | Gemini 3.8 TTS Interactions; WAV or headerless PCM16 LE/μ-law/A-law, 24 kHz common default | Multiple speakers, custom voices, native Live |
| Vertex AI | Unavailable through a dedicated file ASR adapter | Distinct Vertex Gemini TTS model IDs, Google Cloud bearer token, explicit project/location, PCM16 LE 24 kHz mono | Location-specific model support, multiple speakers; no agent realtime claim |
| MiniMax | Native ASR, 50 MB upload limit | Native T2A model/format controls; explicit discovered voice; PCM16 LE 24 kHz mono common selection | HTTP output streaming, asynchronous synthesis, incremental text TTS |
| Qwen | Provider-only asynchronous URL-input ASR tasks | Qwen3-TTS-Flash complete text → SSE PCM16 LE 24 kHz mono; explicit workspace scope | ASR polling, AudioGen, live ASR, incremental TTS, native realtime |
| GLM | Hosted GLM-ASR-2512, WAV/MP3, 25 MB; documented 30-second duration ceiling retained in catalog | Mainland hosted TTS remains provider-only because encoding/voice geometry is opaque | International hosted TTS is unsupported; raw ASR streams and realtime retain typed native APIs |
| OpenRouter | Models explicitly discovered with output modality `transcription` | Models explicitly discovered with output modality `speech`; explicit voice; MP3/PCM | No guessed model/voice default. PCM geometry stays unknown and cannot be played as assumed OpenAI PCM |

Profiles using fixed provider-native services must name that provider's documented origin. API account region is reported separately from the client's presentation region. Qwen requires the selected profile's `extra.workspace_id`; Vertex uses the selected profile's `signing.project` and `signing.region` (catalogued global default if absent). These values never come from another profile. Model/voice catalogs describe API contracts, not credential/account entitlements; credential readiness and live connectivity remain host concerns.

`TranscriptionResult` retains normalized text/timing, exact route provenance, request identity, native fields, and reported usage. `AudioStream` yields encoded audio bytes unchanged and is pull-driven. Dropping it closes the underlying response. `collect(max_bytes)` requires an explicit positive limit; exceeding it reports `MediaTooLarge` with dispatch `Accepted`. Qwen usage is available only after the provider's terminal completion is validated. Headerless PCM has explicit byte order/rate/channel metadata; no container header or resampling is fabricated. Provider-owned URLs are not downloaded. Missing provider usage stays optional rather than becoming zero.

`AudioError.kind` supplies a host-facing class; `dispatch` distinguishes `NotSent`, `Rejected`, `Unknown`, and `Accepted`. The original typed provider error remains as `source`. Unsupported operation/model/format and known file limits are checked before reading input or transport. A deadline cancels the operation and conservatively reports uncertain provider acceptance; it never authorizes replay.

Native realtime uses the typed agent convenience connectors (`OpenAiClient::connect_agent_realtime` and `GoogleClient::connect_agent_live`). `agent_conversation` and `native_realtime_contract` describe their enabled normalization contract. The connected `RealtimeControl::capabilities()` reports actual configuration-specific flags. The host must still route every tool call/cancellation through its permissions and execution lifecycle. xAI, GLM, Qwen, and Vertex do not acquire agent conversation support merely by providing audio frames or a realtime endpoint.

Verification uses mock transports, including exact-profile rejection, unsupported/oversize input admission, independent defaults, single dispatch after uncertainty, PCM preservation, byte-limited collection, and terminal usage. These tests do not verify provider credentials, account entitlements, hardware playback, or live provider billing.
