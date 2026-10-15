# Provider protocol and feature review — 2026-10-12

The SDK separates provider identity, account, wire format, and feature support.
Changing the conversation format does not grant a different account's tools,
audio services, model entitlement, region, or billing product.

This review used official provider documentation and local request/response
contract tests. It did not make authenticated live requests. Documentation
support, implementation checks, and live account availability are separate
facts; unknown cells remain unknown.

## Format selection

The precedence is `RequestOptions.protocol`, then the connection's persisted
`default_protocol`, then its original `protocol`. Callers configure the default
with `ClientSettings::set_default_protocol`; an explicit request overrides it.
Only routes in `connection.protocol_profiles` qualify. Selection checks provider,
credential configuration, billing product, target model, and region before using
the target endpoint. An unavailable format fails before HTTP.

New format profiles are hidden and opt-in. They do not add duplicate entries to
unscoped model resolution. A format-specific model, such as `gpt-audio-1.5`, can
be requested through its explicitly named account even if the default endpoint
does not serve that model.

## Official conversation and search contracts

| Provider / account | Formats wired into the SDK | Relevant feature boundary and official source |
| --- | --- | --- |
| OpenAI API key | Responses, Chat Completions | [Tools](https://developers.openai.com/api/docs/guides/tools) belong to their documented endpoint and model. Chat audio uses a dedicated model and Chat fields. Responses background operations remain a Responses resource. |
| Codex / ChatGPT authentication | Its existing Responses transport | API-key hosted tool declarations do not authorize them on the account transport. The host search service uses the supported search execution route; no API-key credential substitution occurs. |
| Anthropic API key | Messages, OpenAI-compatible Chat | The [compatibility API](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk) ignores structured output, reasoning controls, and caching. Typed requests for those controls are rejected on Chat; select Messages to use native features. |
| DeepSeek | Chat, Messages, Responses | [Responses](https://api-docs.deepseek.com/api/create-response) is stateless and ignores built-in tools. Native search stays on its documented Messages route. Images in the SDK's normalized message roles belong to user messages. |
| Gemini Developer API | GenerateContent, OpenAI-compatible Chat; Interactions through its native service | [OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai) includes functions, JSON and audio input. [Grounding](https://ai.google.dev/gemini-api/docs/google-search) and other native tools retain their native formats. An unrelated Responses endpoint is not inferred. |
| Qwen | Regional Chat / Responses; Beijing and Singapore Messages | The [Messages API](https://www.alibabacloud.com/help/en/model-studio/anthropic-api-messages) has its own model set and `output_config` effort/JSON controls. [Search](https://www.alibabacloud.com/help/en/model-studio/web-search) has format-specific parameters. The Messages search client-identifier prerequisite is not fabricated. |
| MiniMax | Regional Messages, Chat, Responses | [Server tools](https://platform.minimax.io/docs/guides/server-tools) support Messages and Responses search, with different tool types. [Responses](https://platform.minimax.io/docs/api-reference/responses-create) permits `auto`/`none` tool choice. M3 thinking is enabled by non-none effort; M2.x cannot disable thinking. |
| Kimi platform | Regional Chat / Responses; documented international Messages models | [Messages](https://platform.kimi.ai/docs/api/messages) supports top-level cache writes with 5m/1h TTL; content markers are ignored and rejected by the SDK. [K3 Chat](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart) uses effort, fixed temperature, and inline/provider-file vision inputs instead of public image URLs. |
| Kimi Code subscription | Chat, Messages, Responses on the existing Coding account | [Coding endpoints](https://www.kimi.com/en/help/kimi-code/membership-guide) and model IDs differ from metered Kimi. [Current model controls](https://www.kimi.com/code/docs/en/kimi-code/models) distinguish K3, K2.8 Preview and HighSpeed; K3's default effort is high. Model/context entitlement remains account-dependent. |
| GLM / Z.AI | Metered Chat; existing regional Coding Messages; documented international Coding Chat / Responses | [Coding endpoint documentation](https://docs.z.ai/devpack/tool/others) specifies separate quota endpoints. Metered audio, embeddings and search are not inherited by Coding profiles. Responses thinking is not exposed without an established parameter contract. |
| xAI | Chat, Messages, Responses | [Responses](https://docs.x.ai/developers/model-capabilities/text/generate-text) declares continuation; [web search](https://docs.x.ai/developers/tools/web-search) uses Responses tools. Continuation still needs the caller's explicit storage/account scope. |
| OpenRouter | Chat, Responses, Messages | Its [server web-search tool](https://openrouter.ai/tool/web-search/) supports all three formats. Provider-native compatibility does not establish every model's audio, JSON, or reasoning features. |
| GitHub Copilot | Existing host-authenticated route | The [supported-model listing](https://docs.github.com/en/copilot/reference/ai-models/supported-models) does not establish a public hosted-search or audio inference contract. Those capabilities are not inferred from a listed model name. |

China and international credentials are independent. Endpoint variants that were
not established by the reviewed regional documentation are not invented.

## Audio and independent resources

Conversation audio, file transcription, speech synthesis, streaming synthesis,
and realtime sessions are separate operations. Conversation compatibility does
not authorize the other services.

| Service | Contract checked and implementation implication |
| --- | --- |
| OpenAI Chat audio | [Audio Chat Completions](https://developers.openai.com/api/docs/guides/audio-chat-completions): user `input_audio` is base64 WAV/MP3; output uses `modalities` plus voice/format. Streaming output requires PCM16. Non-streaming `message.audio` is preserved, and later turns replay its audio ID. The [GPT Audio model](https://developers.openai.com/api/docs/models/gpt-audio-1.5) does not support structured output. |
| OpenAI STT / TTS / Realtime | [Audio overview](https://developers.openai.com/api/docs/guides/audio): these have independent native endpoints, requests and session lifecycles. Switching API-key conversation format keeps explicitly configured service routes. Codex authentication does not acquire API-key audio services. |
| Gemini | [OpenAI audio input](https://ai.google.dev/gemini-api/docs/openai) and [native speech generation](https://ai.google.dev/gemini-api/docs/speech-generation) have different contracts. Speech generation and Live remain native operations. |
| Qwen | [TTS API](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-api) uses the native multimodal-generation contract and regional account; SSE audio and completion URL fields are retained. ASR, realtime and bidirectional TTS use their existing independent services. |
| MiniMax | [HTTP TTS](https://platform.minimax.io/docs/api-reference/speech-t2a-http) carries hexadecimal audio or an explicitly selected URL result. Ordinary streaming TTS and bidirectional TTS retain different state machines. Chat compatibility does not make text models audio-input models. |
| xAI | [TTS](https://docs.x.ai/developers/model-capabilities/audio/text-to-speech) uses `/v1/tts`, native voice/output settings and byte output. [STT](https://docs.x.ai/developers/model-capabilities/audio/speech-to-text) uses `/v1/stt`, the transcription model and file upload ordering. Voice sessions retain their native transport. |
| GLM | [Cloud transcription](https://docs.z.ai/api-reference/audio/audio-transcriptions) uses the dedicated model/endpoint. Self-hosted SGLang audio and realtime APIs remain separate services; a Coding quota route does not inherit cloud transcription. |
| OpenRouter | Audio remains subject to the selected routed model and its documented input/output settings. The OpenAI-specific typed audio output option is not redirected to OpenRouter. |

Embeddings, retrieval/File Search, batch jobs, background responses, remote MCP,
tool search and code execution also use their existing typed service contracts.
The new conversation formats do not create undocumented service endpoints. In
particular, Gemini's independently configured File Search resource remains
usable after selecting Chat; its provider, native service endpoint, credentials
and file-reference scopes still apply.

## Corrections from this review

- Added 17 opt-in format profiles and explicit format mappings.
- Added MiniMax Responses search; enabled xAI Responses search/continuation and
  OpenRouter search across its three documented formats.
- Added typed OpenAI Chat audio output, input validation, lossless audio payload
  preservation and audio-ID replay; permitted documented Gemini Chat audio input.
- Rejected Anthropic compatibility options that the upstream silently ignores.
- Corrected Anthropic per-message effort's beta header to
  `mid-conversation-output-config-2026-07-01`, per the
  [effort documentation](https://platform.claude.com/docs/en/build-with-claude/effort).
- Corrected MiniMax M2.7 context limits and Kimi Code model/effort metadata from
  their official model documentation.
- Enforced Kimi's documented cache and K3 Chat input constraints.
- Added MiniMax Chat video URL, base64, resolved attachment and scoped uploaded
  file encoding, with format/size limits. New Chat routes use the documented
  completion-token field and request streaming usage. Models without tunable
  effort reject effort settings instead of silently ignoring them.
- Preserved HTTP status classification and Retry-After for malformed/truncated
  error bodies; malformed successful responses still fail strict decoding.
- Kept host provider identity independent of wire format, and shared credentials
  only across declared routes on the same region and billing product.

## Evidence and verification

Final local results: the isolated SDK full run passed 2,450 tests across 204
suites with `realtime-websocket` enabled; the canonical working checkout passed
15 protocol/feature regressions; Harness passed 77 provider-config tests and
3 DeepSeek hosted-search integration tests. All reported runs exited 0.
The temporary Harness Cargo path override was removed from its lockfile.

The isolated SDK full-run command is
```sh
cargo test --offline --tests --features realtime-websocket --no-fail-fast
```
Local transport tests require permission to bind loopback ports. These results
do not constitute a full Harness suite or desktop account acceptance.

`data/capability-matrix-protocols.json` separately dates the new format evidence.
It contains per-model operation cells and source links, with no inferred account
entitlement. The three previous matrices keep their original evidence dates.
`capability_matrix_all` checks that the combined matrices cover every current
builtin model exactly once.

`tests/protocol_feature_contracts.rs` checks endpoint/body selection, unsupported
controls, hosted search shapes, audio input/output/replay, cache semantics,
regional constraints and error classification without account credentials.
`tests/protocol_selection.rs` checks explicit/default precedence, persistence and
provider/account/billing isolation. The complete SDK test run also exercises the
existing service codecs and optional realtime WebSocket implementation.

Harness verification uses the updated SDK through a temporary Cargo path patch:
provider-config tests validate account/region/product mapping, and the DeepSeek
hosted-search integration tests validate execution, credential reuse and native
result parsing. The patch is a test override, not a published dependency pin.

Live account validation, current provider quotas, model rollout eligibility and
regional deployment access remain unverified. This document does not certify
every remote account or mark documentation-only capabilities as live success.
