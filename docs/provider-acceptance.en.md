# Live provider acceptance

Local contract tests, first-party documentation evidence and live account acceptance are recorded separately. A successful read-only request does not establish inference, audio/Realtime, custom voice entitlement or availability in another region.

## xAI built-in voice catalog

`examples/xai_voice_acceptance.rs` uses the library's `XaiAudioService` with its fixed official default endpoint. It calls `list_voices()`, then `get_voice()` for the first returned voice. This sends at most two GET requests with no generation, mutation, retry or custom voice access. The detail response ID must exactly match the requested ID. An empty catalog leaves the detail check `not_run` and the overall result incomplete.

The default prints a plan without reading credentials or accessing the network:

```sh
cargo run --locked --offline --example xai_voice_acceptance
```

Once live use of the account credential is authorized, use the configured `XAI_API_KEY`:

```sh
cargo run --locked --offline --example xai_voice_acceptance -- --run-read-only
```

Do not place credentials in command-line arguments. Reports contain only check status, error category, HTTP status when available, voice count and timing. They omit credentials, provider error bodies, voice metadata and account identifiers. `environment-xai-key` is only a local scope label, not verified account identity. Each request has a 20-second timeout; failed or incomplete checks exit nonzero. Ordinary `cargo test` does not make these live requests.

2026-09-27 run status: the default preview works. One restricted-environment execution returned a transport error with no provider response. The subsequent network escalation was rejected by automatic approval review because sending the environment credential to xAI had not been explicitly authorized. No bypass or live success followed. Execution of the two read-only GETs awaits explicit user authorization. The evidence matrix retains `live_validation: not_run`; local mocks do not change that status.

Contract: [xAI Voice REST reference](https://docs.x.ai/developers/rest-api-reference/inference/voice).

## Other acceptance work

Other providers require the intended account, region, service entitlement and test input. Generation, upload, asynchronous job and Realtime tests need a concrete operation scope. Device recording/playback and end-to-end interaction also require a host application rather than additional device-layer code in this library.

Live probing is not a substitute for an undocumented contract. The reviewed [Speech API reference](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create) and [TTS guide](https://developers.openai.com/api/docs/guides/text-to-speech) do not establish OpenAI Speech SSE text/audio alignment fields. Do not send guessed fields. The individual service guides record the precise evidence and limits for Z.AI international TTS and discovery of Anthropic inline by-value tool definitions.
