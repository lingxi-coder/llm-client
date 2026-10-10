# OpenAI / Gemini capability evidence matrix

[中文](capability-matrix-openai-gemini.md)

The [machine-readable matrix](../data/capability-matrix-openai-gemini.json) records first-party documentation evidence reviewed on **2026-10-09**. It covers every model in the raw OpenAI and Gemini catalogs, including embedding, realtime, image, music, video and agent entries excluded from ordinary completion presets. Older model and source records retain their original review dates. This P0 documentation audit does not modify runtime routing or attest to client implementation or live account validation.

## Coverage and interpretation

| Profile | Models | Operations per model | Supported | Unsupported | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `openai` | 50 | 21 | 332 | 44 | 674 |
| `gemini` | 35 | 21 | 150 | 213 | 372 |
| Total | 85 | 1,785 model-operation cells | 482 | 257 | 1,046 |

Three removed Gemini preview rows remain in `retired_models` as historical evidence, outside active coverage: **63 historical cells** (9 Supported, 12 Unsupported, 42 Unknown). Flash-Lite Preview shut down on 2026-05-25; Pro Image Preview and Flash Image Preview shut down on 2026-06-25. [Google deprecations](https://ai.google.dev/gemini-api/docs/deprecations).

There are also **41 independent service operations** for vector stores, cache resources, File Search stores, the Gemini model directory, batches, background results and audio services, each with affirmative documentation evidence. The matrix references **97 first-party sources**. A service endpoint does not establish that every catalog model can use it.

Gemini's native `models.list` and `models.get` are recorded as service operations and do not add model capability columns. The official list returns `supportedGenerationMethods`; filtering for `embedContent` is a client-side catalog projection, not evidence that every returned model supports embeddings. Embedding support remains grounded in each model's documentation. These API references do not verify live account or regional access.

Gemini Batch submission is split by API route: `gemini.batch.generate_content.submit` (`batchGenerateContent`) and `gemini.batch.embed_content.submit` (`asyncBatchEmbedContent`). Only `gemini-embedding-001` and `gemini-embedding-2` have affirmative evidence for the embedding-specific operation. The former is covered by the Embeddings guide's Batch API statement for the Gemini embedding model family; the latter also has a direct create example in the Batch API guide. Other models remain `unknown` for embedding batches. Creation is model-scoped; get, list, cancel, delete and results are service-scoped operations on shared Batch resources. `interactions.batch` is a separate API and does not establish support for Gemini Batch.

- `supported`: an official source affirmatively documents the operation, including an explicitly identified model family or alias where applicable.
- `unsupported`: an official source explicitly excludes the operation for the model or endpoint. Omission from documentation is insufficient.
- `unknown`: the reviewed evidence does not establish the model-and-operation combination. Its source identifies the reviewed context, not negative proof.

Every row retains `live_validation: not_run` and `account_region_validation: unknown`. Scope is the first-party international API used by these two profiles, without extrapolation to Azure, Vertex, arbitrary regions, subscriptions or accounts. Missing model pages, similar names and floating `latest` aliases do not inherit another model's capabilities.

## Structure and use

`operations` defines each operation, capability category, model/service scope and endpoint. `sources` holds official URLs, evidence dates and relevant sections. `profiles[].models[].cells` contains an explicit status, source IDs and evidence basis for every model-operation combination. `services` records independent resource operations. `retired_models` preserves removed model rows with their original operation evidence and explicit shutdown availability. Every cell's references must resolve in the source registry.

Separate `availability` records use `documented`, `restricted`, `deprecated`, `shut_down` or `unknown`. `documented` means the official page lists the model, not that the current account can call it. Feature cells describe documented model contracts; `shut_down` overrides callability even when historical feature cells remain supported.

`audio.input` and `audio.output` describe modalities only, without establishing dedicated STT, TTS or Live endpoints. Music output is not speech synthesis, and audio understanding is not a file-transcription API. The computer-use operations describe the tool contract; client actions remain the host's responsibility.

## Differences preserved in the evidence

| Case | Matrix treatment |
| --- | --- |
| `gpt-4o-2024-05-13` | JSON Schema is Unsupported; later snapshots' Structured Outputs are not inherited. |
| `gpt-5.2-pro`, `gpt-5.4-pro` | Model pages explicitly exclude Structured Outputs; GPT-5.4 Pro also excludes Code Interpreter and hosted shell. |
| GPT-5.6 and later | Explicit cache breakpoints and `prompt_cache_options.ttl=30m` have separate operations from earlier `prompt_cache_retention=24h`. |
| `gpt-5.6` | Its documented alias to `gpt-5.6-sol` establishes the target, without guessing from its name. |
| `gpt-5.3-codex-spark` | Reviewed sources do not establish operation-level HTTP API contracts; cells remain Unknown. |
| Gemini Interactions | Explicit caching and Batch are explicitly unsupported; independent cache resources and Batch API remain separately documented. |
| Gemini 3 versus Deep Research | Ordinary Gemini 3 models exclude Interactions Remote MCP; Deep Research agents have separate positive MCP evidence and require background execution. |
| Retired Gemini preview IDs | Flash-Lite Preview (2026-05-25), Pro Image Preview and Flash Image Preview (2026-06-25) are outside active coverage and retained in `retired_models`. |
| Gemini 2.5 Flash / Flash-Lite / Pro | Official access restriction to prior active users is recorded as `restricted`; account eligibility is untested. |
| `lyria-3-pro-preview` | Interactions explicitly lists the ID, but the attempted model page identifies another model; that feature table is not copied. |

Sources: [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [OpenAI caching](https://developers.openai.com/api/docs/guides/prompt-caching), [GPT-5.4 Pro](https://developers.openai.com/api/docs/models/gpt-5.4-pro), [Gemini Models API](https://ai.google.dev/api/models), [Gemini Interactions](https://ai.google.dev/gemini-api/docs/interactions-overview), [Gemini Deep Research](https://ai.google.dev/gemini-api/docs/deep-research), [Flash-Lite Preview shutdown](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-preview), and [Gemini 2.5 Flash access restriction](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash). Other operation-specific sources are resolved through each cell's `source_ids`.

OpenAI background requests may use `store=false`, with temporary server storage for polling; starting a resumed stream requires creation with `stream=true`. Gemini Interactions rejects the combination of background execution and `store=false`. These storage contracts must remain distinct. [OpenAI Background](https://developers.openai.com/api/docs/guides/background), [Gemini Interactions](https://ai.google.dev/gemini-api/docs/interactions-overview)

A generic Gemini model-card claim of caching support does not independently establish that every model accepts explicit `cachedContent` references. Those model cells can remain Unknown while `cachedContents` create/get/list/update/delete operations have independent API evidence. Fill such gaps with model-and-operation evidence rather than service-level inference.

## Maintenance and validation

Add a complete matrix row when a raw catalog model changes. Move explicitly shut-down rows into `retired_models` when removing them from the catalog; preserve their historical cells and shutdown sources. A new operation requires an explicit cell for every model in the profile; start with Unknown when evidence is missing. Documentation review dates are not live validation dates, and unperformed live calls must never be marked passed.

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 cargo test --test capability_matrix_openai_gemini
```

The [tests](../tests/capability_matrix_openai_gemini.rs) require exact active raw-TOML model coverage, disjoint complete retired rows with shutdown evidence, complete operation rows, all seven capability categories, valid evidence bases, traceable first-party sources and dates, independent service scope, and regressions for snapshot/cache-generation/agent/endpoint distinctions. They make no network or provider calls and do not treat a parseable URL as an ongoing online link check.

OpenAI `GET /v1/models` is recorded as an independent service operation. It returns account-visible models and basic metadata, without embedding capability flags. Filtering documented embedding IDs does not change other model capability cells; the complete directory is retained for callers. [Models list](https://developers.openai.com/api/reference/resources/models/methods/list).

GPT-Live primary WebSocket has its own service record (`openai.live.connect`), separately from Realtime: the fixed `/v1/live/sessions` route uses session.start/started and session.close/closed, with explicit Responses delegation and cumulative session usage. The service evidence does not create a Chat catalog row or establish account access. [Primary WebSocket guide](https://developers.openai.com/api/docs/guides/voice-websockets).
