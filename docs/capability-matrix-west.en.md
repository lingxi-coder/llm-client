# Western provider capability evidence matrix

[简体中文](capability-matrix-west.md)

The [machine-readable matrix](../data/capability-matrix-west.json) covers every built-in catalog outside the OpenAI/Gemini and China matrices: Anthropic, xAI, OpenRouter and GitHub Copilot. It contains **4 providers, 6 profiles, 408 raw model rows and 7,878 model-operation cells**, plus **102 independent service records and 62 first-party sources**. The matrix was most recently reviewed on **2026-09-27**; older records retain their individual evidence dates.

| Profile | Model rows | Operations per row | Cells | Supported | Unsupported | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anthropic` | 12 | 20 | 240 | 121 | 25 | 94 |
| `github-copilot` | 33 | 14 | 462 | 0 | 0 | 462 |
| `grok-anthropic` | 7 | 18 | 126 | 3 | 4 | 119 |
| `grok-responses` | 7 | 14 | 98 | 26 | 11 | 61 |
| `grok` | 7 | 16 | 112 | 16 | 4 | 92 |
| `openrouter` | 342 | 20 | 6840 | 1246 | 658 | 4936 |
| Total | 408 | — | 7,878 | 1,412 | 702 | 5,764 |

This is a documentation evidence snapshot, **not runtime configuration, an implementation completion checklist, or account acceptance testing**. Every model and service has `live_validation=not_run` and `account_region_validation=unknown`. Reading the public OpenRouter directory does not call a model or validate account access, prices, regions, tool execution or actual responses.

## Reading the evidence

The schema matches the [OpenAI/Gemini matrix](capability-matrix-openai-gemini.en.md): `operations` defines precise operations and endpoints; `sources` provides first-party URLs, review dates and sections; `profiles[].models[].cells` records each model/operation pair; `services` records independent operations. Each profile covers structured output, caching, hosted tools, retrieval, embeddings, batch/background, and audio/realtime; Anthropic also has separate client-toolset operations.

- `supported`: affirmative documentation for the operation or an explicitly bounded model family. Parameter combinations and every upstream route are not guaranteed.
- `unsupported`: documentation explicitly excludes the model, endpoint or combination.
- `unknown`: reviewed evidence is insufficient. Missing parameters, missing directory entries, similar names or failed page fetches are not negative evidence.
- `availability` is independent. A model page or public router entry marked `documented` is not an account-level callability guarantee.

Model-level `batch.submit` cells concern an independent Batch service. A Messages profile does not thereby accept Batch parameters at `POST /messages`. xAI service records are indexed under `grok`; this does not copy credentials or endpoint configuration into the other protocol profiles.

## Confirmed boundaries

Anthropic has affirmative evidence for the catalog's JSON Schema, prompt caching and Batch support. Haiku 4.5 supports code execution but excludes programmatic tool calling. Sonnet 5 is not explicitly in the reviewed Tool Search model table, so it remains unknown. Web Search, Web Fetch and MCP support does not spread to every model merely because another tool works. JSON Schema with citations is explicitly excluded. Anthropic documents no first-party embedding model; Voyage support is not assigned to Claude. [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [code execution](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool), [embeddings](https://platform.claude.com/docs/en/build-with-claude/embeddings).

Stable Browser and Computer client toolsets are separate operations: `browser_toolset_20260801` and `computer_toolset_20260801`. Their compatibility tables name eight model versions; Anthropic's platform model-ID tables give the corresponding IDs: `claude-fable-5`, `claude-fable-5-1`, `claude-mythos-5`, `claude-mythos-5-1`, `claude-opus-4-8`, `claude-opus-5`, `claude-opus-5-5`, and `claude-sonnet-5`. Six matching IDs (Fable 5 and 5.1, Opus 4.8, 5 and 5.5, and Sonnet 5) appear as raw rows in the current catalog. Limited-availability Mythos 5 and 5.1 are not in that catalog, so the matrix does not fabricate model rows for them. The other six Anthropic catalog models remain unknown; support is not inferred from a neighboring model. Platform-level evidence records Google Cloud/Agent Platform as supported and these stable versions on Microsoft Foundry as unsupported; Foundry's older beta computer tool is a separate operation. These are vendor-contract findings, not claims about client implementation, account permissions or regional availability. [Browser use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/browser-use-tool), [Computer use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool), [Claude on Google Cloud](https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai), [Claude in Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry).

xAI's three primary protocols have separate evidence. The Responses schema marks `background` as currently unused compatibility metadata; acceptance or rejection of `background=true` was not live-tested. Deferred Chat results can be retrieved once within 24 hours, unlike reusable persisted Responses. Model cards support Batch for Grok 4.20, 4.20 Multi Agent and 4.3, and explicitly exclude 4.5, 4.6, 4.7 and Grok Build 0.1. Collection management uses a Management API key while search uses the inference API key. Dedicated voice-service evidence covers REST TTS, bidirectional TTS WebSocket, built-in voice listing, and Custom Voices create, list, get, metadata update, delete, and reference-audio download operations. The Custom Voices API create endpoint is Enterprise-gated; its official page says Custom Voices is available only in the United States except Illinois. These remain service-level records: live account and regional access stay unknown, and no Grok catalog model gets voice-operation support inferred from a service entry. The TTS page says it was last updated 2026-09-19; the Custom Voices page says 2026-08-04. [TTS and streaming TTS](https://docs.x.ai/developers/model-capabilities/audio/text-to-speech), [Custom Voices](https://docs.x.ai/developers/model-capabilities/audio/custom-voices), [Responses reference](https://docs.x.ai/developers/rest-api-reference/inference/responses), [Deferred](https://docs.x.ai/developers/advanced-api-usage/deferred-chat-completions), [Grok 4.7 card](https://docs.x.ai/developers/models/grok-4.7), [Collections](https://docs.x.ai/developers/files/collections/api).

OpenRouter response caching, prompt caching and independent services are distinct. Claude's `cache_control.ttl=1h` does not stand in for GPT-5.6-and-later `prompt_cache_options.ttl=30m`. Chat audio input/output does not establish standalone STT/TTS. Embeddings, reranking, audio and Batch have independent service evidence; reranking does not establish a managed vector store. Hosted shell and Tool Search are documented for Responses/Messages, so those Chat operations are unsupported. Batch deletion applies to terminal resources; cancellation remains unknown. [Prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching), [server tools](https://openrouter.ai/docs/guides/features/server-tools), [Batch](https://openrouter.ai/docs/batch-quickstart).

All operations for the 33 GitHub Copilot catalog rows remain unknown. Public REST documentation covers administration and usage, while CLI/SDK documentation covers agent products. Neither establishes native OpenAI, Claude or Gemini capabilities at the catalog's raw `api.githubcopilot.com` HTTP route. [REST documentation](https://docs.github.com/en/rest/copilot), [agent product boundary](https://docs.github.com/en/copilot/responsible-use/agents).

Foundry service evidence separates hosting options and tool versions. It adds no model rows and does not copy platform support into model cells. The five added records are:

| Service record | Documented scope |
| --- | --- |
| `anthropic.foundry.tool_search` | Both hosting options support regex/BM25 `20251119`, subject to the tool's model compatibility table. |
| `anthropic.foundry.azure.web_fetch` | Azure hosting supports only `web_fetch_20250910`; later versions, dynamic filtering, `use_cache` and `response_inclusion` are excluded. |
| `anthropic.foundry.anthropic.web_fetch` | Anthropic hosting supports the four current versions: `20250910`, `20260209`, `20260309` and `20260318`, with their model and parameter restrictions. |
| `anthropic.foundry.remote_mcp` | Both hosting options support the `mcp-client-2025-11-20` beta connector; `mcp_toolset` is not a stable toolset. |
| `anthropic.api.mcp_tool_list_pinning` | Pinned lists and `mcp_tool_listing` replay under `mcp-client-2026-09-15` have affirmative evidence for the Claude API only. Foundry remains unestablished, not a tested rejection. |

The Tool Search and MCP rows in [Features overview](https://platform.claude.com/docs/en/build-with-claude/overview) lack the Anthropic-hosting-only marker. [Web Fetch](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool) gives the explicit hosting/version split, and [MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector) separately scopes the 2026 feature. The existing unsupported stable Browser/Computer toolset records for Foundry are preserved. These platform records do not validate actual deployments or accounts.

## OpenRouter directory evidence

The review used the [public directory with all output modalities](https://openrouter.ai/api/v1/models?output_modalities=all) because the default list filters to text output. It matched 328 of 342 local rows. The 14 missing IDs are retained in `scope.openrouter_directory.missing_catalog_models` and remain unknown; absence does not prove retirement.

Only affirmative metadata establishes support. `supported_parameters.structured_outputs` supports JSON Schema; `response_format` alone does not. Architecture metadata establishes 27 audio-input and 4 audio-output rows. A matching `:batch` variant establishes Batch evidence for 63 model rows. Aggregated model metadata still requires a suitable provider endpoint and compatible parameters. Dynamic router metadata does not promise a fixed upstream model. [Model directory documentation](https://openrouter.ai/docs/guides/overview/models), [structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).

## Validation and maintenance

`tests/capability_matrix_west.rs` checks every remaining raw provider TOML row, complete operations across the seven common categories plus Anthropic client toolsets, dated first-party sources, closed source/operation references, independent services, and the key limitations above. New models or providers require explicit cells; Unknown is valid.

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 cargo test --test capability_matrix_west
```

Tests validate local evidence structure and recorded facts only. They do not access the network, run inference or validate accounts. Inspect service implementations and their contract tests separately for runtime support.

2026-09-26 follow-up: the client selects the latest first-party Anthropic Code Execution version, `code_execution_20260521` without a beta header, with models checked against the official compatibility table. [Typed execution, file input, and container reuse](anthropic-code-execution.en.md) preserve native usage without converting execution counts into time-based fees. Documentation evidence remains separate from live account validation.

Anthropic Skills create/list/get/delete and the corresponding version operations now have independent service evidence. This does not establish model execution or account validation. See the [Skills API](https://platform.claude.com/docs/en/api/http/skills).

Four further Foundry service records distinguish Code Execution and Programmatic Tool Calling by hosting: explicitly supported on Anthropic hosting and excluded on Azure hosting. Model compatibility, resource endpoints and account access require separate validation. [Foundry hosting limits](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry).

Files service records separately cover upload, list, metadata, download, delete and `ids[]` lookup for each Foundry hosting option: supported on Anthropic hosting, excluded on Azure hosting. First-party batch metadata lookup has its own record. Downloads require generated downloadable files; batch lookup can omit missing IDs. Account availability remains unverified. [Files API](https://platform.claude.com/docs/en/build-with-claude/files).

Foundry Skills are recorded separately by hosting: Anthropic hosting supports custom Skill and version CRUD/list operations, while Azure hosting does not. Version content download is explicitly excluded on Foundry, including Anthropic hosting. These 18 service rows preserve the distinction between documented API support and untested account entitlement. [Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview).


xAI built-in voice metadata (`GET /v1/tts/voices/{voice_id}`) has independent operation evidence, separate from custom voice management. Live acceptance remains unverified.
