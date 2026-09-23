# Project Review and Fix Record

[简体中文](review.md)

The initial review covered client routing and failover, authentication and transport boundaries, built-in codecs, stream parsing, the model catalog, usage and pricing calculations, and API documentation builds in an independent repository. That round of fixes preserved the existing API shape and added no dependencies. The later built-in HTTP functionality is described separately below.

## Issues Fixed

| Priority | Issue and impact | Fix location |
| --- | --- | --- |
| High | `HttpRequest` Debug output contained authentication headers, URL credentials, and prompts, which could leak into logs | `src/transport.rs`: hide the URL, header values, and body; retain only the request method, header names, lengths, and timeout |
| High | `stream()` encoded a non-streaming request when using default options; `complete()` could also send a streaming request unintentionally | `src/client/failover.rs`: let the high-level method determine the mode without changing the caller's options |
| High | In-stream Chat/Gemini errors were ignored, and Responses failed events lost their error classification | `src/codecs/*/stream.rs`: recognize errors and preserve their semantic classification |
| High | EOF without a termination marker was treated as normal completion, potentially passing truncated output to later processing | Four base stream decoders: return `StreamInterrupted` while preserving the normal termination path |
| Medium | A qualified `profile/model` reference could select another connection in a group with the same name | `src/client/resolve.rs`: prioritize exact profile names |
| Medium | Anthropic model catalog requests lacked a version header | `src/directory/anthropic.rs`: send `anthropic-version` and support a configured version |
| Medium | Some codecs treated non-2xx responses, invalid JSON, or missing required fields as empty successes | Anthropic/Gemini/Responses decode: validate status and response shape |
| Medium | Text documents were not Base64-encoded; Responses URL files used the wrong field | Gemini/Responses encode: encode text and use `file_url` for URLs |
| Medium | SSE parsing mishandled CR line endings, CRLF split across chunks, and data fields without a colon | `src/framing/sse.rs`: normalize line endings and preserve empty data lines |
| Medium | Gemini token addition could overflow; cache reads and writes totaling more than the input were still considered valid | Gemini decode and `src/client/usage.rs`: prevent overflow and validate count consistency |
| Medium | Invalid rates, multipliers, or amount overflow still yielded successful cost estimates | `src/client/pricing.rs`: return `CostUnavailable` |
| Medium | Large numbers in peak-period windows could panic; a window ending at midnight could not be represented | `crates/agent-api/src/protocol/provider.rs`: validate before calculating and support the end time `24:00` |
| Medium | Broken internal Rustdoc links caused strict documentation builds to fail | Shared protocol documentation: correct nonexistent links |

## Documentation and Verification

- [API documentation](api.en.md) covers integration, requests, events, configuration, authentication, transport, the catalog, costs, errors, and extension points.
- Guides are included in Rustdoc with `include_str!`, and doctests compile-check their Rust examples.
- Behavior fixes have regression tests using mock transport, so no real API key is required.
- Initial review verification (excluding the later built-in HTTP functionality): 228 unit/integration tests and 4 documentation examples passed; 24 behavior regression tests were added relative to baseline. Formatting, Clippy (warnings treated as errors), and strict Rustdoc builds passed, and all local documentation link targets existed.

```sh
cargo test --workspace --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked
```

## Later Fixes and Boundaries the Host Still Handles

- This round of fixes scoped the primary connection's credentials to that connection; the host must supply separate credentials for backup connections by profile. Complete and streaming responses expose the connection that actually succeeded so callers can price usage against it.
- These subsequent fixes addressed Gemini tool IDs and signatures, OpenAI message replay and truncated responses, request deadlines, SSE/AWS frame size limits, and peak price configuration validation. They are not included in the initial review's test counts above.
- A built-in HTTP client was subsequently implemented; WebSocket scheduling, OAuth refresh, and automatic catalog merging are still unavailable.
- No real provider network requests, actual billing checks, or online availability checks for every provider configuration were performed; the static catalog still needs maintenance.

## Follow-up: Built-in HTTP Client

`LlmClientBuilder::new(&profiles)?` provides a built-in `HttpTransport`, `SystemClock`, and API key / Bearer authenticators by default. `LlmServices` was removed, so callers no longer need to assemble services; use `with_transport()` to inject custom transport or `with_clock()` to set a test clock. This changes how the builder is integrated, and previous service-injection calls should migrate to these entry points. Internally, it uses reqwest 0.12 and Rustls. Callers only need a Tokio runtime and request credentials; they do not need to import reqwest or write an HTTP adapter.

Redirects and automatic retries are disabled by default, and the connection timeout is 30 seconds; the total timeout specified by a request covers stream reads. HTTP status and error bodies are preserved for codec handling, while network error messages avoid exposing sensitive request data. See the [transport API](api.en.md#transport-api). The test counts above record the initial review; use current test results for the added functionality.

## Protocol Sources Checked

- [Claude Models API](https://platform.claude.com/docs/en/api/models/list): model-list request header.
- [OpenAI File inputs](https://developers.openai.com/api/docs/guides/file-inputs): file URL and Base64 fields.
- [Gemini generateContent](https://ai.google.dev/api/generate-content): inlineData format.
- [WHATWG Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html): line endings and field parsing.

Verification of the built-in HTTP and clock changes: all 236 unit/integration tests and 4 documentation examples passed; formatting, Clippy (all targets, warnings treated as errors), and strict Rustdoc passed. Seven new local TCP HTTP tests cover requests/responses, redirects, immediate stream reads, total timeouts, interrupted streams, and built-in authentication; a fixed clock verifies costs at different times. No real provider or paid requests were used; the HTTPS trust chain was not verified online.

The HTTP implementation follows the [reqwest 0.12 ClientBuilder documentation](https://docs.rs/reqwest/0.12.28/reqwest/struct.ClientBuilder.html) for connection timeout settings and disabling redirects and automatic retries. TLS connection errors are currently classified as `Transport`; certificate errors are not classified separately.

## API and State Fixes After the Design Review

This round kept the existing separation among codecs, authenticators, transport, and catalog resolvers while fixing request target and credential binding, protocol terminal states, and configuration sync boundaries; it added no dependencies.

- Added `complete_in`, `stream_in`, `web_search_in`, and `web_search_stream_in`. These methods explicitly select a starting profile or group and use the same resolved route to execute the request. When an unqualified native model ID and `profile/model` point to different targets, they return `AmbiguousNativeAndQualified`, preventing primary credentials from being sent to another connection when OpenAI and OpenRouter are both configured. Valid, unambiguous native IDs containing `/` remain usable.
- Anthropic stream decoding preserves `redacted_thinking`. Usage is considered complete only when the final output count is present and counts are numerically consistent; an initial seed or a standalone end marker cannot promote partial usage to final usage.
- Gemini prompt blocks and Responses refusal content retain consistent refusal semantics in complete and streaming responses; truncation and provider errors retain their existing priority.
- When a streaming entry point receives an unsuccessful HTTP status, it preserves the status, collected content, and `Retry-After` even if the error body is interrupted, and evaluates failover under the existing policy. An interruption after a successful stream has already been returned is still not replayed automatically.
- The Anthropic catalog returns an error when the boolean `has_more` is absent, or when `has_more: true` lacks a valid `last_id`, avoiding silent acceptance of an incomplete catalog.
- The client's persisted overrides, cache, and tracking state were consolidated in an internal `ProviderStore`, separate from the effective profile snapshot used for requests. `prepare_provider_sync` creates an operation that does not borrow the client; `fetch` retrieves the catalog, and `apply_provider_sync` validates and commits it. File locks and file I/O involved in sync run on Tokio's blocking thread pool. Before committing, it checks fresh disk state, connection configuration, and the configuration directory generation.
- The default `agent` feature of `lingxi-agent-api` retains the full original API. The LLM client disables this feature and re-exports the protocol through `lingxi_llm_client::protocol`. CI checks the minimal feature set separately so workspace feature unification cannot hide problems.
- Models gained optional tri-state capability information distinguishing unknown, supported, and unsupported. The old Boolean fields remain: explicit new metadata takes precedence, an old `true` can establish support, and an old `false` remains unknown. Newly cataloged models without capability facts remain unknown. Capability metadata remains available to the host for decisions, without introducing a new runtime rejection.

During migration, existing JSON configurations remain readable; Rust `ModelProfile` literals need `capability_support: None`. Callers relying on unqualified model names should switch to the explicit-profile `_in` methods if they encounter ambiguity. Old imports directly from `lingxi-agent-api` remain valid.

The fetch phase of configuration sync can be canceled without writing to disk. Once the commit enters the file-writing phase, cancellation can still leave updated files and an outdated in-memory snapshot; the host should reload configuration in that case. Synchronous configuration-management methods remain blocking interfaces, so async hosts should choose an appropriate execution environment.

Verification for this round: 335 unit/integration tests and 12 doctests passed; formatting, Clippy (warnings treated as errors), and strict Rustdoc passed. Builds/tests for both the default and minimal feature sets of the shared protocol passed, as did strict Rustdoc with minimal features. Offline packaging and an in-package build of the shared crate passed; the client package file list excluded local `.omx` runtime state. Only mock transport and loopback HTTP were used in this round; no real provider was called and live rates were not verified.

## Fixes After the Second Full Verification Pass

This round continued to use GPT-6 Luna (max) for implementation and an independent reviewer for review. All six issues were first reproduced with regression cases or an independent downstream project; no runtime dependencies were added.

- Non-streaming and streaming entry points share bounded error-body reading. After a non-2xx HTTP status has been received, a connection interruption or error-body timeout no longer erases the status, response headers, or `Retry-After`; at most 64 KiB is retained. Successful responses still require a complete body, and the total request deadline continues to limit backup-connection attempts.
- Ordinary Responses responses preserve all reasoning summary text blocks in order, matching streaming summary content.
- Anthropic usage validation covers separate cache read and write counts: when present, fields must be unsigned integers, and the normalized total must not overflow. Omitted values, numeric zero, and valid cache counts greater than uncached input are allowed.
- Gemini uses explicit `supportedGenerationMethods` to exclude models that cannot be called with `generateContent`; the static catalog also records the corresponding method metadata. Missing method information still means unknown and cannot clear an existing unsupported fact. Sync stores exclusions by connection, preventing old models from reappearing from cache or later allowlist expansion; a new observation that explicitly supports generation or an explicit configuration replacement can remove an exclusion. Other models absent from the catalog remain under the existing merge semantics.
- An invalid Gemini pagination token type fails sync and leaves the old configuration unchanged; an absent value, empty string, and compatible null still indicate the last page. An optional method with a default implementation was added to the catalog trait to convey operation-compatibility facts, preserving existing uses of `decode_page`, `ModelPage`, and `LiveModel`.
- Documentation consistently imports types through `lingxi_llm_client::protocol`. A separate `tests/downstream-docs` project was added to compile Rust examples from the README and two guides using the direct dependencies declared in the README; CI runs this check separately so workspace development dependencies cannot hide downstream compilation errors.

Final verification: all 352 unit/integration tests and 12 workspace doctests passed; 13 documentation examples in the independent downstream project passed, with 17 behavior regression tests added compared with before this round of fixes. Formatting, Clippy (all targets, warnings treated as errors), strict Rustdoc, minimal-feature tests of the shared protocol, and a standalone client build passed. Offline packaging and an in-package build of the shared crate passed; the client package file list contained no local runtime state, downstream test project, or build artifacts. Independent review found no remaining substantive issues with the six fixes.

Catalog observations for a given profile have no version number, and concurrent fetch results are applied in host commit order. If the most recently started refresh must take priority, the host should serialize refreshes or discard stale results before commit. Verification used only mock transport and local loopback HTTP; no real provider was called and live bills were not checked.
