# LLM client implementation handoff

## Host runtime integration merge (2026-09-27)

The user authorized merging `codex/lingxi-runtime-integration` (`9b0323f10f76c5834acafedaa04470c4e96346f2`, including `ec8876c`) into main after the provider-capability and review-fix commits (`85946bf`, `95aee15`). This section supersedes the historical no-commit/dirty-tree status below.

The host execution API now uses the current `ClientSnapshot` architecture: `prepare_draft_on`, `prepare_on`, exact token counting, single-dispatch calls, immutable pricing observations, chunk-level usage observations, Responses WebSocket sessions, native request controls, exact UTF-16 serialization, and AWS SigV4. Main's ChatRequest/ChatResponse names, scoped continuation, unified prompt-cache policy, provider capabilities and file lifecycle validations remain authoritative. Prepared dispatch rechecks provider file expiry before the host dispatch marker; batch stream observation retains continuation and Anthropic container metadata.

This merges library functionality, not a downstream dependency upgrade. LingXi's external harness-runtime still pins the older integration revision and uses the older request API; adopting the new main commit requires a separate host API migration and pin update. No changes to that repository, no push and no live provider calls were performed.

Verification (all Cargo runs offline; no live provider calls):

- All-feature tests: `/private/tmp/llm-merge-all3.log` covered the full suite. Final outcomes total 1,906 passed / 8 ignored after targeted reruns: `/private/tmp/llm-merge-retest.log` passes all 14 HTTP loopback tests, 18 prepared-call tests and 6 streaming-upload tests; `/private/tmp/llm-merge-docfinal.log` passes 39 library doctests / 8 ignored. Initial failures were sandbox-denied loopback binds, an incomplete new test fixture, and missing doctest build artifacts during overlapping Cargo commands; final reruns were serialized.
- Independent downstream docs: `/private/tmp/llm-merge-downstream-final.log`, 273 doctests plus 1 unit test passed; 62 ignored.
- Strict all-target/all-feature Clippy: `/private/tmp/llm-merge-clippy-final.log`, passed.
- Warnings-as-errors all-feature Rustdoc: `/private/tmp/llm-merge-rustdoc.log`, passed.
- Formatting, diff whitespace and default dependency isolation passed; tokenizer backends remain absent from default normal dependencies.
- Offline packaged-source verification: `/private/tmp/llm-merge-package.log` (677 files). This handoff is the only status-only edit following packaging.

The historical round-26 results below describe the earlier provider feature inventory, not the separate integration branch's delivery status.

## Round 26 unified verification complete (2026-09-27)

Latest user instruction: finish the remaining work, with functionality first and unified tests afterwards. Both phases are now complete for the finite confirmed feature inventory. GPT-6 Luna max workers added bounded regressions in parallel; all lanes are frozen/completed. No persistent goal, commit, push, publication, or live provider call. Preserve the shared dirty main tree (baseline a419da374ff51448c42f9039eeeefeb066bb706a). Snapshot/cache architecture remains outside this session's ownership.

## Round 25 features, now locally verified

- OpenAI Code Interpreter: `CodeInterpreterConfig.container` accepts a scoped existing `OpenAiContainerRef`; `files` accepts automatic-container file mounts. Official endpoint, profile/account, safe/unique file IDs, readiness/expiry checks; explicit mode rejects auto-only settings. Qwen rejects OpenAI-only options. Executor pins Code Interpreter calls and disables missing-file replay.
- Gemini Chat: typed `ContentBlock::Audio` maps to inlineData on GenerateContent/Vertex. User role, standard Base64, declared formats, OpenRouter output-setting rejection, 20 MB complete-body cap. Mixed automatic attachments are checked in their full inline representation before uploads; larger files can be uploaded explicitly and passed as scoped references.
- Gemini Live: JPEG/PNG input frames, batched tool results, corrected tool table byte accounting, atomic incoming call batch validation/registration, and cleanup after successful outbound queue acceptance. QueueFull/Closed/frame failures retain retryable names. `RealtimeCodec::input_queued` is a default no-op callback used for this local cleanup, not a provider delivery ACK.
- OpenAI Realtime: ClearAudio plus shared RetrieveItem/DeleteItem/TruncateAudio commands. Unsupported providers reject the new variants explicitly. GPT-Live gets typed startup history (128-message local cap, provider token cap) and opt-in store, default false.
- xAI Realtime: typed VAD/reasoning/transcription/tools/speed/replacements/resumption settings; JSON/binary audio including Opus; clear, scripted messages and per-response instructions; lifecycle event projection. `XaiRealtimeResumeRef` binds model, normalized route and explicit credential scope. Connect rejects raw/unscoped or conflicting query identity.
- Qwen Omni and GLM Realtime: ClearAudio. No guessed FinishSession event.
- Gemini Speech: typed two-speaker dialogue and per-turn styles, sharing single/SSE synthesis paths.
- Gemini Voices Beta: new `gemini_speech/voices.rs` exports create/list/get/delete, typed prompted/replicated input, explicit storage, filters/pagination, scoped references, ID/key mutual exclusion, redacted Debug and scoped speech-request helper. No automatic retries or downloads.
- Vertex TTS: new public `hosting::vertex::speech` module. Explicit project/location/account/model and caller bearer token, single/two speakers, unary/SSE responses, bounded audio/native data, PCM MIME checks, finish/block reason and premature EOF. Does not add Cloud Text-to-Speech API or legacy first-party GenerateContent TTS compatibility.

New paired guides: gemini-chat-audio, gemini-voices, vertex-speech. Existing affected paired guides and README links updated. New guides registered in downstream-docs source. OpenAI/Gemini matrix now 41 service rows / 97 sources / 77 operation definitions (added four Voices operations); West remains 102/62/148, China 606/86/95. All live_validation values remain not_run.

## Round 26 verification and closure

All confirmed feature tasks and local delivery checks are complete. Do not restart broad provider research or infer new coding tasks from Unknown matrix cells or historical "next round" notes. No pending local tests remain for this slice.

New regressions cover Gemini audio/Base64/roles/full-body/pre-upload limits and valid mixed attachments; Live atomic batches, queue-failure retention and 1,025 completed calls; Code Interpreter network/500 no-replay in complete/stream plus automatic missing-file repair suppression; OpenAI/xAI lifecycle/history/settings/binary/scope; Qwen/GLM clearing; Gemini multi-speaker and Voices CRUD/scope/mutation/ID-key/redaction; Vertex regional auth/wire/PCM/SSE terminal and premature EOF. Fixed a real Debug disclosure: GeminiVoicesError no longer prints raw Provider/InvalidResponse bodies that may contain voice keys. Corrected test assumptions/races and strict Clippy style issues. Chinese Voices guide is translated.

Final logs use `/private/tmp/llm-client-round26-`:

- `all-final.log`: cargo test --locked --offline --all-features --no-fail-fast, exit 0; 1,858 passed / 0 failed / 8 ignored.
- `default.log`: default tests, exit 0; 1,836 passed / 0 failed / 8 ignored.
- `downstream.log`: downstream crate, exit 0; 273 doctests + 1 unit test passed, 62 doctests ignored.
- `clippy-default.log`, `clippy-final.log`: strict all-targets default/all-feature Clippy, exit 0.
- `rustdoc-default.log`, `rustdoc.log`: warnings-as-errors default/all-feature Rustdoc, exit 0.
- `fmt-final.log`, git diff --check: pass.
- `dependency-tree.log`: default normal dependency tree excludes tokenizers/onig/tiktoken-rs/xz2.
- `package.log`: offline package --allow-dirty and packaged source compilation pass; 667 files, 21.0 MiB / 11.0 MiB compressed. Only status documentation was updated after packaging.

All Cargo used CARGO_INCREMENTAL=0 and CARGO_BUILD_JOBS=2. Early failures are retained separately: stale/mistaken test fixtures, a test helper lifetime, and sandbox-denied localhost binds were resolved; final tests ran with local loopback allowed. No external provider calls. All Cargo sessions and worker lanes are closed.

## Remaining evidence boundaries

The bounded current-surface P4/P5 audit found no confirmed missing lifecycle operation in the checked Gemini File Search, GLM Knowledge, Beijing Qwen RAG, xAI Collections, or Gemini/GLM/Qwen/xAI Batch services. Alibaba signed ROA OpenAPI connector update/delete is a different service/auth surface, not a guessed bearer RAG route.

Contracts still unestablished: OpenAI Speech SSE text/audio alignment, Z.AI international TTS, Anthropic Tool Search discoverability of message-only by-value definitions, xAI conversation-item delete/truncate exact request fields. xAI item retrieval is documented unsupported. These are distinct from implemented-but-untested work; do not invent wire payloads or equate Unknown matrix cells with missing code.

Live acceptance is deferred. In round24, sending the environment XAI_API_KEY externally was rejected by automatic approval review because credential use/live calls lacked explicit authorization. No bypass occurred. The prepared xai_voice_acceptance example defaults to safe preview; its live mode would make at most two GETs and must not run without later explicit authorization. No API credential should be printed. No manual compaction tool exists; this handoff is not compaction.
