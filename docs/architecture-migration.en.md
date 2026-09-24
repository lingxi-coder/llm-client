# Architecture and API update

[简体中文](architecture-migration.md)

This update keeps one standalone Rust crate and changes the public extension APIs. Old configuration and response formats are not supported. Recreate a version 2 configuration with the public configuration APIs; loading version 1 returns `UnsupportedVersion` without modifying the file.

## Services and execution

`LlmClient` coordinates the builder, request executor, attachment manager, account service and configuration coordinator. Routing, listings and prices read one immutable runtime snapshot. A successful configuration transaction installs a new snapshot and invalidates changed account bindings and attachment caches. The repository only locks, reads, validates the format, and atomically replaces the file.

Complete and streamed requests share route selection, attachment resolution, preparation, authentication, stale-file retry and failover. Continuations remain pinned to one connection, and fallback connections need their own credentials. Monotonic deadlines cover asynchronous authentication, upload/poll/read and cleanup. Complete requests keep the 120-second default (video retains its longer policy); streams have no total deadline unless configured. Wall-clock time is used only for timestamps and pricing.

## Extension API changes

| Interface | Contract |
| --- | --- |
| `Transport` | One async `send(HttpRequest) -> Result<StreamResponse, LlmError>`. Return raw bytes and all HTTP statuses; disable redirects and automatic retries. |
| `HttpExecutor` | Shared response collection, body limits and deadlines. Error bodies preserve status, headers and at most 64 KiB. |
| `WireCodec` | `encode_request(EncodeRequest, &CodecContext)`, `encoded_body_len`, `decode_response`, `stream_decoder`. The default size implementation encodes once; built-ins count with their serialization writer. |
| `EncodeRequest` | A borrowed request plus shared media and per-block bindings. Custom codecs call `block(original)` before encoding each content block, and `inline_media` for app bytes. |
| `CodecContext` | Connection data (including capability restrictions), wire model, `RequestMode` and file account scope; no credential values, route chain or file service. Use `CodecContext::for_model` to select a specific catalog row when several rows share a wire ID. |
| `StreamDecoder` | `push_bytes`, `finish`, `usage_report`. Accept arbitrary network fragments; return ordered `Vec<Result<StreamEvent, LlmError>>` so valid events preceding an error survive. Terminal/error states ignore later input. |
| `RequestOptions` | No `stream` field; the entry point or codec context chooses the mode. |
| `UsageReport` | `Option<Usage>` and `Missing / Partial / Complete / Invalid`. Full responses and streams use the same report. |
| `AccountUsageSource` | `fetch(&AccountFetchContext, &mut AccountReport)`. Commit each metric as it arrives; a timeout only fails unfinished fields. |

Profile-specific codec variants, `FrameStream`, `UrlOpener`, and the unused Transport WebSocket API are removed. SSE framing belongs to ordinary codec decoders; Bedrock decoders own EventStream framing. Hosted adapters modify the base request before its single serialization. Native reasoning envelopes, signatures, tool IDs, multimodal ordering and message/attachment reference structures remain unchanged.

Actual cost now reads the response's complete report:

```rust
use lingxi_llm_client::{LlmClient, ResolvedRoute};
use lingxi_llm_client::protocol::{CompletionResponse, LlmError, Submission};
fn report(client: &LlmClient, route: &ResolvedRoute, response: &CompletionResponse)
    -> Result<(), LlmError>
{
    let cost = client.estimate_actual_cost(route, response, Submission::Interactive)?;
    println!("{cost:?}");
    Ok(())
}
```

For a completed stream, call `estimate_stream_cost(&route, &stream, submission)` to include the observed service tier. Partial or invalid reports return `CostUnavailable`. A raw `Usage` can still be used for a preflight estimate through `estimate_cost`.

## Version 2 configuration

Static definitions, user settings, account observations and the runtime snapshot have separate roles. Describe caller defaults with the builder's profile slice. `add_builtin_profile(name)` / `add_builtin_profiles()` explicitly registers built-in definitions; `restore_builtin(name)` also uses a built-in definition reference. Definition references record `builtin` or `caller`; they are never guessed by comparing values.

Description, context and output limits merge as **user override > account observation > current definition**. Other model fields merge as **user override > current definition**. `add_provider` is a full replacement with explicit model fields. Use per-field overrides to keep other fields current. `Clear` sets the field's empty/default value; `Inherit` removes the override and keeps valid observations.

```rust
use lingxi_llm_client::{LlmClient, ProviderStoreError};
use lingxi_llm_client::configuration::{FieldOverride, ModelField};
fn edit(client: &mut LlmClient, profile: &str) -> Result<(), ProviderStoreError> {
    let rows = client.configured_models(profile)?;
    if let Some(row) = rows.first() {
        client.set_model_override(profile, &row.row_id, ModelField::Description,
            FieldOverride::Set(serde_json::json!("My model")))?;
        client.clear_model_override(profile, &row.row_id, ModelField::Description)?;
    }
    Ok(())
}
```

Rows retain order and independent IDs even when wire IDs repeat. Definitions with both identical wire IDs and display names are identified by occurrence; callers must preserve the relative order of these identically named rows when updating defaults. Wire-based visibility edits reject ambiguous rows; row-based edits are explicit. `replace_model` replaces one row without resurrecting its old catalog entry. Observed models later added to the catalog adopt defaults without duplicating the row.

The allowlist controls tracking, saved model metadata and presentation; it does not change explicit routing. Compatibility controls execution independently, so tracked metadata survives temporary incompatibility. The saved fallback definition is allowlist-filtered and refreshed after successful commits. It is used only if the whole definition is missing, including an empty builder; it cannot revive a model removed from an available definition.

Transactions lock and reread the latest state, validate the candidate, sync a temporary file and rename it atomically. Failed writes do not install a new runtime snapshot. Credentials are never persisted. Directory sync retains pagination limits, region-independent account management and `ProfileChanged` checks. Version 1 migration, legacy baselines and backups have been removed.

## Accounts and tokenizer features

`AccountQuery.execution` defaults to 60 seconds total and 30 seconds per HTTP/RPC. The budget starts when an execution slot is acquired; pages and subsequent calls share it. Batch concurrency defaults to 4, configured with `with_account_concurrency(NonZeroUsize)`. Scheduling is unordered; returned results retain profile order.

```toml
[dependencies]
lingxi-llm-client = { version = "0.1", features = ["tokenizer-openai", "tokenizer-qwen"] }
```

Features: `tokenizer-openai`, `tokenizer-deepseek`, `tokenizer-qwen`, `tokenizer-kimi`, `tokenizer-glm`, or `tokenizers-all`. `default = []`. Known models without their feature return `FeatureDisabled`; unknown models return `UnsupportedModel`. Assets and licenses remain in the source archive, while optional backends and assets are absent from default builds. See [verification and measurements](architecture-validation.md).
