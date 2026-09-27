# Shared clients, configuration updates, and performance

[简体中文](client-reuse.md)

Build one long-lived `LlmClient` at application startup and pass `clone()` handles to concurrent tasks. Cloning shares runtime resources and the configuration publication slot; it does not parse the catalog, create an HTTP connection pool, or copy model configuration. `model`, `thinking.effort`, `service_tier` (Fast), and credentials are request parameters, not client reuse keys. Build another client when region, transport/proxy/TLS, extension services, or configuration ownership must be independent.

## Initialization and dynamic configuration

`builtin_catalog()` returns a read-only built-in catalog parsed once per process. `builtin_providers()` returns an independent copy when editing is needed. `build()` returns a read-only request handle. For dynamic configuration, synchronous `build_managed()` returns a request handle and configuration manager:

```rust,no_run
use lingxi_llm_client::{builtin_catalog, LlmClientBuilder};
use lingxi_llm_client::protocol::Region;

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let (client, config) = LlmClientBuilder::new(builtin_catalog()?)?
    .with_region(Region::International)
    .build_managed()?;
config.set_config_dir("./config").await?;

let worker_client = client.clone();
// Move worker_client into a concurrent task without creating a connection pool.
assert_eq!(worker_client.region(), client.region());
config.set_tracked_models("openai", ["gpt-4.1-mini".to_owned()]).await?;
// Existing client handles automatically use the new configuration on later operations.
# Ok(())
# }
```

`ClientConfigManager` mutations, persistence, catalog sync, account-source rebinding, and management queries are async `&self` methods. The manager independently serializes configuration writes, with file locks and I/O running on the blocking thread pool. Requests hold no configuration lock while waiting on the network. The manager is not `Clone`; use `Arc<ClientConfigManager>` to share management access when needed. Dropping it leaves client handles usable with the last published configuration.

Publication installs connections, model indexes, account-source bindings, and cache generations together. New requests use the new state; in-flight operations retain the old state throughout failover, batch account queries, stream decoding, and file cleanup. Validation or write failures do not publish a new state. Cancellation can stop a commit before file writing starts. Once writing starts, the worker finishes installing and publishing the persisted result even if the caller stops awaiting it.

`prepare_provider_sync(...).await` → `operation.fetch().await` → `apply_provider_sync(...).await` supports independently scheduled catalog requests. Fetching holds no management lock. Committing rereads and merges the file; switching configuration directories invalidates previously prepared sync results.

## Pinning a sequence to one configuration

A retained live service handle captures the latest snapshot when each async operation first runs; synchronous queries capture at entry. To keep route resolution, estimation, execution, and pricing on one configuration, retain `client.snapshot()`:

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::{ChatRequest, PricingContext, Submission, Usage};

async fn run(client: &LlmClient, request: &ChatRequest, options: &RequestOptions)
    -> Result<(), Box<dyn std::error::Error>>
{
    let snapshot = client.snapshot();
    let route = snapshot.resolve_in(&request.model, Some("openai"))?;
    let assumed_usage = Usage { input_tokens: 1000, output_tokens: 500, ..Default::default() };
    let pricing = PricingContext {
        service_tier: request.service_tier,
        input_tokens: Some(1000),
        ..Default::default()
    };
    let estimate = snapshot.estimate_cost(&route, &assumed_usage, &pricing)?;
    let response = snapshot.chat().complete_in("openai", request, options).await?;
    let actual = snapshot.estimate_actual_cost(&route, &response, Submission::Interactive)?;
    println!("estimated: {estimate:?}; actual: {actual:?}");
    Ok(())
}
```

`ClientSnapshot` exposes all services plus routing, token, pricing, and account queries. Borrowing `profiles()` and `provider()` queries move to the snapshot; bind the snapshot to a local variable before retaining references. Live-client listing and pricing conveniences return owned results, but independently capture configuration on each call and cannot recover historical prices. Old snapshots retain old account-source bindings for their operations. Keeping a snapshot for a long time extends the lifetime of that configuration.

## Cache boundaries

Client clones share HTTP transport, codecs, authenticators, and attachment-upload caches. Credentials remain request-scoped and are not persisted. Upload caching and concurrent-upload deduplication use the same full key: endpoint, stable file-account scope, attachment version, configuration namespace, and profile generation. Without a stable `file_account_scope`, uploads are not reused across requests. Sharing a client does not share one account's uploads with another account.

Changing configuration directories or relevant connection settings gives subsequent requests a new cache generation. Completion of an old upload or a stale-file 404 affects only the old generation. Qwen rate limits remain shared by endpoint and stable account identity; automatically cleaned-up Qwen uploads are not reused. The host's attachment resolver still owns raw-content loading and access checks. Model responses and account balances are not cached, and retry policy is unchanged.

`AttachmentResolver::validate_content_reuse` defaults to `false`, preserving a `resolve` call per request for existing implementations. A host may check the current caller's access, continued availability, and that the complete reference identifies the same immutable content, then return `true` to explicitly allow byte reuse. This check runs before cache hits, misses, and waits for concurrent reads; errors never fall back to cached bytes. Clones of one client share a content cache keyed by the full attachment reference, limited to 256 entries and 64 MiB total. On insertion, the visible bytes are copied into compact storage shared by the current request and subsequent hits, so a small slice cannot keep an oversized backing allocation alive. Source bytes may be reused across configuration snapshots; remote file references remain isolated by account and configuration generation.

Each request validates all attachment metadata, total size, and duplicate-reference consistency before running up to four independent reads or uploads concurrently. Duplicate attachments remain deduplicated, with message ordering, total deadlines, Qwen pacing, and cancellation cleanup preserved.

Compiled structured-output schemas are reused by content, with at most 16 entries and 4 MiB of serialized schema source retained; this is not a bound on the actual memory occupied by compiled validators. Concurrent initialization of the same cached entry is shared, and failures use the same bounded cache. Model capabilities, strict subsets, and native profile-field conflicts are checked against the current request. Cache matching distinguishes numeric representations with different wire lengths, including `0.0` and `-0.0`. Request schemas retain their 1 MiB limit; larger schemas used only for local response validation bypass the cache.

## Migration and measurement

- Replace per-task `build()` calls with startup construction and `client.clone()` in tasks.
- Replace `let mut client = ...build()?` and client configuration APIs with `let (client, config) = ...build_managed()?`; invoke management methods as `config.method(...).await`.
- Move `client.profiles()` / `client.provider(name)` borrowing queries to a local snapshot. `deleted_builtin_profiles()`, `tracked_models()`, and `configured_models()` return owned data asynchronously through config.
- Use one snapshot for multi-step operations requiring consistent configuration. Operations through the live client capture the latest state when they start.

Run the release benchmark without contacting a real provider:

```sh
cargo run --release --example client_reuse_bench -- --iterations 1024
```

It separates cold/hot built-in catalog access, editable catalog copying, construction, cloning, snapshot capture, and route resolution. Mock requests with 1/16/64 concurrent workers report local preparation and overall latency percentiles, allocation counts, and allocated bytes. Add `--mixed` to measure concurrent persisted configuration updates separately. CSV output includes `p50_ns`, `p95_ns`, `allocations_per_sample`, and `allocated_bytes_per_sample`. Allocated bytes are cumulative allocation requests (reallocations count their new size), not RSS. Thread-local request measurements exclude the configuration writer. Preparation ends at entry to the mock transport; overall latency also includes mock response decoding. Compare on the same machine with identical build options and working-tree contents. The old shared path uses `Arc<LlmClient>` as its baseline, so an already-shareable use case is not misrepresented as a new optimization. Results measure local preparation costs, not real-provider generation speed, and are not machine-dependent CI thresholds.


### Initial measurements on 2026-09-25

Environment: macOS 15.7.8, arm64, Rust 1.94.0, release with default features, 1,024 iterations per worker. After stopping concurrent builds and tests, the compiled baseline and optimized binaries ran serially in alternating order for three rounds. Tables show the median of each round's p50/p95; full results with minimum/maximum values are in the [summary CSV](benchmarks/client-reuse-2026-09-25/summary.csv). The baseline is a copy of the dirty working tree before this refactor. Account/cache changes from this refactor that raced with the copy were restored to their original interfaces; other pre-existing edits were retained.

| Operation | Baseline p50 | Optimized p50 | Allocations: baseline → optimized |
| --- | ---: | ---: | ---: |
| Warm editable built-in catalog | 15.453 ms | 0.418 ms | 113,067 → 10,510 |
| Single-profile client construction (mock transport) | 3.333 µs | 3.292 µs | 84 → 82 |
| Shared handle clone + drop | 4 ns (`Arc<LlmClient>`) | 5 ns (`LlmClient`) | 0 → 0 |
| Snapshot capture + drop | — | 22 ns | 0 |

Warm editable catalog access is about 37 times faster, with cumulative allocated bytes falling from 34,237,122 to 1,102,702. The first access still parses the catalog. Warm read-only `builtin_catalog()` access allocates nothing. Handle and read-only catalog samples batch 128 operations and divide elapsed time per operation to reduce timer-resolution artifacts; nanosecond results indicate scale only.

Preparation for requests through an already-shared client has a small additional hot-path cost:

| Concurrent workers | Baseline p50 / p95 (µs) | Optimized p50 / p95 (µs) |
| --- | ---: | ---: |
| 1 | 9.667 / 11.584 | 10.791 / 12.833 |
| 16 | 10.291 / 21.375 | 11.333 / 21.625 |
| 64 | 10.500 / 20.792 | 11.917 / 22.417 |

Preparation p50 increased by about 1.0–1.4 µs. It still makes 95 allocations per request; cumulative allocated bytes changed from 34,130 to 34,354. Across the three rounds, 16-worker p95 ranged from 14.500–21.416 µs for the baseline and 15.583–22.833 µs after optimization; scheduling affects tail latency. The gains are avoiding repeated catalog parsing and initialization, plus allowing configuration updates during in-flight requests. These measurements do not establish a speedup for already-shared requests.

A separate [mixed run](benchmarks/client-reuse-2026-09-25/optimized-mixed.csv) with the CPU otherwise idle completed 19 persisted configuration commits. Preparation p95 was 12.292/21.833/22.500 µs at 1/16/64 workers. This demonstrates concurrent progress under that workload; deterministic concurrency-contract tests establish correctness. Raw records: baseline [round 1](benchmarks/client-reuse-2026-09-25/baseline-round1.csv), [round 2](benchmarks/client-reuse-2026-09-25/baseline-round2.csv), [round 3](benchmarks/client-reuse-2026-09-25/baseline-round3.csv); optimized [round 1](benchmarks/client-reuse-2026-09-25/optimized-round1.csv), [round 2](benchmarks/client-reuse-2026-09-25/optimized-round2.csv), [round 3](benchmarks/client-reuse-2026-09-25/optimized-round3.csv).


### Request workload follow-up

The next pass optimizes schema compilation reuse, borrowed text/tool-schema serialization, bounded attachment preparation, token counting, and catalog materialization. See the [workload measurements and raw CSV](benchmarks/request-workloads-2026-09-25/README.md) for the three-round comparison, including 16/64 workers and the limits of the measured gains. Run `cargo run --release --example performance_workloads -- --iterations 128 --parallel` to measure these workloads locally.
