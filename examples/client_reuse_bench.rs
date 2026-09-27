//! Local lifecycle benchmark: no network or provider credentials.
//!
//! Run `cargo run --release --example client_reuse_bench -- --iterations 128`.
//! Add `--mixed` to measure requests while durable configuration updates run.
//! For a pre-refactor checkout, copy this example there and compile it with
//! `cargo rustc --release --example client_reuse_bench -- --cfg client_reuse_baseline`.
//! The baseline measures Arc<LlmClient> reuse and omits unavailable snapshot APIs.
#![allow(unexpected_cfgs)]

use async_trait::async_trait;
use lingxi_llm_client::{protocol::*, *};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::Instant;

#[derive(Clone, Copy, Default)]
struct Allocations {
    count: u64,
    bytes: u64,
}

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<Allocations> = const { Cell::new(Allocations { count: 0, bytes: 0 }) };
    static STARTED: Cell<Option<Instant>> = const { Cell::new(None) };
    static PREPARED: Cell<Option<Sample>> = const { Cell::new(None) };
}

struct CountingAllocator;

fn allocated(size: usize) {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|counter| {
            let mut value = counter.get();
            value.count += 1;
            value.bytes += size as u64;
            counter.set(value);
        });
    }
}

// These methods only forward the same allocation contract to System. The
// counters are allocation-free thread locals, so they cannot recurse. Realloc
// counts one allocation with its requested size; bytes are cumulative requested
// bytes, not live memory or RSS. Library code itself remains unsafe-free.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() {
            allocated(layout.size());
        }
        result
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() {
            allocated(layout.size());
        }
        result
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            allocated(size);
        }
        result
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Default)]
struct Sample {
    nanos: u128,
    allocations: Allocations,
}

fn capture(started: Instant) -> Sample {
    Sample {
        nanos: started.elapsed().as_nanos(),
        allocations: ALLOCATIONS.with(Cell::get),
    }
}

fn measure<T>(operation: impl FnOnce() -> T) -> (T, Sample, Option<Sample>) {
    ALLOCATIONS.with(|value| value.set(Allocations::default()));
    PREPARED.with(|value| value.set(None));
    let started = Instant::now();
    STARTED.with(|value| value.set(Some(started)));
    COUNTING.with(|value| value.set(true));
    let result = operation();
    COUNTING.with(|value| value.set(false));
    let sample = capture(started);
    STARTED.with(|value| value.set(None));
    (result, sample, PREPARED.with(Cell::get))
}

fn report(label: &str, concurrency: usize, samples: &mut [Sample]) {
    samples.sort_unstable_by_key(|value| value.nanos);
    let length = samples.len();
    let percentile = |percent: usize| samples[(length * percent).div_ceil(100) - 1].nanos;
    let allocations: u64 = samples.iter().map(|value| value.allocations.count).sum();
    let bytes: u64 = samples.iter().map(|value| value.allocations.bytes).sum();
    println!(
        "{label},{concurrency},{length},{},{},{:.2},{:.2}",
        percentile(50),
        percentile(95),
        allocations as f64 / length as f64,
        bytes as f64 / length as f64,
    );
}

fn bench<T>(label: &str, iterations: usize, mut operation: impl FnMut() -> T) {
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let (result, sample, _) = measure(&mut operation);
        black_box(result);
        samples.push(sample);
    }
    report(label, 1, &mut samples);
}

// Amortize timer granularity for sub-microsecond handle operations. A sample
// includes destruction of each returned handle, keeping refcounts balanced.
fn bench_handles<T>(label: &str, iterations: usize, mut operation: impl FnMut() -> T) {
    const BATCH: u64 = 128;
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let (_, mut sample, _) = measure(|| {
            for _ in 0..BATCH {
                black_box(operation());
            }
        });
        sample.nanos /= u128::from(BATCH);
        sample.allocations.count /= BATCH;
        sample.allocations.bytes /= BATCH;
        samples.push(sample);
    }
    report(label, 1, &mut samples);
}

struct Mock;

#[async_trait]
impl Transport for Mock {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        STARTED.with(|started| {
            if let Some(started) = started.get() {
                PREPARED.with(|prepared| prepared.set(Some(capture(started))));
            }
        });
        tokio::task::yield_now().await;
        Ok(HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: bytes::Bytes::from_static(
                br#"{"id":"bench","object":"chat.completion","model":"bench-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#,
            ),
        }
        .into())
    }
}

fn profiles() -> Vec<ProviderProfile> {
    serde_json::from_value(serde_json::json!([{
        "provider_id": "openai",
        "profile_name": "bench",
        "base_url": "https://benchmark.invalid/v1",
        "protocol": "open_ai_chat",
        "auth": "none",
        "models": [{
            "display_model": "bench-model",
            "request_model": "bench-model",
            "billing_model": "bench-model"
        }]
    }]))
    .unwrap()
}

fn builder(profiles: &[ProviderProfile]) -> LlmClientBuilder {
    LlmClientBuilder::with_transport(Arc::new(Mock), profiles).with_region(Region::International)
}

#[cfg(client_reuse_baseline)]
type SharedClient = Arc<LlmClient>;
#[cfg(not(client_reuse_baseline))]
type SharedClient = LlmClient;

fn requests(client: &SharedClient, concurrency: usize, iterations: usize, label: &str) {
    let barrier = Barrier::new(concurrency);
    let results = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..concurrency)
            .map(|_| {
                let barrier = &barrier;
                let client = client.clone();
                scope.spawn(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_time()
                        .build()
                        .unwrap();
                    let request: ChatRequest = serde_json::from_value(serde_json::json!({
                        "model": "bench-model", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
                    })).unwrap();
                    let options = RequestOptions::default();
                    // Warm codec/runtime paths outside the samples.
                    runtime.block_on(client.chat().complete(&request, &options)).unwrap();
                    let mut totals = Vec::with_capacity(iterations);
                    let mut preparations = Vec::with_capacity(iterations);
                    barrier.wait();
                    for _ in 0..iterations {
                        let (response, total, prepared) = measure(|| {
                            runtime.block_on(client.chat().complete(&request, &options))
                        });
                        black_box(response.unwrap());
                        totals.push(total);
                        preparations.push(prepared.expect("mock transport was reached"));
                    }
                    (totals, preparations)
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    let (mut totals, mut preparations): (Vec<_>, Vec<_>) = results.into_iter().fold(
        (Vec::new(), Vec::new()),
        |(mut totals, mut preparations), (worker_total, worker_prepared)| {
            totals.extend(worker_total);
            preparations.extend(worker_prepared);
            (totals, preparations)
        },
    );
    report(&format!("{label}_prepare"), concurrency, &mut preparations);
    report(&format!("{label}_complete"), concurrency, &mut totals);
}

#[cfg(not(client_reuse_baseline))]
fn mixed_requests(profiles: &[ProviderProfile], iterations: usize) {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    let path = std::env::temp_dir().join(format!(
        "llm-client-reuse-bench-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (client, manager) = builder(profiles).build_managed().unwrap();
    runtime.block_on(manager.set_config_dir(&path)).unwrap();
    let stop = AtomicBool::new(false);
    let commits = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !stop.load(Ordering::Relaxed) {
                let models = if commits.load(Ordering::Relaxed).is_multiple_of(2) {
                    vec!["bench-model".into()]
                } else {
                    Vec::new()
                };
                runtime
                    .block_on(manager.set_tracked_models("openai", models))
                    .unwrap();
                commits.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        for concurrency in [1, 16, 64] {
            requests(&client, concurrency, iterations, "mixed");
        }
        stop.store(true, Ordering::Relaxed);
    });
    eprintln!(
        "mixed durable commits: {} (writer allocations excluded)",
        commits.load(Ordering::Relaxed)
    );
    std::fs::remove_dir_all(path).unwrap();
}

fn main() {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let iterations = arguments
        .windows(2)
        .find(|pair| pair[0] == "--iterations")
        .map(|pair| pair[1].parse::<usize>().expect("positive iteration count"))
        .unwrap_or(128);
    assert!(iterations > 0);
    println!("operation,concurrency,samples,p50_ns,p95_ns,allocations_per_sample,allocated_bytes_per_sample");
    eprintln!("release={}, baseline={}, iterations={iterations}; times include thread-local allocation counters; requested bytes include realloc and exclude other threads", !cfg!(debug_assertions), cfg!(client_reuse_baseline));

    #[cfg(client_reuse_baseline)]
    bench("builtin_cold_parse_owned", 1, || {
        builtin_providers().unwrap()
    });
    #[cfg(not(client_reuse_baseline))]
    {
        bench("builtin_cold_parse_borrowed", 1, || {
            builtin_catalog().unwrap()
        });
        bench_handles("builtin_warm_borrowed", iterations, || {
            black_box(builtin_catalog().unwrap())
        });
    }
    bench("builtin_warm_owned", iterations, || {
        builtin_providers().unwrap()
    });
    let profiles = profiles();
    bench("codec_context_for_model", iterations, || {
        lingxi_llm_client::codecs::CodecContext::for_model(
            &profiles[0],
            &profiles[0].models[0],
            lingxi_llm_client::codecs::RequestMode::Complete,
        )
    });
    bench("build_one_profile_mock_transport", iterations, || {
        builder(&profiles).build().unwrap()
    });
    #[cfg(client_reuse_baseline)]
    let client = Arc::new(builder(&profiles).build().unwrap());
    #[cfg(not(client_reuse_baseline))]
    let client = builder(&profiles).build().unwrap();
    #[cfg(client_reuse_baseline)]
    bench_handles("baseline_arc_clone_drop", iterations, || client.clone());
    #[cfg(not(client_reuse_baseline))]
    {
        bench_handles("client_clone_drop", iterations, || client.clone());
        bench_handles("snapshot_capture_drop", iterations, || client.snapshot());
        let snapshot = client.snapshot();
        bench("snapshot_resolve", iterations, || {
            snapshot.resolve("bench-model").unwrap()
        });
    }
    bench("live_resolve", iterations, || {
        client.resolve("bench-model").unwrap()
    });
    for concurrency in [1, 16, 64] {
        requests(&client, concurrency, iterations, "shared");
    }
    if arguments.iter().any(|argument| argument == "--mixed") {
        #[cfg(not(client_reuse_baseline))]
        mixed_requests(&profiles, iterations);
        #[cfg(client_reuse_baseline)]
        eprintln!(
            "mixed updates skipped: immutable pre-refactor Arc<LlmClient> cannot publish updates"
        );
    }
}
