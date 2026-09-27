//! Local preparation costs for long history, tools, and structured output.
//! Run with --iterations 128, optionally --parallel for 16/64 schema workers.
//! All transport is mocked. Compilation rows bypass the library schema cache;
//! cached-validation rows measure only validation, not compilation speedup.
//! Allocation bytes are cumulative allocation requests (not peak memory).

use async_trait::async_trait;
use lingxi_llm_client::{protocol::*, *};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;
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

fn schema(properties: usize) -> serde_json::Value {
    let fields: serde_json::Map<String, serde_json::Value> = (0..properties)
        .map(|i| (format!("field_{i}"), serde_json::json!({"type":"string", "description":"An output field with an explicit string type"})))
        .collect();
    let required: Vec<_> = fields.keys().cloned().collect();
    serde_json::json!({"type":"object", "properties":fields, "required":required, "additionalProperties":false})
}
fn base_request() -> ChatRequest {
    serde_json::from_value(serde_json::json!({"model":"bench-model", "messages":[{"role":"user", "content":[{"type":"text", "text":"hello"}]}]})).unwrap()
}
fn bench_request(label: &str, req: &ChatRequest, iterations: usize) {
    let client = builder(&profiles()).build().unwrap();
    let options = RequestOptions::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    runtime
        .block_on(client.chat().complete(req, &options))
        .unwrap();
    let mut preparations = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let (result, _, prepared) =
            measure(|| runtime.block_on(client.chat().complete(req, &options)));
        black_box(result.unwrap());
        preparations.push(prepared.unwrap());
    }
    report(label, 1, &mut preparations);
}
fn main() {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let n = arguments
        .windows(2)
        .find(|pair| pair[0] == "--iterations")
        .map(|pair| pair[1].parse::<usize>().expect("positive iteration count"))
        .unwrap_or(64);
    assert!(n > 0);
    println!(
        "operation,concurrency,samples,p50_ns,p95_ns,allocations_per_sample,allocated_bytes_per_sample"
    );
    let mut req = base_request();
    bench_request("simple_prepare", &req, n);
    for count in [16, 128] {
        req = base_request();
        req.messages = (0..count).map(|_| serde_json::from_value(serde_json::json!({"role":"user", "content":[{"type":"text", "text":"x".repeat(4096)}]})).unwrap()).collect();
        bench_request(&format!("history_{count}_x4096_prepare"), &req, n);
    }
    for count in [16, 64] {
        req = base_request();
        req.tools = (0..count)
            .map(|i| ToolSpec {
                name: format!("tool_{i}"),
                description: "Tool description ".repeat(32),
                input_schema: schema(20),
                strict: true,
                defer_loading: false,
                allowed_callers: vec![],
            })
            .collect();
        bench_request(&format!("tools_{count}_x20fields_prepare"), &req, n);
    }
    for count in [1, 50, 200] {
        req = base_request();
        let spec = schema(count);
        req.output_format = OutputFormat::JsonSchema {
            name: "result".into(),
            schema: spec.clone(),
            strict: true,
        };
        bench_request(&format!("schema_{count}_prepare"), &req, n);
        if count == 200 && arguments.iter().any(|arg| arg == "--parallel") {
            for workers in [16, 64] {
                bench_parallel("schema_200_prepare", &req, workers, n);
            }
        }
        bench(&format!("schema_{count}_compile"), n, || {
            jsonschema::options()
                .with_draft(jsonschema::Draft::Draft202012)
                .should_validate_formats(true)
                .should_ignore_unknown_formats(false)
                .build(&spec)
                .unwrap()
        });
        let validator = jsonschema::options()
            .with_draft(jsonschema::Draft::Draft202012)
            .should_validate_formats(true)
            .should_ignore_unknown_formats(false)
            .build(&spec)
            .unwrap();
        let value = serde_json::Value::Object(
            (0..count)
                .map(|i| (format!("field_{i}"), serde_json::Value::String("ok".into())))
                .collect(),
        );
        bench(&format!("schema_{count}_cached_validate"), n, || {
            validator.validate(&value).unwrap()
        });
    }
    bench_catalog(n);
}

fn bench_parallel(label: &str, request: &ChatRequest, workers: usize, iterations: usize) {
    let client = builder(&profiles()).build().unwrap();
    let barrier = std::sync::Barrier::new(workers);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let client = &client;
                let barrier = &barrier;
                scope.spawn(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_time()
                        .build()
                        .unwrap();
                    let options = RequestOptions::default();
                    runtime
                        .block_on(client.chat().complete(request, &options))
                        .unwrap();
                    let mut samples = Vec::with_capacity(iterations);
                    barrier.wait();
                    for _ in 0..iterations {
                        let (response, _, prepared) =
                            measure(|| runtime.block_on(client.chat().complete(request, &options)));
                        black_box(response.unwrap());
                        samples.push(prepared.unwrap());
                    }
                    samples
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut results = results;
    report(label, workers, &mut results);
}

fn bench_catalog(iterations: usize) {
    let mut profiles = profiles();
    let seed = profiles[0].models[0].clone();
    profiles[0].models = (0..1000)
        .map(|i| {
            let mut model = seed.clone();
            let name = format!("model-{i}");
            model.display_model = name.clone();
            model.request_model = name.clone();
            model.billing_model = name;
            model
        })
        .collect();
    let (client, manager) = builder(&profiles).build_managed().unwrap();
    bench("chat_models_1000", iterations, || client.chat().models());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    runtime
        .block_on(manager.configured_models("bench"))
        .unwrap();
    // Management work runs on another thread; allocation columns intentionally
    // count only this measuring thread, so compare its latency rather than bytes.
    bench("configured_models_1000", iterations, || {
        runtime
            .block_on(manager.configured_models("bench"))
            .unwrap()
    });
}
