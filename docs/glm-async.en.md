# GLM asynchronous chat completion

`glm_async` provides one-shot asynchronous chat submission and result retrieval for Zhipu BigModel. The mainland official docs list `POST /api/paas/v4/async/chat/completions` and `GET /api/paas/v4/async-result/{id}`. Submission takes the typed `ChatRequest` and uses the OpenAI Chat codec. The caller chooses query intervals, persists references, and handles completed results.

```rust,ignore
use lingxi_llm_client::{
    providers::zhipu::async_tasks::{GlmAsyncConfig, GlmAsyncCredentials, GlmAsyncRegion},
    protocol::Secret,
};

let snapshot = client.snapshot();
let profile = snapshot.profile("glm-mainland").expect("configured GLM profile").clone();
let config = GlmAsyncConfig::new(
    profile,
    GlmAsyncRegion::ChinaMainland,
    "tenant-7/zhipu-account",
    "zhipu-key-slot-2",
)?;
let service = snapshot.glm_async(config)?;
let credentials = GlmAsyncCredentials::new(
    "zhipu-key-slot-2",
    Secret::new(api_key),
);

let accepted = service.submit(&chat_request, &credentials).await?;
// Persist accepted.reference. The host chooses when to query again.
let latest = service.get(&accepted.reference, &credentials).await?;
println!("status: {:?}; provider payload: {}", latest.task_status, latest.native);
```

`GlmAsyncConfig` explicitly binds the profile, region, account, and a non-secret credential-slot identity. Each operation receives the API key separately. Job references bind the provider, profile, region, endpoint, account, credential slot, and model; queries from another scope are rejected before sending. The credential-slot identity must match `GlmAsyncCredentials`. Serialized references never contain the API key.

Submission returns a task snapshot with its native JSON. `PROCESSING`, `SUCCESS`, and `FAIL` map to known statuses; future values remain in `Other`. A response `error` is also exposed as `native_error`. Each query sends one GET. The service does not poll, resubmit, or switch accounts. If submission is interrupted, returns HTTP 408/5xx, or succeeds without a task ID, the service returns `SubmitOutcomeUnknown` and preserves any available HTTP status, request ID, and native response. Check the task before deciding whether another submission is safe.

The async Chat docs do not declare a cancellation operation, so this service has no `cancel` method. Dropping a request future does not mean the provider task was cancelled.

Mainland API references: [Async chat completion](https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%AF%B9%E8%AF%9D%E8%A1%A5%E5%85%A8%E5%BC%82%E6%AD%A5), [Retrieve async result](https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E6%9F%A5%E8%AF%A2%E5%BC%82%E6%AD%A5%E7%BB%93%E6%9E%9C), and the [official documentation index](https://docs.bigmodel.cn/llms.txt).

An international profile can be explicitly selected with `GlmAsyncRegion::International`, but the current Z.AI docs list synchronous Chat Completion only and omit the asynchronous Chat route. Configuration therefore returns `UnsupportedRegion` before sending; the service does not guess an international endpoint. The region can be enabled after Z.AI publishes its async route. Mainland and international accounts and keys are not interchangeable.
