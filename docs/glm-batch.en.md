# GLM Batch processing

[中文](glm-batch.md)

`LlmClient::glm_batch()` exposes a bounded lifecycle for Zhipu GLM's native Batch API: upload JSONL input, submit a 24-hour job, read state once, list one page of jobs, request cancellation, and stream successful or error result files. The only input endpoint exposed here is `/v4/chat/completions`, shown in Zhipu's official Java SDK Batch example. Each JSONL row uses `custom_id`, `method`, `url`, and `body`; the official example uses the `24h` completion window. See the [official Zhipu SDK Batch example](https://github.com/MetaGLM/zhipuai-sdk-java-v4#batch-processing), [Zhipu API introduction](https://docs.bigmodel.cn/cn/api/introduction), [List Batch Tasks API](https://docs.bigmodel.cn/api-reference/%E6%89%B9%E5%A4%84%E7%90%86-api/%E5%88%97%E5%87%BA%E6%89%B9%E5%A4%84%E7%90%86%E4%BB%BB%E5%8A%A1), and [official OpenAPI specification](https://docs.bigmodel.cn/openapi/openapi.json).

The service is pinned to the mainland BigModel root, `https://open.bigmodel.cn/api/paas/v4`. `GlmBatchRegion::International` is rejected when constructing a scope: the current [official Z.AI documentation index](https://docs.z.ai/llms.txt) does not publish a Batch lifecycle, so this module does not reuse mainland routing or credentials for the international service. The Zhipu SDK documents the API family and a Batch example; this module does not claim the international Z.AI host is a verified Batch endpoint.

```rust,no_run
use lingxi_llm_client::{
    glm_batch::{
        GlmBatchChatRequest, GlmBatchInput, GlmBatchLine, GlmBatchMessage,
        GlmBatchMetadata, GlmBatchRegion, GlmBatchScope,
    },
    protocol::Secret,
    LlmClient,
};
use serde_json::json;

async fn submit_glm_batch(
    client: &LlmClient,
    api_key: String,
) -> Result<String, Box<dyn std::error::Error>> {
    let scope = GlmBatchScope::new(
        "glm-mainland",
        "zhipu-account-42",
        GlmBatchRegion::ChinaMainland,
    )?;
    let service = client.glm_batch(Secret::new(api_key), scope)?;
    let body = GlmBatchChatRequest::new(
        "glm-5.3",
        vec![GlmBatchMessage {
            role: "user".into(),
            content: json!("Summarize this record."),
        }],
    )?;
    let input = GlmBatchInput::new(vec![GlmBatchLine::new("record-1", body)?])?;
    let input_file = service.upload_input(&input).await?;
    let job = service
        .submit(&input_file, &GlmBatchMetadata::default())
        .await?;
    Ok(job.reference.batch_id().to_owned())
}
```

Every row needs a unique `custom_id`, and one file can contain requests for only one model. Message `content` and additional model parameters retain their JSON representation; streaming is not accepted. Local input bounds are 50,000 rows, 1 MB per row, and 200 MB total.

Upload uses multipart `purpose=batch`. The resulting `GlmBatchInputFileRef` can only be submitted through a service with the same profile, account, region, and endpoint scope. Job and output/error file references carry that same scope fingerprint and cannot be reused across accounts or endpoints. `get` performs one status read; `list` fetches one page with the documented `after` cursor and `limit`; when `has_more` is true, pass `last_id` to a later explicit call. `cancel` sends one cancellation request; `stream_result` opens the selected JSONL file. The host controls polling, later pages, and result parsing.

Uploads, submissions, and cancellations are never retried automatically. Connection failures, provider 5xx responses, or malformed success responses are surfaced as `OutcomeUnknown` or `OutcomeUnknownResponse`. A submission may have been accepted, so check provider state before deciding whether to submit again. Explicit 4xx errors are returned as provider rejections. Result downloads yield raw byte chunks without buffering the complete file. Tests use a mock transport; they make no live or paid calls.
