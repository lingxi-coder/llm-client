# Qwen Batch inference

`qwen_batch` provides a typed interface to Alibaba Cloud Model Studio's Batch API. It uses the OpenAI-compatible routes documented for Beijing (`https://dashscope.aliyuncs.com/compatible-mode/v1`) and Singapore (`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`). API keys are region-specific.

The current interface covers text Chat Completions and Embeddings, with JSONL upload, task submission, query, one-page task listing, cancellation, and streamed download of success and error files. Input rows use typed request bodies; callers cannot supply arbitrary HTTP methods or URLs. Every input file uses one endpoint, one model, and one `enable_thinking` mode when applicable. Multimodal content, tool calls, and Qwen's test endpoint are outside the current request types.

```rust,no_run
// `client` and `api_key` come from the host application.
# async fn example(client: &lingxi_llm_client::LlmClient, api_key: String) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    protocol::Secret,
    qwen_batch::{
        QwenBatchChatMessage, QwenBatchChatRequest, QwenBatchChatRole, QwenBatchCompletionWindow,
        QwenBatchInput, QwenBatchLine, QwenBatchListOptions, QwenBatchMetadata, QwenBatchRegion,
        QwenBatchRequestBody, QwenBatchScope, QwenBatchSubmitOptions,
    },
};

let scope = QwenBatchScope::new(
    "qwen-production",
    "account-123",
    QwenBatchRegion::Beijing,
    Some("workspace-456".into()),
)?;
let service = client.qwen_batch(Secret::new(api_key), scope)?;
let input = QwenBatchInput::new(vec![QwenBatchLine {
    custom_id: "case-1".into(),
    body: QwenBatchRequestBody::Chat(QwenBatchChatRequest {
        model: "qwen-plus".into(),
        messages: vec![QwenBatchChatMessage {
            role: QwenBatchChatRole::User,
            content: "Introduce Hangzhou in one sentence.".into(),
        }],
        temperature: None,
        top_p: None,
        max_tokens: Some(128),
        enable_thinking: Some(false),
    }),
}])?;
let file = service.upload_input(&input).await?;
let options = QwenBatchSubmitOptions {
    completion_window: QwenBatchCompletionWindow::from_hours(24)?,
    metadata: QwenBatchMetadata {
        name: Some("nightly-eval".into()),
        description: None,
    },
};
let job = service.submit(&file, &options).await?;
let _page = service.list(&QwenBatchListOptions::new().limit(10)).await?;
let latest = service.query(&job.reference).await?;
if let Some(output) = latest.output_ref() {
    let mut bytes = service.stream_result(&output).await?;
    // Consume the JSONL byte stream as needed.
}
if let Some(errors) = latest.error_ref() {
    let mut bytes = service.stream_result(&errors).await?;
    // Consume failed-request details as JSONL.
}
# Ok(())
# }
```

Callers decide when to query for terminal state; the service does not poll, retry, or save files automatically. Batch input is limited to 50,000 rows and 500 MB, with a 1 MB row limit. `custom_id` values must be unique and no longer than 256 characters. Completion windows range from 24 through 336 hours. The official API only allows querying tasks created in the last 30 days. Query responses must identify the requested task; a mismatched task ID is rejected. A successful cancellation can first return `cancelling`; query again for the final state. If a successful cancel response identifies a different task, the service reports an unknown cancellation outcome and does not retry. Snapshots expose optional `file-batch_output-` and `file-batch_error-` IDs through `output_ref()` and `error_ref()`. Both references use the documented `/files/{file_id}/content` route through `stream_result()`.

`list` fetches one page and supports the documented `after`, `limit`, task-name, input-file, status, and creation-time filters. Limits range from 1 to 100, and creation-time filters use `yyyyMMddHHmmss`. When `has_more` is true, pass `last_id` as `after` in a later explicit call. The service rejects invalid filters, duplicate task IDs, and a page whose cursor does not advance before returning it.

File, job, and result references bind the provider, profile, HTTP endpoint fingerprint, account scope, region, and optional workspace ID. A reference from another account, connection, region, or workspace is rejected before network access. Workspace ID currently participates only in local reference scoping: the Batch-specific documentation does not define a workspace-specific URL or header, so the module does not invent one. Supply the API key for the declared region and workspace.

See Alibaba Cloud's official [Batch inference guide](https://help.aliyun.com/en/model-studio/batch-inference) and [OpenAI-compatible Batch API reference](https://help.aliyun.com/en/model-studio/batch-interfaces-compatible-with-openai) for input limits, lifecycle states, and result-file routes.
