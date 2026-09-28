# Gemini Batch API

`gemini_batch` 提供 Google Gemini Developer API `generateContent` Batch 操作，支持 inline 请求和 JSONL 文件：创建、更新、读取、单页列表、请求取消、删除，以及 inline 或文件形式的结果。另有独立的异步 `EmbedContent` Batch typed 生命周期、inline 单项结果和逐行解析的 typed JSONL 结果流。文件输入使用官方文档中的 Gemini Files API 可恢复上传流程。服务只保存 transport 和非密钥作用域；宿主在每次操作时传入 `&Secret<String>`，并自行负责凭据刷新和轮换。服务不会自动轮询、翻页或重试。generateContent JSONL 输出仍以原始字节提供；embedding JSONL 输出会逐行解码。

路由和 JSON 结构依据 Google 的 [Batch API guide](https://ai.google.dev/gemini-api/docs/batch-api)、[Batch API REST reference](https://ai.google.dev/api/batch-api) 与 [Files API reference](https://ai.google.dev/api/files)：创建使用 `POST /v1beta/models/{model}:batchGenerateContent`；更新使用 `PATCH /v1beta/batches/{batchId}:updateGenerateContentBatch`；状态读取使用 `GET /v1beta/batches/{batchId}`；列表使用 `GET /v1beta/batches?pageSize=...&pageToken=...`；取消使用 `POST /v1beta/batches/{batchId}:cancel`；删除使用 `DELETE /v1beta/batches/{batchId}`。独立构造的服务通过 `x-goog-api-key` 发送 API key；`GoogleClient::batch(scope)` 的绑定服务使用 profile 注册的鉴权器。

```rust,no_run
use lingxi_llm_client::{
    providers::google::batch::{
        GeminiBatchCreateRequest, GeminiBatchError, GeminiBatchGenerateContentRequest,
        GeminiBatchInput, GeminiBatchListOptions, GeminiBatchRequest, GeminiBatchScope,
        GeminiBatchService,
    },
    protocol::Secret,
    transport::Transport,
};
use serde_json::json;

async fn run_batch(
    transport: &dyn Transport,
    api_key: Secret<String>,
) -> Result<(), GeminiBatchError> {
    let scope = GeminiBatchScope::new(
        "google-production",
        "account-123",
        "https://generativelanguage.googleapis.com/v1beta",
    )?;
    let service = GeminiBatchService::new(transport, scope)?;

    let request = GeminiBatchGenerateContentRequest::new(vec![json!({
        "role": "user",
        "parts": [{"text": "Summarize this text."}]
    })])?;
    let row = GeminiBatchRequest::new(request).with_key("summary-1")?;
    let input = GeminiBatchInput::new(vec![row])?;
    let create = GeminiBatchCreateRequest::new(
        "gemini-3.8-flash",
        "summaries-september",
        input,
    )?;

    // Creation enqueues work and is non-idempotent. Persist this scoped ref.
    let created = service.create(&create, &api_key).await?;

    // The caller controls when to check status; each call makes one request.
    let current = service.get(&created.reference, &api_key).await?;
    let _page = service.list(&GeminiBatchListOptions::new(), &api_key).await?;
    let results = service.results(&current.reference, &api_key).await?;
    if let Some(items) = results.items {
        for item in items {
            if let Some(response) = item.response {
                let _native_generate_content_response = response;
            } else if let Some(error) = item.error {
                let _per_item_error = error;
            }
        }
    }

    // Cancellation is best effort. Call get later to confirm the final state.
    // service.cancel(&created.reference, &api_key).await?;
    Ok(())
}
```

inline 输入对象采用 Gemini REST schema 的 `batch.input_config.requests.requests[]` 形状。`GeminiBatchGenerateContentRequest::from_value` 会验证请求是对象且带有非空 `contents` 数组，同时保留其他 GenerateContent 字段。序列化后的完整创建 body 必须小于 20,000,000 字节。

文件输入的 JSONL 每行包含调用方提供且不重复的 `key`，以及一个原生 REST `request` 对象。`GeminiBatchJsonlInput` 会校验数据并编码为每行一个 JSON 对象；`upload_input_jsonl` 会逐行流式编码，并通过 Files API 可恢复上传流程上传文件，最后返回绑定了当前作用域的 `GeminiBatchFileRef`。已有本地大文件可以通过 `upload_input_stream` 上传：传入准确的字节数和只消费一次的流即可避免在服务中缓冲整份文件。两种上传方式都遵守官方 2 GB 文件上限，且不会重试结果不确定的上传。将返回的文件引用传给 `GeminiBatchInput::from_file`，创建请求会发送 `batch.input_config.file_name`。

```rust,no_run
use futures::StreamExt;
use lingxi_llm_client::providers::google::batch::{
    GeminiBatchCreateRequest, GeminiBatchError, GeminiBatchFileRef,
    GeminiBatchGenerateContentRequest, GeminiBatchInput, GeminiBatchJsonlInput,
    GeminiBatchJsonlRequest, GeminiBatchScope, GeminiBatchService, GeminiBatchState,
};
use lingxi_llm_client::{protocol::Secret, transport::Transport};
use serde_json::json;

async fn run_file_batch(
    transport: &dyn Transport,
    api_key: Secret<String>,
) -> Result<(), GeminiBatchError> {
    let scope = GeminiBatchScope::new(
        "google-production",
        "account-123",
        "https://generativelanguage.googleapis.com/v1beta",
    )?;
    let service = GeminiBatchService::new(transport, scope)?;
    let request = GeminiBatchGenerateContentRequest::new(vec![json!({
        "role": "user",
        "parts": [{"text": "Summarize this text."}]
    })])?;
    let input = GeminiBatchJsonlInput::new(vec![
        GeminiBatchJsonlRequest::new("summary-1", request)?,
    ])?;
    let uploaded = service
        .upload_input_jsonl("summaries.jsonl", input, &api_key)
        .await?;
    let create = GeminiBatchCreateRequest::new(
        "gemini-3.8-flash",
        "summaries-september",
        GeminiBatchInput::from_file(uploaded),
    )?;
    let created = service.create(&create, &api_key).await?;

    // 显式读取一次状态；生产调用方可以持久化并复用该引用。
    let current = service.get(&created.reference, &api_key).await?;
    if current.state == Some(GeminiBatchState::Succeeded) {
        if let Some(output_file) = current.output_file {
            let mut output = service.download_results(&output_file, &api_key).await?;
            while let Some(chunk) = output.next().await {
                let _jsonl_bytes = chunk?;
            }
        }
    }

    // 也可以将其他上传流程返回的资源名绑定到此作用域。
    let _same_scope_file = GeminiBatchFileRef::from_resource_name(
        service.scope(),
        "files/preuploaded-123",
    )?;
    Ok(())
}
```

Google 将 inline 输出放在 operation response 的 `output.inlinedResponses.inlinedResponses` 中；每一项包含 `metadata` 和 `response` 或 `error`。结果仍是 provider 原生 JSON，逐项错误不会被提升成整批请求失败。若 operation 尚未包含 inline 输出，`GeminiBatchResults::items` 为 `None`；如果含有结果，则 `Some` 包含每一个成功或失败条目。

文件输入任务的 operation response 也可能返回 `response.output.responsesFile`。解码后的文件资源会放在 `GeminiBatchSnapshot::output_file`；`download_results` 会以流的形式返回 JSONL 字节。输出文件每行是 provider 原生响应或状态对象。调用方自行决定何时读取状态和下载结果；本模块不会自动轮询或缓冲结果文件。

`delete` 需要同一作用域内的 Batch 引用，并验证 Google 文档规定的空 JSON 对象响应。删除操作记录只表示客户端不再关心结果，不会停止正在执行的任务；需要停止任务时请单独调用 `cancel`。

批次引用会绑定 `google` provider ID、profile、API endpoint 指纹和调用方提供的非密钥 `account_scope`。跨任一作用域使用引用都会在发请求前拒绝。持久化时应连同引用一起保存这些身份字段。

Batch 创建、更新和文件上传都不会自动重试。上传、创建、更新、取消或删除期间遇到传输错误时，服务会返回 `OutcomeUnknown`；收到成功 HTTP 状态但结果无法解码时会返回 `OutcomeUnknownResponse`。删除操作记录不会取消任务。取消接口只承诺尽力停止后续处理，成功返回表示取消请求被接受；调用方应稍后单独读取状态确认结果。

`update_generate_content_batch` 使用官方 `PATCH /v1beta/batches/{batchId}:updateGenerateContentBatch`。`GeminiBatchUpdateRequest` 会编码必需的 `model`、`displayName`、`inputConfig`；可选 `priority` 按十进制字符串发送 int64，Google 文档允许负值。可选 `updateMask` 只接受资源字段名 `model`、`displayName`、`inputConfig` 和 `priority`。文件输入的资源字段使用 `inputConfig.fileName`。inline 更新资源使用与 inline 创建相同的客户端 20,000,000 字节上限；更大的批次请使用文件输入。Google 直接返回 `GenerateContentBatch` 资源，因此客户端会校验响应名称与带作用域的引用一致，并解析资源必需字段。传输错误或无法解码的 2xx 响应会携带引用返回 `OutcomeUnknown` 或 `OutcomeUnknownResponse`，且不会重试。REST 文档没有规定额外的 priority 范围或批次状态前置条件，客户端也不自行添加。

```rust,no_run
use lingxi_llm_client::providers::google::batch::{
    GeminiBatchError, GeminiBatchGenerateContentRequest, GeminiBatchInput,
    GeminiBatchRequest, GeminiBatchSnapshot, GeminiBatchUpdateField,
    GeminiBatchUpdateRequest, GeminiBatchService,
};
use lingxi_llm_client::protocol::Secret;
use serde_json::json;

async fn update_example(
    service: &GeminiBatchService<'_>,
    created: &GeminiBatchSnapshot,
    api_key: &Secret<String>,
) -> Result<(), GeminiBatchError> {
    let replacement_input = GeminiBatchInput::new(vec![GeminiBatchRequest::new(
        GeminiBatchGenerateContentRequest::new(vec![json!({
            "role": "user",
            "parts": [{"text": "updated prompt"}]
        })])?,
    )
    .with_key("updated-row")?])?;
    let update = GeminiBatchUpdateRequest::new(
        "gemini-3.8-flash",
        "renamed-batch",
        replacement_input,
    )?
    .with_priority(-5)
    .with_update_mask([
        GeminiBatchUpdateField::DisplayName,
        GeminiBatchUpdateField::Priority,
    ])?;
    let _updated = service
        .update_generate_content_batch(&created.reference, &update, api_key)
        .await?;
    Ok(())
}
```

## Embedding Batch

同一服务也提供独立的异步 `EmbedContent` Batch API typed 接口，创建路由为 `POST /v1beta/models/{model}:asyncBatchEmbedContent`。模型会写入每条原生 `EmbedContentRequest`，当前配置放在 `embedContentConfig`；不会发送已弃用的顶层 `taskType`、`title` 或 `outputDimensionality`。Google 文档规定 `gemini-embedding-001` 和 `gemini-embedding-2` 的输出维度为 128–3072；`gemini-embedding-2` 不接受 `taskType`，且 `title` 只适用于 `RETRIEVAL_DOCUMENT`。

```rust,no_run
use lingxi_llm_client::{
    providers::google::batch::{
        GeminiBatchEmbeddingConfig, GeminiBatchEmbedContentItem,
        GeminiBatchEmbedContentRequest, GeminiBatchListOptions,
        GeminiEmbeddingBatchCreateRequest, GeminiEmbeddingBatchInput,
        GeminiBatchError, GeminiBatchScope, GeminiBatchService,
    },
    protocol::Secret,
    transport::Transport,
};

async fn embed_corpus(
    transport: &dyn Transport,
    api_key: Secret<String>,
) -> Result<(), GeminiBatchError> {
    let scope = GeminiBatchScope::new(
        "google-production",
        "account-123",
        "https://generativelanguage.googleapis.com/v1beta",
    )?;
    let service = GeminiBatchService::new(transport, scope)?;
    let config = GeminiBatchEmbeddingConfig::new()
        .with_output_dimensionality(768)?;
    let request = GeminiBatchEmbedContentRequest::text("批量生成 embedding。")?
        .with_config(config);
    let input = GeminiEmbeddingBatchInput::new(vec![
        GeminiBatchEmbedContentItem::new(request).with_key("row-1")?,
    ])?;
    let create = GeminiEmbeddingBatchCreateRequest::new(
        "gemini-embedding-2",
        "corpus-september",
        input,
    )?;

    let created = service.create_embedding_batch(&create, &api_key).await?;
    let current = service
        .get_embedding_batch(&created.reference, &api_key)
        .await?;
    let _page = service
        .list_embedding_batches(&GeminiBatchListOptions::new(), &api_key)
        .await?;
    let results = service
        .embedding_batch_results(&current.reference, &api_key)
        .await?;
    if let Some(items) = results.items {
        for item in items {
            if let Some(response) = item.response {
                // Google 的 first-party 响应可能没有 values，只返回 tensor shape。
                let _shape = response.embedding.shape;
            } else if let Some(error) = item.error {
                let _per_request_error = error;
            }
        }
    }
    Ok(())
}
```

文件输入可以用 `GeminiEmbeddingBatchJsonlInput` 构造带 key 的原生 embedding 请求行，再调用 `upload_embedding_input_jsonl`。编码器会在开始上传前校验 key 唯一且非空，并遵守官方 2 GB 文件上限。返回的 `GeminiEmbeddingBatchFileRef` 会保留模型、输入顺序中的 request key 和输出维度；将其传给 `GeminiEmbeddingBatchInput::from_file` 即可创建任务。如果宿主已有 JSONL 流，也可以使用 `upload_embedding_input_stream`。

```rust,no_run
use lingxi_llm_client::providers::google::batch::{
    GeminiBatchEmbeddingConfig, GeminiBatchEmbedContentRequest,
    GeminiEmbeddingBatchCreateRequest, GeminiEmbeddingBatchInput,
    GeminiEmbeddingBatchJsonlInput, GeminiEmbeddingBatchJsonlRequest,
    GeminiBatchError, GeminiBatchScope, GeminiBatchService,
};
use lingxi_llm_client::{protocol::Secret, transport::Transport};

async fn embed_from_file(
    transport: &dyn Transport,
    api_key: Secret<String>,
) -> Result<(), GeminiBatchError> {
    let scope = GeminiBatchScope::new(
        "google-production",
        "account-123",
        "https://generativelanguage.googleapis.com/v1beta",
    )?;
    let service = GeminiBatchService::new(transport, scope)?;
    let input = GeminiEmbeddingBatchJsonlInput::new(
        "gemini-embedding-2",
        vec![GeminiEmbeddingBatchJsonlRequest::new(
            "row-1",
            GeminiBatchEmbedContentRequest::text("一条文档记录。")?
                .with_config(GeminiBatchEmbeddingConfig::new()
                    .with_output_dimensionality(768)?),
        )?],
    )?;
    let file = service
        .upload_embedding_input_jsonl("embeddings.jsonl", input, &api_key)
        .await?;
    let create = GeminiEmbeddingBatchCreateRequest::new(
        "gemini-embedding-2",
        "file-backed-embeddings",
        GeminiEmbeddingBatchInput::from_file(file),
    )?;
    let created = service.create_embedding_batch(&create, &api_key).await?;
    let current = service.get_embedding_batch(&created.reference, &api_key).await?;
    if let Some(file) = current.output_file {
        let mut rows = service.download_embedding_results(&file, &api_key).await?;
        use futures::StreamExt;
        while let Some(row) = rows.next().await {
            let _typed_embedding_or_item_error = row?;
        }
    }
    Ok(())
}
```

Embedding Batch 引用会序列化明确的操作类型，不能传给 generateContent 生命周期方法。创建返回的引用保留模型和已知维度；若 Google 的列表响应没有这些信息，列表引用会明确保持 unknown。Google 文档说明 inline 和文件输出都与输入请求顺序一致（[REST 输出 schema](https://ai.google.dev/api/embeddings#EmbedContentBatchOutput)）。文件输出按 JSONL 行逐条流式解码；对于 typed JSONL 输入，会按行校验已知维度和 key，返回的 key 必须与输入顺序一致；无 key 的行会继承对应输入 key。流会拒绝缺行、多行和维度不匹配。若调用方通过原始流上传文件且未提供请求元数据，可为文件引用设置统一预期维度以校验每个向量。

`GeminiEmbeddingBatchResultStream` 每次解析一条 JSONL 结果；单行最大 64 MiB，整份文件不受该限制。状态读取和下载时机由调用方控制，服务不会自动轮询或翻页。取消仍是 best effort；删除操作只移除客户端对 operation 的关注，不会停止任务。
