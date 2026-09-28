# Batch 批处理

[English](batches.en.md)

先用 `client.provider::<OpenAiClient>(profile)?` 绑定具体 profile，再通过 `provider.batches()` 调用资源。每次操作传入 `RequestOptions`，client 不保存凭证。

`provider.batches()` 是独立于 Chat 的 OpenAI Batch 生命周期入口。内置 `openai` profile 配置了独立 Batch 与结果文件路由；配置 v3 可继承、替换或禁用。接口依据 [OpenAI Batch 指南](https://developers.openai.com/api/docs/guides/batch) 和 [Batch API](https://developers.openai.com/api/reference/resources/batches/methods/create)。其他提供方使用各自的 Batch 服务；Background 和 deferred 任务使用独立资源 API。

用 `encode_jsonl()` 将最多 50,000 条请求编码成 JSONL，或用 `write_jsonl()` 写入调用方持有的文件并取得准确字节数。每条必须有唯一 `custom_id`；同一输入不能混用模型，也不能启用 `stream`。`write_jsonl()` 会在首次写入前校验全部请求；I/O 写入失败可能留下部分文件，调用方应丢弃。将 JSONL 以 `.jsonl` 文件名和 `application/x-ndjson` 类型通过同一 profile 的 `FileService::upload(..., FilePurpose::Batch)` 上传。这个便捷入口仍会在内存中构造 multipart。

大输入可使用 `FileService::upload_batch_stream(filename, media_type, size_bytes, chunks, timeout)`。传入文件/生成器的字节流和**准确**长度；客户端逐块发送 OpenAI multipart，校验流的实际长度，最多 200 MB，要求非空 `account_scope`。自定义 `Transport` 需实现 `send_stream()`；默认实现会在消耗输入前明确拒绝，不退回内存缓冲。上传和 Batch 提交是两步，不会自动重试；上传中断后的服务端结果可能未知，调用方应先核对文件列表再决定是否重传。输入及结果文件不随 Chat 附件的自动清理结束；OpenAI 文档规定 `purpose=batch` 文件默认在 30 天后过期，调用方仍应管理任务期间的引用和清理策略。

### 每条请求的持久文件输入

Batch 请求 JSONL 可以引用预先上传的文件：Responses `input_file.file_id`、Responses `input_image.file_id`，以及 Chat Completions `file` 内容部分中的 PDF 文件 ID。它们与最外层 JSONL 输入文件不同：应使用 `FilePurpose::ModelInput`（OpenAI `user_data`）上传，并保留对应 `ProviderFileRef`。OpenAI 的[文件输入指南](https://developers.openai.com/api/docs/guides/file-inputs)说明模型文件输入可用 `user_data`，且 Chat Completions 的上传文件 ID 仅支持 PDF；[图像指南](https://developers.openai.com/api/docs/guides/images-vision)也记录了 Responses 图像文件 ID，并在示例中使用 `vision` purpose。Responses 图像文件清单接受文档支持的两种 purpose。适配器拒绝 Responses `file_url` 和远程图像 URL，因为它无法确认任务运行时这些地址仍然可用。内嵌 data URL 的内容本身位于 JSONL 中，无需额外文件清单。

使用 `encode_jsonl_with_attachments()` 或 `write_jsonl_with_attachments()`，将 JSONL 中每个文件 ID 与显式的账户作用域引用绑定。普通编码入口会拒绝缺少清单的文件 ID；清单必须与 JSONL 中的文件 ID 完全一致。调用 `submit_with_attachments()` 时传入相同引用：服务会检查 provider、profile、端点和账户作用域，然后先逐个获取文件元数据，再创建 Batch。它会核对远端 ID 和 purpose，并拒绝在 Batch 24 小时完成窗口、文档说明的取消协调时间和 5 分钟时钟/请求缓冲之前到期的文件。这只是提交前检查，并不锁定文件：在 Batch 进入终态且所需结果已读取前，请勿删除这些文件。[Files API 文档](https://developers.openai.com/api/reference/typescript/resources/files/methods/create)说明未设置到期时间的文件会保留至人工删除。客户端不会自动上传、延长有效期、重试或删除这些附件文件。若创建 Batch 的传输中断，结果可能未知；应通过 `list()`/`get()` 核对，不能盲目重试。

```rust,no_run
use lingxi_llm_client::providers::openai::batches::{
    BatchAttachmentRef, BatchEndpoint, BatchError, BatchJob, BatchLine,
    encode_jsonl_with_attachments,
};
use lingxi_llm_client::files::ProviderFileRef;
use lingxi_llm_client::{LlmClient, RequestOptions};
use serde_json::json;

fn input_jsonl(pdf: &ProviderFileRef) -> Result<Vec<u8>, BatchError> {
    let attachment = BatchAttachmentRef::chat_completions_pdf(pdf.clone());
    encode_jsonl_with_attachments(
        BatchEndpoint::ChatCompletions,
        &[BatchLine {
            custom_id: "document-1".into(),
            body: json!({"model":"gpt-6-sol","messages":[{"role":"user","content":[
                {"type":"file","file":{"file_id":pdf.file_id.clone()}},
                {"type":"text","text":"总结这个 PDF。"}
            ]}]}),
        }],
        &[attachment],
    )
}

async fn submit_with_pdf(
    client: &LlmClient,
    jsonl_file: &ProviderFileRef,
    pdf: &ProviderFileRef,
    options: &RequestOptions,
) -> Result<BatchJob, Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    let attachments = [BatchAttachmentRef::chat_completions_pdf(pdf.clone())];
    provider.batches().submit_with_attachments(
        jsonl_file, BatchEndpoint::ChatCompletions, None, &attachments, options,
    ).await.map_err(Into::into)
}
```

Responses 文件和图像也可分别使用 `BatchAttachmentRef::responses_file()` 与 `BatchAttachmentRef::responses_image()`。API 无法检查已上传的 JSONL 文件内容，因此调用方必须提交生成该文件时使用的同一份清单。元数据核对后，调用方仍可通过其他客户端删除远端文件；本服务不声称提供原子保留或阻止外部删除的保证。

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::files::ProviderFileRef;
use lingxi_llm_client::providers::openai::batches::{BatchEndpoint, BatchError, BatchJob, BatchLine, encode_jsonl, write_jsonl};
use serde_json::json;

fn input_jsonl() -> Result<Vec<u8>, BatchError> {
    encode_jsonl(BatchEndpoint::Responses, &[
        BatchLine { custom_id: "item-1".into(), body: json!({"model":"gpt-6-sol","input":"你好"}) },
        BatchLine { custom_id: "item-2".into(), body: json!({"model":"gpt-6-sol","input":"再见"}) },
    ])
}

fn write_input_jsonl(output: &mut impl std::io::Write) -> Result<u64, BatchError> {
    write_jsonl(BatchEndpoint::Responses, &[
        BatchLine { custom_id: "item-1".into(), body: json!({"model":"gpt-6-sol","input":"你好"}) },
    ], output)
}

async fn submit_uploaded(client: &LlmClient, input: &ProviderFileRef, options: &RequestOptions) -> Result<BatchJob, Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    provider.batches().submit(input, BatchEndpoint::Responses, None, options).await.map_err(Into::into)
}

async fn consume_result(client: &LlmClient, job: &BatchJob, options: &RequestOptions) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>(&job.reference.profile_name)?;
    use futures::StreamExt;
    if let Some(reference) = job.output_ref() {
        let mut rows = provider.batches().stream_result(&reference, options).await?;
        while let Some(row) = rows.next().await {
            let row = row?;
            // 通过 row.custom_id 与原始请求关联，并自行持久化需要的结果。
            let _ = row;
        }
    }
    Ok(())
}
```

`submit()` 要求上传引用的 purpose 为 `batch`，并校验 provider、profile、文件端点和 `RequestOptions.account_scope`。JSONL 引用了持久 Files API 文件时，改用 `submit_with_attachments()`。两个入口都返回带相同作用域的 `BatchJobRef`；调用 `get()`、`list()` 和 `cancel()` 管理任务。状态 `cancelling` 不是终态，已完成请求可保留部分结果；`request_counts`、错误和用量原样保留。提交或取消遇到传输中断返回 `BatchError::OutcomeUnknown`，不得盲目重提；可先用 `list()`/`get()` 核对。

任务提供 `output_ref()` 与 `error_ref()`。`stream_result()` 从配置的 Files 路由逐行读取 JSONL，返回 `futures::stream::BoxStream`，保留 `custom_id`、原生 `response` 和 `error`；结果顺序不一定与输入一致。它不缓存整个结果文件，单行最多 16 MiB、最多 50,000 个唯一 `custom_id`；丢弃流会停止读取。`read_result()` 仍提供便捷的整文件读取，但限制为 64 MiB。两个入口都校验账户、Batch 路由及结果文件路由，且不会自动重试读取中断的流。该服务不自动轮询，也不删除输入或结果文件。测试仅使用 mock，尚未进行真实账户验收。
