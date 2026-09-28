# Qwen 知识库 Chunk 管理

[English](qwen-knowledge-chunks.en.md)

`QwenKnowledgeService` 增加了百炼知识库的手动 chunk 操作，使用 Model Studio 原生 RAG API，与 OpenAI Vector Stores 分开。服务支持新增、分页列出、更新和批量删除 chunk。路由及字段依据阿里云官方的 [RAG API 总览](https://help.aliyun.com/en/model-studio/rag-api-overview)、[Add Chunk](https://help.aliyun.com/en/model-studio/rag-api-add-chunk)、[List Chunks](https://help.aliyun.com/en/model-studio/rag-api-list-chunks)、[Update Chunk](https://help.aliyun.com/en/model-studio/rag-api-update-chunk) 和 [Delete Chunks](https://help.aliyun.com/en/model-studio/rag-api-delete-chunk) 文档。

该服务使用每次 `RequestOptions` 传入的 Bearer key、Beijing workspace endpoint 和知识库引用作用域。chunk 引用同时绑定知识库与可选源文档；更新要求源文档 ID，若 list response 不含 `metadata.doc_id`，该引用可以删除但不能更新。调用者必须为 document 和 multimedia 知识库的 list 请求提供文档引用；provider 文档指出这两类必须传 `docId`。

```rust,no_run
use lingxi_llm_client::providers::qwen::knowledge::{
    QwenKnowledgeAddChunkRequest, QwenKnowledgeChunkFields,
    QwenKnowledgeListChunksRequest, QwenKnowledgeUpdateChunkRequest,
};

# use lingxi_llm_client::providers::qwen::knowledge::{
#     QwenKnowledgeDocumentRef, QwenKnowledgeError, QwenKnowledgeRef, QwenKnowledgeService,
# };
# async fn manage_chunks(
#     request_options: &lingxi_llm_client::RequestOptions,
#     service: &QwenKnowledgeService<'_>,
#     knowledge: &QwenKnowledgeRef,
#     document: &QwenKnowledgeDocumentRef,
# ) -> Result<(), QwenKnowledgeError> {
let page = service
    .list_chunks(
        knowledge,
        &QwenKnowledgeListChunksRequest::default().for_document(document),
        request_options,
    )
    .await?;
let Some(chunk) = page.chunks.first() else {
    return Ok(());
};

service
    .update_chunk(
        &chunk.reference,
        &QwenKnowledgeUpdateChunkRequest::new(
            "Updated content with at least ten characters.",
            true,
        )
        .with_title("Updated title"),
        request_options,
    )
    .await?;

let fields = QwenKnowledgeChunkFields::document("A new indexed paragraph.")
    .with_title("New paragraph")?;
let add = QwenKnowledgeAddChunkRequest::for_document(document, fields);
service.add_chunk(knowledge, &add, request_options).await?;
service.delete_chunks(knowledge, &[chunk.reference.clone()], request_options).await?;
# Ok(())
# }
```

接口均使用 POST JSON。更新和删除接口的成功示例只包含 `code: Success`、`status_code: 200` 和 `request_id`，虽然总览定义的公共响应 envelope 还包含 `success`。客户端仅对这两个接口接受文档示例中的缺省 `success`；仍要求业务状态为 200，若提供 `success: false` 或非 2xx `status_code` 则按失败处理。新增和列举操作仍要求公共 envelope 中的 `success: true`。监控查询也按其单独文档样例接受缺省 `success`。

List Chunks 使用 `indexId`, `pageNum`, `pageSize` 与可选 `docId`，页码从 1 开始，每页 1–100 条，默认 20。新增 document chunk 的 `content` 最多 6000 个字符、`title` 最多 50 个字符、`image_urls` 最多 10 个；Add Chunk 文档注明每秒最多 10 次调用，调用方负责控制频率。更新 `content` 必须为 10–6000 个字符，标题可为 0–50 个字符；传空字符串可清空标题，省略标题会保留原值。删除每次接受 1–10 个 chunk，操作不可恢复。对表格或图片知识库，可使用动态字段 map；其中字符串值只按文档注明的 6000 字符上限预检。

新增 API 文档称其幂等，但 client 仍只发起一次请求，不自动重试。新增、更新或删除发生传输中断时会返回 `OutcomeUnknown`；宿主应检查远端状态后再决定是否再次提交。删除 chunk 会立即影响远端知识库。合同测试使用 mock transport，没有真实账户调用。
