# Qwen knowledge-base chunk management

[简体中文](qwen-knowledge-chunks.md)

`QwenKnowledgeService` now supports Model Studio's native RAG chunk operations: add, paginated list, update, and batch delete. These routes are separate from OpenAI Vector Stores. The implementation follows Alibaba Cloud's official [RAG API overview](https://help.aliyun.com/en/model-studio/rag-api-overview), [Add Chunk](https://help.aliyun.com/en/model-studio/rag-api-add-chunk), [List Chunks](https://help.aliyun.com/en/model-studio/rag-api-list-chunks), [Update Chunk](https://help.aliyun.com/en/model-studio/rag-api-update-chunk), and [Delete Chunks](https://help.aliyun.com/en/model-studio/rag-api-delete-chunk) documentation.

The service reuses `QwenKnowledgeService` credentials, the Beijing workspace endpoint, and knowledge-base scope checks. A chunk reference binds to its knowledge base and, when available, its source document. Updates require the source document ID; a chunk whose list metadata has no `doc_id` can be deleted but cannot be updated. The provider requires `docId` when listing chunks for document and multimedia knowledge bases, so callers must pass a scoped document reference for those types.

```rust,no_run
use lingxi_llm_client::qwen_knowledge::{
    QwenKnowledgeAddChunkRequest, QwenKnowledgeChunkFields,
    QwenKnowledgeListChunksRequest, QwenKnowledgeUpdateChunkRequest,
};

# use lingxi_llm_client::qwen_knowledge::{
#     QwenKnowledgeDocumentRef, QwenKnowledgeError, QwenKnowledgeRef, QwenKnowledgeService,
# };
# async fn manage_chunks(
#     service: &QwenKnowledgeService<'_>,
#     knowledge: &QwenKnowledgeRef,
#     document: &QwenKnowledgeDocumentRef,
# ) -> Result<(), QwenKnowledgeError> {
let page = service
    .list_chunks(
        knowledge,
        &QwenKnowledgeListChunksRequest::default().for_document(document),
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
    )
    .await?;

let fields = QwenKnowledgeChunkFields::document("A new indexed paragraph.")
    .with_title("New paragraph")?;
let add = QwenKnowledgeAddChunkRequest::for_document(document, fields);
service.add_chunk(knowledge, &add).await?;
service.delete_chunks(knowledge, &[chunk.reference.clone()]).await?;
# Ok(())
# }
```

All routes use POST JSON. The Update and Delete endpoint success examples contain only `code: Success`, `status_code: 200`, and `request_id`, although the overview's common envelope also includes `success`. The client accepts a missing `success` field only for those two operations and only when the business status is 200; an explicit `success: false` or non-2xx `status_code` remains a failure. Add and List still require `success: true` in the common envelope. The monitoring query also accepts the omission shown in its separate endpoint sample.

List Chunks accepts `indexId`, `pageNum`, `pageSize`, and optional `docId`; page numbers start at 1, page size is 1–100, and the default is 20. For document chunks, `content` may be up to 6,000 characters, `title` up to 50, and `image_urls` up to 10 entries. The Add Chunk page specifies a maximum rate of 10 calls per second; the caller controls request pacing. Update content must be 10–6,000 characters; title may be 0–50 characters. An empty title clears it, while an omitted title preserves the old one. Delete accepts 1–10 chunks per call and is irreversible. For table or image knowledge bases, callers can pass dynamic spreadsheet fields; string values are prechecked only against the documented 6,000-character limit.

The add API is documented as idempotent, but the client sends each request once and never retries automatically. A transport interruption during add, update, or delete returns `OutcomeUnknown`; the host should inspect remote state before deciding whether to submit again. Chunk deletion changes remote data. Contract tests use a mock transport and do not call a live account.
