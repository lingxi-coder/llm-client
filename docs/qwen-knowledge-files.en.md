# Qwen Data-Center File Management

`QwenKnowledgeService` provides Beijing RAG data-center file listing, deletion, and batch tag updates. A data-center file is the raw asset; a document imported into a knowledge base is a separate resource. Deleting the data-center file does not issue a separate document deletion or rebuild the affected index.

```rust,no_run
# async fn example(
#     http: &dyn lingxi_llm_client::transport::Transport,
#     api_key: String,
# ) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    protocol::Secret,
    qwen_knowledge::{
        QwenKnowledgeFileListRequest, QwenKnowledgeFileTagUpdate,
        QwenKnowledgeFileTagUpdateMode, QwenKnowledgeFileTagUpdateRequest,
        QwenKnowledgeRegion, QwenKnowledgeScope, QwenKnowledgeService,
    },
};

let scope = QwenKnowledgeScope::new(
    "qwen-profile",
    "team/account-1",
    QwenKnowledgeRegion::Beijing,
    "llm-workspace-1",
)?;
let service = QwenKnowledgeService::new(http, Secret::new(api_key), scope)?;

let request = QwenKnowledgeFileListRequest::new("cate-1").with_max_result(20);
let page = service.list_files(&request).await?;
if let Some(next_token) = page.next_token.clone() {
    let next = QwenKnowledgeFileListRequest::new("cate-1").with_next_token(next_token);
    let _next_page = service.list_files(&next).await?;
}

if let Some(file) = page.files.first() {
    let tags = QwenKnowledgeFileTagUpdateRequest::new([
        QwenKnowledgeFileTagUpdate::new(file.reference.clone(), ["FAQ", "returns"]),
    ])
    .with_update_mode(QwenKnowledgeFileTagUpdateMode::Overwrite);
    let _tag_result = service.update_file_tags(&tags).await?;
}
# Ok(())
# }
```

All three APIs are single `POST` requests under `/api/v1/connector/dash/`, using the workspace-bound Beijing endpoint and the service's Bearer API key. They send and receive JSON. The client does not paginate automatically, retry writes, or perform follow-up document or index operations.

`list_files` requires a `category_id` and returns one page. The optional `file_name` is an exact name without its extension. `file_ids` accepts up to 20 IDs. `next_token` continues a prior page, and `max_result` uses the provider's exact singular field name; the documented default is 20, with no client-invented maximum. When `hasNext` is true, the response must include a nonempty advancing token. Pass it unchanged in a new request to fetch the next page. Each file includes a `QwenKnowledgeFileRef` bound to the current scope and retains its complete native response row.

`delete_file` permanently deletes the data-center asset and cannot be undone. The provider warns that if a knowledge base references the file, its document index becomes invalid. The result preserves the provider response and exposes returned `fileId` and `DELETED` status when present; the official success example also permits an empty `data` object. The client does not remove a separate knowledge-base document or rebuild an index.

`update_file_tags` accepts 1–20 file entries in one API batch request. Each entry may have up to 100 tags, each at most 32 characters, with at most 700 tag characters total per file. `Overwrite` replaces existing tags; `Append` adds tags. The response may include per-file `success` results; these are returned individually, including `success: false`, rather than being collapsed into batch-level success. Some documented success responses omit `data.results`, so `per_file_results` is optional and the complete native response remains available.

Before HTTP dispatch, every supplied file reference is checked against the current provider, profile, account, region, workspace, and endpoint. A timeout, interrupted write, or successful HTTP status with an unconfirmed mutation response is reported as an unknown outcome. The service never retries automatically; callers decide whether to inspect state or try again.

Official references: [RAG API overview](https://help.aliyun.com/en/model-studio/rag-api-overview), [List Files](https://help.aliyun.com/en/model-studio/rag-api-list-file), [Delete File](https://help.aliyun.com/en/model-studio/rag-api-delete-file), and [Batch Update File Tags](https://help.aliyun.com/en/model-studio/rag-api-batch-update-tag).
