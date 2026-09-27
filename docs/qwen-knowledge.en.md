# Qwen Model Studio Knowledge Bases and Documents

[中文](qwen-knowledge.md)

`client.qwen_knowledge(api_key, scope)` provides Alibaba Cloud Model Studio RAG REST operations for knowledge bases, documents, and import jobs, while retaining low-level `retrieve`. Knowledge-base, document, and import-job references bind the account scope, profile, region, workspace, and parent knowledge-base ID so they cannot be reused across connections or bases.

The current REST overview documents the Beijing workspace URL: `https://{workspace_id}.cn-beijing.maas.aliyuncs.com`. `QwenKnowledgeRegion::Singapore` remains available to describe the origin of a reference, but this service rejects every Singapore RAG REST call before sending it rather than guessing a route. This applies to retrieval, reads, and writes.

```rust,ignore
use lingxi_llm_client::{
    protocol::Secret,
    qwen_knowledge::{
        QwenKnowledgeCreateRequest, QwenKnowledgeDocumentPageRequest,
        QwenKnowledgeRegion, QwenKnowledgeScope,
    },
};

let scope = QwenKnowledgeScope::new(
    "qwen-beijing-account",
    "tenant-42",
    QwenKnowledgeRegion::Beijing,
    "llm-your-workspace-id",
)?;
let service = client.qwen_knowledge(Secret::new(api_key), scope)?;

// create_v2 creates the knowledge base and starts its initial import job.
let created = service
    .create_knowledge_base(&QwenKnowledgeCreateRequest::new(
        "Returns",
        "Return policy documents",
        ["registered-file-id"],
    ))
    .await?;

// The caller triggers later operations separately; the service does not poll.
let documents = service
    .list_documents(&created.knowledge, QwenKnowledgeDocumentPageRequest::default())
    .await?;
let import = service
    .get_import_job_status(
        &created.job,
        QwenKnowledgeDocumentPageRequest::default(),
    )
    .await?;
let _ = (documents, import);
```

## Implemented REST operations

| Operation | Route | Description |
| --- | --- | --- |
| Create and import | `POST /api/v1/indices/rag/index/create_v2` | Create a base and start an initial import using registered file IDs |
| List knowledge bases | `GET /api/v1/indices/rag/index/list` | Paginate through the workspace's knowledge bases |
| Find a knowledge-base detail | Documented list route | There is no separate detail route; `find_knowledge_base` scans pages and matches the exact ID |
| Update a knowledge base | `POST /api/v1/indices/rag/index/update` | Update name, description, or rerank minimum score |
| Delete a knowledge base | `POST /api/v1/indices/rag/index/delete` | Permanently delete the base, documents, and chunks |
| List documents | `GET /api/v1/indices/rag/index/files` | Paginate through document states |
| Document details | `POST /api/v1/indices/rag/list/index/file/details` | Read parsing and chunk settings; at most 10 items per page |
| Delete documents | `POST /api/v1/indices/rag/index/delete_file` | Delete selected documents and their chunks |
| Submit an appended import | `POST /api/v1/indices/rag/index/job/create` | Start an import from explicit file or category IDs |
| Get import-job status | `GET /api/v1/indices/rag/index_job/status` | Query one job and per-document processing state |

Each endpoint keeps its documented parameter casing. Knowledge-base and document lists use `page_number` and `page_size` in the query string. Document-details POST uses `indexId`, `pageNumber`, and `pageSize`; import status requires both `index_id` and `job_id` in the query. Document list pages support up to 100 rows; details pages support up to 10. Create uses `docIds`; update uses `id`; knowledge-base delete uses `index_id`; document delete uses `index_id` and `doc_ids`.

Create requires file IDs already registered in the workspace data center, along with `sourceType` and `dataSources`. Appended imports use `QwenKnowledgeImportRequest::from_files` or `from_categories` to select a source explicitly. Because the API imports the entire data center when `sourceType` is omitted, the service never constructs that request. The service also supports the documented three-call file flow below; the calls remain explicit and separately observable.

## File upload and registration

The current data-import REST reference documents `applyFileUploadLease` → OSS `PUT` → `addFile`, plus a one-shot `describeFile` lookup. The Rust service exposes each call separately. It does not calculate the caller-provided MD5, chain steps, retry an upload, follow up on `PARSING`, or infer a lease expiration time.

```rust,no_run
use lingxi_llm_client::{
    files::UploadFileStream,
    protocol::LlmError,
    qwen_knowledge::{
        QwenKnowledgeFileUploadRequest, QwenKnowledgeParser,
        QwenKnowledgeRegisterFileRequest, QwenKnowledgeService,
        QwenKnowledgeError,
    },
};

async fn upload_file(
    service: &QwenKnowledgeService<'_>,
    content_md5: &str,
) -> Result<(), QwenKnowledgeError> {
    let lease = service
        .request_file_upload_lease(&QwenKnowledgeFileUploadRequest::new(
            "default",
            "guide.md",
            11,
            content_md5, // computed by the caller before streaming
        ))
        .await?;
    service
        .upload_file_content(
            &lease,
            UploadFileStream::new(
                "guide.md",
                "text/plain",
                11,
                futures::stream::iter([Ok::<_, LlmError>(b"hello world".to_vec().into())]),
            ),
        )
        .await?;
    let registered = service
        .register_file(
            &lease,
            &QwenKnowledgeRegisterFileRequest::new(QwenKnowledgeParser::AutoSelect),
        )
        .await?;
    let details = service.describe_file(&registered.reference).await?;
    let _ = details;
    Ok(())
}
```

The lease request sends `sizeBytes` as a decimal string and sends the supplied `contentMd5` unchanged. Alibaba's lease reference describes the digest as Base64 but shows a hexadecimal example; this client does not guess a conversion. The stream filename and declared byte count must match the lease. The returned OSS URL must be HTTPS on the documented OSS endpoint shape, and the validated headers from the lease are sent verbatim; they remain authoritative even if the stream's descriptive media type differs. The workspace API key is sent only to the RAG workspace endpoint, never to OSS. `Transport` implementations are required not to retry or follow redirects.

The lease is tied to the exact profile, account scope, region, workspace, and endpoint. After the caller successfully PUTs the bytes, it explicitly calls `register_file`; the method sends `leaseId`, the lease's original `category` and `categoryType`, and caller-selected parser options. `DASH_QWEN_VL_PARSER` requires `parser_config`; its prompt may be multiline and follows the documented 1–1500 character limit. A registration response can report `PARSING`; `describe_file` returns one snapshot and leaves any further polling to the caller.

`QwenKnowledgeFileRef` keeps the same connection scope and any known category type. Use `create_knowledge_base_with_files` or `submit_import_job_with_files` to ensure typed references match the IDs in the request. A known `SESSION_FILE` reference is rejected for long-term knowledge-base imports because Model Studio documents those files as session-only. When binding a pre-existing ID with `scope.file_ref`, its category type is unknown; callers must only use the typed knowledge-base helpers for files known to be in a compatible category.

Lease creation, OSS upload, and registration are each mutations sent once. A transport failure, uncertain HTTP status, or malformed successful mutation response reports `QwenKnowledgeError::dispatch() == Unknown`; inspect provider state before deciding whether to continue, and do not retry automatically. A known provider rejection reports `Rejected`. Short, overlong, or interrupted byte streams also report an unknown upload outcome because dispatch has begun.

The file operations follow Alibaba Cloud's first-party [Request Upload Lease](https://help.aliyun.com/en/model-studio/rag-api-upload-lease), [Register File](https://help.aliyun.com/en/model-studio/rag-api-add-file), and [Describe File](https://help.aliyun.com/en/model-studio/rag-api-describe-file) references. The lease's `status` and the registration/describe `status` or `status_code` business envelope must be checked in addition to HTTP status.

Model Studio does not currently document a dedicated knowledge-base detail REST route. `find_knowledge_base` uses the official paginated list route and its `id` field to locate an exact record, issuing one read request per page. The update route's identifier field is `id`; it must not be substituted with `index_id` or `indexId`.

Knowledge-base deletion is irreversible. Document deletion also removes its chunks from the index. Each write is sent once. Transport errors, HTTP 408/5xx, or a response without a conclusive success envelope return `Unknown` through `QwenKnowledgeError::dispatch()`. The server may have accepted the request, so callers should inspect the knowledge-base list or import-job status before deciding what to do; do not retry automatically. Definite parameter or business errors return `Rejected`. Local validation and Singapore-region checks run before dispatch.

The implementation follows Alibaba Cloud's first-party [RAG API overview](https://help.aliyun.com/en/model-studio/rag-api-overview), [create and import](https://help.aliyun.com/en/model-studio/rag-api-create-index), [knowledge-base list](https://help.aliyun.com/en/model-studio/rag-api-list-indices), [update](https://help.aliyun.com/en/model-studio/rag-api-update-index), [delete](https://help.aliyun.com/en/model-studio/rag-api-delete-index), [document list](https://help.aliyun.com/en/model-studio/rag-api-list-documents), [document details](https://help.aliyun.com/en/model-studio/rag-api-list-file-details), [document delete](https://help.aliyun.com/en/model-studio/rag-api-delete-document), [submit import job](https://help.aliyun.com/en/model-studio/rag-api-submit-sync-job), and [job status](https://help.aliyun.com/en/model-studio/rag-api-get-sync-job-status) references. Crate tests use mocks only and make no live account calls.

## Chunks, file management, and monitoring

[Chunk operations](qwen-knowledge-chunks.en.md) support creation, paginated listing, updates, and batch deletion. [Data-center file management](qwen-knowledge-files.en.md) supports cursor listing, deletion, and batch tag updates. Source files and knowledge-base documents are distinct resources; deleting a source file affects knowledge bases that reference it.

`get_knowledge_base_monitoring` reads a scoped knowledge base over a window of at most 30 days. Supply Unix seconds; the client does not poll. The official field table and example disagree on nested shapes, so `data` preserves raw JSON rather than coercing storage/QPS fields.

```rust,no_run
# async fn example(
#     service: &lingxi_llm_client::qwen_knowledge::QwenKnowledgeService<'_>,
#     knowledge: &lingxi_llm_client::qwen_knowledge::QwenKnowledgeRef,
# ) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::qwen_knowledge::QwenKnowledgeMonitoringRequest;
let window = QwenKnowledgeMonitoringRequest::new(1_750_000_000, 1_750_086_400);
let result = service.get_knowledge_base_monitoring(knowledge, &window).await?;
let _ = (result.data, result.request_id);
# Ok(())
# }
```

Source: [Knowledge base monitoring](https://help.aliyun.com/en/model-studio/rag-api-get-index-monitor).

[Category management](qwen-knowledge-categories.en.md) supports listing, creation, and deletion. [Connectors and OSS import](qwen-knowledge-connectors.en.md) support creating/reading file connectors and importing explicit object keys from an already authorized bucket. Category and connector references are scope-checked before dispatch; the client does not configure cloud authorization.

[Knowledge Search](qwen-knowledge-search.en.md) queries a published hosted retrieval service with text/images and optional per-knowledge-base runtime filters. Its protocol is separate from single-knowledge-base `retrieve`.

[Knowledge Chat](qwen-knowledge-chat.en.md) calls a published Q&A service with caller-supplied full history and native SSE phases. Hosted tool results are retained for replay, not dispatched to host tool execution.
