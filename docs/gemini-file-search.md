# Gemini File Search

Gemini File Search 为检索增强生成托管索引。该模块管理 File Search stores，支持直接上传字节或导入已有 Gemini Files API 文件，查询两类索引 operation 状态，以及列出、读取和删除 documents。请求仍由调用方提供 API key 和稳定的非秘密 `account_scope`。

```rust,ignore
use lingxi_llm_client::{
    gemini_file_search::{GeminiFileSearchMetadata, GeminiFileSearchMetadataValue,
        GeminiFileSearchOperationState, GeminiFileSearchWhiteSpaceChunking},
    LlmClient, ProviderFileRef, RequestOptions,
};

async fn manage(
    client: &LlmClient,
    options: &RequestOptions,
    file: &ProviderFileRef,
) -> Result<(), Box<dyn std::error::Error>> {
    let service = client.gemini_file_search();
    let store = service
        .create_store(
            "gemini",
            Some("Product documents"),
            Some("models/gemini-embedding-2"),
            options,
        )
        .await?;

    // `file` is a ProviderFileRef returned by the Gemini Files API for this
    // provider profile and file account scope.
    let operation = service
        .import_file(
            &store.reference,
            file,
            &[GeminiFileSearchMetadata {
                key: "department".into(),
                value: GeminiFileSearchMetadataValue::String("support".into()),
            }],
            Some(GeminiFileSearchWhiteSpaceChunking {
                max_tokens_per_chunk: 200,
                max_overlap_tokens: 20,
            }),
            options,
        )
        .await?;

    // The import returns a Google long-running Operation. The host controls when
    // to fetch it again; this library does not sleep or poll automatically.
    let latest = service.get_operation(&operation.reference, options).await?;
    if latest.state == GeminiFileSearchOperationState::Succeeded {
        let documents = service
            .list_documents(&store.reference, Some(10), None, options)
            .await?;
        let ready_count = documents
            .items
            .iter()
            .filter(|doc| doc.state.is_ready())
            .count();
        let _ = ready_count;
    }
    Ok(())
}
```

若文件已由 Gemini Files API 上传，可用 `import_file()` 将其导入 store。也可以通过 `upload_to_store()` 使用官方的 resumable upload 路径直接导入字节；该方法提交一个完整上传块并返回独立的 upload operation 引用：

```rust,ignore
use bytes::Bytes;
use lingxi_llm_client::gemini_file_search::GeminiFileSearchUploadRequest;

let upload = GeminiFileSearchUploadRequest::new(
    Bytes::from_static(b"Product documentation"),
    "text/plain",
)
.with_display_name("product-docs.txt");
let operation = service.upload_to_store(&store.reference, &upload, options).await?;
let latest = service
    .get_upload_operation(&operation.reference, options)
    .await?;
```

`list_stores` 和 `list_documents` 只接受 Gemini 返回的 `nextPageToken`，并原样传回 `pageToken`；每页大小遵循官方上限 20。未知 document 状态与原始 JSON 会保留。创建 store、导入文件和删除资源不会自动重试；传输中断时返回 `GeminiFileSearchError::OutcomeUnknown`，调用方应先读取远端状态再决定是否重交。

资源引用绑定 provider profile、File Search endpoint 指纹和 `account_scope`。导入的 `ProviderFileRef` 还必须来自相同 Google profile 的 Gemini Files endpoint，并匹配 `file_account_scope`。`force=false` 是删除 store/document 的默认值；设为 `true` 会连同服务端关联的数据一起删除。

删除前可以先列出 store 中的 documents，并按需选择是否连同索引数据一起删除：

```rust,ignore
let mut documents = Vec::new();
let mut page_token = None;
loop {
    let page = service
        .list_documents(&store.reference, Some(20), page_token.as_deref(), options)
        .await?;
    documents.extend(page.items.into_iter().map(|document| document.reference));
    page_token = page.next_page_token;
    if page_token.is_none() {
        break;
    }
}
for document in documents {
    service
        .delete_document(&document, true, options)
        .await?;
}
service.delete_store(&store.reference, false, options).await?;
```

若 document 已包含 chunks，Google 要求删除时显式传入 `true` 才会一并清理 chunks。删除完所有 documents 后，可用 `false` 删除空 store；直接删除仍有 documents 的 store 时也必须显式选择级联删除。

直接上传路径最多接受 100 MiB 的内存字节；更大的输入应通过 Gemini Files API 上传后再调用 `import_file()`。两种上传方式都只提交一次，不自动重试或等待索引完成。`import_file()` 返回的 operation 使用 `/operations/{id}`，直接上传使用独立的 `/upload/operations/{id}`，分别通过 `get_operation()` 和 `get_upload_operation()` 查询。

接口依据 Google 官方 [File Search 指南](https://ai.google.dev/gemini-api/docs/file-search)、[File Search Stores API](https://ai.google.dev/api/file-search/file-search-stores) 和 [Documents API](https://ai.google.dev/api/file-search/documents)。
