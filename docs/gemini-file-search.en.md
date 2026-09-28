# Gemini File Search

Bind `client.provider::<GoogleClient>(profile)?` to an exact profile, then use `provider.file_search()`. Each operation takes `RequestOptions`; the client retains no credential.

Gemini File Search provides a managed index for retrieval-augmented generation. This module manages File Search stores, supports direct byte uploads or imports existing Gemini Files API resources, reads both kinds of indexing operation, and lists, gets, or deletes documents. Callers provide the API key and a stable, non-secret `account_scope` on each request.

```rust,ignore
use lingxi_llm_client::{
    providers::google::file_search::{GeminiFileSearchMetadata, GeminiFileSearchMetadataValue,
        GeminiFileSearchOperationState, GeminiFileSearchWhiteSpaceChunking},
    LlmClient, ProviderFileRef, RequestOptions,
};

async fn manage(
    client: &LlmClient,
    options: &RequestOptions,
    file: &ProviderFileRef,
) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::google::GoogleClient>("gemini")?;
    let service = provider.file_search();
    let store = service
        .create_store(
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

    // Import returns a Google long-running Operation. The host chooses when to
    // fetch it again; this library does not sleep or poll automatically.
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

Use `import_file()` when a file has already been uploaded through the Gemini Files API. `upload_to_store()` uses Google's documented resumable upload path to send bytes directly to a store and returns a separate upload-operation reference:

```rust,ignore
use bytes::Bytes;
use lingxi_llm_client::providers::google::file_search::GeminiFileSearchUploadRequest;

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

`list_stores` and `list_documents` accept only the `nextPageToken` returned by Gemini and pass it back unchanged as `pageToken`; page sizes follow the documented maximum of 20. Unknown document states and native JSON are preserved. Store creation, file import, and resource deletion are never retried automatically. A transport interruption returns `GeminiFileSearchError::OutcomeUnknown`; callers should inspect remote state before resubmitting.

Resource references bind the provider profile, File Search endpoint fingerprint, and `account_scope`. An imported `ProviderFileRef` must also come from the same Google profile's Gemini Files endpoint and match `file_account_scope`. `force=false` is the default when deleting a store or document; setting it to `true` also removes related server-side data.

List a store's documents before deleting them when you want to keep deletion explicit:

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

If a document contains chunks, Google requires `true` to delete it along with those chunks. After deleting every document, you can delete the empty store with `false`; deleting a store that still contains documents also requires opting into cascading deletion.

Direct upload accepts at most 100 MiB of in-memory bytes; larger inputs should go through the Gemini Files API and then `import_file()`. Both paths submit once without automatic retry or indexing polling. `import_file()` returns operations under `/operations/{id}`; direct uploads use a separate `/upload/operations/{id}` path and are queried through `get_upload_operation()`.

The API follows Google's official [File Search guide](https://ai.google.dev/gemini-api/docs/file-search), [File Search Stores API](https://ai.google.dev/api/file-search/file-search-stores), and [Documents API](https://ai.google.dev/api/file-search/documents).
