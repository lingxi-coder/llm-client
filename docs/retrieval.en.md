# Managed retrieval

[简体中文](retrieval.md)

Bind `client.provider::<OpenAiClient>(profile)?` to an exact profile, then use `provider.retrieval()`. Each operation takes `RequestOptions`; the client retains no credential.

`provider.retrieval()` uses a service route independent of Chat and Embeddings. The current OpenAI Vector Stores adapter creates, lists, retrieves and deletes stores; attaches uploaded files, checks indexing status and removes files; and performs semantic search. The built-in `openai` profile supplies the full operation URL. Other providers expose their managed knowledge resources through their own typed clients. This adapter follows the [OpenAI Retrieval guide](https://developers.openai.com/api/docs/guides/retrieval) and [Vector Stores API](https://developers.openai.com/api/reference/resources/vector_stores).

File upload and indexing are separate operations. Upload through the same provider/profile's `FileService` to obtain a `ProviderFileRef`, then pass it to `attach_file()`. This returns an `IndexTask`, possibly `InProgress`; call `get_file()` later and treat the file as indexed only when `status.is_ready()` is true. The client does not poll or reupload automatically, and HTTP success does not imply indexing readiness. Removal from a store is eventually consistent, so search may briefly return deleted content.

To filter by file attributes, attach a file with `attach_file_with_attributes()` and a `BTreeMap<String, serde_json::Value>`. Up to 16 attributes are accepted; keys and scalar values are validated before sending. `update_file_attributes()` replaces attributes on an indexed file. To control chunk size and overlap, call `attach_file_with_options()` with `FileChunking::Static`; size must be 100–4096 tokens and overlap no more than half.

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions, ProviderFileRef};
use lingxi_llm_client::providers::openai::retrieval::RetrievalError;

async fn index_file(
    client: &LlmClient,
    uploaded_file: &ProviderFileRef,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    // options.account_scope must identify the same provider account as uploaded_file.account_scope.
    let store = provider.retrieval().create_store("FAQ", options).await?;
    let task = provider.retrieval().attach_file(&store.reference, uploaded_file, options).await?;
    let current = provider.retrieval().get_file(&task.reference, options).await?;
    if current.status.is_ready() {
        let results = provider.retrieval().search(&store.reference, "return policy", 10, options).await?;
        for hit in results.hits {
            println!("{}: {:?}", hit.file.file_id, hit.content);
        }
    }
    Ok(())
}
```

`RetrievalStoreRef` and `RetrievalFileRef` bind the provider, profile, retrieval endpoint and `RequestOptions.account_scope`. Attaching a file also requires its `ProviderFileRef` to match the provider, profile, file endpoint and account scope. Cross-account, cross-connection or unavailable-region use fails before HTTP. `ProviderProfile.retrieval` can be inherited, replaced or disabled in persisted v3 configuration; its URL is never derived from the Chat URL.

`create_file_batch()` submits 1–2000 uploaded files with the same scope for asynchronous indexing. Each `BatchFileInput` may specify attributes and chunking. `IndexBatchTask.status` reports overall progress and `file_counts` retains counts by state. Only `all_files_ready()` confirms that every file completed; `status == Completed` alone is insufficient. Use `get_file_batch()` to check progress, `list_batch_files()` with its `last_id` cursor to inspect files in one batch, or `list_store_files()` to inspect every file in the store. `cancel_file_batch()` attempts cancellation without rolling back files already completed. These operations follow the [OpenAI File Batches API](https://developers.openai.com/api/reference/resources/vector_stores/subresources/file_batches/methods/create).

```rust,no_run
use lingxi_llm_client::{LlmClient, ProviderFileRef, RequestOptions};
use lingxi_llm_client::providers::openai::retrieval::{BatchFileInput, FileChunking, RetrievalStoreRef, RetrievalError};

async fn index_many(client: &LlmClient, store: &RetrievalStoreRef, uploaded: Vec<ProviderFileRef>, options: &RequestOptions) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>(&store.profile_name)?;
    let files: Vec<_> = uploaded.into_iter().map(|file| BatchFileInput {
        file,
        attributes: None,
        chunking: Some(FileChunking::Auto),
    }).collect();
    let batch = provider.retrieval().create_file_batch(store, &files, options).await?;
    let current = provider.retrieval().get_file_batch(&batch.reference, options).await?;
    println!("{:?}: {:?}", current.status, current.file_counts);
    Ok(())
}
```

`create_store()`, `attach_file()` and `create_file_batch()` never retry automatically. A transport failure during a state-changing request yields `RetrievalError::OutcomeUnknown`; check remote state before resubmitting. There is no batch-list operation to reconcile an uncertain batch creation directly; use `list_store_files()` to inspect file state instead of blindly submitting again. Read-only `search()` keeps the transport error classification. Requests and responses are limited to 64 MiB; the default total deadline is 120 seconds. A search requests at most 50 hits and returns chunks, scores, file attributes and the native page cursor.

For advanced search, use `SearchRequest` with `search_with()` to combine file-attribute filters, a ranker, a score threshold and query rewriting. Filters and thresholds are checked before HTTP, and `search_query` preserves the provider's rewritten query. These fields follow the [OpenAI search API](https://developers.openai.com/api/reference/resources/vector_stores/methods/search).

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::providers::openai::retrieval::{RetrievalStoreRef, RetrievalFilter, SearchRequest, SearchRanking, SearchRanker, RetrievalError};
use serde_json::json;

async fn filtered_search(client: &LlmClient, store: &RetrievalStoreRef, options: &RequestOptions) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>(&store.profile_name)?;
    let mut request = SearchRequest::new("return policy", 20);
    request.filters = Some(RetrievalFilter::Eq { key: "region".into(), value: json!("us") });
    request.ranking = Some(SearchRanking { ranker: Some(SearchRanker::Auto), score_threshold: Some(0.7) });
    request.rewrite_query = Some(true);
    let result = provider.retrieval().search_with(store, &request, options).await?;
    println!("rewritten query: {}", result.search_query);
    Ok(())
}
```

`list_store_files_with()` accepts `IndexFileListRequest` with an `after` or `before` cursor, file-status filter, and ascending or descending creation order. It rejects both cursors together before sending. These options follow the [OpenAI file-list API](https://developers.openai.com/api/reference/resources/vector_stores/subresources/files/methods/list).

Search still exposes only the first page. Although the response has `next_page`, the search request reference does not document a subsequent-page cursor parameter, so the client does not guess one. The host decides how to use hits in a Chat prompt.

Deleting a store or attached file changes remote data; the host decides when to call those methods. Index storage and search may be charged separately from tokens, so token price estimates exclude them. Local tests use mocks; no live account acceptance was run.
