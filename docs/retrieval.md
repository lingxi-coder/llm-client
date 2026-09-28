# 托管知识库检索

[English](retrieval.en.md)

先用 `client.provider::<OpenAiClient>(profile)?` 绑定具体 profile，再通过 `provider.retrieval()` 调用资源。每次操作传入 `RequestOptions`，client 不保存凭证。

`provider.retrieval()` 使用独立于 Chat 和 Embeddings 的服务路由。当前实现 OpenAI Vector Stores 的索引创建、列举、查询、删除，关联已上传文件、查询索引状态、移除文件，以及语义搜索。内置 `openai` profile 配置了完整操作 URL；其他提供方的托管知识库接口尚未接入。接口依据 [OpenAI Retrieval 指南](https://developers.openai.com/api/docs/guides/retrieval) 与 [Vector Stores API](https://developers.openai.com/api/reference/resources/vector_stores)。

Files 上传和索引处理是两个独立操作。调用方先通过该 provider/profile 的 `FileService` 上传文件，取得 `ProviderFileRef`，再将其传给 `attach_file()`。后者返回 `IndexTask`，状态可能是 `InProgress`；通过 `get_file()` 查询后续状态，只有 `status.is_ready()` 时才把该文件视为已完成索引。库不会擅自轮询、重传文件或把 HTTP 成功当成索引就绪。文件从索引删除后，搜索结果可能短暂仍包含其内容，调用方应按提供方的最终一致性处理。

需要按文件属性筛选时，关联文件可调用 `attach_file_with_attributes()`，传入 `BTreeMap<String, serde_json::Value>`。最多 16 个属性；键和标量值按 API 限制在发送前校验。`update_file_attributes()` 可替换已有索引文件的属性。若要指定分块大小和重叠，调用 `attach_file_with_options()` 并传入 `FileChunking::Static`；大小为 100–4096 token，重叠不超过一半。

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions, ProviderFileRef};
use lingxi_llm_client::providers::openai::retrieval::RetrievalError;

async fn index_file(
    client: &LlmClient,
    uploaded_file: &ProviderFileRef,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>("openai")?;
    // options.account_scope 与上传文件的 account_scope 必须是同一非密钥账号标识。
    let store = provider.retrieval().create_store("FAQ", options).await?;
    let task = provider.retrieval().attach_file(&store.reference, uploaded_file, options).await?;
    let current = provider.retrieval().get_file(&task.reference, options).await?;
    if current.status.is_ready() {
        let results = provider.retrieval().search(&store.reference, "退货期限", 10, options).await?;
        for hit in results.hits {
            println!("{}: {:?}", hit.file.file_id, hit.content);
        }
    }
    Ok(())
}
```

`RetrievalStoreRef` 和 `RetrievalFileRef` 绑定 provider、profile、检索端点和 `RequestOptions.account_scope`。文件关联还要求 `ProviderFileRef` 具有同一 provider、profile、文件端点和账户作用域。跨账户、跨连接或跨区域使用会在 HTTP 前被拒绝。持久化配置 v3 中，`ProviderProfile.retrieval` 可继承、替换或显式禁用；服务 URL 不从 Chat URL 推导。

批量索引使用 `create_file_batch()`，一次提交 1–2000 个同作用域、已上传文件。每个 `BatchFileInput` 可有自己的属性和分块策略，响应的 `IndexBatchTask.status` 表示整体异步进度；`file_counts` 保留各状态计数。只有 `all_files_ready()` 为真才表示整批文件均完成，不能只看 `status == Completed`。调用方用 `get_file_batch()` 查询状态，用 `list_batch_files()` 按 `last_id` 翻页检查批次中的文件，或用 `list_store_files()` 检查整个 store 的文件状态；也可用 `cancel_file_batch()` 尝试取消。取消并不保证已完成文件回滚。接口依据 [OpenAI File Batches API](https://developers.openai.com/api/reference/resources/vector_stores/subresources/file_batches/methods/create)。

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

`create_store()`、`attach_file()` 和 `create_file_batch()` 不自动重试。状态变更请求传输中断时返回 `RetrievalError::OutcomeUnknown`；调用方先核对远端状态，再决定是否重新提交。批量创建没有列出批次的 API，提交结果未知时可用 `list_store_files()` 检查文件状态，不能盲目重提。`search()` 是只读请求，传输失败保留原错误类别。请求和响应上限为 64 MiB，默认总时限为 120 秒。搜索最多请求 50 条，返回分块内容、分数、文件属性和原生分页游标。

高级搜索使用 `SearchRequest` 和 `search_with()`，可组合文件属性过滤器、排名器、分数阈值及查询重写。过滤器和阈值在 HTTP 前校验；服务端重写后的查询保留在 `search_query`。字段依据 [OpenAI 搜索接口](https://developers.openai.com/api/reference/resources/vector_stores/methods/search)。

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::providers::openai::retrieval::{RetrievalStoreRef, RetrievalFilter, SearchRequest, SearchRanking, SearchRanker, RetrievalError};
use serde_json::json;

async fn filtered_search(client: &LlmClient, store: &RetrievalStoreRef, options: &RequestOptions) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::openai::OpenAiClient>(&store.profile_name)?;
    let mut request = SearchRequest::new("退货期限", 20);
    request.filters = Some(RetrievalFilter::Eq { key: "region".into(), value: json!("us") });
    request.ranking = Some(SearchRanking { ranker: Some(SearchRanker::Auto), score_threshold: Some(0.7) });
    request.rewrite_query = Some(true);
    let result = provider.retrieval().search_with(store, &request, options).await?;
    println!("rewritten query: {}", result.search_query);
    Ok(())
}
```

`list_store_files_with()` 可使用 `IndexFileListRequest` 指定 `after` 或 `before` 游标、文件状态过滤和创建时间升降序；两种游标不能同时使用，错误参数在发送前拒绝。字段依据 [OpenAI 文件列表接口](https://developers.openai.com/api/reference/resources/vector_stores/subresources/files/methods/list)。

目前只提供第一页搜索；官方接口虽返回 `next_page`，其搜索请求文档未列出后续页游标参数，因此尚未实现后续页请求。该服务不会执行 RAG 合成，宿主自行将结果放入 Chat 提示。

删除索引或关联文件会影响远端数据，请由宿主决定何时调用。索引存储和搜索可能单独计费；token 价格估算不包含这些费用。本地测试使用 mock，没有真实账户验收。
