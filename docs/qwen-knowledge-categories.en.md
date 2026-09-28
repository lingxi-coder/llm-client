# Qwen Data-Center Categories

`QwenKnowledgeService` provides Beijing RAG data-center category listing, creation, and deletion. Categories organize data-center files. Deleting a category does not delete its files; those files become uncategorized, and their former category association cannot be restored.

```rust,no_run
# async fn example(
#     http: &dyn lingxi_llm_client::transport::Transport,
#     api_key: String,
# ) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    protocol::Secret,
    providers::qwen::knowledge::{
        QwenKnowledgeCategoryCreateRequest, QwenKnowledgeCategoryListRequest,
        QwenKnowledgeRegion, QwenKnowledgeScope, QwenKnowledgeService,
    },
};

let scope = QwenKnowledgeScope::new(
    "qwen-profile",
    "team/account-1",
    QwenKnowledgeRegion::Beijing,
    "llm-workspace-1",
)?;
let request_options = lingxi_llm_client::RequestOptions {
    credential: Some(Secret::new(api_key)),
    ..Default::default()
};
let service = QwenKnowledgeService::new(http, scope)?;

let page = service
    .list_categories(&QwenKnowledgeCategoryListRequest::new(), &request_options)
    .await?;
if let Some(next_token) = page.next_token.clone() {
    let next_page = QwenKnowledgeCategoryListRequest::new().with_next_token(next_token);
    let _next = service.list_categories(&next_page, &request_options).await?;
}

let parent = service.scope().category_ref("cate-parent-1")?;
let connector = service.scope().connector_ref("file_conn_1")?;
let created = service
    .create_category(
        &QwenKnowledgeCategoryCreateRequest::new("Product guides")
            .with_parent_category(parent)
            .with_connector_ref(connector),
        &request_options,
    )
    .await?;
// Deletion is irreversible; files in this category become uncategorized.
let _deleted = service.delete_category(&created.reference, &request_options).await?;
# Ok(())
# }
```

All three APIs are JSON `POST` requests using the workspace-bound Beijing endpoint and a Bearer API key: list uses `/api/v1/connector/dash/listCategory`, create uses `/api/v1/connector/dash/addCategory`, and delete uses `/api/v1/connector/dash/deleteCategory`. Category references are bound to the provider, profile, account, region, workspace, endpoint, and `UNSTRUCTURED` category namespace; the service checks that scope before dispatch. The client does not paginate or retry writes automatically.

`list_categories` returns one page at a time and always sends `type: "UNSTRUCTURED"`. Optional filters are a scoped parent-category reference sent as `parentId`, exact-match `categoryName`, and a connector reference bound to the current scope sent as `connectorId`. Pagination uses `nextToken`. The provider's page-size field is singular `maxResult`, with a documented default of 20. When `hasNext` is true, the response must include a nonempty advancing `nextToken`; pass it unchanged in another request to fetch the next page. Each category exposes a typed reference, name, type, and default-category flag while retaining its complete native row and response.

`create_category` creates an `UNSTRUCTURED` category. Its name must contain 1–20 characters. The optional `parentCategoryId` creates a child category, and a connector reference bound to the current scope associates the category with that connector ID. A successful response returns a scope-bound category reference and retains the native response.

`delete_category` permanently deletes the specified category. The provider states that its data-center files become uncategorized and the former category association cannot be restored. This method does not delete files or perform separate knowledge-base document or index operations.

If create or delete times out, is interrupted during writing, or receives a successful HTTP response that does not confirm the operation, the client reports an unknown outcome and does not retry automatically. Callers can inspect category state before deciding what to do next.

Official references: [RAG API overview](https://help.aliyun.com/en/model-studio/rag-api-overview), [List Categories](https://help.aliyun.com/en/model-studio/rag-api-list-category), [Create Category](https://help.aliyun.com/en/model-studio/rag-api-add-category), and [Delete Category](https://help.aliyun.com/en/model-studio/rag-api-delete-category).
