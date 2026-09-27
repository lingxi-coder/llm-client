# Qwen 数据中心类目

`QwenKnowledgeService` 提供北京地域 RAG 数据中心类目的列表、创建和删除操作。类目用于组织数据中心文件；删除类目不会删除其中的文件，而是将这些文件变为未归类状态，原有类目关联无法恢复。

```rust,no_run
# async fn example(
#     http: &dyn lingxi_llm_client::transport::Transport,
#     api_key: String,
# ) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    protocol::Secret,
    qwen_knowledge::{
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
let service = QwenKnowledgeService::new(http, Secret::new(api_key), scope)?;

let page = service
    .list_categories(&QwenKnowledgeCategoryListRequest::new())
    .await?;
if let Some(next_token) = page.next_token.clone() {
    let next_page = QwenKnowledgeCategoryListRequest::new().with_next_token(next_token);
    let _next = service.list_categories(&next_page).await?;
}

let parent = service.scope().category_ref("cate-parent-1")?;
let connector = service.scope().connector_ref("file_conn_1")?;
let created = service
    .create_category(
        &QwenKnowledgeCategoryCreateRequest::new("Product guides")
            .with_parent_category(parent)
            .with_connector_ref(connector),
    )
    .await?;
// Deletion is irreversible; files in this category become uncategorized.
let _deleted = service.delete_category(&created.reference).await?;
# Ok(())
# }
```

三个接口都是使用 workspace 对应的北京服务地址和 Bearer API Key 的 JSON `POST`：列表为 `/api/v1/connector/dash/listCategory`，创建为 `/api/v1/connector/dash/addCategory`，删除为 `/api/v1/connector/dash/deleteCategory`。类目引用绑定 provider、profile、account、region、workspace、endpoint 和 `UNSTRUCTURED` 类目命名空间；调用前会检查引用作用域。客户端不自动翻页或重试写操作。

`list_categories` 每次请求一页，查询类型固定为 `type: "UNSTRUCTURED"`。可选过滤条件包括通过 `parentId` 指定的父类目引用、精确匹配的 `categoryName` 和绑定当前作用域的 `connectorId` 引用。分页使用 `nextToken`，页大小字段的供应商拼写是单数 `maxResult`，文档默认值为 20。若响应中的 `hasNext` 为 true，必须返回非空且前进的 `nextToken`；调用方可原样将其传入下一次请求。返回类目包含类型化引用、名称、类型和默认类目标记，并保留完整原始行与响应。

`create_category` 创建一个 `UNSTRUCTURED` 类目。名称必须为 1–20 个字符；可选的 `parentCategoryId` 用于创建子类目，绑定作用域的 connector 引用会通过其 `connectorId` 关联类目。成功响应会提供作用域绑定的类目引用，并保留原始响应。

`delete_category` 会永久删除指定类目。供应商说明，类目中的数据中心文件将变为未归类状态，且无法恢复原有类目关系。该方法不会删除文件或另行执行知识库文档、索引操作。

创建或删除遇到超时、写入中断，或成功 HTTP 响应无法确认操作结果时，客户端会返回结果未知，不会自动重试。调用方可以检查类目状态后再决定如何处理。

官方文档：[RAG API 概览](https://help.aliyun.com/en/model-studio/rag-api-overview)、[查询类目列表](https://help.aliyun.com/en/model-studio/rag-api-list-category)、[新增类目](https://help.aliyun.com/en/model-studio/rag-api-add-category)和[删除类目](https://help.aliyun.com/en/model-studio/rag-api-delete-category)。
