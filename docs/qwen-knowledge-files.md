# Qwen 数据中心文件管理

`QwenKnowledgeService` 提供北京地域 RAG 数据中心文件的列表、删除和批量标签更新操作。数据中心文件是原始资源；导入知识库后的文档是另一种资源。删除数据中心文件不会另外删除知识库文档，也不会重建受影响的索引。

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

三个接口都是 `/api/v1/connector/dash/` 下的单次 `POST`，使用绑定 workspace 的北京服务地址和 service 中的 Bearer API Key，以 JSON 收发数据。客户端不会自动翻页或重试写操作，也不会自动执行后续的文档和索引操作。

`list_files` 必须提供 `category_id`，每次返回一页。可选的 `file_name` 是不带扩展名的精确文件名；`file_ids` 最多可包含 20 个 ID；`next_token` 用于请求后续页面；`max_result` 使用供应商规定的单数拼写，文档默认值为 20，客户端不设置文档未说明的最大值。若响应中的 `hasNext` 为 true，必须同时提供非空且向前推进的 token。调用方可将该 token 原样传入新请求以继续分页。每个文件都附带绑定当前作用域的 `QwenKnowledgeFileRef`，并保留完整原始响应行。

`delete_file` 会永久删除数据中心文件，且无法撤销。供应商说明，如果知识库引用了该文件，对应的文档索引将失效。结果会保留供应商响应；响应包含 `fileId` 或 `DELETED` 状态时，也会提供对应的类型化字段。官方成功示例允许 `data` 为空对象。客户端不会另行删除知识库文档或重建索引。

`update_file_tags` 在一次 API 批量请求中接受 1–20 个文件条目。每个文件最多可有 100 个标签，每个标签最多 32 个字符，每个文件的标签字符总数最多为 700。`Overwrite` 会替换现有标签，`Append` 会追加标签。响应可能包含逐文件的 `success` 结果；客户端会分别返回这些结果，包括 `success: false`，不会将其合并为批次级成功。一些官方成功响应没有 `data.results`，因此 `per_file_results` 是可选字段，完整原始响应始终保留。

发送 HTTP 请求前，客户端会将每个文件引用与当前 provider、profile、account、region、workspace 和 endpoint 逐项核对。超时、写入中断，或 HTTP 成功但无法确认变更结果时，操作会返回结果未知。客户端不会自动重试；调用方可以自行检查状态并决定是否再次尝试。

官方文档：[RAG API 概览](https://help.aliyun.com/en/model-studio/rag-api-overview)、[文件列表](https://help.aliyun.com/en/model-studio/rag-api-list-file)、[删除文件](https://help.aliyun.com/en/model-studio/rag-api-delete-file)和[批量更新文件标签](https://help.aliyun.com/en/model-studio/rag-api-batch-update-tag)。
