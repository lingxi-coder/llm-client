# Qwen 百炼知识库与文档

[English](qwen-knowledge.en.md)

`client.provider::<QwenClient>(profile)?.knowledge(scope)` 提供阿里云 Model Studio（百炼）RAG REST 工作空间内的知识库、文档和导入任务操作，并保留原有低层 `retrieve`。知识库、文档和导入任务引用都绑定 account scope、profile、地域、workspace 和知识库 ID，不能跨连接或知识库复用。

当前 REST API 总览只文档化北京的工作空间地址：`https://{workspace_id}.cn-beijing.maas.aliyuncs.com`。`QwenKnowledgeRegion::Singapore` 仍可用于描述引用来源，但此服务会在发送任何 RAG REST 请求前拒绝新加坡地域，不会猜测端点。这个限制覆盖检索、读操作和写操作。

```rust,ignore
use lingxi_llm_client::{
    protocol::Secret,
    providers::qwen::knowledge::{
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
let provider = client.provider::<lingxi_llm_client::providers::qwen::QwenClient>(scope.profile_name())?;
let request_options = lingxi_llm_client::RequestOptions {
    credential: Some(Secret::new(api_key)),
    ..Default::default()
};
let service = provider.knowledge(scope)?;

// create_v2 同时创建知识库并提交初始文件索引任务。
let created = service
    .create_knowledge_base(&QwenKnowledgeCreateRequest::new(
        "Returns",
        "Return policy documents",
        ["registered-file-id"],
    ), &request_options)
    .await?;

// 后续操作由调用方分别触发；该服务不会自动轮询导入任务。
let documents = service
    .list_documents(&created.knowledge, QwenKnowledgeDocumentPageRequest::default(), &request_options)
    .await?;
let import = service
    .get_import_job_status(
        &created.job,
        QwenKnowledgeDocumentPageRequest::default(),
        &request_options,
    )
    .await?;
let _ = (documents, import);
```

## 已实现的 REST 操作

| 操作 | 路由 | 说明 |
| --- | --- | --- |
| 创建并导入 | `POST /api/v1/indices/rag/index/create_v2` | 创建知识库，并用已注册文件 ID 启动初始导入 |
| 列出知识库 | `GET /api/v1/indices/rag/index/list` | 分页列出工作空间知识库 |
| 查找知识库详情 | 文档化的列表路由 | 没有单独的详情路由；`find_knowledge_base` 按页扫描列表并精确匹配 ID |
| 更新知识库 | `POST /api/v1/indices/rag/index/update` | 更新名称、描述或 rerank 最低分数 |
| 删除知识库 | `POST /api/v1/indices/rag/index/delete` | 永久删除知识库、文档和 chunks |
| 列出文档 | `GET /api/v1/indices/rag/index/files` | 分页查看文档状态 |
| 文档详情 | `POST /api/v1/indices/rag/list/index/file/details` | 查看文档解析和分块配置，单页最多 10 条 |
| 删除文档 | `POST /api/v1/indices/rag/index/delete_file` | 删除指定文档及其 chunks |
| 提交追加导入 | `POST /api/v1/indices/rag/index/job/create` | 从显式文件 ID 或类别 ID 启动任务 |
| 查询导入任务 | `GET /api/v1/indices/rag/index_job/status` | 查询一个任务及文档处理状态 |

各路由遵循各自文档的参数大小写：知识库列表和文档列表使用 query 中的 `page_number`、`page_size`；文档详情的 POST body 使用 `indexId`、`pageNumber`、`pageSize`；导入任务状态 query 需要同时带 `index_id` 和 `job_id`。文档列表单页最多 100 条，文档详情单页最多 10 条。创建使用 `docIds`；更新使用 `id`；删除知识库使用 `index_id`；删除文档使用 `index_id` 和 `doc_ids`。

创建知识库的 `docIds` 必须是已注册到工作空间数据中心的文件 ID。初始导入请求还需明确提供 `sourceType` 和 `dataSources`。追加导入通过 `QwenKnowledgeImportRequest::from_files` 或 `from_categories` 显式选择来源；该 API 在省略 `sourceType` 时会导入整个数据中心，因此此服务不会生成这种请求。服务也实现了下面文档化的三步文件流程；每一步仍由调用方单独触发和观察。

## 文件上传与注册

当前数据导入 REST 文档描述了 `applyFileUploadLease` → OSS `PUT` → `addFile`，以及单次 `describeFile` 查询。Rust 服务将每一步作为独立调用提供，不会计算调用方提供的 MD5、自动串联步骤、重试上传、自动跟进 `PARSING` 状态，也不会推断租约过期时间。

```rust,no_run
use lingxi_llm_client::{
    files::UploadFileStream,
    protocol::LlmError,
    providers::qwen::knowledge::{
        QwenKnowledgeFileUploadRequest, QwenKnowledgeParser,
        QwenKnowledgeRegisterFileRequest, QwenKnowledgeService,
        QwenKnowledgeError,
    },
};

async fn upload_file(
    request_options: &lingxi_llm_client::RequestOptions,
    service: &QwenKnowledgeService<'_>,
    content_md5: &str,
) -> Result<(), QwenKnowledgeError> {
    let lease = service
        .request_file_upload_lease(&QwenKnowledgeFileUploadRequest::new(
            "default",
            "guide.md",
            11,
            content_md5, // 调用方预先计算的摘要字符串
        ), request_options)
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
            request_options,
        )
        .await?;
    let registered = service
        .register_file(
            &lease,
            &QwenKnowledgeRegisterFileRequest::new(QwenKnowledgeParser::AutoSelect),
            request_options,
        )
        .await?;
    let details = service.describe_file(&registered.reference, request_options).await?;
    let _ = details;
    Ok(())
}
```

租约请求会把 `sizeBytes` 作为十进制字符串发送，并原样发送调用方提供的 `contentMd5`。阿里云租约文档的字段表称摘要为 Base64，但示例却是十六进制；客户端不猜测或转换格式。流文件名和声明字节数必须与租约匹配。返回的 OSS URL 必须是 HTTPS 且符合已文档化的 OSS 主机格式，租约返回的请求头会经过校验后原样发送；即使流的描述性媒体类型不同，也以租约请求头为准。工作空间 API Key 只发送到 RAG 工作空间接口，不会发送给 OSS。`Transport` 实现必须禁止自动重试和重定向。

租约绑定到确切的 profile、account scope、region、workspace 和 endpoint。调用方成功 PUT 文件字节后，再显式调用 `register_file`；该方法发送 `leaseId`、租约原有的 `category` 和 `categoryType`，以及调用方选择的解析器参数。`DASH_QWEN_VL_PARSER` 必须提供 `parser_config`，其提示词可以多行，长度遵循文档中的 1–1500 字符限制。注册响应可能表示文件仍在 `PARSING`；`describe_file` 只返回一次状态快照，后续轮询由调用方负责。

`QwenKnowledgeFileRef` 保留相同连接 scope 和已知的类别类型。可使用 `create_knowledge_base_with_files` 或 `submit_import_job_with_files` 确认类型化引用与请求中的文件 ID 一致。Model Studio 文档说明已知的 `SESSION_FILE` 只能用于当前会话，因此不能导入长期知识库。通过 `scope.file_ref` 绑定已有 ID 时，类别类型未知；只有确认文件类别兼容后，调用方才应把它用于知识库类型化导入方法。

创建租约、OSS 上传、注册均是只发送一次的变更操作。传输失败、状态码结果不确定或成功响应格式不完整时，`QwenKnowledgeError::dispatch()` 会报告 `Unknown`；调用方应先检查服务端状态再决定下一步，不要自动重试。明确的服务端拒绝会报告 `Rejected`。由于短流、超长流或中断发生在请求已分发后，上传结果也会报告为未知。

文件流程依据阿里云第一方 [申请上传租约](https://help.aliyun.com/zh/model-studio/rag-api-upload-lease)、[注册文件](https://help.aliyun.com/zh/model-studio/rag-api-add-file)和[查询文件详情](https://help.aliyun.com/zh/model-studio/rag-api-describe-file)文档实现。除 HTTP 状态外，还必须检查租约的 `status` 以及注册/查询响应中的 `status` 或 `status_code` 业务状态。

百炼目前没有文档化的独立知识库详情 REST 路由。`find_knowledge_base` 使用官方列表接口的分页和 `id` 字段查找精确知识库记录；每页会触发一次读取请求。更新路由的 ID 字段是 `id`，不可替换成 `index_id` 或 `indexId`。

删除知识库不可恢复。删除文档也会立即从知识库索引移除其 chunks。写操作只发送一次请求：传输错误、HTTP 408/5xx 或无法判定成功 envelope 的响应会通过 `QwenKnowledgeError::dispatch()` 报告 `Unknown`。这表示服务端可能已接受原请求，调用方应先查询知识库或导入任务状态，不能自动重试。明确的参数/业务错误为 `Rejected`。本地校验和新加坡地域预检发生在发送请求之前。

协议和参数按阿里云第一方 [RAG API 总览](https://help.aliyun.com/en/model-studio/rag-api-overview)、[创建并导入](https://help.aliyun.com/en/model-studio/rag-api-create-index)、[知识库列表](https://help.aliyun.com/en/model-studio/rag-api-list-indices)、[更新](https://help.aliyun.com/en/model-studio/rag-api-update-index)、[删除](https://help.aliyun.com/en/model-studio/rag-api-delete-index)、[文档列表](https://help.aliyun.com/en/model-studio/rag-api-list-documents)、[文档详情](https://help.aliyun.com/en/model-studio/rag-api-list-file-details)、[删除文档](https://help.aliyun.com/en/model-studio/rag-api-delete-document)、[提交导入任务](https://help.aliyun.com/en/model-studio/rag-api-submit-sync-job)和[查询任务状态](https://help.aliyun.com/en/model-studio/rag-api-get-sync-job-status)实现。crate 测试只使用 mock，不访问真实账户。

## 分块、文件管理与监控

[分块接口](qwen-knowledge-chunks.md)提供创建、分页列举、更新及批量删除；[数据中心文件管理](qwen-knowledge-files.md)提供游标列举、删除及批量标签更新。文件与知识库文档是不同资源；删除数据中心源文件会影响引用它的知识库。

`get_knowledge_base_monitoring` 查询指定知识库在最多 30 天时间窗内的监控信息。时间按秒传入，不自动轮询。官方字段表与示例对子结构的描述不同，因此 `data` 保留原始 JSON，而不强制转换存储/QPS 字段。

```rust,no_run
# async fn example(
#     request_options: &lingxi_llm_client::RequestOptions,
#     service: &lingxi_llm_client::providers::qwen::knowledge::QwenKnowledgeService<'_>,
#     knowledge: &lingxi_llm_client::providers::qwen::knowledge::QwenKnowledgeRef,
# ) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::providers::qwen::knowledge::QwenKnowledgeMonitoringRequest;
let window = QwenKnowledgeMonitoringRequest::new(1_750_000_000, 1_750_086_400);
let result = service.get_knowledge_base_monitoring(knowledge, &window, request_options).await?;
let _ = (result.data, result.request_id);
# Ok(())
# }
```

来源：[知识库监控](https://help.aliyun.com/en/model-studio/rag-api-get-index-monitor)。

[分类管理](qwen-knowledge-categories.md)支持列举、创建和删除；[connector 与 OSS 导入](qwen-knowledge-connectors.md)支持创建/查询文件连接器，以及从已授权 bucket 导入明确的 object key。分类与 connector 引用会在发送前校验作用域，客户端不自动配置云端授权。

[Knowledge Search](qwen-knowledge-search.md)通过已发布的托管检索服务进行图文查询，可提供按知识库绑定的运行时过滤；它与单个知识库的 `retrieve` 使用不同协议。

[Knowledge Chat](qwen-knowledge-chat.md)调用已发布问答服务，使用调用方提供的完整历史和原生 SSE 阶段；托管工具结果保留供回放，不交给宿主执行。
