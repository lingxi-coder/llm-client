# GLM 托管知识库

[English](glm-knowledge.en.md)

`client.glm_knowledge()` 提供智谱个人知识库的原生 API，与 `client.retrieval()` 的 OpenAI Vector Stores 路由分开。当前支持知识库分页列表、创建、详情、部分更新和删除；文档分页列表、URL 导入、流式 multipart 文件上传和知识库检索。接口依据智谱当前的[知识库列表](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E7%9F%A5%E8%AF%86%E5%BA%93%E5%88%97%E8%A1%A8)、[知识库详情](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E7%9F%A5%E8%AF%86%E5%BA%93%E8%AF%A6%E6%83%85)、[编辑知识库](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E7%BC%96%E8%BE%91%E7%9F%A5%E8%AF%86%E5%BA%93)、[删除知识库](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E5%88%A0%E9%99%A4%E7%9F%A5%E8%AF%86%E5%BA%93)、[文档列表](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E6%96%87%E6%A1%A3%E5%88%97%E8%A1%A8)、[上传 URL 文档](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E4%B8%8A%E4%BC%A0url%E6%96%87%E6%A1%A3)、[上传文件文档](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E4%B8%8A%E4%BC%A0%E6%96%87%E4%BB%B6%E6%96%87%E6%A1%A3)和[知识库检索](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E7%9F%A5%E8%AF%86%E5%BA%93%E6%A3%80%E7%B4%A2)文档。

把独立服务根配置在 Zhipu/GLM profile 的 `glm_knowledge` 路由中，API key 和账户身份按请求传入。`account_scope` 必须是调用方提供的非密钥、稳定账户标识；服务不读取或保存密钥。

```toml
glm_knowledge = { mode = "enabled", value = {
  endpoint = "https://open.bigmodel.cn/api/llm-application/open",
  auth = { type = "bearer" }
} }
```

创建知识库需要智谱支持的 Embedding ID（3、11 或 12）。URL 导入只提交提供方任务，不代表文档已经完成索引；可以传回调配置，索引状态的后续检查由宿主负责。更改绑定的向量化模型可能需要重新构建知识库，宿主应按智谱文档配置回调并安排后续处理。

```rust,no_run
# async fn example(client: &lingxi_llm_client::LlmClient, api_key: String) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{protocol::Secret, RequestOptions};
use lingxi_llm_client::files::UploadFileStream;
use lingxi_llm_client::glm_knowledge::{
    GlmCreateKnowledgeRequest, GlmKnowledgeDocumentListRequest, GlmKnowledgeEmbedding,
    GlmKnowledgeListRequest, GlmKnowledgeRetrieveRequest, GlmKnowledgeService,
    GlmUpdateKnowledgeRequest, GlmUrlDocumentInput, GlmUploadFileDocumentsRequest,
    GlmUploadUrlDocumentsRequest,
};

let options = RequestOptions {
    account_scope: Some("zhipu-user-42".into()),
    credential: Some(Secret::new(api_key)),
    ..Default::default()
};
let service: GlmKnowledgeService<'_> = client.glm_knowledge();
let knowledge = service.create_knowledge(
    "glm-mainland",
    &GlmCreateKnowledgeRequest {
        embedding_id: GlmKnowledgeEmbedding::Embedding3,
        name: "Product docs".into(),
        embedding_model: None,
        contextual: None,
        description: None,
        background: None,
        icon: None,
    },
    &options,
).await?;
service.upload_url_documents(
    &knowledge.reference,
    &GlmUploadUrlDocumentsRequest {
        documents: vec![GlmUrlDocumentInput {
            url: "https://example.org/returns".into(),
            knowledge_type: None,
            custom_separator: None,
            sentence_size: None,
            callback_url: None,
            callback_header: None,
        }],
    },
    &options,
).await?;
let uploaded = service.upload_file_documents(
    &knowledge.reference,
    vec![UploadFileStream::from_bytes(
        "returns.pdf",
        "application/pdf",
        b"caller-owned PDF bytes".to_vec(),
    )],
    &GlmUploadFileDocumentsRequest {
        knowledge_type: Some(5),
        custom_separator: Some(vec!["###".into()]),
        sentence_size: Some(400),
        parse_image: Some(true),
        ..Default::default()
    },
    &options,
).await?;
let page = service.list_knowledge(
    "glm-mainland",
    &GlmKnowledgeListRequest { page: Some(1), size: Some(10) },
    &options,
).await?;
let detail = service.get_knowledge(&knowledge.reference, &options).await?;
let documents = service.list_documents(
    &knowledge.reference,
    &GlmKnowledgeDocumentListRequest { page: Some(1), size: Some(10), word: None },
    &options,
).await?;
service.update_knowledge(
    &knowledge.reference,
    &GlmUpdateKnowledgeRequest { name: Some("Product handbook".into()), ..Default::default() },
    &options,
).await?;
let hits = service.retrieve(
    &[knowledge.reference.clone()],
    &GlmKnowledgeRetrieveRequest {
        query: "What is the return window?".into(),
        request_id: None,
        documents: vec![],
        top_k: Some(8),
        top_n: None,
        recall_method: None,
        recall_ratio: None,
        rerank: None,
        rerank_model: None,
        fractional_threshold: None,
    },
    &options,
).await?;
service.delete_knowledge(&knowledge.reference, &options).await?;
# Ok(())
# }
```

知识库引用和文档引用都绑定 provider、profile、endpoint 与 `account_scope`。详情、更新、删除、文档列表、URL 导入和文件上传都会在发请求前检查引用作用域。知识库与文档列表使用一页起始的 `page` / `size` 参数；文档列表还可用 `word` 搜索文档名称。文件上传使用可重复的 multipart `files` 字段，并支持 `knowledge_type`、`custom_separator`、`sentence_size`、`parse_image`、`callback_url`、`callback_header`、`word_num_limit` 和 `request_id`（线路字段为 `req_id`）。客户端每次接受 1–100 个文件、一次性流式发送并校验声明字节数；当前 API 文档未公布文件大小上限，客户端不会自行假定提供方限制。自定义分隔符和句子大小只可用于自定义切片（`knowledge_type = 5`）。

上传结果分别保留成功、失败记录。无法识别的成功记录放入 `unresolved`；响应数组未覆盖到的输入文件名放入 `missing_files`。上传成功不代表解析或索引已经完成，宿主应使用提供方回调或自行检查后续状态；客户端不会轮询或自动重试。发送后遇到传输错误，或收到 HTTP 408/5xx，会标记为未知结果；宿主应先查询提供方状态，再决定是否重新提交。请求 `Debug` 会隐藏回调 URL 和回调头值。

当前官方 API 参考提供文档列表与 URL/文件导入，没有单文档详情、更新或删除路由。文件上传接口说明了切片选项和逐文件成功/失败数组，见[上传文件文档](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E4%B8%8A%E4%BC%A0%E6%96%87%E4%BB%B6%E6%96%87%E6%A1%A3)。

`GlmKnowledgeRef` 和 `GlmKnowledgeDocumentRef` 携带 provider、profile、endpoint 指纹与 `account_scope`。知识库管理方法使用带作用域的引用，而不是裸 ID；provider、profile、endpoint 或账户不匹配时，会在发 HTTP 请求前拒绝。该模块只封装 GLM 原生 API，不适配 OpenAI Vector Stores，也不承诺跨提供方兼容。

检索支持文档化的向量、关键词和混合召回，可选重排与分数阈值。查询长度最多 1000 个字符，`top_k` 为 1–20，`top_n` 为 1–100，混合召回比例为 1–99。服务保留原生响应，并提供类型化文本、分数和来源元数据。它不会生成最终答案，也不会轮询索引状态。状态变更请求遇到传输错误时会返回 `OutcomeUnknown`；宿主应先查询提供方状态，再决定是否重试。契约测试使用 mock transport，不会调用线上服务。
