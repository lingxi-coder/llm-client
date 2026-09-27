# xAI Collections

`xai_collections` 将 xAI 的托管知识库作为独立服务提供，与聊天和对话附件分开。当前支持创建、分页列出、读取、更新配置和删除 collection；上传文件并将文件加入 collection；分页列出、读取、批量读取、重新索引和移除文档；以及语义搜索。搜索结果携带绑定到 collection 的文档引用。

xAI 将管理和搜索拆成两种凭据与两个 API 根地址。管理 collection、通过 `upload_document_stream` 直传文档、添加文件和管理文档时使用 **Management API key**，请求发往 `https://management-api.x.ai/v1`；两步上传中的 `upload_file` 和语义搜索使用普通 **API key**，请求发往 `https://api.x.ai/v1`。库不会查找或持久化这些密钥。每次操作都通过 `XaiCollectionsCredentials` 显式传入密钥；该类型不能序列化，调试输出也会隐藏密钥。

```rust,ignore
use lingxi_llm_client::{
    files::UploadFileStream,
    protocol::Secret,
    xai_collections::{
        XaiCollectionListRequest, XaiCollectionsConfig, XaiCollectionsCredentials,
        XaiCreateCollectionRequest, XaiRetrievalMode, XaiSearchRequest,
        XaiUpdateCollectionRequest,
    },
};

let service = client.xai_collections(XaiCollectionsConfig::new(
    "xai-primary",
    "xai-team/account-42", // 调用方提供的稳定、非敏感账户范围
))?;
let credentials = XaiCollectionsCredentials::new(
    Secret::new(api_key),
    Secret::new(management_api_key),
);
let collection = service
    .create_collection(&XaiCreateCollectionRequest {
        name: "研究资料".into(),
        description: Some("内部报告".into()),
        index_configuration: None,
        chunk_configuration: None,
        field_definitions: vec![],
    }, &credentials)
    .await?;
let page = service
    .list_collections(&XaiCollectionListRequest::default(), &credentials)
    .await?;
let renamed = service
    .update_collection(
        &collection.reference,
        &XaiUpdateCollectionRequest {
            name: Some("研究笔记 2026".into()),
            ..Default::default()
        },
        &credentials,
    )
    .await?;
assert_eq!(renamed.name, "研究笔记 2026");
```

`XaiCollectionListRequest` 提供官方集合列表的 `filter`、`order` 和 `sort_by` 查询参数；`XaiDocumentListRequest` 提供文档列表的 `filter`、`order` 和 `sort_by`。API 参考把 `order` 和 `sort_by` 定义为字符串，但没有公布允许值枚举，因此客户端仅拒绝空值或包含控制字符的值，具体取值仍由 xAI 校验。文档列表使用 `filter`；xAI 参考已将旧的 `name` 查询参数标记为弃用。

`update_collection` 使用 Management API key 向 xAI 的 `PUT /v1/collections/{collection_id}` 发送请求，并返回更新后的 collection 对象。可单独或组合提供文档化字段：`name`、`description`、`chunk_configuration` 和非空 `field_definitions`。请求至少要包含一个字段；发送前会检查名称、分块配置对象和字段定义的 key，并核对响应中的 collection ID。操作只发送一次。如果派发后传输失败，xAI 可能已经应用更新；再次提交前先用 `get_collection` 检查当前状态。

上传按 xAI 文档分成两步。`upload_file` 先返回带范围的文件引用，`add_document` 再把它加入指定 collection，并可附上元数据字段。如果添加失败，调用方仍能保留已上传的文件 ID 并重试添加。

```rust,ignore
let uploaded = service.upload_file(&file, &credentials).await?;
let document = service
    .add_document(
        &collection.reference,
        &uploaded.reference,
        Some(&serde_json::json!({"author":"Mina","year":"2026"})),
        &credentials,
    )
    .await?;
let indexed = service.get_document(&document, &credentials).await?;
service.reindex_document(&document, &credentials).await?;
let indexed_again = service.get_document(&document, &credentials).await?;
```

也可以用 `upload_document_stream` 通过 Management API 的 multipart route 一步上传并加入 collection。它接受 `UploadFileStream`、可选 JSON 对象形式的元数据 `fields`，并直接返回 collection 范围的 `XaiDocument`：

```rust,ignore
let document = service
    .upload_document_stream(
        &collection.reference,
        UploadFileStream::from_bytes("paper.pdf", "application/pdf", pdf_bytes),
        Some(&serde_json::json!({"author":"Mina"})),
        &credentials,
    )
    .await?;
```

`batch_get_documents` 接受同一 collection 中的一个或多个 `XaiDocumentRef`，并返回 xAI 提供的文档元数据子集。官方接口没有说明缺失 ID 如何表示，因此客户端不会要求返回数量与请求数量一致；响应中的未请求 ID 或重复 ID 会被拒绝。

用 `list_documents` 或 `get_document` 查看 xAI 返回的索引状态。`reindex_document` 对已加入 collection 的文档发送官方 `PATCH /v1/collections/{collection_id}/documents/{file_id}` 操作，使用 Management API key；它只提交重新索引请求，不会等待完成。调用方可以稍后读取文档状态。服务不会自动轮询，何时再次查询由调用方决定。Collection 搜索接受一个或多个 `XaiCollectionRef` 以及可选的 xAI 过滤表达式；每条结果以 `XaiDocumentRef` 返回，并绑定到 xAI 在结果中报告的 collection。如果响应引用了本次搜索未请求的 collection，服务会将其视为格式错误。

搜索默认使用 xAI 的混合检索。要显式选择文档化模式，可将 `XaiSearchRequest::retrieval_mode` 设为 `Some(XaiRetrievalMode::Keyword)`、`Semantic` 或 `Hybrid`。线上请求会把它编码为 `{"type":"keyword"}`、`{"type":"semantic"}` 或 `{"type":"hybrid"}`；`None` 会省略该字段，让提供方应用默认值。

```rust,ignore
let results = service
    .search(
        &XaiSearchRequest {
            query: "查找收入指引".into(),
            collections: vec![collection.reference.clone()],
            filter: None,
            retrieval_mode: Some(XaiRetrievalMode::Keyword),
        },
        &credentials,
    )
    .await?;
```

Collection、上传文件和文档引用包含 provider ID、profile 名称、两个 API 根地址的非敏感指纹以及调用方提供的账户范围。把引用用于其他 profile、endpoint 或账户时，请求会在发送前被拒绝。引用可以序列化，密钥不能。

xAI 的 Collections 指南标注单文件上限为 100 MB；`upload_document_stream` 对应这一直接 Collections 上传路径，并在发送前限制为 100,000,000 字节。两步流程中的 `upload_file` 会调用 Files API 的 `POST /v1/files`，该独立接口文档标注 50 MB，本实现限制为 50,000,000 字节。上传和索引也可能要求账户有可用额度。

`upload_document_stream` 是一次性 mutation，不会自动重试。如果派发后传输失败或成功响应无法解码，结果可能未知；重试前应先用 `list_documents` 或 `get_document` 核对 collection 状态。全局 Files API 删除可通过 crate 的独立文件服务执行。`reindex_document` 使用 Management API key 提交 xAI 文档化的 `PATCH /v1/collections/{collection_id}/documents/{file_id}` 操作，并立即返回；调用方稍后可读取文档状态。如果请求传输中断，远端可能已经接受请求，调用方应先读取文档状态再决定是否重试。服务不会自动等待索引完成，也没有执行真实 provider 请求；契约测试使用注入的 mock transport。

参考：[xAI Collections 概览](https://docs.x.ai/developers/files/collections)、[Collections REST API](https://docs.x.ai/developers/rest-api-reference/collections)、[Collection 管理参考](https://docs.x.ai/developers/rest-api-reference/collections/collection)、[Collections API 指南](https://docs.x.ai/developers/files/collections/api)、[搜索 API](https://docs.x.ai/developers/rest-api-reference/collections/search)、[Files 上传 API](https://docs.x.ai/developers/rest-api-reference/files/upload)。

## 接续普通文件服务

通过 `upload_file` 上传的结果可以转换后交给普通 `FileService` 查询、下载或删除。转换会检查 provider、profile、API 根地址、协议与账户作用域，并保留已知文件元数据，包括到期信息；转换本身不请求网络，也不证明文件仍然可用。

```rust,no_run
use lingxi_llm_client::{files::ProviderFileRef, protocol::{LlmError, ProviderProfile}, xai_collections::XaiUploadedFile};
fn file_reference(uploaded: &XaiUploadedFile, profile: &ProviderProfile, account: &str) -> Result<ProviderFileRef, LlmError> {
    uploaded.to_provider_file_ref(profile, account)
}
```

把返回的引用传给配置相同 profile 和账户作用域的 `FileService`。移除集合成员关系与删除底层全局文件是两个独立的显式操作。
