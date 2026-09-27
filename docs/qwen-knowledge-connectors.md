# Qwen 百炼数据连接器与 OSS 导入

[English](qwen-knowledge-connectors.en.md)

`QwenKnowledgeService` 可创建和查询 Model Studio `FILE` 数据连接器，也可按明确指定的对象 key，从已授权 OSS bucket 批量导入文件。路由、字段和长度限制依据阿里云官方的 [RAG API 总览](https://help.aliyun.com/en/model-studio/rag-api-overview)、[Create Connector](https://help.aliyun.com/en/model-studio/rag-api-add-connector)、[Get Connector](https://help.aliyun.com/en/model-studio/rag-api-get-connector) 和 [Import from OSS](https://help.aliyun.com/en/model-studio/rag-api-oss-import) 文档。

所有请求都使用 `QwenKnowledgeService` 当前的 API key、北京 workspace 地址和 profile/account scope。创建或查询得到的 `QwenKnowledgeConnectorRef` 绑定 provider、profile、account scope、region、workspace 和 endpoint；跨连接的引用会在发送 HTTP 请求前被拒绝。

```rust,no_run
use lingxi_llm_client::qwen_knowledge::{
    QwenKnowledgeConnectorCreateRequest, QwenKnowledgeConnectorLookup,
    QwenKnowledgeCategoryRef, QwenKnowledgeError, QwenKnowledgeOssImportFile,
    QwenKnowledgeOssImportRequest, QwenKnowledgeService,
};

async fn use_connectors(
    service: &QwenKnowledgeService<'_>,
    category: &QwenKnowledgeCategoryRef,
) -> Result<(), QwenKnowledgeError> {
    let created = service
        .create_connector(&QwenKnowledgeConnectorCreateRequest::new(
            "product docs",
            "Connector for product documentation",
        ))
        .await?;
    let details = service
        .get_connector(&QwenKnowledgeConnectorLookup::by_id(
            created.reference.clone(),
        ))
        .await?;

    let request = QwenKnowledgeOssImportRequest::for_category(
        category.clone(),
        "my-docs-bucket",
        "cn-beijing",
        [
            QwenKnowledgeOssImportFile::new(
                "product-guide.pdf",
                "docs/product-guide.pdf",
            ),
            QwenKnowledgeOssImportFile::new("faq.docx", "docs/faq.docx"),
        ],
    );
    let imported = service.import_files_from_oss(&request).await?;
    let _ = (details, imported);
    Ok(())
}
```

## 数据连接器

创建调用 `POST /api/v1/connector/dash/addConnector`。当前 API 只支持 `connectorType: FILE`。请求必须包含 1–20 个字符的 `connectorName`、1–200 个字符的 `description` 和 `fileConnectorConfig.storeType`。`PLATFORM` 使用平台托管存储；`CUSTOM` 必须同时提供用户 OSS 的 `regionId` 和 `bucketName`。`QwenKnowledgeConnectorCreateRequest::new` 默认选择 `PLATFORM`，调用 `with_custom_oss` 可选择 `CUSTOM`。

查询调用 `POST /api/v1/connector/dash/getConnector`。可以传入已绑定作用域的 connector ID、名称，或同时传入二者。provider 要求至少一个查询条件；如果两者都传，优先使用 `connectorId`。名称最多 20 个字符。当前 RAG 接口目录只列出创建和查询，本模块不推测连接器的列表、更新或删除路由。

## 从 OSS 导入

导入调用 `POST /api/v1/connector/dash/addFilesFromAuthorizedOss`。首次调用前，必须按官方指南授权 Model Studio 的 OSS 服务关联角色；RAM 控制台中应能看到 `AliyunServiceRoleForBailian`。目标 bucket 还需要带有 `bailian-datahub-access=read` 标签，详见阿里云[文件连接器指南](https://help.aliyun.com/en/model-studio/connector/file)。请求包含 category ID、bucket、region 和 1–10 个明确的对象 key。`fileName` 必须为 1–500 个字符，`ossKey` 必须为 1–256 个字符。可选 tags 最多 10 个。只有显式选择后才发送 `overWriteFileByOssKey`；provider 默认值为 `false`。

使用 `QwenKnowledgeOssImportRequest::for_category` 和 `QwenKnowledgeCategoryRef`，可在请求发送时继续校验 category 的 profile 和 account scope。调用者也可以主动传入原始 category ID 并使用 `new`；此时服务只检查必填 ID 非空，没有来源引用可供比较。

API 还文档化了以下可选解析器：`AUTO_SELECT`、`DOCMIND`、`DOCMIND_DIGITAL`、`DOCMIND_LLM_VERSION`、`DASH_QWEN_VL_PARSER` 和 `DOCMIND_LLM_VERSION_MEDIA`。`DASH_QWEN_VL_PARSER` 必须配套 `parserConfig`，模型名固定为 `qwen3-vl-plus`，提示词长度为 1–1500 个字符。对于 `SESSION_FILE`，Model Studio 使用默认解析器，不允许覆盖解析设置。服务只预检文档明确规定的必填项和长度限制。

客户端每次只发送一个导入请求。若传输响应丢失，不会自动重试，而是返回 `QwenKnowledgeError::OutcomeUnknown`。客户端不会检查或修改 OSS 权限、列举 bucket、上传本地文件，也不会轮询导入进度。官方成功响应示例返回 `data: {}`，而字段表又提到 `data.fileIds`；因此 `imported_files` 是可选值，客户端不会据此推断 job ID 或完成状态。如果响应提供 `fileIds`，客户端会将它们转换为绑定当前 scope 的 `QwenKnowledgeFileRef`。完整的 provider envelope 保存在 `native`。

mock 合同测试覆盖文档请求字段、引用作用域、输入限制、缺省的导入文件 ID、格式错误响应和状态未知的写入结果。测试不调用真实阿里云 API。
