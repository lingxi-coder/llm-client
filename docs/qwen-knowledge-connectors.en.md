# Qwen Model Studio Data Connectors and OSS Imports

[中文](qwen-knowledge-connectors.md)

`QwenKnowledgeService` can create and query Model Studio `FILE` data connectors and batch-import files from an already-authorized OSS bucket by explicit object key. Routes, fields, and limits follow Alibaba Cloud's official [RAG API overview](https://help.aliyun.com/en/model-studio/rag-api-overview), [Create Connector](https://help.aliyun.com/en/model-studio/rag-api-add-connector), [Get Connector](https://help.aliyun.com/en/model-studio/rag-api-get-connector), and [Import from OSS](https://help.aliyun.com/en/model-studio/rag-api-oss-import) documentation.

All calls use the API key, Beijing workspace endpoint, and profile/account scope already held by `QwenKnowledgeService`. A `QwenKnowledgeConnectorRef` binds the provider, profile, account scope, region, workspace, and endpoint. A reference from another connection is rejected before HTTP dispatch.

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

## Data connectors

Create calls `POST /api/v1/connector/dash/addConnector`. The API currently supports only `connectorType: FILE`. The request requires a `connectorName` from 1 to 20 characters, a `description` from 1 to 200 characters, and `fileConnectorConfig.storeType`. `PLATFORM` selects platform-managed storage. `CUSTOM` requires the caller's OSS `regionId` and `bucketName`. `QwenKnowledgeConnectorCreateRequest::new` selects `PLATFORM`; call `with_custom_oss` for `CUSTOM`.

Get calls `POST /api/v1/connector/dash/getConnector`. Supply a scoped connector ID, a name, or both. The provider requires at least one selector and gives `connectorId` precedence if both are present. Names are limited to 20 characters. The current RAG catalog documents create and get only, so this module does not add connector list, update, or delete calls.

## Import from OSS

Import calls `POST /api/v1/connector/dash/addFilesFromAuthorizedOss`. Before the first call, authorize Model Studio's OSS service-linked role; the official endpoint guide says the RAM console should show `AliyunServiceRoleForBailian`. The target bucket also needs the `bailian-datahub-access=read` tag, as described in Alibaba Cloud's [File connector guide](https://help.aliyun.com/en/model-studio/connector/file). A request includes a category ID, bucket, region, and 1–10 explicit object keys. `fileName` must contain 1–500 characters and `ossKey` 1–256 characters. Optional tags allow at most 10 entries. `overWriteFileByOssKey` is omitted unless explicitly set and defaults to false on the provider.

Use `QwenKnowledgeOssImportRequest::for_category` with a `QwenKnowledgeCategoryRef` to keep the category's profile and account scope bound through dispatch. `new` is also available when the caller intentionally supplies a raw category ID; the service checks that this required ID is nonempty, but has no source reference to compare.

The API documents these optional parsers: `AUTO_SELECT`, `DOCMIND`, `DOCMIND_DIGITAL`, `DOCMIND_LLM_VERSION`, `DASH_QWEN_VL_PARSER`, and `DOCMIND_LLM_VERSION_MEDIA`. `DASH_QWEN_VL_PARSER` requires `parserConfig` with the fixed model name `qwen3-vl-plus` and a prompt from 1 to 1,500 characters. For `SESSION_FILE`, Model Studio uses its default parser and does not allow parser overrides. The service preflights only these documented limits and required fields.

The client sends one request and does not retry an import after a lost transport response; it returns `QwenKnowledgeError::OutcomeUnknown`. It does not inspect or change OSS permissions, list buckets, upload local files, or poll import progress. The OSS import copies files into Model Studio's data center. The documented success example returns `data: {}`, while the response-field table also mentions `data.fileIds`; `imported_files` is therefore optional, and the client infers no job ID or completion state. When `fileIds` are present, the client returns scope-bound `QwenKnowledgeFileRef` values. It preserves the full provider envelope in `native`.

Mock contract tests cover the documented payloads, scope binding, preflight limits, missing optional import IDs, malformed responses, and unknown mutation outcomes. No live Alibaba Cloud requests are made.
