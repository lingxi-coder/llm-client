# GLM managed knowledge bases

[简体中文](glm-knowledge.md)

`client.glm_knowledge()` exposes Zhipu's native personal Knowledge Base API separately from `client.retrieval()`, which uses OpenAI Vector Stores. It supports paginated knowledge-base listing, create, detail, partial update and delete; paginated document listing, URL import and streamed multipart file upload; and retrieval. The routes follow Zhipu's current [knowledge-base list](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E7%9F%A5%E8%AF%86%E5%BA%93%E5%88%97%E8%A1%A8), [knowledge-base detail](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E7%9F%A5%E8%AF%86%E5%BA%93%E8%AF%A6%E6%83%85), [edit knowledge base](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E7%BC%96%E8%BE%91%E7%9F%A5%E8%AF%86%E5%BA%93), [delete knowledge base](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E5%88%A0%E9%99%A4%E7%9F%A5%E8%AF%86%E5%BA%93), [document list](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E6%96%87%E6%A1%A3%E5%88%97%E8%A1%A8), [URL document upload](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E4%B8%8A%E4%BC%A0url%E6%96%87%E6%A1%A3), [file document upload](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E4%B8%8A%E4%BC%A0%E6%96%87%E4%BB%B6%E6%96%87%E6%A1%A3), and [knowledge retrieval](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E7%9F%A5%E8%AF%86%E5%BA%93%E6%A3%80%E7%B4%A2) references.

Configure the independent service root on a Zhipu/GLM profile. Pass the API key and a stable, non-secret account identity on each request; the client does not look up or store credentials.

```toml
glm_knowledge = { mode = "enabled", value = {
  endpoint = "https://open.bigmodel.cn/api/llm-application/open",
  auth = { type = "bearer" }
} }
```

Knowledge-base creation requires a documented Zhipu Embedding ID (3, 11, or 12). URL ingestion submits provider indexing work; a successful submission does not mean indexing has finished. The host can supply callback settings and remains responsible for checking later indexing state. Changing the bound embedding model may require rebuilding the knowledge base; configure callbacks and follow-up work according to Zhipu's documentation.

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

Knowledge-base and document references bind the provider, profile, endpoint fingerprint, and `account_scope`. Detail, update, delete, document listing, URL import and file upload check this scope before sending requests. List methods use one-based `page` and `size`; document listing also accepts `word` to search document names. File uploads use repeated multipart `files` parts and can include `knowledge_type`, `custom_separator`, `sentence_size`, `parse_image`, `callback_url`, `callback_header`, `word_num_limit`, and `request_id` (sent as `req_id`). The client accepts 1–100 files per call, streams each one once, checks the declared byte count, and does not impose a provider file-size limit that the current API page does not publish. `custom_separator` and `sentence_size` are accepted only with custom slicing (`knowledge_type = 5`).

The file-upload result retains accepted and rejected rows separately. Unrecognized success rows remain in `unresolved`; any submitted file absent from both response arrays appears in `missing_files`. A successful upload response does not mean document parsing or indexing has finished; use provider callbacks or your own follow-up checks. The client does not poll or retry uploads. If transport fails after dispatch, HTTP 408/5xx is returned as an unknown outcome, and the host should inspect provider state before resubmitting. Callback URLs and header values are redacted from request `Debug` output.

The current official API reference exposes document listing and URL/file upload, with no per-document detail, update, or delete route. It describes document-level parsing options and per-file success/failure arrays; see [Upload file document](https://docs.bigmodel.cn/api-reference/%E7%9F%A5%E8%AF%86%E5%BA%93-api/%E4%B8%8A%E4%BC%A0%E6%96%87%E4%BB%B6%E6%96%87%E6%A1%A3).

`GlmKnowledgeRef` and `GlmKnowledgeDocumentRef` carry scoped identities instead of raw IDs; mismatched providers, profiles, endpoints, or accounts are rejected before HTTP. This module is GLM-specific and does not adapt OpenAI Vector Store requests or promise cross-provider compatibility.

Retrieval supports the documented embedding, keyword, and mixed strategies, optional reranking, and score thresholds. Queries are limited to 1,000 characters, `top_k` to 1–20, `top_n` to 1–100, and the mixed-recall ratio to 1–99. Responses preserve provider-native data and expose typed text, score, and source metadata. The client does not generate an answer or poll indexing. If a transport error interrupts a state-changing request, the service returns `OutcomeUnknown`; the host should check provider state before deciding to resubmit. Contract tests use a mock transport, with no live provider calls.
