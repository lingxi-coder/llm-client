# Qwen Model Studio Knowledge Search

[中文](qwen-knowledge-search.md)

`QwenKnowledgeService::knowledge_search` calls Alibaba Cloud Model Studio's native, published Knowledge Search service. It uses the dedicated `/api/v1/indices/knowledge/search` route, separate from lower-level single-knowledge-base `retrieve` and host-side RAG orchestration. Routes, fields, filter rules, and response shape follow Alibaba Cloud's official [Knowledge Search API](https://help.aliyun.com/en/model-studio/knowledgesearch) documentation.

The caller must create and publish a Knowledge Retrieval service in the console, then bind its `agent_id` as a `QwenKnowledgeSearchRef`. The client does not create, configure, or publish an Agent and does not take over multi-KB routing, weights, reranking, or other service strategies. Those settings remain in the published service. `QwenKnowledgeSearchRef` binds the provider, profile, account scope, region, workspace, and endpoint.

```rust,no_run
use lingxi_llm_client::qwen_knowledge::{
    QwenKnowledgeError, QwenKnowledgeRef, QwenKnowledgeSearchKbConfig,
    QwenKnowledgeSearchRequest, QwenKnowledgeService,
};
use serde_json::json;

async fn search(
    service: &QwenKnowledgeService<'_>,
    knowledge: &QwenKnowledgeRef,
) -> Result<(), QwenKnowledgeError> {
    let published_service = service.scope().knowledge_search_ref("aid-published-123")?;
    let request = QwenKnowledgeSearchRequest::new(
        published_service,
        "Find the product warranty period.",
    )
    .with_kb_search_config(
        QwenKnowledgeSearchKbConfig::new(knowledge.clone()).with_search_filters([
            json!({"doc_id":["warranty-a","warranty-b"]}),
            json!({"tags":["warranty"]}),
            json!({"price":{"gte":100,"lte":500}}),
        ]),
    );

    let result = service.knowledge_search(&request).await?;
    for node in result.nodes {
        println!("score={:?} text={:?} metadata={}", node.score, node.text, node.metadata);
    }
    Ok(())
}
```

The request uses the Beijing workspace `POST /api/v1/indices/knowledge/search` endpoint. It sends only the published service ID, optional `agent_version`, query text and image URLs, and optional per-Knowledge-Base `kb_search_configs`. Strategy override fields are not part of the request type.

At least one of `query` and `images` is required. Use `QwenKnowledgeSearchRequest::new(..., query)` for text or multimodal search, optionally followed by `with_images`. For image-only search, use `images_only`; it sends the empty `query` string required by the docs. Each image URL must be an absolute HTTP(S) URL with a host and be publicly accessible to Model Studio. The client checks only URL structure: it does not request the URL, resolve DNS, infer whether an IP address is public, or upload the image.

Filter configurations take a `QwenKnowledgeRef` through `QwenKnowledgeSearchKbConfig`. Before dispatch, the service verifies that every reference belongs to the current connection and rejects duplicate Knowledge Base IDs. Each filter item must be a JSON object; other fields, operators, and metadata keys pass through unchanged. Total serialized `search_filters` data is limited to 80,000 bytes. The docs cap `tags` arrays at 1,000 items for Unstructured Knowledge Bases only. If the caller knows a KB's type, `QwenKnowledgeSearchKbConfig::for_unstructured` enables this local preflight. The regular `new` constructor does not infer the KB type and leaves that type-specific validation to Model Studio. Filter IDs must already be bound to the published service. The client cannot inspect that private service config, so Model Studio validates membership. Runtime filters are appended to the service's offline filters and combined with AND. Passing a filter config does not restrict retrieval to only those Knowledge Bases: the published service may search its other bound Knowledge Bases too.

The result preserves the complete response envelope, each node, and its complete `metadata`, including provider fields not yet modeled. When metadata includes `workspace_id`, the client verifies it matches the current workspace. When it includes `pipeline_id`, the client binds that ID to a `QwenKnowledgeRef` in the current workspace. Results may come from any KB bound to the Agent; they do not have to belong to the request's filter configs.

Knowledge Search has a default limit of 25 QPS per user. The service sends each request once, does not retry automatically, and does not create or publish a service. Tests use a mock transport and make no live Alibaba Cloud calls.
