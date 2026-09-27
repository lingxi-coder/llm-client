# Qwen 百炼 Knowledge Search

[English](qwen-knowledge-search.en.md)

`QwenKnowledgeService::knowledge_search` 调用阿里云 Model Studio 已发布的原生 Knowledge Search 服务。它使用专门的 `/api/v1/indices/knowledge/search` 路由，与低层单知识库 `retrieve` 和宿主侧 RAG 编排分开。路由、字段、过滤规则和响应格式依据阿里云官方 [Knowledge Search API](https://help.aliyun.com/en/model-studio/knowledgesearch) 文档。

调用者需先在控制台创建并发布 Knowledge Retrieval service，再将其 `agent_id` 绑定为 `QwenKnowledgeSearchRef`。客户端不会创建、配置或发布 Agent，也不会接管多知识库路由、权重、重排等策略；这些策略由已发布服务保存。`QwenKnowledgeSearchRef` 绑定 provider、profile、account scope、region、workspace 和 endpoint。

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

此请求使用北京 workspace 的 `POST /api/v1/indices/knowledge/search`。它只发送已发布服务 ID、可选 `agent_version`、查询文本和图片 URL，以及可选的 per-Knowledge-Base `kb_search_configs`。策略覆写字段不在 API 请求类型中。

`query` 与 `images` 至少提供一个。文本和图文检索使用 `QwenKnowledgeSearchRequest::new(..., query)`，可选地再调用 `with_images`。仅图片检索用 `images_only`；它会按文档要求发送空字符串 `query`。图片 URL 必须是带 host 的绝对 HTTP(S) URL，并能被 Model Studio 公开访问；客户端只检查 URL 结构，不请求 URL、解析 DNS 或推断 IP 是否可公开访问，也不上传图片。

过滤配置必须通过 `QwenKnowledgeSearchKbConfig` 传入 `QwenKnowledgeRef`。服务在发送请求前检查所有引用属于当前连接，并拒绝重复 Knowledge Base ID。每个过滤项必须是 JSON object，其他字段、运算符和元数据键保持原样。总 `search_filters` JSON 字节数最多 80,000。文档只对 Unstructured KB 说明 `tags` 数组最多 1,000 项；如果调用者明确知道 KB 类型，可用 `QwenKnowledgeSearchKbConfig::for_unstructured` 启用本地预检。普通 `new` 不推断 KB 类型，把该类型相关校验留给 provider。过滤 ID 还必须已绑定到已发布服务，客户端无法读取该服务的后台配置，因此该成员关系由 provider 检查。运行时过滤会追加到服务的离线过滤条件，按 AND 合并。传入过滤配置不会把检索范围限制为这些 KB；已发布服务仍按自己的配置检索其他绑定 KB。

响应保留整个 envelope、每个 node 和完整 `metadata`，包括尚未建模的 provider 字段。若 metadata 返回 `workspace_id`，客户端会验证它与当前 workspace 一致；若返回 `pipeline_id`，客户端会将其绑定为当前 workspace 的 `QwenKnowledgeRef`。搜索结果可来自 Agent 绑定的任一 KB，不要求只属于请求过滤配置中的 KB。

Knowledge Search 默认限额为每用户 25 QPS。服务每次只发送一次请求，不自动重试，也不创建或发布服务。测试使用 mock transport，不调用真实阿里云 API。
