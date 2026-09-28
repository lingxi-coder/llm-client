# Gemini Developer API：显式上下文缓存

`gemini_context_cache` 提供 Google Gemini Developer API 的 `cachedContents` 资源生命周期：创建、分页列出、读取、更新过期时间和删除。它独立于聊天路由，必须显式提供完整集合 URL：

```text
https://generativelanguage.googleapis.com/v1beta/cachedContents
```

服务不会从聊天 profile 的 `base_url` 推导缓存 URL。`GeminiContextCacheScope` 将 profile 名、调用方定义的非秘密 account scope 和这个 URL 绑定起来。返回的 `GeminiContextCacheRef` 也绑定同一范围；换账号、profile 或 endpoint 后，不能复用该引用。

```rust,no_run
use lingxi_llm_client::{
    providers::google::context_cache::{
        GeminiContextCacheCreateRequest, GeminiContextCacheExpiration,
        GeminiContextCacheScope, GeminiContextCacheService,
    },
    protocol::Secret,
    transport::HttpTransport,
};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let transport = HttpTransport::new()?;
let scope = GeminiContextCacheScope::new(
    "gemini-production",
    "customer-account-42",
    "https://generativelanguage.googleapis.com/v1beta/cachedContents",
)?;
let service = GeminiContextCacheService::new(&transport, scope)?;
let api_key = Secret::new("caller-managed-key".to_owned());

let request = GeminiContextCacheCreateRequest::new("models/gemini-2.5-flash")?
    .with_display_name("Support handbook")?
    .with_contents(vec![serde_json::json!({
        "role": "user",
        "parts": [{"text": "Long-lived source material"}]
    })])?
    .with_expiration(GeminiContextCacheExpiration::Ttl("3600s".into()))?;
let cached = service.create(&request, &api_key).await?;

// Use cached.reference.resource_name() as `cachedContent` on a later
// GenerateContent request, through the caller's normal chat/client flow.
let _cache_resource_name = cached.reference.resource_name();
# Ok(())
# }
```

API key 在每次调用时传入，并通过 `x-goog-api-key` header 发送；它不会保存在 service 或 URL 中。令牌和密钥存储、刷新以及调用方账户身份均由宿主负责。

创建请求保留 Gemini REST 的原生 JSON：`contents`、`systemInstruction`、`tools` 和 `toolConfig`。这些输入结构以 `serde_json::Value` 提供；服务检查对象/数组外形，并要求 system instruction 是仅含文本 part 的 `Content`，但不代替 Gemini 校验其他嵌套字段。`model` 必须使用 `models/{model}` 资源名；`displayName` 最多 128 个 Unicode 字符。`ttl` 和 `expireTime` 是互斥的过期方式。显式缓存的 `contents`、工具和系统指令在创建后不可修改；PATCH 只改变过期时间。

`list` 每次只取一页，最大 `pageSize` 为 1000，后续调用应原样传入 `nextPageToken`。get/list 返回缓存元数据，不会重新提供创建时的缓存内容。缓存内容可在后续 Gemini `generateContent` 请求中通过 `cachedContent` 字段引用。

创建、过期更新和删除各发送一次，不进行自动重试。若传输在请求发送后失败，错误会标记结果为未知；调用方可用 scoped 引用执行一次 get，或显式 list 来核对资源状态，再决定下一步。

参考：[Gemini Caching REST API](https://ai.google.dev/api/caching)、[Gemini context caching guide](https://ai.google.dev/gemini-api/docs/caching)、[Gemini API authentication](https://ai.google.dev/api)。
