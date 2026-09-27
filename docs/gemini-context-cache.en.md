# Gemini Developer API: explicit context cache

`gemini_context_cache` implements the Gemini Developer API `cachedContents` resource lifecycle: create, paginated list, get, expiration update, and delete. It is independent of chat routing and requires the full collection URL explicitly:

```text
https://generativelanguage.googleapis.com/v1beta/cachedContents
```

The service never derives its cache URL from a chat profile's `base_url`. `GeminiContextCacheScope` binds a profile name, a caller-defined non-secret account scope, and this URL. Each returned `GeminiContextCacheRef` carries the same binding; it cannot be reused after switching account, profile, or endpoint.

```rust,no_run
use lingxi_llm_client::{
    gemini_context_cache::{
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
// GenerateContent request through the caller's normal chat/client flow.
let _cache_resource_name = cached.reference.resource_name();
# Ok(())
# }
```

Pass the API key on each call. The service sends it in the `x-goog-api-key` header and does not store it or put it in the URL. Credential storage, refresh, and caller account identity remain with the host application.

Create requests preserve Gemini REST JSON for `contents`, `systemInstruction`, `tools`, and `toolConfig`. These native values use `serde_json::Value`; the service checks their outer object/array shape and requires the system instruction to be a `Content` object with text-only parts, but it does not duplicate Gemini's other nested field validation. `model` must be a `models/{model}` resource name, and `displayName` is limited to 128 Unicode characters. `ttl` and `expireTime` are mutually exclusive expiration forms. Cached contents, tools, and system instructions are immutable after creation; PATCH changes expiration only.

Each `list` call fetches one page. `pageSize` is limited to 1000, and callers pass the returned `nextPageToken` unchanged on the next call. Get/list return cache metadata; they do not return the original cached input. A cache can be referenced by the `cachedContent` field on a later Gemini `generateContent` request.

Create, expiration update, and delete are each sent once with no automatic retry. If transport fails after dispatch, the service marks the outcome as unknown. The caller can reconcile with one get using the scoped reference or an explicit list, then decide what to do next.

References: [Gemini Caching REST API](https://ai.google.dev/api/caching), [Gemini context caching guide](https://ai.google.dev/gemini-api/docs/caching), and [Gemini API authentication](https://ai.google.dev/api).
