use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    gemini_context_cache::*,
    protocol::{LlmError, Secret},
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

const COLLECTION: &str = "https://generativelanguage.googleapis.com/v1beta/cachedContents";

struct Reply {
    method: &'static str,
    url: &'static str,
    status: u16,
    body: Value,
}

struct Mock {
    replies: Mutex<VecDeque<Reply>>,
    sent: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected HTTP call");
        assert_eq!(request.method, reply.method);
        assert_eq!(request.url, reply.url);
        self.sent.lock().unwrap().push(request);
        let bytes = serde_json::to_vec(&reply.body).unwrap();
        Ok(StreamResponse {
            status: reply.status,
            headers: vec![("x-goog-request-id".into(), "ctx-cache-req".into())],
            body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
        })
    }
}

fn reply(method: &'static str, url: &'static str, body: Value) -> Reply {
    Reply {
        method,
        url,
        status: 200,
        body,
    }
}

fn cache_resource(id: &str) -> Value {
    json!({
        "name": format!("cachedContents/{id}"),
        "model": "models/gemini-2.5-flash",
        "displayName": "Support handbook",
        "createTime": "2026-09-25T10:00:00Z",
        "expireTime": "2026-09-25T11:00:00Z",
        "usageMetadata": {"totalTokenCount": 8192}
    })
}

fn scope(account: &str) -> GeminiContextCacheScope {
    GeminiContextCacheScope::new("gemini-prod", account, COLLECTION).unwrap()
}

fn credential(value: &str) -> Secret<String> {
    Secret::new(value.to_owned())
}

#[tokio::test]
async fn create_list_get_patch_delete_use_exact_cached_content_wire() {
    let mock = Mock {
        replies: Mutex::new(
            vec![
                reply("POST", COLLECTION, cache_resource("cache-a")),
                reply(
                    "GET",
                    "https://generativelanguage.googleapis.com/v1beta/cachedContents?pageSize=2&pageToken=next+token",
                    json!({
                        "cachedContents": [cache_resource("cache-a")],
                        "nextPageToken": "next page"
                    }),
                ),
                reply(
                    "GET",
                    "https://generativelanguage.googleapis.com/v1beta/cachedContents/cache-a",
                    cache_resource("cache-a"),
                ),
                reply(
                    "PATCH",
                    "https://generativelanguage.googleapis.com/v1beta/cachedContents/cache-a?updateMask=ttl",
                    cache_resource("cache-a"),
                ),
                reply(
                    "DELETE",
                    "https://generativelanguage.googleapis.com/v1beta/cachedContents/cache-a",
                    json!({}),
                ),
            ]
            .into(),
        ),
        sent: Mutex::new(Vec::new()),
    };
    let service = GeminiContextCacheService::new(&mock, scope("account-a")).unwrap();
    let api_key = credential("key-one");
    let create = GeminiContextCacheCreateRequest::new("models/gemini-2.5-flash")
        .unwrap()
        .with_display_name("Support handbook")
        .unwrap()
        .with_contents(vec![json!({
            "role": "user",
            "parts": [{"text": "Cached source material"}]
        })])
        .unwrap()
        .with_system_instruction(json!({
            "parts": [{"text": "Answer using the supplied material."}]
        }))
        .unwrap()
        .with_tools(vec![json!({
            "functionDeclarations": [{"name": "lookup", "description": "Look up a record"}]
        })])
        .unwrap()
        .with_tool_config(json!({"functionCallingConfig": {"mode": "AUTO"}}))
        .unwrap()
        .with_expiration(GeminiContextCacheExpiration::Ttl("3600s".into()))
        .unwrap();

    let created = service.create(&create, &api_key).await.unwrap();
    assert_eq!(created.reference.resource_name(), "cachedContents/cache-a");
    assert_eq!(created.model, "models/gemini-2.5-flash");
    assert_eq!(created.display_name.as_deref(), Some("Support handbook"));
    assert_eq!(created.expire_time.as_deref(), Some("2026-09-25T11:00:00Z"));

    let options = GeminiContextCacheListOptions::new()
        .with_page_size(2)
        .unwrap()
        .with_page_token("next token")
        .unwrap();
    let page = service.list(&options, &api_key).await.unwrap();
    assert_eq!(page.items.len(), 1);
    assert_eq!(page.next_page_token.as_deref(), Some("next page"));

    let fetched = service.get(&created.reference, &api_key).await.unwrap();
    assert_eq!(fetched.reference, created.reference);
    let updated = service
        .update_expiration(
            &created.reference,
            GeminiContextCacheExpiration::Ttl("7200s".into()),
            &api_key,
        )
        .await
        .unwrap();
    assert_eq!(updated.reference, created.reference);
    let rotated_key = credential("key-two");
    service
        .delete(&created.reference, &rotated_key)
        .await
        .unwrap();

    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent.len(), 5);
    assert_eq!(
        sent[0].headers[0],
        ("x-goog-api-key".into(), "key-one".into())
    );
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "models/gemini-2.5-flash",
            "displayName": "Support handbook",
            "contents": [{"role":"user","parts":[{"text":"Cached source material"}]}],
            "systemInstruction": {"parts":[{"text":"Answer using the supplied material."}]},
            "tools": [{"functionDeclarations":[{"name":"lookup","description":"Look up a record"}]}],
            "toolConfig": {"functionCallingConfig":{"mode":"AUTO"}},
            "ttl": "3600s"
        })
    );
    let patch_body: Value = serde_json::from_slice(&sent[3].body).unwrap();
    assert_eq!(patch_body, json!({"ttl": "7200s"}));
    assert!(sent[1].body.is_empty());
    assert!(sent[2].body.is_empty());
    assert!(sent[4].body.is_empty());
    assert_eq!(
        sent[4].headers[0],
        ("x-goog-api-key".into(), "key-two".into())
    );
}

#[tokio::test]
async fn update_expire_time_uses_its_documented_union_member_and_mask() {
    let mock = Mock {
        replies: Mutex::new(
            vec![reply(
                "PATCH",
                "https://generativelanguage.googleapis.com/v1beta/cachedContents/cache-a?updateMask=expireTime",
                cache_resource("cache-a"),
            )]
            .into(),
        ),
        sent: Mutex::new(Vec::new()),
    };
    let service = GeminiContextCacheService::new(&mock, scope("account-a")).unwrap();
    let reference =
        GeminiContextCacheRef::from_resource_name(&scope("account-a"), "cachedContents/cache-a")
            .unwrap();
    let updated = service
        .update_expiration(
            &reference,
            GeminiContextCacheExpiration::ExpireTime("2030-05-01T12:00:00Z".into()),
            &credential("key-one"),
        )
        .await
        .unwrap();
    assert_eq!(updated.reference, reference);
    let sent = mock.sent.lock().unwrap();
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body, json!({"expireTime": "2030-05-01T12:00:00Z"}));
}

#[tokio::test]
async fn references_and_inputs_are_preflighted_without_network_calls() {
    let mock = Mock {
        replies: Mutex::new(VecDeque::new()),
        sent: Mutex::new(Vec::new()),
    };
    let service = GeminiContextCacheService::new(&mock, scope("account-a")).unwrap();
    let api_key = credential("key-one");

    let foreign =
        GeminiContextCacheRef::from_resource_name(&scope("account-b"), "cachedContents/cache-a")
            .unwrap();
    assert!(matches!(
        service.get(&foreign, &api_key).await,
        Err(GeminiContextCacheError::Llm(
            LlmError::PermissionDenied { .. }
        ))
    ));
    assert!(
        GeminiContextCacheRef::from_resource_name(&scope("account-a"), "cachedContents/a/b")
            .is_err()
    );
    assert!(GeminiContextCacheScope::new(
        "gemini-prod",
        "account-a",
        "https://example.test/v1beta/cachedContents"
    )
    .is_err());
    assert!(GeminiContextCacheCreateRequest::new("gemini-2.5-flash").is_err());
    assert!(
        GeminiContextCacheCreateRequest::new("models/gemini-2.5-flash")
            .unwrap()
            .with_display_name("x".repeat(129))
            .is_err()
    );
    assert!(
        GeminiContextCacheCreateRequest::new("models/gemini-2.5-flash")
            .unwrap()
            .with_expiration(GeminiContextCacheExpiration::Ttl("1.1234567890s".into()))
            .is_err()
    );
    assert!(GeminiContextCacheListOptions::new()
        .with_page_size(1001)
        .is_err());
    assert!(mock.sent.lock().unwrap().is_empty());
}

struct FailingTransport {
    calls: Mutex<usize>,
}

#[async_trait]
impl Transport for FailingTransport {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        *self.calls.lock().unwrap() += 1;
        Err(LlmError::Transport {
            message: "connection was interrupted".into(),
        })
    }
}

#[tokio::test]
async fn uncertain_create_is_reported_and_never_retried() {
    let transport = FailingTransport {
        calls: Mutex::new(0),
    };
    let service = GeminiContextCacheService::new(&transport, scope("account-a")).unwrap();
    let request = GeminiContextCacheCreateRequest::new("models/gemini-2.5-flash").unwrap();
    let result = service.create(&request, &credential("key-one")).await;
    assert!(matches!(
        result,
        Err(GeminiContextCacheError::OutcomeUnknown {
            operation: "create",
            ..
        })
    ));
    assert_eq!(*transport.calls.lock().unwrap(), 1);
}
