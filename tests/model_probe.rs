use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{
    directory::probe::*, protocol::LlmError, HttpRequest, StreamResponse, Transport,
};
use std::sync::Mutex;
struct Pages {
    seen: Mutex<Vec<HttpRequest>>,
    pages: Mutex<std::collections::VecDeque<serde_json::Value>>,
}
#[async_trait]
impl Transport for Pages {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.seen.lock().unwrap().push(request);
        let body = self
            .pages
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected repeated request");
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: futures::stream::once(async move { Ok(body.to_string().into()) }).boxed(),
        })
    }
}
fn pages(values: Vec<serde_json::Value>) -> Pages {
    Pages {
        seen: Mutex::new(vec![]),
        pages: Mutex::new(values.into()),
    }
}
fn credential(bearer: bool) -> ProbeCredential {
    ProbeCredential {
        token: "secret".to_string().into(),
        bearer,
        account_id: None,
        fedramp: false,
    }
}
#[tokio::test]
async fn directory_authentication_and_pagination_live_in_sdk() {
    let http = pages(vec![
        serde_json::json!({"data":[{"id":"a"}],"has_more":true,"last_id":"a"}),
        serde_json::json!({"data":[{"id":"b"}],"has_more":false}),
    ]);
    let result = probe(
        &http,
        "https://example.test/v1",
        ProbeProtocol::Anthropic,
        &credential(true),
        std::time::Duration::from_secs(2),
    )
    .await
    .unwrap();
    assert_eq!(result.model_ids.unwrap(), vec!["a", "b"]);
    let seen = http.seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert!(seen[0].url.starts_with("https://example.test/v1/models?"));
    assert!(seen[1].url.contains("after_id=a"));
    assert!(seen[0]
        .headers
        .iter()
        .any(|(k, v)| k.eq_ignore_ascii_case("authorization") && v == "Bearer secret"));
    assert!(seen[0]
        .headers
        .iter()
        .any(|(k, v)| k == "anthropic-beta" && v.contains("oauth-")));
}
#[tokio::test]
async fn validates_base_before_sending_and_does_not_leak_userinfo() {
    let http = pages(vec![]);
    let err = probe(
        &http,
        "https://secret@example.test",
        ProbeProtocol::OpenAi,
        &credential(false),
        std::time::Duration::from_secs(1),
    )
    .await
    .unwrap_err();
    assert!(!err.to_string().contains("secret"));
    assert!(http.seen.lock().unwrap().is_empty());
}
#[tokio::test]
async fn chatgpt_directory_uses_account_identity_and_slug() {
    let http = pages(vec![serde_json::json!({"models":[{"slug":"gpt-model"}]})]);
    let mut cred = credential(true);
    cred.account_id = Some("account".into());
    cred.fedramp = true;
    let result = probe(
        &http,
        "https://example.test/backend-api",
        ProbeProtocol::ChatGpt,
        &cred,
        std::time::Duration::from_secs(1),
    )
    .await
    .unwrap();
    assert_eq!(result.model_ids.unwrap(), vec!["gpt-model"]);
    assert!(http.seen.lock().unwrap()[0]
        .headers
        .iter()
        .any(|(k, v)| k == "ChatGPT-Account-ID" && v == "account"));
}
#[tokio::test]
async fn repeated_pagination_cursor_is_not_followed_forever() {
    let body = serde_json::json!({"data":[{"id":"a"}],"has_more":true,"last_id":"a"});
    let http = pages(vec![body.clone(), body]);
    assert!(probe(
        &http,
        "https://example.test",
        ProbeProtocol::Anthropic,
        &credential(false),
        std::time::Duration::from_secs(1)
    )
    .await
    .is_err());
    assert_eq!(http.seen.lock().unwrap().len(), 2);
}
