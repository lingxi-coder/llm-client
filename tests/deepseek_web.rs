//! Direct Search/Fetch contracts; no external network or credentials.
use lingxi_llm_client::protocol::{LlmError, LlmErrorKind, ProviderProfile, Secret};
use lingxi_llm_client::providers::{deepseek::web::*, DeepSeekClient};
use lingxi_llm_client::{
    HttpRequest, HttpResponse, LlmClientBuilder, RequestOptions, StreamResponse, Transport,
};
use serde_json::json;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

struct Mock {
    requests: Mutex<Vec<HttpRequest>>,
    responses: Mutex<VecDeque<HttpResponse>>,
}
#[async_trait::async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        Ok(self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request")
            .into())
    }
}
fn setup(profile_name: &str, responses: Vec<HttpResponse>) -> (DeepSeekClient, Arc<Mock>) {
    let profile = lingxi_llm_client::builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == profile_name)
        .unwrap();
    from_profile(profile, responses)
}
fn from_profile(
    profile: ProviderProfile,
    responses: Vec<HttpResponse>,
) -> (DeepSeekClient, Arc<Mock>) {
    let mock = Arc::new(Mock {
        requests: Mutex::new(vec![]),
        responses: Mutex::new(responses.into()),
    });
    let name = profile.profile_name.clone();
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
    (client.provider::<DeepSeekClient>(&name).unwrap(), mock)
}
fn opts() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("test-key".into())),
        ..Default::default()
    }
}
fn response(status: u16, body: serde_json::Value) -> HttpResponse {
    HttpResponse {
        status,
        headers: vec![
            ("x-request-id".into(), "req-1".into()),
            ("retry-after".into(), "2".into()),
        ],
        body: body.to_string().into(),
    }
}
fn hit() -> serde_json::Value {
    json!({"rerank_score":2.4,"title":"新闻","url":"https://example.com/news","snippet":"短摘要","summary":"长摘要","content":"正文","site_name":null,"site_icon":null,"date":null,"authority":-1})
}
#[tokio::test]
async fn direct_apis_use_bearer_auth_on_each_chat_wire_and_preserve_fields() {
    for profile in ["deepseek", "deepseek-search", "deepseek-responses"] {
        let (client, mock) = setup(
            profile,
            vec![
                response(200, json!({"results":[hit()]})),
                response(200, json!({"title":"文章","content":"# 正文"})),
            ],
        );
        let mut request = DeepSeekSearchRequest::new("天气😀");
        request.count = 50;
        request.need_content = true;
        request.ori_question = Some("武汉天气？".into());
        request.localization = Some("127.0.0.1".parse().unwrap());
        let result = client.web().search(&request, &opts()).await.unwrap();
        assert_eq!(result.results[0].summary, "长摘要");
        assert_eq!(result.results[0].authority, -1);
        assert!(result.results[0].site_name.is_none());
        assert_eq!(result.billable_requests(), 1);
        assert_eq!(result.cost_usd(), 0.00075);
        let fetched = client
            .web()
            .fetch(
                &DeepSeekFetchRequest::new("https://example.com/article"),
                &opts(),
            )
            .await
            .unwrap();
        assert_eq!(fetched.content, "# 正文");
        assert_eq!(fetched.cost_usd(), 0.00075);
        let requests = mock.requests.lock().unwrap();
        for request in requests.iter() {
            assert_eq!(request.method, "POST");
            assert!(request
                .timeout
                .is_some_and(|timeout| timeout <= Duration::from_secs(60) && !timeout.is_zero()));
            assert!(request
                .headers
                .iter()
                .any(|(k, v)| k.eq_ignore_ascii_case("authorization") && v == "Bearer test-key"));
            assert!(!request
                .headers
                .iter()
                .any(|(k, _)| k.eq_ignore_ascii_case("x-api-key")));
        }
        assert_eq!(requests[0].url, "https://api.deepseek.com/v1/tools/search");
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["count"], 50);
        assert_eq!(body["need_content"], true);
        assert_eq!(body["localization"], "127.0.0.1");
        assert!(body.get("model").is_none());
        assert_eq!(requests[1].url, "https://api.deepseek.com/v1/tools/fetch");
    }
}
#[tokio::test]
async fn empty_results_and_content_are_successful_and_not_billed() {
    let (client, _) = setup(
        "deepseek",
        vec![
            response(200, json!({"results":[]})),
            response(200, json!({"title":"","content":""})),
        ],
    );
    let result = client
        .web()
        .search(&DeepSeekSearchRequest::new("none"), &opts())
        .await
        .unwrap();
    assert_eq!(result.billable_requests(), 0);
    assert_eq!(result.cost_usd(), 0.0);
    assert_eq!(
        client
            .web()
            .fetch(&DeepSeekFetchRequest::new("https://example.com"), &opts())
            .await
            .unwrap()
            .billable_requests(),
        0
    );
}
#[tokio::test]
async fn invalid_requests_fail_before_network() {
    let (client, mock) = setup("deepseek", vec![]);
    for query in ["".to_owned(), " \n".to_owned(), "😀".repeat(129)] {
        assert!(client
            .web()
            .search(&DeepSeekSearchRequest::new(query), &opts())
            .await
            .is_err());
    }
    for count in [0, 51] {
        let mut req = DeepSeekSearchRequest::new("q");
        req.count = count;
        assert!(client.web().search(&req, &opts()).await.is_err());
    }
    let mut req = DeepSeekSearchRequest::new("q");
    req.ori_question = Some(" ".into());
    assert!(client.web().search(&req, &opts()).await.is_err());
    for url in [
        "file:///tmp/a",
        "https://user:pass@example.com",
        "https://example.com/".to_owned().repeat(600).as_str(),
    ] {
        assert!(client
            .web()
            .fetch(&DeepSeekFetchRequest::new(url), &opts())
            .await
            .is_err());
    }
    assert!(mock.requests.lock().unwrap().is_empty());
    assert!(DeepSeekSearchRequest::new("😀".repeat(128))
        .validate()
        .is_ok());
}
#[test]
fn domain_operators_are_validated_and_query_limit_applies_after_encoding() {
    for query in ["", "   "] {
        assert!(DeepSeekSearchRequest::new(query)
            .with_domains(&["example.com".into()], &[])
            .is_err());
    }
    let req = DeepSeekSearchRequest::new("news")
        .with_domains(
            &["example.com".into(), "docs.example.com".into()],
            &["ads.example.com".into()],
        )
        .unwrap();
    assert_eq!(
        req.query,
        "news site:example.com OR site:docs.example.com -site:ads.example.com"
    );
    for domain in [
        "example.com OR evil.test",
        "https://example.com",
        "example.com/path",
        "-example.com",
    ] {
        assert!(DeepSeekSearchRequest::new("news")
            .with_domains(&[domain.into()], &[])
            .is_err());
    }
    assert!(DeepSeekSearchRequest::new("a".repeat(128))
        .with_domains(&["example.com".into()], &[])
        .is_err());
}
#[tokio::test]
async fn custom_origins_and_missing_credentials_are_not_dispatched() {
    for base in [
        "https://gateway.example/v1",
        "https://api.deepseek.com/other",
        "http://api.deepseek.com",
        "https://user@api.deepseek.com",
        "https://api.deepseek.com?token=a",
    ] {
        let mut profile = lingxi_llm_client::builtin_providers()
            .unwrap()
            .into_iter()
            .find(|p| p.profile_name == "deepseek")
            .unwrap();
        profile.base_url = base.into();
        let (client, mock) = from_profile(profile, vec![]);
        assert!(client
            .web()
            .search(&DeepSeekSearchRequest::new("q"), &opts())
            .await
            .is_err());
        assert!(mock.requests.lock().unwrap().is_empty());
    }
    let (client, mock) = setup("deepseek", vec![]);
    assert!(client
        .web()
        .search(&DeepSeekSearchRequest::new("q"), &Default::default())
        .await
        .is_err());
    assert!(mock.requests.lock().unwrap().is_empty());
}
#[tokio::test]
async fn provider_errors_keep_status_retry_after_and_do_not_retry() {
    for (status, kind) in [
        (400, LlmErrorKind::InvalidRequest),
        (401, LlmErrorKind::Authentication),
        (402, LlmErrorKind::QuotaExceeded),
        (403, LlmErrorKind::PermissionDenied),
        (404, LlmErrorKind::InvalidRequest),
        (408, LlmErrorKind::ProviderTimeout),
        (413, LlmErrorKind::RequestTooLarge),
        (504, LlmErrorKind::ProviderTimeout),
        (429, LlmErrorKind::RateLimited),
        (500, LlmErrorKind::ProviderInternal),
    ] {
        let (client, mock) = setup(
            "deepseek",
            vec![response(
                status,
                json!({"error":{"message":"failure","type":"error","param":null,"code":"error"}}),
            )],
        );
        let error = client
            .web()
            .search(&DeepSeekSearchRequest::new("q"), &opts())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), kind, "status {status}");
        assert!(
            matches!(error.classification_source(), LlmError::ProviderResponse {status:s, retry_after:Some(t),request_id:Some(_),..} if *s==status && *t==Duration::from_secs(2))
        );
        assert_eq!(mock.requests.lock().unwrap().len(), 1);
    }
}
#[tokio::test]
async fn malformed_success_and_expired_deadlines_fail() {
    let (client, mock) = setup(
        "deepseek",
        vec![response(200, json!({"results":[{"title":"incomplete"}]}))],
    );
    assert!(client
        .web()
        .search(&DeepSeekSearchRequest::new("q"), &opts())
        .await
        .is_err());
    let options = RequestOptions {
        total_timeout: Some(Duration::ZERO),
        ..opts()
    };
    assert_eq!(
        client
            .web()
            .fetch(&DeepSeekFetchRequest::new("https://example.com"), &options)
            .await
            .unwrap_err()
            .kind(),
        LlmErrorKind::TransportTimeout
    );
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

struct PendingTransport {
    body_only: bool,
}
#[async_trait::async_trait]
impl Transport for PendingTransport {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        use futures::StreamExt;
        if !self.body_only {
            return futures::future::pending().await;
        }
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: futures::stream::pending().boxed(),
        })
    }
}
#[tokio::test]
async fn deadlines_bound_dispatch_and_response_reads() {
    for body_only in [false, true] {
        let profile = lingxi_llm_client::builtin_providers()
            .unwrap()
            .into_iter()
            .find(|p| p.profile_name == "deepseek")
            .unwrap();
        let client =
            LlmClientBuilder::with_transport(Arc::new(PendingTransport { body_only }), &[profile])
                .with_region(lingxi_llm_client::protocol::Region::International)
                .build()
                .unwrap();
        let provider = client.provider::<DeepSeekClient>("deepseek").unwrap();
        let options = RequestOptions {
            total_timeout: Some(Duration::from_millis(5)),
            ..opts()
        };
        let error = provider
            .web()
            .search(&DeepSeekSearchRequest::new("q"), &options)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), LlmErrorKind::TransportTimeout);
    }
}
