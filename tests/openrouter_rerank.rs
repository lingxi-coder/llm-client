use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::openrouter::rerank::{
        OpenRouterRerankDispatch, OpenRouterRerankError, OpenRouterRerankRequest,
        OpenRouterRerankScope, OpenRouterRerankService, OPENROUTER_RERANK_ENDPOINT,
    },
    HttpRequest, RequestOptions, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

struct Reply {
    status: u16,
    body: Vec<u8>,
}

#[derive(Debug, Clone)]
struct Sent {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

struct MockTransport {
    replies: Mutex<VecDeque<Result<Reply, LlmError>>>,
    sent: Mutex<Vec<Sent>>,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Result<Reply, LlmError>>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            sent: Mutex::new(Vec::new()),
        }
    }

    fn sent(&self) -> Vec<Sent> {
        self.sent.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(Sent {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body.to_vec(),
        });
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected OpenRouter rerank request")?;
        Ok(StreamResponse {
            status: reply.status,
            headers: Vec::new(),
            body: stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
        })
    }
}

fn scope() -> OpenRouterRerankScope {
    OpenRouterRerankScope::new("openrouter-prod", "account-17", OPENROUTER_RERANK_ENDPOINT).unwrap()
}

fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("openrouter-account-key".to_owned())),
        account_scope: Some("account-17".to_owned()),
        ..Default::default()
    }
}

fn reply(status: u16, value: Value) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body: serde_json::to_vec(&value).unwrap(),
    })
}

fn success_response() -> Value {
    json!({
        "model": "cohere/rerank-v3.5",
        "results": [
            {
                "document": {"text": "Paris is the capital of France."},
                "index": 0,
                "relevance_score": 0.98
            },
            {
                "document": {"text": "Berlin is the capital of Germany."},
                "index": 1,
                "relevance_score": 0.12
            }
        ],
        "id": "gen-rerank-1234567890-abc",
        "provider": "Cohere",
        "usage": {"search_units": 1, "total_tokens": 150}
    })
}

#[tokio::test]
async fn sends_documented_route_and_preserves_ranked_indices_scores_and_native_data() {
    let transport = MockTransport::new([reply(200, success_response())]);
    let service = OpenRouterRerankService::new(&transport, scope()).unwrap();
    let request = OpenRouterRerankRequest::new(
        "cohere/rerank-v3.5",
        "What is the capital of France?",
        [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
        ],
    )
    .with_top_n(1);

    let result = service.rerank(&request, &options()).await.unwrap();
    assert_eq!(result.id.as_deref(), Some("gen-rerank-1234567890-abc"));
    assert_eq!(result.model.as_deref(), Some("cohere/rerank-v3.5"));
    assert_eq!(result.provider.as_deref(), Some("Cohere"));
    assert_eq!(result.results.len(), 2);
    assert_eq!(result.results[0].index, 0);
    assert_eq!(result.results[0].relevance_score, 0.98);
    assert_eq!(
        result.results[0].document.as_ref().unwrap()["text"],
        "Paris is the capital of France."
    );
    let usage = result.usage.as_ref().unwrap();
    assert_eq!(usage.search_units, Some(1));
    assert_eq!(usage.total_tokens, Some(150));
    assert_eq!(result.scope().endpoint(), OPENROUTER_RERANK_ENDPOINT);
    assert_eq!(result.native["usage"]["total_tokens"], 150);

    let sent = transport.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].method, "POST");
    assert_eq!(sent[0].url, OPENROUTER_RERANK_ENDPOINT);
    assert!(sent[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("authorization") && value == "Bearer openrouter-account-key"
    }));
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["model"], "cohere/rerank-v3.5");
    assert_eq!(body["query"], "What is the capital of France?");
    assert_eq!(body["documents"].as_array().unwrap().len(), 2);
    assert_eq!(body["top_n"], 1);
}

#[tokio::test]
async fn validates_request_and_endpoint_before_sending() {
    let transport = MockTransport::new([]);
    let service = OpenRouterRerankService::new(&transport, scope()).unwrap();
    let cases = [
        OpenRouterRerankRequest::new(" ", "query", ["document"]),
        OpenRouterRerankRequest::new("model", " ", ["document"]),
        OpenRouterRerankRequest::new("model", "query", std::iter::empty::<String>()),
        OpenRouterRerankRequest::new("model", "query", [" "]),
        OpenRouterRerankRequest::new("model", "query", ["document"]).with_top_n(0),
    ];
    for request in cases {
        let error = service.rerank(&request, &options()).await.unwrap_err();
        assert_eq!(error.dispatch(), OpenRouterRerankDispatch::NotSent);
        assert!(matches!(error, OpenRouterRerankError::InvalidInput(_)));
    }
    assert!(matches!(
        OpenRouterRerankScope::new(
            "profile",
            "account",
            "https://attacker.example/api/v1/rerank"
        ),
        Err(OpenRouterRerankError::InvalidInput(_))
    ));
    assert!(transport.sent().is_empty());
}

#[tokio::test]
async fn exact_account_scope_and_key_are_required() {
    let transport = MockTransport::new([]);
    let service = OpenRouterRerankService::new(&transport, scope()).unwrap();
    let request = OpenRouterRerankRequest::new("cohere/rerank-v3.5", "query", ["document"]);

    let wrong_account = RequestOptions {
        account_scope: Some("another-account".into()),
        ..options()
    };
    assert!(matches!(
        service.rerank(&request, &wrong_account).await,
        Err(OpenRouterRerankError::ScopeMismatch)
    ));
    let missing_key = RequestOptions {
        credential: None,
        account_scope: Some("account-17".into()),
        ..Default::default()
    };
    let error = service.rerank(&request, &missing_key).await.unwrap_err();
    assert_eq!(error.dispatch(), OpenRouterRerankDispatch::NotSent);
    assert!(matches!(
        error,
        OpenRouterRerankError::Llm(LlmError::Authentication { .. })
    ));
    assert!(transport.sent().is_empty());
}

#[tokio::test]
async fn transport_failure_is_unknown_and_never_retried() {
    let transport = MockTransport::new([Err(LlmError::Transport {
        message: "connection lost after request send".into(),
    })]);
    let service = OpenRouterRerankService::new(&transport, scope()).unwrap();
    let error = service
        .rerank(
            &OpenRouterRerankRequest::new("cohere/rerank-v3.5", "query", ["document"]),
            &options(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), OpenRouterRerankDispatch::Unknown);
    assert_eq!(transport.sent().len(), 1);
}

#[tokio::test]
async fn distinguishes_http_rejection_from_malformed_success_response() {
    let rejected_transport = MockTransport::new([reply(
        401,
        json!({"error":{"code":401,"message":"Invalid API key"}}),
    )]);
    let rejected_service = OpenRouterRerankService::new(&rejected_transport, scope()).unwrap();
    let rejected = rejected_service
        .rerank(
            &OpenRouterRerankRequest::new("cohere/rerank-v3.5", "query", ["document"]),
            &options(),
        )
        .await
        .unwrap_err();
    assert_eq!(rejected.dispatch(), OpenRouterRerankDispatch::Rejected);
    assert!(matches!(
        rejected,
        OpenRouterRerankError::Rejected { status: 401, .. }
    ));

    let malformed_transport = MockTransport::new([reply(
        200,
        json!({"id":"accepted-id","results":[{"index":4,"relevance_score":0.9}]}),
    )]);
    let malformed_service = OpenRouterRerankService::new(&malformed_transport, scope()).unwrap();
    let malformed = malformed_service
        .rerank(
            &OpenRouterRerankRequest::new("cohere/rerank-v3.5", "query", ["document"]),
            &options(),
        )
        .await
        .unwrap_err();
    assert_eq!(malformed.dispatch(), OpenRouterRerankDispatch::Accepted);
    assert!(matches!(
        malformed,
        OpenRouterRerankError::AcceptedInvalidResponse {
            id: Some(ref id),
            ..
        } if id == "accepted-id"
    ));
}
