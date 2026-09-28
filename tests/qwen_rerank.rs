use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::qwen::rerank::{
        QwenRerankDispatchOutcome, QwenRerankError, QwenRerankRegion, QwenRerankRequest,
        QwenRerankScope, QwenRerankService,
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
            .expect("unexpected Qwen rerank request")?;
        Ok(StreamResponse {
            status: reply.status,
            headers: Vec::new(),
            body: stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
        })
    }
}

fn scope() -> QwenRerankScope {
    QwenRerankScope::new(
        "qwen-beijing",
        "account-42",
        QwenRerankRegion::Beijing,
        "workspace-abc",
    )
    .unwrap()
}

fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("beijing-key".to_owned())),
        account_scope: Some("account-42".to_owned()),
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
        "output": {
            "results": [
                {"index": 2, "relevance_score": 0.93},
                {"index": 0, "relevance_score": 0.42}
            ]
        },
        "usage": {"prompt_tokens": 41, "total_tokens": 41},
        "request_id": "rerank-request-1"
    })
}

#[tokio::test]
async fn sends_official_native_contract_and_preserves_original_indices_and_scores() {
    let transport = MockTransport::new([reply(200, success_response())]);
    let service = QwenRerankService::new(&transport, scope()).unwrap();
    let request = QwenRerankRequest::new(
        "What is the return window?",
        [
            "Returns are accepted within 30 days.",
            "Our office is in London.",
            "You can return products for a refund within thirty days.",
        ],
    )
    .with_top_n(2)
    .with_instruction("Retrieve passages that answer the query.");

    let result = service.rerank(&request, &options()).await.unwrap();
    assert_eq!(result.request_id.as_deref(), Some("rerank-request-1"));
    assert_eq!(result.scope().workspace_id(), "workspace-abc");
    assert_eq!(result.results[0].index, 2);
    assert_eq!(result.results[0].relevance_score, 0.93);
    assert_eq!(result.results[1].index, 0);
    assert_eq!(result.usage.as_ref().unwrap().total_tokens, Some(41));

    let sent = transport.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].method, "POST");
    assert_eq!(
        sent[0].url,
        "https://workspace-abc.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    );
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer beijing-key"));
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["model"], "qwen3.7-text-rerank");
    assert_eq!(body["input"]["query"], "What is the return window?");
    assert_eq!(body["input"]["documents"].as_array().unwrap().len(), 3);
    assert_eq!(body["parameters"]["top_n"], 2);
    assert_eq!(
        body["parameters"]["instruct"],
        "Retrieve passages that answer the query."
    );
}

#[tokio::test]
async fn validates_inputs_and_exact_account_scope_before_sending() {
    let transport = MockTransport::new([]);
    let service = QwenRerankService::new(&transport, scope()).unwrap();

    let cases = [
        QwenRerankRequest::new(" ", ["document"]),
        QwenRerankRequest::new("query", std::iter::empty::<String>()),
        QwenRerankRequest::new("query", [" "]),
        QwenRerankRequest::new("query", ["document"]).with_top_n(0),
        QwenRerankRequest::new("query", ["document"]).with_instruction(" "),
    ];
    for request in cases {
        let error = service.rerank(&request, &options()).await.unwrap_err();
        assert_eq!(error.dispatch_outcome(), QwenRerankDispatchOutcome::NotSent);
        assert!(matches!(error, QwenRerankError::InvalidInput(_)));
    }

    let too_many = QwenRerankRequest::new("query", std::iter::repeat_n("doc", 501));
    assert!(matches!(
        service.rerank(&too_many, &options()).await,
        Err(QwenRerankError::InvalidInput(_))
    ));

    let wrong_account = RequestOptions {
        account_scope: Some("another-account".into()),
        ..options()
    };
    assert!(matches!(
        service
            .rerank(
                &QwenRerankRequest::new("query", ["document"]),
                &wrong_account
            )
            .await,
        Err(QwenRerankError::ScopeMismatch)
    ));
    assert!(transport.sent().is_empty());
}

#[tokio::test]
async fn missing_key_and_account_identity_fail_before_http() {
    let transport = MockTransport::new([]);
    let service = QwenRerankService::new(&transport, scope()).unwrap();
    let request = QwenRerankRequest::new("query", ["document"]);

    assert!(matches!(
        service.rerank(&request, &RequestOptions::default()).await,
        Err(QwenRerankError::ScopeMismatch)
    ));
    let no_key = RequestOptions {
        account_scope: Some("account-42".into()),
        ..Default::default()
    };
    let error = service.rerank(&request, &no_key).await.unwrap_err();
    assert_eq!(error.dispatch_outcome(), QwenRerankDispatchOutcome::NotSent);
    assert!(matches!(
        error,
        QwenRerankError::Llm(LlmError::Authentication { .. })
    ));
    assert!(transport.sent().is_empty());
}

#[tokio::test]
async fn network_failure_is_unknown_and_never_retried() {
    let transport = MockTransport::new([Err(LlmError::Transport {
        message: "connection dropped after request send".into(),
    })]);
    let service = QwenRerankService::new(&transport, scope()).unwrap();
    let error = service
        .rerank(&QwenRerankRequest::new("query", ["document"]), &options())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch_outcome(), QwenRerankDispatchOutcome::Unknown);
    assert_eq!(transport.sent().len(), 1);
}

#[tokio::test]
async fn distinguishes_rejection_from_accepted_malformed_result() {
    let rejected_transport = MockTransport::new([reply(
        400,
        json!({"code":"InvalidParameter","message":"query too long","request_id":"bad-1"}),
    )]);
    let rejected_service = QwenRerankService::new(&rejected_transport, scope()).unwrap();
    let rejected = rejected_service
        .rerank(&QwenRerankRequest::new("query", ["document"]), &options())
        .await
        .unwrap_err();
    assert_eq!(
        rejected.dispatch_outcome(),
        QwenRerankDispatchOutcome::Rejected
    );
    assert!(matches!(
        rejected,
        QwenRerankError::Rejected { status: 400, .. }
    ));

    let malformed_transport = MockTransport::new([reply(
        200,
        json!({
            "request_id":"accepted-1",
            "output":{"results":[{"index":5,"relevance_score":0.2}]}
        }),
    )]);
    let malformed_service = QwenRerankService::new(&malformed_transport, scope()).unwrap();
    let malformed = malformed_service
        .rerank(&QwenRerankRequest::new("query", ["document"]), &options())
        .await
        .unwrap_err();
    assert_eq!(
        malformed.dispatch_outcome(),
        QwenRerankDispatchOutcome::Accepted
    );
    assert!(matches!(
        malformed,
        QwenRerankError::AcceptedInvalidResponse {
            request_id: Some(ref id),
            ..
        } if id == "accepted-1"
    ));
}

#[tokio::test]
async fn validates_scope_workspace_dns_label() {
    assert!(matches!(
        QwenRerankScope::new(
            "qwen",
            "account",
            QwenRerankRegion::Beijing,
            "workspace/path"
        ),
        Err(QwenRerankError::InvalidInput(_))
    ));
}
