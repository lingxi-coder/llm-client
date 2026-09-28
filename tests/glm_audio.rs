use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::zhipu::audio::{
        GlmAsrDispatch, GlmAsrError, GlmAsrRequest, GlmAsrRoute, GlmAsrScope, GlmAsrService,
    },
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::sync::Mutex;

enum Reply {
    Response { status: u16, body: Value },
    Error(LlmError),
}

struct Mock {
    reply: Mutex<Option<Reply>>,
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        match self.reply.lock().unwrap().take().expect("unexpected retry") {
            Reply::Error(error) => Err(error),
            Reply::Response { status, body } => {
                let body = serde_json::to_vec(&body).unwrap();
                Ok(StreamResponse {
                    status,
                    headers: vec![("x-request-id".into(), "glm-asr-trace-1".into())],
                    body: futures::stream::once(async move { Ok(Bytes::from(body)) }).boxed(),
                })
            }
        }
    }
}

fn setup(reply: Reply) -> (GlmAsrService<'static>, &'static Mock) {
    let transport: &'static Mock = Box::leak(Box::new(Mock {
        reply: Mutex::new(Some(reply)),
        requests: Mutex::new(Vec::new()),
    }));
    let route = GlmAsrRoute::new("http://127.0.0.1:8000/v1").unwrap();
    let scope = GlmAsrScope::new("glm-asr-local", "acct-local-1", &route).unwrap();
    let service = GlmAsrService::new(transport, route, scope).unwrap();
    (service, transport)
}

#[tokio::test]
async fn transcribes_using_the_documented_sglang_chat_request() {
    let (service, mock) = setup(Reply::Response {
        status: 200,
        body: json!({
            "id":"chatcmpl-glm-asr-1",
            "model":"glm-asr",
            "choices":[{"message":{"role":"assistant","content":"你好，欢迎。"}}],
            "usage":{"prompt_tokens":42,"completion_tokens":8,"total_tokens":50}
        }),
    });
    let result = service
        .transcribe(
            &GlmAsrRequest::new("example_zh.wav").unwrap(),
            &request_options(),
        )
        .await
        .unwrap();

    assert_eq!(result.text, "你好，欢迎。");
    assert_eq!(result.response_id.as_deref(), Some("chatcmpl-glm-asr-1"));
    assert_eq!(result.model.as_deref(), Some("glm-asr"));
    assert_eq!(result.request_id.as_deref(), Some("glm-asr-trace-1"));
    assert_eq!(result.scope.account_scope(), "acct-local-1");
    assert_eq!(result.native["usage"]["total_tokens"], 50);

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].url, "http://127.0.0.1:8000/v1/chat/completions");
    assert_eq!(
        requests[0]
            .headers
            .iter()
            .find(|(name, _)| name == "authorization")
            .unwrap()
            .1,
        "Bearer EMPTY"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({
            "model":"glm-asr",
            "messages":[{"role":"user","content":[
                {"type":"audio_url","audio_url":{"url":"example_zh.wav"}},
                {"type":"text","text":"Please transcribe this audio into text"}
            ]}],
            "max_tokens":1024
        })
    );
}

#[tokio::test]
async fn empty_or_control_character_audio_references_are_rejected_before_send() {
    let (service, mock) = setup(Reply::Response {
        status: 200,
        body: json!({"choices":[{"message":{"content":"should not run"}}]}),
    });
    for audio_url in [" ", "audio\nfile.wav"] {
        let request: GlmAsrRequest =
            serde_json::from_value(json!({"audio_url":audio_url})).unwrap();
        let error = service
            .transcribe(&request, &request_options())
            .await
            .unwrap_err();
        assert!(matches!(error, GlmAsrError::InvalidInput(_)));
    }
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn scope_endpoint_mismatch_is_rejected_at_construction() {
    let first_route = GlmAsrRoute::new("http://127.0.0.1:8000/v1").unwrap();
    let other_route = GlmAsrRoute::new("http://127.0.0.1:8001/v1").unwrap();
    let scope = GlmAsrScope::new("glm-asr-local", "acct-local-1", &first_route).unwrap();
    let transport: &'static Mock = Box::leak(Box::new(Mock {
        reply: Mutex::new(None),
        requests: Mutex::new(Vec::new()),
    }));

    let error = match GlmAsrService::new(transport, other_route, scope) {
        Ok(_) => panic!("endpoint mismatch must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        GlmAsrError::Llm(LlmError::PermissionDenied { .. })
    ));
}

#[test]
fn route_requires_a_documented_v1_root_and_secure_transport() {
    assert!(GlmAsrRoute::new("http://asr.example/v1").is_err());
    assert!(GlmAsrRoute::new("https://asr.example/chat/completions").is_err());
    assert!(GlmAsrRoute::new("https://user:pass@asr.example/v1").is_err());
    assert!(GlmAsrRoute::new("https://asr.example/v1?key=secret").is_err());
    assert!(GlmAsrRoute::new("http://127.0.0.1:8000/v1").is_ok());
    assert!(GlmAsrRoute::new("https://asr.example/v1").is_ok());
}

#[tokio::test]
async fn provider_rejection_and_uncertain_transport_failures_are_not_retried() {
    let (service, mock) = setup(Reply::Response {
        status: 401,
        body: json!({"error":{"message":"bad credential"}}),
    });
    let error = service
        .transcribe(
            &GlmAsrRequest::new("example.wav").unwrap(),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        GlmAsrError::Provider {
            status: 401,
            dispatch: GlmAsrDispatch::Rejected,
            ..
        }
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);

    let (service, mock) = setup(Reply::Error(LlmError::Transport {
        message: "connection closed".into(),
    }));
    let error = service
        .transcribe(
            &GlmAsrRequest::new("example.wav").unwrap(),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, GlmAsrError::OutcomeUnknown { .. }));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn success_without_transcript_is_marked_accepted_and_invalid() {
    let (service, mock) = setup(Reply::Response {
        status: 200,
        body: json!({"id":"accepted-but-empty","choices":[]}),
    });
    let error = service
        .transcribe(
            &GlmAsrRequest::new("example.wav").unwrap(),
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, GlmAsrError::InvalidResponse { .. }));
    assert_eq!(error.dispatch(), GlmAsrDispatch::Accepted);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

fn request_options() -> lingxi_llm_client::RequestOptions {
    lingxi_llm_client::RequestOptions {
        credential: Some(Secret::new("EMPTY".to_owned())),
        ..Default::default()
    }
}

#[tokio::test]
async fn credentials_are_required_per_operation_and_can_rotate() {
    let (service, mock) = setup(Reply::Response {
        status: 401,
        body: json!({"error":"rejected"}),
    });
    let request = GlmAsrRequest::new("sample.wav").unwrap();
    for credential in [None, Some(Secret::new(" ".into()))] {
        let options = lingxi_llm_client::RequestOptions {
            credential,
            ..Default::default()
        };
        assert!(service.transcribe(&request, &options).await.is_err());
    }
    assert!(mock.requests.lock().unwrap().is_empty());
    for key in ["first-key", "rotated-key"] {
        *mock.reply.lock().unwrap() = Some(Reply::Response {
            status: 401,
            body: json!({"error":"rejected"}),
        });
        let options = lingxi_llm_client::RequestOptions {
            credential: Some(Secret::new(key.into())),
            total_timeout: Some(std::time::Duration::from_secs(5)),
            ..Default::default()
        };
        assert!(service.transcribe(&request, &options).await.is_err());
    }
    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    for (request, key) in requests.iter().zip(["first-key", "rotated-key"]) {
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
                && value == &format!("Bearer {key}")));
        assert!(request.timeout.is_some_and(
            |timeout| !timeout.is_zero() && timeout <= std::time::Duration::from_secs(5)
        ));
    }
}
