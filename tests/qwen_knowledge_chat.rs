use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::qwen::knowledge::{
        QwenKnowledgeChatContentPart, QwenKnowledgeChatError, QwenKnowledgeChatMessage,
        QwenKnowledgeChatRequest, QwenKnowledgeChatToolCall, QwenKnowledgeDispatch,
        QwenKnowledgeRegion, QwenKnowledgeScope, QwenKnowledgeService,
    },
    transport::{HttpRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

enum Reply {
    Stream {
        status: u16,
        headers: Vec<(String, String)>,
        chunks: Vec<Result<Bytes, LlmError>>,
    },
    Failure,
    DropTracked(Arc<AtomicBool>),
}

struct DropGuard(Arc<AtomicBool>);

impl Drop for DropGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct Mock {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl Mock {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<HttpRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        match self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected Knowledge Chat request")
        {
            Reply::Stream {
                status,
                headers,
                chunks,
            } => Ok(StreamResponse {
                status,
                headers,
                body: stream::iter(chunks).boxed(),
            }),
            Reply::Failure => Err(LlmError::TransportTimeout {
                message: "mock chat timeout".into(),
            }),
            Reply::DropTracked(flag) => {
                let body = stream::unfold(DropGuard(flag), |guard| async move {
                    let _guard = guard;
                    None::<(Result<Bytes, LlmError>, DropGuard)>
                })
                .boxed();
                Ok(StreamResponse {
                    status: 200,
                    headers: vec![("content-type".into(), "text/event-stream".into())],
                    body,
                })
            }
        }
    }
}

fn scope(account: &str) -> QwenKnowledgeScope {
    QwenKnowledgeScope::new(
        "qwen-main",
        account,
        QwenKnowledgeRegion::Beijing,
        "llm-chat-workspace",
    )
    .unwrap()
}

fn make_service<'a>(mock: &'a Mock, account: &str) -> QwenKnowledgeService<'a> {
    QwenKnowledgeService::new(mock, scope(account)).unwrap()
}

fn chat_ref(
    scope: &QwenKnowledgeScope,
) -> lingxi_llm_client::providers::qwen::knowledge::QwenKnowledgeChatRef {
    scope.knowledge_chat_ref("aid-published-chat-1").unwrap()
}

fn sse(value: Value) -> Vec<u8> {
    [
        b"data: ".as_slice(),
        serde_json::to_string(&value).unwrap().as_bytes(),
        b"\n\n",
    ]
    .concat()
}

fn successful_frame(step: &str, step_change: &str) -> Value {
    json!({
        "code":"200",
        "message":"Success",
        "request_id":"chat-request-1",
        "output":{
            "choices":[{
                "message":{
                    "role":"assistant",
                    "content":"partial",
                    "extra":{"group":"planning","step":step,"step_change":step_change},
                    "response_metadata":{"native_extension":{"kept":true}}
                },
                "finish_reason":""
            }]
        },
        "future_field":{"kept":true}
    })
}

fn final_frame() -> Value {
    json!({
        "code":"200",
        "message":"Success",
        "request_id":"chat-request-1",
        "output":{"choices":[{
            "message":{
                "role":"assistant",
                "content":"answer",
                "extra":{"group":"generating","step":"generating","step_change":"generation_end"}
            },
            "finish_reason":"stop"
        }]},
        "usage":{"input_tokens":10,"output_tokens":3,"future_usage":7}
    })
}

fn stream_reply(chunks: impl IntoIterator<Item = Vec<u8>>) -> Reply {
    Reply::Stream {
        status: 200,
        headers: vec![
            (
                "content-type".into(),
                "text/event-stream; charset=utf-8".into(),
            ),
            ("x-request-id".into(), "header-chat-request".into()),
        ],
        chunks: chunks
            .into_iter()
            .map(|chunk| Ok(Bytes::from(chunk)))
            .collect(),
    }
}

#[tokio::test]
async fn sends_official_native_wire_with_full_tool_history_images_session_files_and_cache_flag() {
    let mock = Mock::new([stream_reply([sse(final_frame())])]);
    let service = make_service(&mock, "account-1");
    let session_file = service
        .scope()
        .file_ref("session_file_registered_1")
        .unwrap();
    let request = QwenKnowledgeChatRequest::new(
        chat_ref(service.scope()),
        [
            QwenKnowledgeChatMessage::user_text("Find the answer in this picture."),
            QwenKnowledgeChatMessage::assistant_with_tool_calls(
                "",
                [QwenKnowledgeChatToolCall::new(
                    "call-1",
                    "semantic_search",
                    r#"{"query":"answer","target_ids":["kb-1"]}"#,
                )
                .with_index(0)],
            ),
            QwenKnowledgeChatMessage::tool_return("call-1", "Retrieved two relevant passages."),
            QwenKnowledgeChatMessage::assistant_text("Prior answer."),
            QwenKnowledgeChatMessage::user_parts([
                QwenKnowledgeChatContentPart::text("What is shown?"),
                QwenKnowledgeChatContentPart::image_url(
                    "https://images.example.test/evidence.jpg?revision=4",
                ),
            ]),
        ],
    )
    .with_request_id("caller-request-1")
    .with_session_files([session_file])
    .with_cache_control(true);

    let mut output = service
        .knowledge_chat(&request, &request_options())
        .await
        .unwrap();
    let final_event = output.next_event().await.unwrap().unwrap();
    assert!(final_event.is_complete());
    assert_eq!(final_event.native()["usage"]["future_usage"], 7);
    assert!(output.next_event().await.unwrap().is_none());
    assert_eq!(output.request_id(), Some("chat-request-1"));

    let requests = mock.requests();
    assert_eq!(requests.len(), 1, "a chat turn is sent exactly once");
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://llm-chat-workspace.cn-beijing.maas.aliyuncs.com/api/v2/apps/knowledge/chat"
    );
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer qwen-chat-api-key"));
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "accept" && value == "text/event-stream"));
    assert!(requests[0]
        .timeout
        .is_some_and(|timeout| !timeout.is_zero() && timeout <= Duration::from_secs(120)));
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(body["input"]["request_id"], "caller-request-1");
    assert_eq!(body["input"]["messages"].as_array().unwrap().len(), 5);
    assert_eq!(
        body["input"]["messages"][1]["tool_calls"][0]["type"],
        "function"
    );
    assert_eq!(body["input"]["messages"][1]["tool_calls"][0]["index"], 0);
    assert_eq!(
        body["input"]["messages"][1]["tool_calls"][0]["function"]["arguments"],
        r#"{"query":"answer","target_ids":["kb-1"]}"#
    );
    assert_eq!(body["input"]["messages"][2]["tool_call_id"], "call-1");
    assert_eq!(
        body["input"]["messages"][4]["content"][1]["type"],
        "image_url"
    );
    assert_eq!(
        body["input"]["messages"][4]["content"][1]["image_url"]["url"],
        "https://images.example.test/evidence.jpg?revision=4"
    );
    assert_eq!(
        body["parameters"]["agent_options"]["agent_id"],
        "aid-published-chat-1"
    );
    assert_eq!(
        body["parameters"]["agent_options"]["session_files"],
        json!(["session_file_registered_1"])
    );
    assert_eq!(
        body["parameters"]["agent_options"]["enable_cache_control"],
        true
    );
    assert!(body.get("session_id").is_none());
}

#[tokio::test]
async fn intermediate_native_phases_are_preserved_and_stop_waits_for_clean_eof() {
    let mut frame = sse(successful_frame("planning", "plan_start"));
    frame.extend(sse(final_frame()));
    let bytes = frame;
    let mock = Mock::new([stream_reply(bytes.into_iter().map(|byte| vec![byte]))]);
    let service = make_service(&mock, "account-1");
    let mut output = service
        .knowledge_chat(
            &QwenKnowledgeChatRequest::new(
                chat_ref(service.scope()),
                [QwenKnowledgeChatMessage::user_text("Question")],
            ),
            &request_options(),
        )
        .await
        .unwrap();

    let first = output.next_event().await.unwrap().unwrap();
    assert!(!first.is_complete());
    assert_eq!(first.step(), Some("planning"));
    assert_eq!(first.step_change(), Some("plan_start"));
    assert_eq!(first.native()["future_field"]["kept"], true);
    let final_event = output.next_event().await.unwrap().unwrap();
    assert!(final_event.is_complete());
    assert_eq!(
        final_event.native()["output"]["choices"][0]["finish_reason"],
        "stop"
    );
    assert!(output.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn provider_error_after_stop_prevents_false_completion_and_keeps_native_error() {
    let mut bytes = sse(final_frame());
    bytes.extend(b"event: error\ndata: {\"code\":\"AgentApp.NotFound\",\"message\":\"not published\",\"request_id\":\"err-id\"}\n\n");
    let mock = Mock::new([stream_reply([bytes])]);
    let service = make_service(&mock, "account-1");
    let mut output = service
        .knowledge_chat(
            &QwenKnowledgeChatRequest::new(
                chat_ref(service.scope()),
                [QwenKnowledgeChatMessage::user_text("Question")],
            ),
            &request_options(),
        )
        .await
        .unwrap();
    let error = output.next_event().await.expect_err("expected an error");
    match error {
        QwenKnowledgeChatError::Provider {
            dispatch: QwenKnowledgeDispatch::Accepted,
            error,
            ..
        } => {
            assert_eq!(error.code.as_deref(), Some("AgentApp.NotFound"));
            assert_eq!(error.request_id.as_deref(), Some("err-id"));
            assert_eq!(error.native["message"], "not published");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn malformed_success_envelopes_with_stop_never_complete() {
    for malformed in [
        {
            let mut value = final_frame();
            value["code"] = json!(200);
            value
        },
        {
            let mut value = final_frame();
            value["status_code"] = json!("200");
            value
        },
    ] {
        let mock = Mock::new([stream_reply([sse(malformed)])]);
        let service = make_service(&mock, "account-1");
        let mut output = service
            .knowledge_chat(
                &QwenKnowledgeChatRequest::new(
                    chat_ref(service.scope()),
                    [QwenKnowledgeChatMessage::user_text("Question")],
                ),
                &request_options(),
            )
            .await
            .unwrap();
        assert!(matches!(
            output.next_event().await,
            Err(QwenKnowledgeChatError::InvalidResponse { .. })
        ));
    }
}

#[tokio::test]
async fn non_sse_success_response_retains_native_body_in_error() {
    let mock = Mock::new([Reply::Stream {
        status: 200,
        headers: vec![(
            "content-type".into(),
            "application/json; charset=utf-8".into(),
        )],
        chunks: vec![Ok(Bytes::from_static(
            br#"{"code":"Unexpected","message":"wrong response mode","request_id":"bad-mode"}"#,
        ))],
    }]);
    let service = make_service(&mock, "account-1");
    let error = service
        .knowledge_chat(
            &QwenKnowledgeChatRequest::new(
                chat_ref(service.scope()),
                [QwenKnowledgeChatMessage::user_text("Question")],
            ),
            &request_options(),
        )
        .await
        .err()
        .expect("expected an error");
    match error {
        QwenKnowledgeChatError::InvalidResponse {
            request_id, native, ..
        } => {
            assert_eq!(request_id.as_deref(), Some("bad-mode"));
            assert_eq!(native["message"], "wrong response mode");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn eof_without_stop_and_body_interruption_are_unknown_not_success() {
    let mock = Mock::new([
        stream_reply([sse(successful_frame("generating", "generation_start"))]),
        Reply::Stream {
            status: 200,
            headers: vec![("content-type".into(), "text/event-stream".into())],
            chunks: vec![
                Ok(Bytes::from(sse(successful_frame(
                    "generating",
                    "generation_start",
                )))),
                Err(LlmError::StreamInterrupted {
                    message: "mock late disconnect".into(),
                }),
            ],
        },
    ]);
    let service = make_service(&mock, "account-1");
    let request = || {
        QwenKnowledgeChatRequest::new(
            chat_ref(service.scope()),
            [QwenKnowledgeChatMessage::user_text("Question")],
        )
    };

    let mut clean_eof = service
        .knowledge_chat(&request(), &request_options())
        .await
        .unwrap();
    let _ = clean_eof.next_event().await.unwrap().unwrap();
    let error = clean_eof.next_event().await.expect_err("expected an error");
    assert!(matches!(
        &error,
        QwenKnowledgeChatError::StreamInterrupted { .. }
    ));
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Unknown);

    let mut interrupted = service
        .knowledge_chat(&request(), &request_options())
        .await
        .unwrap();
    let _ = interrupted.next_event().await.unwrap().unwrap();
    let error = interrupted
        .next_event()
        .await
        .expect_err("expected an error");
    assert!(matches!(
        &error,
        QwenKnowledgeChatError::StreamInterrupted { .. }
    ));
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Unknown);
    assert_eq!(mock.requests().len(), 2);
}

#[tokio::test]
async fn scope_image_and_session_file_limits_fail_before_http() {
    let mock = Mock::new([]);
    let service = make_service(&mock, "account-1");
    let other_scope = scope("account-2");
    let foreign_ref = other_scope.knowledge_chat_ref("aid-other").unwrap();
    let error = service
        .knowledge_chat(
            &QwenKnowledgeChatRequest::new(
                foreign_ref,
                [QwenKnowledgeChatMessage::user_text("Question")],
            ),
            &request_options(),
        )
        .await
        .err()
        .expect("expected an error");
    assert!(matches!(
        error,
        QwenKnowledgeChatError::Llm(LlmError::PermissionDenied { .. })
    ));

    let invalid_image = QwenKnowledgeChatRequest::new(
        chat_ref(service.scope()),
        [QwenKnowledgeChatMessage::user_parts([
            QwenKnowledgeChatContentPart::image_url("file:///private/image.png"),
        ])],
    );
    assert!(matches!(
        service
            .knowledge_chat(&invalid_image, &request_options())
            .await,
        Err(QwenKnowledgeChatError::InvalidInput(_))
    ));

    let files = (0..11)
        .map(|index| service.scope().file_ref(format!("file-{index}")).unwrap())
        .collect::<Vec<_>>();
    let too_many = QwenKnowledgeChatRequest::new(
        chat_ref(service.scope()),
        [QwenKnowledgeChatMessage::user_text("Question")],
    )
    .with_session_files(files);
    assert!(matches!(
        service.knowledge_chat(&too_many, &request_options()).await,
        Err(QwenKnowledgeChatError::InvalidInput(_))
    ));
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn transport_failure_is_not_retried_and_http_errors_keep_native_status() {
    let mock = Mock::new([
        Reply::Failure,
        Reply::Stream {
            status: 401,
            headers: vec![("x-request-id".into(), "header-401".into())],
            chunks: vec![Ok(Bytes::from_static(
                br#"{"code":"InvalidApiKey","message":"bad key","request_id":"body-401"}"#,
            ))],
        },
        Reply::Stream {
            status: 500,
            headers: Vec::new(),
            chunks: vec![Ok(Bytes::from_static(
                br#"{"code":"AgentApp.NotFound","message":"not published"}"#,
            ))],
        },
    ]);
    let service = make_service(&mock, "account-1");
    let request = || {
        QwenKnowledgeChatRequest::new(
            chat_ref(service.scope()),
            [QwenKnowledgeChatMessage::user_text("Question")],
        )
    };

    let error = service
        .knowledge_chat(&request(), &request_options())
        .await
        .err()
        .expect("expected an error");
    assert!(matches!(
        error,
        QwenKnowledgeChatError::OutcomeUnknown { .. }
    ));
    assert_eq!(error.dispatch(), QwenKnowledgeDispatch::Unknown);

    let error = service
        .knowledge_chat(&request(), &request_options())
        .await
        .err()
        .expect("expected an error");
    match error {
        QwenKnowledgeChatError::Provider {
            status: 401,
            dispatch: QwenKnowledgeDispatch::Rejected,
            error,
        } => {
            assert_eq!(error.request_id.as_deref(), Some("body-401"));
            assert_eq!(error.native["code"], "InvalidApiKey");
        }
        other => panic!("unexpected error: {other:?}"),
    }

    let error = service
        .knowledge_chat(&request(), &request_options())
        .await
        .err()
        .expect("expected an error");
    match error {
        QwenKnowledgeChatError::Provider {
            status: 500,
            dispatch: QwenKnowledgeDispatch::Unknown,
            error,
        } => assert_eq!(error.code.as_deref(), Some("AgentApp.NotFound")),
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(mock.requests().len(), 3);
}

#[tokio::test]
async fn body_limit_is_enforced_and_dropping_stream_cancels_body() {
    let mock = Mock::new([stream_reply([vec![b'x'; 8 * 1024 * 1024 + 1]])]);
    let service = make_service(&mock, "account-1");

    let large_request = QwenKnowledgeChatRequest::new(
        chat_ref(service.scope()),
        [QwenKnowledgeChatMessage::user_text(
            "x".repeat(8 * 1024 * 1024),
        )],
    );
    assert!(matches!(
        service
            .knowledge_chat(&large_request, &request_options())
            .await,
        Err(QwenKnowledgeChatError::InvalidInput(_))
    ));
    assert!(mock.requests().is_empty());

    let mut overlong_event = service
        .knowledge_chat(
            &QwenKnowledgeChatRequest::new(
                chat_ref(service.scope()),
                [QwenKnowledgeChatMessage::user_text("Question")],
            ),
            &request_options(),
        )
        .await
        .unwrap();
    assert!(matches!(
        overlong_event.next_event().await,
        Err(QwenKnowledgeChatError::StreamInterrupted { .. })
    ));

    let dropped = Arc::new(AtomicBool::new(false));
    let drop_mock = Mock::new([Reply::DropTracked(dropped.clone())]);
    let drop_service = make_service(&drop_mock, "account-1");
    let stream = drop_service
        .knowledge_chat(
            &QwenKnowledgeChatRequest::new(
                chat_ref(drop_service.scope()),
                [QwenKnowledgeChatMessage::user_text("Question")],
            ),
            &request_options(),
        )
        .await
        .unwrap();
    drop(stream);
    assert!(dropped.load(Ordering::SeqCst));
}

fn request_options() -> lingxi_llm_client::RequestOptions {
    lingxi_llm_client::RequestOptions {
        credential: Some(Secret::new("qwen-chat-api-key".to_owned())),
        ..Default::default()
    }
}
