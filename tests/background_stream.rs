use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{protocol::*, providers::openai::background::*, *};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

struct Reply {
    status: u16,
    frames: Vec<Result<Vec<u8>, LlmError>>,
}
struct Mock {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<HttpRequest>>,
}
#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected HTTP call");
        Ok(StreamResponse {
            status: reply.status,
            headers: vec![("content-type".into(), "text/event-stream".into())],
            body: futures::stream::iter(
                reply.frames.into_iter().map(|frame| frame.map(Into::into)),
            )
            .boxed(),
        })
    }
}
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openai", "profile_name":"openai", "base_url":"https://api.openai.com/v1",
        "protocol":"open_ai_responses", "auth":"none",
        "extra":{"supports_previous_response_id":true},
        "models":[{"display_model":"test","request_model":"test","billing_model":"test"}],
        "background":{"mode":"enabled","value":{"endpoint":"https://api.openai.com/v1/responses","auth":{"type":"bearer"}}}
    })).unwrap()
}
fn request() -> ChatRequest {
    serde_json::from_value(json!({"model":"test","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]})).unwrap()
}
fn options(scope: &str) -> RequestOptions {
    RequestOptions {
        account_scope: Some(scope.into()),
        credential: Some(Secret::new("key".into())),
        ..Default::default()
    }
}
fn setup(replies: Vec<Reply>) -> (LlmClient, Arc<Mock>) {
    let mock = Arc::new(Mock {
        replies: Mutex::new(replies.into()),
        requests: Mutex::new(vec![]),
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    (client, mock)
}
fn event(kind: &str, seq: u64, with_id: bool) -> Vec<u8> {
    let mut event = json!({"type":kind,"sequence_number":seq});
    if with_id {
        event["response"] = json!({"id":"resp_1"});
    }
    format!("data: {}\n\n", event).into_bytes()
}

fn full_terminal_event(kind: &str, seq: u64, status: &str, output: Value) -> Vec<u8> {
    let response = json!({
        "id":"resp_1",
        "model":"test",
        "status":status,
        "output":output,
        "usage":{"input_tokens":2,"output_tokens":3}
    });
    format!(
        "data: {}\n\n",
        json!({"type":kind,"sequence_number":seq,"response":response})
    )
    .into_bytes()
}

fn assistant_output(text: &str) -> Value {
    json!([{
        "id":"msg_1",
        "type":"message",
        "role":"assistant",
        "content":[{"type":"output_text","text":text,"annotations":[]}]
    }])
}

fn delta_event(seq: u64, block: usize, text: &str) -> Vec<u8> {
    format!(
        "data: {}\n\n",
        json!({
            "type":"response.output_text.delta",
            "sequence_number":seq,
            "output_index":block,
            "delta":text
        })
    )
    .into_bytes()
}

#[tokio::test]
async fn chat_stream_decodes_provider_neutral_events_and_reconstructs_terminal_response() {
    let mut output = assistant_output("hello from the background stream");
    output.as_array_mut().unwrap().push(json!({
        "type":"web_search_call","id":"ws_1","status":"completed",
        "action":{"type":"open_page","url":"https://example.test"}
    }));
    let completed = full_terminal_event("response.completed", 2, "completed", output);
    let (client, _) = setup(vec![Reply {
        status: 200,
        frames: vec![
            Ok(event("response.created", 0, true)),
            Ok(delta_event(1, 0, "hello")),
            Ok(completed),
        ],
    }]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit_chat_stream(&request(), &options("acct"))
        .await
        .unwrap();

    let start = stream.next_event().await.unwrap().unwrap();
    assert!(
        matches!(start.events.as_slice(), [StreamEvent::Start { response_id: Some(id), .. }] if id.as_str() == "resp_1")
    );
    assert_eq!(start.cursor.sequence_number, 0);

    let delta = stream.next_event().await.unwrap().unwrap();
    assert!(delta
        .events
        .iter()
        .any(|event| matches!(event, StreamEvent::TextDelta { text, .. } if text == "hello")));
    assert_eq!(delta.cursor.sequence_number, 1);

    let terminal = stream.next_event().await.unwrap().unwrap();
    assert!(terminal.terminal);
    assert_eq!(terminal.native["type"], "response.completed");
    assert!(terminal.events.iter().any(|event| matches!(event, StreamEvent::ProviderContent { value, .. } if value["type"] == "web_search_call")));
    assert!(terminal
        .events
        .iter()
        .any(|event| matches!(event, StreamEvent::End { .. })));
    let response = terminal.response.unwrap();
    assert_eq!(response.message.text(), "hello from the background stream");
    assert_eq!(response.response_id.as_ref().unwrap().as_str(), "resp_1");
    assert_eq!(
        response.continuation.as_ref().unwrap().account_scope,
        "acct"
    );
    assert!(response.message.content.iter().any(|block| matches!(block, ContentBlock::ProviderContent { value, .. } if value["type"] == "web_search_call")));
    assert!(stream.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn chat_stream_resume_keeps_cursor_and_rebuilds_response_without_resubmitting() {
    let (client, mock) = setup(vec![
        Reply {
            status: 200,
            frames: vec![
                Ok(event("response.created", 0, true)),
                Ok(delta_event(1, 0, "partial")),
                Err(LlmError::Transport {
                    message: "disconnected".into(),
                }),
            ],
        },
        Reply {
            status: 200,
            frames: vec![Ok(full_terminal_event(
                "response.completed",
                2,
                "completed",
                assistant_output("complete answer after resume"),
            ))],
        },
    ]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit_chat_stream(&request(), &options("acct"))
        .await
        .unwrap();
    let _ = stream.next_event().await.unwrap().unwrap();
    let delta = stream.next_event().await.unwrap().unwrap();
    assert!(delta
        .events
        .iter()
        .any(|event| matches!(event, StreamEvent::TextDelta { text, .. } if text == "partial")));
    let error = stream.next_event().await.unwrap_err();
    let BackgroundChatStreamError::Native(BackgroundStreamError::Interrupted {
        cursor: Some(cursor),
        ..
    }) = error
    else {
        panic!("expected resumable interruption")
    };
    assert_eq!(cursor.sequence_number, 1);

    let mut resumed = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .resume_chat_stream(*cursor, &options("acct"))
        .await
        .unwrap();
    let terminal = resumed.next_event().await.unwrap().unwrap();
    assert_eq!(terminal.cursor.sequence_number, 2);
    assert_eq!(
        terminal.response.unwrap().message.text(),
        "complete answer after resume"
    );
    assert!(resumed.next_event().await.unwrap().is_none());

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[1].method, "GET");
    assert_eq!(
        requests[1].url,
        "https://api.openai.com/v1/responses/resp_1?stream=true"
    );
}

#[tokio::test]
async fn chat_stream_incomplete_is_a_response_but_failed_keeps_native_error_and_cursor() {
    let (client, _) = setup(vec![Reply {
        status: 200,
        frames: vec![Ok(full_terminal_event(
            "response.incomplete",
            0,
            "incomplete",
            assistant_output("partial answer"),
        ))],
    }]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit_chat_stream(&request(), &options("acct"))
        .await
        .unwrap();
    let terminal = stream.next_event().await.unwrap().unwrap();
    assert!(terminal.terminal);
    let response = terminal.response.unwrap();
    assert_eq!(response.message.text(), "partial answer");
    assert_eq!(response.stop_reason, StopReason::Other("incomplete".into()));

    let failed_event = json!({
        "type":"response.failed","sequence_number":1,
        "response":{"id":"resp_1","model":"test","status":"failed","output":[],"error":{"message":"provider failed"}}
    });
    let (client, _) = setup(vec![Reply {
        status: 200,
        frames: vec![
            Ok(event("response.created", 0, true)),
            Ok(format!("data: {}\n\n", failed_event).into_bytes()),
        ],
    }]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit_chat_stream(&request(), &options("acct"))
        .await
        .unwrap();
    let _ = stream.next_event().await.unwrap().unwrap();
    let error = stream.next_event().await.unwrap_err();
    let BackgroundChatStreamError::Decode { cursor, native, .. } = error else {
        panic!("expected provider decode error")
    };
    assert_eq!(cursor.sequence_number, 1);
    assert_eq!(native["type"], "response.failed");
}

#[tokio::test]
async fn submit_disconnect_resume_tracks_exact_sequence_and_scope() {
    let created = event("response.created", 0, true);
    let delta = event("response.output_text.delta", 1, false);
    let (client, mock) = setup(vec![
        Reply {
            status: 200,
            frames: vec![
                Ok(created[..10].to_vec()),
                Ok(created[10..].to_vec()),
                Ok(delta),
                Err(LlmError::Transport {
                    message: "disconnected".into(),
                }),
            ],
        },
        Reply {
            status: 200,
            frames: vec![Ok(event("response.completed", 2, true))],
        },
    ]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit_stream(&request(), &options("a"))
        .await
        .unwrap();
    let first = stream.next_event().await.unwrap().unwrap();
    assert_eq!(first.cursor.sequence_number, 0);
    assert_eq!(first.cursor.reference.response_id, "resp_1");
    assert!(!first.terminal);
    let second = stream.next_event().await.unwrap().unwrap();
    assert_eq!(second.cursor.sequence_number, 1);
    let interrupted = stream.next_event().await.unwrap_err();
    let BackgroundStreamError::Interrupted {
        cursor: Some(cursor),
        ..
    } = interrupted
    else {
        panic!("expected resumable interruption")
    };
    assert_eq!(cursor.sequence_number, 1);
    assert_eq!(cursor.reference.account_scope, "a");

    let mut resumed = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .resume_stream(*cursor, &options("a"))
        .await
        .unwrap();
    let final_event = resumed.next_event().await.unwrap().unwrap();
    assert!(final_event.terminal);
    assert_eq!(final_event.cursor.sequence_number, 2);
    assert!(resumed.next_event().await.unwrap().is_none());
    let sent = mock.requests.lock().unwrap();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0].method, "POST");
    assert_eq!(
        sent[1].url,
        "https://api.openai.com/v1/responses/resp_1?stream=true&starting_after=1"
    );
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["background"], true);
    assert_eq!(body["stream"], true);
}

#[tokio::test]
async fn resume_wrong_account_fails_before_network() {
    let (client, mock) = setup(vec![Reply {
        status: 200,
        frames: vec![Ok(event("response.created", 0, true))],
    }]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit_stream(&request(), &options("a"))
        .await
        .unwrap();
    let cursor = stream.next_event().await.unwrap().unwrap().cursor;
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .background()
            .resume_stream(cursor, &options("b"))
            .await,
        Err(BackgroundError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn missing_id_and_non_monotonic_sequence_are_rejected() {
    let (client, _) = setup(vec![Reply {
        status: 200,
        frames: vec![Ok(event("response.created", 0, false))],
    }]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit_stream(&request(), &options("a"))
        .await
        .unwrap();
    assert!(matches!(
        stream.next_event().await,
        Err(BackgroundStreamError::InvalidEvent { cursor: None, .. })
    ));

    let (client, _) = setup(vec![Reply {
        status: 200,
        frames: vec![
            Ok(event("response.created", 1, true)),
            Ok(event("response.output_text.delta", 1, false)),
        ],
    }]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit_stream(&request(), &options("a"))
        .await
        .unwrap();
    let _ = stream.next_event().await.unwrap();
    assert!(matches!(
        stream.next_event().await,
        Err(BackgroundStreamError::InvalidEvent {
            cursor: Some(_),
            ..
        })
    ));
}

#[tokio::test]
async fn clean_eof_before_terminal_retains_cursor() {
    let (client, _) = setup(vec![Reply {
        status: 200,
        frames: vec![Ok(event("response.created", 4, true))],
    }]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit_stream(&request(), &options("a"))
        .await
        .unwrap();
    let _ = stream.next_event().await.unwrap();
    let error = stream.next_event().await.unwrap_err();
    let BackgroundStreamError::Interrupted {
        cursor: Some(cursor),
        ..
    } = error
    else {
        panic!("expected resumable interruption")
    };
    assert_eq!(cursor.sequence_number, 4);
}

#[tokio::test]
async fn provider_http_error_is_not_exposed_as_an_event_stream() {
    let (client, mock) = setup(vec![Reply {
        status: 429,
        frames: vec![Ok(serde_json::to_vec(
            &json!({"error":{"message":"slow down"}}),
        )
        .unwrap())],
    }]);
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .background()
            .submit_stream(&request(), &options("a"))
            .await,
        Err(BackgroundError::Provider { status: 429, .. })
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

fn tool_added_event() -> Vec<u8> {
    format!("data: {}\n\n", json!({
        "type":"response.output_item.added", "sequence_number":1, "output_index":0,
        "item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":""}
    })).into_bytes()
}

fn tool_arguments_event() -> Vec<u8> {
    format!(
        "data: {}\n\n",
        json!({
            "type":"response.function_call_arguments.delta", "sequence_number":2,
            "output_index":0,"item_id":"fc_1","delta":"{\"city\":\"Paris\"}"
        })
    )
    .into_bytes()
}

#[tokio::test]
async fn chat_resume_rebuilds_tool_mapping_without_redelivering_history() {
    let (client, mock) = setup(vec![
        Reply {
            status: 200,
            frames: vec![
                Ok(event("response.created", 0, true)),
                Ok(tool_added_event()),
                Err(LlmError::Transport {
                    message: "disconnect".into(),
                }),
            ],
        },
        Reply {
            status: 200,
            frames: vec![
                Ok(event("response.created", 0, true)),
                Ok(tool_added_event()),
                Ok(tool_arguments_event()),
                Ok(full_terminal_event(
                    "response.completed",
                    3,
                    "completed",
                    json!([{
                        "type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":"{\"city\":\"Paris\"}"
                    }]),
                )),
            ],
        },
    ]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit_chat_stream(&request(), &options("acct"))
        .await
        .unwrap();
    stream.next_event().await.unwrap();
    let added = stream.next_event().await.unwrap().unwrap();
    assert!(added
        .events
        .iter()
        .any(|e| matches!(e, StreamEvent::ToolCallDelta { .. })));
    assert!(stream.next_event().await.is_err());
    // Only the durable cursor is retained across recovery, not the old decoder.
    let cursor = serde_json::from_str(&serde_json::to_string(&added.cursor).unwrap()).unwrap();
    let mut resumed = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .resume_chat_stream(cursor, &options("acct"))
        .await
        .unwrap();
    let delta = resumed.next_event().await.unwrap().unwrap();
    assert_eq!(delta.cursor.sequence_number, 2);
    assert!(
        matches!(delta.events.as_slice(), [StreamEvent::ToolCallDelta { id, name, arguments_fragment, .. }]
        if id.as_str() == "call_1" && name == "weather" && arguments_fragment == "{\"city\":\"Paris\"}")
    );
    let terminal = resumed.next_event().await.unwrap().unwrap();
    assert!(terminal.terminal);
    assert!(terminal.response.is_some());
    assert!(resumed.next_event().await.unwrap().is_none());
    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].method, "GET");
    assert_eq!(
        requests[1].url,
        "https://api.openai.com/v1/responses/resp_1?stream=true"
    );
}

#[tokio::test]
async fn interrupted_chat_replay_preserves_last_delivered_cursor() {
    let (client, _) = setup(vec![
        Reply {
            status: 200,
            frames: vec![
                Ok(event("response.created", 0, true)),
                Ok(tool_added_event()),
            ],
        },
        Reply {
            status: 200,
            frames: vec![
                Ok(event("response.created", 0, true)),
                Err(LlmError::Transport {
                    message: "replay disconnected".into(),
                }),
            ],
        },
        Reply {
            status: 200,
            frames: vec![
                Ok(event("response.created", 0, true)),
                Ok(tool_added_event()),
                Ok(tool_arguments_event()),
            ],
        },
    ]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit_chat_stream(&request(), &options("acct"))
        .await
        .unwrap();
    stream.next_event().await.unwrap();
    let checkpoint = stream.next_event().await.unwrap().unwrap().cursor;
    let mut resumed = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .resume_chat_stream(checkpoint.clone(), &options("acct"))
        .await
        .unwrap();
    let error = resumed.next_event().await.unwrap_err();
    let BackgroundChatStreamError::Native(BackgroundStreamError::Interrupted {
        cursor: Some(cursor),
        ..
    }) = error
    else {
        panic!("expected interruption")
    };
    assert_eq!(*cursor, checkpoint);
    let mut resumed = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .resume_chat_stream(*cursor, &options("acct"))
        .await
        .unwrap();
    assert_eq!(
        resumed
            .next_event()
            .await
            .unwrap()
            .unwrap()
            .cursor
            .sequence_number,
        2
    );
}

#[tokio::test]
async fn chat_replay_rejects_a_different_response_id() {
    let (client, _) = setup(vec![
        Reply {
            status: 200,
            frames: vec![Ok(event("response.created", 0, true))],
        },
        Reply {
            status: 200,
            frames: vec![Ok(String::from_utf8(event("response.created", 0, true))
                .unwrap()
                .replace("resp_1", "resp_other")
                .into_bytes())],
        },
    ]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .submit_chat_stream(&request(), &options("acct"))
        .await
        .unwrap();
    let cursor = stream.next_event().await.unwrap().unwrap().cursor;
    let mut resumed = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .background()
        .resume_chat_stream(cursor.clone(), &options("acct"))
        .await
        .unwrap();
    let error = resumed.next_event().await.unwrap_err();
    assert!(
        matches!(error,BackgroundChatStreamError::Native(BackgroundStreamError::InvalidEvent {cursor:Some(checkpoint),..}) if *checkpoint == cursor)
    );
}
