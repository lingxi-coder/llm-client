use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

struct Reply {
    status: u16,
    body: Vec<u8>,
}
#[derive(Default)]
struct Mock {
    sent: Mutex<Vec<HttpRequest>>,
    replies: Mutex<VecDeque<Result<Reply, LlmError>>>,
}
impl Mock {
    fn json(&self, status: u16, body: Value) {
        self.raw(status, serde_json::to_vec(&body).unwrap());
    }
    fn raw(&self, status: u16, body: Vec<u8>) {
        self.replies
            .lock()
            .unwrap()
            .push_back(Ok(Reply { status, body }));
    }
    fn bodies(&self) -> Vec<Value> {
        self.sent
            .lock()
            .unwrap()
            .iter()
            .map(|r| serde_json::from_slice(&r.body).unwrap())
            .collect()
    }
}
#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected generation")?;
        // Split SSE/JSON at arbitrary byte boundaries, including UTF-8 units.
        let chunks: Vec<_> = reply
            .body
            .chunks(7)
            .map(|chunk| Ok(bytes::Bytes::copy_from_slice(chunk)))
            .collect();
        Ok(StreamResponse {
            status: reply.status,
            headers: vec![],
            body: futures::stream::iter(chunks).boxed(),
        })
    }
}
fn profile(family: &str) -> ProviderProfile {
    let (provider, endpoint, model, extra) = match family {
        "anthropic_messages" => (
            "anthropic",
            "https://api.anthropic.com",
            "claude-sonnet-4-6",
            json!({"supports_message_threads":true}),
        ),
        "gemini_interactions" => (
            "google",
            "https://generativelanguage.googleapis.com/v1beta",
            "gemini-3.8-flash",
            json!({}),
        ),
        _ => (
            "fixture",
            "https://fixture.test/v1",
            "m",
            json!({"supports_previous_response_id":true}),
        ),
    };
    serde_json::from_value(json!({"provider_id":provider,"profile_name":"one","base_url":endpoint,"protocol":family,"auth":"none","extra":extra,"models":[{"display_model":"m","request_model":model,"billing_model":model}]})).unwrap()
}
fn client(mock: &Arc<Mock>, profile: ProviderProfile) -> LlmClient {
    LlmClientBuilder::with_transport(mock.clone(), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap()
}
fn options() -> RequestOptions {
    RequestOptions {
        account_scope: Some("account-a".into()),
        ..Default::default()
    }
}
fn request() -> ChatRequest {
    let mut request = ChatRequest::new("m");
    request
        .messages
        .push(ConversationMessage::user_text("remember 4921"));
    request
}
fn response(family: &str, id: &str, text: &str) -> Value {
    match family {
        "anthropic_messages" => {
            json!({"id":id,"type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":text}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2}})
        }
        "gemini_interactions" => {
            json!({"id":id,"status":"completed","model":"gemini-3.8-flash","steps":[{"type":"model_output","content":[{"type":"text","text":text}]}],"usage":{"total_input_tokens":5,"total_output_tokens":2,"total_tokens":7}})
        }
        _ => {
            json!({"id":id,"status":"completed","model":"m","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":text,"annotations":[]}]}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}})
        }
    }
}
async fn complete(
    session: &mut ConversationSession,
    client: &LlmClient,
    request: &ChatRequest,
    options: &RequestOptions,
) -> ChatResponse {
    session
        .complete_in(
            &client.snapshot(),
            "one",
            request,
            options,
            || Ok(()),
            |_| {},
        )
        .await
        .unwrap()
}
fn followup(request: &mut ChatRequest, response: ChatResponse) {
    request.messages.push(response.message);
    request
        .messages
        .push(ConversationMessage::user_text("what was the number?"));
}

#[tokio::test]
async fn three_turns_use_only_new_input_on_supported_protocols() {
    for family in [
        "open_ai_responses",
        "anthropic_messages",
        "gemini_interactions",
    ] {
        let mock = Arc::new(Mock::default());
        for id in ["first", "second", "third"] {
            mock.json(200, response(family, id, "4921"));
        }
        let client = client(&mock, profile(family));
        let mut session = ConversationSession::new();
        let mut request = request();
        let first = complete(&mut session, &client, &request, &options()).await;
        followup(&mut request, first);
        let second = complete(&mut session, &client, &request, &options()).await;
        assert_eq!(
            session.last_request().mode,
            ConversationRequestMode::Continue,
            "{family}"
        );
        followup(&mut request, second);
        complete(&mut session, &client, &request, &options()).await;
        let bodies = mock.bodies();
        let input = if family == "anthropic_messages" {
            "/messages"
        } else if family == "gemini_interactions" {
            "/input/steps"
        } else {
            "/input"
        };
        assert_eq!(
            (*bodies[1].pointer(input).unwrap())
                .as_array()
                .unwrap()
                .len(),
            1,
            "{family}"
        );
        assert_eq!(
            (*bodies[2].pointer(input).unwrap())
                .as_array()
                .unwrap()
                .len(),
            1,
            "{family}"
        );
        assert!(
            !serde_json::to_string(&(*bodies[1].pointer(input).unwrap()))
                .unwrap()
                .contains("remember")
        );
        if family == "anthropic_messages" {
            assert_eq!(bodies[0]["thread"]["type"], "create");
            assert_eq!(bodies[2]["thread"]["previous_message_id"], "second");
            assert!(mock.sent.lock().unwrap().iter().all(|r| r
                .headers
                .iter()
                .any(|(key, value)| key == "anthropic-beta"
                    && value.contains("message-threads-2026-08-12"))));
        }
    }
}

#[tokio::test]
async fn exact_rejection_replays_once_and_reenters_admission_and_accounting() {
    for family in [
        "open_ai_responses",
        "anthropic_messages",
        "gemini_interactions",
    ] {
        let mock = Arc::new(Mock::default());
        mock.json(200, response(family, "first", "4921"));
        let code = if family == "anthropic_messages" {
            "thread_not_found"
        } else if family == "gemini_interactions" {
            "interaction_not_found"
        } else {
            "previous_response_not_found"
        };
        mock.json(404, json!({"error":{"type":"not_found_error","code":code,"message":"expired","details":{"error_code":code}}}));
        mock.json(200, response(family, "recovered", "4921"));
        let client = client(&mock, profile(family));
        let mut session = ConversationSession::new();
        let mut request = request();
        let first = complete(&mut session, &client, &request, &options()).await;
        followup(&mut request, first);
        let mut admissions = 0;
        let mut statuses = Vec::new();
        session
            .complete_in(
                &client.snapshot(),
                "one",
                &request,
                &options(),
                || {
                    admissions += 1;
                    Ok(())
                },
                |r| statuses.push(r.response().status),
            )
            .await
            .unwrap();
        assert_eq!(admissions, 2);
        assert_eq!(statuses, [404, 200]);
        assert_eq!(
            session.last_request().mode,
            ConversationRequestMode::Recreate
        );
        let bodies = mock.bodies();
        let input = if family == "anthropic_messages" {
            "/messages"
        } else if family == "gemini_interactions" {
            "/input/steps"
        } else {
            "/input"
        };
        assert_eq!(
            (*bodies[1].pointer(input).unwrap())
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            (*bodies[2].pointer(input).unwrap())
                .as_array()
                .unwrap()
                .len(),
            3
        );
    }
}

#[tokio::test]
async fn generic_404_transport_failure_and_second_rejection_never_loop() {
    for kind in ["generic", "transport", "twice"] {
        let mock = Arc::new(Mock::default());
        mock.json(200, response("open_ai_responses", "first", "4921"));
        if kind == "transport" {
            mock.replies
                .lock()
                .unwrap()
                .push_back(Err(LlmError::TransportTimeout {
                    message: "outcome unknown".into(),
                }));
        } else {
            mock.json(404, json!({"error":{"code":if kind == "twice" {"previous_response_not_found"} else {"model_not_found"},"message":"missing"}}));
            if kind == "twice" {
                mock.json(404, json!({"error":{"code":"previous_response_not_found","message":"still missing"}}));
            }
        }
        let client = client(&mock, profile("open_ai_responses"));
        let mut session = ConversationSession::new();
        let mut request = request();
        let first = complete(&mut session, &client, &request, &options()).await;
        followup(&mut request, first);
        assert!(session
            .complete_in(
                &client.snapshot(),
                "one",
                &request,
                &options(),
                || Ok(()),
                |_| {}
            )
            .await
            .is_err());
        assert_eq!(mock.bodies().len(), if kind == "twice" { 3 } else { 2 });
    }
}

#[tokio::test]
async fn account_store_policy_history_and_fixed_field_changes_rebuild() {
    for change in ["account", "store", "history", "tools", "reset"] {
        let mock = Arc::new(Mock::default());
        for id in ["first", "second"] {
            mock.json(200, response("open_ai_responses", id, "4921"));
        }
        let client = client(&mock, profile("open_ai_responses"));
        let mut session = ConversationSession::new();
        let mut request = request();
        let mut opts = options();
        let first = complete(&mut session, &client, &request, &opts).await;
        followup(&mut request, first);
        match change {
            "account" => opts.account_scope = Some("account-b".into()),
            "store" => request.controls.responses.store = Some(false),
            "history" => request.messages[0] = ConversationMessage::user_text("edited"),
            "tools" => request.system.push(SystemBlock::new("changed")),
            _ => session.reset(),
        }
        let second = complete(&mut session, &client, &request, &opts).await;
        assert_ne!(
            session.last_request().mode,
            ConversationRequestMode::Continue,
            "{change}"
        );
        assert!(mock.bodies()[1].get("previous_response_id").is_none());
        if change == "store" {
            assert!(second.continuation.is_none());
        }
    }
}

#[tokio::test]
async fn unsupported_routes_and_missing_scope_keep_full_history() {
    for scoped in [false, true] {
        let mock = Arc::new(Mock::default());
        for id in ["first", "second"] {
            mock.json(200, response("open_ai_responses", id, "4921"));
        }
        let mut p = profile("open_ai_responses");
        if scoped {
            p.extra = json!({"supports_previous_response_id":false});
        }
        let client = client(&mock, p);
        let mut session = ConversationSession::new();
        let mut request = request();
        let mut opts = options();
        if !scoped {
            opts.account_scope = None;
        }
        let first = complete(&mut session, &client, &request, &opts).await;
        followup(&mut request, first);
        complete(&mut session, &client, &request, &opts).await;
        assert_eq!(mock.bodies()[1]["input"].as_array().unwrap().len(), 3);
    }
}

#[tokio::test]
async fn thread_continuation_omits_system_tools_and_rebuilds_on_changes() {
    let mock = Arc::new(Mock::default());
    for id in ["first", "second", "third"] {
        mock.json(200, response("anthropic_messages", id, "4921"));
    }
    let client = client(&mock, profile("anthropic_messages"));
    let mut session = ConversationSession::new();
    let mut request = request();
    request.system.push(SystemBlock::new("support assistant"));
    request.tools = serde_json::from_value(
        json!([{"name":"lookup","description":"lookup","input_schema":{"type":"object"}}]),
    )
    .unwrap();
    let first = complete(&mut session, &client, &request, &options()).await;
    followup(&mut request, first);
    request.max_tokens = Some(900);
    let second = complete(&mut session, &client, &request, &options()).await;
    followup(&mut request, second);
    request.system.push(SystemBlock::new("changed"));
    complete(&mut session, &client, &request, &options()).await;
    let bodies = mock.bodies();
    assert!(bodies[1].get("system").is_none());
    assert!(bodies[1].get("tools").is_none());
    assert_eq!(bodies[1]["max_tokens"], 900);
    assert_eq!(bodies[2]["thread"]["type"], "create");
    assert!(bodies[2].get("system").is_some());
}

fn sse(events: &[Value]) -> Vec<u8> {
    events
        .iter()
        .map(|v| format!("data: {}\n\n", serde_json::to_string(v).unwrap()))
        .collect::<String>()
        .into_bytes()
}
#[tokio::test]
async fn streaming_commits_only_at_terminal_and_http_recovery_is_bounded() {
    let family = "open_ai_responses";
    let mock = Arc::new(Mock::default());
    for id in ["first", "second"] {
        if id == "second" {
            mock.json(
                400,
                json!({"error":{"code":"previous_response_not_found","message":"expired"}}),
            );
        }
        mock.raw(200, sse(&[json!({"type":"response.created","response":{"id":id,"model":"m"}}), json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"4921"}), json!({"type":"response.completed","response":response(family,id,"4921")})]));
    }
    let client = client(&mock, profile(family));
    let mut session = ConversationSession::new();
    let mut request = request();
    let mut stream = session
        .stream_in(
            &client.snapshot(),
            "one",
            &request,
            &options(),
            || Ok(()),
            |_| {},
        )
        .await
        .unwrap();
    let mut accumulator = StreamAccumulator::new();
    while let Some(event) = stream.next().await {
        accumulator.observe(&event.unwrap());
    }
    followup(&mut request, accumulator.snapshot().response);
    let mut stream = session
        .stream_in(
            &client.snapshot(),
            "one",
            &request,
            &options(),
            || Ok(()),
            |_| {},
        )
        .await
        .unwrap();
    while let Some(event) = stream.next().await {
        event.unwrap();
    }
    assert_eq!(mock.bodies().len(), 3);
    assert_eq!(mock.bodies()[1]["input"].as_array().unwrap().len(), 1);
    assert_eq!(mock.bodies()[2]["input"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn dropping_a_stream_or_reusing_an_old_observer_cannot_seed_a_new_turn() {
    let mock = Arc::new(Mock::default());
    mock.raw(
        200,
        sse(&[json!({"type":"response.created","response":{"id":"unfinished","model":"m"}})]),
    );
    mock.json(200, response("open_ai_responses", "second", "4921"));
    let client = client(&mock, profile("open_ai_responses"));
    let mut session = ConversationSession::new();
    let mut request = request();
    let stream = session
        .stream_in(
            &client.snapshot(),
            "one",
            &request,
            &options(),
            || Ok(()),
            |_| {},
        )
        .await
        .unwrap();
    drop(stream);
    request
        .messages
        .push(ConversationMessage::user_text("after cancel"));
    complete(&mut session, &client, &request, &options()).await;
    assert!(mock.bodies()[1].get("previous_response_id").is_none());
}

#[tokio::test]
async fn rejection_of_recovery_admission_sends_no_second_generation() {
    let mock = Arc::new(Mock::default());
    mock.json(200, response("open_ai_responses", "first", "4921"));
    mock.json(
        404,
        json!({"error":{"code":"previous_response_not_found","message":"expired"}}),
    );
    let client = client(&mock, profile("open_ai_responses"));
    let mut session = ConversationSession::new();
    let mut request = request();
    let first = complete(&mut session, &client, &request, &options()).await;
    followup(&mut request, first);
    let mut attempts = 0;
    assert!(session
        .complete_in(
            &client.snapshot(),
            "one",
            &request,
            &options(),
            || {
                attempts += 1;
                if attempts == 2 {
                    Err(LlmError::PermissionDenied {
                        message: "revoked".into(),
                    })
                } else {
                    Ok(())
                }
            },
            |_| {}
        )
        .await
        .is_err());
    assert_eq!(mock.bodies().len(), 2);
}

#[test]
fn provider_presets_only_advertise_verified_continuation() {
    let profiles = builtin_providers().unwrap();
    for name in [
        "qwen-search",
        "qwen-search-intl",
        "qwen-search-hk",
        "qwen-search-us",
        "grok-responses",
        "openai",
    ] {
        assert_eq!(
            ContinuationCapability::for_profile(
                profiles.iter().find(|p| p.profile_name == name).unwrap()
            ),
            ContinuationCapability::Responses
        );
    }
    for name in [
        "deepseek-responses",
        "minimax-responses",
        "minimax-responses-intl",
        "openrouter-responses",
        "kimi-code-responses",
        "zai-coding-responses",
    ] {
        assert_eq!(
            ContinuationCapability::for_profile(
                profiles.iter().find(|p| p.profile_name == name).unwrap()
            ),
            ContinuationCapability::Unsupported
        );
    }
}

#[tokio::test]
async fn anthropic_streaming_tool_results_and_automatic_cache_continue() {
    let family = "anthropic_messages";
    let mock = Arc::new(Mock::default());
    mock.raw(200,sse(&[
        json!({"type":"message_start","message":{"id":"first","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call1","name":"lookup","input":{}}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"n\":4921}"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}),
        json!({"type":"message_stop"})]));
    mock.json(200, response(family, "second", "found"));
    let client = client(&mock, profile(family));
    let mut session = ConversationSession::new();
    let mut request = request();
    request.tools=serde_json::from_value(json!([{"name":"lookup","description":"lookup","input_schema":{"type":"object","properties":{"n":{"type":"integer"}}}}])).unwrap();
    request.prompt_cache.automatic = Some(CacheTtl::FiveMinutes);
    let mut stream = session
        .stream_in(
            &client.snapshot(),
            "one",
            &request,
            &options(),
            || Ok(()),
            |_| {},
        )
        .await
        .unwrap();
    let mut accumulator = StreamAccumulator::new();
    while let Some(event) = stream.next().await {
        accumulator.observe(&event.unwrap());
    }
    request
        .messages
        .push(accumulator.snapshot().response.message);
    request.messages.push(serde_json::from_value(json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"call1","content":"result","is_error":false}]})).unwrap());
    complete(&mut session, &client, &request, &options()).await;
    assert_eq!(
        session.last_request().mode,
        ConversationRequestMode::Continue
    );
    assert_eq!(mock.bodies()[1]["messages"].as_array().unwrap().len(), 1);
    assert_eq!(
        mock.bodies()[1]["messages"][0]["content"][0]["tool_use_id"],
        "call1"
    );
    assert!(mock.bodies()[1].get("cache_control").is_some());
}

#[tokio::test]
async fn gemini_streaming_terminal_without_steps_retains_exact_decoder_steps() {
    let mock = Arc::new(Mock::default());
    mock.raw(200,sse(&[
        json!({"event_type":"interaction.created","interaction":{"id":"first"}}),
        json!({"event_type":"step.start","index":0,"step":{"type":"model_output","content":[]}}),
        json!({"event_type":"step.delta","index":0,"delta":{"type":"text","text":"4921"}}),
        json!({"event_type":"step.stop","index":0}),
        json!({"event_type":"interaction.completed","interaction":{"id":"first","status":"completed","model":"gemini-3.8-flash","usage":{"total_input_tokens":5,"total_output_tokens":2,"total_tokens":7}}})]));
    mock.json(200, response("gemini_interactions", "second", "4921"));
    let client = client(&mock, profile("gemini_interactions"));
    let mut session = ConversationSession::new();
    let mut request = request();
    let mut stream = session
        .stream_in(
            &client.snapshot(),
            "one",
            &request,
            &options(),
            || Ok(()),
            |_| {},
        )
        .await
        .unwrap();
    let mut accumulator = StreamAccumulator::new();
    while let Some(event) = stream.next().await {
        accumulator.observe(&event.unwrap());
    }
    followup(&mut request, accumulator.snapshot().response);
    complete(&mut session, &client, &request, &options()).await;
    assert_eq!(
        session.last_request().mode,
        ConversationRequestMode::Continue
    );
    assert_eq!(
        mock.bodies()[1]["input"]["steps"].as_array().unwrap().len(),
        1
    );
}

struct WsMock {
    inner: Arc<Mock>,
    connects: std::sync::atomic::AtomicUsize,
}
struct WsConnection {
    inner: Arc<Mock>,
}
#[async_trait]
impl Transport for WsMock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.inner.send(request).await
    }
    async fn connect_websocket(
        &self,
        _: HttpRequest,
    ) -> Result<Box<dyn transport::WebSocketConnection>, LlmError> {
        self.connects
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(Box::new(WsConnection {
            inner: self.inner.clone(),
        }))
    }
}
#[async_trait]
impl transport::WebSocketConnection for WsConnection {
    async fn send(&mut self, payload: bytes::Bytes) -> Result<StreamResponse, LlmError> {
        self.inner.sent.lock().unwrap().push(HttpRequest {
            method: "POST".into(),
            url: "wss://fixture.test/v1/responses".into(),
            body: payload,
            headers: vec![],
            http1_header_layout: None,
            timeout: None,
        });
        let reply = self.inner.replies.lock().unwrap().pop_front().unwrap()?;
        let events: Vec<Value> = serde_json::from_slice(&reply.body).unwrap();
        Ok(StreamResponse {
            status: reply.status,
            headers: vec![],
            body: futures::stream::iter(
                events
                    .into_iter()
                    .map(|v| Ok(serde_json::to_vec(&v).unwrap().into())),
            )
            .boxed(),
        })
    }
    async fn close(&mut self) -> Result<(), LlmError> {
        Ok(())
    }
}

#[tokio::test]
async fn websocket_store_false_recovers_from_pre_generation_error_and_reconnects() {
    let inner = Arc::new(Mock::default());
    let ws = Arc::new(WsMock {
        inner: inner.clone(),
        connects: Default::default(),
    });
    let mut p = profile("open_ai_responses");
    p.supports_websockets = true;
    let client = LlmClientBuilder::with_transport(ws.clone(), &[p])
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut session = ConversationSession::new();
    let mut request = request();
    request.controls.responses.store = Some(false);
    for id in ["first", "second"] {
        if id == "second" {
            inner.json(200,json!([{"type":"error","status":400,"error":{"code":"previous_response_not_found","message":"expired"}}]));
        }
        inner.json(
            200,
            json!([
            {"type":"response.created","response":{"id":id,"model":"m"}},
            {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"4921"},
            {"type":"response.completed","response":response("open_ai_responses",id,"4921")}]),
        );
    }
    let mut stream = session
        .stream_in(
            &client.snapshot(),
            "one",
            &request,
            &options(),
            || Ok(()),
            |_| {},
        )
        .await
        .unwrap();
    let mut accumulator = StreamAccumulator::new();
    while let Some(event) = stream.next().await {
        accumulator.observe(&event.unwrap());
    }
    followup(&mut request, accumulator.snapshot().response);
    let mut admitted = 0;
    let mut stream = session
        .stream_in(
            &client.snapshot(),
            "one",
            &request,
            &options(),
            || {
                admitted += 1;
                Ok(())
            },
            |_| {},
        )
        .await
        .unwrap();
    while let Some(event) = stream.next().await {
        event.unwrap();
    }
    assert_eq!(admitted, 2);
    assert_eq!(inner.bodies().len(), 3);
    assert_eq!(inner.bodies()[1]["previous_response_id"], "first");
    assert!(inner.bodies()[2].get("previous_response_id").is_none());
    assert_eq!(ws.connects.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[tokio::test]
async fn glm_requires_opt_in_storage_and_preserves_account_product() {
    for store in [None, Some(true), Some(false)] {
        let mock = Arc::new(Mock::default());
        for id in ["first", "second"] {
            mock.json(200, response("open_ai_responses", id, "4921"));
        }
        let mut p = builtin_providers()
            .unwrap()
            .into_iter()
            .find(|p| p.profile_name == "glm-responses")
            .unwrap();
        p.auth = AuthStrategy::None;
        p.profile_name = "one".into();
        let client = LlmClientBuilder::with_transport(mock.clone(), &[p])
            .with_region(Region::ChinaMainland)
            .build()
            .unwrap();
        let mut session = ConversationSession::new();
        let mut request = request();
        request.model = "glm-5.3".into();
        request.controls.responses.store = store;
        let first = complete(&mut session, &client, &request, &options()).await;
        assert_eq!(first.continuation.is_some(), store == Some(true));
        followup(&mut request, first);
        complete(&mut session, &client, &request, &options()).await;
        assert_eq!(
            mock.bodies()[1].get("previous_response_id").is_some(),
            store == Some(true)
        );
        assert!(mock
            .sent
            .lock()
            .unwrap()
            .iter()
            .all(|r| r.url == "https://open.bigmodel.cn/api/v1/responses"));
    }
}

#[tokio::test]
async fn explicit_branches_do_not_seed_implicit_full_history() {
    let mock = Arc::new(Mock::default());
    for id in ["first", "branch", "last"] {
        mock.json(200, response("open_ai_responses", id, "4921"));
    }
    let client = client(&mock, profile("open_ai_responses"));
    let mut session = ConversationSession::new();
    let mut request = request();
    let first = complete(&mut session, &client, &request, &options()).await;
    let reference = first.continuation.clone();
    followup(&mut request, first);
    let mut branch = request.clone();
    branch.messages = vec![ConversationMessage::user_text("branch")];
    branch.continuation = reference;
    complete(&mut session, &client, &branch, &options()).await;
    complete(&mut session, &client, &request, &options()).await;
    assert_eq!(mock.bodies()[1]["previous_response_id"], "first");
    assert!(mock.bodies()[2].get("previous_response_id").is_none());
}

#[tokio::test]
async fn session_preparation_future_is_send() {
    fn assert_send(_: impl Send) {}
    let mock = Arc::new(Mock::default());
    let client = client(&mock, profile("open_ai_responses"));
    let mut draft = client
        .prepare_draft_on("one", &request(), &options(), RequestMode::Complete)
        .await
        .unwrap();
    let mut session = ConversationSession::new();
    assert_send(session.prepare(&mut draft));
}

#[tokio::test]
async fn unsupported_thread_downgrades_once_and_stays_unthreaded_for_that_identity() {
    let mock = Arc::new(Mock::default());
    mock.json(400,json!({"error":{"type":"invalid_request_error","message":"organization setting","details":{"error_code":"thread_unsupported_request","unsupported":["hipaa"]}}}));
    mock.json(200, response("anthropic_messages", "first", "4921"));
    mock.json(200, response("anthropic_messages", "second", "4921"));
    let client = client(&mock, profile("anthropic_messages"));
    let mut session = ConversationSession::new();
    let mut request = request();
    let first = complete(&mut session, &client, &request, &options()).await;
    followup(&mut request, first);
    complete(&mut session, &client, &request, &options()).await;
    let bodies = mock.bodies();
    assert_eq!(bodies.len(), 3);
    assert!(bodies[0].get("thread").is_some());
    assert!(bodies[1].get("thread").is_none());
    assert!(bodies[2].get("thread").is_none());
}

#[tokio::test]
async fn refusal_truncation_and_fallback_never_seed_a_continuation() {
    for reason in ["refusal", "max_tokens", "fallback"] {
        let mock = Arc::new(Mock::default());
        let mut first = response("anthropic_messages", "first", "4921");
        if reason == "fallback" {
            first["usage"]["iterations"] = json!([{"type":"fallback_message","model":"other","input_tokens":5,"output_tokens":2}]);
        } else {
            first["stop_reason"] = json!(reason);
        }
        mock.json(200, first);
        mock.json(200, response("anthropic_messages", "second", "4921"));
        let client = client(&mock, profile("anthropic_messages"));
        let mut session = ConversationSession::new();
        let mut request = request();
        let first = complete(&mut session, &client, &request, &options()).await;
        followup(&mut request, first);
        complete(&mut session, &client, &request, &options()).await;
        assert_eq!(mock.bodies()[1]["thread"]["type"], "create", "{reason}");
    }
}

#[tokio::test]
async fn retired_stream_observer_cannot_overwrite_a_newer_completed_chain() {
    let mock = Arc::new(Mock::default());
    mock.raw(200,sse(&[json!({"type":"response.created","response":{"id":"old","model":"m"}}),json!({"type":"response.completed","response":response("open_ai_responses","old","old")})]));
    mock.json(200, response("open_ai_responses", "new", "4921"));
    mock.json(200, response("open_ai_responses", "last", "4921"));
    let client = client(&mock, profile("open_ai_responses"));
    let mut session = ConversationSession::new();
    let mut request = request();
    let mut old = session
        .stream_in(
            &client.snapshot(),
            "one",
            &request,
            &options(),
            || Ok(()),
            |_| {},
        )
        .await
        .unwrap();
    request.messages[0] = ConversationMessage::user_text("new branch");
    let newer = complete(&mut session, &client, &request, &options()).await;
    while let Some(event) = old.next().await {
        event.unwrap();
    }
    followup(&mut request, newer);
    complete(&mut session, &client, &request, &options()).await;
    assert_eq!(mock.bodies()[2]["previous_response_id"], "new");
}

#[tokio::test]
async fn exact_json_metadata_and_utf16_survive_incremental_preparation() {
    let mock = Arc::new(Mock::default());
    mock.json(200, response("open_ai_responses", "first", "ok"));
    let client = client(&mock, profile("open_ai_responses"));
    let mut session = ConversationSession::new();
    let mut request = request();
    request.metadata =
        JsonPayload::parse(r#"{"opaque":"\udfff","integer":9007199254740993}"#).unwrap();
    let mut draft = client
        .prepare_draft_on("one", &request, &options(), RequestMode::Complete)
        .await
        .unwrap();
    let source = std::str::from_utf8(&draft.request().body)
        .unwrap()
        .replace("remember 4921", r#"\ud800"#);
    draft.set_body_bytes(source.into_bytes().into());
    session.prepare(&mut draft).await.unwrap();
    let collected = session
        .dispatch(draft.seal().await.unwrap(), || Ok(()))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let first = collected.decode().unwrap();
    collected.finish().await;
    followup(&mut request, first);
    let mut draft = client
        .prepare_draft_on("one", &request, &options(), RequestMode::Complete)
        .await
        .unwrap();
    let source = std::str::from_utf8(&draft.request().body)
        .unwrap()
        .replace("remember 4921", r#"\ud800"#);
    draft.set_body_bytes(source.into_bytes().into());
    let metadata = draft
        .semantic_body_source()
        .unwrap()
        .payload_at("/metadata")
        .unwrap();
    session.prepare(&mut draft).await.unwrap();
    assert_eq!(
        session.last_request().mode,
        ConversationRequestMode::Continue
    );
    let wire = JsonPayload::parse(std::str::from_utf8(&draft.request().body).unwrap()).unwrap();
    assert!(metadata
        .semantically_eq(&wire.payload_at("/metadata").unwrap())
        .unwrap());
    assert!(std::str::from_utf8(&draft.request().body)
        .unwrap()
        .contains(r#"\udfff"#));
}

#[tokio::test]
async fn glm_single_content_object_is_decoded_and_can_continue() {
    let mock = Arc::new(Mock::default());
    let mut first = response("open_ai_responses", "first", "4921");
    first["output"][0]["content"] = first["output"][0]["content"][0].clone();
    mock.json(200, first);
    mock.json(200, response("open_ai_responses", "second", "4921"));
    let mut p = profile("open_ai_responses");
    p.provider_id = "zhipu".into();
    p.base_url = "https://open.bigmodel.cn/api/v1".into();
    p.extra = json!({"supports_previous_response_id":true,"response_store_default":false,"responses_contract":"glm"});
    let client = client(&mock, p);
    let mut session = ConversationSession::new();
    let mut request = request();
    request.controls.responses.store = Some(true);
    let first = complete(&mut session, &client, &request, &options()).await;
    assert!(!first.message.content.is_empty());
    followup(&mut request, first);
    complete(&mut session, &client, &request, &options()).await;
    assert_eq!(
        session.last_request().mode,
        ConversationRequestMode::Continue
    );
}

#[tokio::test]
async fn changing_transports_never_reuses_another_clients_history_reference() {
    let first_http = Arc::new(Mock::default());
    first_http.json(200, response("open_ai_responses", "first", "4921"));
    let second_http = Arc::new(Mock::default());
    second_http.json(200, response("open_ai_responses", "second", "4921"));
    let first_client = client(&first_http, profile("open_ai_responses"));
    let second_client = client(&second_http, profile("open_ai_responses"));
    let mut session = ConversationSession::new();
    let mut request = request();
    let first = complete(&mut session, &first_client, &request, &options()).await;
    followup(&mut request, first);
    complete(&mut session, &second_client, &request, &options()).await;
    assert!(second_http.bodies()[0]
        .get("previous_response_id")
        .is_none());
}

#[tokio::test]
async fn prepared_deltas_update_content_length_and_reject_caller_edits() {
    let mock = Arc::new(Mock::default());
    mock.json(200, response("open_ai_responses", "first", "4921"));
    let client = client(&mock, profile("open_ai_responses"));
    let mut session = ConversationSession::new();
    let mut request = request();
    let first = complete(&mut session, &client, &request, &options()).await;
    followup(&mut request, first);
    let mut draft = client
        .prepare_draft_on("one", &request, &options(), RequestMode::Complete)
        .await
        .unwrap();
    let full_length = draft.request().body.len();
    draft
        .request_mut()
        .headers
        .push(("Content-Length".into(), full_length.to_string()));
    session.prepare(&mut draft).await.unwrap();
    assert_eq!(
        draft
            .request()
            .headers
            .iter()
            .find(|(name, _)| name == "Content-Length")
            .unwrap()
            .1,
        draft.request().body.len().to_string()
    );
    assert!(draft.request().body.len() < full_length);
    draft
        .request_mut()
        .headers
        .push(("x-caller-edit".into(), "changed".into()));
    let call = draft.seal().await.unwrap();
    assert!(session.dispatch(call, || Ok(())).await.is_err());
    assert_eq!(mock.bodies().len(), 1);
}
