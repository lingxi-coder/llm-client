//! Gate 32: a request walks `ResolvedRoute::connection_chain`, and gate 30's
//! half that this crate can hold on its own — nothing here knows a provider by
//! name, so two profiles that differ only in settings behave differently.
//!
//! The fakes are a `Transport` and a `WireCodec`, per the review: a fake
//! provider is built at the transport/codec layer, never by stubbing the
//! client itself.

use async_trait::async_trait;
use bytes::Bytes;
use futures::executor::block_on;
use futures::stream;
use lingxi_agent_api::protocol::{
    CompletionRequest, CompletionResponse, ContentBlock, ConversationMessage, LlmError,
    MessageRole, ProtocolFamily, ProviderProfile, ResponseId, StopReason, StreamEvent, ToolChoice,
    Usage,
};
use lingxi_llm_client::{
    HttpRequest, HttpResponse, LlmClientBuilder, RequestOptions, ResolveError, ResolvedRoute,
    StreamDecoder, StreamResponse, Transport, WebSocketSession, WireCodec,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// --- fakes -----------------------------------------------------------------

/// Answers per URL: the script says what each base URL does. Records the order
/// it was called in, which is the thing gate 32 is actually about.
#[derive(Default)]
struct ScriptedTransport {
    answers: Mutex<Vec<(String, Result<u16, LlmError>)>>,
    seen: Mutex<Vec<String>>,
    calls: AtomicUsize,
    seen_auth: Mutex<Vec<Option<String>>>,
    seen_timeouts: Mutex<Vec<Option<std::time::Duration>>>,
}

impl ScriptedTransport {
    fn new(answers: Vec<(&str, Result<u16, LlmError>)>) -> Arc<Self> {
        Arc::new(Self {
            answers: Mutex::new(
                answers
                    .into_iter()
                    .map(|(u, r)| (u.to_owned(), r))
                    .collect(),
            ),
            ..Default::default()
        })
    }

    fn answer_for(&self, url: &str) -> Result<HttpResponse, LlmError> {
        self.seen.lock().unwrap().push(url.to_owned());
        self.calls.fetch_add(1, Ordering::SeqCst);
        let answers = self.answers.lock().unwrap();
        let hit = answers
            .iter()
            .find(|(u, _)| url.starts_with(u.as_str()))
            .map(|(_, r)| r.clone());
        match hit {
            Some(Ok(status)) => Ok(HttpResponse {
                status,
                // Real providers report quota state in headers on the success
                // path too, so the fake does the same: it is the only way to
                // show they survive to the caller of a *stream*.
                headers: vec![
                    (
                        "Anthropic-RateLimit-Unified-5h-Utilization".to_owned(),
                        "0.42".to_owned(),
                    ),
                    ("retry-after".to_owned(), "7".to_owned()),
                ],
                body: Bytes::from_static(b"{\"text\":\"hi\"}"),
            }),
            Some(Err(e)) => Err(e),
            None => Err(LlmError::Transport {
                message: format!("no script for {url}"),
            }),
        }
    }

    fn hops(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }

    fn auth_headers(&self) -> Vec<Option<String>> {
        self.seen_auth.lock().unwrap().clone()
    }

    fn timeouts(&self) -> Vec<Option<std::time::Duration>> {
        self.seen_timeouts.lock().unwrap().clone()
    }

    fn record_request(&self, req: &HttpRequest) {
        self.seen_auth.lock().unwrap().push(
            req.headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                .map(|(_, value)| value.clone()),
        );
        self.seen_timeouts.lock().unwrap().push(req.timeout);
    }
}

#[async_trait]
impl Transport for ScriptedTransport {
    async fn execute(&self, req: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.record_request(&req);
        self.answer_for(&req.url)
    }
    async fn open_stream(&self, req: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.record_request(&req);
        let answered = self.answer_for(&req.url)?;
        let frames: Vec<Result<Bytes, LlmError>> = vec![
            Ok(Bytes::from_static(b"one")),
            Ok(Bytes::from_static(b"two")),
        ];
        Ok(StreamResponse {
            status: answered.status,
            headers: answered.headers,
            body: Box::pin(stream::iter(frames)),
        })
    }
    async fn open_responses_websocket_session(
        &self,
        _req: HttpRequest,
    ) -> Result<Box<dyn WebSocketSession>, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "no websockets in this fake".to_owned(),
        })
    }
}

/// Encodes to `<base_url>/chat`, decodes 200 to one text block. One frame
/// decodes to one `TextDelta`, so a test can count frames as events.
struct FakeCodec;

#[async_trait]
impl WireCodec for FakeCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::OpenAiChat
    }
    fn encode_request(
        &self,
        req: &CompletionRequest,
        profile: &ProviderProfile,
        route: &ResolvedRoute,
        _opts: &RequestOptions,
    ) -> Result<HttpRequest, LlmError> {
        // The endpoint comes from the connection's own profile, which is why
        // the codec is handed one: it is registered per protocol family and
        // shared by every profile using it.
        let base = &profile.base_url;
        Ok(HttpRequest {
            method: "POST".to_owned(),
            url: format!("{base}/chat"),
            headers: vec![],
            body: Bytes::from(
                serde_json::to_vec(
                    &json!({ "model": route.request_model, "n": req.messages.len() }),
                )
                .unwrap(),
            ),
            timeout: None,
        })
    }
    fn decode_response(&self, resp: &HttpResponse) -> Result<CompletionResponse, LlmError> {
        if resp.status != 200 {
            return Err(LlmError::ProviderInternal {
                message: format!("status {}", resp.status),
            });
        }
        Ok(CompletionResponse {
            web_search: None,
            message: ConversationMessage {
                role: MessageRole::Assistant,
                content: vec![ContentBlock::Text {
                    text: "hi".to_owned(),
                    thought_signature: None,
                }],
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
            model: "fake".to_owned(),
            response_id: None,
            executed_profile: None,
        })
    }
    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(FakeDecoder { blocks: 0 })
    }
    fn response_usage(&self, _resp: &HttpResponse) -> Option<Usage> {
        None
    }
}

/// The same fake on the wire that keeps turn state at the endpoint. Only the
/// family differs — so a test using it isolates what the *client* does with a
/// continuation, not what a real encoder does.
struct FakeStatefulCodec;

#[async_trait]
impl WireCodec for FakeStatefulCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::OpenAiResponses
    }
    fn encode_request(
        &self,
        req: &CompletionRequest,
        profile: &ProviderProfile,
        route: &ResolvedRoute,
        opts: &RequestOptions,
    ) -> Result<HttpRequest, LlmError> {
        FakeCodec.encode_request(req, profile, route, opts)
    }
    fn decode_response(&self, resp: &HttpResponse) -> Result<CompletionResponse, LlmError> {
        FakeCodec.decode_response(resp)
    }
    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        FakeCodec.stream_decoder()
    }
    fn response_usage(&self, resp: &HttpResponse) -> Option<Usage> {
        FakeCodec.response_usage(resp)
    }
}

struct FakeDecoder {
    blocks: usize,
}

impl StreamDecoder for FakeDecoder {
    fn decode_frame(&mut self, frame: &[u8]) -> Result<Vec<StreamEvent>, LlmError> {
        let block = self.blocks;
        self.blocks += 1;
        Ok(vec![StreamEvent::TextDelta {
            block,
            text: String::from_utf8_lossy(frame).into_owned(),
        }])
    }
    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        Ok(vec![StreamEvent::End {
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }])
    }
    fn observed_usage(&self) -> Option<Usage> {
        None
    }

    fn usage_is_complete(&self) -> bool {
        false
    }
    fn set_provider_metadata(&mut self, _meta: Value) {}
}

// --- fixtures --------------------------------------------------------------

/// A provider entry written the way a user writes settings. No code in this
/// crate mentions any of these names (gate 30).
#[allow(clippy::too_many_arguments)]
fn conn(
    profile_name: &str,
    base: &str,
    models: Value,
    group: Option<&str>,
    order: u32,
    billing: &str,
    hidden: bool,
) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme",
        "profile_name": profile_name,
        "base_url": base,
        "protocol": "open_ai_chat",
        "auth": "none",
        "models": models,
        "pricing": { "billingMode": billing },
        "connection": {
            "group": group,
            "order": order,
            "hidden": hidden,
            "failover": { "rateLimit": true, "overloaded": true, "serverError": true, "network": true, "auth": true },
        },
    }))
    .expect("profile fixture parses")
}

/// A standalone profile: its own one-connection group, no failover.
fn solo(profile_name: &str, base: &str, models: Value) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme",
        "profile_name": profile_name,
        "base_url": base,
        "protocol": "open_ai_chat",
        "auth": "none",
        "models": models,
    }))
    .expect("profile fixture parses")
}

fn model(display: &str, wire: &str) -> Value {
    json!([{
        "display_model": display,
        "request_model": wire,
        "billing_model": wire,
        "metadata": { "contextWindowTokens": 128000 },
    }])
}

fn client(
    profiles: &[ProviderProfile],
    http: Arc<ScriptedTransport>,
) -> lingxi_llm_client::LlmClient {
    let mut b = LlmClientBuilder::with_transport(http, profiles);
    b.register_codec(Arc::new(FakeCodec));
    b.build().expect("every profile's protocol has a codec")
}

fn request(model: &str) -> CompletionRequest {
    CompletionRequest {
        model: model.to_owned(),
        web_search: None,
        previous_response_id: None,
        system: vec![],
        messages: vec![ConversationMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: "hi".to_owned(),
                thought_signature: None,
            }],
        }],
        tools: vec![],
        tool_choice: ToolChoice::auto(),
        max_tokens: None,
        temperature: None,
        thinking: None,
        stop_sequences: vec![],
        metadata: Value::Null,
    }
}

// --- resolve ---------------------------------------------------------------

#[test]
fn siblings_of_one_group_are_the_chain_in_connection_order() {
    let c = client(
        &[
            conn(
                "acme:intl",
                "https://intl.test",
                model("m-1", "m-1"),
                Some("acme"),
                1,
                "per_token",
                false,
            ),
            conn(
                "acme:cn",
                "https://cn.test",
                model("m-1", "m-1"),
                Some("acme"),
                0,
                "per_token",
                false,
            ),
        ],
        ScriptedTransport::new(vec![]),
    );

    let route = c.resolve("m-1").unwrap();

    assert_eq!(route.profile_name, "acme:cn", "lowest order starts");
    assert_eq!(
        route
            .connection_chain
            .iter()
            .map(|h| h.profile_name.as_str())
            .collect::<Vec<_>>(),
        vec!["acme:intl"],
        "the rest of the group is the chain, and the head is not in it"
    );
}

#[test]
fn scoping_to_one_connection_still_offers_the_whole_group() {
    let c = client(
        &[
            conn(
                "acme:cn",
                "https://cn.test",
                model("m-1", "m-1"),
                Some("acme"),
                0,
                "per_token",
                false,
            ),
            conn(
                "acme:intl",
                "https://intl.test",
                model("m-1", "m-1"),
                Some("acme"),
                1,
                "per_token",
                false,
            ),
        ],
        ScriptedTransport::new(vec![]),
    );

    // A picker hands back a connection, so a session stores `acme:intl` and
    // every later request arrives scoped to it. Scoping picks where to start;
    // it does not decide whether the rest of the group may be used.
    let route = c.resolve_in("m-1", Some("acme:intl")).unwrap();

    assert_eq!(route.profile_name, "acme:intl");
    assert_eq!(
        route.connection_chain.len(),
        1,
        "the chain comes from the head's group, not from what was in scope"
    );
}

#[test]
fn a_hop_must_serve_the_same_wire_model() {
    let c = client(
        &[
            conn(
                "acme:a",
                "https://a.test",
                model("m-1", "wire-1"),
                Some("acme"),
                0,
                "per_token",
                false,
            ),
            conn(
                "acme:b",
                "https://b.test",
                model("m-1", "wire-2"),
                Some("acme"),
                1,
                "per_token",
                false,
            ),
        ],
        ScriptedTransport::new(vec![]),
    );

    let route = c.resolve("m-1").unwrap();

    assert_eq!(route.request_model, "wire-1");
    assert!(
        route.connection_chain.is_empty(),
        "failing over re-points the endpoint, not the model: a sibling serving a \
         different wire id would silently answer as something else"
    );
}

#[test]
fn a_hop_must_be_billed_the_same_way() {
    let c = client(
        &[
            conn(
                "zhipu:plan",
                "https://plan.test",
                model("glm", "glm"),
                Some("zhipu"),
                0,
                "subscription",
                false,
            ),
            conn(
                "zhipu:metered",
                "https://metered.test",
                model("glm", "glm"),
                Some("zhipu"),
                1,
                "per_token",
                false,
            ),
        ],
        ScriptedTransport::new(vec![]),
    );

    let route = c.resolve("glm").unwrap();

    assert_eq!(route.profile_name, "zhipu:plan");
    assert!(
        route.connection_chain.is_empty(),
        "moving a rate-limited subscription request onto the metered endpoint \
         would start charging real money for what the plan already covers"
    );
}

#[test]
fn a_hidden_connection_is_reachable_by_failover_but_never_offered() {
    let c = client(
        &[
            conn(
                "acme:key1",
                "https://one.test",
                model("m-1", "m-1"),
                Some("acme"),
                0,
                "per_token",
                false,
            ),
            conn(
                "acme:key2",
                "https://two.test",
                model("m-1", "m-1"),
                Some("acme"),
                1,
                "per_token",
                true,
            ),
        ],
        ScriptedTransport::new(vec![]),
    );

    assert_eq!(
        c.models().iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        vec!["m-1"],
        "the spare key slot is not something a user can pick"
    );
    assert_eq!(
        c.resolve("m-1").unwrap().connection_chain.len(),
        1,
        "but it is still there to fail over onto"
    );
}

#[test]
fn matching_across_two_groups_is_a_real_ambiguity() {
    let c = client(
        &[
            solo("openai", "https://a.test", model("gpt", "gpt")),
            solo("azure", "https://b.test", model("gpt", "gpt")),
        ],
        ScriptedTransport::new(vec![]),
    );

    match c.resolve("gpt").unwrap_err() {
        ResolveError::AmbiguousAcrossGroups { groups, .. } => {
            assert_eq!(groups, vec!["azure".to_owned(), "openai".to_owned()]);
        }
        other => panic!("expected an ambiguity, got {other:?}"),
    }

    // The qualified form a picker shows routes.
    let route = c.resolve("azure/gpt").unwrap();
    assert_eq!(route.profile_name, "azure");
}

#[test]
fn two_models_of_one_profile_answering_to_one_name_is_an_error() {
    let c = client(
        &[solo(
            "acme",
            "https://a.test",
            json!([
                {"display_model": "a", "request_model": "wire-a", "billing_model": "wire-a", "aliases": ["dup"]},
                {"display_model": "b", "request_model": "wire-b", "billing_model": "wire-b", "aliases": ["dup"]},
            ]),
        )],
        ScriptedTransport::new(vec![]),
    );

    assert_eq!(
        c.resolve("dup").unwrap_err(),
        ResolveError::DuplicateOnProfile {
            model: "dup".to_owned(),
            profile_name: "acme".to_owned(),
        },
        "two models on one endpoint: which to send is genuinely unknown"
    );
}

#[test]
fn an_unknown_model_says_so() {
    let c = client(
        &[solo("acme", "https://a.test", model("m-1", "m-1"))],
        ScriptedTransport::new(vec![]),
    );
    assert_eq!(
        c.resolve("nope").unwrap_err(),
        ResolveError::UnknownModel {
            model: "nope".to_owned()
        }
    );
}

// --- gate 32: the walk -----------------------------------------------------

/// The same two connections as `pair`, on the wire that keeps turn state at
/// the endpoint, with the continuation opt-in the encoder requires.
fn stateful_pair(http: Arc<ScriptedTransport>) -> lingxi_llm_client::LlmClient {
    let responses = |name: &str, base: &str, order: u32| -> ProviderProfile {
        let mut p = conn(
            name,
            base,
            model("m-1", "m-1"),
            Some("acme"),
            order,
            "per_token",
            false,
        );
        p.protocol = ProtocolFamily::OpenAiResponses;
        p.extra = json!({"supports_previous_response_id": true});
        p
    };
    let profiles = [
        responses("acme:one", "https://one.test", 0),
        responses("acme:two", "https://two.test", 1),
    ];
    let mut b = LlmClientBuilder::with_transport(http, &profiles);
    b.register_codec(Arc::new(FakeStatefulCodec));
    b.build().expect("every profile's protocol has a codec")
}

fn continuing(model: &str) -> CompletionRequest {
    let mut req = request(model);
    req.previous_response_id = Some(ResponseId::new("resp_previous"));
    req
}

/// A continuation id is state the *endpoint* holds, so the second connection
/// has never heard of it. Failing over would send a turn that silently loses
/// everything before it — the one failure a retry is supposed to prevent.
///
/// Stated against the same script as the plain failover test above: the only
/// difference between the two is the continuation, so this cannot pass by the
/// chain being empty.
#[test]
fn a_continuation_does_not_fail_over_to_a_connection_that_never_saw_it() {
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::Overloaded {
                message: "busy".to_owned(),
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let c = stateful_pair(http.clone());

    let err = block_on(c.complete(&continuing("m-1"), &RequestOptions::default()))
        .expect_err("the connection holding the state is the only one that can serve this");
    assert!(matches!(err, LlmError::Overloaded { .. }), "{err}");
    assert_eq!(
        http.hops(),
        vec!["https://one.test/chat"],
        "the sibling was never tried; it does not have the previous response"
    );

    // The same script, without the continuation, walks both — so the stop
    // above is the continuation and not the route.
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::Overloaded {
                message: "busy".to_owned(),
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let c = stateful_pair(http.clone());
    block_on(c.complete(&request("m-1"), &RequestOptions::default())).unwrap();
    assert_eq!(http.hops().len(), 2);
}

/// A wire with no endpoint-side turn state cannot continue one. Refusing at
/// the client is the layer that matters: a codec that never looks at the field
/// would send a fresh turn, and the caller would be told nothing.
#[test]
fn a_continuation_on_a_wire_that_has_no_such_state_is_refused_before_it_is_sent() {
    let http = ScriptedTransport::new(vec![("https://one.test", Ok(200))]);
    let c = pair(http.clone());

    let err = block_on(c.complete(&continuing("m-1"), &RequestOptions::default()))
        .expect_err("this wire keeps no turn state to continue");
    match err {
        LlmError::UnsupportedCapability { message } => assert!(
            message.contains("acme:one"),
            "the refusal must name the connection: {message}"
        ),
        other => panic!("{other}"),
    }
    assert!(
        http.hops().is_empty(),
        "nothing may go on the wire before the refusal"
    );
}

fn pair(http: Arc<ScriptedTransport>) -> lingxi_llm_client::LlmClient {
    client(
        &[
            conn(
                "acme:one",
                "https://one.test",
                model("m-1", "m-1"),
                Some("acme"),
                0,
                "per_token",
                false,
            ),
            conn(
                "acme:two",
                "https://two.test",
                model("m-1", "m-1"),
                Some("acme"),
                1,
                "per_token",
                false,
            ),
        ],
        http,
    )
}

#[test]
fn a_failover_trigger_moves_the_request_to_the_next_connection() {
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::Overloaded {
                message: "busy".to_owned(),
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let c = pair(http.clone());

    let resp = block_on(c.complete(&request("m-1"), &RequestOptions::default())).unwrap();

    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert_eq!(
        http.hops(),
        vec!["https://one.test/chat", "https://two.test/chat"],
        "both connections were tried, in order"
    );
}

#[test]
fn an_error_another_endpoint_would_repeat_stops_at_the_first_connection() {
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::ContextOverflow {
                message: "too long".to_owned(),
                limit: None,
                actual: None,
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let c = pair(http.clone());

    let err = block_on(c.complete(&request("m-1"), &RequestOptions::default())).unwrap_err();

    assert!(matches!(err, LlmError::ContextOverflow { .. }));
    assert_eq!(
        http.hops(),
        vec!["https://one.test/chat"],
        "a transcript that does not fit is not an endpoint problem; retrying it \
         elsewhere would only bury the reason"
    );
}

#[test]
fn the_last_error_survives_when_the_group_is_spent() {
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::Overloaded {
                message: "one busy".to_owned(),
            }),
        ),
        (
            "https://two.test",
            Err(LlmError::ProviderInternal {
                message: "two broke".to_owned(),
            }),
        ),
    ]);
    let c = pair(http.clone());

    let err = block_on(c.complete(&request("m-1"), &RequestOptions::default())).unwrap_err();

    assert_eq!(
        err,
        LlmError::ProviderInternal {
            message: "two broke".to_owned()
        },
        "the caller sees why the last connection failed, not the first"
    );
    assert_eq!(http.hops().len(), 2);
}

#[test]
fn streaming_walks_the_same_connections_and_decodes_through_the_codec() {
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::TransportTimeout {
                message: "no answer".to_owned(),
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let c = pair(http.clone());

    let events = block_on(async {
        let mut s = c
            .stream(
                &request("m-1"),
                &RequestOptions {
                    stream: true,
                    ..RequestOptions::default()
                },
            )
            .await
            .unwrap();
        let mut out = vec![];
        while let Some(ev) = s.next().await {
            out.push(ev.unwrap());
        }
        out
    });

    assert_eq!(http.hops().len(), 2, "the stream failed over too");
    assert_eq!(
        events.len(),
        3,
        "two frames decoded to two deltas, and `finish` emitted the End"
    );
    assert!(matches!(events[0], StreamEvent::TextDelta { block: 0, .. }));
    assert!(matches!(events[2], StreamEvent::End { .. }));
}

#[test]
fn a_standalone_profile_is_one_connection() {
    let http = ScriptedTransport::new(vec![(
        "https://one.test",
        Err(LlmError::Overloaded {
            message: "busy".to_owned(),
        }),
    )]);
    let c = client(
        &[solo("only", "https://one.test", model("m-1", "m-1"))],
        http.clone(),
    );

    let err = block_on(c.complete(&request("m-1"), &RequestOptions::default())).unwrap_err();

    assert!(matches!(err, LlmError::Overloaded { .. }));
    assert_eq!(
        http.hops().len(),
        1,
        "a trigger with nowhere to go is still just the error"
    );
}

/// The streaming path used to return frames and nothing else, so every response
/// header was unreachable on the one path an agent turn actually takes — a
/// streamed 429 could not even say how long to wait. `open_stream` carries the
/// status and headers now; this is what that buys.
#[test]
fn a_streamed_response_still_has_its_headers() {
    let http = ScriptedTransport::new(vec![("https://one.test", Ok(200))]);
    let c = pair(http.clone());

    let s = block_on(async {
        c.stream(
            &request("m-1"),
            &RequestOptions {
                stream: true,
                ..RequestOptions::default()
            },
        )
        .await
        .unwrap()
    });

    assert_eq!(s.status(), 200);
    assert_eq!(
        s.header("anthropic-ratelimit-unified-5h-utilization"),
        Some("0.42"),
        "header lookup is case-insensitive; the provider sent it capitalized"
    );
    assert_eq!(
        s.header("Retry-After"),
        Some("7"),
        "the header a streamed 429 needs, and could not reach before"
    );
    assert_eq!(s.header("x-absent"), None);
    assert_eq!(s.headers().len(), 2);
}

/// This crate does not hold, fetch or store credentials: the caller owns them
/// and hands one over per request. There is no credential store to reach into
/// and no `CredentialProvider` to consult — whatever the caller puts on
/// `RequestOptions` is exactly what the authenticator is given.
mod credentials_come_from_the_caller {
    use super::*;
    use std::sync::Mutex;

    /// Records the credential it was handed, so a test can prove which one
    /// arrived rather than that *some* request went out.
    struct Recording(Arc<Mutex<Vec<Option<String>>>>);

    #[async_trait]
    impl lingxi_llm_client::Authenticator for Recording {
        async fn apply(
            &self,
            _req: &mut HttpRequest,
            _profile: &ProviderProfile,
            credential: Option<&lingxi_agent_api::protocol::Secret<String>>,
        ) -> Result<(), LlmError> {
            self.0
                .lock()
                .unwrap()
                .push(credential.map(|s| s.expose_secret().to_owned()));
            Ok(())
        }
    }

    fn keyed_profile() -> ProviderProfile {
        serde_json::from_value(json!({
            "provider_id": "acme",
            "profile_name": "acme",
            "base_url": "https://one.test",
            "protocol": "open_ai_chat",
            "auth": "api_key",
            "models": [{"display_model": "m-1", "request_model": "m-1", "billing_model": "m-1"}],
        }))
        .expect("profile fixture parses")
    }

    #[test]
    fn the_authenticator_is_handed_the_caller_s_credential() {
        let seen = Arc::new(Mutex::new(vec![]));
        let http = ScriptedTransport::new(vec![("https://one.test", Ok(200))]);
        let profiles = [keyed_profile()];
        let mut b = LlmClientBuilder::with_transport(http, &profiles);
        b.register_codec(Arc::new(FakeCodec));
        b.register_authenticator(
            lingxi_agent_api::protocol::AuthStrategy::ApiKey,
            Arc::new(Recording(seen.clone())),
        );
        let c = b.build().expect("the profile's protocol has a codec");

        block_on(async {
            c.complete(
                &request("m-1"),
                &RequestOptions {
                    credential: Some(lingxi_agent_api::protocol::Secret::new(
                        "sk-from-the-caller".to_owned(),
                    )),
                    ..RequestOptions::default()
                },
            )
            .await
            .unwrap();
        });

        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[Some("sk-from-the-caller".to_owned())],
            "the secret the caller passed in, not one this crate went looking for"
        );
    }

    #[test]
    fn a_caller_that_passes_none_leaves_the_authenticator_with_none() {
        // There is nowhere else for one to come from. A crate that fell back to
        // an environment variable or a keychain would be storing credentials.
        let seen = Arc::new(Mutex::new(vec![]));
        let http = ScriptedTransport::new(vec![("https://one.test", Ok(200))]);
        let profiles = [keyed_profile()];
        let mut b = LlmClientBuilder::with_transport(http, &profiles);
        b.register_codec(Arc::new(FakeCodec));
        b.register_authenticator(
            lingxi_agent_api::protocol::AuthStrategy::ApiKey,
            Arc::new(Recording(seen.clone())),
        );
        let c = b.build().unwrap();

        block_on(async {
            c.complete(&request("m-1"), &RequestOptions::default())
                .await
                .unwrap();
        });

        assert_eq!(seen.lock().unwrap().as_slice(), &[None]);
    }
}

/// Failover re-points the endpoint, not the bill. The connection-level rule
/// already stops a subscription request sliding onto a metered endpoint; an
/// aggregator needs the same rule one level down, because its free tier and its
/// metered models share one endpoint and one key.
#[test]
fn a_free_model_does_not_fail_over_onto_a_metered_one() {
    let free = json!([{
        "display_model": "m-1", "request_model": "m-1", "billing_model": "m-1",
        "billing_mode": "free",
    }]);
    let metered = json!([{
        "display_model": "m-1", "request_model": "m-1", "billing_model": "m-1",
    }]);
    let profiles = [
        conn(
            "a",
            "https://one.test",
            free.clone(),
            Some("g"),
            0,
            "per_token",
            false,
        ),
        conn(
            "b",
            "https://two.test",
            metered,
            Some("g"),
            1,
            "per_token",
            false,
        ),
    ];
    let c = client(&profiles, ScriptedTransport::new(vec![]));
    let route = c.resolve("m-1").expect("the free model resolves");
    assert!(
        route.connection_chain.is_empty(),
        "the metered sibling serves the same wire model over the same key, and \
         falling onto it would start charging: {:?}",
        route.connection_chain
    );

    // Two free connections are interchangeable, which is what makes the check
    // above a billing rule rather than a ban on failover.
    let both_free = [
        conn(
            "a",
            "https://one.test",
            free.clone(),
            Some("g"),
            0,
            "per_token",
            false,
        ),
        conn(
            "b",
            "https://two.test",
            free,
            Some("g"),
            1,
            "per_token",
            false,
        ),
    ];
    let c = client(&both_free, ScriptedTransport::new(vec![]));
    let route = c.resolve("m-1").unwrap();
    assert_eq!(route.connection_chain.len(), 1);
}

/// Production HTTP delivers arbitrary byte chunks rather than parsed events.
mod raw_http_streams {
    use super::*;
    use std::collections::VecDeque;
    use std::time::Duration;

    struct RawTransport {
        responses: Mutex<VecDeque<StreamResponse>>,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Transport for RawTransport {
        async fn execute(&self, _: HttpRequest) -> Result<HttpResponse, LlmError> {
            unreachable!("stream tests do not make buffered requests")
        }
        async fn open_stream(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("scripted response"))
        }
        async fn open_responses_websocket_session(
            &self,
            _: HttpRequest,
        ) -> Result<Box<dyn WebSocketSession>, LlmError> {
            unreachable!("HTTP test")
        }
    }

    fn response(status: u16, body: &[u8], chunk_size: usize) -> StreamResponse {
        let chunks: Vec<_> = body
            .chunks(chunk_size)
            .map(|b| Ok(Bytes::copy_from_slice(b)))
            .collect();
        StreamResponse {
            status,
            headers: vec![
                (
                    "Content-Type".into(),
                    "Text/Event-Stream; charset=utf-8".into(),
                ),
                ("Retry-After".into(), "7".into()),
            ],
            body: Box::pin(stream::iter(chunks)),
        }
    }

    fn setup(
        responses: Vec<StreamResponse>,
        failover: bool,
    ) -> (lingxi_llm_client::LlmClient, Arc<RawTransport>) {
        let http = Arc::new(RawTransport {
            responses: Mutex::new(responses.into()),
            calls: AtomicUsize::new(0),
        });
        let mut profiles = vec![conn(
            "one",
            "https://one.test",
            model("m-1", "m-1"),
            Some("g"),
            0,
            "per_token",
            false,
        )];
        if failover {
            profiles.push(conn(
                "two",
                "https://two.test",
                model("m-1", "m-1"),
                Some("g"),
                1,
                "per_token",
                false,
            ));
        }
        (
            LlmClientBuilder::with_transport(http.clone(), &profiles)
                .build()
                .unwrap(),
            http,
        )
    }

    const SSE: &[u8] = concat!(
        ": keepalive\r\n\r\n",
        "event: message\r\ndata: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"你好\"}}]}\r\n\r\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]"
    ).as_bytes();

    #[test]
    fn sse_decodes_at_every_byte_boundary_and_coalesced_events() {
        for chunk_size in [1, 2, 7, SSE.len()] {
            let (client, _) = setup(vec![response(200, SSE, chunk_size)], false);
            block_on(async {
                let mut stream = client
                    .stream(&request("m-1"), &RequestOptions::default())
                    .await
                    .unwrap();
                let mut text = String::new();
                let mut ends = 0;
                while let Some(event) = stream.next().await {
                    match event.unwrap() {
                        StreamEvent::TextDelta { text: delta, .. } => text.push_str(&delta),
                        StreamEvent::End { .. } => ends += 1,
                        _ => {}
                    }
                }
                assert_eq!(text, "你好", "chunk size {chunk_size}");
                assert_eq!(ends, 1);
            });
        }
    }

    #[test]
    fn http_auth_and_rate_limit_statuses_fail_over_before_returning_a_stream() {
        for status in [401, 429] {
            let (client, http) = setup(
                vec![
                    response(status, br#"{"error":{"message":"try next"}}"#, 3),
                    response(200, SSE, 4),
                ],
                true,
            );
            block_on(async {
                let mut stream = client
                    .stream(&request("m-1"), &RequestOptions::default())
                    .await
                    .unwrap();
                assert_eq!(stream.status(), 200);
                while let Some(event) = stream.next().await {
                    event.unwrap();
                }
            });
            assert_eq!(http.calls.load(Ordering::SeqCst), 2);
        }
    }

    #[test]
    fn exhausted_http_errors_keep_the_provider_message_and_retry_after() {
        for status in [401, 429] {
            let (client, _) = setup(
                vec![response(
                    status,
                    br#"{"error":{"message":"quota unavailable"}}"#,
                    2,
                )],
                false,
            );
            let result = block_on(client.stream(&request("m-1"), &RequestOptions::default()));
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("HTTP error returned a stream"),
            };
            match error {
                LlmError::RateLimited {
                    message,
                    retry_after,
                } if status == 429 => {
                    assert!(message.contains("quota unavailable"));
                    assert_eq!(retry_after, Some(Duration::from_secs(7)));
                }
                LlmError::Authentication { message } if status == 401 => {
                    assert!(message.contains("quota unavailable"))
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn oversized_error_body_is_bounded_and_does_not_poll_the_tail() {
        let mut resp = response(429, &vec![b'x'; 64 * 1024], 4096);
        use futures::StreamExt;
        resp.body = Box::pin(resp.body.chain(stream::once(async {
            panic!("error bodies must not be collected past the limit");
            #[allow(unreachable_code)]
            Ok(Bytes::new())
        })));
        let (client, _) = setup(vec![resp], false);
        assert!(matches!(
            block_on(client.stream(&request("m-1"), &RequestOptions::default())),
            Err(LlmError::RateLimited { .. })
        ));
    }
}

#[test]
fn qualified_profile_wins_over_its_namesake_group() {
    let c = client(
        &[
            conn(
                "acme",
                "https://primary.test",
                model("m", "primary-model"),
                Some("acme"),
                1,
                "per_token",
                false,
            ),
            conn(
                "spare",
                "https://spare.test",
                model("m", "other-model"),
                Some("acme"),
                0,
                "per_token",
                false,
            ),
        ],
        ScriptedTransport::new(vec![]),
    );
    for scope in [None, Some("acme")] {
        let route = c.resolve_in("acme/m", scope).unwrap();
        assert_eq!(route.profile_name, "acme");
        assert_eq!(route.request_model, "primary-model");
    }
}

struct ModeCheckingCodec(bool);

#[async_trait]
impl WireCodec for ModeCheckingCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::OpenAiChat
    }
    fn encode_request(
        &self,
        req: &CompletionRequest,
        profile: &ProviderProfile,
        route: &ResolvedRoute,
        opts: &RequestOptions,
    ) -> Result<HttpRequest, LlmError> {
        assert_eq!(
            opts.stream, self.0,
            "client method must determine wire mode"
        );
        FakeCodec.encode_request(req, profile, route, opts)
    }
    fn decode_response(&self, resp: &HttpResponse) -> Result<CompletionResponse, LlmError> {
        FakeCodec.decode_response(resp)
    }
    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        FakeCodec.stream_decoder()
    }
    fn response_usage(&self, resp: &HttpResponse) -> Option<Usage> {
        FakeCodec.response_usage(resp)
    }
}

#[test]
fn client_methods_determine_wire_mode() {
    for streaming in [false, true] {
        let http = ScriptedTransport::new(vec![("https://x.test", Ok(200))]);
        let mut builder = LlmClientBuilder::with_transport(
            http,
            &[solo("acme", "https://x.test", model("m", "m"))],
        );
        builder.register_codec(Arc::new(ModeCheckingCodec(streaming)));
        let client = builder.build().unwrap();
        let opts = RequestOptions {
            stream: !streaming,
            ..Default::default()
        };
        if streaming {
            assert!(block_on(client.stream(&request("m"), &opts)).is_ok());
        } else {
            assert!(block_on(client.complete(&request("m"), &opts)).is_ok());
        }
        assert_eq!(opts.stream, !streaming, "caller options stay unchanged");
    }
}

#[test]
fn a_fallback_without_its_own_credential_is_never_sent() {
    use lingxi_agent_api::protocol::Secret;

    let mut primary = conn(
        "primary",
        "https://one.test",
        model("m", "m"),
        Some("g"),
        0,
        "per_token",
        false,
    );
    let mut secondary = conn(
        "secondary",
        "https://two.test",
        model("m", "m"),
        Some("g"),
        1,
        "per_token",
        true,
    );
    primary.auth = lingxi_agent_api::protocol::AuthStrategy::ApiKey;
    secondary.auth = lingxi_agent_api::protocol::AuthStrategy::ApiKey;
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::Overloaded {
                message: "busy".into(),
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let client = client(&[primary, secondary], http.clone());
    let opts = RequestOptions {
        credential: Some(Secret::new("primary-secret".to_owned())),
        ..Default::default()
    };
    assert!(block_on(client.complete(&request("m"), &opts)).is_err());
    assert_eq!(http.hops(), vec!["https://one.test/chat"]);
}

#[test]
fn a_fallback_uses_its_explicit_credential_and_reports_the_actual_profile() {
    use lingxi_agent_api::protocol::{Secret, Submission, TokenPricing};
    use std::collections::BTreeMap;

    let mut primary = conn(
        "primary",
        "https://one.test",
        model("m", "m"),
        Some("g"),
        0,
        "per_token",
        false,
    );
    let mut secondary = conn(
        "secondary",
        "https://two.test",
        model("m", "m"),
        Some("g"),
        1,
        "per_token",
        true,
    );
    primary.auth = lingxi_agent_api::protocol::AuthStrategy::ApiKey;
    secondary.auth = lingxi_agent_api::protocol::AuthStrategy::ApiKey;
    primary.models[0].pricing = Some(TokenPricing {
        input_per_million: Some(1.0),
        ..Default::default()
    });
    secondary.models[0].pricing = Some(TokenPricing {
        input_per_million: Some(2.0),
        ..Default::default()
    });
    let http = ScriptedTransport::new(vec![
        (
            "https://one.test",
            Err(LlmError::Overloaded {
                message: "busy".into(),
            }),
        ),
        ("https://two.test", Ok(200)),
    ]);
    let client = client(&[primary, secondary], http.clone());
    let opts = RequestOptions {
        credential: Some(Secret::new("primary-secret".to_owned())),
        fallback_credentials: BTreeMap::from([(
            "secondary".to_owned(),
            Secret::new("secondary-secret".to_owned()),
        )]),
        ..Default::default()
    };
    let response = block_on(client.complete(&request("m"), &opts)).unwrap();
    assert_eq!(response.executed_profile.as_deref(), Some("secondary"));
    let route = client.resolve("m").unwrap();
    let usage = Usage {
        input_tokens: 1_000_000,
        ..Default::default()
    };
    assert_eq!(
        client
            .estimate_cost(&route, &usage, Submission::Interactive)
            .unwrap()
            .unwrap()
            .total_usd,
        1.0
    );
    assert_eq!(
        client
            .estimate_actual_cost(&route, &response, &usage, Submission::Interactive)
            .unwrap()
            .unwrap()
            .total_usd,
        2.0
    );
    assert_eq!(
        http.hops(),
        vec!["https://one.test/chat", "https://two.test/chat"]
    );
    assert_eq!(
        http.auth_headers(),
        vec![
            Some("Bearer primary-secret".into()),
            Some("Bearer secondary-secret".into())
        ]
    );
    let stream = block_on(client.stream(&request("m"), &opts)).unwrap();
    assert_eq!(stream.executed_profile(), "secondary");
    assert_eq!(
        client
            .estimate_cost_for_profile(
                &route,
                stream.executed_profile(),
                &usage,
                Submission::Interactive
            )
            .unwrap()
            .unwrap()
            .total_usd,
        2.0
    );
}

#[test]
fn request_timeout_defaults_to_120_seconds_and_can_be_overridden() {
    use std::time::Duration;
    let http = ScriptedTransport::new(vec![("https://one.test", Ok(200))]);
    let client = client(
        &[solo("only", "https://one.test", model("m", "m"))],
        http.clone(),
    );
    block_on(client.complete(&request("m"), &RequestOptions::default())).unwrap();
    block_on(client.complete(
        &request("m"),
        &RequestOptions {
            total_timeout: Some(Duration::from_secs(7)),
            ..Default::default()
        },
    ))
    .unwrap();
    let timeouts = http.timeouts();
    assert!(
        timeouts[0].is_some_and(|t| t <= Duration::from_secs(120) && t > Duration::from_secs(119))
    );
    assert!(timeouts[1].is_some_and(|t| t <= Duration::from_secs(7) && t > Duration::from_secs(6)));
    block_on(client.stream(&request("m"), &RequestOptions::default())).unwrap();
    assert_eq!(
        http.timeouts()[2],
        None,
        "streaming has no default total deadline"
    );
}

#[tokio::test]
async fn total_timeout_expires_during_authentication_before_a_request_is_sent() {
    use lingxi_agent_api::protocol::{AuthStrategy, Secret};
    use std::time::Duration;

    struct SlowAuthenticator;

    #[async_trait]
    impl lingxi_llm_client::Authenticator for SlowAuthenticator {
        async fn apply(
            &self,
            _req: &mut HttpRequest,
            _profile: &ProviderProfile,
            _credential: Option<&Secret<String>>,
        ) -> Result<(), LlmError> {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(())
        }
    }

    for streaming in [false, true] {
        let mut profile = solo("only", "https://one.test", model("m", "m"));
        profile.auth = AuthStrategy::ApiKey;
        let http = ScriptedTransport::new(vec![("https://one.test", Ok(200))]);
        let mut builder = LlmClientBuilder::with_transport(http.clone(), &[profile]);
        builder.register_authenticator(AuthStrategy::ApiKey, Arc::new(SlowAuthenticator));
        let client = builder.build().unwrap();
        let options = RequestOptions {
            total_timeout: Some(Duration::from_millis(10)),
            ..Default::default()
        };

        let result = if streaming {
            client.stream(&request("m"), &options).await.map(|_| ())
        } else {
            client.complete(&request("m"), &options).await.map(|_| ())
        };
        assert!(matches!(result, Err(LlmError::TransportTimeout { .. })));
        assert!(http.hops().is_empty(), "expired work must never be sent");
    }
}
