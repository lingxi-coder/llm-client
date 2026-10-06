use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
struct Reply {
    status: u16,
    body: Value,
}

struct FixtureTransport {
    replies: Mutex<Vec<Reply>>,
    sent: Mutex<Vec<HttpRequest>>,
}

impl FixtureTransport {
    fn new(replies: Vec<Reply>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies),
            sent: Mutex::new(Vec::new()),
        })
    }

    fn bodies(&self) -> Vec<Value> {
        self.sent
            .lock()
            .unwrap()
            .iter()
            .map(|request| serde_json::from_slice(&request.body).unwrap())
            .collect()
    }
}

#[async_trait]
impl Transport for FixtureTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        let reply = self.replies.lock().unwrap().remove(0);
        Ok(HttpResponse {
            status: reply.status,
            headers: vec![],
            body: Bytes::from(serde_json::to_vec(&reply.body).unwrap()),
        }
        .into())
    }
}

fn profile(name: &str, model: &str) -> ProviderProfile {
    let mut profile = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|profile| profile.profile_name == name)
        .unwrap();
    profile.auth = AuthStrategy::None;
    profile.credential = CredentialConfig::None;
    profile
        .models
        .retain(|candidate| candidate.request_model == model);
    assert_eq!(profile.models.len(), 1);
    profile
}

fn client(
    profiles: &[ProviderProfile],
    replies: Vec<Reply>,
    region: Region,
) -> (LlmClient, Arc<FixtureTransport>) {
    let transport = FixtureTransport::new(replies);
    let client = LlmClientBuilder::with_transport(transport.clone(), profiles)
        .with_region(region)
        .build()
        .unwrap();
    (client, transport)
}

struct WeakResponsesCodec;

impl WireCodec for WeakResponsesCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::OpenAiResponses
    }

    fn encode_request(
        &self,
        request: EncodeRequest<'_>,
        context: &CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let mut http = OpenAiResponsesCodec.encode_request(request, context)?;
        let mut body: Value = serde_json::from_slice(&http.body).unwrap();
        body.as_object_mut().unwrap().remove("text");
        http.body = serde_json::to_vec(&body).unwrap().into();
        Ok(http)
    }

    fn decode_response(
        &self,
        response: &HttpResponse,
        context: &CodecContext,
    ) -> Result<ChatResponse, LlmError> {
        OpenAiResponsesCodec.decode_response(response, context)
    }

    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        OpenAiResponsesCodec.stream_decoder(context)
    }
}

#[tokio::test]
async fn replacing_a_builtin_codec_disables_decisions_on_that_wire() {
    let transport = FixtureTransport::new(vec![]);
    let mut builder =
        LlmClientBuilder::with_transport(transport.clone(), &[profile("openai", "gpt-6-luna")]);
    builder.register_codec(Arc::new(WeakResponsesCodec));
    let client = builder.with_region(Region::International).build().unwrap();
    assert!(client.decisions().models().is_empty());
    assert!(matches!(
        client
            .decisions()
            .decide(&request("gpt-6-luna", false), &RequestOptions::default())
            .await,
        Err(DecisionError::Unsupported(_))
    ));
    assert!(transport.bodies().is_empty());
}

fn request(model: &str, image: bool) -> DecisionRequest {
    let mut context = vec![DecisionContextPart::Text {
        text: "A customer needs a refund.".into(),
    }];
    if image {
        context.push(DecisionContextPart::Image {
            source: Box::new(ImageSource::Base64 {
                media_type: "image/png".into(),
                data: "aGVsbG8=".into(),
            }),
        });
    }
    DecisionRequest {
        model: model.into(),
        context,
        questions: vec![
            DecisionQuestion {
                id: "route".into(),
                prompt: "Where should this go?".into(),
                options: vec![
                    DecisionOption {
                        id: "sales".into(),
                        label: "Sales".into(),
                    },
                    DecisionOption {
                        id: "support".into(),
                        label: "Support".into(),
                    },
                ],
            },
            DecisionQuestion {
                id: "priority".into(),
                prompt: "How urgent?".into(),
                options: vec![
                    DecisionOption {
                        id: "normal".into(),
                        label: "Normal".into(),
                    },
                    DecisionOption {
                        id: "urgent".into(),
                        label: "Urgent".into(),
                    },
                ],
            },
        ],
    }
}

fn reply(provider: &str, model: &str, text: &str, stop: &str) -> Reply {
    let body = match provider {
        "openai" => json!({
            "id":"resp_decision", "model":model,
            "status":if stop == "refusal" { "completed" } else if stop == "max_tokens" { "incomplete" } else { "completed" },
            "incomplete_details":if stop == "max_tokens" { json!({"reason":"max_output_tokens"}) } else { Value::Null },
            "output":[{"type":"message","role":"assistant","content":[
                if stop == "refusal" { json!({"type":"refusal","refusal":"cannot comply"}) }
                else { json!({"type":"output_text","text":text}) }
            ]}],
            "usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}
        }),
        "anthropic" => json!({
            "id":"msg_decision", "type":"message", "role":"assistant", "model":model,
            "content":[{"type":"text","text":text}],
            "stop_reason":if stop == "max_tokens" { "max_tokens" } else { "end_turn" },
            "usage":{"input_tokens":3,"output_tokens":4}
        }),
        "google" => json!({
            "modelVersion":model,
            "candidates":[{"content":{"role":"model","parts":[{"text":text}]},"finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"totalTokenCount":7}
        }),
        _ => json!({
            "id":"chat_decision", "model":model,
            "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}
        }),
    };
    Reply { status: 200, body }
}

#[tokio::test]
async fn strict_wire_and_batched_answer_mapping_on_each_first_batch_provider() {
    for (provider, model, region, image) in [
        ("openai", "gpt-6-luna", Region::International, true),
        ("anthropic", "claude-sonnet-5", Region::International, true),
        ("gemini", "gemini-3.7-flash", Region::International, true),
        ("grok", "grok-4.20", Region::International, false),
        ("qwen", "qwen3.8-flash", Region::ChinaMainland, false),
    ] {
        let profile = profile(provider, model);
        let provider_id = profile.provider_id.as_str().to_owned();
        let request = request(model, image);
        let (client, transport) = client(
            &[profile],
            vec![reply(
                &provider_id,
                model,
                r#"{"q0":"support","q1":"urgent"}"#,
                "end_turn",
            )],
            region,
        );
        let listed = client.decisions().models();
        assert_eq!(listed.len(), 1, "{provider}");
        assert_eq!(listed[0].image, image, "{provider}");
        let answer = client
            .decisions()
            .decide_in(provider, &request, &RequestOptions::default())
            .await
            .unwrap();
        assert_eq!(
            answer.answers,
            BTreeMap::from([
                ("route".into(), "support".into()),
                ("priority".into(), "urgent".into()),
            ]),
            "{provider}"
        );
        assert_eq!(answer.report.executed_profile.as_deref(), Some(provider));
        assert!(answer.report.usage.usage.is_some());
        let body = transport.bodies().pop().unwrap();
        let schema = match provider {
            "openai" => &body["text"]["format"]["schema"],
            "anthropic" => &body["output_config"]["format"]["schema"],
            "gemini" => &body["generationConfig"]["responseFormat"]["text"]["schema"],
            _ => &body["response_format"]["json_schema"]["schema"],
        };
        assert_eq!(
            schema["properties"]["q0"]["enum"],
            json!(["sales", "support"]),
            "{provider}"
        );
        assert_eq!(
            schema["properties"]["q1"]["enum"],
            json!(["normal", "urgent"]),
            "{provider}"
        );
        assert_eq!(schema["required"], json!(["q0", "q1"]), "{provider}");
        assert_eq!(schema["additionalProperties"], false, "{provider}");
        match provider {
            "openai" => assert_eq!(body["text"]["format"]["strict"], true),
            "anthropic" => assert_eq!(body["output_config"]["format"]["type"], "json_schema"),
            "gemini" => assert_eq!(
                body["generationConfig"]["responseFormat"]["text"]["mimeType"],
                "application/json"
            ),
            _ => assert_eq!(body["response_format"]["json_schema"]["strict"], true),
        }
        assert_eq!(body.to_string().contains("aGVsbG8="), image, "{provider}");
    }
}

#[tokio::test]
async fn request_and_modality_errors_never_dispatch() {
    let profile = profile("qwen", "qwen3.8-flash");
    let (client, transport) = client(&[profile], vec![], Region::ChinaMainland);
    let mut bad = request("qwen3.8-flash", false);
    bad.questions.clear();
    assert!(matches!(
        client
            .decisions()
            .decide(&bad, &RequestOptions::default())
            .await,
        Err(DecisionError::InvalidRequest(_))
    ));
    bad = request("qwen3.8-flash", false);
    bad.questions[1].id = bad.questions[0].id.clone();
    assert!(matches!(
        client
            .decisions()
            .decide(&bad, &RequestOptions::default())
            .await,
        Err(DecisionError::InvalidRequest(_))
    ));
    bad = request("qwen3.8-flash", false);
    bad.questions[0].options[1].id = bad.questions[0].options[0].id.clone();
    assert!(matches!(
        client
            .decisions()
            .decide(&bad, &RequestOptions::default())
            .await,
        Err(DecisionError::InvalidRequest(_))
    ));
    bad = request("qwen3.8-flash", false);
    bad.questions[0].options.clear();
    assert!(matches!(
        client
            .decisions()
            .decide(&bad, &RequestOptions::default())
            .await,
        Err(DecisionError::InvalidRequest(_))
    ));
    bad = request("qwen3.8-flash", true);
    assert!(matches!(
        client
            .decisions()
            .decide(&bad, &RequestOptions::default())
            .await,
        Err(DecisionError::Unsupported(_))
    ));
    assert!(transport.bodies().is_empty());
}

#[tokio::test]
async fn unresolved_image_attachment_never_dispatches() {
    let (client, transport) = client(
        &[profile("openai", "gpt-6-luna")],
        vec![],
        Region::International,
    );
    let mut request = request("gpt-6-luna", false);
    request.context.push(DecisionContextPart::Image {
        source: Box::new(ImageSource::Attachment {
            attachment: AttachmentRef {
                attachment_id: "photo".into(),
                revision: "1".into(),
                filename: "photo.png".into(),
                media_type: "image/png".into(),
                size_bytes: 32,
            },
        }),
    });
    assert!(client
        .decisions()
        .decide(&request, &RequestOptions::default())
        .await
        .is_err());
    assert!(transport.bodies().is_empty());
}

#[tokio::test]
async fn failover_skips_ineligible_connection_and_keeps_eligible_one() {
    let mut first = profile("openai", "gpt-6-luna");
    first.connection.group = Some("decision-group".into());
    first.connection.connection_id = Some("primary".into());
    first.connection.order = 0;
    first.connection.failover.rate_limit = true;
    let mut spare = first.clone();
    spare.profile_name = "spare".into();
    spare.connection.connection_id = Some("spare".into());
    spare.connection.order = 1;
    spare.connection.hidden = true;
    let limited = Reply {
        status: 429,
        body: json!({"error":{"message":"limited"}}),
    };

    spare.decisions = None;
    let (first_client, transport) = client(
        &[first.clone(), spare.clone()],
        vec![limited.clone()],
        Region::International,
    );
    assert!(first_client
        .decisions()
        .decide(&request("gpt-6-luna", false), &RequestOptions::default())
        .await
        .is_err());
    assert_eq!(transport.bodies().len(), 1);

    spare.decisions = first.decisions;
    let (client, transport) = client(
        &[first, spare],
        vec![
            limited,
            reply(
                "openai",
                "gpt-6-luna",
                r#"{"q0":"support","q1":"urgent"}"#,
                "end_turn",
            ),
        ],
        Region::International,
    );
    let result = client
        .decisions()
        .decide(&request("gpt-6-luna", false), &RequestOptions::default())
        .await
        .unwrap();
    assert_eq!(result.report.executed_profile.as_deref(), Some("spare"));
    assert!(result.report.prior_attempts.is_empty());
    assert_eq!(transport.bodies().len(), 2);
}

#[tokio::test]
async fn unscoped_decision_promotes_an_eligible_connection() {
    let mut first = profile("openai", "gpt-6-luna");
    first.connection.group = Some("decision-group".into());
    first.connection.connection_id = Some("primary".into());
    first.connection.order = 0;
    let mut eligible = first.clone();
    eligible.profile_name = "eligible".into();
    eligible.connection.connection_id = Some("eligible".into());
    eligible.connection.order = 1;
    first.decisions = None;
    let (client, transport) = client(
        &[first, eligible],
        vec![reply(
            "openai",
            "gpt-6-luna",
            r#"{"q0":"support","q1":"urgent"}"#,
            "end_turn",
        )],
        Region::International,
    );
    assert_eq!(client.decisions().models().len(), 1);
    assert!(matches!(
        client
            .decisions()
            .decide_in(
                "openai",
                &request("gpt-6-luna", false),
                &RequestOptions::default(),
            )
            .await,
        Err(DecisionError::Unsupported(_))
    ));
    assert!(transport.bodies().is_empty());
    let result = client
        .decisions()
        .decide(&request("gpt-6-luna", false), &RequestOptions::default())
        .await
        .unwrap();
    assert_eq!(result.report.executed_profile.as_deref(), Some("eligible"));
    assert_eq!(transport.bodies().len(), 1);
}

#[tokio::test]
async fn promoted_connection_uses_only_its_named_credential() {
    let mut primary = profile("openai", "gpt-6-luna");
    primary.auth = AuthStrategy::ApiKey;
    primary.credential = CredentialConfig::Env {
        var: "OPENAI_API_KEY".into(),
    };
    primary.connection.group = Some("decision-group".into());
    primary.connection.connection_id = Some("primary".into());
    primary.connection.order = 0;
    let mut spare = primary.clone();
    spare.profile_name = "spare".into();
    spare.connection.connection_id = Some("spare".into());
    spare.connection.order = 1;
    primary.decisions = None;

    let (client, transport) = client(
        &[primary, spare],
        vec![reply(
            "openai",
            "gpt-6-luna",
            r#"{"q0":"support","q1":"urgent"}"#,
            "end_turn",
        )],
        Region::International,
    );
    let mut options = RequestOptions {
        credential: Some(Secret::new("primary-secret".into())),
        authenticator: Some(lingxi_llm_client::client::options::RequestAuthenticator(
            Arc::new(RedirectDecisionAuthentication),
        )),
        ..Default::default()
    };
    assert!(matches!(
        client
            .decisions()
            .decide(&request("gpt-6-luna", false), &options)
            .await,
        Err(DecisionError::Provider(error)) if matches!(*error, LlmError::Authentication { .. })
    ));
    assert!(transport.sent.lock().unwrap().is_empty());

    options
        .fallback_credentials
        .insert("spare".into(), Secret::new("spare-secret".into()));
    let result = client
        .decisions()
        .decide(&request("gpt-6-luna", false), &options)
        .await
        .unwrap();
    assert_eq!(result.report.executed_profile.as_deref(), Some("spare"));
    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].url, "https://api.openai.com/v1/responses");
    assert!(sent[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("authorization") && value == "Bearer spare-secret"
    }));
    assert!(!sent[0]
        .headers
        .iter()
        .any(|(_, value)| value.contains("primary-secret")));
}

#[tokio::test]
async fn hidden_spare_is_never_promoted_by_an_unscoped_decision() {
    let mut primary = profile("openai", "gpt-6-luna");
    primary.connection.group = Some("decision-group".into());
    primary.connection.connection_id = Some("primary".into());
    primary.connection.order = 0;
    let mut spare = primary.clone();
    spare.profile_name = "spare".into();
    spare.connection.connection_id = Some("spare".into());
    spare.connection.order = 1;
    spare.connection.hidden = true;
    primary.decisions = None;
    let (decision_client, transport) = client(
        &[primary, spare.clone()],
        vec![reply(
            "openai",
            "gpt-6-luna",
            r#"{"q0":"support","q1":"urgent"}"#,
            "end_turn",
        )],
        Region::International,
    );
    assert!(decision_client.decisions().models().is_empty());
    assert!(matches!(
        decision_client
            .decisions()
            .decide(&request("gpt-6-luna", false), &RequestOptions::default())
            .await,
        Err(DecisionError::Unsupported(_))
    ));
    assert!(transport.sent.lock().unwrap().is_empty());

    let (hidden_only, hidden_transport) = client(&[spare], vec![], Region::International);
    assert!(hidden_only.decisions().models().is_empty());
    assert!(matches!(
        hidden_only
            .decisions()
            .decide(&request("gpt-6-luna", false), &RequestOptions::default())
            .await,
        Err(DecisionError::Unsupported(_))
    ));
    assert!(hidden_transport.sent.lock().unwrap().is_empty());

    let result = decision_client
        .decisions()
        .decide_in(
            "spare",
            &request("gpt-6-luna", false),
            &RequestOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(result.report.executed_profile.as_deref(), Some("spare"));
}

#[derive(Debug)]
struct RewriteDecisionBody;

impl lingxi_llm_client::client::options::RequestFinalizer for RewriteDecisionBody {
    fn finalize(&self, request: &mut HttpRequest, _: &ProviderProfile) -> Result<(), LlmError> {
        request.body = Bytes::from_static(br#"{"model":"gpt-6-luna"}"#);
        Ok(())
    }
}

#[derive(Debug)]
struct RedirectDecisionAuthentication;

#[async_trait]
impl Authenticator for RedirectDecisionAuthentication {
    async fn apply(
        &self,
        request: &mut HttpRequest,
        _: &ProviderProfile,
        _: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        request.url = "https://example.invalid/weak-json".into();
        Ok(())
    }
}

#[tokio::test]
async fn decision_hooks_cannot_rewrite_the_strict_wire() {
    let (basic_client, basic_transport) = client(
        &[profile("openai", "gpt-6-luna")],
        vec![],
        Region::International,
    );
    let options = RequestOptions {
        finalizer: Some(Arc::new(RewriteDecisionBody)),
        ..Default::default()
    };
    assert!(matches!(
        basic_client
            .decisions()
            .decide(&request("gpt-6-luna", false), &options)
            .await,
        Err(DecisionError::Provider(error)) if matches!(*error, LlmError::InvalidRequest { .. })
    ));
    assert!(basic_transport.bodies().is_empty());

    let mut authenticated = profile("openai", "gpt-6-luna");
    authenticated.auth = AuthStrategy::ApiKey;
    authenticated.credential = CredentialConfig::Env {
        var: "OPENAI_API_KEY".into(),
    };
    let (auth_client, auth_transport) = client(&[authenticated], vec![], Region::International);
    let options = RequestOptions {
        authenticator: Some(lingxi_llm_client::client::options::RequestAuthenticator(
            Arc::new(RedirectDecisionAuthentication),
        )),
        credential: Some(Secret::new("test-key".into())),
        ..Default::default()
    };
    assert!(matches!(
        auth_client
            .decisions()
            .decide(&request("gpt-6-luna", false), &options)
            .await,
        Err(DecisionError::Provider(error)) if matches!(*error, LlmError::InvalidRequest { .. })
    ));
    assert!(auth_transport.bodies().is_empty());
}

struct FailingSpareTransport {
    first: Reply,
    sent: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for FailingSpareTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let mut sent = self.sent.lock().unwrap();
        sent.push(request);
        if sent.len() == 1 {
            Ok(HttpResponse {
                status: self.first.status,
                headers: vec![],
                body: Bytes::from(serde_json::to_vec(&self.first.body).unwrap()),
            }
            .into())
        } else {
            Err(LlmError::Transport {
                message: "spare connection unavailable".into(),
            })
        }
    }
}

#[tokio::test]
async fn reported_usage_survives_when_spare_connection_fails_before_response() {
    let mut first = profile("openai", "gpt-6-luna");
    first.connection.group = Some("decision-group".into());
    first.connection.connection_id = Some("primary".into());
    first.connection.failover.rate_limit = true;
    let mut spare = first.clone();
    spare.profile_name = "spare".into();
    spare.connection.connection_id = Some("spare".into());
    spare.connection.order = 1;
    spare.connection.hidden = true;
    let transport = Arc::new(FailingSpareTransport {
        first: Reply {
            status: 200,
            body: json!({
                "model":"gpt-6-luna", "status":"failed", "output":[],
                "error":{"code":"insufficient_quota","message":"quota exhausted"},
                "usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}
            }),
        },
        sent: Mutex::new(Vec::new()),
    });
    let client = LlmClientBuilder::with_transport(transport.clone(), &[first, spare])
        .with_region(Region::International)
        .build()
        .unwrap();
    let error = client
        .decisions()
        .decide(&request("gpt-6-luna", false), &RequestOptions::default())
        .await
        .unwrap_err();
    let DecisionError::ProviderWithReport { error, report } = error else {
        panic!("{error:?}");
    };
    assert!(matches!(*error, LlmError::Transport { .. }));
    assert_eq!(report.executed_profile.as_deref(), Some("spare"));
    assert_eq!(report.usage.state, UsageState::Missing);
    assert_eq!(report.prior_attempts.len(), 1);
    assert_eq!(report.prior_attempts[0].executed_profile, "openai");
    assert_eq!(
        report.prior_attempts[0].usage.usage.unwrap().input_tokens,
        5
    );
    assert_eq!(transport.sent.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn preparation_failure_does_not_mark_spare_as_executed() {
    let mut first = profile("openai", "gpt-6-luna");
    first.connection.group = Some("decision-group".into());
    first.connection.connection_id = Some("primary".into());
    first.connection.failover.rate_limit = true;
    let mut spare = first.clone();
    spare.profile_name = "spare".into();
    spare.connection.connection_id = Some("spare".into());
    spare.connection.order = 1;
    spare.auth = AuthStrategy::ApiKey;
    spare.credential = CredentialConfig::Env {
        var: "OPENAI_API_KEY".into(),
    };
    let (client, transport) = client(
        &[first, spare],
        vec![Reply {
            status: 200,
            body: json!({
                "model":"gpt-6-luna", "status":"failed", "output":[],
                "error":{"code":"insufficient_quota","message":"quota exhausted"},
                "usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}
            }),
        }],
        Region::International,
    );
    let error = client
        .decisions()
        .decide(&request("gpt-6-luna", false), &RequestOptions::default())
        .await
        .unwrap_err();
    let DecisionError::ProviderWithReport { error, report } = error else {
        panic!("{error:?}");
    };
    assert!(matches!(*error, LlmError::Authentication { .. }));
    assert_eq!(report.executed_profile, None);
    assert_eq!(report.prior_attempts.len(), 1);
    assert_eq!(report.prior_attempts[0].executed_profile, "openai");
    assert_eq!(
        report.prior_attempts[0].usage.usage.unwrap().input_tokens,
        5
    );
    assert_eq!(transport.bodies().len(), 1);
}

#[tokio::test]
async fn malformed_refused_and_incomplete_results_retain_usage_without_retry() {
    for (text, stop, kind) in [
        (r#"{"q0":"foreign","q1":"urgent"}"#, "end_turn", "invalid"),
        (r#"{"q0":"support"}"#, "end_turn", "invalid"),
        (
            r#"{"q0":"support","q1":"urgent","q2":"extra"}"#,
            "end_turn",
            "invalid",
        ),
        ("not JSON", "end_turn", "invalid"),
        ("", "refusal", "refused"),
        ("", "max_tokens", "incomplete"),
    ] {
        let (client, transport) = client(
            &[profile("openai", "gpt-6-luna")],
            vec![reply("openai", "gpt-6-luna", text, stop)],
            Region::International,
        );
        let error = client
            .decisions()
            .decide(&request("gpt-6-luna", false), &RequestOptions::default())
            .await
            .unwrap_err();
        let report = match (kind, error) {
            ("invalid", DecisionError::InvalidResult { report, .. }) => report,
            ("refused", DecisionError::Refused { report }) => report,
            ("incomplete", DecisionError::Incomplete { report }) => report,
            (_, error) => panic!("unexpected error: {error:?}"),
        };
        assert!(report.usage.usage.is_some());
        assert_eq!(transport.bodies().len(), 1);
    }
}

#[tokio::test]
async fn missing_terminal_marker_is_incomplete_on_every_strict_wire() {
    for (provider, model, region) in [
        ("openai", "gpt-6-luna", Region::International),
        ("anthropic", "claude-sonnet-5", Region::International),
        ("gemini", "gemini-3.7-flash", Region::International),
        ("grok", "grok-4.20", Region::International),
        ("qwen", "qwen3.8-flash", Region::ChinaMainland),
    ] {
        let mut response = reply(
            if provider == "gemini" {
                "google"
            } else {
                provider
            },
            model,
            r#"{"q0":"support","q1":"urgent"}"#,
            "end_turn",
        );
        match provider {
            "openai" => {
                response.body.as_object_mut().unwrap().remove("status");
            }
            "anthropic" => {
                response.body.as_object_mut().unwrap().remove("stop_reason");
            }
            "gemini" => {
                response.body["candidates"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("finishReason");
            }
            _ => {
                response.body["choices"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("finish_reason");
            }
        }
        let (client, transport) = client(&[profile(provider, model)], vec![response], region);
        let error = client
            .decisions()
            .decide(&request(model, false), &RequestOptions::default())
            .await
            .unwrap_err();
        let DecisionError::Incomplete { report } = error else {
            panic!("{provider}: {error:?}");
        };
        assert_eq!(report.usage.usage.unwrap().input_tokens, 3, "{provider}");
        assert_eq!(transport.bodies().len(), 1, "{provider}");
    }
}

struct InterruptedBodyTransport {
    sent: Mutex<Vec<HttpRequest>>,
    body: Bytes,
}

#[async_trait]
impl Transport for InterruptedBodyTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let mut sent = self.sent.lock().unwrap();
        sent.push(request);
        assert_eq!(
            sent.len(),
            1,
            "a partially received decision must not be retried"
        );
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: futures::stream::iter(vec![
                Ok(self.body.clone()),
                Err(LlmError::Transport {
                    message: "body interrupted".into(),
                }),
            ])
            .boxed(),
        })
    }
}

#[tokio::test]
async fn interrupted_200_body_never_redecides_on_spare_connection() {
    let mut first = profile("openai", "gpt-6-luna");
    first.connection.group = Some("decision-group".into());
    first.connection.connection_id = Some("primary".into());
    let mut spare = first.clone();
    spare.profile_name = "spare".into();
    spare.connection.connection_id = Some("spare".into());
    spare.connection.order = 1;
    spare.connection.hidden = true;
    let response = reply(
        "openai",
        "gpt-6-luna",
        r#"{"q0":"support","q1":"urgent"}"#,
        "end_turn",
    );
    let transport = Arc::new(InterruptedBodyTransport {
        sent: Mutex::new(Vec::new()),
        body: Bytes::from(serde_json::to_vec(&response.body).unwrap()),
    });
    let client = LlmClientBuilder::with_transport(transport.clone(), &[first, spare])
        .with_region(Region::International)
        .build()
        .unwrap();
    let error = client
        .decisions()
        .decide(&request("gpt-6-luna", false), &RequestOptions::default())
        .await
        .unwrap_err();
    let DecisionError::Incomplete { report } = error else {
        panic!("{error:?}");
    };
    assert_eq!(report.usage.usage.unwrap().input_tokens, 3);
    assert_eq!(transport.sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn responses_failed_200_keeps_provider_classification_and_eligible_failover() {
    let mut first = profile("openai", "gpt-6-luna");
    first.connection.group = Some("decision-group".into());
    first.connection.connection_id = Some("primary".into());
    first.connection.failover.rate_limit = true;
    let mut spare = first.clone();
    spare.profile_name = "spare".into();
    spare.connection.connection_id = Some("spare".into());
    spare.connection.order = 1;
    spare.connection.hidden = true;
    let failed = Reply {
        status: 200,
        body: json!({
            "model":"gpt-6-luna", "status":"failed", "output":[],
            "error":{"code":"insufficient_quota","message":"quota exhausted"},
            "usage":{"input_tokens":3,"output_tokens":0,"total_tokens":3}
        }),
    };
    let (first_client, transport) = client(
        &[first.clone()],
        vec![failed.clone()],
        Region::International,
    );
    let error = first_client
        .decisions()
        .decide(&request("gpt-6-luna", false), &RequestOptions::default())
        .await
        .unwrap_err();
    let DecisionError::ProviderWithReport { error, report } = error else {
        panic!("{error:?}");
    };
    assert!(matches!(*error, LlmError::QuotaExceeded { .. }));
    assert_eq!(report.usage.usage.unwrap().input_tokens, 3);
    assert_eq!(transport.bodies().len(), 1);

    let (client, transport) = client(
        &[first, spare],
        vec![
            failed,
            reply(
                "openai",
                "gpt-6-luna",
                r#"{"q0":"support","q1":"urgent"}"#,
                "end_turn",
            ),
        ],
        Region::International,
    );
    let result = client
        .decisions()
        .decide(&request("gpt-6-luna", false), &RequestOptions::default())
        .await
        .unwrap();
    assert_eq!(result.report.executed_profile.as_deref(), Some("spare"));
    assert_eq!(result.report.usage.usage.unwrap().input_tokens, 3);
    assert_eq!(result.report.prior_attempts.len(), 1);
    assert_eq!(result.report.prior_attempts[0].executed_profile, "openai");
    assert_eq!(
        result.report.prior_attempts[0]
            .usage
            .usage
            .unwrap()
            .input_tokens,
        3
    );
    assert_eq!(transport.bodies().len(), 2);
}

#[tokio::test]
async fn malformed_provider_envelope_retains_usage_and_never_fails_over() {
    let mut first = profile("openai", "gpt-6-luna");
    first.connection.group = Some("decision-group".into());
    first.connection.connection_id = Some("primary".into());
    first.connection.order = 0;
    first.connection.failover.server_error = true;
    let mut spare = first.clone();
    spare.profile_name = "spare".into();
    spare.connection.connection_id = Some("spare".into());
    spare.connection.order = 1;
    spare.connection.hidden = true;
    let malformed = Reply {
        status: 200,
        body: json!({
            "model":"gpt-6-luna",
            "status":"completed",
            "output":null,
            "usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}
        }),
    };
    let (client, transport) = client(
        &[first, spare],
        vec![
            malformed,
            reply(
                "openai",
                "gpt-6-luna",
                r#"{"q0":"support","q1":"urgent"}"#,
                "end_turn",
            ),
        ],
        Region::International,
    );
    let error = client
        .decisions()
        .decide(&request("gpt-6-luna", false), &RequestOptions::default())
        .await
        .unwrap_err();
    let DecisionError::InvalidResult { report, .. } = error else {
        panic!("expected invalid result, got {error:?}");
    };
    assert_eq!(report.model, "gpt-6-luna");
    assert_eq!(report.executed_profile.as_deref(), Some("openai"));
    assert_eq!(report.usage.usage.unwrap().input_tokens, 3);
    assert_eq!(report.usage.usage.unwrap().output_tokens, 4);
    assert_eq!(transport.bodies().len(), 1);
}

#[tokio::test]
async fn malformed_envelopes_preserve_usage_on_other_strict_wires() {
    for (provider, model, region, body) in [
        (
            "anthropic",
            "claude-sonnet-5",
            Region::International,
            json!({
                "model":"claude-sonnet-5", "content":null,
                "usage":{"input_tokens":3,"output_tokens":4}
            }),
        ),
        (
            "gemini",
            "gemini-3.7-flash",
            Region::International,
            json!({
                "modelVersion":"gemini-3.7-flash", "candidates":null,
                "usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"totalTokenCount":7}
            }),
        ),
        (
            "grok",
            "grok-4.20",
            Region::International,
            json!({
                "model":"grok-4.20", "choices":null,
                "usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}
            }),
        ),
        (
            "qwen",
            "qwen3.8-flash",
            Region::ChinaMainland,
            json!({
                "model":"qwen3.8-flash", "choices":null,
                "usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}
            }),
        ),
    ] {
        let (client, transport) = client(
            &[profile(provider, model)],
            vec![Reply { status: 200, body }],
            region,
        );
        let error = client
            .decisions()
            .decide(&request(model, false), &RequestOptions::default())
            .await
            .unwrap_err();
        let DecisionError::InvalidResult { report, .. } = error else {
            panic!("expected invalid result for {provider}, got {error:?}");
        };
        assert_eq!(report.model, model, "{provider}");
        assert_eq!(report.usage.usage.unwrap().input_tokens, 3, "{provider}");
        assert_eq!(report.usage.usage.unwrap().output_tokens, 4, "{provider}");
        assert_eq!(transport.bodies().len(), 1, "{provider}");
    }
}

#[test]
fn first_party_presets_are_opted_in_and_weak_routes_are_not() {
    let profiles = builtin_providers().unwrap();
    for name in ["openai", "anthropic", "gemini", "grok", "qwen"] {
        assert!(profiles
            .iter()
            .find(|p| p.profile_name == name)
            .unwrap()
            .decisions
            .is_some());
    }
    for name in [
        "deepseek",
        "openrouter",
        "kimi",
        "glm",
        "minimax",
        "github-copilot",
        "qwen-intl",
    ] {
        assert!(profiles
            .iter()
            .find(|p| p.profile_name == name)
            .unwrap()
            .decisions
            .is_none());
    }
}

#[tokio::test]
async fn decision_admission_requires_verified_provider_protocol_endpoint_model_and_region() {
    let valid = profile("openai", "gpt-6-luna");
    let (valid_client, _) = client(std::slice::from_ref(&valid), vec![], Region::International);
    assert_eq!(valid_client.snapshot().decisions().models().len(), 1);

    let old_model = profile("openai", "gpt-3.5-turbo");
    let (old_client, _) = client(&[old_model], vec![], Region::International);
    assert!(old_client.decisions().models().is_empty());

    let (other_region_client, _) =
        client(std::slice::from_ref(&valid), vec![], Region::ChinaMainland);
    assert!(other_region_client.decisions().models().is_empty());

    for mut changed in [
        {
            let mut p = valid.clone();
            p.provider_id = "openrouter".into();
            p
        },
        {
            let mut p = valid.clone();
            p.protocol = ProtocolFamily::OpenAiChat;
            p
        },
        {
            let mut p = valid.clone();
            p.base_url = "https://api.openai.com:8443/v1".into();
            p
        },
    ] {
        // The OpenAI preset also carries independent, endpoint-bound services.
        // This fixture changes only the Decision route under test.
        changed.embeddings = Default::default();
        changed.retrieval = Default::default();
        changed.batches = Default::default();
        changed.background = Default::default();
        changed.audio = Default::default();
        let (client, transport) = client(&[changed], vec![], Region::International);
        assert!(client.decisions().models().is_empty());
        assert!(matches!(
            client
                .decisions()
                .decide(&request("gpt-6-luna", false), &RequestOptions::default())
                .await,
            Err(DecisionError::Unsupported(_))
        ));
        assert!(transport.bodies().is_empty());
    }
}

#[tokio::test]
async fn generic_structured_output_catalog_flag_does_not_admit_a_decision_model() {
    let gemma = profile("gemini", "gemma-4-26b-a4b-it");
    assert_eq!(
        gemma.models[0].capability_support_for(ModelCapability::StructuredOutput),
        CapabilitySupport::Supported
    );
    let (client, transport) = client(&[gemma], vec![], Region::International);
    assert!(client.decisions().models().is_empty());
    assert!(matches!(
        client
            .decisions()
            .decide(
                &request("gemma-4-26b-a4b-it", false),
                &RequestOptions::default(),
            )
            .await,
        Err(DecisionError::Unsupported(_))
    ));
    assert!(transport.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn old_gpt_4o_snapshot_is_not_listed_for_json_schema_decisions() {
    let (client, transport) = client(
        &[profile("openai", "gpt-4o-2024-05-13")],
        vec![],
        Region::International,
    );
    assert!(client.decisions().models().is_empty());
    assert!(matches!(
        client
            .decisions()
            .decide(
                &request("gpt-4o-2024-05-13", false),
                &RequestOptions::default(),
            )
            .await,
        Err(DecisionError::Unsupported(_))
    ));
    assert!(transport.bodies().is_empty());
}

#[tokio::test]
async fn qwen_explicitly_unsupported_row_is_not_admitted() {
    let mut profile = profile("qwen", "qwen3.8-flash");
    profile.models[0].capability_support = Some(ModelCapabilitySupport {
        structured_output: CapabilitySupport::Unsupported,
        ..Default::default()
    });
    let (client, transport) = client(&[profile], vec![], Region::ChinaMainland);
    assert!(client.decisions().models().is_empty());
    assert!(matches!(
        client
            .decisions()
            .decide(&request("qwen3.8-flash", false), &RequestOptions::default())
            .await,
        Err(DecisionError::Unsupported(_))
    ));
    assert!(transport.bodies().is_empty());
}

#[tokio::test]
async fn chatgpt_plan_profile_is_not_admitted_for_nonstreaming_decisions() {
    let mut profile = profile("openai", "gpt-6-luna");
    profile.auth = AuthStrategy::ChatGptPlan;
    profile.pricing.billing_mode = BillingMode::Subscription;
    profile.model_list = DirectoryRoute::NotPublished;
    profile.images = Default::default();
    profile.embeddings = Default::default();
    profile.retrieval = Default::default();
    profile.batches = Default::default();
    profile.background = Default::default();
    profile.audio = Default::default();
    let (client, transport) = client(&[profile], vec![], Region::International);
    assert!(client.decisions().models().is_empty());
    assert!(matches!(
        client
            .decisions()
            .decide(&request("gpt-6-luna", false), &RequestOptions::default())
            .await,
        Err(DecisionError::Unsupported(_))
    ));
    assert!(transport.bodies().is_empty());
}

#[tokio::test]
async fn model_listing_uses_each_row_when_wire_ids_are_shared() {
    let mut profile = profile("openai", "gpt-6-luna");
    let first_display = profile.models[0].display_model.clone();
    let mut second = profile.models[0].clone();
    second.display_model = "second-catalog-row".into();
    second.aliases.clear();
    second
        .capability_support
        .as_mut()
        .unwrap()
        .structured_output = CapabilitySupport::Unsupported;
    profile.models.push(second);

    let (first_client, transport) = client(&[profile.clone()], vec![], Region::International);
    let listings = first_client.decisions().models();
    assert_eq!(listings.len(), 1);
    assert_eq!(listings[0].model.id, first_display);
    assert!(matches!(
        first_client
            .decisions()
            .decide_in(
                "openai",
                &request("second-catalog-row", false),
                &RequestOptions::default(),
            )
            .await,
        Err(DecisionError::Unsupported(_))
    ));
    assert!(transport.bodies().is_empty());

    profile.models[0]
        .capability_support
        .as_mut()
        .unwrap()
        .structured_output = CapabilitySupport::Unsupported;
    profile.models[1]
        .capability_support
        .as_mut()
        .unwrap()
        .structured_output = CapabilitySupport::Supported;
    let (second_client, _) = client(&[profile], vec![], Region::International);
    let listings = second_client.decisions().models();
    assert_eq!(listings.len(), 1);
    assert_eq!(listings[0].model.id, "second-catalog-row");
}
