//! A host capability observation belongs to one request and primary connection.
use async_trait::async_trait;
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Capture(Mutex<Vec<HttpRequest>>);
#[async_trait]
impl Transport for Capture {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.lock().unwrap().push(request);
        Ok(HttpResponse {
            status: 503,
            headers: vec![],
            body: bytes::Bytes::from_static(
                br#"{"type":"error","error":{"type":"overloaded_error","message":"fixture"}}"#,
            ),
        }
        .into())
    }
}
fn profile() -> ProviderProfile {
    let mut p = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "anthropic")
        .unwrap();
    p.auth = AuthStrategy::None;
    p.credential = CredentialConfig::None;
    p.models.retain(|m| m.request_model == "claude-opus-4-7");
    assert_eq!(p.models.len(), 1);
    assert_eq!(
        p.models[0].info.features.fast,
        CapabilitySupport::Unsupported
    );
    p
}
fn request() -> ChatRequest {
    serde_json::from_value(
        json!({"model":"claude-opus-4-7","messages":[],"max_tokens":1024,"service_tier":"fast"}),
    )
    .unwrap()
}
#[tokio::test]
async fn request_observation_admits_encoding_without_mutating_shared_catalog() {
    let p = profile();
    let capture = Arc::new(Capture::default());
    let client = LlmClientBuilder::with_transport(capture.clone(), std::slice::from_ref(&p))
        .with_region(Region::International)
        .build()
        .unwrap();
    let req = request();
    for mode in [RequestMode::Complete, RequestMode::Stream] {
        assert!(matches!(
            client
                .prepare_draft_on("anthropic", &req, &Default::default(), mode)
                .await,
            Err(LlmError::UnsupportedCapability { .. })
        ));
        let draft = client
            .prepare_draft_on(
                "anthropic",
                &req,
                &RequestOptions {
                    fast_capability: Some(CapabilitySupport::Supported),
                    ..Default::default()
                },
                mode,
            )
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&draft.request().body).unwrap();
        assert_eq!(body["speed"], "fast");
        assert!(matches!(
            client
                .prepare_draft_on(
                    "anthropic",
                    &req,
                    &RequestOptions {
                        fast_capability: Some(CapabilitySupport::Unsupported),
                        ..Default::default()
                    },
                    mode
                )
                .await,
            Err(LlmError::UnsupportedCapability { .. })
        ));
        assert!(matches!(
            client
                .prepare_draft_on("anthropic", &req, &Default::default(), mode)
                .await,
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
    assert!(matches!(
        client
            .prepare_draft_on(
                "anthropic",
                &req,
                &RequestOptions {
                    fast_capability: Some(CapabilitySupport::Unknown),
                    ..Default::default()
                },
                RequestMode::Complete
            )
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert_eq!(
        client.models()[0].info.features.fast,
        CapabilitySupport::Unsupported
    );
    assert!(capture.0.lock().unwrap().is_empty());
}
#[tokio::test]
async fn primary_observation_is_not_applied_to_a_failover_connection() {
    let profiles: Vec<_> = (0..2)
        .map(|index| {
            let mut p = profile();
            p.profile_name = format!("primary-{index}");
            p.connection.group = Some("fast-fixture".into());
            p.connection.connection_id = Some(format!("slot-{index}"));
            p.connection.order = index;
            p.connection.failover =
                serde_json::from_value(json!({"overloaded":true,"serverError":true})).unwrap();
            p
        })
        .collect();
    let capture = Arc::new(Capture::default());
    let client = LlmClientBuilder::with_transport(capture.clone(), &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let result = client
        .chat()
        .complete(
            &request(),
            &RequestOptions {
                fast_capability: Some(CapabilitySupport::Supported),
                ..Default::default()
            },
        )
        .await;
    assert!(
        matches!(result, Err(LlmError::UnsupportedCapability { .. })),
        "{result:?}"
    );
    let calls = capture.0.lock().unwrap();
    assert_eq!(
        calls.len(),
        1,
        "fallback must fail its own unsupported capability before dispatch"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&calls[0].body).unwrap()["speed"],
        "fast"
    );
}

#[test]
fn an_observation_does_not_invent_a_fast_wire_mapping() {
    let mut p = profile();
    p.inference.fast = Some(FastWire::Unavailable);
    let context = CodecContext::new(&p, "claude-opus-4-7", RequestMode::Complete)
        .with_fast_capability(Some(CapabilitySupport::Supported));
    assert!(matches!(
        AnthropicMessagesCodec.encode_request(EncodeRequest::new(&request()), &context),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}
