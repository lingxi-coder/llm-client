//! Exercise actual codec bodies and per-connection preparation with no network.
use async_trait::async_trait;
use lingxi_llm_client::protocol::{ChatRequest, LlmError, ProviderProfile, ThinkingBudget};
use lingxi_llm_client::{
    HttpRequest, HttpResponse, LlmClientBuilder, RequestOptions, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Capture(Mutex<Vec<HttpRequest>>);
#[async_trait]
impl Transport for Capture {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let first = request.url.contains("primary.test");
        self.0.lock().unwrap().push(request);
        let (status, body) = if first {
            (
                429,
                json!({"type":"error","error":{"type":"rate_limit_error","message":"try backup"}}),
            )
        } else {
            (
                200,
                json!({"id":"msg_ok","type":"message","role":"assistant","model":"shared","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}),
            )
        };
        Ok(HttpResponse {
            status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: serde_json::to_vec(&body).unwrap().into(),
        }
        .into())
    }
}
fn profiles() -> Vec<ProviderProfile> {
    [("primary",32768),("backup",8192)].into_iter().enumerate().map(|(order,(name,cap))| {
        serde_json::from_value(json!({"provider_id":"acme","profile_name":name,"protocol":"anthropic_messages","auth":"none","base_url":format!("https://{name}.test"),
        "models":[{"display_model":"Shared","request_model":"shared","billing_model":"shared","metadata":{"applicationOutputTokens":32768,"maxOutputTokens":cap}}],
        "connection":{"group":"chain","order":order,"failover":{"rateLimit":true}}
        })).unwrap()
    }).collect()
}
#[tokio::test]
async fn fallback_re_resolves_default_and_automatic_thinking_without_mutating_intent() {
    let transport = Arc::new(Capture::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &profiles())
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
    let request: ChatRequest = serde_json::from_value(json!({"model":"Shared","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],"thinking":{"mode":"enabled","budget":{"tokens":16384}}})).unwrap();
    client
        .chat()
        .complete_in(
            "chain",
            &request,
            &RequestOptions {
                automatic_thinking_budget: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let seen = transport.0.lock().unwrap();
    assert_eq!(seen.len(), 2);
    for (request, cap, thinking) in [(&seen[0], 32768, 16384), (&seen[1], 8192, 8191)] {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["max_tokens"], cap);
        assert_eq!(body["thinking"]["budget_tokens"], thinking);
    }
    assert_eq!(request.max_tokens, None);
    assert_eq!(
        request.thinking.unwrap().budget,
        Some(ThinkingBudget::Tokens(16384))
    );
}
#[tokio::test]
async fn explicit_thinking_is_not_rewritten_to_hide_a_codec_rejection() {
    let mut rows = profiles();
    rows.remove(0);
    rows[0].connection = Default::default();
    let transport = Arc::new(Capture::default());
    let client = LlmClientBuilder::with_transport(transport.clone(), &rows)
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
    let request:ChatRequest=serde_json::from_value(json!({"model":"Shared","messages":[],"thinking":{"mode":"enabled","budget":{"tokens":16384}}})).unwrap();
    assert!(client
        .chat()
        .complete(&request, &RequestOptions::default())
        .await
        .is_err());
    assert!(transport.0.lock().unwrap().is_empty());
}

#[tokio::test]
async fn optional_protocols_omit_unconfigured_limits_and_encode_only_configured_defaults() {
    use lingxi_llm_client::client::OutputBudgetSource;
    use lingxi_llm_client::RequestMode;
    for (protocol, pointer) in [
        ("open_ai_chat", "/max_tokens"),
        ("open_ai_responses", "/max_output_tokens"),
        (
            "gemini_generate_content",
            "/generationConfig/maxOutputTokens",
        ),
    ] {
        for configured in [false, true] {
            let mut profile: Value = serde_json::to_value(&profiles()[1]).unwrap();
            profile["protocol"] = json!(protocol);
            profile["connection"] = json!({});
            if !configured {
                profile["models"][0]["metadata"]
                    .as_object_mut()
                    .unwrap()
                    .remove("applicationOutputTokens");
            }
            let profile: ProviderProfile = serde_json::from_value(profile).unwrap();
            let transport = Arc::new(Capture::default());
            let client = LlmClientBuilder::with_transport(transport.clone(), &[profile])
                .with_region(lingxi_llm_client::protocol::Region::International)
                .build()
                .unwrap();
            let request:ChatRequest=serde_json::from_value(json!({"model":"Shared","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]})).unwrap();
            let prepared = client
                .prepare_on(
                    "backup",
                    &request,
                    &RequestOptions::default(),
                    RequestMode::Complete,
                )
                .await
                .unwrap();
            let body: Value = serde_json::from_slice(&prepared.request().body).unwrap();
            if configured {
                assert_eq!(body.pointer(pointer), Some(&json!(8192)), "{protocol}");
                assert_eq!(
                    prepared.output_budget().unwrap().source,
                    OutputBudgetSource::ClampedApplicationDefault
                );
            } else {
                assert!(body.pointer(pointer).is_none(), "{protocol}");
                assert_eq!(
                    prepared.output_budget().unwrap().source,
                    OutputBudgetSource::ProviderDefault
                );
            }
            assert_eq!(request.max_tokens, None);
            assert!(transport.0.lock().unwrap().is_empty());
        }
    }
}

#[tokio::test]
async fn automatic_thinking_is_omitted_below_the_manual_minimum() {
    use lingxi_llm_client::RequestMode;
    let mut rows = profiles();
    rows.remove(0);
    rows[0].connection = Default::default();
    rows[0].models[0].metadata.max_output_tokens = Some(512);
    let client = LlmClientBuilder::with_transport(Arc::new(Capture::default()), &rows)
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
    let request:ChatRequest=serde_json::from_value(json!({"model":"Shared","messages":[],"thinking":{"mode":"enabled","budget":{"tokens":16384}}})).unwrap();
    let prepared = client
        .prepare_on(
            "backup",
            &request,
            &RequestOptions {
                automatic_thinking_budget: true,
                ..Default::default()
            },
            RequestMode::Complete,
        )
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&prepared.request().body).unwrap();
    assert_eq!(body["max_tokens"], 512);
    assert!(body.get("thinking").is_none());
    assert_eq!(
        request.thinking.unwrap().budget,
        Some(ThinkingBudget::Tokens(16384))
    );
}

#[tokio::test]
async fn a_larger_fallback_restores_the_original_automatic_thinking_intent() {
    let transport = Arc::new(Capture::default());
    let mut rows = profiles();
    rows[0].models[0].metadata.max_output_tokens = Some(8192);
    rows[1].models[0].metadata.max_output_tokens = Some(32768);
    let client = LlmClientBuilder::with_transport(transport.clone(), &rows)
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
    let request:ChatRequest=serde_json::from_value(json!({"model":"Shared","max_tokens":65536,"messages":[],"thinking":{"mode":"enabled","budget":{"tokens":16384}}})).unwrap();
    client
        .chat()
        .complete_in(
            "chain",
            &request,
            &RequestOptions {
                automatic_thinking_budget: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let seen = transport.0.lock().unwrap();
    assert_eq!(seen.len(), 2);
    for (request, cap, thinking) in [(&seen[0], 8192, 8191), (&seen[1], 32768, 16384)] {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["max_tokens"], cap);
        assert_eq!(body["thinking"]["budget_tokens"], thinking);
    }
    assert_eq!(request.max_tokens, Some(65536));
    assert_eq!(
        request.thinking.unwrap().budget,
        Some(ThinkingBudget::Tokens(16384))
    );
}

#[tokio::test]
async fn claude_automatic_manual_thinking_does_not_leak_into_other_protocols() {
    let mut row = profiles().remove(1);
    row.connection = Default::default();
    row.protocol = lingxi_llm_client::protocol::ProtocolFamily::OpenAiChat;
    row.models[0].metadata.application_output_tokens = None;
    let client = LlmClientBuilder::with_transport(Arc::new(Capture::default()), &[row])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
    let request:ChatRequest=serde_json::from_value(json!({"model":"Shared","messages":[],"thinking":{"mode":"enabled","budget":{"tokens":16384}}})).unwrap();
    let prepared = client
        .prepare_on(
            "backup",
            &request,
            &RequestOptions {
                automatic_thinking_budget: true,
                ..Default::default()
            },
            lingxi_llm_client::RequestMode::Complete,
        )
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&prepared.request().body).unwrap();
    for key in [
        "max_tokens",
        "max_completion_tokens",
        "thinking",
        "reasoning",
        "reasoning_effort",
    ] {
        assert!(body.get(key).is_none(), "{key}");
    }
    assert_eq!(
        request.thinking.unwrap().budget,
        Some(ThinkingBudget::Tokens(16384))
    );
}

#[tokio::test]
async fn omitting_automatic_manual_thinking_preserves_explicit_effort() {
    use lingxi_llm_client::{protocol::ProtocolFamily, RequestMode};
    for (protocol, output, pointer) in [
        (
            ProtocolFamily::AnthropicMessages,
            512,
            "/output_config/effort",
        ),
        (ProtocolFamily::OpenAiChat, 8192, "/reasoning_effort"),
        (ProtocolFamily::OpenAiResponses, 8192, "/reasoning/effort"),
    ] {
        let mut row = profiles().remove(1);
        row.connection = Default::default();
        row.protocol = protocol;
        let client = LlmClientBuilder::with_transport(Arc::new(Capture::default()), &[row])
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build()
            .unwrap();
        let request: ChatRequest = serde_json::from_value(json!({
            "model":"Shared", "messages":[], "max_tokens":output,
            "thinking":{"mode":"enabled","budget":{"tokens":16384},"effort":"high"}
        }))
        .unwrap();
        let prepared = client
            .prepare_on(
                "backup",
                &request,
                &RequestOptions {
                    automatic_thinking_budget: true,
                    ..Default::default()
                },
                RequestMode::Complete,
            )
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&prepared.request().body).unwrap();
        assert_eq!(body.pointer(pointer), Some(&json!("high")), "{protocol:?}");
        assert!(body.get("thinking").is_none(), "{protocol:?}");
        assert_eq!(
            request.thinking.as_ref().unwrap().budget,
            Some(ThinkingBudget::Tokens(16384))
        );
    }
}

#[tokio::test]
async fn preserved_explicit_effort_still_requires_selected_model_support() {
    use lingxi_llm_client::{
        protocol::{CapabilitySupport, ProtocolFamily},
        RequestMode,
    };
    let mut row = profiles().remove(1);
    row.connection = Default::default();
    row.protocol = ProtocolFamily::OpenAiResponses;
    row.models[0].info.features.effort.support = CapabilitySupport::Unsupported;
    let capture = Arc::new(Capture::default());
    let client = LlmClientBuilder::with_transport(capture.clone(), &[row])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
    let request: ChatRequest = serde_json::from_value(json!({
        "model":"Shared", "messages":[],
        "thinking":{"mode":"enabled","budget":{"tokens":16384},"effort":"high"}
    }))
    .unwrap();
    let result = client
        .prepare_on(
            "backup",
            &request,
            &RequestOptions {
                automatic_thinking_budget: true,
                ..Default::default()
            },
            RequestMode::Complete,
        )
        .await;
    assert!(matches!(
        result,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert!(capture.0.lock().unwrap().is_empty());
}
