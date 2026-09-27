#[path = "support/wire_api.rs"]
mod wire_api;

use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::openai_containers::{OpenAiContainerRef, OpenAiContainerScope};
use lingxi_llm_client::protocol::{
    ChatRequest, CodeInterpreterConfig, CodeInterpreterMemoryLimit, ConnectionSpec, ContentBlock,
    FailoverTriggers, HostedTool, LlmError, ProtocolFamily, ProviderFileSource, ProviderProfile,
    Region,
};
use lingxi_llm_client::{
    files::provider_file_endpoint_fingerprint, transport::HttpRequest, HttpResponse,
    LlmClientBuilder, OpenAiResponsesCodec, RequestOptions, StreamResponse, Transport, WireCodec,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

mod support;

fn profile(provider: &str, enabled: bool) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": provider,
        "profile_name": provider,
        "protocol": "open_ai_responses",
        "base_url": if provider == "openai" {
            "https://api.openai.com/v1"
        } else {
            "https://api.example.test/v1"
        },
        "auth": "none",
        "extra": if enabled { json!({"code_interpreter":"openai_responses"}) } else { Value::Null },
        "models": [{"display_model":"m","request_model":"m","billing_model":"m"}]
    }))
    .unwrap()
}

fn request(limit: Option<CodeInterpreterMemoryLimit>) -> ChatRequest {
    let mut req: ChatRequest = serde_json::from_value(json!({
        "model":"m", "messages":[{"role":"user","content":[{"type":"text","text":"Calculate"}]}]
    }))
    .unwrap();
    req.hosted_tools
        .push(HostedTool::CodeInterpreter(CodeInterpreterConfig {
            memory_limit: limit,
            ..CodeInterpreterConfig::default()
        }));
    req
}

fn provider_file(
    profile: &ProviderProfile,
    file_id: &str,
    account_scope: &str,
) -> ProviderFileSource {
    ProviderFileSource {
        protocol: ProtocolFamily::OpenAiResponses,
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&profile.base_url),
        account_scope: Some(account_scope.into()),
        file_id: file_id.into(),
        uri: None,
        expires_at: None,
        processing_status: None,
        media_type: None,
        purpose: Some("user_data".into()),
    }
}

#[test]
fn openai_responses_encodes_auto_container_and_requests_execution_outputs() {
    let profile = profile("openai", true);
    let client =
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), std::slice::from_ref(&profile))
            .with_region(Region::International)
            .build()
            .unwrap();
    let route = client.resolve("m").unwrap();
    for (limit, expected) in [
        (None, Value::Null),
        (Some(CodeInterpreterMemoryLimit::FourG), json!("4g")),
        (Some(CodeInterpreterMemoryLimit::SixtyFourG), json!("64g")),
    ] {
        let req = request(limit);
        let http = OpenAiResponsesCodec
            .encode_request(
                lingxi_llm_client::EncodeRequest::new(&req),
                &wire_api::context(&profile, &route.request_model, &RequestOptions::default()),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(body["tools"][0]["type"], "code_interpreter");
        assert_eq!(body["tools"][0]["container"]["type"], "auto");
        assert_eq!(body["tools"][0]["container"]["memory_limit"], expected);
        assert!(body["include"]
            .as_array()
            .unwrap()
            .contains(&json!("code_interpreter_call.outputs")));
    }

    let item = json!({
        "type":"code_interpreter_call","id":"ci-1","container_id":"cntr-1",
        "code":"print(2)","outputs":[{"type":"logs","logs":"2"}]
    });
    let decoded = OpenAiResponsesCodec
        .decode_response(
            &HttpResponse {
                status: 200,
                headers: vec![],
                body: serde_json::to_vec(&json!({
                    "id":"resp-1","status":"completed","model":"m","output":[item]
                }))
                .unwrap()
                .into(),
            },
            &wire_api::context(&profile, &route.request_model, &RequestOptions::default()),
        )
        .unwrap();
    assert_eq!(
        decoded.message.content[0],
        ContentBlock::ProviderContent {
            protocol: ProtocolFamily::OpenAiResponses,
            value: item
        }
    );
}

#[test]
fn automatic_container_mounts_scoped_file_ids_and_rejects_duplicate_ids() {
    let profile = profile("openai", true);
    let files_scope = "openai-files-project";
    let mut req = request(None);
    req.hosted_tools = vec![HostedTool::CodeInterpreter(
        CodeInterpreterConfig::default().with_files([
            provider_file(&profile, "file-abc", files_scope),
            provider_file(&profile, "file-def", files_scope),
        ]),
    )];
    let options = RequestOptions {
        file_account_scope: Some(files_scope.into()),
        ..RequestOptions::default()
    };
    let context = wire_api::context(&profile, "m", &options)
        .with_file_validation_time(std::time::SystemTime::now());
    let encoded = OpenAiResponsesCodec
        .encode_request(lingxi_llm_client::EncodeRequest::new(&req), &context)
        .unwrap();
    let body: Value = serde_json::from_slice(&encoded.body).unwrap();
    assert_eq!(
        body["tools"][0]["container"],
        json!({"type":"auto", "file_ids":["file-abc", "file-def"]})
    );

    let mut duplicate = request(None);
    duplicate.hosted_tools = vec![HostedTool::CodeInterpreter(
        CodeInterpreterConfig::default().with_files([
            provider_file(&profile, "file-abc", files_scope),
            provider_file(&profile, "file-abc", files_scope),
        ]),
    )];
    assert!(matches!(
        OpenAiResponsesCodec
            .encode_request(lingxi_llm_client::EncodeRequest::new(&duplicate), &context),
        Err(LlmError::InvalidRequest { .. })
    ));

    let mut expired = request(None);
    let mut expired_file = provider_file(&profile, "file-old", files_scope);
    expired_file.expires_at = Some("2000-01-01T00:00:00Z".into());
    expired.hosted_tools = vec![HostedTool::CodeInterpreter(
        CodeInterpreterConfig::default().with_files([expired_file]),
    )];
    assert!(matches!(
        OpenAiResponsesCodec
            .encode_request(lingxi_llm_client::EncodeRequest::new(&expired), &context),
        Err(LlmError::InvalidRequest { .. })
    ));

    let mut mismatched_file = request(None);
    mismatched_file.hosted_tools = vec![HostedTool::CodeInterpreter(
        CodeInterpreterConfig::default().with_files([provider_file(
            &profile,
            "file-other-account",
            "another-files-scope",
        )]),
    )];
    assert!(matches!(
        OpenAiResponsesCodec.encode_request(
            lingxi_llm_client::EncodeRequest::new(&mismatched_file),
            &context
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut unsafe_id = request(None);
    unsafe_id.hosted_tools = vec![HostedTool::CodeInterpreter(
        CodeInterpreterConfig::default().with_files([provider_file(
            &profile,
            "file/escape",
            files_scope,
        )]),
    )];
    assert!(matches!(
        OpenAiResponsesCodec
            .encode_request(lingxi_llm_client::EncodeRequest::new(&unsafe_id), &context),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn explicit_container_reuse_is_scope_bound_and_auto_settings_are_rejected() {
    let profile = profile("openai", true);
    let scope = OpenAiContainerScope::new("openai", "container-account").unwrap();
    let container = OpenAiContainerRef::from_id(&scope, "cntr_abc123").unwrap();
    let mut req = request(None);
    req.hosted_tools = vec![HostedTool::CodeInterpreter(
        CodeInterpreterConfig::default().with_container(container.clone()),
    )];
    let context = wire_api::context(&profile, "m", &RequestOptions::default())
        .with_account_scope(Some("container-account"));
    let encoded = OpenAiResponsesCodec
        .encode_request(lingxi_llm_client::EncodeRequest::new(&req), &context)
        .unwrap();
    let body: Value = serde_json::from_slice(&encoded.body).unwrap();
    assert_eq!(body["tools"][0]["container"], "cntr_abc123");

    let mismatched_context = context.clone().with_account_scope(Some("other-account"));
    assert!(matches!(
        OpenAiResponsesCodec.encode_request(
            lingxi_llm_client::EncodeRequest::new(&req),
            &mismatched_context
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    req.hosted_tools = vec![HostedTool::CodeInterpreter(CodeInterpreterConfig {
        memory_limit: Some(CodeInterpreterMemoryLimit::FourG),
        container: Some(container),
        ..CodeInterpreterConfig::default()
    })];
    assert!(matches!(
        OpenAiResponsesCodec.encode_request(lingxi_llm_client::EncodeRequest::new(&req), &context),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[tokio::test]
async fn compatible_profiles_without_first_party_adapter_are_refused_before_network() {
    for (provider, enabled) in [("openrouter", true), ("openai", false)] {
        let p = profile(provider, enabled);
        let client = LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[p])
            .with_region(Region::International)
            .build()
            .unwrap();
        assert!(matches!(
            client
                .chat()
                .complete(&request(None), &RequestOptions::default())
                .await,
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
}

#[derive(Clone, Copy)]
enum DispatchFailure {
    Network,
    ServerError,
}

struct CodeInterpreterFailureTransport {
    failure: DispatchFailure,
    calls: AtomicUsize,
    encoded_tool_requests: AtomicUsize,
}

#[async_trait]
impl Transport for CodeInterpreterFailureTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        if body["tools"]
            .as_array()
            .is_some_and(|tools| tools.iter().any(|tool| tool["type"] == "code_interpreter"))
        {
            self.encoded_tool_requests.fetch_add(1, Ordering::Relaxed);
        }
        match self.failure {
            DispatchFailure::Network => Err(LlmError::Transport {
                message: "connection interrupted after dispatch".into(),
            }),
            DispatchFailure::ServerError => Ok(HttpResponse {
                status: 500,
                headers: vec![],
                body: Bytes::from_static(br#"{"error":{"message":"server error"}}"#),
            }
            .into()),
        }
    }
}

fn failover_profile(name: &str, order: u32) -> ProviderProfile {
    let mut profile = profile("openai", true);
    profile.profile_name = name.into();
    profile.connection = ConnectionSpec {
        group: Some("openai-code-interpreter-no-replay".into()),
        connection_id: Some(name.into()),
        order,
        hidden: false,
        failover: FailoverTriggers {
            network: true,
            server_error: true,
            ..FailoverTriggers::default()
        },
    };
    profile
}

#[tokio::test]
async fn code_interpreter_dispatch_is_never_replayed_after_ambiguous_failure() {
    for failure in [DispatchFailure::Network, DispatchFailure::ServerError] {
        for streaming in [false, true] {
            let transport = Arc::new(CodeInterpreterFailureTransport {
                failure,
                calls: AtomicUsize::new(0),
                encoded_tool_requests: AtomicUsize::new(0),
            });
            let profiles = [
                failover_profile("primary", 0),
                failover_profile("secondary", 1),
            ];
            let client = LlmClientBuilder::with_transport(transport.clone(), &profiles)
                .with_region(Region::International)
                .build()
                .unwrap();
            let req = request(None);
            let result = if streaming {
                client
                    .chat()
                    .stream(&req, &RequestOptions::default())
                    .await
                    .map(|_| ())
            } else {
                client
                    .chat()
                    .complete(&req, &RequestOptions::default())
                    .await
                    .map(|_| ())
            };
            assert!(result.is_err());
            assert_eq!(transport.calls.load(Ordering::Relaxed), 1);
            assert_eq!(transport.encoded_tool_requests.load(Ordering::Relaxed), 1);
        }
    }
}
