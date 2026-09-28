use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::providers::anthropic::types::*;
use lingxi_llm_client::{
    files::{FilePurpose, FileService, UploadFile},
    protocol::*,
    CodecContext, EncodeRequest, FoundryClaudeCodec, HttpRequest, HttpResponse, RequestMode,
    StreamResponse, Transport, WireCodec,
};
use serde_json::{json, Value};
use std::sync::Mutex;

#[path = "support/wire_api.rs"]
mod wire_api;

const PROFILE: &str = "foundry-anthropic";
const ENDPOINT: &str = "https://resource-a.services.ai.azure.com/anthropic";
const ACCOUNT: &str = "foundry-resource-a";
const DEPLOYMENT: &str = "custom-claude-deployment";
const MODEL_ID: &str = "claude-opus-5-5";

fn profile(hosting: FoundryHosting, model_id: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"custom-foundry", "profile_name":PROFILE,
        "protocol":"foundry_claude", "base_url":ENDPOINT,
        "auth":"none", "regions":["international"],
        "models":[{
            "display_model":"selected", "request_model":DEPLOYMENT,
            "billing_model":"not-a-model-id",
            "aliases":["selected-alias"],
            "foundry":{
                "hosting":match hosting {
                    FoundryHosting::Azure => "azure",
                    FoundryHosting::Anthropic => "anthropic",
                },
                "model_id":model_id
            }
        }]
    }))
    .unwrap()
}

fn request() -> ChatRequest {
    let mut request: ChatRequest = serde_json::from_value(json!({
        "model":DEPLOYMENT,
        "max_tokens":1024,
        "messages":[{"role":"user","content":[{"type":"text","text":"Compute 6 times 7."}]}]
    }))
    .unwrap();
    request.hosted_tools.push(
        lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(
            Default::default(),
        )
        .into(),
    );
    request
}

fn context(profile: &ProviderProfile, mode: RequestMode) -> CodecContext {
    CodecContext::new(profile, DEPLOYMENT, mode)
        .with_account_scope(Some(ACCOUNT))
        .with_file_scope(Some(ACCOUNT))
}

fn encode(request: &ChatRequest, profile: &ProviderProfile) -> Result<(String, Value), LlmError> {
    let wire = FoundryClaudeCodec.encode_request(
        EncodeRequest::new(request),
        &context(profile, RequestMode::Complete),
    )?;
    Ok((wire.url, serde_json::from_slice(&wire.body).unwrap()))
}

fn container_scope(profile: &ProviderProfile) -> AnthropicContainerScope {
    let deployment = profile.models[0].foundry.clone().unwrap();
    AnthropicContainerScope::new_foundry(PROFILE, ENDPOINT, ACCOUNT, DEPLOYMENT, deployment)
        .unwrap()
}

fn programmatic_tool() -> ToolSpec {
    ToolSpec {
        tool_type: None,
        extra: serde_json::Value::Null,
        name: "lookup".into(),
        description: "Find one record".into(),
        input_schema: json!({"type":"object","properties":{"query":{"type":"string"}}}),
        strict: false,
        defer_loading: false,
        native_options: vec![lingxi_llm_client::protocol::NativeExtension::from_typed(
            lingxi_llm_client::providers::anthropic::native::AnthropicToolOptions {
                allowed_callers: vec![AnthropicToolCaller::CodeExecution20260120],
            },
        )
        .unwrap()],
    }
}

fn response(container: Value) -> Value {
    json!({
        "id":"msg_foundry", "type":"message", "role":"assistant",
        "model":DEPLOYMENT, "content":[], "stop_reason":"end_turn",
        "usage":{"input_tokens":2,"output_tokens":1,
            "server_tool_use":{"code_execution_requests":1}},
        "container":container
    })
}

fn foundry_file(
    endpoint: &str,
    account_scope: &str,
    protocol: ProtocolFamily,
    id: &str,
) -> ProviderFileSource {
    ProviderFileSource {
        protocol,
        provider_id: "custom-foundry".into(),
        profile_name: PROFILE.into(),
        endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
            endpoint,
        ),
        account_scope: Some(account_scope.into()),
        expires_at: None,
        processing_status: None,
        file_id: id.into(),
        uri: None,
        media_type: Some("text/csv".into()),
        purpose: None,
    }
}

#[derive(Default)]
struct FoundryFileTransport {
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for FoundryFileTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        Ok(HttpResponse {
            status: 200,
            headers: vec![("content-type".into(), "application/json".into())],
            body: json!({
                "id":"file_uploaded_from_foundry",
                "filename":"input.csv",
                "size_bytes":3,
                "mime_type":"text/csv",
                "expires_at":"2099-01-01T00:00:00Z"
            })
            .to_string()
            .into(),
        }
        .into())
    }
}

#[test]
fn anthropic_hosted_foundry_encodes_code_execution_on_its_resource_route() {
    let profile = profile(FoundryHosting::Anthropic, MODEL_ID);
    let (url, body) = encode(&request(), &profile).unwrap();
    assert_eq!(
        url,
        "https://resource-a.services.ai.azure.com/anthropic/v1/messages"
    );
    assert_eq!(body["model"], DEPLOYMENT);
    assert_eq!(
        body["tools"],
        json!([{"type":"code_execution_20260521","name":"code_execution"}])
    );
    assert!(body.get("foundry").is_none());
    assert!(body.get("model_id").is_none());
}

#[test]
fn container_reuse_binds_foundry_resource_account_deployment_and_underlying_model() {
    let profile = profile(FoundryHosting::Anthropic, MODEL_ID);
    let scope = container_scope(&profile);
    assert_eq!(scope.endpoint(), ENDPOINT);
    assert_eq!(
        scope.foundry_deployment(),
        profile.models[0].foundry.as_ref()
    );

    let reference = AnthropicContainerRef::new("container_foundry", scope).unwrap();
    let mut request = request();
    request.hosted_tools[0].edit_native::<lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool, _>(|tool| {
        if let lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(config) = tool {
        config.container = Some(reference);
    } else { unreachable!(); }
    }).unwrap();
    let (_, body) = encode(&request, &profile).unwrap();
    assert_eq!(body["container"], "container_foundry");

    let mut wrong_model = profile.clone();
    wrong_model.models[0].foundry.as_mut().unwrap().model_id = "claude-sonnet-5".into();
    assert!(matches!(
        encode(&request, &wrong_model),
        Err(LlmError::InvalidRequest { .. })
    ));

    let mut wrong_account_context = CodecContext::new(&profile, DEPLOYMENT, RequestMode::Complete)
        .with_account_scope(Some("another-resource"));
    let wire_result =
        FoundryClaudeCodec.encode_request(EncodeRequest::new(&request), &wrong_account_context);
    assert!(matches!(wire_result, Err(LlmError::InvalidRequest { .. })));
    wrong_account_context = wrong_account_context.with_account_scope(Some(ACCOUNT));
    assert!(FoundryClaudeCodec
        .encode_request(EncodeRequest::new(&request), &wrong_account_context)
        .is_ok());
}

#[test]
fn code_execution_and_programmatic_calling_require_exact_anthropic_hosting_identity() {
    let azure = profile(FoundryHosting::Azure, MODEL_ID);
    assert!(matches!(
        encode(&request(), &azure),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut programmatic = request();
    programmatic.tools.push(programmatic_tool());
    let (_, body) = encode(&programmatic, &profile(FoundryHosting::Anthropic, MODEL_ID)).unwrap();
    assert!(body["tools"].as_array().unwrap().iter().any(|tool| {
        tool["name"] == "lookup" && tool["allowed_callers"] == json!(["code_execution_20260120"])
    }));
    assert!(matches!(
        encode(&programmatic, &azure),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let haiku = profile(FoundryHosting::Anthropic, "claude-haiku-4-5");
    assert!(encode(&request(), &haiku).is_ok());
    assert!(matches!(
        encode(&programmatic, &haiku),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let preview = profile(FoundryHosting::Anthropic, "claude-mythos-preview");
    assert!(encode(&request(), &preview).is_ok());
    assert!(matches!(
        encode(&programmatic, &preview),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn foundry_container_scope_is_explicit_and_rejects_other_routes() {
    let profile = profile(FoundryHosting::Anthropic, MODEL_ID);
    let deployment = profile.models[0].foundry.clone().unwrap();
    assert!(AnthropicContainerScope::new(PROFILE, ENDPOINT, ACCOUNT, DEPLOYMENT).is_err());
    assert!(AnthropicContainerScope::new_foundry(
        PROFILE,
        ENDPOINT,
        ACCOUNT,
        DEPLOYMENT,
        FoundryDeployment {
            hosting: FoundryHosting::Azure,
            model_id: MODEL_ID.into(),
        },
    )
    .is_err());

    for invalid_endpoint in [
        "http://resource-a.services.ai.azure.com/anthropic",
        "https://resource-a.services.ai.azure.com/anthropic/v1",
        "https://resource-a.services.ai.azure.com/anthropic?redirect=true",
        "https://proxy.example.test/anthropic",
        "https://api.anthropic.com",
    ] {
        assert!(AnthropicContainerScope::new_foundry(
            PROFILE,
            invalid_endpoint,
            ACCOUNT,
            DEPLOYMENT,
            deployment.clone(),
        )
        .is_err());
    }
}

#[test]
fn foundry_file_references_are_resource_and_account_scoped_not_model_scoped() {
    let profile = profile(FoundryHosting::Anthropic, MODEL_ID);
    let mut request = request();
    request.hosted_tools[0].edit_native::<lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool, _>(|tool| {
        if let lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(config) = tool {
        config.files.push(foundry_file(
            ENDPOINT,
            ACCOUNT,
            ProtocolFamily::FoundryClaude,
            "foundry_file_123",
        ));
    } else { unreachable!(); }
    }).unwrap();
    let (_, body) = encode(&request, &profile).unwrap();
    assert!(
        body["messages"][0]["content"]
            .as_array()
            .unwrap()
            .iter()
            .any(|block| block["type"] == "container_upload"
                && block["file_id"] == "foundry_file_123")
    );

    // The same resource and account can use the file from another supported
    // deployment because this FileService reference does not capture a model.
    let mut another_model = profile.clone();
    another_model.models[0].foundry.as_mut().unwrap().model_id = "claude-sonnet-5".into();
    assert!(encode(&request, &another_model).is_ok());
}

#[tokio::test]
async fn foundry_file_service_upload_reference_is_usable_by_code_execution() {
    let foundry_profile = profile(FoundryHosting::Anthropic, MODEL_ID);
    let transport = FoundryFileTransport::default();
    assert!(
        !FileService::new(&transport, &foundry_profile, None, None, Some(ACCOUNT))
            .capabilities(DEPLOYMENT, "text/csv")
            .upload
    );
    let azure_profile = profile(FoundryHosting::Azure, MODEL_ID);
    assert!(FileService::new_foundry_for_model(
        &transport,
        &azure_profile,
        &azure_profile.models[0],
        None,
        None,
        ACCOUNT,
    )
    .is_err());

    let service = FileService::new_foundry_for_model(
        &transport,
        &foundry_profile,
        &foundry_profile.models[0],
        None,
        None,
        ACCOUNT,
    )
    .unwrap();
    let uploaded = service
        .upload(
            &UploadFile {
                filename: "input.csv".into(),
                media_type: "text/csv".into(),
                bytes: Bytes::from_static(b"a,b"),
            },
            FilePurpose::ModelInput,
        )
        .await
        .unwrap();

    assert_eq!(uploaded.protocol, ProtocolFamily::FoundryClaude);
    assert_eq!(uploaded.provider_id.as_str(), "custom-foundry");
    assert_eq!(uploaded.profile_name, PROFILE);
    assert_eq!(uploaded.account_scope.as_deref(), Some(ACCOUNT));
    assert_eq!(
        uploaded.endpoint_fingerprint,
        lingxi_llm_client::files::provider_file_endpoint_fingerprint(ENDPOINT)
    );
    assert_eq!(uploaded.expires_at.as_deref(), Some("2099-01-01T00:00:00Z"));

    let mut request = request();
    request.hosted_tools[0].edit_native::<lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool, _>(|tool| {
        if let lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(config) = tool {
        config.files.push(uploaded.model_reference());
    } else { unreachable!(); }
    }).unwrap();
    let (_, body) = encode(&request, &foundry_profile).unwrap();
    assert!(body["messages"][0]["content"]
        .as_array()
        .unwrap()
        .iter()
        .any(|block| {
            block["type"] == "container_upload" && block["file_id"] == "file_uploaded_from_foundry"
        }));
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].url,
        "https://resource-a.services.ai.azure.com/anthropic/v1/files"
    );
    assert!(String::from_utf8_lossy(&requests[0].body).contains("filename=\"input.csv\""));
}

#[test]
fn foundry_file_references_reject_other_resource_route_protocol_and_account() {
    let foundry_profile = profile(FoundryHosting::Anthropic, MODEL_ID);
    let mismatched = [
        foundry_file(
            "https://another-resource.services.ai.azure.com/anthropic",
            ACCOUNT,
            ProtocolFamily::FoundryClaude,
            "file_other_resource",
        ),
        foundry_file(
            ENDPOINT,
            "another-account",
            ProtocolFamily::FoundryClaude,
            "file_other_account",
        ),
        foundry_file(
            ENDPOINT,
            ACCOUNT,
            ProtocolFamily::AnthropicMessages,
            "file_first_party",
        ),
    ];
    for file in mismatched {
        let mut request = request();
        request.hosted_tools[0].edit_native::<lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool, _>(|tool| {
        if let lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(config) = tool {
            config.files.push(file);
        } else { unreachable!(); }
    }).unwrap();
        assert!(matches!(
            encode(&request, &foundry_profile),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }

    let mut duplicate_request = request();
    duplicate_request.hosted_tools[0].edit_native::<lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool, _>(|tool| {
        if let lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(config) = tool {
        let duplicate = foundry_file(
            ENDPOINT,
            ACCOUNT,
            ProtocolFamily::FoundryClaude,
            "file_duplicate",
        );
        config.files.extend([duplicate.clone(), duplicate]);
    } else { unreachable!(); }
    }).unwrap();
    assert!(matches!(
        encode(&duplicate_request, &foundry_profile),
        Err(LlmError::InvalidRequest { .. })
    ));

    let mut expired_request = request();
    expired_request.hosted_tools[0].edit_native::<lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool, _>(|tool| {
        if let lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(config) = tool {
        let mut expired = foundry_file(
            ENDPOINT,
            ACCOUNT,
            ProtocolFamily::FoundryClaude,
            "file_expired",
        );
        expired.expires_at = Some("0".into());
        config.files.push(expired);
    } else { unreachable!(); }
    }).unwrap();
    assert!(matches!(
        encode(&expired_request, &foundry_profile),
        Err(LlmError::InvalidRequest { .. })
    ));

    let azure = profile(FoundryHosting::Azure, MODEL_ID);
    let mut azure_request = request();
    azure_request.hosted_tools[0].edit_native::<lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool, _>(|tool| {
        if let lingxi_llm_client::providers::anthropic::native::AnthropicHostedTool::CodeExecution(config) = tool {
        config.files.push(foundry_file(
            ENDPOINT,
            ACCOUNT,
            ProtocolFamily::FoundryClaude,
            "file_azure",
        ));
    } else { unreachable!(); }
    }).unwrap();
    assert!(matches!(
        encode(&azure_request, &azure),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn foundry_anthropic_host_response_and_stream_keep_container_and_usage_metadata() {
    let profile = profile(FoundryHosting::Anthropic, MODEL_ID);
    let complete_context = context(&profile, RequestMode::Complete);
    let container = json!({"id":"container_foundry","expires_at":"later","future_field":true});
    let http = HttpResponse {
        status: 200,
        headers: vec![],
        body: response(container.clone()).to_string().into(),
    };
    let decoded = FoundryClaudeCodec
        .decode_response(&http, &complete_context)
        .unwrap();
    assert_eq!(decoded.anthropic_container().unwrap().envelope, container);
    assert_eq!(
        decoded.anthropic_usage().unwrap()["server_tool_use"]["code_execution_requests"],
        1
    );

    let stream_context = context(&profile, RequestMode::Stream);
    let mut decoder = FoundryClaudeCodec.stream_decoder(&stream_context);
    let start = json!({
        "type":"message_start",
        "message":response(json!({"id":"container_start","extra":1}))
    });
    let mut events = wire_api::decode_frame(&mut *decoder, start.to_string().as_bytes()).unwrap();
    let delta = json!({
        "type":"message_delta",
        "delta":{"container":{"id":"container_delta","extra":2},"stop_reason":"end_turn"},
        "usage":{"output_tokens":1,"server_tool_use":{"code_execution_requests":1}}
    });
    events.extend(wire_api::decode_frame(&mut *decoder, delta.to_string().as_bytes()).unwrap());
    events.extend(wire_api::decode_frame(&mut *decoder, br#"{"type":"message_stop"}"#).unwrap());
    assert!(events.iter().any(|event| matches!(event,
        StreamEvent::ProviderEvent { payload, .. }
            if payload["message"]["container"]["id"] == "container_start"
    )));
    assert!(events.iter().any(|event| matches!(event,
        StreamEvent::ProviderEvent { payload, .. }
            if payload["delta"]["container"]["id"] == "container_delta"
    )));
    assert!(events.iter().any(|event| matches!(event,
        StreamEvent::End { usage, .. }
            if usage.usage.as_ref()
                .and_then(|usage| usage.server_tool_usage.as_ref())
                .and_then(|usage| usage.code_interpreter_requests) == Some(1)
    )));
}
