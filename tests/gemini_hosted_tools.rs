//! First-party Gemini GenerateContent hosted tools, native context replay,
//! and preflight routing.

#[path = "support/wire_api.rs"]
mod wire_api;

use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::protocol::{
    AttachmentRef, AuthStrategy, ChatRequest, ContentBlock, DocumentSource, HostedTool, LlmError,
    ProtocolFamily, ProviderFileSource, ProviderId, ProviderProfile, Region, Secret, StreamEvent,
    ToolSpec, WebSearchConfig,
};
use lingxi_llm_client::providers::google::types::{GeminiLatLng, GeminiMapsGroundingConfig};
use lingxi_llm_client::{
    AttachmentResolver, Authenticator, EncodeRequest, GeminiCodec, HttpRequest, HttpResponse,
    LlmClientBuilder, RequestOptions, StreamResponse, Transport, VertexGeminiCodec, WireCodec,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

const GEMINI_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

fn gemini_profile(model: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "google",
        "profile_name": "gemini",
        "base_url": GEMINI_BASE,
        "protocol": "gemini_generate_content",
        "auth": "none",
        "extra": {"web_search": "gemini"},
        "models": [{"display_model": model, "request_model": model, "billing_model": model}]
    }))
    .unwrap()
}

fn new_request(model: &str) -> ChatRequest {
    serde_json::from_value(json!({
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Explain this result."}]}]
    }))
    .unwrap()
}

fn context(profile: &ProviderProfile, model: &str) -> lingxi_llm_client::CodecContext {
    lingxi_llm_client::CodecContext::new(profile, model, lingxi_llm_client::RequestMode::Complete)
}

fn body(request: &HttpRequest) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

fn encode(request: &ChatRequest, model: &str) -> HttpRequest {
    let profile = gemini_profile(model);
    GeminiCodec
        .encode_request(EncodeRequest::new(request), &context(&profile, model))
        .unwrap()
}

fn response(parts: Value, candidate_metadata: Value) -> HttpResponse {
    let mut candidate = json!({
        "content": {"role": "model", "parts": parts},
        "finishReason": "STOP"
    });
    if let Some(fields) = candidate_metadata.as_object() {
        for (key, value) in fields {
            candidate[key] = value.clone();
        }
    }
    HttpResponse {
        status: 200,
        headers: vec![],
        body: serde_json::to_vec(&json!({
            "modelVersion": "gemini-3.8-flash",
            "candidates": [candidate],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7}
        }))
        .unwrap()
        .into(),
    }
}

#[test]
fn first_party_tools_use_native_generate_content_shapes() {
    let model = "gemini-3.8-flash";
    let mut code = new_request(model);
    code.hosted_tools
        .push(lingxi_llm_client::providers::google::native::GoogleHostedTool::CodeExecution.into());
    let code_body = body(&encode(&code, model));
    assert_eq!(code_body["tools"], json!([{"codeExecution": {}}]));
    assert_eq!(
        code_body["toolConfig"],
        json!({
            "includeServerSideToolInvocations": true,
            "functionCallingConfig": {"mode": "VALIDATED"}
        })
    );

    let model = "gemini-2.5-flash";
    let mut url = new_request(model);
    url.hosted_tools
        .push(lingxi_llm_client::providers::google::native::GoogleHostedTool::UrlContext.into());
    let url_body = body(&encode(&url, model));
    assert_eq!(url_body["tools"], json!([{"urlContext": {}}]));
    assert!(url_body.get("toolConfig").is_none());

    let mut maps = new_request(model);
    maps.hosted_tools.push(
        lingxi_llm_client::providers::google::native::GoogleHostedTool::MapsGrounding(
            GeminiMapsGroundingConfig {
                enable_widget: Some(false),
                lat_lng: Some(GeminiLatLng::new(37.78193, -122.40476).unwrap()),
            },
        )
        .into(),
    );
    let maps_body = body(&encode(&maps, model));
    assert_eq!(
        maps_body["tools"],
        json!([{"googleMaps": {"enableWidget": false}}])
    );
    assert_eq!(
        maps_body["toolConfig"]["retrievalConfig"]["latLng"],
        json!({"latitude": 37.78193, "longitude": -122.40476})
    );
}

#[test]
fn gemini_3_functions_and_google_search_use_tool_context_circulation() {
    let model = "gemini-3.8-flash";
    let mut request = new_request(model);
    request.hosted_tools = vec![
        HostedTool::WebSearch(WebSearchConfig::default()),
        lingxi_llm_client::providers::google::native::GoogleHostedTool::CodeExecution.into(),
    ];
    request.tools.push(ToolSpec {
        input_schema_json: None,
        tool_type: None,
        extra: serde_json::Value::Null,
        name: "get_weather".into(),
        description: "Get the weather for a city".into(),
        input_schema: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        strict: false,
        defer_loading: false,
        native_options: Vec::new(),
    });
    let encoded = body(&encode(&request, model));
    assert_eq!(
        encoded["tools"][0]["functionDeclarations"][0]["name"],
        "get_weather"
    );
    assert_eq!(encoded["tools"][1], json!({"googleSearch": {}}));
    assert_eq!(encoded["tools"][2], json!({"codeExecution": {}}));
    assert!(encoded["toolConfig"]["includeServerSideToolInvocations"]
        .as_bool()
        .unwrap());
    assert_eq!(
        encoded["toolConfig"]["functionCallingConfig"]["mode"],
        "VALIDATED"
    );
}

#[test]
fn preflight_refuses_undocumented_routes_models_and_tool_combinations() {
    let model = "gemini-2.5-flash";
    let mut request = new_request(model);
    request
        .hosted_tools
        .push(lingxi_llm_client::providers::google::native::GoogleHostedTool::CodeExecution.into());
    request.tools.push(ToolSpec {
        input_schema_json: None,
        tool_type: None,
        extra: serde_json::Value::Null,
        name: "f".into(),
        description: "f".into(),
        input_schema: json!({"type": "object"}),
        strict: false,
        defer_loading: false,
        native_options: Vec::new(),
    });
    let profile = gemini_profile(model);
    assert!(matches!(
        GeminiCodec.validate_request(&request, &context(&profile, model)),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let model = "gemini-3.1-pro-preview";
    let mut maps_search = new_request(model);
    maps_search.tools.clear();
    maps_search.hosted_tools = vec![
        lingxi_llm_client::providers::google::native::GoogleHostedTool::MapsGrounding(
            GeminiMapsGroundingConfig::default(),
        )
        .into(),
        HostedTool::WebSearch(WebSearchConfig::default()),
    ];
    let profile = gemini_profile(model);
    assert!(matches!(
        GeminiCodec.validate_request(&maps_search, &context(&profile, model)),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let model = "gemini-3.8-flash";
    let mut multiple = new_request(model);
    multiple.hosted_tools = vec![
        lingxi_llm_client::providers::google::native::GoogleHostedTool::UrlContext.into(),
        lingxi_llm_client::providers::google::native::GoogleHostedTool::MapsGrounding(
            GeminiMapsGroundingConfig::default(),
        )
        .into(),
    ];
    let profile = gemini_profile(model);
    assert!(matches!(
        GeminiCodec.validate_request(&multiple, &context(&profile, model)),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let model = "gemini-2.5-flash-image";
    let mut unsupported_model = new_request(model);
    unsupported_model
        .hosted_tools
        .push(lingxi_llm_client::providers::google::native::GoogleHostedTool::CodeExecution.into());
    let profile = gemini_profile(model);
    assert!(matches!(
        GeminiCodec.validate_request(&unsupported_model, &context(&profile, model)),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut gateway = gemini_profile("gemini-3.8-flash");
    gateway.base_url = "https://gateway.example/v1beta".into();
    let mut hosted = new_request("gemini-3.8-flash");
    hosted
        .hosted_tools
        .push(lingxi_llm_client::providers::google::native::GoogleHostedTool::UrlContext.into());
    assert!(matches!(
        GeminiCodec.validate_request(&hosted, &context(&gateway, "gemini-3.8-flash")),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut vertex = gemini_profile("gemini-3.8-flash");
    vertex.protocol = ProtocolFamily::VertexGemini;
    let vertex_context = context(&vertex, "gemini-3.8-flash");
    assert!(matches!(
        VertexGeminiCodec.validate_request(&hosted, &vertex_context),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn maps_location_and_multimodal_constraints_are_preflighted() {
    assert!(GeminiLatLng::new(90.01, 0.0).is_err());
    assert!(GeminiLatLng::new(0.0, f64::INFINITY).is_err());

    let model = "gemini-3.8-flash";
    let mut multimodal = new_request(model);
    multimodal.messages[0].content.push(ContentBlock::Image {
        source: lingxi_llm_client::protocol::ImageSource::Base64 {
            media_type: "image/png".into(),
            data: "aGVsbG8=".into(),
        },
    });
    multimodal.hosted_tools.push(
        lingxi_llm_client::providers::google::native::GoogleHostedTool::MapsGrounding(
            GeminiMapsGroundingConfig::default(),
        )
        .into(),
    );
    let profile = gemini_profile(model);
    assert!(matches!(
        GeminiCodec.validate_request(&multimodal, &context(&profile, model)),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut inline_history = new_request(model);
    inline_history
        .messages
        .push(lingxi_llm_client::protocol::ConversationMessage {
            native_options: Vec::new(),
            role: lingxi_llm_client::protocol::MessageRole::Assistant,
            content: vec![ContentBlock::ProviderContent {
                protocol: ProtocolFamily::GeminiGenerateContent,
                value: json!({"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}),
            }],
        });
    inline_history.hosted_tools.push(
        lingxi_llm_client::providers::google::native::GoogleHostedTool::MapsGrounding(
            GeminiMapsGroundingConfig::default(),
        )
        .into(),
    );
    assert!(matches!(
        GeminiCodec.validate_request(&inline_history, &context(&profile, model)),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let mut multimodal_output_profile = gemini_profile(model);
    multimodal_output_profile.extra = json!({
        "web_search": "gemini",
        "body": {"generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}}
    });
    let mut maps = new_request(model);
    maps.hosted_tools.push(
        lingxi_llm_client::providers::google::native::GoogleHostedTool::MapsGrounding(
            GeminiMapsGroundingConfig::default(),
        )
        .into(),
    );
    assert!(matches!(
        GeminiCodec.validate_request(&maps, &context(&multimodal_output_profile, model)),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn native_code_artifact_and_server_tool_parts_round_trip_without_rewriting() {
    let code_parts = json!([
        {"executableCode": {"language": "PYTHON", "code": "print(1)"}, "thoughtSignature": "opaque"},
        {"codeExecutionResult": {"outcome": "OUTCOME_UNSPECIFIED", "output": "1"}},
        {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}, "thoughtSignature": "image-sig"}
    ]);
    let profile = gemini_profile("gemini-3.8-flash");
    let decoded = GeminiCodec
        .decode_response(
            &response(code_parts.clone(), Value::Null),
            &context(&profile, "gemini-3.8-flash"),
        )
        .unwrap();
    assert_eq!(decoded.message.content.len(), 3);
    for (block, original) in decoded
        .message
        .content
        .iter()
        .zip(code_parts.as_array().unwrap())
    {
        assert!(
            matches!(block, ContentBlock::ProviderContent { protocol: ProtocolFamily::GeminiGenerateContent, value } if value == original)
        );
    }

    let mut follow_up = new_request("gemini-3.8-flash");
    follow_up
        .hosted_tools
        .push(lingxi_llm_client::providers::google::native::GoogleHostedTool::CodeExecution.into());
    follow_up.messages.push(decoded.message);
    let encoded = body(&encode(&follow_up, "gemini-3.8-flash"));
    assert_eq!(encoded["contents"][1]["parts"], code_parts);

    let tool_parts = json!([
        {"toolCall": {"toolType": "URL_CONTEXT", "toolName": "url_context", "id": "server-1", "args": {"url": "https://example.com"}}, "thoughtSignature": "s"},
        {"toolResponse": {"toolType": "URL_CONTEXT", "id": "server-1", "response": {"status": "SUCCESS"}}}
    ]);
    let decoded = GeminiCodec
        .decode_response(
            &response(tool_parts.clone(), Value::Null),
            &context(&profile, "gemini-3.8-flash"),
        )
        .unwrap();
    let mut follow_up = new_request("gemini-3.8-flash");
    follow_up
        .hosted_tools
        .push(lingxi_llm_client::providers::google::native::GoogleHostedTool::UrlContext.into());
    follow_up.messages.push(decoded.message);
    let encoded = body(&encode(&follow_up, "gemini-3.8-flash"));
    assert_eq!(encoded["contents"][1]["parts"], tool_parts);
}

#[test]
fn unary_and_stream_outputs_keep_url_maps_and_native_parts_separate() {
    let model = "gemini-3.8-flash";
    let candidate_metadata = json!({
        "groundingMetadata": {"groundingChunks": [{"maps": {
            "uri": "https://maps.google.com/?cid=123",
            "title": "Library",
            "placeId": "places/abc"
        }}]},
        "urlContextMetadata": {"urlMetadata": [{
            "retrievedUrl": "https://example.com/article",
            "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"
        }]}
    });
    let native_part = json!({"toolCall": {"toolType": "GOOGLE_MAPS", "id": "map-1", "args": {}}});
    let response = response(
        json!([native_part.clone(), {"text": "nearby"}]),
        candidate_metadata.clone(),
    );
    let profile = gemini_profile(model);
    let decoded = GeminiCodec
        .decode_response(&response, &context(&profile, model))
        .unwrap();
    let search = decoded.web_search.unwrap();
    assert_eq!(
        search.metadata["groundingMetadata"],
        candidate_metadata["groundingMetadata"]
    );
    assert_eq!(
        search.metadata["urlContextMetadata"],
        candidate_metadata["urlContextMetadata"]
    );
    assert!(search.citations.iter().any(|citation| {
        citation.url == "https://maps.google.com/?cid=123"
            && citation.title.as_deref() == Some("Library")
    }));
    assert!(search.citations.iter().any(|citation| {
        citation.url == "https://example.com/article" && citation.title.is_none()
    }));
    assert_eq!(decoded.message.content.len(), 2);
    assert!(matches!(
        &decoded.message.content[0],
        ContentBlock::ProviderContent { value, .. } if value == &native_part
    ));

    let stream_profile = gemini_profile(model);
    let stream_context = context(&stream_profile, model);
    let mut decoder = GeminiCodec.stream_decoder(&stream_context);
    let frame = json!({
        "modelVersion": model,
        "candidates": [{
            "content": {"role": "model", "parts": [native_part.clone(), {"text": "nearby"}]},
            "groundingMetadata": candidate_metadata["groundingMetadata"].clone(),
            "urlContextMetadata": candidate_metadata["urlContextMetadata"].clone()
        }]
    });
    let events = wire_api::decode_frame(&mut *decoder, frame.to_string().as_bytes()).unwrap();
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ProviderContent { protocol: ProtocolFamily::GeminiGenerateContent, value, .. }
            if value == &native_part
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::WebSearch { result }
            if result.metadata.get("groundingMetadata").is_some()
                && result.metadata.get("urlContextMetadata").is_some()
                && result.citations.iter().any(|citation| citation.url == "https://maps.google.com/?cid=123")
    )));
}

struct CountingResolver(Arc<AtomicUsize>);

#[async_trait]
impl AttachmentResolver for CountingResolver {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(Bytes::from_static(b"document"))
    }
}

struct CountingAuth(Arc<AtomicUsize>);

#[async_trait]
impl Authenticator for CountingAuth {
    async fn apply(
        &self,
        _: &mut HttpRequest,
        _: &ProviderProfile,
        _: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

struct CountingTransport(Arc<AtomicUsize>);

#[async_trait]
impl Transport for CountingTransport {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Err(LlmError::Transport {
            message: "unexpected network dispatch".into(),
        })
    }
}

#[tokio::test]
async fn unsupported_cross_protocol_tool_fails_before_attachment_auth_or_network() {
    let attachment_reads = Arc::new(AtomicUsize::new(0));
    let auth_calls = Arc::new(AtomicUsize::new(0));
    let network_calls = Arc::new(AtomicUsize::new(0));
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id": "acme",
        "profile_name": "acme",
        "base_url": "https://api.example/v1",
        "protocol": "open_ai_chat",
        "auth": "api_key",
        "models": [{
            "display_model": "m",
            "request_model": "m",
            "billing_model": "m",
            "metadata": {"inputModalities": ["text", "file"]}
        }]
    }))
    .unwrap();
    let mut request = new_request("m");
    request
        .hosted_tools
        .push(lingxi_llm_client::providers::google::native::GoogleHostedTool::CodeExecution.into());
    request.messages[0].content.push(ContentBlock::Document {
        source: DocumentSource::Attachment {
            attachment: AttachmentRef {
                attachment_id: "doc".into(),
                revision: "1".into(),
                filename: "doc.pdf".into(),
                media_type: "application/pdf".into(),
                size_bytes: 8,
            },
        },
        title: None,
    });

    let mut builder = LlmClientBuilder::with_transport(
        Arc::new(CountingTransport(network_calls.clone())),
        &[profile],
    )
    .with_region(Region::International);
    builder.with_attachment_resolver(Arc::new(CountingResolver(attachment_reads.clone())));
    builder.register_authenticator(
        AuthStrategy::ApiKey,
        Arc::new(CountingAuth(auth_calls.clone())),
    );
    let client = builder.build().unwrap();
    let result = client
        .chat()
        .complete(
            &request,
            &RequestOptions {
                credential: Some(Secret::new("key".into())),
                ..RequestOptions::default()
            },
        )
        .await;
    assert!(matches!(
        result,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert_eq!(attachment_reads.load(Ordering::Relaxed), 0);
    assert_eq!(auth_calls.load(Ordering::Relaxed), 0);
    assert_eq!(network_calls.load(Ordering::Relaxed), 0);
}

#[derive(Clone, Copy)]
enum RetryFailure {
    Network,
    ServerError,
}

struct HostedRetryTransport {
    failure: RetryFailure,
    calls: AtomicUsize,
}

#[async_trait]
impl Transport for HostedRetryTransport {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed);
        if call == 0 {
            return match self.failure {
                RetryFailure::Network => Err(LlmError::Transport {
                    message: "outcome unknown".into(),
                }),
                RetryFailure::ServerError => Ok(HttpResponse {
                    status: 500,
                    headers: vec![],
                    body: Bytes::from_static(
                        br#"{"error":{"status":"INTERNAL","message":"server error"}}"#,
                    ),
                }
                .into()),
            };
        }
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: Bytes::from_static(
                br#"{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}"#,
            ),
        }
        .into())
    }
}

fn failover_profile(name: &str, order: u32) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "google",
        "profile_name": name,
        "base_url": GEMINI_BASE,
        "protocol": "gemini_generate_content",
        "auth": "none",
        "regions": ["international"],
        "models": [{
            "display_model": "gemini-3.8-flash",
            "request_model": "gemini-3.8-flash",
            "billing_model": "gemini-3.8-flash",
            "metadata": {"inputModalities": ["text", "file"]},
            "capability_support": {"documents": "supported"}
        }],
        "connection": {
            "group": "gemini-hosted-tool-failover",
            "order": order,
            "failover": {
                "rateLimit": true,
                "overloaded": true,
                "serverError": true,
                "network": true,
                "auth": true
            }
        }
    }))
    .unwrap()
}

#[tokio::test]
async fn server_side_tools_are_not_replayed_after_uncertain_outcomes() {
    for failure in [RetryFailure::Network, RetryFailure::ServerError] {
        for streaming in [false, true] {
            let transport = Arc::new(HostedRetryTransport {
                failure,
                calls: AtomicUsize::new(0),
            });
            let profiles = [
                failover_profile("primary", 0),
                failover_profile("secondary", 1),
            ];
            let client = LlmClientBuilder::with_transport(transport.clone(), &profiles)
                .with_region(Region::International)
                .build()
                .unwrap();
            let mut req = new_request("gemini-3.8-flash");
            req.hosted_tools.push(
                lingxi_llm_client::providers::google::native::GoogleHostedTool::CodeExecution
                    .into(),
            );
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
        }
    }
}

struct MissingFileTransport(AtomicUsize);

#[async_trait]
impl Transport for MissingFileTransport {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(HttpResponse {
            status: 404,
            headers: vec![],
            body: Bytes::from_static(b"file files/missing-file not found"),
        }
        .into())
    }
}

#[tokio::test]
async fn missing_provider_file_does_not_automatically_repeat_hosted_execution() {
    let transport = Arc::new(MissingFileTransport(AtomicUsize::new(0)));
    let profiles = [
        failover_profile("primary", 0),
        failover_profile("secondary", 1),
    ];
    let primary = &profiles[0];
    let client = LlmClientBuilder::with_transport(transport.clone(), &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let mut req = new_request("gemini-3.8-flash");
    req.hosted_tools
        .push(lingxi_llm_client::providers::google::native::GoogleHostedTool::CodeExecution.into());
    req.messages[0].content.push(ContentBlock::Document {
        source: DocumentSource::ProviderFile {
            file: ProviderFileSource {
                protocol: ProtocolFamily::GeminiGenerateContent,
                provider_id: ProviderId::new("google"),
                profile_name: primary.profile_name.clone(),
                endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
                    &primary.base_url,
                ),
                account_scope: Some("account-1".into()),
                expires_at: None,
                processing_status: None,
                file_id: "files/missing-file".into(),
                uri: Some(format!("{GEMINI_BASE}/files/missing-file")),
                media_type: Some("application/pdf".into()),
                purpose: None,
            },
        },
        title: None,
    });
    let result = client
        .chat()
        .complete(
            &req,
            &RequestOptions {
                account_scope: Some("account-1".into()),
                file_account_scope: Some("account-1".into()),
                ..RequestOptions::default()
            },
        )
        .await;
    assert!(result.is_err());
    assert_eq!(transport.0.load(Ordering::Relaxed), 1);
}
