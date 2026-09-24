//! Public API regressions from the architecture review.
use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

fn profile(protocol: ProtocolFamily) -> ProviderProfile {
    serde_json::from_value(json!({
        "profile_name":"p", "provider_id":"openai", "base_url":"https://api.openai.com/v1",
        "protocol":protocol, "auth":"none",
        "models":[{"request_model":"wire", "display_model":"shared", "billing_model":"wire", "aliases":["first"]}]
    })).unwrap()
}
fn fast_request(model: &str) -> CompletionRequest {
    serde_json::from_value(json!({"model":model,"messages":[],"service_tier":"fast"})).unwrap()
}

#[derive(Default)]
struct Http(Mutex<Vec<HttpRequest>>);
#[async_trait]
impl Transport for Http {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let body = if request.url.ends_with("/files") {
            serde_json::to_vec(
                &json!({"id":"file-review","filename":"paper.pdf","purpose":"user_data","bytes":4}),
            )
            .unwrap()
        } else {
            let input: Value = serde_json::from_slice(&request.body).unwrap();
            if input["stream"] == true {
                b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n".to_vec()
            } else if request.url.ends_with("/responses") {
                serde_json::to_vec(&json!({"id":"resp-review","model":"wire","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":0}})).unwrap()
            } else {
                serde_json::to_vec(&json!({"model":"wire","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}})).unwrap()
            }
        };
        self.0.lock().unwrap().push(request);
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: body.into(),
        }
        .into())
    }
}

#[test]
fn direct_codec_context_keeps_connection_restrictions() {
    let mut p = profile(ProtocolFamily::OpenAiChat);
    p.models[0].info.features.fast = CapabilitySupport::Supported;
    p.info.features.fast = CapabilitySupport::Unsupported;
    let req = fast_request("wire");
    for context in [
        CodecContext::new(&p, "wire", RequestMode::Complete),
        CodecContext::for_model(&p, &p.models[0], RequestMode::Stream),
    ] {
        assert_eq!(
            context.profile().info.features.fast,
            CapabilitySupport::Unsupported
        );
        assert!(matches!(
            OpenAiChatCodec.encode_request(EncodeRequest::new(&req), &context),
            Err(LlmError::UnsupportedCapability { .. })
        ));
        assert!(matches!(
            OpenAiChatCodec.encoded_body_len(EncodeRequest::new(&req), &context),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }
}

#[tokio::test]
async fn complete_and_stream_keep_the_exact_aliased_row() {
    for reverse in [false, true] {
        let mut p = profile(ProtocolFamily::OpenAiChat);
        p.models[0].info.features.fast = CapabilitySupport::Unsupported;
        let mut second = p.models[0].clone();
        second.aliases = vec!["second".into()];
        second.info.features.fast = CapabilitySupport::Supported;
        p.models.push(second);
        if reverse {
            p.models.reverse();
        }
        let http = Arc::new(Http::default());
        let client = LlmClientBuilder::with_transport(http.clone(), &[p])
            .with_region(Region::International)
            .build()
            .unwrap();
        let opts = RequestOptions::default();
        assert!(matches!(
            client.complete(&fast_request("first"), &opts).await,
            Err(LlmError::UnsupportedCapability { .. })
        ));
        assert!(matches!(
            client.stream(&fast_request("first"), &opts).await,
            Err(LlmError::UnsupportedCapability { .. })
        ));
        assert!(http.0.lock().unwrap().is_empty());
        client
            .complete(&fast_request("second"), &opts)
            .await
            .unwrap();
        let mut stream = client.stream(&fast_request("second"), &opts).await.unwrap();
        let mut ended = false;
        while let Some(event) = stream.next().await {
            ended |= matches!(event.unwrap(), StreamEvent::End { .. });
        }
        assert!(ended);
        let requests = http.0.lock().unwrap();
        assert_eq!(requests.len(), 2);
        for request in requests.iter() {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            assert_eq!(body["model"], "wire");
            assert_eq!(body["service_tier"], "priority");
        }
    }
}

struct Attachment;
#[async_trait]
impl AttachmentResolver for Attachment {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        Ok(Bytes::from_static(b"%PDF"))
    }
}
#[derive(Default)]
struct Auth(Mutex<Vec<usize>>);
#[async_trait]
impl Authenticator for Auth {
    async fn apply(
        &self,
        _: &mut HttpRequest,
        p: &ProviderProfile,
        _: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        self.0.lock().unwrap().push(p.models.len());
        Ok(())
    }
}

#[tokio::test]
async fn attachment_planning_uses_the_selected_row_and_auth_keeps_the_connection() {
    let mut p = profile(ProtocolFamily::OpenAiResponses);
    p.auth = AuthStrategy::ApiKey;
    let mut second = p.models[0].clone();
    second.aliases = vec!["second".into()];
    second.capability_support.get_or_insert_default().documents = CapabilitySupport::Supported;
    second.metadata.input_modalities = vec!["text".into(), "file".into()];
    p.models.push(second);
    let http = Arc::new(Http::default());
    let auth = Arc::new(Auth::default());
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[p]);
    builder.with_attachment_resolver(Arc::new(Attachment));
    builder.register_authenticator(AuthStrategy::ApiKey, auth.clone());
    let client = builder.with_region(Region::International).build().unwrap();
    let req: CompletionRequest = serde_json::from_value(json!({"model":"second","messages":[{"role":"user","content":[{
        "type":"document","source":{"type":"attachment","attachment":{"attachment_id":"pdf","revision":"1","filename":"paper.pdf","media_type":"application/pdf","size_bytes":4}}
    }]}]})).unwrap();
    client
        .complete(
            &req,
            &RequestOptions {
                file_account_scope: Some("account".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let requests = http.0.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].url.ends_with("/files"));
    let body: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(body["input"][0]["content"][0]["file_id"], "file-review");
    assert_eq!(*auth.0.lock().unwrap(), vec![2, 2]);
}

#[test]
fn gemini_preserves_explicit_thought_output_settings() {
    for typed in [false, true] {
        for include in [None, Some(false), Some(true)] {
            let mut p = profile(ProtocolFamily::GeminiGenerateContent);
            let mut req: CompletionRequest =
                serde_json::from_value(json!({"model":"wire","messages":[]})).unwrap();
            if typed {
                req.thinking = Some(ThinkingConfig {
                    budget: Some(ThinkingBudget::Tokens(2048)),
                    ..Default::default()
                });
            } else {
                p.extra["body"]["generationConfig"]["thinkingConfig"]["thinkingBudget"] =
                    json!(2048);
            }
            if let Some(include) = include {
                p.extra["body"]["generationConfig"]["thinkingConfig"]["includeThoughts"] =
                    json!(include);
            }
            let context = CodecContext::new(&p, "wire", RequestMode::Complete);
            let http = GeminiCodec
                .encode_request(EncodeRequest::new(&req), &context)
                .unwrap();
            let body: Value = serde_json::from_slice(&http.body).unwrap();
            assert_eq!(
                body["generationConfig"]["thinkingConfig"]["includeThoughts"],
                include.unwrap_or(true)
            );
            assert_eq!(
                GeminiCodec
                    .encoded_body_len(EncodeRequest::new(&req), &context)
                    .unwrap(),
                http.body.len()
            );
        }
    }
}

#[cfg(feature = "tokenizer-openai")]
#[test]
fn encrypted_reasoning_is_never_tokenized_as_plaintext() {
    for protocol in [
        ProtocolFamily::OpenAiChat,
        ProtocolFamily::OpenAiResponses,
        ProtocolFamily::AzureOpenAi,
        ProtocolFamily::GeminiGenerateContent,
        ProtocolFamily::VertexGemini,
        ProtocolFamily::AnthropicMessages,
        ProtocolFamily::VertexClaude,
        ProtocolFamily::FoundryClaude,
        ProtocolFamily::BedrockClaude,
    ] {
        let mut p = profile(protocol);
        p.models[0].request_model = "gpt-4o".into();
        let client =
            LlmClientBuilder::with_transport(Arc::new(Http::default()), std::slice::from_ref(&p))
                .with_region(Region::International)
                .build()
                .unwrap();
        let plain: CompletionRequest = serde_json::from_value(json!({"model":"gpt-4o","messages":[{"role":"assistant","content":[{"type":"text","text":"hello"}]}]})).unwrap();
        let mut encrypted = plain.clone();
        encrypted.messages[0].content.insert(
            0,
            ContentBlock::RedactedThinking {
                data: "ciphertext abc123XYZ".repeat(10_000),
            },
        );
        let base = client.estimate_local_tokens(&plain).unwrap();
        let estimate = client.estimate_local_tokens(&encrypted).unwrap();
        assert_eq!(estimate.input_tokens, base.input_tokens);
        let replays = matches!(
            protocol,
            ProtocolFamily::AnthropicMessages
                | ProtocolFamily::VertexClaude
                | ProtocolFamily::FoundryClaude
                | ProtocolFamily::BedrockClaude
        );
        assert_eq!(estimate.is_partial, replays);
        assert_eq!(
            estimate.uncounted_components,
            if replays {
                vec![LocalTokenEstimateOmission::ProviderOpaqueContent]
            } else {
                vec![]
            }
        );
        if protocol == ProtocolFamily::OpenAiChat {
            let context = CodecContext::new(&p, "gpt-4o", RequestMode::Complete);
            assert_eq!(
                OpenAiChatCodec
                    .encode_request(EncodeRequest::new(&plain), &context)
                    .unwrap()
                    .body,
                OpenAiChatCodec
                    .encode_request(EncodeRequest::new(&encrypted), &context)
                    .unwrap()
                    .body
            );
        }
    }
}

#[test]
fn failover_requires_one_row_with_the_same_wire_model_and_billing_mode() {
    let mut primary = profile(ProtocolFamily::OpenAiChat);
    primary.profile_name = "primary".into();
    primary.connection.group = Some("group".into());
    primary.connection.connection_id = Some("primary".into());
    let mut backup = primary.clone();
    backup.profile_name = "backup".into();
    backup.connection.connection_id = Some("backup".into());
    let mut alternate = backup.models[0].clone();
    alternate.aliases = vec!["alternate".into()];
    backup.models.push(alternate);
    let make = |backup: ProviderProfile| {
        LlmClientBuilder::with_transport(Arc::new(Http::default()), &[primary.clone(), backup])
            .with_region(Region::International)
            .build()
            .unwrap()
    };
    assert!(make(backup.clone())
        .resolve_in("first", Some("primary"))
        .unwrap()
        .connection_chain
        .is_empty());
    backup.models[0].billing_mode = Some(BillingMode::Subscription);
    let route = make(backup).resolve_in("first", Some("primary")).unwrap();
    assert_eq!(route.connection_chain.len(), 1);
    assert_eq!(route.connection_chain[0].profile_name, "backup");
}
