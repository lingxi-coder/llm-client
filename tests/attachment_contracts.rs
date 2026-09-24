use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::{protocol::*, *};
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
struct Resolver;
#[async_trait]
impl AttachmentResolver for Resolver {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        Ok(Bytes::from_static(b"data"))
    }
}
struct NoIo(AtomicUsize);
#[async_trait]
impl Transport for NoIo {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        panic!("plan must fail before sending an upload")
    }
}
fn attachment(id: &str, mime: &str) -> AttachmentRef {
    AttachmentRef {
        attachment_id: id.into(),
        revision: "1".into(),
        filename: "名\"\\.pdf".into(),
        media_type: mime.into(),
        size_bytes: 4,
    }
}

struct GeminiUpload(AtomicUsize);
#[async_trait]
impl Transport for GeminiUpload {
    async fn send(&self, req: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        assert!(req
            .timeout
            .is_some_and(|timeout| !timeout.is_zero() && timeout <= Duration::from_secs(1)));
        let (headers, body) = if req.url.contains("/upload/v1beta/files") {
            (
                vec![(
                    "x-goog-upload-url".into(),
                    "https://generativelanguage.googleapis.com/session".into(),
                )],
                Bytes::new(),
            )
        } else if req.url.ends_with("/session") {
            (vec![], Bytes::from(json!({"file":{"name":"files/test","uri":"https://generativelanguage.googleapis.com/v1beta/files/test","state":"ACTIVE","mimeType":"application/pdf"}}).to_string()))
        } else {
            let response = json!({"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}});
            let body = if req.url.contains(":streamGenerateContent") {
                format!("data: {response}\n\n")
            } else {
                response.to_string()
            };
            (vec![], Bytes::from(body))
        };
        Ok(HttpResponse {
            status: 200,
            headers,
            body,
        }
        .into())
    }
}

#[tokio::test]
async fn gemini_attachments_keep_usable_short_request_budgets() {
    let p: ProviderProfile = serde_json::from_value(json!({"profile_name":"p","provider_id":"google","base_url":"https://generativelanguage.googleapis.com/v1beta","protocol":"gemini_generate_content","auth":"none","models":[{"display_model":"m","request_model":"m","billing_model":"m","metadata":{"inputModalities":["text","file"]},"capability_support":{"documents":"supported"}}]})).unwrap();
    let req: CompletionRequest = serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":[{"type":"document","source":{"type":"attachment","attachment":attachment("pdf","application/pdf")}}]}]})).unwrap();
    for timeout in [Duration::from_millis(500), Duration::from_secs(1)] {
        for streaming in [false, true] {
            let http = Arc::new(GeminiUpload(AtomicUsize::new(0)));
            let mut builder =
                LlmClientBuilder::with_transport(http.clone(), std::slice::from_ref(&p));
            builder.with_attachment_resolver(Arc::new(Resolver));
            let c = builder.with_region(Region::International).build().unwrap();
            let opts = RequestOptions {
                total_timeout: Some(timeout),
                ..Default::default()
            };
            if streaming {
                let mut stream = c.stream(&req, &opts).await.unwrap();
                let mut ended = false;
                while let Some(event) = stream.next().await {
                    ended |= matches!(event.unwrap(), StreamEvent::End { .. });
                }
                assert!(ended);
            } else {
                assert_eq!(
                    c.complete(&req, &opts).await.unwrap().stop_reason,
                    StopReason::EndTurn
                );
            }
            assert_eq!(http.0.load(Ordering::Relaxed), 3);
        }
    }
    let http = Arc::new(NoIo(AtomicUsize::new(0)));
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[p]);
    builder.with_attachment_resolver(Arc::new(Resolver));
    let c = builder.with_region(Region::International).build().unwrap();
    let opts = RequestOptions {
        total_timeout: Some(Duration::ZERO),
        ..Default::default()
    };
    assert!(matches!(
        c.complete(&req, &opts).await,
        Err(LlmError::TransportTimeout { .. })
    ));
    assert_eq!(http.0.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn all_attachment_decisions_are_validated_before_the_first_upload() {
    let profile:ProviderProfile=serde_json::from_value(json!({"profile_name":"p","provider_id":"openai","base_url":"https://api.openai.com/v1","protocol":"open_ai_responses","auth":"none","models":[{"display_model":"m","request_model":"m","billing_model":"m","metadata":{"inputModalities":["text","image","file"]},"capability_support":{"vision":"unsupported","documents":"supported"}}]})).unwrap();
    let request:CompletionRequest=serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":[{"type":"document","source":{"type":"attachment","attachment":attachment("doc","application/pdf")}},{"type":"image","source":{"type":"attachment","attachment":attachment("image","image/png")}}]}]})).unwrap();
    let original = request.clone();
    let http = Arc::new(NoIo(AtomicUsize::new(0)));
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[profile]);
    builder.with_attachment_resolver(Arc::new(Resolver));
    let client = builder.with_region(Region::International).build().unwrap();
    assert!(matches!(
        client.complete(&request, &RequestOptions::default()).await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert_eq!(http.0.load(Ordering::Relaxed), 0);
    assert_eq!(request, original);
}
#[test]
fn anthropic_counting_and_serialization_share_escaped_titles_and_lazy_bytes() {
    let profile:ProviderProfile=serde_json::from_value(json!({"profile_name":"p","provider_id":"anthropic","base_url":"https://api.anthropic.com","protocol":"anthropic_messages","auth":"none","models":[]})).unwrap();
    let attachment = attachment("doc", "application/pdf");
    let request:CompletionRequest=serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":[{"type":"document","source":{"type":"attachment","attachment":attachment}}]}]})).unwrap();
    let media = [PreparedMedia {
        attachment: &attachment,
        bytes: b"data",
    }];
    let view = EncodeRequest::new(&request).with_media(&media);
    let context = CodecContext::new(&profile, "m", RequestMode::Complete);
    let encoded = AnthropicMessagesCodec
        .encode_request(view, &context)
        .unwrap();
    assert_eq!(
        AnthropicMessagesCodec
            .encoded_body_len(view, &context)
            .unwrap(),
        encoded.body.len()
    );
    let json: serde_json::Value = serde_json::from_slice(&encoded.body).unwrap();
    assert_eq!(
        json["messages"][0]["content"][0]["title"],
        attachment.filename
    );
    assert_eq!(
        json["messages"][0]["content"][0]["source"]["data"],
        "ZGF0YQ=="
    );
}
#[tokio::test]
async fn an_inline_wire_rejection_prevents_earlier_file_uploads() {
    let profile:ProviderProfile=serde_json::from_value(json!({"profile_name":"p","provider_id":"openai","base_url":"https://api.openai.com/v1","protocol":"open_ai_chat","auth":"none","models":[{"display_model":"m","request_model":"m","billing_model":"m","metadata":{"inputModalities":["text","file"]},"capability_support":{"documents":"supported"}}]})).unwrap();
    let request:CompletionRequest=serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":[{"type":"document","source":{"type":"attachment","attachment":attachment("pdf","application/pdf")}},{"type":"document","source":{"type":"attachment","attachment":attachment("txt","text/plain")}}]}]})).unwrap();
    let http = Arc::new(NoIo(AtomicUsize::new(0)));
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[profile]);
    builder.with_attachment_resolver(Arc::new(Resolver));
    let client = builder.with_region(Region::International).build().unwrap();
    assert!(matches!(
        client.complete(&request, &RequestOptions::default()).await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert_eq!(http.0.load(Ordering::Relaxed), 0);
}
#[test]
fn custom_content_bindings_are_validated_as_the_effective_request() {
    let p:ProviderProfile=serde_json::from_value(json!({"profile_name":"p","provider_id":"openai","base_url":"https://api.openai.com/v1","protocol":"open_ai_responses","auth":"none","models":[{"display_model":"m","request_model":"m","billing_model":"m","metadata":{"inputModalities":["text","image","file"]}}]})).unwrap();
    let request:CompletionRequest=serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":[{"type":"document","source":{"type":"attachment","attachment":attachment("d","application/pdf")}}]}]})).unwrap();
    let file = ProviderFileSource {
        protocol: p.protocol,
        provider_id: p.provider_id.clone(),
        profile_name: p.profile_name.clone(),
        endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
            &p.base_url,
        ),
        account_scope: Some("a".into()),
        file_id: "file-example".into(),
        uri: None,
        media_type: Some("image/png".into()),
        purpose: None,
    };
    let bindings = [lingxi_llm_client::codecs::ContentBinding {
        original: &request.messages[0].content[0],
        replacement: ContentBlock::Document {
            source: DocumentSource::ProviderFile { file },
            title: None,
        },
    }];
    let view = EncodeRequest::new(&request).with_bindings(&bindings);
    let context = CodecContext::new(&p, "m", RequestMode::Complete).with_file_scope(Some("a"));
    assert!(matches!(
        OpenAiResponsesCodec.encode_request(view, &context),
        Err(LlmError::InvalidRequest { .. })
    ));
}
