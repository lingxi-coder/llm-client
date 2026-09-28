use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::codecs::openai::chat::OpenAiChatCodec;
use lingxi_llm_client::protocol::{
    ChatRequest, ChatResponse, LlmError, ProtocolFamily, ProviderProfile, Region,
};
use lingxi_llm_client::{
    CodecContext, EncodeRequest, HttpRequest, HttpResponse, LlmClientBuilder, RequestMode,
    RequestOptions, StreamDecoder, StreamResponse, Transport, WireCodec,
};
use serde_json::json;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Observation {
    validated: Mutex<Vec<String>>,
    encoded: Mutex<Vec<String>>,
    decoded: Mutex<Vec<String>>,
    streamed: Mutex<Vec<String>>,
    sent: Mutex<Vec<HttpRequest>>,
}
struct InjectedCodec {
    observation: Arc<Observation>,
    reject: bool,
}
impl WireCodec for InjectedCodec {
    fn family(&self) -> ProtocolFamily {
        ProtocolFamily::OpenAiChat
    }
    fn validate_request(&self, _: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
        self.observation
            .validated
            .lock()
            .unwrap()
            .push(context.profile().provider_id.as_str().to_owned());
        if self.reject {
            Err(LlmError::UnsupportedCapability {
                message: "injected provider rejection".into(),
            })
        } else {
            Ok(())
        }
    }
    fn encode_request(
        &self,
        input: EncodeRequest<'_>,
        context: &CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        self.observation
            .encoded
            .lock()
            .unwrap()
            .push(context.profile().provider_id.as_str().to_owned());
        let mut request = OpenAiChatCodec.encode_request(input, context)?;
        request
            .headers
            .push(("x-injected-codec".into(), "yes".into()));
        Ok(request)
    }
    fn decode_response(
        &self,
        response: &HttpResponse,
        context: &CodecContext,
    ) -> Result<ChatResponse, LlmError> {
        self.observation
            .decoded
            .lock()
            .unwrap()
            .push(context.profile().provider_id.as_str().to_owned());
        OpenAiChatCodec.decode_response(response, context)
    }
    fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
        self.observation
            .streamed
            .lock()
            .unwrap()
            .push(context.profile().provider_id.as_str().to_owned());
        OpenAiChatCodec.stream_decoder(context)
    }
}
struct InjectedTransport(Arc<Observation>);
#[async_trait]
impl Transport for InjectedTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name == "x-injected-codec" && value == "yes"));
        let streaming =
            serde_json::from_slice::<serde_json::Value>(&request.body).unwrap()["stream"] == true;
        self.0.sent.lock().unwrap().push(request);
        if streaming {
            let data = format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({"model":"wire-m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}], "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})
            );
            return Ok(StreamResponse {
                status: 200,
                headers: vec![],
                body: futures::stream::iter(vec![Ok(data.into())]).boxed(),
            });
        }
        Ok(HttpResponse { status: 200, headers: vec![], body: json!({"model":"wire-m", "choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}], "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}).to_string().into() }.into())
    }
}
fn profile(provider: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "profile_name":"selected", "provider_id":provider, "base_url":"https://api.example.test/v1",
        "auth":"none", "protocol":"open_ai_chat",
        "models":[{"request_model":"wire-m","display_model":"m","billing_model":"wire-m"}]
    }))
    .unwrap()
}
fn request() -> ChatRequest {
    serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]})).unwrap()
}
const PROVIDERS: &[&str] = &[
    "openai",
    "anthropic",
    "google",
    "qwen",
    "minimax",
    "zhipu",
    "kimi",
    "xai",
    "openrouter",
    "deepseek",
    "github-copilot",
    "custom-compatible",
];

#[tokio::test]
async fn builtin_and_compatible_dispatch_share_injected_codec_and_transport() {
    for provider in PROVIDERS {
        let observed = Arc::new(Observation::default());
        let mut builder = LlmClientBuilder::with_transport(
            Arc::new(InjectedTransport(observed.clone())),
            &[profile(provider)],
        )
        .with_region(Region::International);
        builder.register_codec(Arc::new(InjectedCodec {
            observation: observed.clone(),
            reject: false,
        }));
        let client = builder.build().unwrap();
        let req = request();
        client
            .chat()
            .complete_in("selected", &req, &RequestOptions::default())
            .await
            .unwrap();
        let collected = client
            .snapshot()
            .prepare_on(
                "selected",
                &req,
                &RequestOptions::default(),
                RequestMode::Complete,
            )
            .await
            .unwrap()
            .dispatch_once()
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        collected.decode().unwrap();
        collected.finish().await;
        assert_eq!(
            *observed.decoded.lock().unwrap(),
            vec![provider.to_string(), provider.to_string()]
        );
        let mut stream = client
            .chat()
            .stream_in("selected", &req, &RequestOptions::default())
            .await
            .unwrap();
        let mut ended = false;
        while let Some(event) = stream.next().await {
            ended |= matches!(
                event.unwrap(),
                lingxi_llm_client::protocol::StreamEvent::End { .. }
            );
        }
        assert!(ended);
        assert_eq!(
            *observed.streamed.lock().unwrap(),
            vec![provider.to_string()]
        );
        assert_eq!(observed.sent.lock().unwrap().len(), 3);
        assert!(observed
            .encoded
            .lock()
            .unwrap()
            .iter()
            .all(|identity| identity == provider));
    }
}

#[tokio::test]
async fn provider_rejection_never_falls_back_to_a_default_codec() {
    for provider in PROVIDERS {
        let observed = Arc::new(Observation::default());
        let mut builder = LlmClientBuilder::with_transport(
            Arc::new(InjectedTransport(observed.clone())),
            &[profile(provider)],
        )
        .with_region(Region::International);
        builder.register_codec(Arc::new(InjectedCodec {
            observation: observed.clone(),
            reject: true,
        }));
        let client = builder.build().unwrap();
        let error = client
            .chat()
            .complete_in("selected", &request(), &RequestOptions::default())
            .await
            .unwrap_err();
        assert!(
            matches!(error, LlmError::UnsupportedCapability { message } if message == "injected provider rejection")
        );
        assert_eq!(observed.validated.lock().unwrap().len(), 1);
        assert!(observed.encoded.lock().unwrap().is_empty());
        assert!(observed.sent.lock().unwrap().is_empty());
    }
}
