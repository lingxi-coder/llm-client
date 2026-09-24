use lingxi_llm_client::protocol::PricingContext;
#[path = "support/wire_api.rs"]
mod wire_api;
use async_trait::async_trait;
use futures::{stream, StreamExt};
use lingxi_llm_client::codecs::openai::chat::OpenAiChatCodec;
use lingxi_llm_client::protocol::{AuthStrategy, CompletionRequest, LlmError, Region, StreamEvent};
use lingxi_llm_client::{
    builtin_providers, HttpRequest, HttpResponse, LlmClientBuilder, RequestOptions, StreamResponse,
    Transport, WireCodec,
};
use serde_json::{json, Value};
use std::sync::Arc;

fn usage() -> Value {
    json!({"prompt_tokens":32,"completion_tokens":9,"total_tokens":135,
        "prompt_tokens_details":{"cached_tokens":6},"completion_tokens_details":{"reasoning_tokens":94}})
}
struct Http {
    status: u16,
}
#[async_trait]
impl Transport for Http {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        if serde_json::from_slice::<serde_json::Value>(&request.body)
            .ok()
            .and_then(|body| body.get("stream").and_then(serde_json::Value::as_bool))
            == Some(true)
        {
            self.stream_response(request).await
        } else {
            self.response(request).await.map(Into::into)
        }
    }
}
impl Http {
    async fn response(&self, _: HttpRequest) -> Result<HttpResponse, LlmError> {
        Ok(HttpResponse { status:self.status, headers:vec![], body:json!({"model":"grok-4.20","choices":[{"message":{"content":"answer"},"finish_reason":"stop"}],"usage":usage()}).to_string().into() })
    }
    async fn stream_response(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        Ok(StreamResponse {status:self.status,headers:vec![],body:stream::iter(vec![
            Ok(format!("data: {}\n\n",json!({"model":"grok-4.20","choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}],"usage":usage()})).into()),
            Ok("data: [DONE]\n\n".into())
        ]).boxed()})
    }
}

#[tokio::test]
async fn builtin_grok_normalizes_buffered_and_streamed_reasoning_for_pricing() {
    let mut profile = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "grok")
        .unwrap();
    assert_eq!(profile.extra["reasoning_tokens_separate"], true);
    profile.auth = AuthStrategy::None;
    let client = LlmClientBuilder::with_transport(
        Arc::new(Http { status: 200 }),
        std::slice::from_ref(&profile),
    )
    .with_region(Region::International)
    .build()
    .unwrap();
    let req: CompletionRequest =
        serde_json::from_value(json!({"model":"grok-4.20","messages":[]})).unwrap();
    let complete = client
        .complete(&req, &RequestOptions::default())
        .await
        .unwrap();
    assert_eq!(complete.usage.usage.as_ref().unwrap().input_tokens, 26);
    assert_eq!(complete.usage.usage.as_ref().unwrap().cache_read_tokens, 6);
    assert_eq!(complete.usage.usage.as_ref().unwrap().output_tokens, 103);
    assert_eq!(complete.usage.usage.as_ref().unwrap().reasoning_tokens, 94);
    let route = client.resolve(&req.model).unwrap();
    let cost = client
        .estimate_cost(
            &route,
            complete.usage.usage.as_ref().unwrap(),
            &PricingContext::default(),
        )
        .unwrap();
    assert!(cost.total_cost > 0.0);
    let mut streamed = client
        .stream(&req, &RequestOptions::default())
        .await
        .unwrap();
    let mut final_usage = None;
    while let Some(event) = streamed.next().await {
        if let StreamEvent::End { usage, .. } = event.unwrap() {
            final_usage = Some(usage);
        }
    }
    assert_eq!(final_usage, Some(complete.usage.clone()));
    assert_eq!(
        crate::wire_api::observed_usage(&streamed),
        complete.usage.usage
    );
    assert!(crate::wire_api::usage_is_complete(&streamed));

    // Other Chat providers keep the existing output-includes-reasoning contract.
    profile.extra = Value::Null;
    let response = HttpResponse {
        status: 200,
        headers: vec![],
        body: json!({"usage":usage()}).to_string().into(),
    };
    let unchanged = wire_api::response_usage(
        &OpenAiChatCodec,
        &response,
        &lingxi_llm_client::CodecContext::new(
            &profile,
            "",
            lingxi_llm_client::RequestMode::Complete,
        ),
    )
    .unwrap();
    assert_eq!(unchanged.output_tokens, 9);
    let mut decoder = OpenAiChatCodec.stream_decoder(&lingxi_llm_client::CodecContext::new(
        &profile,
        "",
        lingxi_llm_client::RequestMode::Stream,
    ));
    wire_api::decode_frame(
        &mut *decoder,
        json!({"usage":usage(),"choices":[]}).to_string().as_bytes(),
    )
    .unwrap();
    assert!(!crate::wire_api::usage_is_complete(&decoder));
}

struct ProfileCodec;
impl WireCodec for ProfileCodec {
    fn family(&self) -> lingxi_llm_client::protocol::ProtocolFamily {
        OpenAiChatCodec.family()
    }

    fn encode_request(
        &self,
        input: lingxi_llm_client::EncodeRequest<'_>,
        context: &lingxi_llm_client::CodecContext,
    ) -> Result<HttpRequest, LlmError> {
        let req = input.request();
        let profile = context.profile();
        let route = &wire_api::route(context);
        let opts = &wire_api::options(context);

        OpenAiChatCodec.encode_request(
            lingxi_llm_client::EncodeRequest::new(req),
            &wire_api::context(profile, &(route).request_model, opts),
        )
    }
    fn decode_response(
        &self,
        response: &HttpResponse,
        context: &lingxi_llm_client::CodecContext,
    ) -> Result<lingxi_llm_client::protocol::CompletionResponse, LlmError> {
        assert_eq!(response.status, 429);
        assert_eq!(context.profile().profile_name, "grok");
        Err(LlmError::QuotaExceeded {
            message: "profile hook".into(),
        })
    }
    fn stream_decoder(
        &self,
        _context: &lingxi_llm_client::CodecContext,
    ) -> Box<dyn lingxi_llm_client::StreamDecoder> {
        OpenAiChatCodec.stream_decoder(&wire_api::decode_context())
    }
}

#[tokio::test]
async fn profile_error_hook_is_used_for_both_request_modes() {
    let mut profile = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "grok")
        .unwrap();
    profile.auth = AuthStrategy::None;
    let mut builder = LlmClientBuilder::with_transport(Arc::new(Http { status: 429 }), &[profile]);
    builder.register_codec(Arc::new(ProfileCodec));
    let client = builder.with_region(Region::International).build().unwrap();
    let req: CompletionRequest =
        serde_json::from_value(json!({"model":"grok-4.20","messages":[]})).unwrap();
    assert!(matches!(
        client.complete(&req, &RequestOptions::default()).await,
        Err(LlmError::QuotaExceeded { .. })
    ));
    assert!(matches!(
        client.stream(&req, &RequestOptions::default()).await,
        Err(LlmError::QuotaExceeded { .. })
    ));
}
