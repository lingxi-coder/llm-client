#![allow(dead_code)]
use lingxi_llm_client::{protocol::*, *};
use serde_json::json;

#[derive(Default)]
pub struct EncodingOptions {
    pub stream: bool,
    pub file_account_scope: Option<String>,
}
pub trait WireOptions {
    fn stream(&self) -> bool;
    fn scope(&self) -> Option<&str>;
}
impl WireOptions for RequestOptions {
    fn stream(&self) -> bool {
        false
    }
    fn scope(&self) -> Option<&str> {
        self.file_account_scope.as_deref()
    }
}
impl WireOptions for EncodingOptions {
    fn stream(&self) -> bool {
        self.stream
    }
    fn scope(&self) -> Option<&str> {
        self.file_account_scope.as_deref()
    }
}
pub fn context(profile: &ProviderProfile, model: &str, options: &impl WireOptions) -> CodecContext {
    CodecContext::new(
        profile,
        model,
        if options.stream() {
            RequestMode::Stream
        } else {
            RequestMode::Complete
        },
    )
    .with_file_scope(options.scope())
}
pub fn decode_context() -> CodecContext {
    let profile:ProviderProfile=serde_json::from_value(json!({"provider_id":"test","profile_name":"test","base_url":"https://test.invalid","protocol":"open_ai_chat","auth":"none"})).unwrap();
    CodecContext::new(&profile, "", RequestMode::Complete)
}
pub fn route(context: &CodecContext) -> ResolvedRoute {
    ResolvedRoute {
        provider_id: context.profile().provider_id.clone(),
        profile_name: context.profile().profile_name.clone(),
        request_model: context.request_model().into(),
        display_model: context.request_model().into(),
        pricing_model: PricingModelRef {
            pricing_provider_id: context.profile().provider_id.clone(),
            billing_model: context.request_model().into(),
            request_model: context.request_model().into(),
            display_model: context.request_model().into(),
        },
        capability_support: Default::default(),
        connection_chain: vec![],
        failover: Default::default(),
    }
}
pub fn options(context: &CodecContext) -> EncodingOptions {
    EncodingOptions {
        stream: context.mode() == RequestMode::Stream,
        file_account_scope: context.file_scope().map(str::to_owned),
    }
}
pub fn decode_frame(
    decoder: &mut dyn StreamDecoder,
    data: &[u8],
) -> Result<Vec<StreamEvent>, LlmError> {
    let bytes = if data.first() == Some(&0) {
        data.to_vec()
    } else {
        let mut bytes = b"data: ".to_vec();
        bytes.extend_from_slice(data);
        bytes.extend_from_slice(b"\n\n");
        bytes
    };
    decoder.push_bytes(&bytes).into_iter().collect()
}
pub fn finish(decoder: &mut dyn StreamDecoder) -> Result<Vec<StreamEvent>, LlmError> {
    decoder.finish().into_iter().collect()
}
pub trait UsageSource {
    fn report(&self) -> UsageReport;
}
impl UsageSource for Box<dyn StreamDecoder> {
    fn report(&self) -> UsageReport {
        self.usage_report()
    }
}
impl UsageSource for ModelStream {
    fn report(&self) -> UsageReport {
        self.usage_report()
    }
}
pub fn observed_usage(decoder: &impl UsageSource) -> Option<Usage> {
    decoder.report().usage
}
pub fn usage_is_complete(decoder: &impl UsageSource) -> bool {
    decoder.report().state == UsageState::Complete
}
pub fn response_usage(
    codec: &dyn WireCodec,
    response: &HttpResponse,
    context: &CodecContext,
) -> Option<Usage> {
    let mut value: serde_json::Value = serde_json::from_slice(&response.body).ok()?;
    let object = value.as_object_mut()?;
    match codec.family() {
        ProtocolFamily::OpenAiChat | ProtocolFamily::AzureOpenAi => {
            object
                .entry("choices")
                .or_insert(json!([{"message":{},"finish_reason":"stop"}]));
        }
        ProtocolFamily::OpenAiResponses => {
            object.entry("output").or_insert(json!([]));
            object.entry("status").or_insert(json!("completed"));
        }
        ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini => {
            object
                .entry("candidates")
                .or_insert(json!([{"content":{"parts":[]},"finishReason":"STOP"}]));
        }
        _ => {
            object.entry("content").or_insert(json!([]));
        }
    }
    let response = HttpResponse {
        status: response.status,
        headers: response.headers.clone(),
        body: value.to_string().into(),
    };
    codec.decode_response(&response, context).ok()?.usage.usage
}
