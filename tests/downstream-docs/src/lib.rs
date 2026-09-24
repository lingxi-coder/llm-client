#![doc = concat!(
    include_str!("../../../README.md"),
    "\n\n",
    include_str!("../../../docs/api.md"),
    "\n\n",
    include_str!("../../../docs/web-search.md"),
    "\n\n",
    include_str!("../../../README.en.md"),
    "\n\n",
    include_str!("../../../docs/api.en.md"),
    "\n\n",
    include_str!("../../../docs/web-search.en.md"),
    "\n\n",
    include_str!("../../../docs/file-attachments.zh.md"),
    "\n\n",
    include_str!("../../../docs/file-attachments.md"),
    "\n\n",
    include_str!("../../../docs/architecture-migration.md"),
    "\n\n",
    include_str!("../../../docs/architecture-migration.en.md"),
    "\n\n",
    include_str!("../../../docs/inference.md"),
    "\n\n",
    include_str!("../../../docs/inference.en.md"),
)]

// External-crate compilation catches accidental private types in extension contracts.
#[cfg(test)]
mod extensions {
    use async_trait::async_trait;
    use lingxi_llm_client::{protocol::*, *};
    struct Http;
    #[async_trait]
    impl Transport for Http {
        async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
            Ok(HttpResponse {
                status: 200,
                headers: vec![],
                body: Vec::new().into(),
            }
            .into())
        }
    }
    struct Codec;
    impl WireCodec for Codec {
        fn family(&self) -> ProtocolFamily {
            ProtocolFamily::OpenAiChat
        }
        fn encode_request(
            &self,
            request: EncodeRequest<'_>,
            context: &CodecContext,
        ) -> Result<HttpRequest, LlmError> {
            let _ = request.blocks().count();
            let _ = request.media();
            OpenAiChatCodec.encode_request(request, context)
        }
        fn decode_response(
            &self,
            response: &HttpResponse,
            context: &CodecContext,
        ) -> Result<CompletionResponse, LlmError> {
            OpenAiChatCodec.decode_response(response, context)
        }
        fn stream_decoder(&self, context: &CodecContext) -> Box<dyn StreamDecoder> {
            OpenAiChatCodec.stream_decoder(context)
        }
    }
    struct Decoder;
    impl StreamDecoder for Decoder {
        fn push_bytes(&mut self, _: &[u8]) -> Vec<Result<StreamEvent, LlmError>> {
            Vec::new()
        }
        fn finish(&mut self) -> Vec<Result<StreamEvent, LlmError>> {
            Vec::new()
        }
        fn usage_report(&self) -> UsageReport {
            UsageReport::default()
        }
    }
    struct Account;
    #[async_trait]
    impl AccountUsageSource for Account {
        async fn fetch(
            &self,
            _: &AccountFetchContext<'_>,
            report: &mut AccountReport,
        ) -> Result<(), AccountFailure> {
            report.balance = Some(AccountMetric::NotReported);
            Ok(())
        }
    }
    #[test]
    fn public_extension_contracts_are_implementable() {
        let _transport: Box<dyn Transport> = Box::new(Http);
        let _codec: Box<dyn WireCodec> = Box::new(Codec);
        let _decoder: Box<dyn StreamDecoder> = Box::new(Decoder);
        let _source: Box<dyn AccountUsageSource> = Box::new(Account);
    }
}
