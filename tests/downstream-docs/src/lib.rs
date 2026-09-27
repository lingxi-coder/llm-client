#![doc = concat!(
    include_str!("../../../docs/openai-responses-prompt-cache.md"),
    "\n\n",
    include_str!("../../../docs/openai-responses-prompt-cache.en.md"),
    "\n\n",
    include_str!("../../../docs/anthropic-programmatic-tools.md"),
    "\n\n",
    include_str!("../../../docs/anthropic-programmatic-tools.en.md"),
    "\n\n",
    include_str!("../../../docs/openai-tool-search.md"),
    "\n\n",
    include_str!("../../../docs/openai-tool-search.en.md"),
    "\n\n",
    include_str!("../../../docs/embedding-limits.md"),
    "\n\n",
    include_str!("../../../docs/embedding-limits.en.md"),
    "\n\n",
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
    include_str!("../../../docs/client-reuse.md"),
    "\n\n",
    include_str!("../../../docs/client-reuse.en.md"),
    "\n\n",
    include_str!("../../../docs/architecture-migration.md"),
    "\n\n",
    include_str!("../../../docs/architecture-migration.en.md"),
    "\n\n",
    include_str!("../../../docs/inference.md"),
    "\n\n",
    include_str!("../../../docs/inference.en.md"),
    "\n\n",
    include_str!("../../../docs/services.md"),
    "\n\n",
    include_str!("../../../docs/services.en.md"),
    include_str!("../../../docs/retrieval.md"),
    include_str!("../../../docs/retrieval.en.md"),
    include_str!("../../../docs/batches.md"),
    include_str!("../../../docs/batches.en.md"),
    include_str!("../../../docs/deferred.md"),
    include_str!("../../../docs/deferred.en.md"),
    include_str!("../../../docs/background.md"),
    include_str!("../../../docs/background.en.md"),
    include_str!("../../../docs/audio.md"),
    include_str!("../../../docs/audio.en.md"),
    include_str!("../../../docs/interactions.md"),
    include_str!("../../../docs/interactions.en.md"),
    include_str!("../../../docs/anthropic-tools.md"),
    include_str!("../../../docs/anthropic-tools.en.md"),
    include_str!("../../../docs/anthropic-mcp.md"),
    include_str!("../../../docs/anthropic-mcp.en.md"),
    include_str!("../../../docs/anthropic-conversation.md"),
    include_str!("../../../docs/anthropic-conversation.en.md"),
    include_str!("../../../docs/anthropic-vertex.md"),
    include_str!("../../../docs/anthropic-vertex.en.md"),
    include_str!("../../../docs/anthropic-foundry.md"),
    include_str!("../../../docs/anthropic-foundry.en.md"),
    include_str!("../../../docs/anthropic-client-toolsets.md"),
    include_str!("../../../docs/anthropic-client-toolsets.en.md"),
    include_str!("../../../docs/anthropic-web-fetch.md"),
    include_str!("../../../docs/anthropic-web-fetch.en.md"),
    include_str!("../../../docs/anthropic-code-execution.md"),
    include_str!("../../../docs/anthropic-code-execution.en.md"),
    include_str!("../../../docs/anthropic-skills.md"),
    include_str!("../../../docs/anthropic-skills.en.md"),
    include_str!("../../../docs/anthropic-native-content.md"),
    include_str!("../../../docs/anthropic-native-content.en.md"),
    include_str!("../../../docs/minimax-audio.md"),
    include_str!("../../../docs/minimax-audio.en.md"),
    include_str!("../../../docs/minimax-voices.md"),
    include_str!("../../../docs/minimax-voices.en.md"),
    include_str!("../../../docs/xai-collections.md"),
    include_str!("../../../docs/xai-collections.en.md"),
    include_str!("../../../docs/qwen-batch.md"),
    include_str!("../../../docs/qwen-batch.en.md"),
    include_str!("../../../docs/gemini-embedding.md"),
    include_str!("../../../docs/gemini-embedding.en.md"),
    include_str!("../../../docs/gemini-file-search.md"),
    include_str!("../../../docs/gemini-file-search.en.md"),
    include_str!("../../../docs/openrouter-batch.md"),
    include_str!("../../../docs/openrouter-batch.en.md"),
    include_str!("../../../docs/openrouter-audio.md"),
    include_str!("../../../docs/openrouter-audio.en.md"),
    include_str!("../../../docs/openrouter-server-tools.md"),
    include_str!("../../../docs/openrouter-server-tools.en.md"),
    include_str!("../../../docs/glm-async.md"),
    include_str!("../../../docs/glm-async.en.md"),
    include_str!("../../../docs/realtime.md"),
    include_str!("../../../docs/realtime.en.md"),
    include_str!("../../../docs/glm-knowledge.md"),
    include_str!("../../../docs/glm-knowledge.en.md"),
    include_str!("../../../docs/kimi-batch.md"),
    include_str!("../../../docs/kimi-batch.en.md"),
    include_str!("../../../docs/anthropic-batch.md"),
    include_str!("../../../docs/anthropic-batch.en.md"),
    include_str!("../../../docs/qwen-asr.md"),
    include_str!("../../../docs/qwen-asr.en.md"),
    include_str!("../../../docs/qwen-asr-realtime.md"),
    include_str!("../../../docs/qwen-asr-realtime.en.md"),
    include_str!("../../../docs/qwen-knowledge.md"),
    include_str!("../../../docs/qwen-knowledge.en.md"),
    include_str!("../../../docs/qwen-knowledge-chunks.md"),
    include_str!("../../../docs/qwen-knowledge-chunks.en.md"),
    include_str!("../../../docs/qwen-knowledge-files.md"),
    include_str!("../../../docs/qwen-knowledge-files.en.md"),
    include_str!("../../../docs/qwen-knowledge-categories.md"),
    include_str!("../../../docs/qwen-knowledge-categories.en.md"),
    include_str!("../../../docs/qwen-knowledge-connectors.md"),
    include_str!("../../../docs/qwen-knowledge-connectors.en.md"),
    include_str!("../../../docs/qwen-knowledge-search.md"),
    include_str!("../../../docs/qwen-knowledge-search.en.md"),
    include_str!("../../../docs/qwen-knowledge-chat.md"),
    include_str!("../../../docs/qwen-knowledge-chat.en.md"),
    include_str!("../../../docs/xai-audio.md"),
    include_str!("../../../docs/xai-audio.en.md"),
    include_str!("../../../docs/xai-stt.md"),
    include_str!("../../../docs/xai-stt.en.md"),
    include_str!("../../../docs/xai-streaming-tts.md"),
    include_str!("../../../docs/xai-streaming-tts.en.md"),
    include_str!("../../../docs/xai-custom-voices.md"),
    include_str!("../../../docs/xai-custom-voices.en.md"),
    include_str!("../../../docs/xai-batch.md"),
    include_str!("../../../docs/xai-batch.en.md"),
    include_str!("../../../docs/qwen-tts.md"),
    include_str!("../../../docs/qwen-tts.en.md"),
    include_str!("../../../docs/qwen-audio-generation.md"),
    include_str!("../../../docs/qwen-audio-generation.en.md"),
    include_str!("../../../docs/qwen-tts-realtime.md"),
    include_str!("../../../docs/qwen-tts-realtime.en.md"),
    include_str!("../../../docs/glm-audio.md"),
    include_str!("../../../docs/glm-audio.en.md"),
    include_str!("../../../docs/minimax-tts.md"),
    include_str!("../../../docs/minimax-tts.en.md"),
    include_str!("../../../docs/gemini-batch.md"),
    include_str!("../../../docs/gemini-batch.en.md"),
    include_str!("../../../docs/qwen-rerank.md"),
    include_str!("../../../docs/qwen-rerank.en.md"),
    include_str!("../../../docs/openrouter-rerank.md"),
    include_str!("../../../docs/openrouter-rerank.en.md"),
    include_str!("../../../docs/glm-batch.md"),
    include_str!("../../../docs/glm-batch.en.md"),
    include_str!("../../../docs/gemini-context-cache.md"),
    include_str!("../../../docs/gemini-context-cache.en.md"),
    include_str!("../../../docs/openai-containers.md"),
    include_str!("../../../docs/openai-containers.en.md"),
    include_str!("../../../docs/minimax-async-tts.md"),
    include_str!("../../../docs/minimax-async-tts.en.md"),
    include_str!("../../../docs/gemini-live.md"),
    include_str!("../../../docs/gemini-live.en.md"),
    include_str!("../../../docs/xai-realtime.md"),
    include_str!("../../../docs/xai-realtime.en.md"),
    include_str!("../../../docs/openai-hosted-extended.md"),
    include_str!("../../../docs/openai-hosted-extended.en.md"),
    include_str!("../../../docs/openrouter-chat-audio.md"),
    include_str!("../../../docs/openrouter-chat-audio.en.md"),
    include_str!("../../../docs/glm-cloud-audio.md"),
    include_str!("../../../docs/glm-cloud-audio.en.md"),
    include_str!("../../../docs/gemini-speech.md"),
    include_str!("../../../docs/gemini-speech.en.md"),
    include_str!("../../../docs/gemini-chat-audio.md"),
    include_str!("../../../docs/gemini-chat-audio.en.md"),
    include_str!("../../../docs/gemini-voices.md"),
    include_str!("../../../docs/gemini-voices.en.md"),
    include_str!("../../../docs/vertex-speech.md"),
    include_str!("../../../docs/vertex-speech.en.md"),
    include_str!("../../../docs/capability-matrix-openai-gemini.md"),
    include_str!("../../../docs/capability-matrix-openai-gemini.en.md"),
    include_str!("../../../docs/capability-matrix-china.md"),
    include_str!("../../../docs/capability-matrix-china.en.md"),
    include_str!("../../../docs/glm-realtime.md"),
    include_str!("../../../docs/glm-realtime.en.md"),
    include_str!("../../../docs/capability-matrix-west.md"),
    include_str!("../../../docs/capability-matrix-west.en.md"),
    include_str!("../../../docs/xai-remote-mcp.md"),
    include_str!("../../../docs/xai-remote-mcp.en.md"),
    include_str!("../../../docs/minimax-streaming-tts.md"),
    include_str!("../../../docs/minimax-streaming-tts.en.md"),
    include_str!("../../../docs/minimax-bidi-tts.md"),
    include_str!("../../../docs/minimax-bidi-tts.en.md"),
    include_str!("../../../docs/qwen-prompt-cache.md"),
    include_str!("../../../docs/qwen-prompt-cache.en.md"),
    include_str!("../../../docs/qwen-hosted.md"),
    include_str!("../../../docs/qwen-hosted.en.md"),
    include_str!("../../../docs/qwen-realtime.md"),
    include_str!("../../../docs/qwen-realtime.en.md"),
    "\n\n",
    include_str!("../../../docs/qwen-translate.md"),
    "\n\n",
    include_str!("../../../docs/qwen-translate.en.md"),
    "\n\n",
    include_str!("../../../docs/openai-live.md"),
    "\n\n",
    include_str!("../../../docs/openai-live.en.md"),
    include_str!("../../../docs/openrouter-prompt-cache.md"),
    include_str!("../../../docs/openrouter-prompt-cache.en.md"),
    include_str!("../../../docs/qwen-web-extractor.md"),
    include_str!("../../../docs/qwen-web-extractor.en.md"),
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
        ) -> Result<ChatResponse, LlmError> {
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
