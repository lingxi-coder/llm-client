//! Traverse request content independently from tokenizer model mapping.
use super::{backends::Encoder, model::encoder_for, types::*};
use crate::{
    client::LlmClient,
    protocol::{
        CompletionRequest, ContentBlock, DocumentSource, ImageSource, MessageRole, ProtocolFamily,
        VideoSource,
    },
};
impl LlmClient {
    /// Estimate input tokens locally for the route selected by `request.model`.
    ///
    /// The estimate is synchronous and never sends a request or reads a
    /// credential. The route's first profile selects the tokenizer; if that
    /// route later fails over, this result does not count the fallback model.
    pub fn estimate_local_tokens(
        &self,
        request: &CompletionRequest,
    ) -> Result<LocalTokenEstimate, LocalTokenCountError> {
        self.estimate_local_tokens_resolved(request, self.resolve(&request.model)?)
    }

    /// Estimate input tokens locally after resolving `request.model` within a
    /// specific profile or connection group.
    pub fn estimate_local_tokens_in(
        &self,
        profile_or_group: &str,
        request: &CompletionRequest,
    ) -> Result<LocalTokenEstimate, LocalTokenCountError> {
        let route = self.resolve_in(&request.model, Some(profile_or_group))?;
        self.estimate_local_tokens_resolved(request, route)
    }

    fn estimate_local_tokens_resolved(
        &self,
        request: &CompletionRequest,
        route: crate::client::route::ResolvedRoute,
    ) -> Result<LocalTokenEstimate, LocalTokenCountError> {
        let encoder = encoder_for(&route.provider_id, &route.request_model)?.ok_or_else(|| {
            LocalTokenCountError::UnsupportedModel {
                profile_name: route.profile_name.clone(),
                provider_id: route.provider_id.clone(),
                request_model: route.request_model.clone(),
            }
        })?;

        let profile = self.profile(&route.profile_name).expect("resolved profile");
        let mut accumulator = Accumulator::new(&encoder, profile.protocol);
        for block in &request.system {
            accumulator.add_text(&block.text)?;
            accumulator.add_framing(2);
        }

        for message in &request.messages {
            accumulator.add_text(match message.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::System => "system",
            })?;
            // Estimate role/message framing. Providers do not expose a stable
            // local formula for all message variants and model generations.
            accumulator.add_framing(4);
            for block in &message.content {
                accumulator.add_content(block)?;
            }
        }

        for tool in &request.tools {
            accumulator.add_text(&tool.name)?;
            accumulator.add_text(&tool.description)?;
            accumulator.add_json(&tool.input_schema)?;
            if tool.strict {
                accumulator.add_text("strict")?;
            }
            accumulator.add_framing(8);
        }

        match &request.tool_choice {
            crate::protocol::ToolChoice::Auto => accumulator.add_text("auto")?,
            crate::protocol::ToolChoice::Any => accumulator.add_text("any")?,
            crate::protocol::ToolChoice::None => accumulator.add_text("none")?,
            crate::protocol::ToolChoice::Tool { name } => {
                accumulator.add_text("tool")?;
                accumulator.add_text(name)?;
            }
        }
        accumulator.add_framing(3); // estimated assistant-turn priming

        if request.thinking.is_some() {
            accumulator.add_framing(2);
        }
        if request.web_search.is_some() {
            accumulator.omit(LocalTokenEstimateOmission::HostedWebSearchContext);
        }
        if request.file_search.is_some() {
            accumulator.omit(LocalTokenEstimateOmission::HostedFileSearchContext);
        }
        if request.previous_response_id.is_some() {
            accumulator.omit(LocalTokenEstimateOmission::PreviousResponseState);
        }
        if !request.metadata.is_null() {
            accumulator.omit(LocalTokenEstimateOmission::ProviderMetadata);
        }

        let is_partial = !accumulator.uncounted_components.is_empty();
        Ok(LocalTokenEstimate {
            input_tokens: accumulator.input_tokens,
            profile_name: route.profile_name,
            provider_id: route.provider_id,
            request_model: route.request_model,
            tokenizer: encoder.name().to_owned(),
            is_estimate: true,
            is_partial,
            uncounted_components: accumulator.uncounted_components,
        })
    }
}

struct Accumulator<'a> {
    encoder: &'a Encoder,
    protocol: ProtocolFamily,
    input_tokens: u64,
    uncounted_components: Vec<LocalTokenEstimateOmission>,
}

impl<'a> Accumulator<'a> {
    fn new(encoder: &'a Encoder, protocol: ProtocolFamily) -> Self {
        Self {
            encoder,
            protocol,
            input_tokens: 0,
            uncounted_components: Vec::new(),
        }
    }

    fn add_text(&mut self, text: &str) -> Result<(), LocalTokenCountError> {
        self.input_tokens = self.input_tokens.saturating_add(self.encoder.count(text)?);
        Ok(())
    }

    fn add_json(&mut self, value: &serde_json::Value) -> Result<(), LocalTokenCountError> {
        let encoded = serde_json::to_string(value)
            .map_err(|error| LocalTokenCountError::Serialization(error.to_string()))?;
        self.add_text(&encoded)
    }

    fn add_framing(&mut self, tokens: u64) {
        self.input_tokens = self.input_tokens.saturating_add(tokens);
    }

    fn omit(&mut self, component: LocalTokenEstimateOmission) {
        self.uncounted_components.push(component);
    }

    fn add_content(&mut self, block: &ContentBlock) -> Result<(), LocalTokenCountError> {
        match block {
            ContentBlock::Text {
                text,
                thought_signature,
            } => {
                self.add_text(text)?;
                if thought_signature.is_some() {
                    self.omit(LocalTokenEstimateOmission::ProviderSignature);
                }
            }
            ContentBlock::ToolUse {
                id,
                name,
                input,
                provider_id,
                thought_signature,
            } => {
                self.add_text(id.as_str())?;
                self.add_text(name)?;
                self.add_json(input)?;
                if let Some(provider_id) = provider_id {
                    self.add_text(provider_id)?;
                }
                self.add_framing(2);
                if thought_signature.is_some() {
                    self.omit(LocalTokenEstimateOmission::ProviderSignature);
                }
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                blocks,
                ..
            } => {
                self.add_text(tool_use_id.as_str())?;
                self.add_text(content)?;
                if blocks.is_some() {
                    self.omit(LocalTokenEstimateOmission::ProviderOpaqueContent);
                }
            }
            ContentBlock::Thinking { text, signature } => {
                self.add_text(text)?;
                if signature.is_some() {
                    self.omit(LocalTokenEstimateOmission::ProviderSignature);
                }
            }
            ContentBlock::RedactedThinking { .. } => {
                // Only the Anthropic wire replays this encrypted payload. Its
                // contents cannot be recovered by tokenizing the ciphertext.
                if matches!(
                    self.protocol,
                    ProtocolFamily::AnthropicMessages
                        | ProtocolFamily::BedrockClaude
                        | ProtocolFamily::VertexClaude
                        | ProtocolFamily::FoundryClaude
                ) {
                    self.omit(LocalTokenEstimateOmission::ProviderOpaqueContent);
                }
            }
            ContentBlock::ProviderContent { .. } => {
                self.omit(LocalTokenEstimateOmission::ProviderOpaqueContent);
            }
            ContentBlock::Image { source } => match source {
                ImageSource::ProviderFile { .. } => {
                    self.omit(LocalTokenEstimateOmission::ProviderFileInput);
                }
                ImageSource::Base64 { .. }
                | ImageSource::Url { .. }
                | ImageSource::Attachment { .. } => {
                    self.omit(LocalTokenEstimateOmission::ImageInput);
                }
            },
            ContentBlock::Document { source, title } => {
                if let Some(title) = title {
                    self.add_text(title)?;
                }
                match source {
                    DocumentSource::Text { data, .. } => self.add_text(data)?,
                    DocumentSource::ProviderFile { .. } => {
                        self.omit(LocalTokenEstimateOmission::ProviderFileInput);
                    }
                    DocumentSource::Base64 { .. }
                    | DocumentSource::Url { .. }
                    | DocumentSource::Attachment { .. } => {
                        self.omit(LocalTokenEstimateOmission::DocumentInput);
                    }
                }
            }
            ContentBlock::Video { source } => match source {
                VideoSource::ProviderFile { .. } => {
                    self.omit(LocalTokenEstimateOmission::ProviderFileInput);
                }
                VideoSource::Base64 { .. }
                | VideoSource::Url { .. }
                | VideoSource::Attachment { .. } => {
                    self.omit(LocalTokenEstimateOmission::VideoInput);
                }
            },
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::LlmClientBuilder;
    #[cfg(feature = "tokenizers-all")]
    use crate::protocol::ProviderId;
    use crate::protocol::{ConversationMessage, ToolChoice};
    #[cfg(feature = "tokenizer-deepseek")]
    use crate::protocol::{
        FileSearchConfig, ResponseId, SystemBlock, ThinkingConfig, ToolSpec, WebSearchConfig,
    };
    #[cfg(feature = "tokenizer-openai")]
    use crate::token_count::backends::openai_encoder;
    #[cfg(feature = "tokenizer-deepseek")]
    use serde_json::json;

    fn client(profile_name: &str, region: crate::protocol::Region) -> LlmClient {
        let profiles = crate::presets::builtin()
            .unwrap()
            .into_iter()
            .filter(|profile| profile.profile_name == profile_name)
            .collect::<Vec<_>>();
        assert_eq!(profiles.len(), 1);
        LlmClientBuilder::new(&profiles)
            .unwrap()
            .with_region(region)
            .build()
            .unwrap()
    }

    #[cfg(feature = "tokenizer-openai")]
    #[test]
    fn openai_mapping_counts_with_model_specific_encodings() {
        let o200k = openai_encoder("gpt-4o").expect("known OpenAI model");
        let cl100k = openai_encoder("gpt-4").expect("known OpenAI model");
        assert_eq!(o200k.count("tiktoken is great!").unwrap(), 6);
        assert_eq!(cl100k.count("hello world").unwrap(), 2);
        assert!(openai_encoder("gpt-6-astra").is_none());
    }

    #[cfg(feature = "tokenizers-all")]
    #[test]
    fn bundled_provider_assets_load_and_count_text() {
        let cases = [
            (
                ProviderId::from("deepseek"),
                "deepseek-v4-pro",
                "你好，世界",
            ),
            (
                ProviderId::from("deepseek"),
                "deepseek-flash",
                "Hello world",
            ),
            (ProviderId::from("qwen"), "qwen3.8-flash", "你好，世界"),
            (ProviderId::from("kimi"), "kimi-k3", "Hello, world!"),
            (ProviderId::from("zhipu"), "glm-5", "Hello, world!"),
        ];
        for (provider, model, text) in cases {
            let encoder = encoder_for(&provider, model)
                .unwrap()
                .expect("model has a pinned tokenizer");
            assert!(encoder.count(text).unwrap() > 0, "{provider}/{model}");
        }
    }

    #[cfg(feature = "tokenizer-deepseek")]
    #[test]
    fn estimates_visible_conversation_tools_and_reports_omissions() {
        let client = client("deepseek", crate::protocol::Region::ChinaMainland);
        let request = CompletionRequest {
            controls: Default::default(),
            service_tier: None,
            model: "deepseek-flash".to_owned(),
            web_search: Some(WebSearchConfig::default()),
            file_search: Some(FileSearchConfig {
                knowledge_base_id: "kb-1".to_owned(),
                workspace_id: "workspace-1".to_owned(),
            }),
            previous_response_id: Some(ResponseId::new("resp_previous")),
            system: vec![SystemBlock {
                cache_control: None,
                text: "system prompt".to_owned(),
                cacheable: false,
            }],
            messages: vec![
                ConversationMessage::user_text("hello world"),
                ConversationMessage::assistant(vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "lookup".to_owned(),
                    input: json!({"query": "local token count"}),
                    provider_id: Some("provider-call-1".to_owned()),
                    thought_signature: None,
                }]),
                ConversationMessage {
                    role: MessageRole::User,
                    content: vec![
                        ContentBlock::ToolResult {
                            tool_use_id: "call-1".into(),
                            content: "found a result".to_owned(),
                            is_error: false,
                            blocks: None,
                        },
                        ContentBlock::Text {
                            text: "follow-up text".to_owned(),
                            thought_signature: Some("opaque-signature".to_owned()),
                        },
                        ContentBlock::Image {
                            source: ImageSource::Url {
                                url: "https://example.invalid/image.png".to_owned(),
                            },
                        },
                        ContentBlock::Document {
                            source: DocumentSource::Text {
                                media_type: "text/plain".to_owned(),
                                data: "counted document text".to_owned(),
                            },
                            title: Some("notes".to_owned()),
                        },
                        ContentBlock::Document {
                            source: DocumentSource::Url {
                                url: "https://example.invalid/remote.pdf".to_owned(),
                            },
                            title: None,
                        },
                        ContentBlock::Video {
                            source: VideoSource::Url {
                                url: "https://example.invalid/clip.mp4".to_owned(),
                            },
                        },
                        ContentBlock::ProviderContent {
                            protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
                            value: json!({"type": "opaque_provider_block"}),
                        },
                    ],
                },
            ],
            tools: vec![ToolSpec {
                tool_type: None,
                defer_loading: None,
                extra: serde_json::Value::Null,
                name: "lookup".to_owned(),
                description: "search local records".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {"query": {"type": "string"}}
                }),
                strict: true,
            }],
            tool_choice: ToolChoice::Tool {
                name: "lookup".to_owned(),
            },
            max_tokens: Some(2048),
            temperature: Some(0.2),
            thinking: Some(ThinkingConfig {
                budget: Some(crate::protocol::ThinkingBudget::Tokens(512)),
                ..ThinkingConfig::default()
            }),
            stop_sequences: vec!["END".to_owned()],
            metadata: json!({"provider_hint": "opaque"}),
        };

        let estimate = client.estimate_local_tokens(&request).unwrap();
        assert!(estimate.input_tokens > 0);
        assert!(estimate.is_estimate);
        assert!(estimate.is_partial);
        assert_eq!(estimate.profile_name, "deepseek");
        assert_eq!(estimate.request_model, "deepseek-flash");
        for omitted in [
            LocalTokenEstimateOmission::ImageInput,
            LocalTokenEstimateOmission::DocumentInput,
            LocalTokenEstimateOmission::VideoInput,
            LocalTokenEstimateOmission::HostedWebSearchContext,
            LocalTokenEstimateOmission::HostedFileSearchContext,
            LocalTokenEstimateOmission::PreviousResponseState,
            LocalTokenEstimateOmission::ProviderOpaqueContent,
            LocalTokenEstimateOmission::ProviderSignature,
            LocalTokenEstimateOmission::ProviderMetadata,
        ] {
            assert!(
                estimate.uncounted_components.contains(&omitted),
                "{omitted:?}"
            );
        }

        let mut output_cap_changed = request.clone();
        output_cap_changed.max_tokens = Some(16_384);
        assert_eq!(
            estimate.input_tokens,
            client
                .estimate_local_tokens_in("deepseek", &output_cap_changed)
                .unwrap()
                .input_tokens
        );
    }

    #[test]
    fn unsupported_openai_model_does_not_fall_back_to_another_tokenizer() {
        let openai = client("openai", crate::protocol::Region::International);
        let request = CompletionRequest {
            controls: Default::default(),
            service_tier: None,
            model: "gpt-6-astra".to_owned(),
            web_search: None,
            file_search: None,
            previous_response_id: None,
            system: Vec::new(),
            messages: vec![ConversationMessage::user_text("hello")],
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            max_tokens: None,
            temperature: None,
            thinking: None,
            stop_sequences: Vec::new(),
            metadata: serde_json::Value::Null,
        };
        assert!(matches!(
            openai.estimate_local_tokens(&request),
            Err(LocalTokenCountError::UnsupportedModel { .. })
        ));

        let anthropic = client("anthropic", crate::protocol::Region::International);
        let mut online_counter_only = request;
        online_counter_only.model = "claude-fable-5".to_owned();
        assert!(matches!(
            anthropic.estimate_local_tokens(&online_counter_only),
            Err(LocalTokenCountError::UnsupportedModel { .. })
        ));
    }
}
