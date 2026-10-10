//! Pure request data passed to wire codecs. No credentials or execution state.
use crate::protocol::{ChatRequest, CredentialConfig, ModelProfile, ProviderInfo, ProviderProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestMode {
    Complete,
    Stream,
    /// Exact provider-side input counting, without generating a response.
    CountTokens,
}

/// A connection snapshot containing only the data used to encode a request.
/// Credential configuration and prices are removed when the snapshot is made.
#[derive(Debug, Clone)]
pub struct CodecContext {
    pub(crate) profile: ProviderProfile,
    pub(crate) request_model: String,
    pub(crate) stream: bool,
    mode: RequestMode,
    pub(crate) file_account_scope: Option<String>,
    account_scope: Option<String>,
    native_options: Vec<crate::protocol::NativeExtension>,
    anthropic_request_kind: crate::providers::anthropic::request_policy::AnthropicRequestKind,
    file_validation_time: Option<std::time::SystemTime>,
    native_anthropic_context: crate::client::NativeAnthropicContextInput,
    native_anthropic_endpoint_admitted: bool,
    native_anthropic_thinking_resumption_beta_requested: Option<bool>,
    model_complete_empty_cache_blocks: bool,
}
impl CodecContext {
    /// Build a context from a wire ID. Use [`Self::for_model`] when selecting
    /// one of several catalog rows that share that ID.
    pub fn new(
        profile: &ProviderProfile,
        request_model: impl Into<String>,
        mode: RequestMode,
    ) -> Self {
        let request_model = request_model.into();
        Self::from_models(
            profile,
            &request_model,
            profile
                .models
                .iter()
                .filter(|model| model.request_model == request_model),
            mode,
        )
    }
    /// Encode one selected catalog row, including when multiple rows share a wire ID.
    pub fn for_model(profile: &ProviderProfile, model: &ModelProfile, mode: RequestMode) -> Self {
        Self::from_models(profile, &model.request_model, std::iter::once(model), mode)
    }
    fn from_models<'a>(
        profile: &ProviderProfile,
        request_model: &str,
        models: impl Iterator<Item = &'a ModelProfile>,
        mode: RequestMode,
    ) -> Self {
        let connection = ProviderProfile {
            inference: profile.inference.clone(),
            chat_enabled: profile.chat_enabled,
            regions: Vec::new(),
            provider_id: profile.provider_id.clone(),
            profile_name: profile.profile_name.clone(),
            base_url: profile.base_url.clone(),
            protocol: profile.protocol,
            model_list: Default::default(),
            auth: profile.auth,
            credential: CredentialConfig::None,
            models: models
                .map(|model| {
                    let mut model = model.clone();
                    model.pricing = None;
                    model.info.pricing = None;
                    model.billing_mode = None;
                    model
                })
                .collect(),
            images: Default::default(),
            embeddings: Default::default(),
            decisions: profile.decisions,
            retrieval: Default::default(),
            batches: Default::default(),
            deferred: Default::default(),
            background: Default::default(),
            audio: Default::default(),
            interactions: Default::default(),
            gemini_file_search: Default::default(),
            glm_knowledge: Default::default(),
            pricing: Default::default(),
            signing: profile.signing.clone(),
            azure: profile.azure.clone(),
            supports_websockets: false,
            supports_websocket_compression: false,
            websocket_connect_timeout_ms: None,
            connection: Default::default(),
            info: ProviderInfo {
                features: profile.info.features.clone(),
                ..Default::default()
            },
            extra: profile.extra.clone(),
        };
        Self {
            profile: connection,
            request_model: request_model.to_owned(),
            stream: mode == RequestMode::Stream,
            mode,
            file_account_scope: None,
            account_scope: None,
            native_options: Vec::new(),
            anthropic_request_kind: Default::default(),
            file_validation_time: None,
            native_anthropic_context: crate::client::NativeAnthropicContextInput::default(),
            native_anthropic_endpoint_admitted: false,
            native_anthropic_thinking_resumption_beta_requested: None,
            model_complete_empty_cache_blocks: false,
        }
    }
    /// Set the caller's non-secret identity for account-bound execution state.
    pub fn with_account_scope(mut self, scope: Option<&str>) -> Self {
        self.account_scope = scope.map(str::to_owned);
        self
    }
    pub fn with_native_options(mut self, options: &[crate::protocol::NativeExtension]) -> Self {
        self.native_options = options.to_vec();
        self
    }
    pub fn native_options(&self) -> &[crate::protocol::NativeExtension] {
        &self.native_options
    }
    /// Bind Anthropic request-local encoding policy to the selected host call.
    pub fn with_anthropic_request_kind(
        mut self,
        kind: crate::providers::anthropic::request_policy::AnthropicRequestKind,
    ) -> Self {
        self.anthropic_request_kind = kind;
        self
    }
    /// Apply a request-local capability observation to this selected model.
    /// The connection's wire vocabulary and every other capability are retained.
    /// An unknown observation retains the configured catalog fact.
    pub fn with_fast_capability(
        mut self,
        support: Option<crate::protocol::CapabilitySupport>,
    ) -> Self {
        if let Some(support) =
            support.filter(|value| *value != crate::protocol::CapabilitySupport::Unknown)
        {
            for model in &mut self.profile.models {
                model.info.features.fast = support;
            }
        }
        self
    }
    pub fn account_scope(&self) -> Option<&str> {
        self.account_scope.as_deref()
    }
    /// Supply the wall-clock instant used for provider-file expiry validation.
    /// High-level calls refresh this from the client clock before dispatch.
    /// Direct callers normally omit it and use the current system time.
    pub fn with_file_validation_time(mut self, now: std::time::SystemTime) -> Self {
        self.file_validation_time = Some(now);
        self
    }
    /// Supply exact Native context observations from the selected current
    /// route. Unknown observations remain absent; generic directory defaults
    /// are never promoted into Native facts.
    pub fn with_native_anthropic_context(
        mut self,
        context: crate::client::NativeAnthropicContextInput,
    ) -> Self {
        self.native_anthropic_thinking_resumption_beta_requested = self
            .native_anthropic_error_route()
            .and(context.thinking_resumption_beta_requested);
        self.native_anthropic_context = context;
        self
    }

    /// Bind request-local Native Messages behavior to the exact profile
    /// selected by the host. A mismatched profile never receives admission.
    pub fn with_native_anthropic_endpoint_admission(
        mut self,
        admission: Option<&crate::client::NativeAnthropicEndpointAdmission>,
    ) -> Self {
        self.native_anthropic_endpoint_admitted =
            admission.is_some_and(|admission| admission.matches_profile(&self.profile));
        self
    }

    pub(crate) fn native_anthropic_endpoint_admitted(&self) -> bool {
        self.native_anthropic_endpoint_admitted
    }

    /// Native-only wire normalization for a host-selected first-party main
    /// generation. Other Anthropic-compatible routes retain their own shape.
    pub(crate) fn native_anthropic_main_request(&self) -> bool {
        self.native_anthropic_endpoint_admitted
            && self.profile.protocol == crate::protocol::ProtocolFamily::AnthropicMessages
            && self.anthropic_main_generation_request()
    }

    pub(crate) fn anthropic_main_generation_request(&self) -> bool {
        matches!(self.mode, RequestMode::Complete | RequestMode::Stream)
            && self.anthropic_request_kind
                == crate::providers::anthropic::request_policy::AnthropicRequestKind::Main
    }

    /// Native's query/resumption implementation runs only on its Anthropic
    /// Messages providers. Arbitrary Anthropic-compatible profiles remain
    /// outside this route even when they use the same wire format.
    pub(crate) fn native_anthropic_error_route(
        &self,
    ) -> Option<(crate::protocol::ProviderId, crate::protocol::ProtocolFamily)> {
        let profile = &self.profile;
        let selected = matches!(
            (profile.protocol, profile.provider_id.as_str()),
            (
                crate::protocol::ProtocolFamily::AnthropicMessages,
                "anthropic"
            ) | (
                crate::protocol::ProtocolFamily::AnthropicMessages,
                "anthropicAws"
            ) | (
                crate::protocol::ProtocolFamily::AnthropicMessages,
                "anthropicGoogleCloud"
            ) | (
                crate::protocol::ProtocolFamily::AnthropicMessages,
                "foundry"
            ) | (
                crate::protocol::ProtocolFamily::AnthropicMessages,
                "gateway"
            ) | (crate::protocol::ProtocolFamily::AnthropicMessages, "mantle")
                | (crate::protocol::ProtocolFamily::BedrockClaude, _)
                | (crate::protocol::ProtocolFamily::VertexClaude, _)
                | (crate::protocol::ProtocolFamily::FoundryClaude, _)
        );
        selected.then(|| (profile.provider_id.clone(), profile.protocol))
    }

    /// Native Aet consults strict membership in the request's planned `betas`
    /// array, not the post-authentication HTTP header collection.
    pub(crate) fn native_anthropic_thinking_resumption_beta_requested(&self) -> Option<bool> {
        self.native_anthropic_thinking_resumption_beta_requested
    }

    pub(crate) fn set_native_anthropic_thinking_resumption_beta_requested(
        &mut self,
        requested: Option<bool>,
    ) {
        self.native_anthropic_thinking_resumption_beta_requested =
            self.native_anthropic_error_route().and(requested);
    }

    pub(crate) fn set_native_anthropic_context_1m_beta_requested(
        &mut self,
        requested: Option<bool>,
    ) {
        self.native_anthropic_context.context_1m_beta_requested =
            self.native_anthropic_error_route().and(requested);
    }

    /// Resolve only context sources that are present in the current request or
    /// Native process. A missing Native registry/dynamic observation stays
    /// unknown instead of inheriting the unrelated generic 200k fallback.
    pub(crate) fn native_anthropic_effective_context_window(&self) -> Option<f64> {
        self.native_anthropic_error_route()?;
        if std::env::var("DISABLE_COMPACT").is_ok_and(|value| !value.is_empty()) {
            if let Ok(value) = std::env::var("CLAUDE_CODE_MAX_CONTEXT_TOKENS") {
                if let Ok(window) = value.parse::<f64>() {
                    if window.is_finite() && window > 0.0 {
                        return Some(window);
                    }
                }
            }
        }
        if let Some(window) = self
            .native_anthropic_context
            .dynamic_context_window
            .filter(|window| *window > 0)
        {
            return Some(window as f64);
        }
        if self.request_model.ends_with("[1m]")
            && self.native_anthropic_context.model_supports_context_1m == Some(true)
        {
            return Some(1_000_000.0);
        }
        let one_million_beta_requested = self.native_anthropic_context.context_1m_beta_requested
            == Some(true)
            && self.native_anthropic_context.model_supports_context_1m == Some(true);
        if one_million_beta_requested {
            return Some(1_000_000.0);
        }
        self.native_anthropic_context
            .model_registry_context_window
            .filter(|window| *window > 0)
            .map(|window| window as f64)
    }
    pub fn file_validation_time(&self) -> std::time::SystemTime {
        self.file_validation_time
            .unwrap_or_else(std::time::SystemTime::now)
    }
    /// Preserve Native model.complete empty text blocks when they carry an
    /// explicit cache marker. Other request paths retain the nonempty rule.
    pub fn with_model_complete_empty_cache_blocks(mut self, allow: bool) -> Self {
        self.model_complete_empty_cache_blocks = allow;
        self
    }
    pub(crate) fn allows_model_complete_empty_cache_blocks(&self) -> bool {
        self.model_complete_empty_cache_blocks
    }
    pub fn with_file_scope(mut self, scope: Option<&str>) -> Self {
        self.file_account_scope = scope.map(str::to_owned);
        self
    }
    pub fn profile(&self) -> &ProviderProfile {
        &self.profile
    }
    pub fn request_model(&self) -> &str {
        &self.request_model
    }
    pub fn mode(&self) -> RequestMode {
        self.mode
    }
    pub fn file_scope(&self) -> Option<&str> {
        self.file_account_scope.as_deref()
    }
}

/// Borrowed model input. Media bindings are prepared separately from the
/// caller-owned conversation and never carry upload/cache or credential state.
#[derive(Clone, Copy)]
pub struct EncodeRequest<'a> {
    request: &'a ChatRequest,
    media: &'a [PreparedMedia<'a>],
    bindings: &'a [ContentBinding<'a>],
}
impl<'a> EncodeRequest<'a> {
    pub fn new(request: &'a ChatRequest) -> Self {
        Self {
            request,
            media: &[],
            bindings: &[],
        }
    }
    pub fn request(self) -> &'a ChatRequest {
        self.request
    }
    pub fn blocks(self) -> impl Iterator<Item = &'a crate::protocol::ContentBlock> {
        self.request
            .messages
            .iter()
            .flat_map(|m| &m.content)
            .map(move |block| self.block(block))
    }
    pub fn media(self) -> &'a [PreparedMedia<'a>] {
        self.media
    }
}

/// Immutable attachment data resolved by the caller or attachment service.
#[derive(Clone, Copy)]
pub struct PreparedMedia<'a> {
    pub attachment: &'a crate::protocol::AttachmentRef,
    pub bytes: &'a [u8],
}

/// An attempt-local replacement for a single caller-owned content block.
/// Pointer identity associates a binding with its original block without cloning the conversation.
pub struct ContentBinding<'a> {
    pub original: &'a crate::protocol::ContentBlock,
    pub replacement: crate::protocol::ContentBlock,
}
impl<'a> EncodeRequest<'a> {
    pub fn with_media(mut self, media: &'a [PreparedMedia<'a>]) -> Self {
        self.media = media;
        self
    }
    pub fn with_bindings(mut self, bindings: &'a [ContentBinding<'a>]) -> Self {
        self.bindings = bindings;
        self
    }
    pub fn block<'b>(
        self,
        original: &'b crate::protocol::ContentBlock,
    ) -> &'b crate::protocol::ContentBlock
    where
        'a: 'b,
    {
        self.bindings
            .iter()
            .find(|binding| std::ptr::eq(binding.original, original))
            .map_or(original, |binding| &binding.replacement)
    }
    pub fn inline_media(
        self,
        block: &crate::protocol::ContentBlock,
    ) -> Result<Option<PreparedMedia<'a>>, crate::protocol::LlmError> {
        use crate::protocol::{ContentBlock, DocumentSource, ImageSource, LlmError, VideoSource};
        let attachment = match block {
            ContentBlock::Image {
                source: ImageSource::Attachment { attachment },
            }
            | ContentBlock::Document {
                source: DocumentSource::Attachment { attachment },
                ..
            }
            | ContentBlock::Video {
                source: VideoSource::Attachment { attachment },
            } => attachment,
            _ => return Ok(None),
        };
        let media = self
            .media
            .iter()
            .find(|media| media.attachment == attachment)
            .copied()
            .ok_or_else(super::unresolved_attachment_error)?;
        if media.bytes.len() as u64 != attachment.size_bytes {
            return Err(LlmError::InvalidRequest {
                message: "prepared media length does not match attachment metadata".into(),
            });
        }
        Ok(Some(media))
    }
}
