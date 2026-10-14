//! Pure request data passed to wire codecs. No credentials or execution state.
use crate::protocol::{
    ChatRequest, CredentialConfig, JsonPayload, ModelProfile, ProviderInfo, ProviderProfile,
};

/// Native removes these exact context-window markers from a selected model
/// before sending it to Claude. Work on UTF-16 units so a surrounding lone
/// surrogate remains part of the source identity.
pub(crate) fn normalized_model_source_units(
    source: &JsonPayload,
) -> Result<Vec<u16>, crate::protocol::LlmError> {
    source.validate_root_string("ChatRequest.model_source")?;
    let units = source
        .root_string_code_units()
        .expect("validated model source has a JSON string root");
    let mut normalized = Vec::with_capacity(units.len());
    let mut offset = 0;
    while offset < units.len() {
        let marker_matches = units
            .get(offset..offset.saturating_add(4))
            .is_some_and(|marker| {
                marker.len() == 4
                    && marker[0] == b'[' as u16
                    && (marker[1] == b'1' as u16 || marker[1] == b'2' as u16)
                    && (marker[2] == b'm' as u16 || marker[2] == b'M' as u16)
                    && marker[3] == b']' as u16
            });
        if marker_matches {
            offset += 4;
        } else {
            normalized.push(units[offset]);
            offset += 1;
        }
    }
    Ok(normalized)
}

fn utf16_ends_with_ascii(units: &[u16], suffix: &str) -> bool {
    let suffix = suffix.as_bytes();
    units.len() >= suffix.len()
        && units[units.len() - suffix.len()..]
            .iter()
            .zip(suffix)
            .all(|(unit, expected)| *unit == u16::from(*expected))
}

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
    selected_route_model: Option<ModelProfile>,
    model_identity_units: Vec<u16>,
    model_source: Option<JsonPayload>,
    model_source_has_1m_suffix: bool,
    pub(crate) stream: bool,
    mode: RequestMode,
    pub(crate) file_account_scope: Option<String>,
    account_scope: Option<String>,
    native_options: Vec<crate::protocol::NativeExtension>,
    anthropic_request_kind: crate::providers::anthropic::request_policy::AnthropicRequestKind,
    file_validation_time: Option<std::time::SystemTime>,
    native_anthropic_context: crate::client::NativeAnthropicContextInput,
    native_anthropic_request_context: Option<crate::client::NativeAnthropicRequestContext>,
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
        let models = models
            .map(|model| {
                let mut model = model.clone();
                model.pricing = None;
                model.info.pricing = None;
                model.billing_mode = None;
                model
            })
            .collect::<Vec<_>>();
        let selected_route_model = models.first().cloned();
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
            models,
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
            selected_route_model,
            model_identity_units: request_model.encode_utf16().collect(),
            model_source: None,
            model_source_has_1m_suffix: false,
            stream: mode == RequestMode::Stream,
            mode,
            file_account_scope: None,
            account_scope: None,
            native_options: Vec::new(),
            anthropic_request_kind: Default::default(),
            file_validation_time: None,
            native_anthropic_context: crate::client::NativeAnthropicContextInput::default(),
            native_anthropic_request_context: None,
            native_anthropic_endpoint_admitted: false,
            native_anthropic_thinking_resumption_beta_requested: None,
            model_complete_empty_cache_blocks: false,
        }
    }
    /// Replace the selected catalog identity with the Native source model for
    /// model-scoped capability, inference, and wire decisions. The route still
    /// supplies connection, transport, and authentication configuration.
    pub(crate) fn with_model_source(
        mut self,
        source: Option<&JsonPayload>,
        source_profile: &ProviderProfile,
    ) -> Result<Self, crate::protocol::LlmError> {
        let Some(source) = source else {
            return Ok(self);
        };
        source.validate_root_string("ChatRequest.model_source")?;
        let original_units = source
            .root_string_code_units()
            .expect("validated model source has a JSON string root");
        let normalized_units = normalized_model_source_units(source)?;
        let normalized_string = String::from_utf16(&normalized_units).ok();
        let selected_route_model = normalized_string
            .as_ref()
            .and_then(|model| {
                self.selected_route_model
                    .as_ref()
                    .filter(|candidate| candidate.request_model == *model)
            })
            .cloned();
        self.model_identity_units.clone_from(&normalized_units);
        self.model_source_has_1m_suffix = utf16_ends_with_ascii(&original_units, "[1m]");
        self.model_source = Some(source.clone());
        self.profile.models.clear();
        match normalized_string {
            Some(model) => {
                self.request_model.clone_from(&model);
                let mut exact_rows = source_profile
                    .models
                    .iter()
                    .filter(|candidate| candidate.request_model == model);
                let source_model = selected_route_model.or_else(|| {
                    let first = exact_rows.next()?.clone();
                    exact_rows.next().is_none().then_some(first)
                });
                if let Some(mut source_model) = source_model {
                    source_model.pricing = None;
                    source_model.info.pricing = None;
                    source_model.billing_mode = None;
                    self.profile.models.push(source_model);
                }
            }
            None => {
                // A malformed UTF-16 selector has no Rust string key. Keep its
                // code units as the authoritative identity, with no route
                // model's catalog capabilities attached to it.
                self.request_model.clear();
            }
        }
        Ok(self)
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
        mut context: crate::client::NativeAnthropicContextInput,
    ) -> Self {
        self.native_anthropic_thinking_resumption_beta_requested = self
            .native_anthropic_error_route()
            .and(context.thinking_resumption_beta_requested);
        context.native_thinking_binding_controls_beta_requested = self
            .native_thinking_binding_controls_beta_route()
            .then_some(context.native_thinking_binding_controls_beta_requested)
            .flatten();
        self.native_anthropic_context = context;
        self
    }

    /// Carry trusted current-request Native facts to the selected codec.
    /// The codec exposes them only for an admitted Native main-generation route.
    pub fn with_native_anthropic_request_context(
        mut self,
        context: Option<crate::client::NativeAnthropicRequestContext>,
    ) -> Self {
        self.native_anthropic_request_context = context;
        self
    }

    pub(crate) fn native_anthropic_request_context(
        &self,
    ) -> Option<&crate::client::NativeAnthropicRequestContext> {
        (self.native_anthropic_main_request() || self.native_foundry_main_request())
            .then_some(self.native_anthropic_request_context.as_ref())
            .flatten()
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

    /// A main Foundry request can carry the host's narrowly-scoped Native
    /// per-turn effort marker. This does not admit other Native-only options.
    pub(crate) fn native_foundry_main_request(&self) -> bool {
        self.profile.protocol == crate::protocol::ProtocolFamily::FoundryClaude
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

    /// Exact planned membership for Native's thinking-binding-controls beta.
    /// Its producer fact is accepted only on a recognized Native Anthropic route.
    pub(crate) fn native_thinking_binding_controls_beta_requested(&self) -> Option<bool> {
        self.native_thinking_binding_controls_beta_route()
            .then_some(
                self.native_anthropic_context
                    .native_thinking_binding_controls_beta_requested,
            )
            .flatten()
    }

    fn native_thinking_binding_controls_beta_route(&self) -> bool {
        self.native_anthropic_error_route().is_some()
    }

    pub(crate) fn set_native_thinking_binding_controls_beta_requested(
        &mut self,
        requested: Option<bool>,
    ) {
        self.native_anthropic_context
            .native_thinking_binding_controls_beta_requested = self
            .native_thinking_binding_controls_beta_route()
            .then_some(requested)
            .flatten();
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
        if (self.model_source_has_1m_suffix
            || (self.model_source.is_none() && self.request_model.ends_with("[1m]")))
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

    /// Host-directed refusal retry suppression, limited to Native Anthropic
    /// routes. This is applied to the final request before authentication.
    pub(crate) fn native_anthropic_context_1m_beta_suppressed(&self) -> bool {
        self.native_anthropic_context.context_1m_beta_suppressed
            && self.native_anthropic_error_route().is_some()
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
    /// Exact Native-normalized model identity, including isolated UTF-16
    /// surrogates that cannot be represented by `request_model`.
    pub(crate) fn model_identity_matches(&self, model: &str) -> bool {
        self.model_identity_units == model.encode_utf16().collect::<Vec<_>>()
    }
    pub(crate) fn model_identity_matches_any(&self, models: &[&str]) -> bool {
        models
            .iter()
            .any(|model| self.model_identity_matches(model))
    }
    pub(crate) fn model_identity_display(&self) -> String {
        String::from_utf16_lossy(&self.model_identity_units)
    }
    pub(crate) fn has_model_source(&self) -> bool {
        self.model_source.is_some()
    }
    pub(crate) fn selected_route_model(&self) -> Option<&ModelProfile> {
        self.selected_route_model.as_ref()
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

#[cfg(test)]
mod model_source_context_tests {
    use super::*;
    use crate::protocol::ProtocolFamily;
    use serde_json::json;

    fn profile() -> ProviderProfile {
        serde_json::from_value(json!({
            "provider_id": "anthropic",
            "profile_name": "anthropic",
            "base_url": "https://api.anthropic.com",
            "protocol": "anthropic_messages",
            "auth": "none",
            "models": [
                {
                    "display_model": "route-selector",
                    "request_model": "route-selector",
                    "billing_model": "route-selector"
                },
                {
                    "display_model": "claude-sonnet-5-5",
                    "request_model": "claude-sonnet-5-5",
                    "billing_model": "claude-sonnet-5-5"
                }
            ]
        }))
        .expect("fixture profile")
    }

    #[test]
    fn source_model_replaces_route_model_identity_and_keeps_context_marker_semantics() {
        let profile = profile();
        let route_model = profile
            .models
            .iter()
            .find(|model| model.request_model == "route-selector")
            .unwrap();
        let source = JsonPayload::string("claude-sonnet-5-5[1m]");
        let context = CodecContext::for_model(&profile, route_model, RequestMode::Complete)
            .with_model_source(Some(&source), &profile)
            .expect("valid Native model source");

        assert_eq!(context.request_model(), "claude-sonnet-5-5");
        assert!(context.model_identity_matches("claude-sonnet-5-5"));
        assert!(!context.model_identity_matches("route-selector"));
        assert!(context.model_source_has_1m_suffix);
        assert_eq!(context.profile().models.len(), 1);
        assert_eq!(
            context.profile().models[0].request_model,
            "claude-sonnet-5-5"
        );
        assert_eq!(
            context.profile().protocol,
            ProtocolFamily::AnthropicMessages
        );
        assert_eq!(
            context.selected_route_model().unwrap().request_model,
            "route-selector"
        );
    }

    #[test]
    fn malformed_source_keeps_distinct_units_and_ascii_family_identity() {
        let profile = profile();
        let route_model = &profile.models[0];
        let high_units = [vec![0xD800], "claude-sonnet-5-5".encode_utf16().collect()].concat();
        let low_units = [vec![0xDC00], "claude-sonnet-5-5".encode_utf16().collect()].concat();
        let high = JsonPayload::string_from_utf16_code_units(&high_units);
        let low = JsonPayload::string_from_utf16_code_units(&low_units);
        let high_context = CodecContext::for_model(&profile, route_model, RequestMode::Complete)
            .with_model_source(Some(&high), &profile)
            .expect("lone surrogates are valid JSON string sources");
        let low_context = CodecContext::for_model(&profile, route_model, RequestMode::Complete)
            .with_model_source(Some(&low), &profile)
            .expect("lone surrogates are valid JSON string sources");

        assert_eq!(high_context.request_model(), "");
        assert!(!high_context.model_identity_matches("route-selector"));
        assert_ne!(
            high_context.model_identity_units,
            low_context.model_identity_units
        );
        assert!(high_context.profile().models.is_empty());
    }

    #[test]
    fn tbc_membership_preserves_tri_state_only_for_native_anthropic_routes() {
        fn tbc_profile(protocol: &str, provider_id: &str) -> ProviderProfile {
            serde_json::from_value(serde_json::json!({
                "provider_id": provider_id,
                "profile_name": "tbc-route-test",
                "base_url": "https://example.test",
                "protocol": protocol,
                "auth": "none",
                "models": [{
                    "display_model": "wire-model",
                    "request_model": "wire-model",
                    "billing_model": "wire-model"
                }]
            }))
            .unwrap()
        }

        for (protocol, provider_id) in [
            ("anthropic_messages", "anthropic"),
            ("anthropic_messages", "anthropicAws"),
            ("anthropic_messages", "anthropicGoogleCloud"),
            ("anthropic_messages", "foundry"),
            ("anthropic_messages", "gateway"),
            ("anthropic_messages", "mantle"),
            ("foundry_claude", "foundry"),
            ("bedrock_claude", "bedrock"),
            ("vertex_claude", "vertex"),
        ] {
            let profile = tbc_profile(protocol, provider_id);
            for requested in [None, Some(false), Some(true)] {
                let context = CodecContext::new(&profile, "wire-model", RequestMode::Complete)
                    .with_native_anthropic_context(crate::client::NativeAnthropicContextInput {
                        native_thinking_binding_controls_beta_requested: requested,
                        ..Default::default()
                    });
                assert_eq!(
                    context.native_thinking_binding_controls_beta_requested(),
                    requested,
                    "protocol={protocol} requested={requested:?}"
                );
                let mut post_encode =
                    CodecContext::new(&profile, "wire-model", RequestMode::Complete)
                        .with_native_anthropic_context(Default::default());
                post_encode.set_native_thinking_binding_controls_beta_requested(requested);
                assert_eq!(
                    post_encode.native_thinking_binding_controls_beta_requested(),
                    requested,
                    "late setter protocol={protocol} requested={requested:?}"
                );
            }
        }

        let generic_compatible = tbc_profile("anthropic_messages", "third-party-compatible");
        let blocked = CodecContext::new(&generic_compatible, "wire-model", RequestMode::Complete)
            .with_native_anthropic_context(crate::client::NativeAnthropicContextInput {
                native_thinking_binding_controls_beta_requested: Some(true),
                ..Default::default()
            });
        assert_eq!(
            blocked.native_thinking_binding_controls_beta_requested(),
            None
        );
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
