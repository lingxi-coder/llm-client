//! Typed source facts for the current Native `model.complete` cache resolver.

use super::model_complete_catalog::{
    canonical_model_id_for_provider_id, family_default_catalog_id, native_model_key,
    provider_model_id, static_family_model_id,
};
use super::{JsonPayload, LlmError, ReasoningEffort};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

pub use super::model_complete_catalog::{NativeModelFamily, NativeModelProvider};

/// Current Native `p5t` limit applied to `maxTokens` before dispatch.
pub(crate) const MODEL_COMPLETE_MAX_OUTPUT_TOKENS: u32 = 64_000;
/// Current Native `p5t` clamp for the per-call timeout passed to `setTimeout`.
pub(crate) const MODEL_COMPLETE_MAX_TIMEOUT_MS: u64 = 2_147_483_647;

/// A source fact whose absence has not been observed is different from a
/// source that was observed and had no value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeModelObservation<T> {
    Unobserved,
    Observed(T),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeModelGateIdentity {
    pub requested_selector: String,
    pub effective_selector: String,
}

impl NativeModelGateIdentity {
    pub fn unchanged(selector: impl Into<String>) -> Self {
        let selector = selector.into();
        Self {
            requested_selector: selector.clone(),
            effective_selector: selector,
        }
    }

    pub fn managed_deny_replacement(
        requested_selector: impl Into<String>,
        current_main_selector: impl Into<String>,
    ) -> Self {
        Self {
            requested_selector: requested_selector.into(),
            effective_selector: current_main_selector.into(),
        }
    }
}

/// Host facts for Native `tt()` and its merged `modelOverrides` lookup.
///
/// `Observed(None)` means the current merged settings were read and did not
/// contain `modelOverrides`; `Unobserved` means no settings snapshot was read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeModelSettingsSnapshot {
    pub current_main_selector: String,
    pub model_overrides: NativeModelObservation<Option<BTreeMap<String, String>>>,
}

impl NativeModelSettingsSnapshot {
    pub fn new(
        current_main_selector: impl Into<String>,
        model_overrides: NativeModelObservation<Option<BTreeMap<String, String>>>,
    ) -> Self {
        Self {
            current_main_selector: current_main_selector.into(),
            model_overrides,
        }
    }
}

/// The family rows Native has already selected from `v9t(servedCatalog)`.
/// These are catalog IDs, not provider model IDs or `modelKnowledge` data.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeServedCatalogFamilies {
    pub haiku: Option<String>,
    pub sonnet: Option<String>,
    pub opus: Option<String>,
}

impl NativeServedCatalogFamilies {
    fn family_id(&self, family: NativeModelFamily) -> Option<&str> {
        match family {
            NativeModelFamily::Haiku => self.haiku.as_deref(),
            NativeModelFamily::Sonnet => self.sonnet.as_deref(),
            NativeModelFamily::Opus => self.opus.as_deref(),
            NativeModelFamily::Fable | NativeModelFamily::Mythos => None,
        }
    }
}

/// SDK-owned observations used by Native family/default helpers.
///
/// `served_catalog_families` represents the output of Native's catalog-family
/// selection after its separate `modelKnowledge` eligibility lookup. It is not
/// a `modelKnowledge` map and must not be synthesized from route/display IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeModelRegistrySnapshot {
    pub served_catalog_families: NativeModelObservation<Option<NativeServedCatalogFamilies>>,
    /// Whether the selected first-party credential satisfies Native `zcs()`:
    /// no auth token and an API credential accepted as a first-party API key.
    /// This fact is consulted only by the small-fast Haiku branch.
    pub first_party_api_key_eligible: NativeModelObservation<bool>,
    /// Exact result of the current Native `h_()` lookup when the SDK has the
    /// source facts needed to resolve it. `None` is an observed absent result.
    pub small_fast_selector: NativeModelObservation<Option<String>>,
}

impl NativeModelRegistrySnapshot {
    pub fn unobserved() -> Self {
        Self {
            served_catalog_families: NativeModelObservation::Unobserved,
            first_party_api_key_eligible: NativeModelObservation::Unobserved,
            small_fast_selector: NativeModelObservation::Unobserved,
        }
    }
}

/// One text block accepted by the current Native model.complete prompt/system
/// surface. Text stays a JSON string carrier until the SDK performs the
/// documented prefix comparison and final provider-wire conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCompleteTextBlock {
    pub text: JsonPayload,
    pub cache: Option<bool>,
}

impl ModelCompleteTextBlock {
    pub fn new(text: JsonPayload, cache: Option<bool>) -> Result<Self, LlmError> {
        text.validate_root_string("model.complete text block text")?;
        Ok(Self { text, cache })
    }
}

/// Typed current Native model.complete input, lowered through the ordinary
/// SDK route and codec pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCompleteInput {
    pub model: NativeModelGateIdentity,
    pub prompt: JsonPayload,
    pub system: Option<JsonPayload>,
    pub prompt_blocks: Option<Vec<ModelCompleteTextBlock>>,
    pub system_blocks: Option<Vec<ModelCompleteTextBlock>>,
    pub max_tokens: Option<u32>,
    pub effort: Option<ReasoningEffort>,
    pub timeout_ms: Option<u64>,
    pub native_settings: NativeModelSettingsSnapshot,
}

impl ModelCompleteInput {
    pub fn new(
        model: NativeModelGateIdentity,
        prompt: JsonPayload,
        native_settings: NativeModelSettingsSnapshot,
    ) -> Self {
        Self {
            model,
            prompt,
            system: None,
            prompt_blocks: None,
            system_blocks: None,
            max_tokens: None,
            effort: None,
            timeout_ms: None,
            native_settings,
        }
    }

    pub fn validate(&self) -> Result<(), LlmError> {
        self.prompt.validate_root_string("model.complete prompt")?;
        if let Some(system) = &self.system {
            system.validate_root_string("model.complete system")?;
        }
        for (field, blocks) in [
            ("model.complete promptBlocks", self.prompt_blocks.as_deref()),
            ("model.complete systemBlocks", self.system_blocks.as_deref()),
        ] {
            if let Some(blocks) = blocks {
                for block in blocks {
                    block.text.validate_root_string(&format!("{field} text"))?;
                }
            }
        }
        if self.model.effective_selector.trim().is_empty() {
            return Err(LlmError::InvalidRequest {
                message: "model.complete effective model selector must be non-empty".into(),
            });
        }
        if self.native_settings.current_main_selector.trim().is_empty() {
            return Err(LlmError::InvalidRequest {
                message: "model.complete requires the current main model selector".into(),
            });
        }
        if self
            .max_tokens
            .is_some_and(|tokens| tokens == 0 || tokens > MODEL_COMPLETE_MAX_OUTPUT_TOKENS)
        {
            return Err(LlmError::InvalidRequest {
                message: "model.complete max_tokens must be between 1 and 64000".into(),
            });
        }
        if self.timeout_ms == Some(0) {
            return Err(LlmError::InvalidRequest {
                message: "model.complete timeout_ms must be a positive integer".into(),
            });
        }
        if self.effort.is_some_and(|effort| {
            !matches!(
                effort,
                ReasoningEffort::Low
                    | ReasoningEffort::Medium
                    | ReasoningEffort::High
                    | ReasoningEffort::XHigh
                    | ReasoningEffort::Max
            )
        }) {
            return Err(LlmError::InvalidRequest {
                message: "model.complete effort must be low, medium, high, xhigh, or max".into(),
            });
        }
        Ok(())
    }

    pub(crate) fn cache_gate(&self, registry: NativeModelRegistrySnapshot) -> NativeModelCacheGate {
        NativeModelCacheGate {
            model: self.model.clone(),
            settings: self.native_settings.clone(),
            registry,
            environment: NativeModelEnvironmentSnapshot::capture_process(),
        }
    }

    pub(crate) fn effective_timeout(&self) -> Option<Duration> {
        self.timeout_ms
            .map(|timeout_ms| Duration::from_millis(timeout_ms.min(MODEL_COMPLETE_MAX_TIMEOUT_MS)))
    }
}

/// Per-operation, non-wire SDK state for applying the Native cache gate to
/// each current route attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeModelCacheGate {
    pub model: NativeModelGateIdentity,
    pub settings: NativeModelSettingsSnapshot,
    pub registry: NativeModelRegistrySnapshot,
    pub environment: NativeModelEnvironmentSnapshot,
}

impl NativeModelCacheGate {
    pub fn decision(&self, provider: NativeModelProvider) -> NativeModelCacheDecision {
        model_complete_cache_decision(
            &self.model,
            provider,
            &self.settings,
            &self.registry,
            &self.environment,
        )
    }
}

impl NativeModelProvider {
    /// Map only explicit current provider/protocol identities. Unknown custom
    /// profiles remain outside the Native provider-specific cache resolver.
    pub fn for_profile(profile: &super::ProviderProfile) -> Option<Self> {
        use super::ProtocolFamily;
        match (profile.protocol, profile.provider_id.as_str()) {
            (ProtocolFamily::AnthropicMessages, "anthropic") => Some(Self::FirstParty),
            (ProtocolFamily::AnthropicMessages, "anthropicAws") => Some(Self::AnthropicAws),
            (ProtocolFamily::AnthropicMessages, "anthropicGoogleCloud") => {
                Some(Self::AnthropicGoogleCloud)
            }
            (ProtocolFamily::AnthropicMessages, "mantle") => Some(Self::Mantle),
            (ProtocolFamily::AnthropicMessages, "gateway") => Some(Self::Gateway),
            (ProtocolFamily::BedrockClaude, _) => Some(Self::Bedrock),
            (ProtocolFamily::VertexClaude, _) => Some(Self::Vertex),
            (ProtocolFamily::FoundryClaude, _) => Some(Self::Foundry),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeModelEnvironmentSnapshot {
    pub disable_all: bool,
    pub disable_haiku: bool,
    pub disable_sonnet: bool,
    pub disable_opus: bool,
    pub disable_fable: bool,
    pub disable_mythos: bool,
    pub default_haiku_model: Option<String>,
    pub default_sonnet_model: Option<String>,
    pub default_opus_model: Option<String>,
    pub default_fable_model: Option<String>,
    pub small_fast_model: Option<String>,
    pub assume_first_party_base_url: bool,
    pub anthropic_base_url: Option<String>,
}

impl NativeModelEnvironmentSnapshot {
    /// Capture process environment once at the SDK model.complete entry.
    pub fn capture_process() -> Self {
        Self {
            disable_all: native_bool_env("DISABLE_PROMPT_CACHING"),
            disable_haiku: native_bool_env("DISABLE_PROMPT_CACHING_HAIKU"),
            disable_sonnet: native_bool_env("DISABLE_PROMPT_CACHING_SONNET"),
            disable_opus: native_bool_env("DISABLE_PROMPT_CACHING_OPUS"),
            disable_fable: native_bool_env("DISABLE_PROMPT_CACHING_FABLE"),
            disable_mythos: native_bool_env("DISABLE_PROMPT_CACHING_MYTHOS"),
            default_haiku_model: string_env("ANTHROPIC_DEFAULT_HAIKU_MODEL"),
            default_sonnet_model: string_env("ANTHROPIC_DEFAULT_SONNET_MODEL"),
            default_opus_model: string_env("ANTHROPIC_DEFAULT_OPUS_MODEL"),
            default_fable_model: string_env("ANTHROPIC_DEFAULT_FABLE_MODEL"),
            small_fast_model: string_env("ANTHROPIC_SMALL_FAST_MODEL"),
            assume_first_party_base_url: native_bool_env(
                "_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL",
            ),
            anthropic_base_url: string_env("ANTHROPIC_BASE_URL"),
        }
    }

    fn native_first_party_base_url(&self) -> bool {
        if self.assume_first_party_base_url {
            return true;
        }
        match self.anthropic_base_url.as_deref() {
            None => true,
            Some(value) => url::Url::parse(value).ok().is_some_and(|url| {
                url.host_str() == Some("api.anthropic.com") && url.port().is_none()
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeModelCacheDecision {
    Enabled,
    Disabled,
    Undetermined,
}

/// Evaluate the source-backed subset of Native `qZ(kCe(input.model))`.
/// `Undetermined` stays distinct from enabled; the caller must not report
/// complete Native gate parity when a required SDK registry fact is missing.
pub fn model_complete_cache_decision(
    model: &NativeModelGateIdentity,
    provider: NativeModelProvider,
    host: &NativeModelSettingsSnapshot,
    registry: &NativeModelRegistrySnapshot,
    environment: &NativeModelEnvironmentSnapshot,
) -> NativeModelCacheDecision {
    if environment.disable_all {
        return NativeModelCacheDecision::Disabled;
    }

    let selector = model.effective_selector.as_str();
    let mut undetermined = false;

    if environment.disable_haiku {
        let haiku = family_model(
            NativeModelFamily::Haiku,
            provider,
            host,
            registry,
            environment.default_haiku_model.as_deref(),
            environment,
        );
        match haiku {
            NativeModelObservation::Observed(Some(haiku)) => {
                match same_native_identity(selector, &haiku, host) {
                    Some(true) => return NativeModelCacheDecision::Disabled,
                    Some(false) => {}
                    None => undetermined = true,
                }
            }
            NativeModelObservation::Observed(None) | NativeModelObservation::Unobserved => {
                undetermined = true;
            }
        }

        match small_fast_branch_enabled(provider, registry, environment) {
            NativeModelObservation::Observed(false) => {}
            NativeModelObservation::Observed(true) => match &registry.small_fast_selector {
                NativeModelObservation::Observed(Some(small_fast)) => {
                    if small_fast != &host.current_main_selector && selector == small_fast {
                        return NativeModelCacheDecision::Disabled;
                    }
                }
                NativeModelObservation::Observed(None) => {}
                NativeModelObservation::Unobserved => undetermined = true,
            },
            NativeModelObservation::Unobserved => undetermined = true,
        }
    }

    if environment.disable_sonnet {
        match family_model(
            NativeModelFamily::Sonnet,
            provider,
            host,
            registry,
            environment.default_sonnet_model.as_deref(),
            environment,
        ) {
            NativeModelObservation::Observed(Some(sonnet)) if selector == sonnet => {
                return NativeModelCacheDecision::Disabled;
            }
            NativeModelObservation::Observed(_) => {}
            NativeModelObservation::Unobserved => undetermined = true,
        }
    }
    if environment.disable_opus {
        match family_model(
            NativeModelFamily::Opus,
            provider,
            host,
            registry,
            environment.default_opus_model.as_deref(),
            environment,
        ) {
            NativeModelObservation::Observed(Some(opus)) if selector == opus => {
                return NativeModelCacheDecision::Disabled;
            }
            NativeModelObservation::Observed(_) => {}
            NativeModelObservation::Unobserved => undetermined = true,
        }
    }
    if environment.disable_fable
        && (selector.contains("claude-fable-")
            || environment
                .default_fable_model
                .as_deref()
                .is_some_and(|fable| same_model_spelling(selector, fable)))
    {
        return NativeModelCacheDecision::Disabled;
    }
    if environment.disable_mythos && selector.contains("claude-mythos-") {
        return NativeModelCacheDecision::Disabled;
    }

    if undetermined {
        NativeModelCacheDecision::Undetermined
    } else {
        NativeModelCacheDecision::Enabled
    }
}

/// Return the source-backed equivalent of Native `Be(selector, {identity:true})`.
/// The selector is the effective current request model, never a display label
/// or a provider route's wire-rewritten model name.
pub fn native_model_identity(
    selector: &str,
    model_overrides: &NativeModelObservation<Option<BTreeMap<String, String>>>,
) -> NativeModelObservation<String> {
    match model_overrides {
        NativeModelObservation::Unobserved => NativeModelObservation::Unobserved,
        NativeModelObservation::Observed(overrides) => {
            NativeModelObservation::Observed(canonical_identity(selector, overrides.as_ref()))
        }
    }
}

/// Evaluate Native `MEe(response,current)` model compatibility from the facts
/// that function actually consults. `served_models_by_requested_model` is the
/// current host request-latch observation (`kZt`), not the separate
/// modelKnowledge or served-catalog registry.
///
/// Native accepts a missing/non-string response model. For string responses,
/// compatibility can remain undetermined when the SDK has not observed the
/// current `modelOverrides` or request-latch set.
pub fn native_model_response_compatible(
    response_model: Option<&str>,
    current_selector: &str,
    model_overrides: &NativeModelObservation<Option<BTreeMap<String, String>>>,
    served_models_by_requested_model: &NativeModelObservation<BTreeMap<String, BTreeSet<String>>>,
) -> NativeModelObservation<bool> {
    let Some(response_model) = response_model else {
        return NativeModelObservation::Observed(true);
    };
    // Native accepts its synthetic assistant identity before the ordinary
    // selector or identity checks. The raw current selector is used by both
    // checks below; Hn normalization belongs only to the served-model latch.
    if response_model == "<synthetic>" || response_model == current_selector {
        return NativeModelObservation::Observed(true);
    }

    let identity_equal = match model_overrides {
        NativeModelObservation::Observed(overrides) => Some(
            canonical_identity(response_model, overrides.as_ref())
                == canonical_identity(current_selector, overrides.as_ref()),
        ),
        NativeModelObservation::Unobserved => None,
    };
    if identity_equal == Some(true) {
        return NativeModelObservation::Observed(true);
    }

    let served_match = match served_models_by_requested_model {
        NativeModelObservation::Observed(latches) => Some(
            latches
                .get(&native_served_models_latch_key(current_selector))
                .is_some_and(|models| models.contains(response_model)),
        ),
        NativeModelObservation::Unobserved => None,
    };
    match (identity_equal, served_match) {
        (_, Some(true)) => NativeModelObservation::Observed(true),
        (Some(false), Some(false)) => NativeModelObservation::Observed(false),
        _ => NativeModelObservation::Unobserved,
    }
}

/// Normalize the request-latch key with Native `Hn`: remove every `[1m]` and
/// `[2m]` marker, case-insensitively, before `servedModelsByRequestedModel`
/// lookup.
pub fn native_served_models_latch_key(requested_selector: &str) -> String {
    requested_selector
        .replace("[1m]", "")
        .replace("[1M]", "")
        .replace("[2m]", "")
        .replace("[2M]", "")
}

fn family_model(
    family: NativeModelFamily,
    provider: NativeModelProvider,
    host: &NativeModelSettingsSnapshot,
    registry: &NativeModelRegistrySnapshot,
    environment_override: Option<&str>,
    environment: &NativeModelEnvironmentSnapshot,
) -> NativeModelObservation<Option<String>> {
    if let Some(value) = environment_override {
        return NativeModelObservation::Observed(Some(value.to_owned()));
    }

    if provider == NativeModelProvider::FirstParty && environment.native_first_party_base_url() {
        match &registry.served_catalog_families {
            NativeModelObservation::Unobserved => return NativeModelObservation::Unobserved,
            NativeModelObservation::Observed(Some(catalog)) => {
                if let Some(catalog_id) = catalog.family_id(family) {
                    return resolve_served_catalog_id(catalog_id, provider, host);
                }
            }
            NativeModelObservation::Observed(None) => {}
        }
    }

    let catalog_id = family_default_catalog_id(family, provider);
    match &host.model_overrides {
        NativeModelObservation::Unobserved => NativeModelObservation::Unobserved,
        NativeModelObservation::Observed(overrides) => {
            let provider_id = overrides
                .as_ref()
                .and_then(|overrides| overrides.get(catalog_id))
                .cloned()
                .or_else(|| static_family_model_id(family, provider).map(str::to_owned));
            NativeModelObservation::Observed(provider_id)
        }
    }
}

fn resolve_served_catalog_id(
    catalog_id: &str,
    provider: NativeModelProvider,
    host: &NativeModelSettingsSnapshot,
) -> NativeModelObservation<Option<String>> {
    if native_model_key(catalog_id).is_none() {
        // Native YI returns the served catalog ID itself when it is not in the
        // compiled alias-key table.
        return NativeModelObservation::Observed(Some(catalog_id.to_owned()));
    }
    match &host.model_overrides {
        NativeModelObservation::Unobserved => NativeModelObservation::Unobserved,
        NativeModelObservation::Observed(overrides) => {
            let provider_id = overrides
                .as_ref()
                .and_then(|overrides| overrides.get(catalog_id))
                .cloned()
                .or_else(|| provider_model_id(catalog_id, provider).map(str::to_owned));
            NativeModelObservation::Observed(provider_id)
        }
    }
}

fn small_fast_branch_enabled(
    provider: NativeModelProvider,
    registry: &NativeModelRegistrySnapshot,
    environment: &NativeModelEnvironmentSnapshot,
) -> NativeModelObservation<bool> {
    if environment.small_fast_model.is_some() || environment.default_haiku_model.is_some() {
        return NativeModelObservation::Observed(true);
    }
    match provider {
        NativeModelProvider::AnthropicAws | NativeModelProvider::AnthropicGoogleCloud => {
            NativeModelObservation::Observed(true)
        }
        NativeModelProvider::FirstParty if environment.native_first_party_base_url() => {
            NativeModelObservation::Observed(true)
        }
        NativeModelProvider::FirstParty => registry.first_party_api_key_eligible.clone(),
        _ => NativeModelObservation::Observed(false),
    }
}

/// Compare the result of Native `Be(value, {identity:true})` for two selectors.
fn same_native_identity(
    left: &str,
    right: &str,
    host: &NativeModelSettingsSnapshot,
) -> Option<bool> {
    if left == right {
        return Some(true);
    }
    let (left, right) = match &host.model_overrides {
        NativeModelObservation::Unobserved => return None,
        NativeModelObservation::Observed(overrides) => (
            canonical_identity(left, overrides.as_ref()),
            canonical_identity(right, overrides.as_ref()),
        ),
    };
    Some(left == right)
}

fn canonical_identity(value: &str, overrides: Option<&BTreeMap<String, String>>) -> String {
    if let Some((catalog_id, _)) = overrides.and_then(|overrides| {
        overrides
            .iter()
            .find(|(_, provider_id)| provider_id.as_str() == value)
    }) {
        return canonical_name(catalog_id);
    }
    canonical_static_identity(value).unwrap_or_else(|| canonical_name(value))
}

fn canonical_static_identity(value: &str) -> Option<String> {
    let value = strip_context_suffix(value);
    canonical_model_id_for_provider_id(value).map(str::to_owned)
}

fn canonical_name(value: &str) -> String {
    strip_context_suffix(value).to_ascii_lowercase()
}

fn strip_context_suffix(value: &str) -> &str {
    value
        .strip_suffix("[1m]")
        .or_else(|| value.strip_suffix("[2m]"))
        .unwrap_or(value)
}

fn same_model_spelling(left: &str, right: &str) -> bool {
    canonical_name(left) == canonical_name(right)
}

fn native_bool_env(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn string_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

#[cfg(test)]
mod tests {
    use super::{
        model_complete_cache_decision, NativeModelCacheDecision, NativeModelEnvironmentSnapshot,
        NativeModelGateIdentity, NativeModelObservation, NativeModelProvider,
        NativeModelRegistrySnapshot, NativeModelSettingsSnapshot, NativeServedCatalogFamilies,
    };
    use super::{
        ModelCompleteInput, MODEL_COMPLETE_MAX_OUTPUT_TOKENS, MODEL_COMPLETE_MAX_TIMEOUT_MS,
    };
    use crate::protocol::{JsonPayload, ReasoningEffort};
    use std::{collections::BTreeMap, time::Duration};

    fn env() -> NativeModelEnvironmentSnapshot {
        NativeModelEnvironmentSnapshot {
            disable_all: false,
            disable_haiku: false,
            disable_sonnet: false,
            disable_opus: false,
            disable_fable: false,
            disable_mythos: false,
            default_haiku_model: None,
            default_sonnet_model: None,
            default_opus_model: None,
            default_fable_model: None,
            small_fast_model: None,
            assume_first_party_base_url: false,
            anthropic_base_url: None,
        }
    }

    fn host() -> NativeModelSettingsSnapshot {
        NativeModelSettingsSnapshot::new(
            "claude-sonnet-5-5",
            NativeModelObservation::Observed(None),
        )
    }

    fn registry() -> NativeModelRegistrySnapshot {
        NativeModelRegistrySnapshot {
            served_catalog_families: NativeModelObservation::Observed(None),
            first_party_api_key_eligible: NativeModelObservation::Observed(false),
            small_fast_selector: NativeModelObservation::Observed(None),
        }
    }

    #[test]
    fn known_absent_first_party_catalog_uses_source_pinned_static_default() {
        let mut env = env();
        env.disable_sonnet = true;
        let model = NativeModelGateIdentity::unchanged("claude-sonnet-5-5");
        assert_eq!(
            model_complete_cache_decision(
                &model,
                NativeModelProvider::FirstParty,
                &host(),
                &registry(),
                &env,
            ),
            NativeModelCacheDecision::Disabled
        );
    }

    #[test]
    fn missing_first_party_catalog_observation_stays_undetermined() {
        let mut env = env();
        env.disable_sonnet = true;
        let model = NativeModelGateIdentity::unchanged("claude-sonnet-5-5");
        let mut registry = registry();
        registry.served_catalog_families = NativeModelObservation::Unobserved;
        assert_eq!(
            model_complete_cache_decision(
                &model,
                NativeModelProvider::FirstParty,
                &host(),
                &registry,
                &env,
            ),
            NativeModelCacheDecision::Undetermined
        );
    }

    #[test]
    fn served_catalog_family_is_resolved_through_static_alias_and_settings_override() {
        let mut env = env();
        env.disable_sonnet = true;
        let model = NativeModelGateIdentity::unchanged("custom-sonnet");
        let mut host = host();
        host.model_overrides = NativeModelObservation::Observed(Some(BTreeMap::from([(
            "claude-sonnet-5-5".to_owned(),
            "custom-sonnet".to_owned(),
        )])));
        let mut registry = registry();
        registry.served_catalog_families =
            NativeModelObservation::Observed(Some(NativeServedCatalogFamilies {
                sonnet: Some("claude-sonnet-5-5".to_owned()),
                ..NativeServedCatalogFamilies::default()
            }));
        assert_eq!(
            model_complete_cache_decision(
                &model,
                NativeModelProvider::FirstParty,
                &host,
                &registry,
                &env,
            ),
            NativeModelCacheDecision::Disabled
        );
    }

    #[test]
    fn unobserved_model_overrides_do_not_become_known_absent() {
        let mut env = env();
        env.disable_sonnet = true;
        let model = NativeModelGateIdentity::unchanged("custom-sonnet");
        let host = NativeModelSettingsSnapshot::new(
            "claude-sonnet-5-5",
            NativeModelObservation::Unobserved,
        );
        assert_eq!(
            model_complete_cache_decision(
                &model,
                NativeModelProvider::Bedrock,
                &host,
                &registry(),
                &env,
            ),
            NativeModelCacheDecision::Undetermined
        );
        assert_eq!(
            super::native_model_identity("claude-sonnet-5-5", &host.model_overrides),
            NativeModelObservation::Unobserved
        );
    }

    #[test]
    fn global_disable_flag_is_decisive_without_registry_or_settings_facts() {
        let mut env = env();
        env.disable_all = true;
        let host = NativeModelSettingsSnapshot::new(
            "claude-sonnet-5-5",
            NativeModelObservation::Unobserved,
        );
        let model = NativeModelGateIdentity::unchanged("arbitrary-selector");
        assert_eq!(
            model_complete_cache_decision(
                &model,
                NativeModelProvider::FirstParty,
                &host,
                &NativeModelRegistrySnapshot::unobserved(),
                &env,
            ),
            NativeModelCacheDecision::Disabled
        );
    }

    #[test]
    fn fable_and_mythos_match_current_native_substring_tests() {
        let model = NativeModelGateIdentity::unchanged("us.anthropic.claude-fable-5-1-v1:0");
        let mut env = env();
        env.disable_fable = true;
        assert_eq!(
            model_complete_cache_decision(
                &model,
                NativeModelProvider::Bedrock,
                &host(),
                &registry(),
                &env,
            ),
            NativeModelCacheDecision::Disabled
        );
        env.disable_fable = false;
        env.disable_mythos = true;
        let model = NativeModelGateIdentity::unchanged("provider.claude-mythos-5-deployment");
        assert_eq!(
            model_complete_cache_decision(
                &model,
                NativeModelProvider::Bedrock,
                &host(),
                &registry(),
                &env,
            ),
            NativeModelCacheDecision::Disabled
        );
    }

    #[test]
    fn response_model_compatibility_uses_raw_selector_identity_and_normalized_latch_key() {
        use super::{
            native_model_response_compatible as compatible, NativeModelObservation as Obs,
        };
        use std::collections::BTreeSet;

        let no_overrides = Obs::Observed(None);
        let no_served_models = Obs::Observed(BTreeMap::new());
        assert_eq!(
            compatible(
                Some("claude-sonnet-4-5-20250929"),
                "claude-sonnet-4-5",
                &no_overrides,
                &no_served_models,
            ),
            Obs::Observed(true)
        );
        assert_eq!(
            compatible(None, "requested", &no_overrides, &no_served_models),
            Obs::Observed(true)
        );
        assert_eq!(
            compatible(
                Some("served-alias"),
                "requested",
                &no_overrides,
                &Obs::Observed(BTreeMap::from([(
                    "requested".to_owned(),
                    BTreeSet::from(["served-alias".to_owned()]),
                )])),
            ),
            Obs::Observed(true)
        );
        assert_eq!(
            compatible(
                Some("different"),
                "requested",
                &no_overrides,
                &no_served_models,
            ),
            Obs::Observed(false)
        );
        assert_eq!(
            compatible(
                Some("possibly-overridden"),
                "requested",
                &Obs::Unobserved,
                &no_served_models,
            ),
            Obs::Unobserved
        );
    }

    #[test]
    fn response_model_compatibility_keeps_context_tags_out_of_raw_identity_checks() {
        use super::{
            native_model_response_compatible as compatible, NativeModelObservation as Obs,
        };
        use std::collections::BTreeSet;

        let no_overrides = Obs::Observed(None);
        let no_served_models = Obs::Observed(BTreeMap::new());
        for (tag, normalized) in [("[1m]", "requestedalias"), ("[2m]", "requestedalias")] {
            let current = format!("requested{tag}alias");

            // Native Be(identity) sees the raw current selector. For these
            // interior markers, it does not turn the selector into the Hn
            // latch key, so a response matching only that key is rejected.
            assert_eq!(
                compatible(Some(normalized), &current, &no_overrides, &no_served_models,),
                Obs::Observed(false)
            );

            // Hn removes both marker spellings for the served-model lookup
            // only, while the response entry itself remains exact.
            let served_models = Obs::Observed(BTreeMap::from([(
                normalized.to_owned(),
                BTreeSet::from(["served-alias".to_owned()]),
            )]));
            assert_eq!(
                compatible(
                    Some("served-alias"),
                    &current,
                    &no_overrides,
                    &served_models,
                ),
                Obs::Observed(true)
            );

            // Native Be still treats a terminal context-size tag as the same
            // model identity, independently of Hn's latch normalization.
            let tagged = format!("claude-sonnet-4-5{tag}");
            assert_eq!(
                compatible(
                    Some("claude-sonnet-4-5"),
                    &tagged,
                    &no_overrides,
                    &no_served_models,
                ),
                Obs::Observed(true)
            );
        }
    }

    #[test]
    fn synthetic_response_model_is_always_compatible() {
        use super::{
            native_model_response_compatible as compatible, NativeModelObservation as Obs,
        };

        assert_eq!(
            compatible(
                Some("<synthetic>"),
                "any-current-selector",
                &Obs::Unobserved,
                &Obs::Unobserved
            ),
            Obs::Observed(true)
        );
    }

    #[test]
    fn first_party_base_url_matches_native_host_including_nondefault_port() {
        let mut environment = env();
        environment.anthropic_base_url = Some("https://api.anthropic.com:8443".into());
        assert!(!environment.native_first_party_base_url());
        environment.anthropic_base_url = Some("https://api.anthropic.com".into());
        assert!(environment.native_first_party_base_url());
        environment.assume_first_party_base_url = true;
        environment.anthropic_base_url = Some("https://proxy.invalid".into());
        assert!(environment.native_first_party_base_url());
    }

    #[test]
    fn model_complete_input_checks_native_positive_limits_and_effort_values() {
        let mut input = ModelCompleteInput::new(
            NativeModelGateIdentity::unchanged("claude-sonnet-5-5"),
            JsonPayload::string("prompt"),
            NativeModelSettingsSnapshot::new(
                "claude-sonnet-5-5",
                NativeModelObservation::Observed(None),
            ),
        );
        input.max_tokens = Some(0);
        assert!(input.validate().is_err());
        input.max_tokens = Some(MODEL_COMPLETE_MAX_OUTPUT_TOKENS + 1);
        assert!(input.validate().is_err());
        input.max_tokens = None;
        input.timeout_ms = Some(0);
        assert!(input.validate().is_err());
        input.timeout_ms = None;
        input.effort = Some(ReasoningEffort::None);
        assert!(input.validate().is_err());
        input.effort = Some(ReasoningEffort::Minimal);
        assert!(input.validate().is_err());
        input.effort = Some(ReasoningEffort::XHigh);
        assert!(input.validate().is_ok());
    }

    #[test]
    fn model_complete_timeout_uses_native_timer_ceiling() {
        let input = ModelCompleteInput {
            timeout_ms: Some(MODEL_COMPLETE_MAX_TIMEOUT_MS + 1),
            ..ModelCompleteInput::new(
                NativeModelGateIdentity::unchanged("claude-sonnet-5-5"),
                JsonPayload::string("prompt"),
                NativeModelSettingsSnapshot::new(
                    "claude-sonnet-5-5",
                    NativeModelObservation::Observed(None),
                ),
            )
        };
        assert_eq!(
            input.effective_timeout(),
            Some(Duration::from_millis(MODEL_COMPLETE_MAX_TIMEOUT_MS))
        );
    }
}
