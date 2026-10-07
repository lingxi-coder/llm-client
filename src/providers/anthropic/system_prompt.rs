//! Claude Code compatible system-prompt source projection and cache layout.
//!
//! Prompt source sections are assembled by the host. This module owns the
//! provider-specific source-vector marker and its transient conversion into
//! Anthropic system blocks and positional cache policy.

use crate::protocol::{
    CacheBreakpoint, CachePosition, CacheScope, CacheTtl, ChatRequest, ContentBlock,
    ProtocolFamily, ProviderProfile, SystemBlock,
};
use std::sync::Arc;

/// Native separator marker carried as a standalone source-vector element.
pub const DYNAMIC_BOUNDARY: &str = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__";

const SECTION_SEP: &str = "\n\n";
const FIRST_PARTY_API_HOST: &str = "api.anthropic.com";

/// Native's current GrowthBook allowlist feature and shipped fallback values.
pub const PROMPT_CACHE_1H_ALLOWLIST_FEATURE: &str = "tengu_prompt_cache_1h_config";
pub const PROMPT_CACHE_1H_ALLOWLIST_DEFAULT: &[&str] =
    &["repl_main_thread*", "sdk", "auto_mode", "memdir_relevance"];

/// One JavaScript string carried across the host boundary without losing
/// lone UTF-16 surrogates. `display_text` is the ECMAScript well-formed view
/// used by the provider wire request; `utf16_code_units` remains the source
/// value for Native array comparisons and source-vector joins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptText {
    display_text: String,
    utf16_code_units: Vec<u16>,
}

impl PromptText {
    /// Build an exact prompt string from UTF-16 units, using the same U+FFFD
    /// display conversion as Native's final provider request egress.
    #[must_use]
    pub fn from_utf16(utf16_code_units: Vec<u16>) -> Self {
        let display_text = String::from_utf16_lossy(&utf16_code_units);
        Self {
            display_text,
            utf16_code_units,
        }
    }

    /// Build a prompt string from a Rust string without changing its value.
    #[must_use]
    pub fn from_string(display_text: impl Into<String>) -> Self {
        let display_text = display_text.into();
        let utf16_code_units = display_text.encode_utf16().collect();
        Self {
            display_text,
            utf16_code_units,
        }
    }

    /// Display-safe UTF-8 view used by the provider's JSON body.
    #[must_use]
    pub fn display_text(&self) -> &str {
        &self.display_text
    }

    /// Exact JavaScript string units used for source-vector classification.
    #[must_use]
    pub fn utf16_code_units(&self) -> &[u16] {
        &self.utf16_code_units
    }

    /// Whether the JavaScript string has no code units.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.utf16_code_units.is_empty()
    }

    pub fn equals_ascii(&self, value: &str) -> bool {
        self.utf16_code_units
            .iter()
            .copied()
            .eq(value.encode_utf16())
    }

    fn starts_with_ascii(&self, value: &str) -> bool {
        self.utf16_code_units
            .starts_with(&value.encode_utf16().collect::<Vec<_>>())
    }
}

impl From<&str> for PromptText {
    fn from(value: &str) -> Self {
        Self::from_string(value)
    }
}

impl From<String> for PromptText {
    fn from(value: String) -> Self {
        Self::from_string(value)
    }
}

/// Current prompt input. Static source elements and per-query system context
/// remain separate until provider-specific source-vector egress. Custom prompt
/// parsing is performed by the Host at the field's Native input boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemPromptInput {
    /// Ordered static sections plus the separately recomputed system context.
    SourceVector {
        elements: Vec<PromptText>,
        dynamic_context: Option<PromptText>,
        /// Host-owned identity member; never inserted into prompt text.
        /// Native brand treatment applies only to an exact UTF-16 member match.
        host_branding_identity: Option<PromptText>,
    },
    /// Opaque custom text, such as Native `overrideSystemPrompt`, which feeds
    /// one source element to the provider projector without marker parsing.
    CustomPrompt { text: PromptText },
    /// A Native `customSystemPrompt` value that the Host already split with
    /// Native's current line-marker rule. The original is retained for display;
    /// every provider projector consumes `source_elements`.
    NativeCustomPrompt {
        text: PromptText,
        source_elements: Vec<PromptText>,
    },
}

impl SystemPromptInput {
    /// Create a source-vector request.
    ///
    /// `host_branding_identity` is non-content Host metadata. The Anthropic
    /// projector classifies a final vector member as this brand block only
    /// when its full UTF-16 sequence exactly matches the identity. `None`
    /// leaves only Native's built-in exact brand identities classified specially.
    #[must_use]
    pub fn source_vector(
        elements: Vec<PromptText>,
        dynamic_context: Option<PromptText>,
        host_branding_identity: Option<PromptText>,
    ) -> Self {
        Self::SourceVector {
            elements,
            dynamic_context,
            host_branding_identity,
        }
    }

    /// Create a one-element Native CustomPrompt request.
    #[must_use]
    pub fn custom_prompt(text: PromptText) -> Self {
        Self::CustomPrompt { text }
    }

    /// Create a current Native customSystemPrompt request after Host-side
    /// source-vector parsing. The SDK never splits the original string.
    #[must_use]
    pub fn native_custom_prompt(text: PromptText, source_elements: Vec<PromptText>) -> Self {
        Self::NativeCustomPrompt {
            text,
            source_elements,
        }
    }

    /// Visible combined text used by host previews and approximate counting.
    #[must_use]
    pub fn display_text(&self) -> String {
        let elements = match self {
            Self::SourceVector {
                elements,
                dynamic_context,
                ..
            } => {
                let mut elements = elements.clone();
                if let Some(dynamic_context) =
                    dynamic_context.as_ref().filter(|text| !text.is_empty())
                {
                    elements.push(dynamic_context.clone());
                }
                elements
            }
            Self::CustomPrompt { text } => vec![text.clone()],
            Self::NativeCustomPrompt { text, .. } => vec![text.clone()],
        };
        elements
            .iter()
            .map(PromptText::display_text)
            .collect::<Vec<_>>()
            .join(SECTION_SEP)
    }

    fn host_branding_identity(&self) -> Option<&PromptText> {
        match self {
            Self::SourceVector {
                host_branding_identity,
                ..
            } => host_branding_identity.as_ref(),
            Self::CustomPrompt { .. } | Self::NativeCustomPrompt { .. } => None,
        }
    }

    fn source_elements(&self) -> Vec<PromptText> {
        match self {
            Self::SourceVector {
                elements,
                dynamic_context,
                ..
            } => {
                let mut result = elements.clone();
                if let Some(dynamic_context) =
                    dynamic_context.as_ref().filter(|text| !text.is_empty())
                {
                    result.push(dynamic_context.clone());
                }
                result
            }
            Self::CustomPrompt { text } => vec![text.clone()],
            Self::NativeCustomPrompt {
                source_elements, ..
            } => source_elements.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Source-prompt caching group used by Native's system-prompt composer.
pub enum SectionScope {
    /// Included in the shared cacheable prefix when the global gate is open.
    Shared,
    /// Included after shared sections and kept in the session cache scope.
    Session,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One host-composed static prompt block and its cache grouping.
pub struct SourceSection {
    /// Source prompt bytes, before system-context additions.
    pub text: PromptText,
    /// Whether this prompt block is shared or session-specific.
    pub scope: SectionScope,
}

/// Host facts needed by the provider-owned marker gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatePolicy {
    /// Current host compliance-taint state (`false` in the Native-equivalent
    /// initialized-empty taint store until a host collector updates it).
    pub hipaa_tainted: bool,
    /// Current experimental-beta kill-switch state, sampled by the SDK.
    pub experimental_betas_disabled: bool,
    /// SDK-sampled Native `ni()` process endpoint gate.
    pub process_base_url_allowed: bool,
    /// SDK-sampled Native `_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL` override.
    pub assume_first_party_base_url: bool,
}

impl Default for GatePolicy {
    fn default() -> Self {
        Self {
            hipaa_tainted: false,
            experimental_betas_disabled: false,
            process_base_url_allowed: true,
            assume_first_party_base_url: false,
        }
    }
}

impl GatePolicy {
    /// Sample Native's process beta and endpoint gates in the SDK and combine
    /// them with the host's Native-equivalent compliance taint state.
    pub fn from_process(hipaa_tainted: bool) -> Self {
        let assume_first_party_base_url = env_truthy("_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL");
        let process_base_url_allowed = native_base_url_gate(
            std::env::var("ANTHROPIC_BASE_URL").ok().as_deref(),
            assume_first_party_base_url,
        );
        Self {
            hipaa_tainted,
            experimental_betas_disabled: env_truthy("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"),
            process_base_url_allowed,
            assume_first_party_base_url,
        }
    }
}

/// Per-agent prompt cache TTL carried from current agent frontmatter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentPromptCacheTtlOverride {
    /// Explicit Native `experimental.cacheTtl: "5m"`.
    FiveMinutes,
    /// Explicit Native `experimental.cacheTtl: "1h"`.
    OneHour,
}

/// Native request role used only to select the main or subagent cache-TTL
/// environment branch. A child tool is represented as a role, not by reusing
/// the unrelated COGS/query-source label carried on the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCacheQuerySource<'a> {
    /// A current Native `querySource` value.
    Named(&'a str),
    /// A Host `SubagentApiRequest`; it selects the Native non-main TTL branch.
    Subagent,
    /// A request for which the host has no Native query-source fact.
    Unspecified,
}

/// Retention selected by the host's current `promptCacheTtl` settings fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCacheTtl {
    /// Five-minute retention.
    FiveMinutes,
    /// One-hour retention.
    OneHour,
}

/// Main and non-main TTL settings loaded by the host for this request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptCacheTtlSettings {
    /// Current `promptCacheTtl` value.
    pub prompt_cache_ttl: Option<PromptCacheTtl>,
    /// Current `subagentPromptCacheTtl` value.
    pub subagent_prompt_cache_ttl: Option<PromptCacheTtl>,
}

/// Host's confidence that the active request uses Native's subscriber route.
/// Unknown custom routes stay distinct from a known non-subscriber route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCacheSubscriberState {
    /// The registered first-party OAuth route has a subscriber entitlement.
    Subscriber,
    /// The registered route or account state is known not to be subscribed.
    NotSubscriber,
    /// The Host cannot establish Native's account/provider gate.
    Unknown,
}

/// Request-local Host facts used by Native's subscriber TTL fallback. The
/// allowlist callback is invoked only after Native's force/env/settings/agent
/// gates and subscriber/overage checks have all fallen through.
#[derive(Clone)]
pub struct PromptCacheTtlInputs {
    /// Trusted subscriber and route/authentication classification.
    pub subscriber: PromptCacheSubscriberState,
    /// Native `ga().isUsingOverage === true`, from first-party response headers.
    pub is_using_overage: bool,
    /// Lazily reads the Host's feature snapshot; implementations should latch
    /// the first resolved list, matching Native `xgo`/`Pgo`.
    pub subscriber_allowlist: Arc<dyn Fn() -> Vec<String> + Send + Sync>,
}

impl std::fmt::Debug for PromptCacheTtlInputs {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PromptCacheTtlInputs")
            .field("subscriber", &self.subscriber)
            .field("is_using_overage", &self.is_using_overage)
            .field("subscriber_allowlist", &"<lazy>")
            .finish()
    }
}

impl Default for PromptCacheTtlInputs {
    fn default() -> Self {
        Self {
            subscriber: PromptCacheSubscriberState::Unknown,
            is_using_overage: false,
            subscriber_allowlist: Arc::new(|| {
                PROMPT_CACHE_1H_ALLOWLIST_DEFAULT
                    .iter()
                    .map(|source| (*source).to_owned())
                    .collect()
            }),
        }
    }
}

impl PromptCacheQuerySource<'_> {
    fn is_main(self) -> bool {
        match self {
            Self::Named(source) => {
                source.starts_with("repl_main_thread")
                    || matches!(source, "sdk" | "auto_mode" | "memdir_relevance")
            }
            Self::Subagent | Self::Unspecified => false,
        }
    }
}

/// App cache preferences passed to the SDK. Endpoint and experimental-beta
/// checks remain SDK-owned.
#[derive(Debug, Clone)]
pub struct CachePolicy {
    /// Current host compliance-taint state.
    pub hipaa_tainted: bool,
    /// Explicit per-agent Native TTL; `None` leaves environment policy in charge.
    pub agent_prompt_cache_ttl_override: Option<AgentPromptCacheTtlOverride>,
    /// Whether the Native source allowlist classifies this request as main.
    pub query_source_is_main: bool,
    /// Exact current `querySource`, when the Host has one. Retained separately
    /// from the main/subagent branch because Native's subscriber allowlist
    /// matches the actual source value.
    pub query_source: Option<String>,
    /// Native main and subagent settings after the host's settings merge.
    pub prompt_cache_ttl_settings: PromptCacheTtlSettings,
    /// Subscriber, overage and lazy allowlist facts for Native's one-hour TTL.
    pub prompt_cache_ttl_inputs: PromptCacheTtlInputs,
    /// Whether experimental-beta features are disabled.
    pub experimental_betas_disabled: bool,
    /// SDK-sampled Native `ni()` process endpoint gate.
    pub process_base_url_allowed: bool,
    /// SDK-sampled Native `_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL` override.
    pub assume_first_party_base_url: bool,
    /// Native `skipGlobalCacheForSystemPrompt`, computed from the active
    /// query's registered tool facts before provider tool JSON is flattened.
    pub skip_global_cache_for_system_prompt: bool,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            hipaa_tainted: false,
            agent_prompt_cache_ttl_override: None,
            query_source_is_main: false,
            query_source: None,
            prompt_cache_ttl_settings: PromptCacheTtlSettings::default(),
            prompt_cache_ttl_inputs: PromptCacheTtlInputs::default(),
            experimental_betas_disabled: false,
            process_base_url_allowed: true,
            assume_first_party_base_url: false,
            skip_global_cache_for_system_prompt: false,
        }
    }
}

impl CachePolicy {
    /// Sample SDK process policy while accepting the host-owned compliance
    /// state. Native's global source boundary has no subscription or local
    /// opt-in condition.
    pub fn from_process(
        hipaa_tainted: bool,
        agent_prompt_cache_ttl_override: Option<AgentPromptCacheTtlOverride>,
        skip_global_cache_for_system_prompt: bool,
        query_source: PromptCacheQuerySource<'_>,
        prompt_cache_ttl_settings: PromptCacheTtlSettings,
        prompt_cache_ttl_inputs: PromptCacheTtlInputs,
    ) -> Self {
        let assume_first_party_base_url = env_truthy("_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL");
        let process_base_url_allowed = native_base_url_gate(
            std::env::var("ANTHROPIC_BASE_URL").ok().as_deref(),
            assume_first_party_base_url,
        );
        let query_source_value = match query_source {
            PromptCacheQuerySource::Named(source) => Some(source.to_owned()),
            PromptCacheQuerySource::Subagent | PromptCacheQuerySource::Unspecified => None,
        };
        Self {
            hipaa_tainted,
            agent_prompt_cache_ttl_override,
            query_source_is_main: query_source.is_main(),
            query_source: query_source_value,
            prompt_cache_ttl_settings,
            prompt_cache_ttl_inputs,
            experimental_betas_disabled: env_truthy("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"),
            process_base_url_allowed,
            assume_first_party_base_url,
            skip_global_cache_for_system_prompt,
        }
    }

    /// Reduce cache settings to the inputs that govern global source boundaries.
    pub fn gate(&self) -> GatePolicy {
        GatePolicy {
            hipaa_tainted: self.hipaa_tainted,
            experimental_betas_disabled: self.experimental_betas_disabled,
            process_base_url_allowed: self.process_base_url_allowed,
            assume_first_party_base_url: self.assume_first_party_base_url,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PromptCacheTtlEnvironment {
    force_five_minutes: bool,
    main_ttl: Option<PromptCacheTtl>,
    subagent_ttl: Option<PromptCacheTtl>,
    enable_one_hour: bool,
    enable_bedrock_one_hour: bool,
}

impl PromptCacheTtlEnvironment {
    fn from_process() -> Self {
        Self {
            force_five_minutes: env_truthy("FORCE_PROMPT_CACHING_5M"),
            main_ttl: parse_prompt_cache_ttl_env("CLAUDE_CODE_PROMPT_CACHE_TTL"),
            subagent_ttl: parse_prompt_cache_ttl_env("CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL"),
            enable_one_hour: env_truthy("ENABLE_PROMPT_CACHING_1H"),
            enable_bedrock_one_hour: env_truthy("ENABLE_PROMPT_CACHING_1H_BEDROCK"),
        }
    }
}

fn parse_prompt_cache_ttl_env(name: &str) -> Option<PromptCacheTtl> {
    match std::env::var(name).ok().as_deref() {
        Some("5m") => Some(PromptCacheTtl::FiveMinutes),
        Some("1h") => Some(PromptCacheTtl::OneHour),
        _ => None,
    }
}

fn resolve_prompt_cache_ttl(
    protocol: ProtocolFamily,
    policy: &CachePolicy,
    env: PromptCacheTtlEnvironment,
) -> CacheTtl {
    if env.force_five_minutes {
        return CacheTtl::FiveMinutes;
    }

    let source_env = if policy.query_source_is_main {
        env.main_ttl
    } else {
        env.subagent_ttl
    };
    if let Some(ttl) = source_env {
        return match ttl {
            PromptCacheTtl::FiveMinutes => CacheTtl::FiveMinutes,
            PromptCacheTtl::OneHour => CacheTtl::OneHour,
        };
    }

    let source_setting = if policy.query_source_is_main {
        policy.prompt_cache_ttl_settings.prompt_cache_ttl
    } else {
        policy.prompt_cache_ttl_settings.subagent_prompt_cache_ttl
    };
    if let Some(ttl) = source_setting {
        return match ttl {
            PromptCacheTtl::FiveMinutes => CacheTtl::FiveMinutes,
            PromptCacheTtl::OneHour => CacheTtl::OneHour,
        };
    }

    match policy.agent_prompt_cache_ttl_override {
        Some(AgentPromptCacheTtlOverride::FiveMinutes) => CacheTtl::FiveMinutes,
        Some(AgentPromptCacheTtlOverride::OneHour)
            if !(policy.prompt_cache_ttl_inputs.subscriber
                == PromptCacheSubscriberState::Subscriber
                && policy.prompt_cache_ttl_inputs.is_using_overage) =>
        {
            CacheTtl::OneHour
        }
        _ if env.enable_one_hour
            || protocol == ProtocolFamily::BedrockClaude && env.enable_bedrock_one_hour =>
        {
            CacheTtl::OneHour
        }
        _ => {
            if policy.prompt_cache_ttl_inputs.subscriber != PromptCacheSubscriberState::Subscriber
                || policy.prompt_cache_ttl_inputs.is_using_overage
            {
                return CacheTtl::FiveMinutes;
            }
            let Some(source) = policy.query_source.as_deref() else {
                return CacheTtl::FiveMinutes;
            };
            let allowlist = (policy.prompt_cache_ttl_inputs.subscriber_allowlist)();
            let allowed = allowlist.iter().any(|entry| {
                entry
                    .strip_suffix('*')
                    .map_or(entry.as_str() == source, |prefix| {
                        source.starts_with(prefix)
                    })
            });
            if allowed {
                CacheTtl::OneHour
            } else {
                CacheTtl::FiveMinutes
            }
        }
    }
}

/// One encoded block plus the cache breakpoint to apply at that position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedSystemBlock {
    /// Provider-independent block content passed into the protocol request.
    pub block: SystemBlock,
    /// Cache checkpoint to attach at this system-block position.
    pub cache_breakpoint: Option<CacheBreakpoint>,
}

/// Native `UDn` projection: remove whole sections that equal the marker,
/// group shared blocks before session blocks, and insert a standalone marker
/// only when Native's provider/taint/beta/base-URL gate is open and a shared
/// block exists. Native has no subscriber or local opt-in condition here.
pub fn snapshot_source_vector(
    sections: &[SourceSection],
    profile: Option<&ProviderProfile>,
    policy: GatePolicy,
) -> Vec<PromptText> {
    let mut shared = Vec::new();
    let mut session = Vec::new();
    for section in sections {
        if section.text.equals_ascii(DYNAMIC_BOUNDARY) {
            continue;
        }
        match section.scope {
            SectionScope::Shared => shared.push(section.text.clone()),
            SectionScope::Session => session.push(section.text.clone()),
        }
    }

    let mut result = shared;
    if global_scope_enabled(profile, policy) && !result.is_empty() {
        result.push(PromptText::from_string(DYNAMIC_BOUNDARY));
    }
    result.extend(session);
    result
}

/// Return whether the Native source boundary can carry global cache scope.
///
/// The selected profile is required. Boot/no-route callers therefore leave
/// the marker out rather than assuming Anthropic. Native gates this on clear
/// compliance taints, enabled experimental betas, the configured process
/// base URL, and the selected provider; subscription and local opt-in state do
/// not participate.
pub fn global_scope_enabled(profile: Option<&ProviderProfile>, policy: GatePolicy) -> bool {
    if policy.hipaa_tainted || policy.experimental_betas_disabled {
        return false;
    }
    let Some(profile) = profile else {
        return false;
    };
    native_profile_allows_global_scope(
        profile,
        policy.process_base_url_allowed,
        policy.assume_first_party_base_url,
    )
}

fn native_profile_allows_global_scope(
    profile: &ProviderProfile,
    process_base_url_allowed: bool,
    assume_first_party_base_url: bool,
) -> bool {
    if profile.protocol != ProtocolFamily::AnthropicMessages || !process_base_url_allowed {
        return false;
    }
    match profile.provider_id.as_str() {
        // Native `He() === "firstParty"`: the selected first-party API
        // endpoint and process-level ANTHROPIC_BASE_URL gate must both pass.
        "anthropic" => official_anthropic_base_url_with_override(
            &profile.base_url,
            assume_first_party_base_url,
        ),
        // Native `He() === "anthropicAws"`: this route is allowed by the
        // process-level gate but may use its own AWS regional endpoint.
        "anthropicAws" => true,
        _ => false,
    }
}

/// Project the current source-vector shape into provider system blocks. This
/// is the sole system prompt egress conversion used by the request builder.
pub fn project_system_prompt(
    system: &SystemPromptInput,
    profile: Option<&ProviderProfile>,
    model: &str,
    protocol: ProtocolFamily,
    policy: CachePolicy,
) -> Vec<EncodedSystemBlock> {
    let cache_enabled = prompt_caching_enabled(model, protocol);
    let global_scope = global_scope_enabled(
        profile,
        GatePolicy {
            hipaa_tainted: policy.hipaa_tainted,
            experimental_betas_disabled: policy.experimental_betas_disabled,
            process_base_url_allowed: policy.process_base_url_allowed,
            assume_first_party_base_url: policy.assume_first_party_base_url,
        },
    );
    let ttl =
        resolve_prompt_cache_ttl(protocol, &policy, PromptCacheTtlEnvironment::from_process());
    let one_hour = ttl == CacheTtl::OneHour;
    let elements = system.source_elements();
    if !matches!(
        protocol,
        ProtocolFamily::AnthropicMessages
            | ProtocolFamily::VertexClaude
            | ProtocolFamily::BedrockClaude
            | ProtocolFamily::FoundryClaude
    ) {
        let elements = match system {
            SystemPromptInput::SourceVector { .. }
            | SystemPromptInput::NativeCustomPrompt { .. } => elements
                .into_iter()
                .filter(|element| !element.equals_ascii(DYNAMIC_BOUNDARY))
                .collect::<Vec<_>>(),
            SystemPromptInput::CustomPrompt { .. } => elements,
        };
        let text = join_source_units(
            elements
                .iter()
                .map(|element| element.utf16_code_units.as_slice()),
        );
        return if text.is_empty() {
            Vec::new()
        } else {
            vec![encoded_block(text, None, None, 0)]
        };
    }

    project_native_source_vector(
        &elements,
        cache_enabled,
        global_scope,
        policy.skip_global_cache_for_system_prompt,
        one_hour,
        system.host_branding_identity(),
    )
}

/// Default prompt-cache setting, using the same model-family environment gates
/// as the host had before this provider conversion moved into the SDK.
pub fn prompt_caching_enabled(model: &str, protocol: ProtocolFamily) -> bool {
    if !matches!(
        protocol,
        ProtocolFamily::AnthropicMessages
            | ProtocolFamily::VertexClaude
            | ProtocolFamily::BedrockClaude
            | ProtocolFamily::FoundryClaude
    ) {
        return false;
    }
    if env_truthy("DISABLE_PROMPT_CACHING") {
        return false;
    }
    let model = model.to_ascii_lowercase();
    if model.contains("haiku") && env_truthy("DISABLE_PROMPT_CACHING_HAIKU") {
        return false;
    }
    if model.contains("sonnet") && env_truthy("DISABLE_PROMPT_CACHING_SONNET") {
        return false;
    }
    if model.contains("opus") && env_truthy("DISABLE_PROMPT_CACHING_OPUS") {
        return false;
    }
    if model.contains("fable") && env_truthy("DISABLE_PROMPT_CACHING_FABLE") {
        return false;
    }
    if model.contains("mythos") && env_truthy("DISABLE_PROMPT_CACHING_MYTHOS") {
        return false;
    }
    true
}

/// Attach the default last-content breakpoint from the shared request cache
/// pipeline. Existing state at the same position is replaced by the default
/// five-minute marker, matching the host's former projection.
pub fn apply_last_message_breakpoint(request: &mut ChatRequest, enabled: bool) {
    if !enabled {
        return;
    }
    let Some(message_index) = request.messages.len().checked_sub(1) else {
        return;
    };
    let Some(block_index) = request.messages[message_index]
        .content
        .iter()
        .enumerate()
        .rev()
        .find(|(_, block)| {
            !matches!(
                block,
                ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. }
            )
        })
        .map(|(block_index, _)| block_index)
    else {
        return;
    };
    if !matches!(
        &request.messages[message_index].content[block_index],
        ContentBlock::Text { .. }
            | ContentBlock::TextJsUtf16 { .. }
            | ContentBlock::ToolResult { .. }
    ) {
        return;
    }
    let position = CachePosition::Message {
        index: message_index,
        block: block_index,
    };
    let breakpoint = CacheBreakpoint {
        position,
        scope: None,
        ttl: CacheTtl::FiveMinutes,
    };
    if let Some(existing) = request
        .prompt_cache
        .breakpoints
        .iter_mut()
        .find(|existing| existing.position == position)
    {
        *existing = breakpoint;
    } else {
        request.prompt_cache.breakpoints.push(breakpoint);
    }
}

fn project_native_source_vector(
    elements: &[PromptText],
    enabled: bool,
    global_scope: bool,
    skip_global_cache_for_system_prompt: bool,
    one_hour: bool,
    host_branding_identity: Option<&PromptText>,
) -> Vec<EncodedSystemBlock> {
    let org_ttl = if one_hour {
        CacheTtl::OneHour
    } else {
        CacheTtl::FiveMinutes
    };
    let boundary = elements
        .iter()
        .position(|element| element.equals_ascii(DYNAMIC_BOUNDARY));
    if global_scope && skip_global_cache_for_system_prompt && boundary.is_none() {
        let (billing_header, native_brand, reporting, ordinary) =
            classify_native_sections(elements, host_branding_identity);
        let include_reporting = billing_header.is_some() && native_brand.is_some();
        let mut blocks = Vec::new();
        push_uncached(&mut blocks, billing_header);
        push_cached(&mut blocks, native_brand, enabled, org_ttl, None);
        if include_reporting {
            push_uncached(&mut blocks, reporting);
        }
        let ordinary = join_source_units(ordinary.iter().map(Vec::as_slice));
        if !ordinary.is_empty() {
            push_cached(&mut blocks, Some(ordinary), enabled, org_ttl, None);
        }
        return blocks;
    }

    if global_scope {
        if let Some(boundary) = boundary {
            let (billing_header, native_brand, reporting, before_and_after) =
                classify_native_sections_with_boundary(elements, boundary, host_branding_identity);
            let include_reporting = billing_header.is_some() && native_brand.is_some();
            let mut blocks = Vec::new();
            push_uncached(&mut blocks, billing_header);
            push_uncached(&mut blocks, native_brand);
            if include_reporting {
                push_uncached(&mut blocks, reporting);
            }
            let static_text = join_source_units(before_and_after.0.iter().map(Vec::as_slice));
            if !static_text.is_empty() {
                push_cached(
                    &mut blocks,
                    Some(static_text),
                    enabled,
                    org_ttl,
                    Some(CacheScope::Global),
                );
            }
            let dynamic_text = join_source_units(before_and_after.1.iter().map(Vec::as_slice));
            if !dynamic_text.is_empty() {
                push_cached(&mut blocks, Some(dynamic_text), enabled, org_ttl, None);
            }
            return blocks;
        }
    }

    let (billing_header, native_brand, reporting, ordinary) =
        classify_native_sections(elements, host_branding_identity);
    let include_reporting = billing_header.is_some() && native_brand.is_some();
    let mut blocks = Vec::new();
    push_uncached(&mut blocks, billing_header);
    push_cached(&mut blocks, native_brand, enabled, org_ttl, None);
    if include_reporting {
        push_uncached(&mut blocks, reporting);
    }
    let ordinary = join_source_units(ordinary.iter().map(Vec::as_slice));
    if !ordinary.is_empty() {
        push_cached(&mut blocks, Some(ordinary), enabled, org_ttl, None);
    }
    blocks
}

const NATIVE_BILLING_HEADER: &str = "x-anthropic-billing-header:";
const NATIVE_BRAND_SECTIONS: [&str; 3] = [
    "You are Claude Code, Anthropic's official CLI for Claude.",
    "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK.",
    "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
];
const NATIVE_REPORTING_SECTION: &str = "# Reporting outcomes\n\nReport what actually happened, not what you intended. When you say something is done, sent, saved, fixed, or verified, that claim must rest on a result you observed in this session — tool output, the file as it now reads, the page as it now loads — not on what the step should have produced. If you did not check, say you did not check. If any step failed, was skipped, or came back different from what you expected, say so in the first sentence of your report, before anything else, even when the rest of the work succeeded. Never quietly work around a failure in a way that makes it look resolved; a problem the user can see is recoverable, one your summary hides is not. When you stop before the task is complete, your first line says so plainly and names what is left. Do not describe partial work as done, and do not let a summary read as more certain than the evidence behind it.";

fn is_native_brand_section(
    element: &PromptText,
    host_branding_identity: Option<&PromptText>,
) -> bool {
    NATIVE_BRAND_SECTIONS
        .iter()
        .any(|value| element.equals_ascii(value))
        || host_branding_identity
            .is_some_and(|identity| element.utf16_code_units() == identity.utf16_code_units())
}

fn classify_native_sections(
    elements: &[PromptText],
    host_branding_identity: Option<&PromptText>,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Vec<Vec<u16>>,
) {
    let mut billing_header = None;
    let mut native_brand = None;
    let mut reporting = None;
    let mut ordinary = Vec::new();
    for element in elements {
        if element.is_empty() || element.equals_ascii(DYNAMIC_BOUNDARY) {
            continue;
        }
        if element.starts_with_ascii(NATIVE_BILLING_HEADER) {
            billing_header = Some(element.display_text.clone());
        } else if is_native_brand_section(element, host_branding_identity) {
            native_brand = Some(element.display_text.clone());
        } else if element.equals_ascii(NATIVE_REPORTING_SECTION) {
            reporting = Some(element.display_text.clone());
        } else {
            ordinary.push(element.utf16_code_units.clone());
        }
    }
    (billing_header, native_brand, reporting, ordinary)
}

fn classify_native_sections_with_boundary(
    elements: &[PromptText],
    boundary: usize,
    host_branding_identity: Option<&PromptText>,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    (Vec<Vec<u16>>, Vec<Vec<u16>>),
) {
    let mut billing_header = None;
    let mut native_brand = None;
    let mut reporting = None;
    let mut before = Vec::new();
    let mut after = Vec::new();
    for (index, element) in elements.iter().enumerate() {
        if element.is_empty() || element.equals_ascii(DYNAMIC_BOUNDARY) {
            continue;
        }
        if element.starts_with_ascii(NATIVE_BILLING_HEADER) {
            billing_header = Some(element.display_text.clone());
        } else if is_native_brand_section(element, host_branding_identity) {
            native_brand = Some(element.display_text.clone());
        } else if element.equals_ascii(NATIVE_REPORTING_SECTION) {
            reporting = Some(element.display_text.clone());
        } else if index < boundary {
            before.push(element.utf16_code_units.clone());
        } else {
            after.push(element.utf16_code_units.clone());
        }
    }
    (billing_header, native_brand, reporting, (before, after))
}

fn join_source_units<'a>(elements: impl IntoIterator<Item = &'a [u16]>) -> String {
    let mut units = Vec::new();
    let mut first = true;
    for element in elements.into_iter().filter(|element| !element.is_empty()) {
        if !first {
            units.extend(SECTION_SEP.encode_utf16());
        }
        units.extend_from_slice(element);
        first = false;
    }
    String::from_utf16_lossy(&units)
}

fn push_uncached(blocks: &mut Vec<EncodedSystemBlock>, text: Option<String>) {
    if let Some(text) = text.filter(|text| !text.is_empty()) {
        let index = blocks.len();
        blocks.push(encoded_block(text, None, None, index));
    }
}

fn push_cached(
    blocks: &mut Vec<EncodedSystemBlock>,
    text: Option<String>,
    enabled: bool,
    ttl: CacheTtl,
    scope: Option<CacheScope>,
) {
    if let Some(text) = text.filter(|text| !text.is_empty()) {
        let index = blocks.len();
        blocks.push(encoded_block(text, enabled.then_some(ttl), scope, index));
    }
}

fn encoded_block(
    text: String,
    ttl: Option<CacheTtl>,
    scope: Option<CacheScope>,
    index: usize,
) -> EncodedSystemBlock {
    let cache_breakpoint = ttl.map(|ttl| CacheBreakpoint {
        position: CachePosition::System { index },
        scope,
        ttl,
    });
    EncodedSystemBlock {
        block: SystemBlock { text },
        cache_breakpoint,
    }
}

fn official_anthropic_base_url_with_override(value: &str, assume_first_party: bool) -> bool {
    if assume_first_party {
        return true;
    }
    url::Url::parse(value).ok().is_some_and(|url| {
        url.host_str().is_some_and(|host| match url.port() {
            Some(port) => format!("{host}:{port}") == FIRST_PARTY_API_HOST,
            None => host == FIRST_PARTY_API_HOST,
        })
    })
}

/// Native `ni()` checks the configured process base URL independently of the
/// resolved provider profile. An absent/empty value is allowed; a present URL
/// must have the exact official API host unless the explicit Native override
/// is set.
fn native_base_url_gate(process_base_url: Option<&str>, assume_first_party: bool) -> bool {
    assume_first_party
        || process_base_url.is_none_or(str::is_empty)
        || process_base_url.is_some_and(|value| {
            official_anthropic_base_url_with_override(value, assume_first_party)
        })
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    static GATE_ENV_LOCK: Mutex<()> = Mutex::new(());

    struct GateEnvRestore {
        anthropic_base_url: Option<String>,
        assume_first_party: Option<String>,
        lingxi_global_cache_scope: Option<String>,
        disable_experimental_betas: Option<String>,
        disable_prompt_caching: Option<String>,
        disable_prompt_caching_haiku: Option<String>,
        disable_prompt_caching_sonnet: Option<String>,
        disable_prompt_caching_opus: Option<String>,
        disable_prompt_caching_fable: Option<String>,
        disable_prompt_caching_mythos: Option<String>,
        force_prompt_caching_5m: Option<String>,
        main_prompt_cache_ttl: Option<String>,
        subagent_prompt_cache_ttl: Option<String>,
        enable_prompt_caching_1h: Option<String>,
        enable_prompt_caching_1h_bedrock: Option<String>,
    }

    impl GateEnvRestore {
        fn clear() -> Self {
            let restore = Self {
                anthropic_base_url: std::env::var("ANTHROPIC_BASE_URL").ok(),
                assume_first_party: std::env::var("_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL").ok(),
                lingxi_global_cache_scope: std::env::var("LINGXI_GLOBAL_CACHE_SCOPE").ok(),
                disable_experimental_betas: std::env::var("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS")
                    .ok(),
                disable_prompt_caching: std::env::var("DISABLE_PROMPT_CACHING").ok(),
                disable_prompt_caching_haiku: std::env::var("DISABLE_PROMPT_CACHING_HAIKU").ok(),
                disable_prompt_caching_sonnet: std::env::var("DISABLE_PROMPT_CACHING_SONNET").ok(),
                disable_prompt_caching_opus: std::env::var("DISABLE_PROMPT_CACHING_OPUS").ok(),
                disable_prompt_caching_fable: std::env::var("DISABLE_PROMPT_CACHING_FABLE").ok(),
                disable_prompt_caching_mythos: std::env::var("DISABLE_PROMPT_CACHING_MYTHOS").ok(),
                force_prompt_caching_5m: std::env::var("FORCE_PROMPT_CACHING_5M").ok(),
                main_prompt_cache_ttl: std::env::var("CLAUDE_CODE_PROMPT_CACHE_TTL").ok(),
                subagent_prompt_cache_ttl: std::env::var("CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL")
                    .ok(),
                enable_prompt_caching_1h: std::env::var("ENABLE_PROMPT_CACHING_1H").ok(),
                enable_prompt_caching_1h_bedrock: std::env::var("ENABLE_PROMPT_CACHING_1H_BEDROCK")
                    .ok(),
            };
            std::env::remove_var("ANTHROPIC_BASE_URL");
            std::env::remove_var("_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL");
            std::env::remove_var("LINGXI_GLOBAL_CACHE_SCOPE");
            std::env::remove_var("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS");
            std::env::remove_var("DISABLE_PROMPT_CACHING");
            std::env::remove_var("DISABLE_PROMPT_CACHING_HAIKU");
            std::env::remove_var("DISABLE_PROMPT_CACHING_SONNET");
            std::env::remove_var("DISABLE_PROMPT_CACHING_OPUS");
            std::env::remove_var("DISABLE_PROMPT_CACHING_FABLE");
            std::env::remove_var("DISABLE_PROMPT_CACHING_MYTHOS");
            std::env::remove_var("FORCE_PROMPT_CACHING_5M");
            std::env::remove_var("CLAUDE_CODE_PROMPT_CACHE_TTL");
            std::env::remove_var("CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL");
            std::env::remove_var("ENABLE_PROMPT_CACHING_1H");
            std::env::remove_var("ENABLE_PROMPT_CACHING_1H_BEDROCK");
            restore
        }
    }

    impl Drop for GateEnvRestore {
        fn drop(&mut self) {
            match &self.anthropic_base_url {
                Some(value) => std::env::set_var("ANTHROPIC_BASE_URL", value),
                None => std::env::remove_var("ANTHROPIC_BASE_URL"),
            }
            match &self.assume_first_party {
                Some(value) => std::env::set_var("_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL", value),
                None => std::env::remove_var("_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL"),
            }
            match &self.lingxi_global_cache_scope {
                Some(value) => std::env::set_var("LINGXI_GLOBAL_CACHE_SCOPE", value),
                None => std::env::remove_var("LINGXI_GLOBAL_CACHE_SCOPE"),
            }
            match &self.disable_experimental_betas {
                Some(value) => std::env::set_var("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", value),
                None => std::env::remove_var("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"),
            }
            match &self.disable_prompt_caching {
                Some(value) => std::env::set_var("DISABLE_PROMPT_CACHING", value),
                None => std::env::remove_var("DISABLE_PROMPT_CACHING"),
            }
            match &self.disable_prompt_caching_haiku {
                Some(value) => std::env::set_var("DISABLE_PROMPT_CACHING_HAIKU", value),
                None => std::env::remove_var("DISABLE_PROMPT_CACHING_HAIKU"),
            }
            match &self.disable_prompt_caching_sonnet {
                Some(value) => std::env::set_var("DISABLE_PROMPT_CACHING_SONNET", value),
                None => std::env::remove_var("DISABLE_PROMPT_CACHING_SONNET"),
            }
            match &self.disable_prompt_caching_opus {
                Some(value) => std::env::set_var("DISABLE_PROMPT_CACHING_OPUS", value),
                None => std::env::remove_var("DISABLE_PROMPT_CACHING_OPUS"),
            }
            match &self.disable_prompt_caching_fable {
                Some(value) => std::env::set_var("DISABLE_PROMPT_CACHING_FABLE", value),
                None => std::env::remove_var("DISABLE_PROMPT_CACHING_FABLE"),
            }
            match &self.disable_prompt_caching_mythos {
                Some(value) => std::env::set_var("DISABLE_PROMPT_CACHING_MYTHOS", value),
                None => std::env::remove_var("DISABLE_PROMPT_CACHING_MYTHOS"),
            }
            match &self.force_prompt_caching_5m {
                Some(value) => std::env::set_var("FORCE_PROMPT_CACHING_5M", value),
                None => std::env::remove_var("FORCE_PROMPT_CACHING_5M"),
            }
            match &self.main_prompt_cache_ttl {
                Some(value) => std::env::set_var("CLAUDE_CODE_PROMPT_CACHE_TTL", value),
                None => std::env::remove_var("CLAUDE_CODE_PROMPT_CACHE_TTL"),
            }
            match &self.subagent_prompt_cache_ttl {
                Some(value) => std::env::set_var("CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL", value),
                None => std::env::remove_var("CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL"),
            }
            match &self.enable_prompt_caching_1h {
                Some(value) => std::env::set_var("ENABLE_PROMPT_CACHING_1H", value),
                None => std::env::remove_var("ENABLE_PROMPT_CACHING_1H"),
            }
            match &self.enable_prompt_caching_1h_bedrock {
                Some(value) => std::env::set_var("ENABLE_PROMPT_CACHING_1H_BEDROCK", value),
                None => std::env::remove_var("ENABLE_PROMPT_CACHING_1H_BEDROCK"),
            }
        }
    }

    fn profile(base_url: &str) -> ProviderProfile {
        profile_for("anthropic", base_url)
    }

    fn profile_for(provider_id: &str, base_url: &str) -> ProviderProfile {
        serde_json::from_value(json!({
            "provider_id":provider_id,
            "profile_name":"prompt-cache-test",
            "base_url":base_url,
            "protocol":"anthropic_messages",
            "auth":"none",
            "regions":["international"],
            "models":[]
        }))
        .expect("provider profile")
    }

    fn native_route_allows_global_scope(
        profile: &ProviderProfile,
        process_base_url: Option<&str>,
        assume_first_party_base_url: bool,
    ) -> bool {
        native_profile_allows_global_scope(
            profile,
            native_base_url_gate(process_base_url, assume_first_party_base_url),
            assume_first_party_base_url,
        )
    }

    fn sections() -> Vec<SourceSection> {
        vec![
            SourceSection {
                text: "session section".into(),
                scope: SectionScope::Session,
            },
            SourceSection {
                text: "shared section".into(),
                scope: SectionScope::Shared,
            },
            SourceSection {
                text: DYNAMIC_BOUNDARY.into(),
                scope: SectionScope::Session,
            },
        ]
    }

    fn cache_policy(skip_global_cache_for_system_prompt: bool) -> CachePolicy {
        CachePolicy {
            skip_global_cache_for_system_prompt,
            ..CachePolicy::default()
        }
    }

    #[test]
    fn native_source_marker_defaults_on_without_subscription_or_local_opt_in() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        let official = profile("https://api.anthropic.com");
        assert_eq!(
            snapshot_source_vector(&sections(), Some(&official), GatePolicy::default()),
            vec![
                PromptText::from_string("shared section"),
                PromptText::from_string(DYNAMIC_BOUNDARY),
                PromptText::from_string("session section")
            ]
        );
        let aws = profile_for(
            "anthropicAws",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
        );
        assert_eq!(
            snapshot_source_vector(&sections(), Some(&aws), GatePolicy::default()),
            vec![
                PromptText::from_string("shared section"),
                PromptText::from_string(DYNAMIC_BOUNDARY),
                PromptText::from_string("session section")
            ]
        );
    }

    #[test]
    fn native_process_endpoint_gate_is_sampled_by_the_sdk() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        let official = profile("https://api.anthropic.com");
        let sections = sections();
        let expected = vec![
            PromptText::from_string("shared section"),
            PromptText::from_string(DYNAMIC_BOUNDARY),
            PromptText::from_string("session section"),
        ];
        assert_eq!(
            snapshot_source_vector(&sections, Some(&official), GatePolicy::from_process(false)),
            expected
        );

        std::env::set_var("ANTHROPIC_BASE_URL", "https://proxy.example.test");
        assert_eq!(
            snapshot_source_vector(&sections, Some(&official), GatePolicy::from_process(false)),
            vec![
                PromptText::from_string("shared section"),
                PromptText::from_string("session section")
            ]
        );

        std::env::set_var("_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL", "1");
        assert_eq!(
            snapshot_source_vector(&sections, Some(&official), GatePolicy::from_process(false)),
            expected
        );
    }

    #[test]
    fn snapshot_boundary_requires_selected_route_clear_taint_and_enabled_betas() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        let official = profile("https://api.anthropic.com");
        let proxy = profile("https://proxy.example.test");
        let nonstandard_port = profile("https://api.anthropic.com:8443");
        let enabled = GatePolicy::default();
        assert!(!global_scope_enabled(None, enabled));
        assert!(!global_scope_enabled(Some(&proxy), enabled));
        assert!(!global_scope_enabled(Some(&nonstandard_port), enabled));
        assert!(!global_scope_enabled(
            Some(&official),
            GatePolicy {
                hipaa_tainted: true,
                ..enabled
            }
        ));
        assert!(!global_scope_enabled(
            Some(&official),
            GatePolicy {
                experimental_betas_disabled: true,
                ..enabled
            }
        ));
    }

    #[test]
    fn native_global_scope_provider_and_process_url_gates_match_supported_routes() {
        let direct = profile("https://api.anthropic.com");
        let direct_proxy = profile("https://proxy.example.test");
        let aws = profile_for(
            "anthropicAws",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
        );
        let mut bedrock = profile_for(
            "bedrock-claude",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
        );
        bedrock.protocol = ProtocolFamily::BedrockClaude;
        let mut third_party = profile_for("openai", "https://api.openai.com");
        third_party.protocol = ProtocolFamily::OpenAiChat;
        let google_cloud = profile_for(
            "anthropicGoogleCloud",
            "https://us-east5-aiplatform.googleapis.com",
        );
        let mut foundry = profile_for("foundry-claude", "https://foundry.example.test");
        foundry.protocol = ProtocolFamily::FoundryClaude;

        assert!(native_route_allows_global_scope(&direct, None, false));
        assert!(native_route_allows_global_scope(
            &direct,
            Some("https://api.anthropic.com"),
            false
        ));
        assert!(!native_route_allows_global_scope(
            &direct,
            Some("https://proxy.example.test"),
            false
        ));
        assert!(!native_base_url_gate(
            Some("https://api.anthropic.com:8443"),
            false
        ));
        assert!(!native_base_url_gate(Some("not a URL"), false));
        assert!(native_base_url_gate(
            Some("https://proxy.example.test"),
            true
        ));
        assert!(!native_route_allows_global_scope(
            &direct_proxy,
            None,
            false
        ));
        assert!(native_route_allows_global_scope(&direct_proxy, None, true));

        // The Native anthropicAws provider is allowed to use an AWS regional
        // egress URL; ni() still evaluates only ANTHROPIC_BASE_URL.
        assert!(native_route_allows_global_scope(&aws, None, false));
        assert!(native_route_allows_global_scope(
            &aws,
            Some("https://api.anthropic.com"),
            false
        ));
        assert!(!native_route_allows_global_scope(
            &aws,
            Some("https://proxy.example.test"),
            false
        ));

        // The workspace's selected Bedrock profile uses its Bedrock Converse
        // protocol and is Native's distinct `bedrock` route, not
        // `anthropicAws`.
        assert!(!native_route_allows_global_scope(&bedrock, None, false));
        assert!(!native_route_allows_global_scope(&third_party, None, false));
        assert!(!native_route_allows_global_scope(
            &google_cloud,
            None,
            false
        ));
        assert!(!native_route_allows_global_scope(&foundry, None, false));
        // Ose checks route identity plus ni's process URL gate, without an
        // additional region-domain predicate on the resolved AWS endpoint.
        assert!(native_route_allows_global_scope(
            &profile_for("anthropicAws", "https://aws.example.test"),
            None,
            false
        ));
    }

    #[test]
    fn system_projection_uses_only_a_standalone_source_element_as_boundary() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        let official = profile("https://api.anthropic.com");
        let system = SystemPromptInput::source_vector(
            vec![
                PromptText::from_string(format!("shared {DYNAMIC_BOUNDARY} remains text")),
                PromptText::from_string(DYNAMIC_BOUNDARY),
                PromptText::from_string("session context"),
            ],
            None,
            None,
        );
        let blocks = project_system_prompt(
            &system,
            Some(&official),
            "claude-sonnet-4-20250514",
            ProtocolFamily::AnthropicMessages,
            cache_policy(false),
        );
        assert_eq!(
            blocks
                .iter()
                .map(|block| block.block.text.as_str())
                .collect::<Vec<_>>(),
            vec![
                format!("shared {DYNAMIC_BOUNDARY} remains text").as_str(),
                "session context"
            ]
        );
        assert_eq!(
            blocks[0].cache_breakpoint,
            Some(CacheBreakpoint {
                position: CachePosition::System { index: 0 },
                scope: Some(CacheScope::Global),
                ttl: CacheTtl::FiveMinutes,
            })
        );
        assert_eq!(
            blocks[1].cache_breakpoint,
            Some(CacheBreakpoint {
                position: CachePosition::System { index: 1 },
                scope: None,
                ttl: CacheTtl::FiveMinutes,
            })
        );
    }

    #[test]
    fn host_branding_identity_is_explicit_exact_and_anthropic_only() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        let official = profile("https://api.anthropic.com");
        let identity_units = "You are LingXi, an agentic command-line coding assistant."
            .encode_utf16()
            .collect::<Vec<_>>();
        let identity = PromptText::from_utf16(identity_units.clone());
        let exact_member = PromptText::from_utf16(identity_units);
        let project = |input: &SystemPromptInput, profile: Option<&ProviderProfile>, protocol| {
            project_system_prompt(
                input,
                profile,
                "claude-sonnet-4-20250514",
                protocol,
                cache_policy(false),
            )
        };

        let explicit = SystemPromptInput::source_vector(
            vec![
                exact_member.clone(),
                PromptText::from_string("static instructions"),
                PromptText::from_string(DYNAMIC_BOUNDARY),
                PromptText::from_string("dynamic instructions"),
            ],
            None,
            Some(identity.clone()),
        );
        let native_layout = project(
            &explicit,
            Some(&official),
            ProtocolFamily::AnthropicMessages,
        );
        assert_eq!(
            native_layout
                .iter()
                .map(|block| block.block.text.as_str())
                .collect::<Vec<_>>(),
            vec![
                identity.display_text(),
                "static instructions",
                "dynamic instructions"
            ]
        );
        assert_eq!(native_layout[0].cache_breakpoint, None);
        assert_eq!(
            native_layout[1]
                .cache_breakpoint
                .as_ref()
                .map(|point| point.scope),
            Some(Some(CacheScope::Global))
        );
        assert_eq!(
            native_layout[2]
                .cache_breakpoint
                .as_ref()
                .map(|point| point.scope),
            Some(None)
        );

        // A rewritten member does not inherit the identity's special block.
        let rewritten = SystemPromptInput::source_vector(
            vec![
                PromptText::from_string("You are LingXi, a rewritten assistant."),
                PromptText::from_string("static instructions"),
                PromptText::from_string(DYNAMIC_BOUNDARY),
                PromptText::from_string("dynamic instructions"),
            ],
            None,
            Some(identity.clone()),
        );
        let rewritten_layout = project(
            &rewritten,
            Some(&official),
            ProtocolFamily::AnthropicMessages,
        );
        assert_eq!(rewritten_layout.len(), 2);
        assert_eq!(
            rewritten_layout[0].block.text,
            "You are LingXi, a rewritten assistant.\n\nstatic instructions"
        );
        assert_eq!(
            rewritten_layout[0]
                .cache_breakpoint
                .as_ref()
                .map(|point| point.scope),
            Some(Some(CacheScope::Global))
        );

        // The same exact text remains ordinary without the explicit Host fact.
        let untyped = SystemPromptInput::source_vector(
            vec![
                exact_member,
                PromptText::from_string("static instructions"),
                PromptText::from_string(DYNAMIC_BOUNDARY),
                PromptText::from_string("dynamic instructions"),
            ],
            None,
            None,
        );
        let untyped_layout = project(&untyped, Some(&official), ProtocolFamily::AnthropicMessages);
        assert_eq!(untyped_layout.len(), 2);
        assert_eq!(
            untyped_layout[0].block.text,
            format!("{}\n\nstatic instructions", identity.display_text())
        );

        // With the Native global gate closed, the exact Host identity remains
        // a separate org-cache member; an untyped Header joins its neighbor.
        let unmarked_explicit = SystemPromptInput::source_vector(
            vec![
                identity.clone(),
                PromptText::from_string("static instructions"),
            ],
            None,
            Some(identity.clone()),
        );
        let gate_closed_policy = CachePolicy {
            process_base_url_allowed: false,
            ..cache_policy(false)
        };
        let explicit_closed_layout = project_system_prompt(
            &unmarked_explicit,
            Some(&official),
            "claude-sonnet-4-20250514",
            ProtocolFamily::AnthropicMessages,
            gate_closed_policy.clone(),
        );
        assert_eq!(
            explicit_closed_layout
                .iter()
                .map(|block| block.block.text.as_str())
                .collect::<Vec<_>>(),
            vec![identity.display_text(), "static instructions"]
        );
        assert!(explicit_closed_layout.iter().all(|block| {
            block
                .cache_breakpoint
                .as_ref()
                .is_some_and(|point| point.scope.is_none())
        }));

        let untyped_unmarked = SystemPromptInput::source_vector(
            vec![
                PromptText::from_string(identity.display_text()),
                PromptText::from_string("static instructions"),
            ],
            None,
            None,
        );
        let untyped_closed_layout = project_system_prompt(
            &untyped_unmarked,
            Some(&official),
            "claude-sonnet-4-20250514",
            ProtocolFamily::AnthropicMessages,
            gate_closed_policy,
        );
        assert_eq!(untyped_closed_layout.len(), 1);
        assert_eq!(
            untyped_closed_layout[0].block.text,
            format!("{}\n\nstatic instructions", identity.display_text())
        );

        // Custom input is one opaque source member and carries no Host identity.
        let custom = SystemPromptInput::custom_prompt(PromptText::from_string(format!(
            "{}\n\nstatic instructions",
            identity.display_text()
        )));
        let custom_layout = project(&custom, Some(&official), ProtocolFamily::AnthropicMessages);
        assert_eq!(custom_layout.len(), 1);
        assert_eq!(
            custom_layout[0].block.text,
            format!("{}\n\nstatic instructions", identity.display_text())
        );

        // Branding metadata is ignored by other provider protocols.
        let openai = profile_for("openai", "https://api.openai.com");
        let other_provider = project(&explicit, Some(&openai), ProtocolFamily::OpenAiChat);
        assert_eq!(other_provider.len(), 1);
        assert!(other_provider[0]
            .block
            .text
            .starts_with(identity.display_text()));
    }

    #[test]
    fn skip_global_cache_changes_only_unmarked_vectors() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        let official = profile("https://api.anthropic.com");
        let unmarked = SystemPromptInput::source_vector(
            vec![
                PromptText::from_string("base"),
                PromptText::from_string("dynamic"),
            ],
            None,
            None,
        );
        let skipped = project_system_prompt(
            &unmarked,
            Some(&official),
            "claude-sonnet-4-20250514",
            ProtocolFamily::AnthropicMessages,
            cache_policy(true),
        );
        assert_eq!(
            skipped
                .iter()
                .map(|block| block.block.text.as_str())
                .collect::<Vec<_>>(),
            vec!["base\n\ndynamic"]
        );
        assert_eq!(
            skipped[0]
                .cache_breakpoint
                .as_ref()
                .map(|breakpoint| breakpoint.scope),
            Some(None)
        );

        let marked = SystemPromptInput::source_vector(
            vec![
                PromptText::from_string("base"),
                PromptText::from_string(DYNAMIC_BOUNDARY),
                PromptText::from_string("dynamic"),
            ],
            None,
            None,
        );
        let honored_marker = project_system_prompt(
            &marked,
            Some(&official),
            "claude-sonnet-4-20250514",
            ProtocolFamily::AnthropicMessages,
            cache_policy(true),
        );
        assert_eq!(
            honored_marker
                .iter()
                .map(|block| block.block.text.as_str())
                .collect::<Vec<_>>(),
            vec!["base", "dynamic"]
        );
        assert_eq!(
            honored_marker[0]
                .cache_breakpoint
                .as_ref()
                .map(|breakpoint| breakpoint.scope),
            Some(Some(CacheScope::Global))
        );
        assert_eq!(
            honored_marker[1]
                .cache_breakpoint
                .as_ref()
                .map(|breakpoint| breakpoint.scope),
            Some(None)
        );
    }

    #[test]
    fn custom_prompt_and_utf16_units_reach_the_native_projection_without_join_parsing() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        let official = profile("https://api.anthropic.com");
        let custom = SystemPromptInput::custom_prompt(PromptText::from_string(format!(
            "before {DYNAMIC_BOUNDARY} after"
        )));
        let custom_blocks = project_system_prompt(
            &custom,
            Some(&official),
            "claude-sonnet-4-20250514",
            ProtocolFamily::AnthropicMessages,
            cache_policy(false),
        );
        assert_eq!(custom_blocks.len(), 1);
        assert_eq!(
            custom_blocks[0].block.text,
            format!("before {DYNAMIC_BOUNDARY} after")
        );

        let exact = PromptText::from_utf16(vec![0xd800, 0x03a9, 0xdc00]);
        assert_eq!(exact.utf16_code_units(), &[0xd800, 0x03a9, 0xdc00]);
        assert_eq!(exact.display_text(), "�Ω�");
        let exact_blocks = project_system_prompt(
            &SystemPromptInput::source_vector(vec![exact], None, None),
            Some(&official),
            "claude-sonnet-4-20250514",
            ProtocolFamily::AnthropicMessages,
            cache_policy(false),
        );
        assert_eq!(exact_blocks[0].block.text, "�Ω�");
    }

    #[test]
    fn cold_snapshot_utf16_values_project_like_native_replacement_values() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        let official = profile("https://api.anthropic.com");
        let mut snapshot_units = "snapshot ".encode_utf16().collect::<Vec<_>>();
        snapshot_units.extend([0xd800, 0x03a9, 0xdc00]);
        let mut context_units = "cwd: /repo/".encode_utf16().collect::<Vec<_>>();
        context_units.push(0xd800);

        let exact_snapshot = SystemPromptInput::source_vector(
            vec![
                PromptText::from_string("snapshot prefix"),
                PromptText::from_string(DYNAMIC_BOUNDARY),
                PromptText::from_utf16(snapshot_units),
            ],
            Some(PromptText::from_utf16(context_units)),
            None,
        );
        let cold_native_snapshot = SystemPromptInput::source_vector(
            vec![
                PromptText::from_string("snapshot prefix"),
                PromptText::from_string(DYNAMIC_BOUNDARY),
                PromptText::from_string("snapshot �Ω�"),
            ],
            Some(PromptText::from_string("cwd: /repo/�")),
            None,
        );

        let exact_blocks = project_system_prompt(
            &exact_snapshot,
            Some(&official),
            "claude-sonnet-4-6",
            ProtocolFamily::AnthropicMessages,
            CachePolicy::default(),
        );
        let cold_blocks = project_system_prompt(
            &cold_native_snapshot,
            Some(&official),
            "claude-sonnet-4-6",
            ProtocolFamily::AnthropicMessages,
            CachePolicy::default(),
        );
        assert_eq!(exact_blocks, cold_blocks);
        assert_eq!(
            exact_blocks
                .iter()
                .map(|block| block.block.text.as_str())
                .collect::<Vec<_>>(),
            vec!["snapshot prefix", "snapshot �Ω�\n\ncwd: /repo/�"]
        );
        assert_eq!(
            exact_blocks[0]
                .cache_breakpoint
                .as_ref()
                .map(|breakpoint| breakpoint.scope),
            Some(Some(CacheScope::Global))
        );
        assert_eq!(
            exact_blocks[1]
                .cache_breakpoint
                .as_ref()
                .map(|breakpoint| breakpoint.scope),
            Some(None)
        );
    }

    #[test]
    fn host_parsed_native_custom_prompt_uses_source_elements_across_protocols() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        let official = profile("https://api.anthropic.com");
        let openai = profile_for("openai", "https://api.openai.com");
        let original =
            PromptText::from_string(format!("custom before\n{DYNAMIC_BOUNDARY}\ncustom after"));
        let native_custom = SystemPromptInput::native_custom_prompt(
            original.clone(),
            vec![
                PromptText::from_string("custom before\n"),
                PromptText::from_string(DYNAMIC_BOUNDARY),
                PromptText::from_string("\ncustom after"),
            ],
        );

        let claude_blocks = project_system_prompt(
            &native_custom,
            Some(&official),
            "claude-sonnet-4-20250514",
            ProtocolFamily::AnthropicMessages,
            cache_policy(false),
        );
        assert_eq!(
            claude_blocks
                .iter()
                .map(|block| block.block.text.as_str())
                .collect::<Vec<_>>(),
            vec!["custom before\n", "\ncustom after"]
        );
        assert_eq!(
            claude_blocks[0]
                .cache_breakpoint
                .as_ref()
                .map(|breakpoint| breakpoint.scope),
            Some(Some(CacheScope::Global))
        );
        assert_eq!(
            claude_blocks[1]
                .cache_breakpoint
                .as_ref()
                .map(|breakpoint| breakpoint.scope),
            Some(None)
        );

        let openai_blocks = project_system_prompt(
            &native_custom,
            Some(&openai),
            "gpt-6.1",
            ProtocolFamily::OpenAiChat,
            cache_policy(false),
        );
        assert_eq!(openai_blocks.len(), 1);
        assert_eq!(
            openai_blocks[0].block.text,
            "custom before\n\n\n\ncustom after"
        );

        // Native `overrideSystemPrompt` bypasses `hwe`; an opaque override
        // containing the marker on its own line stays one literal section.
        let override_text = format!("custom before\n{DYNAMIC_BOUNDARY}\ncustom after");
        let override_input =
            SystemPromptInput::custom_prompt(PromptText::from_string(override_text.clone()));
        let claude_override_blocks = project_system_prompt(
            &override_input,
            Some(&official),
            "claude-sonnet-4-20250514",
            ProtocolFamily::AnthropicMessages,
            cache_policy(false),
        );
        assert_eq!(claude_override_blocks.len(), 1);
        assert_eq!(claude_override_blocks[0].block.text, override_text);
        let override_blocks = project_system_prompt(
            &override_input,
            Some(&openai),
            "gpt-6.1",
            ProtocolFamily::OpenAiChat,
            cache_policy(false),
        );
        assert_eq!(override_blocks.len(), 1);
        assert_eq!(override_blocks[0].block.text, override_text);
    }

    #[test]
    fn non_claude_routes_drop_only_structural_marker_elements() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        let openai = profile_for("openai", "https://api.openai.com");
        let input = SystemPromptInput::source_vector(
            vec![
                PromptText::from_string(format!("literal {DYNAMIC_BOUNDARY} text")),
                PromptText::from_string(DYNAMIC_BOUNDARY),
                PromptText::from_string("dynamic"),
            ],
            None,
            None,
        );
        let blocks = project_system_prompt(
            &input,
            Some(&openai),
            "gpt-6.1",
            ProtocolFamily::OpenAiChat,
            cache_policy(false),
        );
        assert_eq!(blocks.len(), 1);
        assert_eq!(
            blocks[0].block.text,
            format!("literal {DYNAMIC_BOUNDARY} text\n\ndynamic")
        );
        assert_eq!(blocks[0].cache_breakpoint, None);
    }

    #[test]
    fn native_unmarked_mcp_gate_keeps_special_sections_and_one_hour_policy() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        let official = profile("https://api.anthropic.com");
        let text = SystemPromptInput::source_vector(
            vec![
                PromptText::from_string("x-anthropic-billing-header: metadata"),
                PromptText::from_string(NATIVE_BRAND_SECTIONS[0]),
                PromptText::from_string(NATIVE_REPORTING_SECTION),
                PromptText::from_string("ordinary context"),
            ],
            None,
            None,
        );
        let blocks = project_system_prompt(
            &text,
            Some(&official),
            "claude-sonnet-4-20250514",
            ProtocolFamily::AnthropicMessages,
            CachePolicy {
                agent_prompt_cache_ttl_override: Some(AgentPromptCacheTtlOverride::OneHour),
                skip_global_cache_for_system_prompt: true,
                ..CachePolicy::default()
            },
        );
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].block.text, "x-anthropic-billing-header: metadata");
        assert_eq!(blocks[0].cache_breakpoint, None);
        assert_eq!(blocks[1].block.text, NATIVE_BRAND_SECTIONS[0]);
        assert_eq!(blocks[2].block.text, NATIVE_REPORTING_SECTION);
        assert_eq!(blocks[2].cache_breakpoint, None);
        assert_eq!(blocks[3].block.text, "ordinary context");
        assert!(blocks[1..]
            .iter()
            .filter_map(|block| block.cache_breakpoint.as_ref())
            .all(|breakpoint| breakpoint.ttl == CacheTtl::OneHour));
    }

    #[test]
    fn prompt_cache_source_uses_native_main_query_allowlist() {
        assert!(PromptCacheQuerySource::Named("repl_main_thread").is_main());
        assert!(PromptCacheQuerySource::Named("repl_main_thread:outputStyle").is_main());
        assert!(PromptCacheQuerySource::Named("sdk").is_main());
        assert!(PromptCacheQuerySource::Named("auto_mode").is_main());
        assert!(PromptCacheQuerySource::Named("memdir_relevance").is_main());
        assert!(!PromptCacheQuerySource::Named("agent:custom").is_main());
        assert!(!PromptCacheQuerySource::Named("fusion_panel").is_main());
        assert!(!PromptCacheQuerySource::Subagent.is_main());
        assert!(!PromptCacheQuerySource::Unspecified.is_main());
    }

    #[test]
    fn source_specific_ttl_env_precedes_agent_and_global_1h_controls() {
        let mut policy = CachePolicy {
            agent_prompt_cache_ttl_override: Some(AgentPromptCacheTtlOverride::FiveMinutes),
            query_source_is_main: true,
            ..CachePolicy::default()
        };
        let env = PromptCacheTtlEnvironment {
            force_five_minutes: false,
            main_ttl: Some(PromptCacheTtl::OneHour),
            subagent_ttl: Some(PromptCacheTtl::FiveMinutes),
            enable_one_hour: true,
            enable_bedrock_one_hour: false,
        };
        assert_eq!(
            resolve_prompt_cache_ttl(ProtocolFamily::AnthropicMessages, &policy, env),
            CacheTtl::OneHour,
            "main source env overrides per-agent 5m"
        );

        policy.query_source_is_main = false;
        assert_eq!(
            resolve_prompt_cache_ttl(ProtocolFamily::AnthropicMessages, &policy, env),
            CacheTtl::FiveMinutes,
            "subagent source env overrides global 1h"
        );

        assert_eq!(
            resolve_prompt_cache_ttl(
                ProtocolFamily::AnthropicMessages,
                &policy,
                PromptCacheTtlEnvironment {
                    force_five_minutes: true,
                    ..env
                }
            ),
            CacheTtl::FiveMinutes,
            "force 5m remains first"
        );
    }

    #[test]
    fn source_specific_ttl_settings_follow_env_and_precede_agent_and_global_controls() {
        let settings = PromptCacheTtlSettings {
            prompt_cache_ttl: Some(PromptCacheTtl::FiveMinutes),
            subagent_prompt_cache_ttl: Some(PromptCacheTtl::OneHour),
        };
        let mut policy = CachePolicy {
            agent_prompt_cache_ttl_override: Some(AgentPromptCacheTtlOverride::OneHour),
            query_source_is_main: true,
            prompt_cache_ttl_settings: settings,
            ..CachePolicy::default()
        };
        let env = PromptCacheTtlEnvironment {
            force_five_minutes: false,
            main_ttl: None,
            subagent_ttl: None,
            enable_one_hour: true,
            enable_bedrock_one_hour: false,
        };
        assert_eq!(
            resolve_prompt_cache_ttl(ProtocolFamily::AnthropicMessages, &policy, env),
            CacheTtl::FiveMinutes,
            "main setting overrides explicit agent 1h"
        );

        policy.query_source_is_main = false;
        policy.agent_prompt_cache_ttl_override = Some(AgentPromptCacheTtlOverride::FiveMinutes);
        assert_eq!(
            resolve_prompt_cache_ttl(ProtocolFamily::AnthropicMessages, &policy, env),
            CacheTtl::OneHour,
            "subagent setting selects the non-main branch and precedes agent 5m"
        );

        assert_eq!(
            resolve_prompt_cache_ttl(
                ProtocolFamily::AnthropicMessages,
                &policy,
                PromptCacheTtlEnvironment {
                    subagent_ttl: Some(PromptCacheTtl::FiveMinutes),
                    ..env
                }
            ),
            CacheTtl::FiveMinutes,
            "matching env setting precedes the settings field"
        );

        assert_eq!(
            resolve_prompt_cache_ttl(
                ProtocolFamily::AnthropicMessages,
                &policy,
                PromptCacheTtlEnvironment {
                    force_five_minutes: true,
                    subagent_ttl: Some(PromptCacheTtl::OneHour),
                    ..env
                }
            ),
            CacheTtl::FiveMinutes,
            "force 5m remains the first gate"
        );
    }

    #[test]
    fn subscriber_ttl_uses_native_overage_gate_and_lazy_source_allowlist() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        use std::sync::atomic::{AtomicUsize, Ordering};

        let reads = Arc::new(AtomicUsize::new(0));
        let reads_from_source = Arc::clone(&reads);
        let mut policy = CachePolicy {
            query_source_is_main: true,
            query_source: Some("repl_main_thread:resume".into()),
            prompt_cache_ttl_inputs: PromptCacheTtlInputs {
                subscriber: PromptCacheSubscriberState::Subscriber,
                is_using_overage: false,
                subscriber_allowlist: Arc::new(move || {
                    reads_from_source.fetch_add(1, Ordering::SeqCst);
                    vec!["repl_main_thread*".into()]
                }),
            },
            ..CachePolicy::default()
        };
        let env = PromptCacheTtlEnvironment {
            force_five_minutes: false,
            main_ttl: None,
            subagent_ttl: None,
            enable_one_hour: false,
            enable_bedrock_one_hour: false,
        };
        assert_eq!(
            resolve_prompt_cache_ttl(ProtocolFamily::AnthropicMessages, &policy, env),
            CacheTtl::OneHour,
        );
        assert_eq!(reads.load(Ordering::SeqCst), 1);

        policy.agent_prompt_cache_ttl_override = Some(AgentPromptCacheTtlOverride::FiveMinutes);
        assert_eq!(
            resolve_prompt_cache_ttl(ProtocolFamily::AnthropicMessages, &policy, env),
            CacheTtl::FiveMinutes,
        );
        assert_eq!(
            reads.load(Ordering::SeqCst),
            1,
            "explicit TTL skips the allowlist"
        );

        policy.agent_prompt_cache_ttl_override = Some(AgentPromptCacheTtlOverride::OneHour);
        policy.prompt_cache_ttl_inputs.is_using_overage = true;
        assert_eq!(
            resolve_prompt_cache_ttl(ProtocolFamily::AnthropicMessages, &policy, env),
            CacheTtl::FiveMinutes,
            "subscriber overage suppresses only the explicit agent 1h and subscriber fallback",
        );
        assert_eq!(
            reads.load(Ordering::SeqCst),
            1,
            "overage skips allowlist fetch"
        );
        assert_eq!(
            resolve_prompt_cache_ttl(
                ProtocolFamily::AnthropicMessages,
                &policy,
                PromptCacheTtlEnvironment {
                    enable_one_hour: true,
                    ..env
                },
            ),
            CacheTtl::OneHour,
            "Native's later ENABLE_PROMPT_CACHING_1H gate still wins during overage",
        );

        policy.prompt_cache_ttl_inputs.is_using_overage = false;
        policy.prompt_cache_ttl_inputs.subscriber = PromptCacheSubscriberState::Unknown;
        policy.agent_prompt_cache_ttl_override = None;
        assert_eq!(
            resolve_prompt_cache_ttl(ProtocolFamily::AnthropicMessages, &policy, env),
            CacheTtl::FiveMinutes,
            "unknown custom route/account remains separate from confirmed subscriber",
        );
        assert_eq!(
            reads.load(Ordering::SeqCst),
            1,
            "unknown gate skips allowlist fetch"
        );

        policy.prompt_cache_ttl_inputs.subscriber = PromptCacheSubscriberState::NotSubscriber;
        policy.agent_prompt_cache_ttl_override = Some(AgentPromptCacheTtlOverride::OneHour);
        assert_eq!(
            resolve_prompt_cache_ttl(ProtocolFamily::AnthropicMessages, &policy, env),
            CacheTtl::OneHour,
            "Native explicit agent 1h is suppressed by subscriber overage, not by non-subscriber status",
        );
    }

    #[test]
    fn subscriber_allowlist_fallback_matches_native_prefix_entries_only_after_gates() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();
        let list = || PromptCacheTtlInputs {
            subscriber: PromptCacheSubscriberState::Subscriber,
            is_using_overage: false,
            subscriber_allowlist: Arc::new(|| vec!["repl_main_thread*".into(), "sdk".into()]),
        };
        for (source, expected) in [
            ("repl_main_thread", CacheTtl::OneHour),
            ("repl_main_thread:outputStyle", CacheTtl::OneHour),
            ("sdk", CacheTtl::OneHour),
            ("fusion_panel", CacheTtl::FiveMinutes),
        ] {
            let policy = CachePolicy {
                query_source_is_main: PromptCacheQuerySource::Named(source).is_main(),
                query_source: Some(source.into()),
                prompt_cache_ttl_inputs: list(),
                ..CachePolicy::default()
            };
            assert_eq!(
                resolve_prompt_cache_ttl(
                    ProtocolFamily::AnthropicMessages,
                    &policy,
                    PromptCacheTtlEnvironment {
                        force_five_minutes: false,
                        main_ttl: None,
                        subagent_ttl: None,
                        enable_one_hour: false,
                        enable_bedrock_one_hour: false,
                    },
                ),
                expected,
                "source {source:?}",
            );
        }
    }

    #[test]
    fn v290_cache_ttl_environment_gates_and_model_families_match_native() {
        let _lock = GATE_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env = GateEnvRestore::clear();

        assert!(prompt_caching_enabled(
            "claude-fable-5-1",
            ProtocolFamily::AnthropicMessages
        ));
        assert!(prompt_caching_enabled(
            "claude-mythos-5-1",
            ProtocolFamily::AnthropicMessages
        ));
        std::env::set_var("DISABLE_PROMPT_CACHING_FABLE", "1");
        assert!(!prompt_caching_enabled(
            "claude-fable-5-1",
            ProtocolFamily::AnthropicMessages
        ));
        assert!(prompt_caching_enabled(
            "claude-mythos-5-1",
            ProtocolFamily::AnthropicMessages
        ));
        std::env::set_var("DISABLE_PROMPT_CACHING_MYTHOS", "1");
        assert!(!prompt_caching_enabled(
            "claude-mythos-5-1",
            ProtocolFamily::AnthropicMessages
        ));

        let official = profile("https://api.anthropic.com");
        let one_hour_request =
            SystemPromptInput::source_vector(vec![PromptText::from_string("one hour")], None, None);
        std::env::set_var("FORCE_PROMPT_CACHING_5M", "1");
        let forced_five_minutes = project_system_prompt(
            &one_hour_request,
            Some(&official),
            "claude-sonnet-4-6",
            ProtocolFamily::AnthropicMessages,
            CachePolicy {
                agent_prompt_cache_ttl_override: Some(AgentPromptCacheTtlOverride::OneHour),
                ..CachePolicy::default()
            },
        );
        assert_eq!(
            forced_five_minutes[0]
                .cache_breakpoint
                .as_ref()
                .map(|breakpoint| breakpoint.ttl),
            Some(CacheTtl::FiveMinutes)
        );

        std::env::remove_var("FORCE_PROMPT_CACHING_5M");
        std::env::set_var("ENABLE_PROMPT_CACHING_1H", "1");
        let explicit_five_minutes = project_system_prompt(
            &one_hour_request,
            Some(&official),
            "claude-sonnet-4-6",
            ProtocolFamily::AnthropicMessages,
            CachePolicy {
                agent_prompt_cache_ttl_override: Some(AgentPromptCacheTtlOverride::FiveMinutes),
                ..CachePolicy::default()
            },
        );
        assert_eq!(
            explicit_five_minutes[0]
                .cache_breakpoint
                .as_ref()
                .map(|breakpoint| breakpoint.ttl),
            Some(CacheTtl::FiveMinutes)
        );
        std::env::remove_var("ENABLE_PROMPT_CACHING_1H");
        std::env::set_var("ENABLE_PROMPT_CACHING_1H_BEDROCK", "1");
        let mut bedrock = profile_for(
            "bedrock-claude",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
        );
        bedrock.protocol = ProtocolFamily::BedrockClaude;
        let bedrock_blocks = project_system_prompt(
            &one_hour_request,
            Some(&bedrock),
            "claude-sonnet-4-6",
            ProtocolFamily::BedrockClaude,
            CachePolicy::default(),
        );
        assert_eq!(
            bedrock_blocks[0]
                .cache_breakpoint
                .as_ref()
                .map(|breakpoint| breakpoint.ttl),
            Some(CacheTtl::OneHour)
        );
    }
}
