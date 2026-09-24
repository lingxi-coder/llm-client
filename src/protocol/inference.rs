//! Inference controls and discoverable, endpoint-specific support facts.
use super::{CapabilitySupport, ProtocolFamily, ProviderProfile, TokenPricing};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
    Max,
}
impl ReasoningEffort {
    pub const ALL: [Self; 7] = [
        Self::None,
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::XHigh,
        Self::Max,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingMode {
    Disabled,
    Enabled,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingBudget {
    Tokens(u32),
    Dynamic,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ThinkingConfig {
    pub mode: Option<ThinkingMode>,
    pub budget: Option<ThinkingBudget>,
    pub effort: Option<ReasoningEffort>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    #[default]
    Standard,
    Fast,
}

/// None means unknown; an empty modes/levels list explicitly allows no values.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InferenceFeatures {
    pub thinking: CapabilitySupport,
    pub modes: Option<Vec<ThinkingMode>>,
    /// Per-mode observations and combination restrictions. Missing entries are unknown.
    pub mode_support: BTreeMap<ThinkingMode, ThinkingModeSupport>,
    pub default_mode: Option<ThinkingMode>,
    pub budget: BudgetSupport,
    pub effort: EffortSupport,
    pub fast: CapabilitySupport,
    pub default_service_tier: Option<ServiceTier>,
    pub requirements: Vec<String>,
    pub sources: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ThinkingModeSupport {
    pub support: CapabilitySupport,
    /// Additional effort restriction while this thinking mode is active.
    pub effort_levels: Option<Vec<ReasoningEffort>>,
    pub forced_tool_choice: CapabilitySupport,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BudgetSupport {
    pub support: CapabilitySupport,
    pub min_tokens: Option<u32>,
    pub max_tokens: Option<u32>,
    pub dynamic: CapabilitySupport,
    /// A target is not a guaranteed spending or token cap.
    pub hard_limit: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EffortSupport {
    pub support: CapabilitySupport,
    /// Whether a non-None effort can accompany an explicit Disabled mode.
    /// Model-specific effort restrictions still apply.
    pub with_disabled_thinking: CapabilitySupport,
    pub levels: Option<Vec<ReasoningEffort>>,
    /// Explicit facts from directories that report only some effort levels.
    pub level_support: BTreeMap<ReasoningEffort, CapabilitySupport>,
    pub default: Option<ReasoningEffort>,
}

/// Wire vocabulary, declared by the connection rather than inferred from its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningWire {
    OpenAiChat,
    OpenAiResponses,
    Anthropic,
    Gemini,
    ThinkingType,
    EnableThinking,
    ReasoningObject,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastWire {
    Priority,
    Fast,
    Speed,
    Unavailable,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InferenceWire {
    pub reasoning: Option<ReasoningWire>,
    pub fast: Option<FastWire>,
    pub fast_beta: Option<String>,
    pub tier_response_header: Option<String>,
    pub enabled_requires_budget: Option<bool>,
}

impl ProviderProfile {
    pub(crate) fn reasoning_wire(&self) -> ReasoningWire {
        self.inference.reasoning.unwrap_or(match self.protocol {
            ProtocolFamily::OpenAiChat | ProtocolFamily::AzureOpenAi => ReasoningWire::OpenAiChat,
            ProtocolFamily::OpenAiResponses => ReasoningWire::OpenAiResponses,
            ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini => {
                ReasoningWire::Gemini
            }
            _ => ReasoningWire::Anthropic,
        })
    }

    /// Project wire constraints into the same facts used by model selectors.
    pub(crate) fn inference_features(&self) -> InferenceFeatures {
        let mut constraints = InferenceFeatures::default();
        constraints.effort.with_disabled_thinking = match self.reasoning_wire() {
            ReasoningWire::Anthropic => CapabilitySupport::Supported,
            ReasoningWire::Unavailable => CapabilitySupport::Unknown,
            _ => CapabilitySupport::Unsupported,
        };
        self.info.features.on_connection(&constraints)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelInfo {
    pub features: InferenceFeatures,
    /// Read-only projection of ModelProfile.pricing, populated by the client.
    /// Configure prices through ModelProfile.pricing or ModelField::Pricing.
    pub pricing: Option<TokenPricing>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderPricingInfo {
    pub currencies: Vec<String>,
    pub units: Vec<String>,
    pub dimensions: Vec<String>,
    pub sources: Vec<String>,
}

/// Observations remain distinct from requested settings. Absence is not standard.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InferenceReport {
    /// Client dispatch time for the successful attempt, in UTC Unix seconds.
    /// This is a local observation, not a provider confirmation.
    pub executed_at: Option<u64>,
    pub requested_raw_service_tier: Option<String>,
    pub requested_effort: Option<ReasoningEffort>,
    pub requested_service_tier: Option<ServiceTier>,
    pub reported_effort: Option<ReasoningEffort>,
    pub service_tier: Option<ServiceTier>,
    pub raw_effort: Option<String>,
    pub raw_service_tier: Option<String>,
    pub raw_speed: Option<String>,
}

impl InferenceFeatures {
    pub(crate) fn effort_mode_error(
        &self,
        mode: Option<ThinkingMode>,
        effort: ReasoningEffort,
    ) -> Option<&'static str> {
        match mode {
            Some(ThinkingMode::Enabled | ThinkingMode::Adaptive)
                if effort == ReasoningEffort::None =>
            {
                Some("enabled thinking conflicts with effort none")
            }
            Some(ThinkingMode::Disabled)
                if effort != ReasoningEffort::None
                    && self.effort.with_disabled_thinking == CapabilitySupport::Unsupported =>
            {
                Some("disabled thinking conflicts with a reasoning effort")
            }
            _ => None,
        }
    }

    /// Effective support, including partial directory observations.
    pub fn supports_mode(&self, mode: ThinkingMode) -> CapabilitySupport {
        use CapabilitySupport::*;
        if mode != ThinkingMode::Disabled && self.thinking == Unsupported {
            return Unsupported;
        }
        let explicit = self.mode_support.get(&mode).map_or(Unknown, |s| s.support);
        if explicit == Unsupported || self.modes.as_ref().is_some_and(|v| !v.contains(&mode)) {
            Unsupported
        } else if explicit == Supported || self.modes.is_some() {
            Supported
        } else {
            Unknown
        }
    }

    /// Shared by request validation and callers building effort selectors.
    pub fn supports_effort(
        &self,
        mode: Option<ThinkingMode>,
        effort: ReasoningEffort,
    ) -> CapabilitySupport {
        use CapabilitySupport::*;
        if self.effort_mode_error(mode, effort).is_some() {
            return Unsupported;
        }
        let mode = mode.or(self.default_mode);
        if self.effort.support == Unsupported
            || self
                .effort
                .levels
                .as_ref()
                .is_some_and(|v| !v.contains(&effort))
            || self.effort.level_support.get(&effort) == Some(&Unsupported)
            || (effort == ReasoningEffort::None
                && self.supports_mode(ThinkingMode::Disabled) == Unsupported)
            || mode.is_some_and(|m| {
                self.supports_mode(m) == Unsupported
                    || self
                        .mode_support
                        .get(&m)
                        .and_then(|s| s.effort_levels.as_ref())
                        .is_some_and(|v| !v.contains(&effort))
            })
        {
            Unsupported
        } else if self.effort.levels.is_some()
            || self.effort.level_support.get(&effort) == Some(&Supported)
        {
            Supported
        } else {
            Unknown
        }
    }

    pub fn forced_tool_choice(&self, mode: ThinkingMode) -> CapabilitySupport {
        if self.supports_mode(mode) == CapabilitySupport::Unsupported {
            return CapabilitySupport::Unsupported;
        }
        self.mode_support
            .get(&mode)
            .map_or(CapabilitySupport::Unknown, |s| s.forced_tool_choice)
    }

    /// Merge explicit observations only; missing/unknown fields preserve facts.
    pub fn overlay(&mut self, other: &Self) {
        use CapabilitySupport::Unknown;
        if other.thinking != Unknown {
            self.thinking = other.thinking;
        }
        if let Some(modes) = &other.modes {
            self.modes.clone_from(&other.modes);
            for (mode, support) in &mut self.mode_support {
                support.support = if modes.contains(mode) {
                    CapabilitySupport::Supported
                } else {
                    CapabilitySupport::Unsupported
                };
            }
        }
        for (mode, observed) in &other.mode_support {
            let current = self.mode_support.entry(*mode).or_default();
            if observed.support != Unknown {
                current.support = observed.support;
                update_allowed(&mut self.modes, *mode, observed.support);
            }
            if observed.effort_levels.is_some() {
                current.effort_levels.clone_from(&observed.effort_levels);
            }
            if observed.forced_tool_choice != Unknown {
                current.forced_tool_choice = observed.forced_tool_choice;
            }
        }
        if other.default_mode.is_some() {
            self.default_mode = other.default_mode;
        }
        if other.budget.support != Unknown {
            self.budget.support = other.budget.support;
        }
        if other.budget.dynamic != Unknown {
            self.budget.dynamic = other.budget.dynamic;
        }
        if other.budget.min_tokens.is_some() {
            self.budget.min_tokens = other.budget.min_tokens;
        }
        if other.budget.max_tokens.is_some() {
            self.budget.max_tokens = other.budget.max_tokens;
        }
        if other.budget.hard_limit.is_some() {
            self.budget.hard_limit = other.budget.hard_limit;
        }
        if other.effort.support != Unknown {
            self.effort.support = other.effort.support;
        }
        if other.effort.with_disabled_thinking != Unknown {
            self.effort.with_disabled_thinking = other.effort.with_disabled_thinking;
        }
        if other.effort.levels.is_some() {
            self.effort.levels.clone_from(&other.effort.levels);
            self.effort.level_support.clear();
        }
        for (level, support) in &other.effort.level_support {
            if *support != Unknown {
                self.effort.level_support.insert(*level, *support);
                update_allowed(&mut self.effort.levels, *level, *support);
            }
        }
        if other.effort.default.is_some() {
            self.effort.default = other.effort.default;
        }
        if other.fast != Unknown {
            self.fast = other.fast;
        }
        if other.default_service_tier.is_some() {
            self.default_service_tier = other.default_service_tier;
        }
        for value in &other.requirements {
            if !self.requirements.contains(value) {
                self.requirements.push(value.clone());
            }
        }
        for value in &other.sources {
            if !self.sources.contains(value) {
                self.sources.push(value.clone());
            }
        }
        self.restrict_to_modes();
    }

    fn restrict_to_modes(&mut self) {
        use CapabilitySupport::Unsupported;
        if let Some(modes) = &mut self.modes {
            modes.retain(|m| {
                self.mode_support
                    .get(m)
                    .is_none_or(|s| s.support != Unsupported)
            });
        }
        if let Some(levels) = &mut self.effort.levels {
            levels.retain(|e| self.effort.level_support.get(e) != Some(&Unsupported));
        }
        self.default_mode = self
            .default_mode
            .filter(|mode| self.supports_mode(*mode) != Unsupported);
        self.effort.default = self
            .effort
            .default
            .filter(|effort| self.supports_effort(None, *effort) != Unsupported);
        if self.thinking == Unsupported {
            if let Some(modes) = &mut self.modes {
                modes.retain(|mode| *mode == ThinkingMode::Disabled);
            }
            self.default_mode = self
                .default_mode
                .filter(|mode| *mode == ThinkingMode::Disabled);
            self.budget.support = Unsupported;
        }
        if self.budget.support == Unsupported {
            self.budget.dynamic = Unsupported;
            self.budget.min_tokens = None;
            self.budget.max_tokens = None;
            self.budget.hard_limit = None;
        }
        if self.effort.support == Unsupported {
            self.effort.levels = Some(Vec::new());
        }
        if let Some(levels) = &self.effort.levels {
            self.effort.default = self.effort.default.filter(|effort| levels.contains(effort));
        }
        if self.fast == Unsupported && self.default_service_tier == Some(ServiceTier::Fast) {
            self.default_service_tier = None;
        }
        let Some(modes) = &self.modes else {
            return;
        };
        // An explicit new restriction invalidates conflicting older facts,
        // even when a directory update omits defaults and effort levels.
        self.default_mode = self.default_mode.filter(|mode| modes.contains(mode));
        if !modes.contains(&ThinkingMode::Disabled) {
            if let Some(levels) = &mut self.effort.levels {
                levels.retain(|effort| *effort != ReasoningEffort::None);
            }
            self.effort.default = self
                .effort
                .default
                .filter(|effort| *effort != ReasoningEffort::None);
        }
    }

    /// Connection restrictions bound a model. Positive connection facts alone
    /// cannot establish support on an unknown model.
    pub fn on_connection(&self, connection: &Self) -> Self {
        use CapabilitySupport::Unsupported;
        let mut result = self.clone();
        result.default_service_tier = result
            .default_service_tier
            .or(connection.default_service_tier);
        for (mode, limit) in &connection.mode_support {
            let own = result.mode_support.entry(*mode).or_default();
            if limit.support == Unsupported {
                own.support = Unsupported;
            }
            if limit.forced_tool_choice == Unsupported {
                own.forced_tool_choice = Unsupported;
            }
            if let Some(allowed) = &limit.effort_levels {
                own.effort_levels = Some(own.effort_levels.as_ref().map_or_else(
                    || allowed.clone(),
                    |v| v.iter().filter(|e| allowed.contains(e)).copied().collect(),
                ));
            }
        }
        for (effort, support) in &connection.effort.level_support {
            if *support == Unsupported {
                result.effort.level_support.insert(*effort, Unsupported);
            }
        }
        if connection.thinking == Unsupported {
            result.thinking = Unsupported;
        }
        if connection.budget.support == Unsupported {
            result.budget.support = Unsupported;
        }
        if connection.budget.dynamic == Unsupported {
            result.budget.dynamic = Unsupported;
        }
        if connection.effort.support == Unsupported {
            result.effort.support = Unsupported;
        }
        if result.effort.with_disabled_thinking == CapabilitySupport::Unknown
            || connection.effort.with_disabled_thinking == Unsupported
        {
            result.effort.with_disabled_thinking = connection.effort.with_disabled_thinking;
        }
        if connection.fast == Unsupported {
            result.fast = Unsupported;
        }
        if let Some(allowed) = &connection.modes {
            result.modes = Some(result.modes.as_ref().map_or_else(
                || allowed.clone(),
                |own| {
                    own.iter()
                        .filter(|v| allowed.contains(v))
                        .copied()
                        .collect()
                },
            ));
        }
        if let Some(allowed) = &connection.effort.levels {
            result.effort.levels = Some(result.effort.levels.as_ref().map_or_else(
                || allowed.clone(),
                |own| {
                    own.iter()
                        .filter(|v| allowed.contains(v))
                        .copied()
                        .collect()
                },
            ));
        }
        result.budget.min_tokens = match (result.budget.min_tokens, connection.budget.min_tokens) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        result.budget.max_tokens = match (result.budget.max_tokens, connection.budget.max_tokens) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        for value in &connection.requirements {
            if !result.requirements.contains(value) {
                result.requirements.push(value.clone());
            }
        }
        for value in &connection.sources {
            if !result.sources.contains(value) {
                result.sources.push(value.clone());
            }
        }
        result.restrict_to_modes();
        result
    }
}

fn update_allowed<T: Copy + PartialEq>(
    allowed: &mut Option<Vec<T>>,
    value: T,
    support: CapabilitySupport,
) {
    let Some(allowed) = allowed else { return };
    if support == CapabilitySupport::Unsupported {
        allowed.retain(|v| *v != value);
    } else if support == CapabilitySupport::Supported && !allowed.contains(&value) {
        allowed.push(value);
    }
}
