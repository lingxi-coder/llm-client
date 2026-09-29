//! Provider and model reasoning controls, derived from SDK presets.

use crate::protocol::ProtocolFamily;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningSelection {
    Automatic,
    Disabled,
    Enabled,
    Level(String),
    TokenBudget(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenBudgetRange {
    pub min: u32,
    pub max: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReasoningControlSpec {
    pub levels: Vec<String>,
    pub token_budget: Option<TokenBudgetRange>,
    pub can_disable: bool,
    pub can_enable: bool,
    pub mandatory_selection: Option<ReasoningSelection>,
}

impl ReasoningControlSpec {
    #[must_use]
    pub fn automatic_only() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn supports(&self, selection: &ReasoningSelection) -> bool {
        // Automatic never sends an override; provider defaults remain
        // authoritative even for models that mandate internal reasoning.
        if matches!(selection, ReasoningSelection::Automatic) {
            return true;
        }
        if let Some(mandatory) = &self.mandatory_selection {
            return mandatory == selection;
        }
        match selection {
            ReasoningSelection::Automatic => true,
            ReasoningSelection::Disabled => self.can_disable,
            ReasoningSelection::Enabled => self.can_enable,
            ReasoningSelection::Level(level) => self.levels.iter().any(|item| item == level),
            ReasoningSelection::TokenBudget(tokens) => self
                .token_budget
                .is_some_and(|range| *tokens >= range.min && *tokens <= range.max),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReasoningTarget<'a> {
    pub profile_name: Option<&'a str>,
    pub protocol: &'a ProtocolFamily,
    pub base_url: &'a str,
    pub model: &'a str,
}

/// Apply a validated user selection to canonical model input. An unsupported
/// selection leaves the request untouched; Automatic removes any override.
pub fn apply_reasoning_selection(
    request: &mut crate::protocol::ChatRequest,
    target: ReasoningTarget<'_>,
    selection: ReasoningSelection,
) -> Result<(), String> {
    use crate::protocol::{ReasoningEffort, ThinkingBudget, ThinkingConfig, ThinkingMode};
    validate_reasoning_selection(target, &selection)?;
    request.thinking = match selection {
        ReasoningSelection::Automatic => None,
        ReasoningSelection::Disabled => Some(ThinkingConfig {
            mode: Some(ThinkingMode::Disabled),
            ..Default::default()
        }),
        ReasoningSelection::Enabled => Some(ThinkingConfig {
            mode: Some(ThinkingMode::Enabled),
            ..Default::default()
        }),
        ReasoningSelection::Level(level) => {
            let effort = ReasoningEffort::ALL
                .into_iter()
                .find(|effort| effort.as_str() == level)
                .ok_or_else(|| format!("unsupported reasoning effort {level}"))?;
            Some(ThinkingConfig {
                effort: Some(effort),
                ..Default::default()
            })
        }
        ReasoningSelection::TokenBudget(tokens) => Some(ThinkingConfig {
            mode: Some(ThinkingMode::Enabled),
            budget: Some(ThinkingBudget::Tokens(tokens)),
            ..Default::default()
        }),
    };
    Ok(())
}

/// Validate a user selection without mutating a request.
pub fn validate_reasoning_selection(
    target: ReasoningTarget<'_>,
    selection: &ReasoningSelection,
) -> Result<(), String> {
    if reasoning_control_spec(target).supports(selection) {
        Ok(())
    } else {
        Err(format!(
            "reasoning selection {selection:?} is unsupported for model {}",
            target.model
        ))
    }
}

pub fn reasoning_control_spec(target: ReasoningTarget<'_>) -> ReasoningControlSpec {
    if let Some(spec) = match target.profile_name {
        Some("zai") => Some(catalog_control_spec(zai_catalog(), target.model)),
        Some("glm-coding") => Some(catalog_control_spec(glm_coding_catalog(), target.model)),
        Some("github-copilot") => {
            Some(catalog_control_spec(github_copilot_catalog(), target.model))
        }
        _ => None,
    } {
        return spec;
    }
    match target.protocol {
        ProtocolFamily::AnthropicMessages
        | ProtocolFamily::BedrockClaude
        | ProtocolFamily::FoundryClaude
        | ProtocolFamily::VertexClaude => anthropic_spec(target.model),
        ProtocolFamily::OpenAiResponses => openai_responses_spec(target.base_url, target.model),
        ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini => {
            gemini_spec(target.model)
        }
        ProtocolFamily::OpenAiChat => {
            openai_chat_spec(target.profile_name, target.base_url, target.model)
        }
        ProtocolFamily::AzureOpenAi => ReasoningControlSpec::automatic_only(),
    }
}

fn catalog_control_spec(
    catalog: &HashMap<String, CatalogReasoningSpec>,
    model: &str,
) -> ReasoningControlSpec {
    let Some(spec) = catalog.get(model) else {
        return ReasoningControlSpec::automatic_only();
    };
    if !spec.reasoning {
        return ReasoningControlSpec::automatic_only();
    }
    let can_disable = spec.toggle || spec.levels.iter().any(|level| level == "none");
    ReasoningControlSpec {
        levels: spec
            .levels
            .iter()
            .filter(|level| level.as_str() != "none")
            .cloned()
            .collect(),
        token_budget: spec.token_budget,
        can_disable,
        can_enable: spec.toggle && spec.levels.is_empty() && spec.token_budget.is_none(),
        mandatory_selection: None,
    }
}

#[derive(Debug, Clone)]
struct CatalogReasoningSpec {
    reasoning: bool,
    levels: Vec<String>,
    token_budget: Option<TokenBudgetRange>,
    toggle: bool,
}

fn anthropic_catalog() -> &'static HashMap<String, CatalogReasoningSpec> {
    static CATALOG: OnceLock<HashMap<String, CatalogReasoningSpec>> = OnceLock::new();
    CATALOG.get_or_init(|| {
        let mut catalog = upstream_catalog("anthropic");
        if let Some(fable) = catalog.get("claude-fable-5-1").cloned() {
            catalog.insert("claude-mythos-5-1".into(), fable);
        }
        catalog
    })
}

fn anthropic_spec(model: &str) -> ReasoningControlSpec {
    let canonical = normalize_model_id(model);
    // These historical host controls offered effort only, not token budgets
    // or mode toggles. Keep that user-facing contract as catalogs evolve.
    if canonical == "claude-opus-4-5" || canonical.starts_with("claude-opus-4-5-") {
        return ReasoningControlSpec::automatic_only();
    }
    let canonical = if canonical == "claude-mythos-5-1" {
        "claude-fable-5-1"
    } else {
        &canonical
    };
    anthropic_catalog().get(canonical).map_or_else(
        ReasoningControlSpec::automatic_only,
        catalog_spec_for_openai,
    )
}

fn known_wrapper_suffix(suffix: &str) -> bool {
    if suffix.is_empty() || suffix == "-eap" {
        return true;
    }
    if let Some(date) = suffix.strip_prefix('@') {
        return date.len() == 8 && date.bytes().all(|b| b.is_ascii_digit());
    }
    if let Some(version) = suffix.strip_prefix("-v") {
        let mut parts = version.split(':');
        return parts
            .next()
            .is_some_and(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
            && parts
                .next()
                .is_some_and(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
            && parts.next().is_none();
    }
    if let Some(date) = suffix.strip_prefix('-') {
        if date.len() == 8 && date.bytes().all(|b| b.is_ascii_digit()) {
            return true;
        }
        if let Some((date, version)) = date.split_once("-v") {
            let mut version_parts = version.split(':');
            return date.len() == 8
                && date.bytes().all(|b| b.is_ascii_digit())
                && version_parts.next().is_some_and(|part| {
                    !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit())
                })
                && version_parts.next().is_some_and(|part| {
                    !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit())
                })
                && version_parts.next().is_none();
        }
    }
    false
}

fn claude_transport_candidate(model_id: &str) -> Option<&str> {
    if model_id.starts_with("claude-") {
        return Some(model_id);
    }
    if let Some(candidate) = model_id.strip_prefix("anthropic.") {
        return candidate.starts_with("claude-").then_some(candidate);
    }
    let (region, candidate) = model_id.split_once(".anthropic.")?;
    matches!(region, "us" | "eu" | "apac" | "global")
        .then_some(candidate)
        .filter(|candidate| candidate.starts_with("claude-"))
}

/// Strip the decorations a model id can carry before a registry lookup:
/// a provider/profile prefix (`openrouter/anthropic/…`), a `[1m]` context
/// suffix, an `-eap` early-access suffix, and the standard dated/cloud
/// transport wrappers around a canonical Claude id.
#[must_use]
fn normalize_model_id(model_id: &str) -> String {
    let bare = model_id.rsplit('/').next().unwrap_or(model_id);
    let bare = bare.split('[').next().unwrap_or(bare);
    let lower = bare.trim().to_ascii_lowercase();
    let Some(candidate) = claude_transport_candidate(&lower) else {
        return lower;
    };
    let without_eap = candidate.strip_suffix("-eap").unwrap_or(candidate);
    anthropic_catalog()
        .keys()
        .find(|known| {
            without_eap
                .strip_prefix(known.as_str())
                .is_some_and(known_wrapper_suffix)
        })
        .map_or_else(|| without_eap.to_string(), |known| known.to_string())
}

fn openai_responses_spec(base_url: &str, model: &str) -> ReasoningControlSpec {
    if base_url.contains("chatgpt.com/backend-api/codex") {
        return match model {
            "gpt-5-codex" | "gpt-5.3-codex" => ReasoningControlSpec {
                levels: vec![
                    "minimal".to_string(),
                    "low".to_string(),
                    "medium".to_string(),
                    "high".to_string(),
                ],
                token_budget: None,
                can_disable: false,
                can_enable: false,
                mandatory_selection: None,
            },
            _ => openai_catalog().get(model).map_or_else(
                ReasoningControlSpec::automatic_only,
                catalog_spec_for_openai,
            ),
        };
    }

    if let Some(spec) = openai_catalog().get(model) {
        return catalog_spec_for_openai(spec);
    }

    if base_url.contains("aliyuncs.com") && model.starts_with("qwen") {
        return ReasoningControlSpec {
            levels: vec![
                "minimal".to_string(),
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
            ],
            token_budget: None,
            can_disable: true,
            can_enable: false,
            mandatory_selection: None,
        };
    }

    ReasoningControlSpec::automatic_only()
}

fn catalog_spec_for_openai(spec: &CatalogReasoningSpec) -> ReasoningControlSpec {
    if !spec.reasoning || spec.levels.is_empty() {
        return ReasoningControlSpec::automatic_only();
    }
    let can_disable = spec.levels.iter().any(|level| level == "none");
    let levels = spec
        .levels
        .iter()
        .filter(|level| level.as_str() != "none")
        .cloned()
        .collect();
    ReasoningControlSpec {
        levels,
        token_budget: None,
        can_disable,
        can_enable: false,
        mandatory_selection: None,
    }
}

fn gemini_spec(model: &str) -> ReasoningControlSpec {
    let Some(spec) = gemini_catalog().get(model) else {
        return ReasoningControlSpec::automatic_only();
    };
    if !spec.reasoning {
        return ReasoningControlSpec::automatic_only();
    }
    ReasoningControlSpec {
        levels: spec.levels.clone(),
        token_budget: spec.token_budget,
        can_disable: spec.toggle && spec.token_budget.is_some(),
        can_enable: spec.toggle && spec.levels.is_empty() && spec.token_budget.is_none(),
        mandatory_selection: None,
    }
}

fn openai_chat_spec(
    profile_name: Option<&str>,
    base_url: &str,
    model: &str,
) -> ReasoningControlSpec {
    if is_openrouter_profile(profile_name, base_url) {
        return openrouter_spec(model);
    }
    if is_deepseek_profile(profile_name, base_url) {
        return deepseek_spec(model);
    }
    if is_kimi_profile(profile_name) {
        return kimi_spec(model);
    }
    ReasoningControlSpec::automatic_only()
}

fn openrouter_spec(model: &str) -> ReasoningControlSpec {
    let Some(spec) = openrouter_catalog().get(model) else {
        return ReasoningControlSpec::automatic_only();
    };
    if !spec.reasoning {
        return ReasoningControlSpec::automatic_only();
    }
    if spec.levels.is_empty() && spec.token_budget.is_none() && !spec.toggle {
        return ReasoningControlSpec::automatic_only();
    }
    let can_disable = spec.toggle || spec.levels.iter().any(|level| level == "none");
    let levels = spec
        .levels
        .iter()
        .filter(|level| level.as_str() != "none")
        .cloned()
        .collect();
    ReasoningControlSpec {
        levels,
        token_budget: spec.token_budget,
        can_disable,
        can_enable: spec.toggle && spec.levels.is_empty() && spec.token_budget.is_none(),
        mandatory_selection: None,
    }
}

fn deepseek_spec(model: &str) -> ReasoningControlSpec {
    match model {
        "deepseek-reasoner" => ReasoningControlSpec {
            levels: Vec::new(),
            token_budget: None,
            can_disable: false,
            can_enable: false,
            mandatory_selection: Some(ReasoningSelection::Enabled),
        },
        _ => {
            let Some(spec) = deepseek_catalog().get(model) else {
                return ReasoningControlSpec::automatic_only();
            };
            if !spec.reasoning {
                return ReasoningControlSpec::automatic_only();
            }
            ReasoningControlSpec {
                levels: spec.levels.clone(),
                token_budget: None,
                can_disable: spec.toggle,
                can_enable: false,
                mandatory_selection: None,
            }
        }
    }
}

fn kimi_spec(model: &str) -> ReasoningControlSpec {
    match model {
        "kimi-k2-thinking"
        | "k2-thinking"
        | "kimi-k2-thinking-preview"
        | "kimi-k2.7-code"
        | "kimi-for-coding"
        | "kimi-for-coding-highspeed" => ReasoningControlSpec {
            levels: Vec::new(),
            token_budget: None,
            can_disable: false,
            can_enable: false,
            mandatory_selection: Some(ReasoningSelection::Enabled),
        },
        _ => {
            let spec = kimi_catalog()
                .get(model)
                .or_else(|| kimi_code_catalog().get(model));
            let Some(spec) = spec else {
                return ReasoningControlSpec::automatic_only();
            };
            if !spec.reasoning {
                return ReasoningControlSpec::automatic_only();
            }
            let can_enable = spec.toggle && spec.levels.is_empty() && spec.token_budget.is_none();
            // K3 exposes discrete effort levels, not a separate off switch.
            // Only toggle-only K2.x entries may advertise Disable.
            let can_disable = spec.toggle && spec.levels.is_empty() && spec.token_budget.is_none();
            ReasoningControlSpec {
                levels: spec.levels.clone(),
                token_budget: None,
                can_disable,
                can_enable,
                mandatory_selection: None,
            }
        }
    }
}

pub(crate) fn is_openrouter_profile(profile_name: Option<&str>, base_url: &str) -> bool {
    profile_name == Some("openrouter") || base_url.contains("openrouter.ai")
}

pub(crate) fn is_deepseek_profile(profile_name: Option<&str>, base_url: &str) -> bool {
    profile_name == Some("deepseek")
        || matches!(
            base_url.trim_end_matches('/'),
            "https://api.deepseek.com" | "https://api.deepseek.com/v1"
        )
}

pub(crate) fn is_kimi_profile(profile_name: Option<&str>) -> bool {
    matches!(profile_name, Some("kimi" | "kimi-code"))
}

fn openai_catalog() -> &'static HashMap<String, CatalogReasoningSpec> {
    static OPENAI: OnceLock<HashMap<String, CatalogReasoningSpec>> = OnceLock::new();
    OPENAI.get_or_init(|| upstream_catalog("openai"))
}

fn gemini_catalog() -> &'static HashMap<String, CatalogReasoningSpec> {
    static GEMINI: OnceLock<HashMap<String, CatalogReasoningSpec>> = OnceLock::new();
    GEMINI.get_or_init(|| upstream_catalog("gemini"))
}

fn deepseek_catalog() -> &'static HashMap<String, CatalogReasoningSpec> {
    static DEEPSEEK: OnceLock<HashMap<String, CatalogReasoningSpec>> = OnceLock::new();
    DEEPSEEK.get_or_init(|| upstream_catalog("deepseek"))
}

fn kimi_catalog() -> &'static HashMap<String, CatalogReasoningSpec> {
    static KIMI: OnceLock<HashMap<String, CatalogReasoningSpec>> = OnceLock::new();
    KIMI.get_or_init(|| upstream_catalog("kimi"))
}

fn kimi_code_catalog() -> &'static HashMap<String, CatalogReasoningSpec> {
    static KIMI_CODE: OnceLock<HashMap<String, CatalogReasoningSpec>> = OnceLock::new();
    KIMI_CODE.get_or_init(|| upstream_catalog("kimi-code"))
}

fn openrouter_catalog() -> &'static HashMap<String, CatalogReasoningSpec> {
    static OPENROUTER: OnceLock<HashMap<String, CatalogReasoningSpec>> = OnceLock::new();
    OPENROUTER.get_or_init(|| upstream_catalog("openrouter"))
}

fn zai_catalog() -> &'static HashMap<String, CatalogReasoningSpec> {
    static CATALOG: OnceLock<HashMap<String, CatalogReasoningSpec>> = OnceLock::new();
    CATALOG.get_or_init(|| upstream_catalog("zai"))
}

fn glm_coding_catalog() -> &'static HashMap<String, CatalogReasoningSpec> {
    static CATALOG: OnceLock<HashMap<String, CatalogReasoningSpec>> = OnceLock::new();
    CATALOG.get_or_init(|| upstream_catalog("glm-coding"))
}

fn github_copilot_catalog() -> &'static HashMap<String, CatalogReasoningSpec> {
    static CATALOG: OnceLock<HashMap<String, CatalogReasoningSpec>> = OnceLock::new();
    CATALOG.get_or_init(|| upstream_catalog("github-copilot"))
}

fn upstream_catalog(profile_name: &str) -> HashMap<String, CatalogReasoningSpec> {
    use crate::protocol::{CapabilitySupport, ThinkingMode};
    let Some(profile) = crate::builtin_providers()
        .expect("pinned catalog parses")
        .into_iter()
        .find(|p| p.profile_name == profile_name)
    else {
        return HashMap::new();
    };
    profile
        .models
        .iter()
        .flat_map(|model| {
            let features = model.info.features.on_connection(&profile.info.features);
            let levels = features
                .effort
                .levels
                .unwrap_or_default()
                .into_iter()
                .filter_map(|effort| {
                    serde_json::to_value(effort)
                        .ok()?
                        .as_str()
                        .map(str::to_owned)
                })
                .collect();
            let token_budget = match (features.budget.min_tokens, features.budget.max_tokens) {
                (Some(min), Some(max))
                    if features.budget.support == CapabilitySupport::Supported =>
                {
                    Some(TokenBudgetRange { min, max })
                }
                _ => None,
            };
            let toggle = features.modes.as_ref().is_some_and(|m| {
                m.contains(&ThinkingMode::Enabled) && m.contains(&ThinkingMode::Disabled)
            });
            let spec = CatalogReasoningSpec {
                reasoning: features.thinking == CapabilitySupport::Supported,
                levels,
                token_budget,
                toggle,
            };
            std::iter::once(model.request_model.clone())
                .chain(model.aliases.iter().cloned())
                .map(move |id| (id, spec.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_changes_only_typed_thinking_after_validation() {
        use crate::protocol::{
            ChatRequest, ReasoningEffort, ThinkingBudget, ThinkingConfig, ThinkingMode,
        };
        let mut request = ChatRequest::new("gpt-5");
        let target = ReasoningTarget {
            profile_name: None,
            protocol: &ProtocolFamily::OpenAiResponses,
            base_url: "https://api.openai.com/v1",
            model: "gpt-5",
        };
        apply_reasoning_selection(
            &mut request,
            target,
            ReasoningSelection::Level("high".into()),
        )
        .unwrap();
        assert_eq!(
            request.thinking,
            Some(ThinkingConfig {
                effort: Some(ReasoningEffort::High),
                ..Default::default()
            })
        );
        let before = request.clone();
        assert!(apply_reasoning_selection(
            &mut request,
            target,
            ReasoningSelection::Level("unknown".into())
        )
        .is_err());
        assert_eq!(request, before);
        apply_reasoning_selection(&mut request, target, ReasoningSelection::Automatic).unwrap();
        assert_eq!(request, ChatRequest::new("gpt-5"));
        let target = ReasoningTarget {
            profile_name: Some("gemini"),
            protocol: &ProtocolFamily::GeminiGenerateContent,
            base_url: "https://generativelanguage.googleapis.com",
            model: "gemini-2.5-flash",
        };
        apply_reasoning_selection(&mut request, target, ReasoningSelection::TokenBudget(1024))
            .unwrap();
        assert_eq!(
            request.thinking,
            Some(ThinkingConfig {
                mode: Some(ThinkingMode::Enabled),
                budget: Some(ThinkingBudget::Tokens(1024)),
                effort: None
            })
        );
        apply_reasoning_selection(&mut request, target, ReasoningSelection::Disabled).unwrap();
        assert_eq!(
            request.thinking,
            Some(ThinkingConfig {
                mode: Some(ThinkingMode::Disabled),
                ..Default::default()
            })
        );
    }

    #[test]
    fn anthropic_transport_wrappers_keep_catalog_levels() {
        for model in [
            "CLAUDE-OPUS-5",
            "claude-opus-5[1m]",
            "claude-opus-5-eap",
            "openrouter/anthropic/claude-opus-5",
            "  claude-opus-5  ",
            "claude-opus-5@20260728",
            "us.anthropic.claude-opus-5-20260728-v1:0",
        ] {
            assert_eq!(
                anthropic_spec(model),
                anthropic_spec("claude-opus-5"),
                "{model}"
            );
        }
        assert_eq!(
            anthropic_spec("eu.anthropic.claude-mythos-5-1-v1:0"),
            anthropic_spec("claude-fable-5-1")
        );
        for model in [
            "vendor-compat-claude-opus-5",
            "not-anthropic.claude-opus-5",
            "claude-opus-5-custom",
            "claude-unknown",
        ] {
            assert_eq!(
                anthropic_spec(model),
                ReasoningControlSpec::automatic_only(),
                "{model}"
            );
        }
        assert_eq!(
            anthropic_spec("claude-opus-4-5"),
            ReasoningControlSpec::automatic_only()
        );
    }

    #[test]
    fn validation_respects_automatic_mandatory_and_budget_bounds() {
        let target = ReasoningTarget {
            profile_name: Some("deepseek"),
            protocol: &ProtocolFamily::OpenAiChat,
            base_url: "https://api.deepseek.com",
            model: "deepseek-reasoner",
        };
        assert!(validate_reasoning_selection(target, &ReasoningSelection::Automatic).is_ok());
        assert!(validate_reasoning_selection(target, &ReasoningSelection::Enabled).is_ok());
        assert!(validate_reasoning_selection(target, &ReasoningSelection::Disabled).is_err());
        let target = ReasoningTarget {
            profile_name: Some("gemini"),
            protocol: &ProtocolFamily::GeminiGenerateContent,
            base_url: "https://generativelanguage.googleapis.com",
            model: "gemini-2.5-flash",
        };
        for tokens in [1, 24_576] {
            assert!(
                validate_reasoning_selection(target, &ReasoningSelection::TokenBudget(tokens))
                    .is_ok()
            );
        }
        for tokens in [0, 24_577] {
            assert!(
                validate_reasoning_selection(target, &ReasoningSelection::TokenBudget(tokens))
                    .is_err()
            );
        }
    }

    #[test]
    fn anthropic_capability_levels_follow_model_registry() {
        let spec = anthropic_spec("claude-opus-5");
        assert_eq!(spec.levels, vec!["low", "medium", "high", "xhigh", "max"]);
        assert!(!spec.can_disable);

        let unknown = anthropic_spec("claude-unknown");
        assert_eq!(unknown, ReasoningControlSpec::automatic_only());
    }

    #[test]
    fn openai_and_chatgpt_codex_specs_are_provider_aware() {
        let openai = openai_responses_spec("https://api.openai.com/v1", "gpt-5");
        assert_eq!(openai.levels, vec!["minimal", "low", "medium", "high"]);
        assert!(!openai.can_disable);

        let chatgpt = openai_responses_spec("https://chatgpt.com/backend-api/codex", "gpt-5-codex");
        assert_eq!(chatgpt.levels, vec!["minimal", "low", "medium", "high"]);
    }

    #[test]
    fn gemini_spec_distinguishes_budget_and_level_models() {
        let budget = gemini_spec("gemini-2.5-flash");
        assert_eq!(
            budget.token_budget,
            Some(TokenBudgetRange {
                min: 1,
                max: 24_576
            })
        );
        assert!(budget.can_disable);
        assert!(budget.levels.is_empty());

        let levels = gemini_spec("gemini-3.1-pro-preview");
        assert_eq!(levels.levels, vec!["low", "medium", "high"]);
        assert!(!levels.can_disable);
    }

    #[test]
    fn deepseek_and_kimi_specs_only_expose_verified_controls() {
        let deepseek = deepseek_spec("deepseek-flash");
        assert_eq!(deepseek.levels, vec!["low", "high", "max"]);
        assert!(deepseek.can_disable);
        assert!(!deepseek.can_enable);

        let kimi_k3 = kimi_spec("kimi-k3");
        assert_eq!(kimi_k3.levels, vec!["low", "high", "max"]);
        assert!(!kimi_k3.can_disable);

        let kimi_k27 = kimi_spec("kimi-k2.7-code");
        assert_eq!(
            kimi_k27.mandatory_selection,
            Some(ReasoningSelection::Enabled)
        );
    }

    #[test]
    fn openrouter_spec_uses_catalog_reasoning_options() {
        let spec = openrouter_spec("google/gemini-3.5-flash");
        assert!(
            spec.levels.is_empty(),
            "upstream does not publish effort levels for this route"
        );
        assert!(!spec.can_disable);

        let unknown = openrouter_spec("openrouter/auto");
        assert_eq!(unknown, ReasoningControlSpec::automatic_only());
    }

    #[test]
    fn subscription_and_glm_profiles_use_their_route_catalog() {
        let protocol = ProtocolFamily::AnthropicMessages;
        let glm = reasoning_control_spec(ReasoningTarget {
            profile_name: Some("glm-coding"),
            protocol: &protocol,
            base_url: "https://open.bigmodel.cn/api/anthropic",
            model: "glm-5.3",
        });
        assert_eq!(glm.levels, vec!["low", "high", "max"]);

        let chat = ProtocolFamily::OpenAiChat;
        let copilot = reasoning_control_spec(ReasoningTarget {
            profile_name: Some("github-copilot"),
            protocol: &chat,
            base_url: "https://api.githubcopilot.com",
            model: "gpt-5.6-sol",
        });
        assert_eq!(
            copilot,
            ReasoningControlSpec::automatic_only(),
            "unknown upstream controls must not be inferred from the model name"
        );
    }
}
