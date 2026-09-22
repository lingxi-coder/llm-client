//! The built-in provider presets.
//!
//! One file per provider under `data/providers/`, each carrying the route and
//! the models together. The file name is the profile name, so adding a provider
//! is adding a file — no central table, no code, which is what makes "add a
//! provider" a settings change by construction rather than by convention
//! (gate 30, enforced by `scripts/check-provider-names.sh`).
//!
//! Everything above the first `[[model]]` is hand-authored and carries the
//! reasons; the `[[model]]` blocks are generated from an upstream catalog by
//! `scripts/vendor-catalog.py`, which replaces only that section. Refreshing a
//! provider is therefore one command, and the comments survive it.
//!
//! The route stays hand-authored on purpose: a catalog's advisory endpoint
//! field is not authoritative about which wire an endpoint speaks. One vendor
//! here serves a protocol family its own slice describes as the other one.
//!
//! `model_list` is the other hand-authored route key that is not derivable:
//! which shape an endpoint publishes its model directory in, and `none` when it
//! publishes none. It defaults to `protocol`, which is right for nine of the
//! twelve presets here and wrong for the three that say otherwise — a wire is
//! not evidence about a list endpoint.
//!
//! The quirk flags a route may set, each carrying what the wire cannot express:
//!
//! - `preserve_reasoning_content` — thinking is on by default and the endpoint
//!   rejects a later turn that omits that turn's reasoning;
//! - `thinking_rejects_forced_tool_choice` — with thinking on, a forced tool
//!   choice comes back 400, while `none`, `auto` and the tools are accepted;
//! - `stream_usage_opt_in` — the terminal usage object is omitted from SSE
//!   unless it is asked for.
//! - `supports_previous_response_id` — this Responses endpoint persists state
//!   and honors the typed continuation id instead of accepting and ignoring it.

use lingxi_agent_api::protocol::{
    AuthStrategy, BillingMode, ConnectionSpec, CredentialConfig, DirectoryRoute, ModelCapabilities,
    ModelMetadata, ModelProfile, PeakSchedule, PricingConfig, ProtocolFamily, ProviderId,
    ProviderInfo, ProviderProfile, TokenPricing,
};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

// Generated from the directory listing; see build.rs.
include!(concat!(env!("OUT_DIR"), "/presets.rs"));

/// One preset file: the route, then the models.
#[derive(Debug, Deserialize)]
struct Preset {
    provider_id: ProviderId,
    base_url: String,
    protocol: ProtocolFamily,
    /// Which shape this endpoint publishes its model list in, when that is not
    /// the wire it speaks otherwise. A flat key for the same reason as the six
    /// below.
    #[serde(default)]
    model_list: DirectoryRoute,
    auth: AuthStrategy,
    credential_env: String,
    billing_mode: BillingMode,
    /// Presentation and set-up links. Flat keys, not a `[info]` table: TOML
    /// binds every bare key after a table header into that table, and the route
    /// header has bare keys after these.
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    console_url: Option<String>,
    #[serde(default)]
    api_key_url: Option<String>,
    #[serde(default)]
    docs_url: Option<String>,
    #[serde(default)]
    credential_hint: Option<String>,
    /// The vendor's billing schedule, if it charges less outside peak hours.
    #[serde(default)]
    pricing: RoutePricing,
    #[serde(default)]
    connection: ConnectionSpec,
    #[serde(default)]
    extra: Value,
    #[serde(default)]
    model: Vec<CatalogModel>,
}

/// The provider-level half of pricing: when the listed rates apply.
#[derive(Debug, Default, Deserialize)]
struct RoutePricing {
    #[serde(default)]
    peak: Option<PeakSchedule>,
}

/// One generated `[[model]]` block. Every field beyond the id is optional:
/// absent upstream stays absent rather than becoming an invented default that
/// would silently drive the compaction threshold.
#[derive(Debug, Deserialize)]
struct CatalogModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    knowledge_cutoff: Option<String>,
    #[serde(default)]
    release_date: Option<String>,
    #[serde(default)]
    last_updated: Option<String>,
    #[serde(default)]
    context_window: Option<u64>,
    #[serde(default)]
    max_input: Option<u64>,
    #[serde(default)]
    max_output: Option<u64>,
    #[serde(default)]
    input_modalities: Vec<String>,
    #[serde(default)]
    output_modalities: Vec<String>,
    #[serde(default)]
    tool_call: bool,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    structured_output: bool,
    #[serde(default)]
    temperature: Option<bool>,
    #[serde(default)]
    attachment: Option<bool>,
    #[serde(default)]
    open_weights: Option<bool>,
    /// Published prices, carried through from the catalog.
    #[serde(default)]
    pricing: Option<TokenPricing>,
    /// Set only when this model is billed differently from the rest of the
    /// connection — an aggregator's free tier alongside its metered one.
    #[serde(default)]
    billing_mode: Option<BillingMode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PresetError {
    #[error("the preset for {profile_name:?} is not valid: {message}")]
    Invalid {
        profile_name: String,
        message: String,
    },
    #[error("the preset for {profile_name:?} declares no model, so nothing can resolve to it")]
    NoModels { profile_name: String },
}

/// Every preset, by file name.
pub fn builtin() -> Result<Vec<ProviderProfile>, PresetError> {
    PRESETS
        .iter()
        .map(|(name, text)| parse(name, text))
        .collect()
}

fn parse(profile_name: &str, text: &str) -> Result<ProviderProfile, PresetError> {
    let p: Preset = ::toml::from_str(text).map_err(|e| PresetError::Invalid {
        profile_name: profile_name.to_owned(),
        message: e.to_string(),
    })?;
    if p.model.is_empty() {
        return Err(PresetError::NoModels {
            profile_name: profile_name.to_owned(),
        });
    }
    Ok(ProviderProfile {
        provider_id: p.provider_id,
        profile_name: profile_name.to_owned(),
        base_url: p.base_url,
        protocol: p.protocol,
        model_list: p.model_list,
        auth: p.auth,
        credential: CredentialConfig::Env {
            var: p.credential_env,
        },
        models: disambiguate(p.model.iter().map(model_profile).collect()),
        pricing: PricingConfig {
            billing_mode: p.billing_mode,
            peak: p.pricing.peak.clone(),
            ..PricingConfig::default()
        },
        signing: None,
        azure: None,
        supports_websockets: false,
        supports_websocket_compression: false,
        websocket_connect_timeout_ms: None,
        vision_delegate: None,
        connection: p.connection,
        info: ProviderInfo {
            display_name: p.display_name,
            description: p.description,
            console_url: p.console_url,
            api_key_url: p.api_key_url,
            docs_url: p.docs_url,
            credential_hint: p.credential_hint,
        },
        extra: p.extra,
    })
}

/// The presets a user has not replaced, merged under their own entries.
///
/// A user entry wins on profile name: someone who redeclares a preset with a
/// different base URL means it, and keeping the preset beside theirs would
/// leave the group's failover chain reaching an endpoint they removed.
pub fn merge(user: Vec<ProviderProfile>) -> Result<Vec<ProviderProfile>, PresetError> {
    let named: std::collections::BTreeSet<_> =
        user.iter().map(|p| p.profile_name.clone()).collect();
    let mut out = user;
    out.extend(
        builtin()?
            .into_iter()
            .filter(|p| !named.contains(&p.profile_name)),
    );
    Ok(out)
}

/// Give a model back its wire id as a display name when the catalog hands two
/// models the same one.
///
/// The catalog names a model and its preview twin identically. That makes the
/// listing row ambiguous — two rows read the same — and it makes the name
/// unresolvable, because `resolve_in` refuses a ref matching more than one model
/// on a profile rather than guessing. The wire id is unique by construction, so
/// the colliding pair falls back to it and everything else keeps its name.
fn disambiguate(mut models: Vec<ModelProfile>) -> Vec<ModelProfile> {
    let mut seen: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for m in &models {
        *seen.entry(m.display_model.clone()).or_default() += 1;
    }
    for m in &mut models {
        if seen.get(&m.display_model).is_some_and(|n| *n > 1) {
            m.display_model.clone_from(&m.request_model);
        }
    }
    models
}

fn model_profile(m: &CatalogModel) -> ModelProfile {
    let accepts = |what: &str| m.input_modalities.iter().any(|i| i == what);
    ModelProfile {
        display_model: m.name.clone().unwrap_or_else(|| m.id.clone()),
        request_model: m.id.clone(),
        billing_model: m.id.clone(),
        aliases: Vec::new(),
        description: m.description.clone(),
        metadata: ModelMetadata {
            family: m.family.clone(),
            status: None,
            release_date: m.release_date.clone(),
            last_updated: m.last_updated.clone(),
            knowledge_cutoff: m.knowledge_cutoff.clone(),
            input_modalities: m.input_modalities.clone(),
            output_modalities: m.output_modalities.clone(),
            context_window_tokens: m.context_window,
            max_input_tokens: m.max_input,
            max_output_tokens: m.max_output,
            open_weights: m.open_weights,
            attachments: m.attachment,
            temperature_control: m.temperature,
        },
        pricing: m.pricing.clone(),
        billing_mode: m.billing_mode,
        capabilities: ModelCapabilities {
            streaming: true,
            tools: m.tool_call,
            vision: accepts("image"),
            documents: accepts("pdf") || accepts("file"),
            reasoning: m.reasoning,
            structured_output: m.structured_output,
            signed_reasoning: false,
        },
    }
}
