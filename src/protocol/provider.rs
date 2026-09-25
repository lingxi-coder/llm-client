//! Connection configuration and provider metadata.
//!
//! `provider_id` is an open string, allowing callers to configure additional
//! providers that use an existing protocol family.

use crate::protocol::{
    CapabilitySupport, ModelCapability, ModelCapabilitySupport, ProviderId, Secret,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Closed set of 9. Adding a family means adding a codec crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolFamily {
    AnthropicMessages,
    OpenAiChat,
    OpenAiResponses,
    GeminiGenerateContent,
    VertexClaude,
    VertexGemini,
    BedrockClaude,
    AzureOpenAi,
    FoundryClaude,
}

impl ProtocolFamily {
    pub const ALL: [ProtocolFamily; 9] = [
        ProtocolFamily::AnthropicMessages,
        ProtocolFamily::OpenAiChat,
        ProtocolFamily::OpenAiResponses,
        ProtocolFamily::GeminiGenerateContent,
        ProtocolFamily::VertexClaude,
        ProtocolFamily::VertexGemini,
        ProtocolFamily::BedrockClaude,
        ProtocolFamily::AzureOpenAi,
        ProtocolFamily::FoundryClaude,
    ];
}

/// Which wire a connection's *model directory* speaks, when it publishes one.
///
/// Separate from `protocol` because the two disagree in the shipped data, in
/// both directions. One connection speaks one wire for completions while the
/// same host publishes its model list in another wire's shape; another
/// publishes no list at all. Assuming the completion wire would run the wrong
/// parser over a real 200 and produce a plausible list with every field
/// silently absent — worse than failing, because nothing downstream can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DirectoryRoute {
    /// Not stated: the directory speaks the same wire as the rest of the
    /// endpoint, which is true of most connections.
    #[default]
    SameAsProtocol,
    /// This endpoint publishes no directory. Distinct from not stating one: a
    /// refresh must not fall back to a path that answers 404, which reads as
    /// "this provider withdrew every model".
    NotPublished,
    /// The directory speaks this wire, whatever the endpoint speaks otherwise.
    Shape(ProtocolFamily),
}

impl DirectoryRoute {
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::SameAsProtocol
    }

    /// Which shape to decode a directory page with, given what this endpoint
    /// speaks otherwise. `None` when it publishes no directory at all.
    #[must_use]
    pub fn shape(self, protocol: ProtocolFamily) -> Option<ProtocolFamily> {
        match self {
            Self::SameAsProtocol => Some(protocol),
            Self::NotPublished => None,
            Self::Shape(family) => Some(family),
        }
    }
}

impl<'de> Deserialize<'de> for DirectoryRoute {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        if raw == "same_as_protocol" {
            return Ok(Self::SameAsProtocol);
        }
        if raw == "none" {
            return Ok(Self::NotPublished);
        }
        ProtocolFamily::deserialize(serde::de::value::StrDeserializer::<D::Error>::new(&raw))
            .map(Self::Shape)
            .map_err(|e: D::Error| {
                serde::de::Error::custom(format!(
                    "{e}, or \"none\" when the endpoint publishes no model directory"
                ))
            })
    }
}

impl Serialize for DirectoryRoute {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::SameAsProtocol => serializer.serialize_str("same_as_protocol"),
            Self::NotPublished => serializer.serialize_str("none"),
            Self::Shape(family) => family.serialize(serializer),
        }
    }
}

#[cfg(test)]
mod directory_route_tests {
    use super::{DirectoryRoute, ProtocolFamily};

    #[test]
    fn serialized_routes_round_trip() {
        for route in [
            DirectoryRoute::SameAsProtocol,
            DirectoryRoute::NotPublished,
            DirectoryRoute::Shape(ProtocolFamily::OpenAiChat),
        ] {
            let json = serde_json::to_string(&route).unwrap();
            assert_eq!(
                serde_json::from_str::<DirectoryRoute>(&json).unwrap(),
                route
            );
        }
    }
}

/// Supported credential schemes; hosts register cloud authenticators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthStrategy {
    ApiKey,
    Bearer,
    OAuthBearer,
    CopilotBearer,
    ChatGptOAuth,
    GcpToken,
    AwsSigV4,
    AzureToken,
    None,
}

impl AuthStrategy {
    pub const ALL: [AuthStrategy; 9] = [
        AuthStrategy::ApiKey,
        AuthStrategy::Bearer,
        AuthStrategy::OAuthBearer,
        AuthStrategy::CopilotBearer,
        AuthStrategy::ChatGptOAuth,
        AuthStrategy::GcpToken,
        AuthStrategy::AwsSigV4,
        AuthStrategy::AzureToken,
        AuthStrategy::None,
    ];
}

/// Where the credential comes from. `Static` deserializes into a `Secret` and
/// can never be serialized back out.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum CredentialConfig {
    Env {
        var: String,
    },
    Static {
        value: Secret<String>,
    },
    /// An opaque key the host can use to look up a credential.
    /// The client never performs this lookup.
    HostManaged {
        key: String,
    },
    #[default]
    None,
}

impl Serialize for CredentialConfig {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        match self {
            Self::Static { .. } => Err(serde::ser::Error::custom(
                "static credentials cannot be persisted",
            )),
            Self::Env { var } => {
                let mut value = serializer.serialize_struct("CredentialConfig", 2)?;
                value.serialize_field("source", "env")?;
                value.serialize_field("var", var)?;
                value.end()
            }
            Self::HostManaged { key } => {
                let mut value = serializer.serialize_struct("CredentialConfig", 2)?;
                value.serialize_field("source", "host_managed")?;
                value.serialize_field("key", key)?;
                value.end()
            }
            Self::None => {
                let mut value = serializer.serialize_struct("CredentialConfig", 1)?;
                value.serialize_field("source", "none")?;
                value.end()
            }
        }
    }
}

/// Whether a route charges per token, is covered by a plan, or is free.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum BillingMode {
    /// Published per-token prices apply.
    PerToken,
    /// Usage is governed by a subscription or membership plan.
    Subscription,
    /// The provider explicitly publishes the route as free.
    Free,
    /// No reliable billing semantics are available. Distinct from `Free`.
    #[default]
    Unknown,
}

/// Which failures move a request to the next connection of the group.
///
/// Deliberately not a list of every `LlmErrorKind`: `ContextOverflow`,
/// `RequestTooLarge` and `UnsupportedCapability` are excluded because another
/// endpoint would reject them identically, and retrying them elsewhere only
/// hides the real reason.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FailoverTriggers {
    /// 429 from this connection.
    #[serde(default)]
    pub rate_limit: bool,
    /// 529 overloaded.
    #[serde(default)]
    pub overloaded: bool,
    /// 5xx / provider-internal.
    #[serde(default)]
    pub server_error: bool,
    /// Transport failure or timeout reaching this endpoint.
    #[serde(default)]
    pub network: bool,
    /// 401/403 — the usual reason to rotate to the next key.
    #[serde(default)]
    pub auth: bool,
}

impl FailoverTriggers {
    /// What a provider gets when it declares connections without naming
    /// triggers: everything another endpoint or key could plausibly answer.
    pub const DEFAULT: Self = Self {
        rate_limit: true,
        overloaded: true,
        server_error: true,
        network: true,
        auth: true,
    };

    /// No trigger set — never fail over.
    pub const NONE: Self = Self {
        rate_limit: false,
        overloaded: false,
        server_error: false,
        network: false,
        auth: false,
    };

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        !(self.rate_limit || self.overloaded || self.server_error || self.network || self.auth)
    }

    /// Whether `error` should move the request to the next connection.
    ///
    /// The `_ => false` arm is the point of the type: an error another endpoint
    /// would answer identically (a context overflow, a body over the byte
    /// limit, a capability the model does not have) is not a failover trigger,
    /// and retrying it elsewhere only buries the real reason.
    #[must_use]
    pub fn matches(self, error: &crate::protocol::LlmError) -> bool {
        use crate::protocol::LlmError;
        match error {
            LlmError::RateLimited { .. } | LlmError::QuotaExceeded { .. } => self.rate_limit,
            LlmError::Overloaded { .. } => self.overloaded,
            LlmError::ProviderInternal { .. } => self.server_error,
            LlmError::Transport { .. } | LlmError::TransportTimeout { .. } => self.network,
            LlmError::Authentication { .. } | LlmError::PermissionDenied { .. } => self.auth,
            _ => false,
        }
    }
}

/// Group and ordering identity of one connection.
///
/// Two endpoints of one provider, or two API keys for one endpoint, are one
/// group with several connections — not several providers. The default is a
/// profile that stands alone, so an entry written before groups existed keeps
/// behaving exactly as it did.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionSpec {
    /// Group this connection belongs to. `None` = the profile stands alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    /// Connection id within the group, unique per group.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connection_id: Option<String>,
    /// Position within the group; lower is tried first. Ties break on
    /// `profile_name`, so the order is total and stable.
    #[serde(default)]
    pub order: u32,
    /// Never offered in a model picker. Set on the second and later key slots of
    /// one connection, which exist only to be failed over onto.
    #[serde(default)]
    pub hidden: bool,
    /// Which failures move to the next connection of this group. Shared by every
    /// connection in the group: it is the provider's policy, not the endpoint's.
    #[serde(default, skip_serializing_if = "FailoverTriggers::is_empty")]
    pub failover: FailoverTriggers,
}

impl ConnectionSpec {
    /// Whether this is the default (standalone, visible, first) identity, so an
    /// untouched profile serializes unchanged.
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Provider-published display metadata. Informational: it never participates in
/// routing.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelMetadata {
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub release_date: Option<String>,
    #[serde(default)]
    pub last_updated: Option<String>,
    #[serde(default)]
    pub knowledge_cutoff: Option<String>,
    #[serde(default)]
    pub input_modalities: Vec<String>,
    #[serde(default)]
    pub output_modalities: Vec<String>,
    #[serde(default)]
    pub context_window_tokens: Option<u64>,
    #[serde(default)]
    pub max_input_tokens: Option<u64>,
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub open_weights: Option<bool>,
    #[serde(default)]
    pub attachments: Option<bool>,
    #[serde(default)]
    pub temperature_control: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    #[serde(default)]
    pub info: super::ModelInfo,
    /// Human-facing model label. What a picker shows and what an override in
    /// `pricing` is keyed by.
    pub display_model: String,
    /// Provider-local model value sent on the wire.
    pub request_model: String,
    /// Model id used by pricing lookup. Often equal to `request_model`, but a
    /// provider may bill a family under one id and serve several wire ids.
    pub billing_model: String,
    /// Exclude this model from picker listings while keeping it addressable.
    #[serde(default)]
    pub hidden: bool,
    /// Alternate names route resolution accepts.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// One-line human description for the model picker.
    #[serde(default)]
    pub description: Option<String>,
    /// Provider-published display metadata. Never participates in routing.
    #[serde(default)]
    pub metadata: ModelMetadata,
    /// Explicit capability facts; absent fields are unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_support: Option<ModelCapabilitySupport>,
    /// Published prices for this model. Absent means unpriced, which is not
    /// the same as free: a cost estimate has to say it does not know.
    #[serde(default)]
    pub pricing: Option<TokenPricing>,
    /// How this one model is billed, when that differs from the rest of the
    /// connection. Absent means the connection's own mode applies.
    ///
    /// An aggregator is the case this exists for: it serves metered models and
    /// free ones over one endpoint and one key, so the mode cannot be a
    /// property of the connection alone.
    ///
    /// Note this is not derivable from the price. A model priced at zero on a
    /// subscription connection is not free — it is covered by the plan, and
    /// treating the two alike would let a request move off the plan and start
    /// charging.
    #[serde(default)]
    pub billing_mode: Option<BillingMode>,
}

impl ModelProfile {
    /// Returns the explicit support fact, or Unknown when absent.
    #[must_use]
    pub fn capability_support_for(&self, capability: ModelCapability) -> CapabilitySupport {
        self.capability_support
            .map(|support| support.get(capability))
            .unwrap_or_default()
    }

    /// How this model is billed on the given connection.
    #[must_use]
    pub fn billing_mode_on(&self, connection: &PricingConfig) -> BillingMode {
        self.billing_mode.unwrap_or(connection.billing_mode)
    }
}

impl ModelProfile {
    /// Whether any of this model's names answers to `wanted`.
    #[must_use]
    pub fn answers_to(&self, wanted: &str) -> bool {
        self.display_model == wanted
            || self.request_model == wanted
            || self.aliases.iter().any(|a| a == wanted)
    }
}

/// Pricing behaviour for a profile.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct PricingConfig {
    /// Whether this provider charges per token, via subscription, or is
    /// explicitly free. `Unknown` is distinct from `Free`.
    #[serde(default, rename = "billingMode")]
    pub billing_mode: BillingMode,
    /// When this provider charges its listed rates and when it charges less.
    /// A provider's billing policy covers every model it serves, which is why
    /// it sits here and the rates themselves sit on the model.
    #[serde(default)]
    pub peak: Option<PeakSchedule>,
}

/// A provider that charges a lower rate outside published hours.
///
/// The rates on a model are the *peak* rates, because that is how a provider
/// publishes them and how a catalog records them. Outside the windows the bill
/// is the listed rate times `off_peak_multiplier` — so a cost estimate that
/// ignores this is wrong by that factor for most of the week.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PeakSchedule {
    /// `"HH:MM-HH:MM"` in UTC. A window that wraps midnight is written as two;
    /// `24:00` is allowed only as a window's end.
    pub utc_windows: Vec<String>,
    /// Peak applies on Monday to Friday only; weekends are entirely off-peak.
    #[serde(default)]
    pub weekdays_only: bool,
    /// What the listed rates are multiplied by outside peak.
    pub off_peak_multiplier: f64,
}

impl PeakSchedule {
    /// Validate a configured schedule before it is used for pricing.
    /// `is_peak` remains total for callers reading older malformed data.
    pub fn validate(&self) -> Result<(), String> {
        if self.utc_windows.is_empty() {
            return Err("at least one UTC window is required".to_owned());
        }
        if !self.off_peak_multiplier.is_finite() || self.off_peak_multiplier < 0.0 {
            return Err("off-peak multiplier must be finite and non-negative".to_owned());
        }
        for window in &self.utc_windows {
            if parse_window(window).is_none() {
                return Err(format!("invalid UTC window {window:?}"));
            }
        }
        Ok(())
    }

    /// Whether `unix_seconds` falls inside a peak window.
    ///
    /// Hand-rolled rather than pulling in a calendar crate: this needs the UTC
    /// weekday and minute-of-day and nothing else, and protocol data should not
    /// grow a dependency for arithmetic. 1970-01-01 was a Thursday, which is
    /// where the weekday offset comes from.
    #[must_use]
    pub fn is_peak(&self, unix_seconds: u64) -> bool {
        let day = unix_seconds / 86_400;
        let minute_of_day = ((unix_seconds % 86_400) / 60) as u32;
        if self.weekdays_only {
            // 0 = Monday. The epoch fell on a Thursday, hence + 3.
            let weekday = (day + 3) % 7;
            if weekday >= 5 {
                return false;
            }
        }
        self.utc_windows
            .iter()
            .filter_map(|w| parse_window(w))
            .any(|(from, to)| minute_of_day >= from && minute_of_day < to)
    }
}

/// `"HH:MM-HH:MM"` to minutes of day. A malformed window matches nothing
/// rather than matching everything. Invalid windows never establish peak time.
fn parse_window(w: &str) -> Option<(u32, u32)> {
    let (from, to) = w.split_once('-')?;
    let minutes = |s: &str, is_end: bool| -> Option<u32> {
        let (h, m) = s.trim().split_once(':')?;
        let (h, m) = (h.parse::<u32>().ok()?, m.parse::<u32>().ok()?);
        if (h < 24 && m < 60) || (is_end && h == 24 && m == 0) {
            Some(h * 60 + m)
        } else {
            None
        }
    };
    let (from, to) = (minutes(from, false)?, minutes(to, true)?);
    (from < to).then_some((from, to))
}

/// Published prices and selection rules for one model, in its stated currency per million tokens.
///
/// Cache reads and writes are priced separately from input because they are
/// billed separately: a read is typically a tenth of the input rate and a write
/// can exceed it, so folding either into `input` misprices every cached turn.
/// `reasoning` is separate from `output` for the same reason — a provider that
/// bills thinking tokens at a different rate is not describable without it.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct TokenPricing {
    /// ISO currency; defaults to USD when omitted.
    #[serde(default)]
    pub currency: Option<String>,
    #[serde(default)]
    pub cache_write_1h_per_million: Option<f64>,
    #[serde(default)]
    pub rules: Vec<super::PriceRule>,
    #[serde(default)]
    pub quota: Vec<super::QuotaPrice>,
    #[serde(default)]
    pub verified_at: Option<String>,
    #[serde(default)]
    pub input_per_million: Option<f64>,
    #[serde(default)]
    pub output_per_million: Option<f64>,
    #[serde(default)]
    pub cache_read_per_million: Option<f64>,
    #[serde(default)]
    pub cache_write_per_million: Option<f64>,
    #[serde(default)]
    pub reasoning_per_million: Option<f64>,
    /// Where these numbers came from, so a stale estimate can be traced.
    #[serde(default)]
    pub source: Option<String>,
    /// What the same model costs submitted as a batch job, when the vendor
    /// publishes it. Absent means unknown, not unavailable.
    #[serde(default)]
    pub batch: Option<BatchPricing>,
}

/// Published rates for the same model submitted as a batch job.
///
/// Separate buckets rather than one multiplier. Across the whole shipped
/// catalog the discount happens to be a flat half, but nothing makes a vendor
/// discount input and output alike, and a scalar could not record it if one
/// did. No `source` of its own: these arrive alongside the listed rates and
/// share their provenance.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct BatchPricing {
    #[serde(default)]
    pub cache_write_1h_per_million: Option<f64>,
    #[serde(default)]
    pub input_per_million: Option<f64>,
    #[serde(default)]
    pub output_per_million: Option<f64>,
    #[serde(default)]
    pub cache_read_per_million: Option<f64>,
    #[serde(default)]
    pub cache_write_per_million: Option<f64>,
    #[serde(default)]
    pub reasoning_per_million: Option<f64>,
}

/// How a request is submitted, which is what selects between a model's listed
/// and batch rates. It is not a property of the model, so it is not stored on
/// one — the same model is billed either way depending on the endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Submission {
    /// The ordinary synchronous or streaming endpoint.
    #[default]
    Interactive,
    /// The vendor's batch endpoint: cheaper, asynchronous, results later.
    Batch,
}

/// Request signing (SigV4 for Bedrock, GCP tokens for Vertex). Required when
/// `auth = AwsSigV4`; missing is an `InvalidRequest` at auth time that names
/// the profile and the field.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct SigningConfig {
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub service: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
}

/// Azure OpenAI api-version configuration. Required when
/// `protocol = AzureOpenAi`; missing is an `InvalidRequest` at codec-build time.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct AzureConfig {
    #[serde(default)]
    pub api_version: Option<String>,
    #[serde(default)]
    pub deployment: Option<String>,
}

/// One provider entry in settings; several may describe one provider's several
/// connections (see [`ConnectionSpec`]).
/// What an app needs to present a provider and send a user to the right page.
///
/// Every field is optional because a user-declared provider has none of it, and
/// none of it participates in routing — a wrong URL costs a user a wasted click,
/// not a failed turn. The shipped presets do fill `display_name` and
/// `api_key_url`, which is what a "you have not set this up yet" prompt needs.
///
/// Hand-authored in each preset's route header rather than derived: the vendor's
/// own name for itself is not the profile name, and the page where a key is
/// created is not derivable from the base URL.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProviderInfo {
    #[serde(default)]
    pub features: super::InferenceFeatures,
    #[serde(default)]
    pub pricing: super::ProviderPricingInfo,
    /// The vendor's name for itself, for a picker. Not the profile name, which
    /// is a file stem.
    #[serde(default)]
    pub display_name: Option<String>,
    /// One line on what this provider is.
    #[serde(default)]
    pub description: Option<String>,
    /// Where a user manages the account — usage, billing, limits.
    #[serde(default)]
    pub console_url: Option<String>,
    /// The page where a user creates or manages the credential. The one field
    /// an app needs to turn "you have no key for this" into a link.
    #[serde(default)]
    pub api_key_url: Option<String>,
    /// The vendor's API documentation.
    #[serde(default)]
    pub docs_url: Option<String>,
    /// What the credential looks like, so a user can tell they pasted the right
    /// thing — a prefix, never a secret.
    #[serde(default)]
    pub credential_hint: Option<String>,
}

/// Product usage region, independent of an endpoint's deployment or signing region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Region {
    ChinaMainland,
    International,
}

impl Region {
    pub const ALL: [Self; 2] = [Self::ChinaMainland, Self::International];

    /// Unscoped custom profiles are available in both usage regions.
    pub fn all() -> Vec<Self> {
        Self::ALL.to_vec()
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderProfile {
    #[serde(default)]
    pub inference: super::InferenceWire,
    /// Allowed usage regions. Missing means both; an empty list disables execution.
    #[serde(default = "Region::all")]
    pub regions: Vec<Region>,
    /// Explicit provider identity used after route resolution. An open string
    /// allows additional providers to use an existing protocol family.
    pub provider_id: ProviderId,
    /// Human/config profile name.
    pub profile_name: String,
    /// Base URL for this connection.
    ///
    /// How much of the path belongs here differs by protocol family — some want
    /// the bare origin, some the versioned root, some the resource endpoint
    /// without the deployment segment — and each codec documents its own shape
    /// and appends the rest.
    pub base_url: String,
    pub protocol: ProtocolFamily,
    /// Which shape this connection's model directory speaks. Defaults to
    /// `protocol`, which is right for most connections and wrong for the two
    /// this key exists for.
    #[serde(default, skip_serializing_if = "DirectoryRoute::is_default")]
    pub model_list: DirectoryRoute,
    pub auth: AuthStrategy,
    #[serde(default)]
    pub credential: CredentialConfig,
    #[serde(default)]
    pub models: Vec<ModelProfile>,
    /// Image routes and models are independent of the chat model directory.
    #[serde(default)]
    pub images: super::ImageServiceConfig,
    #[serde(default)]
    pub pricing: PricingConfig,
    #[serde(default)]
    pub signing: Option<SigningConfig>,
    #[serde(default)]
    pub azure: Option<AzureConfig>,
    #[serde(default)]
    pub supports_websockets: bool,
    #[serde(default)]
    pub supports_websocket_compression: bool,
    #[serde(default)]
    pub websocket_connect_timeout_ms: Option<u64>,
    /// Group and ordering identity. The default is "this profile is its own
    /// one-connection group".
    #[serde(default, skip_serializing_if = "ConnectionSpec::is_default")]
    pub connection: ConnectionSpec,
    /// Presentation and set-up links for this provider.
    #[serde(default)]
    pub info: ProviderInfo,
    /// Provider-opaque extras a codec may read.
    #[serde(default)]
    pub extra: Value,
}

impl ProviderProfile {
    /// Whether this connection may be offered and used in the selected region.
    #[must_use]
    pub fn supports_region(&self, region: Region) -> bool {
        self.regions.contains(&region)
    }

    /// The provider group this profile belongs to. Falls back to
    /// `profile_name`, so a standalone profile is a group of one and no caller
    /// has to special-case it.
    #[must_use]
    pub fn group(&self) -> &str {
        self.connection
            .group
            .as_deref()
            .unwrap_or(&self.profile_name)
    }

    /// This profile's connection id within its group (`"default"` when it
    /// stands alone).
    #[must_use]
    pub fn connection_id(&self) -> &str {
        self.connection
            .connection_id
            .as_deref()
            .unwrap_or("default")
    }

    /// Order within the group. Ties break on the profile name, so the order is
    /// total and stable across runs.
    #[must_use]
    pub fn connection_sort_key(&self) -> (u32, &str) {
        (self.connection.order, self.profile_name.as_str())
    }
}

#[cfg(test)]
mod pricing_tests {
    use super::*;

    /// 2026-01-05T00:00:00Z is a Monday.
    const MONDAY: u64 = 1_767_571_200;

    fn at(day: u64, hour: u64, minute: u64) -> u64 {
        MONDAY + day * 86_400 + hour * 3_600 + minute * 60
    }

    fn schedule() -> PeakSchedule {
        PeakSchedule {
            utc_windows: vec!["01:00-04:00".to_owned(), "06:00-10:00".to_owned()],
            weekdays_only: true,
            off_peak_multiplier: 0.5,
        }
    }

    #[test]
    fn a_window_is_half_open_so_its_end_is_already_off_peak() {
        let s = schedule();
        assert!(s.is_peak(at(0, 1, 0)), "the window's start is peak");
        assert!(s.is_peak(at(0, 3, 59)));
        assert!(
            !s.is_peak(at(0, 4, 0)),
            "an inclusive end would overlap the next window's start elsewhere"
        );
        assert!(!s.is_peak(at(0, 0, 59)));
    }

    #[test]
    fn the_gap_between_two_windows_is_off_peak() {
        assert!(!schedule().is_peak(at(0, 5, 0)));
    }

    #[test]
    fn a_window_can_end_at_midnight() {
        let s = PeakSchedule {
            utc_windows: vec!["23:00-24:00".into(), "00:00-01:00".into()],
            weekdays_only: false,
            off_peak_multiplier: 0.5,
        };
        assert!(s.is_peak(at(0, 23, 59)));
        assert!(s.is_peak(at(1, 0, 30)));
        assert!(!s.is_peak(at(1, 1, 0)));
    }

    #[test]
    fn very_large_window_components_are_rejected_without_panicking() {
        for invalid in [
            "4294967295:00-01:00",
            "00:4294967295-01:00",
            "24:00-24:00",
            "23:00-24:01",
        ] {
            assert_eq!(parse_window(invalid), None);
        }
    }

    #[test]
    fn the_weekend_is_off_peak_at_every_hour() {
        let s = schedule();
        for day in [5, 6] {
            assert!(!s.is_peak(at(day, 2, 0)), "day {day} is a weekend");
        }
        assert!(s.is_peak(at(4, 2, 0)), "Friday is still a weekday");
        assert!(s.is_peak(at(7, 2, 0)), "and the next Monday is peak again");
    }

    #[test]
    fn a_malformed_window_matches_nothing_rather_than_everything() {
        // Over-charging is the safer failure: reporting a discount that did not
        // apply understates a bill a user has already paid.
        for bad in ["", "01:00", "1000-1400", "25:00-26:00", "10:00-06:00"] {
            let s = PeakSchedule {
                utc_windows: vec![bad.to_owned()],
                weekdays_only: false,
                off_peak_multiplier: 0.5,
            };
            assert!(!s.is_peak(at(0, 2, 0)), "{bad:?} must not match");
        }
    }

    #[test]
    fn invalid_pricing_schedules_are_rejected_before_use() {
        assert!(schedule().validate().is_ok());
        for windows in [vec![], vec!["25:00-26:00".to_owned()]] {
            let mut candidate = schedule();
            candidate.utc_windows = windows;
            assert!(candidate.validate().is_err());
        }
        let mut candidate = schedule();
        candidate.off_peak_multiplier = f64::NAN;
        assert!(candidate.validate().is_err());
    }
}
