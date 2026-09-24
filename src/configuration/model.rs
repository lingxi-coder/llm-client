//! Persisted provenance. Model rows have identities independent of wire IDs.
use crate::protocol::{ModelProfile, ProviderProfile};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Where an account's default definition is obtained on the next load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum DefinitionRef {
    Builtin { name: String },
    Caller { name: String },
}
impl DefinitionRef {
    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Builtin { name } | Self::Caller { name } => name,
        }
    }
}

/// Inherit the latest value, explicitly set it, or clear the value.
/// `Inherit` removes the explicit value so the active definition or observation applies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "value", rename_all = "snake_case")]
pub enum FieldOverride<T> {
    Inherit,
    Set(T),
    Clear,
}

/// Editable fields. Model identity changes use `replace_model` instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelField {
    DisplayModel,
    BillingModel,
    Hidden,
    Aliases,
    Description,
    Pricing,
    BillingMode,
    CapabilitySupport,
    InferenceFeatures,
    Family,
    Status,
    ReleaseDate,
    LastUpdated,
    KnowledgeCutoff,
    InputModalities,
    OutputModalities,
    ContextWindowTokens,
    MaxInputTokens,
    MaxOutputTokens,
    OpenWeights,
    Attachments,
    TemperatureControl,
}
impl ModelField {
    pub(crate) const ALL: [Self; 22] = [
        Self::DisplayModel,
        Self::BillingModel,
        Self::Hidden,
        Self::Aliases,
        Self::Description,
        Self::Pricing,
        Self::BillingMode,
        Self::CapabilitySupport,
        Self::InferenceFeatures,
        Self::Family,
        Self::Status,
        Self::ReleaseDate,
        Self::LastUpdated,
        Self::KnowledgeCutoff,
        Self::InputModalities,
        Self::OutputModalities,
        Self::ContextWindowTokens,
        Self::MaxInputTokens,
        Self::MaxOutputTokens,
        Self::OpenWeights,
        Self::Attachments,
        Self::TemperatureControl,
    ];
    // Defined separately below to keep serialized field names stable.
    pub(crate) fn path(self) -> (&'static str, &'static str) {
        match self {
            Self::DisplayModel => ("display_model", ""),
            Self::BillingModel => ("billing_model", ""),
            Self::Hidden => ("hidden", ""),
            Self::Aliases => ("aliases", ""),
            Self::Description => ("description", ""),
            Self::Pricing => ("pricing", ""),
            Self::BillingMode => ("billing_mode", ""),
            Self::CapabilitySupport => ("capability_support", ""),
            Self::InferenceFeatures => ("features", "info"),
            Self::Family => ("family", "metadata"),
            Self::Status => ("status", "metadata"),
            Self::ReleaseDate => ("releaseDate", "metadata"),
            Self::LastUpdated => ("lastUpdated", "metadata"),
            Self::KnowledgeCutoff => ("knowledgeCutoff", "metadata"),
            Self::InputModalities => ("inputModalities", "metadata"),
            Self::OutputModalities => ("outputModalities", "metadata"),
            Self::ContextWindowTokens => ("contextWindowTokens", "metadata"),
            Self::MaxInputTokens => ("maxInputTokens", "metadata"),
            Self::MaxOutputTokens => ("maxOutputTokens", "metadata"),
            Self::OpenWeights => ("openWeights", "metadata"),
            Self::Attachments => ("attachments", "metadata"),
            Self::TemperatureControl => ("temperatureControl", "metadata"),
        }
    }
}

pub(crate) type Values = BTreeMap<ModelField, Value>;
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct ModelKey {
    pub wire: String,
    pub label: String,
    pub occurrence: usize,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ModelRow {
    pub id: String,
    /// A default-catalog row. Missing defaults never resurrect from fallback.
    pub definition_key: Option<ModelKey>,
    /// Explicitly added or observed models have no definition key.
    pub initial: Option<ModelProfile>,
    #[serde(default)]
    pub replacement: bool,
    #[serde(default)]
    pub observed: bool,
    #[serde(default)]
    pub overrides: Values,
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Observation {
    pub description: Option<String>,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub inference_features: Option<crate::protocol::InferenceFeatures>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SavedProfile {
    pub definition: DefinitionRef,
    /// Account connection settings, without model rows or credential values.
    pub connection: ProviderProfile,
    /// Used only when the whole referenced definition is unavailable.
    pub fallback: ProviderProfile,
    #[serde(default)]
    pub replace_models: bool,
    #[serde(default)]
    pub removed_defaults: BTreeSet<ModelKey>,
    #[serde(default)]
    pub models: Vec<ModelRow>,
    #[serde(default)]
    pub observations: BTreeMap<String, Observation>,
    #[serde(default)]
    pub incompatible_models: BTreeSet<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SavedConfig {
    pub version: u32,
    #[serde(default)]
    pub tracked_models: BTreeMap<String, BTreeSet<String>>,
    #[serde(default)]
    pub deleted_profiles: BTreeSet<String>,
    #[serde(default)]
    pub providers: Vec<SavedProfile>,
}
impl Default for SavedConfig {
    fn default() -> Self {
        Self {
            version: 2,
            tracked_models: BTreeMap::new(),
            deleted_profiles: BTreeSet::new(),
            providers: Vec::new(),
        }
    }
}
/// A configured row, including metadata for a temporarily incompatible model.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfiguredModel {
    pub row_id: String,
    pub model: ModelProfile,
    pub compatible: bool,
}

pub(crate) fn empty_model(wire: &str) -> ModelProfile {
    ModelProfile {
        info: Default::default(),
        display_model: wire.into(),
        request_model: wire.into(),
        billing_model: wire.into(),
        hidden: false,
        aliases: Vec::new(),
        description: None,
        metadata: Default::default(),
        capability_support: None,
        pricing: None,
        billing_mode: None,
    }
}
pub(crate) fn values(model: &ModelProfile) -> Values {
    let value = serde_json::to_value(model).expect("model is serializable");
    ModelField::ALL
        .into_iter()
        .map(|field| {
            let (key, metadata) = field.path();
            (
                field,
                if !metadata.is_empty() {
                    value[metadata][key].clone()
                } else {
                    value[key].clone()
                },
            )
        })
        .collect()
}
pub(crate) fn apply(model: &mut ModelProfile, fields: &Values) -> Result<(), serde_json::Error> {
    let mut value = serde_json::to_value(&*model)?;
    for (field, setting) in fields {
        let (key, metadata) = field.path();
        if !metadata.is_empty() {
            value[metadata][key] = setting.clone();
        } else {
            value[key] = setting.clone();
        }
    }
    *model = serde_json::from_value(value)?;
    Ok(())
}
