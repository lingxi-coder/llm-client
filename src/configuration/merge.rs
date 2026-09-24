//! Pure merging of defaults, explicit settings, observations.
use super::{model::*, ProviderStoreError};
use crate::protocol::ProviderProfile;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
pub(crate) struct Definitions {
    pub caller: Vec<ProviderProfile>,
    pub builtin_names: BTreeSet<String>,
}
impl Definitions {
    pub fn new(caller: Vec<ProviderProfile>) -> Self {
        Self {
            caller,
            builtin_names: BTreeSet::new(),
        }
    }
    pub fn reference(&self, name: &str) -> DefinitionRef {
        if self.builtin_names.contains(name) {
            DefinitionRef::Builtin { name: name.into() }
        } else {
            DefinitionRef::Caller { name: name.into() }
        }
    }
    pub fn get(&self, reference: &DefinitionRef) -> Option<&ProviderProfile> {
        let list: &[ProviderProfile] = match reference {
            DefinitionRef::Caller { .. } => &self.caller,
            DefinitionRef::Builtin { .. } => {
                static BUILTINS: std::sync::OnceLock<Vec<ProviderProfile>> =
                    std::sync::OnceLock::new();
                BUILTINS.get_or_init(|| {
                    crate::presets::builtin().expect("validated embedded provider catalog")
                })
            }
        };
        list.iter().find(|p| p.profile_name == reference.name())
    }
}
fn keys(profile: &ProviderProfile) -> Vec<ModelKey> {
    let mut occurrences = BTreeMap::new();
    profile
        .models
        .iter()
        .map(|m| {
            let n = occurrences
                .entry((m.request_model.clone(), m.display_model.clone()))
                .or_insert(0);
            let key = ModelKey {
                wire: m.request_model.clone(),
                label: m.display_model.clone(),
                occurrence: *n,
            };
            *n += 1;
            key
        })
        .collect()
}
fn row_id(key: &ModelKey) -> String {
    format!(
        "definition:{}",
        serde_json::to_string(key).expect("model key is serializable")
    )
}
impl SavedProfile {
    pub(crate) fn inherited(profile: &ProviderProfile, definition: DefinitionRef) -> Self {
        let mut connection = profile.clone();
        connection.models.clear();
        if matches!(definition, DefinitionRef::Builtin { .. }) {
            // Built-in image routes are inherited just like built-in model rows.
            connection.images = Default::default();
        }
        let models = keys(profile)
            .into_iter()
            .map(|key| ModelRow {
                id: row_id(&key),
                definition_key: Some(key),
                initial: None,
                replacement: false,
                observed: false,
                overrides: Values::new(),
            })
            .collect();
        Self {
            definition,
            connection,
            fallback: profile.clone(),
            replace_models: false,
            removed_defaults: BTreeSet::new(),
            models,
            observations: BTreeMap::new(),
            incompatible_models: BTreeSet::new(),
        }
    }
    pub(crate) fn replacement(profile: &ProviderProfile, definition: DefinitionRef) -> Self {
        let mut saved = Self::inherited(profile, definition);
        saved.connection.images = profile.images.clone();
        saved.replace_models = true;
        for (row, model) in saved.models.iter_mut().zip(&profile.models) {
            row.replacement = true;
            row.initial = Some(empty_model(&model.request_model));
            row.overrides = values(model);
        }
        saved
    }
    pub(crate) fn rows(
        &self,
        definitions: &Definitions,
    ) -> Result<Vec<ConfiguredModel>, ProviderStoreError> {
        let definition = definitions.get(&self.definition).unwrap_or(&self.fallback);
        let keyed: Vec<_> = keys(definition)
            .into_iter()
            .zip(&definition.models)
            .collect();
        let mut rows = self.models.clone();
        self.adopt_rows(&mut rows, &keyed);
        if !self.replace_models {
            for (key, _) in &keyed {
                if !self.removed_defaults.iter().any(|removed| {
                    removed == key
                        || (removed.wire == key.wire
                            && keyed
                                .iter()
                                .filter(|(candidate, _)| candidate.wire == key.wire)
                                .count()
                                == 1)
                }) && !rows.iter().any(|r| r.definition_key.as_ref() == Some(key))
                {
                    rows.push(ModelRow {
                        id: row_id(key),
                        definition_key: Some(key.clone()),
                        initial: None,
                        replacement: false,
                        observed: false,
                        overrides: Values::new(),
                    });
                }
            }
        }
        let mut result = Vec::new();
        for row in rows {
            let base = if row.replacement {
                let initial = row.initial.as_ref();
                initial
                    .and_then(|initial| {
                        let candidates: Vec<_> = keyed
                            .iter()
                            .filter(|(key, _)| key.wire == initial.request_model)
                            .collect();
                        candidates
                            .iter()
                            .find(|(key, _)| row.definition_key.as_ref() == Some(key))
                            .or_else(|| (candidates.len() == 1).then(|| &candidates[0]))
                            .map(|(_, model)| *model)
                    })
                    .or(initial)
            } else {
                match &row.definition_key {
                    Some(key) => keyed
                        .iter()
                        .find(|(k, _)| k == key)
                        .map(|(_, m)| *m)
                        .or_else(|| row.observed.then_some(row.initial.as_ref()).flatten()),
                    None => row.initial.as_ref(),
                }
            };
            let Some(base) = base else {
                continue;
            };
            let mut model = base.clone();
            if let Some(observation) = self.observations.get(&model.request_model) {
                if let Some(description) = &observation.description {
                    model.description = Some(description.clone());
                }
                if let Some(context) = observation.context_window {
                    model.metadata.context_window_tokens = Some(context);
                }
                if let Some(features) = &observation.inference_features {
                    model.info.features.overlay(features);
                }
                if let Some(output) = observation.max_output_tokens {
                    model.metadata.max_output_tokens = Some(output);
                }
            }
            apply(&mut model, &row.overrides)?;
            model.info.pricing = model.pricing.clone();
            model.info.features = model
                .info
                .features
                .on_connection(&self.connection.inference_features());
            let compatible = !self.incompatible_models.contains(&model.request_model);
            result.push(ConfiguredModel {
                row_id: row.id,
                model,
                compatible,
            });
        }
        Ok(result)
    }
    fn adopt_rows(
        &self,
        rows: &mut [ModelRow],
        keyed: &[(ModelKey, &crate::protocol::ModelProfile)],
    ) {
        for row in rows.iter_mut() {
            if let Some(old) = &row.definition_key {
                if !keyed.iter().any(|(key, _)| key == old)
                    && keyed.iter().filter(|(key, _)| key.wire == old.wire).count() == 1
                    && self
                        .models
                        .iter()
                        .filter(|r| {
                            r.definition_key
                                .as_ref()
                                .is_some_and(|key| key.wire == old.wire)
                        })
                        .count()
                        == 1
                {
                    row.definition_key = keyed
                        .iter()
                        .find(|(key, _)| key.wire == old.wire)
                        .map(|(key, _)| key.clone());
                }
            }
        }
        for (key, _) in keyed {
            if key.occurrence == 0 && !rows.iter().any(|r| r.definition_key.as_ref() == Some(key)) {
                if let Some(row) = rows.iter_mut().find(|r| {
                    r.definition_key.is_none()
                        && !r.replacement
                        && r.id.starts_with("observed:")
                        && r.initial
                            .as_ref()
                            .is_some_and(|m| m.request_model == key.wire)
                }) {
                    row.definition_key = Some(key.clone());
                    // Keep the observation's own seed if a later static
                    // definition withdraws this model.
                }
            }
        }
    }
    fn adopt_observed(&mut self, definitions: &Definitions) {
        let definition = definitions.get(&self.definition).unwrap_or(&self.fallback);
        let keyed: Vec<_> = keys(definition)
            .into_iter()
            .zip(&definition.models)
            .collect();
        let mut rows = self.models.clone();
        self.adopt_rows(&mut rows, &keyed);
        self.models = rows;
    }
    pub(crate) fn ensure_row(
        &mut self,
        id: &str,
        definitions: &Definitions,
    ) -> Result<&mut ModelRow, ProviderStoreError> {
        self.adopt_observed(definitions);
        if !self.models.iter().any(|r| r.id == id) {
            let definition = definitions.get(&self.definition).unwrap_or(&self.fallback);
            let key = keys(definition)
                .into_iter()
                .find(|key| row_id(key) == id)
                .ok_or_else(|| ProviderStoreError::UnknownModel {
                    profile_name: self.connection.profile_name.clone(),
                    model: id.into(),
                })?;
            self.models.push(ModelRow {
                id: id.into(),
                definition_key: Some(key),
                initial: None,
                replacement: false,
                observed: false,
                overrides: Values::new(),
            });
        }
        Ok(self.models.iter_mut().find(|r| r.id == id).unwrap())
    }
}
impl SavedConfig {
    pub(crate) fn ensure_definitions(&mut self, definitions: &Definitions) {
        for profile in &definitions.caller {
            if !self.deleted_profiles.contains(&profile.profile_name)
                && !self
                    .providers
                    .iter()
                    .any(|p| p.connection.profile_name == profile.profile_name)
            {
                self.providers.push(SavedProfile::inherited(
                    profile,
                    definitions.reference(&profile.profile_name),
                ));
            }
        }
    }
    pub(crate) fn profiles(
        &self,
        definitions: &Definitions,
        executable: bool,
    ) -> Result<Vec<ProviderProfile>, ProviderStoreError> {
        let mut merged = self.clone();
        merged.ensure_definitions(definitions);
        merged
            .providers
            .iter()
            .filter(|p| !self.deleted_profiles.contains(&p.connection.profile_name))
            .map(|p| {
                let mut profile = p.connection.clone();
                if !p.replace_models
                    && profile.images.routes.is_empty()
                    && profile.images.models.is_empty()
                {
                    if let Some(definition) = definitions.get(&p.definition) {
                        profile.images = definition.images.clone();
                    }
                }
                profile.info.features = profile.inference_features();
                profile.models = p
                    .rows(definitions)?
                    .into_iter()
                    .filter(|r| !executable || r.compatible)
                    .map(|r| r.model)
                    .collect();
                Ok(profile)
            })
            .collect()
    }
    pub(crate) fn refresh_fallbacks(&mut self, definitions: &Definitions) {
        for profile in &mut self.providers {
            profile.adopt_observed(definitions);
            if let Some(current) = definitions.get(&profile.definition) {
                profile.fallback = current.clone();
            }
        }
    }
    pub(crate) fn persisted(&self) -> Self {
        let mut result = self.clone();
        for profile in &mut result.providers {
            let tracked = result
                .tracked_models
                .get(profile.connection.provider_id.as_str());
            let tracks = |wire: &str| tracked.is_some_and(|ids| ids.contains(wire));
            profile.fallback.models.retain(|m| tracks(&m.request_model));
            profile.models.retain(|r| {
                tracks(if r.replacement {
                    r.initial
                        .as_ref()
                        .map(|m| m.request_model.as_str())
                        .unwrap_or("")
                } else {
                    r.definition_key
                        .as_ref()
                        .map(|k| k.wire.as_str())
                        .or_else(|| r.initial.as_ref().map(|m| m.request_model.as_str()))
                        .unwrap_or("")
                })
            });
            profile.observations.retain(|wire, _| tracks(wire));
            // Negative compatibility survives an allowlist expansion, independently of metadata.
        }
        result
    }
    /// Keep untracked session-only definitions only while the durable account is unchanged.
    pub(crate) fn reconcile_session(&mut self, previous: &Self) {
        let durable = previous.persisted();
        for profile in &mut self.providers {
            if durable
                .providers
                .iter()
                .find(|p| p.connection.profile_name == profile.connection.profile_name)
                == Some(profile)
            {
                if let Some(old) = previous
                    .providers
                    .iter()
                    .find(|p| p.connection.profile_name == profile.connection.profile_name)
                {
                    *profile = old.clone();
                }
            }
        }
    }
}
pub(crate) fn same_connection(left: &ProviderProfile, right: &ProviderProfile) -> bool {
    let mut left = left.clone();
    let mut right = right.clone();
    left.models.clear();
    right.models.clear();
    left == right
}
