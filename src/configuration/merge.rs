//! Pure merging of defaults, explicit settings, observations.
use super::{model::*, ProviderStoreError};
use crate::protocol::{ModelProfile, ProviderProfile};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct Definitions {
    pub caller: Arc<[ProviderProfile]>,
    pub builtin_names: BTreeSet<String>,
}
impl Definitions {
    pub fn new(caller: Vec<ProviderProfile>) -> Self {
        Self {
            caller: caller.into(),
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
                crate::presets::builtin_catalog().expect("validated embedded provider catalog")
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

/// Definition lookup shared by all rows in one merge. Catalog order remains in
/// `keyed`; a wire id only identifies a row when the definition has one match.
struct DefinitionRows<'a> {
    keyed: Vec<(ModelKey, &'a ModelProfile)>,
    by_key: BTreeMap<ModelKey, usize>,
    unique_by_wire: BTreeMap<String, Option<usize>>,
}
impl<'a> DefinitionRows<'a> {
    fn new(profile: &'a ProviderProfile) -> Self {
        let keyed: Vec<_> = keys(profile).into_iter().zip(&profile.models).collect();
        let mut by_key = BTreeMap::new();
        let mut unique_by_wire = BTreeMap::new();
        for (index, (key, _)) in keyed.iter().enumerate() {
            by_key.insert(key.clone(), index);
            unique_by_wire
                .entry(key.wire.clone())
                .and_modify(|entry| *entry = None)
                .or_insert(Some(index));
        }
        Self {
            keyed,
            by_key,
            unique_by_wire,
        }
    }

    fn model(&self, key: &ModelKey) -> Option<&'a ModelProfile> {
        self.by_key.get(key).map(|index| self.keyed[*index].1)
    }

    fn unique(&self, wire: &str) -> Option<&(ModelKey, &'a ModelProfile)> {
        self.unique_by_wire
            .get(wire)
            .and_then(|index| index.map(|index| &self.keyed[index]))
    }
}
impl SavedProfile {
    pub(crate) fn inherited(profile: &ProviderProfile, definition: DefinitionRef) -> Self {
        let mut connection = profile.clone();
        connection.models.clear();
        connection.embeddings = Default::default();
        connection.retrieval = Default::default();
        connection.batches = Default::default();
        connection.deferred = Default::default();
        connection.background = Default::default();
        connection.audio = Default::default();
        connection.interactions = Default::default();
        connection.gemini_file_search = Default::default();
        connection.glm_knowledge = Default::default();
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
        saved.connection.embeddings = profile.embeddings.clone();
        saved.connection.retrieval = profile.retrieval.clone();
        saved.connection.batches = profile.batches.clone();
        saved.connection.deferred = profile.deferred.clone();
        saved.connection.background = profile.background.clone();
        saved.connection.audio = profile.audio.clone();
        saved.connection.interactions = profile.interactions.clone();
        saved.connection.gemini_file_search = profile.gemini_file_search.clone();
        saved.connection.glm_knowledge = profile.glm_knowledge.clone();
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
        let definition = DefinitionRows::new(definition);
        let mut rows = self.models.clone();
        self.adopt_rows(&mut rows, &definition);
        if !self.replace_models {
            let removed_wires: BTreeSet<_> = self
                .removed_defaults
                .iter()
                .map(|key| key.wire.as_str())
                .collect();
            let mut present: BTreeSet<_> = rows
                .iter()
                .filter_map(|row| row.definition_key.clone())
                .collect();
            for (key, _) in &definition.keyed {
                let removed = self.removed_defaults.contains(key)
                    || (definition.unique(&key.wire).is_some()
                        && removed_wires.contains(key.wire.as_str()));
                if !removed && present.insert(key.clone()) {
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
        let connection_features = self.connection.inference_features();
        let mut result = Vec::with_capacity(rows.len());
        for row in rows {
            let base = if row.replacement {
                let initial = row.initial.as_ref();
                initial
                    .and_then(|initial| {
                        row.definition_key
                            .as_ref()
                            .filter(|key| key.wire == initial.request_model)
                            .and_then(|key| definition.model(key))
                            .or_else(|| {
                                definition
                                    .unique(&initial.request_model)
                                    .map(|(_, model)| *model)
                            })
                    })
                    .or(initial)
            } else {
                match &row.definition_key {
                    Some(key) => definition
                        .model(key)
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
            model.info.features = model.info.features.on_connection(&connection_features);
            let compatible = !self.incompatible_models.contains(&model.request_model);
            result.push(ConfiguredModel {
                row_id: row.id,
                model,
                compatible,
            });
        }
        Ok(result)
    }
    fn adopt_rows(&self, rows: &mut [ModelRow], definition: &DefinitionRows<'_>) {
        let mut saved_wire_counts = BTreeMap::new();
        for key in self.models.iter().filter_map(|r| r.definition_key.as_ref()) {
            *saved_wire_counts.entry(key.wire.as_str()).or_insert(0) += 1;
        }
        for row in rows.iter_mut() {
            if let Some(old) = &row.definition_key {
                if !definition.by_key.contains_key(old)
                    && saved_wire_counts.get(old.wire.as_str()) == Some(&1)
                {
                    if let Some((key, _)) = definition.unique(&old.wire) {
                        row.definition_key = Some(key.clone());
                    }
                }
            }
        }
        let mut present: BTreeSet<_> = rows
            .iter()
            .filter_map(|row| row.definition_key.clone())
            .collect();
        let mut observed: BTreeMap<String, VecDeque<usize>> = BTreeMap::new();
        for (index, row) in rows.iter().enumerate() {
            if row.definition_key.is_none() && !row.replacement && row.id.starts_with("observed:") {
                if let Some(initial) = &row.initial {
                    observed
                        .entry(initial.request_model.clone())
                        .or_default()
                        .push_back(index);
                }
            }
        }
        for (key, _) in &definition.keyed {
            if key.occurrence == 0 && !present.contains(key) {
                if let Some(index) = observed.get_mut(&key.wire).and_then(VecDeque::pop_front) {
                    rows[index].definition_key = Some(key.clone());
                    present.insert(key.clone());
                    // Keep the observation's own seed if a later static
                    // definition withdraws this model.
                }
            }
        }
    }
    fn adopt_observed(&mut self, definitions: &Definitions) {
        let definition = definitions.get(&self.definition).unwrap_or(&self.fallback);
        let definition = DefinitionRows::new(definition);
        let mut rows = self.models.clone();
        self.adopt_rows(&mut rows, &definition);
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
        let mut present: BTreeSet<_> = self
            .providers
            .iter()
            .map(|p| p.connection.profile_name.clone())
            .collect();
        for profile in definitions.caller.iter() {
            if !self.deleted_profiles.contains(&profile.profile_name)
                && present.insert(profile.profile_name.clone())
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
        let mut present: BTreeSet<_> = self
            .providers
            .iter()
            .map(|p| p.connection.profile_name.as_str())
            .collect();
        let inherited: Vec<_> = definitions
            .caller
            .iter()
            .filter(|p| {
                !self.deleted_profiles.contains(&p.profile_name)
                    && present.insert(p.profile_name.as_str())
            })
            .map(|p| SavedProfile::inherited(p, definitions.reference(&p.profile_name)))
            .collect();
        self.providers
            .iter()
            .chain(&inherited)
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
                if matches!(profile.embeddings, crate::protocol::ServiceSetting::Inherit) {
                    profile.embeddings = definitions
                        .get(&p.definition)
                        .unwrap_or(&p.fallback)
                        .embeddings
                        .clone();
                }
                if matches!(profile.retrieval, crate::protocol::ServiceSetting::Inherit) {
                    profile.retrieval = definitions
                        .get(&p.definition)
                        .unwrap_or(&p.fallback)
                        .retrieval
                        .clone();
                }
                if matches!(profile.batches, crate::protocol::ServiceSetting::Inherit) {
                    profile.batches = definitions
                        .get(&p.definition)
                        .unwrap_or(&p.fallback)
                        .batches
                        .clone();
                }
                if matches!(profile.deferred, crate::protocol::ServiceSetting::Inherit) {
                    profile.deferred = definitions
                        .get(&p.definition)
                        .unwrap_or(&p.fallback)
                        .deferred
                        .clone();
                }
                if matches!(profile.background, crate::protocol::ServiceSetting::Inherit) {
                    profile.background = definitions
                        .get(&p.definition)
                        .unwrap_or(&p.fallback)
                        .background
                        .clone();
                }
                if matches!(profile.audio, crate::protocol::ServiceSetting::Inherit) {
                    profile.audio = definitions
                        .get(&p.definition)
                        .unwrap_or(&p.fallback)
                        .audio
                        .clone();
                }
                if matches!(
                    profile.interactions,
                    crate::protocol::ServiceSetting::Inherit
                ) {
                    profile.interactions = definitions
                        .get(&p.definition)
                        .unwrap_or(&p.fallback)
                        .interactions
                        .clone();
                }
                if matches!(
                    profile.gemini_file_search,
                    crate::protocol::ServiceSetting::Inherit
                ) {
                    profile.gemini_file_search = definitions
                        .get(&p.definition)
                        .unwrap_or(&p.fallback)
                        .gemini_file_search
                        .clone();
                }
                if matches!(
                    profile.glm_knowledge,
                    crate::protocol::ServiceSetting::Inherit
                ) {
                    profile.glm_knowledge = definitions
                        .get(&p.definition)
                        .unwrap_or(&p.fallback)
                        .glm_knowledge
                        .clone();
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
