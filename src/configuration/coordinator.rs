//! Transaction coordination produces changes; service invalidation belongs to the client.
use super::{merge::*, model::*, repository::Repository, ProviderStoreError};
use crate::protocol::ProviderProfile;
use std::{collections::BTreeSet, path::Path};

pub(crate) struct Coordinator {
    pub definitions: Definitions,
    pub state: SavedConfig,
    pub repository: Option<Repository>,
    pub generation: u64,
    pub deleted_builtin_profiles: BTreeSet<String>,
}
#[derive(Default)]
pub(crate) struct ChangeSet {
    pub connections: BTreeSet<String>,
    pub profiles: BTreeSet<String>,
    pub all_bindings: bool,
}
impl ChangeSet {
    pub fn between(old: &[ProviderProfile], new: &[ProviderProfile]) -> Self {
        let names: BTreeSet<_> = old
            .iter()
            .chain(new)
            .map(|p| p.profile_name.clone())
            .collect();
        let mut result = Self::default();
        for name in names {
            let before = old.iter().find(|p| p.profile_name == name);
            let after = new.iter().find(|p| p.profile_name == name);
            if before != after {
                result.profiles.insert(name.clone());
            }
            if !matches!((before,after),(Some(a),Some(b))if same_connection(a,b)) {
                result.connections.insert(name);
            }
        }
        result
    }
}
impl Coordinator {
    pub fn new(profiles: Vec<ProviderProfile>) -> Self {
        Self {
            definitions: Definitions::new(profiles),
            state: SavedConfig::default(),
            repository: None,
            generation: 0,
            deleted_builtin_profiles: BTreeSet::new(),
        }
    }
    pub fn load(
        &mut self,
        path: &Path,
        validate: impl FnOnce(&[ProviderProfile]) -> Result<(), ProviderStoreError>,
    ) -> Result<Vec<ProviderProfile>, ProviderStoreError> {
        let repo = Repository::open(path)?;
        let state = repo.read_locked()?;
        let profiles = state.profiles(&self.definitions, true)?;
        validate(&profiles)?;
        self.repository = Some(repo);
        self.generation = self.generation.wrapping_add(1);
        self.install(state);
        Ok(profiles)
    }
    pub fn install(&mut self, state: SavedConfig) {
        self.deleted_builtin_profiles = state
            .deleted_profiles
            .iter()
            .filter(|name| crate::presets::is_builtin_profile(name))
            .cloned()
            .collect();
        self.state = state;
    }
    pub fn update<T>(
        &mut self,
        change: impl FnOnce(&mut SavedConfig, &Definitions) -> Result<T, ProviderStoreError>,
        validate: impl FnOnce(&[ProviderProfile]) -> Result<(), ProviderStoreError>,
    ) -> Result<(T, Vec<ProviderProfile>), ProviderStoreError> {
        let repo = self
            .repository
            .as_ref()
            .ok_or(ProviderStoreError::NotConfigured)?;
        let _lock = repo.lock()?;
        let mut state = repo.read()?;
        state.reconcile_session(&self.state);
        state.ensure_definitions(&self.definitions);
        let result = change(&mut state, &self.definitions)?;
        state.refresh_fallbacks(&self.definitions);
        let profiles = state.profiles(&self.definitions, true)?;
        validate(&profiles)?;
        repo.write(&state)?;
        self.install(state);
        Ok((result, profiles))
    }
    pub fn configured_models(
        &self,
        name: &str,
    ) -> Result<Vec<ConfiguredModel>, ProviderStoreError> {
        let mut state = self.state.clone();
        state.ensure_definitions(&self.definitions);
        profile(&mut state, name)?.rows(&self.definitions)
    }
}
pub(crate) fn profile<'a>(
    state: &'a mut SavedConfig,
    name: &str,
) -> Result<&'a mut SavedProfile, ProviderStoreError> {
    if state.deleted_profiles.contains(name) {
        return Err(ProviderStoreError::UnknownProfile(name.into()));
    }
    state
        .providers
        .iter_mut()
        .find(|p| p.connection.profile_name == name)
        .ok_or_else(|| ProviderStoreError::UnknownProfile(name.into()))
}
