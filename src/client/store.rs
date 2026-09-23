//! Local provider overrides and account-specific model directory refresh.

use super::{validate_profiles, BuildError, LlmClient};
use crate::directory::LiveModel;
use crate::presets::PresetError;
use lingxi_agent_api::protocol::{
    AuthStrategy, CredentialConfig, LlmError, ModelProfile, ProviderProfile, Secret,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

const FILE_NAME: &str = "providers.json";
const MAX_PAGES: usize = 100;
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum ProviderStoreError {
    #[error("provider configuration directory has not been set")]
    NotConfigured,
    #[error("provider profile {0:?} was not found")]
    UnknownProfile(String),
    #[error("provider profile {0:?} is not a built-in preset")]
    NotBuiltin(String),
    #[error("provider profile {0:?} changed while its model directory was being synced")]
    ProfileChanged(String),
    #[error("model {model:?} was not found on provider profile {profile_name:?}")]
    UnknownModel { profile_name: String, model: String },
    #[error("model {model:?} is not tracked for provider {provider_id:?}")]
    UntrackedModel { provider_id: String, model: String },
    #[error("provider profile {0:?} does not publish a supported model directory")]
    NoDirectory(String),
    #[error("provider profile {0:?} contains a static credential that cannot be saved")]
    StaticCredential(String),
    #[error("provider profile {0:?} is duplicated in the saved configuration")]
    DuplicateProfile(String),
    #[error("unsupported provider configuration version {0}")]
    UnsupportedVersion(u32),
    #[error("model directory pagination exceeded {MAX_PAGES} pages or repeated a cursor")]
    InvalidPagination,
    #[error("provider configuration I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("provider configuration JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Build(#[from] BuildError),
    #[error(transparent)]
    Preset(#[from] PresetError),
    #[error(transparent)]
    Directory(#[from] LlmError),
}

#[derive(Serialize, Deserialize)]
struct SavedProviders {
    version: u32,
    #[serde(default)]
    tracked_models: BTreeMap<String, BTreeSet<String>>,
    #[serde(default)]
    deleted_profiles: BTreeSet<String>,
    providers: Vec<ProviderProfile>,
}

fn reject_static(profiles: &[ProviderProfile]) -> Result<(), ProviderStoreError> {
    if let Some(profile) = profiles
        .iter()
        .find(|p| matches!(p.credential, CredentialConfig::Static { .. }))
    {
        return Err(ProviderStoreError::StaticCredential(
            profile.profile_name.clone(),
        ));
    }
    Ok(())
}

fn validate_unique(profiles: &[ProviderProfile]) -> Result<(), ProviderStoreError> {
    let mut names = BTreeSet::new();
    for profile in profiles {
        if !names.insert(&profile.profile_name) {
            return Err(ProviderStoreError::DuplicateProfile(
                profile.profile_name.clone(),
            ));
        }
    }
    Ok(())
}

fn replace(profiles: &mut Vec<ProviderProfile>, profile: ProviderProfile) {
    if let Some(old) = profiles
        .iter_mut()
        .find(|p| p.profile_name == profile.profile_name)
    {
        *old = profile;
    } else {
        profiles.push(profile);
    }
}

fn same_profile_config(left: &ProviderProfile, right: &ProviderProfile) -> bool {
    let mut left = left.clone();
    let mut right = right.clone();
    left.models.clear();
    right.models.clear();
    left == right
}

fn read(dir: &Path) -> Result<SavedProviders, ProviderStoreError> {
    let bytes = match fs::read(dir.join(FILE_NAME)) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(SavedProviders {
                version: 1,
                tracked_models: BTreeMap::new(),
                deleted_profiles: BTreeSet::new(),
                providers: Vec::new(),
            });
        }
        Err(error) => return Err(error.into()),
    };
    let saved: SavedProviders = serde_json::from_slice(&bytes)?;
    if saved.version != 1 {
        return Err(ProviderStoreError::UnsupportedVersion(saved.version));
    }
    validate_unique(&saved.providers)?;
    reject_static(&saved.providers)?;
    Ok(saved)
}

fn filtered_profiles(saved: &SavedProviders) -> Vec<ProviderProfile> {
    let mut filtered = saved.providers.clone();
    for profile in &mut filtered {
        let tracked = saved.tracked_models.get(profile.provider_id.as_str());
        profile
            .models
            .retain(|model| tracked.is_some_and(|ids| ids.contains(&model.request_model)));
    }
    filtered
}

fn write(dir: &Path, saved: &SavedProviders) -> Result<(), ProviderStoreError> {
    reject_static(&saved.providers)?;
    let bytes = serde_json::to_vec_pretty(&SavedProviders {
        version: 1,
        tracked_models: saved.tracked_models.clone(),
        deleted_profiles: saved.deleted_profiles.clone(),
        providers: filtered_profiles(saved),
    })?;
    let path = dir.join(FILE_NAME);
    let temp = dir.join(format!(
        ".{FILE_NAME}.{}.{}.tmp",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temp, &path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result.map_err(Into::into)
}

fn merge_model(profile: &mut ProviderProfile, live: LiveModel) {
    if let Some(existing) = profile
        .models
        .iter_mut()
        .find(|m| m.request_model == live.request_model)
    {
        if let Some(description) = live.description {
            existing.description = Some(description);
        }
        if let Some(window) = live.context_window {
            existing.metadata.context_window_tokens = Some(window);
        }
        if let Some(output) = live.max_output_tokens {
            existing.metadata.max_output_tokens = Some(output);
        }
    } else {
        let mut model = ModelProfile {
            display_model: live.request_model.clone(),
            request_model: live.request_model.clone(),
            billing_model: live.request_model,
            hidden: false,
            aliases: Vec::new(),
            description: live.description.or(live.display_name),
            metadata: Default::default(),
            capabilities: Default::default(),
            pricing: None,
            billing_mode: None,
        };
        model.metadata.context_window_tokens = live.context_window;
        model.metadata.max_output_tokens = live.max_output_tokens;
        profile.models.push(model);
    }
}

impl LlmClient {
    fn remember_models(&mut self, profile: ProviderProfile) {
        if self.invalidated_model_sources.remove(&profile.profile_name) {
            self.model_sources
                .insert(profile.profile_name.clone(), profile);
            return;
        }
        if !self.model_sources.contains_key(&profile.profile_name) {
            if let Some(base) = self.base_profiles.iter().find(|base| {
                base.profile_name == profile.profile_name && same_profile_config(base, &profile)
            }) {
                self.model_sources
                    .insert(profile.profile_name.clone(), base.clone());
            }
        }
        if let Some(source) = self.model_sources.get_mut(&profile.profile_name) {
            if same_profile_config(source, &profile) {
                for model in profile.models {
                    if let Some(existing) = source
                        .models
                        .iter_mut()
                        .find(|old| old.request_model == model.request_model)
                    {
                        *existing = model;
                    } else {
                        source.models.push(model);
                    }
                }
                return;
            }
        }
        self.model_sources
            .insert(profile.profile_name.clone(), profile);
    }

    fn profiles_from_saved(
        &self,
        saved: &SavedProviders,
        locally_removed: &BTreeSet<String>,
    ) -> Vec<ProviderProfile> {
        let mut profiles: Vec<_> = self
            .base_profiles
            .iter()
            .filter(|p| !locally_removed.contains(&p.profile_name))
            .cloned()
            .collect();
        for profile in &saved.providers {
            replace(&mut profiles, profile.clone());
        }
        profiles.retain(|p| !saved.deleted_profiles.contains(&p.profile_name));
        profiles
    }

    fn install_saved(&mut self, saved: SavedProviders, profiles: Vec<ProviderProfile>) {
        self.persisted_profiles = filtered_profiles(&saved);
        self.profiles = profiles;
        self.deleted_profiles = saved.deleted_profiles;
        self.tracked_models = saved.tracked_models;
    }

    /// Apply one change to the latest on-disk state while other clients wait.
    fn update_saved<T>(
        &mut self,
        change: impl FnOnce(
            &mut SavedProviders,
            &mut Vec<ProviderProfile>,
        ) -> Result<T, ProviderStoreError>,
    ) -> Result<T, ProviderStoreError> {
        let dir = self
            .config_dir
            .as_ref()
            .ok_or(ProviderStoreError::NotConfigured)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join(".providers.json.lock"))?;
        lock.lock()?;
        let mut saved = read(dir)?;
        let names: BTreeSet<_> = self
            .persisted_profiles
            .iter()
            .chain(saved.providers.iter())
            .map(|profile| profile.profile_name.as_str())
            .collect();
        let changed_profiles: Vec<_> = names
            .into_iter()
            .filter(|name| {
                self.persisted_profiles
                    .iter()
                    .find(|p| p.profile_name == *name)
                    != saved.providers.iter().find(|p| p.profile_name == *name)
            })
            .map(str::to_owned)
            .collect();
        let mut profiles = self.profiles_from_saved(&saved, &self.locally_removed_profiles);
        let result = change(&mut saved, &mut profiles)?;
        validate_profiles(&profiles, &self.codecs, &self.authenticators)?;
        write(dir, &saved)?;
        for name in changed_profiles {
            self.model_sources.remove(&name);
            self.invalidated_model_sources.insert(name);
        }
        self.install_saved(saved, profiles);
        Ok(result)
    }

    /// Set the local configuration directory and load its saved profiles.
    /// Saved profiles replace current profiles with the same name.
    pub fn set_config_dir(&mut self, path: impl AsRef<Path>) -> Result<(), ProviderStoreError> {
        let dir = path.as_ref();
        fs::create_dir_all(dir)?;
        let dir = dir.canonicalize()?;
        let saved = read(&dir)?;
        let candidate = self.profiles_from_saved(&saved, &BTreeSet::new());
        validate_profiles(&candidate, &self.codecs, &self.authenticators)?;
        self.install_saved(saved, candidate);
        self.locally_removed_profiles.clear();
        self.model_sources.clear();
        self.invalidated_model_sources.clear();
        self.config_dir = Some(dir);
        Ok(())
    }

    /// Add or replace one account profile, keeping it across client restarts.
    pub fn add_provider(&mut self, profile: ProviderProfile) -> Result<(), ProviderStoreError> {
        let name = profile.profile_name.clone();
        let source = profile.clone();
        self.update_saved(|saved, profiles| {
            replace(profiles, profile.clone());
            replace(&mut saved.providers, profile);
            saved.deleted_profiles.remove(&name);
            Ok(())
        })?;
        self.locally_removed_profiles.remove(&name);
        self.invalidated_model_sources.remove(&name);
        self.model_sources.insert(name, source);
        Ok(())
    }

    /// Return the full configuration for one account profile.
    pub fn provider(&self, profile_name: &str) -> Option<&ProviderProfile> {
        self.profiles
            .iter()
            .find(|p| p.profile_name == profile_name)
    }

    /// Names of built-in profiles currently soft-deleted in this directory.
    pub fn deleted_builtin_profiles(&self) -> &BTreeSet<String> {
        &self.deleted_profiles
    }

    /// Remove one account profile. Built-ins are soft-deleted so they do not
    /// reappear on restart; custom profiles are removed from the local file.
    pub fn remove_provider(&mut self, profile_name: &str) -> Result<(), ProviderStoreError> {
        self.update_saved(|saved, profiles| {
            if !profiles.iter().any(|p| p.profile_name == profile_name) {
                return Err(ProviderStoreError::UnknownProfile(profile_name.to_owned()));
            }
            profiles.retain(|p| p.profile_name != profile_name);
            saved.providers.retain(|p| p.profile_name != profile_name);
            if crate::presets::is_builtin_profile(profile_name) {
                saved.deleted_profiles.insert(profile_name.to_owned());
            } else {
                saved.deleted_profiles.remove(profile_name);
            }
            Ok(())
        })?;
        if self
            .base_profiles
            .iter()
            .any(|p| p.profile_name == profile_name)
        {
            self.locally_removed_profiles
                .insert(profile_name.to_owned());
        }
        self.model_sources.remove(profile_name);
        self.invalidated_model_sources.remove(profile_name);
        Ok(())
    }

    /// Restore a soft-deleted built-in profile from the bundled preset.
    pub fn restore_builtin(&mut self, profile_name: &str) -> Result<(), ProviderStoreError> {
        if !crate::presets::is_builtin_profile(profile_name) {
            return Err(ProviderStoreError::NotBuiltin(profile_name.to_owned()));
        }
        let preset = crate::presets::builtin()?
            .into_iter()
            .find(|p| p.profile_name == profile_name)
            .ok_or_else(|| ProviderStoreError::UnknownProfile(profile_name.to_owned()))?;
        let source = preset.clone();
        self.update_saved(|saved, profiles| {
            replace(profiles, preset.clone());
            replace(&mut saved.providers, preset);
            saved.deleted_profiles.remove(profile_name);
            Ok(())
        })?;
        self.locally_removed_profiles.remove(profile_name);
        self.invalidated_model_sources.remove(profile_name);
        self.model_sources.insert(profile_name.to_owned(), source);
        Ok(())
    }

    /// Replace the tracked request-model IDs for every account of a provider.
    /// Unlisted models are omitted from listings and saved snapshots.
    pub fn set_tracked_models(
        &mut self,
        provider_id: &str,
        request_models: impl IntoIterator<Item = String>,
    ) -> Result<(), ProviderStoreError> {
        let models: BTreeSet<String> = request_models.into_iter().collect();
        let expected_profiles = self.persisted_profiles.clone();
        let sources: BTreeMap<_, _> = self
            .base_profiles
            .iter()
            .chain(self.model_sources.values())
            .filter(|profile| {
                profile.provider_id.as_str() == provider_id
                    && !self
                        .invalidated_model_sources
                        .contains(&profile.profile_name)
            })
            .map(|profile| (profile.profile_name.clone(), profile.clone()))
            .collect();
        self.update_saved(|saved, profiles| {
            let newly_tracked: BTreeSet<_> = match saved.tracked_models.get(provider_id) {
                Some(previous) => models.difference(previous).cloned().collect(),
                None => models.clone(),
            };
            saved.tracked_models.insert(provider_id.to_owned(), models);
            for profile in profiles
                .iter_mut()
                .filter(|p| p.provider_id.as_str() == provider_id)
            {
                let Some(source) = sources.get(&profile.profile_name) else {
                    continue;
                };
                if saved
                    .providers
                    .iter()
                    .find(|p| p.profile_name == profile.profile_name)
                    != expected_profiles
                        .iter()
                        .find(|p| p.profile_name == profile.profile_name)
                {
                    continue;
                }
                if !same_profile_config(profile, source) {
                    continue;
                }
                for model in &source.models {
                    if newly_tracked.contains(&model.request_model)
                        && !profile
                            .models
                            .iter()
                            .any(|m| m.request_model == model.request_model)
                    {
                        profile.models.push(model.clone());
                    }
                }
                if saved
                    .providers
                    .iter()
                    .any(|p| p.profile_name == profile.profile_name)
                {
                    replace(&mut saved.providers, profile.clone());
                }
            }
            Ok(())
        })
    }

    /// Stop tracking one model for every account of a provider.
    pub fn untrack_model(
        &mut self,
        provider_id: &str,
        request_model: &str,
    ) -> Result<(), ProviderStoreError> {
        self.update_saved(|saved, _| {
            let models = saved.tracked_models.get_mut(provider_id).ok_or_else(|| {
                ProviderStoreError::UntrackedModel {
                    provider_id: provider_id.to_owned(),
                    model: request_model.to_owned(),
                }
            })?;
            if !models.remove(request_model) {
                return Err(ProviderStoreError::UntrackedModel {
                    provider_id: provider_id.to_owned(),
                    model: request_model.to_owned(),
                });
            }
            Ok(())
        })
    }

    /// Request-model IDs currently tracked for a provider.
    pub fn tracked_models(&self, provider_id: &str) -> Option<&BTreeSet<String>> {
        self.tracked_models.get(provider_id)
    }

    /// Change whether one account's model appears in `models()` listings.
    /// The model remains available to explicitly qualified requests.
    pub fn set_model_visibility(
        &mut self,
        profile_name: &str,
        request_model: &str,
        visible: bool,
    ) -> Result<(), ProviderStoreError> {
        self.update_saved(|saved, profiles| {
            let profile = profiles
                .iter_mut()
                .find(|p| p.profile_name == profile_name)
                .ok_or_else(|| ProviderStoreError::UnknownProfile(profile_name.to_owned()))?;
            let model = profile
                .models
                .iter_mut()
                .find(|m| m.request_model == request_model)
                .ok_or_else(|| ProviderStoreError::UnknownModel {
                    profile_name: profile_name.to_owned(),
                    model: request_model.to_owned(),
                })?;
            model.hidden = !visible;
            replace(&mut saved.providers, profile.clone());
            Ok(())
        })?;
        if let Some(profile) = self.provider(profile_name).cloned() {
            self.remember_models(profile);
        }
        Ok(())
    }

    /// Refresh one account's model list with that account's own credential.
    /// Other accounts in the same connection group are untouched.
    pub async fn sync_provider(
        &mut self,
        profile_name: &str,
        credential: Option<&Secret<String>>,
    ) -> Result<usize, ProviderStoreError> {
        if self.config_dir.is_none() {
            return Err(ProviderStoreError::NotConfigured);
        }
        let dir = self
            .config_dir
            .as_ref()
            .ok_or(ProviderStoreError::NotConfigured)?;
        let current = read(dir)?;
        let profile = self
            .profiles_from_saved(&current, &self.locally_removed_profiles)
            .into_iter()
            .find(|p| p.profile_name == profile_name)
            .ok_or_else(|| ProviderStoreError::UnknownProfile(profile_name.to_owned()))?;
        let source_profile = profile.clone();
        let directory = self
            .directory_for(&profile)
            .ok_or_else(|| ProviderStoreError::NoDirectory(profile_name.to_owned()))?;
        let mut cursor = None;
        let mut seen = BTreeSet::new();
        let mut live = Vec::new();
        for _ in 0..MAX_PAGES {
            let mut request = directory.list_request(&profile, cursor.as_deref());
            if profile.auth != AuthStrategy::None {
                let auth = self.authenticators.get(&profile.auth).ok_or_else(|| {
                    ProviderStoreError::Build(BuildError::MissingAuthenticator {
                        profile_name: profile_name.to_owned(),
                        strategy: profile.auth,
                    })
                })?;
                auth.apply(&mut request, &profile, credential).await?;
            }
            let response = self.http.execute(request).await?;
            let page = directory.decode_page(&response)?;
            live.extend(page.models);
            match page.next_cursor {
                None => {
                    let count = self.update_saved(|saved, profiles| {
                        let profile = profiles
                            .iter_mut()
                            .find(|p| p.profile_name == profile_name)
                            .ok_or_else(|| {
                                ProviderStoreError::UnknownProfile(profile_name.to_owned())
                            })?;
                        if !same_profile_config(profile, &source_profile) {
                            return Err(ProviderStoreError::ProfileChanged(
                                profile_name.to_owned(),
                            ));
                        }
                        let tracked = saved.tracked_models.get(profile.provider_id.as_str());
                        let mut count = 0;
                        for model in live {
                            if tracked.is_some_and(|ids| ids.contains(&model.request_model)) {
                                merge_model(profile, model);
                                count += 1;
                            }
                        }
                        replace(&mut saved.providers, profile.clone());
                        Ok(count)
                    })?;
                    if let Some(profile) = self.provider(profile_name).cloned() {
                        self.remember_models(profile);
                    }
                    return Ok(count);
                }
                Some(next) if seen.insert(next.clone()) => cursor = Some(next),
                Some(_) => return Err(ProviderStoreError::InvalidPagination),
            }
        }
        Err(ProviderStoreError::InvalidPagination)
    }
}
