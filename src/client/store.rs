//! Local provider overrides and account-specific model directory refresh.

use super::{validate_profiles, BuildError, LlmClient};
use crate::auth::Authenticator;
use crate::directory::{LiveModel, ModelDirectory};
use crate::presets::PresetError;
use crate::transport::Transport;
use lingxi_agent_api::protocol::{
    AuthStrategy, CredentialConfig, LlmError, ModelProfile, ProviderProfile, Secret,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
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
    #[error("provider profile {0:?} has a credential-bearing header in extra.headers")]
    CredentialHeader(String),
    #[error("provider profile {0:?} is duplicated in the saved configuration")]
    DuplicateProfile(String),
    #[error("unsupported provider configuration version {0}")]
    UnsupportedVersion(u32),
    #[error("model directory pagination exceeded {MAX_PAGES} pages or repeated a cursor")]
    InvalidPagination,
    #[error("provider configuration worker failed: {0}")]
    Worker(String),
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
    /// Negative availability is separate from the whitelist: a provider may
    /// explicitly list a model that this client's completion protocol cannot
    /// call, and that fact must survive a later whitelist expansion.
    #[serde(default)]
    incompatible_models: BTreeMap<String, BTreeSet<String>>,
    #[serde(default)]
    deleted_profiles: BTreeSet<String>,
    providers: Vec<ProviderProfile>,
}

/// Persistence overlays and caches kept separately from the execution snapshot.
pub(super) struct ProviderStore {
    base_profiles: Vec<ProviderProfile>,
    locally_removed_profiles: BTreeSet<String>,
    model_sources: BTreeMap<String, ProviderProfile>,
    invalidated_model_sources: BTreeSet<String>,
    persisted_profiles: Vec<ProviderProfile>,
    config_dir: Option<PathBuf>,
    config_generation: u64,
    deleted_profiles: BTreeSet<String>,
    tracked_models: BTreeMap<String, BTreeSet<String>>,
}

/// An owned provider-directory fetch prepared from one client snapshot.
///
/// It contains no reference to `LlmClient`, so callers can drop any client
/// borrow before awaiting [`Self::fetch`].
pub struct ProviderSyncOperation {
    directory_path: PathBuf,
    config_generation: u64,
    profile_name: String,
    source_profile: ProviderProfile,
    base_profiles: Vec<ProviderProfile>,
    locally_removed_profiles: BTreeSet<String>,
    directory: std::sync::Arc<dyn ModelDirectory>,
    authenticator: Option<std::sync::Arc<dyn Authenticator>>,
    http: std::sync::Arc<dyn Transport>,
    credential: Option<Secret<String>>,
}

/// Results fetched by [`ProviderSyncOperation`], ready for a short commit.
///
/// Results are tied to the profile and configuration-directory generation
/// from which they were prepared. Applying an obsolete result returns
/// [`ProviderStoreError::ProfileChanged`].
pub struct ProviderSyncResult {
    directory_path: PathBuf,
    config_generation: u64,
    profile_name: String,
    source_profile: ProviderProfile,
    live: Vec<LiveModel>,
    incompatible_models: Vec<String>,
    explicitly_compatible_models: Vec<String>,
}

struct SyncCommit {
    saved: SavedProviders,
    profiles: Vec<ProviderProfile>,
    changed_profiles: Vec<String>,
    count: usize,
}

struct CancelCommitOnDrop(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Drop for CancelCommitOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
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

fn reject_credential_headers(profiles: &[ProviderProfile]) -> Result<(), ProviderStoreError> {
    for profile in profiles {
        let Some(headers) = profile
            .extra
            .get("headers")
            .and_then(serde_json::Value::as_object)
        else {
            continue;
        };
        if headers
            .keys()
            .any(|name| crate::codecs::extras::is_credential_header(profile, name))
        {
            return Err(ProviderStoreError::CredentialHeader(
                profile.profile_name.clone(),
            ));
        }
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
                incompatible_models: BTreeMap::new(),
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
    reject_credential_headers(&saved.providers)?;
    Ok(saved)
}

fn store_lock(dir: &Path) -> Result<File, ProviderStoreError> {
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join(".providers.json.lock"))?;
    lock.lock()?;
    Ok(lock)
}

fn read_locked(dir: &Path) -> Result<SavedProviders, ProviderStoreError> {
    let _lock = store_lock(dir)?;
    read(dir)
}

fn filtered_profiles(saved: &SavedProviders) -> Vec<ProviderProfile> {
    let mut filtered = saved.providers.clone();
    for profile in &mut filtered {
        let tracked = saved.tracked_models.get(profile.provider_id.as_str());
        let incompatible = saved.incompatible_models.get(&profile.profile_name);
        profile.models.retain(|model| {
            tracked.is_some_and(|ids| ids.contains(&model.request_model))
                && !incompatible.is_some_and(|ids| ids.contains(&model.request_model))
        });
    }
    filtered
}

fn profiles_from_saved(
    base_profiles: &[ProviderProfile],
    saved: &SavedProviders,
    locally_removed: &BTreeSet<String>,
) -> Vec<ProviderProfile> {
    let mut profiles: Vec<_> = base_profiles
        .iter()
        .filter(|profile| !locally_removed.contains(&profile.profile_name))
        .cloned()
        .collect();
    for profile in &saved.providers {
        replace(&mut profiles, profile.clone());
    }
    profiles.retain(|profile| !saved.deleted_profiles.contains(&profile.profile_name));
    for profile in &mut profiles {
        if let Some(incompatible) = saved.incompatible_models.get(&profile.profile_name) {
            profile
                .models
                .retain(|model| !incompatible.contains(&model.request_model));
        }
    }
    profiles
}

fn changed_profile_names(
    persisted_profiles: &[ProviderProfile],
    saved: &SavedProviders,
) -> Vec<String> {
    let names: BTreeSet<_> = persisted_profiles
        .iter()
        .chain(saved.providers.iter())
        .map(|profile| profile.profile_name.as_str())
        .collect();
    names
        .into_iter()
        .filter(|name| {
            persisted_profiles
                .iter()
                .find(|profile| profile.profile_name == *name)
                != saved
                    .providers
                    .iter()
                    .find(|profile| profile.profile_name == *name)
        })
        .map(str::to_owned)
        .collect()
}

fn write(dir: &Path, saved: &SavedProviders) -> Result<(), ProviderStoreError> {
    reject_static(&saved.providers)?;
    reject_credential_headers(&saved.providers)?;
    let bytes = serde_json::to_vec_pretty(&SavedProviders {
        version: 1,
        tracked_models: saved.tracked_models.clone(),
        incompatible_models: saved.incompatible_models.clone(),
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
            capability_support: None,
            pricing: None,
            billing_mode: None,
        };
        model.metadata.context_window_tokens = live.context_window;
        model.metadata.max_output_tokens = live.max_output_tokens;
        profile.models.push(model);
    }
}

fn worker_error(error: tokio::task::JoinError) -> ProviderStoreError {
    ProviderStoreError::Worker(error.to_string())
}

impl ProviderSyncOperation {
    /// Fetch every page from the provider using only owned state.
    ///
    /// Filesystem reads needed to validate the snapshot run on Tokio's blocking
    /// pool. This future owns all inputs and does not borrow the client.
    pub async fn fetch(self) -> Result<ProviderSyncResult, ProviderStoreError> {
        let directory_path = self.directory_path.clone();
        let base_profiles = self.base_profiles.clone();
        let locally_removed_profiles = self.locally_removed_profiles.clone();
        let profile_name = self.profile_name.clone();
        let source_profile = self.source_profile.clone();
        let current_profile = tokio::task::spawn_blocking(move || {
            let saved = read_locked(&directory_path)?;
            let profile = profiles_from_saved(&base_profiles, &saved, &locally_removed_profiles)
                .into_iter()
                .find(|profile| profile.profile_name == profile_name)
                .ok_or_else(|| ProviderStoreError::UnknownProfile(profile_name.clone()))?;
            if !same_profile_config(&profile, &source_profile) {
                return Err(ProviderStoreError::ProfileChanged(profile_name));
            }
            Ok(profile)
        })
        .await
        .map_err(worker_error)??;

        let mut cursor = None;
        let mut seen = BTreeSet::new();
        let mut live = Vec::new();
        let mut incompatible_models = BTreeSet::new();
        let mut explicitly_compatible_models = BTreeSet::new();
        for _ in 0..MAX_PAGES {
            let mut request = self
                .directory
                .list_request(&current_profile, cursor.as_deref());
            if current_profile.auth != AuthStrategy::None {
                let authenticator = self.authenticator.as_ref().ok_or_else(|| {
                    ProviderStoreError::Build(BuildError::MissingAuthenticator {
                        profile_name: self.profile_name.clone(),
                        strategy: current_profile.auth,
                    })
                })?;
                authenticator
                    .apply(&mut request, &current_profile, self.credential.as_ref())
                    .await?;
            }
            let response = self.http.execute(request).await?;
            let decoded = self.directory.decode_page_with_exclusions(&response)?;
            live.extend(decoded.page.models);
            incompatible_models.extend(decoded.incompatible_model_ids);
            explicitly_compatible_models.extend(decoded.explicitly_compatible_model_ids);
            match decoded.page.next_cursor {
                None => {
                    return Ok(ProviderSyncResult {
                        directory_path: self.directory_path,
                        config_generation: self.config_generation,
                        profile_name: self.profile_name,
                        source_profile: self.source_profile,
                        live,
                        incompatible_models: incompatible_models.into_iter().collect(),
                        explicitly_compatible_models: explicitly_compatible_models
                            .into_iter()
                            .collect(),
                    });
                }
                Some(next) if seen.insert(next.clone()) => cursor = Some(next),
                Some(_) => return Err(ProviderStoreError::InvalidPagination),
            }
        }
        Err(ProviderStoreError::InvalidPagination)
    }
}

impl ProviderStore {
    pub(super) fn new(base_profiles: Vec<ProviderProfile>) -> Self {
        Self {
            base_profiles,
            locally_removed_profiles: BTreeSet::new(),
            model_sources: BTreeMap::new(),
            invalidated_model_sources: BTreeSet::new(),
            persisted_profiles: Vec::new(),
            config_dir: None,
            config_generation: 0,
            deleted_profiles: BTreeSet::new(),
            tracked_models: BTreeMap::new(),
        }
    }

    pub(super) fn tracks(&self, profile: &ProviderProfile, model: &ModelProfile) -> bool {
        self.config_dir.is_none()
            || self
                .tracked_models
                .get(profile.provider_id.as_str())
                .is_some_and(|ids| ids.contains(&model.request_model))
    }

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

    fn profiles_from_saved(&self, saved: &SavedProviders) -> Vec<ProviderProfile> {
        profiles_from_saved(&self.base_profiles, saved, &self.locally_removed_profiles)
    }

    fn install_saved(
        &mut self,
        execution_profiles: &mut Vec<ProviderProfile>,
        saved: SavedProviders,
        profiles: Vec<ProviderProfile>,
    ) {
        self.persisted_profiles = filtered_profiles(&saved);
        *execution_profiles = profiles;
        self.deleted_profiles = saved.deleted_profiles;
        self.tracked_models = saved.tracked_models;
    }

    /// Apply one change to the latest on-disk state while other clients wait.
    fn update_saved<T>(
        &mut self,
        execution_profiles: &mut Vec<ProviderProfile>,
        codecs: &BTreeMap<
            lingxi_agent_api::protocol::ProtocolFamily,
            std::sync::Arc<dyn crate::codecs::WireCodec>,
        >,
        authenticators: &BTreeMap<AuthStrategy, std::sync::Arc<dyn Authenticator>>,
        change: impl FnOnce(
            &mut SavedProviders,
            &mut Vec<ProviderProfile>,
        ) -> Result<T, ProviderStoreError>,
    ) -> Result<T, ProviderStoreError> {
        let dir = self
            .config_dir
            .as_ref()
            .ok_or(ProviderStoreError::NotConfigured)?;
        let _lock = store_lock(dir)?;
        let mut saved = read(dir)?;
        let changed_profiles = changed_profile_names(&self.persisted_profiles, &saved);
        let mut profiles = self.profiles_from_saved(&saved);
        let result = change(&mut saved, &mut profiles)?;
        validate_profiles(&profiles, codecs, authenticators)?;
        write(dir, &saved)?;
        for name in changed_profiles {
            self.model_sources.remove(&name);
            self.invalidated_model_sources.insert(name);
        }
        self.install_saved(execution_profiles, saved, profiles);
        Ok(result)
    }
}

impl LlmClient {
    fn retain_account_sources_for_unchanged_profiles(&mut self, previous: &[ProviderProfile]) {
        let unchanged: BTreeSet<_> = self
            .profiles
            .iter()
            .filter(|profile| {
                previous.iter().any(|old| {
                    old.profile_name == profile.profile_name && same_profile_config(old, profile)
                })
            })
            .map(|profile| profile.profile_name.as_str())
            .collect();
        self.profile_account_sources
            .retain(|(name, _), _| unchanged.contains(name.as_str()));
    }

    fn update_saved<T>(
        &mut self,
        change: impl FnOnce(
            &mut SavedProviders,
            &mut Vec<ProviderProfile>,
        ) -> Result<T, ProviderStoreError>,
    ) -> Result<T, ProviderStoreError> {
        let previous = self.profiles.clone();
        let result = self.store.update_saved(
            &mut self.profiles,
            &self.codecs,
            &self.authenticators,
            change,
        )?;
        self.retain_account_sources_for_unchanged_profiles(&previous);
        Ok(result)
    }

    fn remember_models(&mut self, profile: ProviderProfile) {
        self.store.remember_models(profile);
    }

    /// Set the local configuration directory and load its saved profiles.
    /// Saved profiles replace current profiles with the same name.
    pub fn set_config_dir(&mut self, path: impl AsRef<Path>) -> Result<(), ProviderStoreError> {
        let dir = path.as_ref();
        fs::create_dir_all(dir)?;
        let dir = dir.canonicalize()?;
        let saved = read(&dir)?;
        let candidate = profiles_from_saved(&self.store.base_profiles, &saved, &BTreeSet::new());
        validate_profiles(&candidate, &self.codecs, &self.authenticators)?;
        let previous = self.profiles.clone();
        let switched_directory = self
            .store
            .config_dir
            .as_ref()
            .is_some_and(|current| current != &dir);
        self.store
            .install_saved(&mut self.profiles, saved, candidate);
        if switched_directory {
            self.profile_account_sources.clear();
        } else {
            self.retain_account_sources_for_unchanged_profiles(&previous);
        }
        self.store.locally_removed_profiles.clear();
        self.store.model_sources.clear();
        self.store.invalidated_model_sources.clear();
        self.store.config_dir = Some(dir);
        self.store.config_generation = self.store.config_generation.wrapping_add(1);
        Ok(())
    }

    /// Add or replace one account profile, keeping it across client restarts.
    pub fn add_provider(&mut self, profile: ProviderProfile) -> Result<(), ProviderStoreError> {
        let name = profile.profile_name.clone();
        let source = profile.clone();
        self.update_saved(|saved, profiles| {
            saved.incompatible_models.remove(&name);
            replace(profiles, profile.clone());
            replace(&mut saved.providers, profile);
            saved.deleted_profiles.remove(&name);
            Ok(())
        })?;
        self.store.locally_removed_profiles.remove(&name);
        self.store.invalidated_model_sources.remove(&name);
        self.profile_account_sources
            .retain(|(profile_name, _), _| profile_name != &name);
        self.store.model_sources.insert(name, source);
        Ok(())
    }

    /// Return the full configuration for one account profile.
    pub fn provider(&self, profile_name: &str) -> Option<&ProviderProfile> {
        self.profiles
            .iter()
            .find(|p| p.profile_name == profile_name)
    }

    /// Bind a signed-in account source after a profile was replaced or loaded.
    /// A changed profile drops its old binding to prevent account mix-ups.
    pub fn register_profile_account_source(
        &mut self,
        profile_name: &str,
        identity: super::account::AccountIdentity,
        source: std::sync::Arc<dyn super::account::AccountUsageSource>,
    ) -> Result<(), ProviderStoreError> {
        if self.provider(profile_name).is_none() {
            return Err(ProviderStoreError::UnknownProfile(profile_name.to_owned()));
        }
        self.profile_account_sources
            .insert((profile_name.to_owned(), identity), source);
        Ok(())
    }

    /// Names of built-in profiles currently soft-deleted in this directory.
    pub fn deleted_builtin_profiles(&self) -> &BTreeSet<String> {
        &self.store.deleted_profiles
    }

    /// Remove one account profile. Built-ins are soft-deleted so they do not
    /// reappear on restart; custom profiles are removed from the local file.
    pub fn remove_provider(&mut self, profile_name: &str) -> Result<(), ProviderStoreError> {
        self.update_saved(|saved, profiles| {
            if !profiles.iter().any(|p| p.profile_name == profile_name) {
                return Err(ProviderStoreError::UnknownProfile(profile_name.to_owned()));
            }
            saved.incompatible_models.remove(profile_name);
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
            .store
            .base_profiles
            .iter()
            .any(|p| p.profile_name == profile_name)
        {
            self.store
                .locally_removed_profiles
                .insert(profile_name.to_owned());
        }
        self.store.model_sources.remove(profile_name);
        self.store.invalidated_model_sources.remove(profile_name);
        self.profile_account_sources
            .retain(|(name, _), _| name != profile_name);
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
            saved.incompatible_models.remove(profile_name);
            replace(profiles, preset.clone());
            replace(&mut saved.providers, preset);
            saved.deleted_profiles.remove(profile_name);
            Ok(())
        })?;
        self.store.locally_removed_profiles.remove(profile_name);
        self.store.invalidated_model_sources.remove(profile_name);
        self.profile_account_sources
            .retain(|(name, _), _| name != profile_name);
        self.store
            .model_sources
            .insert(profile_name.to_owned(), source);
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
        let expected_profiles = self.store.persisted_profiles.clone();
        let sources: BTreeMap<_, _> = self
            .store
            .base_profiles
            .iter()
            .chain(self.store.model_sources.values())
            .filter(|profile| {
                profile.provider_id.as_str() == provider_id
                    && !self
                        .store
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
                let incompatible = saved.incompatible_models.get(&profile.profile_name);
                for model in &source.models {
                    if newly_tracked.contains(&model.request_model)
                        && !incompatible.is_some_and(|ids| ids.contains(&model.request_model))
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
        self.store.tracked_models.get(provider_id)
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

    /// Prepare an owned directory fetch from the current profile snapshot.
    ///
    /// This method performs no filesystem access and returns no borrow of the
    /// client. The resulting operation validates the snapshot on Tokio's
    /// blocking pool before its first network request.
    pub fn prepare_provider_sync(
        &self,
        profile_name: &str,
        credential: Option<&Secret<String>>,
    ) -> Result<ProviderSyncOperation, ProviderStoreError> {
        let directory_path = self
            .store
            .config_dir
            .clone()
            .ok_or(ProviderStoreError::NotConfigured)?;
        let profile = self
            .provider(profile_name)
            .cloned()
            .ok_or_else(|| ProviderStoreError::UnknownProfile(profile_name.to_owned()))?;
        let directory = self
            .directory_for(&profile)
            .ok_or_else(|| ProviderStoreError::NoDirectory(profile_name.to_owned()))?;
        let authenticator = if profile.auth == AuthStrategy::None {
            None
        } else {
            Some(
                self.authenticators
                    .get(&profile.auth)
                    .cloned()
                    .ok_or_else(|| {
                        ProviderStoreError::Build(BuildError::MissingAuthenticator {
                            profile_name: profile_name.to_owned(),
                            strategy: profile.auth,
                        })
                    })?,
            )
        };
        Ok(ProviderSyncOperation {
            directory_path,
            config_generation: self.store.config_generation,
            profile_name: profile_name.to_owned(),
            source_profile: profile,
            base_profiles: self.store.base_profiles.clone(),
            locally_removed_profiles: self.store.locally_removed_profiles.clone(),
            directory,
            authenticator,
            http: self.http.clone(),
            credential: credential.cloned(),
        })
    }

    /// Apply a fetched directory after checking it against current client and
    /// locked on-disk state. Filesystem work runs on Tokio's blocking pool.
    ///
    /// The worker never mutates this client. A cancellation flag is checked
    /// before the file-write phase begins. If this future is dropped after
    /// that phase starts, the blocking worker may finish its atomic rename,
    /// leaving the durable file updated while this in-memory snapshot remains
    /// unchanged. Call [`Self::set_config_dir`] again or rebuild the client to
    /// reload that state.
    pub async fn apply_provider_sync(
        &mut self,
        result: ProviderSyncResult,
    ) -> Result<usize, ProviderStoreError> {
        let profile_name = result.profile_name.clone();
        let current_dir = self
            .store
            .config_dir
            .as_ref()
            .ok_or_else(|| ProviderStoreError::ProfileChanged(profile_name.clone()))?;
        if current_dir != &result.directory_path
            || self.store.config_generation != result.config_generation
            || self
                .provider(&profile_name)
                .is_none_or(|profile| !same_profile_config(profile, &result.source_profile))
        {
            return Err(ProviderStoreError::ProfileChanged(profile_name));
        }

        let directory_path = result.directory_path;
        let worker_profile_name = profile_name.clone();
        let source_profile = result.source_profile;
        let live = result.live;
        let incompatible_models = result.incompatible_models;
        let explicitly_compatible_models = result.explicitly_compatible_models;
        let base_profiles = self.store.base_profiles.clone();
        let locally_removed_profiles = self.store.locally_removed_profiles.clone();
        let persisted_profiles = self.store.persisted_profiles.clone();
        let codecs = self.codecs.clone();
        let authenticators = self.authenticators.clone();
        let cancellation = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let _cancel_on_drop = CancelCommitOnDrop(cancellation.clone());

        let commit = tokio::task::spawn_blocking(move || {
            let _lock = store_lock(&directory_path)?;
            if cancellation.load(Ordering::Acquire) {
                return Err(ProviderStoreError::Worker(
                    "provider sync was cancelled before commit".to_owned(),
                ));
            }
            let mut saved = read(&directory_path)?;
            let changed_profiles = changed_profile_names(&persisted_profiles, &saved);
            let mut profiles =
                profiles_from_saved(&base_profiles, &saved, &locally_removed_profiles);
            let profile = profiles
                .iter_mut()
                .find(|profile| profile.profile_name == worker_profile_name)
                .ok_or_else(|| ProviderStoreError::UnknownProfile(worker_profile_name.clone()))?;
            if !same_profile_config(profile, &source_profile) {
                return Err(ProviderStoreError::ProfileChanged(worker_profile_name));
            }
            let incompatible = saved
                .incompatible_models
                .entry(worker_profile_name.clone())
                .or_default();
            incompatible.extend(incompatible_models);
            incompatible.retain(|model| !explicitly_compatible_models.contains(model));
            let incompatible = incompatible.clone();
            if incompatible.is_empty() {
                saved.incompatible_models.remove(&worker_profile_name);
            }
            profile
                .models
                .retain(|model| !incompatible.contains(&model.request_model));
            let tracked = saved.tracked_models.get(profile.provider_id.as_str());
            let mut count = 0;
            for model in live {
                if tracked.is_some_and(|ids| ids.contains(&model.request_model))
                    && !incompatible.contains(&model.request_model)
                {
                    merge_model(profile, model);
                    count += 1;
                }
            }
            replace(&mut saved.providers, profile.clone());
            validate_profiles(&profiles, &codecs, &authenticators)?;
            if cancellation.load(Ordering::Acquire) {
                return Err(ProviderStoreError::Worker(
                    "provider sync was cancelled before commit".to_owned(),
                ));
            }
            write(&directory_path, &saved)?;
            Ok(SyncCommit {
                saved,
                profiles,
                changed_profiles,
                count,
            })
        })
        .await
        .map_err(worker_error)??;

        for changed in commit.changed_profiles {
            self.store.model_sources.remove(&changed);
            self.store.invalidated_model_sources.insert(changed);
        }
        let previous = self.profiles.clone();
        self.store
            .install_saved(&mut self.profiles, commit.saved, commit.profiles);
        self.retain_account_sources_for_unchanged_profiles(&previous);
        if let Some(profile) = self.provider(&profile_name).cloned() {
            self.remember_models(profile);
        }
        Ok(commit.count)
    }

    /// Refresh one account's model list with that account's own credential.
    /// Other accounts in the same connection group are untouched.
    ///
    /// For concurrent fetches or client mutations during network I/O, use
    /// [`Self::prepare_provider_sync`], await [`ProviderSyncOperation::fetch`],
    /// then call [`Self::apply_provider_sync`].
    pub async fn sync_provider(
        &mut self,
        profile_name: &str,
        credential: Option<&Secret<String>>,
    ) -> Result<usize, ProviderStoreError> {
        let operation = self.prepare_provider_sync(profile_name, credential)?;
        let result = operation.fetch().await?;
        self.apply_provider_sync(result).await
    }
}
