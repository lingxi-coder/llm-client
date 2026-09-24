//! Configuration facade. Durable commits emit changes before services are invalidated.
use super::{snapshot::RuntimeSnapshot, validate_profiles, LlmClient};
use crate::{
    account,
    configuration::{self as config, *},
    protocol::{AuthStrategy, ProviderProfile, Secret},
};
pub use config::{ProviderStoreError, ProviderSyncOperation, ProviderSyncResult};
use std::{
    collections::BTreeSet,
    path::Path,
    sync::{atomic::AtomicBool, Arc},
};

impl LlmClient {
    fn install_configuration(&mut self, profiles: Vec<ProviderProfile>, all_bindings: bool) {
        let mut changes = config::ChangeSet::between(&self.snapshot.profiles, &profiles);
        changes.all_bindings = all_bindings;
        let changed_providers: BTreeSet<_> = self
            .snapshot
            .profiles
            .iter()
            .chain(&profiles)
            .filter(|p| changes.connections.contains(&p.profile_name))
            .map(|p| p.provider_id.as_str().to_owned())
            .collect();
        self.accounts.invalidate(
            &changes.connections,
            &changed_providers,
            changes.all_bindings,
        );
        self.attachments.invalidate_profiles(&changes.profiles);
        self.snapshot = Arc::new(RuntimeSnapshot::new(
            self.region,
            profiles,
            Some(self.store.state.tracked_models.clone()),
        ));
    }
    fn update_configuration<T>(
        &mut self,
        change: impl FnOnce(
            &mut config::SavedConfig,
            &config::Definitions,
        ) -> Result<T, ProviderStoreError>,
    ) -> Result<T, ProviderStoreError> {
        let codecs = &self.codecs;
        let auth = &self.authenticators;
        let (result, profiles) = self.store.update(change, |profiles| {
            validate_profiles(profiles, codecs, auth).map_err(Into::into)
        })?;
        self.install_configuration(profiles, false);
        Ok(result)
    }
    fn invalidate_binding(&mut self, name: &str, provider: &str) {
        self.accounts.invalidate(
            &BTreeSet::from([name.to_owned()]),
            &BTreeSet::from([provider.to_owned()]),
            false,
        );
    }

    /// Load a v2 configuration without modifying its file. Unsupported versions are rejected.
    pub fn set_config_dir(&mut self, path: impl AsRef<Path>) -> Result<(), ProviderStoreError> {
        let previous = self.store.repository.as_ref().map(|r| r.path.clone());
        let codecs = &self.codecs;
        let auth = &self.authenticators;
        let profiles = self.store.load(path.as_ref(), |profiles| {
            validate_profiles(profiles, codecs, auth).map_err(Into::into)
        })?;
        let switched =
            previous.is_some_and(|p| Some(&p) != self.store.repository.as_ref().map(|r| &r.path));
        self.install_configuration(profiles, switched);
        Ok(())
    }
    /// Fully replace an account and its models. Every supplied model field is an explicit override.
    /// Use `set_model_override` to let other fields continue following the catalog.
    pub fn add_provider(&mut self, profile: ProviderProfile) -> Result<(), ProviderStoreError> {
        let name = profile.profile_name.clone();
        let provider = profile.provider_id.as_str().to_owned();
        self.update_configuration(|state, definitions| {
            let saved = SavedProfile::replacement(&profile, definitions.reference(&name));
            if let Some(old) = state
                .providers
                .iter_mut()
                .find(|p| p.connection.profile_name == name)
            {
                *old = saved;
            } else {
                state.providers.push(saved);
            }
            state.deleted_profiles.remove(&name);
            Ok(())
        })?;
        self.invalidate_binding(&name, &provider);
        Ok(())
    }
    pub fn provider(&self, name: &str) -> Option<&ProviderProfile> {
        self.snapshot.profile(name)
    }
    pub fn register_profile_account_source(
        &mut self,
        profile_name: &str,
        identity: account::AccountIdentity,
        source: Arc<dyn account::AccountUsageSource>,
    ) -> Result<(), ProviderStoreError> {
        if self.provider(profile_name).is_none() {
            return Err(ProviderStoreError::UnknownProfile(profile_name.into()));
        }
        self.accounts.bind(profile_name.into(), identity, source);
        Ok(())
    }
    pub fn deleted_builtin_profiles(&self) -> &BTreeSet<String> {
        &self.store.deleted_builtin_profiles
    }
    pub fn remove_provider(&mut self, name: &str) -> Result<(), ProviderStoreError> {
        self.update_configuration(|state, _| {
            config::profile(state, name)?;
            state
                .providers
                .retain(|p| p.connection.profile_name != name);
            state.deleted_profiles.insert(name.into());
            Ok(())
        })
    }
    pub fn restore_builtin(&mut self, name: &str) -> Result<(), ProviderStoreError> {
        let preset = crate::presets::builtin()?
            .into_iter()
            .find(|p| p.profile_name == name)
            .ok_or_else(|| ProviderStoreError::NotBuiltin(name.into()))?;
        let provider = preset.provider_id.as_str().to_owned();
        self.update_configuration(|state, _| {
            let saved =
                SavedProfile::inherited(&preset, DefinitionRef::Builtin { name: name.into() });
            if let Some(old) = state
                .providers
                .iter_mut()
                .find(|p| p.connection.profile_name == name)
            {
                *old = saved;
            } else {
                state.providers.push(saved);
            }
            state.deleted_profiles.remove(name);
            Ok(())
        })?;
        self.invalidate_binding(name, &provider);
        Ok(())
    }
    /// Allowlist controls tracking, persistence and presentation, never explicit routing.
    pub fn set_tracked_models(
        &mut self,
        provider: &str,
        models: impl IntoIterator<Item = String>,
    ) -> Result<(), ProviderStoreError> {
        let models = models.into_iter().collect();
        self.update_configuration(|state, _| {
            state.tracked_models.insert(provider.into(), models);
            Ok(())
        })
    }
    pub fn untrack_model(&mut self, provider: &str, model: &str) -> Result<(), ProviderStoreError> {
        self.update_configuration(|state, _| {
            if !state
                .tracked_models
                .get_mut(provider)
                .is_some_and(|ids| ids.remove(model))
            {
                return Err(ProviderStoreError::UntrackedModel {
                    provider_id: provider.into(),
                    model: model.into(),
                });
            }
            Ok(())
        })
    }
    pub fn tracked_models(&self, provider: &str) -> Option<&BTreeSet<String>> {
        self.store.state.tracked_models.get(provider)
    }
    /// Return ordered rows, including temporarily incompatible models and stable row identities.
    pub fn configured_models(
        &self,
        profile: &str,
    ) -> Result<Vec<ConfiguredModel>, ProviderStoreError> {
        self.store.configured_models(profile)
    }
    pub fn set_model_visibility(
        &mut self,
        name: &str,
        wire: &str,
        visible: bool,
    ) -> Result<(), ProviderStoreError> {
        self.update_configuration(|state, definitions| {
            let profile = config::profile(state, name)?;
            let rows = profile.rows(definitions)?;
            let mut matches = rows
                .into_iter()
                .filter(|r| r.model.request_model == wire && r.compatible);
            let row = matches
                .next()
                .ok_or_else(|| ProviderStoreError::UnknownModel {
                    profile_name: name.into(),
                    model: wire.into(),
                })?;
            if matches.next().is_some() {
                return Err(ProviderStoreError::InvalidModelOverride(
                    "wire model matches multiple rows; select a row ID".into(),
                ));
            }
            profile
                .ensure_row(&row.row_id, definitions)?
                .overrides
                .insert(ModelField::Hidden, serde_json::Value::Bool(!visible));
            Ok(())
        })
    }
    /// Set one explicit field on an unambiguous row. `Clear` resets its value; `Inherit` removes the override.
    pub fn set_model_override(
        &mut self,
        profile_name: &str,
        row_id: &str,
        field: ModelField,
        value: FieldOverride<serde_json::Value>,
    ) -> Result<(), ProviderStoreError> {
        self.update_configuration(|state, definitions| {
            let profile = config::profile(state, profile_name)?;
            let current = profile
                .rows(definitions)?
                .into_iter()
                .find(|r| r.row_id == row_id)
                .ok_or_else(|| ProviderStoreError::UnknownModel {
                    profile_name: profile_name.into(),
                    model: row_id.into(),
                })?;
            let row = profile.ensure_row(row_id, definitions)?;
            match value {
                FieldOverride::Inherit => {
                    row.overrides.remove(&field);
                }
                FieldOverride::Set(value) => {
                    row.overrides.insert(field, value);
                }
                FieldOverride::Clear => {
                    row.overrides.insert(
                        field,
                        config::values(&config::empty_model(&current.model.request_model))[&field]
                            .clone(),
                    );
                }
            }
            profile.rows(definitions)?;
            Ok(())
        })
    }
    pub fn clear_model_override(
        &mut self,
        profile: &str,
        row: &str,
        field: ModelField,
    ) -> Result<(), ProviderStoreError> {
        self.set_model_override(profile, row, field, FieldOverride::Inherit)
    }
    /// Replace all fields of one row while preserving its identity and position.
    pub fn replace_model(
        &mut self,
        name: &str,
        row_id: &str,
        model: crate::protocol::ModelProfile,
    ) -> Result<(), ProviderStoreError> {
        self.update_configuration(|state, definitions| {
            let profile = config::profile(state, name)?;
            if !profile
                .rows(definitions)?
                .iter()
                .any(|r| r.row_id == row_id)
            {
                return Err(ProviderStoreError::UnknownModel {
                    profile_name: name.into(),
                    model: row_id.into(),
                });
            }
            let key = profile
                .ensure_row(row_id, definitions)?
                .definition_key
                .clone();
            if let Some(key) = key {
                profile.removed_defaults.insert(key);
            }
            let row = profile.ensure_row(row_id, definitions)?;
            row.replacement = true;
            row.initial = Some(config::empty_model(&model.request_model));
            row.overrides = config::values(&model);
            Ok(())
        })
    }
    pub fn prepare_provider_sync(
        &self,
        name: &str,
        credential: Option<&Secret<String>>,
    ) -> Result<ProviderSyncOperation, ProviderStoreError> {
        let repository = self
            .store
            .repository
            .clone()
            .ok_or(ProviderStoreError::NotConfigured)?;
        let source = self
            .provider(name)
            .cloned()
            .ok_or_else(|| ProviderStoreError::UnknownProfile(name.into()))?;
        let directory = self
            .directory_for(&source)
            .ok_or_else(|| ProviderStoreError::NoDirectory(name.into()))?;
        let authenticator = if source.auth == AuthStrategy::None {
            None
        } else {
            Some(
                self.authenticators
                    .get(&source.auth)
                    .cloned()
                    .ok_or_else(|| super::BuildError::MissingAuthenticator {
                        profile_name: name.into(),
                        strategy: source.auth,
                    })?,
            )
        };
        Ok(ProviderSyncOperation {
            repository,
            generation: self.store.generation,
            source,
            definitions: self.store.definitions.clone(),
            directory,
            authenticator,
            http: self.http.clone(),
            credential: credential.cloned(),
        })
    }
    pub async fn apply_provider_sync(
        &mut self,
        result: ProviderSyncResult,
    ) -> Result<usize, ProviderStoreError> {
        let name = result.source.profile_name.clone();
        if self
            .store
            .repository
            .as_ref()
            .is_none_or(|r| r.path != result.repository.path)
            || self.store.generation != result.generation
            || self
                .provider(&name)
                .is_none_or(|p| !config::same_connection(p, &result.source))
        {
            return Err(ProviderStoreError::ProfileChanged(name));
        }
        let definitions = self.store.definitions.clone();
        let previous = self.store.state.clone();
        let codecs = self.codecs.clone();
        let auth = self.authenticators.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let _guard = config::CancelCommit(cancel.clone());
        let (state, profiles, count) = tokio::task::spawn_blocking(move || {
            result.commit(&definitions, &previous, &cancel, |profiles| {
                validate_profiles(profiles, &codecs, &auth).map_err(Into::into)
            })
        })
        .await
        .map_err(|e| ProviderStoreError::Worker(e.to_string()))??;
        self.store.install(state);
        self.install_configuration(profiles, false);
        Ok(count)
    }
    pub async fn sync_provider(
        &mut self,
        name: &str,
        credential: Option<&Secret<String>>,
    ) -> Result<usize, ProviderStoreError> {
        let operation = self.prepare_provider_sync(name, credential)?;
        self.apply_provider_sync(operation.fetch().await?).await
    }
}
