//! Serialized configuration transactions and atomic publication to shared clients.
use super::{snapshot::RuntimeSnapshot, validate_profiles, LlmClient, PublishedState};
use crate::{
    account,
    configuration::{self as config, *},
    protocol::{AuthStrategy, ProviderProfile, Secret},
};
pub use config::{ProviderStoreError, ProviderSyncOperation, ProviderSyncResult};
use std::{
    collections::BTreeSet,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

/// The configuration writer paired with a shared [`LlmClient`].
///
/// File access, validation and publication run on blocking workers. Transactions
/// are serialized without holding a request-side lock across I/O. Once a commit
/// starts writing, cancellation cannot prevent its snapshot from being published.
pub struct ClientConfigManager {
    client: LlmClient,
    store: Arc<Mutex<config::Coordinator>>,
}

impl ClientConfigManager {
    pub(crate) fn new(client: LlmClient, store: config::Coordinator) -> Self {
        Self {
            client,
            store: Arc::new(Mutex::new(store)),
        }
    }

    async fn run<T: Send + 'static>(
        &self,
        operation: impl FnOnce(
                &mut config::Coordinator,
                &LlmClient,
                &AtomicBool,
            ) -> Result<T, ProviderStoreError>
            + Send
            + 'static,
    ) -> Result<T, ProviderStoreError> {
        let store = self.store.clone();
        let client = self.client.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let _guard = config::CancelCommit(cancel.clone());
        tokio::task::spawn_blocking(move || {
            let mut store = store
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if cancel.load(Ordering::Acquire) {
                return Err(ProviderStoreError::Worker(
                    "configuration operation was cancelled before commit".into(),
                ));
            }
            operation(&mut store, &client, &cancel)
        })
        .await
        .map_err(|error| ProviderStoreError::Worker(error.to_string()))?
    }

    fn install_published(client: &LlmClient, state: PublishedState) {
        let state = Arc::new(state);
        let previous = {
            let mut published = client
                .published
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::mem::replace(&mut *published, state)
        };
        // Releasing the last reader can destroy a full catalog; do that after
        // releasing the publication lock as well.
        drop(previous);
    }

    fn publish(
        client: &LlmClient,
        store: &mut config::Coordinator,
        profiles: Vec<ProviderProfile>,
        switched_directory: bool,
        forced_binding: Option<&str>,
    ) {
        #[cfg(test)]
        if let Some(before_publication) = store.before_publication.take() {
            before_publication();
        }
        let previous = client
            .published
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let revision = previous
            .revision
            .checked_add(1)
            .expect("configuration revision exhausted");
        let mut changes = config::ChangeSet::between(&previous.config.profiles, &profiles);
        changes.all_bindings = switched_directory;
        if let Some(name) = forced_binding {
            changes.connections.insert(name.to_owned());
        }
        let changed_providers: BTreeSet<_> = previous
            .config
            .profiles
            .iter()
            .chain(&profiles)
            .filter(|profile| changes.connections.contains(&profile.profile_name))
            .map(|profile| profile.provider_id.as_str().to_owned())
            .collect();
        let mut registry = (*previous.account_registry).clone();
        registry.invalidate(
            &changes.connections,
            &changed_providers,
            changes.all_bindings,
        );
        let mut cache_generations = previous.cache_generations.clone();
        for name in &changes.profiles {
            // Publication revisions never repeat, including after remove/recreate.
            cache_generations.insert(name.clone(), revision);
        }
        cache_generations
            .retain(|name, _| profiles.iter().any(|profile| &profile.profile_name == name));
        let state = PublishedState {
            revision,
            config: Arc::new(RuntimeSnapshot::new(
                client.runtime.region,
                profiles,
                Some(store.state.tracked_models.clone()),
            )),
            account_registry: Arc::new(registry),
            cache_namespace: if switched_directory {
                previous
                    .cache_namespace
                    .checked_add(1)
                    .expect("cache namespace exhausted")
            } else {
                previous.cache_namespace
            },
            cache_generations,
        };
        Self::install_published(client, state);
    }

    async fn update_configuration<T: Send + 'static>(
        &self,
        forced_binding: Option<String>,
        change: impl FnOnce(&mut config::SavedConfig, &config::Definitions) -> Result<T, ProviderStoreError>
            + Send
            + 'static,
    ) -> Result<T, ProviderStoreError> {
        self.run(move |store, client, cancel| {
            let (result, profiles) = store.update(
                change,
                |profiles| {
                    validate_profiles(
                        profiles,
                        &client.runtime.codecs,
                        &client.runtime.authenticators,
                    )
                    .map_err(Into::into)
                },
                cancel,
            )?;
            Self::publish(client, store, profiles, false, forced_binding.as_deref());
            Ok(result)
        })
        .await
    }

    /// Load v3 configuration without rewriting its file. All client clones see
    /// the new configuration on their next operation after this call succeeds.
    pub async fn set_config_dir(&self, path: impl AsRef<Path>) -> Result<(), ProviderStoreError> {
        let path = path.as_ref().to_owned();
        self.run(move |store, client, cancel| {
            let previous = store
                .repository
                .as_ref()
                .map(|repository| repository.path.clone());
            let profiles = store.load(
                &path,
                |profiles| {
                    validate_profiles(
                        profiles,
                        &client.runtime.codecs,
                        &client.runtime.authenticators,
                    )
                    .map_err(Into::into)
                },
                cancel,
            )?;
            let switched = previous.is_some_and(|path| {
                Some(&path) != store.repository.as_ref().map(|repository| &repository.path)
            });
            Self::publish(client, store, profiles, switched, None);
            Ok(())
        })
        .await
    }

    /// Fully replace an account and its models. Supplied model fields become explicit overrides.
    pub async fn add_provider(&self, profile: ProviderProfile) -> Result<(), ProviderStoreError> {
        let name = profile.profile_name.clone();
        self.update_configuration(Some(name.clone()), move |state, definitions| {
            let saved = SavedProfile::replacement(&profile, definitions.reference(&name));
            if let Some(old) = state
                .providers
                .iter_mut()
                .find(|profile| profile.connection.profile_name == name)
            {
                *old = saved;
            } else {
                state.providers.push(saved);
            }
            state.deleted_profiles.remove(&name);
            Ok(())
        })
        .await
    }

    /// Bind one host-owned account source and publish it with the current connections.
    pub async fn register_profile_account_source(
        &self,
        profile_name: &str,
        identity: account::AccountIdentity,
        source: Arc<dyn account::AccountUsageSource>,
    ) -> Result<(), ProviderStoreError> {
        let profile_name = profile_name.to_owned();
        self.run(move |_store, client, _cancel| {
            let previous = client
                .published
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            if previous.config.profile(&profile_name).is_none() {
                return Err(ProviderStoreError::UnknownProfile(profile_name));
            }
            let mut registry = (*previous.account_registry).clone();
            registry.bind(profile_name, identity, source);
            let state = PublishedState {
                revision: previous
                    .revision
                    .checked_add(1)
                    .expect("configuration revision exhausted"),
                config: previous.config.clone(),
                account_registry: Arc::new(registry),
                cache_namespace: previous.cache_namespace,
                cache_generations: previous.cache_generations.clone(),
            };
            Self::install_published(client, state);
            Ok(())
        })
        .await
    }

    pub async fn deleted_builtin_profiles(&self) -> Result<BTreeSet<String>, ProviderStoreError> {
        self.run(|store, _, _| Ok(store.deleted_builtin_profiles.clone()))
            .await
    }

    pub async fn remove_provider(&self, name: &str) -> Result<(), ProviderStoreError> {
        let name = name.to_owned();
        self.update_configuration(None, move |state, _| {
            config::profile(state, &name)?;
            state
                .providers
                .retain(|profile| profile.connection.profile_name != name);
            state.deleted_profiles.insert(name);
            Ok(())
        })
        .await
    }

    pub async fn restore_builtin(&self, name: &str) -> Result<(), ProviderStoreError> {
        let name = name.to_owned();
        self.update_configuration(Some(name.clone()), move |state, _| {
            let preset = crate::presets::builtin()?
                .into_iter()
                .find(|profile| profile.profile_name == name)
                .ok_or_else(|| ProviderStoreError::NotBuiltin(name.clone()))?;
            let saved =
                SavedProfile::inherited(&preset, DefinitionRef::Builtin { name: name.clone() });
            if let Some(old) = state
                .providers
                .iter_mut()
                .find(|profile| profile.connection.profile_name == name)
            {
                *old = saved;
            } else {
                state.providers.push(saved);
            }
            state.deleted_profiles.remove(&name);
            Ok(())
        })
        .await
    }

    /// Allowlisting controls tracking and presentation, never explicit routing.
    pub async fn set_tracked_models(
        &self,
        provider: &str,
        models: impl IntoIterator<Item = String>,
    ) -> Result<(), ProviderStoreError> {
        let provider = provider.to_owned();
        let models = models.into_iter().collect();
        self.update_configuration(None, move |state, _| {
            state.tracked_models.insert(provider, models);
            Ok(())
        })
        .await
    }

    pub async fn untrack_model(
        &self,
        provider: &str,
        model: &str,
    ) -> Result<(), ProviderStoreError> {
        let provider = provider.to_owned();
        let model = model.to_owned();
        self.update_configuration(None, move |state, _| {
            if !state
                .tracked_models
                .get_mut(&provider)
                .is_some_and(|ids| ids.remove(&model))
            {
                return Err(ProviderStoreError::UntrackedModel {
                    provider_id: provider,
                    model,
                });
            }
            Ok(())
        })
        .await
    }

    pub async fn tracked_models(
        &self,
        provider: &str,
    ) -> Result<Option<BTreeSet<String>>, ProviderStoreError> {
        let provider = provider.to_owned();
        self.run(move |store, _, _| Ok(store.state.tracked_models.get(&provider).cloned()))
            .await
    }

    /// Return ordered rows, including incompatible models and stable row identities.
    pub async fn configured_models(
        &self,
        profile: &str,
    ) -> Result<Vec<ConfiguredModel>, ProviderStoreError> {
        let profile = profile.to_owned();
        self.run(move |store, _, _| store.configured_models(&profile))
            .await
    }

    pub async fn set_model_visibility(
        &self,
        name: &str,
        wire: &str,
        visible: bool,
    ) -> Result<(), ProviderStoreError> {
        let name = name.to_owned();
        let wire = wire.to_owned();
        self.update_configuration(None, move |state, definitions| {
            let profile = config::profile(state, &name)?;
            let rows = profile.rows(definitions)?;
            let mut matches = rows
                .into_iter()
                .filter(|row| row.model.request_model == wire && row.compatible);
            let row = matches
                .next()
                .ok_or_else(|| ProviderStoreError::UnknownModel {
                    profile_name: name.clone(),
                    model: wire.clone(),
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
        .await
    }

    /// Set one row field. `Clear` resets it; `Inherit` removes the override.
    pub async fn set_model_override(
        &self,
        profile_name: &str,
        row_id: &str,
        field: ModelField,
        value: FieldOverride<serde_json::Value>,
    ) -> Result<(), ProviderStoreError> {
        let profile_name = profile_name.to_owned();
        let row_id = row_id.to_owned();
        self.update_configuration(None, move |state, definitions| {
            let profile = config::profile(state, &profile_name)?;
            let current = profile
                .rows(definitions)?
                .into_iter()
                .find(|row| row.row_id == row_id)
                .ok_or_else(|| ProviderStoreError::UnknownModel {
                    profile_name: profile_name.clone(),
                    model: row_id.clone(),
                })?;
            let row = profile.ensure_row(&row_id, definitions)?;
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
        .await
    }

    pub async fn clear_model_override(
        &self,
        profile: &str,
        row: &str,
        field: ModelField,
    ) -> Result<(), ProviderStoreError> {
        self.set_model_override(profile, row, field, FieldOverride::Inherit)
            .await
    }

    /// Replace all fields of one row while preserving its identity and position.
    pub async fn replace_model(
        &self,
        name: &str,
        row_id: &str,
        model: crate::protocol::ModelProfile,
    ) -> Result<(), ProviderStoreError> {
        let name = name.to_owned();
        let row_id = row_id.to_owned();
        self.update_configuration(None, move |state, definitions| {
            let profile = config::profile(state, &name)?;
            if !profile
                .rows(definitions)?
                .iter()
                .any(|row| row.row_id == row_id)
            {
                return Err(ProviderStoreError::UnknownModel {
                    profile_name: name.clone(),
                    model: row_id.clone(),
                });
            }
            let key = profile
                .ensure_row(&row_id, definitions)?
                .definition_key
                .clone();
            if let Some(key) = key {
                profile.removed_defaults.insert(key);
            }
            let row = profile.ensure_row(&row_id, definitions)?;
            row.replacement = true;
            row.initial = Some(config::empty_model(&model.request_model));
            row.overrides = config::values(&model);
            Ok(())
        })
        .await
    }

    /// Capture an owned fetch operation. Its network work holds no manager lock.
    pub async fn prepare_provider_sync(
        &self,
        name: &str,
        credential: Option<&Secret<String>>,
    ) -> Result<ProviderSyncOperation, ProviderStoreError> {
        let name = name.to_owned();
        let credential = credential.cloned();
        self.run(move |store, client, _| {
            let repository = store
                .repository
                .clone()
                .ok_or(ProviderStoreError::NotConfigured)?;
            let published = client
                .published
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            let source = published
                .config
                .profile(&name)
                .cloned()
                .ok_or_else(|| ProviderStoreError::UnknownProfile(name.clone()))?;
            let directory = source
                .model_list
                .shape(source.protocol)
                .and_then(|shape| client.runtime.directories.get(&shape))
                .cloned()
                .ok_or_else(|| ProviderStoreError::NoDirectory(name.clone()))?;
            let authenticator = if source.auth == AuthStrategy::None {
                None
            } else {
                Some(
                    client
                        .runtime
                        .authenticators
                        .get(&source.auth)
                        .cloned()
                        .ok_or_else(|| super::BuildError::MissingAuthenticator {
                            profile_name: name.clone(),
                            strategy: source.auth,
                        })?,
                )
            };
            Ok(ProviderSyncOperation {
                repository,
                generation: store.generation,
                source,
                definitions: store.definitions.clone(),
                directory,
                authenticator,
                http: client.runtime.http.clone(),
                credential,
            })
        })
        .await
    }

    /// Commit fetched observations and publish before the worker releases the writer lock.
    pub async fn apply_provider_sync(
        &self,
        result: ProviderSyncResult,
    ) -> Result<usize, ProviderStoreError> {
        self.run(move |store, client, cancel| {
            let name = result.source.profile_name.clone();
            let published = client
                .published
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            if store
                .repository
                .as_ref()
                .is_none_or(|repository| repository.path != result.repository.path)
                || store.generation != result.generation
                || published
                    .config
                    .profile(&name)
                    .is_none_or(|profile| !config::same_connection(profile, &result.source))
            {
                return Err(ProviderStoreError::ProfileChanged(name));
            }
            let (state, profiles, count) =
                result.commit(&store.definitions, &store.state, cancel, |profiles| {
                    validate_profiles(
                        profiles,
                        &client.runtime.codecs,
                        &client.runtime.authenticators,
                    )
                    .map_err(Into::into)
                })?;
            store.install(state);
            Self::publish(client, store, profiles, false, None);
            Ok(count)
        })
        .await
    }

    pub async fn sync_provider(
        &self,
        name: &str,
        credential: Option<&Secret<String>>,
    ) -> Result<usize, ProviderStoreError> {
        let operation = self.prepare_provider_sync(name, credential).await?;
        self.apply_provider_sync(operation.fetch().await?).await
    }
}

#[cfg(test)]
mod publication_tests {
    use super::*;
    use crate::{protocol::Region, LlmClientBuilder};
    use serde_json::json;
    use std::time::Duration;

    struct Directory(std::path::PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[tokio::test]
    async fn cancelling_after_durable_commit_still_publishes_before_the_next_transaction() {
        let directory = Directory(std::env::temp_dir().join(format!(
            "llm-client-commit-publication-{}",
            std::process::id()
        )));
        let _ = std::fs::remove_dir_all(&directory.0);
        std::fs::create_dir_all(&directory.0).unwrap();
        let profile: ProviderProfile = serde_json::from_value(json!({
            "provider_id": "acme", "profile_name": "p", "protocol": "open_ai_chat",
            "auth": "none", "base_url": "https://example.test/v1",
            "models": [{"request_model": "m", "display_model": "m", "billing_model": "m"}]
        }))
        .unwrap();
        let (client, manager) = LlmClientBuilder::new(&[profile])
            .unwrap()
            .with_region(Region::International)
            .build_managed()
            .unwrap();
        let manager = Arc::new(manager);
        manager.set_config_dir(&directory.0).await.unwrap();
        manager
            .set_tracked_models("acme", ["m".into()])
            .await
            .unwrap();
        let previous = client.snapshot();
        let prior_bytes = std::fs::read(directory.0.join("providers.json")).unwrap();
        let (committed, notification) = tokio::sync::oneshot::channel();
        let (release, released) = std::sync::mpsc::channel();
        manager.store.lock().unwrap().before_publication = Some(Box::new(move || {
            committed.send(()).unwrap();
            released.recv_timeout(Duration::from_secs(5)).unwrap();
        }));
        let writing = manager.clone();
        let task = tokio::spawn(async move { writing.set_model_visibility("p", "m", false).await });
        tokio::time::timeout(Duration::from_secs(5), notification)
            .await
            .unwrap()
            .unwrap();
        assert_ne!(
            std::fs::read(directory.0.join("providers.json")).unwrap(),
            prior_bytes
        );
        assert_eq!(client.snapshot().revision(), previous.revision());
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        release.send(()).unwrap();
        // This read acquires the same writer lock, so observing its result also
        // establishes that the cancelled worker finished installing its commit.
        let rows = tokio::time::timeout(Duration::from_secs(5), manager.configured_models("p"))
            .await
            .unwrap()
            .unwrap();
        assert!(rows[0].model.hidden);
        assert!(client.snapshot().profile("p").unwrap().models[0].hidden);
        assert!(client.snapshot().revision() > previous.revision());
        assert!(!previous.profile("p").unwrap().models[0].hidden);
    }
}
