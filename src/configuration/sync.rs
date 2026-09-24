//! Owned account directory fetches and short, locked commits.
use super::{merge::*, model::*, repository::Repository, ProviderStoreError, MAX_PAGES};
use crate::{
    auth::Authenticator,
    directory::{LiveModel, ModelDirectory},
    protocol::{AuthStrategy, ProviderProfile, Secret},
    transport::{HttpExecutor, Transport},
};
use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// An owned directory operation. Fetching never borrows the client.
pub struct ProviderSyncOperation {
    pub(crate) repository: Repository,
    pub(crate) generation: u64,
    pub(crate) source: ProviderProfile,
    pub(crate) definitions: Definitions,
    pub(crate) directory: Arc<dyn ModelDirectory>,
    pub(crate) authenticator: Option<Arc<dyn Authenticator>>,
    pub(crate) http: Arc<dyn Transport>,
    pub(crate) credential: Option<Secret<String>>,
}
/// Fetched observations tied to the connection and directory generation.
pub struct ProviderSyncResult {
    pub(crate) repository: Repository,
    pub(crate) generation: u64,
    pub(crate) source: ProviderProfile,
    live: Vec<LiveModel>,
    incompatible: Vec<String>,
    compatible: Vec<String>,
}
impl ProviderSyncOperation {
    pub async fn fetch(self) -> Result<ProviderSyncResult, ProviderStoreError> {
        let repo = self.repository.clone();
        let definitions = self.definitions.clone();
        let source = self.source.clone();
        tokio::task::spawn_blocking(move || {
            let state = repo.read_locked()?;
            let profiles = state.profiles(&definitions, false)?;
            let current = profiles
                .iter()
                .find(|p| p.profile_name == source.profile_name)
                .ok_or_else(|| ProviderStoreError::UnknownProfile(source.profile_name.clone()))?;
            if !same_connection(current, &source) {
                return Err(ProviderStoreError::ProfileChanged(source.profile_name));
            }
            Ok(())
        })
        .await
        .map_err(|e| ProviderStoreError::Worker(e.to_string()))??;
        let mut cursor = None;
        let mut seen = BTreeSet::new();
        let mut live = Vec::new();
        let mut incompatible = Vec::new();
        let mut compatible = Vec::new();
        for _ in 0..MAX_PAGES {
            let mut request = self.directory.list_request(&self.source, cursor.as_deref());
            let deadline = crate::runtime::Deadline::after(request.timeout);
            if self.source.auth != AuthStrategy::None {
                let auth = self.authenticator.as_ref().ok_or_else(|| {
                    ProviderStoreError::Build(crate::client::BuildError::MissingAuthenticator {
                        profile_name: self.source.profile_name.clone(),
                        strategy: self.source.auth,
                    })
                })?;
                deadline
                    .run(auth.apply(&mut request, &self.source, self.credential.as_ref()))
                    .await??;
            }
            request.timeout = deadline.remaining()?;
            let response = HttpExecutor::new(self.http.as_ref())
                .with_deadline(deadline)
                .execute(request)
                .await?;
            let decoded = self.directory.decode_page_with_exclusions(&response)?;
            live.extend(decoded.page.models);
            incompatible.extend(decoded.incompatible_model_ids);
            compatible.extend(decoded.explicitly_compatible_model_ids);
            match decoded.page.next_cursor {
                None => {
                    return Ok(ProviderSyncResult {
                        repository: self.repository,
                        generation: self.generation,
                        source: self.source,
                        live,
                        incompatible,
                        compatible,
                    })
                }
                Some(next) if seen.insert(next.clone()) => cursor = Some(next),
                Some(_) => return Err(ProviderStoreError::InvalidPagination),
            }
        }
        Err(ProviderStoreError::InvalidPagination)
    }
}
pub(crate) struct CancelCommit(pub Arc<AtomicBool>);
impl Drop for CancelCommit {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
impl ProviderSyncResult {
    pub(crate) fn commit(
        self,
        definitions: &Definitions,
        previous: &SavedConfig,
        cancel: &AtomicBool,
        validate: impl FnOnce(&[ProviderProfile]) -> Result<(), ProviderStoreError>,
    ) -> Result<(SavedConfig, Vec<ProviderProfile>, usize), ProviderStoreError> {
        let _lock = self.repository.lock()?;
        let mut state = self.repository.read()?;
        state.reconcile_session(previous);
        state.ensure_definitions(definitions);
        let tracked = state
            .tracked_models
            .get(self.source.provider_id.as_str())
            .cloned()
            .unwrap_or_default();
        let current = state
            .profiles(definitions, false)?
            .into_iter()
            .find(|p| p.profile_name == self.source.profile_name)
            .ok_or_else(|| ProviderStoreError::UnknownProfile(self.source.profile_name.clone()))?;
        if !same_connection(&current, &self.source) {
            return Err(ProviderStoreError::ProfileChanged(self.source.profile_name));
        }
        let profile = super::profile(&mut state, &self.source.profile_name)?;
        profile.incompatible_models.extend(self.incompatible);
        profile
            .incompatible_models
            .retain(|id| !self.compatible.contains(id));
        let mut count = 0;
        for live in self.live {
            if !tracked.contains(&live.request_model)
                || profile.incompatible_models.contains(&live.request_model)
            {
                continue;
            }
            let exists = profile
                .rows(definitions)?
                .iter()
                .any(|r| r.model.request_model == live.request_model);
            if !exists {
                let mut restored = false;
                for row in profile.models.iter_mut().filter(|row| {
                    !row.replacement
                        && row
                            .definition_key
                            .as_ref()
                            .is_some_and(|key| key.wire == live.request_model)
                }) {
                    row.observed = true;
                    row.initial = Some(empty_model(&live.request_model));
                    restored = true;
                }
                if !restored {
                    let mut n = 0;
                    let id = loop {
                        let id = format!("observed:{}:{n}", live.request_model);
                        if !profile.models.iter().any(|r| r.id == id) {
                            break id;
                        }
                        n += 1;
                    };
                    profile.models.push(ModelRow {
                        id,
                        definition_key: None,
                        initial: Some(empty_model(&live.request_model)),
                        replacement: false,
                        observed: true,
                        overrides: Values::new(),
                    });
                }
            }
            let observation = profile.observations.entry(live.request_model).or_default();
            if let Some(description) = live
                .description
                .or_else(|| (!exists).then_some(live.display_name).flatten())
            {
                observation.description = Some(description);
            }
            if let Some(context) = live.context_window {
                observation.context_window = Some(context);
            }
            if let Some(features) = live.inference_features {
                observation
                    .inference_features
                    .get_or_insert_with(Default::default)
                    .overlay(&features);
            }
            if let Some(output) = live.max_output_tokens {
                observation.max_output_tokens = Some(output);
            }
            count += 1;
        }
        state.refresh_fallbacks(definitions);
        let profiles = state.profiles(definitions, true)?;
        validate(&profiles)?;
        if cancel.load(Ordering::Acquire) {
            return Err(ProviderStoreError::Worker(
                "provider sync was cancelled before commit".into(),
            ));
        }
        self.repository.write(&state)?;
        Ok((state, profiles, count))
    }
}
