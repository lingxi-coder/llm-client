//! Account source bindings and query scheduling.
use super::*;
use crate::transport::Clock;
use futures::{stream::iter, StreamExt};
use std::sync::Arc;
use std::{collections::BTreeSet, num::NonZeroUsize};
type Sources = BTreeMap<(String, AccountIdentity), Arc<dyn AccountUsageSource>>;
#[derive(Default)]
struct Registry {
    providers: Sources,
    profiles: Sources,
    invalidated: BTreeSet<(String, AccountIdentity)>,
}
pub(crate) struct Service {
    registry: Registry,
    http: Arc<dyn Transport>,
    clock: Arc<dyn Clock>,
    concurrency: NonZeroUsize,
}
impl Service {
    pub fn new(
        providers: Sources,
        profiles: Sources,
        http: Arc<dyn Transport>,
        clock: Arc<dyn Clock>,
        concurrency: NonZeroUsize,
    ) -> Self {
        Self {
            registry: Registry {
                providers,
                profiles,
                ..Default::default()
            },
            http,
            clock,
            concurrency,
        }
    }
    pub fn bind(
        &mut self,
        name: String,
        identity: AccountIdentity,
        source: Arc<dyn AccountUsageSource>,
    ) {
        self.registry.profiles.insert((name, identity), source);
    }
    pub fn invalidate(
        &mut self,
        names: &BTreeSet<String>,
        providers: &BTreeSet<String>,
        all: bool,
    ) {
        self.registry
            .profiles
            .retain(|(name, _), _| !all && !names.contains(name));
        self.registry.invalidated.extend(
            self.registry
                .providers
                .keys()
                .filter(|(provider, _)| all || providers.contains(provider))
                .cloned(),
        );
    }
    /// Read an account's available balance, token history, quota windows and
    /// subscription evidence without sending a model request.
    pub async fn query(
        &self,
        profiles: &[ProviderProfile],
        profile_name: &str,
        query: &AccountQuery,
    ) -> Result<AccountSnapshot, AccountUsageError> {
        let profile = profiles
            .iter()
            .find(|p| p.profile_name == profile_name)
            .ok_or_else(|| AccountUsageError::UnknownProfile(profile_name.to_owned()))?;
        let now = self
            .clock
            .now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let range = query.range(now)?;
        let profile_key = (profile.profile_name.clone(), query.identity);
        let provider_key = (profile.provider_id.as_str().to_owned(), query.identity);
        let profile_source = self.registry.profiles.get(&profile_key);
        let source = profile_source.or_else(|| self.registry.providers.get(&provider_key));
        if profile_source.is_none()
            && source.is_some_and(|source| source.requires_profile_binding(query))
            && (self.registry.invalidated.contains(&provider_key)
                || profiles
                    .iter()
                    .filter(|candidate| candidate.provider_id == profile.provider_id)
                    .count()
                    > 1)
        {
            return Err(AccountUsageError::AmbiguousAccountSource(
                profile_name.to_owned(),
            ));
        }
        if query.execution.total_timeout.is_zero() || query.execution.operation_timeout.is_zero() {
            return Err(AccountUsageError::InvalidExecutionOptions);
        }
        let Some(source) = source else {
            return Ok(AccountSnapshot::unsupported(profile, query.identity, now));
        };
        let context = AccountFetchContext::new(profile, query, range, now, self.http.clone());
        Ok(context.collect(source.as_ref()).await)
    }

    /// Query every configured connection independently, including hidden ones.
    /// Missing query entries and individual provider failures stay local to
    /// their profile's result.
    pub async fn query_all(
        &self,
        profiles: &[ProviderProfile],
        queries_by_profile: &BTreeMap<String, AccountQuery>,
    ) -> Vec<(String, Result<AccountSnapshot, AccountUsageError>)> {
        let mut results = iter(
            profiles
                .iter()
                .enumerate()
                .map(|(index, profile)| async move {
                    let name = profile.profile_name.clone();
                    let result = match queries_by_profile.get(&name) {
                        Some(query) => self.query(profiles, &name, query).await,
                        None => Err(AccountUsageError::MissingQuery(name.clone())),
                    };
                    (index, name, result)
                }),
        )
        .buffer_unordered(self.concurrency.get())
        .collect::<Vec<_>>()
        .await;
        results.sort_by_key(|(index, _, _)| *index);
        results
            .into_iter()
            .map(|(_, name, result)| (name, result))
            .collect()
    }
}
