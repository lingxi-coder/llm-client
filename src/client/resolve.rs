//! `resolve` / `resolve_in`: a model id becomes a route and its failover
//! chain. Ported from the previous project's `registry.rs::resolve_in`.

use super::route::{ConnectionHop, PricingModelRef, ResolvedRoute};
use super::LlmClient;
use lingxi_agent_api::protocol::{LlmError, ModelProfile, ProviderProfile};
use thiserror::Error;

/// Why `resolve` could not produce a route. Each names what a user has to
/// change, which is why they are distinct variants rather than one string.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResolveError {
    #[error("no available profile declares model {model:?}")]
    UnknownModel { model: String },
    #[error(
        "model reference {model:?} is ambiguous across profiles: {} — qualify it, e.g. {}/{model}",
        groups.join(", "), groups.first().map(String::as_str).unwrap_or("<group>")
    )]
    AmbiguousAcrossGroups { model: String, groups: Vec<String> },
    #[error(
        "ambiguous model reference {model:?} matches a native model id on profiles {native_profiles:?} \
         and a qualified profile reference on profiles {qualified_profiles:?}; specify a profile scope"
    )]
    AmbiguousNativeAndQualified {
        model: String,
        native_profiles: Vec<String>,
        qualified_profiles: Vec<String>,
    },
    #[error(
        "model reference {model:?} matches more than one model on profile {profile_name:?} \
         — rename or remove the duplicate alias"
    )]
    DuplicateOnProfile { model: String, profile_name: String },
}

impl From<ResolveError> for LlmError {
    fn from(e: ResolveError) -> Self {
        LlmError::ModelUnavailable {
            message: e.to_string(),
        }
    }
}

impl LlmClient {
    /// Which connection serves `model`, under what wire name, and which
    /// siblings may take over. Ported from the previous project's
    /// `registry.rs::resolve_in`.
    ///
    /// `profile = Some(p)` scopes to connections whose profile name **or group**
    /// is `p`; `None` matches across all providers. Scoping selects the
    /// connection to start on — it does not decide whether the rest of the
    /// group may be used.
    pub fn resolve(&self, model: &str) -> Result<ResolvedRoute, ResolveError> {
        self.resolve_in(model, None)
    }

    pub fn resolve_in(
        &self,
        requested: &str,
        profile: Option<&str>,
    ) -> Result<ResolvedRoute, ResolveError> {
        // A qualifier naming one connection means that connection, even when a
        // group shares the name. One preset is its own group's namesake and has
        // a sibling that serves a different wire model under the same display
        // name, so falling straight to the group would answer a connection-
        // qualified ref with the other connection's model. Failover still
        // reaches the rest of the group — scoping picks where to start.
        let names_a_connection = profile.is_some_and(|scoped| {
            self.profiles
                .iter()
                .any(|p| p.supports_region(self.region) && p.profile_name == scoped)
        });
        let in_scope = |p: &ProviderProfile| match profile {
            // A group name scopes to every connection in that group, so a
            // session that stored the group-qualified ref a picker showed still
            // routes.
            Some(scoped) if names_a_connection => p.profile_name == scoped,
            Some(scoped) => p.group() == scoped,
            None => true,
        };

        let mut matches: Vec<(&ProviderProfile, &ModelProfile)> = Vec::new();
        for p in self
            .profiles
            .iter()
            .filter(|p| p.supports_region(self.region) && in_scope(p))
        {
            for m in &p.models {
                if m.answers_to(requested) {
                    matches.push((p, m));
                }
            }
        }

        // Structured clients expose `profile/model` refs so two providers
        // serving the same model stay distinguishable, but a provider's protocol
        // accepts only its native model. A slash-bearing string can also be a
        // provider's native wire id, so collect both interpretations before
        // selecting one. If they identify different route heads, an unscoped
        // call cannot safely guess which credential and endpoint the caller
        // intended.
        let mut qualified_matches: Vec<(&ProviderProfile, &ModelProfile)> = Vec::new();
        if let Some((qualifier, bare)) = requested.split_once('/') {
            if profile.is_none_or(|scoped| scoped == qualifier) {
                let names_a_connection = self
                    .profiles
                    .iter()
                    .any(|p| p.supports_region(self.region) && p.profile_name == qualifier);
                for p in self
                    .profiles
                    .iter()
                    .filter(|p| p.supports_region(self.region))
                {
                    if if names_a_connection {
                        p.profile_name != qualifier
                    } else {
                        p.group() != qualifier
                    } {
                        continue;
                    }
                    for m in &p.models {
                        if m.answers_to(bare) {
                            qualified_matches.push((p, m));
                        }
                    }
                }
            }
        }

        if profile.is_none() && !matches.is_empty() && !qualified_matches.is_empty() {
            let mut native_route = matches.clone();
            let mut qualified_route = qualified_matches.clone();
            sort_matches(&mut native_route, true);
            sort_matches(&mut qualified_route, true);
            let same_target = match (native_route.first(), qualified_route.first()) {
                (
                    Some((native_provider, native_model)),
                    Some((qualified_provider, qualified_model)),
                ) => {
                    std::ptr::eq(*native_provider, *qualified_provider)
                        && std::ptr::eq(*native_model, *qualified_model)
                }
                _ => false,
            };
            if !same_target {
                let profile_names = |candidates: &[(&ProviderProfile, &ModelProfile)]| {
                    let mut names: Vec<_> = candidates
                        .iter()
                        .map(|(provider, _)| provider.profile_name.clone())
                        .collect();
                    names.sort();
                    names.dedup();
                    names
                };
                return Err(ResolveError::AmbiguousNativeAndQualified {
                    model: requested.to_owned(),
                    native_profiles: profile_names(&matches),
                    qualified_profiles: profile_names(&qualified_matches),
                });
            }
        }

        if matches.is_empty() {
            matches = qualified_matches;
        }

        if matches.is_empty() {
            return Err(ResolveError::UnknownModel {
                model: requested.to_owned(),
            });
        }

        // More than one group in play is a real ambiguity; within one group the
        // extra matches are the failover chain.
        let mut groups: Vec<&str> = matches.iter().map(|(p, _)| p.group()).collect();
        groups.sort_unstable();
        groups.dedup();
        if groups.len() > 1 {
            return Err(ResolveError::AmbiguousAcrossGroups {
                model: requested.to_owned(),
                groups: groups.iter().map(|g| (*g).to_owned()).collect(),
            });
        }

        // Two models of ONE profile answering to the same string is a genuine
        // ambiguity: they are different models on the same endpoint, so which to
        // send is unknown.
        if let Some((dup, _)) = matches.iter().find(|(p, _)| {
            matches
                .iter()
                .filter(|(q, _)| q.profile_name == p.profile_name)
                .count()
                > 1
        }) {
            return Err(ResolveError::DuplicateOnProfile {
                model: requested.to_owned(),
                profile_name: dup.profile_name.clone(),
            });
        }

        // A bare model reference starts on a connection the picker can show.
        // An explicitly scoped profile remains addressable even when hidden.
        sort_matches(&mut matches, profile.is_none());
        let (provider, model) = matches[0];

        // Siblings come from the head's GROUP, not from whatever was in scope.
        // A picker hands back a connection profile, so a session stores
        // `deepseek:cn` and every real request arrives scoped to one connection;
        // building the chain from the in-scope matches left it empty exactly
        // when failover was needed.
        //
        // A hop must serve the SAME wire model — failing over only re-points the
        // endpoint and the credential, so a hop that changed the model would
        // silently answer as something else.
        //
        // Only connections billed the same way are offered. A provider can
        // publish one model on a subscription endpoint and a metered one, and
        // moving a rate-limited subscription request onto the metered endpoint
        // would start charging real money for what the plan already covers.
        //
        // Compared per model, not per connection: an aggregator serves free and
        // metered models over one endpoint, and a free model that fell over
        // onto a metered one would start charging just the same.
        let billing = model.billing_mode_on(&provider.pricing);
        let group = provider.group();
        let mut siblings: Vec<(&ProviderProfile, &ModelProfile)> = self
            .profiles
            .iter()
            .filter(|c| {
                c.supports_region(self.region)
                    && c.group() == group
                    && c.profile_name != provider.profile_name
            })
            .filter_map(|c| {
                c.models
                    .iter()
                    .find(|m| m.request_model == model.request_model)
                    .map(|m| (c, m))
            })
            .filter(|(c, m)| m.billing_mode_on(&c.pricing) == billing)
            .collect();
        siblings.sort_by(|(a, _), (b, _)| a.connection_sort_key().cmp(&b.connection_sort_key()));

        Ok(ResolvedRoute {
            provider_id: provider.provider_id.clone(),
            profile_name: provider.profile_name.clone(),
            request_model: model.request_model.clone(),
            display_model: model.display_model.clone(),
            pricing_model: PricingModelRef {
                pricing_provider_id: provider.provider_id.clone(),
                billing_model: model.billing_model.clone(),
                request_model: model.request_model.clone(),
                display_model: model.display_model.clone(),
            },
            capabilities: model.capabilities,
            connection_chain: siblings
                .into_iter()
                .map(|(p, m)| ConnectionHop {
                    profile_name: p.profile_name.clone(),
                    request_model: m.request_model.clone(),
                })
                .collect(),
            failover: provider.connection.failover,
        })
    }
}

fn sort_matches(matches: &mut [(&ProviderProfile, &ModelProfile)], prefer_visible: bool) {
    matches.sort_by(|(a, _), (b, _)| {
        let visibility = if prefer_visible {
            a.connection.hidden.cmp(&b.connection.hidden)
        } else {
            std::cmp::Ordering::Equal
        };
        visibility.then_with(|| a.connection_sort_key().cmp(&b.connection_sort_key()))
    });
}
