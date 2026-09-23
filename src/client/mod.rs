//! `LlmClientBuilder` / `LlmClient`: the provider-neutral machine.
//!
//! `resolve` turns a model id into a route and its failover chain; `complete`
//! and `stream` walk that chain (gate 32). Nothing here knows a provider by
//! name — a new OpenAI-compatible provider is a `ProviderProfile` in settings
//! and no code change (gate 30). The builder refuses to build a client whose
//! profile names a protocol with no codec (gate 33).

mod failover;
pub mod options;
pub mod pricing;
mod resolve;
pub mod route;
mod stream;
pub(crate) mod usage;

pub use options::RequestOptions;
pub use resolve::ResolveError;
pub use stream::ModelStream;

use crate::auth::Authenticator;
use crate::codecs::WireCodec;
use crate::directory::ModelDirectory;
use crate::transport::{Clock, HttpTransport, SystemClock, Transport};
use lingxi_agent_api::protocol::{
    AuthStrategy, CredentialConfig, LlmError, ModelListing, ProtocolFamily, ProviderListing,
    ProviderProfile, Submission, Usage,
};
use route::ResolvedRoute;
use std::collections::BTreeMap;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BuildError {
    #[error("provider profile {profile_name:?} uses protocol {family:?} but no codec for it is registered")]
    MissingCodec {
        profile_name: String,
        family: ProtocolFamily,
    },
    #[error("provider profile {profile_name:?} uses auth {strategy:?} but no authenticator for it is registered")]
    MissingAuthenticator {
        profile_name: String,
        strategy: AuthStrategy,
    },
    #[error("provider profile name {profile_name:?} is declared twice")]
    DuplicateProfile { profile_name: String },
    #[error("provider profile {profile_name:?} has an invalid peak price schedule: {reason}")]
    InvalidPeakSchedule {
        profile_name: String,
        reason: String,
    },
}

pub struct LlmClientBuilder {
    http: Arc<dyn Transport>,
    clock: Arc<dyn Clock>,
    codecs: BTreeMap<ProtocolFamily, Arc<dyn WireCodec>>,
    directories: BTreeMap<ProtocolFamily, Arc<dyn ModelDirectory>>,
    authenticators: BTreeMap<AuthStrategy, Arc<dyn Authenticator>>,
    profiles: Vec<ProviderProfile>,
}

impl LlmClientBuilder {
    /// Use the built-in HTTP/HTTPS transport and system clock, with API-key
    /// and bearer authentication registered. No custom services are needed.
    /// Requests require a Tokio runtime with I/O and time enabled.
    pub fn new(profiles: &[ProviderProfile]) -> Result<Self, LlmError> {
        Ok(Self::with_transport(
            Arc::new(HttpTransport::new()?),
            profiles,
        ))
    }

    /// Use a custom transport with the system clock. Registers all built-in
    /// codecs, model directories, API-key and bearer authenticators, just like
    /// [`Self::new`]. This constructor does not create a network client.
    pub fn with_transport(http: Arc<dyn Transport>, profiles: &[ProviderProfile]) -> Self {
        let mut codecs: BTreeMap<ProtocolFamily, Arc<dyn WireCodec>> = BTreeMap::new();
        for codec in crate::codecs::builtin() {
            codecs.insert(codec.family(), codec);
        }
        let mut directories: BTreeMap<ProtocolFamily, Arc<dyn ModelDirectory>> = BTreeMap::new();
        for directory in crate::directory::builtin() {
            directories.insert(directory.shape(), directory);
        }
        let mut builder = Self {
            http,
            clock: Arc::new(SystemClock),
            codecs,
            directories,
            authenticators: BTreeMap::new(),
            profiles: profiles.to_vec(),
        };
        builder.register_authenticator(AuthStrategy::ApiKey, Arc::new(crate::ApiKeyAuthenticator));
        builder.register_authenticator(AuthStrategy::Bearer, Arc::new(crate::BearerAuthenticator));
        builder
    }

    /// Replace the default system clock, for example to test price schedules.
    pub fn with_clock(&mut self, clock: Arc<dyn Clock>) -> &mut Self {
        self.clock = clock;
        self
    }

    /// Later registrations for the same family replace earlier ones; the
    /// composition root decides the order.
    pub fn register_codec(&mut self, codec: Arc<dyn WireCodec>) -> &mut Self {
        self.codecs.insert(codec.family(), codec);
        self
    }

    /// Replace or add the directory for a shape. Unlike a codec, a shape with
    /// none is allowed — see `ModelDirectory` for why the asymmetry is
    /// deliberate.
    pub fn register_directory(&mut self, directory: Arc<dyn ModelDirectory>) -> &mut Self {
        self.directories.insert(directory.shape(), directory);
        self
    }

    pub fn register_authenticator(
        &mut self,
        strategy: AuthStrategy,
        auth: Arc<dyn Authenticator>,
    ) -> &mut Self {
        self.authenticators.insert(strategy, auth);
        self
    }

    pub fn add_profile(&mut self, profile: ProviderProfile) -> &mut Self {
        self.profiles.push(profile);
        self
    }

    pub fn codec_families(&self) -> Vec<ProtocolFamily> {
        self.codecs.keys().copied().collect()
    }

    /// Every profile's protocol must have a codec and its auth strategy an
    /// authenticator (`AuthStrategy::None` needs none). Fails before the first
    /// request, naming the profile (gate 33).
    pub fn build(self) -> Result<LlmClient, BuildError> {
        let mut seen = std::collections::BTreeSet::new();
        for p in &self.profiles {
            if !seen.insert(p.profile_name.clone()) {
                return Err(BuildError::DuplicateProfile {
                    profile_name: p.profile_name.clone(),
                });
            }
        }
        for p in &self.profiles {
            if let Some(peak) = &p.pricing.peak {
                peak.validate()
                    .map_err(|reason| BuildError::InvalidPeakSchedule {
                        profile_name: p.profile_name.clone(),
                        reason,
                    })?;
            }
            if !self.codecs.contains_key(&p.protocol) {
                return Err(BuildError::MissingCodec {
                    profile_name: p.profile_name.clone(),
                    family: p.protocol,
                });
            }
            if p.auth != AuthStrategy::None && !self.authenticators.contains_key(&p.auth) {
                return Err(BuildError::MissingAuthenticator {
                    profile_name: p.profile_name.clone(),
                    strategy: p.auth,
                });
            }
        }
        // A profile whose `model_list` names a shape with no directory is
        // deliberately NOT an error here. It can run every turn it could run
        // before; all it cannot do is refresh its own list.
        Ok(LlmClient {
            http: self.http,
            clock: self.clock,
            codecs: self.codecs,
            directories: self.directories,
            authenticators: self.authenticators,
            profiles: self.profiles,
        })
    }
}

/// The provider-neutral client. Holds every registered codec and profile;
/// routing and requests are M1.
pub struct LlmClient {
    http: Arc<dyn Transport>,
    clock: Arc<dyn Clock>,
    codecs: BTreeMap<ProtocolFamily, Arc<dyn WireCodec>>,
    directories: BTreeMap<ProtocolFamily, Arc<dyn ModelDirectory>>,
    authenticators: BTreeMap<AuthStrategy, Arc<dyn Authenticator>>,
    profiles: Vec<ProviderProfile>,
}

impl LlmClient {
    /// Every model a picker may offer. A hidden connection is skipped: those
    /// exist only to be failed over onto, so offering them would let a user
    /// pick the spare key directly.
    ///
    /// `id` is the display model, which is the ref a session stores and
    /// `resolve` accepts back.
    pub fn models(&self) -> Vec<ModelListing> {
        self.profiles
            .iter()
            .filter(|p| !p.connection.hidden)
            .flat_map(|p| {
                p.models.iter().map(move |m| ModelListing {
                    id: m.display_model.clone(),
                    profile_name: p.profile_name.clone(),
                    request_model: m.request_model.clone(),
                    billing_mode: m.billing_mode_on(&p.pricing),
                    pricing: m.pricing.clone(),
                    // The catalog's display name, not its description: the
                    // latter is a paragraph of vendor prose and was never a
                    // name. It keeps its own field below.
                    display_name: m.display_model.clone(),
                    description: m.description.clone(),
                    provider_id: p.provider_id.clone(),
                    context_window: m.metadata.context_window_tokens,
                    max_output_tokens: m.metadata.max_output_tokens,
                    capabilities: m.capabilities,
                })
            })
            .collect()
    }

    /// Every configured provider, including the ones with no credential and the
    /// spare connections a picker should not offer.
    ///
    /// Unfiltered on purpose, which is the opposite of `models()`: an app needs
    /// the unconfigured ones to offer "set this up", and it needs to see the
    /// spares to explain a group. `hidden` says which is which.
    pub fn providers(&self) -> Vec<ProviderListing> {
        self.profiles
            .iter()
            .map(|p| ProviderListing {
                provider_id: p.provider_id.clone(),
                profile_name: p.profile_name.clone(),
                group: p.group().to_owned(),
                info: p.info.clone(),
                protocol: p.protocol,
                auth: p.auth,
                credential_env: match &p.credential {
                    CredentialConfig::Env { var } => Some(var.clone()),
                    _ => None,
                },
                billing_mode: p.pricing.billing_mode,
                model_count: p.models.len(),
                hidden: p.connection.hidden,
            })
            .collect()
    }

    pub fn profiles(&self) -> &[ProviderProfile] {
        &self.profiles
    }

    pub fn codec_families(&self) -> Vec<ProtocolFamily> {
        self.codecs.keys().copied().collect()
    }

    /// How to ask this connection what it serves, when it publishes that at
    /// all and this crate can read the shape it publishes it in.
    ///
    /// `None` is an ordinary answer, not a failure: a connection that declares
    /// no directory, and one whose directory speaks a shape with no reader,
    /// both keep working from the shipped catalog.
    #[must_use]
    pub fn directory_for(&self, profile: &ProviderProfile) -> Option<Arc<dyn ModelDirectory>> {
        let shape = profile.model_list.shape(profile.protocol)?;
        self.directories.get(&shape).cloned()
    }

    pub fn directory_shapes(&self) -> Vec<ProtocolFamily> {
        self.directories.keys().copied().collect()
    }

    pub(super) fn profile(&self, name: &str) -> Option<&ProviderProfile> {
        self.profiles.iter().find(|p| p.profile_name == name)
    }

    /// Preflight estimate for the route's first connection. For a completed
    /// request that may have failed over, use [`Self::estimate_actual_cost`].
    /// What a finished request on `route` cost, priced from the catalog at
    /// the rates in force now. `Ok(None)` is a model the catalog does not
    /// price on a connection that tolerates that; a connection that set
    /// `require_priced` gets the error instead.
    pub fn estimate_cost(
        &self,
        route: &ResolvedRoute,
        usage: &Usage,
        submission: Submission,
    ) -> Result<Option<pricing::CostEstimate>, LlmError> {
        let unpriced = || LlmError::CostUnavailable {
            message: format!(
                "{:?} publishes no price for {:?}",
                route.profile_name, route.display_model
            ),
        };
        let profile = self.profile(&route.profile_name).ok_or_else(unpriced)?;
        let model = profile
            .models
            .iter()
            .find(|m| m.display_model == route.display_model)
            .ok_or_else(unpriced)?;
        let Some(pricing) = &model.pricing else {
            return if profile.pricing.require_priced {
                Err(unpriced())
            } else {
                Ok(None)
            };
        };
        let now = self
            .clock
            .now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        pricing::estimate(
            pricing,
            submission,
            profile.pricing.peak.as_ref(),
            now,
            usage,
            &route.pricing_model,
        )
        .map(Some)
    }

    /// Price usage against the connection that actually served a response.
    pub fn estimate_actual_cost(
        &self,
        route: &ResolvedRoute,
        response: &lingxi_agent_api::protocol::CompletionResponse,
        usage: &Usage,
        submission: Submission,
    ) -> Result<Option<pricing::CostEstimate>, LlmError> {
        let name =
            response
                .executed_profile
                .as_deref()
                .ok_or_else(|| LlmError::CostUnavailable {
                    message: "response has no executed profile".into(),
                })?;
        self.estimate_cost_for_profile(route, name, usage, submission)
    }

    /// Price usage from a streamed request using `ModelStream::executed_profile`.
    pub fn estimate_cost_for_profile(
        &self,
        route: &ResolvedRoute,
        name: &str,
        usage: &Usage,
        submission: Submission,
    ) -> Result<Option<pricing::CostEstimate>, LlmError> {
        if name != route.profile_name
            && !route
                .connection_chain
                .iter()
                .any(|hop| hop.profile_name == name)
        {
            return Err(LlmError::CostUnavailable {
                message: format!("profile {name:?} is not on this route"),
            });
        }
        let profile = self
            .profile(name)
            .ok_or_else(|| LlmError::CostUnavailable {
                message: format!("executed profile {name:?} is unavailable"),
            })?;
        let model = profile
            .models
            .iter()
            .find(|m| m.request_model == route.request_model)
            .ok_or_else(|| LlmError::CostUnavailable {
                message: format!(
                    "executed profile {name:?} does not serve {:?}",
                    route.request_model
                ),
            })?;
        let mut actual_route = route.clone();
        actual_route.profile_name = name.to_owned();
        actual_route.provider_id = profile.provider_id.clone();
        actual_route.display_model = model.display_model.clone();
        actual_route.pricing_model.pricing_provider_id = profile.provider_id.clone();
        actual_route.pricing_model.billing_model = model.billing_model.clone();
        actual_route.pricing_model.display_model = model.display_model.clone();
        self.estimate_cost(&actual_route, usage, submission)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{
        HttpRequest, HttpResponse, StreamResponse, Transport, WebSocketSession,
    };
    use async_trait::async_trait;
    use lingxi_agent_api::protocol::LlmError;

    struct NoHttp;
    #[async_trait]
    impl Transport for NoHttp {
        async fn execute(&self, _req: HttpRequest) -> Result<HttpResponse, LlmError> {
            Err(LlmError::Transport {
                message: "test transport".into(),
            })
        }
        async fn open_stream(&self, _req: HttpRequest) -> Result<StreamResponse, LlmError> {
            Err(LlmError::Transport {
                message: "test transport".into(),
            })
        }
        async fn open_responses_websocket_session(
            &self,
            _req: HttpRequest,
        ) -> Result<Box<dyn WebSocketSession>, LlmError> {
            Err(LlmError::Transport {
                message: "test transport".into(),
            })
        }
    }
    fn profile(name: &str, protocol: &str) -> ProviderProfile {
        serde_json::from_value(serde_json::json!({
            "provider_id": "acme", "profile_name": name, "base_url": "https://x.test",
            "protocol": protocol, "auth": "none",
            "models": [{"display_model": "m1", "request_model": "m1", "billing_model": "m1"}]
        }))
        .unwrap()
    }

    /// Gate 33's error path. Every family in the closed set now ships a codec,
    /// so this is only reachable by removing one or by adding a tenth family —
    /// which is exactly what it is here to catch. `tests/families.rs` asserts
    /// the covering invariant that keeps it unreachable in practice.
    #[test]
    fn build_names_the_profile_and_the_family_when_a_codec_is_missing() {
        let mut b = LlmClientBuilder::with_transport(
            Arc::new(NoHttp),
            &[profile("p1", "gemini_generate_content")],
        );
        b.codecs.remove(&ProtocolFamily::GeminiGenerateContent);
        assert_eq!(
            b.build()
                .err()
                .expect("a profile with no codec cannot build"),
            BuildError::MissingCodec {
                profile_name: "p1".into(),
                family: ProtocolFamily::GeminiGenerateContent
            }
        );
    }

    #[test]
    fn a_family_this_crate_speaks_needs_no_capability_entry() {
        let b =
            LlmClientBuilder::with_transport(Arc::new(NoHttp), &[profile("p1", "open_ai_chat")]);
        assert!(
            b.build().is_ok(),
            "an OpenAI-compatible provider is a settings entry and nothing else (gate 30)"
        );
    }

    #[test]
    fn build_rejects_duplicate_profile_names() {
        let b = LlmClientBuilder::with_transport(
            Arc::new(NoHttp),
            &[profile("p1", "open_ai_chat"), profile("p1", "open_ai_chat")],
        );
        assert_eq!(
            b.build().err().unwrap(),
            BuildError::DuplicateProfile {
                profile_name: "p1".into()
            }
        );
    }

    #[test]
    fn empty_builder_builds_and_lists_nothing() {
        let c = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[])
            .build()
            .unwrap();
        assert!(c.models().is_empty());
    }

    #[test]
    fn build_rejects_empty_and_malformed_peak_windows() {
        use lingxi_agent_api::protocol::PeakSchedule;

        for windows in [vec![], vec!["25:00-26:00".to_owned()]] {
            let mut profile = profile("p1", "open_ai_chat");
            profile.pricing.peak = Some(PeakSchedule {
                utc_windows: windows,
                weekdays_only: false,
                off_peak_multiplier: 0.5,
            });
            assert!(matches!(
                LlmClientBuilder::with_transport(Arc::new(NoHttp), &[profile]).build(),
                Err(BuildError::InvalidPeakSchedule { .. })
            ));
        }
    }

    #[test]
    fn clock_override_controls_time_based_pricing() {
        use lingxi_agent_api::protocol::{PeakSchedule, TokenPricing};
        use std::time::{Duration, SystemTime, UNIX_EPOCH};

        struct FixedClock(u64);
        impl Clock for FixedClock {
            fn now(&self) -> SystemTime {
                UNIX_EPOCH + Duration::from_secs(self.0)
            }
        }

        let mut p = profile("p1", "open_ai_chat");
        p.models[0].pricing = Some(TokenPricing {
            input_per_million: Some(2.0),
            ..Default::default()
        });
        p.pricing.peak = Some(PeakSchedule {
            weekdays_only: false,
            utc_windows: vec!["09:00-17:00".into()],
            off_peak_multiplier: 0.5,
        });
        for (hour, expected) in [(12, 2.0), (20, 1.0)] {
            let mut builder = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[p.clone()]);
            builder.with_clock(Arc::new(FixedClock(hour * 3600)));
            let client = builder.build().unwrap();
            let route = client.resolve("m1").unwrap();
            let cost = client
                .estimate_cost(
                    &route,
                    &Usage {
                        input_tokens: 1_000_000,
                        ..Default::default()
                    },
                    Submission::Interactive,
                )
                .unwrap()
                .unwrap();
            assert_eq!(cost.total_usd, expected);
        }
    }
}
