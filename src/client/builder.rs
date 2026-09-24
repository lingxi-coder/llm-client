//! Composition and validation of client services.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BuildError {
    #[error("invalid image configuration on {profile_name:?}: {reason}")]
    InvalidImage {
        profile_name: String,
        reason: String,
    },
    #[error("invalid inference or pricing configuration on {profile_name:?}: {reason}")]
    InvalidInference {
        profile_name: String,
        reason: String,
    },
    #[error("a usage region must be explicitly selected with with_region")]
    MissingRegion,
    #[error(
        "provider profile {profile_name:?} uses protocol {family:?} but no codec for it is registered"
    )]
    MissingCodec {
        profile_name: String,
        family: ProtocolFamily,
    },
    #[error(
        "provider profile {profile_name:?} uses auth {strategy:?} but no authenticator for it is registered"
    )]
    MissingAuthenticator {
        profile_name: String,
        strategy: AuthStrategy,
    },
    #[error("provider profile name {profile_name:?} is declared twice")]
    DuplicateProfile { profile_name: String },
    #[error("connection id {connection_id:?} is declared twice in group {group:?}")]
    DuplicateConnectionId {
        group: String,
        connection_id: String,
    },
    #[error("provider profile {profile_name:?} has an invalid peak price schedule: {reason}")]
    InvalidPeakSchedule {
        profile_name: String,
        reason: String,
    },
}

pub struct LlmClientBuilder {
    account_concurrency: std::num::NonZeroUsize,
    region: Option<Region>,
    builtin_definitions: BTreeSet<String>,
    http: Arc<dyn Transport>,
    clock: Arc<dyn Clock>,
    pub(super) codecs: BTreeMap<ProtocolFamily, Arc<dyn WireCodec>>,
    image_adapters: BTreeMap<crate::protocol::ImageApi, Arc<dyn crate::images::ImageAdapter>>,
    image_authenticators: BTreeMap<String, Arc<dyn crate::images::ImageAuthenticator>>,
    directories: BTreeMap<ProtocolFamily, Arc<dyn ModelDirectory>>,
    authenticators: BTreeMap<AuthStrategy, Arc<dyn Authenticator>>,
    account_sources:
        BTreeMap<(String, account::AccountIdentity), Arc<dyn account::AccountUsageSource>>,
    profile_account_sources:
        BTreeMap<(String, account::AccountIdentity), Arc<dyn account::AccountUsageSource>>,
    attachment_resolver: Option<Arc<dyn AttachmentResolver>>,
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
            account_concurrency: std::num::NonZeroUsize::new(4).unwrap(),
            region: None,
            builtin_definitions: BTreeSet::new(),
            http,
            clock: Arc::new(SystemClock),
            codecs,
            image_adapters: crate::images::builtin_adapters(),
            image_authenticators: BTreeMap::new(),
            directories,
            authenticators: BTreeMap::new(),
            account_sources: account::builtin_sources(),
            profile_account_sources: BTreeMap::new(),
            attachment_resolver: None,
            profiles: profiles.to_vec(),
        };
        builder.register_authenticator(AuthStrategy::ApiKey, Arc::new(crate::ApiKeyAuthenticator));
        builder.register_authenticator(AuthStrategy::Bearer, Arc::new(crate::BearerAuthenticator));
        builder
    }

    /// Choose the required usage region. Build a new client to change regions.
    #[must_use]
    pub fn with_region(mut self, region: Region) -> Self {
        self.region = Some(region);
        self
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

    /// Register a provider image protocol independently of chat codecs.
    pub fn register_image_adapter(
        &mut self,
        adapter: Arc<dyn crate::images::ImageAdapter>,
    ) -> &mut Self {
        self.image_adapters.insert(adapter.api(), adapter);
        self
    }

    pub fn register_image_authenticator(
        &mut self,
        name: impl Into<String>,
        authenticator: Arc<dyn crate::images::ImageAuthenticator>,
    ) -> &mut Self {
        self.image_authenticators.insert(name.into(), authenticator);
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

    /// Register an account source for one provider and principal kind.
    /// Host-owned OAuth and local service sessions can be supplied here.
    /// Session-bound sources require one unchanged account. After changing
    /// that account, rebind with `register_profile_account_source`.
    pub fn register_account_source(
        &mut self,
        provider_id: impl Into<String>,
        identity: account::AccountIdentity,
        source: Arc<dyn account::AccountUsageSource>,
    ) -> &mut Self {
        self.account_sources
            .insert((provider_id.into(), identity), source);
        self
    }

    /// Register a session-bound account source for exactly one connection.
    /// Use this for separate Codex/Copilot logins under the same provider.
    pub fn register_profile_account_source(
        &mut self,
        profile_name: impl Into<String>,
        identity: account::AccountIdentity,
        source: Arc<dyn account::AccountUsageSource>,
    ) -> &mut Self {
        self.profile_account_sources
            .insert((profile_name.into(), identity), source);
        self
    }

    /// Resolve app-owned attachments before provider-specific request
    /// preparation. The application remains responsible for storage and
    /// cross-device authorization.
    pub fn with_attachment_resolver(&mut self, resolver: Arc<dyn AttachmentResolver>) -> &mut Self {
        self.attachment_resolver = Some(resolver);
        self
    }

    /// Maximum independently scheduled account queries; output order remains stable.
    pub fn with_account_concurrency(&mut self, concurrency: std::num::NonZeroUsize) -> &mut Self {
        self.account_concurrency = concurrency;
        self
    }

    /// Register bundled defaults with explicit builtin provenance for persistence.
    pub fn add_builtin_profiles(&mut self) -> Result<&mut Self, crate::presets::PresetError> {
        for profile in crate::presets::builtin()? {
            self.builtin_definitions
                .insert(profile.profile_name.clone());
            if let Some(existing) = self
                .profiles
                .iter_mut()
                .find(|p| p.profile_name == profile.profile_name)
            {
                *existing = profile;
            } else {
                self.profiles.push(profile);
            }
        }
        Ok(self)
    }

    /// Register one bundled definition, preserving its builtin source on disk.
    pub fn add_builtin_profile(
        &mut self,
        name: &str,
    ) -> Result<&mut Self, crate::configuration::ProviderStoreError> {
        let profile = crate::presets::builtin()?
            .into_iter()
            .find(|p| p.profile_name == name)
            .ok_or_else(|| crate::configuration::ProviderStoreError::NotBuiltin(name.into()))?;
        self.builtin_definitions.insert(name.into());
        if let Some(existing) = self.profiles.iter_mut().find(|p| p.profile_name == name) {
            *existing = profile;
        } else {
            self.profiles.push(profile);
        }
        Ok(self)
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
    /// request, naming the profile (gate 33). A region must be selected first.
    pub fn build(self) -> Result<LlmClient, BuildError> {
        let region = self.region.ok_or(BuildError::MissingRegion)?;
        validate_profiles(&self.profiles, &self.codecs, &self.authenticators)?;
        for profile in &self.profiles {
            for route in profile.images.routes.values() {
                if !self.image_adapters.contains_key(&route.api) {
                    return Err(BuildError::InvalidImage {
                        profile_name: profile.profile_name.clone(),
                        reason: format!("missing image adapter {:?}", route.api),
                    });
                }
                if let Some(auth) = &route.authenticator {
                    if !self.image_authenticators.contains_key(auth) {
                        return Err(BuildError::InvalidImage {
                            profile_name: profile.profile_name.clone(),
                            reason: format!("missing image authenticator {auth:?}"),
                        });
                    }
                }
            }
        }
        let mut store = crate::configuration::Coordinator::new(self.profiles.clone());
        store.definitions.builtin_names = self.builtin_definitions;
        Ok(LlmClient {
            accounts: account::Service::new(
                self.account_sources,
                self.profile_account_sources,
                self.http.clone(),
                self.clock.clone(),
                self.account_concurrency,
            ),
            region,
            attachments: AttachmentManager::new(self.http.clone(), self.attachment_resolver),
            http: self.http,
            clock: self.clock,
            codecs: self.codecs,
            image_adapters: self.image_adapters,
            image_authenticators: self.image_authenticators,
            directories: self.directories,
            authenticators: self.authenticators,
            snapshot: Arc::new(super::snapshot::RuntimeSnapshot::new(
                region,
                self.profiles,
                None,
            )),
            store,
        })
    }
}

pub(super) fn validate_profiles(
    profiles: &[ProviderProfile],
    codecs: &BTreeMap<ProtocolFamily, Arc<dyn WireCodec>>,
    authenticators: &BTreeMap<AuthStrategy, Arc<dyn Authenticator>>,
) -> Result<(), BuildError> {
    let mut seen = std::collections::BTreeSet::new();
    let mut connections = std::collections::BTreeSet::new();
    for p in profiles {
        for (name, route) in &p.images.routes {
            for (label, raw) in [
                ("base_url", &route.base_url),
                (
                    "task_base_url",
                    route.task_base_url.as_ref().unwrap_or(&route.base_url),
                ),
            ] {
                let parsed = url::Url::parse(raw).map_err(|e| BuildError::InvalidImage {
                    profile_name: p.profile_name.clone(),
                    reason: format!("route {name:?} {label}: {e}"),
                })?;
                if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
                    return Err(BuildError::InvalidImage {
                        profile_name: p.profile_name.clone(),
                        reason: format!("route {name:?} {label} must be an HTTP(S) URL"),
                    });
                }
            }
            if route.api_key_header.as_ref().is_some_and(|header| {
                header.is_empty()
                    || !header
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'-')
            }) {
                return Err(BuildError::InvalidImage {
                    profile_name: p.profile_name.clone(),
                    reason: format!("route {name:?} has an invalid API-key header"),
                });
            }
        }
        let mut image_names = BTreeSet::new();
        for model in &p.images.models {
            if !p.images.routes.contains_key(&model.route) {
                return Err(BuildError::InvalidImage {
                    profile_name: p.profile_name.clone(),
                    reason: format!(
                        "model {:?} references missing route {:?}",
                        model.request_model, model.route
                    ),
                });
            }
            if (model.capabilities.async_generate || model.capabilities.async_edit)
                && p.images.routes.get(&model.route).is_some_and(|route| {
                    matches!(
                        route.api,
                        crate::protocol::ImageApi::Qwen | crate::protocol::ImageApi::Wan
                    ) && route.task_base_url.is_none()
                })
            {
                return Err(BuildError::InvalidImage {
                    profile_name: p.profile_name.clone(),
                    reason: format!("model {:?} needs a task_base_url", model.request_model),
                });
            }
            let selectors: BTreeSet<_> = std::iter::once(&model.display_model)
                .chain(std::iter::once(&model.request_model))
                .chain(&model.aliases)
                .cloned()
                .collect();
            for selector in selectors {
                if selector.is_empty() || !image_names.insert(selector.clone()) {
                    return Err(BuildError::InvalidImage {
                        profile_name: p.profile_name.clone(),
                        reason: format!("ambiguous or empty image selector {selector:?}"),
                    });
                }
            }
        }
        if !seen.insert(p.profile_name.clone()) {
            return Err(BuildError::DuplicateProfile {
                profile_name: p.profile_name.clone(),
            });
        }
        if let Some(connection_id) = &p.connection.connection_id {
            let identity = (p.group().to_owned(), connection_id.clone());
            if !connections.insert(identity.clone()) {
                return Err(BuildError::DuplicateConnectionId {
                    group: identity.0,
                    connection_id: identity.1,
                });
            }
        }
    }
    for p in profiles {
        for model in &p.models {
            if let Some(prices) = &model.pricing {
                super::price_query::validate_prices(prices).map_err(|reason| {
                    BuildError::InvalidInference {
                        profile_name: p.profile_name.clone(),
                        reason,
                    }
                })?;
            }
            let f = model.info.features.on_connection(&p.info.features);
            if f.budget
                .min_tokens
                .zip(f.budget.max_tokens)
                .is_some_and(|(min, max)| min > max)
            {
                return Err(BuildError::InvalidInference {
                    profile_name: p.profile_name.clone(),
                    reason: "empty thinking budget range".into(),
                });
            }
        }
        if let Some(peak) = &p.pricing.peak {
            peak.validate()
                .map_err(|reason| BuildError::InvalidPeakSchedule {
                    profile_name: p.profile_name.clone(),
                    reason,
                })?;
        }
        if !codecs.contains_key(&p.protocol) {
            return Err(BuildError::MissingCodec {
                profile_name: p.profile_name.clone(),
                family: p.protocol,
            });
        }
        if p.auth != AuthStrategy::None && !authenticators.contains_key(&p.auth) {
            return Err(BuildError::MissingAuthenticator {
                profile_name: p.profile_name.clone(),
                strategy: p.auth,
            });
        }
    }
    // A profile whose `model_list` names a shape with no directory is
    // deliberately NOT an error here. It can run every turn it could run
    // before; all it cannot do is refresh its own list.
    Ok(())
}
