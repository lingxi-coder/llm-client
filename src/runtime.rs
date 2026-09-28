//! Shared monotonic deadlines. Wall-clock policy lives in `Clock`.

use crate::protocol::LlmError;
use std::{future::Future, time::Duration};
use tokio::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Deadline(Option<Instant>);

impl Deadline {
    pub(crate) fn after(timeout: Option<Duration>) -> Self {
        let now = Instant::now();
        Self(timeout.map(|duration| now.checked_add(duration).unwrap_or(now)))
    }

    pub(crate) fn at(instant: Option<std::time::Instant>) -> Self {
        Self(instant.map(Instant::from_std))
    }

    pub(crate) fn remaining(self) -> Result<Option<Duration>, LlmError> {
        self.0
            .map(|end| {
                end.checked_duration_since(Instant::now())
                    .filter(|duration| !duration.is_zero())
                    .ok_or_else(timeout_error)
            })
            .transpose()
    }

    pub(crate) fn cap(self, timeout: Option<Duration>) -> Self {
        let other = Self::after(timeout);
        Self(match (self.0, other.0) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        })
    }

    pub(crate) async fn run<F: Future>(self, future: F) -> Result<F::Output, LlmError> {
        let Some(remaining) = self.remaining()? else {
            return Ok(future.await);
        };
        match futures::future::select(Box::pin(future), Box::pin(delay(remaining))).await {
            futures::future::Either::Left((result, _)) => Ok(result),
            futures::future::Either::Right((_, _)) => Err(timeout_error()),
        }
    }
}

pub(crate) fn timeout_error() -> LlmError {
    LlmError::TransportTimeout {
        message: "request deadline elapsed".into(),
    }
}

/// Also works with an embedding application's non-Tokio executor. Dropping
/// the future interrupts the fallback sleeper instead of leaving it running.
pub(crate) async fn delay(duration: Duration) {
    if tokio::runtime::Handle::try_current().is_ok() {
        tokio::time::sleep(duration).await;
        return;
    }
    let (cancel_send, cancel_receive) = std::sync::mpsc::channel::<()>();
    let (send, receive) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let _ = cancel_receive.recv_timeout(duration);
        let _ = send.send(());
    });
    let _cancel_on_drop = cancel_send;
    let _ = receive.await;
}

use crate::account;
use crate::auth::Authenticator;
use crate::codecs::WireCodec;
use crate::directory::ModelDirectory;
use crate::protocol::{AuthStrategy, ProtocolFamily, Region};
use crate::transport::{Clock, Transport};
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

/// An owned, immutable view of one configuration revision.
#[derive(Clone)]
pub struct ClientSnapshot {
    pub(crate) runtime: Arc<RuntimeResources>,
    pub(crate) state: Arc<PublishedState>,
    pub(crate) bound_profile: Option<Arc<str>>,
}

pub(crate) struct RuntimeResources {
    pub(crate) region: Region,
    pub(crate) http: Arc<dyn Transport>,
    pub(crate) clock: Arc<dyn Clock>,
    pub(crate) codecs: BTreeMap<ProtocolFamily, Arc<dyn WireCodec>>,
    pub(crate) image_adapters:
        BTreeMap<crate::protocol::ImageApi, Arc<dyn crate::images::ImageAdapter>>,
    pub(crate) image_authenticators: BTreeMap<String, Arc<dyn crate::images::ImageAuthenticator>>,
    pub(crate) directories: BTreeMap<ProtocolFamily, Arc<dyn ModelDirectory>>,
    pub(crate) authenticators: BTreeMap<AuthStrategy, Arc<dyn Authenticator>>,
    pub(crate) accounts: account::Service,
    pub(crate) attachments: crate::client::AttachmentManager,
}

pub(crate) struct PublishedState {
    pub(crate) revision: u64,
    pub(crate) config: Arc<crate::client::snapshot::RuntimeSnapshot>,
    pub(crate) account_registry: Arc<account::Registry>,
    pub(crate) cache_namespace: u64,
    pub(crate) cache_generations: BTreeMap<String, u64>,
}

/// Shared execution source, independent of the public client facades.
#[derive(Clone)]
pub(crate) struct OwnedClientSource {
    pub(crate) runtime: Arc<RuntimeResources>,
    state: SourceState,
}
#[derive(Clone)]
enum SourceState {
    Live(Arc<RwLock<Arc<PublishedState>>>),
    Fixed(Arc<PublishedState>),
}
impl OwnedClientSource {
    pub(crate) fn live(
        runtime: &Arc<RuntimeResources>,
        published: &Arc<RwLock<Arc<PublishedState>>>,
    ) -> Self {
        Self {
            runtime: runtime.clone(),
            state: SourceState::Live(published.clone()),
        }
    }
    pub(crate) fn fixed(snapshot: &ClientSnapshot) -> Self {
        Self {
            runtime: snapshot.runtime.clone(),
            state: SourceState::Fixed(snapshot.state.clone()),
        }
    }
    pub(crate) fn snapshot(&self) -> ClientSnapshot {
        let state = match &self.state {
            SourceState::Live(published) => published
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
            SourceState::Fixed(state) => state.clone(),
        };
        ClientSnapshot {
            runtime: self.runtime.clone(),
            state,
            bound_profile: None,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ClientSource<'a> {
    Live {
        runtime: &'a Arc<RuntimeResources>,
        published: &'a Arc<RwLock<Arc<PublishedState>>>,
    },
    Snapshot(&'a ClientSnapshot),
    Bound {
        source: &'a OwnedClientSource,
        profile_name: &'a str,
        provider_id: &'static str,
    },
}
impl<'a> ClientSource<'a> {
    pub(crate) fn live(
        runtime: &'a Arc<RuntimeResources>,
        published: &'a Arc<RwLock<Arc<PublishedState>>>,
    ) -> Self {
        Self::Live { runtime, published }
    }
    pub(crate) fn fixed(snapshot: &'a ClientSnapshot) -> Self {
        Self::Snapshot(snapshot)
    }
    pub(crate) fn snapshot(self) -> ClientSnapshot {
        match self {
            Self::Live { runtime, published } => {
                OwnedClientSource::live(runtime, published).snapshot()
            }
            Self::Snapshot(snapshot) => snapshot.clone(),
            Self::Bound {
                source,
                profile_name,
                ..
            } => {
                let mut snapshot = source.snapshot();
                snapshot.bound_profile = Some(Arc::from(profile_name));
                snapshot
            }
        }
    }
    pub(crate) fn profile_name(self) -> Result<&'a str, LlmError> {
        match self {
            Self::Bound { profile_name, .. } => Ok(profile_name),
            _ => Err(LlmError::InvalidRequest {
                message: "provider resource requires a typed provider binding".into(),
            }),
        }
    }
    pub(crate) fn pin(self) -> Result<ClientSnapshot, LlmError> {
        let snapshot = self.snapshot();
        if let Self::Bound {
            profile_name,
            provider_id,
            ..
        } = self
        {
            let profile =
                snapshot
                    .profile(profile_name)
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: format!("provider profile {profile_name:?} no longer exists"),
                    })?;
            if profile.provider_id.as_str() != provider_id
                || !profile.supports_region(snapshot.runtime.region)
            {
                return Err(LlmError::InvalidRequest { message: format!("provider profile {profile_name:?} no longer matches its bound provider and region") });
            }
        }
        Ok(snapshot)
    }
}
