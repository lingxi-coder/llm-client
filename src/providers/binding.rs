//! Exact-profile binding shared by typed provider clients and native resources.
use crate::{
    protocol::LlmError,
    runtime::{ClientSnapshot, ClientSource, OwnedClientSource},
    transport::Transport,
};

/// A failed exact-profile binding. Routing groups are never accepted here.
#[derive(Debug, thiserror::Error)]
pub enum ProviderBindingError {
    #[error("unknown provider profile {profile_name:?}; expected an exact profile name")]
    UnknownProfile { profile_name: String },
    #[error("profile {profile_name:?} belongs to provider {actual:?}, expected {expected:?}")]
    ProviderMismatch {
        profile_name: String,
        expected: &'static str,
        actual: String,
    },
    #[error("profile {profile_name:?} is unavailable in this client's region")]
    UnavailableRegion { profile_name: String },
}

/// Construction contract for the concrete provider clients.
///
/// This trait is sealed: use one of the provider client types supplied by this crate.
pub trait ProviderClient: private::Sealed + Sized {
    /// Stable provider identity, independent of a profile's display name or group.
    const PROVIDER_ID: &'static str;
    #[doc(hidden)]
    fn from_binding(binding: ProviderBinding) -> Self;
}
pub(crate) mod private {
    pub trait Sealed {}
}

/// Runtime binding carried by a typed client. It never retains credentials.
#[doc(hidden)]
#[derive(Clone)]
pub struct ProviderBinding {
    pub(crate) source: OwnedClientSource,
    profile_name: String,
    provider_id: &'static str,
}
impl ProviderBinding {
    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }
    pub(crate) fn provider_id(&self) -> &'static str {
        self.provider_id
    }
    pub(crate) fn source(&self) -> ClientSource<'_> {
        ClientSource::Bound {
            source: &self.source,
            profile_name: &self.profile_name,
            provider_id: self.provider_id,
        }
    }
    /// Capture and validate once at the start of one native operation.
    pub(crate) fn pin(&self) -> Result<ClientSnapshot, LlmError> {
        self.source().pin()
    }
    pub(crate) fn pinned(&self) -> Result<Self, LlmError> {
        let snapshot = self.pin()?;
        Ok(Self {
            source: OwnedClientSource::fixed(&snapshot),
            profile_name: self.profile_name.clone(),
            provider_id: self.provider_id,
        })
    }
    pub(crate) fn transport(&self) -> &dyn Transport {
        self.source.runtime.http.as_ref()
    }
    pub(crate) fn validate_scope(
        &self,
        profile_name: &str,
        account_scope: &str,
    ) -> Result<(), LlmError> {
        self.pin()?;
        if profile_name != self.profile_name || account_scope.trim().is_empty() {
            return Err(LlmError::InvalidRequest { message: "native resource scope must name the bound profile and an explicit nonempty account scope".into() });
        }
        Ok(())
    }
}

pub(crate) fn bind<T: ProviderClient>(
    source: OwnedClientSource,
    profile_name: &str,
) -> Result<T, ProviderBindingError> {
    let snapshot = source.snapshot();
    let profile =
        snapshot
            .profile(profile_name)
            .ok_or_else(|| ProviderBindingError::UnknownProfile {
                profile_name: profile_name.into(),
            })?;
    if profile.provider_id.as_str() != T::PROVIDER_ID {
        return Err(ProviderBindingError::ProviderMismatch {
            profile_name: profile_name.into(),
            expected: T::PROVIDER_ID,
            actual: profile.provider_id.to_string(),
        });
    }
    if !profile.supports_region(snapshot.runtime.region) {
        return Err(ProviderBindingError::UnavailableRegion {
            profile_name: profile_name.into(),
        });
    }
    Ok(T::from_binding(ProviderBinding {
        source,
        profile_name: profile_name.into(),
        provider_id: T::PROVIDER_ID,
    }))
}

/// Pin the provider identity for one connection and verify its explicit account scope.
pub(crate) fn pin_realtime_scope(
    binding: &ProviderBinding,
    profile_name: &str,
    account_scope: &str,
) -> Result<ProviderBinding, crate::realtime::RealtimeError> {
    let map_error = |error: LlmError| crate::realtime::RealtimeError::InvalidConfig {
        message: error.to_string(),
    };
    let pinned = binding.pinned().map_err(map_error)?;
    pinned
        .validate_scope(profile_name, account_scope)
        .map_err(map_error)?;
    Ok(pinned)
}

#[cfg(test)]
mod tests {
    use crate::{
        protocol::{PricingContext, ProviderProfile},
        LlmClientBuilder,
    };
    use serde_json::json;

    #[test]
    fn native_scope_restriction_does_not_hide_snapshot_pricing_profiles() {
        let profiles: Vec<ProviderProfile> = ["primary", "backup"]
            .into_iter()
            .map(|name| {
                serde_json::from_value(json!({
                "provider_id":"openai", "profile_name":name,
                "protocol":"open_ai_chat", "base_url":"https://api.example.test/v1", "auth":"none",
                "models":[{"request_model":"wire-m","display_model":"m","billing_model":"wire-m"}]
            })).unwrap()
            })
            .collect();
        let client = LlmClientBuilder::new(&profiles)
            .unwrap()
            .with_region(crate::protocol::Region::International)
            .build()
            .unwrap();
        let typed = client
            .provider::<crate::providers::OpenAiClient>("primary")
            .unwrap();
        let pinned = typed.binding.pin().unwrap();
        assert!(pinned.native_profile("primary").is_some());
        assert!(pinned.native_profile("backup").is_none());
        assert!(pinned.profile("backup").is_some());
        let route = pinned.resolve_in("m", Some("backup")).unwrap();
        pinned
            .price_quote_for_route(&route, &PricingContext::default())
            .unwrap();
    }
}
