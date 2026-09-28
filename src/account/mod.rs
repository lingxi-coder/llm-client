//! Read-only account usage, balances and plan entitlements.
//!
//! Account data has different authority and scope from one model response.
//! Sources are selected by provider *and* the caller's explicit account kind;
//! `AuthStrategy::Bearer` alone cannot distinguish a user from an API key.

mod execution;
mod service;
pub use execution::{AccountExecutionOptions, AccountFetchContext, AccountReport};
pub(crate) use service::{Registry, Service};
pub(crate) mod support;

/// An RPC connection already authenticated and owned by the embedding host.
/// This crate neither starts a CLI process nor reads its credential store.
/// `call` returns the method's JSON-RPC `result` payload.
#[async_trait]
pub trait AccountRpc: Send + Sync + 'static {
    async fn call(&self, method: &str, params: &Value) -> Result<Value, AccountFailure>;
}

use crate::protocol::{ProviderId, ProviderProfile, Secret};
use crate::transport::{HttpRequest, Transport};
use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;
use thiserror::Error;

const MAX_ACCOUNT_BODY: usize = 2 * 1024 * 1024;

mod types;
pub use types::*;

#[async_trait]
pub trait AccountUsageSource: Send + Sync + 'static {
    /// A source backed by one signed-in session must be registered for a
    /// specific profile when the provider has multiple configured profiles.
    fn requires_profile_binding(&self, _query: &AccountQuery) -> bool {
        false
    }

    async fn fetch(
        &self,
        context: &AccountFetchContext<'_>,
        report: &mut AccountReport,
    ) -> Result<(), AccountFailure>;
}

pub(crate) fn builtin_sources(
) -> BTreeMap<(String, AccountIdentity), std::sync::Arc<dyn AccountUsageSource>> {
    let mut sources: BTreeMap<(String, AccountIdentity), std::sync::Arc<dyn AccountUsageSource>> =
        BTreeMap::new();
    crate::providers::openai::account::register(&mut sources);
    crate::providers::anthropic::account::register(&mut sources);
    crate::providers::google::account::register(&mut sources);
    crate::providers::xai::account::register(&mut sources);
    crate::providers::openrouter::account::register(&mut sources);
    crate::providers::deepseek::account::register(&mut sources);
    crate::providers::kimi::account::register(&mut sources);
    crate::providers::qwen::account::register(&mut sources);
    crate::providers::minimax::account::register(&mut sources);
    sources
}
pub(crate) mod http;
pub(crate) use http::*;
#[cfg(test)]
mod tests {
    use super::{parse_iso_utc, AccountIdentity, AccountQuery};

    #[test]
    fn account_history_defaults_to_last_thirty_days() {
        let now = 40 * 86_400;
        assert_eq!(
            AccountQuery::new(AccountIdentity::ApiKey).range(now),
            Ok((10 * 86_400, now))
        );
    }

    #[test]
    fn account_dates_are_validated_and_normalized() {
        assert_eq!(parse_iso_utc("1970-01-01"), Some(0));
        assert_eq!(parse_iso_utc("1970-01-01T01:00:00+01:00"), Some(0));
        assert_eq!(
            parse_iso_utc("2024-02-29T00:00:00.123Z"),
            Some(1_709_164_800)
        );
        assert_eq!(parse_iso_utc("2023-02-29"), None);
        assert_eq!(parse_iso_utc("2026-09-23T99:00:00Z"), None);
    }
}
