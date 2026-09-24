//! Account data supplied by an authenticated local host or official loopback service.

use super::{
    field_from_result, get_json, header_bearer, parse_iso_utc, AccountBalance, AccountFailure,
    AccountMetric, AccountQuery, AccountQuotaWindow, AccountScope, AccountScopeKind,
    AccountSubscription, AccountTokenBucket, AccountTokenUsage, AccountUsageSource,
    SubscriptionStatus,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::net::IpAddr;
use std::sync::Arc;

/// An RPC connection already authenticated and owned by the embedding host.
/// This crate neither starts a CLI process nor reads its credential store.
/// `call` returns the method's JSON-RPC `result` payload, not its outer envelope.
#[async_trait]
pub trait AccountRpc: Send + Sync + 'static {
    async fn call(&self, method: &str, params: &Value) -> Result<Value, AccountFailure>;
}

mod codex;
pub use codex::*;

mod copilot;
pub use copilot::*;

mod kimi;
pub use kimi::*;
fn user_scope(id: Option<String>) -> AccountScope {
    AccountScope::new(AccountScopeKind::User, id)
}

fn optional_unix(value: Option<&Value>) -> Result<Option<u64>, AccountFailure> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or(AccountFailure::InvalidResponse),
    }
}

fn optional_time(value: Option<&Value>) -> Result<Option<u64>, AccountFailure> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .and_then(parse_iso_utc)
            .map(Some)
            .ok_or(AccountFailure::InvalidResponse),
    }
}
