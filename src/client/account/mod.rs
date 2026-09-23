//! Read-only account usage, balances and plan entitlements.
//!
//! Account data has different authority and scope from one model response.
//! Sources are selected by provider *and* the caller's explicit account kind;
//! `AuthStrategy::Bearer` alone cannot distinguish a user from an API key.

mod admin;
mod local;
mod minimax;
mod public;
mod qwen;

pub use local::{AccountRpc, CodexAccountSource, CopilotAccountSource, KimiCodeAccountSource};

use crate::transport::{HttpRequest, Transport};
use async_trait::async_trait;
use bytes::Bytes;
use lingxi_agent_api::protocol::{ProviderId, ProviderProfile, Secret};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;
use thiserror::Error;

const ACCOUNT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_ACCOUNT_BODY: usize = 2 * 1024 * 1024;

/// The kind of principal whose account is queried, independent of the wire's
/// authentication header or the connection's billing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountIdentity {
    ApiKey,
    AuthUser,
}

/// IDs needed to select the precise scope of a provider's account API.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AccountSelector {
    pub organization_id: Option<String>,
    pub project_id: Option<String>,
    pub workspace_id: Option<String>,
    pub team_id: Option<String>,
    pub api_key_id: Option<String>,
}

/// Optional Alibaba Cloud AccessKey credentials for Model Studio's signed
/// billing-trend API. These values belong to one query and are never stored.
#[derive(Debug, Clone)]
pub struct AlibabaAccessKey {
    pub id: String,
    pub secret: Secret<String>,
    pub security_token: Option<Secret<String>>,
}

/// Credentials belong to this query and are never persisted by the client.
#[derive(Debug, Clone)]
pub struct AccountQuery {
    pub identity: AccountIdentity,
    /// The ordinary API key or user token, when a source uses one directly.
    pub credential: Option<Secret<String>>,
    /// A separate Admin/Management credential, when required by a source.
    pub management_credential: Option<Secret<String>>,
    /// Local service bearer token, used by a host-provided Kimi Code server.
    pub service_credential: Option<Secret<String>>,
    /// Host-provided loopback URL for an official local service.
    pub service_url: Option<String>,
    /// Separate Alibaba Cloud RAM AccessKey used only to sign GetBillingTrend.
    pub alibaba_access_key: Option<AlibabaAccessKey>,
    pub selector: AccountSelector,
    /// Inclusive UTC Unix seconds. Defaults to 30 days before the query.
    pub since_unix: Option<u64>,
    /// Exclusive UTC Unix seconds. Defaults to the query time.
    pub until_unix: Option<u64>,
}

impl AccountQuery {
    pub fn new(identity: AccountIdentity) -> Self {
        Self {
            identity,
            credential: None,
            management_credential: None,
            service_credential: None,
            service_url: None,
            alibaba_access_key: None,
            selector: AccountSelector::default(),
            since_unix: None,
            until_unix: None,
        }
    }

    pub(crate) fn range(&self, now: u64) -> Result<(u64, u64), AccountUsageError> {
        let until = self.until_unix.unwrap_or(now);
        let since = self
            .since_unix
            .unwrap_or_else(|| until.saturating_sub(30 * 86_400));
        if since >= until {
            return Err(AccountUsageError::InvalidTimeRange);
        }
        Ok((since, until))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AccountUsageError {
    #[error("unknown provider profile {0:?}")]
    UnknownProfile(String),
    #[error("no account query was supplied for provider profile {0:?}")]
    MissingQuery(String),
    #[error("account source for provider profile {0:?} must be registered per profile")]
    AmbiguousAccountSource(String),
    #[error("account usage start must be earlier than end")]
    InvalidTimeRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountScopeKind {
    ApiKey,
    User,
    Project,
    Workspace,
    Organization,
    Team,
    Account,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountScope {
    pub kind: AccountScopeKind,
    pub id: Option<String>,
}

impl AccountScope {
    pub fn new(kind: AccountScopeKind, id: Option<String>) -> Self {
        Self { kind, id }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountFailure {
    Transport,
    Unauthorized,
    PermissionDenied,
    RateLimited,
    InvalidResponse,
    ProviderError,
}

/// Per-field availability. A failed balance fetch does not erase valid usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AccountMetric<T> {
    Available {
        scope: AccountScope,
        source: String,
        value: T,
    },
    Unsupported,
    CredentialRequired,
    NotReported,
    Failed {
        reason: AccountFailure,
    },
}

impl<T> AccountMetric<T> {
    pub fn available(scope: AccountScope, source: &str, value: T) -> Self {
        Self::Available {
            scope,
            source: source.to_owned(),
            value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountBalance {
    /// ISO currency code or the provider's documented credit unit.
    pub unit: String,
    /// Decimal text preserves the provider's monetary precision.
    pub remaining: String,
    pub total: Option<String>,
    /// Provider-reported ability to spend this balance, when exposed.
    pub is_available: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountTokenBucket {
    pub start_unix: u64,
    pub end_unix: u64,
    pub model: Option<String>,
    /// Total input, including cached input when the provider reports it so.
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// A subset of `input_tokens`, never an additional total.
    pub cached_input_tokens: Option<u64>,
    /// Cache-creation input, also a subset of `input_tokens` when reported.
    pub cache_write_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountTokenUsage {
    pub lifetime_tokens: Option<u64>,
    pub buckets: Vec<AccountTokenBucket>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountCostBucket {
    pub start_unix: u64,
    pub end_unix: u64,
    pub currency: String,
    /// Exact decimal amount in major currency units (for example USD).
    pub amount: String,
    pub model: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountCostUsage {
    pub buckets: Vec<AccountCostBucket>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountQuotaWindow {
    pub name: String,
    pub duration_mins: Option<u64>,
    pub used_percent: Option<f64>,
    pub remaining_percent: Option<f64>,
    pub resets_at_unix: Option<u64>,
    pub limit: Option<u64>,
    pub used: Option<u64>,
    pub remaining: Option<u64>,
    /// Exact decimal quantities for fractional or monetary limits.
    pub limit_decimal: Option<String>,
    pub remaining_decimal: Option<String>,
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubscriptionStatus {
    VerifiedActive,
    VerifiedInactive,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountSubscription {
    pub plan_name: Option<String>,
    pub status: SubscriptionStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub profile_name: String,
    pub provider_id: ProviderId,
    pub identity: AccountIdentity,
    pub fetched_at_unix: u64,
    pub balance: AccountMetric<Vec<AccountBalance>>,
    pub token_usage: AccountMetric<AccountTokenUsage>,
    pub cost_usage: AccountMetric<AccountCostUsage>,
    pub quota_windows: AccountMetric<Vec<AccountQuotaWindow>>,
    pub subscription: AccountMetric<AccountSubscription>,
}

impl AccountSnapshot {
    pub fn unsupported(profile: &ProviderProfile, identity: AccountIdentity, now: u64) -> Self {
        Self {
            profile_name: profile.profile_name.clone(),
            provider_id: profile.provider_id.clone(),
            identity,
            fetched_at_unix: now,
            balance: AccountMetric::Unsupported,
            token_usage: AccountMetric::Unsupported,
            cost_usage: AccountMetric::Unsupported,
            quota_windows: AccountMetric::Unsupported,
            subscription: AccountMetric::Unsupported,
        }
    }
}

#[async_trait]
pub trait AccountUsageSource: Send + Sync + 'static {
    /// A source backed by one signed-in session must be registered for a
    /// specific profile when the provider has multiple configured profiles.
    fn requires_profile_binding(&self, _query: &AccountQuery) -> bool {
        false
    }

    async fn fetch(
        &self,
        profile: &ProviderProfile,
        query: &AccountQuery,
        range: (u64, u64),
        fetched_at_unix: u64,
        http: &dyn Transport,
    ) -> AccountSnapshot;
}

pub(crate) fn builtin_sources(
) -> BTreeMap<(String, AccountIdentity), std::sync::Arc<dyn AccountUsageSource>> {
    let mut sources: BTreeMap<(String, AccountIdentity), std::sync::Arc<dyn AccountUsageSource>> =
        BTreeMap::new();
    public::register(&mut sources);
    admin::register(&mut sources);
    qwen::register(&mut sources);
    minimax::register(&mut sources);
    sources
}

pub(super) async fn get_json(
    http: &dyn Transport,
    url: String,
    headers: Vec<(String, String)>,
) -> Result<Value, AccountFailure> {
    let response = http
        .execute_no_follow_bounded(
            HttpRequest {
                method: "GET".into(),
                url,
                headers,
                body: Bytes::new(),
                timeout: Some(ACCOUNT_REQUEST_TIMEOUT),
            },
            MAX_ACCOUNT_BODY,
        )
        .await
        .map_err(|_| AccountFailure::Transport)?;
    match response.status {
        200..=299 => {}
        401 => return Err(AccountFailure::Unauthorized),
        403 => return Err(AccountFailure::PermissionDenied),
        429 => return Err(AccountFailure::RateLimited),
        _ => return Err(AccountFailure::ProviderError),
    }
    if response.body.len() > MAX_ACCOUNT_BODY {
        return Err(AccountFailure::InvalidResponse);
    }
    serde_json::from_slice(&response.body).map_err(|_| AccountFailure::InvalidResponse)
}

pub(super) fn header_bearer(key: &Secret<String>) -> Vec<(String, String)> {
    vec![(
        "authorization".into(),
        format!("Bearer {}", key.expose_secret()),
    )]
}

pub(super) fn field_from_result<T>(
    result: Result<T, AccountFailure>,
    scope: AccountScope,
    source: &str,
) -> AccountMetric<T> {
    match result {
        Ok(value) => AccountMetric::available(scope, source, value),
        Err(reason) => AccountMetric::Failed { reason },
    }
}

/// Parse an ISO calendar day or RFC3339 timestamp to UTC Unix seconds without
/// introducing a date-time dependency for account metadata.
pub(super) fn parse_iso_utc(value: &str) -> Option<u64> {
    let date = value.get(..10)?;
    let bytes = date.as_bytes();
    if bytes.get(4) != Some(&b'-') || bytes.get(7) != Some(&b'-') {
        return None;
    }
    let year = date.get(..4)?.parse::<i64>().ok()?;
    let month = date.get(5..7)?.parse::<u32>().ok()?;
    let day = date.get(8..10)?.parse::<u32>().ok()?;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let month_days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return None,
    };
    if day == 0 || day > month_days {
        return None;
    }
    let year_adj = year - i64::from(month <= 2);
    let era = year_adj.div_euclid(400);
    let yoe = year_adj - era * 400;
    let month_adj = i64::from(month) + if month > 2 { -3 } else { 9 };
    let doy = (153 * month_adj + 2) / 5 + i64::from(day) - 1;
    let days = era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468;
    let mut seconds = days.checked_mul(86_400)?;
    if value.len() == 10 {
        return seconds.try_into().ok();
    }
    if value.as_bytes().get(10) != Some(&b'T') {
        return None;
    }
    let hour = value.get(11..13)?.parse::<i64>().ok()?;
    let minute = value.get(14..16)?.parse::<i64>().ok()?;
    let second = value.get(17..19)?.parse::<i64>().ok()?;
    if value.as_bytes().get(13) != Some(&b':')
        || value.as_bytes().get(16) != Some(&b':')
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    seconds = seconds.checked_add(hour * 3600 + minute * 60 + second)?;
    let mut suffix = value.get(19..)?;
    if suffix.starts_with('.') {
        let digits = suffix.as_bytes()[1..]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        if digits == 0 {
            return None;
        }
        suffix = suffix.get(1 + digits..)?;
    }
    let offset = if suffix == "Z" {
        0
    } else {
        let sign = match suffix.as_bytes().first()? {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if suffix.len() != 6 || suffix.as_bytes().get(3) != Some(&b':') {
            return None;
        }
        let offset_hours = suffix.get(1..3)?.parse::<i64>().ok()?;
        let offset_minutes = suffix.get(4..6)?.parse::<i64>().ok()?;
        if offset_hours > 23 || offset_minutes > 59 {
            return None;
        }
        sign * (offset_hours * 3600 + offset_minutes * 60)
    };
    seconds.checked_sub(offset)?.try_into().ok()
}

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
