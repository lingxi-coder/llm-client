//! Account data supplied by an authenticated local host or official loopback service.

use super::{
    field_from_result, get_json, header_bearer, parse_iso_utc, AccountBalance, AccountFailure,
    AccountMetric, AccountQuery, AccountQuotaWindow, AccountScope, AccountScopeKind,
    AccountSnapshot, AccountSubscription, AccountTokenBucket, AccountTokenUsage,
    AccountUsageSource, SubscriptionStatus,
};
use crate::transport::Transport;
use async_trait::async_trait;
use lingxi_agent_api::protocol::ProviderProfile;
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

pub struct CodexAccountSource {
    rpc: Arc<dyn AccountRpc>,
}

impl CodexAccountSource {
    pub fn new(rpc: Arc<dyn AccountRpc>) -> Self {
        Self { rpc }
    }
}

#[async_trait]
impl AccountUsageSource for CodexAccountSource {
    fn requires_profile_binding(&self, _query: &AccountQuery) -> bool {
        true
    }

    async fn fetch(
        &self,
        profile: &ProviderProfile,
        query: &AccountQuery,
        range: (u64, u64),
        fetched_at_unix: u64,
        _http: &dyn Transport,
    ) -> AccountSnapshot {
        let mut snapshot = AccountSnapshot::unsupported(profile, query.identity, fetched_at_unix);
        let scope = user_scope(None);
        let account = self
            .rpc
            .call("account/read", &json!({"refreshToken": false}))
            .await;
        let limits = self.rpc.call("account/rateLimits/read", &json!({})).await;
        let usage = self.rpc.call("account/usage/read", &json!({})).await;

        let plan = account
            .as_ref()
            .ok()
            .and_then(codex_plan)
            .map(|name| (name, "codex account/read"))
            .or_else(|| {
                limits
                    .as_ref()
                    .ok()
                    .and_then(codex_plan)
                    .map(|name| (name, "codex account/rateLimits/read"))
            });
        snapshot.subscription = match plan {
            Some((name, source)) => AccountMetric::available(
                scope.clone(),
                source,
                AccountSubscription {
                    plan_name: Some(name.to_owned()),
                    status: match name {
                        "free" => SubscriptionStatus::VerifiedInactive,
                        "unknown" => SubscriptionStatus::Unknown,
                        _ => SubscriptionStatus::VerifiedActive,
                    },
                },
            ),
            None if account.is_ok() || limits.is_ok() => AccountMetric::NotReported,
            None => AccountMetric::Failed {
                reason: *account.as_ref().unwrap_err(),
            },
        };

        snapshot.balance = match &limits {
            Ok(value) => codex_credits(value).map_or(AccountMetric::NotReported, |balance| {
                AccountMetric::available(
                    scope.clone(),
                    "codex account/rateLimits/read",
                    vec![balance],
                )
            }),
            Err(reason) => AccountMetric::Failed { reason: *reason },
        };
        snapshot.quota_windows = field_from_result(
            limits.and_then(|value| codex_windows(&value)),
            scope.clone(),
            "codex account/rateLimits/read",
        );
        snapshot.token_usage = field_from_result(
            usage.and_then(|value| codex_usage(&value, range)),
            scope,
            "codex account/usage/read",
        );
        snapshot
    }
}

fn codex_plan(value: &Value) -> Option<&str> {
    value
        .pointer("/account/planType")
        .or_else(|| value.pointer("/rateLimits/planType"))
        .and_then(Value::as_str)
        .filter(|plan| !plan.is_empty())
}

fn codex_credits(value: &Value) -> Option<AccountBalance> {
    let credits = value.pointer("/rateLimits/credits").or_else(|| {
        value
            .get("rateLimitsByLimitId")?
            .as_object()?
            .values()
            .find_map(|group| group.get("credits").filter(|credits| !credits.is_null()))
    })?;
    let balance = credits.get("balance")?.as_str()?;
    Some(AccountBalance {
        unit: "credits".into(),
        remaining: balance.to_owned(),
        total: None,
        is_available: None,
    })
}

fn codex_windows(value: &Value) -> Result<Vec<AccountQuotaWindow>, AccountFailure> {
    let mut windows = Vec::new();
    if let Some(groups) = value.get("rateLimitsByLimitId").and_then(Value::as_object) {
        for (name, group) in groups {
            push_codex_windows(&mut windows, name, group)?;
        }
    }
    if windows.is_empty() {
        let group = value
            .get("rateLimits")
            .ok_or(AccountFailure::InvalidResponse)?;
        push_codex_windows(&mut windows, "codex", group)?;
    }
    Ok(windows)
}

fn push_codex_windows(
    windows: &mut Vec<AccountQuotaWindow>,
    name: &str,
    group: &Value,
) -> Result<(), AccountFailure> {
    for part in ["primary", "secondary"] {
        let Some(window) = group.get(part).filter(|value| !value.is_null()) else {
            continue;
        };
        let used = window
            .get("usedPercent")
            .and_then(Value::as_f64)
            .filter(|value| (0.0..=100.0).contains(value))
            .ok_or(AccountFailure::InvalidResponse)?;
        windows.push(AccountQuotaWindow {
            name: format!("{name}/{part}"),
            duration_mins: window.get("windowDurationMins").and_then(Value::as_u64),
            used_percent: Some(used),
            remaining_percent: Some(100.0 - used),
            resets_at_unix: optional_unix(window.get("resetsAt"))?,
            limit: None,
            used: None,
            remaining: None,
            limit_decimal: None,
            remaining_decimal: None,
            unit: None,
        });
    }
    Ok(())
}

fn codex_usage(value: &Value, range: (u64, u64)) -> Result<AccountTokenUsage, AccountFailure> {
    let summary = value
        .get("summary")
        .ok_or(AccountFailure::InvalidResponse)?;
    let lifetime_tokens = summary.get("lifetimeTokens").and_then(Value::as_u64);
    let mut buckets = Vec::new();
    if let Some(daily) = value.get("dailyUsageBuckets").and_then(Value::as_array) {
        for day in daily {
            let start = day
                .get("startDate")
                .and_then(Value::as_str)
                .and_then(parse_iso_utc)
                .ok_or(AccountFailure::InvalidResponse)?;
            let tokens = day
                .get("tokens")
                .and_then(Value::as_u64)
                .ok_or(AccountFailure::InvalidResponse)?;
            let end = start
                .checked_add(86_400)
                .ok_or(AccountFailure::InvalidResponse)?;
            // The service reports whole UTC days. Include any day that
            // overlaps the requested interval; do not imply hourly precision.
            if end <= range.0 || start >= range.1 {
                continue;
            }
            buckets.push(AccountTokenBucket {
                start_unix: start,
                end_unix: end,
                model: None,
                input_tokens: None,
                output_tokens: None,
                cached_input_tokens: None,
                cache_write_tokens: None,
                total_tokens: Some(tokens),
            });
        }
    }
    Ok(AccountTokenUsage {
        lifetime_tokens,
        buckets,
    })
}

pub struct CopilotAccountSource {
    rpc: Arc<dyn AccountRpc>,
}

impl CopilotAccountSource {
    pub fn new(rpc: Arc<dyn AccountRpc>) -> Self {
        Self { rpc }
    }
}

#[async_trait]
impl AccountUsageSource for CopilotAccountSource {
    fn requires_profile_binding(&self, query: &AccountQuery) -> bool {
        query.credential.is_none()
    }

    async fn fetch(
        &self,
        profile: &ProviderProfile,
        query: &AccountQuery,
        _range: (u64, u64),
        fetched_at_unix: u64,
        _http: &dyn Transport,
    ) -> AccountSnapshot {
        let mut snapshot = AccountSnapshot::unsupported(profile, query.identity, fetched_at_unix);
        let params = query.credential.as_ref().map_or_else(
            || json!({}),
            |token| json!({"gitHubToken": token.expose_secret()}),
        );
        let quota = self
            .rpc
            .call("account.getQuota", &params)
            .await
            .and_then(|value| copilot_windows(&value));
        // Quota exists for free-tier users too; this RPC does not attest a paid plan.
        snapshot.subscription = AccountMetric::NotReported;
        snapshot.quota_windows =
            field_from_result(quota, user_scope(None), "copilot account.getQuota");
        snapshot
    }
}

fn copilot_windows(value: &Value) -> Result<Vec<AccountQuotaWindow>, AccountFailure> {
    let snapshots = value
        .get("quotaSnapshots")
        .and_then(Value::as_object)
        .ok_or(AccountFailure::InvalidResponse)?;
    snapshots
        .iter()
        .map(|(name, quota)| {
            let raw_limit = quota
                .get("entitlementRequests")
                .and_then(Value::as_i64)
                .ok_or(AccountFailure::InvalidResponse)?;
            let limit = match raw_limit {
                -1 => None, // Documented unlimited sentinel.
                0.. => Some(raw_limit as u64),
                _ => return Err(AccountFailure::InvalidResponse),
            };
            let used = quota.get("usedRequests").and_then(Value::as_u64);
            let remaining_percent = quota
                .get("remainingPercentage")
                .and_then(Value::as_f64)
                .filter(|number| (0.0..=100.0).contains(number));
            let resets_at_unix = optional_time(quota.get("resetDate"))?;
            Ok(AccountQuotaWindow {
                name: name.clone(),
                duration_mins: None,
                used_percent: remaining_percent.map(|value| 100.0 - value),
                remaining_percent,
                resets_at_unix,
                limit,
                used,
                remaining: limit
                    .zip(used)
                    .map(|(limit, used)| limit.saturating_sub(used)),
                limit_decimal: None,
                remaining_decimal: None,
                unit: Some("requests".into()),
            })
        })
        .collect()
}

#[derive(Default)]
pub struct KimiCodeAccountSource;

impl KimiCodeAccountSource {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AccountUsageSource for KimiCodeAccountSource {
    async fn fetch(
        &self,
        profile: &ProviderProfile,
        query: &AccountQuery,
        _range: (u64, u64),
        fetched_at_unix: u64,
        http: &dyn Transport,
    ) -> AccountSnapshot {
        let mut snapshot = AccountSnapshot::unsupported(profile, query.identity, fetched_at_unix);
        let Some(base) = query.service_url.as_deref() else {
            snapshot.balance = AccountMetric::NotReported;
            snapshot.quota_windows = AccountMetric::NotReported;
            snapshot.subscription = AccountMetric::NotReported;
            return snapshot;
        };
        let Some(credential) = query.service_credential.as_ref() else {
            snapshot.balance = AccountMetric::CredentialRequired;
            snapshot.quota_windows = AccountMetric::CredentialRequired;
            snapshot.subscription = AccountMetric::CredentialRequired;
            return snapshot;
        };
        let (usage_url, user_url) = match kimi_urls(base) {
            Ok(urls) => urls,
            Err(reason) => {
                snapshot.balance = AccountMetric::Failed { reason };
                snapshot.quota_windows = AccountMetric::Failed { reason };
                snapshot.subscription = AccountMetric::Failed { reason };
                return snapshot;
            }
        };
        let headers = header_bearer(credential);
        let usage = get_json(http, usage_url, headers.clone())
            .await
            .and_then(kimi_data);
        let user = get_json(http, user_url, headers).await.and_then(kimi_data);
        let scope = user_scope(
            user.as_ref()
                .ok()
                .and_then(|value| value.pointer("/userInfo/userId"))
                .and_then(Value::as_str)
                .map(str::to_owned),
        );
        snapshot.balance = match &usage {
            Ok(value) => match kimi_balance(value) {
                Some(balance) => AccountMetric::available(
                    scope.clone(),
                    "kimi-code /api/v1/oauth/usage",
                    vec![balance],
                ),
                None => AccountMetric::NotReported,
            },
            Err(reason) => AccountMetric::Failed { reason: *reason },
        };
        snapshot.quota_windows = field_from_result(
            usage.and_then(|value| kimi_windows(&value)),
            scope.clone(),
            "kimi-code /api/v1/oauth/usage",
        );
        snapshot.subscription = match user {
            Ok(value) => kimi_subscription(&value).map_or(AccountMetric::NotReported, |plan| {
                AccountMetric::available(scope, "kimi-code /api/v1/oauth/userinfo", plan)
            }),
            Err(reason) => AccountMetric::Failed { reason },
        };
        snapshot
    }
}

fn kimi_urls(base: &str) -> Result<(String, String), AccountFailure> {
    let url = reqwest::Url::parse(base).map_err(|_| AccountFailure::InvalidResponse)?;
    let host = url.host_str().ok_or(AccountFailure::InvalidResponse)?;
    let is_loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .trim_matches(['[', ']'])
            .parse::<IpAddr>()
            .map(|address| address.is_loopback())
            .unwrap_or(false);
    if url.scheme() != "http"
        || !is_loopback
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != "/"
    {
        return Err(AccountFailure::InvalidResponse);
    }
    Ok((
        url.join("/api/v1/oauth/usage")
            .map_err(|_| AccountFailure::InvalidResponse)?
            .to_string(),
        url.join("/api/v1/oauth/userinfo")
            .map_err(|_| AccountFailure::InvalidResponse)?
            .to_string(),
    ))
}

fn kimi_data(value: Value) -> Result<Value, AccountFailure> {
    if value.get("code").and_then(Value::as_i64) != Some(0) {
        return Err(AccountFailure::ProviderError);
    }
    let data = value.get("data").ok_or(AccountFailure::InvalidResponse)?;
    match data.get("kind").and_then(Value::as_str) {
        Some("ok") => Ok(data.clone()),
        Some("error") => Err(match data.get("status").and_then(Value::as_u64) {
            Some(401) => AccountFailure::Unauthorized,
            Some(403) => AccountFailure::PermissionDenied,
            Some(429) => AccountFailure::RateLimited,
            _ => AccountFailure::ProviderError,
        }),
        _ => Err(AccountFailure::InvalidResponse),
    }
}

fn kimi_windows(value: &Value) -> Result<Vec<AccountQuotaWindow>, AccountFailure> {
    let usages = value
        .pointer("/quota/usages")
        .and_then(Value::as_object)
        .ok_or(AccountFailure::InvalidResponse)?;
    usages
        .iter()
        .map(|(name, quota)| {
            let ratio = quota
                .get("usedRatio")
                .and_then(Value::as_f64)
                .filter(|ratio| (0.0..=1.0).contains(ratio))
                .ok_or(AccountFailure::InvalidResponse)?;
            Ok(AccountQuotaWindow {
                name: name.clone(),
                duration_mins: match name.as_str() {
                    "limit5h" => Some(300),
                    "limit7d" => Some(10_080),
                    _ => None,
                },
                used_percent: Some(ratio * 100.0),
                remaining_percent: Some((1.0 - ratio) * 100.0),
                resets_at_unix: optional_time(quota.get("resetAt"))?,
                limit: None,
                used: None,
                remaining: None,
                limit_decimal: None,
                remaining_decimal: None,
                unit: None,
            })
        })
        .collect()
}

fn kimi_balance(value: &Value) -> Option<AccountBalance> {
    let wallet = value.pointer("/quota/extraUsage")?.as_object()?;
    let remaining = wallet.get("balanceCents").and_then(Value::as_i64)?;
    let total = wallet
        .get("totalCents")
        .and_then(Value::as_i64)
        .map(cents_text);
    Some(AccountBalance {
        unit: wallet.get("currency")?.as_str()?.to_owned(),
        remaining: cents_text(remaining),
        total,
        is_available: None,
    })
}

fn cents_text(cents: i64) -> String {
    let negative = if cents < 0 { "-" } else { "" };
    let absolute = cents.unsigned_abs();
    format!("{negative}{}.{:02}", absolute / 100, absolute % 100)
}

fn kimi_subscription(value: &Value) -> Option<AccountSubscription> {
    let info = value.get("userInfo")?;
    let plan_name = info
        .get("userLevelName")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())?
        .to_owned();
    // The documented userInfo status is an account field, not a documented
    // subscription-state contract. A plan name alone cannot prove payment.
    Some(AccountSubscription {
        plan_name: Some(plan_name),
        status: SubscriptionStatus::Unknown,
    })
}

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
