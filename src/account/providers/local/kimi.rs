use super::*;
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
        context: &crate::account::AccountFetchContext<'_>,
        report: &mut crate::account::AccountReport,
    ) -> Result<(), AccountFailure> {
        let query = context.query;
        let _now = context.fetched_at_unix;
        let http = context.http();
        let snapshot = report;
        snapshot.unsupported();
        snapshot.balance = None;
        snapshot.quota_windows = None;
        snapshot.subscription = None;
        let Some(base) = query.service_url.as_deref() else {
            snapshot.balance = Some(AccountMetric::NotReported);
            snapshot.quota_windows = Some(AccountMetric::NotReported);
            snapshot.subscription = Some(AccountMetric::NotReported);
            return Ok(());
        };
        let Some(credential) = query.service_credential.as_ref() else {
            snapshot.balance = Some(AccountMetric::CredentialRequired);
            snapshot.quota_windows = Some(AccountMetric::CredentialRequired);
            snapshot.subscription = Some(AccountMetric::CredentialRequired);
            return Ok(());
        };
        let (usage_url, user_url) = match kimi_urls(base) {
            Ok(urls) => urls,
            Err(reason) => {
                snapshot.balance = Some(AccountMetric::Failed { reason });
                snapshot.quota_windows = Some(AccountMetric::Failed { reason });
                snapshot.subscription = Some(AccountMetric::Failed { reason });
                return Ok(());
            }
        };
        let headers = header_bearer(credential);
        let usage = get_json(http, usage_url, headers.clone())
            .await
            .and_then(kimi_data);
        let scope = user_scope(None);
        snapshot.balance = Some(match &usage {
            Ok(value) => match kimi_balance(value) {
                Some(balance) => AccountMetric::available(
                    scope.clone(),
                    "kimi-code /api/v1/oauth/usage",
                    vec![balance],
                ),
                None => AccountMetric::NotReported,
            },
            Err(reason) => AccountMetric::Failed { reason: *reason },
        });
        snapshot.quota_windows = Some(field_from_result(
            usage.and_then(|value| kimi_windows(&value)),
            scope.clone(),
            "kimi-code /api/v1/oauth/usage",
        ));
        let user = get_json(http, user_url, headers).await.and_then(kimi_data);
        let scope = user_scope(
            user.as_ref()
                .ok()
                .and_then(|value| value.pointer("/userInfo/userId"))
                .and_then(Value::as_str)
                .map(str::to_owned),
        );
        if let Some(AccountMetric::Available { scope: current, .. }) = &mut snapshot.balance {
            *current = scope.clone();
        }
        if let Some(AccountMetric::Available { scope: current, .. }) = &mut snapshot.quota_windows {
            *current = scope.clone();
        }
        snapshot.subscription = Some(match user {
            Ok(value) => kimi_subscription(&value).map_or(AccountMetric::NotReported, |plan| {
                AccountMetric::available(scope, "kimi-code /api/v1/oauth/userinfo", plan)
            }),
            Err(reason) => AccountMetric::Failed { reason },
        });
        Ok(())
    }
}

fn kimi_urls(base: &str) -> Result<(String, String), AccountFailure> {
    let url = url::Url::parse(base).map_err(|_| AccountFailure::InvalidResponse)?;
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
