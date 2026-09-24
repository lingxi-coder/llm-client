use super::*;
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
        context: &crate::account::AccountFetchContext<'_>,
        report: &mut crate::account::AccountReport,
    ) -> Result<(), AccountFailure> {
        let range = context.range;
        let _now = context.fetched_at_unix;
        let _http = context.http();
        let snapshot = report;
        snapshot.unsupported();
        snapshot.balance = None;
        snapshot.token_usage = None;
        snapshot.quota_windows = None;
        snapshot.subscription = None;
        let scope = user_scope(None);
        let account = context
            .rpc(
                self.rpc.as_ref(),
                "account/read",
                &json!({"refreshToken": false}),
            )
            .await;
        if let Some(name) = account.as_ref().ok().and_then(codex_plan) {
            snapshot.subscription = Some(codex_subscription(
                name,
                "codex account/read",
                scope.clone(),
            ));
        }
        let limits = context
            .rpc(self.rpc.as_ref(), "account/rateLimits/read", &json!({}))
            .await;

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
        snapshot.subscription = Some(match plan {
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
        });

        snapshot.balance = Some(match &limits {
            Ok(value) => codex_credits(value).map_or(AccountMetric::NotReported, |balance| {
                AccountMetric::available(
                    scope.clone(),
                    "codex account/rateLimits/read",
                    vec![balance],
                )
            }),
            Err(reason) => AccountMetric::Failed { reason: *reason },
        });
        snapshot.quota_windows = Some(field_from_result(
            limits.and_then(|value| codex_windows(&value)),
            scope.clone(),
            "codex account/rateLimits/read",
        ));
        let usage = context
            .rpc(self.rpc.as_ref(), "account/usage/read", &json!({}))
            .await;
        snapshot.token_usage = Some(field_from_result(
            usage.and_then(|value| codex_usage(&value, range)),
            scope,
            "codex account/usage/read",
        ));
        Ok(())
    }
}

fn codex_subscription(
    name: &str,
    source: &str,
    scope: AccountScope,
) -> AccountMetric<AccountSubscription> {
    AccountMetric::available(
        scope,
        source,
        AccountSubscription {
            plan_name: Some(name.into()),
            status: match name {
                "free" => SubscriptionStatus::VerifiedInactive,
                "unknown" => SubscriptionStatus::Unknown,
                _ => SubscriptionStatus::VerifiedActive,
            },
        },
    )
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
