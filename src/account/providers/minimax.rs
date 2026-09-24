//! MiniMax Token Plan quota windows.

use super::{
    get_json, header_bearer, AccountFailure, AccountIdentity, AccountMetric, AccountQuotaWindow,
    AccountScope, AccountScopeKind, AccountUsageSource,
};
use crate::protocol::ProviderProfile;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

pub(super) fn register(
    sources: &mut BTreeMap<(String, AccountIdentity), Arc<dyn AccountUsageSource>>,
) {
    sources.insert(
        ("minimax".into(), AccountIdentity::ApiKey),
        Arc::new(MiniMax),
    );
}

struct MiniMax;

#[async_trait]
impl AccountUsageSource for MiniMax {
    async fn fetch(
        &self,
        context: &super::AccountFetchContext<'_>,
        report: &mut super::AccountReport,
    ) -> Result<(), AccountFailure> {
        let profile = context.profile;
        let query = context.query;
        let now = context.fetched_at_unix;
        let http = context.http();
        let snapshot = report;
        snapshot.unsupported();
        let Some(host) = official_host(profile) else {
            return Ok(());
        };
        snapshot.quota_windows = None;
        let source = format!("https://{host}/v1/token_plan/remains");
        snapshot.quota_windows = Some(match &query.credential {
            Some(key) => {
                let result = get_json(http, source.clone(), header_bearer(key))
                    .await
                    .and_then(|body| parse_quota_windows(body, now));
                match result {
                    Ok(windows) => AccountMetric::available(
                        AccountScope::new(AccountScopeKind::ApiKey, None),
                        &source,
                        windows,
                    ),
                    Err(reason) => AccountMetric::Failed { reason },
                }
            }
            None => AccountMetric::CredentialRequired,
        });
        Ok(())
    }
}

fn official_host(profile: &ProviderProfile) -> Option<&'static str> {
    if profile.provider_id.as_str() != "minimax" {
        return None;
    }
    let host = url::Url::parse(&profile.base_url)
        .ok()?
        .host_str()?
        .to_owned();
    match host.as_str() {
        "api.minimaxi.com" => Some("api.minimaxi.com"),
        "api.minimax.io" => Some("api.minimax.io"),
        _ => None,
    }
}

fn parse_quota_windows(body: Value, now: u64) -> Result<Vec<AccountQuotaWindow>, AccountFailure> {
    let base_resp = body
        .get("base_resp")
        .ok_or(AccountFailure::InvalidResponse)?;
    let code = base_resp
        .get("status_code")
        .and_then(as_u64)
        .ok_or(AccountFailure::InvalidResponse)?;
    if code != 0 {
        return Err(if matches!(code, 1004 | 1005) {
            AccountFailure::Unauthorized
        } else {
            AccountFailure::ProviderError
        });
    }
    let rows = body
        .get("model_remains")
        .or_else(|| body.pointer("/data/model_remains"))
        .and_then(Value::as_array);
    let mut windows = Vec::new();
    if let Some(rows) = rows {
        for row in rows {
            let model = row
                .get("model_name")
                .or_else(|| row.get("model"))
                .and_then(Value::as_str)
                .unwrap_or("MiniMax");
            if let Some(window) = quota_window(row, "current_interval", "5_hour", 300, now) {
                windows.push(named(window, model, "5_hour"));
            }
            if let Some(window) = quota_window(row, "current_weekly", "weekly", 10_080, now) {
                windows.push(named(window, model, "weekly"));
            }
        }
    } else {
        let row = body.get("data").unwrap_or(&body);
        if let Some(window) = quota_window(row, "current_interval", "5_hour", 300, now) {
            windows.push(named(window, "MiniMax", "5_hour"));
        }
        if let Some(window) = quota_window(row, "current_weekly", "weekly", 10_080, now) {
            windows.push(named(window, "MiniMax", "weekly"));
        }
    }
    Ok(windows)
}

fn quota_window(
    value: &Value,
    prefix: &str,
    label: &str,
    default_duration_mins: u64,
    now: u64,
) -> Option<AccountQuotaWindow> {
    let total = value.get(format!("{prefix}_total_count")).and_then(as_u64);
    // `*_usage_count` is not used: its remaining-vs-consumed semantics are
    // disputed by MiniMax's own client users. Explicit remaining fields and
    // remaining percentages are safe to report.
    let remaining = value
        .get(format!("{prefix}_remaining_count"))
        .or_else(|| value.get(format!("{prefix}_remains_count")))
        .and_then(as_u64);
    let reported_remaining_percent = value
        .get(format!("{prefix}_remaining_percent"))
        .and_then(as_f64)
        .filter(|percent| percent.is_finite() && (0.0..=100.0).contains(percent));
    let computed_remaining_percent = remaining.zip(total).and_then(|(remaining, total)| {
        (total > 0 && remaining <= total).then(|| 100.0 * remaining as f64 / total as f64)
    });
    let remaining_percent = reported_remaining_percent.or(computed_remaining_percent);
    if total.is_none() && remaining.is_none() && remaining_percent.is_none() {
        return None;
    }
    let (start_key, end_key, remaining_key) = if prefix == "current_weekly" {
        (
            "weekly_start_time",
            "weekly_end_time",
            "weekly_remains_time",
        )
    } else {
        ("start_time", "end_time", "remains_time")
    };
    let start = value
        .get(start_key)
        .and_then(as_u64)
        .map(milliseconds_to_seconds);
    let end = value
        .get(end_key)
        .and_then(as_u64)
        .map(milliseconds_to_seconds);
    let duration_mins = start
        .zip(end)
        .and_then(|(start, end)| {
            let seconds = end.checked_sub(start)?;
            (seconds % 60 == 0).then_some(seconds / 60)
        })
        .or(Some(default_duration_mins));
    let reset = end.or_else(|| {
        value
            .get(remaining_key)
            .and_then(as_u64)
            .map(|millis| now.saturating_add(millis / 1_000))
    });
    Some(AccountQuotaWindow {
        name: label.to_owned(),
        duration_mins,
        used_percent: remaining_percent.map(|remaining| 100.0 - remaining),
        remaining_percent,
        resets_at_unix: reset,
        limit: total,
        used: total
            .zip(remaining)
            .and_then(|(total, remaining)| (remaining <= total).then(|| total - remaining)),
        remaining,
        limit_decimal: None,
        remaining_decimal: None,
        unit: Some("tokens".into()),
    })
}

fn named(mut window: AccountQuotaWindow, model: &str, period: &str) -> AccountQuotaWindow {
    window.name = format!("{model}:{period}");
    window
}

fn as_u64(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| value.as_str()?.parse().ok())
}

fn as_f64(value: &Value) -> Option<f64> {
    value.as_f64().or_else(|| value.as_str()?.parse().ok())
}

fn milliseconds_to_seconds(value: u64) -> u64 {
    if value >= 1_000_000_000_000 {
        value / 1_000
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn maps_explicit_remaining_aliases_and_does_not_guess_usage_count() {
        let windows = parse_quota_windows(
            json!({
                "base_resp":{"status_code":0},
                "model_remains":[{
                    "model_name":"MiniMax-M3",
                    "start_time":1782381600000_u64,
                    "end_time":1782399600000_u64,
                    "current_interval_total_count":100,
                    "current_interval_usage_count":80,
                    "current_interval_remaining_percent":20,
                    "current_weekly_total_count":500,
                    "current_weekly_remaining_count":400
                }]
            }),
            1_782_381_600,
        )
        .unwrap();
        assert_eq!(windows.len(), 2);
        let interval = &windows[0];
        assert_eq!(interval.limit, Some(100));
        assert_eq!(interval.used, None);
        assert_eq!(interval.remaining, None);
        assert_eq!(interval.remaining_percent, Some(20.0));
        assert_eq!(interval.resets_at_unix, Some(1_782_399_600));
        let weekly = &windows[1];
        assert_eq!(weekly.limit, Some(500));
        assert_eq!(weekly.remaining, Some(400));
        assert_eq!(weekly.used, Some(100));
    }

    #[test]
    fn malformed_percentages_and_inconsistent_counts_are_not_clamped_into_valid_quotas() {
        let now = 1_782_381_600;
        let over_limit = quota_window(
            &json!({
                "current_interval_total_count": 100,
                "current_interval_remaining_percent": 150
            }),
            "current_interval",
            "5_hour",
            300,
            now,
        )
        .unwrap();
        assert_eq!(over_limit.remaining_percent, None);
        assert_eq!(over_limit.used_percent, None);

        let inconsistent = quota_window(
            &json!({
                "current_weekly_total_count": 0,
                "current_weekly_remaining_count": 10
            }),
            "current_weekly",
            "weekly",
            10_080,
            now,
        )
        .unwrap();
        assert_eq!(inconsistent.remaining, Some(10));
        assert_eq!(inconsistent.used, None);
        assert_eq!(inconsistent.remaining_percent, None);

        let fallback = quota_window(
            &json!({
                "current_interval_total_count": 100,
                "current_interval_remaining_count": 40,
                "current_interval_remaining_percent": "NaN"
            }),
            "current_interval",
            "5_hour",
            300,
            now,
        )
        .unwrap();
        assert_eq!(fallback.remaining_percent, Some(40.0));
        assert_eq!(fallback.used_percent, Some(60.0));
    }
    #[test]
    fn weekly_window_uses_its_own_clock_fields() {
        let mut value = json!({"current_interval_total_count":100,"current_weekly_total_count":500,
            "start_time":1782381600000_u64,"end_time":1782399600000_u64,
            "weekly_start_time":1782086400000_u64,"weekly_end_time":1782691200000_u64,
            "remains_time":60000,"weekly_remains_time":120000});
        let hourly = quota_window(&value, "current_interval", "5_hour", 300, 100).unwrap();
        let weekly = quota_window(&value, "current_weekly", "weekly", 10080, 100).unwrap();
        assert_eq!(hourly.duration_mins, Some(300));
        assert_eq!(weekly.duration_mins, Some(10080));
        assert_eq!(weekly.resets_at_unix, Some(1782691200));
        value.as_object_mut().unwrap().remove("weekly_end_time");
        assert_eq!(
            quota_window(&value, "current_weekly", "weekly", 10080, 100)
                .unwrap()
                .resets_at_unix,
            Some(220)
        );
        value.as_object_mut().unwrap().remove("weekly_remains_time");
        assert_eq!(
            quota_window(&value, "current_weekly", "weekly", 10080, 100)
                .unwrap()
                .resets_at_unix,
            None
        );
    }
}
