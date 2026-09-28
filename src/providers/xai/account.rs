//! Provider-owned organization account reporting.
use crate::account::support::*;
use crate::account::*;
use crate::transport::Transport;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use url::Url;

pub(crate) struct AccountSource;

pub(crate) fn register(
    sources: &mut BTreeMap<(String, AccountIdentity), Arc<dyn AccountUsageSource>>,
) {
    sources.insert(
        ("xai".into(), AccountIdentity::ApiKey),
        Arc::new(AccountSource),
    );
}

#[async_trait]
impl AccountUsageSource for AccountSource {
    async fn fetch(
        &self,
        context: &AccountFetchContext<'_>,
        snapshot: &mut AccountReport,
    ) -> Result<(), AccountFailure> {
        let query = context.query;
        let range = context.range;
        let http = context.http();
        snapshot.unsupported();
        snapshot.balance = None;
        snapshot.cost_usage = None;
        let Some(key) = &query.management_credential else {
            snapshot.balance = Some(AccountMetric::CredentialRequired);
            snapshot.cost_usage = Some(AccountMetric::CredentialRequired);
            return Ok(());
        };

        if let Some(team_id) = query.selector.team_id.as_deref().filter(|s| !s.is_empty()) {
            let scope = AccountScope::new(AccountScopeKind::Team, Some(team_id.into()));
            snapshot.balance = Some(field_from_result(
                xai_balance(team_id, key.expose_secret(), http).await,
                scope.clone(),
                "https://management-api.x.ai/v1/billing/teams/{team_id}/prepaid/balance",
            ));
            snapshot.cost_usage = Some(field_from_result(
                xai_costs(team_id, range, key.expose_secret(), http).await,
                scope,
                "https://management-api.x.ai/v1/billing/teams/{team_id}/usage",
            ));
        } else {
            snapshot.balance = Some(AccountMetric::NotReported);
            snapshot.cost_usage = Some(AccountMetric::NotReported);
        }

        Ok(())
    }
}

fn xai_endpoint(team_id: &str, tail: &[&str]) -> Result<String, AccountFailure> {
    let mut endpoint = Url::parse("https://management-api.x.ai/v1/billing/teams/")
        .map_err(|_| AccountFailure::InvalidResponse)?;
    let mut path = endpoint
        .path_segments_mut()
        .map_err(|_| AccountFailure::InvalidResponse)?;
    path.pop_if_empty().push(team_id);
    for segment in tail {
        path.push(segment);
    }
    drop(path);
    Ok(endpoint.into())
}

async fn xai_costs(
    team_id: &str,
    range: (u64, u64),
    key: &str,
    http: &dyn Transport,
) -> Result<AccountCostUsage, AccountFailure> {
    // xAI's timeRange end is inclusive, whereas AccountQuery.until_unix is exclusive.
    let start = format_rfc3339(range.0)?
        .replace('T', " ")
        .trim_end_matches('Z')
        .to_owned();
    let end = format_rfc3339(range.1 - 1)?
        .replace('T', " ")
        .trim_end_matches('Z')
        .to_owned();
    let request = serde_json::json!({"analyticsRequest": {
        "timeRange": {"startTime": start, "endTime": end, "timezone": "Etc/GMT"},
        "timeUnit": "TIME_UNIT_DAY",
        "values": [{"name": "usd", "aggregation": "AGGREGATION_SUM"}],
        "groupBy": ["description"], "filters": []
    }});
    let body = post_json(http, xai_endpoint(team_id, &["usage"])?, key, &request).await?;
    if body.get("limitReached").and_then(Value::as_bool) == Some(true) {
        return Err(AccountFailure::ProviderError);
    }
    let series = match body.get("timeSeries") {
        None => return Ok(AccountCostUsage::default()),
        Some(Value::Array(series)) => series,
        Some(_) => return Err(AccountFailure::InvalidResponse),
    };
    let mut buckets = Vec::new();
    for item in series {
        let points = match item.get("dataPoints") {
            None => continue,
            Some(Value::Array(points)) => points,
            Some(_) => return Err(AccountFailure::InvalidResponse),
        };
        for point in points {
            let start = parse_rfc3339(
                point
                    .get("timestamp")
                    .and_then(Value::as_str)
                    .ok_or(AccountFailure::InvalidResponse)?,
            )?;
            let end = start
                .checked_add(86_400)
                .ok_or(AccountFailure::InvalidResponse)?;
            let amount = point
                .pointer("/values/0")
                .and_then(Value::as_number)
                .ok_or(AccountFailure::InvalidResponse)?;
            buckets.push(AccountCostBucket {
                start_unix: start,
                end_unix: end,
                currency: "USD".into(),
                amount: amount.to_string(),
                model: None,
            });
        }
    }
    Ok(AccountCostUsage { buckets })
}

async fn xai_balance(
    team_id: &str,
    key: &str,
    http: &dyn Transport,
) -> Result<Vec<AccountBalance>, AccountFailure> {
    let body = get_json(
        http,
        xai_endpoint(team_id, &["prepaid", "balance"])?,
        vec![("authorization".into(), format!("Bearer {key}"))],
    )
    .await?;
    let cents = body
        .pointer("/total/val")
        .and_then(Value::as_str)
        .ok_or(AccountFailure::InvalidResponse)?
        .parse::<i128>()
        .map_err(|_| AccountFailure::InvalidResponse)?;
    // xAI's prepaid ledger records purchased credit as a negative liability.
    // Negating that signed ledger total yields the remaining usable credit.
    let available = cents.checked_neg().ok_or(AccountFailure::InvalidResponse)?;
    let abs = available.unsigned_abs();
    let remaining = format!(
        "{}{}.{:02}",
        if available < 0 { "-" } else { "" },
        abs / 100,
        abs % 100
    );
    Ok(vec![AccountBalance {
        unit: "USD".into(),
        remaining,
        total: None,
        is_available: None,
    }])
}
