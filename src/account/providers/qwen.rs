//! Qwen Model Studio account quotas and Aliyun-signed billing trends.

use super::{
    field_from_result, get_json, header_bearer, parse_iso_utc, AccountCostBucket, AccountCostUsage,
    AccountFailure, AccountIdentity, AccountMetric, AccountQuotaWindow, AccountScope,
    AccountScopeKind, AccountUsageSource,
};
use crate::protocol::ProviderProfile;
use crate::transport::Transport;
use async_trait::async_trait;
use ring::{
    digest, hmac,
    rand::{SecureRandom, SystemRandom},
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

const SIGNING_VERSION: &str = "2026-02-10";
const QUOTAS_PAGE_SIZE: usize = 100;
const MAX_QUOTAS_PAGES: usize = 100;

pub(super) fn register(
    sources: &mut BTreeMap<(String, AccountIdentity), Arc<dyn AccountUsageSource>>,
) {
    sources.insert(("qwen".into(), AccountIdentity::ApiKey), Arc::new(Qwen));
}

struct Qwen;

#[async_trait]
impl AccountUsageSource for Qwen {
    async fn fetch(
        &self,
        context: &super::AccountFetchContext<'_>,
        report: &mut super::AccountReport,
    ) -> Result<(), AccountFailure> {
        let profile = context.profile;
        let query = context.query;
        let range = context.range;
        let now = context.fetched_at_unix;
        let http = context.http();
        let snapshot = report;
        snapshot.unsupported();
        let Some(region) = region_for_profile(profile) else {
            return Ok(());
        };
        snapshot.quota_windows = None;
        snapshot.cost_usage = if query.alibaba_access_key.is_none() {
            Some(AccountMetric::CredentialRequired)
        } else if query
            .selector
            .api_key_id
            .as_deref()
            .is_none_or(|id| id.trim().is_empty())
        {
            Some(AccountMetric::NotReported)
        } else {
            None
        };
        let workspace = query
            .selector
            .workspace_id
            .as_deref()
            .filter(|value| valid_workspace_id(value));
        snapshot.quota_windows = Some(match (&query.credential, workspace) {
            (None, _) => AccountMetric::CredentialRequired,
            (Some(_), None) => AccountMetric::NotReported,
            (Some(key), Some(workspace)) => {
                let url = format!(
                    "https://{workspace}.{}.maas.aliyuncs.com/api/v1/quotas",
                    region.domain
                );
                let result = fetch_quotas(http, &url, key).await;
                field_from_result(
                    result,
                    AccountScope::new(AccountScopeKind::Workspace, Some(workspace.to_owned())),
                    &url,
                )
            }
        });

        snapshot.cost_usage = Some(
            match (
                &query.alibaba_access_key,
                query
                    .selector
                    .api_key_id
                    .as_deref()
                    .filter(|id| !id.trim().is_empty()),
            ) {
                (None, _) => AccountMetric::CredentialRequired,
                (Some(_), None) => AccountMetric::NotReported,
                (Some(key), Some(api_key_id)) => {
                    let url = billing_url(region, range, api_key_id);
                    let result = signed_get_json(http, &url, key, now)
                        .await
                        .and_then(parse_costs);
                    field_from_result(
                        result,
                        AccountScope::new(AccountScopeKind::ApiKey, Some(api_key_id.to_owned())),
                        &format!(
                            "https://modelstudio.{}.aliyuncs.com/modelstudio/billing/trend",
                            region.region_id
                        ),
                    )
                }
            },
        );
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Region {
    region_id: &'static str,
    domain: &'static str,
}

fn region_for_profile(profile: &ProviderProfile) -> Option<Region> {
    if profile.provider_id.as_str() != "qwen" {
        return None;
    }
    let host = url::Url::parse(&profile.base_url)
        .ok()?
        .host_str()?
        .to_owned();
    match host.as_str() {
        "dashscope.aliyuncs.com" => Some(Region {
            region_id: "cn-beijing",
            domain: "cn-beijing",
        }),
        "dashscope-intl.aliyuncs.com" => Some(Region {
            region_id: "ap-southeast-1",
            domain: "ap-southeast-1",
        }),
        "dashscope-us.aliyuncs.com" => Some(Region {
            region_id: "us-east-1",
            domain: "us-east-1",
        }),
        "cn-hongkong.dashscope.aliyuncs.com" => Some(Region {
            region_id: "cn-hongkong",
            domain: "cn-hongkong",
        }),
        _ => None,
    }
}

fn valid_workspace_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

#[cfg(test)]
fn parse_quotas(body: Value) -> Result<Vec<AccountQuotaWindow>, AccountFailure> {
    if body.get("success").and_then(Value::as_bool) == Some(false) {
        return Err(AccountFailure::ProviderError);
    }
    let rows = body
        .pointer("/output/quotas")
        .and_then(Value::as_array)
        .ok_or(AccountFailure::InvalidResponse)?;
    parse_quota_rows(rows)
}

struct QuotaPage {
    total: usize,
    row_count: usize,
    fingerprint: String,
    windows: Vec<AccountQuotaWindow>,
}

fn parse_quota_page(body: Value) -> Result<QuotaPage, AccountFailure> {
    if body.get("success").and_then(Value::as_bool) == Some(false) {
        return Err(AccountFailure::ProviderError);
    }
    let output = body.get("output").ok_or(AccountFailure::InvalidResponse)?;
    let total = output
        .get("total")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(AccountFailure::InvalidResponse)?;
    if total > QUOTAS_PAGE_SIZE.saturating_mul(MAX_QUOTAS_PAGES) {
        return Err(AccountFailure::InvalidResponse);
    }
    let rows = output
        .get("quotas")
        .and_then(Value::as_array)
        .ok_or(AccountFailure::InvalidResponse)?;
    Ok(QuotaPage {
        total,
        row_count: rows.len(),
        fingerprint: serde_json::to_string(rows).map_err(|_| AccountFailure::InvalidResponse)?,
        windows: parse_quota_rows(rows)?,
    })
}

async fn fetch_quotas(
    http: &dyn Transport,
    endpoint: &str,
    key: &crate::protocol::Secret<String>,
) -> Result<Vec<AccountQuotaWindow>, AccountFailure> {
    let mut page_no = 1_usize;
    let mut total = None;
    let mut rows_returned = 0_usize;
    let mut fingerprints = BTreeSet::new();
    let mut windows = Vec::new();
    loop {
        let url = format!("{endpoint}?page_no={page_no}&page_size={QUOTAS_PAGE_SIZE}");
        let body = get_json(http, url, header_bearer(key)).await?;
        let page = parse_quota_page(body)?;
        if total.is_some_and(|expected| expected != page.total)
            || !fingerprints.insert(page.fingerprint)
        {
            return Err(AccountFailure::InvalidResponse);
        }
        total = Some(page.total);
        rows_returned = rows_returned.saturating_add(page.row_count);
        if page.row_count == 0 && rows_returned < page.total {
            return Err(AccountFailure::InvalidResponse);
        }
        windows.extend(page.windows);
        if rows_returned >= page.total {
            return Ok(windows);
        }
        if page_no >= MAX_QUOTAS_PAGES {
            return Err(AccountFailure::InvalidResponse);
        }
        page_no += 1;
    }
}

fn parse_quota_rows(rows: &[Value]) -> Result<Vec<AccountQuotaWindow>, AccountFailure> {
    let mut windows = Vec::new();
    for row in rows {
        let model = row
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| !model.is_empty())
            .unwrap_or("model");
        for (scope, limit) in [
            ("model", row.get("model_limit")),
            ("workspace", row.get("workspace_limit")),
        ] {
            let Some(limit) = limit.filter(|value| !value.is_null()) else {
                continue;
            };
            if let Some(count) = limit.get("request_limit").and_then(Value::as_u64) {
                let seconds = limit.get("request_limit_period").and_then(Value::as_u64);
                windows.push(AccountQuotaWindow {
                    name: format!("{model}:{scope}:requests"),
                    duration_mins: seconds
                        .and_then(|seconds| (seconds % 60 == 0).then_some(seconds / 60)),
                    used_percent: None,
                    remaining_percent: None,
                    resets_at_unix: None,
                    limit: Some(count),
                    used: None,
                    remaining: None,
                    limit_decimal: None,
                    remaining_decimal: None,
                    unit: Some("requests".into()),
                });
            }
            if let Some(count) = limit.get("usage_limit").and_then(Value::as_u64) {
                let seconds = limit.get("usage_limit_period").and_then(Value::as_u64);
                let unit = limit
                    .get("usage_limit_field")
                    .and_then(Value::as_str)
                    .unwrap_or("usage");
                windows.push(AccountQuotaWindow {
                    name: format!("{model}:{scope}:{unit}"),
                    duration_mins: seconds
                        .and_then(|seconds| (seconds % 60 == 0).then_some(seconds / 60)),
                    used_percent: None,
                    remaining_percent: None,
                    resets_at_unix: None,
                    limit: Some(count),
                    used: None,
                    remaining: None,
                    limit_decimal: None,
                    remaining_decimal: None,
                    unit: Some(unit.into()),
                });
            }
        }
    }
    Ok(windows)
}

fn billing_url(region: Region, range: (u64, u64), api_key_id: &str) -> String {
    let start = unix_day(range.0);
    let end = unix_day(range.1.saturating_sub(1));
    let filter = json!({"dimensions":[
        {"code":"API_KEY_ID","values":[api_key_id],"selectType":"IN"},
        {"code":"BUSINESS_REGION","values":[region.region_id],"selectType":"IN"}
    ]});
    let params = [
        ("filter", filter.to_string()),
        ("granularity", "DAY".to_owned()),
        ("groupBy", json!([{"code":"BASE_MODEL"}]).to_string()),
        ("locale", "en-US".to_owned()),
        ("regionId", region.region_id.to_owned()),
        ("timePeriod", json!({"start":start,"end":end}).to_string()),
        ("topNum", "20".to_owned()),
        ("zeroFilter", "true".to_owned()),
    ];
    let canonical_query = params
        .iter()
        .map(|(key, value)| format!("{}={}", percent_encode(key), percent_encode(value)))
        .collect::<Vec<_>>()
        .join("&");
    format!(
        "https://modelstudio.{}.aliyuncs.com/modelstudio/billing/trend?{canonical_query}",
        region.region_id
    )
}

async fn signed_get_json(
    http: &dyn Transport,
    url: &str,
    access_key: &super::AlibabaAccessKey,
    now: u64,
) -> Result<Value, AccountFailure> {
    let parsed = url::Url::parse(url).map_err(|_| AccountFailure::InvalidResponse)?;
    let host = parsed.host_str().ok_or(AccountFailure::InvalidResponse)?;
    let canonical_uri = parsed.path();
    let canonical_query = parsed.query().unwrap_or_default();
    let date = unix_acs_date(now);
    let mut nonce_bytes = [0_u8; 16];
    SystemRandom::new()
        .fill(&mut nonce_bytes)
        .map_err(|_| AccountFailure::Transport)?;
    let nonce = hex(&nonce_bytes);
    let payload_hash = hex(digest::digest(&digest::SHA256, b"").as_ref());
    let mut signed_headers = vec![
        ("host", host.to_owned()),
        ("x-acs-action", "GetBillingTrend".to_owned()),
        ("x-acs-content-sha256", payload_hash.clone()),
        ("x-acs-date", date.clone()),
        ("x-acs-signature-nonce", nonce.clone()),
        ("x-acs-version", SIGNING_VERSION.to_owned()),
    ];
    if let Some(token) = &access_key.security_token {
        signed_headers.push(("x-acs-security-token", token.expose_secret().clone()));
    }
    signed_headers.sort_by_key(|(name, _)| *name);
    let signed_header_names = signed_headers
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(";");
    let canonical_headers = signed_headers
        .iter()
        .map(|(name, value)| format!("{name}:{}\n", value.trim()))
        .collect::<String>();
    let canonical_request = format!(
        "GET\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_header_names}\n{payload_hash}"
    );
    let canonical_hash =
        hex(digest::digest(&digest::SHA256, canonical_request.as_bytes()).as_ref());
    let string_to_sign = format!("ACS3-HMAC-SHA256\n{canonical_hash}");
    let hmac_key = hmac::Key::new(
        hmac::HMAC_SHA256,
        access_key.secret.expose_secret().as_bytes(),
    );
    let signature = hex(hmac::sign(&hmac_key, string_to_sign.as_bytes()).as_ref());
    let authorization = format!(
        "ACS3-HMAC-SHA256 Credential={},SignedHeaders={},Signature={signature}",
        access_key.id, signed_header_names
    );
    let mut headers = vec![
        ("authorization".into(), authorization),
        ("x-acs-action".into(), "GetBillingTrend".into()),
        ("x-acs-content-sha256".into(), payload_hash),
        ("x-acs-date".into(), date),
        ("x-acs-signature-nonce".into(), nonce),
        ("x-acs-version".into(), SIGNING_VERSION.into()),
    ];
    if let Some(token) = &access_key.security_token {
        headers.push(("x-acs-security-token".into(), token.expose_secret().clone()));
    }
    get_json(http, url.to_owned(), headers).await
}

fn parse_costs(body: Value) -> Result<AccountCostUsage, AccountFailure> {
    if body.get("success").and_then(Value::as_bool) != Some(true)
        && body.get("code").and_then(scalar_string).as_deref() != Some("200")
    {
        return Err(AccountFailure::ProviderError);
    }
    let data = body.get("data").ok_or(AccountFailure::InvalidResponse)?;
    let result_by_time = data
        .get("resultByTime")
        .and_then(Value::as_array)
        .ok_or(AccountFailure::InvalidResponse)?;
    let default_currency = data
        .pointer("/costTotals/currency")
        .and_then(Value::as_str)
        .ok_or(AccountFailure::InvalidResponse)?;
    let mut buckets = Vec::new();
    for day in result_by_time {
        let period = day
            .get("period")
            .and_then(Value::as_str)
            .ok_or(AccountFailure::InvalidResponse)?;
        let date = if period.len() == 8 && period.bytes().all(|byte| byte.is_ascii_digit()) {
            format!("{}-{}-{}", &period[..4], &period[4..6], &period[6..8])
        } else {
            return Err(AccountFailure::InvalidResponse);
        };
        let start_unix = parse_iso_utc(&date).ok_or(AccountFailure::InvalidResponse)?;
        let currency = day
            .pointer("/total/currency")
            .and_then(Value::as_str)
            .unwrap_or(default_currency);
        let details = day.get("periodDetails").and_then(Value::as_array);
        if let Some(details) = details.filter(|details| !details.is_empty()) {
            for detail in details {
                buckets.push(AccountCostBucket {
                    start_unix,
                    end_unix: start_unix.saturating_add(86_400),
                    currency: currency.to_owned(),
                    amount: scalar_string(
                        detail
                            .get("amount")
                            .ok_or(AccountFailure::InvalidResponse)?,
                    )
                    .ok_or(AccountFailure::InvalidResponse)?,
                    model: detail.get("key").and_then(Value::as_str).map(str::to_owned),
                });
            }
        } else {
            let total = day.get("total").ok_or(AccountFailure::InvalidResponse)?;
            buckets.push(AccountCostBucket {
                start_unix,
                end_unix: start_unix.saturating_add(86_400),
                currency: currency.to_owned(),
                amount: scalar_string(total.get("amount").ok_or(AccountFailure::InvalidResponse)?)
                    .ok_or(AccountFailure::InvalidResponse)?,
                model: None,
            });
        }
    }
    Ok(AccountCostUsage { buckets })
}

fn scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn unix_day(seconds: u64) -> String {
    let days = i64::try_from(seconds / 86_400).unwrap_or(i64::MAX);
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}")
}

fn unix_acs_date(seconds: u64) -> String {
    let day = unix_day(seconds);
    let time = seconds % 86_400;
    format!(
        "{day}T{:02}:{:02}:{:02}Z",
        time / 3_600,
        (time % 3_600) / 60,
        time % 60
    )
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    (year + i64::from(month <= 2), month as u32, day as u32)
}

fn percent_encode(value: &str) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(char::from(byte));
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(HEX[(byte >> 4) as usize]));
        out.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_quota_windows_report_limits_without_fabricating_consumption() {
        let windows = parse_quotas(json!({"success":true,"output":{"quotas":[{
            "model":"qwen3.8-max",
            "model_limit":{"request_limit":5,"request_limit_period":1,"usage_limit":100000,"usage_limit_field":"total_tokens","usage_limit_period":60},
            "workspace_limit":null
        }]}})).unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].limit, Some(5));
        assert_eq!(windows[0].used, None);
        assert_eq!(windows[0].remaining, None);
        assert_eq!(windows[1].duration_mins, Some(1));
    }

    #[test]
    fn day_conversion_and_query_encoding_are_stable() {
        assert_eq!(unix_day(1_735_689_600), "2025-01-01");
        assert_eq!(unix_acs_date(1_735_689_600), "2025-01-01T00:00:00Z");
        assert_eq!(percent_encode("[\"x y\"]"), "%5B%22x%20y%22%5D");
    }

    #[test]
    fn cost_rows_preserve_provider_decimal_strings_and_model_grouping() {
        let costs = parse_costs(json!({
            "success":true,
            "data":{"costTotals":{"currency":"CNY"},"resultByTime":[{
                "period":"20260801","total":{"currency":"CNY","amount":"3.40"},
                "periodDetails":[{"key":"qwen-max","amount":"3.40"}]
            }]}
        }))
        .unwrap();
        assert_eq!(costs.buckets[0].amount, "3.40");
        assert_eq!(costs.buckets[0].currency, "CNY");
        assert_eq!(costs.buckets[0].model.as_deref(), Some("qwen-max"));
    }
    #[test]
    fn malformed_cost_period_is_an_error_without_panicking() {
        for period in ["2026年x", "abcdefgh", "2026010", "20261301"] {
            assert!(matches!(
                parse_costs(
                    json!({"success":true,"data":{"costTotals":{"currency":"CNY"},"resultByTime":[{"period":period,"total":{"amount":"1"}}]}})
                ),
                Err(AccountFailure::InvalidResponse)
            ));
        }
    }
}
