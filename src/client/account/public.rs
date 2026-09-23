//! Account APIs reachable with ordinary provider keys or an optional
//! OpenRouter management key. URLs are fixed to the providers' own hosts.

use super::{
    field_from_result, get_json, header_bearer, AccountBalance, AccountFailure, AccountIdentity,
    AccountMetric, AccountQuery, AccountQuotaWindow, AccountScope, AccountScopeKind,
    AccountSnapshot, AccountUsageSource,
};
use crate::transport::Transport;
use async_trait::async_trait;
use lingxi_agent_api::protocol::ProviderProfile;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

const OPENROUTER_KEY: &str = "https://openrouter.ai/api/v1/key";
const OPENROUTER_CREDITS: &str = "https://openrouter.ai/api/v1/credits";
const DEEPSEEK_BALANCE: &str = "https://api.deepseek.com/user/balance";
const MOONSHOT_CN_BALANCE: &str = "https://api.moonshot.cn/v1/users/me/balance";
const MOONSHOT_GLOBAL_BALANCE: &str = "https://api.moonshot.ai/v1/users/me/balance";

pub(super) fn register(
    sources: &mut BTreeMap<(String, AccountIdentity), Arc<dyn AccountUsageSource>>,
) {
    sources.insert(
        ("openrouter".into(), AccountIdentity::ApiKey),
        Arc::new(OpenRouter),
    );
    sources.insert(
        ("deepseek".into(), AccountIdentity::ApiKey),
        Arc::new(DeepSeek),
    );
    sources.insert(("kimi".into(), AccountIdentity::ApiKey), Arc::new(Moonshot));
}

struct OpenRouter;
struct DeepSeek;
struct Moonshot;

fn snapshot(profile: &ProviderProfile, query: &AccountQuery, now: u64) -> AccountSnapshot {
    AccountSnapshot::unsupported(profile, query.identity, now)
}

fn user_scope() -> AccountScope {
    AccountScope::new(AccountScopeKind::User, None)
}

fn key_scope() -> AccountScope {
    // /key returns the credential used for this request, without a canonical
    // key ID. A caller-supplied selector cannot identify that credential.
    AccountScope::new(AccountScopeKind::ApiKey, None)
}

#[async_trait]
impl AccountUsageSource for OpenRouter {
    async fn fetch(
        &self,
        profile: &ProviderProfile,
        query: &AccountQuery,
        _range: (u64, u64),
        now: u64,
        http: &dyn Transport,
    ) -> AccountSnapshot {
        let mut result = snapshot(profile, query, now);
        let key_details = match &query.credential {
            Some(key) => Some(
                get_json(http, OPENROUTER_KEY.into(), header_bearer(key))
                    .await
                    .and_then(parse_openrouter_key),
            ),
            None => None,
        };

        result.quota_windows = match &key_details {
            Some(Ok(details)) => details
                .quota
                .clone()
                .map(|quota| AccountMetric::available(key_scope(), OPENROUTER_KEY, vec![quota]))
                .unwrap_or(AccountMetric::NotReported),
            Some(Err(reason)) => AccountMetric::Failed { reason: *reason },
            None => AccountMetric::CredentialRequired,
        };

        result.balance = if let Some(key) = &query.management_credential {
            let balance = get_json(http, OPENROUTER_CREDITS.into(), header_bearer(key))
                .await
                .and_then(parse_openrouter_credits);
            field_from_result(
                balance.map(|value| vec![value]),
                user_scope(),
                OPENROUTER_CREDITS,
            )
        } else {
            AccountMetric::CredentialRequired
        };
        result
    }
}

struct OpenRouterKeyDetails {
    quota: Option<AccountQuotaWindow>,
}

fn parse_openrouter_key(body: Value) -> Result<OpenRouterKeyDetails, AccountFailure> {
    let data = body
        .get("data")
        .and_then(Value::as_object)
        .ok_or(AccountFailure::InvalidResponse)?;
    let Some(limit) = data.get("limit").filter(|value| !value.is_null()) else {
        return Ok(OpenRouterKeyDetails { quota: None });
    };
    let limit = decimal(limit)?;
    let remaining = decimal(
        data.get("limit_remaining")
            .ok_or(AccountFailure::InvalidResponse)?,
    )?;
    let limit_text = limit.to_text();
    let remaining_text = remaining.to_text();
    let duration_mins = match data.get("limit_reset").and_then(Value::as_str) {
        Some("daily") => Some(1_440),
        Some("weekly") => Some(10_080),
        Some("monthly") | None => None,
        Some(_) => None,
    };
    let remaining_percent =
        if limit.units > 0 && remaining.units >= 0 && limit.subtract(remaining)?.units >= 0 {
            let fraction = remaining.to_f64() / limit.to_f64();
            Some(fraction * 100.0)
        } else {
            None
        };
    let quota = AccountQuotaWindow {
        name: "api_key_spend_limit".into(),
        duration_mins,
        used_percent: remaining_percent.map(|value| 100.0 - value),
        remaining_percent,
        resets_at_unix: None,
        // Money can have fractional units; these integer slots must stay empty.
        limit: None,
        used: None,
        remaining: None,
        limit_decimal: Some(limit_text),
        remaining_decimal: Some(remaining_text),
        unit: Some("USD".into()),
    };
    Ok(OpenRouterKeyDetails { quota: Some(quota) })
}

fn parse_openrouter_credits(body: Value) -> Result<AccountBalance, AccountFailure> {
    let data = body.get("data").ok_or(AccountFailure::InvalidResponse)?;
    let total = decimal(
        data.get("total_credits")
            .ok_or(AccountFailure::InvalidResponse)?,
    )?;
    let used = decimal(
        data.get("total_usage")
            .ok_or(AccountFailure::InvalidResponse)?,
    )?;
    Ok(AccountBalance {
        unit: "USD".into(),
        remaining: total.subtract(used)?.to_text(),
        total: Some(total.to_text()),
        is_available: None,
    })
}

#[async_trait]
impl AccountUsageSource for DeepSeek {
    async fn fetch(
        &self,
        profile: &ProviderProfile,
        query: &AccountQuery,
        _range: (u64, u64),
        now: u64,
        http: &dyn Transport,
    ) -> AccountSnapshot {
        let mut result = snapshot(profile, query, now);
        result.balance = match &query.credential {
            Some(key) => {
                let balance = get_json(http, DEEPSEEK_BALANCE.into(), header_bearer(key))
                    .await
                    .and_then(parse_deepseek_balance);
                field_from_result(balance, user_scope(), DEEPSEEK_BALANCE)
            }
            None => AccountMetric::CredentialRequired,
        };
        result
    }
}

fn parse_deepseek_balance(body: Value) -> Result<Vec<AccountBalance>, AccountFailure> {
    let is_available = body
        .get("is_available")
        .and_then(Value::as_bool)
        .ok_or(AccountFailure::InvalidResponse)?;
    let rows = body
        .get("balance_infos")
        .and_then(Value::as_array)
        .ok_or(AccountFailure::InvalidResponse)?;
    rows.iter()
        .map(|row| {
            let unit = row
                .get("currency")
                .and_then(Value::as_str)
                .filter(|unit| matches!(*unit, "USD" | "CNY"))
                .ok_or(AccountFailure::InvalidResponse)?;
            let remaining = amount(
                row.get("total_balance")
                    .ok_or(AccountFailure::InvalidResponse)?,
            )?;
            Ok(AccountBalance {
                unit: unit.into(),
                remaining,
                total: None,
                is_available: Some(is_available),
            })
        })
        .collect()
}

#[async_trait]
impl AccountUsageSource for Moonshot {
    async fn fetch(
        &self,
        profile: &ProviderProfile,
        query: &AccountQuery,
        _range: (u64, u64),
        now: u64,
        http: &dyn Transport,
    ) -> AccountSnapshot {
        let mut result = snapshot(profile, query, now);
        if profile.profile_name == "kimi-code" {
            return result;
        }
        let (url, unit) = if official_origin(&profile.base_url, "https://api.moonshot.cn") {
            (MOONSHOT_CN_BALANCE, "CNY")
        } else if official_origin(&profile.base_url, "https://api.moonshot.ai") {
            (MOONSHOT_GLOBAL_BALANCE, "USD")
        } else {
            // Kimi Code has the same provider ID but a separate subscription API.
            return result;
        };
        result.balance = match &query.credential {
            Some(key) => {
                let balance = get_json(http, url.into(), header_bearer(key))
                    .await
                    .and_then(|body| parse_moonshot_balance(body, unit));
                field_from_result(balance.map(|row| vec![row]), user_scope(), url)
            }
            None => AccountMetric::CredentialRequired,
        };
        result
    }
}

fn official_origin(base: &str, origin: &str) -> bool {
    base == origin
        || base
            .strip_prefix(origin)
            .is_some_and(|tail| tail.starts_with('/'))
}

fn parse_moonshot_balance(body: Value, unit: &str) -> Result<AccountBalance, AccountFailure> {
    let code = body
        .get("code")
        .and_then(Value::as_i64)
        .ok_or(AccountFailure::InvalidResponse)?;
    let status = body
        .get("status")
        .and_then(Value::as_bool)
        .ok_or(AccountFailure::InvalidResponse)?;
    if code != 0 || !status {
        return Err(AccountFailure::ProviderError);
    }
    let remaining = amount(
        body.get("data")
            .and_then(|data| data.get("available_balance"))
            .ok_or(AccountFailure::InvalidResponse)?,
    )?;
    Ok(AccountBalance {
        unit: unit.into(),
        remaining,
        total: None,
        is_available: None,
    })
}

/// A bounded decimal representation avoids floating-point rounding in credits.
#[derive(Clone, Copy)]
struct Decimal {
    units: i128,
    scale: u32,
}

fn decimal(value: &Value) -> Result<Decimal, AccountFailure> {
    let raw = match value {
        Value::String(raw) => raw.as_str(),
        Value::Number(number) => return Decimal::parse(&number.to_string()),
        _ => return Err(AccountFailure::InvalidResponse),
    };
    Decimal::parse(raw)
}

fn amount(value: &Value) -> Result<String, AccountFailure> {
    let parsed = decimal(value)?;
    Ok(match value {
        Value::String(raw) => raw.clone(),
        _ => parsed.to_text(),
    })
}

impl Decimal {
    fn parse(raw: &str) -> Result<Self, AccountFailure> {
        let (negative, unsigned) = match raw.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, raw),
        };
        let (mantissa, exponent) = match unsigned.find(['e', 'E']) {
            Some(index) => {
                let exponent = unsigned[index + 1..]
                    .parse::<i64>()
                    .map_err(|_| AccountFailure::InvalidResponse)?;
                (&unsigned[..index], exponent)
            }
            None => (unsigned, 0),
        };
        let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        if whole.is_empty()
            || !whole.bytes().all(|c| c.is_ascii_digit())
            || !fraction.bytes().all(|c| c.is_ascii_digit())
            || (mantissa.contains('.') && fraction.is_empty())
        {
            return Err(AccountFailure::InvalidResponse);
        }
        let digits = format!("{whole}{fraction}");
        let mut units = digits
            .parse::<i128>()
            .map_err(|_| AccountFailure::InvalidResponse)?;
        let scale = i64::try_from(fraction.len())
            .ok()
            .and_then(|length| length.checked_sub(exponent))
            .ok_or(AccountFailure::InvalidResponse)?;
        if !(-38..=18).contains(&scale) {
            return Err(AccountFailure::InvalidResponse);
        }
        if scale < 0 {
            let multiplier = 10_i128
                .checked_pow((-scale) as u32)
                .ok_or(AccountFailure::InvalidResponse)?;
            units = units
                .checked_mul(multiplier)
                .ok_or(AccountFailure::InvalidResponse)?;
        }
        if negative {
            units = -units;
        }
        Ok(Self {
            units,
            scale: scale.max(0) as u32,
        })
    }

    fn subtract(self, other: Self) -> Result<Self, AccountFailure> {
        let scale = self.scale.max(other.scale);
        let left = self
            .units
            .checked_mul(10_i128.pow(scale - self.scale))
            .ok_or(AccountFailure::InvalidResponse)?;
        let right = other
            .units
            .checked_mul(10_i128.pow(scale - other.scale))
            .ok_or(AccountFailure::InvalidResponse)?;
        Ok(Self {
            units: left
                .checked_sub(right)
                .ok_or(AccountFailure::InvalidResponse)?,
            scale,
        })
    }

    fn to_text(self) -> String {
        let sign = if self.units < 0 { "-" } else { "" };
        let digits = self.units.unsigned_abs().to_string();
        if self.scale == 0 {
            return format!("{sign}{digits}");
        }
        let width = self.scale as usize + 1;
        let padded = format!("{digits:0>width$}");
        let split = padded.len() - self.scale as usize;
        format!("{sign}{}.{}", &padded[..split], &padded[split..])
    }

    fn to_f64(self) -> f64 {
        self.units as f64 / 10_f64.powi(self.scale as i32)
    }
}
