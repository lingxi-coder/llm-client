//! Account APIs reachable with ordinary provider keys or an optional
//! OpenRouter management key. URLs are fixed to the providers' own hosts.

use super::{
    field_from_result, get_json, header_bearer, AccountBalance, AccountFailure, AccountIdentity,
    AccountMetric, AccountQuotaWindow, AccountScope, AccountScopeKind, AccountUsageSource,
};
use async_trait::async_trait;
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

fn user_scope() -> AccountScope {
    AccountScope::new(AccountScopeKind::User, None)
}

fn key_scope() -> AccountScope {
    // /key returns the credential used for this request, without a canonical
    // key ID. A caller-supplied selector cannot identify that credential.
    AccountScope::new(AccountScopeKind::ApiKey, None)
}

mod openrouter;

mod deepseek;

mod moonshot;
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
