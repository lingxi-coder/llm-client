//! Shared account report parsing and bounded HTTP helpers.
use crate::account::*;
use crate::transport::{HttpRequest, Transport};
use bytes::Bytes;
use serde_json::Value;
use url::Url;

pub(crate) const MAX_PAGES: usize = 100;

pub(crate) fn url(base: &str, params: &[(&str, String)]) -> Result<String, AccountFailure> {
    let mut url = Url::parse(base).map_err(|_| AccountFailure::InvalidResponse)?;
    url.query_pairs_mut()
        .extend_pairs(params.iter().map(|(key, value)| (*key, value.as_str())));
    Ok(url.to_string())
}

pub(crate) fn next_page<'a>(
    body: &'a Value,
    has_more_name: &str,
    page_name: &str,
) -> Result<Option<&'a str>, AccountFailure> {
    let more = body
        .get(has_more_name)
        .and_then(Value::as_bool)
        .ok_or(AccountFailure::InvalidResponse)?;
    if more {
        body.get(page_name)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(Some)
            .ok_or(AccountFailure::InvalidResponse)
    } else {
        Ok(None)
    }
}

pub(crate) fn number(value: &Value, key: &str) -> Result<u64, AccountFailure> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or(AccountFailure::InvalidResponse)
}

pub(crate) fn optional_number(value: &Value, key: &str) -> Result<Option<u64>, AccountFailure> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or(AccountFailure::InvalidResponse),
    }
}

pub(crate) fn add(current: &mut Option<u64>, value: u64) -> Result<(), AccountFailure> {
    *current = Some(
        current
            .unwrap_or(0)
            .checked_add(value)
            .ok_or(AccountFailure::InvalidResponse)?,
    );
    Ok(())
}

pub(crate) fn valid_currency(value: &str) -> bool {
    value.len() == 3 && value.bytes().all(|b| b.is_ascii_alphabetic())
}

pub(crate) async fn post_json(
    http: &dyn Transport,
    url: String,
    key: &str,
    value: &Value,
) -> Result<Value, AccountFailure> {
    let body = serde_json::to_vec(value).map_err(|_| AccountFailure::InvalidResponse)?;
    let response = crate::transport::HttpExecutor::new(http)
        .execute_bounded(
            HttpRequest {
                method: "POST".into(),
                url,
                headers: vec![
                    ("authorization".into(), format!("Bearer {key}")),
                    ("content-type".into(), "application/json".into()),
                ],
                body: Bytes::from(body),
                timeout: None,
            },
            crate::account::MAX_ACCOUNT_BODY,
        )
        .await
        .map_err(crate::account::execution::map_transport_error)?;
    match response.status {
        200..=299 => {}
        401 => return Err(AccountFailure::Unauthorized),
        403 => return Err(AccountFailure::PermissionDenied),
        429 => return Err(AccountFailure::RateLimited),
        _ => return Err(AccountFailure::ProviderError),
    }
    if response.body.len() > crate::account::MAX_ACCOUNT_BODY {
        return Err(AccountFailure::InvalidResponse);
    }
    serde_json::from_slice(&response.body).map_err(|_| AccountFailure::InvalidResponse)
}

pub(crate) fn format_rfc3339(seconds: u64) -> Result<String, AccountFailure> {
    let days = i64::try_from(seconds / 86_400).map_err(|_| AccountFailure::InvalidResponse)?;
    let (year, month, day) = civil_from_days(days).ok_or(AccountFailure::InvalidResponse)?;
    let day_seconds = seconds % 86_400;
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        day_seconds / 3_600,
        day_seconds % 3_600 / 60,
        day_seconds % 60
    ))
}

pub(crate) fn parse_rfc3339(value: &str) -> Result<u64, AccountFailure> {
    parse_iso_utc(value).ok_or(AccountFailure::InvalidResponse)
}

pub(crate) fn civil_from_days(days: i64) -> Option<(i64, u32, u32)> {
    let z = days.checked_add(719_468)?;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    Some((year, month as u32, day as u32))
}
#[derive(Clone, Copy)]
pub(crate) struct Decimal {
    pub(crate) units: i128,
    scale: u32,
}

pub(crate) fn decimal(value: &Value) -> Result<Decimal, AccountFailure> {
    let raw = match value {
        Value::String(raw) => raw.as_str(),
        Value::Number(number) => return Decimal::parse(&number.to_string()),
        _ => return Err(AccountFailure::InvalidResponse),
    };
    Decimal::parse(raw)
}

pub(crate) fn amount(value: &Value) -> Result<String, AccountFailure> {
    let parsed = decimal(value)?;
    Ok(match value {
        Value::String(raw) => raw.clone(),
        _ => parsed.to_text(),
    })
}

impl Decimal {
    pub(crate) fn parse(raw: &str) -> Result<Self, AccountFailure> {
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

    pub(crate) fn subtract(self, other: Self) -> Result<Self, AccountFailure> {
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

    pub(crate) fn to_text(self) -> String {
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

    pub(crate) fn to_f64(self) -> f64 {
        self.units as f64 / 10_f64.powi(self.scale as i32)
    }
}
pub(crate) fn optional_unix(value: Option<&Value>) -> Result<Option<u64>, AccountFailure> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or(AccountFailure::InvalidResponse),
    }
}

pub(crate) fn optional_time(value: Option<&Value>) -> Result<Option<u64>, AccountFailure> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .and_then(parse_iso_utc)
            .map(Some)
            .ok_or(AccountFailure::InvalidResponse),
    }
}
