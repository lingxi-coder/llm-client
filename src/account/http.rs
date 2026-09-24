//! Bounded account HTTP, pagination values and parsing helpers.
use super::*;

pub(super) async fn get_json(
    http: &dyn Transport,
    url: String,
    headers: Vec<(String, String)>,
) -> Result<Value, AccountFailure> {
    let response = crate::transport::HttpExecutor::new(http)
        .execute_bounded(
            HttpRequest {
                method: "GET".into(),
                url,
                headers,
                body: Bytes::new(),
                timeout: None,
            },
            MAX_ACCOUNT_BODY,
        )
        .await
        .map_err(execution::map_transport_error)?;
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
