//! Anthropic quota facts decoded from provider response headers.
//!
//! These names and their JavaScript coercion rules are provider protocol
//! details. Hosts consume the typed result and keep account/session state.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

const PERSISTENT_RESET_CAP: Duration = Duration::from_secs(6 * 60 * 60);

/// The quota fact Native uses when choosing prompt-cache retention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotaStatus {
    /// Native `zve(...).base.isUsingOverage`.
    pub is_using_overage: bool,
}

/// A quota-window reset that qualifies for Native's retry-wait callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotaWindowWait {
    /// Remaining wait through the unified reset, capped at six hours.
    pub retry_delay: Duration,
    /// The decoded quota status applied before Native waits.
    pub status: QuotaStatus,
}

/// Decode the current overage fact from a response header set.
///
/// Native defaults a missing/empty unified status to `allowed`; overage is in
/// use only when the unified status is `rejected` and the overage window is
/// `allowed` or `allowed_warning`. The independent `...overage-in-use` header
/// does not feed this boolean.
#[must_use]
pub fn quota_status_from_headers(headers: &[(String, String)]) -> QuotaStatus {
    let unified_status = header(headers, "anthropic-ratelimit-unified-status")
        .filter(|value| !value.is_empty())
        .unwrap_or("allowed");
    let overage_status = header(headers, "anthropic-ratelimit-unified-overage-status");
    QuotaStatus {
        is_using_overage: unified_status == "rejected"
            && matches!(overage_status, Some("allowed" | "allowed_warning")),
    }
}

/// Decode the Native error-path overage fact for a 429.
///
/// `claudeAiLimits.extractQuotaStatusFromError` applies `Fet`, which forces the
/// unified status to `rejected` after parsing the response headers. Therefore
/// the 429 `isUsingOverage` result depends on the overage-window status even
/// when the response omits `anthropic-ratelimit-unified-status`.
#[must_use]
pub fn quota_status_from_429_headers(headers: &[(String, String)]) -> QuotaStatus {
    let overage_status = header(headers, "anthropic-ratelimit-unified-overage-status");
    QuotaStatus {
        is_using_overage: matches!(overage_status, Some("allowed" | "allowed_warning")),
    }
}

/// Decode Native's `onQuotaWindowWait` fact for a 429 retry that is about to
/// sleep. Callers must invoke this only after their retry policy selects an
/// actual wait; the decoder enforces the remaining Native gates:
/// retry-watchdog enabled, 429 status, positive unified reset, and unified
/// status exactly `rejected`.
#[must_use]
pub fn retry_quota_window_wait_from_headers(
    status_code: u16,
    headers: &[(String, String)],
    retry_watchdog_enabled: bool,
    now: SystemTime,
) -> Option<QuotaWindowWait> {
    if !retry_watchdog_enabled || status_code != 429 {
        return None;
    }
    if header(headers, "anthropic-ratelimit-unified-status")? != "rejected" {
        return None;
    }
    let retry_delay = unified_reset_delay(headers, now)?;
    Some(QuotaWindowWait {
        retry_delay,
        status: quota_status_from_429_headers(headers),
    })
}

fn unified_reset_delay(headers: &[(String, String)], now: SystemTime) -> Option<Duration> {
    let raw = header(headers, "anthropic-ratelimit-unified-reset")?;
    let reset_seconds = javascript_number(raw)?;
    if !reset_seconds.is_finite() {
        return None;
    }
    let now_ms = now.duration_since(UNIX_EPOCH).ok()?.as_millis() as f64;
    let remaining_ms = (reset_seconds * 1000.0 - now_ms).round();
    if remaining_ms.is_nan() || remaining_ms <= 0.0 {
        return None;
    }
    // Clamp before converting to u64 so huge but finite server values cannot
    // overflow. Native's `Math.min(delayMs, 6h)` applies this same cap.
    let cap_ms = PERSISTENT_RESET_CAP.as_millis() as f64;
    Some(Duration::from_millis(remaining_ms.min(cap_ms) as u64))
}

fn javascript_number(value: &str) -> Option<f64> {
    let value = value.trim_matches(is_javascript_whitespace);
    if value.is_empty() {
        return Some(0.0);
    }

    let radix = if let Some(digits) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        Some((digits, 16))
    } else if let Some(digits) = value
        .strip_prefix("0b")
        .or_else(|| value.strip_prefix("0B"))
    {
        Some((digits, 2))
    } else {
        value
            .strip_prefix("0o")
            .or_else(|| value.strip_prefix("0O"))
            .map(|digits| (digits, 8))
    };
    if let Some((digits, base)) = radix {
        if digits.is_empty() {
            return None;
        }
        let mut number = 0.0;
        for character in digits.chars() {
            let digit = character.to_digit(base)?;
            number = number * f64::from(base) + f64::from(digit);
        }
        return Some(number);
    }

    value.parse::<f64>().ok()
}

fn is_javascript_whitespace(character: char) -> bool {
    matches!(
        character,
        '\u{0009}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{0020}'
            | '\u{00A0}'
            | '\u{FEFF}'
            | '\u{000A}'
            | '\u{000D}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header_vec(entries: &[(&str, &str)]) -> Vec<(String, String)> {
        entries
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    fn now() -> SystemTime {
        UNIX_EPOCH + Duration::from_millis(1_000_000)
    }

    #[test]
    fn overage_fact_matches_unified_status_and_overage_window_only() {
        for overage_status in ["allowed", "allowed_warning"] {
            assert!(
                quota_status_from_headers(&header_vec(&[
                    ("Anthropic-RateLimit-Unified-Status", "rejected"),
                    ("anthropic-ratelimit-unified-overage-status", overage_status),
                    ("anthropic-ratelimit-unified-overage-in-use", "false"),
                ]))
                .is_using_overage
            );
        }
        for pair in [
            ("allowed", "allowed"),
            ("rejected", "rejected"),
            ("rejected", "missing"),
        ] {
            let mut entries = vec![("anthropic-ratelimit-unified-status", pair.0)];
            if pair.1 != "missing" {
                entries.push(("anthropic-ratelimit-unified-overage-status", pair.1));
            }
            assert!(!quota_status_from_headers(&header_vec(&entries)).is_using_overage);
        }
        assert!(
            !quota_status_from_headers(&header_vec(&[(
                "anthropic-ratelimit-unified-overage-in-use",
                "true"
            ),]))
            .is_using_overage
        );
        assert!(
            !quota_status_from_headers(&header_vec(&[
                ("anthropic-ratelimit-unified-status", ""),
                ("anthropic-ratelimit-unified-overage-status", "allowed"),
            ]))
            .is_using_overage
        );
    }

    #[test]
    fn retry_wait_requires_native_watchdog_429_rejected_and_positive_reset() {
        let headers = header_vec(&[
            ("anthropic-ratelimit-unified-status", "rejected"),
            (
                "anthropic-ratelimit-unified-overage-status",
                "allowed_warning",
            ),
            ("anthropic-ratelimit-unified-reset", "1001"),
        ]);
        let wait = retry_quota_window_wait_from_headers(429, &headers, true, now()).unwrap();
        assert_eq!(wait.retry_delay, Duration::from_secs(1));
        assert!(wait.status.is_using_overage);
        assert!(retry_quota_window_wait_from_headers(429, &headers, false, now()).is_none());
        assert!(retry_quota_window_wait_from_headers(200, &headers, true, now()).is_none());

        for reset in ["", "bad", "NaN", "1000"] {
            let invalid = header_vec(&[
                ("anthropic-ratelimit-unified-status", "rejected"),
                ("anthropic-ratelimit-unified-reset", reset),
            ]);
            assert!(retry_quota_window_wait_from_headers(429, &invalid, true, now()).is_none());
        }
        let not_rejected = header_vec(&[
            ("anthropic-ratelimit-unified-status", "allowed"),
            ("anthropic-ratelimit-unified-reset", "1001"),
        ]);
        assert!(retry_quota_window_wait_from_headers(429, &not_rejected, true, now()).is_none());
    }

    #[test]
    fn retry_wait_caps_large_reset_at_six_hours() {
        let headers = header_vec(&[
            ("anthropic-ratelimit-unified-status", "rejected"),
            ("anthropic-ratelimit-unified-reset", "999999999"),
        ]);
        let wait = retry_quota_window_wait_from_headers(429, &headers, true, now()).unwrap();
        assert_eq!(wait.retry_delay, PERSISTENT_RESET_CAP);

        let finite_before_multiplication = header_vec(&[
            ("anthropic-ratelimit-unified-status", "rejected"),
            ("anthropic-ratelimit-unified-reset", "1e308"),
        ]);
        let wait =
            retry_quota_window_wait_from_headers(429, &finite_before_multiplication, true, now())
                .unwrap();
        assert_eq!(wait.retry_delay, PERSISTENT_RESET_CAP);
    }

    #[test]
    fn retry_reset_uses_javascript_number_forms() {
        for reset in ["0x3e9", "0b1111101001", "0o1751", "\u{feff}1001\u{feff}"] {
            let headers = header_vec(&[
                ("anthropic-ratelimit-unified-status", "rejected"),
                ("anthropic-ratelimit-unified-reset", reset),
            ]);
            let wait = retry_quota_window_wait_from_headers(429, &headers, true, now())
                .unwrap_or_else(|| panic!("Native Number should accept reset {reset:?}"));
            assert_eq!(wait.retry_delay, Duration::from_secs(1));
        }
    }

    #[test]
    fn terminal_429_uses_native_forced_rejected_error_status() {
        assert!(
            quota_status_from_429_headers(&header_vec(&[(
                "anthropic-ratelimit-unified-overage-status",
                "allowed"
            ),]))
            .is_using_overage
        );
        assert!(
            !quota_status_from_429_headers(&header_vec(&[(
                "anthropic-ratelimit-unified-status",
                "rejected"
            ),]))
            .is_using_overage
        );
    }
}
