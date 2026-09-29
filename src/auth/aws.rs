//! AWS STS credential DTO and exported JSON parser. The host owns command execution.

use std::fmt;

/// Credentials produced by `awsCredentialExport` (`t2d` return shape).
#[derive(Clone, PartialEq, Eq)]
pub struct AwsExportedCredentials {
    /// `AccessKeyId`.
    pub access_key_id: String,
    /// `SecretAccessKey`.
    pub secret_access_key: String,
    /// `SessionToken`.
    pub session_token: String,
    /// `Expiration` parsed to epoch-milliseconds (`Date.parse`); `None` when
    /// absent or unparseable (`Number.isFinite` gate).
    pub expiration_ms: Option<i64>,
}

impl fmt::Debug for AwsExportedCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AwsExportedCredentials")
            .field("access_key_id", &"[REDACTED]")
            .field("secret_access_key", &"[REDACTED]")
            .field("session_token", &"[REDACTED]")
            .field("expiration_ms", &self.expiration_ms)
            .finish()
    }
}

/// `Rdi(e)`: the three key fields are non-empty strings.
fn is_sts_credentials_shape(v: &serde_json::Value) -> bool {
    let non_empty_str = |k: &str| {
        v.get(k)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|s| !s.is_empty())
    };
    v.is_object()
        && non_empty_str("AccessKeyId")
        && non_empty_str("SecretAccessKey")
        && non_empty_str("SessionToken")
}

/// `wdi(e)`: accept `{Credentials: {...}}` (nested STS output) or the flat
/// credentials object; anything else is `None`.
#[must_use]
pub fn parse_sts_output(v: &serde_json::Value) -> Option<AwsExportedCredentials> {
    let creds = match v.get("Credentials") {
        Some(nested) if is_sts_credentials_shape(nested) => nested,
        _ if is_sts_credentials_shape(v) => v,
        _ => return None,
    };
    let s = |k: &str| {
        creds
            .get(k)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    // `Date.parse(o)` → ms; `Number.isFinite(s) ? s : void 0`.
    let expiration_ms = creds
        .get("Expiration")
        .and_then(serde_json::Value::as_str)
        .and_then(|raw| {
            chrono::DateTime::parse_from_rfc3339(raw)
                .ok()
                .map(|dt| dt.timestamp_millis())
        });
    Some(AwsExportedCredentials {
        access_key_id: s("AccessKeyId"),
        secret_access_key: s("SecretAccessKey"),
        session_token: s("SessionToken"),
        expiration_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_sts_output_accepts_nested_credentials() {
        let v: serde_json::Value = serde_json::json!({
            "Credentials": {
                "AccessKeyId": "AKIA123",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
                "Expiration": "2026-07-02T12:00:00+00:00"
            }
        });
        let c = parse_sts_output(&v).expect("nested Credentials accepted");
        assert_eq!(c.access_key_id, "AKIA123");
        assert_eq!(
            c.expiration_ms,
            Some(1_782_993_600_000),
            "2026-07-02T12:00:00Z in epoch ms"
        );
    }

    #[test]
    fn parse_sts_output_accepts_flat_shape_and_tolerates_bad_expiration() {
        let v: serde_json::Value = serde_json::json!({
            "AccessKeyId": "AKIA123",
            "SecretAccessKey": "secret",
            "SessionToken": "token",
            "Expiration": "not-a-date"
        });
        let c = parse_sts_output(&v).expect("flat shape accepted");
        assert_eq!(
            c.expiration_ms, None,
            "unparseable Expiration → None (Number.isFinite gate)"
        );
    }

    #[test]
    fn parse_sts_output_rejects_empty_or_missing_fields() {
        // Rdi: three non-empty strings required.
        for bad in [
            serde_json::json!({"AccessKeyId": "", "SecretAccessKey": "s", "SessionToken": "t"}),
            serde_json::json!({"AccessKeyId": "a", "SecretAccessKey": "s"}),
            serde_json::json!("not an object"),
            serde_json::json!(null),
        ] {
            assert!(parse_sts_output(&bad).is_none(), "must reject: {bad}");
        }
    }
}
