//! Native session identity facts, separate from model OAuth credentials.
use base64::{
    alphabet,
    engine::{
        general_purpose::{GeneralPurpose, GeneralPurposeConfig},
        DecodePaddingMode,
    },
    Engine,
};
use serde::{Deserialize, Serialize};

/// Raw environment values plus a host's already captured session-token slot.
/// Tokens are borrowed and deliberately have no Debug representation here.
#[derive(Default)]
pub struct SessionEnvironment<'a> {
    pub access_token: Option<&'a str>,
    pub cached_access_token: Option<&'a str>,
    pub remote: Option<&'a str>,
    pub environment_kind: Option<&'a str>,
    pub entrypoint: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionIdentity {
    pub has_session_token: bool,
    pub no_user_account: bool,
    pub agent_owned_remote: bool,
}

fn js_trim(value: &str) -> &str {
    value.trim_matches(|c| matches!(c, '\u{0009}'..='\u{000D}' | '\u{0020}' | '\u{00A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}' | '\u{2028}' | '\u{2029}' | '\u{202F}' | '\u{205F}' | '\u{3000}' | '\u{FEFF}'))
}
fn env_string(value: Option<&str>) -> Option<&str> {
    value.map(js_trim).filter(|value| !value.is_empty())
}
fn remote_flag(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            js_trim(value).to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}
/// Native wU: permissive Node base64url decoding followed by JSON parsing.
/// This reads claims for runtime policy; it does not verify a JWT signature.
fn claims(token: &str) -> Option<super::session_claims::Claims> {
    let token = token.strip_prefix("sk-ant-si-").unwrap_or(token);
    let mut parts = token.split('.');
    let _header = parts.next()?;
    let payload = parts.next()?;
    let _signature = parts.next()?;
    if parts.next().is_some() || payload.is_empty() {
        return None;
    }
    // Buffer's base64 string decoder consumes the low byte of UTF-16 units.
    let mut encoded: Vec<u8> = payload
        .encode_utf16()
        .map(|unit| (unit & 255) as u8)
        .take_while(|byte| *byte != b'=')
        .filter_map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' => Some(byte),
            b'+' => Some(b'-'),
            b'/' => Some(b'_'),
            _ => None,
        })
        .collect();
    if encoded.len() % 4 == 1 {
        encoded.pop();
    }
    let decoder = GeneralPurpose::new(
        &alphabet::URL_SAFE,
        GeneralPurposeConfig::new()
            .with_decode_padding_mode(DecodePaddingMode::Indifferent)
            .with_decode_allow_trailing_bits(true),
    );
    let bytes = decoder.decode(encoded).ok()?;
    super::session_claims::parse(&String::from_utf8_lossy(&bytes))
}

/// Current lc/Zvn, Fyt/VJn/e_ and ma/Pz/Ln decisions from actual session input.
pub fn session_identity(input: SessionEnvironment<'_>) -> SessionIdentity {
    // lc uses raw process.env truthiness, unlike the parsed environment below.
    let token = input
        .access_token
        .filter(|value| !value.is_empty())
        .or(input.cached_access_token);
    let remote = remote_flag(input.remote);
    let kind = env_string(input.environment_kind);
    let entrypoint = env_string(input.entrypoint);
    let payload = token.and_then(claims);
    let object = payload.as_ref();
    let no_user_account = object.is_some_and(|object| {
        let service = !object.contains("account_uuid")
            && !object.contains("sub")
            && object.nonempty("org_service_name")
            && object.nonempty("code_agent_id");
        let worker = remote
            && kind == Some("byoc")
            && object.session_worker()
            && !object.contains("account_uuid")
            && !object.contains("ccr:account_id")
            && object.nonempty("code_agent_id");
        service || worker
    });
    SessionIdentity {
        has_session_token: token.is_some(),
        no_user_account,
        agent_owned_remote: remote
            && kind.is_none()
            && matches!(entrypoint, Some("remote_cowork" | "remote_cowork_trigger")),
    }
}
