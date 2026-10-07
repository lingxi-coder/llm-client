//! Native GGt guards over a current APIError envelope. Keep IR recognition
//! separate from pF policy guards: only IR suppresses failed-probe accounting.
use super::beta_repair::{capability_rejected, error_message, prefix_rejected};
use serde_json::Value;
use std::sync::OnceLock;

// These native expressions have ASCII literals and use non-Unicode JavaScript
// regex semantics. Project each UTF-16 unit to one byte so dot/quantifiers and
// ASCII word boundaries retain those semantics, including surrogate pairs.
fn regex_units(message: &str) -> Vec<u8> {
    message
        .encode_utf16()
        .map(|unit| match unit {
            0x0d | 0x2028 | 0x2029 => b'\n',
            0x00a0 | 0x1680 | 0x2000..=0x200a | 0x202f | 0x205f | 0x3000 | 0xfeff => 0x81,
            0..=0x7f => unit as u8,
            _ => 0x80,
        })
        .collect()
}
macro_rules! js_match {
    ($pattern:literal, $units:expr) => {{
        static RE: OnceLock<regex::bytes::Regex> = OnceLock::new();
        RE.get_or_init(|| {
            regex::bytes::RegexBuilder::new(&$pattern.replace(r"\s", r"[\t\n\x0B\x0C\r \x81]"))
                .unicode(false)
                .build()
                .expect("static native regex")
        })
        .is_match($units)
    }};
}

fn header_rejected(message: &str, header: &str) -> bool {
    (message.contains(header)
        && (message.contains("anthropic-beta") || message.contains("anthropic_beta")))
        || capability_rejected(message, &format!("beta_header:{header}"))
}

fn thinking_type_rejected(message: &str, units: &[u8]) -> bool {
    js_match!(
        r"(?i)thinking\.type[^a-z]{1,8}(enabled|adaptive)(?s:.*?)not supported",
        units
    ) || js_match!(r"(?i)\badaptive thinking is not supported", units)
        || capability_rejected(message, "thinking_type:enabled")
        || capability_rejected(message, "thinking_type:adaptive")
}

fn cache_field_rejected(message: &str, units: &[u8]) -> bool {
    if !message.contains("cache_control")
        || js_match!(r"(?i)system messages?\b|role .{0,2}system", units)
        || message.contains("empty text block")
        || js_match!(r"\bsystem\.[0-9]+\.", units)
        || message.contains("tool_result")
        || js_match!(r"(?i)\bttl\b", units)
    {
        return false;
    }
    let lower = message.to_lowercase();
    [
        "not permitted",
        "cannot be set",
        "unknown name",
        "unknown field",
        "unrecognized",
        "additional propert",
    ]
    .iter()
    .any(|text| lower.contains(text))
}

fn media_rejected(message: &str, units: &[u8]) -> bool {
    let lower = message.to_lowercase();
    ["image_block", "document_block", "media_budget"]
        .iter()
        .any(|name| capability_rejected(message, name))
        || js_match!(
            r"messages[.\[]([0-9]+)[.\]]+content[.\[]([0-9]+)[.\]]+(?:tool_result[.\[]content[.\[]([0-9]+)[.\]]+)?(image|document|pdf)",
            units
        )
        || [
            "could not process image",
            "image exceeds",
            "image dimensions exceed",
            "image does not match the provided media type",
            "image cannot be empty",
            "exceeds api limit",
            "images exceed the api limit",
            "unable to resize image",
            "unable to compress image",
            "image file is empty",
            "could not process pdf",
            "pdf pages",
            "the pdf specified was not valid",
            "the pdf specified is password protected",
            "pdf cannot be empty",
            "does not support pdf input",
            "does not support pdfs",
            "too much media",
        ]
        .iter()
        .any(|copy| lower.contains(copy))
}

/// Complete native pF boolean composition. Model availability is a routing
/// fact, supplied by its owner rather than inferred from a generic 400/404.
/// First-party Gft returns unknown, so its main Messages caller supplies false.
pub fn policy_denied(status: u16, body: &Value, model_refused: bool) -> bool {
    if model_refused {
        return true;
    }
    let message = error_message(status, body);
    let units = regex_units(&message);
    let advisor_tag = js_match!(r"Input tag 'advisor_[0-9]+'", &units);
    let inline_tag = js_match!(r"Input tag 'tool_definition'", &units);
    let change_tag = js_match!(r"Input tag 'tool_(addition|removal)'", &units);
    if status == 422 {
        return advisor_tag || inline_tag || change_tag;
    }
    if status != 400 {
        return false;
    }
    let signature = super::beta_repair::signature_rejected(status, &message);
    let prefix = prefix_rejected(status, &message);
    let all_named: Vec<_> = super::beta_repair::Beta::ALL
        .into_iter()
        .map(|beta| beta.header().to_owned())
        .collect();
    let normalized = message.to_lowercase().replace('`', "");
    let encrypted = [
        "invalid encrypted_content in search_result block",
        "invalid encrypted_index in text block",
        "failed to decrypt web search result content",
        "invalid encrypted_stdout in encrypted_code_execution_result block",
    ]
    .iter()
    .any(|copy| normalized.contains(copy));
    let thinking_type = thinking_type_rejected(&message, &units);
    let structured = message
        .to_lowercase()
        .contains("output_config.format: extra inputs are not permitted")
        || capability_rejected(&message, "structured_outputs_unsupported");
    let cache_field = cache_field_rejected(&message, &units);
    let mid_system_position = js_match!(
        r"(?i)(?:messages\.([0-9]{1,6}): )?(?:role .{0,2}system.{0,2} must (?:precede an|follow a)|use the top-level .{0,2}system.{0,2} parameter for the initial system prompt)",
        &units
    );
    let effort_without_thinking = js_match!(
        r"(?i)effort '([a-z]+)' is not supported when thinking is disabled",
        &units
    );
    let per_turn_effort = !structured
        && (!effort_without_thinking
            || js_match!(r"messages\.[0-9]+:\s*output_config\.effort", &units))
        && !mid_system_position
        && (header_rejected(&message, "per-turn-control-2026-07-01")
            || (message.contains("output_config") && !message.contains("output_config.timing"))
            || capability_rejected(&message, "effort_unsupported"));
    let lower = message.to_lowercase();
    let effort = !thinking_type
        && ((lower.contains("effort parameter") && lower.contains("not support"))
            || (lower.contains("output_config")
                && lower.contains("effort")
                && lower.contains("extra inputs are not permitted"))
            || capability_rejected(&message, "effort_unsupported"));
    let inline = inline_tag
        || header_rejected(&message, "inline-tools-2026-09-15")
        || (message.contains("are not available on this platform")
            && js_match!(r"tool definitions|tool_addition", &units))
        || message.contains("cannot yet be defined in a message")
        || (message.contains("is already used by a")
            && js_match!(r"tool\.definition|tool_addition", &units));
    let tool_change = change_tag
        || header_rejected(&message, "mid-conversation-tool-changes-2026-07-01")
        || (js_match!(r"\btool_addition\b", &units) && !cache_field);
    header_rejected(&message, "afk-mode-2026-01-31")
        || message.contains("Advisor tool result content could not be processed")
        || message.contains("found in advisor_tool_result blocks")
        || message.contains("the advisor tool is not available")
        || message.contains("cannot be used as an advisor")
        || js_match!(r"tools\.[0-9]+\.model: ", &units)
        || advisor_tag
        || encrypted
        || media_rejected(&message, &units)
        || !super::beta_repair::named_rejections(status, body, &all_named).is_empty()
        || header_rejected(&message, "cache-keepalive-2026-09-03")
        || header_rejected(&message, "thinking-resumption-2026-07-17")
        || (!signature
            && !prefix
            && (header_rejected(&message, "thinking-binding-controls-2026-08-01")
                || js_match!(
                    r"thinking\.(adaptive|enabled)\.block_binding(\.prefix_mismatch_behavior)?: Extra inputs are not permitted",
                    &units
                )))
        || thinking_type
        || signature
        || mid_system_copy(&message)
        || capability_rejected(&message, "mid_conv_system")
        || capability_rejected(&message, &format!("beta_header:{MID_SYSTEM}"))
        || tool_change
        || inline
        || cache_field
        || capability_rejected(&message, "cache_control_field")
        || per_turn_effort
        || header_rejected(&message, "timing-2026-09-09")
        || message.contains("output_config.timing")
        || effort
        || structured
}

const MID_SYSTEM: &str = "mid-conversation-system-2026-04-07";

fn gap_to<'a>(after: &'a str, literal: &str) -> Option<&'a str> {
    for (index, _) in after
        .char_indices()
        .chain(std::iter::once((after.len(), '\0')))
    {
        let gap = &after[..index];
        if gap.encode_utf16().count() > 2 || gap.contains(['\n', '\r', '\u{2028}', '\u{2029}']) {
            return None;
        }
        if let Some(rest) = after[index..].strip_prefix(literal) {
            return Some(rest);
        }
    }
    None
}
fn role_system_tails(text: &str) -> impl Iterator<Item = &str> {
    text.match_indices("role ")
        .filter_map(|(index, _)| gap_to(&text[index + 5..], "system"))
}
fn mid_system_copy(message: &str) -> bool {
    if message.contains(MID_SYSTEM)
        && (message.contains("anthropic-beta") || message.contains("anthropic_beta"))
    {
        return true;
    }
    let lower = message.to_ascii_lowercase();
    if role_system_tails(&lower).any(|tail| {
        gap_to(tail, " must ")
            .is_some_and(|tail| tail.starts_with("precede an") || tail.starts_with("follow a"))
    }) {
        return true;
    }
    if lower.match_indices("use the top-level ").any(|(index, _)| {
        gap_to(&lower[index + 18..], "system")
            .is_some_and(|tail| gap_to(tail, " parameter for the initial system prompt").is_some())
    }) {
        return true;
    }
    if lower.match_indices("text-only ").any(|(index, _)| {
        lower[index + 10..]
            .strip_prefix("role ")
            .and_then(|after| gap_to(after, "system"))
            .is_some_and(|tail| gap_to(tail, " messages require").is_some())
    }) {
        return true;
    }
    if message.contains("Unexpected role") && message.contains("input message role") {
        return true;
    }
    if message.contains("cache_control")
        && (role_system_tails(&lower).next().is_some()
            || lower.match_indices("system message").any(|(index, _)| {
                let tail = &lower[index + 14..];
                let tail = tail.strip_prefix('s').unwrap_or(tail);
                tail.as_bytes()
                    .first()
                    .is_none_or(|byte| !byte.is_ascii_alphanumeric() && *byte != b'_')
            }))
    {
        return true;
    }
    message.contains("not supported") && role_system_tails(&lower).next().is_some()
}

/// GGt's prefix-healing marker belongs to the error object, so its owner must
/// supply it explicitly. No SDK error-category or HTTP auth inference is used.
pub fn known_error(status: u16, body: &Value, prefix_heal_declined: bool) -> bool {
    let message = error_message(status, body);
    let lower = message.to_lowercase();
    if (prefix_rejected(status, &message) && !prefix_heal_declined)
        || lower.contains("prompt is too long")
        || lower.contains("input is too long for requested model")
        || capability_rejected(&message, "prompt_too_long")
        || lower.contains("input length and `max_tokens` exceed context limit")
        || capability_rejected(&message, "max_tokens_context_overflow")
        || lower.contains("credit balance is too low")
        || lower.contains("organization has been disabled")
        || body["error"]["details"]["error_code"] == "dlp_request_denied"
    {
        return true;
    }
    if status != 400 {
        return false;
    }
    let normalized = lower.replace('`', "");
    message.contains("Fast mode is not enabled")
        || message.contains("cannot be used as an advisor when the request model is")
        || body["error"]["type"] == "policy_blocked"
        || message.contains("Output blocked by content filtering policy")
        || mid_system_copy(&message)
        || capability_rejected(&message, "mid_conv_system")
        || capability_rejected(&message, &format!("beta_header:{MID_SYSTEM}"))
        || [
            "invalid encrypted_content in search_result block",
            "invalid encrypted_index in text block",
            "failed to decrypt web search result content",
            "invalid encrypted_stdout in encrypted_code_execution_result block",
        ]
        .iter()
        .any(|copy| normalized.contains(copy))
}

/// Host routing/history facts used by native IR's x4e branch.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecognitionContext<'a> {
    pub request_model: &'a str,
    pub refusal_fallback_target: bool,
    pub previous_fast_rejection: bool,
    pub prefix_heal_declined: bool,
}

/// Native oFe. Parse only the first matching copy, like String.match, and let
/// the host resolve identities/aliases without moving provider parsing there.
pub fn speed_rejected_for(
    status: u16,
    body: &Value,
    model: &str,
    identity: impl Fn(&str) -> String,
) -> bool {
    if status != 400 {
        return false;
    }
    let message = error_message(status, body);
    for (start, _) in message.match_indices('\'') {
        let tail = &message[start + 1..];
        let Some(end) = tail.find('\'') else {
            continue;
        };
        if end != 0 && tail[end..].starts_with("' does not support the `speed` parameter") {
            return identity(&tail[..end]) == identity(model);
        }
    }
    false
}

pub fn fast_not_enabled(status: u16, body: &Value) -> bool {
    status == 400 && error_message(status, body).contains("Fast mode is not enabled")
}

/// Complete native IR = GGt || x4e. Recognition is independent of whether the
/// current request enabled fast mode; only the actual repair uses that fact.
pub fn recognized(
    status: u16,
    body: &Value,
    context: RecognitionContext<'_>,
    identity: impl Fn(&str) -> String,
) -> bool {
    known_error(status, body, context.prefix_heal_declined)
        || ((context.refusal_fallback_target || context.previous_fast_rejection)
            && speed_rejected_for(status, body, context.request_model, identity))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn policy_fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/error_policy_2_1_288.json"
        ))
        .unwrap()
    }
    #[test]
    fn full_native_ir_matches_speed_model_target_and_rejection_history() {
        let f: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/fast_refusal_2_1_288.json"
        ))
        .unwrap();
        let cases = f["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 2520);
        for case in cases {
            let identity =
                |model: &str| f["identities"][model].as_str().unwrap_or(model).to_owned();
            let status = case["status"].as_u64().unwrap() as u16;
            let model = case["request_model"].as_str().unwrap();
            assert_eq!(
                fast_not_enabled(status, &case["body"]),
                case["fastNotEnabled"].as_bool().unwrap(),
                "{case}"
            );
            assert_eq!(
                speed_rejected_for(status, &case["body"], model, identity),
                case["speedMatches"].as_bool().unwrap(),
                "{case}"
            );
            let context = RecognitionContext {
                request_model: model,
                refusal_fallback_target: case["target"].as_bool().unwrap(),
                previous_fast_rejection: case["rejected"].as_bool().unwrap(),
                prefix_heal_declined: case["declined"].as_bool().unwrap(),
            };
            assert_eq!(
                recognized(status, &case["body"], context, identity),
                case["expected"].as_bool().unwrap(),
                "{case}"
            );
        }
    }
    #[test]
    fn complete_native_pf_composition_matches_messages_and_host_refusal_fact() {
        let f = policy_fixture();
        let cases = f["policyCases"].as_array().unwrap();
        assert_eq!(cases.len(), 4140);
        for case in cases {
            assert_eq!(
                policy_denied(
                    case["status"].as_u64().unwrap() as u16,
                    &case["body"],
                    case["model_refused"].as_bool().unwrap()
                ),
                case["expected"].as_bool().unwrap(),
                "{case}"
            );
        }
    }
    #[test]
    fn full_pf_blocks_probe_admission_without_suppressing_failed_probe_accounting() {
        use super::super::thinking_display::{
            probe_admission, DisplayProbe, DisplayProbeBudget, ProbeAdmission, ProbeState,
        };
        let f = policy_fixture();
        let cases = f["displayCases"].as_array().unwrap();
        assert_eq!(cases.len(), 690);
        for case in cases {
            let mut probe = DisplayProbe::default();
            let budget = DisplayProbeBudget::default();
            let initial: ProbeState = serde_json::from_value(case["state"].clone()).unwrap();
            if initial != ProbeState::Idle {
                assert!(probe.on_error(
                    400,
                    ProbeAdmission {
                        header_sent: true,
                        ..Default::default()
                    },
                    &budget
                ));
                if initial == ProbeState::Spent {
                    assert!(probe
                        .on_success(&super::super::beta_repair::ConversationBetaState::default()));
                }
            }
            let status = case["status"].as_u64().unwrap() as u16;
            let retry = probe.on_error(
                status,
                probe_admission(
                    status,
                    &case["body"],
                    true,
                    false,
                    RecognitionContext::default(),
                    str::to_owned,
                ),
                &budget,
            );
            assert_eq!(retry, case["expected"]["retry"].is_string(), "{case}");
            assert_eq!(
                serde_json::to_value(probe.state()).unwrap(),
                case["expected"]["state"],
                "{case}"
            );
            assert_eq!(
                budget.failures(),
                case["expected"]["failures"].as_u64().unwrap() as u32,
                "{case}"
            );
        }
    }
    #[test]
    fn complete_native_ggt_composition_matches_error_envelopes() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/beta_repair_2_1_288.json"
        ))
        .unwrap();
        let cases = fixture["recognitionCases"].as_array().unwrap();
        assert_eq!(cases.len(), 724);
        for case in cases {
            assert_eq!(
                known_error(
                    case["status"].as_u64().unwrap() as u16,
                    &case["body"],
                    case["declined"].as_bool().unwrap()
                ),
                case["expected"].as_bool().unwrap(),
                "{case}"
            );
        }
    }
}
