//! Current named Messages beta rejection policy and conversation/model latches.
//! The host supplies model identity and owns conversation lifetimes. Error
//! recognition, request mutation and the process model cache belong to the SDK.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc, Mutex, OnceLock,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Beta {
    CacheDiagnosis,
    PromptCachingEvict,
    ThinkingDisplayUpdates,
    ThinkingTokenCount,
}
impl Beta {
    pub const ALL: [Self; 4] = [
        Self::CacheDiagnosis,
        Self::PromptCachingEvict,
        Self::ThinkingDisplayUpdates,
        Self::ThinkingTokenCount,
    ];
    pub const fn header(self) -> &'static str {
        match self {
            Self::CacheDiagnosis => "cache-diagnosis-2026-04-07",
            Self::PromptCachingEvict => "prompt-caching-evict-2026-05-12",
            Self::ThinkingDisplayUpdates => "thinking-display-updates-2026-08-18",
            Self::ThinkingTokenCount => "thinking-token-count-2026-05-13",
        }
    }
    const fn mask(self) -> u8 {
        1 << self as u8
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConversationBetaState(Arc<AtomicU8>);
impl PartialEq for ConversationBetaState {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl ConversationBetaState {
    pub fn rejected(&self, beta: Beta) -> bool {
        self.0.load(Ordering::Acquire) & beta.mask() != 0
    }
    pub fn reject(&self, beta: Beta) {
        self.0.fetch_or(beta.mask(), Ordering::AcqRel);
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelBetaRejections(Arc<Mutex<BTreeMap<String, u8>>>);
impl ModelBetaRejections {
    pub fn for_process() -> Self {
        static CACHE: OnceLock<ModelBetaRejections> = OnceLock::new();
        CACHE.get_or_init(Self::default).clone()
    }
    pub fn rejected(&self, model: &str, beta: Beta) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(model)
            .is_some_and(|mask| mask & beta.mask() != 0)
    }
    fn reject(&self, model: &str, beta: Beta) {
        *self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(model.into())
            .or_default() |= beta.mask();
    }
}

/// Native APIError.makeMessage: top-level message precedes the complete JSON
/// payload. Using only error.message loses escaping and other payload fields.
pub fn error_message(status: u16, body: &Value) -> String {
    fn truthy(v: &Value) -> bool {
        match v {
            Value::Null => false,
            Value::Bool(v) => *v,
            Value::String(v) => !v.is_empty(),
            Value::Number(v) => v.as_f64() != Some(0.0),
            _ => true,
        }
    }
    fn stringify(v: &Value) -> String {
        String::from_utf8(
            crate::exact_json::serialize(
                v,
                &BTreeMap::new(),
                crate::exact_json::JsonEncoding::JavaScript,
            )
            .expect("JSON payload is serializable"),
        )
        .expect("JSON is UTF-8")
    }
    let message = body
        .get("message")
        .filter(|v| truthy(v))
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| stringify(v))
        })
        .unwrap_or_else(|| {
            if truthy(body) {
                stringify(body)
            } else {
                String::new()
            }
        });
    if status != 0 && !message.is_empty() {
        format!("{status} {message}")
    } else if status != 0 {
        format!("{status} status code (no body)")
    } else if !message.is_empty() {
        message
    } else {
        "(no status code or body)".into()
    }
}

pub(crate) fn capability_rejected(message: &str, capability: &str) -> bool {
    let marker = format!("capability_rejected: {capability}");
    message.match_indices(&marker).any(|(index, _)| {
        message
            .as_bytes()
            .get(index + marker.len())
            .is_none_or(|byte| !byte.is_ascii_alphanumeric() && !b"_:.-".contains(byte))
    })
}
pub(crate) fn prefix_rejected(status: u16, message: &str) -> bool {
    let lower = message.to_lowercase();
    status == 400
        && (lower.contains("not created in this conversation")
            || lower.contains("bound to a different conversation"))
}
pub(crate) fn signature_rejected(status: u16, message: &str) -> bool {
    let lower = message.to_lowercase().replace('`', "");
    status == 400
        && (lower.contains("signature in thinking block")
            || lower.contains("invalid data in redacted_thinking block")
            || (lower.contains("thinking.signature") && lower.contains("field required"))
            || ((lower.contains("thinking block") || lower.contains("redacted_thinking"))
                && (lower.contains("cannot be modified") || lower.contains("invalid signature")))
            || capability_rejected(message, "thinking_signature"))
}
fn display_shape_rejected(message: &str) -> bool {
    [
        "thinking.adaptive.display: Input should be ",
        "thinking.enabled.display: Input should be ",
    ]
    .iter()
    .any(|text| message.contains(text))
}
fn alternate_rejection(beta: Beta, message: &str) -> bool {
    match beta {
        Beta::PromptCachingEvict => {
            message.contains("evict_on_complete") && message.contains("beta")
        }
        Beta::ThinkingDisplayUpdates => display_shape_rejected(message),
        _ => false,
    }
}
fn header_rejected(message: &str, beta: Beta) -> bool {
    (message.contains(beta.header())
        && (message.contains("anthropic-beta") || message.contains("anthropic_beta")))
        || capability_rejected(message, &format!("beta_header:{}", beta.header()))
}
fn unexpected_header(first_line: &str, header: &str) -> bool {
    let Some(mut rest) = first_line.strip_prefix("Unexpected value(s) ") else {
        return false;
    };
    let mut found = false;
    loop {
        let Some(after_open) = rest.strip_prefix('`') else {
            return false;
        };
        let Some(end) = after_open.find('`').filter(|end| *end != 0) else {
            return false;
        };
        found |= &after_open[..end] == header;
        rest = &after_open[end + 1..];
        if let Some(after_comma) = rest.strip_prefix(", ") {
            rest = after_comma;
        }
        if rest.starts_with('`') {
            continue;
        }
        return found
            && (rest.starts_with(" for the `anthropic-beta")
                || rest.starts_with(" for the `anthropic_beta"));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rejection {
    pub beta: Beta,
    pub strong: bool,
}

/// Native kMe/SMe, in native declaration order. Prefix/signature repairs have
/// priority even when the same payload also mentions one of these betas.
pub fn named_rejections(status: u16, body: &Value, sent: &[String]) -> Vec<Rejection> {
    let message = error_message(status, body);
    if status != 400 || prefix_rejected(status, &message) || signature_rejected(status, &message) {
        return Vec::new();
    }
    let fallback = if message
        .as_bytes()
        .get(..4)
        .is_some_and(|prefix| prefix[..3].iter().all(u8::is_ascii_digit) && prefix[3] == b' ')
    {
        &message[4..]
    } else {
        &message
    };
    let first_line = body["error"]["message"]
        .as_str()
        .unwrap_or(fallback)
        .split('\n')
        .next()
        .unwrap_or("");
    Beta::ALL
        .into_iter()
        .filter(|beta| {
            sent.iter().any(|h| h == beta.header())
                && (header_rejected(&message, *beta) || alternate_rejection(*beta, &message))
        })
        .map(|beta| Rejection {
            beta,
            strong: unexpected_header(first_line, beta.header())
                || capability_rejected(&message, &format!("beta_header:{}", beta.header()))
                || alternate_rejection(beta, &message),
        })
        .collect()
}

/// First-party E3e precedes T3e: an unnamed invalid-beta refusal retires
/// token-count for this conversation, without promoting a model-wide refusal.
pub fn request_rejections(status: u16, body: &Value, sent: &[String]) -> Vec<Rejection> {
    if status == 400
        && sent
            .iter()
            .any(|header| header == Beta::ThinkingTokenCount.header())
        && error_message(status, body)
            .to_lowercase()
            .contains("invalid beta flag")
    {
        return vec![Rejection {
            beta: Beta::ThinkingTokenCount,
            strong: false,
        }];
    }
    named_rejections(status, body, sent)
}

/// Gt commits conversation rejection immediately. Strong refusals additionally
/// apply to new conversations on the same host-supplied model identity.
pub fn reject_named(
    rejections: &[Rejection],
    conversation: &ConversationBetaState,
    model: Option<&str>,
    cache: &ModelBetaRejections,
) {
    for rejected in rejections {
        conversation.reject(rejected.beta);
        if let Some(model) = model.filter(|_| rejected.strong) {
            cache.reject(model, rejected.beta);
        }
    }
}

pub fn request_betas(headers: &BTreeMap<String, String>) -> Vec<String> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .map(|(_, value)| {
            value
                .split(',')
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Filter computed betas before explicit extra-body/header overrides. Do not
/// erase explicit body fields: the native final spread still owns them.
pub fn apply_rejections(
    headers: &mut BTreeMap<String, String>,
    body: &mut Value,
    conversation: &ConversationBetaState,
    model: Option<&str>,
    cache: &ModelBetaRejections,
) {
    for beta in Beta::ALL {
        if model.is_some_and(|model| cache.rejected(model, beta)) {
            conversation.reject(beta);
        }
    }
    let rejected: BTreeSet<_> = Beta::ALL
        .into_iter()
        .filter(|beta| {
            conversation.rejected(*beta) || model.is_some_and(|model| cache.rejected(model, *beta))
        })
        .map(Beta::header)
        .collect();
    let mut betas = request_betas(headers);
    let previous = betas.len();
    betas.retain(|beta| !rejected.contains(beta.as_str()));
    if betas.len() != previous {
        super::request_policy::set_header(headers, "anthropic-beta", &betas.join(","));
    }
    if rejected.contains(Beta::CacheDiagnosis.header()) {
        if let Some(body) = body.as_object_mut() {
            body.shift_remove("diagnostics");
        }
    }
    if rejected.contains(Beta::PromptCachingEvict.header())
        && body
            .get("cache_control")
            .is_some_and(|value| value.get("evict_on_complete").is_some())
    {
        if let Some(body) = body.as_object_mut() {
            body.shift_remove("cache_control");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/beta_repair_2_1_288.json"
        ))
        .unwrap()
    }
    #[test]
    fn named_and_earlier_token_repairs_match_actual_native_callbacks() {
        let f = fixture();
        let cases = f["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 6336);
        for case in cases {
            let status = case["status"].as_u64().unwrap() as u16;
            let sent = serde_json::from_value::<Vec<String>>(case["sent"].clone()).unwrap();
            let named = named_rejections(status, &case["body"], &sent);
            assert_eq!(
                serde_json::to_value(&named).unwrap(),
                case["named"],
                "{case}"
            );
            let repairs = request_rejections(status, &case["body"], &sent);
            let scope = ConversationBetaState::default();
            let cache = ModelBetaRejections::default();
            reject_named(&repairs, &scope, Some("fixture"), &cache);
            let rejected: Vec<_> = Beta::ALL
                .into_iter()
                .filter(|beta| scope.rejected(*beta))
                .map(Beta::header)
                .collect();
            let model_rejected: Vec<_> = Beta::ALL
                .into_iter()
                .filter(|beta| cache.rejected("fixture", *beta))
                .map(Beta::header)
                .collect();
            assert_eq!(
                serde_json::to_value(rejected).unwrap(),
                case["expected"]["rejected"],
                "{case}"
            );
            assert_eq!(
                serde_json::to_value(model_rejected).unwrap(),
                case["expected"]["modelRejected"],
                "{case}"
            );
            assert_eq!(
                !repairs.is_empty(),
                case["expected"]["directive"].is_string(),
                "{case}"
            );
        }
    }
    #[test]
    fn native_error_message_bytes_include_complete_envelope_and_top_level_precedence() {
        for case in fixture()["recognitionCases"].as_array().unwrap() {
            assert_eq!(
                error_message(case["status"].as_u64().unwrap() as u16, &case["body"]),
                case["messageBytes"].as_str().unwrap(),
                "{case}"
            );
        }
    }
    #[test]
    fn rejected_computed_fields_retire_without_erasing_explicit_later_spread() {
        let scope = ConversationBetaState::default();
        let cache = ModelBetaRejections::default();
        reject_named(
            &Beta::ALL.map(|beta| Rejection { beta, strong: true }),
            &scope,
            Some("model-a"),
            &cache,
        );
        let fresh = ConversationBetaState::default();
        let mut headers = BTreeMap::from([(
            "anthropic-beta".into(),
            Beta::ALL.map(Beta::header).join(","),
        )]);
        let mut body = serde_json::json!({"diagnostics":{"previous_message_id":"msg"},"cache_control":{"type":"ephemeral","evict_on_complete":true}});
        apply_rejections(&mut headers, &mut body, &fresh, Some("model-a"), &cache);
        assert_eq!(headers["anthropic-beta"], "");
        assert!(body.get("diagnostics").is_none());
        assert!(body.get("cache_control").is_none());
        assert!(fresh.rejected(Beta::ThinkingDisplayUpdates));
        assert!(!cache.rejected("model-b", Beta::ThinkingDisplayUpdates));
        let unaffected = ConversationBetaState::default();
        let mut clean_headers = BTreeMap::from([(
            "anthropic-beta".into(),
            Beta::ThinkingDisplayUpdates.header().into(),
        )]);
        apply_rejections(
            &mut clean_headers,
            &mut Value::Null,
            &unaffected,
            Some("model-b"),
            &cache,
        );
        assert_eq!(
            clean_headers["anthropic-beta"],
            Beta::ThinkingDisplayUpdates.header()
        );
    }
}
