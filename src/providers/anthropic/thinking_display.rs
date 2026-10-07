//! Native thinking-display request selection and unclaimed-error probe state.
//! The host supplies routing/admission facts and owns conversation lifetimes;
//! payload mutation and provider request policy remain in the SDK.
use super::beta_repair::{Beta, ConversationBetaState};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

pub const UPDATES_BETA: &str = "thinking-display-updates-2026-08-18";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectorMode {
    None,
    ConnectorText,
    ThinkingAndConnectorText,
}

#[derive(Debug, Clone, Copy)]
pub struct UpdatesAdmission {
    pub mode: ConnectorMode,
    pub supports_interleaved: bool,
    pub extra_has_thinking: bool,
    pub simulated_proxy: bool,
}

/// Native Cjt: explicit display intent and summary preference precede the flag.
pub fn connector_mode(
    display: Option<&str>,
    explicit: bool,
    summaries: bool,
    updates: bool,
) -> ConnectorMode {
    if matches!(display, Some("summarized" | "highlights")) {
        ConnectorMode::ThinkingAndConnectorText
    } else if display == Some("omitted") && !explicit {
        ConnectorMode::None
    } else if display != Some("omitted") && summaries {
        ConnectorMode::ThinkingAndConnectorText
    } else if updates {
        ConnectorMode::ConnectorText
    } else {
        ConnectorMode::None
    }
}

/// Native host/process budget Ajt/Pjt, separate from ordinary API retries.
#[derive(Debug, Clone, Default)]
pub struct DisplayProbeBudget(Arc<AtomicU32>);
impl DisplayProbeBudget {
    /// Native probe failures are shared by API clients on one process host.
    pub fn for_process() -> Self {
        static BUDGET: std::sync::OnceLock<DisplayProbeBudget> = std::sync::OnceLock::new();
        BUDGET.get_or_init(Self::default).clone()
    }
    pub fn failures(&self) -> u32 {
        self.0.load(Ordering::Acquire)
    }
    pub fn available(&self) -> bool {
        self.failures() < 2
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeState {
    #[default]
    Idle,
    Retrying,
    Spent,
}

/// Supplied facts match the dependencies of native P4e. This policy does not
/// infer error recognition, policy denial or endpoint trust from model input.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProbeAdmission {
    pub header_sent: bool,
    pub billing_error: bool,
    pub recognized_error: bool,
    pub policy_denied: bool,
    pub trusted_first_party_endpoint: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DisplayProbe {
    state: ProbeState,
}

/// Decode complete IR and pF before an unclaimed display probe. The host
/// supplies trusted model-target/history facts and an identity resolver.
/// First-party Gft is unknown rather than an inferred model refusal.
pub fn probe_admission(
    status: u16,
    body: &Value,
    header_sent: bool,
    trusted_first_party_endpoint: bool,
    recognition: super::error_recognition::RecognitionContext<'_>,
    identity: impl Fn(&str) -> String,
) -> ProbeAdmission {
    let recognized_error =
        super::error_recognition::recognized(status, body, recognition, identity);
    ProbeAdmission {
        header_sent,
        billing_error: body["error"]["type"] == "billing_error",
        recognized_error,
        // pF blocks admission; only IR suppresses failed-trial accounting.
        policy_denied: super::error_recognition::policy_denied(status, body, false),
        trusted_first_party_endpoint,
    }
}

pub fn request_carries_updates(headers: &BTreeMap<String, String>) -> bool {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .is_some_and(|(_, value)| value.split(',').any(|beta| beta == UPDATES_BETA))
}
impl DisplayProbe {
    pub fn state(&self) -> ProbeState {
        self.state
    }
    pub fn suppress_updates(&self) -> bool {
        self.state != ProbeState::Idle
    }

    /// P4e only observes HTTP 400/422. A failed trial spends the probe and
    /// increments the process budget only if its next error is unrecognized.
    pub fn on_error(
        &mut self,
        status: u16,
        admission: ProbeAdmission,
        budget: &DisplayProbeBudget,
    ) -> bool {
        if !matches!(status, 400 | 422) {
            return false;
        }
        if self.state == ProbeState::Retrying {
            self.state = ProbeState::Spent;
            if !admission.recognized_error {
                budget.0.fetch_add(1, Ordering::AcqRel);
            }
            return false;
        }
        if self.state != ProbeState::Idle
            || !admission.header_sent
            || admission.billing_error
            || admission.recognized_error
            || admission.policy_denied
            || !budget.available()
            || admission.trusted_first_party_endpoint
        {
            return false;
        }
        self.state = ProbeState::Retrying;
        true
    }

    /// I4e commits the sticky rejection only when the trial succeeded.
    pub fn on_success(&mut self, conversation: &ConversationBetaState) -> bool {
        if self.state != ProbeState::Retrying {
            return false;
        }
        self.state = ProbeState::Spent;
        conversation.reject(Beta::ThinkingDisplayUpdates);
        true
    }
}

/// Apply the current connector-only request arm before the extra-body spread.
/// Explicit extra thinking and simulated proxy usage suppress this automatic
/// field. Inserting display preserves the existing thinking property order.
pub fn apply_updates(
    body: &mut Value,
    betas: &mut Vec<String>,
    admission: UpdatesAdmission,
    conversation: &ConversationBetaState,
    probe: &DisplayProbe,
) -> bool {
    if admission.mode != ConnectorMode::ConnectorText
        || !admission.supports_interleaved
        || admission.extra_has_thinking
        || admission.simulated_proxy
        || conversation.rejected(Beta::ThinkingDisplayUpdates)
        || probe.suppress_updates()
    {
        return false;
    }
    let Some(thinking) = body.get_mut("thinking").and_then(Value::as_object_mut) else {
        return false;
    };
    if !matches!(
        thinking.get("type").and_then(Value::as_str),
        Some("adaptive" | "enabled")
    ) {
        return false;
    }
    thinking.insert("display".into(), Value::String("updates".into()));
    if !betas.iter().any(|beta| beta == UPDATES_BETA) {
        betas.push(UPDATES_BETA.into());
    }
    betas.retain(|beta| beta != "redact-thinking-2026-02-12");
    true
}

/// Apply the display field and coupled beta to a prepared Messages request.
/// Hosts call before spreading explicit extra body and before signing/sealing.
pub fn apply_request_updates(
    body: &mut Value,
    headers: &mut BTreeMap<String, String>,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
    admission: UpdatesAdmission,
    conversation: &ConversationBetaState,
    probe: &DisplayProbe,
) -> bool {
    let mut betas: Vec<String> = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .map(|(_, value)| {
            value
                .split(',')
                .filter(|beta| !beta.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if !apply_updates(body, &mut betas, admission, conversation, probe) {
        return false;
    }
    string_overrides
        .retain(|path, _| path != "/thinking/display" && !path.starts_with("/thinking/display/"));
    super::request_policy::set_header(headers, "anthropic-beta", &betas.join(","));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/thinking_display_2_1_288.json"
        ))
        .unwrap()
    }
    fn state(value: &Value) -> ProbeState {
        serde_json::from_value(value.clone()).unwrap()
    }
    #[test]
    fn connector_selection_matches_native_cjt() {
        let fixture = fixture();
        let cases = fixture["modeCases"].as_array().unwrap();
        assert_eq!(cases.len(), 56);
        for case in cases {
            let actual = connector_mode(
                case["display"].as_str(),
                case["explicit"].as_bool().unwrap(),
                case["summaries"].as_bool().unwrap(),
                case["updates"].as_bool().unwrap(),
            );
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                case["expected"],
                "{case}"
            );
        }
    }
    #[test]
    fn known_signature_fast_and_context_repairs_precede_display_probe() {
        for case in fixture()["knownErrorCases"].as_array().unwrap() {
            let admission = probe_admission(
                case["status"].as_u64().unwrap() as u16,
                &serde_json::json!({"error":{"type":"invalid_request_error","message":case["message"]}}),
                true,
                false,
                super::super::error_recognition::RecognitionContext::default(),
                str::to_owned,
            );
            assert_eq!(
                admission.recognized_error || admission.policy_denied,
                case["expected"].as_bool().unwrap(),
                "{case}"
            );
            assert_eq!(
                admission.recognized_error,
                case["recognized"].as_bool().unwrap(),
                "{case}"
            );
            assert_eq!(
                admission.policy_denied,
                case["denied"].as_bool().unwrap(),
                "{case}"
            );
        }
    }

    #[test]
    fn automatic_request_field_replaces_only_its_owned_exact_string() {
        let mut body = serde_json::json!({"thinking":{"type":"adaptive","display":"omitted"}});
        let mut headers = BTreeMap::from([(
            "Anthropic-Beta".into(),
            "base-beta,redact-thinking-2026-02-12".into(),
        )]);
        let mut strings = BTreeMap::from([
            ("/thinking/display".into(), vec![0xD800]),
            ("/messages/0/content/0/text".into(), vec![0xD801]),
        ]);
        assert!(apply_request_updates(
            &mut body,
            &mut headers,
            &mut strings,
            UpdatesAdmission {
                mode: ConnectorMode::ConnectorText,
                supports_interleaved: true,
                extra_has_thinking: false,
                simulated_proxy: false,
            },
            &ConversationBetaState::default(),
            &DisplayProbe::default(),
        ));
        assert_eq!(body["thinking"]["display"], "updates");
        assert_eq!(
            headers["anthropic-beta"],
            format!("base-beta,{UPDATES_BETA}")
        );
        assert_eq!(headers.len(), 1);
        assert_eq!(
            strings,
            BTreeMap::from([("/messages/0/content/0/text".into(), vec![0xD801])])
        );
    }
    #[test]
    fn unclaimed_probe_matches_native_state_and_process_budget() {
        let fixture = fixture();
        let cases = fixture["probeCases"].as_array().unwrap();
        assert_eq!(cases.len(), 1920);
        for case in cases {
            let mut probe = DisplayProbe {
                state: state(&case["state"]),
            };
            let budget = DisplayProbeBudget(Arc::new(AtomicU32::new(
                case["failures"].as_u64().unwrap() as u32,
            )));
            let retry = probe.on_error(
                case["status"].as_u64().unwrap() as u16,
                ProbeAdmission {
                    header_sent: case["header_sent"].as_bool().unwrap(),
                    billing_error: case["billing_error"].as_bool().unwrap(),
                    recognized_error: case["recognized_error"].as_bool().unwrap(),
                    policy_denied: case["policy_denied"].as_bool().unwrap(),
                    trusted_first_party_endpoint: case["trusted_first_party_endpoint"]
                        .as_bool()
                        .unwrap(),
                },
                &budget,
            );
            assert_eq!(
                retry,
                case["expected"]["retry"] == "retry:thinking-display-updates-unclaimed",
                "{case}"
            );
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
    fn only_healed_trial_commits_conversation_rejection() {
        let fixture = fixture();
        for case in fixture["successCases"].as_array().unwrap() {
            let conversation = ConversationBetaState::default();
            if case["disabled"].as_bool().unwrap() {
                conversation.reject(Beta::ThinkingDisplayUpdates);
            }
            let mut probe = DisplayProbe {
                state: state(&case["state"]),
            };
            assert_eq!(probe.on_success(&conversation), case["state"] == "retrying");
            assert_eq!(
                serde_json::to_value(probe.state()).unwrap(),
                case["expected"]["state"],
                "{case}"
            );
            assert_eq!(
                conversation.rejected(Beta::ThinkingDisplayUpdates),
                case["expected"]["disabled"].as_bool().unwrap(),
                "{case}"
            );
            assert!(!ConversationBetaState::default().rejected(Beta::ThinkingDisplayUpdates));
        }
    }
    #[test]
    fn automatic_updates_wire_matches_native_before_extra_body() {
        let fixture = fixture();
        let cases = fixture["wireCases"].as_array().unwrap();
        assert_eq!(cases.len(), 192);
        for case in cases {
            let mut body = case["body"].clone();
            let mut betas: Vec<String> = serde_json::from_value(case["betas"].clone()).unwrap();
            let conversation = ConversationBetaState::default();
            if case["disabled"].as_bool().unwrap() {
                conversation.reject(Beta::ThinkingDisplayUpdates);
            }
            let probe = DisplayProbe {
                state: state(&case["state"]),
            };
            apply_updates(
                &mut body,
                &mut betas,
                UpdatesAdmission {
                    mode: ConnectorMode::ConnectorText,
                    supports_interleaved: case["enabled"].as_bool().unwrap(),
                    extra_has_thinking: case["extra"].as_bool().unwrap(),
                    simulated_proxy: case["simulated"].as_bool().unwrap(),
                },
                &conversation,
                &probe,
            );
            assert_eq!(body["thinking"], case["expected"]["thinking"], "{case}");
            assert_eq!(
                serde_json::to_string(&body["thinking"]).unwrap(),
                case["thinkingBytes"].as_str().unwrap(),
                "{case}"
            );
            assert_eq!(
                betas.join(","),
                case["betaBytes"].as_str().unwrap(),
                "{case}"
            );
            assert_eq!(
                serde_json::to_value(betas).unwrap(),
                case["expected"]["betas"],
                "{case}"
            );
        }
    }
}
