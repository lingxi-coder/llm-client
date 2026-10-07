//! Native NHt request parameters and conversation-owned server beta latches.
//! The caller supplies a lane admitted by its model/feature policy.
use super::request_policy::merge_beta_header;
use crate::protocol::LlmError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub const EXPLICIT_BETA: &str = "server-side-fallback-2026-06-01";
pub const DEFAULT_BETA: &str = "server-side-fallback-2026-07-01";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneMode {
    #[default]
    Explicit,
    Default,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerLane {
    pub for_model: String,
    pub model: String,
    pub mode: LaneMode,
}

/// The native July server-fallback beta rejection causes that can repair a
/// conversation-scoped default-lane latch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerFallbackRepairCause {
    CategoryBetaHeader,
    DefaultUnconfigured,
}

/// A wire-grounded repair fact. The host owns the conversation beta latch and
/// the query-local retry loop; `rejected_mode` names the beta being retired,
/// not necessarily the mode of the current lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerFallbackRepairFacts {
    pub cause: ServerFallbackRepairCause,
    pub rejected_mode: LaneMode,
}

/// Classify the two July server-fallback causes recognized by native bMe.
/// This intentionally observes only the API error status and message; request
/// preparation and retry eligibility are evaluated separately below.
pub fn classify_server_fallback_rejection(
    status: u16,
    error_body: &Value,
) -> Option<ServerFallbackRepairCause> {
    if status != 400 {
        return None;
    }

    let message = super::beta_repair::error_message(status, error_body);
    if message.contains(&format!("`{DEFAULT_BETA}`")) && message.contains("anthropic-beta") {
        return Some(ServerFallbackRepairCause::CategoryBetaHeader);
    }
    // Native bMe returns the generic beta_header cause before it considers
    // default_unconfigured. That generic cause does not enter the category
    // strip retry branch handled by this module.
    if message.contains("`server-side-fallback-") && message.contains("anthropic-beta") {
        return None;
    }
    message
        .contains("has no default fallback configuration")
        .then_some(ServerFallbackRepairCause::DefaultUnconfigured)
}

/// Apply native's server-fallback retry guard to an already classified error.
/// `fallback_parameter_added` is a1: whether the lane parameter survived the
/// SDK transport gate before extra-body parameters are merged. Category beta
/// errors do not depend on a1; default-unconfigured errors do. The caller owns
/// the query-local one-shot latch represented by `repair_already_attempted`.
pub fn server_fallback_repair_facts(
    cause: ServerFallbackRepairCause,
    fallback_parameter_added: bool,
    repair_already_attempted: bool,
) -> Option<ServerFallbackRepairFacts> {
    let retry_guard_matches = !repair_already_attempted
        && (cause == ServerFallbackRepairCause::CategoryBetaHeader || fallback_parameter_added);
    retry_guard_matches.then_some(ServerFallbackRepairFacts {
        cause,
        rejected_mode: LaneMode::Default,
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BetaSnapshot {
    pub explicit_active: bool,
    pub default_active: bool,
    pub explicit_rejected: bool,
    pub default_rejected: bool,
}

/// This identity follows one conversation, independently of a shared client.
#[derive(Debug, Clone, Default)]
pub struct ServerBetaState(Arc<Mutex<BetaSnapshot>>);
impl PartialEq for ServerBetaState {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl ServerBetaState {
    pub fn snapshot(&self) -> BetaSnapshot {
        *self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    pub fn from_snapshot(snapshot: BetaSnapshot) -> Self {
        Self(Arc::new(Mutex::new(snapshot)))
    }
    pub fn reject(&self, mode: LaneMode) {
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match mode {
            LaneMode::Explicit => {
                state.explicit_rejected = true;
                state.explicit_active = false;
            }
            LaneMode::Default => {
                state.default_rejected = true;
                state.default_active = false;
            }
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RequestPolicy {
    pub lane: Option<ServerLane>,
    /// Native IFo has already resolved model identity and category targets.
    pub explicit_target_eligible: bool,
    pub silent_arm: bool,
    pub threaded_request: bool,
    pub beta_transport_enabled: bool,
    pub simulated_proxy_usage: bool,
}

impl RequestPolicy {
    /// Run before the native extra-body spread and request sealing. The model
    /// comparison is raw equality; wire target normalization is native Mn.
    pub fn apply(
        &self,
        request_model: &str,
        first_party: bool,
        state: &ServerBetaState,
        body: &mut Value,
        headers: &mut BTreeMap<String, String>,
        string_overrides: &mut BTreeMap<String, Vec<u16>>,
    ) -> Result<(), LlmError> {
        let mut snapshot = state
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let base = self
            .lane
            .as_ref()
            .is_some_and(|lane| lane.for_model == request_model)
            && !self.threaded_request
            && first_party
            && !snapshot.explicit_rejected;
        let default = base
            && !snapshot.default_rejected
            && self
                .lane
                .as_ref()
                .is_some_and(|lane| lane.mode == LaneMode::Default);
        let admitted = base && (default || self.explicit_target_eligible);
        if admitted {
            if default {
                snapshot.default_active = true;
            } else {
                snapshot.explicit_active = true;
            }
        }
        let mut betas = Vec::new();
        if !self.silent_arm && self.beta_transport_enabled && !self.simulated_proxy_usage {
            for (active, rejected, beta) in [
                (
                    snapshot.explicit_active,
                    snapshot.explicit_rejected,
                    EXPLICIT_BETA,
                ),
                (
                    snapshot.default_active,
                    snapshot.default_rejected,
                    DEFAULT_BETA,
                ),
            ] {
                if active && !rejected {
                    betas.push(beta.to_owned());
                }
            }
        }
        drop(snapshot);
        if !self.beta_transport_enabled || self.simulated_proxy_usage {
            let mut retained = super::beta_repair::request_betas(headers);
            retained.retain(|beta| beta != EXPLICIT_BETA && beta != DEFAULT_BETA);
            headers.retain(|key, _| !key.eq_ignore_ascii_case("anthropic-beta"));
            if !retained.is_empty() {
                super::request_policy::set_header(headers, "anthropic-beta", &retained.join(","));
            }
        }
        if !betas.is_empty() {
            merge_beta_header(headers, &betas);
        }
        if !admitted || !self.beta_transport_enabled || self.simulated_proxy_usage {
            return Ok(());
        }
        let selected_beta = if default { DEFAULT_BETA } else { EXPLICIT_BETA };
        if !super::beta_repair::request_betas(headers)
            .iter()
            .any(|beta| beta == selected_beta)
        {
            return Ok(());
        }
        let lane = self.lane.as_ref().expect("admitted lane");
        let parameters = if default {
            json!("default")
        } else {
            json!([{"model":wire_model(&lane.model)}])
        };
        body.as_object_mut()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Messages request body must be an object".into(),
            })?
            .insert("fallbacks".into(), parameters);
        string_overrides.retain(|path, _| path != "/fallbacks" && !path.starts_with("/fallbacks/"));
        Ok(())
    }
}

fn wire_model(model: &str) -> String {
    let bytes = model.as_bytes();
    let mut out = String::with_capacity(model.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index..].len() >= 4
            && bytes[index] == b'['
            && matches!(bytes[index + 1], b'1' | b'2')
            && bytes[index + 2].eq_ignore_ascii_case(&b'm')
            && bytes[index + 3] == b']'
        {
            index += 4;
            continue;
        }
        let character = model[index..].chars().next().expect("UTF-8 boundary");
        out.push(character);
        index += character.len_utf8();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_native_request_and_transport_beta_gates_match() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/fallback_request_2_1_288.json"
        ))
        .unwrap();
        for case in fixture["requests"].as_array().unwrap() {
            let policy: RequestPolicy = serde_json::from_value(case["policy"].clone()).unwrap();
            let state = ServerBetaState::from_snapshot(
                serde_json::from_value(case["initial"].clone()).unwrap(),
            );
            let mut headers = BTreeMap::new();
            let initial = case["initialBetas"]
                .as_array()
                .unwrap()
                .iter()
                .map(|beta| beta.as_str().unwrap())
                .collect::<Vec<_>>()
                .join(",");
            if !initial.is_empty() {
                headers.insert("anthropic-beta".into(), initial);
            }
            let mut body = json!({"model":case["model"]});
            let mut strings = BTreeMap::from([("/fallbacks/0/model".into(), vec![b'x' as u16])]);
            policy
                .apply(
                    case["model"].as_str().unwrap(),
                    case["firstParty"].as_bool().unwrap(),
                    &state,
                    &mut body,
                    &mut headers,
                    &mut strings,
                )
                .unwrap();
            assert_eq!(
                body.get("fallbacks"),
                case["expected"]["parameters"].get("fallbacks"),
                "{case}"
            );
            assert_eq!(
                json!(super::super::beta_repair::request_betas(&headers)),
                case["expected"]["betas"],
                "{case}"
            );
            assert_eq!(
                serde_json::to_value(state.snapshot()).unwrap(),
                case["expected"]["state"],
                "{case}"
            );
            if body.get("fallbacks").is_some() {
                assert!(strings.is_empty());
            }
        }
    }
}
