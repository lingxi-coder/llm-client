//! Responses WebSocket envelopes and turn-scoped continuation state.
//! Hosts own connection lifetime and retry/admission decisions.
use crate::protocol::LlmError;
use std::sync::{Arc, Mutex};

/// Encode one Responses request as a WebSocket response.create message.
pub fn request_payload(body: &[u8]) -> Result<bytes::Bytes, LlmError> {
    let mut value: serde_json::Value =
        serde_json::from_slice(body).map_err(|_| LlmError::InvalidRequest {
            message: "Responses request is not valid JSON".into(),
        })?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "Responses request must be an object".into(),
        })?;
    object.remove("stream");
    object.insert("type".into(), serde_json::json!("response.create"));
    serde_json::to_vec(&value)
        .map(Into::into)
        .map_err(|_| LlmError::InvalidRequest {
            message: "Responses request cannot be serialized".into(),
        })
}

#[derive(Debug, Default)]
pub struct ResponsesWebSocketSessionState {
    pub fallback_to_http: bool,
    pub connection_healthy: bool,
    pub last_request_body: Option<serde_json::Value>,
    pub last_response_id: Option<String>,
    pub last_added_response_items: Vec<serde_json::Value>,
    pub last_response_from_prewarm: bool,
    pub last_logical_request_body: Option<serde_json::Value>,
    pub last_wire_request_body: Option<serde_json::Value>,
    pub last_wire_used_previous_response_id: bool,
    pub last_wire_used_prewarm_response_id: bool,
}

/// Debug/telemetry snapshot for the most recent Responses WebSocket send.
///
/// `logical_request_body` is the full model-visible request the caller meant
/// to send. `wire_request_body` is the compressed WebSocket payload body that
/// may contain `previous_response_id` and only newly-added `input` items.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResponsesWebSocketRequestSnapshot {
    pub logical_request_body: Option<serde_json::Value>,
    pub wire_request_body: Option<serde_json::Value>,
    pub wire_used_previous_response_id: bool,
    pub wire_used_prewarm_response_id: bool,
}

pub fn set_responses_generate(
    body: &mut serde_json::Value,
    generate: bool,
) -> Result<(), LlmError> {
    let serde_json::Value::Object(map) = body else {
        return Err(LlmError::InvalidRequest {
            message: "OpenAI Responses request body must be a JSON object".to_string(),
        });
    };
    map.insert("generate".to_string(), serde_json::Value::Bool(generate));
    Ok(())
}

pub fn incremental_responses_body(
    state: &Arc<Mutex<ResponsesWebSocketSessionState>>,
    logical_body: &serde_json::Value,
) -> Option<serde_json::Value> {
    let state = state.lock().expect("responses ws state");
    let previous_id = state.last_response_id.as_ref()?;
    let previous_body = state.last_request_body.as_ref()?;
    if non_input_responses_body(previous_body) != non_input_responses_body(logical_body) {
        return None;
    }
    let previous_input = previous_body.get("input")?.as_array()?;
    let current_input = logical_body.get("input")?.as_array()?;
    if current_input.len() < previous_input.len() {
        return None;
    }
    if !previous_input
        .iter()
        .zip(current_input.iter())
        .all(|(previous, current)| previous == current)
    {
        return None;
    }

    let mut body = logical_body.clone();
    let serde_json::Value::Object(map) = &mut body else {
        return None;
    };
    map.insert(
        "previous_response_id".to_string(),
        serde_json::Value::String(previous_id.clone()),
    );
    map.insert(
        "input".to_string(),
        serde_json::Value::Array(current_input[previous_input.len()..].to_vec()),
    );
    Some(body)
}

pub fn record_responses_wire_request(
    state: &Arc<Mutex<ResponsesWebSocketSessionState>>,
    logical_body: &serde_json::Value,
    wire_body: &serde_json::Value,
) {
    let mut state = state.lock().expect("responses ws state");
    let used_previous_response_id = wire_body.get("previous_response_id").is_some();
    state.last_logical_request_body = Some(logical_body.clone());
    state.last_wire_request_body = Some(wire_body.clone());
    state.last_wire_used_previous_response_id = used_previous_response_id;
    state.last_wire_used_prewarm_response_id =
        used_previous_response_id && state.last_response_from_prewarm;
}

fn non_input_responses_body(body: &serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(map) = body else {
        return body.clone();
    };
    let mut copy = map.clone();
    copy.remove("input");
    copy.remove("previous_response_id");
    copy.remove("generate");
    serde_json::Value::Object(copy)
}

/// Observe one raw JSON frame and retain continuation state only after completion.
pub fn observe_response_frame(
    state: &Arc<Mutex<ResponsesWebSocketSessionState>>,
    logical_body: &serde_json::Value,
    from_prewarm: bool,
    items_added: &mut Vec<serde_json::Value>,
    terminal_seen: &mut bool,
    bytes: &[u8],
) {
    let Ok(root) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return;
    };
    match root.get("type").and_then(serde_json::Value::as_str) {
        Some("response.output_item.added") => {
            if let Some(item) = root.get("item") {
                items_added.push(item.clone());
            }
        }
        Some("response.completed") => {
            *terminal_seen = true;
            let response_id = root
                .get("response")
                .and_then(|response| response.get("id"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            let mut state = state.lock().expect("responses ws state");
            state.last_request_body = Some(logical_body.clone());
            state.last_response_id = response_id;
            state.last_added_response_items = items_added.clone();
            state.last_response_from_prewarm = from_prewarm;
            state.connection_healthy = true;
        }
        Some("response.incomplete" | "response.failed" | "error") => {
            *terminal_seen = true;
            clear_continuation(
                state,
                matches!(
                    root.get("type").and_then(serde_json::Value::as_str),
                    Some("response.failed" | "error")
                ),
            );
        }
        _ => {}
    }
}
/// Invalidate continuation after interruption or a failed response.
pub fn clear_continuation(
    state: &Arc<Mutex<ResponsesWebSocketSessionState>>,
    connection_unhealthy: bool,
) {
    let mut state = state.lock().expect("responses ws state");
    state.last_request_body = None;
    state.last_response_id = None;
    state.last_added_response_items.clear();
    state.last_response_from_prewarm = false;
    if connection_unhealthy {
        state.connection_healthy = false;
    }
}
