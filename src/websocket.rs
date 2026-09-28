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
    /// Authoritative completed output, not output_item.added placeholders.
    pub last_response_output: Option<Vec<serde_json::Value>>,
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
    // An explicit continuation selects a branch, including a deliberate null
    // that starts a new one. Only implicit full-history requests may be shrunk
    // against the session's cached response.
    if logical_body.get("previous_response_id").is_some() {
        return None;
    }
    let state = state.lock().expect("responses ws state");
    let previous_id = state.last_response_id.as_ref()?;
    let previous_body = state.last_request_body.as_ref()?;
    // A cached response built on an explicit branch is not a representation of
    // this caller's full history and cannot seed a later implicit continuation.
    if previous_body.get("previous_response_id").is_some() {
        return None;
    }
    if non_input_responses_body(previous_body) != non_input_responses_body(logical_body) {
        return None;
    }
    let previous_input = previous_body.get("input")?.as_array()?;
    let current_input = logical_body.get("input")?.as_array()?;
    let previous_output = state.last_response_output.as_ref()?;
    let prefix_len = previous_input.len().checked_add(previous_output.len())?;
    if current_input.len() < prefix_len {
        return None;
    }
    if !previous_input
        .iter()
        .zip(current_input.iter())
        .all(|(previous, current)| previous == current)
    {
        return None;
    }

    if !previous_output
        .iter()
        .zip(&current_input[previous_input.len()..prefix_len])
        .all(|(output, replay)| {
            response_item_for_comparison(output) == response_item_for_comparison(replay)
        })
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
        serde_json::Value::Array(current_input[prefix_len..].to_vec()),
    );
    Some(body)
}

// Strip only response-envelope/rendering metadata for item types whose replay
// representation is known. Unknown items require an exact match; uncertainty
// falls back to full history instead of guessing a continuation prefix.
fn response_item_for_comparison(item: &serde_json::Value) -> serde_json::Value {
    let mut item = item.clone();
    let Some(map) = item.as_object_mut() else {
        return item;
    };
    let message = map.get("type").and_then(serde_json::Value::as_str) == Some("message")
        || (!map.contains_key("type")
            && map.get("role").and_then(serde_json::Value::as_str) == Some("assistant"));
    let function = map.get("type").and_then(serde_json::Value::as_str) == Some("function_call");
    if message || function {
        map.remove("id");
        map.remove("status");
    }
    if message {
        map.insert("type".into(), serde_json::json!("message"));
        if let Some(parts) = map
            .get_mut("content")
            .and_then(serde_json::Value::as_array_mut)
        {
            for part in parts {
                if part.get("type").and_then(serde_json::Value::as_str) == Some("output_text") {
                    if let Some(part) = part.as_object_mut() {
                        part.remove("annotations");
                        part.remove("logprobs");
                    }
                }
            }
        }
    }
    item
}

pub fn record_responses_wire_request(
    state: &Arc<Mutex<ResponsesWebSocketSessionState>>,
    logical_body: &serde_json::Value,
    wire_body: &serde_json::Value,
) {
    let mut state = state.lock().expect("responses ws state");
    let previous_response_id = wire_body
        .get("previous_response_id")
        .and_then(serde_json::Value::as_str);
    let used_previous_response_id = previous_response_id.is_some();
    state.last_logical_request_body = Some(logical_body.clone());
    state.last_wire_request_body = Some(wire_body.clone());
    state.last_wire_used_previous_response_id = used_previous_response_id;
    state.last_wire_used_prewarm_response_id = used_previous_response_id
        && state.last_response_from_prewarm
        && previous_response_id == state.last_response_id.as_deref();
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
            state.last_response_output = root
                .get("response")
                .and_then(|response| response.get("output"))
                .and_then(serde_json::Value::as_array)
                .cloned()
                .or_else(|| from_prewarm.then(Vec::new));
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
    state.last_response_output = None;
    state.last_response_from_prewarm = false;
    if connection_unhealthy {
        state.connection_healthy = false;
    }
}

#[cfg(test)]
mod continuation_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn continuation_requires_completed_output_and_matching_full_prefix() {
        let state = Arc::new(Mutex::new(ResponsesWebSocketSessionState::default()));
        let input =
            json!({"type":"message","role":"user","content":[{"type":"input_text","text":"Hi"}]});
        let call =
            json!({"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"});
        let output = json!({"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"lookup","arguments":"{}"});
        let result = json!({"type":"function_call_output","call_id":"call_1","output":"done"});
        let previous = json!({"model":"wire","input":[input.clone()]});
        let current = json!({"model":"wire","input":[input.clone(),call.clone(),result.clone()]});
        let observe = |frame| {
            observe_response_frame(
                &state,
                &previous,
                false,
                &mut vec![],
                &mut false,
                &serde_json::to_vec(&frame).unwrap(),
            )
        };
        // Missing authoritative output must not be confused with an empty output.
        observe(json!({"type":"response.completed","response":{"id":"r1"}}));
        assert!(incremental_responses_body(&state, &current).is_none());
        observe(json!({"type":"response.completed","response":{"id":"r1","output":[output]}}));
        assert_eq!(
            incremental_responses_body(&state, &current).unwrap()["input"],
            json!([result])
        );
        let mut changed = current.clone();
        changed["input"][1]["arguments"] = json!("{\"different\":true}");
        assert!(incremental_responses_body(&state, &changed).is_none());
        assert!(incremental_responses_body(&state, &previous).is_none());
        clear_continuation(&state, false);
        assert!(incremental_responses_body(&state, &current).is_none());
    }
}
