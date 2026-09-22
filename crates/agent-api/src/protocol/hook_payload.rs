//! The event-specific fields of every hook payload, by name.
//!
//! Ported from LingXi-Next's `hook_payload.rs`, which spelled each event as
//! a struct. The fields every event shares — `hook_event_name`,
//! `session_id`, `cwd`, `permission_mode` — are added by the interpreter's
//! envelope; these builders produce the rest, so the wire shape a hook
//! author reads about is decided in one file and the reducer never spells
//! a key. Optional fields are left out when absent, as the source's
//! `skip_serializing_if` did.

use serde_json::{json, Map, Value};

fn obj(pairs: Vec<(&str, Value)>) -> Value {
    let mut m = Map::new();
    for (k, v) in pairs {
        if !v.is_null() {
            m.insert(k.to_owned(), v);
        }
    }
    Value::Object(m)
}

fn opt(s: Option<&str>) -> Value {
    s.map_or(Value::Null, |s| Value::String(s.to_owned()))
}

pub fn pre_tool_use(tool_name: &str, tool_input: &Value, tool_use_id: &str) -> Value {
    obj(vec![
        ("tool_name", json!(tool_name)),
        ("tool_input", tool_input.clone()),
        ("tool_use_id", json!(tool_use_id)),
    ])
}

pub fn post_tool_use(
    tool_name: &str,
    tool_input: &Value,
    tool_response: &Value,
    tool_use_id: &str,
    duration_ms: Option<u64>,
) -> Value {
    obj(vec![
        ("tool_name", json!(tool_name)),
        ("tool_input", tool_input.clone()),
        ("tool_response", tool_response.clone()),
        ("tool_use_id", json!(tool_use_id)),
        ("duration_ms", duration_ms.map_or(Value::Null, |d| json!(d))),
    ])
}

pub fn post_tool_use_failure(
    tool_name: &str,
    tool_input: &Value,
    tool_use_id: &str,
    error: &str,
    is_interrupt: Option<bool>,
    duration_ms: Option<u64>,
) -> Value {
    obj(vec![
        ("tool_name", json!(tool_name)),
        ("tool_input", tool_input.clone()),
        ("tool_use_id", json!(tool_use_id)),
        ("error", json!(error)),
        (
            "is_interrupt",
            is_interrupt.map_or(Value::Null, |b| json!(b)),
        ),
        ("duration_ms", duration_ms.map_or(Value::Null, |d| json!(d))),
    ])
}

pub fn user_prompt_submit(prompt: &str) -> Value {
    obj(vec![("prompt", json!(prompt))])
}

pub fn stop(stop_hook_active: bool, last_assistant_message: Option<&str>) -> Value {
    obj(vec![
        ("stop_hook_active", json!(stop_hook_active)),
        ("last_assistant_message", opt(last_assistant_message)),
    ])
}

pub fn stop_failure(
    error: &str,
    error_details: Option<&str>,
    last_assistant_message: Option<&str>,
) -> Value {
    obj(vec![
        ("error", json!(error)),
        ("error_details", opt(error_details)),
        ("last_assistant_message", opt(last_assistant_message)),
    ])
}

/// `source` is `startup`, `resume`, `clear` or `compact`.
pub fn session_start(source: &str, model: Option<&str>) -> Value {
    obj(vec![("source", json!(source)), ("model", opt(model))])
}

/// `reason` is `clear`, `logout`, `prompt_input_exit` or `other`.
pub fn session_end(reason: &str) -> Value {
    obj(vec![("reason", json!(reason))])
}

pub fn setup(trigger: &str) -> Value {
    obj(vec![("trigger", json!(trigger))])
}

pub fn pre_compact(trigger: &str, custom_instructions: Option<&str>) -> Value {
    obj(vec![
        ("trigger", json!(trigger)),
        ("custom_instructions", opt(custom_instructions)),
    ])
}

pub fn post_compact(trigger: &str, compact_summary: &str) -> Value {
    obj(vec![
        ("trigger", json!(trigger)),
        ("compact_summary", json!(compact_summary)),
    ])
}

pub fn permission_request(
    tool_name: &str,
    tool_input: &Value,
    suggestions: Option<&[Value]>,
) -> Value {
    obj(vec![
        ("tool_name", json!(tool_name)),
        ("tool_input", tool_input.clone()),
        (
            "permission_suggestions",
            suggestions.map_or(Value::Null, |s| json!(s)),
        ),
    ])
}

pub fn permission_denied(
    tool_name: &str,
    tool_input: &Value,
    tool_use_id: &str,
    reason: &str,
) -> Value {
    obj(vec![
        ("tool_name", json!(tool_name)),
        ("tool_input", tool_input.clone()),
        ("tool_use_id", json!(tool_use_id)),
        ("reason", json!(reason)),
    ])
}

/// One call of a batch: `tool_name`, `tool_input`, `tool_use_id`, and
/// `tool_response` when it finished.
pub fn post_tool_batch(tool_calls: Vec<Value>) -> Value {
    obj(vec![("tool_calls", Value::Array(tool_calls))])
}

pub fn notification(message: &str, title: Option<&str>, notification_type: &str) -> Value {
    obj(vec![
        ("message", json!(message)),
        ("title", opt(title)),
        ("notification_type", json!(notification_type)),
    ])
}

pub fn cwd_changed(old_cwd: &str, new_cwd: &str) -> Value {
    obj(vec![
        ("old_cwd", json!(old_cwd)),
        ("new_cwd", json!(new_cwd)),
    ])
}

pub fn instructions_loaded(
    file_path: &str,
    memory_type: &str,
    load_reason: &str,
    globs: Option<&[String]>,
    trigger_file_path: Option<&str>,
    parent_file_path: Option<&str>,
) -> Value {
    obj(vec![
        ("file_path", json!(file_path)),
        ("memory_type", json!(memory_type)),
        ("load_reason", json!(load_reason)),
        ("globs", globs.map_or(Value::Null, |g| json!(g))),
        ("trigger_file_path", opt(trigger_file_path)),
        ("parent_file_path", opt(parent_file_path)),
    ])
}

pub fn subagent_start(agent_id: &str, agent_type: &str) -> Value {
    obj(vec![
        ("agent_id", json!(agent_id)),
        ("agent_type", json!(agent_type)),
    ])
}

pub fn subagent_stop(
    agent_id: &str,
    agent_type: &str,
    status: &str,
    last_assistant_message: Option<&str>,
) -> Value {
    obj(vec![
        ("agent_id", json!(agent_id)),
        ("agent_type", json!(agent_type)),
        ("status", json!(status)),
        ("last_assistant_message", opt(last_assistant_message)),
    ])
}

pub fn agent_spawn(
    agent_type: &str,
    model: Option<&str>,
    background: bool,
    parent_agent_id: Option<&str>,
) -> Value {
    obj(vec![
        ("agent_type", json!(agent_type)),
        ("model", opt(model)),
        ("background", json!(background)),
        ("parent_agent_id", opt(parent_agent_id)),
    ])
}

pub fn config_change(source: &str, file_path: Option<&str>) -> Value {
    obj(vec![
        ("source", json!(source)),
        ("file_path", opt(file_path)),
    ])
}

pub fn directory_added(directory: &str, source: &str) -> Value {
    obj(vec![
        ("directory", json!(directory)),
        ("source", json!(source)),
    ])
}

pub fn file_changed(file_path: &str, event: &str) -> Value {
    obj(vec![
        ("file_path", json!(file_path)),
        ("event", json!(event)),
    ])
}

pub fn worktree_create(name: &str) -> Value {
    obj(vec![("name", json!(name))])
}

pub fn worktree_remove(worktree_path: &str) -> Value {
    obj(vec![("worktree_path", json!(worktree_path))])
}

pub fn elicitation(
    mcp_server_name: &str,
    message: &str,
    mode: Option<&str>,
    url: Option<&str>,
    elicitation_id: Option<&str>,
    requested_schema: Option<&Value>,
) -> Value {
    obj(vec![
        ("mcp_server_name", json!(mcp_server_name)),
        ("message", json!(message)),
        ("mode", opt(mode)),
        ("url", opt(url)),
        ("elicitation_id", opt(elicitation_id)),
        (
            "requested_schema",
            requested_schema.cloned().unwrap_or(Value::Null),
        ),
    ])
}

pub fn elicitation_result(
    mcp_server_name: &str,
    elicitation_id: Option<&str>,
    mode: Option<&str>,
    action: &str,
    content: Option<&Value>,
) -> Value {
    obj(vec![
        ("mcp_server_name", json!(mcp_server_name)),
        ("elicitation_id", opt(elicitation_id)),
        ("mode", opt(mode)),
        ("action", json!(action)),
        ("content", content.cloned().unwrap_or(Value::Null)),
    ])
}

/// Shared by `PreModelSwitch` and `PostModelSwitch`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSwitch<'a> {
    pub from_model: &'a str,
    pub to_model: &'a str,
    pub requested_model: Option<&'a str>,
    pub source: &'a str,
    pub context_tokens: u64,
    pub prompt_cache_warm: bool,
    pub cache_ttl: &'a str,
    pub estimated_cache_write_usd: f64,
    pub pricing: &'a str,
}

pub fn model_switch(s: &ModelSwitch<'_>) -> Value {
    obj(vec![
        ("from_model", json!(s.from_model)),
        ("to_model", json!(s.to_model)),
        ("requested_model", opt(s.requested_model)),
        ("source", json!(s.source)),
        ("context_tokens", json!(s.context_tokens)),
        ("prompt_cache_warm", json!(s.prompt_cache_warm)),
        ("cache_ttl", json!(s.cache_ttl)),
        (
            "estimated_cache_write_usd",
            json!(s.estimated_cache_write_usd),
        ),
        ("pricing", json!(s.pricing)),
    ])
}

pub fn teammate_idle(teammate_name: &str, team_name: &str) -> Value {
    obj(vec![
        ("teammate_name", json!(teammate_name)),
        ("team_name", json!(team_name)),
    ])
}

pub fn task_created(
    task_id: &str,
    task_subject: &str,
    task_description: Option<&str>,
    teammate_name: Option<&str>,
    team_name: Option<&str>,
) -> Value {
    obj(vec![
        ("task_id", json!(task_id)),
        ("task_subject", json!(task_subject)),
        ("task_description", opt(task_description)),
        ("teammate_name", opt(teammate_name)),
        ("team_name", opt(team_name)),
    ])
}

pub fn task_completed(
    task_id: &str,
    status: &str,
    task_subject: &str,
    task_description: Option<&str>,
    teammate_name: Option<&str>,
    team_name: Option<&str>,
) -> Value {
    obj(vec![
        ("task_id", json!(task_id)),
        ("status", json!(status)),
        ("task_subject", json!(task_subject)),
        ("task_description", opt(task_description)),
        ("teammate_name", opt(teammate_name)),
        ("team_name", opt(team_name)),
    ])
}

pub fn user_prompt_expansion(
    expansion_type: &str,
    command_name: &str,
    command_args: &str,
    command_source: Option<&str>,
    prompt: &str,
) -> Value {
    obj(vec![
        ("expansion_type", json!(expansion_type)),
        ("command_name", json!(command_name)),
        ("command_args", json!(command_args)),
        ("command_source", opt(command_source)),
        ("prompt", json!(prompt)),
    ])
}

pub fn message_display(
    turn_id: &str,
    message_id: &str,
    index: u64,
    is_final: bool,
    delta: &str,
) -> Value {
    obj(vec![
        ("turn_id", json!(turn_id)),
        ("message_id", json!(message_id)),
        ("index", json!(index)),
        ("is_final", json!(is_final)),
        ("delta", json!(delta)),
    ])
}

/// This project's own event (§10): a work mode changed.
pub fn mode_changed(from: &str, to: &str, by: &str) -> Value {
    obj(vec![
        ("from", json!(from)),
        ("to", json!(to)),
        ("by", json!(by)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(v: &Value) -> Vec<&str> {
        v.as_object().unwrap().keys().map(String::as_str).collect()
    }

    #[test]
    fn every_builder_spells_the_source_fields_and_drops_absent_options() {
        assert_eq!(
            keys(&pre_tool_use("Bash", &json!({}), "t1")),
            ["tool_name", "tool_input", "tool_use_id"]
        );
        assert_eq!(
            keys(&post_tool_use(
                "Bash",
                &json!({}),
                &json!("ok"),
                "t1",
                Some(3)
            )),
            [
                "tool_name",
                "tool_input",
                "tool_response",
                "tool_use_id",
                "duration_ms"
            ]
        );
        assert_eq!(
            keys(&post_tool_use_failure(
                "Bash",
                &json!({}),
                "t1",
                "boom",
                None,
                None
            )),
            ["tool_name", "tool_input", "tool_use_id", "error"]
        );
        assert_eq!(keys(&user_prompt_submit("hi")), ["prompt"]);
        assert_eq!(keys(&stop(false, None)), ["stop_hook_active"]);
        assert_eq!(
            keys(&stop_failure("rate_limit", Some("429"), None)),
            ["error", "error_details"]
        );
        assert_eq!(
            keys(&session_start("startup", Some("m"))),
            ["source", "model"]
        );
        assert_eq!(keys(&session_end("other")), ["reason"]);
        assert_eq!(keys(&pre_compact("auto", None)), ["trigger"]);
        assert_eq!(
            keys(&post_compact("manual", "summary")),
            ["trigger", "compact_summary"]
        );
        assert_eq!(
            keys(&permission_request("Bash", &json!({}), None)),
            ["tool_name", "tool_input"]
        );
        assert_eq!(
            keys(&permission_denied("Bash", &json!({}), "t1", "no")),
            ["tool_name", "tool_input", "tool_use_id", "reason"]
        );
        assert_eq!(keys(&cwd_changed("/a", "/b")), ["old_cwd", "new_cwd"]);
        assert_eq!(
            keys(&instructions_loaded(
                "/p/rules.md",
                "Project",
                "session_start",
                None,
                None,
                None
            )),
            ["file_path", "memory_type", "load_reason"]
        );
        assert_eq!(
            keys(&notification("m", None, "idle")),
            ["message", "notification_type"]
        );
        assert_eq!(
            keys(&subagent_start("a1", "Explore")),
            ["agent_id", "agent_type"]
        );
        assert_eq!(
            keys(&elicitation("srv", "m", None, None, None, None)),
            ["mcp_server_name", "message"]
        );
        let sw = ModelSwitch {
            from_model: "a",
            to_model: "b",
            requested_model: None,
            source: "user",
            context_tokens: 1,
            prompt_cache_warm: false,
            cache_ttl: "5m",
            estimated_cache_write_usd: 0.0,
            pricing: "same",
        };
        assert_eq!(keys(&model_switch(&sw)).len(), 8);
        assert_eq!(
            keys(&mode_changed("chat", "code", "user")),
            ["from", "to", "by"]
        );
    }
}
