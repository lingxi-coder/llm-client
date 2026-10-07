//! Wire merging for explicit host policy overrides. Environment and feature decisions remain in the host.
use crate::protocol::LlmError;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Explicit caller identity; preserving an authenticator's identity is the
/// default for providers that require their own User-Agent.
#[derive(Debug, Clone, Copy)]
pub enum UserAgentPolicy<'a> {
    Replace(&'a str),
    IfAbsent(&'a str),
}

/// Set one HTTP header, replacing any differently-cased spelling.
pub fn set_header(headers: &mut BTreeMap<String, String>, name: &str, value: &str) {
    headers.retain(|key, _| !key.eq_ignore_ascii_case(name));
    headers.insert(name.to_ascii_lowercase(), value.to_owned());
}

pub fn apply_user_agent(headers: &mut BTreeMap<String, String>, policy: UserAgentPolicy<'_>) {
    let value = match policy {
        UserAgentPolicy::Replace(value) => value,
        UserAgentPolicy::IfAbsent(value) => {
            if headers
                .keys()
                .any(|key| key.eq_ignore_ascii_case("user-agent"))
            {
                return;
            }
            value
        }
    };
    set_header(headers, "user-agent", value);
}

/// Preserve existing (including auth-injected) betas first, then append the
/// caller-selected values, without duplicate beta tokens or header spellings.
pub fn merge_beta_header(headers: &mut BTreeMap<String, String>, betas: &[String]) {
    let mut parts = Vec::<String>::new();
    for value in headers
        .iter()
        .filter(|(key, _)| key.eq_ignore_ascii_case("anthropic-beta"))
        .map(|(_, value)| value.as_str())
        .chain(betas.iter().map(String::as_str))
    {
        for beta in value
            .split(',')
            .map(str::trim)
            .filter(|beta| !beta.is_empty())
        {
            if !parts.iter().any(|part| part == beta) {
                parts.push(beta.to_owned());
            }
        }
    }
    set_header(headers, "anthropic-beta", &parts.join(","));
}

/// Wire policy chosen by the host. Apply before request sealing/signing.
/// The SDK neither reads flags/environment nor makes routing/retry decisions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AnthropicRequestKind {
    #[default]
    Main,
    SideQuery,
    /// Native `hook_prompt` classifier/completion request. The exact JSON
    /// serializer applies Native's Anthropic body `toWellFormed` behavior at
    /// the final egress boundary for this request kind only.
    HookPrompt,
}

/// Native YMe admission chosen by the host for the selected model. Wire policy
/// does not resolve catalogs, flags, session effort or provider capability.
#[derive(Debug, Clone)]
pub struct AnthropicEffortPolicy {
    pub supported: bool,
    /// Host-resolved YMe input. None means undefined; nonstrings have no wire
    /// effort. Main explicit config is inspected before this automatic value.
    pub value: Option<Value>,
}

#[derive(Debug, Clone, Default)]
pub struct AnthropicRequestPolicy {
    pub request_kind: AnthropicRequestKind,
    pub extra_body: Map<String, Value>,
    pub body_betas: Vec<String>,
    pub effort: Option<AnthropicEffortPolicy>,
    /// Messages SDK virtual parameters travel in headers, including hosted variants.
    pub message_header_parameters: bool,
    /// A beta token whose fast-mode fields must be removed for this route.
    pub disallowed_fast_beta: Option<String>,
}

/// Native nonstream fallback owns the final stream flag after the extra-body
/// spread. Preserve an existing property's position and clear its exact-string
/// sidecars before serialization. Call before sealing or authentication.
pub fn nonstream_fallback_parameters(
    body: &mut Value,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
) -> Result<(), LlmError> {
    let body = body
        .as_object_mut()
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "Messages request body must be an object".into(),
        })?;
    body.insert("stream".into(), Value::Bool(false));
    string_overrides.retain(|path, _| path != "/stream" && !path.starts_with("/stream/"));
    Ok(())
}

/// Native extra-body cleanup, before any automatic feature decisions. Policy
/// admission and environment access belong to the host; wire cleanup belongs here.
pub fn sanitize_extra_body(mut extra: Map<String, Value>) -> Result<Map<String, Value>, LlmError> {
    for key in ["betas", "anthropic_beta"] {
        let Some(value) = extra.get(key) else {
            continue;
        };
        let text = javascript_string(value)?;
        let tokens: Vec<_> = text
            .split(',')
            .map(javascript_trim)
            .filter(|s| !s.is_empty())
            .collect();
        let retained: Vec<_> = tokens
            .iter()
            .copied()
            .filter(|s| !s.to_lowercase().contains("afk-mode"))
            .collect();
        if retained.len() == tokens.len() {
            continue;
        }
        if retained.is_empty() {
            extra.shift_remove(key);
        } else {
            extra.insert(
                key.into(),
                Value::Array(
                    retained
                        .into_iter()
                        .map(|s| Value::String(s.into()))
                        .collect(),
                ),
            );
        }
    }
    if let Some(Value::Object(metadata)) = extra.get_mut("metadata") {
        if let Some(Value::String(user_id)) = metadata.get_mut("user_id") {
            if let Some(cleaned) = super::metadata_json::remove_identity_tk(user_id) {
                *user_id = cleaned;
            }
        }
    }
    Ok(extra)
}

fn javascript_trim(text: &str) -> &str {
    text.trim_matches(|c| matches!(c, '\u{0009}'..='\u{000D}' | '\u{0020}' | '\u{00A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}' | '\u{2028}' | '\u{2029}' | '\u{202F}' | '\u{205F}' | '\u{3000}' | '\u{FEFF}'))
}

fn javascript_string(value: &Value) -> Result<String, LlmError> {
    Ok(match value {
        Value::Null => "null".into(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => javascript_number(value),
        Value::String(value) => value.clone(),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                if item.is_null() {
                    Ok(String::new())
                } else {
                    javascript_string(item)
                }
            })
            .collect::<Result<Vec<_>, _>>()?
            .join(","),
        Value::Object(object) => {
            // JSON cannot carry callable methods. An own toString masks the
            // inherited method and default valueOf still returns the object.
            if object.contains_key("toString") {
                return Err(LlmError::InvalidRequest {
                    message: "Cannot convert object to primitive value".into(),
                });
            }
            "[object Object]".into()
        }
    })
}

fn javascript_number(number: &serde_json::Number) -> String {
    crate::exact_json::javascript_number(number.as_f64().expect("JSON number"))
}

impl AnthropicRequestPolicy {
    /// Update body and exact-string sidecars together before sealing/signing.
    pub fn apply(
        mut self,
        body: &mut Value,
        headers: &mut BTreeMap<String, String>,
        string_overrides: &mut BTreeMap<String, Vec<u16>>,
        url: &mut String,
    ) -> Result<(), LlmError> {
        self.extra_body = sanitize_extra_body(self.extra_body)?;
        // An explicit top-level spread owns its string leaves. Main handles
        // output_config separately, retaining untouched automatic fields.
        string_overrides.retain(|pointer, _| {
            let Some(path) = pointer.strip_prefix('/') else {
                return true;
            };
            let mut segments = path.split('/');
            let key = segments
                .next()
                .unwrap_or("")
                .replace("~1", "/")
                .replace("~0", "~");
            if key == "output_config" && self.request_kind == AnthropicRequestKind::Main {
                let field = segments
                    .next()
                    .unwrap_or("")
                    .replace("~1", "/")
                    .replace("~0", "~");
                return !explicit_config_contains(self.extra_body.get(&key), &field);
            }
            !self.extra_body.contains_key(&key)
        });
        let previous_effort = body.pointer("/output_config/effort").cloned();
        let overrides = merge_extra(
            body,
            beta_body(self.extra_body, &self.body_betas),
            self.request_kind,
            self.effort.as_ref(),
        );
        string_overrides.extend(overrides);
        if self.effort.is_some()
            && (body.pointer("/output_config/effort").is_none()
                || body.pointer("/output_config/effort") != previous_effort.as_ref())
        {
            string_overrides.retain(|path, _| {
                path != "/output_config/effort" && !path.starts_with("/output_config/effort/")
            });
        }
        if self.message_header_parameters {
            normalize_message_parameters(
                body,
                headers,
                string_overrides,
                url,
                crate::RequestMode::Complete,
            )?;
        } else {
            normalize_output_format(body, string_overrides)?;
        }
        if let Some(beta) = self.disallowed_fast_beta {
            remove_fast(body, headers, string_overrides, &beta);
        }
        Ok(())
    }
}

/// Current beta Messages SDK virtual parameters. CountTokens spreads iterable
/// betas before appending its mandatory token-counting beta; create calls
/// toString directly. Apply before sealing/signing the request.
pub fn normalize_message_parameters(
    body: &mut Value,
    headers: &mut BTreeMap<String, String>,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
    url: &mut String,
    mode: crate::RequestMode,
) -> Result<(), LlmError> {
    normalize_output_format(body, string_overrides)?;
    materialize_message_headers(body, headers, string_overrides, mode)?;
    let mut parsed = url::Url::parse(url).map_err(|error| LlmError::InvalidRequest {
        message: format!("Invalid Messages URL: {error}"),
    })?;
    if !parsed
        .query_pairs()
        .any(|(key, value)| key == "beta" && value == "true")
    {
        let pairs: Vec<_> = parsed
            .query_pairs()
            .filter(|(key, _)| key != "beta")
            .map(|(key, value)| (key.into_owned(), value.into_owned()))
            .collect();
        parsed.set_query(None);
        parsed
            .query_pairs_mut()
            .extend_pairs(pairs)
            .append_pair("beta", "true");
        *url = parsed.into();
    }
    Ok(())
}

pub const TOKEN_COUNTING: &str = "token-counting-2024-11-01";

// The current native beta Messages SDK still performs this parameter
// normalization before create/countTokens. This is not a Harness API alias.
fn normalize_output_format(
    body: &mut Value,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
) -> Result<(), LlmError> {
    let Some(object) = body.as_object_mut() else {
        return Ok(());
    };
    let Some(format) = object
        .get("output_format")
        .filter(|value| javascript_truthy(value))
        .cloned()
    else {
        return Ok(());
    };
    if object
        .get("output_config")
        .and_then(|config| config.get("format"))
        .is_some_and(javascript_truthy)
    {
        return Err(LlmError::InvalidRequest { message: "Both output_format and output_config.format were provided. Please use only output_config.format (output_format is deprecated).".into() });
    }
    let mut spread_overrides = BTreeMap::new();
    let mut config = spread_config(object.get("output_config").cloned(), &mut spread_overrides);
    // A retained top-level exact string must spread its actual UTF-16 units.
    if object.get("output_config").is_some_and(Value::is_string) {
        if let Some(units) = string_overrides.get("/output_config") {
            config.clear();
            spread_overrides.clear();
            for (index, &unit) in units.iter().enumerate() {
                let key = index.to_string();
                config.insert(
                    key.clone(),
                    Value::String(String::from_utf16_lossy(&[unit])),
                );
                if (0xD800..=0xDFFF).contains(&unit) {
                    spread_overrides.insert(format!("/output_config/{key}"), vec![unit]);
                }
            }
        }
    }
    let mut remapped = BTreeMap::new();
    string_overrides.retain(|path, units| {
        if path == "/output_config"
            || path == "/output_config/format"
            || path.starts_with("/output_config/format/")
        {
            return false;
        }
        if let Some(suffix) = path
            .strip_prefix("/output_format")
            .filter(|suffix| suffix.is_empty() || suffix.starts_with('/'))
        {
            remapped.insert(format!("/output_config/format{suffix}"), units.clone());
            return false;
        }
        true
    });
    config.insert("format".into(), format);
    javascript_key_order(&mut config);
    object.shift_remove("output_format");
    // Inserting over an existing key preserves its original object position.
    object.insert("output_config".into(), Value::Object(config));
    string_overrides.extend(spread_overrides);
    string_overrides.extend(remapped);
    Ok(())
}

fn javascript_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

fn materialize_message_headers(
    body: &mut Value,
    headers: &mut BTreeMap<String, String>,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
    mode: crate::RequestMode,
) -> Result<(), LlmError> {
    let Some(object) = body.as_object_mut() else {
        return Ok(());
    };
    // Convert before mutating; JSON values cannot have callable toString.
    let mut parameters = Vec::new();
    for (key, header) in [
        ("betas", "anthropic-beta"),
        ("user_profile_id", "anthropic-user-profile-id"),
        ("workspace_id", "anthropic-workspace-id"),
    ] {
        let counting = key == "betas" && mode == crate::RequestMode::CountTokens;
        if let Some(value) = object
            .get(key)
            .filter(|value| !value.is_null())
            .or_else(|| counting.then_some(&Value::Null))
        {
            let mut value = value.clone();
            let pointer = format!("/{key}");
            for (path, units) in string_overrides.iter() {
                let Some(relative) = path
                    .strip_prefix(&pointer)
                    .filter(|suffix| suffix.is_empty() || suffix.starts_with('/'))
                else {
                    continue;
                };
                if !crate::exact_json::canonical_string_pointer(&value, relative) {
                    return Err(LlmError::InvalidRequest {
                        message: format!(
                            "UTF-16 JSON override does not target a string leaf: {path}"
                        ),
                    });
                }
                let Some(Value::String(text)) = value.pointer_mut(relative) else {
                    return Err(LlmError::InvalidRequest {
                        message: format!(
                            "UTF-16 JSON override does not target a string leaf: {path}"
                        ),
                    });
                };
                *text = String::from_utf16(units).map_err(|_| LlmError::InvalidRequest {
                    message: format!("Invalid HTTP header value for {header}"),
                })?;
            }
            if counting {
                let mut betas = match value {
                    Value::Null => Vec::new(),
                    Value::Array(items) => items,
                    Value::String(text) => text
                        .chars()
                        .map(|ch| Value::String(ch.to_string()))
                        .collect(),
                    _ => {
                        return Err(LlmError::InvalidRequest {
                            message: "(h ?? []) is not iterable".into(),
                        })
                    }
                };
                betas.push(Value::String(TOKEN_COUNTING.into()));
                value = Value::Array(betas);
            }
            let items = if key != "betas" {
                value.as_array().map(Vec::as_slice)
            } else {
                None
            };
            let items = items.unwrap_or_else(|| std::slice::from_ref(&value));
            if items.is_empty() {
                continue;
            }
            let mut combined = None;
            for item in items {
                if item.is_null() {
                    combined = None;
                    continue;
                }
                let value = javascript_string(item)?;
                let value = value.trim_matches([' ', '\t', '\r', '\n']);
                if value.contains(['\0', '\r', '\n']) || value.chars().any(|ch| u32::from(ch) > 255)
                {
                    return Err(LlmError::InvalidRequest {
                        message: format!("Invalid HTTP header value for {header}"),
                    });
                }
                combined = Some(match combined {
                    Some(previous) => format!("{previous}, {value}"),
                    None => value.to_owned(),
                });
            }
            parameters.push((header, combined));
        }
    }
    for key in ["betas", "user_profile_id", "workspace_id"] {
        object.shift_remove(key);
        let pointer = format!("/{key}");
        string_overrides.retain(|path, _| {
            path != &pointer
                && !path
                    .strip_prefix(&pointer)
                    .is_some_and(|tail| tail.starts_with('/'))
        });
    }
    for (header, value) in parameters {
        match value {
            Some(value) => set_header(headers, header, value.trim_matches([' ', '\t', '\r', '\n'])),
            None => headers.retain(|key, _| !key.eq_ignore_ascii_case(header)),
        }
    }
    Ok(())
}

fn explicit_config_contains(config: Option<&Value>, field: &str) -> bool {
    match config {
        Some(Value::Object(map)) => map.contains_key(field),
        Some(Value::Array(items)) => {
            array_index(field).is_some_and(|index| (index as usize) < items.len())
        }
        Some(Value::String(text)) => {
            array_index(field).is_some_and(|index| (index as usize) < text.encode_utf16().count())
        }
        _ => false,
    }
}

/// Minimum manual thinking budget accepted by Claude.
pub const MIN_MANUAL_THINKING_TOKENS: u32 = 1024;
pub fn beta_body(mut r: Map<String, Value>, betas: &[String]) -> Map<String, Value> {
    if !betas.is_empty() {
        match r.get_mut("anthropic_beta") {
            // Extra body already carries the array → append only the missing
            // entries, preserving the extra body's order (claude-code's
            // `[...o, ...n.filter((s)=>!o.includes(s))]`).
            Some(serde_json::Value::Array(existing)) => {
                for b in betas {
                    if !existing.iter().any(|v| v.as_str() == Some(b.as_str())) {
                        existing.push(serde_json::Value::String(b.clone()));
                    }
                }
            }
            _ => {
                r.insert(
                    "anthropic_beta".to_string(),
                    serde_json::Value::Array(
                        betas
                            .iter()
                            .cloned()
                            .map(serde_json::Value::String)
                            .collect(),
                    ),
                );
            }
        }
    }
    r
}
fn merge_extra(
    body: &mut Value,
    mut extra: Map<String, Value>,
    kind: AnthropicRequestKind,
    effort: Option<&AnthropicEffortPolicy>,
) -> BTreeMap<String, Vec<u16>> {
    let mut overrides = BTreeMap::new();
    let Some(body) = body.as_object_mut() else {
        return overrides;
    };
    // Main builds output_config from the explicit extra config first. Native
    // YMe/YLo/ZLo only insert automatic fields absent from that object. A side
    // query instead spreads the entire extra body after its computed config.
    let main = kind == AnthropicRequestKind::Main;
    let mut output_config = if main {
        spread_config(extra.shift_remove("output_config"), &mut overrides)
    } else {
        Map::new()
    };
    if main {
        if effort.is_some_and(|policy| !policy.supported) {
            output_config.shift_remove("effort");
        } else if let Some(value) = effort
            .and_then(|policy| policy.value.as_ref())
            .filter(|value| value.is_string())
        {
            output_config
                .entry("effort".to_owned())
                .or_insert_with(|| value.clone());
        }
        if let Some(Value::Object(computed)) = body.shift_remove("output_config") {
            let mut computed = computed;
            for key in ["effort", "task_budget", "format"] {
                if let Some(value) = computed.shift_remove(key) {
                    if key == "effort" && effort.is_some() {
                        continue;
                    }
                    if !output_config.contains_key(key) {
                        output_config.insert(key.into(), value);
                    }
                }
            }
            for (key, value) in computed {
                if !output_config.contains_key(&key) {
                    output_config.insert(key, value);
                }
            }
        }
    }
    if !main {
        let computed = body.shift_remove("output_config");
        if computed.is_some() || effort.is_some() {
            let mut computed = computed
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default();
            if let Some(policy) = effort {
                computed.shift_remove("effort");
                if policy.supported {
                    if let Some(value) = policy.value.as_ref().filter(|value| value.is_string()) {
                        computed.insert("effort".into(), value.clone());
                    }
                }
            }
            let mut ordered = Map::new();
            for key in ["format", "effort"] {
                if let Some(value) = computed.shift_remove(key) {
                    ordered.insert(key.into(), value);
                }
            }
            ordered.extend(computed);
            if !ordered.is_empty() {
                body.insert("output_config".into(), Value::Object(ordered));
            }
        }
    }
    let computed_speed = if main {
        body.shift_remove("speed")
    } else {
        None
    };
    let computed_thread = if main {
        body.shift_remove("thread")
    } else {
        None
    };
    let computed_diagnostics = if main {
        body.shift_remove("diagnostics")
    } else {
        None
    };
    let computed_stream = if main {
        body.shift_remove("stream")
    } else {
        None
    };
    let order: &[&str] = if main {
        &[
            "model",
            "messages",
            "system",
            "tools",
            "tool_choice",
            "betas",
            "metadata",
            "max_tokens",
            "thinking",
            "temperature",
            "context_management",
            "safeguards",
        ]
    } else {
        &[
            "model",
            "max_tokens",
            "system",
            "messages",
            "tools",
            "tool_choice",
            "output_config",
            "temperature",
            "stop_sequences",
            "thinking",
            "betas",
            "metadata",
        ]
    };
    let mut remaining = std::mem::take(body);
    for &key in order {
        if let Some(value) = remaining.shift_remove(key) {
            body.insert(key.into(), value);
        }
    }
    body.extend(remaining);
    for (k, v) in extra {
        body.insert(k, v);
    }
    if main && !output_config.is_empty() {
        body.insert("output_config".into(), Value::Object(output_config));
    }
    // Main's computed speed is emitted after output_config and the extra body.
    if let Some(speed) = computed_speed {
        body.insert("speed".to_string(), speed);
    }
    if let Some(thread) = computed_thread {
        body.insert("thread".into(), thread);
    }
    if let Some(diagnostics) = computed_diagnostics {
        body.insert("diagnostics".into(), diagnostics);
    }
    // The streaming main SDK call spreads `stream: true` over the final pd.
    // A collision retains the extra key's position, like the computed tails.
    if let Some(stream) = computed_stream {
        body.insert("stream".into(), stream);
    }
    javascript_key_order(body);
    overrides
}

fn spread_config(
    value: Option<Value>,
    overrides: &mut BTreeMap<String, Vec<u16>>,
) -> Map<String, Value> {
    match value {
        Some(Value::Object(map)) => map,
        Some(Value::Array(items)) => items
            .into_iter()
            .enumerate()
            .map(|(i, value)| (i.to_string(), value))
            .collect(),
        Some(Value::String(text)) => text
            .encode_utf16()
            .enumerate()
            .map(|(i, unit)| {
                let key = i.to_string();
                if (0xD800..=0xDFFF).contains(&unit) {
                    overrides.insert(format!("/output_config/{key}"), vec![unit]);
                }
                (key, Value::String(String::from_utf16_lossy(&[unit])))
            })
            .collect(),
        _ => Map::new(),
    }
}

// Object enumeration emits canonical array-index keys first, numerically.
fn javascript_key_order(map: &mut Map<String, Value>) {
    for value in map.values_mut() {
        match value {
            Value::Object(child) => javascript_key_order(child),
            Value::Array(items) => reorder_array_objects(items),
            _ => {}
        }
    }
    if !map.keys().any(|key| array_index(key).is_some()) {
        return;
    }
    let mut entries: Vec<_> = std::mem::take(map).into_iter().collect();
    entries.sort_by_key(|(key, _)| array_index(key).map_or((1, 0), |index| (0, index)));
    map.extend(entries);
}
fn reorder_array_objects(items: &mut [Value]) {
    for value in items {
        match value {
            Value::Object(map) => javascript_key_order(map),
            Value::Array(items) => reorder_array_objects(items),
            _ => {}
        }
    }
}
fn array_index(key: &str) -> Option<u32> {
    key.parse::<u32>()
        .ok()
        .filter(|&index| index != u32::MAX && index.to_string() == key)
}
pub fn remove_fast(
    body: &mut Value,
    headers: &mut std::collections::BTreeMap<String, String>,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
    beta: &str,
) {
    string_overrides.retain(|pointer, _| pointer != "/speed" && !pointer.starts_with("/speed/"));
    if let Some(body) = body.as_object_mut() {
        body.shift_remove("speed");
    }
    let keys: Vec<_> = headers
        .keys()
        .filter(|key| key.eq_ignore_ascii_case("anthropic-beta"))
        .cloned()
        .collect();
    for key in keys {
        let header = headers.remove(&key).unwrap_or_default();
        let retained = header
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty() && *part != beta)
            .collect::<Vec<_>>()
            .join(",");
        if !retained.is_empty() {
            headers.insert(key, retained);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_fallback_stream_flag_matches_native_final_spread_bytes() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/fallback_stream_flag_2_1_288.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut body = case["body"].clone();
            let mut strings = BTreeMap::new();
            nonstream_fallback_parameters(&mut body, &mut strings).unwrap();
            let bytes = crate::exact_json::serialize(
                &body,
                &strings,
                crate::exact_json::JsonEncoding::JavaScript,
            )
            .unwrap();
            assert_eq!(
                String::from_utf8(bytes).unwrap(),
                case["expected_json"].as_str().unwrap(),
                "{case}"
            );
        }
    }

    #[test]
    fn fallback_clears_overridden_stream_strings_and_retains_other_exact_strings() {
        let mut body = serde_json::json!({"stream":"placeholder","keep":"placeholder","streaming":"unchanged"});
        let mut strings = BTreeMap::from([
            ("/stream".into(), vec![0xd800]),
            ("/stream/nested".into(), vec![0xd800]),
            ("/keep".into(), vec![0xdc00]),
        ]);
        nonstream_fallback_parameters(&mut body, &mut strings).unwrap();
        assert_eq!(strings, BTreeMap::from([("/keep".into(), vec![0xdc00])]));
        let bytes = crate::exact_json::serialize(
            &body,
            &strings,
            crate::exact_json::JsonEncoding::JavaScript,
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            r#"{"stream":false,"keep":"\udc00","streaming":"unchanged"}"#
        );
    }
    use serde_json::json;

    #[test]
    fn native_count_tokens_parameters_match_reference() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/count_tokens_2_1_287.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut body = case["input"]["body"].clone();
            let mut headers =
                serde_json::from_value(case["input"]["default_headers"].clone()).unwrap();
            let mut url = "https://api.anthropic.com/v1/messages/count_tokens".to_owned();
            let result = normalize_message_parameters(
                &mut body,
                &mut headers,
                &mut BTreeMap::new(),
                &mut url,
                crate::RequestMode::CountTokens,
            );
            if let Some(error) = case["expected"]["error"].as_str() {
                let result = result.expect_err(&case.to_string());
                if error.contains("not iterable")
                    || error.starts_with("Both output_format")
                    || error == "Cannot convert object to primitive value"
                {
                    assert_eq!(
                        result,
                        LlmError::InvalidRequest {
                            message: error.into()
                        },
                        "{case}"
                    );
                } else {
                    assert!(matches!(result, LlmError::InvalidRequest { .. }), "{case}");
                }
            } else {
                result.unwrap();
                assert_eq!(
                    url,
                    format!(
                        "https://api.anthropic.com{}",
                        case["expected"]["url"].as_str().unwrap()
                    )
                );
                assert_eq!(
                    serde_json::to_value(headers).unwrap(),
                    case["expected"]["headers"],
                    "{case}"
                );
                assert_eq!(
                    String::from_utf8(
                        crate::exact_json::serialize(
                            &body,
                            &BTreeMap::new(),
                            crate::exact_json::JsonEncoding::JavaScript
                        )
                        .unwrap()
                    )
                    .unwrap(),
                    case["expected"]["body_json"].as_str().unwrap(),
                    "{case}"
                );
            }
        }
    }

    #[test]
    fn native_main_and_side_templates_match_full_layout_bytes() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/request_layout_2_1_287.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut body = case["input"]["base"].clone();
            let mut strings = BTreeMap::new();
            AnthropicRequestPolicy {
                request_kind: if case["input"]["kind"] == "main" {
                    AnthropicRequestKind::Main
                } else {
                    AnthropicRequestKind::SideQuery
                },
                extra_body: case["input"]["extra"].as_object().unwrap().clone(),
                ..Default::default()
            }
            .apply(
                &mut body,
                &mut BTreeMap::new(),
                &mut strings,
                &mut "https://api.anthropic.com/v1/messages".into(),
            )
            .unwrap();
            assert_eq!(
                String::from_utf8(
                    crate::exact_json::serialize(
                        &body,
                        &strings,
                        crate::exact_json::JsonEncoding::JavaScript
                    )
                    .unwrap()
                )
                .unwrap(),
                case["expected"]["body_json"].as_str().unwrap(),
                "{case}"
            );
        }
    }

    #[test]
    fn messages_beta_endpoint_preserves_query_and_other_transports() {
        for (source, expected) in [
            (
                "https://api.anthropic.com/v1/messages",
                "https://api.anthropic.com/v1/messages?beta=true",
            ),
            (
                "https://api.anthropic.com/v1/messages?route=one&beta=false",
                "https://api.anthropic.com/v1/messages?route=one&beta=true",
            ),
            (
                "https://api.anthropic.com/v1/messages?beta=true&route=one",
                "https://api.anthropic.com/v1/messages?beta=true&route=one",
            ),
        ] {
            let mut url = source.to_owned();
            AnthropicRequestPolicy {
                message_header_parameters: true,
                ..Default::default()
            }
            .apply(
                &mut json!({"messages":[]}),
                &mut BTreeMap::new(),
                &mut BTreeMap::new(),
                &mut url,
            )
            .unwrap();
            assert_eq!(url, expected);
        }
        let mut url = "https://cloud/v1/messages?custom=true".to_owned();
        AnthropicRequestPolicy::default()
            .apply(
                &mut json!({"messages":[]}),
                &mut BTreeMap::new(),
                &mut BTreeMap::new(),
                &mut url,
            )
            .unwrap();
        assert_eq!(url, "https://cloud/v1/messages?custom=true");
    }

    #[test]
    fn current_native_output_format_normalization_matches_reference_bytes() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/output_format_2_1_287.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut body = case["body"].clone();
            let mut strings = BTreeMap::new();
            let result = normalize_output_format(&mut body, &mut strings);
            if let Some(message) = case["expected"]["error"].as_str() {
                assert_eq!(
                    result.unwrap_err(),
                    LlmError::InvalidRequest {
                        message: message.into()
                    },
                    "{case}"
                );
            } else {
                result.unwrap();
                assert_eq!(
                    String::from_utf8(
                        crate::exact_json::serialize(
                            &body,
                            &strings,
                            crate::exact_json::JsonEncoding::JavaScript
                        )
                        .unwrap()
                    )
                    .unwrap(),
                    case["expected"]["body_json"].as_str().unwrap(),
                    "{case}"
                );
            }
        }
    }

    #[test]
    fn output_format_normalization_moves_exact_strings_and_retains_config_position() {
        let mut body = json!({"output_config":"display", "tail":true, "output_format":{"description":"display"}});
        let mut strings = BTreeMap::from([
            ("/output_config".into(), vec![0xD800, 65]),
            ("/output_format/description".into(), vec![0xDC00]),
        ]);
        normalize_output_format(&mut body, &mut strings).unwrap();
        assert_eq!(
            String::from_utf8(
                crate::exact_json::serialize(
                    &body,
                    &strings,
                    crate::exact_json::JsonEncoding::JavaScript
                )
                .unwrap()
            )
            .unwrap(),
            r#"{"output_config":{"0":"\ud800","1":"A","format":{"description":"\udc00"}},"tail":true}"#
        );
        let mut body = json!({"output_config":{"format":"already"},"output_format":"new"});
        let mut strings = BTreeMap::from([("/output_format".into(), vec![0xD800])]);
        let before = (body.clone(), strings.clone());
        assert!(normalize_output_format(&mut body, &mut strings).is_err());
        assert_eq!((body, strings), before);
    }

    #[test]
    fn native_messages_parameters_move_to_headers_without_body_residue() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/message_parameters_2_1_287.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut body = case["input"]["body"].clone();
            let mut headers: BTreeMap<_, _> = case["input"]["default_headers"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, value)| (key.clone(), value.as_str().unwrap().to_owned()))
                .collect();
            let mut strings = BTreeMap::new();
            let result = materialize_message_headers(
                &mut body,
                &mut headers,
                &mut strings,
                crate::RequestMode::Complete,
            );
            if case["expected"].get("error").is_some() {
                assert!(
                    matches!(result, Err(LlmError::InvalidRequest { .. })),
                    "{case}"
                );
            } else {
                result.unwrap();
                assert_eq!(body, case["expected"]["body"], "{case}");
                assert_eq!(
                    serde_json::to_value(headers).unwrap(),
                    case["expected"]["headers"],
                    "{case}"
                );
            }
        }
        let mut body =
            json!({"betas":"custom","user_profile_id":"p","workspace_id":"w","speed":"fast"});
        let mut strings = BTreeMap::from([
            ("/betas".into(), "custom".encode_utf16().collect()),
            ("/user_profile_id".into(), vec![112]),
            ("/workspace_id".into(), vec![119]),
            ("/speed".into(), vec![0xD803]),
        ]);
        materialize_message_headers(
            &mut body,
            &mut BTreeMap::new(),
            &mut strings,
            crate::RequestMode::Complete,
        )
        .unwrap();
        assert_eq!(strings, BTreeMap::from([("/speed".into(), vec![0xD803])]));
        assert!(crate::exact_json::serialize(
            &body,
            &strings,
            crate::exact_json::JsonEncoding::JavaScript
        )
        .is_ok());
    }

    #[test]
    fn native_extra_body_sanitation_matches_reference_values_and_metadata_bytes() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/extra_body_2_1_287.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let result = sanitize_extra_body(case["input"]["extra"].as_object().unwrap().clone());
            if let Some(message) = case["expected"]["error"].as_str() {
                assert_eq!(
                    result.unwrap_err(),
                    LlmError::InvalidRequest {
                        message: message.into()
                    },
                    "{case}"
                );
            } else {
                let betas: Vec<_> = case["input"]["betas"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|beta| beta.as_str().unwrap().to_owned())
                    .collect();
                assert_eq!(
                    Value::Object(beta_body(result.unwrap(), &betas)),
                    case["expected"]["value"],
                    "{case}"
                );
            }
        }
    }

    #[test]
    fn effort_admission_clears_only_removed_exact_strings_before_extra_spread() {
        for kind in [AnthropicRequestKind::Main, AnthropicRequestKind::SideQuery] {
            for explicit in [false, true] {
                let mut body = json!({"messages":[],"output_config":{"effort":"automatic","format":{"description":"exact"}}});
                let mut strings = BTreeMap::from([
                    ("/output_config/effort".into(), vec![0xD800]),
                    ("/output_config/format/description".into(), vec![0xDC00]),
                ]);
                AnthropicRequestPolicy {
                    request_kind: kind,
                    effort: Some(AnthropicEffortPolicy {
                        supported: false,
                        value: None,
                    }),
                    extra_body: if explicit {
                        json!({"output_config":{"effort":"explicit","unknown":true}})
                            .as_object()
                            .unwrap()
                            .clone()
                    } else {
                        Map::new()
                    },
                    ..Default::default()
                }
                .apply(
                    &mut body,
                    &mut BTreeMap::new(),
                    &mut strings,
                    &mut "https://api.anthropic.com/v1/messages".into(),
                )
                .unwrap();
                assert!(!strings.contains_key("/output_config/effort"));
                if kind == AnthropicRequestKind::Main || !explicit {
                    assert!(body.pointer("/output_config/effort").is_none());
                    assert_eq!(strings["/output_config/format/description"], [0xDC00]);
                } else {
                    assert_eq!(body["output_config"]["effort"], "explicit");
                    assert!(strings.is_empty());
                }
                crate::exact_json::serialize(
                    &body,
                    &strings,
                    crate::exact_json::JsonEncoding::JavaScript,
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn resolved_effort_replacement_discards_previous_exact_string() {
        for kind in [AnthropicRequestKind::Main, AnthropicRequestKind::SideQuery] {
            let mut body = serde_json::json!({"messages":[],"output_config":{"effort":"high","format":{"description":"keep"}}});
            let mut exact = BTreeMap::from([
                ("/output_config/effort".into(), vec![0xD800]),
                ("/output_config/format/description".into(), vec![0xDC00]),
            ]);
            AnthropicRequestPolicy {
                request_kind: kind,
                effort: Some(AnthropicEffortPolicy {
                    supported: true,
                    value: Some(Value::String("low".into())),
                }),
                ..Default::default()
            }
            .apply(
                &mut body,
                &mut BTreeMap::new(),
                &mut exact,
                &mut "https://api.anthropic.com/v1/messages".into(),
            )
            .unwrap();
            assert_eq!(body["output_config"]["effort"], "low");
            assert!(!exact.contains_key("/output_config/effort"));
            assert_eq!(exact["/output_config/format/description"], [0xDC00]);
            let bytes = crate::exact_json::serialize(
                &body,
                &exact,
                crate::exact_json::JsonEncoding::JavaScript,
            )
            .unwrap();
            assert!(String::from_utf8(bytes)
                .unwrap()
                .contains("\"effort\":\"low\""));
        }
    }

    #[test]
    fn invalid_extra_body_does_not_mutate_body_headers_or_exact_strings() {
        let mut body = json!({"messages":[],"speed":"fast"});
        let mut headers = BTreeMap::from([("anthropic-beta".into(), "computed".into())]);
        let mut overrides = BTreeMap::from([("/speed".into(), vec![0xD800])]);
        let before = (body.clone(), headers.clone(), overrides.clone());
        let result = AnthropicRequestPolicy {
            extra_body: json!({"betas":{"toString":null}})
                .as_object()
                .unwrap()
                .clone(),
            ..Default::default()
        }
        .apply(
            &mut body,
            &mut headers,
            &mut overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
        );
        assert!(matches!(result, Err(LlmError::InvalidRequest { .. })));
        assert_eq!((body, headers, overrides), before);
    }

    #[test]
    fn native_output_config_policy_matches_all_reference_bytes() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/output_config_wire_2_1_287.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut body = case["expected"]["base"].clone();
            let mut headers = BTreeMap::new();
            let mut overrides = BTreeMap::new();
            AnthropicRequestPolicy {
                request_kind: if case["input"]["kind"] == "main" {
                    AnthropicRequestKind::Main
                } else {
                    AnthropicRequestKind::SideQuery
                },
                extra_body: case["input"]["extra"].as_object().unwrap().clone(),
                effort: Some(AnthropicEffortPolicy {
                    supported: case["input"]["supported"].as_bool().unwrap(),
                    value: (!case["input"]["effort"].is_null())
                        .then(|| case["input"]["effort"].clone()),
                }),
                ..Default::default()
            }
            .apply(
                &mut body,
                &mut headers,
                &mut overrides,
                &mut "https://api.anthropic.com/v1/messages".to_owned(),
            )
            .unwrap();
            let wire = crate::exact_json::serialize(
                &body,
                &overrides,
                crate::exact_json::JsonEncoding::JavaScript,
            )
            .unwrap();
            assert_eq!(
                String::from_utf8(wire).unwrap(),
                case["expected"]["body_json"].as_str().unwrap(),
                "{:?}",
                case["input"]
            );
        }
    }

    #[test]
    fn explicit_replacements_remove_stale_exact_string_overrides() {
        for request_kind in [AnthropicRequestKind::Main, AnthropicRequestKind::SideQuery] {
            let mut body = json!({"messages":[{"content":"old"}],"output_config":{"format":{"schema":{"description":"old"}},"effort":"high"}});
            let mut overrides = BTreeMap::from([
                ("/messages/0/content".into(), vec![0xD800]),
                (
                    "/output_config/format/schema/description".into(),
                    vec![0xD801],
                ),
            ]);
            AnthropicRequestPolicy { request_kind, extra_body: json!({"messages":[{"content":"new"}],"output_config":{"format":{"schema":{"description":"new"}}}}).as_object().unwrap().clone(), ..Default::default() }
                .apply(&mut body, &mut BTreeMap::new(), &mut overrides, &mut "https://api.anthropic.com/v1/messages".to_owned()).unwrap();
            assert!(overrides.is_empty());
            assert!(String::from_utf8(
                crate::exact_json::serialize(
                    &body,
                    &overrides,
                    crate::exact_json::JsonEncoding::JavaScript
                )
                .unwrap()
            )
            .unwrap()
            .contains("new"));
        }
        let mut body = json!({"":{"content":"old"}});
        let mut overrides = BTreeMap::from([("//content".into(), vec![0xD800])]);
        AnthropicRequestPolicy {
            extra_body: json!({"content":"new"}).as_object().unwrap().clone(),
            ..Default::default()
        }
        .apply(
            &mut body,
            &mut BTreeMap::new(),
            &mut overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
        )
        .unwrap();
        assert!(
            overrides.contains_key("//content"),
            "an empty top-level key remains untouched"
        );
        AnthropicRequestPolicy {
            extra_body: json!({"":{"content":"new"}}).as_object().unwrap().clone(),
            ..Default::default()
        }
        .apply(
            &mut body,
            &mut BTreeMap::new(),
            &mut overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
        )
        .unwrap();
        assert!(overrides.is_empty());
    }

    #[test]
    fn fast_guard_clears_exact_overrides_for_removed_speed() {
        let mut body = json!({"speed":"fast","messages":[]});
        let mut overrides = BTreeMap::from([("/speed".into(), vec![0xD800])]);
        remove_fast(&mut body, &mut BTreeMap::new(), &mut overrides, "fast");
        assert!(overrides.is_empty());
        assert!(crate::exact_json::serialize(
            &body,
            &overrides,
            crate::exact_json::JsonEncoding::JavaScript
        )
        .is_ok());
    }

    #[test]
    fn beta_merge_preserves_auth_order_and_normalizes_header_spelling() {
        let mut headers = BTreeMap::from([
            ("Anthropic-Beta".into(), "oauth, existing,oauth".into()),
            ("authorization".into(), "Bearer secret".into()),
        ]);
        merge_beta_header(&mut headers, &["existing,computed".into(), "custom".into()]);
        assert_eq!(headers["anthropic-beta"], "oauth,existing,computed,custom");
        assert!(!headers.contains_key("Anthropic-Beta"));
        assert_eq!(headers["authorization"], "Bearer secret");
    }

    #[test]
    fn identity_policy_preserves_auth_identity_or_replaces_all_spellings() {
        let mut headers = BTreeMap::from([("USER-AGENT".into(), "auth-agent".into())]);
        apply_user_agent(&mut headers, UserAgentPolicy::IfAbsent("host-agent"));
        assert_eq!(headers["USER-AGENT"], "auth-agent");
        apply_user_agent(&mut headers, UserAgentPolicy::Replace("host-agent"));
        assert_eq!(
            headers,
            BTreeMap::from([("user-agent".into(), "host-agent".into())])
        );
    }

    #[test]
    fn main_policy_preserves_explicit_output_config_and_computed_speed() {
        let mut body = json!({"speed":"fast", "output_config":{"effort":"high"}, "messages":[]});
        let mut headers = BTreeMap::new();
        let mut overrides = BTreeMap::new();
        AnthropicRequestPolicy {
            extra_body: json!({"speed":"slow", "output_config":{"effort":"low","unknown":true},
                "anthropic_beta":["custom","first"], "provider_extension":{"foo":1}})
            .as_object()
            .unwrap()
            .clone(),
            body_betas: vec!["first".into(), "second".into()],
            ..Default::default()
        }
        .apply(
            &mut body,
            &mut headers,
            &mut overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
        )
        .unwrap();
        assert!(overrides.is_empty());
        assert_eq!(body["speed"], "fast");
        assert_eq!(
            body["output_config"],
            json!({"effort":"low","unknown":true})
        );
        assert_eq!(body["anthropic_beta"], json!(["custom", "first", "second"]));
        assert_eq!(body["provider_extension"], json!({"foo":1}));
    }

    #[test]
    fn fast_guard_runs_after_extra_and_covers_case_insensitive_headers() {
        let mut body = json!({"messages":[]});
        let mut headers = BTreeMap::from([("Anthropic-Beta".into(), "oauth,fast,other".into())]);
        let mut overrides = BTreeMap::new();
        AnthropicRequestPolicy {
            extra_body: json!({"speed":"fast"}).as_object().unwrap().clone(),
            disallowed_fast_beta: Some("fast".into()),
            ..Default::default()
        }
        .apply(
            &mut body,
            &mut headers,
            &mut overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
        )
        .unwrap();
        assert!(overrides.is_empty());
        assert!(body.get("speed").is_none());
        assert_eq!(headers["Anthropic-Beta"], "oauth,other");
    }
}
