//! Wire merging for explicit host policy overrides. Environment and feature decisions remain in the host.
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
#[derive(Debug, Clone, Default)]
pub struct AnthropicRequestPolicy {
    pub extra_body: Map<String, Value>,
    pub body_betas: Vec<String>,
    /// A beta token whose fast-mode fields must be removed for this route.
    pub disallowed_fast_beta: Option<String>,
}

impl AnthropicRequestPolicy {
    pub fn apply(self, body: &mut Value, headers: &mut BTreeMap<String, String>) {
        merge_extra(body, beta_body(self.extra_body, &self.body_betas));
        if let Some(beta) = self.disallowed_fast_beta {
            remove_fast(body, headers, &beta);
        }
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
pub fn merge_extra(body: &mut Value, mut extra: Map<String, Value>) {
    if extra.is_empty() {
        return;
    }
    let Some(body) = body.as_object_mut() else {
        return;
    };
    // Peel the extra body's output_config (claude-code `delete _i.output_config`).
    let extra_output_config = extra.remove("output_config");
    // Capture the codec-computed top-level `speed` so the generic extra spread
    // can't clobber it: claude-code spreads the extra body (`...Vs`) BEFORE the
    // computed `...ze!==void 0&&{speed:ze}`, so a computed speed wins over an
    // extra one. When no speed was computed (`ze` undefined) the spread is
    // skipped and an extra-body `speed` survives.
    let computed_speed = body.get("speed").cloned();
    // Spread the remaining keys first (claude-code `...va, ..._i`).
    for (k, v) in extra {
        body.insert(k, v);
    }
    // Re-apply the computed speed on top (computed wins, position preserved).
    if let Some(speed) = computed_speed {
        body.insert("speed".to_string(), speed);
    }
    // Then merge/emit output_config last (claude-code `...{output_config:Ii}`):
    // start from the extra body's copy, overlay the computed one (computed wins).
    if let Some(serde_json::Value::Object(extra_oc)) = extra_output_config {
        let mut merged = extra_oc;
        if let Some(serde_json::Value::Object(computed)) = body.get("output_config") {
            for (k, v) in computed {
                merged.insert(k.clone(), v.clone());
            }
        }
        if merged.is_empty() {
            body.remove("output_config");
        } else {
            body.insert(
                "output_config".to_string(),
                serde_json::Value::Object(merged),
            );
        }
    }
}
pub fn remove_fast(
    body: &mut Value,
    headers: &mut std::collections::BTreeMap<String, String>,
    beta: &str,
) {
    if let Some(body) = body.as_object_mut() {
        body.remove("speed");
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
    use serde_json::json;

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
    fn explicit_policy_preserves_passthrough_and_computed_precedence() {
        let mut body = json!({"speed":"fast", "output_config":{"effort":"high"}, "messages":[]});
        let mut headers = BTreeMap::new();
        AnthropicRequestPolicy {
            extra_body: json!({"speed":"slow", "output_config":{"effort":"low","unknown":true},
                "anthropic_beta":["custom","first"], "provider_extension":{"foo":1}})
            .as_object()
            .unwrap()
            .clone(),
            body_betas: vec!["first".into(), "second".into()],
            ..Default::default()
        }
        .apply(&mut body, &mut headers);
        assert_eq!(body["speed"], "fast");
        assert_eq!(
            body["output_config"],
            json!({"effort":"high","unknown":true})
        );
        assert_eq!(body["anthropic_beta"], json!(["custom", "first", "second"]));
        assert_eq!(body["provider_extension"], json!({"foo":1}));
    }

    #[test]
    fn fast_guard_runs_after_extra_and_covers_case_insensitive_headers() {
        let mut body = json!({"messages":[]});
        let mut headers = BTreeMap::from([("Anthropic-Beta".into(), "oauth,fast,other".into())]);
        AnthropicRequestPolicy {
            extra_body: json!({"speed":"fast"}).as_object().unwrap().clone(),
            disallowed_fast_beta: Some("fast".into()),
            ..Default::default()
        }
        .apply(&mut body, &mut headers);
        assert!(body.get("speed").is_none());
        assert_eq!(headers["Anthropic-Beta"], "oauth,other");
    }
}
