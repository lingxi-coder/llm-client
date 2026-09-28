//! Wire merging for explicit host policy overrides. Environment and feature decisions remain in the host.
use serde_json::{Map, Value};
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
    let Some(header) = headers.get("anthropic-beta").cloned() else {
        return;
    };
    let retained = header
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty() && *part != beta)
        .collect::<Vec<_>>()
        .join(",");
    if retained.is_empty() {
        headers.remove("anthropic-beta");
    } else {
        headers.insert("anthropic-beta".to_string(), retained);
    }
}
