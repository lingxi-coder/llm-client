//! Declarative inference controls shared by encoding, discovery and pricing.
use super::CodecContext;
use crate::protocol::*;
use serde_json::{json, Map, Value};

fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
fn unsupported(message: impl Into<String>) -> LlmError {
    LlmError::UnsupportedCapability {
        message: message.into(),
    }
}

pub(crate) fn fast_wire(profile: &ProviderProfile) -> FastWire {
    profile.inference.fast.unwrap_or(match profile.protocol {
        ProtocolFamily::OpenAiChat
        | ProtocolFamily::OpenAiResponses
        | ProtocolFamily::GeminiGenerateContent => FastWire::Priority,
        ProtocolFamily::AnthropicMessages => FastWire::Speed,
        _ => FastWire::Unavailable,
    })
}

pub(crate) fn features(profile: &ProviderProfile, model: &str) -> InferenceFeatures {
    profile
        .models
        .iter()
        .find(|m| m.request_model == model)
        .map(|m| m.info.features.clone())
        .unwrap_or_default()
        .on_connection(&profile.inference_features())
}

fn parse_effort(value: &Value) -> Result<Option<ReasoningEffort>, LlmError> {
    if value.is_null() {
        return Ok(None);
    }
    serde_json::from_value(value.clone())
        .map(Some)
        .map_err(|_| invalid("invalid reasoning effort"))
}

fn tier_field(profile: &ProviderProfile) -> &'static str {
    if fast_wire(profile) == FastWire::Speed {
        "speed"
    } else if matches!(
        profile.protocol,
        ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini
    ) {
        "serviceTier"
    } else {
        "service_tier"
    }
}

fn raw_tier(profile: &ProviderProfile) -> Result<&Value, LlmError> {
    let body = &profile.extra["body"];
    let key = tier_field(profile);
    let primary = &body[key];
    let alias = if key == "serviceTier" {
        &body["service_tier"]
    } else {
        &Value::Null
    };
    let same_tier = tier_from_raw(primary.as_str()).is_some()
        && tier_from_raw(primary.as_str()) == tier_from_raw(alias.as_str());
    if !primary.is_null() && !alias.is_null() && primary != alias && !same_tier {
        return Err(invalid(
            "serviceTier conflicts with service_tier in extra.body",
        ));
    }
    let raw = if primary.is_null() { alias } else { primary };
    if !raw.is_null() && !raw.is_string() {
        return Err(invalid("service tier in extra.body must be a string"));
    }
    Ok(raw)
}

/// Recognize existing body defaults so typed controls cannot contradict them,
/// and their cost implications are retained on high-level responses.
pub(crate) fn controls(
    req: &CompletionRequest,
    profile: &ProviderProfile,
) -> Result<(ThinkingConfig, Option<ServiceTier>), LlmError> {
    let b = &profile.extra["body"];
    let mut config = ThinkingConfig::default();
    let (mode, budget, effort) = match profile.reasoning_wire() {
        ReasoningWire::OpenAiChat => (&Value::Null, &Value::Null, &b["reasoning_effort"]),
        ReasoningWire::OpenAiResponses => (&Value::Null, &Value::Null, &b["reasoning"]["effort"]),
        ReasoningWire::Anthropic => (
            &b["thinking"]["type"],
            &b["thinking"]["budget_tokens"],
            &b["output_config"]["effort"],
        ),
        ReasoningWire::Gemini => (
            &Value::Null,
            &b["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            &b["generationConfig"]["thinkingConfig"]["thinkingLevel"],
        ),
        ReasoningWire::ThinkingType => (
            &b["thinking"]["type"],
            &b["thinking"]["budget_tokens"],
            &b["reasoning_effort"],
        ),
        ReasoningWire::EnableThinking => (
            &b["enable_thinking"],
            &b["thinking_budget"],
            &b["reasoning_effort"],
        ),
        ReasoningWire::ReasoningObject => (
            &b["reasoning"]["enabled"],
            &b["reasoning"]["max_tokens"],
            &b["reasoning"]["effort"],
        ),
        ReasoningWire::Unavailable => (&Value::Null, &Value::Null, &Value::Null),
    };
    config.mode = match mode {
        Value::Null => None,
        Value::Bool(true) => Some(ThinkingMode::Enabled),
        Value::Bool(false) => Some(ThinkingMode::Disabled),
        _ => Some(
            serde_json::from_value(mode.clone())
                .map_err(|_| invalid("invalid thinking mode in extra.body"))?,
        ),
    };
    if !budget.is_null() {
        match budget.as_i64() {
            Some(-1) if profile.reasoning_wire() == ReasoningWire::Gemini => {
                config.budget = Some(ThinkingBudget::Dynamic)
            }
            Some(0) if profile.reasoning_wire() == ReasoningWire::Gemini => {
                config.mode = Some(ThinkingMode::Disabled)
            }
            Some(n) if n > 0 && n <= u32::MAX as i64 => {
                config.budget = Some(ThinkingBudget::Tokens(n as u32))
            }
            _ => return Err(invalid("invalid thinking budget in extra.body")),
        }
    }
    config.effort = parse_effort(effort)?;
    if let Some(typed) = &req.thinking {
        merge_option(&mut config.mode, typed.mode, "thinking mode")?;
        merge_option(&mut config.budget, typed.budget, "thinking budget")?;
        merge_option(&mut config.effort, typed.effort, "reasoning effort")?;
    }
    let raw = raw_tier(profile)?;
    let mut tier = tier_from_raw(raw.as_str());
    merge_option(&mut tier, req.service_tier, "service tier")?;
    if req.service_tier.is_some()
        && !raw.is_null()
        && tier_from_raw(raw.as_str()) != req.service_tier
    {
        return Err(invalid("typed service tier conflicts with extra.body"));
    }
    Ok((config, tier))
}

fn beta_headers(profile: &ProviderProfile) -> Vec<&str> {
    let listed = profile.extra["betas"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str);
    let headers = profile.extra["headers"]
        .as_object()
        .into_iter()
        .flat_map(|headers| headers.iter())
        .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .filter_map(|(_, value)| value.as_str())
        .flat_map(|value| value.split(','));
    listed
        .chain(headers)
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .collect()
}

fn effective_thinking(mut thinking: ThinkingConfig, wire: ReasoningWire) -> ThinkingConfig {
    thinking.mode = thinking
        .mode
        .or(thinking.budget.map(|_| ThinkingMode::Enabled));
    if wire == ReasoningWire::Gemini
        && thinking.budget.is_none()
        && thinking.effort.is_none()
        && matches!(
            thinking.mode,
            Some(ThinkingMode::Enabled | ThinkingMode::Adaptive)
        )
    {
        thinking.budget = Some(ThinkingBudget::Dynamic);
    }
    if matches!(
        wire,
        ReasoningWire::OpenAiChat | ReasoningWire::OpenAiResponses
    ) && thinking.mode == Some(ThinkingMode::Disabled)
        && thinking.effort.is_none()
    {
        thinking.effort = Some(ReasoningEffort::None);
    }
    thinking
}
fn merge_option<T: Copy + PartialEq>(
    current: &mut Option<T>,
    typed: Option<T>,
    field: &str,
) -> Result<(), LlmError> {
    if let Some(value) = typed {
        if current.is_some_and(|old| old != value) {
            return Err(invalid(format!("typed {field} conflicts with extra.body")));
        }
        *current = Some(value);
    }
    Ok(())
}
fn tier_from_raw(raw: Option<&str>) -> Option<ServiceTier> {
    match raw {
        Some("priority" | "fast") => Some(ServiceTier::Fast),
        Some("standard" | "default" | "standard_only") => Some(ServiceTier::Standard),
        _ => None,
    }
}

pub(crate) fn validate(
    req: &CompletionRequest,
    profile: &ProviderProfile,
    model: &str,
) -> Result<(), LlmError> {
    let (thinking, tier) = controls(req, profile)?;
    let f = features(profile, model);
    let reject = |what: &str| {
        unsupported(format!(
            "{model:?} on {:?} does not support {what}",
            profile.profile_name
        ))
    };
    let rw = profile.reasoning_wire();
    let thinking = effective_thinking(thinking, rw);
    if thinking != ThinkingConfig::default() && rw == ReasoningWire::Unavailable {
        return Err(reject("a verified reasoning control mapping"));
    }
    if thinking.mode == Some(ThinkingMode::Adaptive)
        && !matches!(rw, ReasoningWire::Anthropic | ReasoningWire::Gemini)
    {
        return Err(reject("adaptive thinking on this wire"));
    }
    if let Some(mode) = thinking.mode {
        if mode == ThinkingMode::Disabled && thinking.budget.is_some() {
            return Err(invalid("disabled thinking cannot have a budget"));
        }
        if let Some(message) = thinking
            .effort
            .and_then(|e| f.effort_mode_error(Some(mode), e))
        {
            return Err(invalid(message));
        }
        if f.supports_mode(mode) == CapabilitySupport::Unsupported {
            return Err(reject("the requested thinking mode"));
        }
    }
    // An explicit mode can override a default that denotes the opposite
    // mode. Only explicit contradictory effort settings are request errors.
    let effort = thinking.effort.or_else(|| {
        f.effort
            .default
            .filter(|e| f.effort_mode_error(thinking.mode, *e).is_none())
    });
    if let Some(effort) = effort {
        if f.supports_effort(thinking.mode, effort) == CapabilitySupport::Unsupported {
            return Err(reject("the requested effort"));
        }
    }
    if let Some(budget) = thinking.budget {
        if f.thinking == CapabilitySupport::Unsupported {
            return Err(reject("thinking"));
        }
        if f.budget.support == CapabilitySupport::Unsupported {
            return Err(reject("a thinking token budget"));
        }
        match budget {
            ThinkingBudget::Dynamic => {
                if rw != ReasoningWire::Gemini || f.budget.dynamic == CapabilitySupport::Unsupported
                {
                    return Err(reject("dynamic thinking budgets"));
                }
            }
            ThinkingBudget::Tokens(n) => {
                if n == 0
                    || f.budget.min_tokens.is_some_and(|min| n < min)
                    || f.budget.max_tokens.is_some_and(|max| n > max)
                {
                    return Err(invalid("thinking budget is outside the supported range; use disabled mode to turn thinking off"));
                }
                if rw == ReasoningWire::Anthropic {
                    let interleaved = !req.tools.is_empty()
                        && beta_headers(profile).contains(&"interleaved-thinking-2025-05-14");
                    if req.temperature.is_some_and(|t| t != 1.0)
                        && profile.inference.enabled_requires_budget.unwrap_or(true)
                    {
                        return Err(invalid(
                            "manual thinking requires temperature 1 when temperature is specified",
                        ));
                    }
                    if n < 1024 || (!interleaved && n >= req.max_tokens.unwrap_or(4096)) {
                        return Err(invalid("manual thinking needs at least 1024 tokens and a budget below max_tokens (except enabled interleaved thinking)"));
                    }
                    if thinking.mode == Some(ThinkingMode::Adaptive) {
                        return Err(invalid("adaptive thinking cannot have a manual budget"));
                    }
                }
            }
        }
        if matches!(
            rw,
            ReasoningWire::OpenAiChat
                | ReasoningWire::OpenAiResponses
                | ReasoningWire::ThinkingType
                | ReasoningWire::Unavailable
        ) {
            return Err(reject("a numeric thinking budget on this wire"));
        }
        if thinking.effort.is_some()
            && matches!(rw, ReasoningWire::Gemini | ReasoningWire::ReasoningObject)
        {
            return Err(invalid(
                "this wire accepts either a budget or an effort, not both",
            ));
        }
    }
    if thinking.mode == Some(ThinkingMode::Enabled)
        && thinking.effort.is_none()
        && matches!(
            rw,
            ReasoningWire::OpenAiChat | ReasoningWire::OpenAiResponses
        )
    {
        return Err(invalid(
            "an explicit effort is required to enable reasoning on this wire",
        ));
    }
    if rw == ReasoningWire::Anthropic
        && thinking.mode == Some(ThinkingMode::Enabled)
        && thinking.budget.is_none()
        && profile.inference.enabled_requires_budget.unwrap_or(true)
    {
        return Err(invalid(
            "manual thinking requires a budget; use adaptive mode on models that support it",
        ));
    }
    if let Some(tier) = tier {
        if fast_wire(profile) == FastWire::Unavailable
            || (tier == ServiceTier::Fast && f.fast == CapabilitySupport::Unsupported)
        {
            return Err(reject("the requested service tier"));
        }
    }
    let mode = thinking.mode.or(f.default_mode);
    let manual = rw == ReasoningWire::Anthropic
        && profile.inference.enabled_requires_budget.unwrap_or(true)
        && mode == Some(ThinkingMode::Enabled);
    let forced_unsupported = manual
        || mode.is_some_and(|mode| f.forced_tool_choice(mode) == CapabilitySupport::Unsupported);
    if (forced_unsupported
        || profile.extra["thinking_rejects_forced_tool_choice"].as_bool() == Some(true))
        && !req.tools.is_empty()
        && matches!(req.tool_choice, ToolChoice::Any | ToolChoice::Tool { .. })
    {
        let disabled =
            mode == Some(ThinkingMode::Disabled) || thinking.effort == Some(ReasoningEffort::None);
        if forced_unsupported || !disabled {
            return Err(reject("forced tool choice while thinking is enabled"));
        }
    }
    Ok(())
}

pub(crate) fn apply(
    req: &CompletionRequest,
    context: &CodecContext,
    body: &mut Map<String, Value>,
    headers: &mut Vec<(String, String)>,
) -> Result<(), LlmError> {
    let p = context.profile();
    validate(req, p, context.request_model())?;
    let (t, tier) = controls(req, p)?;
    let rw = p.reasoning_wire();
    let t = effective_thinking(t, rw);
    let mut fields = json!({});
    let mode = t.mode;
    match rw {
        ReasoningWire::OpenAiChat | ReasoningWire::OpenAiResponses => {
            let effort = if mode == Some(ThinkingMode::Disabled) {
                Some(ReasoningEffort::None)
            } else {
                t.effort
            };
            if let Some(e) = effort {
                if rw == ReasoningWire::OpenAiChat {
                    fields["reasoning_effort"] = json!(e);
                } else {
                    fields["reasoning"]["effort"] = json!(e);
                }
            }
        }
        ReasoningWire::Anthropic => {
            if let Some(m) = mode {
                fields["thinking"]["type"] = json!(m);
            }
            if let Some(ThinkingBudget::Tokens(n)) = t.budget {
                fields["thinking"]["budget_tokens"] = json!(n);
            }
            if let Some(e) = t.effort {
                fields["output_config"]["effort"] = json!(e);
            }
        }
        ReasoningWire::Gemini => {
            let mut config = json!({});
            if mode == Some(ThinkingMode::Disabled) {
                config["thinkingBudget"] = json!(0);
            }
            if let Some(b) = t.budget {
                config["thinkingBudget"] = match b {
                    ThinkingBudget::Tokens(n) => json!(n),
                    ThinkingBudget::Dynamic => json!(-1),
                };
            }
            if let Some(e) = t.effort {
                config["thinkingLevel"] = json!(e);
            }
            if !config.as_object().unwrap().is_empty() {
                if p.extra["body"]["generationConfig"]["thinkingConfig"]
                    .get("includeThoughts")
                    .is_none()
                {
                    config["includeThoughts"] = json!(true);
                }
                fields["generationConfig"]["thinkingConfig"] = config;
            }
        }
        ReasoningWire::ThinkingType | ReasoningWire::EnableThinking => {
            if let Some(m) = mode {
                if rw == ReasoningWire::ThinkingType {
                    fields["thinking"]["type"] = json!(m);
                } else {
                    fields["enable_thinking"] = json!(m != ThinkingMode::Disabled);
                }
            }
            if let Some(ThinkingBudget::Tokens(n)) = t.budget {
                fields["thinking_budget"] = json!(n);
            }
            if let Some(e) = t.effort {
                fields["reasoning_effort"] = json!(e);
            }
        }
        ReasoningWire::ReasoningObject => {
            if let Some(m) = mode {
                fields["reasoning"]["enabled"] = json!(m != ThinkingMode::Disabled);
            }
            if let Some(ThinkingBudget::Tokens(n)) = t.budget {
                fields["reasoning"]["max_tokens"] = json!(n);
            }
            if let Some(e) = t.effort {
                fields["reasoning"]["effort"] = json!(e);
            }
        }
        ReasoningWire::Unavailable => {}
    }
    if rw == ReasoningWire::Anthropic {
        for beta in beta_headers(p) {
            append_beta(headers, beta);
        }
    }
    if let Some(tier) = tier {
        let value = match (fast_wire(p), tier) {
            (FastWire::Speed, ServiceTier::Standard) => "standard",
            (FastWire::Speed | FastWire::Fast, ServiceTier::Fast) => "fast",
            (FastWire::Priority, ServiceTier::Fast) => "priority",
            (_, ServiceTier::Standard) => {
                if matches!(rw, ReasoningWire::Anthropic | ReasoningWire::Gemini) {
                    "standard"
                } else {
                    "default"
                }
            }
            _ => return Err(unsupported("service tier has no verified wire mapping")),
        };
        let key = tier_field(p);
        let raw = raw_tier(p)?;
        fields[key] = if tier_from_raw(raw.as_str()) == Some(tier) {
            raw.clone()
        } else {
            json!(value)
        };
        if tier == ServiceTier::Fast && fast_wire(p) == FastWire::Speed {
            let beta = p
                .inference
                .fast_beta
                .as_deref()
                .unwrap_or("fast-mode-2026-02-01");
            append_beta(headers, beta);
        }
    } else if tier_field(p) == "serviceTier" && !raw_tier(p)?.is_null() {
        // Preserve unrecognized native tiers without inventing a known price.
        fields["serviceTier"] = raw_tier(p)?.clone();
    }
    // Merge only the controlled top-level objects, retaining unrelated siblings.
    for (key, mut value) in fields.as_object().unwrap().clone() {
        if let Some(extra) = p.extra["body"].get(&key) {
            let mut merged = extra.clone();
            merge_json(&mut merged, &value, &key)?;
            value = merged;
        }
        if let Some(existing) = body.get_mut(&key) {
            merge_json(existing, &value, &key)?;
        } else {
            body.insert(key, value);
        }
    }
    Ok(())
}
fn append_beta(headers: &mut Vec<(String, String)>, beta: &str) {
    if let Some((_, value)) = headers
        .iter_mut()
        .find(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
    {
        if !value.split(',').any(|part| part.trim() == beta) {
            if !value.is_empty() {
                value.push(',');
            }
            value.push_str(beta);
        }
    } else {
        headers.push(("anthropic-beta".into(), beta.into()));
    }
}
fn merge_json(target: &mut Value, incoming: &Value, path: &str) -> Result<(), LlmError> {
    if let (Some(dst), Some(src)) = (target.as_object_mut(), incoming.as_object()) {
        for (key, value) in src {
            if let Some(old) = dst.get_mut(key) {
                merge_json(old, value, &format!("{path}.{key}"))?;
            } else {
                dst.insert(key.clone(), value.clone());
            }
        }
    } else if target != incoming {
        return Err(invalid(format!("conflicting request field {path}")));
    }
    Ok(())
}

pub(crate) fn requested(
    req: &CompletionRequest,
    p: &ProviderProfile,
) -> Result<InferenceReport, LlmError> {
    let (t, tier) = controls(req, p)?;
    Ok(InferenceReport {
        requested_effort: t.effort,
        requested_service_tier: tier,
        requested_raw_service_tier: raw_tier(p)?.as_str().map(str::to_owned),
        ..Default::default()
    })
}

pub(crate) fn observe(report: &mut InferenceReport, root: &Value, wire: FastWire) {
    let body = root
        .get("response")
        .or_else(|| root.get("message"))
        .unwrap_or(root);
    if let Some(v) = body
        .get("service_tier")
        .and_then(Value::as_str)
        .or_else(|| body["usage"]["service_tier"].as_str())
        .or_else(|| body["usageMetadata"]["serviceTier"].as_str())
    {
        report.raw_service_tier = Some(v.into());
    }
    if let Some(v) = body["usage"]["speed"]
        .as_str()
        .or_else(|| body["speed"].as_str())
    {
        report.raw_speed = Some(v.into());
    }
    if let Some(v) = body["reasoning"]["effort"]
        .as_str()
        .or_else(|| body["reasoning_effort"].as_str())
        .or_else(|| body["output_config"]["effort"].as_str())
    {
        report.raw_effort = Some(v.into());
        report.reported_effort = serde_json::from_value(json!(v)).ok();
    }
    report.service_tier = if wire == FastWire::Speed {
        // A capacity Priority Tier is different from the fast inference speed.
        if report
            .raw_service_tier
            .as_deref()
            .is_some_and(|s| s != "standard" && s != "default")
            && report.raw_speed.as_deref() != Some("fast")
        {
            None
        } else {
            tier_from_raw(report.raw_speed.as_deref())
        }
    } else {
        tier_from_raw(report.raw_service_tier.as_deref())
    };
}
pub(crate) fn observe_headers(
    report: &mut InferenceReport,
    headers: &[(String, String)],
    p: &ProviderProfile,
) {
    let name = p.inference.tier_response_header.as_deref().or_else(|| {
        (p.protocol == ProtocolFamily::GeminiGenerateContent).then_some("x-gemini-service-tier")
    });
    if let Some(name) = name {
        if let Some((_, value)) = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)) {
            report.raw_service_tier = Some(value.clone());
            report.service_tier = tier_from_raw(Some(value));
        }
    }
}
pub(crate) fn response(
    resp: &crate::transport::HttpResponse,
    p: &ProviderProfile,
) -> InferenceReport {
    let mut report = InferenceReport::default();
    observe_headers(&mut report, &resp.headers, p);
    if let Ok(body) = serde_json::from_slice::<Value>(&resp.body) {
        observe(&mut report, &body, fast_wire(p));
    }
    report
}

#[derive(Debug, Default)]
pub(crate) struct StreamInference {
    pub report: InferenceReport,
    fast: Option<FastWire>,
    header: Option<String>,
}
impl StreamInference {
    pub fn new(context: &CodecContext) -> Self {
        let p = context.profile();
        Self {
            fast: Some(fast_wire(p)),
            header: p.inference.tier_response_header.clone().or_else(|| {
                (p.protocol == ProtocolFamily::GeminiGenerateContent)
                    .then(|| "x-gemini-service-tier".into())
            }),
            ..Self::default()
        }
    }
    pub fn headers(&mut self, headers: &[(String, String)]) {
        if let Some(name) = &self.header {
            if let Some((_, value)) = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)) {
                self.report.raw_service_tier = Some(value.clone());
                self.report.service_tier = tier_from_raw(Some(value));
            }
        }
    }
    pub fn observe(&mut self, body: &Value, out: &mut Vec<StreamEvent>) {
        if body.get("error").is_some_and(|v| !v.is_null()) || body["type"] == "error" {
            return;
        }
        let before = self.report.clone();
        observe(
            &mut self.report,
            body,
            self.fast.unwrap_or(FastWire::Priority),
        );
        if self.report != before {
            out.push(StreamEvent::Inference {
                report: self.report.clone(),
            });
        }
    }
}
