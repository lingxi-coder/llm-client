//! Output policy is resolved for the selected attempt, never a bare model ID.
use crate::protocol::{LlmError, ModelMetadata, ProtocolFamily, ProviderProfile};

/// Published ceilings. Missing facts remain unknown.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OutputTokenLimits {
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
}
impl OutputTokenLimits {
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        Self {
            context_window: metadata.context_window_tokens.filter(|n| *n > 0),
            max_output_tokens: metadata.max_output_tokens.filter(|n| *n > 0),
        }
    }
    /// Automatic generation uses the provider default. Context metadata alone
    /// does not establish that input and output share a window.
    pub fn budget(self, requested: Option<u32>, _input_tokens: u64, _headroom: u64) -> Option<u32> {
        requested.map(|n| {
            self.max_output_tokens.map_or(n, |cap| {
                u64::from(n).min(cap).min(u64::from(u32::MAX)) as u32
            })
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputBudgetSource {
    ProviderDefault,
    ApplicationDefault,
    Explicit,
    ClampedExplicit,
}

/// The effective value for one preparation. None means omit the field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedOutputBudget {
    pub value: Option<u32>,
    pub source: OutputBudgetSource,
    pub maximum: Option<u64>,
}

/// Existing application defaults for explicitly recognized Claude models.
/// These are application policy, not vendor defaults. Unknown models have none.
pub fn claude_application_output_default(model: &str) -> Option<u32> {
    let m = model.to_ascii_lowercase().replace('.', "-");
    let (_, tail) = m.split_once("claude-")?;
    for (family, value) in [
        ("sonnet-5-5", 128_000),
        ("fable-5-1", 128_000),
        ("fable-5", 128_000),
        ("mythos-5-1", 128_000),
        ("opus-5-5", 128_000),
        ("opus-5", 128_000),
        ("sonnet-5", 128_000),
        ("opus-4-8", 64_000),
        ("opus-4-7", 64_000),
        ("opus-4-6", 64_000),
        ("sonnet-4-6", 32_000),
        ("sonnet-4-5", 32_000),
        ("opus-4-5", 32_000),
        ("opus-4-1", 32_000),
        ("haiku-4-5", 32_000),
        ("opus-4", 32_000),
        ("sonnet-4", 32_000),
        ("haiku-4", 32_000),
        ("3-7-sonnet", 32_000),
        ("3-5-sonnet", 8_192),
        ("3-5-haiku", 8_192),
        ("3-opus", 4_096),
        ("3-sonnet", 8_192),
        ("3-haiku", 4_096),
    ] {
        let Some(suffix) = tail.strip_prefix(family) else {
            continue;
        };
        let dated = suffix.strip_prefix('-').is_some_and(|value| {
            value.len() >= 8
                && value.as_bytes().iter().take(8).all(u8::is_ascii_digit)
                && value.starts_with("20")
        });
        if suffix.is_empty()
            || dated
            || suffix == "-latest"
            || suffix.starts_with("[1m]")
            || suffix.starts_with("-v")
        {
            return Some(value);
        }
    }
    None
}

pub fn resolve_output_budget(
    profile: &ProviderProfile,
    model: &str,
    requested: Option<u32>,
) -> Result<ResolvedOutputBudget, LlmError> {
    let selected = profile.models.iter().find(|row| row.request_model == model);
    let maximum = selected
        .and_then(|row| row.metadata.max_output_tokens)
        .filter(|n| *n > 0);
    if requested.is_some()
        && profile.protocol == ProtocolFamily::OpenAiResponses
        && matches!(
            profile.auth,
            crate::protocol::AuthStrategy::ChatGptOAuth
                | crate::protocol::AuthStrategy::ChatGptPlan
        )
    {
        return Err(LlmError::UnsupportedCapability {
            message: "the native ChatGPT Responses endpoint does not accept a hard output budget"
                .into(),
        });
    }
    if let Some(value) = requested {
        if value == 0 {
            return Err(invalid("output budget must be positive"));
        }
        let bounded = maximum.map_or(value, |cap| u64::from(value).min(cap) as u32);
        return Ok(ResolvedOutputBudget {
            value: Some(bounded),
            maximum,
            source: if value == bounded {
                OutputBudgetSource::Explicit
            } else {
                OutputBudgetSource::ClampedExplicit
            },
        });
    }
    if !matches!(
        profile.protocol,
        ProtocolFamily::AnthropicMessages
            | ProtocolFamily::VertexClaude
            | ProtocolFamily::BedrockClaude
            | ProtocolFamily::FoundryClaude
    ) {
        return Ok(ResolvedOutputBudget {
            value: None,
            maximum,
            source: OutputBudgetSource::ProviderDefault,
        });
    }
    let identity = selected
        .and_then(|row| row.foundry.as_ref())
        .map(|deployment| deployment.model_id.as_str())
        .unwrap_or(model);
    let value = selected.and_then(|row| row.metadata.application_output_tokens)
        .or_else(|| claude_application_output_default(identity))
        .ok_or_else(|| invalid("this Messages model requires an explicit output budget or metadata.applicationOutputTokens"))?;
    if value == 0 || maximum.is_some_and(|cap| u64::from(value) > cap) {
        return Err(invalid(
            "application output default must be positive and within the selected model ceiling",
        ));
    }
    Ok(ResolvedOutputBudget {
        value: Some(value),
        maximum,
        source: OutputBudgetSource::ApplicationDefault,
    })
}
fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn optional_defaults_are_omitted() {
        let limits = OutputTokenLimits {
            context_window: Some(100),
            max_output_tokens: Some(80),
        };
        assert_eq!(limits.budget(None, 99, 10), None);
        assert_eq!(limits.budget(Some(120), 0, 0), Some(80));
        assert_eq!(
            OutputTokenLimits::default().budget(Some(120), 0, 0),
            Some(120)
        );
    }
    #[test]
    fn unknown_models_have_no_application_default() {
        assert_eq!(claude_application_output_default("custom"), None);
        assert_eq!(claude_application_output_default("claude-unknown"), None);
        assert_eq!(claude_application_output_default("claude-opus-4-99"), None);
        assert_eq!(
            claude_application_output_default("claude-opus-4-6"),
            Some(64000)
        );
    }
}

/// Compute, but never dispatch, an increase using complete canonical counters.
/// Only documented first-party total-generation contracts are eligible.
pub fn recovery_output_budget(
    profile: &ProviderProfile,
    model: &str,
    usage: &crate::protocol::UsageReport,
    issued: Option<u64>,
    effective_context: Option<u64>,
) -> Option<u32> {
    let counts = usage.complete()?;
    let host = url::Url::parse(&profile.base_url)
        .ok()?
        .host_str()?
        .to_owned();
    let documented_total = matches!(
        (profile.provider_id.as_str(), host.as_str()),
        ("deepseek", "api.deepseek.com")
            | ("anthropic", "api.anthropic.com")
            | ("openai" | "open_ai", "api.openai.com")
            | (
                "kimi",
                "api.moonshot.cn" | "api.moonshot.ai" | "api.kimi.com"
            )
            | ("minimax", "api.minimax.io" | "api.minimax.chat")
    );
    if !documented_total
        || matches!(
            profile.auth,
            crate::protocol::AuthStrategy::ChatGptOAuth
                | crate::protocol::AuthStrategy::ChatGptPlan
        )
    {
        return None;
    }
    let selected = profile
        .models
        .iter()
        .find(|row| row.request_model == model)?;
    let cap = selected.metadata.max_output_tokens.filter(|n| *n > 0)?;
    let base = issued
        .unwrap_or(counts.output_tokens)
        .max(counts.output_tokens);
    let mut next = base.saturating_mul(2).min(cap).min(u64::from(u32::MAX));
    if let Some(window) = effective_context.or(selected.metadata.context_window_tokens) {
        let input = counts
            .input_tokens
            .saturating_add(counts.cache_read_tokens)
            .saturating_add(counts.cache_write_tokens);
        next = next.min(window.saturating_sub(input).saturating_sub(1024));
    }
    (next > base && counts.output_tokens > 0).then_some(next as u32)
}
