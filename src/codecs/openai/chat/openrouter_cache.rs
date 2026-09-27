//! OpenRouter provider-prompt caching controls for Chat Completions.
//!
//! OpenRouter gateway response caching is configured separately through
//! `RequestOptions`. This module encodes only provider-side prompt cache
//! controls documented for specific upstream model routes.

use crate::codecs::{json::WireValue, CodecContext};
use crate::protocol::{
    CachePosition, CacheTtl, ChatRequest, ContentBlock, LlmError, ProtocolFamily,
};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelCacheRoute {
    Anthropic,
    Alibaba,
    OpenAiGpt56Plus,
}

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

pub(crate) fn applies(context: &CodecContext) -> bool {
    context.profile().provider_id.as_str() == "openrouter"
        && context.profile().protocol == ProtocolFamily::OpenAiChat
}

fn official_route(context: &CodecContext) -> bool {
    let Ok(url) = url::Url::parse(&context.profile().base_url) else {
        return false;
    };
    url.scheme() == "https"
        && url.host_str() == Some("openrouter.ai")
        && url.path().trim_end_matches('/') == "/api/v1"
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
}

fn gpt56_or_newer(model: &str) -> bool {
    let model = model.strip_prefix('~').unwrap_or(model);
    let Some(versioned) = model.strip_prefix("openai/gpt-") else {
        return false;
    };
    let version = versioned
        .split_once('-')
        .map(|(version, _)| version)
        .unwrap_or(versioned);
    let mut parts = version.split('.');
    let Some(Ok(major)) = parts.next().map(str::parse::<u32>) else {
        return false;
    };
    let minor = match parts.next() {
        Some(value) => match value.parse::<u32>() {
            Ok(value) => value,
            Err(_) => return false,
        },
        None => 0,
    };
    if parts.next().is_some() {
        return false;
    }
    major > 5 || (major == 5 && minor >= 6)
}

fn model_route(context: &CodecContext) -> Option<ModelCacheRoute> {
    let model = context
        .request_model()
        .strip_prefix('~')
        .unwrap_or(context.request_model());
    if model.starts_with("anthropic/") {
        return Some(ModelCacheRoute::Anthropic);
    }
    if matches!(
        model,
        "deepseek/deepseek-v3.2"
            | "qwen/qwen3-max"
            | "qwen/qwen-plus"
            | "qwen/qwen3.6-plus"
            | "qwen/qwen3-coder-plus"
            | "qwen/qwen3-coder-flash"
    ) {
        return Some(ModelCacheRoute::Alibaba);
    }
    gpt56_or_newer(model).then_some(ModelCacheRoute::OpenAiGpt56Plus)
}

fn raw_cache_override(context: &CodecContext) -> bool {
    [
        "cache_control",
        "prompt_cache_options",
        "prompt_cache_breakpoint",
        "prompt_cache_key",
        "prompt_cache_retention",
    ]
    .iter()
    .any(|key| context.profile().extra["body"].get(*key).is_some())
}

pub(crate) fn validate(request: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    if !applies(context) {
        return Ok(());
    }
    if raw_cache_override(context) {
        return Err(invalid(
            "OpenRouter prompt cache fields must use ChatRequest.prompt_cache, not profile body overrides",
        ));
    }
    let policy = &request.prompt_cache;
    if policy.automatic.is_none() && policy.breakpoints.is_empty() {
        return Ok(());
    }
    if !official_route(context) {
        return Err(unsupported(
            "provider prompt caching requires the official OpenRouter Chat endpoint",
        ));
    }
    let Some(route) = model_route(context) else {
        return Err(unsupported(format!(
            "OpenRouter prompt cache controls are not documented for model {:?}; provider-managed implicit caching needs no request marker",
            context.request_model()
        )));
    };

    let max_breakpoints = match route {
        ModelCacheRoute::Anthropic => {
            if policy.breakpoints.len() + usize::from(policy.automatic.is_some()) > 4 {
                return Err(invalid(
                    "OpenRouter Anthropic caching allows at most four breakpoints including automatic caching",
                ));
            }
            4
        }
        ModelCacheRoute::Alibaba => {
            if policy.automatic.is_some() {
                return Err(unsupported(
                    "OpenRouter Alibaba caching requires explicit content breakpoints and has no caller-selected automatic TTL",
                ));
            }
            1
        }
        ModelCacheRoute::OpenAiGpt56Plus => {
            let max = if policy.automatic.is_some() { 3 } else { 4 };
            if policy.breakpoints.len() > max {
                return Err(invalid(format!(
                    "OpenAI GPT-5.6+ caching allows at most {max} explicit breakpoints for this mode"
                )));
            }
            max
        }
    };
    if policy.breakpoints.len() > max_breakpoints {
        return Err(invalid(format!(
            "this OpenRouter prompt cache route supports at most {max_breakpoints} explicit breakpoint(s)"
        )));
    }

    match route {
        ModelCacheRoute::Anthropic => {
            if policy.automatic == Some(CacheTtl::ThirtyMinutes)
                || policy
                    .breakpoints
                    .iter()
                    .any(|breakpoint| breakpoint.ttl == CacheTtl::ThirtyMinutes)
            {
                return Err(unsupported(
                    "OpenRouter Anthropic caching supports five-minute and one-hour TTLs, not 30 minutes",
                ));
            }
        }
        ModelCacheRoute::Alibaba => {
            if policy
                .breakpoints
                .iter()
                .any(|breakpoint| breakpoint.ttl != CacheTtl::FiveMinutes)
            {
                return Err(unsupported(
                    "OpenRouter Alibaba prompt cache breakpoints have a fixed five-minute TTL",
                ));
            }
        }
        ModelCacheRoute::OpenAiGpt56Plus => {
            if policy.automatic == Some(CacheTtl::FiveMinutes)
                || policy.automatic == Some(CacheTtl::OneHour)
                || policy
                    .breakpoints
                    .iter()
                    .any(|breakpoint| breakpoint.ttl != CacheTtl::ThirtyMinutes)
            {
                return Err(unsupported(
                    "OpenRouter OpenAI GPT-5.6+ prompt cache accepts only the 30-minute TTL",
                ));
            }
        }
    }

    let mut positions = BTreeSet::new();
    for breakpoint in &policy.breakpoints {
        if !positions.insert(breakpoint.position) {
            return Err(invalid("duplicate OpenRouter prompt cache breakpoint"));
        }
        let valid = match breakpoint.position {
            CachePosition::Tool { .. } => false,
            CachePosition::System { index } => request
                .system
                .get(index)
                .is_some_and(|block| !block.text.is_empty()),
            CachePosition::Message { index, block } => request
                .messages
                .get(index)
                .and_then(|message| message.content.get(block))
                .is_some_and(|content| {
                    matches!(content, ContentBlock::Text { text, .. } if !text.is_empty())
                }),
        };
        if !valid {
            return Err(unsupported(
                "OpenRouter cache breakpoints require nonempty text content blocks; tool schemas and multimodal blocks are not accepted",
            ));
        }
    }

    if route == ModelCacheRoute::Anthropic {
        let mut ordered = policy.breakpoints.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|breakpoint| breakpoint.position);
        let mut has_short_ttl = false;
        for ttl in ordered
            .iter()
            .map(|breakpoint| breakpoint.ttl)
            .chain(policy.automatic)
        {
            if ttl == CacheTtl::OneHour && has_short_ttl {
                return Err(invalid(
                    "one-hour OpenRouter cache breakpoints must precede five-minute breakpoints",
                ));
            }
            has_short_ttl |= ttl == CacheTtl::FiveMinutes;
        }
    }
    Ok(())
}

fn control(ttl: CacheTtl) -> Value {
    match ttl {
        CacheTtl::FiveMinutes => json!({"type":"ephemeral"}),
        CacheTtl::ThirtyMinutes => json!({"type":"ephemeral"}),
        CacheTtl::OneHour => json!({"type":"ephemeral","ttl":"1h"}),
    }
}

pub(crate) fn marker(
    request: &ChatRequest,
    context: &CodecContext,
    position: CachePosition,
) -> Option<Value> {
    if !applies(context) {
        return None;
    }
    let route = model_route(context)?;
    request
        .prompt_cache
        .breakpoints
        .iter()
        .find(|breakpoint| breakpoint.position == position)
        .map(|breakpoint| match route {
            ModelCacheRoute::Anthropic | ModelCacheRoute::Alibaba => control(breakpoint.ttl),
            ModelCacheRoute::OpenAiGpt56Plus => json!({"type":"ephemeral"}),
        })
}

pub(crate) fn text_content(text: &str, marker: Option<Value>) -> WireValue<'_> {
    let part = WireValue::from(json!({"type":"text"})).with("text", WireValue::text(text));
    if let Some(marker) = marker {
        WireValue::array(vec![part.with("cache_control", WireValue::from(marker))])
    } else {
        WireValue::text(text)
    }
}

pub(crate) fn mark(mut part: WireValue<'_>, marker: Option<Value>) -> WireValue<'_> {
    if let Some(marker) = marker {
        part.insert("cache_control".into(), marker);
    }
    part
}

pub(crate) fn system<'a>(
    request: &'a ChatRequest,
    context: &CodecContext,
) -> Option<WireValue<'a>> {
    if !applies(context)
        || !request
            .prompt_cache
            .breakpoints
            .iter()
            .any(|breakpoint| matches!(breakpoint.position, CachePosition::System { .. }))
    {
        return None;
    }
    let mut parts = Vec::new();
    for (index, block) in request.system.iter().enumerate() {
        if index > 0 {
            parts.push(json!({"type":"text","text":"\n\n"}).into());
        }
        if block.text.is_empty() {
            continue;
        }
        let part =
            WireValue::from(json!({"type":"text"})).with("text", WireValue::text(&block.text));
        let marker = marker(request, context, CachePosition::System { index });
        parts.push(mark(part, marker));
    }
    Some(WireValue::from(json!({"role":"system"})).with("content", WireValue::array(parts)))
}

pub(crate) fn apply(
    request: &ChatRequest,
    context: &CodecContext,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    validate(request, context)?;
    if !applies(context) {
        return Ok(());
    }
    let Some(route) = model_route(context) else {
        return Ok(());
    };
    match route {
        ModelCacheRoute::Anthropic => {
            if let Some(ttl) = request.prompt_cache.automatic {
                body.insert("cache_control".into(), control(ttl));
            }
        }
        ModelCacheRoute::Alibaba => {}
        ModelCacheRoute::OpenAiGpt56Plus => {
            if request.prompt_cache.automatic.is_some()
                || !request.prompt_cache.breakpoints.is_empty()
            {
                let mode = if request.prompt_cache.automatic.is_some() {
                    "implicit"
                } else {
                    "explicit"
                };
                body.insert(
                    "prompt_cache_options".into(),
                    json!({"mode":mode,"ttl":"30m"}),
                );
            }
        }
    }
    Ok(())
}
