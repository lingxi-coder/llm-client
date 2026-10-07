//! Native prompt-cache controls for the official OpenAI Responses API.

use crate::codecs::{json::WireValue, CodecContext};
use crate::protocol::{
    CachePosition, CacheTtl, ChatRequest, ContentBlock, LlmError, MessageRole,
    OpenAiPromptCacheMode, OpenAiPromptCacheRetention, OpenAiPromptCacheTtl, ProtocolFamily,
};
use serde_json::{json, Map, Value};

const PROFILE_CACHE_FIELDS: [&str; 4] = [
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "prompt_cache_breakpoint",
];

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
    super::responses_policy::is_official_openai_responses_profile(context.profile())
}

fn gpt56_or_later(model: &str) -> bool {
    let model = model.strip_prefix('~').unwrap_or(model);
    let model = model.strip_prefix("openai/").unwrap_or(model);
    let Some(versioned) = model.strip_prefix("gpt-") else {
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
    parts.next().is_none() && (major > 5 || (major == 5 && minor >= 6))
}

fn supports_retention(model: &str, retention: OpenAiPromptCacheRetention) -> bool {
    let model = model.strip_prefix('~').unwrap_or(model);
    if gpt56_or_later(model) {
        return retention == OpenAiPromptCacheRetention::TwentyFourHours;
    }
    match model {
        // Current docs specify 24h only for GPT-5.5 and GPT-5.5 Pro.
        "gpt-5.5" | "gpt-5.5-pro" => retention == OpenAiPromptCacheRetention::TwentyFourHours,
        // Earlier models in the current guide's documented extended-retention list.
        "gpt-5.4"
        | "gpt-5.2"
        | "gpt-5.1-codex-max"
        | "gpt-5.1"
        | "gpt-5.1-codex"
        | "gpt-5.1-codex-mini"
        | "gpt-5.1-chat-latest"
        | "gpt-5"
        | "gpt-5-codex"
        | "gpt-4.1" => true,
        _ => false,
    }
}

fn has_native_fields(request: &ChatRequest) -> bool {
    request.prompt_cache.prompt_cache_key.is_some()
        || request.prompt_cache.prompt_cache_options.is_some()
        || request.prompt_cache.prompt_cache_retention.is_some()
}

pub(crate) fn has_system_breakpoint(request: &ChatRequest) -> bool {
    request
        .prompt_cache
        .breakpoints
        .iter()
        .any(|breakpoint| matches!(breakpoint.position, CachePosition::System { .. }))
}

pub(crate) fn has_message_breakpoint(request: &ChatRequest, message: usize, block: usize) -> bool {
    request.prompt_cache.breakpoints.iter().any(|breakpoint| {
        breakpoint.position
            == (CachePosition::Message {
                index: message,
                block,
            })
    })
}

fn is_markable_input_text(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("input_text")
        && value
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| !text.is_empty())
        && value
            .get("prompt_cache_breakpoint")
            .is_none_or(|marker| marker == &json!({"mode":"explicit"}))
}

fn can_mark_provider_message(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("message")
        && value
            .get("role")
            .and_then(Value::as_str)
            .is_some_and(|role| role != "assistant")
        && value
            .get("content")
            .and_then(Value::as_array)
            .and_then(|parts| parts.last())
            .is_some_and(is_markable_input_text)
}

/// Validate the OpenAI-only policy before any request preparation or encoding.
pub(crate) fn validate(request: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let policy = &request.prompt_cache;
    let has_controls =
        policy.automatic.is_some() || !policy.breakpoints.is_empty() || has_native_fields(request);
    if !has_controls {
        if applies(context)
            && PROFILE_CACHE_FIELDS
                .iter()
                .any(|field| context.profile().extra["body"].get(*field).is_some())
        {
            return Err(invalid(
                "OpenAI Responses prompt-cache fields must be set through ChatRequest.prompt_cache",
            ));
        }
        return Ok(());
    }
    if !applies(context) {
        return Err(unsupported(
            "native OpenAI prompt-cache controls require the official OpenAI Responses endpoint",
        ));
    }
    if PROFILE_CACHE_FIELDS
        .iter()
        .any(|field| context.profile().extra["body"].get(*field).is_some())
    {
        return Err(invalid(
            "OpenAI Responses prompt-cache fields cannot be overridden in profile.extra.body",
        ));
    }
    if policy.automatic.is_some() {
        return Err(unsupported(
            "generic automatic CacheTtl is not an OpenAI Responses cache lifetime; use prompt_cache_options or prompt_cache_retention explicitly",
        ));
    }

    let model = context.request_model();
    let supports_new_options = gpt56_or_later(model);
    if policy.prompt_cache_options.is_some() && !supports_new_options {
        return Err(unsupported(
            "prompt_cache_options is documented only for OpenAI GPT-5.6 and later",
        ));
    }
    if let Some(retention) = policy.prompt_cache_retention {
        if !supports_retention(model, retention) {
            return Err(unsupported(format!(
                "the selected OpenAI model {:?} does not have documented support for this prompt_cache_retention value",
                model
            )));
        }
    }
    if let Some(key) = &policy.prompt_cache_key {
        if key.trim().is_empty() || key.chars().any(char::is_control) {
            return Err(invalid(
                "prompt_cache_key must be nonempty and contain no control characters",
            ));
        }
    }

    if !policy.breakpoints.is_empty() {
        if !supports_new_options {
            return Err(unsupported(
                "explicit Responses cache breakpoints are documented only for OpenAI GPT-5.6 and later",
            ));
        }
        let mode = policy
            .prompt_cache_options
            .as_ref()
            .and_then(|options| options.mode)
            .unwrap_or(OpenAiPromptCacheMode::Implicit);
        let max = if mode == OpenAiPromptCacheMode::Explicit {
            4
        } else {
            3
        };
        if policy.breakpoints.len() > max {
            return Err(invalid(format!(
                "OpenAI Responses {mode:?} prompt caching supports at most {max} explicit breakpoints"
            )));
        }

        let mut positions = std::collections::BTreeSet::new();
        for breakpoint in &policy.breakpoints {
            if breakpoint.ttl != CacheTtl::ThirtyMinutes {
                return Err(unsupported(
                    "OpenAI Responses breakpoints accept only the existing 30-minute cache policy value",
                ));
            }
            if !positions.insert(breakpoint.position) {
                return Err(invalid("duplicate OpenAI Responses cache breakpoint"));
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
                    .and_then(|message| {
                        message
                            .content
                            .get(block)
                            .map(|content| (message.role, content))
                    })
                    .is_some_and(|(role, content)| match content {
                        ContentBlock::Text { text, .. }
                        | ContentBlock::TextJsUtf16 { text, .. } => {
                            role != MessageRole::Assistant && !text.is_empty()
                        }
                        ContentBlock::ToolResult {
                            content, blocks, ..
                        } => blocks
                            .as_ref()
                            .and_then(|parts| parts.last())
                            .map_or_else(|| !content.is_empty(), is_markable_input_text),
                        ContentBlock::ProviderContent { protocol, value } => {
                            *protocol == ProtocolFamily::OpenAiResponses
                                && can_mark_provider_message(value)
                        }
                        ContentBlock::Native { .. }
                        | ContentBlock::Thinking { .. }
                        | ContentBlock::RedactedThinking { .. }
                        | ContentBlock::Image { .. }
                        | ContentBlock::Document { .. }
                        | ContentBlock::Audio { .. }
                        | ContentBlock::Video { .. }
                        | ContentBlock::ToolUse { .. } => false,
                    }),
            };
            if !valid {
                return Err(invalid(
                    "OpenAI Responses cache breakpoints require nonempty input_text content in system, non-assistant messages, or tool results",
                ));
            }
        }
    }

    Ok(())
}

/// Turn top-level system instructions into a developer message only when a
/// system block needs an explicit cache marker.
pub(crate) fn system_input<'a>(request: &'a ChatRequest) -> Option<WireValue<'a>> {
    if !has_system_breakpoint(request) {
        return None;
    }
    let mut content = Vec::with_capacity(request.system.len().saturating_mul(2));
    for (index, block) in request.system.iter().enumerate() {
        if index > 0 {
            content.push(
                WireValue::from(json!({"type":"input_text"})).with("text", WireValue::text("\n\n")),
            );
        }
        let part = WireValue::from(json!({"type":"input_text"}))
            .with("text", WireValue::text(&block.text));
        let part = if request
            .prompt_cache
            .breakpoints
            .iter()
            .any(|breakpoint| breakpoint.position == (CachePosition::System { index }))
        {
            part.with(
                "prompt_cache_breakpoint",
                WireValue::from(json!({"mode":"explicit"})),
            )
        } else {
            part
        };
        content.push(part);
    }
    Some(WireValue::from(json!({"role":"developer"})).with("content", WireValue::array(content)))
}

pub(crate) fn mark_message_block<'a>(
    request: &ChatRequest,
    message: usize,
    block: usize,
    part: WireValue<'a>,
) -> WireValue<'a> {
    if request.prompt_cache.breakpoints.iter().any(|breakpoint| {
        breakpoint.position
            == (CachePosition::Message {
                index: message,
                block,
            })
    }) {
        part.with(
            "prompt_cache_breakpoint",
            WireValue::from(json!({"mode":"explicit"})),
        )
    } else {
        part
    }
}

fn mark_input_text(part: WireValue<'_>) -> WireValue<'_> {
    part.with(
        "prompt_cache_breakpoint",
        WireValue::from(json!({"mode":"explicit"})),
    )
}

pub(crate) fn tool_result_output<'a>(
    request: &ChatRequest,
    message: usize,
    block: usize,
    content: &'a str,
    blocks: Option<&'a [Value]>,
) -> WireValue<'a> {
    let should_mark = has_message_breakpoint(request, message, block);
    let Some(blocks) = blocks else {
        return if should_mark {
            WireValue::array(vec![mark_input_text(
                WireValue::from(json!({"type":"input_text"}))
                    .with("text", WireValue::text(content)),
            )])
        } else {
            WireValue::text(content)
        };
    };

    let last = blocks.len().checked_sub(1);
    WireValue::array(
        blocks
            .iter()
            .enumerate()
            .map(|(index, part)| {
                let part = WireValue::borrowed(part);
                if should_mark && Some(index) == last {
                    mark_input_text(part)
                } else {
                    part
                }
            })
            .collect(),
    )
}

pub(crate) fn mark_provider_message<'a>(value: &'a Value) -> WireValue<'a> {
    let last = value
        .get("content")
        .and_then(Value::as_array)
        .and_then(|parts| parts.len().checked_sub(1));
    WireValue::borrowed(value).map_array_field("content", |index, part| {
        let part = WireValue::from(part);
        if Some(index) == last {
            mark_input_text(part)
        } else {
            part
        }
    })
}

pub(crate) fn apply(
    request: &ChatRequest,
    context: &CodecContext,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    validate(request, context)?;
    let policy = &request.prompt_cache;
    if let Some(key) = &policy.prompt_cache_key {
        body.insert("prompt_cache_key".into(), Value::String(key.clone()));
    }
    if let Some(options) = &policy.prompt_cache_options {
        let mut value = Map::new();
        if let Some(mode) = options.mode {
            value.insert(
                "mode".into(),
                Value::String(
                    match mode {
                        OpenAiPromptCacheMode::Implicit => "implicit",
                        OpenAiPromptCacheMode::Explicit => "explicit",
                    }
                    .into(),
                ),
            );
        }
        if let Some(ttl) = options.ttl {
            value.insert(
                "ttl".into(),
                Value::String(
                    match ttl {
                        OpenAiPromptCacheTtl::ThirtyMinutes => "30m",
                    }
                    .into(),
                ),
            );
        }
        body.insert("prompt_cache_options".into(), Value::Object(value));
    }
    if let Some(retention) = policy.prompt_cache_retention {
        body.insert(
            "prompt_cache_retention".into(),
            Value::String(
                match retention {
                    OpenAiPromptCacheRetention::InMemory => "in_memory",
                    OpenAiPromptCacheRetention::TwentyFourHours => "24h",
                }
                .into(),
            ),
        );
    }
    Ok(())
}
