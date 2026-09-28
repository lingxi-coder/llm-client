use super::{json::WireValue, CodecContext};
use crate::protocol::{
    CachePosition, CacheTtl, ChatRequest, ContentBlock, LlmError, ProtocolFamily,
};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
fn unsupported(message: &str) -> LlmError {
    LlmError::UnsupportedCapability {
        message: message.into(),
    }
}
fn control(ttl: CacheTtl) -> Value {
    match ttl {
        CacheTtl::FiveMinutes => json!({"type":"ephemeral"}),
        CacheTtl::ThirtyMinutes => json!({"type":"ephemeral","ttl":"30m"}),
        CacheTtl::OneHour => json!({"type":"ephemeral","ttl":"1h"}),
    }
}

fn uses_anthropic_marker_rules(context: &CodecContext) -> bool {
    crate::providers::anthropic::code_execution::is_official_profile(context.profile())
        || matches!(
            context.profile().protocol,
            ProtocolFamily::FoundryClaude | ProtocolFamily::VertexClaude
        )
}
pub(crate) fn validate(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let native_openai_responses_controls = req.prompt_cache.prompt_cache_key.is_some()
        || req.prompt_cache.prompt_cache_options.is_some()
        || req.prompt_cache.prompt_cache_retention.is_some();
    if native_openai_responses_controls && !crate::providers::openai::prompt_cache::applies(context)
    {
        return Err(unsupported(
            "native OpenAI prompt-cache controls require the official OpenAI Responses endpoint",
        ));
    }
    if crate::providers::openai::prompt_cache::applies(context) {
        return crate::providers::openai::prompt_cache::validate(req, context);
    }
    if crate::providers::qwen::cache::applies(context) {
        return crate::providers::qwen::cache::validate(req, context);
    }
    if crate::providers::openrouter::prompt_cache::applies(context) {
        return crate::providers::openrouter::prompt_cache::validate(req, context);
    }
    if uses_anthropic_marker_rules(context) {
        return validate_anthropic_markers(req, context);
    }
    let policy = &req.prompt_cache;
    if policy.automatic.is_none() && policy.breakpoints.is_empty() {
        return Ok(());
    }
    if !matches!(
        context.profile().protocol,
        ProtocolFamily::AnthropicMessages
            | ProtocolFamily::VertexClaude
            | ProtocolFamily::BedrockClaude
            | ProtocolFamily::FoundryClaude
    ) {
        return Err(unsupported(
            "explicit prompt cache policy has no encoder for this protocol",
        ));
    }
    crate::providers::minimax::cache::validate(req, context)?;
    if policy.breakpoints.len() + usize::from(policy.automatic.is_some()) > 4 {
        return Err(invalid(
            "at most four cache breakpoints including automatic caching are allowed",
        ));
    }
    if policy.automatic == Some(CacheTtl::ThirtyMinutes)
        || policy
            .breakpoints
            .iter()
            .any(|breakpoint| breakpoint.ttl == CacheTtl::ThirtyMinutes)
    {
        return Err(unsupported(
            "30-minute cache TTL is supported only by OpenRouter OpenAI GPT-5.6+ Chat profiles",
        ));
    }
    validate_breakpoint_positions(req, policy, false)?;
    let mut ordered = policy.breakpoints.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|b| b.position);
    let mut short = false;
    for ttl in ordered.iter().map(|b| b.ttl).chain(policy.automatic) {
        if ttl == CacheTtl::OneHour && short {
            return Err(invalid(
                "one-hour cache breakpoints must precede five-minute breakpoints",
            ));
        }
        short |= ttl == CacheTtl::FiveMinutes;
    }
    if let Some(ttl) = policy.automatic {
        if let Some(native) = context.profile().extra["body"].get("cache_control") {
            if native != &control(ttl) {
                return Err(invalid("conflicting request field cache_control"));
            }
        }
    }
    Ok(())
}

fn validate_anthropic_markers(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let policy = &req.prompt_cache;
    if policy.automatic == Some(CacheTtl::ThirtyMinutes)
        || policy
            .breakpoints
            .iter()
            .any(|breakpoint| breakpoint.ttl == CacheTtl::ThirtyMinutes)
    {
        return Err(unsupported(
            "30-minute cache TTL is supported only by OpenRouter OpenAI GPT-5.6+ Chat profiles",
        ));
    }
    validate_breakpoint_positions(req, policy, true)?;

    // Anthropic's prompt prefix is ordered tools -> system -> messages. The
    // MCP toolsets are emitted after caller tools and before system blocks.
    // Native ProviderContent can already contain message-level cache markers,
    // and profile.extra.body can contain the top-level automatic marker, so
    // include those known first-party inputs in the same four-slot preflight.
    let mut explicit = BTreeMap::new();
    for breakpoint in &policy.breakpoints {
        explicit.insert(breakpoint.position, breakpoint.ttl);
    }
    for (message_index, message) in req.messages.iter().enumerate() {
        for (block_index, block) in message.content.iter().enumerate() {
            let ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value,
            } = block
            else {
                continue;
            };
            let definition = inline_tool_definition(value);
            let outer = value.get("cache_control");
            let nested = definition.and_then(|definition| definition.get("cache_control"));
            let position = CachePosition::Message {
                index: message_index,
                block: block_index,
            };
            if nested.is_some() && (outer.is_some() || explicit.contains_key(&position)) {
                return Err(invalid(
                    "inline tool cache_control belongs on the block or definition, not both",
                ));
            }
            let marker = outer.or(nested);
            if definition.is_some_and(|definition| {
                definition.get("defer_loading").and_then(Value::as_bool) == Some(true)
            }) && (marker.is_some() || explicit.contains_key(&position))
            {
                return Err(invalid("deferred inline tool definitions cannot be cached"));
            }
            let Some(raw) = marker else {
                continue;
            };
            let ttl = parse_anthropic_cache_control(raw)?;
            insert_explicit_marker(
                &mut explicit,
                CachePosition::Message {
                    index: message_index,
                    block: block_index,
                },
                ttl,
            )?;
        }
    }

    // `extra.body.system` is merged additively when the typed request has no
    // system field. Account for only the top-level block markers that will
    // actually be sent; in an MCP request the encoder has already created
    // `tools`, so raw `extra.body.tools` is ignored by that same merge rule.
    if req.system.is_empty() {
        if let Some(blocks) = context
            .profile()
            .extra
            .get("body")
            .and_then(|body| body.get("system"))
            .and_then(Value::as_array)
        {
            for (index, block) in blocks.iter().enumerate() {
                let Some(raw) = block.get("cache_control") else {
                    continue;
                };
                insert_explicit_marker(
                    &mut explicit,
                    CachePosition::System { index },
                    parse_anthropic_cache_control(raw)?,
                )?;
            }
        }
    }

    let profile_cache_control = context
        .profile()
        .extra
        .get("body")
        .and_then(|body| body.get("cache_control"));
    let mut automatic = policy.automatic;
    if let Some(raw) = profile_cache_control {
        let raw_ttl = parse_anthropic_cache_control(raw)?;
        if automatic.is_some_and(|typed_ttl| typed_ttl != raw_ttl) {
            return Err(invalid("conflicting request field cache_control"));
        }
        automatic = Some(raw_ttl);
    }

    let mcp_ttls = crate::providers::anthropic::mcp::cache_ttls(req);
    let client_toolset_ttls = crate::providers::anthropic::client_toolsets::cache_ttls(req);
    let fetch_ttl = req
        .hosted_anthropic_web_fetch()
        .filter(|fetch| !fetch.inline_definition)
        .and_then(|fetch| fetch.cache_control);
    if fetch_ttl == Some(CacheTtl::ThirtyMinutes) {
        return Err(unsupported(
            "Anthropic Web Fetch cache TTL must be five minutes or one hour",
        ));
    }
    if fetch_ttl.is_some()
        && req
            .hosted_anthropic_web_fetch()
            .is_some_and(|fetch| fetch.defer_loading)
    {
        return Err(invalid(
            "deferred Anthropic Web Fetch cannot carry a cache breakpoint",
        ));
    }
    let breakpoint_count = explicit.len()
        + mcp_ttls.len()
        + client_toolset_ttls.len()
        + usize::from(fetch_ttl.is_some())
        + usize::from(automatic.is_some());
    if breakpoint_count > 4 {
        return Err(invalid(
            "at most four cache breakpoints including automatic caching are allowed",
        ));
    }

    let mut ordered = explicit
        .iter()
        .map(|(position, ttl)| (cache_order(*position), *ttl))
        .collect::<Vec<_>>();
    ordered.extend(
        mcp_ttls
            .into_iter()
            .enumerate()
            .map(|(index, ttl)| ((1, index, 0), ttl)),
    );
    if let Some(ttl) = fetch_ttl {
        // Hosted Web Fetch is appended after MCP toolsets and before system.
        ordered.push(((1, usize::MAX, 0), ttl));
    }
    // Client toolset entries follow Web Fetch in declaration order.
    ordered.extend(
        client_toolset_ttls
            .into_iter()
            .enumerate()
            .map(|(index, ttl)| ((1, usize::MAX, index + 1), ttl)),
    );
    if let Some(ttl) = automatic {
        // Profile-level cache_control is the provider's automatic breakpoint;
        // it is ordered after every explicit marker, including MCP toolsets.
        ordered.push(((4, 0, 0), ttl));
    }
    ordered.sort_by_key(|(position, _)| *position);
    validate_ttl_order(ordered.iter().map(|(_, ttl)| *ttl))
}

// Only a direct inline definition is a cache-bearing tool entry. Never scan
// arbitrary nested schemas: their property names are user data.
fn inline_tool_definition(value: &Value) -> Option<&Value> {
    (value.get("type").and_then(Value::as_str) == Some("tool_addition"))
        .then(|| value.get("tool"))
        .flatten()
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("tool_definition"))
        .and_then(|tool| tool.get("definition"))
}

fn validate_breakpoint_positions(
    req: &ChatRequest,
    policy: &crate::protocol::PromptCachePolicy,
    allow_native_content: bool,
) -> Result<(), LlmError> {
    let mut positions = BTreeSet::new();
    for breakpoint in &policy.breakpoints {
        if let CachePosition::Message { index, .. } = breakpoint.position {
            if req.messages.get(index).is_some_and(|message| {
                message.anthropic_options().is_some_and(|options| {
                    options.clear_at
                        == Some(
                            crate::providers::anthropic::types::AnthropicClearAt::NextUserMessage,
                        )
                })
            }) {
                return Err(invalid(
                    "turn-scoped Anthropic system messages cannot carry cache breakpoints",
                ));
            }
        }
        if !positions.insert(breakpoint.position) {
            return Err(invalid("duplicate cache breakpoint"));
        }
        let valid = match breakpoint.position {
            CachePosition::Tool { index } => index < req.tools.len(),
            CachePosition::System { index } => req
                .system
                .get(index)
                .is_some_and(|block| !block.text.is_empty()),
            CachePosition::Message { index, block } => req
                .messages
                .get(index)
                .and_then(|message| message.content.get(block))
                .is_some_and(|block| match block {
                    ContentBlock::Text { text, .. } => !text.is_empty(),
                    ContentBlock::Image { .. }
                    | ContentBlock::Document { .. }
                    | ContentBlock::ToolUse { .. }
                    | ContentBlock::ToolResult { .. } => true,
                    ContentBlock::ProviderContent { .. } => allow_native_content,
                    _ => false,
                }),
        };
        if !valid {
            return Err(invalid(
                "cache breakpoint does not name a cacheable request block",
            ));
        }
    }
    Ok(())
}

fn insert_explicit_marker(
    markers: &mut BTreeMap<CachePosition, CacheTtl>,
    position: CachePosition,
    ttl: CacheTtl,
) -> Result<(), LlmError> {
    if let Some(existing) = markers.get(&position) {
        if *existing != ttl {
            return Err(invalid("conflicting message cache control"));
        }
    } else {
        markers.insert(position, ttl);
    }
    Ok(())
}

fn parse_anthropic_cache_control(value: &Value) -> Result<CacheTtl, LlmError> {
    if value.get("type").and_then(Value::as_str) != Some("ephemeral") {
        return Err(invalid(
            "Anthropic cache_control type must be ephemeral when combined with typed cache policy",
        ));
    }
    match value.get("ttl") {
        None => Ok(CacheTtl::FiveMinutes),
        Some(Value::String(ttl)) if ttl == "5m" => Ok(CacheTtl::FiveMinutes),
        Some(Value::String(ttl)) if ttl == "1h" => Ok(CacheTtl::OneHour),
        Some(_) => Err(invalid("Anthropic cache_control ttl must be 5m or 1h")),
    }
}

fn cache_order(position: CachePosition) -> (u8, usize, usize) {
    match position {
        CachePosition::Tool { index } => (0, index, 0),
        CachePosition::System { index } => (2, index, 0),
        CachePosition::Message { index, block } => (3, index, block),
    }
}

fn validate_ttl_order(ttls: impl IntoIterator<Item = CacheTtl>) -> Result<(), LlmError> {
    let mut short = false;
    for ttl in ttls {
        if ttl == CacheTtl::OneHour && short {
            return Err(invalid(
                "one-hour cache breakpoints must precede five-minute breakpoints",
            ));
        }
        short |= ttl == CacheTtl::FiveMinutes;
    }
    Ok(())
}
pub(crate) fn apply(
    req: &ChatRequest,
    context: &CodecContext,
    body: &mut Map<String, Value>,
    messages: &mut [WireValue<'_>],
) -> Result<(), LlmError> {
    validate(req, context)?;
    if let Some(ttl) = req.prompt_cache.automatic {
        let automatic = if uses_anthropic_marker_rules(context) {
            context
                .profile()
                .extra
                .get("body")
                .and_then(|body| body.get("cache_control"))
                .filter(|native| parse_anthropic_cache_control(native).ok() == Some(ttl))
                .cloned()
                .unwrap_or_else(|| control(ttl))
        } else {
            control(ttl)
        };
        body.insert("cache_control".into(), automatic);
    }
    for breakpoint in &req.prompt_cache.breakpoints {
        let value = control(breakpoint.ttl);
        match breakpoint.position {
            CachePosition::Tool { index } => {
                body.get_mut("tools")
                    .ok_or_else(|| invalid("missing encoded tools"))?[index]["cache_control"] =
                    value
            }
            CachePosition::System { index } => {
                body.get_mut("system")
                    .ok_or_else(|| invalid("missing encoded system"))?[index]["cache_control"] =
                    value
            }
            CachePosition::Message { index, block } => {
                let block = messages
                    .get_mut(index)
                    .and_then(|m| m.array_field_mut("content"))
                    .and_then(|blocks| blocks.get_mut(block))
                    .ok_or_else(|| invalid("cache breakpoint changed during encoding"))?;
                if let Some(existing) = block.get("cache_control") {
                    if uses_anthropic_marker_rules(context)
                        && parse_anthropic_cache_control(existing)? == breakpoint.ttl
                    {
                        // Preserve the provider-native block exactly when the
                        // typed breakpoint describes the same TTL.
                        continue;
                    }
                    if existing != &value {
                        return Err(invalid("conflicting message cache control"));
                    }
                }
                block["cache_control"] = value;
            }
        }
    }
    Ok(())
}
