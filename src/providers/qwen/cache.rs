//! Qwen Chat cache contract, verified against Model Studio context-cache docs.
//! Implicit caching has no request switch. Explicit caching is five-minute,
//! message-content-only prefix caching; it is not Responses session caching.
use crate::codecs::{json::WireValue, CodecContext};
use crate::protocol::{
    CachePosition, CacheTtl, ChatRequest, ContentBlock, LlmError, ProtocolFamily,
};
use serde_json::{json, Value};
use std::collections::BTreeSet;

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

pub(crate) fn applies(context: &CodecContext) -> bool {
    context.profile().provider_id.as_str() == "qwen"
        && context.profile().protocol == ProtocolFamily::OpenAiChat
}

/// Only documented regional Chat roots qualify for explicit cache controls.
fn region(base: &str) -> Option<&'static str> {
    let url = url::Url::parse(base).ok()?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path().trim_end_matches('/') != "/compatible-mode/v1"
    {
        return None;
    }
    let host = url.host_str()?;
    match host {
        "dashscope.aliyuncs.com" => return Some("beijing"),
        "dashscope-intl.aliyuncs.com" => return Some("singapore"),
        "dashscope-us.aliyuncs.com" => return Some("virginia"),
        "cn-hongkong.dashscope.aliyuncs.com" => return Some("hongkong"),
        _ => {}
    }
    for (suffix, region) in [
        (".cn-beijing.maas.aliyuncs.com", "beijing"),
        (".ap-southeast-1.maas.aliyuncs.com", "singapore"),
        (".us-east-1.maas.aliyuncs.com", "virginia"),
        (".cn-hongkong.maas.aliyuncs.com", "hongkong"),
        (".eu-central-1.maas.aliyuncs.com", "frankfurt"),
        (".ap-northeast-1.maas.aliyuncs.com", "tokyo"),
    ] {
        if host.strip_suffix(suffix).is_some_and(|workspace| {
            !workspace.is_empty()
                && !workspace.starts_with('-')
                && !workspace.ends_with('-')
                && workspace
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        }) {
            return Some(region);
        }
    }
    None
}

/// Exact IDs, not an open-ended family prefix. Regional deployment entitlement
/// remains the caller's responsibility even when the contract is documented.
fn model_supported(region: &str, scope: Option<&str>, model: &str) -> bool {
    const GLOBAL_CORE: &[&str] = &[
        "qwen3.8-max",
        "qwen3.8-max-0902",
        "qwen3.8-flash",
        "qwen3.7-max",
        "qwen3.7-max-2026-05-20",
        "qwen3.7-max-2026-06-08",
        "qwen3.7-flash",
        "qwen3.7-flash-2026-07-15",
    ];
    let scope = match (region, scope) {
        ("beijing", None) => "china_mainland",
        ("singapore", None) => "international",
        (_, Some(scope)) => scope,
        _ => return false,
    };
    let global_core = matches!(
        (region, scope),
        ("beijing", "china_mainland")
            | ("singapore", "international")
            | ("virginia" | "hongkong" | "frankfurt" | "tokyo", "global")
    );
    if global_core && GLOBAL_CORE.contains(&model) {
        return true;
    }
    let extra: &[&str] = match (region, scope) {
        ("beijing", "china_mainland") => &[
            "qwen3.6-max-preview",
            "qwen3-max",
            "qwen3.8-2.4t-a95b",
            "qwen3.8-27b",
            "qwen3.7-plus",
            "qwen3.7-plus-2026-05-26",
            "qwen3.6-plus",
            "qwen3.5-plus",
            "qwen3.5-plus-2026-04-20",
            "qwen-plus",
            "qwen3.6-flash",
            "qwen3.5-flash",
            "qwen-flash",
            "qwen3-coder-plus",
            "qwen3-coder-flash",
            "qwen3-vl-plus",
            "qwen3-vl-flash",
            "qwen-plus-character",
            "deepseek-v3.2",
            "kimi-k2.7-code",
            "kimi-k2.6",
            "kimi-k2.5",
            "glm-5.1",
        ],
        ("singapore", "international") => &[
            "qwen3-max",
            "qwen3.8-2.4t-a95b",
            "qwen3.8-27b",
            "qwen3.7-plus",
            "qwen3.7-plus-2026-05-26",
            "qwen3.6-plus",
            "qwen3.5-plus",
            "qwen3.5-plus-2026-04-20",
            "qwen-plus",
            "qwen3.6-flash",
            "qwen3.5-flash",
            "qwen-flash",
            "qwen3-coder-plus",
            "qwen3-coder-flash",
            "qwen3-vl-plus",
            "qwen3-vl-flash",
            "deepseek-v3.2",
        ],
        ("virginia", "global") => &["qwen3-max", "kimi-k2.7-code"],
        ("virginia", "us") => &["qwen3.7-max", "qwen3.7-plus", "qwen3.6-flash"],
        ("hongkong", "global") => &[
            "qwen3.7-plus",
            "qwen3.7-plus-2026-05-26",
            "qwen3.6-plus",
            "qwen3.6-flash",
            "kimi-k2.7-code",
        ],
        ("hongkong", "hong_kong") => &["qwen3-max", "qwen-plus", "qwen3.5-flash", "qwen3-vl-plus"],
        ("frankfurt", "global") => &[
            "qwen3-max",
            "qwen3.7-plus",
            "qwen3.7-plus-2026-05-26",
            "qwen3.6-plus",
            "qwen3.5-plus",
            "qwen-plus",
            "qwen3.5-flash",
            "qwen-flash",
            "qwen3-vl-plus",
            "qwen3-coder-plus",
            "qwen3-coder-flash",
            "kimi-k2.7-code",
            "kimi-k2.5",
        ],
        ("frankfurt", "eu") => &[
            "qwen3-max",
            "qwen-plus",
            "qwen3.6-flash",
            "qwen3.5-flash",
            "qwen3-vl-plus",
            "qwen3-vl-flash",
        ],
        ("tokyo", "global") => &[
            "qwen3-max",
            "qwen3.7-plus",
            "qwen3.7-plus-2026-05-26",
            "qwen3.6-plus",
            "qwen3.5-plus",
            "qwen-plus",
            "qwen3.6-flash",
            "qwen3.5-flash",
            "qwen-flash",
            "kimi-k2.7-code",
        ],
        ("tokyo", "japan") => &["qwen3.7-plus", "qwen3.7-plus-2026-05-26"],
        _ => return false,
    };
    extra.contains(&model)
}

pub(crate) fn validate(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    if !applies(context) {
        return Ok(());
    }
    let extra = &context.profile().extra["body"];
    for field in [
        "cache_control",
        "prompt_cache_options",
        "prompt_cache_key",
        "prompt_cache_retention",
    ] {
        if extra.get(field).is_some() {
            return Err(invalid("Qwen Chat cache controls must use prompt_cache message breakpoints, not top-level native overrides"));
        }
    }
    if context.stream {
        if let Some(options) = extra.get("stream_options") {
            let options = options
                .as_object()
                .ok_or_else(|| invalid("Qwen stream_options must be an object"))?;
            if options
                .get("include_usage")
                .is_some_and(|value| value != &Value::Bool(true))
            {
                return Err(invalid("Qwen streaming requires stream_options.include_usage=true for cache accounting"));
            }
        }
    }
    let policy = &req.prompt_cache;
    if policy.automatic.is_some() {
        return Err(unsupported(
            "Qwen implicit caching is provider-managed with no caller TTL; leave automatic unset",
        ));
    }
    if policy.breakpoints.is_empty() {
        return Ok(());
    }
    let region = region(&context.profile().base_url).ok_or_else(|| {
        unsupported("Qwen explicit caching requires a documented regional Chat endpoint")
    })?;
    let scope = context
        .profile()
        .extra
        .get("qwen_cache_deployment_scope")
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| invalid("qwen_cache_deployment_scope must be a string"))
        })
        .transpose()?;
    if !model_supported(region, scope, context.request_model()) {
        return Err(unsupported("Qwen explicit caching is not documented for this model/region/deployment scope; scoped regions require extra.qwen_cache_deployment_scope"));
    }
    if policy.breakpoints.len() > 4 {
        return Err(invalid("Qwen allows at most four cache breakpoints"));
    }
    let mut positions = BTreeSet::new();
    for breakpoint in &policy.breakpoints {
        if breakpoint.ttl != CacheTtl::FiveMinutes {
            return Err(unsupported(
                "Qwen explicit cache has a fixed five-minute lifetime",
            ));
        }
        if !positions.insert(breakpoint.position) {
            return Err(invalid("duplicate Qwen cache breakpoint"));
        }
        let valid = match breakpoint.position {
            CachePosition::Tool { .. } => return Err(unsupported("Qwen ignores standalone tool-definition cache markers; mark a system or message content block")),
            CachePosition::System { index } => req.system.get(index).is_some_and(|s| !s.text.is_empty()),
            CachePosition::Message { index, block } => req.messages.get(index)
                .and_then(|message| message.content.get(block)).is_some_and(|block| match block {
                    ContentBlock::Text { text, .. } => !text.is_empty(),
                    ContentBlock::Image { .. } => true,
                    ContentBlock::ToolResult { content, .. } => !content.is_empty(),
                    _ => false,
                }),
        };
        if !valid {
            return Err(invalid("Qwen cache breakpoint must name a nonempty text, image or tool-result content block"));
        }
    }
    Ok(())
}

pub(crate) fn marked(req: &ChatRequest, context: &CodecContext, position: CachePosition) -> bool {
    applies(context)
        && req
            .prompt_cache
            .breakpoints
            .iter()
            .any(|b| b.position == position)
}

pub(crate) fn mark(mut part: WireValue<'_>, marked: bool) -> WireValue<'_> {
    if marked {
        part.insert("cache_control".into(), json!({"type":"ephemeral"}));
    }
    part
}

pub(crate) fn text_content(text: &str, marked: bool) -> WireValue<'_> {
    if marked {
        WireValue::array(vec![mark(
            WireValue::from(json!({"type":"text"})).with("text", WireValue::text(text)),
            true,
        )])
    } else {
        WireValue::text(text)
    }
}

pub(crate) fn system<'a>(req: &'a ChatRequest, context: &CodecContext) -> Option<WireValue<'a>> {
    if !applies(context)
        || !req
            .prompt_cache
            .breakpoints
            .iter()
            .any(|b| matches!(b.position, CachePosition::System { .. }))
    {
        return None;
    }
    let mut parts = Vec::new();
    for (index, block) in req.system.iter().enumerate() {
        if index > 0 {
            parts.push(json!({"type":"text","text":"\n\n"}).into());
        }
        if block.text.is_empty() {
            continue;
        }
        parts.push(mark(
            WireValue::from(json!({"type":"text"})).with("text", WireValue::text(&block.text)),
            marked(req, context, CachePosition::System { index }),
        ));
    }
    Some(WireValue::from(json!({"role":"system"})).with("content", WireValue::array(parts)))
}

/// Feed the existing disjoint-bucket accounting validator its canonical write
/// counter. Retain malformed or conflicting data as incomplete, never as zero.
pub(crate) fn normalize_usage(raw: &mut Value) {
    let Some(details) = raw.get_mut("prompt_tokens_details") else {
        return;
    };
    let Some(details) = details.as_object_mut() else {
        *details = json!({"cache_write_tokens":null});
        return;
    };
    if let Some(created) = details.get("cache_creation_input_tokens").cloned() {
        let write = if details
            .get("cache_write_tokens")
            .is_some_and(|existing| *existing != created)
        {
            Value::Null
        } else {
            created
        };
        details.insert("cache_write_tokens".into(), write);
    }
}
