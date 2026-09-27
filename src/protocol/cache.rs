//! Explicit prompt prefix caching. A policy never promises a cache hit.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTtl {
    FiveMinutes,
    /// OpenAI GPT-5.6 and later. On OpenRouter Chat this is a breakpoint TTL;
    /// on Responses it is accepted only as the existing cache-breakpoint
    /// policy's sole supported duration. It is not cache retention.
    ThirtyMinutes,
    OneHour,
}

/// Native Responses prompt-cache write mode for OpenAI GPT-5.6 and later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiPromptCacheMode {
    /// OpenAI chooses an implicit breakpoint in addition to any explicit ones.
    Implicit,
    /// Only explicitly marked content is written; an empty set disables writes.
    Explicit,
}

/// Minimum cache lifetime accepted by the current OpenAI Responses API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiPromptCacheTtl {
    #[serde(rename = "30m")]
    ThirtyMinutes,
}

/// `prompt_cache_options` for OpenAI Responses GPT-5.6 and later.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiPromptCacheOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<OpenAiPromptCacheMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<OpenAiPromptCacheTtl>,
}

/// Maximum prompt-cache retention for explicitly documented model/value pairs.
///
/// This is independent of `OpenAiPromptCacheOptions::ttl`, which sets a
/// minimum lifetime on GPT-5.6 and later. The client never substitutes one
/// field for the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiPromptCacheRetention {
    #[serde(rename = "in_memory")]
    InMemory,
    #[serde(rename = "24h")]
    TwentyFourHours,
}

/// Positions refer to the caller's request, before wire encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CachePosition {
    Tool { index: usize },
    System { index: usize },
    Message { index: usize, block: usize },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheBreakpoint {
    pub position: CachePosition,
    pub ttl: CacheTtl,
}
/// Empty means provider default behavior. This is not gateway response caching.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCachePolicy {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub automatic: Option<CacheTtl>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub breakpoints: Vec<CacheBreakpoint>,
    /// Optional cache-accounting or routing key for OpenAI Responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Native Responses mode and minimum lifetime for GPT-5.6 and later.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_options: Option<OpenAiPromptCacheOptions>,
    /// Explicit maximum retention for a model/value pair that documents support.
    /// Independent of `prompt_cache_options.ttl`; neither field implies the other.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<OpenAiPromptCacheRetention>,
}
