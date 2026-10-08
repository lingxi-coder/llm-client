//! anthropic provider client and resources.
pub(crate) mod account;
pub mod batch;
pub(crate) mod chat;
mod client;
pub(crate) mod client_toolset_history;
pub(crate) mod client_toolsets;
pub(crate) mod code_execution;
pub mod computer;
pub(crate) mod conversation;
pub use conversation::supports_per_message_effort;
pub mod fast_mode;
pub mod fetch_sources;
pub(crate) mod files;
pub(crate) mod mcp;
pub(crate) mod mcp_authorization;
pub mod oauth_source;
mod session_claims;
pub mod session_identity;
pub mod skills;
pub(crate) mod tool_search;
pub mod toolsets;
pub mod types;
pub(crate) mod web_fetch;
pub use client::AnthropicClient;
pub mod native;
pub(crate) mod search;
pub(crate) mod structured;

pub(crate) mod attachments;

pub mod beta_repair;
pub mod error_recognition;
pub mod fallback_request;
pub mod fallback_response;
pub mod limits;
mod metadata_json;
pub mod refusal_fallback;
pub mod request_policy;
pub mod response_policy;
pub mod stream_observation;
pub mod system_prompt;
pub mod thinking_display;

mod connector;
pub use connector::ConnectorTextAccumulator;

/// Explicit conversion to the conservative Messages strict-tool schema subset.
pub mod strict_schema;
