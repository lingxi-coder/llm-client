//! openai provider client and resources.
pub(crate) mod account;
pub mod account_local;
pub mod audio;
pub mod background;
pub mod batches;
pub(crate) mod chat;
pub mod chatgpt_plan;
mod client;
pub mod computer;
pub mod computer_adapter;
pub mod containers;
pub(crate) mod files;
pub(crate) mod images;
pub mod live;
pub(crate) mod mcp_authorization;
pub mod realtime;
pub mod retrieval;
pub mod types;
pub use client::OpenAiClient;
pub mod embeddings;
pub mod native;
pub(crate) mod structured;
pub(crate) mod token_count;

mod realtime_client;
pub(crate) mod search;

pub(crate) mod attachments;

pub(crate) mod responses_policy;

pub(crate) mod code_interpreter;

pub(crate) mod prompt_cache;
