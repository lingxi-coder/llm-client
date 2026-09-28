//! openrouter provider client and resources.
pub(crate) mod account;
pub mod audio;
pub mod batch;
pub(crate) mod chat;
mod client;
pub(crate) mod files;
pub(crate) mod images;
pub mod rerank;
pub mod response_cache;
pub(crate) mod server_tools;
pub mod types;
pub use client::OpenRouterClient;
pub mod embeddings;
pub mod native;
pub(crate) mod search;

pub(crate) mod prompt_cache;

pub(crate) mod chat_audio;
