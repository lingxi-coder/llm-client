//! google provider client and resources.
pub(crate) mod account;
pub mod batch;
pub(crate) mod chat;
mod client;
pub mod computer;
pub mod context_cache;
pub mod file_search;
pub(crate) mod files;
pub mod files_wire;
pub(crate) mod images;
pub mod interactions;
pub mod live;
pub mod speech;
pub mod types;
pub use client::GoogleClient;
pub mod embeddings;
pub mod native;

mod realtime_client;
pub(crate) mod search;

pub(crate) mod attachments;

pub(crate) mod hosted_tools;
