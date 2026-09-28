//! xai provider client and resources.
pub(crate) mod account;
pub mod audio;
pub mod batch;
pub(crate) mod chat;
mod client;
pub mod collections;
pub mod deferred;
pub(crate) mod files;
pub(crate) mod images;
pub mod realtime;
pub mod streaming_tts;
pub mod stt;
pub mod types;
pub use client::XaiClient;
pub mod native;

mod realtime_client;
pub(crate) mod search;

pub(crate) mod responses_policy;
