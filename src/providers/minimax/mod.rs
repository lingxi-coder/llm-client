//! minimax provider client and resources.
pub(crate) mod account;
pub mod async_tts;
pub mod audio;
pub mod bidi_tts;
pub(crate) mod chat;
mod client;
pub(crate) mod files;
pub(crate) mod images;
pub mod streaming_tts;
pub mod tts;
pub mod voices;
pub use client::MiniMaxClient;
pub(crate) mod cache;

mod realtime_client;
pub(crate) mod search;

pub(crate) mod attachments;
