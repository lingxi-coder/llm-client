//! deepseek provider client and resources.

pub(crate) mod account;
pub(crate) mod chat;
mod client;
pub use client::DeepSeekClient;
pub(crate) mod search;
pub(crate) mod structured;
pub(crate) mod token_count;
