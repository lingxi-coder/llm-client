//! github_copilot provider client and resources.

pub mod account_local;
pub(crate) mod chat;
mod client;
pub use client::GithubCopilotClient;
