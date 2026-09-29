//! Provider login and token protocols, independent of platform credential stores.
//! Each network operation uses the injected SDK transport and bounded executor.
//! Callers own polling, persistence and refresh coordination.

pub mod anthropic;
pub mod copilot;
pub mod openai;
pub mod pkce;
