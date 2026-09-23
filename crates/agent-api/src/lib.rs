//! What an agent's parts exchange: messages, ids, scope, origin, LLM
//! request/response/error, permission requests, provider profiles, settings.
//! Plain data: no runtime behavior, I/O, or dependency on the LLM client.
//!
//! This crate depends on no other crate in the workspace, so protocol types
//! remain usable without an agent runtime or LLM transport.
#![forbid(unsafe_code)]

pub mod protocol;
