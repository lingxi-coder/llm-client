//! What an agent's parts exchange: messages, ids, scope, origin, LLM
//! request/response/error, permission requests, provider profiles, settings.
//! Plain data, and a gate keeps it that way — no `dyn`, no future, no lock, no
//! closure bound (`scripts/check-protocol-is-data.sh`).
//!
//! The contracts that used to sit beside this (`capability`) live in
//! `lingxi-agent-runtime` now, next to their default implementations; see
//! `lingxi_agent_runtime::contracts`.
//!
//! This crate depends on no other crate in the workspace (layering rule ③), so
//! nothing in it can reach back up into the runtime or the LLM layer.
#![forbid(unsafe_code)]

pub mod protocol;
