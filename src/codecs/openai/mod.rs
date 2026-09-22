//! OpenAI protocol families.
//!
//! Chat Completions and Responses share a vendor and some error conventions,
//! but remain separate wires with separate encoders, decoders, and streams.

pub mod chat;
pub mod responses;
