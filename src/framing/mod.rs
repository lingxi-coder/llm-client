//! Byte-stream framing for a `Transport` that has no parser of its own: SSE
//! for the HTTP wires, the AWS event-stream binary frame for the hosted one.

pub mod eventstream;
pub mod sse;
