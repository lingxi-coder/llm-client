//! Shared OpenAI-compatible embedding request wire format.
use crate::{embeddings::EmbeddingRequest, protocol::LlmError};
use serde_json::{json, Value};
pub(crate) fn body(req: &EmbeddingRequest) -> Value {
    let mut body = json!({"model":req.model,"input":req.input,"encoding_format":"float"});
    if let Some(dim) = req.dimensions {
        body["dimensions"] = json!(dim);
    }
    body
}
pub(crate) fn reject_task(req: &EmbeddingRequest) -> Result<(), LlmError> {
    if req.task.is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI embedding wire cannot encode task type".into(),
        });
    }
    Ok(())
}
