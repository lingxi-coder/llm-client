//! Explicit deployment identity for hosting- and model-dependent Foundry tools.

use super::CodecContext;
use crate::protocol::{FoundryDeployment, FoundryHosting, LlmError, ProtocolFamily};

// Exact underlying IDs from the Foundry model/hosting table. Deployment names
// are arbitrary and are never parsed or replaced with these IDs on the wire.
const AZURE_MODELS: &[&str] = &[
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-haiku-4-5",
];
const ANTHROPIC_MODELS: &[&str] = &[
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    // The Foundry guide separately documents this invited research preview.
    "claude-mythos-preview",
];

fn unsupported(message: impl Into<String>) -> LlmError {
    LlmError::UnsupportedCapability {
        message: message.into(),
    }
}

/// Resolve caller-supplied facts for a tool requiring a known hosting/model.
/// An ambiguous wire-ID-only context must not choose a different catalog row.
pub(crate) fn require_deployment(context: &CodecContext) -> Result<&FoundryDeployment, LlmError> {
    if context.profile().protocol != ProtocolFamily::FoundryClaude {
        return Err(unsupported(
            "Foundry deployment identity requires the Foundry Claude codec",
        ));
    }
    let mut rows = context
        .profile()
        .models
        .iter()
        .filter(|model| model.request_model == context.request_model());
    let first = rows.next().and_then(|model| model.foundry.as_ref())
        .ok_or_else(|| unsupported("this Foundry tool requires ModelProfile.foundry with explicit hosting and underlying model_id"))?;
    if rows.any(|model| model.foundry.as_ref() != Some(first)) {
        return Err(unsupported("ambiguous Foundry deployment identity; select a model row with CodecContext::for_model"));
    }
    let models = match first.hosting {
        FoundryHosting::Azure => AZURE_MODELS,
        FoundryHosting::Anthropic => ANTHROPIC_MODELS,
    };
    if !models.contains(&first.model_id.as_str()) {
        return Err(unsupported(format!(
            "Foundry model {:?} is not documented for {:?} hosting",
            first.model_id, first.hosting
        )));
    }
    Ok(first)
}
