//! Route and model validation for Anthropic's server-side tool-search tool.

use crate::codecs::CodecContext;
use crate::protocol::{ChatRequest, LlmError, ProtocolFamily};

// These are the exact model IDs in Anthropic's current Tool Search
// compatibility table. The Claude API also documents unpinned convenience
// aliases for the three pre-4.6 snapshots in the table; no other aliases are
// inferred here.
const CLAUDE_API_MODELS_AND_ALIASES: &[&str] = &[
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
];

// Vertex's published model IDs match the Claude API for current dateless
// models. Its versioned pre-4.6 models use `@YYYYMMDD`, so list those exact
// cloud IDs rather than stripping or rewriting arbitrary suffixes.
const VERTEX_MODELS: &[&str] = &[
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5@20251101",
    "claude-sonnet-4-5@20250929",
    "claude-haiku-4-5@20251001",
];

// Microsoft Foundry uses unpinned model IDs for its pre-4.6 Anthropic-hosted
// deployments. Its documented model/hosting intersection is narrower on Azure.
const FOUNDRY_ANTHROPIC_MODELS: &[&str] = &[
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
];

const FOUNDRY_AZURE_MODELS: &[&str] = &[
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-haiku-4-5",
];

fn unsupported(message: impl Into<String>) -> LlmError {
    LlmError::UnsupportedCapability {
        message: message.into(),
    }
}

/// Validate only the hosted Anthropic Tool Search declaration. Bedrock's
/// dedicated codec uses InvokeModel and keeps its modelId (including IDs and
/// ARNs) opaque; Vertex and the Claude API have explicit model-ID tables.
pub(crate) fn validate(request: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    if request.hosted_anthropic_tool_search().is_none() {
        return Ok(());
    }

    let profile = context.profile();
    match profile.protocol {
        ProtocolFamily::AnthropicMessages => {
            if profile.provider_id.as_str() != "anthropic" {
                return Err(unsupported(
                    "Anthropic hosted tool search requires the first-party Anthropic Messages provider",
                ));
            }
            require_model(
                context.request_model(),
                CLAUDE_API_MODELS_AND_ALIASES,
                "Claude API",
            )
        }
        ProtocolFamily::VertexClaude => {
            require_model(context.request_model(), VERTEX_MODELS, "Vertex AI")
        }
        ProtocolFamily::FoundryClaude => {
            let deployment = crate::hosting::foundry::require_deployment(context)?;
            let supported = match deployment.hosting {
                crate::protocol::FoundryHosting::Anthropic => FOUNDRY_ANTHROPIC_MODELS,
                crate::protocol::FoundryHosting::Azure => FOUNDRY_AZURE_MODELS,
            };
            require_model(
                &deployment.model_id,
                supported,
                "Microsoft Foundry Tool Search",
            )
        }
        // The Anthropic compatibility table documents Bedrock InvokeModel,
        // while Bedrock accepts either model IDs or inference-profile ARNs.
        // Keep the established Bedrock route and pass modelId through without
        // guessing the underlying model from a prefix or custom ARN name.
        ProtocolFamily::BedrockClaude => Ok(()),
        _ => Err(unsupported(format!(
            "Anthropic hosted tool search is not supported by the {:?} codec",
            profile.protocol
        ))),
    }
}

fn require_model(model: &str, supported: &[&str], platform: &str) -> Result<(), LlmError> {
    if supported.contains(&model) {
        Ok(())
    } else {
        Err(unsupported(format!(
            "Anthropic hosted tool search is not documented for model {model:?} on {platform}"
        )))
    }
}
