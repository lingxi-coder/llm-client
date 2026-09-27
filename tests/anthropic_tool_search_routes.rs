use lingxi_llm_client::protocol::{
    AnthropicToolSearchConfig, AnthropicToolSearchStrategy, ChatRequest, HostedTool, LlmError,
    ProtocolFamily, ProviderProfile,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, BedrockClaudeCodec, CodecContext, EncodeRequest, RequestMode,
    VertexClaudeCodec, WireCodec,
};
use serde_json::{json, Value};

fn profile(protocol: ProtocolFamily, provider_id: &str, model: &str) -> ProviderProfile {
    let base_url = match protocol {
        ProtocolFamily::BedrockClaude => "https://bedrock-runtime.us-east-1.amazonaws.com",
        ProtocolFamily::VertexClaude => "https://us-east5-aiplatform.googleapis.com/v1",
        _ => "https://api.anthropic.com",
    };
    serde_json::from_value(json!({
        "provider_id": provider_id,
        "profile_name": "anthropic-tool-search-route-test",
        "base_url": base_url,
        "protocol": protocol,
        "auth": "none",
        "models": [{
            "display_model": model,
            "request_model": model,
            "billing_model": model
        }],
        "extra": {}
    }))
    .unwrap()
}

fn request(model: &str) -> ChatRequest {
    serde_json::from_value(json!({
        "model": model,
        "messages": [{"role":"user","content":[{"type":"text","text":"Find a tool"}]}],
        "hosted_tools": [{
            "type": "anthropic_tool_search",
            "config": {"strategy":"regex"}
        }]
    }))
    .unwrap()
}

fn encode(protocol: ProtocolFamily, provider_id: &str, model: &str) -> Result<Value, LlmError> {
    let profile = profile(protocol, provider_id, model);
    let context = CodecContext::new(&profile, model, RequestMode::Complete);
    let request = request(model);
    let http = match protocol {
        ProtocolFamily::AnthropicMessages => {
            AnthropicMessagesCodec.encode_request(EncodeRequest::new(&request), &context)?
        }
        ProtocolFamily::VertexClaude => {
            VertexClaudeCodec.encode_request(EncodeRequest::new(&request), &context)?
        }
        ProtocolFamily::BedrockClaude => {
            BedrockClaudeCodec.encode_request(EncodeRequest::new(&request), &context)?
        }
        _ => unreachable!("this helper covers only Anthropic Tool Search routes"),
    };
    Ok(json!({
        "url": http.url,
        "body": serde_json::from_slice::<Value>(&http.body).unwrap()
    }))
}

const CLAUDE_API_TOOL_SEARCH_MODELS: &[&str] = &[
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
];

const VERTEX_TOOL_SEARCH_MODELS: &[&str] = &[
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

#[test]
fn first_party_route_accepts_the_thirteen_documented_models_and_api_aliases() {
    for model in CLAUDE_API_TOOL_SEARCH_MODELS.iter().copied().chain([
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ]) {
        assert!(
            encode(ProtocolFamily::AnthropicMessages, "anthropic", model).is_ok(),
            "documented Claude API model or alias was rejected: {model}"
        );
    }
}

#[test]
fn first_party_route_rejects_unsupported_models_and_undocumented_aliases() {
    for model in [
        "claude-sonnet-5",
        "claude-opus-4-5-latest",
        "claude-sonnet-4-6-latest",
    ] {
        assert!(
            matches!(
                encode(ProtocolFamily::AnthropicMessages, "anthropic", model),
                Err(LlmError::UnsupportedCapability { .. })
            ),
            "unsupported or undocumented alias unexpectedly passed: {model}"
        );
    }
}

#[test]
fn vertex_accepts_exact_cloud_ids_with_dated_at_suffixes() {
    for model in VERTEX_TOOL_SEARCH_MODELS {
        let encoded = encode(ProtocolFamily::VertexClaude, "my-gcp-proxy", model)
            .unwrap_or_else(|error| panic!("Vertex model {model} was rejected: {error}"));
        assert!(encoded["url"].as_str().unwrap().contains(model));
        assert_eq!(encoded["body"]["anthropic_version"], "vertex-2023-10-16");
    }
}

#[test]
fn vertex_does_not_normalize_api_aliases_or_unknown_wire_suffixes() {
    for model in [
        "claude-opus-4-5",
        "claude-opus-4-5@20251101-preview",
        "claude-sonnet-5",
        "claude-opus-4-5-latest",
    ] {
        assert!(
            matches!(
                encode(
                    ProtocolFamily::VertexClaude,
                    "arbitrary-cloud-profile",
                    model
                ),
                Err(LlmError::UnsupportedCapability { .. })
            ),
            "Vertex model ID was unexpectedly normalized: {model}"
        );
    }
}

#[test]
fn bedrock_invoke_model_preserves_model_ids_and_opaque_arns() {
    for model in [
        "anthropic.claude-opus-5-5",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/team-prod/custom-route-v7",
    ] {
        let encoded = encode(ProtocolFamily::BedrockClaude, "aws-bedrock", model)
            .unwrap_or_else(|error| panic!("Bedrock modelId {model} was rejected: {error}"));
        assert_eq!(
            encoded["url"],
            format!(
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/{}/invoke",
                model.replace('/', "%2F")
            )
        );
        assert_eq!(encoded["body"]["anthropic_version"], "bedrock-2023-05-31");
        assert_eq!(
            encoded["body"]["tools"][0]["type"],
            "tool_search_tool_regex_20251119"
        );
        assert!(encoded["body"].get("model").is_none());
    }
}

#[test]
fn route_test_request_uses_typed_tool_search() {
    // Keep this explicit assertion near the helper so changes to the serde
    // spelling cannot accidentally turn route tests into ordinary requests.
    let request = request("claude-opus-5-5");
    assert!(matches!(
        request.hosted_tools.as_slice(),
        [HostedTool::AnthropicToolSearch(AnthropicToolSearchConfig {
            strategy: AnthropicToolSearchStrategy::Regex
        })]
    ));
}
