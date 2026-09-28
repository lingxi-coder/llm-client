//! Claude structured output and strict tool schema policy.
use crate::codecs::structured::{
    invalid, unsupported, validate_schema_references, validate_subset,
};
use crate::protocol::{
    ChatRequest, ContentBlock, LlmError, MessageRole, OutputFormat, ProtocolFamily, ToolSpec,
};
use serde_json::{Map, Value};

pub(crate) fn is_claude_messages_profile(profile: &crate::protocol::ProviderProfile) -> bool {
    (profile.provider_id.as_str() == "anthropic"
        && profile.protocol == ProtocolFamily::AnthropicMessages)
        || matches!(
            profile.protocol,
            ProtocolFamily::BedrockClaude
                | ProtocolFamily::VertexClaude
                | ProtocolFamily::FoundryClaude
        )
}

/// Claude JSON outputs cannot be combined with citation-enabled input blocks
/// or an assistant-prefilled final message.
pub(crate) fn validate_messages_output_combinations(req: &ChatRequest) -> Result<(), LlmError> {
    if req
        .hosted_anthropic_web_fetch()
        .is_some_and(|fetch| fetch.citations == Some(true))
    {
        return Err(unsupported(
            "Messages JSON output cannot be combined with citation-enabled Web Fetch",
        ));
    }
    if req
        .messages
        .last()
        .is_some_and(|message| message.role == MessageRole::Assistant)
    {
        return Err(unsupported(
            "Messages JSON output cannot be combined with an assistant-prefilled final message",
        ));
    }
    if req.messages.iter().any(|message| {
        message.content.iter().any(|block| match block {
            ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value,
            } => native_content_enables_citations(value),
            ContentBlock::ToolResult {
                blocks: Some(blocks),
                ..
            } => blocks.iter().any(native_content_enables_citations),
            _ => false,
        })
    }) {
        return Err(unsupported(
            "Messages JSON output cannot be combined with citation-enabled document or search-result blocks",
        ));
    }
    Ok(())
}

/// Strict custom-tool schemas share Claude's JSON Schema limitations, whether
/// they were declared in `tools` or by value in a mid-conversation system
/// message. Exact repeated name/schema pairs count once; changed schemas for
/// the same name count separately because all definitions are in this request.
/// Non-strict tools are left to provider handling.
pub(crate) fn validate_messages_inline_tool_schemas(
    req: &ChatRequest,
    inline_tools: &[ToolSpec],
) -> Result<(), LlmError> {
    let strict_fetch = req.hosted_anthropic_web_fetch().is_some_and(|fetch| {
        fetch.strict
            && !req
                .tools
                .iter()
                .chain(inline_tools)
                .any(|tool| tool.name == "web_fetch" && tool.strict)
    });
    let mut tools = Vec::<&ToolSpec>::new();
    let mut seen = std::collections::HashSet::<(String, String)>::new();
    for tool in req.tools.iter().chain(inline_tools) {
        if tool.strict {
            let key = (
                tool.name.clone(),
                serde_json::to_string(&tool.input_schema)
                    .map_err(|_| invalid("strict Messages tool schema is not valid JSON"))?,
            );
            if !seen.insert(key) {
                continue;
            }
            tools.push(tool);
            if tools.len() + usize::from(strict_fetch) > 20 {
                return Err(unsupported(
                    "Messages allows at most 20 strict tools in one request",
                ));
            }
        }
    }
    for tool in &tools {
        validate_messages_strict_schema(&tool.input_schema)?;
    }
    validate_messages_schema_complexity(req, &tools)
}

/// Validate one strict Messages tool schema using the same supported subset
/// and reference rules as strict top-level tool definitions.
pub(crate) fn validate_messages_strict_schema(schema: &Value) -> Result<(), LlmError> {
    validate_subset(schema, true, false, false, true, false)?;
    validate_messages_schema_references(schema)
}

/// Check only native content blocks that can enable Anthropic citations. This
/// avoids interpreting arbitrary user text or unrelated native metadata as a
/// citations request.
fn native_content_enables_citations(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let block_type = object.get("type").and_then(Value::as_str);
    if matches!(block_type, Some("document" | "search_result"))
        && value.pointer("/citations/enabled") == Some(&Value::Bool(true))
    {
        return true;
    }
    match object.get("content") {
        Some(Value::Array(children)) => children.iter().any(native_content_enables_citations),
        Some(content @ Value::Object(_))
            if matches!(
                block_type,
                Some("web_fetch_tool_result" | "web_fetch_result")
            ) =>
        {
            native_content_enables_citations(content)
        }
        _ => false,
    }
}

/// Local references are allowed, but Claude structured outputs do not accept
/// recursive schemas. Resolve local references while tracking the active
/// reference chain so direct and indirect recursion are caught before dispatch.
pub(crate) fn validate_messages_schema_references(schema: &Value) -> Result<(), LlmError> {
    validate_schema_references(
        schema,
        "Messages structured outputs do not support recursive schemas",
    )
}

/// Claude compiles all strict tool and JSON-output schemas in one request.
/// These documented request-wide limits are checked before any attachments
/// are uploaded or a hosted tool can run.
fn validate_messages_schema_complexity(
    req: &ChatRequest,
    strict_tools: &[&ToolSpec],
) -> Result<(), LlmError> {
    let mut complexity = SchemaComplexity::default();
    for tool in strict_tools {
        complexity.visit(&tool.input_schema);
    }
    if let OutputFormat::JsonSchema { schema, .. } = &req.output_format {
        complexity.visit(schema);
    }
    if complexity.optional_parameters > 24 {
        return Err(unsupported(
            "Messages allows at most 24 optional parameters across strict tools and JSON output",
        ));
    }
    if complexity.union_parameters > 16 {
        return Err(unsupported(
            "Messages allows at most 16 union-typed parameters across strict tools and JSON output",
        ));
    }
    Ok(())
}

#[derive(Default)]
struct SchemaComplexity {
    optional_parameters: usize,
    union_parameters: usize,
}

impl SchemaComplexity {
    fn visit(&mut self, schema: &Value) {
        let Some(object) = schema.as_object() else {
            return;
        };
        if schema.get("anyOf").is_some() || schema.get("type").is_some_and(Value::is_array) {
            self.union_parameters = self.union_parameters.saturating_add(1);
        }
        if let Some(properties) = object.get("properties").and_then(Value::as_object) {
            let required = object.get("required").and_then(Value::as_array);
            for (name, child) in properties {
                if !required
                    .is_some_and(|names| names.iter().any(|value| value.as_str() == Some(name)))
                {
                    self.optional_parameters = self.optional_parameters.saturating_add(1);
                }
                self.visit(child);
            }
        }
        for key in ["$defs", "definitions"] {
            if let Some(definitions) = object.get(key).and_then(Value::as_object) {
                for child in definitions.values() {
                    self.visit(child);
                }
            }
        }
        if let Some(items) = object.get("items") {
            self.visit(items);
        }
        for key in ["anyOf", "allOf", "oneOf", "prefixItems"] {
            if let Some(branches) = object.get(key).and_then(Value::as_array) {
                for child in branches {
                    self.visit(child);
                }
            }
        }
    }
}

pub(crate) fn validate_native_schema_node(object: &Map<String, Value>) -> Result<(), LlmError> {
    if object
        .get("format")
        .is_some_and(|value| !value.as_str().is_some_and(messages_format_supported))
    {
        return Err(unsupported("Messages does not support this string format"));
    }
    if object
        .get("allOf")
        .and_then(Value::as_array)
        .is_some_and(|branches| branches.iter().any(|branch| branch.get("$ref").is_some()))
    {
        return Err(unsupported(
            "Messages does not support $ref directly inside allOf",
        ));
    }
    Ok(())
}
fn messages_format_supported(format: &str) -> bool {
    matches!(
        format,
        "date-time"
            | "time"
            | "date"
            | "duration"
            | "email"
            | "hostname"
            | "uri"
            | "ipv4"
            | "ipv6"
            | "uuid"
    )
}
