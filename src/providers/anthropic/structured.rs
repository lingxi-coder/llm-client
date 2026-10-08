//! Claude structured output and strict tool schema policy.
use crate::codecs::structured::{invalid, unsupported};
use crate::protocol::{
    ChatRequest, ContentBlock, JsonPayload, LlmError, MessageRole, OutputFormat, ProtocolFamily,
    ToolSpec,
};

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
            } => native_payload_enables_citations(value),
            ContentBlock::ToolResult {
                blocks: Some(blocks),
                ..
            } => blocks
                .array_items()
                .is_some_and(|items| items.iter().any(native_payload_enables_citations)),
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
                tool.input_schema
                    .canonical_json(crate::exact_json::JsonEncoding::Serde)
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
pub(crate) fn validate_messages_strict_schema(schema: &JsonPayload) -> Result<(), LlmError> {
    validate_messages_tool_schema_payload(schema)?;
    crate::codecs::structured::validate_schema_payload_references(
        schema,
        "Messages structured outputs do not support recursive schemas",
    )
}

/// Validate the provider's known JSON Schema vocabulary directly from the flat
/// source tree. The schema remains opaque to serialization and does not pass
/// through `serde_json::Value` before provider encoding.
fn validate_messages_tool_schema_payload(schema: &JsonPayload) -> Result<(), LlmError> {
    use crate::response_json::FlatJsonKind;

    const COMMON: &[&str] = &[
        "$schema",
        "$defs",
        "definitions",
        "$ref",
        "title",
        "description",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "anyOf",
        "format",
    ];
    const MESSAGES_EXTRA: &[&str] = &["default", "allOf", "minItems", "pattern"];
    const UNSUPPORTED: &[&str] = &[
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "not",
        "if",
        "then",
        "else",
        "dependentRequired",
        "dependentSchemas",
        "patternProperties",
        "uniqueItems",
        "contains",
    ];
    let scanner = schema.scanner()?;
    let mut pending = vec![scanner.root_node()];
    while let Some(node) = pending.pop() {
        if node.kind() != FlatJsonKind::Object {
            return Err(invalid("structured output requires object-form schemas"));
        }
        let entries = node
            .object_entries()
            .ok_or_else(|| invalid("strict Messages schema object cannot be scanned"))?;
        let field = |key: &str| {
            let units = key.encode_utf16().collect::<Vec<_>>();
            entries
                .iter()
                .find(|entry| entry.key_units == units)
                .map(|entry| entry.value)
        };
        for entry in &entries {
            let known = COMMON
                .iter()
                .chain(MESSAGES_EXTRA)
                .any(|key| entry.key_units == key.encode_utf16().collect::<Vec<_>>().as_slice());
            if !known {
                return Err(unsupported(format!(
                    "schema keyword {} is not supported by this encoder",
                    String::from_utf16_lossy(entry.key_units)
                )));
            }
        }
        for key in UNSUPPORTED {
            if field(key).is_some() {
                return Err(unsupported(format!(
                    "schema constraint {key} is not supported by this encoder; constraints are never removed"
                )));
            }
        }

        if let Some(format) = field("format") {
            let Some(units) = format.string_units() else {
                return Err(unsupported("Messages does not support this string format"));
            };
            let format = String::from_utf16(&units)
                .map_err(|_| unsupported("Messages schema format must be valid Unicode"))?;
            if !messages_format_supported(&format) {
                return Err(unsupported("Messages does not support this string format"));
            }
        }
        if let Some(all_of) = field("allOf") {
            if all_of.kind() == FlatJsonKind::Array
                && (0..all_of.array_len().unwrap_or(0)).any(|index| {
                    all_of
                        .array_item(index)
                        .and_then(|branch| branch.object_entries())
                        .is_some_and(|fields| {
                            fields.iter().any(|entry| {
                                entry.key_units == "$ref".encode_utf16().collect::<Vec<_>>()
                            })
                        })
                })
            {
                return Err(unsupported(
                    "Messages does not support $ref directly inside allOf",
                ));
            }
        }
        if field("minItems")
            .and_then(|value| value.number_source())
            .and_then(|source| source.parse::<u64>().ok())
            .filter(|count| *count > 1)
            .is_some()
        {
            return Err(unsupported(
                "Messages only supports minItems of zero or one",
            ));
        }
        if let Some(values) = field("enum") {
            if values.kind() == FlatJsonKind::Array
                && (0..values.array_len().unwrap_or(0)).any(|index| {
                    values.array_item(index).is_some_and(|value| {
                        matches!(value.kind(), FlatJsonKind::Array | FlatJsonKind::Object)
                    })
                })
            {
                return Err(unsupported("Messages does not support complex enum values"));
            }
        }
        if let Some(pattern) = field("pattern") {
            if let Some(units) = pattern.string_units() {
                let pattern = String::from_utf16(&units)
                    .map_err(|_| unsupported("Messages schema patterns must be valid Unicode"))?;
                if pattern.contains("(?")
                    || pattern.contains("\\b")
                    || pattern.contains("\\B")
                    || pattern
                        .as_bytes()
                        .windows(2)
                        .any(|pair| pair[0] == b'\\' && pair[1].is_ascii_digit())
                {
                    return Err(unsupported(
                        "Messages does not support this regex constraint",
                    ));
                }
            }
        }
        if let Some(reference) = field("$ref") {
            if let Some(units) = reference.string_units() {
                let reference = String::from_utf16(&units)
                    .map_err(|_| invalid("tool schema references must be valid Unicode"))?;
                if !reference.starts_with('#') {
                    return Err(invalid("only local schema references are permitted"));
                }
            }
        }

        let type_is_object = field("type").is_some_and(|kind| match kind.kind() {
            FlatJsonKind::String => kind
                .string_units()
                .is_some_and(|units| units == "object".encode_utf16().collect::<Vec<_>>()),
            FlatJsonKind::Array => (0..kind.array_len().unwrap_or(0)).any(|index| {
                kind.array_item(index).is_some_and(|item| {
                    item.string_units()
                        .is_some_and(|units| units == "object".encode_utf16().collect::<Vec<_>>())
                })
            }),
            _ => false,
        });
        let is_object = type_is_object || field("properties").is_some();
        if is_object
            && field("additionalProperties").and_then(|value| value.primitive_source())
                != Some("false")
        {
            return Err(invalid(
                "strict object schemas require additionalProperties: false",
            ));
        }

        for key in ["properties", "$defs", "definitions"] {
            if let Some(children) = field(key).filter(|value| value.kind() == FlatJsonKind::Object)
            {
                pending.extend(
                    children
                        .object_entries()
                        .ok_or_else(|| invalid("strict Messages schema object cannot be scanned"))?
                        .into_iter()
                        .map(|entry| entry.value),
                );
            }
        }
        for key in ["items", "additionalProperties"] {
            if let Some(child) = field(key).filter(|value| value.kind() == FlatJsonKind::Object) {
                pending.push(child);
            }
        }
        for key in ["anyOf", "allOf", "oneOf", "prefixItems"] {
            if let Some(children) = field(key).filter(|value| value.kind() == FlatJsonKind::Array) {
                pending.extend(
                    (0..children.array_len().unwrap_or(0))
                        .filter_map(|index| children.array_item(index)),
                );
            }
        }
    }
    Ok(())
}

fn native_payload_enables_citations(value: &JsonPayload) -> bool {
    use crate::response_json::{FlatJsonKind, JsonNodeRef};

    let Ok(scanner) = value.scanner() else {
        return true;
    };
    let exact_key = |units: &[u16], key: &str| units == key.encode_utf16().collect::<Vec<_>>();
    let mut pending = vec![scanner.root_node()];
    while let Some(node) = pending.pop() {
        match node.kind() {
            FlatJsonKind::Array => {
                if let Some(len) = node.array_len() {
                    for index in (0..len).rev() {
                        if let Some(child) = node.array_item(index) {
                            pending.push(child);
                        }
                    }
                }
            }
            FlatJsonKind::Object => {
                let Some(fields) = node.object_entries() else {
                    continue;
                };
                let mut block_type = None;
                let mut citations: Option<JsonNodeRef<'_>> = None;
                let mut content: Option<JsonNodeRef<'_>> = None;
                for field in fields {
                    if exact_key(field.key_units, "type") {
                        block_type = field.value.string_at().map(str::to_owned);
                    } else if exact_key(field.key_units, "citations") {
                        citations = Some(field.value);
                    } else if exact_key(field.key_units, "content") {
                        content = Some(field.value);
                    }
                }
                let citations_enabled = citations.is_some_and(|citations| {
                    let Some(fields) = citations.object_entries() else {
                        return false;
                    };
                    fields.into_iter().any(|field| {
                        exact_key(field.key_units, "enabled")
                            && field.value.primitive_source() == Some("true")
                    })
                });
                if matches!(block_type.as_deref(), Some("document" | "search_result"))
                    && citations_enabled
                {
                    return true;
                }
                if let Some(content) = content {
                    match (block_type.as_deref(), content.kind()) {
                        (_, FlatJsonKind::Array) => {
                            if let Some(len) = content.array_len() {
                                for index in (0..len).rev() {
                                    if let Some(child) = content.array_item(index) {
                                        pending.push(child);
                                    }
                                }
                            }
                        }
                        (
                            Some("web_fetch_tool_result" | "web_fetch_result"),
                            FlatJsonKind::Object,
                        ) => pending.push(content),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    false
}

/// Local references are allowed, but Claude structured outputs do not accept
/// recursive schemas. Resolve local references while tracking the active
/// reference chain so direct and indirect recursion are caught before dispatch.
pub(crate) fn validate_messages_schema_references(schema: &JsonPayload) -> Result<(), LlmError> {
    crate::codecs::structured::validate_schema_payload_references(
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
        complexity.visit_payload(&tool.input_schema)?;
    }
    if let OutputFormat::JsonSchema { schema, .. } = &req.output_format {
        complexity.visit_payload(schema)?;
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
    fn visit_payload(&mut self, schema: &JsonPayload) -> Result<(), LlmError> {
        use crate::response_json::FlatJsonKind;
        let scanner = schema.scanner()?;
        let key_is =
            |units: &[u16], key: &str| units == key.encode_utf16().collect::<Vec<_>>().as_slice();
        let mut pending = vec![scanner.root_node()];
        while let Some(node) = pending.pop() {
            if node.kind() != FlatJsonKind::Object {
                continue;
            }
            let fields = node
                .object_entries()
                .ok_or_else(|| invalid("Messages schema object cannot be scanned"))?;
            let has_any_of = fields.iter().any(|field| key_is(field.key_units, "anyOf"));
            let type_is_array = fields.iter().any(|field| {
                key_is(field.key_units, "type") && field.value.kind() == FlatJsonKind::Array
            });
            if has_any_of || type_is_array {
                self.union_parameters = self.union_parameters.saturating_add(1);
            }

            let properties = fields
                .iter()
                .find(|field| key_is(field.key_units, "properties"))
                .map(|field| field.value);
            if let Some(properties) = properties.filter(|node| node.kind() == FlatJsonKind::Object)
            {
                let required = fields
                    .iter()
                    .find(|field| key_is(field.key_units, "required"))
                    .map(|field| field.value)
                    .filter(|node| node.kind() == FlatJsonKind::Array)
                    .into_iter()
                    .flat_map(|node| {
                        (0..node.array_len().unwrap_or(0))
                            .filter_map(move |index| node.array_item(index)?.string_units())
                    })
                    .collect::<std::collections::HashSet<_>>();
                for property in properties.object_entries().unwrap_or_default() {
                    if !required.contains(property.key_units) {
                        self.optional_parameters = self.optional_parameters.saturating_add(1);
                    }
                    pending.push(property.value);
                }
            }

            for key in ["$defs", "definitions"] {
                if let Some(definitions) = fields
                    .iter()
                    .find(|field| key_is(field.key_units, key))
                    .map(|field| field.value)
                    .filter(|node| node.kind() == FlatJsonKind::Object)
                {
                    pending.extend(
                        definitions
                            .object_entries()
                            .unwrap_or_default()
                            .into_iter()
                            .map(|field| field.value),
                    );
                }
            }
            if let Some(items) = fields
                .iter()
                .find(|field| key_is(field.key_units, "items"))
                .map(|field| field.value)
            {
                pending.push(items);
            }
            for key in ["anyOf", "allOf", "oneOf", "prefixItems"] {
                if let Some(branches) = fields
                    .iter()
                    .find(|field| key_is(field.key_units, key))
                    .map(|field| field.value)
                    .filter(|node| node.kind() == FlatJsonKind::Array)
                {
                    pending.extend(
                        (0..branches.array_len().unwrap_or(0))
                            .filter_map(|index| branches.array_item(index)),
                    );
                }
            }
        }
        Ok(())
    }
}

pub(crate) fn messages_format_supported(format: &str) -> bool {
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
