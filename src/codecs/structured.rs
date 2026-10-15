//! Output format encoders shared by hosted and direct protocol implementations.
use crate::codecs::CodecContext;
use crate::protocol::{
    CapabilitySupport, ChatRequest, ContentBlock, JsonPayload, LlmError, MessageRole, OutputFormat,
    ProtocolFamily,
};
use serde_json::{json, Map, Value};

pub(crate) fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
pub(crate) fn unsupported(message: impl Into<String>) -> LlmError {
    LlmError::UnsupportedCapability {
        message: message.into(),
    }
}

pub(crate) fn validate(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    validate_contract(req, context)?;
    // Most profiles have no native output fields. Preflight should not build
    // and discard a second copy of the schema just to check that empty case.
    if [
        "response_format",
        "text",
        "output_config",
        "generationConfig",
    ]
    .iter()
    .any(|key| context.profile().extra["body"].get(key).is_some())
    {
        validate_extra_fields(&fields(req, context), req, context)?;
    }
    Ok(())
}

pub(crate) fn apply(
    req: &ChatRequest,
    context: &CodecContext,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    validate_contract(req, context)?;
    let fields = fields(req, context);
    validate_extra_fields(&fields, req, context)?;
    for (key, value) in fields {
        let mut merged = value;
        if let Some(extra) = context.profile().extra["body"].get(&key) {
            preserve_matching_schema_placeholder(req, context, &key, &mut merged, extra)?;
            super::inference::merge_json(&mut merged, extra, &key)?;
        }
        if let Some(existing) = body.get_mut(&key) {
            super::inference::merge_json(existing, &merged, &key)?;
        } else {
            body.insert(key, merged);
        }
    }
    Ok(())
}

fn preserve_matching_schema_placeholder(
    req: &ChatRequest,
    context: &CodecContext,
    key: &str,
    typed: &mut Value,
    extra: &Value,
) -> Result<(), LlmError> {
    let OutputFormat::JsonSchema { schema, .. } = &req.output_format else {
        return Ok(());
    };
    let pointer = match key {
        "response_format" => "/json_schema/schema",
        "text" | "output_config" => "/format/schema",
        "generationConfig" => "/responseFormat/text/schema",
        _ => return Ok(()),
    };
    let Some(native_schema) = extra.pointer(pointer) else {
        return Ok(());
    };
    if !schema_matches_value(schema, native_schema, context.profile().protocol)? {
        return Ok(());
    }
    if typed.pointer(pointer).is_some_and(Value::is_null) {
        *typed
            .pointer_mut(pointer)
            .expect("schema placeholder path was checked") = native_schema.clone();
    }
    Ok(())
}

fn validate_contract(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let profile = context.profile();
    if profile.provider_id.as_str() == "anthropic" && profile.protocol == ProtocolFamily::OpenAiChat {
        if !matches!(req.output_format, OutputFormat::Text) || req.tools.iter().any(|t| t.strict) {
            return Err(unsupported("Anthropic OpenAI compatibility ignores response_format and strict tools; select Anthropic Messages"));
        }
    }
    crate::providers::dispatch::validate_output_contract(req, profile, context.request_model())?;
    super::cache::validate(req, context)?;
    let claude_messages =
        crate::providers::anthropic::structured::is_claude_messages_profile(profile);
    if claude_messages {
        crate::providers::anthropic::structured::validate_messages_inline_tool_schemas(req, &[])?;
        if matches!(req.output_format, OutputFormat::JsonSchema { .. }) {
            crate::providers::anthropic::structured::validate_messages_output_combinations(req)?;
        }
    }
    if matches!(req.output_format, OutputFormat::Text) {
        for pointer in [
            "/response_format",
            "/text/format",
            "/output_config/format",
            "/generationConfig/responseFormat",
            "/generationConfig/responseMimeType",
            "/generationConfig/responseSchema",
            "/generationConfig/responseJsonSchema",
        ] {
            if context.profile().extra["body"].pointer(pointer).is_some() {
                return Err(invalid(
                    "output format must be set through ChatRequest.output_format",
                ));
            }
        }
        return Ok(());
    }
    let p = context.profile();
    crate::providers::openai::structured::validate_output_contract(req, context)?;
    if p.models.iter().any(|m| {
        m.request_model == context.request_model()
            && m.capability_support.unwrap_or_default().structured_output
                == CapabilitySupport::Unsupported
    }) {
        return Err(unsupported(
            "the selected model explicitly does not support structured output",
        ));
    }
    let anthropic = matches!(
        p.protocol,
        ProtocolFamily::AnthropicMessages
            | ProtocolFamily::BedrockClaude
            | ProtocolFamily::VertexClaude
            | ProtocolFamily::FoundryClaude
    );
    let gemini = matches!(
        p.protocol,
        ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini
    );
    match &req.output_format {
        OutputFormat::Text => unreachable!(),
        OutputFormat::JsonObject => {
            if anthropic {
                return Err(unsupported(
                    "Messages requires a JSON schema; unconstrained JSON-object mode is unavailable",
                ));
            }
        }
        OutputFormat::JsonSchema {
            name,
            schema,
            strict,
        } => {
            if name.is_empty()
                || name.len() > 64
                || !name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
            {
                return Err(invalid(
                    "schema name must contain 1–64 ASCII letters, digits, underscores or hyphens",
                ));
            }
            if (anthropic || gemini) && !strict {
                return Err(unsupported(
                    "this protocol cannot express a non-strict schema output contract",
                ));
            }
            crate::protocol::structured::validate_request_schema(schema).map_err(invalid)?;
            if *strict {
                let qwen_strict_output =
                    crate::providers::qwen::structured::is_qwen_openai_chat_profile(profile)
                        && crate::providers::qwen::structured::is_qwen_strict_schema_model(
                            context.request_model(),
                        );
                validate_subset(
                    schema,
                    anthropic,
                    gemini,
                    true,
                    claude_messages,
                    qwen_strict_output,
                )?;
                if claude_messages {
                    crate::providers::anthropic::structured::validate_messages_schema_references(
                        schema,
                    )?;
                }
            }
        }
    }
    if !anthropic
        && !gemini
        && !matches!(
            p.protocol,
            ProtocolFamily::OpenAiChat
                | ProtocolFamily::AzureOpenAi
                | ProtocolFamily::OpenAiResponses
        )
    {
        return Err(unsupported(
            "no structured-output encoder for this protocol",
        ));
    }
    Ok(())
}

pub(crate) fn has_json_prompt_keyword(req: &ChatRequest) -> bool {
    req.system
        .iter()
        .any(|block| {
            block
                .text_payload()
                .root_string_code_units()
                .is_some_and(|units| contains_json_keyword_utf16(&units))
        })
        || req.messages.iter().any(|message| {
            matches!(message.role, MessageRole::System | MessageRole::User)
                && message.content.iter().any(|block| {
                    matches!(block, ContentBlock::Text { text, .. } | ContentBlock::TextJsUtf16 { text, .. } if contains_json_keyword(text))
                })
        })
}

fn contains_json_keyword(text: &str) -> bool {
    text.as_bytes()
        .windows(4)
        .any(|window| window.eq_ignore_ascii_case(b"json"))
}

fn contains_json_keyword_utf16(text: &[u16]) -> bool {
    text.windows(4).any(|window| {
        window
            .iter()
            .copied()
            .map(u8::try_from)
            .collect::<Result<Vec<_>, _>>()
            .is_ok_and(|bytes| bytes.eq_ignore_ascii_case(b"json"))
    })
}

/// Reject recursive local references for Anthropic programmatic tool callers.
/// The caller decides whether this contract applies; ordinary tool schemas do
/// not use this helper.
pub(crate) fn validate_no_recursive_schema_references(
    schema: &JsonPayload,
) -> Result<(), LlmError> {
    validate_schema_payload_references(
        schema,
        "Anthropic programmatic tool calling does not support recursive schema references",
    )
}

pub(crate) fn validate_schema_payload_references(
    schema: &JsonPayload,
    recursive_error: &str,
) -> Result<(), LlmError> {
    let scanner = schema.scanner()?;
    enum Task {
        Visit(String),
        LeaveReference(String),
    }
    let pointer_child = |parent: &str, token: &str| {
        let escaped = token.replace('~', "~0").replace('/', "~1");
        if parent.is_empty() {
            format!("/{escaped}")
        } else {
            format!("{parent}/{escaped}")
        }
    };
    let mut tasks = vec![Task::Visit(String::new())];
    let mut active_refs = Vec::<String>::new();
    let mut completed_refs = std::collections::HashSet::<String>::new();
    while let Some(task) = tasks.pop() {
        let path = match task {
            Task::Visit(path) => path,
            Task::LeaveReference(reference) => {
                let popped = active_refs.pop();
                debug_assert_eq!(popped.as_deref(), Some(reference.as_str()));
                completed_refs.insert(reference);
                continue;
            }
        };
        let reference_path = pointer_child(&path, "$ref");
        if let Some(units) = scanner.string_units_at(&reference_path) {
            let reference = String::from_utf16(&units).map_err(|_| LlmError::InvalidRequest {
                message: "tool schema references must be valid Unicode".into(),
            })?;
            if let Some(fragment) = reference
                .strip_prefix('#')
                .and_then(decode_reference_fragment)
            {
                let target_exists = if fragment.is_empty() {
                    scanner.kind_at("").is_some()
                } else {
                    fragment.starts_with('/') && scanner.kind_at(&fragment).is_some()
                };
                if target_exists {
                    if active_refs.iter().any(|active| active == &fragment) {
                        return Err(unsupported(recursive_error));
                    }
                    if !completed_refs.contains(&fragment) {
                        active_refs.push(fragment.clone());
                        tasks.push(Task::LeaveReference(fragment.clone()));
                        tasks.push(Task::Visit(fragment));
                    }
                }
            }
        }

        let mut children = Vec::new();
        for key in ["properties", "$defs", "definitions"] {
            let object_path = pointer_child(&path, key);
            if scanner.kind_at(&object_path) != Some(crate::response_json::FlatJsonKind::Object) {
                continue;
            }
            let entries = scanner.object_entries_at(&object_path).ok_or_else(|| {
                LlmError::InvalidRequest {
                    message: "tool schema object cannot be scanned".into(),
                }
            })?;
            for (_, units) in entries {
                let token = String::from_utf16(&units).map_err(|_| LlmError::InvalidRequest {
                    message: "tool schema keys must be valid Unicode".into(),
                })?;
                children.push(pointer_child(&object_path, &token));
            }
        }
        for key in ["items", "additionalProperties"] {
            let child = pointer_child(&path, key);
            if scanner.kind_at(&child) == Some(crate::response_json::FlatJsonKind::Object) {
                children.push(child);
            }
        }
        for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
            let array_path = pointer_child(&path, key);
            if scanner.kind_at(&array_path) != Some(crate::response_json::FlatJsonKind::Array) {
                continue;
            }
            let length = scanner.array_len(&array_path).unwrap_or(0);
            children
                .extend((0..length).map(|index| pointer_child(&array_path, &index.to_string())));
        }
        for child in children.into_iter().rev() {
            tasks.push(Task::Visit(child));
        }
    }
    Ok(())
}

/// JSON Schema references use URI fragments, while `Value::pointer` accepts a
/// decoded JSON Pointer. Decode only the URI percent escapes before resolving.
fn decode_reference_fragment(fragment: &str) -> Option<String> {
    let bytes = fragment.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = *bytes.get(index + 1)?;
            let low = *bytes.get(index + 2)?;
            decoded.push((hex_digit(high)? << 4) | hex_digit(low)?);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).ok()
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn fields(req: &ChatRequest, context: &CodecContext) -> Map<String, Value> {
    let mut fields = Map::new();
    if matches!(req.output_format, OutputFormat::Text) {
        return fields;
    }
    let p = context.profile();
    match p.protocol {
        ProtocolFamily::GeminiInteractions => {}
        ProtocolFamily::OpenAiChat | ProtocolFamily::AzureOpenAi => {
            let value = match &req.output_format {
                OutputFormat::JsonSchema { name, strict, .. } => {
                    json!({"type":"json_schema","json_schema":{"name":name,"schema":null,"strict":strict}})
                }
                _ => json!({"type":"json_object"}),
            };
            fields.insert("response_format".into(), value);
        }
        ProtocolFamily::OpenAiResponses => {
            let format = match &req.output_format {
                OutputFormat::JsonSchema { name, strict, .. } => {
                    json!({"type":"json_schema", "name":name, "schema":null, "strict":strict})
                }
                _ => json!({"type":"json_object"}),
            };
            fields.insert("text".into(), json!({"format":format}));
        }
        ProtocolFamily::AnthropicMessages
        | ProtocolFamily::BedrockClaude
        | ProtocolFamily::VertexClaude
        | ProtocolFamily::FoundryClaude => {
            let OutputFormat::JsonSchema { .. } = &req.output_format else {
                unreachable!()
            };
            fields.insert(
                "output_config".into(),
                json!({"format":{"type":"json_schema","schema":Value::Null}}),
            );
        }
        ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini => {
            let mut text = json!({"mimeType":"application/json"});
            if let OutputFormat::JsonSchema { .. } = &req.output_format {
                text["schema"] = Value::Null;
            }
            fields.insert(
                "generationConfig".into(),
                json!({"responseFormat":{"text":text}}),
            );
        }
    }
    fields
}

fn validate_extra_fields(
    fields: &Map<String, Value>,
    req: &ChatRequest,
    context: &CodecContext,
) -> Result<(), LlmError> {
    // Text-mode native conflicts are checked by validate_contract.
    if fields.is_empty() {
        return Ok(());
    }
    let profile = context.profile();
    let output_schema = match &req.output_format {
        OutputFormat::JsonSchema { schema, .. } => Some(schema),
        _ => None,
    };
    for (key, suffix, pointer, schema_suffix) in [
        (
            "response_format",
            "",
            "/response_format",
            Some("/json_schema/schema"),
        ),
        ("text", "/format", "/text/format", Some("/schema")),
        (
            "output_config",
            "/format",
            "/output_config/format",
            Some("/schema"),
        ),
        (
            "generationConfig",
            "/responseFormat",
            "/generationConfig/responseFormat",
            Some("/text/schema"),
        ),
    ] {
        if let Some(native) = profile.extra["body"].pointer(pointer) {
            let Some(expected) = fields.get(key).and_then(|value| value.pointer(suffix)) else {
                return Err(invalid(format!("conflicting request field {pointer}")));
            };
            let mut native = native.clone();
            if let (Some(schema), Some(schema_suffix)) = (output_schema, schema_suffix) {
                if let Some(native_schema) = native.pointer(schema_suffix) {
                    if !schema_matches_value(schema, native_schema, profile.protocol)? {
                        return Err(invalid(format!("conflicting request field {pointer}")));
                    }
                    *native
                        .pointer_mut(schema_suffix)
                        .expect("checked schema path") = Value::Null;
                }
            }
            if &native != expected {
                return Err(invalid(format!("conflicting request field {pointer}")));
            }
        }
    }
    for key in ["responseMimeType", "responseSchema", "responseJsonSchema"] {
        if profile.extra["body"]["generationConfig"].get(key).is_some() {
            return Err(invalid(
                "native Gemini output format conflicts with the typed output contract",
            ));
        }
    }
    // Detect conflicts before attachment uploads or credential acquisition.
    for (key, value) in fields {
        if let Some(extra) = profile.extra["body"].get(key) {
            let mut merged = value.clone();
            preserve_matching_schema_placeholder(req, context, key, &mut merged, extra)?;
            super::inference::merge_json(&mut merged, extra, key)?;
        }
    }
    Ok(())
}

fn schema_matches_value(
    schema: &JsonPayload,
    value: &Value,
    protocol: ProtocolFamily,
) -> Result<bool, LlmError> {
    let generated = JsonPayload::from_generated_value(value)
        .map_err(|error| invalid(format!("native schema extension is invalid JSON: {error}")))?;
    let encoding = crate::exact_json::JsonEncoding::for_protocol(protocol);
    Ok(schema.canonical_json(encoding)? == generated.canonical_json(encoding)?)
}

pub(crate) fn validate_subset(
    schema: &JsonPayload,
    anthropic: bool,
    gemini: bool,
    root: bool,
    native_claude: bool,
    qwen_strict_output: bool,
) -> Result<(), LlmError> {
    use crate::response_json::{FlatJsonKind, JsonNodeRef};

    fn key_is(units: &[u16], key: &str) -> bool {
        units == key.encode_utf16().collect::<Vec<_>>().as_slice()
    }
    fn field<'a>(node: &JsonNodeRef<'a>, key: &str) -> Option<JsonNodeRef<'a>> {
        node.object_entries()?
            .into_iter()
            .find(|entry| key_is(entry.key_units, key))
            .map(|entry| entry.value)
    }
    fn string_units(node: &JsonNodeRef<'_>) -> Option<Vec<u16>> {
        (node.kind() == FlatJsonKind::String)
            .then(|| node.string_units())
            .flatten()
    }
    fn string_is(node: &JsonNodeRef<'_>, text: &str) -> bool {
        string_units(node).is_some_and(|units| key_is(&units, text))
    }
    fn type_has(node: &JsonNodeRef<'_>, expected: &str) -> bool {
        let Some(value) = field(node, "type") else {
            return false;
        };
        if value.kind() == FlatJsonKind::String {
            return string_is(&value, expected);
        }
        if value.kind() != FlatJsonKind::Array {
            return false;
        }
        (0..value.array_len().unwrap_or(0))
            .filter_map(|index| value.array_item(index))
            .any(|item| string_is(&item, expected))
    }
    fn number_u64(node: &JsonNodeRef<'_>) -> Option<u64> {
        node.number_source()?.parse().ok()
    }
    fn schema_object<'a>(
        node: &JsonNodeRef<'a>,
    ) -> Result<Vec<crate::response_json::JsonObjectEntryRef<'a>>, LlmError> {
        if node.kind() != FlatJsonKind::Object {
            return Err(invalid("structured output requires object-form schemas"));
        }
        node.object_entries()
            .ok_or_else(|| invalid("structured schema object cannot be scanned"))
    }

    let scanner = schema.scanner()?;
    let root_node = scanner.root_node();
    let mut pending = vec![(root_node, root)];
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
    let anthropic_extra: &[&str] = &["default", "allOf", "minItems", "pattern"];
    let gemini_extra: &[&str] = &[
        "minimum",
        "maximum",
        "prefixItems",
        "minItems",
        "maxItems",
        "propertyOrdering",
    ];
    let common_extra: &[&str] = &[
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "pattern",
    ];
    let anthropic_unsupported: &[&str] = &[
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
    let gemini_unsupported: &[&str] = &[
        "allOf",
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
    let openai_unsupported: &[&str] = &[
        "allOf",
        "oneOf",
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

    while let Some((node, is_root)) = pending.pop() {
        let entries = schema_object(&node)?;
        let has_any_of = entries.iter().any(|entry| key_is(entry.key_units, "anyOf"));
        if is_root
            && !anthropic
            && !gemini
            && (!field(&node, "type").is_some_and(|value| string_is(&value, "object"))
                || has_any_of)
        {
            return Err(invalid(
                "strict schema output requires an object root without anyOf",
            ));
        }

        let extra = if anthropic {
            anthropic_extra
        } else if gemini {
            gemini_extra
        } else {
            common_extra
        };
        for entry in &entries {
            let name = String::from_utf16_lossy(entry.key_units);
            let supported = COMMON.iter().any(|key| key_is(entry.key_units, key))
                || extra.iter().any(|key| key_is(entry.key_units, key));
            if !supported {
                return Err(unsupported(format!(
                    "schema keyword {name} is not supported by this encoder"
                )));
            }
            let unsupported_keys = if anthropic {
                anthropic_unsupported
            } else if gemini {
                gemini_unsupported
            } else {
                openai_unsupported
            };
            if unsupported_keys
                .iter()
                .any(|key| key_is(entry.key_units, key))
            {
                return Err(unsupported(format!(
                    "schema constraint {name} is not supported by this encoder; constraints are never removed"
                )));
            }
        }

        if native_claude {
            if let Some(format) = field(&node, "format") {
                let supported = string_units(&format)
                    .and_then(|units| String::from_utf16(&units).ok())
                    .is_some_and(|format| {
                        crate::providers::anthropic::structured::messages_format_supported(&format)
                    });
                if !supported {
                    return Err(unsupported("Messages does not support this string format"));
                }
            }
            if let Some(all_of) = field(&node, "allOf") {
                if all_of.kind() == FlatJsonKind::Array
                    && (0..all_of.array_len().unwrap_or(0))
                        .filter_map(|index| all_of.array_item(index))
                        .any(|branch| {
                            branch.object_entries().is_some_and(|entries| {
                                entries.iter().any(|entry| key_is(entry.key_units, "$ref"))
                            })
                        })
                {
                    return Err(unsupported(
                        "Messages does not support $ref directly inside allOf",
                    ));
                }
            }
        }

        if let Some(reference) = field(&node, "$ref") {
            if let Some(units) = string_units(&reference) {
                let reference = String::from_utf16(&units)
                    .map_err(|_| invalid("schema references must use valid Unicode"))?;
                if !reference.starts_with('#') {
                    return Err(invalid("only local schema references are permitted"));
                }
            }
        }

        if anthropic {
            if field(&node, "minItems")
                .and_then(|value| number_u64(&value))
                .is_some_and(|value| value > 1)
            {
                return Err(unsupported(
                    "Messages only supports minItems of zero or one",
                ));
            }
            if let Some(values) = field(&node, "enum") {
                if values.kind() == FlatJsonKind::Array
                    && (0..values.array_len().unwrap_or(0))
                        .filter_map(|index| values.array_item(index))
                        .any(|value| {
                            matches!(value.kind(), FlatJsonKind::Array | FlatJsonKind::Object)
                        })
                {
                    return Err(unsupported("Messages does not support complex enum values"));
                }
            }
            if let Some(pattern) = field(&node, "pattern") {
                if let Some(units) = string_units(&pattern) {
                    let pattern = String::from_utf16(&units)
                        .map_err(|_| invalid("schema patterns must use valid Unicode"))?;
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
        }

        let is_object = type_has(&node, "object") || field(&node, "properties").is_some();
        if is_object && !gemini && !qwen_strict_output {
            if field(&node, "additionalProperties")
                .is_none_or(|value| value.primitive_source() != Some("false"))
            {
                return Err(invalid(
                    "strict object schemas require additionalProperties: false",
                ));
            }
            if !anthropic {
                if let Some(properties) =
                    field(&node, "properties").filter(|value| value.kind() == FlatJsonKind::Object)
                {
                    let required = field(&node, "required")
                        .filter(|value| value.kind() == FlatJsonKind::Array)
                        .map(|array| {
                            (0..array.array_len().unwrap_or(0))
                                .filter_map(|index| array.array_item(index)?.string_units())
                                .collect::<std::collections::HashSet<_>>()
                        })
                        .unwrap_or_default();
                    if properties
                        .object_entries()
                        .unwrap_or_default()
                        .iter()
                        .any(|property| !required.contains(property.key_units))
                    {
                        return Err(invalid(
                            "strict OpenAI schemas require every property to be required",
                        ));
                    }
                }
            }
        }

        for key in ["properties", "$defs", "definitions"] {
            if let Some(children) =
                field(&node, key).filter(|value| value.kind() == FlatJsonKind::Object)
            {
                pending.extend(
                    children
                        .object_entries()
                        .unwrap_or_default()
                        .into_iter()
                        .map(|entry| (entry.value, false)),
                );
            }
        }
        for key in ["items", "additionalProperties"] {
            if let Some(child) =
                field(&node, key).filter(|value| value.kind() == FlatJsonKind::Object)
            {
                pending.push((child, false));
            }
        }
        for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
            if let Some(children) =
                field(&node, key).filter(|value| value.kind() == FlatJsonKind::Array)
            {
                pending.extend(
                    (0..children.array_len().unwrap_or(0))
                        .filter_map(|index| children.array_item(index))
                        .map(|child| (child, false)),
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::RequestMode;
    use crate::protocol::{ProviderProfile, ToolSpec};

    #[test]
    fn messages_rejects_request_wide_schema_complexity_before_dispatch() {
        let profile: ProviderProfile = serde_json::from_value(json!({
            "provider_id":"anthropic", "profile_name":"anthropic",
            "base_url":"https://api.anthropic.com", "protocol":"anthropic_messages",
            "auth":"none"
        }))
        .unwrap();
        let context = CodecContext::new(&profile, "claude-opus-5-5", RequestMode::Complete);
        let mut req: ChatRequest =
            serde_json::from_value(json!({"model":"claude-opus-5-5","messages":[]})).unwrap();
        let tool = |index, strict, schema| ToolSpec {
            tool_type: None,
            extra: JsonPayload::parse("null").unwrap(),
            name: format!("tool_{index}"),
            description: String::new().into(),
            input_schema: crate::protocol::JsonPayload::from_generated_value(&schema).unwrap(),
            strict,
            defer_loading: false,
            native_options: Vec::new(),
        };
        let simple = json!({"type":"object","properties":{},"additionalProperties":false});
        req.tools = (0..20)
            .map(|index| tool(index, true, simple.clone()))
            .collect();
        assert!(validate(&req, &context).is_ok());
        req.tools.push(tool(20, true, simple.clone()));
        assert!(matches!(
            validate(&req, &context),
            Err(LlmError::UnsupportedCapability { .. })
        ));
        let mut gateway = profile.clone();
        gateway.provider_id = "openrouter".into();
        let gateway_context =
            CodecContext::new(&gateway, "anthropic/claude-sonnet-4", RequestMode::Complete);
        assert!(validate(&req, &gateway_context).is_ok());
        req.tools[20].strict = false;
        assert!(validate(&req, &context).is_ok());

        req.tools.clear();
        let optional = |count| {
            let properties = (0..count)
                .map(|index| (format!("field_{index}"), json!({"type":"string"})))
                .collect::<Map<String, Value>>();
            json!({"type":"object","properties":properties,"additionalProperties":false})
        };
        req.tools.push(tool(0, true, optional(12)));
        req.output_format = OutputFormat::JsonSchema {
            name: "answer".into(),
            schema: JsonPayload::from_generated_value(&optional(12)).unwrap(),
            strict: true,
        };
        assert!(validate(&req, &context).is_ok());
        req.output_format = OutputFormat::JsonSchema {
            name: "answer".into(),
            schema: JsonPayload::from_generated_value(&optional(13)).unwrap(),
            strict: true,
        };
        assert!(matches!(
            validate(&req, &context),
            Err(LlmError::UnsupportedCapability { .. })
        ));

        let union_schema = |count| {
            let properties = (0..count)
                .map(|index| {
                    (
                        format!("field_{index}"),
                        json!({"anyOf":[{"type":"string"},{"type":"null"}]}),
                    )
                })
                .collect::<Map<String, Value>>();
            json!({"type":"object","properties":properties,"additionalProperties":false})
        };
        req.tools = vec![tool(0, true, union_schema(8))];
        req.output_format = OutputFormat::JsonSchema {
            name: "answer".into(),
            schema: JsonPayload::from_generated_value(&union_schema(8)).unwrap(),
            strict: true,
        };
        assert!(validate(&req, &context).is_ok());
        req.output_format = OutputFormat::JsonSchema {
            name: "answer".into(),
            schema: JsonPayload::from_generated_value(&union_schema(9)).unwrap(),
            strict: true,
        };
        assert!(matches!(
            validate(&req, &context),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }

    #[test]
    fn preflight_and_encoding_agree_about_native_output_conflicts() {
        let mut req: ChatRequest =
            serde_json::from_value(json!({"model":"test","messages":[]})).unwrap();
        req.output_format = OutputFormat::JsonSchema {
            name: "answer".into(),
            strict: true,
            schema: JsonPayload::from_generated_value(&json!({
                "type":"object","properties":{"answer":{"type":"string"}},
                "required":["answer"],"additionalProperties":false
            }))
            .unwrap(),
        };
        for protocol in [
            ProtocolFamily::OpenAiChat,
            ProtocolFamily::OpenAiResponses,
            ProtocolFamily::AnthropicMessages,
            ProtocolFamily::GeminiGenerateContent,
        ] {
            let mut profile: ProviderProfile = serde_json::from_value(json!({
                "provider_id":"test","profile_name":"test",
                "base_url":"https://test.invalid/v1","protocol":protocol,"auth":"none"
            }))
            .unwrap();
            for extra in [
                json!({}),
                json!({"body":{"text":{"verbosity":"low"}}}),
                json!({"body":{"text":false}}),
                json!({"body":{"response_format":{"type":"json_object"}}}),
                json!({"body":{"output_config":{"format":{"type":"json_object"}}}}),
                json!({"body":{"generationConfig":{"responseMimeType":"application/json"}}}),
                json!({"body":{"generationConfig":false}}),
            ] {
                profile.extra = extra;
                let context = CodecContext::new(&profile, "test", RequestMode::Complete);
                let preflight = validate(&req, &context).map_err(|error| error.to_string());
                let encoded =
                    apply(&req, &context, &mut Map::new()).map_err(|error| error.to_string());
                assert_eq!(preflight, encoded, "{protocol:?}: {}", profile.extra);
            }
        }
    }

    #[test]
    fn messages_json_output_preflight_rejects_prefill_and_enabled_citations() {
        let anthropic: ProviderProfile = serde_json::from_value(json!({
            "provider_id":"anthropic", "profile_name":"anthropic",
            "base_url":"https://api.anthropic.com", "protocol":"anthropic_messages",
            "auth":"none"
        }))
        .unwrap();
        let context = CodecContext::new(&anthropic, "claude-opus-5-5", RequestMode::Complete);
        let mut req: ChatRequest =
            serde_json::from_value(json!({"model":"claude-opus-5-5","messages":[]})).unwrap();
        req.output_format = OutputFormat::JsonSchema {
            name: "answer".into(),
            strict: true,
            schema: JsonPayload::from_generated_value(&json!({
                "type":"object","properties":{"answer":{"type":"string"}},
                "required":["answer"],"additionalProperties":false
            }))
            .unwrap(),
        };

        req.messages = vec![crate::protocol::ConversationMessage::assistant(vec![
            ContentBlock::Text {
                text: "{".into(),
                thought_signature: None,
                citations: None,
            },
        ])];
        assert!(matches!(
            validate(&req, &context),
            Err(LlmError::UnsupportedCapability { .. })
        ));

        req.messages = vec![crate::protocol::ConversationMessage {
            native_options: Vec::new(),
            role: MessageRole::User,
            content: vec![ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value: JsonPayload::from_generated_value(&json!({
                    "type":"document",
                    "source":{"type":"text","media_type":"text/plain","data":"source"},
                    "citations":{"enabled":true}
                }))
                .unwrap(),
            }],
        }];
        assert!(matches!(
            validate(&req, &context),
            Err(LlmError::UnsupportedCapability { .. })
        ));

        req.messages = vec![crate::protocol::ConversationMessage {
            native_options: Vec::new(),
            role: MessageRole::User,
            content: vec![ContentBlock::ProviderContent {
                protocol: ProtocolFamily::AnthropicMessages,
                value: JsonPayload::from_generated_value(&json!({
                    "type":"search_result",
                    "source":"https://example.invalid/page",
                    "title":"Source",
                    "content":[{"type":"text","text":"source"}],
                    "citations":{"enabled":true}
                }))
                .unwrap(),
            }],
        }];
        assert!(matches!(
            validate(&req, &context),
            Err(LlmError::UnsupportedCapability { .. })
        ));

        // A regular user document has no citations flag in the typed API, and
        // text that happens to mention citation JSON is not native metadata.
        req.messages = vec![crate::protocol::ConversationMessage {
            native_options: Vec::new(),
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: r#"{"type":"document","citations":{"enabled":true}}"#.into(),
                thought_signature: None,
                citations: None,
            }],
        }];
        assert!(validate(&req, &context).is_ok());

        req.output_format = OutputFormat::Text;
        req.messages = vec![crate::protocol::ConversationMessage::assistant(vec![
            ContentBlock::Text {
                text: "{".into(),
                thought_signature: None,
                citations: None,
            },
        ])];
        assert!(validate(&req, &context).is_ok());
    }

    #[test]
    fn messages_output_combinations_do_not_expand_to_compatible_gateways_or_non_strict_tools() {
        let mut gateway: ProviderProfile = serde_json::from_value(json!({
            "provider_id":"acme", "profile_name":"gateway",
            "base_url":"https://gateway.invalid", "protocol":"anthropic_messages",
            "auth":"none"
        }))
        .unwrap();
        let mut req: ChatRequest =
            serde_json::from_value(json!({"model":"claude-like","messages":[]})).unwrap();
        req.output_format = OutputFormat::JsonSchema {
            name: "answer".into(),
            strict: true,
            schema: JsonPayload::from_generated_value(&json!({
                "type":"object","properties":{"answer":{"type":"string"}},
                "required":["answer"],"additionalProperties":false
            }))
            .unwrap(),
        };
        req.messages = vec![crate::protocol::ConversationMessage::assistant(vec![
            ContentBlock::Text {
                text: "{".into(),
                thought_signature: None,
                citations: None,
            },
        ])];
        req.tools = (0..21)
            .map(|index| ToolSpec {
                tool_type: None,
                extra: JsonPayload::parse("null").unwrap(),
                name: format!("tool_{index}"),
                description: String::new().into(),
                input_schema: crate::protocol::JsonPayload::from_generated_value(
                    &json!({"type":"object","properties":{},"additionalProperties":false}),
                )
                .unwrap(),
                strict: false,
                defer_loading: false,
                native_options: Vec::new(),
            })
            .collect();
        let context = CodecContext::new(&gateway, "claude-like", RequestMode::Complete);
        assert!(validate(&req, &context).is_ok());

        gateway.provider_id = "anthropic".into();
        let context = CodecContext::new(&gateway, "claude-opus-5-5", RequestMode::Complete);
        assert!(matches!(
            validate(&req, &context),
            Err(LlmError::UnsupportedCapability { .. })
        ));

        req.messages = vec![crate::protocol::ConversationMessage::user_text("continue")];
        assert!(validate(&req, &context).is_ok());
    }
}
