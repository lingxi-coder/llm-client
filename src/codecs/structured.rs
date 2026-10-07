//! Output format encoders shared by hosted and direct protocol implementations.
use crate::codecs::CodecContext;
use crate::protocol::{
    CapabilitySupport, ChatRequest, ContentBlock, LlmError, MessageRole, OutputFormat,
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
        validate_extra_fields(&fields(req, context), context)?;
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
    validate_extra_fields(&fields, context)?;
    for (key, value) in fields {
        let mut merged = value;
        if let Some(extra) = context.profile().extra["body"].get(&key) {
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

fn validate_contract(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let profile = context.profile();
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
                return Err(unsupported("Messages requires a JSON schema; unconstrained JSON-object mode is unavailable"));
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
            crate::protocol::structured::compile_request_schema(schema).map_err(invalid)?;
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
        .any(|block| contains_json_keyword(&block.text))
        || req.messages.iter().any(|message| {
            matches!(message.role, MessageRole::System | MessageRole::User)
                && message.content.iter().any(|block| {
                    matches!(block, ContentBlock::Text { text, .. } if contains_json_keyword(text))
                })
        })
}

fn contains_json_keyword(text: &str) -> bool {
    text.as_bytes()
        .windows(4)
        .any(|window| window.eq_ignore_ascii_case(b"json"))
}

/// Reject recursive local references for Anthropic programmatic tool callers.
/// The caller decides whether this contract applies; ordinary tool schemas do
/// not use this helper.
pub(crate) fn validate_no_recursive_schema_references(schema: &Value) -> Result<(), LlmError> {
    validate_schema_references(
        schema,
        "Anthropic programmatic tool calling does not support recursive schema references",
    )
}

pub(crate) fn validate_schema_references(
    schema: &Value,
    recursive_error: &str,
) -> Result<(), LlmError> {
    fn visit(
        schema: &Value,
        root: &Value,
        active_refs: &mut Vec<String>,
        completed_refs: &mut std::collections::HashSet<String>,
        recursive_error: &str,
    ) -> Result<(), LlmError> {
        if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
            if let Some(fragment) = reference
                .strip_prefix('#')
                .and_then(decode_reference_fragment)
            {
                let target = if fragment.is_empty() {
                    Some(root)
                } else {
                    root.pointer(&fragment)
                };
                if let Some(target) = target {
                    if active_refs.iter().any(|active| active == &fragment) {
                        return Err(unsupported(recursive_error));
                    }
                    if !completed_refs.contains(&fragment) {
                        active_refs.push(fragment.clone());
                        visit(target, root, active_refs, completed_refs, recursive_error)?;
                        active_refs.pop();
                        completed_refs.insert(fragment);
                    }
                }
            }
        }
        for key in ["properties", "$defs", "definitions"] {
            if let Some(children) = schema.get(key).and_then(Value::as_object) {
                for child in children.values() {
                    visit(child, root, active_refs, completed_refs, recursive_error)?;
                }
            }
        }
        for key in ["items", "additionalProperties"] {
            if let Some(child) = schema.get(key).filter(|value| value.is_object()) {
                visit(child, root, active_refs, completed_refs, recursive_error)?;
            }
        }
        for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
            if let Some(children) = schema.get(key).and_then(Value::as_array) {
                for child in children {
                    visit(child, root, active_refs, completed_refs, recursive_error)?;
                }
            }
        }
        Ok(())
    }

    visit(
        schema,
        schema,
        &mut Vec::new(),
        &mut std::collections::HashSet::new(),
        recursive_error,
    )
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
                OutputFormat::JsonSchema {
                    name,
                    schema,
                    strict,
                } => {
                    json!({"type":"json_schema","json_schema":{"name":name,"schema":schema,"strict":strict}})
                }
                _ => json!({"type":"json_object"}),
            };
            fields.insert("response_format".into(), value);
        }
        ProtocolFamily::OpenAiResponses => {
            let format = match &req.output_format {
                OutputFormat::JsonSchema {
                    name,
                    schema,
                    strict,
                } => json!({"type":"json_schema", "name":name, "schema":schema, "strict":strict}),
                _ => json!({"type":"json_object"}),
            };
            fields.insert("text".into(), json!({"format":format}));
        }
        ProtocolFamily::AnthropicMessages
        | ProtocolFamily::BedrockClaude
        | ProtocolFamily::VertexClaude
        | ProtocolFamily::FoundryClaude => {
            let OutputFormat::JsonSchema { schema, .. } = &req.output_format else {
                unreachable!()
            };
            fields.insert(
                "output_config".into(),
                json!({"format":{"type":"json_schema","schema":schema}}),
            );
        }
        ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini => {
            let mut text = json!({"mimeType":"application/json"});
            if let OutputFormat::JsonSchema { schema, .. } = &req.output_format {
                text["schema"] = schema.clone();
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
    context: &CodecContext,
) -> Result<(), LlmError> {
    // Text-mode native conflicts are checked by validate_contract.
    if fields.is_empty() {
        return Ok(());
    }
    let p = context.profile();
    // A native schema must match exactly; a recursive merge must not add unvalidated constraints.
    for (key, suffix, pointer) in [
        ("response_format", "", "/response_format"),
        ("text", "/format", "/text/format"),
        ("output_config", "/format", "/output_config/format"),
        (
            "generationConfig",
            "/responseFormat",
            "/generationConfig/responseFormat",
        ),
    ] {
        if let Some(native) = p.extra["body"].pointer(pointer) {
            if Some(native) != fields.get(key).and_then(|value| value.pointer(suffix)) {
                return Err(invalid(format!("conflicting request field {pointer}")));
            }
        }
    }
    for key in ["responseMimeType", "responseSchema", "responseJsonSchema"] {
        if p.extra["body"]["generationConfig"].get(key).is_some() {
            return Err(invalid(
                "native Gemini output format conflicts with the typed output contract",
            ));
        }
    }
    // Detect conflicts before attachment uploads or credential acquisition.
    for (key, value) in fields {
        if let Some(extra) = p.extra["body"].get(key) {
            super::inference::merge_json(&mut value.clone(), extra, key)?;
        }
    }
    Ok(())
}

pub(crate) fn validate_subset(
    schema: &Value,
    anthropic: bool,
    gemini: bool,
    root: bool,
    native_claude: bool,
    qwen_strict_output: bool,
) -> Result<(), LlmError> {
    let object = schema
        .as_object()
        .ok_or_else(|| invalid("structured output requires object-form schemas"))?;
    if root && !anthropic && !gemini && (schema["type"] != "object" || object.contains_key("anyOf"))
    {
        return Err(invalid(
            "strict schema output requires an object root without anyOf",
        ));
    }
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
    for key in object.keys() {
        let extra = if anthropic {
            &["default", "allOf", "minItems", "pattern"][..]
        } else if gemini {
            &[
                "minimum",
                "maximum",
                "prefixItems",
                "minItems",
                "maxItems",
                "propertyOrdering",
            ][..]
        } else {
            &[
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
                "minItems",
                "maxItems",
                "pattern",
            ][..]
        };
        if !COMMON.contains(&key.as_str()) && !extra.contains(&key.as_str()) {
            return Err(unsupported(format!(
                "schema keyword {key} is not supported by this encoder"
            )));
        }
    }
    if native_claude {
        crate::providers::anthropic::structured::validate_native_schema_node(object)?;
    }
    if anthropic {
        if schema["minItems"].as_u64().is_some_and(|n| n > 1) {
            return Err(unsupported(
                "Messages only supports minItems of zero or one",
            ));
        }
        if schema["enum"]
            .as_array()
            .is_some_and(|values| values.iter().any(|v| v.is_array() || v.is_object()))
        {
            return Err(unsupported("Messages does not support complex enum values"));
        }
        if schema["pattern"].as_str().is_some_and(|pattern| {
            pattern.contains("(?")
                || pattern.contains("\\b")
                || pattern.contains("\\B")
                || pattern
                    .as_bytes()
                    .windows(2)
                    .any(|pair| pair[0] == b'\\' && pair[1].is_ascii_digit())
        }) {
            return Err(unsupported(
                "Messages does not support this regex constraint",
            ));
        }
    }
    let unsupported_keys: &[&str] = if anthropic {
        &[
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
        ]
    } else if gemini {
        &[
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
        ]
    } else {
        &[
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
        ]
    };
    for key in unsupported_keys {
        if object.contains_key(*key) {
            return Err(unsupported(format!("schema constraint {key} is not supported by this encoder; constraints are never removed")));
        }
    }
    if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
        if !reference.starts_with('#') {
            return Err(invalid("only local schema references are permitted"));
        }
    }
    let is_object = schema["type"] == "object"
        || schema["type"]
            .as_array()
            .is_some_and(|types| types.iter().any(|t| t == "object"))
        || object.contains_key("properties");
    if is_object && !gemini && !qwen_strict_output {
        if schema["additionalProperties"] != false {
            return Err(invalid(
                "strict object schemas require additionalProperties: false",
            ));
        }
        if !anthropic {
            if let Some(properties) = schema["properties"].as_object() {
                let required = schema["required"].as_array();
                if properties
                    .keys()
                    .any(|key| !required.is_some_and(|r| r.iter().any(|v| v.as_str() == Some(key))))
                {
                    return Err(invalid(
                        "strict OpenAI schemas require every property to be required",
                    ));
                }
            }
        }
    }
    for key in ["properties", "$defs", "definitions"] {
        if let Some(map) = schema[key].as_object() {
            for child in map.values() {
                validate_subset(
                    child,
                    anthropic,
                    gemini,
                    false,
                    native_claude,
                    qwen_strict_output,
                )?;
            }
        }
    }
    for key in ["items", "additionalProperties"] {
        if schema[key].is_object() {
            validate_subset(
                &schema[key],
                anthropic,
                gemini,
                false,
                native_claude,
                qwen_strict_output,
            )?;
        }
    }
    for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
        if let Some(children) = schema[key].as_array() {
            for child in children {
                validate_subset(
                    child,
                    anthropic,
                    gemini,
                    false,
                    native_claude,
                    qwen_strict_output,
                )?;
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
            extra: Value::Null,
            name: format!("tool_{index}"),
            description: String::new(),
            input_schema: schema,
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
            schema: optional(12),
            strict: true,
        };
        assert!(validate(&req, &context).is_ok());
        req.output_format = OutputFormat::JsonSchema {
            name: "answer".into(),
            schema: optional(13),
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
            schema: union_schema(8),
            strict: true,
        };
        assert!(validate(&req, &context).is_ok());
        req.output_format = OutputFormat::JsonSchema {
            name: "answer".into(),
            schema: union_schema(9),
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
            schema: json!({
                "type":"object","properties":{"answer":{"type":"string"}},
                "required":["answer"],"additionalProperties":false
            }),
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
            schema: json!({
                "type":"object","properties":{"answer":{"type":"string"}},
                "required":["answer"],"additionalProperties":false
            }),
        };

        req.messages = vec![crate::protocol::ConversationMessage::assistant(vec![
            ContentBlock::Text {
                text: "{".into(),
                thought_signature: None,
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
                value: json!({
                    "type":"document",
                    "source":{"type":"text","media_type":"text/plain","data":"source"},
                    "citations":{"enabled":true}
                }),
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
                value: json!({
                    "type":"search_result",
                    "source":"https://example.invalid/page",
                    "title":"Source",
                    "content":[{"type":"text","text":"source"}],
                    "citations":{"enabled":true}
                }),
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
            }],
        }];
        assert!(validate(&req, &context).is_ok());

        req.output_format = OutputFormat::Text;
        req.messages = vec![crate::protocol::ConversationMessage::assistant(vec![
            ContentBlock::Text {
                text: "{".into(),
                thought_signature: None,
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
            schema: json!({
                "type":"object","properties":{"answer":{"type":"string"}},
                "required":["answer"],"additionalProperties":false
            }),
        };
        req.messages = vec![crate::protocol::ConversationMessage::assistant(vec![
            ContentBlock::Text {
                text: "{".into(),
                thought_signature: None,
            },
        ])];
        req.tools = (0..21)
            .map(|index| ToolSpec {
                tool_type: None,
                extra: Value::Null,
                name: format!("tool_{index}"),
                description: String::new(),
                input_schema: json!({"type":"object","properties":{},"additionalProperties":false}),
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
