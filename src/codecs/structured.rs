//! Output format encoders shared by hosted and direct protocol implementations.
use crate::codecs::CodecContext;
use crate::protocol::{
    CapabilitySupport, ChatRequest, ContentBlock, LlmError, MessageRole, OutputFormat,
    ProtocolFamily, ToolSpec,
};
use serde_json::{json, Map, Value};

fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
fn unsupported(message: impl Into<String>) -> LlmError {
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
    validate_qwen_output_contract(req, profile, context.request_model())?;
    super::cache::validate(req, context)?;
    let claude_messages = is_claude_messages_profile(profile);
    if claude_messages {
        validate_messages_inline_tool_schemas(req, &[])?;
        if matches!(req.output_format, OutputFormat::JsonSchema { .. }) {
            validate_messages_output_combinations(req)?;
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
    // OpenAI's May 2024 GPT-4o snapshot supports JSON mode but predates
    // json_schema response formatting (introduced with the August snapshot).
    if p.provider_id.as_str() == "openai"
        && context.request_model() == "gpt-4o-2024-05-13"
        && matches!(req.output_format, OutputFormat::JsonSchema { .. })
    {
        return Err(unsupported(
            "gpt-4o-2024-05-13 does not support JSON Schema output",
        ));
    }
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
                let qwen_strict_output = is_qwen_openai_chat_profile(profile)
                    && is_qwen_strict_schema_model(context.request_model());
                validate_subset(
                    schema,
                    anthropic,
                    gemini,
                    true,
                    claude_messages,
                    qwen_strict_output,
                )?;
                if claude_messages {
                    validate_messages_schema_references(schema)?;
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

/// Provider-specific prompt requirements apply only to first-party
/// OpenAI-compatible Chat endpoints. Keep these checks separate so the
/// executor can run them before resolving attachments and again before
/// acquiring credentials for each attempted route.
pub(crate) fn validate_qwen_output_contract(
    req: &ChatRequest,
    profile: &crate::protocol::ProviderProfile,
    request_model: &str,
) -> Result<(), LlmError> {
    if is_qwen_openai_chat_profile(profile) {
        match &req.output_format {
            OutputFormat::JsonObject if !has_json_prompt_keyword(req) => {
                return Err(invalid(
                    "Qwen JSON Object mode requires JSON in system or user text",
                ));
            }
            OutputFormat::JsonSchema { strict: true, .. }
                if !is_qwen_strict_schema_model(request_model) =>
            {
                return Err(unsupported(
                    "this client currently verifies strict Qwen JSON Schema output only for qwen3.8-flash and qwen3.8-max",
                ));
            }
            _ => {}
        }
    }

    if is_deepseek_chat_profile(profile)
        && matches!(req.output_format, OutputFormat::JsonObject)
        && !has_json_prompt_keyword(req)
    {
        return Err(invalid(
            "DeepSeek JSON Output requires the word JSON in system or user text",
        ));
    }

    Ok(())
}

fn is_deepseek_chat_profile(profile: &crate::protocol::ProviderProfile) -> bool {
    if profile.provider_id.as_str() != "deepseek" || profile.protocol != ProtocolFamily::OpenAiChat
    {
        return false;
    }
    let Ok(url) = url::Url::parse(&profile.base_url) else {
        return false;
    };
    url.scheme() == "https"
        && url.host_str() == Some("api.deepseek.com")
        && url.username().is_empty()
        && url.password().is_none()
        && url.port().is_none()
        && matches!(url.path(), "/" | "/v1" | "/v1/")
        && url.query().is_none()
        && url.fragment().is_none()
}

fn is_qwen_strict_schema_model(request_model: &str) -> bool {
    matches!(request_model, "qwen3.8-flash" | "qwen3.8-max")
}

fn is_qwen_openai_chat_profile(profile: &crate::protocol::ProviderProfile) -> bool {
    if profile.provider_id.as_str() != "qwen" || profile.protocol != ProtocolFamily::OpenAiChat {
        return false;
    }
    let Ok(url) = url::Url::parse(&profile.base_url) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    let direct_host = matches!(
        host,
        "dashscope.aliyuncs.com"
            | "dashscope-intl.aliyuncs.com"
            | "dashscope-us.aliyuncs.com"
            | "cn-hongkong.dashscope.aliyuncs.com"
    );
    let workspace_host = [
        ".cn-beijing.maas.aliyuncs.com",
        ".ap-southeast-1.maas.aliyuncs.com",
    ]
    .iter()
    .any(|suffix| host.strip_suffix(suffix).is_some_and(valid_dns_label));
    let path = url.path();
    let path_matches = path == "/compatible-mode/v1" || path == "/compatible-mode/v1/";
    url.scheme() == "https"
        && url.username().is_empty()
        && url.password().is_none()
        && url.port().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && path_matches
        && (direct_host || workspace_host)
}

fn valid_dns_label(label: &str) -> bool {
    !label.is_empty()
        && label.len() <= 63
        && label
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        && label.as_bytes()[0].is_ascii_alphanumeric()
        && label.as_bytes()[label.len() - 1].is_ascii_alphanumeric()
}

fn has_json_prompt_keyword(req: &ChatRequest) -> bool {
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

fn is_claude_messages_profile(profile: &crate::protocol::ProviderProfile) -> bool {
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
fn validate_messages_output_combinations(req: &ChatRequest) -> Result<(), LlmError> {
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
fn validate_messages_schema_references(schema: &Value) -> Result<(), LlmError> {
    validate_schema_references(
        schema,
        "Messages structured outputs do not support recursive schemas",
    )
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

fn validate_schema_references(schema: &Value, recursive_error: &str) -> Result<(), LlmError> {
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

fn fields(req: &ChatRequest, context: &CodecContext) -> Map<String, Value> {
    let mut fields = Map::new();
    if matches!(req.output_format, OutputFormat::Text) {
        return fields;
    }
    let p = context.profile();
    match p.protocol {
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

fn validate_subset(
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
            name: format!("tool_{index}"),
            description: String::new(),
            input_schema: schema,
            strict,
            defer_loading: false,
            allowed_callers: vec![],
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
            anthropic: None,
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
            anthropic: None,
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
            anthropic: None,
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
                name: format!("tool_{index}"),
                description: String::new(),
                input_schema: json!({"type":"object","properties":{},"additionalProperties":false}),
                strict: false,
                defer_loading: false,
                allowed_callers: vec![],
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
