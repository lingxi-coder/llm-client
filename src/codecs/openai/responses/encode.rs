//! Request encoding for the Responses API.

use crate::codecs::json::{WireRequest, WireValue};
use crate::codecs::{CodecContext, EncodeRequest};
use crate::protocol::{
    ContentBlock, ConversationMessage, DocumentSource, ImageSource, LlmError, MessageRole,
    ProviderProfile, ToolChoice, ToolSpec,
};
use crate::providers::openai::{
    computer::{
        OpenAiComputerCall, OpenAiComputerCallOutput, OpenAiComputerToolConfig,
        OPENAI_COMPUTER_CALL_FORMAT, OPENAI_COMPUTER_CALL_OUTPUT_FORMAT,
        OPENAI_COMPUTER_TOOL_FORMAT,
    },
    types::OpenAiToolSearchExecution,
};

use base64::Engine;
use serde_json::{json, Map, Value};

pub fn request<'a>(
    wire: EncodeRequest<'a>,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<WireRequest<'a>, LlmError> {
    let req = wire.request();
    let computer_tool = validate_computer_request(req, profile)?;
    if req.anthropic_mcp_servers().next().is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI Responses cannot encode the Anthropic MCP connector".into(),
        });
    }
    crate::providers::openai::prompt_cache::validate(req, opts)?;
    crate::providers::anthropic::code_execution::validate(req, opts)?;
    req.validate_hosted_tools()?;
    crate::providers::openrouter::server_tools::validate(req, profile, None, false)?;
    let has_openrouter_tool_search = req.hosted_openrouter_tool_search().is_some();
    let openai_tool_search = req.hosted_openai_tool_search();
    let has_openai_tool_search = openai_tool_search.is_some();
    if req.hosted_anthropic_tool_search().is_some()
        || req.hosted_anthropic_web_fetch().is_some()
        || (req.tools.iter().any(|tool| tool.defer_loading)
            && !has_openrouter_tool_search
            && !has_openai_tool_search)
    {
        return Err(LlmError::UnsupportedCapability {
            message: "Anthropic tool search and Web Fetch require Messages, and defer_loading requires a supported hosted tool-search server tool".into(),
        });
    }
    crate::providers::openai::responses_policy::validate_tool_search(
        req,
        profile,
        &opts.request_model,
    )?;
    crate::files::validate_direct_provider_file_inputs_at(
        wire.blocks(),
        profile,
        &opts.request_model,
        opts.file_scope(),
        opts.file_validation_time(),
    )?;
    if req.has_response_continuation()
        && !profile
            .extra
            .get("supports_previous_response_id")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        return Err(LlmError::UnsupportedCapability {
            message: format!(
                "profile {:?} does not declare previous_response_id support",
                profile.profile_name
            ),
        });
    }
    crate::providers::qwen::responses_policy::validate_file_search_route(req, profile)?;
    if req.remote_mcp_servers().next().is_some() {
        crate::providers::openai::responses_policy::validate_mcp_enabled(profile)?;
    }
    crate::providers::xai::responses_policy::validate_xai_remote_mcp(req, profile)?;
    if req.hosted_file_search().is_some()
        && (req.remote_mcp_servers().next().is_some()
            || req.xai_remote_mcp_servers().next().is_some())
    {
        return Err(LlmError::UnsupportedCapability {
            message: "Qwen File Search and remote MCP cannot share one Responses route".into(),
        });
    }
    if !req.stop_sequences.is_empty() {
        // This wire has no stop parameter. Dropping them would run the model
        // without a limit the caller asked for and never say so.
        return Err(LlmError::InvalidRequest {
            message: "the Responses API has no stop-sequence parameter".to_owned(),
        });
    }

    let mut input = Vec::new();
    if let Some(system) = crate::providers::openai::prompt_cache::system_input(req) {
        input.push(system);
    }
    for (message_index, m) in req.messages.iter().enumerate() {
        encode_message(m, message_index, &mut input, profile, opts, wire)?;
    }

    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        Value::String(opts.request_model.clone()),
    );
    if !req.system.is_empty() && !crate::providers::openai::prompt_cache::has_system_breakpoint(req)
    {
        body.insert(
            "instructions".to_owned(),
            Value::String(
                req.system
                    .iter()
                    .map(|b| b.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            ),
        );
    }
    body.insert("input".to_owned(), Value::Null);
    if let Some(id) = &req.continuation {
        body.insert(
            "previous_response_id".to_owned(),
            Value::String(id.response_id.as_str().to_owned()),
        );
    }
    if !req.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(
                req.tools
                    .iter()
                    .map(|tool| {
                        encode_tool(tool, has_openrouter_tool_search || has_openai_tool_search)
                    })
                    .collect(),
            ),
        );
        body.insert(
            "tool_choice".to_owned(),
            encode_tool_choice(&req.tool_choice),
        );
    }
    if let Some(config) = openai_tool_search {
        body.entry("tools")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Responses hosted tools must encode as an array".into(),
            })?
            .push(encode_openai_tool_search(config));
    }
    if computer_tool.is_some() {
        body.entry("tools")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Responses hosted tools must encode as an array".into(),
            })?
            .push(json!({"type":"computer"}));
    }
    // OpenAI's current web_search wire has its own filter names and supports
    // both allow/block lists together. Other Responses adapters keep using the
    // shared provider-specific adapter.
    crate::providers::openai::responses_policy::apply_web_search(req, profile, &mut body)?;
    crate::providers::qwen::web_extractor::apply(req, profile, &opts.request_model, &mut body)?;
    if !crate::providers::qwen::hosted::apply(req, profile, &opts.request_model, &mut body)? {
        crate::providers::openai::code_interpreter::apply(req, opts, &mut body)?;
    }
    crate::providers::qwen::responses_policy::apply_file_search(
        req,
        profile,
        &opts.request_model,
        &mut body,
    )?;
    let mcp_servers: Vec<_> = req.remote_mcp_servers().collect();
    if !mcp_servers.is_empty() {
        crate::providers::openai::responses_policy::validate_mcp_enabled(profile)?;
        let tools = body.entry("tools").or_insert_with(|| json!([]));
        let Some(tools) = tools.as_array_mut() else {
            return Err(LlmError::InvalidRequest {
                message: "Responses hosted tools must encode as an array".into(),
            });
        };
        for mcp in mcp_servers {
            mcp.validate()?;
            let mut tool = json!({
                "type": "mcp",
                "server_label": mcp.server_label(),
                "server_url": mcp.server_url()
            });
            if let Some(description) = mcp.server_description() {
                tool["server_description"] = Value::String(description.to_owned());
            }
            if mcp.defer_loading() {
                tool["defer_loading"] = Value::Bool(true);
            }
            if !mcp.allowed_tools().is_empty() {
                tool["allowed_tools"] = json!(mcp.allowed_tools());
            }
            if let Some(policy) = mcp.require_approval() {
                tool["require_approval"] = Value::String(
                    match policy {
                        crate::providers::openai::types::McpApprovalPolicy::Always => "always",
                        crate::providers::openai::types::McpApprovalPolicy::Never => "never",
                    }
                    .into(),
                );
            }
            tools.push(tool);
        }
    }
    let xai_mcp_servers: Vec<_> = req.xai_remote_mcp_servers().collect();
    if !xai_mcp_servers.is_empty() {
        let tools = body.entry("tools").or_insert_with(|| json!([]));
        let Some(tools) = tools.as_array_mut() else {
            return Err(LlmError::InvalidRequest {
                message: "Responses hosted tools must encode as an array".into(),
            });
        };
        for mcp in xai_mcp_servers {
            mcp.validate()?;
            let mut tool = json!({
                "type": "mcp",
                "server_label": mcp.server_label(),
                "server_url": mcp.server_url()
            });
            if let Some(description) = mcp.server_description() {
                tool["server_description"] = Value::String(description.to_owned());
            }
            if !mcp.allowed_tools().is_empty() {
                tool["allowed_tools"] = json!(mcp.allowed_tools());
            }
            // xAI's confirmed Responses subset does not include OpenAI's
            // require_approval or connector_id fields.
            tools.push(tool);
        }
    }
    for tool in crate::providers::openrouter::server_tools::response_tool_values(req) {
        body.entry("tools")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Responses hosted tools must encode as an array".into(),
            })?
            .push(tool);
    }
    // Apply the choice to the final tool set, including a sole hosted file
    // search tool. An omitted choice would silently enable automatic search.
    if body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
    {
        if let ToolChoice::Tool { name } = &req.tool_choice {
            if !req.tools.iter().any(|tool| tool.name == *name) {
                return Err(LlmError::InvalidRequest {
                    message: format!("named client tool {name:?} is not advertised"),
                });
            }
        }
        body.entry("tool_choice")
            .or_insert_with(|| encode_tool_choice(&req.tool_choice));
    }
    if let Some(max) = req.max_tokens {
        body.insert("max_output_tokens".to_owned(), Value::from(max));
    }
    if let Some(t) = req.temperature {
        body.insert("temperature".to_owned(), Value::from(t));
    }
    if opts.stream {
        body.insert("stream".to_owned(), Value::Bool(true));
    }

    crate::providers::openai::prompt_cache::apply(req, opts, &mut body)?;
    let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
    crate::codecs::inference::apply(req, opts, &mut body, &mut headers)?;
    crate::codecs::request_controls::apply(req, profile.protocol, &mut body)?;
    crate::codecs::structured::apply(req, opts, &mut body)?;
    crate::wire_options::merge_body(profile, &mut body);
    crate::wire_options::merge_headers(profile, &mut headers);

    let url = if let Some(search) = req.hosted_file_search() {
        crate::providers::qwen::responses_policy::file_search_url(profile, &search.workspace_id)?
    } else {
        format!("{}/responses", profile.base_url.trim_end_matches('/'))
    };
    let mut body = WireValue::from(Value::Object(body)).with("input", WireValue::array(input));
    if !req.tools.is_empty() {
        body = body.map_array_field("tools", |index, tool| {
            let has_schema = tool.get("parameters").is_some();
            let tool = WireValue::from(tool);
            match req.tools.get(index) {
                Some(spec) if has_schema => {
                    tool.with("parameters", WireValue::borrowed(&spec.input_schema))
                }
                _ => tool,
            }
        });
    }
    Ok(WireRequest::new(url, headers, body))
}

/// A message becomes one item; a tool call or result becomes its own sibling
/// item, so any text collected so far is flushed first to keep the order.
fn encode_message<'a>(
    m: &'a ConversationMessage,
    message_index: usize,
    input: &mut Vec<WireValue<'a>>,
    profile: &ProviderProfile,
    opts: &CodecContext,
    wire: EncodeRequest<'a>,
) -> Result<(), LlmError> {
    let role = match m.role {
        MessageRole::Assistant => "assistant",
        MessageRole::User => "user",
        MessageRole::System => "system",
    };
    // The assistant's text is an output part; everyone else's is input.
    let text_part = if m.role == MessageRole::Assistant {
        "output_text"
    } else {
        "input_text"
    };
    let mut parts: Vec<WireValue<'a>> = Vec::new();

    for (block_index, b) in m.content.iter().enumerate() {
        let b = wire.block(b);
        if let Some(media) = inline(wire, b, opts)? {
            parts.push(media);
            continue;
        }
        match b {
            ContentBlock::ProviderContent { protocol, value } => {
                if *protocol != crate::protocol::ProtocolFamily::OpenAiResponses {
                    return Err(LlmError::UnsupportedCapability {
                        message: "native content cannot be replayed on a different protocol"
                            .to_owned(),
                    });
                }
                if matches!(
                    value.get("type").and_then(Value::as_str),
                    Some("computer_call" | "computer_call_output")
                ) {
                    return Err(LlmError::UnsupportedCapability {
                        message: "OpenAI computer calls and outputs must use their validated typed native blocks".into(),
                    });
                }
                validate_provider_item_replay(value, wire.request())?;
                flush(role, &mut parts, input);
                let item = if crate::providers::openai::prompt_cache::has_message_breakpoint(
                    wire.request(),
                    message_index,
                    block_index,
                ) {
                    crate::providers::openai::prompt_cache::mark_provider_message(value)
                } else {
                    WireValue::borrowed(value)
                };
                input.push(item);
            }
            ContentBlock::Text { text, .. } => {
                let part =
                    WireValue::from(json!({"type": text_part})).with("text", WireValue::text(text));
                parts.push(crate::providers::openai::prompt_cache::mark_message_block(
                    wire.request(),
                    message_index,
                    block_index,
                    part,
                ));
            }
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
            ContentBlock::Image { source } => {
                let mut part = json!({"type": "input_image"});
                match source {
                    ImageSource::Base64 { media_type, data } => {
                        part["image_url"] = json!(format!("data:{media_type};base64,{data}"));
                    }
                    ImageSource::Url { url } => part["image_url"] = json!(url),
                    ImageSource::Attachment { .. } => {
                        return Err(crate::codecs::unresolved_attachment_error());
                    }
                    ImageSource::ProviderFile { file } => {
                        let file = crate::codecs::validate_provider_file(file, profile, opts)?;
                        if file.protocol != crate::protocol::ProtocolFamily::OpenAiResponses {
                            return Err(crate::codecs::provider_file_protocol_error());
                        }
                        part["file_id"] = json!(file.file_id);
                    }
                }
                parts.push((part).into());
            }
            ContentBlock::Document { source, title } => {
                let mut part = json!({"type": "input_file"});
                match source {
                    DocumentSource::Url { url } => part["file_url"] = json!(url),
                    DocumentSource::Base64 { media_type, data } => {
                        part["file_data"] = json!(format!("data:{media_type};base64,{data}"));
                    }
                    DocumentSource::Text { media_type, data } => {
                        let encoded =
                            base64::engine::general_purpose::STANDARD.encode(data.as_bytes());
                        part["file_data"] = json!(format!("data:{media_type};base64,{encoded}"));
                    }
                    DocumentSource::Attachment { .. } => {
                        return Err(crate::codecs::unresolved_attachment_error());
                    }
                    DocumentSource::ProviderFile { file } => {
                        let file = crate::codecs::validate_provider_file(file, profile, opts)?;
                        if file.protocol != crate::protocol::ProtocolFamily::OpenAiResponses {
                            return Err(crate::codecs::provider_file_protocol_error());
                        }
                        part["file_id"] = json!(file.file_id);
                    }
                }
                if part.get("file_data").is_some() {
                    part["filename"] = json!(title.as_deref().unwrap_or("document"));
                }
                parts.push((part).into());
            }
            ContentBlock::Audio { .. } => {
                return Err(LlmError::UnsupportedCapability {
                    message: "Responses codec does not support Chat audio blocks".into(),
                });
            }
            ContentBlock::Video { .. } => {
                return Err(LlmError::UnsupportedCapability {
                    message: "Responses input does not support video content blocks".into(),
                });
            }
            ContentBlock::ToolUse {
                id,
                name,
                input: args,
                ..
            } => {
                flush(role, &mut parts, input);
                input.push(
                    (json!({
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        // Arguments go on the wire as a JSON string, as on the
                        // Chat wire.
                        "arguments": args.to_string(),
                    }))
                    .into(),
                );
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                blocks,
                ..
            } => {
                flush(role, &mut parts, input);
                let output = crate::providers::openai::prompt_cache::tool_result_output(
                    wire.request(),
                    message_index,
                    block_index,
                    content,
                    blocks.as_deref(),
                );
                input.push(
                    WireValue::from(json!({
                        "type": "function_call_output",
                        "call_id": tool_use_id,
                    }))
                    .with("output", output),
                );
            }
            ContentBlock::Native { value } => {
                if value.format() == OPENAI_COMPUTER_CALL_FORMAT {
                    // Model-generated calls are output items. Continuations
                    // must send only the caller's matching typed output.
                    OpenAiComputerCall::from_extension(value)?;
                    return Err(LlmError::InvalidRequest {
                        message: "replay OpenAI computer work with its previous_response_id and a typed computer_call_output, not a computer_call input item".into(),
                    });
                }
                if value.format() != OPENAI_COMPUTER_CALL_OUTPUT_FORMAT {
                    return Err(LlmError::UnsupportedCapability {
                        message: format!(
                            "Responses cannot encode native content format {}",
                            value.format()
                        ),
                    });
                }
                if m.role != MessageRole::User {
                    return Err(LlmError::InvalidRequest {
                        message: "computer_call_output must be supplied as user input".into(),
                    });
                }
                if wire.request().continuation.is_none() {
                    return Err(LlmError::InvalidRequest {
                        message: "computer_call_output requires its scoped previous_response_id continuation".into(),
                    });
                }
                if computer_tool_config(wire.request())?.is_none() {
                    return Err(LlmError::InvalidRequest {
                        message: "computer_call_output requires the typed OpenAI computer tool declaration".into(),
                    });
                }
                let output = OpenAiComputerCallOutput::from_extension(value)?;
                output.validate_for_submission()?;
                let item =
                    serde_json::to_value(output).map_err(|error| LlmError::InvalidRequest {
                        message: format!("cannot encode typed computer_call_output: {error}"),
                    })?;
                flush(role, &mut parts, input);
                input.push(item.into());
            }
        }
    }
    flush(role, &mut parts, input);
    Ok(())
}

fn computer_tool_config(
    request: &crate::protocol::ChatRequest,
) -> Result<Option<&OpenAiComputerToolConfig>, LlmError> {
    let mut count = 0;
    for extension in &request.native_options {
        if extension.format() == OPENAI_COMPUTER_TOOL_FORMAT {
            count += 1;
            extension.decode::<OpenAiComputerToolConfig>()?;
        }
    }
    if count > 1 {
        return Err(LlmError::InvalidRequest {
            message: "OpenAI computer tool may be declared only once".into(),
        });
    }
    if count == 0 {
        return Ok(None);
    }
    request
        .openai_computer_tool()
        .map(Some)
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "OpenAI computer tool configuration could not be decoded".into(),
        })
}

fn reject_duplicate_or_raw_computer_declarations(
    request: &crate::protocol::ChatRequest,
    profile: &ProviderProfile,
) -> Result<(), LlmError> {
    if request
        .tools
        .iter()
        .any(|tool| tool.tool_type.as_deref() == Some("computer"))
    {
        return Err(LlmError::InvalidRequest {
            message: "declare OpenAI computer use through the typed native computer tool option, not a raw ToolSpec".into(),
        });
    }
    if request.hosted_tools.iter().any(|tool| {
        matches!(tool, crate::protocol::HostedTool::Native(extension)
            if extension.format() == "openai.hosted_tool.v1" && extension.data()["type"] == "computer")
    }) {
        return Err(LlmError::InvalidRequest {
            message: "OpenAI computer use must use the typed native computer tool option, not a hosted-tool payload".into(),
        });
    }
    if let Some(raw_tools) = profile.extra.pointer("/body/tools") {
        let entries = raw_tools
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(std::slice::from_ref(raw_tools));
        if entries
            .iter()
            .any(|tool| tool.get("type").and_then(Value::as_str) == Some("computer"))
        {
            return Err(LlmError::InvalidRequest {
                message: "profile.extra.body.tools cannot inject or duplicate the typed OpenAI computer tool".into(),
            });
        }
    }
    Ok(())
}

pub(super) fn validate_computer_request<'a>(
    request: &'a crate::protocol::ChatRequest,
    profile: &ProviderProfile,
) -> Result<Option<&'a OpenAiComputerToolConfig>, LlmError> {
    let computer_tool = computer_tool_config(request)?;
    reject_duplicate_or_raw_computer_declarations(request, profile)?;

    let mut has_computer_content = false;
    for message in &request.messages {
        for block in &message.content {
            match block {
                ContentBlock::ProviderContent { value, .. }
                    if matches!(
                        value.get("type").and_then(Value::as_str),
                        Some("computer_call" | "computer_call_output")
                    ) =>
                {
                    return Err(LlmError::UnsupportedCapability {
                        message: "OpenAI computer calls and outputs must use their validated typed native blocks".into(),
                    });
                }
                ContentBlock::Native { value } if value.format() == OPENAI_COMPUTER_CALL_FORMAT => {
                    OpenAiComputerCall::from_extension(value)?;
                    return Err(LlmError::InvalidRequest {
                        message: "replay OpenAI computer work with its previous_response_id and a typed computer_call_output, not a computer_call input item".into(),
                    });
                }
                ContentBlock::Native { value }
                    if value.format() == OPENAI_COMPUTER_CALL_OUTPUT_FORMAT =>
                {
                    has_computer_content = true;
                    if message.role != MessageRole::User {
                        return Err(LlmError::InvalidRequest {
                            message: "computer_call_output must be supplied as user input".into(),
                        });
                    }
                    if request.continuation.is_none() {
                        return Err(LlmError::InvalidRequest {
                            message: "computer_call_output requires its scoped previous_response_id continuation".into(),
                        });
                    }
                    if computer_tool.is_none() {
                        return Err(LlmError::InvalidRequest {
                            message: "computer_call_output requires the typed OpenAI computer tool declaration".into(),
                        });
                    }
                    OpenAiComputerCallOutput::from_extension(value)?.validate_for_submission()?;
                }
                _ => {}
            }
        }
    }

    if (computer_tool.is_some() || has_computer_content)
        && !crate::providers::openai::responses_policy::is_official_openai_responses_profile(
            profile,
        )
    {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI Responses computer use requires the official OpenAI Responses profile"
                .into(),
        });
    }

    Ok(computer_tool)
}

fn validate_provider_item_replay(
    value: &Value,
    request: &crate::protocol::ChatRequest,
) -> Result<(), LlmError> {
    if value.get("type").and_then(Value::as_str) != Some("mcp_approval_response") {
        return Ok(());
    }
    let approval_id = value
        .get("approval_request_id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty() && !id.chars().any(char::is_control));
    if approval_id.is_none() || value.get("approve").and_then(Value::as_bool).is_none() {
        return Err(LlmError::InvalidRequest {
            message: "MCP approval response requires approval_request_id and approve".into(),
        });
    }
    if request.remote_mcp_servers().next().is_none() {
        return Err(LlmError::InvalidRequest {
            message: "MCP approval response requires its remote MCP server definition".into(),
        });
    }
    Ok(())
}

fn flush<'a>(role: &str, parts: &mut Vec<WireValue<'a>>, input: &mut Vec<WireValue<'a>>) {
    if parts.is_empty() {
        return;
    }
    input.push(
        WireValue::from(json!({"type":"message", "role":role}))
            .with("content", WireValue::array(std::mem::take(parts))),
    );
}

fn encode_tool(t: &ToolSpec, include_defer_loading: bool) -> Value {
    let mut tool = json!({
        "type": "function",
        "name": t.name,
        "description": t.description,
        "parameters": Value::Null,
        "strict": t.strict,
    });
    if include_defer_loading && t.defer_loading {
        tool["defer_loading"] = Value::Bool(true);
    }
    crate::codecs::request_controls::tool_extensions(t, tool)
}

fn encode_openai_tool_search(
    config: &crate::providers::openai::types::OpenAiToolSearchConfig,
) -> Value {
    let mut tool = json!({"type":"tool_search"});
    if config.execution == OpenAiToolSearchExecution::Client {
        tool["execution"] = Value::String("client".into());
        tool["description"] = json!(config.description);
        tool["parameters"] = config.parameters.clone().unwrap_or(Value::Null);
    }
    tool
}

fn encode_tool_choice(c: &ToolChoice) -> Value {
    match c {
        ToolChoice::Auto => Value::String("auto".to_owned()),
        ToolChoice::Any => Value::String("required".to_owned()),
        ToolChoice::None => Value::String("none".to_owned()),
        ToolChoice::Tool { name } => json!({"type": "function", "name": name}),
    }
}

fn inline<'a>(
    wire: EncodeRequest<'a>,
    block: &ContentBlock,
    _context: &CodecContext,
) -> Result<Option<WireValue<'a>>, LlmError> {
    let Some(media) = wire.inline_media(block)? else {
        return Ok(None);
    };
    let attachment = media.attachment;
    let (kind, title) = match block {
        ContentBlock::Image { .. } => ("image", None),
        ContentBlock::Document { title, .. } => ("document", title.as_deref()),
        ContentBlock::Video { .. } => ("video", None),
        _ => unreachable!("inline_media accepts only attachment blocks"),
    };
    let _data = || WireValue::base64(media.bytes, String::new());
    let uri = || {
        WireValue::base64(
            media.bytes,
            format!("data:{};base64,", attachment.media_type),
        )
    };
    Ok(Some(match kind {
        "image" => WireValue::from(json!({"type":"input_image"})).with("image_url", uri()),
        "document" => WireValue::from(
            json!({"type":"input_file","filename":title.unwrap_or(&attachment.filename)}),
        )
        .with("file_data", uri()),
        _ => {
            return Err(LlmError::UnsupportedCapability {
                message: "Responses does not support video content blocks".into(),
            });
        }
    }))
}
