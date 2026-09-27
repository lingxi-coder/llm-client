//! Request encoding for the Responses API.

use crate::codecs::json::{WireRequest, WireValue};
use crate::codecs::{CodecContext, EncodeRequest};
use crate::protocol::{
    ContentBlock, ConversationMessage, DocumentSource, ImageSource, LlmError, MessageRole,
    OpenAiToolSearchExecution, ProviderProfile, ToolChoice, ToolSpec,
};

use base64::Engine;
use serde_json::{json, Map, Value};

pub fn request<'a>(
    wire: EncodeRequest<'a>,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<WireRequest<'a>, LlmError> {
    let req = wire.request();
    if req.anthropic_mcp_servers().next().is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI Responses cannot encode the Anthropic MCP connector".into(),
        });
    }
    super::cache::validate(req, opts)?;
    crate::codecs::anthropic_code_execution::validate(req, opts)?;
    req.validate_hosted_tools()?;
    crate::codecs::openrouter_server_tools::validate(req, profile, None, false)?;
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
    let deferred_mcp = req
        .remote_mcp_servers()
        .any(|server| server.defer_loading());
    if deferred_mcp && !has_openai_tool_search {
        return Err(LlmError::UnsupportedCapability {
            message: "deferred OpenAI MCP servers require OpenAI Responses tool search".into(),
        });
    }
    if let Some(config) = openai_tool_search {
        if !is_official_openai_responses_profile(profile) {
            return Err(LlmError::UnsupportedCapability {
                message: "OpenAI tool search requires the official OpenAI Responses profile".into(),
            });
        }
        if !supports_openai_tool_search_model(&opts.request_model) {
            return Err(LlmError::UnsupportedCapability {
                message: format!(
                    "OpenAI Responses tool search requires GPT-5.4 or later; model {:?} is not supported",
                    opts.request_model
                ),
            });
        }
        config.validate()?;
        if deferred_mcp && config.execution != OpenAiToolSearchExecution::Server {
            return Err(LlmError::UnsupportedCapability {
                message: "deferred OpenAI MCP servers require server-executed tool search".into(),
            });
        }
        if config.execution == OpenAiToolSearchExecution::Server
            && !req.tools.iter().any(|tool| tool.defer_loading)
            && !deferred_mcp
        {
            return Err(LlmError::InvalidRequest {
                message:
                    "server-executed OpenAI tool search requires a deferred function or MCP server"
                        .into(),
            });
        }
    }
    crate::files::validate_direct_provider_file_inputs_at(
        wire.blocks(),
        profile,
        &opts.request_model,
        opts.file_scope(),
        opts.file_validation_time(),
    )?;
    if req.continuation.is_some()
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
    if req.hosted_file_search().is_some()
        && profile.extra.get("file_search").and_then(Value::as_str) != Some("qwen")
    {
        return Err(LlmError::UnsupportedCapability {
            message: format!(
                "profile {:?} does not declare Qwen file search",
                profile.profile_name
            ),
        });
    }
    if req.remote_mcp_servers().next().is_some()
        && (!is_official_openai_responses_profile(profile)
            || profile.extra.get("remote_mcp").and_then(Value::as_str) != Some("openai_responses"))
    {
        return Err(LlmError::UnsupportedCapability {
            message: format!(
                "remote MCP is not enabled for OpenAI Responses profile {:?}",
                profile.profile_name
            ),
        });
    }
    validate_xai_remote_mcp(req, profile)?;
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
    if let Some(system) = super::cache::system_input(req) {
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
    if !req.system.is_empty() && !super::cache::has_system_breakpoint(req) {
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
    // OpenAI's current web_search wire has its own filter names and supports
    // both allow/block lists together. Other Responses adapters keep using the
    // shared provider-specific adapter.
    apply_web_search(req, profile, &mut body)?;
    super::qwen_web_extractor::apply(req, profile, &opts.request_model, &mut body)?;
    if !super::qwen_hosted::apply(req, profile, &opts.request_model, &mut body)? {
        crate::codecs::code_interpreter::apply(req, opts, &mut body)?;
    }
    if let Some(search) = req.hosted_file_search() {
        if search.knowledge_base_id.trim().is_empty() {
            return Err(LlmError::InvalidRequest {
                message: "Qwen file search requires a nonempty knowledge_base_id".into(),
            });
        }
        if !search
            .workspace_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            || search.workspace_id.is_empty()
        {
            return Err(LlmError::InvalidRequest {
                message:
                    "Qwen file search workspace_id must contain only letters, digits, or hyphens"
                        .into(),
            });
        }
        if profile.provider_id.as_str() != "qwen"
            || !matches!(opts.request_model.as_str(), "qwen3.8-max" | "qwen3.8-flash")
        {
            return Err(LlmError::UnsupportedCapability {
                message: "Qwen file search is available only for the supported Max and Flash Responses models".into(),
            });
        }
        body.entry("tools")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "Qwen file search requires tools to be an array".into(),
            })?
            .push(json!({"type":"file_search","vector_store_ids":[search.knowledge_base_id]}));
    }
    let mcp_servers: Vec<_> = req.remote_mcp_servers().collect();
    if !mcp_servers.is_empty() {
        if !is_official_openai_responses_profile(profile)
            || profile.extra.get("remote_mcp").and_then(Value::as_str) != Some("openai_responses")
        {
            return Err(LlmError::UnsupportedCapability {
                message: format!(
                    "remote MCP is not enabled for OpenAI Responses profile {:?}",
                    profile.profile_name
                ),
            });
        }
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
                        crate::protocol::llm::McpApprovalPolicy::Always => "always",
                        crate::protocol::llm::McpApprovalPolicy::Never => "never",
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
    for tool in crate::codecs::openrouter_server_tools::response_tool_values(req) {
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

    super::cache::apply(req, opts, &mut body)?;
    let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
    crate::codecs::inference::apply(req, opts, &mut body, &mut headers)?;
    crate::codecs::structured::apply(req, opts, &mut body)?;
    crate::wire_options::merge_body(profile, &mut body);
    crate::wire_options::merge_headers(profile, &mut headers);

    let url = if let Some(search) = req.hosted_file_search() {
        qwen_file_search_url(profile, &search.workspace_id)?
    } else {
        format!("{}/responses", profile.base_url.trim_end_matches('/'))
    };
    let mut body = WireValue::from(Value::Object(body)).with("input", WireValue::array(input));
    if !req.tools.is_empty() {
        body = body.map_array_field("tools", |index, tool| {
            let tool = WireValue::from(tool);
            match req.tools.get(index) {
                Some(spec) => tool.with("parameters", WireValue::borrowed(&spec.input_schema)),
                None => tool,
            }
        });
    }
    Ok(WireRequest::new(url, headers, body))
}

fn apply_web_search(
    req: &crate::protocol::ChatRequest,
    profile: &ProviderProfile,
    body: &mut Map<String, Value>,
) -> Result<(), LlmError> {
    let Some(search) = req.hosted_web_search() else {
        return Ok(());
    };
    let adapter = profile
        .extra
        .get("web_search")
        .and_then(Value::as_str)
        .ok_or_else(|| LlmError::UnsupportedCapability {
            message: format!(
                "web search on profile {:?}: no extra.web_search adapter declared",
                profile.profile_name
            ),
        })?;
    if adapter != "openai_responses" || !is_official_openai_responses_profile(profile) {
        return crate::codecs::web_search::apply(req, profile, body);
    }
    if profile.protocol != crate::protocol::ProtocolFamily::OpenAiResponses {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI hosted web search requires the Responses protocol".into(),
        });
    }
    if search.max_uses.is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI Responses web_search does not expose max_uses".into(),
        });
    }
    if search.allowed_domains.len() > 100 || search.blocked_domains.len() > 100 {
        return Err(LlmError::InvalidRequest {
            message: "OpenAI Responses web_search accepts at most 100 domains per filter".into(),
        });
    }
    for domain in search.allowed_domains.iter().chain(&search.blocked_domains) {
        if domain.trim().is_empty()
            || domain.contains("://")
            || domain.contains('/')
            || domain.chars().any(char::is_whitespace)
        {
            return Err(LlmError::InvalidRequest {
                message: "web search domain filters must be host names without schemes or paths"
                    .into(),
            });
        }
    }
    let mut tool = json!({"type": "web_search"});
    if !search.allowed_domains.is_empty() || !search.blocked_domains.is_empty() {
        let mut filters = Map::new();
        if !search.allowed_domains.is_empty() {
            filters.insert("allowed_domains".into(), json!(search.allowed_domains));
        }
        if !search.blocked_domains.is_empty() {
            filters.insert("blocked_domains".into(), json!(search.blocked_domains));
        }
        tool["filters"] = Value::Object(filters);
    }
    body.entry("tools")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "Responses hosted tools must encode as an array".into(),
        })?
        .push(tool);
    // Include the provider's source records as well as message citations so
    // open_page/fetch actions remain available in native output metadata.
    body.insert("include".into(), json!(["web_search_call.action.sources"]));
    Ok(())
}

pub(crate) fn is_official_openai_responses_profile(profile: &ProviderProfile) -> bool {
    let Ok(url) = url::Url::parse(&profile.base_url) else {
        return false;
    };
    profile.provider_id.as_str() == "openai"
        && profile.protocol == crate::protocol::ProtocolFamily::OpenAiResponses
        && url.scheme() == "https"
        && url.host_str() == Some("api.openai.com")
        && url.path().trim_end_matches('/') == "/v1"
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
}

fn supports_openai_tool_search_model(model: &str) -> bool {
    let Some(version) = model
        .to_ascii_lowercase()
        .strip_prefix("gpt-")
        .map(str::to_owned)
    else {
        return false;
    };
    let major_end = version
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(version.len());
    if major_end == 0
        || version.get(major_end..).is_some_and(|tail| {
            !tail.is_empty() && !tail.starts_with('.') && !tail.starts_with('-')
        })
    {
        return false;
    }
    let Ok(major) = version[..major_end].parse::<u32>() else {
        return false;
    };
    if major > 5 {
        return true;
    }
    if major < 5 || version.as_bytes().get(major_end) != Some(&b'.') {
        return false;
    }
    let minor = &version[major_end + 1..];
    let minor_end = minor
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(minor.len());
    if minor_end == 0
        || minor
            .get(minor_end..)
            .is_some_and(|tail| !tail.is_empty() && !tail.starts_with('-'))
    {
        return false;
    }
    minor[..minor_end]
        .parse::<u32>()
        .is_ok_and(|minor| minor >= 4)
}

pub(crate) fn is_official_xai_responses_profile(profile: &ProviderProfile) -> bool {
    let Ok(url) = url::Url::parse(&profile.base_url) else {
        return false;
    };
    profile.provider_id.as_str() == "xai"
        && profile.protocol == crate::protocol::ProtocolFamily::OpenAiResponses
        && profile.extra.get("xai_remote_mcp").and_then(Value::as_str) == Some("xai_responses")
        && url.scheme() == "https"
        && url.host_str() == Some("api.x.ai")
        && url.path().trim_end_matches('/') == "/v1"
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
}

/// Request-level capability gate shared by the codec and preflight path.
pub(crate) fn validate_xai_remote_mcp(
    request: &crate::protocol::ChatRequest,
    profile: &ProviderProfile,
) -> Result<(), LlmError> {
    request.validate_hosted_tools()?;
    let configs: Vec<_> = request.xai_remote_mcp_servers().collect();
    if configs.is_empty() {
        return Ok(());
    }
    if request.remote_mcp_servers().next().is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI and xAI remote MCP configurations cannot be mixed".into(),
        });
    }
    if !is_official_xai_responses_profile(profile) {
        return Err(LlmError::UnsupportedCapability {
            message: format!(
                "xAI remote MCP requires an enabled official xAI Responses profile; {:?} is not enabled",
                profile.profile_name
            ),
        });
    }
    for config in configs {
        config.validate()?;
    }
    Ok(())
}

fn qwen_file_search_url(profile: &ProviderProfile, workspace_id: &str) -> Result<String, LlmError> {
    let host = url::Url::parse(&profile.base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned));
    let region_domain = match (profile.provider_id.as_str(), host.as_deref()) {
        ("qwen", Some("dashscope.aliyuncs.com")) => "cn-beijing.maas.aliyuncs.com",
        ("qwen", Some("dashscope-intl.aliyuncs.com")) => "ap-southeast-1.maas.aliyuncs.com",
        ("qwen", Some("dashscope-us.aliyuncs.com")) => "us-east-1.maas.aliyuncs.com",
        ("qwen", Some("cn-hongkong.dashscope.aliyuncs.com")) => "cn-hongkong.maas.aliyuncs.com",
        _ => {
            return Err(LlmError::UnsupportedCapability {
                message: "Qwen file search requires a supported regional Model Studio profile"
                    .into(),
            });
        }
    };
    Ok(format!(
        "https://{workspace_id}.{region_domain}/compatible-mode/v1/responses"
    ))
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
                validate_provider_item_replay(value, wire.request())?;
                flush(role, &mut parts, input);
                let item = if super::cache::has_message_breakpoint(
                    wire.request(),
                    message_index,
                    block_index,
                ) {
                    super::cache::mark_provider_message(value)
                } else {
                    WireValue::borrowed(value)
                };
                input.push(item);
            }
            ContentBlock::Text { text, .. } => {
                let part =
                    WireValue::from(json!({"type": text_part})).with("text", WireValue::text(text));
                parts.push(super::cache::mark_message_block(
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
                let output = super::cache::tool_result_output(
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
        }
    }
    flush(role, &mut parts, input);
    Ok(())
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
    tool
}

fn encode_openai_tool_search(config: &crate::protocol::OpenAiToolSearchConfig) -> Value {
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
