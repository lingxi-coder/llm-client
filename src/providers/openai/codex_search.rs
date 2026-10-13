//! Codex's client-executed standalone search protocol, separate from Responses generation.
use crate::codecs::{
    json::{WireRequest, WireValue},
    CodecContext,
};
use crate::protocol::{
    ChatRequest, ChatResponse, ConversationMessage, LlmError, MessageRole, NativeType,
    ProviderProfile, StopReason, WebCitation, WebSearchConfig, WebSearchResult,
};
use crate::response_json::{content_block, DecodedText, FlatJsonKind, ResponseJsonScanner};
use crate::transport::HttpResponse;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodexSearchRequest {
    pub id: String,
    pub query: String,
}
impl NativeType for CodexSearchRequest {
    const FORMAT: &'static str = "openai.codex_search.v1";
}
impl CodexSearchRequest {
    pub fn validate(&self) -> Result<(), LlmError> {
        if self.id.trim().is_empty() || self.query.trim().chars().count() < 2 {
            return Err(LlmError::InvalidRequest {
                message: "Codex search requires an id and a query of at least two characters"
                    .into(),
            });
        }
        Ok(())
    }
}

pub fn supports(profile: &ProviderProfile) -> bool {
    super::search::codex_responses(profile)
        && profile
            .extra
            .get("web_search")
            .and_then(serde_json::Value::as_str)
            == Some("openai_responses")
}
pub(crate) fn validate_config(config: &WebSearchConfig) -> Result<(), LlmError> {
    crate::codecs::web_search::validate_domains(config)?;
    if config.max_uses.is_some() {
        return Err(LlmError::UnsupportedCapability {
            message: "Codex standalone search does not support max_uses".into(),
        });
    }
    Ok(())
}

pub(crate) fn encode<'a>(
    request: &'a ChatRequest,
    profile: &ProviderProfile,
    context: &CodecContext,
) -> Result<Option<WireRequest<'a>>, LlmError> {
    let Some(extension) = request
        .native_options
        .iter()
        .find(|value| value.is::<CodexSearchRequest>())
    else {
        return Ok(None);
    };
    let search = extension.decode::<CodexSearchRequest>()?;
    search.validate()?;
    if !supports(profile) || context.stream {
        return Err(LlmError::UnsupportedCapability { message: "Codex standalone search requires a Codex Responses profile and a complete HTTP request".into() });
    }
    let config = request
        .hosted_web_search()
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "Codex standalone search requires WebSearch configuration".into(),
        })?;
    validate_config(config)?;
    if !request.tools.is_empty()
        || request.hosted_tools.len() != 1
        || request.native_options.len() != 1
        || request.tool_choice == crate::protocol::ToolChoice::None
    {
        return Err(LlmError::UnsupportedCapability {
            message: "Codex standalone search cannot combine generation tools".into(),
        });
    }
    let mut body = json!({
        "id": search.id, "model":context.request_model,
        "commands":{"search_query":[{"q":search.query}]},
        "settings":{"external_web_access":true,"allowed_callers":["direct"]}
    });
    if !config.allowed_domains.is_empty() || !config.blocked_domains.is_empty() {
        let mut filters = json!({});
        if !config.allowed_domains.is_empty() {
            filters["allowed_domains"] = json!(config.allowed_domains);
        }
        if !config.blocked_domains.is_empty() {
            filters["blocked_domains"] = json!(config.blocked_domains);
        }
        body["settings"]["filters"] = filters;
    }
    if let Some(tokens) = request.max_tokens {
        body["max_output_tokens"] = json!(tokens);
    }
    let mut headers = vec![("Content-Type".into(), "application/json".into())];
    let _ = crate::wire_options::merge_headers(profile, &mut headers);
    Ok(Some(WireRequest::new(
        format!("{}/alpha/search", profile.base_url.trim_end_matches('/')),
        headers,
        WireValue::from(body),
    )))
}

pub(crate) fn decode(response: &HttpResponse) -> Result<ChatResponse, LlmError> {
    let scanner =
        ResponseJsonScanner::parse(&response.body, "Codex search body is not valid JSON")?;
    let output = scanner
        .string_at("/output")
        .ok_or_else(|| LlmError::ProviderInternal {
            message: "Codex search response has no text output".into(),
        })?;
    if scanner
        .kind_at("/results")
        .is_some_and(|kind| !matches!(kind, FlatJsonKind::Null | FlatJsonKind::Array))
    {
        return Err(LlmError::ProviderInternal {
            message: "Codex search results must be an array".into(),
        });
    }
    let mut seen = std::collections::BTreeSet::new();
    let citations = (0..scanner.array_len("/results").unwrap_or(0))
        .filter_map(|index| {
            let pointer = format!("/results/{index}");
            let url = scanner.string_at(&format!("{pointer}/url"))?;
            let parsed_url = url::Url::parse(url).ok()?;
            if !matches!(parsed_url.scheme(), "http" | "https") || !seen.insert(url.to_owned()) {
                return None;
            }
            Some(WebCitation {
                url: url.into(),
                title: scanner
                    .string_at(&format!("{pointer}/title"))
                    .map(str::to_owned),
            })
        })
        .collect();
    let units = scanner.string_units_at("/output");
    let text = DecodedText {
        text: output.into(),
        utf16_code_units: units.filter(|units| *units != output.encode_utf16().collect::<Vec<_>>()),
    };
    let metadata = crate::protocol::JsonPayload::parse(
        scanner
            .raw_source_at("")
            .expect("validated response source"),
    )
    .map_err(|error| LlmError::InvalidRequest {
        message: format!("invalid Codex search metadata: {error}"),
    })?
    .without_object_field("output")?
    .with_object_field(
        "adapter",
        &crate::protocol::JsonPayload::string("codex_standalone"),
    )?;
    Ok(ChatResponse {
        inference: Default::default(),
        response_cache: None,
        message: ConversationMessage {
            native_options: Vec::new(),
            role: MessageRole::Assistant,
            content: vec![content_block(text, None, None)],
        },
        web_search: Some(WebSearchResult {
            citations,
            metadata,
        }),
        file_search: None,
        native_metadata: Vec::new(),
        stop_reason: StopReason::EndTurn,
        usage: Default::default(),
        model: String::new(),
        response_id: None,
        continuation: None,
        native_resume_context_exhausted: None,
        executed_profile: None,
    })
}
