//! Wire merging for explicit host policy overrides. Environment and feature decisions remain in the host.
use crate::protocol::{JsonPayload, LlmError, ProtocolFamily, ToolChoice};
use crate::response_json::{FlatJsonKind, JsonNodeRef, ResponseJsonScanner};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

pub const MIN_MANUAL_THINKING_TOKENS: u32 = 1024;

/// Whether `request_url` is the Messages endpoint rooted at `configured_base_url`.
///
/// The configured base must not contain a query or fragment. The request URL
/// must share its scheme, authority, and credentials and use the exact
/// `{base path}/v1/messages` path. Request query strings and fragments are
/// accepted because they do not change that endpoint identity.
#[must_use]
pub fn messages_request_matches_base_url(configured_base_url: &str, request_url: &str) -> bool {
    let (Ok(base), Ok(request)) = (
        url::Url::parse(configured_base_url),
        url::Url::parse(request_url),
    ) else {
        return false;
    };
    if base.query().is_some()
        || base.fragment().is_some()
        || request.scheme() != base.scheme()
        || request.host_str() != base.host_str()
        || request.port_or_known_default() != base.port_or_known_default()
        || request.username() != base.username()
        || request.password() != base.password()
    {
        return false;
    }
    let base_path = base.path().trim_end_matches('/');
    let expected_path = format!("{base_path}/v1/messages");
    request.path() == expected_path
}

/// Explicit caller identity; preserving an authenticator's identity is the
/// default for providers that require their own User-Agent.
#[derive(Debug, Clone, Copy)]
pub enum UserAgentPolicy<'a> {
    Replace(&'a str),
    IfAbsent(&'a str),
}

/// Set one HTTP header, replacing any differently-cased spelling.
pub fn set_header(headers: &mut BTreeMap<String, String>, name: &str, value: &str) {
    headers.retain(|key, _| !key.eq_ignore_ascii_case(name));
    headers.insert(name.to_ascii_lowercase(), value.to_owned());
}

pub fn apply_user_agent(headers: &mut BTreeMap<String, String>, policy: UserAgentPolicy<'_>) {
    let value = match policy {
        UserAgentPolicy::Replace(value) => value,
        UserAgentPolicy::IfAbsent(value) => {
            if headers
                .keys()
                .any(|key| key.eq_ignore_ascii_case("user-agent"))
            {
                return;
            }
            value
        }
    };
    set_header(headers, "user-agent", value);
}

/// Explicit Native-compatible client identity for an admitted Anthropic request.
/// The host captures runtime/platform/session facts; this SDK policy never reads
/// process environment or chooses whether the identity applies to a route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicClientMetadata {
    /// Runtime language reported by the selected client identity.
    pub language: String,
    /// Runtime name reported by the selected client identity.
    pub runtime: String,
    pub package_version: String,
    pub runtime_version: String,
    pub os: String,
    pub arch: String,
    /// Native SDK timeout in seconds, independent of the host turn deadline.
    pub timeout_seconds: u64,
    /// Host-captured origin session. The host may replace this with the
    /// admitted request's session identity immediately before header policy.
    pub session_id: Option<String>,
    pub app_id: String,
}

impl AnthropicClientMetadata {
    /// Metadata defaults captured from the controlled Claude Code 2.1.293
    /// client. Platform labels follow that embedded SDK's mapping.
    pub fn claude_code_2_1_293(os: &str, arch: &str) -> Self {
        Self {
            language: "js".into(),
            runtime: "node".into(),
            package_version: "0.128.0".into(),
            runtime_version: "v26.3.0".into(),
            os: match os.to_ascii_lowercase().as_str() {
                "darwin" | "macos" => "MacOS".into(),
                "win32" | "windows" => "Windows".into(),
                "linux" => "Linux".into(),
                "freebsd" => "FreeBSD".into(),
                "openbsd" => "OpenBSD".into(),
                "android" => "Android".into(),
                value if value.contains("ios") => "iOS".into(),
                "" => "Unknown".into(),
                value => format!("Other:{value}"),
            },
            arch: match arch {
                "aarch64" | "arm64" => "arm64".into(),
                "x86_64" | "x64" => "x64".into(),
                "x32" | "arm" => arch.into(),
                "" => "unknown".into(),
                value => format!("other:{value}"),
            },
            timeout_seconds: 600,
            session_id: None,
            app_id: "cli".into(),
        }
    }
}

/// Add the Native Anthropic client headers without replacing values already
/// supplied by authentication or the caller. Header spelling is canonicalized
/// later only when the transport has selected NativeFetch layout.
pub fn apply_client_metadata(
    headers: &mut BTreeMap<String, String>,
    metadata: &AnthropicClientMetadata,
) {
    let default = |headers: &mut BTreeMap<String, String>, name: &str, value: &str| {
        if !headers.keys().any(|key| key.eq_ignore_ascii_case(name)) {
            set_header(headers, name, value);
        }
    };
    for (name, value) in [
        ("accept", "application/json"),
        ("anthropic-dangerous-direct-browser-access", "true"),
        ("x-app", metadata.app_id.as_str()),
        ("x-stainless-lang", metadata.language.as_str()),
        (
            "x-stainless-package-version",
            metadata.package_version.as_str(),
        ),
        ("x-stainless-os", metadata.os.as_str()),
        ("x-stainless-arch", metadata.arch.as_str()),
        ("x-stainless-runtime", metadata.runtime.as_str()),
        (
            "x-stainless-runtime-version",
            metadata.runtime_version.as_str(),
        ),
        ("x-stainless-retry-count", "0"),
    ] {
        default(headers, name, value);
    }
    default(
        headers,
        "x-stainless-timeout",
        &metadata.timeout_seconds.to_string(),
    );
    if let Some(session_id) = &metadata.session_id {
        default(headers, "x-claude-code-session-id", session_id);
    }
}

/// Canonical spellings emitted by the versioned Native Anthropic SDK client.
/// This is applied by the authenticated request path only after the caller has
/// selected NativeFetch layout. Unknown caller-owned names keep their spelling.
pub fn native_fetch_header_name(name: &str) -> &str {
    match name.to_ascii_lowercase().as_str() {
        "accept" => "Accept",
        "content-type" => "Content-Type",
        "user-agent" => "User-Agent",
        "authorization" => "Authorization",
        "x-claude-code-session-id" => "X-Claude-Code-Session-Id",
        "x-stainless-arch" => "X-Stainless-Arch",
        "x-stainless-lang" => "X-Stainless-Lang",
        "x-stainless-os" => "X-Stainless-OS",
        "x-stainless-package-version" => "X-Stainless-Package-Version",
        "x-stainless-retry-count" => "X-Stainless-Retry-Count",
        "x-stainless-runtime" => "X-Stainless-Runtime",
        "x-stainless-runtime-version" => "X-Stainless-Runtime-Version",
        "x-stainless-timeout" => "X-Stainless-Timeout",
        _ => name,
    }
}

/// Preserve existing (including auth-injected) betas first, then append the
/// caller-selected values, without duplicate beta tokens or header spellings.
pub fn merge_beta_header(headers: &mut BTreeMap<String, String>, betas: &[String]) {
    let mut parts = Vec::<String>::new();
    for value in headers
        .iter()
        .filter(|(key, _)| key.eq_ignore_ascii_case("anthropic-beta"))
        .map(|(_, value)| value.as_str())
        .chain(betas.iter().map(String::as_str))
    {
        for beta in value
            .split(',')
            .map(str::trim)
            .filter(|beta| !beta.is_empty())
        {
            if !parts.iter().any(|part| part == beta) {
                parts.push(beta.to_owned());
            }
        }
    }
    set_header(headers, "anthropic-beta", &parts.join(","));
}

/// Wire policy chosen by the host. Apply before request sealing/signing.
/// The SDK neither reads flags/environment nor makes routing/retry decisions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AnthropicRequestKind {
    #[default]
    Main,
    SideQuery,
    /// Native `hook_prompt` classifier/completion request identity.
    /// All native Claude generation requests normalize string values only
    /// after body assembly, at the final SDK dispatch boundary.
    HookPrompt,
}

/// Native YMe admission chosen by the host for the selected model. Wire policy
/// does not resolve catalogs, flags, session effort or provider capability.
#[derive(Debug, Clone)]
pub struct AnthropicEffortPolicy {
    pub supported: bool,
    /// Host-resolved YMe input. None means undefined; nonstrings have no wire
    /// effort. Main explicit config is inspected before this automatic value.
    pub value: Option<JsonPayload>,
}

/// Native `nxt` inputs captured by the host before Anthropic request preparation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnthropicContextManagement {
    pub has_thinking: bool,
    pub tool_clearing: Option<JsonPayload>,
}

impl AnthropicContextManagement {
    /// Construct the source-backed Native 2.1.293 context-management edit list.
    pub fn to_wire(&self) -> Result<Option<JsonPayload>, LlmError> {
        let mut edits = Vec::new();
        if self.has_thinking {
            edits.push(
                JsonPayload::parse(r#"{"type":"clear_thinking_20251015","keep":"all"}"#)
                    .expect("static context-management edit is valid JSON"),
            );
        }
        if let Some(edit) = &self.tool_clearing {
            edits.push(edit.clone());
        }
        if edits.is_empty() {
            return Ok(None);
        }
        let edits =
            JsonPayload::array_from_items(&edits, crate::exact_json::JsonEncoding::JavaScript)?;
        let mut wire = JsonPayload::parse("{}").expect("empty object is valid JSON");
        set_object_field_source(&mut wire, "edits", &edits)?;
        Ok(Some(wire))
    }
}

#[derive(Debug, Clone)]
pub struct AnthropicRequestPolicy {
    /// The selected wire route. Hosted routes do not necessarily share the
    /// first-party Messages endpoint's query policy.
    pub protocol: ProtocolFamily,
    pub request_kind: AnthropicRequestKind,
    /// Preserve an explicitly empty parent tool prefix for cache-safe forks.
    pub explicit_empty_tools: bool,
    /// Preserve utility callers that supplied plain string message content.
    pub plain_text_message_content: bool,
    /// Source-backed object spread supplied by the host. Unknown provider
    /// values stay in their original JSON representation until serialization.
    pub extra_body: JsonPayload,
    pub body_betas: Vec<String>,
    pub effort: Option<AnthropicEffortPolicy>,
    pub context_management: Option<AnthropicContextManagement>,
    /// Messages SDK virtual parameters travel in headers, including hosted variants.
    pub message_header_parameters: bool,
    /// A beta token whose fast-mode fields must be removed for this route.
    pub disallowed_fast_beta: Option<String>,
}

impl Default for AnthropicRequestPolicy {
    fn default() -> Self {
        Self {
            protocol: ProtocolFamily::AnthropicMessages,
            request_kind: AnthropicRequestKind::Main,
            explicit_empty_tools: false,
            plain_text_message_content: false,
            extra_body: JsonPayload::parse("{}").expect("empty object is valid JSON"),
            body_betas: Vec::new(),
            effort: None,
            context_management: None,
            message_header_parameters: false,
            disallowed_fast_beta: None,
        }
    }
}

/// Native nonstream fallback owns the final stream flag after the extra-body
/// spread. Preserve an existing property's position and clear its exact-string
/// sidecars before serialization. Call before sealing or authentication.
pub fn nonstream_fallback_parameters_source(
    body: &mut JsonPayload,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
) -> Result<(), LlmError> {
    let mut fields = source_object_fields(body)?;
    source_set_field(&mut fields, "stream", "false".into());
    *body = source_object_payload(fields)?;
    string_overrides.retain(|path, _| path != "/stream" && !path.starts_with("/stream/"));
    Ok(())
}

/// Native extra-body cleanup, before any automatic feature decisions. Policy
/// admission and environment access belong to the host; wire cleanup belongs here.
pub fn sanitize_extra_body(extra: JsonPayload) -> Result<JsonPayload, LlmError> {
    let mut fields = source_object_fields(&extra)?;
    for key in ["betas", "anthropic_beta"] {
        let Some(value_source) = source_field_source(&fields, key) else {
            continue;
        };
        let value = JsonPayload::parse(value_source).map_err(|error| LlmError::InvalidRequest {
            message: format!("extra body field {key} is invalid JSON: {error}"),
        })?;
        let text = source_javascript_string_units(value.scanner()?.root_node())?;
        let tokens = split_trim_js_units(&text);
        let token_count = tokens.len();
        let retained = tokens
            .into_iter()
            .filter(|token| !contains_afk_mode(token))
            .collect::<Vec<_>>();
        if retained.len() == token_count {
            continue;
        }
        if retained.is_empty() {
            source_remove_field(&mut fields, key);
        } else {
            let items = retained
                .iter()
                .map(|token| JsonPayload::string_from_utf16_code_units(token))
                .collect::<Vec<_>>();
            let array =
                JsonPayload::array_from_items(&items, crate::exact_json::JsonEncoding::JavaScript)?;
            source_set_field(&mut fields, key, array.as_json_source().to_owned());
        }
    }

    if let Some(metadata_source) = source_field_source(&fields, "metadata") {
        let metadata =
            JsonPayload::parse(metadata_source).map_err(|error| LlmError::InvalidRequest {
                message: format!("extra body metadata is invalid JSON: {error}"),
            })?;
        if metadata.kind_at("") == Some(FlatJsonKind::Object) {
            let mut metadata_fields = source_object_fields(&metadata)?;
            if let Some(user_id_source) = source_field_source(&metadata_fields, "user_id") {
                let user_id = JsonPayload::parse(user_id_source).map_err(|error| {
                    LlmError::InvalidRequest {
                        message: format!("extra body metadata.user_id is invalid JSON: {error}"),
                    }
                })?;
                if let Some(units) = user_id.root_string_code_units() {
                    if let Ok(text) = String::from_utf16(&units) {
                        if let Some(cleaned) = super::metadata_json::remove_identity_tk(&text) {
                            let cleaned = JsonPayload::string(&cleaned);
                            source_set_field(
                                &mut metadata_fields,
                                "user_id",
                                cleaned.as_json_source().to_owned(),
                            );
                            let metadata = source_object_payload(metadata_fields)?;
                            source_set_field(
                                &mut fields,
                                "metadata",
                                metadata.as_json_source().to_owned(),
                            );
                        }
                    }
                }
            }
        }
    }
    source_object_payload(fields)
}

fn javascript_string(value: &Value) -> Result<String, LlmError> {
    Ok(match value {
        Value::Null => "null".into(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => javascript_number(value),
        Value::String(value) => value.clone(),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                if item.is_null() {
                    Ok(String::new())
                } else {
                    javascript_string(item)
                }
            })
            .collect::<Result<Vec<_>, _>>()?
            .join(","),
        Value::Object(object) => {
            // JSON cannot carry callable methods. An own toString masks the
            // inherited method and default valueOf still returns the object.
            if object.contains_key("toString") {
                return Err(LlmError::InvalidRequest {
                    message: "Cannot convert object to primitive value".into(),
                });
            }
            "[object Object]".into()
        }
    })
}

fn javascript_number(number: &serde_json::Number) -> String {
    crate::exact_json::javascript_number(number.as_f64().expect("JSON number"))
}

impl AnthropicRequestPolicy {
    /// Update body and exact-string sidecars together before sealing/signing.
    pub fn apply(
        mut self,
        body: &mut JsonPayload,
        headers: &mut BTreeMap<String, String>,
        string_overrides: &mut BTreeMap<String, Vec<u16>>,
        url: &mut String,
    ) -> Result<(), LlmError> {
        self.extra_body = sanitize_extra_body(self.extra_body)?;
        let explicit_extra = source_object_fields(&self.extra_body)?;
        let body_extra = beta_body(self.extra_body.clone(), &self.body_betas)?;
        let _has_body_extra = !source_object_fields(&body_extra)?.is_empty();
        if self.explicit_empty_tools && body.raw_source_at("/tools").is_none() {
            set_object_field_source(
                body,
                "tools",
                &JsonPayload::parse("[]").expect("empty array"),
            )?;
        }
        if self.plain_text_message_content {
            if let Some(mut messages) = body
                .payload_at("/messages")
                .and_then(|value| value.array_items())
            {
                for (index, message) in messages.iter_mut().enumerate() {
                    let text = message
                        .payload_at("/content")
                        .and_then(|value| value.array_items())
                        .filter(|blocks| blocks.len() == 1)
                        .and_then(|blocks| blocks.into_iter().next())
                        .filter(|block| {
                            source_object_fields(block).is_ok_and(|fields| fields.len() == 2)
                                && block.string_at("/type").as_deref() == Some("text")
                        })
                        .and_then(|block| block.payload_at("/text"))
                        .filter(|text| text.kind_at("") == Some(FlatJsonKind::String));
                    if let Some(text) = text {
                        set_object_field_source(message, "content", &text)?;
                        if let Some(units) =
                            string_overrides.remove(&format!("/messages/{index}/content/0/text"))
                        {
                            string_overrides.insert(format!("/messages/{index}/content"), units);
                        }
                    }
                }
                let messages = JsonPayload::array_from_items(
                    &messages,
                    crate::exact_json::JsonEncoding::JavaScript,
                )?;
                set_object_field_source(body, "messages", &messages)?;
            }
        }
        let context_beta = "context-management-2025-06-27";
        let context_admitted = headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
            .any(|(_, value)| value.split(',').any(|beta| beta.trim() == context_beta))
            || self.body_betas.iter().any(|beta| beta == context_beta);
        let context_management = self
            .context_management
            .as_ref()
            .filter(|_| context_admitted)
            .map(AnthropicContextManagement::to_wire)
            .transpose()?
            .flatten();
        if let Some(context_management) = &context_management {
            set_object_field_source(body, "context_management", context_management)?;
            string_overrides.retain(|path, _| {
                path != "/context_management" && !path.starts_with("/context_management/")
            });
        }
        let format_alias_requested = output_format_alias_requested(body, string_overrides)?;

        // An explicit top-level spread owns its string leaves. Main handles
        // output_config separately, retaining untouched automatic fields.
        string_overrides.retain(|pointer, _| {
            let Some(path) = pointer.strip_prefix('/') else {
                return true;
            };
            let mut segments = path.split('/');
            let key = segments
                .next()
                .unwrap_or("")
                .replace("~1", "/")
                .replace("~0", "~");
            if key == "output_config" && self.request_kind == AnthropicRequestKind::Main {
                let field = segments
                    .next()
                    .unwrap_or("")
                    .replace("~1", "/")
                    .replace("~0", "~");
                return !source_config_contains(source_field_source(&explicit_extra, &key), &field);
            }
            !explicit_extra
                .iter()
                .any(|(units, _)| exact_key(units, &key))
        });
        let previous_effort = body.raw_source_at("/output_config/effort");
        let overrides =
            merge_extra_source(body, body_extra, self.request_kind, self.effort.as_ref())?;
        string_overrides.extend(overrides);
        if self.effort.is_some()
            && (body.raw_source_at("/output_config/effort").is_none()
                || body.raw_source_at("/output_config/effort") != previous_effort)
        {
            string_overrides.retain(|path, _| {
                path != "/output_config/effort" && !path.starts_with("/output_config/effort/")
            });
        }
        if self.message_header_parameters {
            normalize_source_message_parameters(
                body,
                headers,
                string_overrides,
                url,
                crate::RequestMode::Complete,
                None,
                self.protocol,
            )?;
        } else if format_alias_requested {
            let source = crate::exact_json::serialize_json_payload_with_overrides(
                body,
                string_overrides,
                crate::exact_json::JsonEncoding::JavaScript,
            )?;
            let source = String::from_utf8(source).map_err(|error| LlmError::InvalidRequest {
                message: format!("canonical JSON body is not UTF-8: {error}"),
            })?;
            let source = JsonPayload::parse(&source).map_err(|error| LlmError::InvalidRequest {
                message: format!("canonical JSON body is invalid: {error}"),
            })?;
            *body = normalize_output_format_source(source)?;
            string_overrides.clear();
        }
        if let Some(beta) = self.disallowed_fast_beta {
            remove_fast_source(body, headers, string_overrides, &beta)?;
        }
        Ok(())
    }
}

/// Normalize current Anthropic Messages SDK parameters directly on authoritative JSON source.
/// Provider-owned subtrees remain flat/source-backed while aliases and header parameters are
/// moved; this is the current entry point for callers that already hold a `JsonPayload`.
pub fn normalize_message_parameters(
    body: &mut JsonPayload,
    headers: &mut BTreeMap<String, String>,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
    url: &mut String,
    mode: crate::RequestMode,
) -> Result<(), LlmError> {
    normalize_source_message_parameters(
        body,
        headers,
        string_overrides,
        url,
        mode,
        None,
        ProtocolFamily::AnthropicMessages,
    )
}

/// CountTokens variant used when the host assembles the endpoint beta list outside the body.
/// The SDK replaces any body `betas` field with that list before applying its current header
/// conversion, without round-tripping the source body through a mutable `Value`.
pub fn normalize_count_tokens_message_parameters(
    body: &mut JsonPayload,
    headers: &mut BTreeMap<String, String>,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
    url: &mut String,
    betas: &[String],
) -> Result<(), LlmError> {
    let mut overrides = string_overrides.clone();
    overrides.retain(|pointer, _| pointer != "/betas" && !pointer.starts_with("/betas/"));
    normalize_source_message_parameters(
        body,
        headers,
        &mut overrides,
        url,
        crate::RequestMode::CountTokens,
        Some(betas),
        ProtocolFamily::AnthropicMessages,
    )?;
    *string_overrides = overrides;
    Ok(())
}

fn normalize_source_message_parameters(
    body: &mut JsonPayload,
    headers: &mut BTreeMap<String, String>,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
    url: &mut String,
    mode: crate::RequestMode,
    count_token_betas: Option<&[String]>,
    protocol: ProtocolFamily,
) -> Result<(), LlmError> {
    let mut normalized_headers = headers.clone();
    let mut normalized_url = url.clone();
    let apply_output_format_alias = output_format_alias_requested(body, string_overrides)?;
    let message_parameters =
        source_message_header_parameters(body, string_overrides, mode, count_token_betas)?;
    let has_virtual_fields = ["betas", "user_profile_id", "workspace_id"]
        .iter()
        .any(|key| has_field(body, key));
    if apply_output_format_alias || has_virtual_fields {
        let input = crate::exact_json::serialize_json_payload_with_overrides(
            body,
            string_overrides,
            crate::exact_json::JsonEncoding::JavaScript,
        )?;
        let input = String::from_utf8(input).map_err(|error| LlmError::InvalidRequest {
            message: format!("canonical JSON body is not UTF-8: {error}"),
        })?;
        let mut normalized_body =
            JsonPayload::parse(&input).map_err(|error| LlmError::InvalidRequest {
                message: format!("canonical JSON body is invalid: {error}"),
            })?;

        if apply_output_format_alias {
            normalized_body = normalize_output_format_source(normalized_body)?;
        }
        normalized_body = remove_message_header_fields_source(normalized_body)?;
        *body = normalized_body;
        // Exact strings have been moved into the body source, so no stale or duplicate sidecar is
        // required after the source transformation.
        string_overrides.clear();
    }
    apply_message_header_parameters(&mut normalized_headers, message_parameters);
    if protocol == ProtocolFamily::BedrockClaude {
        remove_beta_query(&mut normalized_url)?;
    } else {
        append_beta_query(&mut normalized_url)?;
    }

    *headers = normalized_headers;
    *url = normalized_url;
    Ok(())
}

/// Omit the wire's redundant `{"type":"auto"}` only for the current canonical Auto choice.
/// The caller supplies the semantic choice; the SDK owns the exact provider JSON predicate.
pub fn omit_default_auto_tool_choice(
    body: &mut JsonPayload,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
    choice: &ToolChoice,
) -> Result<bool, LlmError> {
    if !matches!(choice, ToolChoice::Auto) {
        return Ok(false);
    }
    let effective = crate::exact_json::serialize_json_payload_with_overrides(
        body,
        string_overrides,
        crate::exact_json::JsonEncoding::JavaScript,
    )?;
    let effective = String::from_utf8(effective).map_err(|error| LlmError::InvalidRequest {
        message: format!("canonical JSON body is not UTF-8: {error}"),
    })?;
    let effective = JsonPayload::parse(&effective).map_err(|error| LlmError::InvalidRequest {
        message: format!("canonical JSON body is invalid: {error}"),
    })?;
    let scanner = effective.scanner()?;
    let root = scanner.root_node();
    if root.kind() != FlatJsonKind::Object {
        return Ok(false);
    }
    let Some(entry) = root.object_entries().and_then(|entries| {
        entries
            .into_iter()
            .find(|entry| exact_key(entry.key_units, "tool_choice"))
    }) else {
        return Ok(false);
    };
    let matches_auto = entry.value.kind() == FlatJsonKind::Object
        && entry.value.object_entries().is_some_and(|fields| {
            fields.len() == 1
                && exact_key(fields[0].key_units, "type")
                && fields[0].value.kind() == FlatJsonKind::String
                && fields[0]
                    .value
                    .string_units()
                    .is_some_and(|units| exact_key(&units, "auto"))
        });
    if !matches_auto {
        return Ok(false);
    }
    let mut fields = source_object_fields(body)?;
    source_remove_field(&mut fields, "tool_choice");
    *body = source_object_payload(fields)?;
    string_overrides
        .retain(|pointer, _| pointer != "/tool_choice" && !pointer.starts_with("/tool_choice/"));
    Ok(true)
}

/// Bounded SDK-created bodies still use this private Value implementation while the provider
/// receives their generated, non-provider-owned fields. Source-bearing host requests use the
/// public JsonPayload entry point above.
pub(crate) fn normalize_generated_message_parameters(
    body: &mut Value,
    headers: &mut BTreeMap<String, String>,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
    url: &mut String,
    mode: crate::RequestMode,
) -> Result<(), LlmError> {
    normalize_output_format(body, string_overrides)?;
    materialize_message_headers(body, headers, string_overrides, mode)?;
    append_beta_query(url)
}

fn append_beta_query(url: &mut String) -> Result<(), LlmError> {
    let mut parsed = url::Url::parse(url).map_err(|error| LlmError::InvalidRequest {
        message: format!("Invalid Messages URL: {error}"),
    })?;
    if !parsed
        .query_pairs()
        .any(|(key, value)| key == "beta" && value == "true")
    {
        let pairs: Vec<_> = parsed
            .query_pairs()
            .filter(|(key, _)| key != "beta")
            .map(|(key, value)| (key.into_owned(), value.into_owned()))
            .collect();
        parsed.set_query(None);
        parsed
            .query_pairs_mut()
            .extend_pairs(pairs)
            .append_pair("beta", "true");
        *url = parsed.into();
    }
    Ok(())
}

/// Native's Bedrock middleware deletes every `beta` search parameter after it
/// rewrites the request URL. Mutating URLSearchParams also serializes the
/// remaining pairs, preserving their order and duplicates while applying the
/// URL form-encoding rules used by the JavaScript implementation.
pub(crate) fn remove_beta_query(url: &mut String) -> Result<(), LlmError> {
    let mut parsed = url::Url::parse(url).map_err(|error| LlmError::InvalidRequest {
        message: format!("Invalid Bedrock URL: {error}"),
    })?;
    let pairs = parsed
        .query_pairs()
        .filter(|(key, _)| key != "beta")
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    parsed.set_query(None);
    if !pairs.is_empty() {
        parsed.query_pairs_mut().extend_pairs(pairs);
    }
    *url = parsed.into();
    Ok(())
}

type SourceField = (Vec<u16>, String);

fn exact_key(units: &[u16], key: &str) -> bool {
    units.iter().copied().eq(key.encode_utf16())
}

fn source_object_fields(payload: &JsonPayload) -> Result<Vec<SourceField>, LlmError> {
    let scanner = payload.scanner()?;
    let root = scanner.root_node();
    let entries = root
        .object_entries()
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "Messages request body must be an object".into(),
        })?;
    entries
        .into_iter()
        .map(|entry| {
            let source = crate::exact_json::serialize_json_node(
                entry.value,
                crate::exact_json::JsonEncoding::JavaScript,
            )?;
            Ok((entry.key_units.to_vec(), source))
        })
        .collect()
}

/// Test whether every component in a JSON object path exists. Intermediate
/// values must be objects, while a present `null` final member still counts.
pub fn has_field_path(payload: &JsonPayload, path: &[&str]) -> bool {
    if path.is_empty() {
        return true;
    }
    let Ok(scanner) = payload.scanner() else {
        return false;
    };
    let mut node = scanner.root_node();
    for (index, key) in path.iter().enumerate() {
        if node.kind() != FlatJsonKind::Object {
            return false;
        }
        let Some(entry) = node.object_entries().and_then(|entries| {
            entries
                .into_iter()
                .find(|entry| exact_key(entry.key_units, key))
        }) else {
            return false;
        };
        if index + 1 == path.len() {
            return true;
        }
        node = entry.value;
    }
    false
}

pub fn has_field(payload: &JsonPayload, key: &str) -> bool {
    has_field_path(payload, &[key])
}

fn source_field_source<'a>(fields: &'a [SourceField], key: &str) -> Option<&'a str> {
    fields
        .iter()
        .find(|(units, _)| exact_key(units, key))
        .map(|(_, source)| source.as_str())
}

fn source_config_contains(config_source: Option<&str>, field: &str) -> bool {
    let Some(config_source) = config_source else {
        return false;
    };
    let Ok(config) = JsonPayload::parse(config_source) else {
        return false;
    };
    let field_units = field.encode_utf16().collect::<Vec<_>>();
    match config.kind_at("") {
        Some(FlatJsonKind::Object) => has_field(&config, field),
        Some(FlatJsonKind::Array) => source_array_index(&field_units).is_some_and(|index| {
            config
                .array_len("")
                .is_some_and(|length| (index as usize) < length)
        }),
        Some(FlatJsonKind::String) => source_array_index(&field_units).is_some_and(|index| {
            config
                .root_string_code_units()
                .is_some_and(|units| (index as usize) < units.len())
        }),
        _ => false,
    }
}

fn source_javascript_string_units(node: JsonNodeRef<'_>) -> Result<Vec<u16>, LlmError> {
    use FlatJsonKind as Kind;
    Ok(match node.kind() {
        Kind::Null => "null".encode_utf16().collect(),
        Kind::Boolean => node
            .primitive_source()
            .unwrap_or("false")
            .encode_utf16()
            .collect(),
        Kind::Number => {
            let number = node
                .number_source()
                .and_then(|source| source.parse::<f64>().ok())
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "extra body contains an invalid number".into(),
                })?;
            let text = if number.is_finite() {
                crate::exact_json::javascript_number(number)
            } else if number.is_sign_negative() {
                "-Infinity".into()
            } else {
                "Infinity".into()
            };
            text.encode_utf16().collect()
        }
        Kind::String => node.string_units().unwrap_or_default(),
        Kind::Array => {
            let length = node.array_len().unwrap_or_default();
            let mut units = Vec::new();
            for index in 0..length {
                if index > 0 {
                    units.push(u16::from(b','));
                }
                let Some(item) = node.array_item(index) else {
                    continue;
                };
                if item.kind() != Kind::Null {
                    units.extend(source_javascript_string_units(item)?);
                }
            }
            units
        }
        Kind::Object => {
            if node
                .object_entries()
                .unwrap_or_default()
                .iter()
                .any(|entry| exact_key(entry.key_units, "toString"))
            {
                return Err(LlmError::InvalidRequest {
                    message: "Cannot convert object to primitive value".into(),
                });
            }
            "[object Object]".encode_utf16().collect()
        }
    })
}

fn split_trim_js_units(units: &[u16]) -> Vec<Vec<u16>> {
    let mut tokens = Vec::new();
    let mut start = 0;
    for index in 0..=units.len() {
        if index != units.len() && units[index] != u16::from(b',') {
            continue;
        }
        let mut first = start;
        let mut last = index;
        while first < last && is_javascript_whitespace(units[first]) {
            first += 1;
        }
        while last > first && is_javascript_whitespace(units[last - 1]) {
            last -= 1;
        }
        if first < last {
            tokens.push(units[first..last].to_vec());
        }
        start = index + 1;
    }
    tokens
}

fn is_javascript_whitespace(unit: u16) -> bool {
    matches!(
        unit,
        0x0009..=0x000D
            | 0x0020
            | 0x00A0
            | 0x1680
            | 0x2000..=0x200A
            | 0x2028..=0x2029
            | 0x202F
            | 0x205F
            | 0x3000
            | 0xFEFF
    )
}

fn contains_afk_mode(units: &[u16]) -> bool {
    String::from_utf16(units).is_ok_and(|text| text.to_lowercase().contains("afk-mode"))
}

fn source_object_payload(mut fields: Vec<SourceField>) -> Result<JsonPayload, LlmError> {
    fields
        .sort_by_key(|(key, _)| source_array_index(key).map_or((1_u8, 0_u32), |index| (0, index)));
    let mut source = Vec::new();
    source.push(b'{');
    for (index, (key, value)) in fields.into_iter().enumerate() {
        if index > 0 {
            source.push(b',');
        }
        crate::exact_json::write_json_string_from_utf16(&mut source, &key);
        source.push(b':');
        source.extend_from_slice(value.as_bytes());
    }
    source.push(b'}');
    let source = String::from_utf8(source).map_err(|error| LlmError::InvalidRequest {
        message: format!("canonical Messages body is not UTF-8: {error}"),
    })?;
    JsonPayload::parse(&source).map_err(|error| LlmError::InvalidRequest {
        message: format!("canonical Messages body is invalid: {error}"),
    })
}

fn source_array_index(key: &[u16]) -> Option<u32> {
    let text = String::from_utf16(key).ok()?;
    let index = text.parse::<u32>().ok()?;
    (index != u32::MAX && index.to_string() == text).then_some(index)
}

fn source_set_field(fields: &mut Vec<SourceField>, key: &str, source: String) {
    source_set_field_units(fields, &key.encode_utf16().collect::<Vec<_>>(), source);
}

fn source_set_field_units(fields: &mut Vec<SourceField>, key: &[u16], source: String) {
    if let Some((_, existing)) = fields.iter_mut().find(|(units, _)| units == key) {
        *existing = source;
    } else {
        fields.push((key.to_vec(), source));
    }
}

fn source_take_field(fields: &mut Vec<SourceField>, key: &str) -> Option<String> {
    let index = fields.iter().position(|(units, _)| exact_key(units, key))?;
    Some(fields.remove(index).1)
}

fn source_remove_field(fields: &mut Vec<SourceField>, key: &str) {
    fields.retain(|(units, _)| !exact_key(units, key));
}

pub(crate) fn set_object_field_source(
    payload: &mut JsonPayload,
    key: &str,
    value: &JsonPayload,
) -> Result<(), LlmError> {
    let mut fields = source_object_fields(payload)?;
    source_set_field(&mut fields, key, value.as_json_source().to_owned());
    *payload = source_object_payload(fields)?;
    Ok(())
}

fn spread_source_config(source: Option<String>) -> Result<Vec<SourceField>, LlmError> {
    let Some(source) = source else {
        return Ok(Vec::new());
    };
    let config = JsonPayload::parse(&source).map_err(|error| LlmError::InvalidRequest {
        message: format!("output_config is invalid JSON: {error}"),
    })?;
    match config.kind_at("") {
        Some(FlatJsonKind::Object) => source_object_fields(&config),
        Some(FlatJsonKind::Array) => Ok(config
            .array_items()
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                (
                    index.to_string().encode_utf16().collect(),
                    value.as_json_source().to_owned(),
                )
            })
            .collect()),
        Some(FlatJsonKind::String) => Ok(config
            .root_string_code_units()
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(index, unit)| {
                (
                    index.to_string().encode_utf16().collect(),
                    JsonPayload::string_from_utf16_code_units(&[unit])
                        .as_json_source()
                        .to_owned(),
                )
            })
            .collect()),
        _ => Ok(Vec::new()),
    }
}

pub fn beta_body(mut extra: JsonPayload, betas: &[String]) -> Result<JsonPayload, LlmError> {
    let mut fields = source_object_fields(&extra)?;
    if !betas.is_empty() {
        let existing_source = source_take_field(&mut fields, "anthropic_beta");
        let merged = if let Some(existing_source) = existing_source.as_deref() {
            let existing =
                JsonPayload::parse(existing_source).map_err(|error| LlmError::InvalidRequest {
                    message: format!("extra body anthropic_beta is invalid JSON: {error}"),
                })?;
            if existing.kind_at("") == Some(FlatJsonKind::Array) {
                let mut items = existing.array_items().unwrap_or_default();
                for beta in betas {
                    let wanted = beta.encode_utf16().collect::<Vec<_>>();
                    if !items
                        .iter()
                        .any(|item| item.root_string_code_units() == Some(wanted.clone()))
                    {
                        items.push(JsonPayload::string(beta));
                    }
                }
                JsonPayload::array_from_items(&items, crate::exact_json::JsonEncoding::JavaScript)?
            } else {
                JsonPayload::array_from_items(
                    &betas
                        .iter()
                        .map(|beta| JsonPayload::string(beta))
                        .collect::<Vec<_>>(),
                    crate::exact_json::JsonEncoding::JavaScript,
                )?
            }
        } else {
            JsonPayload::array_from_items(
                &betas
                    .iter()
                    .map(|beta| JsonPayload::string(beta))
                    .collect::<Vec<_>>(),
                crate::exact_json::JsonEncoding::JavaScript,
            )?
        };
        source_set_field(
            &mut fields,
            "anthropic_beta",
            merged.as_json_source().to_owned(),
        );
    }
    extra = source_object_payload(fields)?;
    Ok(extra)
}

fn merge_extra_source(
    body: &mut JsonPayload,
    extra: JsonPayload,
    kind: AnthropicRequestKind,
    effort: Option<&AnthropicEffortPolicy>,
) -> Result<BTreeMap<String, Vec<u16>>, LlmError> {
    let main = kind == AnthropicRequestKind::Main;
    let mut body_fields = source_object_fields(body)?;
    let mut extra_fields = source_object_fields(&extra)?;
    let mut output_config = if main {
        spread_source_config(source_take_field(&mut extra_fields, "output_config"))?
    } else {
        Vec::new()
    };

    if main {
        if effort.is_some_and(|policy| !policy.supported) {
            source_remove_field(&mut output_config, "effort");
        } else if let Some(value) = effort
            .and_then(|policy| policy.value.as_ref())
            .filter(|value| value.is_root_string())
        {
            if source_field_source(&output_config, "effort").is_none() {
                source_set_field(
                    &mut output_config,
                    "effort",
                    value.as_json_source().to_owned(),
                );
            }
        }

        if let Some(computed_source) = source_take_field(&mut body_fields, "output_config") {
            let computed =
                JsonPayload::parse(&computed_source).map_err(|error| LlmError::InvalidRequest {
                    message: format!("computed output_config is invalid JSON: {error}"),
                })?;
            if computed.kind_at("") == Some(FlatJsonKind::Object) {
                let computed_fields = source_object_fields(&computed)?;
                for key in ["effort", "task_budget", "format"] {
                    if let Some(value) = source_field_source(&computed_fields, key) {
                        if key == "effort" && effort.is_some() {
                            continue;
                        }
                        if source_field_source(&output_config, key).is_none() {
                            source_set_field(&mut output_config, key, value.to_owned());
                        }
                    }
                }
                for (key, value) in computed_fields {
                    // The first pass deliberately consumes the computed
                    // `effort` field when a host effort policy is present,
                    // even when that policy removes it. Do not reintroduce
                    // that field while copying unrecognized computed keys.
                    if effort.is_some() && exact_key(&key, "effort") {
                        continue;
                    }
                    if !output_config.iter().any(|(units, _)| units == &key) {
                        source_set_field_units(&mut output_config, &key, value);
                    }
                }
            }
        }
    } else {
        let computed_source = source_take_field(&mut body_fields, "output_config");
        if computed_source.is_some() || effort.is_some() {
            let mut computed = match computed_source {
                Some(source) => {
                    let payload =
                        JsonPayload::parse(&source).map_err(|error| LlmError::InvalidRequest {
                            message: format!("computed output_config is invalid JSON: {error}"),
                        })?;
                    if payload.kind_at("") == Some(FlatJsonKind::Object) {
                        source_object_fields(&payload)?
                    } else {
                        Vec::new()
                    }
                }
                None => Vec::new(),
            };
            if let Some(policy) = effort {
                source_remove_field(&mut computed, "effort");
                if policy.supported {
                    if let Some(value) =
                        policy.value.as_ref().filter(|value| value.is_root_string())
                    {
                        source_set_field(
                            &mut computed,
                            "effort",
                            value.as_json_source().to_owned(),
                        );
                    }
                }
            }
            let mut ordered = Vec::new();
            for key in ["format", "effort"] {
                if let Some(value) = source_take_field(&mut computed, key) {
                    source_set_field(&mut ordered, key, value);
                }
            }
            for (key, value) in computed {
                source_set_field_units(&mut ordered, &key, value);
            }
            if !ordered.is_empty() {
                source_set_field(
                    &mut body_fields,
                    "output_config",
                    source_object_payload(ordered)?.as_json_source().to_owned(),
                );
            }
        }
    }

    let mut computed_tail = Vec::new();
    if main {
        for key in ["speed", "thread", "diagnostics", "stream"] {
            if let Some(source) = source_take_field(&mut body_fields, key) {
                computed_tail.push((key, source));
            }
        }
    }

    let order: &[&str] = if main {
        &[
            "model",
            "messages",
            "system",
            "tools",
            "tool_choice",
            "betas",
            "metadata",
            "max_tokens",
            "thinking",
            "temperature",
            "context_management",
            "safeguards",
        ]
    } else {
        &[
            "model",
            "max_tokens",
            "system",
            "messages",
            "tools",
            "tool_choice",
            "output_config",
            "temperature",
            "stop_sequences",
            "thinking",
            "betas",
            "metadata",
        ]
    };
    let mut ordered_body = Vec::with_capacity(body_fields.len() + extra_fields.len());
    for key in order {
        if let Some(source) = source_take_field(&mut body_fields, key) {
            source_set_field(&mut ordered_body, key, source);
        }
    }
    ordered_body.extend(body_fields);
    for (key, value) in extra_fields {
        source_set_field_units(&mut ordered_body, &key, value);
    }
    if main && !output_config.is_empty() {
        source_set_field(
            &mut ordered_body,
            "output_config",
            source_object_payload(output_config)?
                .as_json_source()
                .to_owned(),
        );
    }
    for (key, source) in computed_tail {
        source_set_field(&mut ordered_body, key, source);
    }

    *body = source_object_payload(ordered_body)?;
    Ok(BTreeMap::new())
}

fn source_field<'a>(root: JsonNodeRef<'a>, key: &str) -> Option<JsonNodeRef<'a>> {
    root.object_entries()?
        .into_iter()
        .find(|entry| exact_key(entry.key_units, key))
        .map(|entry| entry.value)
}

fn source_override_nodes(
    scanner: &ResponseJsonScanner,
    string_overrides: &BTreeMap<String, Vec<u16>>,
    roots: &[&str],
) -> Result<BTreeMap<usize, Vec<u16>>, LlmError> {
    let mut result = BTreeMap::new();
    for (pointer, units) in string_overrides {
        let selected = roots.iter().any(|root| {
            pointer == root
                || pointer
                    .strip_prefix(root)
                    .is_some_and(|suffix| suffix.starts_with('/'))
        });
        if !selected {
            continue;
        }
        let node = scanner
            .node_ref_at(pointer)
            .filter(|node| node.kind() == FlatJsonKind::String)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: format!("UTF-16 JSON override does not target a string leaf: {pointer}"),
            })?;
        result.insert(node.identity(), units.clone());
    }
    Ok(result)
}

fn source_node_json(node: JsonNodeRef<'_>) -> Result<String, LlmError> {
    crate::exact_json::serialize_json_node(node, crate::exact_json::JsonEncoding::JavaScript)
}

fn source_truthy(
    node: JsonNodeRef<'_>,
    exact_string_override: Option<&[u16]>,
) -> Result<bool, LlmError> {
    Ok(match node.kind() {
        FlatJsonKind::Null => false,
        FlatJsonKind::Boolean => node.primitive_source() == Some("true"),
        FlatJsonKind::Number => {
            let number = node
                .number_source()
                .and_then(|source| source.parse::<f64>().ok())
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "Messages body contains an invalid number".into(),
                })?;
            number != 0.0
        }
        FlatJsonKind::String => !exact_string_override
            .map(|units| units.to_vec())
            .or_else(|| node.string_units())
            .unwrap_or_default()
            .is_empty(),
        FlatJsonKind::Array | FlatJsonKind::Object => true,
    })
}

fn output_format_alias_requested(
    payload: &JsonPayload,
    string_overrides: &BTreeMap<String, Vec<u16>>,
) -> Result<bool, LlmError> {
    let scanner = payload.scanner()?;
    let root = scanner.root_node();
    if root.kind() != FlatJsonKind::Object {
        return Ok(false);
    }
    let Some(format) = source_field(root, "output_format") else {
        return Ok(false);
    };
    let format_override =
        source_string_override(&scanner, string_overrides, "/output_format", format)?;
    if !source_truthy(format, format_override.as_deref())? {
        return Ok(false);
    }
    if let Some(config) = source_field(root, "output_config") {
        if config.kind() == FlatJsonKind::Object {
            if let Some(config_format) = source_field(config, "format") {
                let config_format_override = source_string_override(
                    &scanner,
                    string_overrides,
                    "/output_config/format",
                    config_format,
                )?;
                if source_truthy(config_format, config_format_override.as_deref())? {
                    return Err(LlmError::InvalidRequest {
                        message: "Both output_format and output_config.format were provided. Please use only output_config.format (output_format is deprecated).".into(),
                    });
                }
            }
        }
    }
    Ok(true)
}

fn source_string_override(
    scanner: &ResponseJsonScanner,
    overrides: &BTreeMap<String, Vec<u16>>,
    pointer: &str,
    node: JsonNodeRef<'_>,
) -> Result<Option<Vec<u16>>, LlmError> {
    let Some(units) = overrides.get(pointer) else {
        return Ok(None);
    };
    if node.kind() != FlatJsonKind::String
        || scanner
            .node_ref_at(pointer)
            .is_none_or(|selected| selected.kind() != FlatJsonKind::String)
    {
        return Err(LlmError::InvalidRequest {
            message: format!("UTF-16 JSON override does not target a string leaf: {pointer}"),
        });
    }
    Ok(Some(units.clone()))
}

fn spread_source_fields(node: JsonNodeRef<'_>) -> Result<Vec<SourceField>, LlmError> {
    match node.kind() {
        FlatJsonKind::Object => node
            .object_entries()
            .unwrap_or_default()
            .into_iter()
            .map(|entry| Ok((entry.key_units.to_vec(), source_node_json(entry.value)?)))
            .collect(),
        FlatJsonKind::Array => {
            let length = node.array_len().unwrap_or(0);
            (0..length)
                .map(|index| {
                    let value = node
                        .array_item(index)
                        .ok_or_else(|| LlmError::InvalidRequest {
                            message: "Messages output_config array child disappeared".into(),
                        })?;
                    Ok((
                        index.to_string().encode_utf16().collect(),
                        source_node_json(value)?,
                    ))
                })
                .collect()
        }
        FlatJsonKind::String => Ok(node
            .string_units()
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(index, unit)| {
                (
                    index.to_string().encode_utf16().collect(),
                    JsonPayload::string_from_utf16_code_units(&[unit])
                        .as_json_source()
                        .to_owned(),
                )
            })
            .collect()),
        FlatJsonKind::Null | FlatJsonKind::Boolean | FlatJsonKind::Number => Ok(Vec::new()),
    }
}

fn normalize_output_format_source(payload: JsonPayload) -> Result<JsonPayload, LlmError> {
    let scanner = payload.scanner()?;
    let root = scanner.root_node();
    if root.kind() != FlatJsonKind::Object {
        return Ok(payload);
    }
    let Some(format) = source_field(root, "output_format") else {
        return Ok(payload);
    };
    let format_source = source_node_json(format)?;
    let mut config_fields = source_field(root, "output_config")
        .map(spread_source_fields)
        .transpose()?
        .unwrap_or_default();
    source_set_field(&mut config_fields, "format", format_source);
    let config_source = source_object_payload(config_fields)?
        .as_json_source()
        .to_owned();

    let mut root_fields = Vec::new();
    let mut had_config = false;
    for entry in root.object_entries().unwrap_or_default() {
        if exact_key(entry.key_units, "output_format") {
            continue;
        }
        if exact_key(entry.key_units, "output_config") {
            root_fields.push((entry.key_units.to_vec(), config_source.clone()));
            had_config = true;
        } else {
            root_fields.push((entry.key_units.to_vec(), source_node_json(entry.value)?));
        }
    }
    if !had_config {
        root_fields.push(("output_config".encode_utf16().collect(), config_source));
    }
    source_object_payload(root_fields)
}

fn source_message_header_parameters(
    payload: &JsonPayload,
    string_overrides: &BTreeMap<String, Vec<u16>>,
    mode: crate::RequestMode,
    count_token_betas: Option<&[String]>,
) -> Result<Vec<(&'static str, Option<String>)>, LlmError> {
    let scanner = payload.scanner()?;
    let root = scanner.root_node();
    if root.kind() != FlatJsonKind::Object {
        return Ok(Vec::new());
    }
    let string_overrides = source_override_nodes(
        &scanner,
        string_overrides,
        &["/betas", "/user_profile_id", "/workspace_id"],
    )?;
    let mut parameters: Vec<(&str, Option<String>)> = Vec::new();
    for (key, header) in [
        ("betas", "anthropic-beta"),
        ("user_profile_id", "anthropic-user-profile-id"),
        ("workspace_id", "anthropic-workspace-id"),
    ] {
        let counting = key == "betas" && mode == crate::RequestMode::CountTokens;
        let value = source_field(root, key);
        if counting {
            let mut combined: Option<Vec<u16>> = None;
            if let Some(betas) = count_token_betas {
                for beta in betas {
                    append_count_beta_value(&mut combined, beta.encode_utf16().collect(), header)?;
                }
            } else {
                match value {
                    None => {}
                    Some(node) if node.kind() == FlatJsonKind::Null => {}
                    Some(node) if node.kind() == FlatJsonKind::Array => {
                        for index in 0..node.array_len().unwrap_or(0) {
                            let item =
                                node.array_item(index)
                                    .ok_or_else(|| LlmError::InvalidRequest {
                                        message:
                                            "Messages count-token beta array child disappeared"
                                                .into(),
                                    })?;
                            append_count_beta_value(
                                &mut combined,
                                javascript_string_units(item, &string_overrides)?,
                                header,
                            )?;
                        }
                    }
                    Some(node) if node.kind() == FlatJsonKind::String => {
                        let units = string_overrides
                            .get(&node.identity())
                            .cloned()
                            .or_else(|| node.string_units())
                            .unwrap_or_default();
                        for units in utf16_iterable_items(&units) {
                            append_count_beta_value(&mut combined, units, header)?;
                        }
                    }
                    Some(_) => {
                        return Err(LlmError::InvalidRequest {
                            message: "(h ?? []) is not iterable".into(),
                        });
                    }
                }
            }
            append_count_beta_value(
                &mut combined,
                TOKEN_COUNTING.encode_utf16().collect(),
                header,
            )?;
            parameters.push((header, combined.map(units_to_string).transpose()?));
            continue;
        }

        let Some(value) = value.filter(|node| node.kind() != FlatJsonKind::Null) else {
            continue;
        };
        let items = if key != "betas" && value.kind() == FlatJsonKind::Array {
            (0..value.array_len().unwrap_or(0))
                .map(|index| {
                    value
                        .array_item(index)
                        .ok_or_else(|| LlmError::InvalidRequest {
                            message: "Messages header parameter array child disappeared".into(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![value]
        };
        if items.is_empty() {
            continue;
        }
        let mut combined: Option<Vec<u16>> = None;
        for item in items {
            if item.kind() == FlatJsonKind::Null {
                combined = None;
                continue;
            }
            append_header_value(
                &mut combined,
                javascript_string_units(item, &string_overrides)?,
                header,
            )?;
        }
        parameters.push((header, combined.map(units_to_string).transpose()?));
    }

    Ok(parameters)
}

fn apply_message_header_parameters(
    headers: &mut BTreeMap<String, String>,
    parameters: Vec<(&str, Option<String>)>,
) {
    for (header, value) in parameters {
        match value {
            Some(value) => set_header(headers, header, value.trim_matches([' ', '\t', '\r', '\n'])),
            None => headers.retain(|key, _| !key.eq_ignore_ascii_case(header)),
        }
    }
}

fn remove_message_header_fields_source(payload: JsonPayload) -> Result<JsonPayload, LlmError> {
    if !payload.is_root_object() {
        return Ok(payload);
    }
    let mut fields = source_object_fields(&payload)?;
    for key in ["betas", "user_profile_id", "workspace_id"] {
        source_remove_field(&mut fields, key);
    }
    source_object_payload(fields)
}

fn utf16_iterable_items(units: &[u16]) -> Vec<Vec<u16>> {
    let mut items = Vec::new();
    let mut index = 0;
    while index < units.len() {
        let unit = units[index];
        if (0xd800..=0xdbff).contains(&unit)
            && units
                .get(index + 1)
                .is_some_and(|next| (0xdc00..=0xdfff).contains(next))
        {
            items.push(vec![unit, units[index + 1]]);
            index += 2;
        } else {
            items.push(vec![unit]);
            index += 1;
        }
    }
    items
}

fn javascript_string_units(
    root: JsonNodeRef<'_>,
    string_overrides: &BTreeMap<usize, Vec<u16>>,
) -> Result<Vec<u16>, LlmError> {
    enum Work<'a> {
        Node(JsonNodeRef<'a>),
        Unit(u16),
    }
    let mut output = Vec::new();
    let mut stack = vec![Work::Node(root)];
    while let Some(work) = stack.pop() {
        match work {
            Work::Unit(unit) => output.push(unit),
            Work::Node(node) => match node.kind() {
                FlatJsonKind::Null => {}
                FlatJsonKind::Boolean => output.extend(
                    node.primitive_source()
                        .ok_or_else(|| LlmError::InvalidRequest {
                            message: "Messages boolean has no source token".into(),
                        })?
                        .encode_utf16(),
                ),
                FlatJsonKind::Number => {
                    let number = node
                        .number_source()
                        .and_then(|source| source.parse::<f64>().ok())
                        .ok_or_else(|| LlmError::InvalidRequest {
                            message: "Messages number has no valid source token".into(),
                        })?;
                    let text = if number.is_infinite() {
                        if number.is_sign_negative() {
                            "-Infinity"
                        } else {
                            "Infinity"
                        }
                        .to_owned()
                    } else if number.is_nan() {
                        "NaN".to_owned()
                    } else {
                        crate::exact_json::javascript_number(number)
                    };
                    output.extend(text.encode_utf16());
                }
                FlatJsonKind::String => output.extend(
                    string_overrides
                        .get(&node.identity())
                        .cloned()
                        .or_else(|| node.string_units())
                        .ok_or_else(|| LlmError::InvalidRequest {
                            message: "Messages string has no UTF-16 source units".into(),
                        })?,
                ),
                FlatJsonKind::Array => {
                    let length = node.array_len().unwrap_or(0);
                    for index in (0..length).rev() {
                        let child =
                            node.array_item(index)
                                .ok_or_else(|| LlmError::InvalidRequest {
                                    message: "Messages array child disappeared".into(),
                                })?;
                        stack.push(Work::Node(child));
                        if index > 0 {
                            stack.push(Work::Unit(b',' as u16));
                        }
                    }
                }
                FlatJsonKind::Object => {
                    let entries = node.object_entries().unwrap_or_default();
                    if entries
                        .iter()
                        .any(|entry| exact_key(entry.key_units, "toString"))
                    {
                        return Err(LlmError::InvalidRequest {
                            message: "Cannot convert object to primitive value".into(),
                        });
                    }
                    output.extend("[object Object]".encode_utf16());
                }
            },
        }
    }
    Ok(output)
}

fn append_header_value(
    combined: &mut Option<Vec<u16>>,
    mut units: Vec<u16>,
    header: &str,
) -> Result<(), LlmError> {
    let start = units
        .iter()
        .position(|unit| !matches!(*unit, 0x20 | 0x09 | 0x0d | 0x0a))
        .unwrap_or(units.len());
    let end = units
        .iter()
        .rposition(|unit| !matches!(*unit, 0x20 | 0x09 | 0x0d | 0x0a))
        .map_or(start, |index| index + 1);
    units = units[start..end].to_vec();
    if units
        .iter()
        .any(|unit| matches!(*unit, 0 | 0x0d | 0x0a) || *unit > 255)
    {
        return Err(LlmError::InvalidRequest {
            message: format!("Invalid HTTP header value for {header}"),
        });
    }
    match combined {
        Some(previous) => {
            previous.extend(", ".encode_utf16());
            previous.extend(units);
        }
        None => *combined = Some(units),
    }
    Ok(())
}

/// CountTokens spreads `betas` into an array and joins it with commas, then
/// trims the complete header value. Preserve whitespace inside each spread
/// item so the final trim matches Native's string-iterator behavior.
fn append_count_beta_value(
    combined: &mut Option<Vec<u16>>,
    units: Vec<u16>,
    header: &str,
) -> Result<(), LlmError> {
    if units
        .iter()
        .any(|unit| matches!(*unit, 0 | 0x0d | 0x0a) || *unit > 255)
    {
        return Err(LlmError::InvalidRequest {
            message: format!("Invalid HTTP header value for {header}"),
        });
    }
    match combined {
        Some(previous) => {
            previous.push(b',' as u16);
            previous.extend(units);
        }
        None => *combined = Some(units),
    }
    Ok(())
}

fn units_to_string(units: Vec<u16>) -> Result<String, LlmError> {
    String::from_utf16(&units).map_err(|_| LlmError::InvalidRequest {
        message: "Invalid HTTP header value contains an unpaired UTF-16 unit".into(),
    })
}

pub const TOKEN_COUNTING: &str = "token-counting-2024-11-01";

// The current native beta Messages SDK still performs this parameter
// normalization before create/countTokens. This is not a Harness API alias.
fn normalize_output_format(
    body: &mut Value,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
) -> Result<(), LlmError> {
    let Some(object) = body.as_object_mut() else {
        return Ok(());
    };
    let Some(format) = object
        .get("output_format")
        .filter(|value| javascript_truthy(value))
        .cloned()
    else {
        return Ok(());
    };
    if object
        .get("output_config")
        .and_then(|config| config.get("format"))
        .is_some_and(javascript_truthy)
    {
        return Err(LlmError::InvalidRequest { message: "Both output_format and output_config.format were provided. Please use only output_config.format (output_format is deprecated).".into() });
    }
    let mut spread_overrides = BTreeMap::new();
    let mut config = spread_config(object.get("output_config").cloned(), &mut spread_overrides);
    // A retained top-level exact string must spread its actual UTF-16 units.
    if object.get("output_config").is_some_and(Value::is_string) {
        if let Some(units) = string_overrides.get("/output_config") {
            config.clear();
            spread_overrides.clear();
            for (index, &unit) in units.iter().enumerate() {
                let key = index.to_string();
                config.insert(
                    key.clone(),
                    Value::String(String::from_utf16_lossy(&[unit])),
                );
                if (0xD800..=0xDFFF).contains(&unit) {
                    spread_overrides.insert(format!("/output_config/{key}"), vec![unit]);
                }
            }
        }
    }
    let mut remapped = BTreeMap::new();
    string_overrides.retain(|path, units| {
        if path == "/output_config"
            || path == "/output_config/format"
            || path.starts_with("/output_config/format/")
        {
            return false;
        }
        if let Some(suffix) = path
            .strip_prefix("/output_format")
            .filter(|suffix| suffix.is_empty() || suffix.starts_with('/'))
        {
            remapped.insert(format!("/output_config/format{suffix}"), units.clone());
            return false;
        }
        true
    });
    config.insert("format".into(), format);
    javascript_key_order(&mut config);
    object.shift_remove("output_format");
    // Inserting over an existing key preserves its original object position.
    object.insert("output_config".into(), Value::Object(config));
    string_overrides.extend(spread_overrides);
    string_overrides.extend(remapped);
    Ok(())
}

fn javascript_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

fn materialize_message_headers(
    body: &mut Value,
    headers: &mut BTreeMap<String, String>,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
    mode: crate::RequestMode,
) -> Result<(), LlmError> {
    let Some(object) = body.as_object_mut() else {
        return Ok(());
    };
    // Convert before mutating; JSON values cannot have callable toString.
    let mut parameters = Vec::new();
    for (key, header) in [
        ("betas", "anthropic-beta"),
        ("user_profile_id", "anthropic-user-profile-id"),
        ("workspace_id", "anthropic-workspace-id"),
    ] {
        let counting = key == "betas" && mode == crate::RequestMode::CountTokens;
        if let Some(value) = object
            .get(key)
            .filter(|value| !value.is_null())
            .or_else(|| counting.then_some(&Value::Null))
        {
            let mut value = value.clone();
            let pointer = format!("/{key}");
            for (path, units) in string_overrides.iter() {
                let Some(relative) = path
                    .strip_prefix(&pointer)
                    .filter(|suffix| suffix.is_empty() || suffix.starts_with('/'))
                else {
                    continue;
                };
                if !crate::exact_json::canonical_string_pointer(&value, relative) {
                    return Err(LlmError::InvalidRequest {
                        message: format!(
                            "UTF-16 JSON override does not target a string leaf: {path}"
                        ),
                    });
                }
                let Some(Value::String(text)) = value.pointer_mut(relative) else {
                    return Err(LlmError::InvalidRequest {
                        message: format!(
                            "UTF-16 JSON override does not target a string leaf: {path}"
                        ),
                    });
                };
                *text = String::from_utf16(units).map_err(|_| LlmError::InvalidRequest {
                    message: format!("Invalid HTTP header value for {header}"),
                })?;
            }
            if counting {
                let mut betas = match value {
                    Value::Null => Vec::new(),
                    Value::Array(items) => items,
                    Value::String(text) => text
                        .chars()
                        .map(|ch| Value::String(ch.to_string()))
                        .collect(),
                    _ => {
                        return Err(LlmError::InvalidRequest {
                            message: "(h ?? []) is not iterable".into(),
                        });
                    }
                };
                betas.push(Value::String(TOKEN_COUNTING.into()));
                value = Value::Array(betas);
            }
            let items = if key != "betas" {
                value.as_array().map(Vec::as_slice)
            } else {
                None
            };
            let items = items.unwrap_or_else(|| std::slice::from_ref(&value));
            if items.is_empty() {
                continue;
            }
            let mut combined = None;
            for item in items {
                if item.is_null() {
                    combined = None;
                    continue;
                }
                let value = javascript_string(item)?;
                let value = value.trim_matches([' ', '\t', '\r', '\n']);
                if value.contains(['\0', '\r', '\n']) || value.chars().any(|ch| u32::from(ch) > 255)
                {
                    return Err(LlmError::InvalidRequest {
                        message: format!("Invalid HTTP header value for {header}"),
                    });
                }
                combined = Some(match combined {
                    Some(previous) => format!("{previous}, {value}"),
                    None => value.to_owned(),
                });
            }
            parameters.push((header, combined));
        }
    }
    for key in ["betas", "user_profile_id", "workspace_id"] {
        object.shift_remove(key);
        let pointer = format!("/{key}");
        string_overrides.retain(|path, _| {
            path != &pointer
                && !path
                    .strip_prefix(&pointer)
                    .is_some_and(|tail| tail.starts_with('/'))
        });
    }
    for (header, value) in parameters {
        match value {
            Some(value) => set_header(headers, header, value.trim_matches([' ', '\t', '\r', '\n'])),
            None => headers.retain(|key, _| !key.eq_ignore_ascii_case(header)),
        }
    }
    Ok(())
}

fn spread_config(
    value: Option<Value>,
    overrides: &mut BTreeMap<String, Vec<u16>>,
) -> Map<String, Value> {
    match value {
        Some(Value::Object(map)) => map,
        Some(Value::Array(items)) => items
            .into_iter()
            .enumerate()
            .map(|(i, value)| (i.to_string(), value))
            .collect(),
        Some(Value::String(text)) => text
            .encode_utf16()
            .enumerate()
            .map(|(i, unit)| {
                let key = i.to_string();
                if (0xD800..=0xDFFF).contains(&unit) {
                    overrides.insert(format!("/output_config/{key}"), vec![unit]);
                }
                (key, Value::String(String::from_utf16_lossy(&[unit])))
            })
            .collect(),
        _ => Map::new(),
    }
}

// Object enumeration emits canonical array-index keys first, numerically.
fn javascript_key_order(map: &mut Map<String, Value>) {
    for value in map.values_mut() {
        match value {
            Value::Object(child) => javascript_key_order(child),
            Value::Array(items) => reorder_array_objects(items),
            _ => {}
        }
    }
    if !map.keys().any(|key| array_index(key).is_some()) {
        return;
    }
    let mut entries: Vec<_> = std::mem::take(map).into_iter().collect();
    entries.sort_by_key(|(key, _)| array_index(key).map_or((1, 0), |index| (0, index)));
    map.extend(entries);
}
fn reorder_array_objects(items: &mut [Value]) {
    for value in items {
        match value {
            Value::Object(map) => javascript_key_order(map),
            Value::Array(items) => reorder_array_objects(items),
            _ => {}
        }
    }
}
fn array_index(key: &str) -> Option<u32> {
    key.parse::<u32>()
        .ok()
        .filter(|&index| index != u32::MAX && index.to_string() == key)
}
pub fn remove_fast_source(
    body: &mut JsonPayload,
    headers: &mut std::collections::BTreeMap<String, String>,
    string_overrides: &mut BTreeMap<String, Vec<u16>>,
    beta: &str,
) -> Result<(), LlmError> {
    let scanner = body.scanner()?;
    let next_body = if scanner.root_node().kind() == FlatJsonKind::Object {
        let mut fields = source_object_fields(body)?;
        source_remove_field(&mut fields, "speed");
        Some(source_object_payload(fields)?)
    } else {
        None
    };
    if let Some(next_body) = next_body {
        *body = next_body;
    }
    string_overrides.retain(|pointer, _| pointer != "/speed" && !pointer.starts_with("/speed/"));
    let keys: Vec<_> = headers
        .keys()
        .filter(|key| key.eq_ignore_ascii_case("anthropic-beta"))
        .cloned()
        .collect();
    for key in keys {
        let header = headers.remove(&key).unwrap_or_default();
        let retained = header
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty() && *part != beta)
            .collect::<Vec<_>>()
            .join(",");
        if !retained.is_empty() {
            headers.insert(key, retained);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_request_matches_only_the_configured_base_endpoint() {
        for (base, request, expected) in [
            (
                "https://api.anthropic.com",
                "https://api.anthropic.com/v1/messages",
                true,
            ),
            (
                "https://proxy.example/api/",
                "https://proxy.example/api/v1/messages?beta=true#fragment",
                true,
            ),
            (
                "https://proxy.example/api///",
                "https://proxy.example/api/v1/messages",
                true,
            ),
            (
                "https://api.anthropic.com",
                "https://api.anthropic.com:443/v1/messages",
                true,
            ),
            (
                "https://user:secret@proxy.example/api",
                "https://user:secret@proxy.example/api/v1/messages",
                true,
            ),
            (
                "https://api.anthropic.com",
                "https://api.anthropic.com/v1/messages/count_tokens",
                false,
            ),
            (
                "https://api.anthropic.com",
                "https://api.anthropic.com.example.test/v1/messages",
                false,
            ),
            (
                "https://api.anthropic.com",
                "http://api.anthropic.com/v1/messages",
                false,
            ),
            (
                "https://api.anthropic.com",
                "https://api.anthropic.com:8443/v1/messages",
                false,
            ),
            (
                "https://user:secret@proxy.example/api",
                "https://other:secret@proxy.example/api/v1/messages",
                false,
            ),
            (
                "https://api.anthropic.com?route=one",
                "https://api.anthropic.com/v1/messages",
                false,
            ),
            (
                "https://api.anthropic.com#fragment",
                "https://api.anthropic.com/v1/messages",
                false,
            ),
            ("not a URL", "https://api.anthropic.com/v1/messages", false),
            ("https://api.anthropic.com", "not a URL", false),
        ] {
            assert_eq!(
                messages_request_matches_base_url(base, request),
                expected,
                "base={base:?}, request={request:?}"
            );
        }
    }

    fn payload(value: &Value) -> JsonPayload {
        JsonPayload::from_generated_value(value).unwrap()
    }

    fn materialize(payload: &JsonPayload) -> Value {
        serde_json::from_str(payload.as_json_source()).unwrap()
    }

    fn wire(payload: &JsonPayload, overrides: &BTreeMap<String, Vec<u16>>) -> String {
        String::from_utf8(
            crate::exact_json::serialize_json_payload_with_overrides(
                payload,
                overrides,
                crate::exact_json::JsonEncoding::JavaScript,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn apply_value_policy(
        policy: AnthropicRequestPolicy,
        body: &mut Value,
        headers: &mut BTreeMap<String, String>,
        overrides: &mut BTreeMap<String, Vec<u16>>,
        url: &mut String,
    ) -> Result<(), LlmError> {
        let mut source = payload(body);
        policy.apply(&mut source, headers, overrides, url)?;
        *body = materialize(&source);
        Ok(())
    }

    #[test]
    fn current_fallback_stream_flag_matches_native_final_spread_bytes() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/fallback_stream_flag_2_1_288.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut body = payload(&case["body"]);
            let mut strings = BTreeMap::new();
            nonstream_fallback_parameters_source(&mut body, &mut strings).unwrap();
            assert_eq!(
                wire(&body, &strings),
                case["expected_json"].as_str().unwrap(),
                "{case}"
            );
        }
    }

    #[test]
    fn fallback_clears_overridden_stream_strings_and_retains_other_exact_strings() {
        let mut body = JsonPayload::parse(
            r#"{"stream":"placeholder","keep":"placeholder","streaming":"unchanged"}"#,
        )
        .unwrap();
        let mut strings = BTreeMap::from([
            ("/stream".into(), vec![0xd800]),
            ("/stream/nested".into(), vec![0xd800]),
            ("/keep".into(), vec![0xdc00]),
        ]);
        nonstream_fallback_parameters_source(&mut body, &mut strings).unwrap();
        assert_eq!(strings, BTreeMap::from([("/keep".into(), vec![0xdc00])]));
        assert_eq!(
            wire(&body, &strings),
            r#"{"stream":false,"keep":"\udc00","streaming":"unchanged"}"#
        );
    }
    use serde_json::json;

    #[test]
    fn native_count_tokens_parameters_match_reference() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/count_tokens_2_1_287.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut body = JsonPayload::from_generated_value(&case["input"]["body"]).unwrap();
            let mut headers =
                serde_json::from_value(case["input"]["default_headers"].clone()).unwrap();
            let mut url = "https://api.anthropic.com/v1/messages/count_tokens".to_owned();
            let result = normalize_message_parameters(
                &mut body,
                &mut headers,
                &mut BTreeMap::new(),
                &mut url,
                crate::RequestMode::CountTokens,
            );
            if let Some(error) = case["expected"]["error"].as_str() {
                let result = result.expect_err(&case.to_string());
                if error.contains("not iterable")
                    || error.starts_with("Both output_format")
                    || error == "Cannot convert object to primitive value"
                {
                    assert_eq!(
                        result,
                        LlmError::InvalidRequest {
                            message: error.into()
                        },
                        "{case}"
                    );
                } else {
                    assert!(matches!(result, LlmError::InvalidRequest { .. }), "{case}");
                }
            } else {
                result.unwrap();
                assert_eq!(
                    url,
                    format!(
                        "https://api.anthropic.com{}",
                        case["expected"]["url"].as_str().unwrap()
                    )
                );
                assert_eq!(
                    serde_json::to_value(headers).unwrap(),
                    case["expected"]["headers"],
                    "{case}"
                );
                assert_eq!(
                    body.canonical_json(crate::exact_json::JsonEncoding::JavaScript)
                        .unwrap(),
                    case["expected"]["body_json"].as_str().unwrap(),
                    "{case}"
                );
            }
        }
    }

    #[test]
    fn source_policy_moves_virtual_fields_without_materializing_deep_or_lone_json() {
        let deep = format!("{}0{}", "[".repeat(2048), "]".repeat(2048));
        let source = format!(
            r#"{{"opaque":{{"\ud800":"\udfff","deep":{deep}}},"output_config":{{"format":false,"tail":"kept"}},"output_format":{{"schema":{{"type":"object"}}}},"betas":["one","two"],"user_profile_id":["first",null,"last"]}}"#
        );
        let mut body = JsonPayload::parse(&source).unwrap();
        let mut headers = BTreeMap::from([("Anthropic-User-Profile-Id".into(), "old".into())]);
        let mut overrides = BTreeMap::new();
        let mut url = "https://api.anthropic.com/v1/messages".to_owned();
        normalize_message_parameters(
            &mut body,
            &mut headers,
            &mut overrides,
            &mut url,
            crate::RequestMode::Complete,
        )
        .unwrap();

        let serialized = body
            .canonical_json(crate::exact_json::JsonEncoding::JavaScript)
            .unwrap();
        assert!(serialized.contains(r#""\ud800":"\udfff""#));
        assert!(serialized.contains(&deep));
        assert!(serialized
            .contains(r#""output_config":{"format":{"schema":{"type":"object"}},"tail":"kept"}"#));
        assert!(!serialized.contains("output_format"));
        assert!(!serialized.contains("\"betas\""));
        assert!(!serialized.contains("user_profile_id"));
        assert_eq!(headers["anthropic-beta"], "one,two");
        assert_eq!(headers["anthropic-user-profile-id"], "last");
        assert!(overrides.is_empty());
        assert_eq!(url, "https://api.anthropic.com/v1/messages?beta=true");
    }

    #[test]
    fn source_output_format_truthiness_and_alias_use_exact_override_units() {
        let mut body = JsonPayload::parse(r#"{"output_format":""}"#).unwrap();
        let mut overrides = BTreeMap::from([("/output_format".into(), vec![0xd800])]);
        normalize_message_parameters(
            &mut body,
            &mut BTreeMap::new(),
            &mut overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
            crate::RequestMode::Complete,
        )
        .unwrap();
        assert!(overrides.is_empty());
        assert_eq!(
            body.canonical_json(crate::exact_json::JsonEncoding::JavaScript)
                .unwrap(),
            r#"{"output_config":{"format":"\ud800"}}"#
        );
    }

    #[test]
    fn source_count_tokens_spreads_utf16_iterables_and_is_transactional_on_error() {
        let mut body =
            JsonPayload::parse(r#"{"messages":[],"betas":"ab","keep":"\ud800"}"#).unwrap();
        let mut headers = BTreeMap::new();
        let mut overrides = BTreeMap::new();
        let mut url =
            "https://api.anthropic.com/v1/messages/count_tokens?beta=false&x=1".to_owned();
        normalize_message_parameters(
            &mut body,
            &mut headers,
            &mut overrides,
            &mut url,
            crate::RequestMode::CountTokens,
        )
        .unwrap();
        assert_eq!(headers["anthropic-beta"], "a,b,token-counting-2024-11-01");
        assert_eq!(
            url,
            "https://api.anthropic.com/v1/messages/count_tokens?x=1&beta=true"
        );
        assert!(body.as_json_source().contains(r#""keep":"\ud800""#));

        let mut invalid = JsonPayload::parse(r#"{"messages":[],"betas":{}}"#).unwrap();
        let before_body = invalid.as_json_source().to_owned();
        let mut headers = BTreeMap::from([("anthropic-beta".into(), "original".into())]);
        let before_headers = headers.clone();
        let mut overrides = BTreeMap::new();
        let mut url = "https://api.anthropic.com/v1/messages/count_tokens".to_owned();
        let before_url = url.clone();
        let error = normalize_message_parameters(
            &mut invalid,
            &mut headers,
            &mut overrides,
            &mut url,
            crate::RequestMode::CountTokens,
        )
        .unwrap_err();
        assert_eq!(
            error,
            LlmError::InvalidRequest {
                message: "(h ?? []) is not iterable".into()
            }
        );
        assert_eq!(invalid.as_json_source(), before_body);
        assert_eq!(headers, before_headers);
        assert_eq!(url, before_url);
    }

    #[test]
    fn source_output_config_spread_preserves_utf16_and_javascript_index_order() {
        let mut body = JsonPayload::parse(
            r#"{"output_config":"\ud800A","output_format":{"10":"ten","3":"three","2":"two"}}"#,
        )
        .unwrap();
        normalize_message_parameters(
            &mut body,
            &mut BTreeMap::new(),
            &mut BTreeMap::new(),
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
            crate::RequestMode::Complete,
        )
        .unwrap();
        assert_eq!(
            body.canonical_json(crate::exact_json::JsonEncoding::JavaScript)
                .unwrap(),
            r#"{"output_config":{"0":"\ud800","1":"A","format":{"2":"two","3":"three","10":"ten"}}}"#
        );

        let mut body =
            JsonPayload::parse(r#"{"output_config":[{"nested":true},null],"output_format":true}"#)
                .unwrap();
        normalize_message_parameters(
            &mut body,
            &mut BTreeMap::new(),
            &mut BTreeMap::new(),
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
            crate::RequestMode::Complete,
        )
        .unwrap();
        assert_eq!(
            body.canonical_json(crate::exact_json::JsonEncoding::JavaScript)
                .unwrap(),
            r#"{"output_config":{"0":{"nested":true},"1":null,"format":true}}"#
        );
    }

    #[test]
    fn default_source_policy_emits_native_main_template_order() {
        let mut body = JsonPayload::parse(
            r#"{ "model":"m", "max_tokens":4096, "messages":[{"role":"user","content":"hi"}] }"#,
        )
        .unwrap();
        let mut headers = BTreeMap::from([("x-existing".into(), "kept".into())]);
        let original_headers = headers.clone();
        let mut overrides = BTreeMap::from([("/messages/0/content".into(), vec![0xd800])]);
        let original_overrides = overrides.clone();
        let mut url = "https://api.anthropic.com/v1/messages?existing=1".to_owned();
        let original_url = url.clone();

        AnthropicRequestPolicy::default()
            .apply(&mut body, &mut headers, &mut overrides, &mut url)
            .unwrap();

        assert_eq!(
            body.as_json_source(),
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":4096}"#
        );
        assert_eq!(headers, original_headers);
        assert_eq!(overrides, original_overrides);
        assert_eq!(url, original_url);
    }

    #[test]
    fn message_header_normalization_keeps_body_source_when_no_virtual_fields_exist() {
        let mut body = JsonPayload::parse(
            r#"{ "model":"m", "max_tokens":4096, "messages":[{"role":"user","content":"hi"}] }"#,
        )
        .unwrap();
        let original = body.as_json_source().to_owned();
        let mut headers = BTreeMap::new();
        let mut overrides = BTreeMap::from([("/messages/0/content".into(), vec![0xd800])]);
        let original_overrides = overrides.clone();
        let mut url = "https://api.anthropic.com/v1/messages?beta=true".to_owned();

        normalize_message_parameters(
            &mut body,
            &mut headers,
            &mut overrides,
            &mut url,
            crate::RequestMode::Complete,
        )
        .unwrap();

        assert_eq!(body.as_json_source(), original);
        assert_eq!(overrides, original_overrides);
    }

    #[test]
    fn source_policy_evaluates_native_number_coercion_before_json_stringification() {
        let mut body = JsonPayload::parse(r#"{"output_format":1e400,"betas":1e400}"#).unwrap();
        let mut headers = BTreeMap::new();
        normalize_message_parameters(
            &mut body,
            &mut headers,
            &mut BTreeMap::new(),
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
            crate::RequestMode::Complete,
        )
        .unwrap();
        assert_eq!(headers["anthropic-beta"], "Infinity");
        assert_eq!(
            body.canonical_json(crate::exact_json::JsonEncoding::JavaScript)
                .unwrap(),
            r#"{"output_config":{"format":null}}"#
        );

        let mut conflict = JsonPayload::parse(
            r#"{"output_config":{"format":1e400},"output_format":{"type":"json"}}"#,
        )
        .unwrap();
        let before_body = conflict.as_json_source().to_owned();
        let mut headers = BTreeMap::new();
        let mut overrides = BTreeMap::new();
        let mut url = "https://api.anthropic.com/v1/messages".to_owned();
        let before_url = url.clone();
        assert_eq!(
            normalize_message_parameters(
                &mut conflict,
                &mut headers,
                &mut overrides,
                &mut url,
                crate::RequestMode::Complete,
            )
            .unwrap_err(),
            LlmError::InvalidRequest {
                message: "Both output_format and output_config.format were provided. Please use only output_config.format (output_format is deprecated).".into()
            }
        );
        assert_eq!(conflict.as_json_source(), before_body);
        assert!(headers.is_empty());
        assert_eq!(url, before_url);
    }

    #[test]
    fn source_helpers_own_count_betas_and_exact_auto_tool_choice_removal() {
        let mut body =
            JsonPayload::parse(r#"{"model":"m","tool_choice":{"type":"auto"}}"#).unwrap();
        let mut overrides = BTreeMap::new();
        assert!(
            omit_default_auto_tool_choice(&mut body, &mut overrides, &ToolChoice::Auto).unwrap()
        );
        assert_eq!(body.as_json_source(), r#"{"model":"m"}"#);
        assert!(
            !omit_default_auto_tool_choice(&mut body, &mut overrides, &ToolChoice::Any).unwrap()
        );

        let mut body =
            JsonPayload::parse(r#"{"tool_choice":{"type":"auto","extra":true}}"#).unwrap();
        assert!(
            !omit_default_auto_tool_choice(&mut body, &mut overrides, &ToolChoice::Auto).unwrap()
        );

        let mut body =
            JsonPayload::parse(r#"{"messages":[],"betas":"old","output_format":1e400}"#).unwrap();
        let mut betas_overrides = BTreeMap::from([("/betas".into(), vec![0xd800])]);
        let mut headers = BTreeMap::new();
        normalize_count_tokens_message_parameters(
            &mut body,
            &mut headers,
            &mut betas_overrides,
            &mut "https://api.anthropic.com/v1/messages/count_tokens".to_owned(),
            &["a".into(), "b".into()],
        )
        .unwrap();
        assert!(betas_overrides.is_empty());
        assert_eq!(headers["anthropic-beta"], "a,b,token-counting-2024-11-01");
        assert_eq!(
            body.canonical_json(crate::exact_json::JsonEncoding::JavaScript)
                .unwrap(),
            r#"{"messages":[],"output_config":{"format":null}}"#
        );
    }

    #[test]
    fn native_main_and_side_templates_match_full_layout_bytes() {
        let fixture = JsonPayload::parse(include_str!(
            "../../../tests/fixtures/request_layout_2_1_287.json"
        ))
        .unwrap();
        let case_count = fixture.array_len("/cases").unwrap();
        for index in 0..case_count {
            let case = format!("/cases/{index}");
            let mut body = fixture
                .payload_at(&format!("{case}/input/base"))
                .expect("base JSON source");
            let extra_body = fixture
                .payload_at(&format!("{case}/input/extra"))
                .expect("extra JSON source");
            let mut strings = BTreeMap::new();
            AnthropicRequestPolicy {
                request_kind: if fixture.string_at(&format!("{case}/input/kind")).as_deref()
                    == Some("main")
                {
                    AnthropicRequestKind::Main
                } else {
                    AnthropicRequestKind::SideQuery
                },
                extra_body,
                ..Default::default()
            }
            .apply(
                &mut body,
                &mut BTreeMap::new(),
                &mut strings,
                &mut "https://api.anthropic.com/v1/messages".into(),
            )
            .unwrap();
            assert_eq!(
                wire(&body, &strings),
                fixture
                    .string_at(&format!("{case}/expected/body_json"))
                    .expect("expected body JSON source"),
                "case {index}"
            );
        }
    }

    #[test]
    fn messages_beta_endpoint_preserves_query_and_other_transports() {
        for (source, expected) in [
            (
                "https://api.anthropic.com/v1/messages",
                "https://api.anthropic.com/v1/messages?beta=true",
            ),
            (
                "https://api.anthropic.com/v1/messages?route=one&beta=false",
                "https://api.anthropic.com/v1/messages?route=one&beta=true",
            ),
            (
                "https://api.anthropic.com/v1/messages?beta=true&route=one",
                "https://api.anthropic.com/v1/messages?beta=true&route=one",
            ),
        ] {
            let mut url = source.to_owned();
            let mut body = JsonPayload::parse(r#"{"messages":[]}"#).unwrap();
            AnthropicRequestPolicy {
                message_header_parameters: true,
                ..Default::default()
            }
            .apply(
                &mut body,
                &mut BTreeMap::new(),
                &mut BTreeMap::new(),
                &mut url,
            )
            .unwrap();
            assert_eq!(url, expected);
        }
        let mut url = "https://cloud/v1/messages?custom=true".to_owned();
        let mut body = JsonPayload::parse(r#"{"messages":[]}"#).unwrap();
        AnthropicRequestPolicy::default()
            .apply(
                &mut body,
                &mut BTreeMap::new(),
                &mut BTreeMap::new(),
                &mut url,
            )
            .unwrap();
        assert_eq!(url, "https://cloud/v1/messages?custom=true");
    }

    #[test]
    fn bedrock_route_deletes_all_beta_query_values_and_keeps_other_pairs_ordered() {
        let mut url = "https://bedrock.example/model?route=one&beta=true&route=two&beta=false&label=a%20b&empty".to_owned();
        let mut body = JsonPayload::parse(r#"{"messages":[]}"#).unwrap();
        AnthropicRequestPolicy {
            protocol: ProtocolFamily::BedrockClaude,
            message_header_parameters: true,
            ..Default::default()
        }
        .apply(
            &mut body,
            &mut BTreeMap::new(),
            &mut BTreeMap::new(),
            &mut url,
        )
        .unwrap();

        assert_eq!(
            url,
            "https://bedrock.example/model?route=one&route=two&label=a+b&empty="
        );
    }

    #[test]
    fn plain_text_message_source_moves_utf16_override_without_flattening_cached_blocks() {
        let mut body = JsonPayload::parse(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"placeholder"}]}]}"#,
        )
        .unwrap();
        let exact_text = "Native insight".encode_utf16().chain([0xd800]).collect();
        let mut overrides = BTreeMap::from([("/messages/0/content/0/text".to_owned(), exact_text)]);
        AnthropicRequestPolicy {
            plain_text_message_content: true,
            ..Default::default()
        }
        .apply(
            &mut body,
            &mut BTreeMap::new(),
            &mut overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
        )
        .unwrap();

        assert_eq!(
            wire(&body, &overrides),
            r#"{"messages":[{"role":"user","content":"Native insight\ud800"}]}"#
        );
        assert!(overrides.contains_key("/messages/0/content"));
        assert!(!overrides.contains_key("/messages/0/content/0/text"));

        let mut cached = JsonPayload::parse(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"cached","cache_control":{"type":"ephemeral"}}]}]}"#,
        )
        .unwrap();
        let cached_text = "cached".encode_utf16().chain([0xd800]).collect();
        let mut cached_overrides =
            BTreeMap::from([("/messages/0/content/0/text".to_owned(), cached_text)]);
        AnthropicRequestPolicy {
            plain_text_message_content: true,
            ..Default::default()
        }
        .apply(
            &mut cached,
            &mut BTreeMap::new(),
            &mut cached_overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
        )
        .unwrap();

        assert!(cached
            .payload_at("/messages/0/content")
            .unwrap()
            .is_root_array());
        assert!(cached_overrides.contains_key("/messages/0/content/0/text"));
    }

    #[test]
    fn current_native_output_format_normalization_matches_reference_bytes() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/output_format_2_1_287.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut body = case["body"].clone();
            let mut strings = BTreeMap::new();
            let result = normalize_output_format(&mut body, &mut strings);
            if let Some(message) = case["expected"]["error"].as_str() {
                assert_eq!(
                    result.unwrap_err(),
                    LlmError::InvalidRequest {
                        message: message.into()
                    },
                    "{case}"
                );
            } else {
                result.unwrap();
                assert_eq!(
                    String::from_utf8(
                        crate::exact_json::serialize(
                            &body,
                            &strings,
                            crate::exact_json::JsonEncoding::JavaScript
                        )
                        .unwrap()
                    )
                    .unwrap(),
                    case["expected"]["body_json"].as_str().unwrap(),
                    "{case}"
                );
            }
        }
    }

    #[test]
    fn output_format_normalization_moves_exact_strings_and_retains_config_position() {
        let mut body = json!({"output_config":"display", "tail":true, "output_format":{"description":"display"}});
        let mut strings = BTreeMap::from([
            ("/output_config".into(), vec![0xD800, 65]),
            ("/output_format/description".into(), vec![0xDC00]),
        ]);
        normalize_output_format(&mut body, &mut strings).unwrap();
        assert_eq!(
            String::from_utf8(
                crate::exact_json::serialize(
                    &body,
                    &strings,
                    crate::exact_json::JsonEncoding::JavaScript
                )
                .unwrap()
            )
            .unwrap(),
            r#"{"output_config":{"0":"\ud800","1":"A","format":{"description":"\udc00"}},"tail":true}"#
        );
        let mut body = json!({"output_config":{"format":"already"},"output_format":"new"});
        let mut strings = BTreeMap::from([("/output_format".into(), vec![0xD800])]);
        let before = (body.clone(), strings.clone());
        assert!(normalize_output_format(&mut body, &mut strings).is_err());
        assert_eq!((body, strings), before);
    }

    #[test]
    fn native_messages_parameters_move_to_headers_without_body_residue() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/message_parameters_2_1_287.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut body = case["input"]["body"].clone();
            let mut headers: BTreeMap<_, _> = case["input"]["default_headers"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, value)| (key.clone(), value.as_str().unwrap().to_owned()))
                .collect();
            let mut strings = BTreeMap::new();
            let result = materialize_message_headers(
                &mut body,
                &mut headers,
                &mut strings,
                crate::RequestMode::Complete,
            );
            if case["expected"].get("error").is_some() {
                assert!(
                    matches!(result, Err(LlmError::InvalidRequest { .. })),
                    "{case}"
                );
            } else {
                result.unwrap();
                assert_eq!(body, case["expected"]["body"], "{case}");
                assert_eq!(
                    serde_json::to_value(headers).unwrap(),
                    case["expected"]["headers"],
                    "{case}"
                );
            }
        }
        let mut body =
            json!({"betas":"custom","user_profile_id":"p","workspace_id":"w","speed":"fast"});
        let mut strings = BTreeMap::from([
            ("/betas".into(), "custom".encode_utf16().collect()),
            ("/user_profile_id".into(), vec![112]),
            ("/workspace_id".into(), vec![119]),
            ("/speed".into(), vec![0xD803]),
        ]);
        materialize_message_headers(
            &mut body,
            &mut BTreeMap::new(),
            &mut strings,
            crate::RequestMode::Complete,
        )
        .unwrap();
        assert_eq!(strings, BTreeMap::from([("/speed".into(), vec![0xD803])]));
        assert!(crate::exact_json::serialize(
            &body,
            &strings,
            crate::exact_json::JsonEncoding::JavaScript
        )
        .is_ok());
    }

    #[test]
    fn native_extra_body_sanitation_matches_reference_values_and_metadata_bytes() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/extra_body_2_1_287.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let result = sanitize_extra_body(payload(&case["input"]["extra"]));
            if let Some(message) = case["expected"]["error"].as_str() {
                assert_eq!(
                    result.unwrap_err(),
                    LlmError::InvalidRequest {
                        message: message.into()
                    },
                    "{case}"
                );
            } else {
                let betas: Vec<_> = case["input"]["betas"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|beta| beta.as_str().unwrap().to_owned())
                    .collect();
                assert_eq!(
                    materialize(&beta_body(result.unwrap(), &betas).unwrap()),
                    case["expected"]["value"],
                    "{case}"
                );
            }
        }
    }

    #[test]
    fn effort_admission_clears_only_removed_exact_strings_before_extra_spread() {
        for kind in [AnthropicRequestKind::Main, AnthropicRequestKind::SideQuery] {
            for explicit in [false, true] {
                let mut body = json!({"messages":[],"output_config":{"effort":"automatic","format":{"description":"exact"}}});
                let mut strings = BTreeMap::from([
                    ("/output_config/effort".into(), vec![0xD800]),
                    ("/output_config/format/description".into(), vec![0xDC00]),
                ]);
                apply_value_policy(
                    AnthropicRequestPolicy {
                        request_kind: kind,
                        effort: Some(AnthropicEffortPolicy {
                            supported: false,
                            value: None,
                        }),
                        extra_body: if explicit {
                            payload(&json!({"output_config":{"effort":"explicit","unknown":true}}))
                        } else {
                            JsonPayload::parse("{}").unwrap()
                        },
                        ..Default::default()
                    },
                    &mut body,
                    &mut BTreeMap::new(),
                    &mut strings,
                    &mut "https://api.anthropic.com/v1/messages".into(),
                )
                .unwrap();
                assert!(!strings.contains_key("/output_config/effort"));
                if kind == AnthropicRequestKind::Main || !explicit {
                    assert!(body.pointer("/output_config/effort").is_none());
                    assert_eq!(strings["/output_config/format/description"], [0xDC00]);
                } else {
                    assert_eq!(body["output_config"]["effort"], "explicit");
                    assert!(strings.is_empty());
                }
                crate::exact_json::serialize(
                    &body,
                    &strings,
                    crate::exact_json::JsonEncoding::JavaScript,
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn resolved_effort_replacement_discards_previous_exact_string() {
        for kind in [AnthropicRequestKind::Main, AnthropicRequestKind::SideQuery] {
            let mut body = serde_json::json!({"messages":[],"output_config":{"effort":"high","format":{"description":"keep"}}});
            let mut exact = BTreeMap::from([
                ("/output_config/effort".into(), vec![0xD800]),
                ("/output_config/format/description".into(), vec![0xDC00]),
            ]);
            apply_value_policy(
                AnthropicRequestPolicy {
                    request_kind: kind,
                    effort: Some(AnthropicEffortPolicy {
                        supported: true,
                        value: Some(JsonPayload::string("low")),
                    }),
                    ..Default::default()
                },
                &mut body,
                &mut BTreeMap::new(),
                &mut exact,
                &mut "https://api.anthropic.com/v1/messages".into(),
            )
            .unwrap();
            assert_eq!(body["output_config"]["effort"], "low");
            assert!(!exact.contains_key("/output_config/effort"));
            assert_eq!(exact["/output_config/format/description"], [0xDC00]);
            let bytes = crate::exact_json::serialize(
                &body,
                &exact,
                crate::exact_json::JsonEncoding::JavaScript,
            )
            .unwrap();
            assert!(String::from_utf8(bytes)
                .unwrap()
                .contains("\"effort\":\"low\""));
        }
    }

    #[test]
    fn invalid_extra_body_does_not_mutate_body_headers_or_exact_strings() {
        let mut body = json!({"messages":[],"speed":"fast"});
        let mut headers = BTreeMap::from([("anthropic-beta".into(), "computed".into())]);
        let mut overrides = BTreeMap::from([("/speed".into(), vec![0xD800])]);
        let before = (body.clone(), headers.clone(), overrides.clone());
        let result = apply_value_policy(
            AnthropicRequestPolicy {
                extra_body: payload(&json!({"betas":{"toString":null}})),
                ..Default::default()
            },
            &mut body,
            &mut headers,
            &mut overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
        );
        assert!(matches!(result, Err(LlmError::InvalidRequest { .. })));
        assert_eq!((body, headers, overrides), before);
    }

    #[test]
    fn native_output_config_policy_matches_all_reference_bytes() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/output_config_wire_2_1_287.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut body = payload(&case["expected"]["base"]);
            let mut headers = BTreeMap::new();
            let mut overrides = BTreeMap::new();
            AnthropicRequestPolicy {
                request_kind: if case["input"]["kind"] == "main" {
                    AnthropicRequestKind::Main
                } else {
                    AnthropicRequestKind::SideQuery
                },
                extra_body: payload(&case["input"]["extra"]),
                effort: Some(AnthropicEffortPolicy {
                    supported: case["input"]["supported"].as_bool().unwrap(),
                    value: (!case["input"]["effort"].is_null())
                        .then(|| payload(&case["input"]["effort"])),
                }),
                ..Default::default()
            }
            .apply(
                &mut body,
                &mut headers,
                &mut overrides,
                &mut "https://api.anthropic.com/v1/messages".to_owned(),
            )
            .unwrap();
            assert_eq!(
                wire(&body, &overrides),
                case["expected"]["body_json"].as_str().unwrap(),
                "{:?}",
                case["input"]
            );
        }
    }

    #[test]
    fn explicit_replacements_remove_stale_exact_string_overrides() {
        for request_kind in [AnthropicRequestKind::Main, AnthropicRequestKind::SideQuery] {
            let mut body = json!({"messages":[{"content":"old"}],"output_config":{"format":{"schema":{"description":"old"}},"effort":"high"}});
            let mut overrides = BTreeMap::from([
                ("/messages/0/content".into(), vec![0xD800]),
                (
                    "/output_config/format/schema/description".into(),
                    vec![0xD801],
                ),
            ]);
            apply_value_policy(AnthropicRequestPolicy { request_kind, extra_body: payload(&json!({"messages":[{"content":"new"}],"output_config":{"format":{"schema":{"description":"new"}}}})), ..Default::default() }, &mut body, &mut BTreeMap::new(), &mut overrides, &mut "https://api.anthropic.com/v1/messages".to_owned()).unwrap();
            assert!(overrides.is_empty());
            assert!(String::from_utf8(
                crate::exact_json::serialize(
                    &body,
                    &overrides,
                    crate::exact_json::JsonEncoding::JavaScript
                )
                .unwrap()
            )
            .unwrap()
            .contains("new"));
        }
        let mut body = json!({"":{"content":"old"}});
        let mut overrides = BTreeMap::from([("//content".into(), vec![0xD800])]);
        apply_value_policy(
            AnthropicRequestPolicy {
                extra_body: payload(&json!({"content":"new"})),
                ..Default::default()
            },
            &mut body,
            &mut BTreeMap::new(),
            &mut overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
        )
        .unwrap();
        assert!(
            overrides.contains_key("//content"),
            "an empty top-level key remains untouched"
        );
        apply_value_policy(
            AnthropicRequestPolicy {
                extra_body: payload(&json!({"":{"content":"new"}})),
                ..Default::default()
            },
            &mut body,
            &mut BTreeMap::new(),
            &mut overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
        )
        .unwrap();
        assert!(overrides.is_empty());
    }

    #[test]
    fn fast_guard_clears_exact_overrides_for_removed_speed() {
        let mut body = payload(&json!({"speed":"fast","messages":[]}));
        let mut overrides = BTreeMap::from([("/speed".into(), vec![0xD800])]);
        remove_fast_source(&mut body, &mut BTreeMap::new(), &mut overrides, "fast").unwrap();
        assert!(overrides.is_empty());
        assert!(crate::exact_json::serialize_json_payload_with_overrides(
            &body,
            &overrides,
            crate::exact_json::JsonEncoding::JavaScript
        )
        .is_ok());
    }

    #[test]
    fn beta_merge_preserves_auth_order_and_normalizes_header_spelling() {
        let mut headers = BTreeMap::from([
            ("Anthropic-Beta".into(), "oauth, existing,oauth".into()),
            ("authorization".into(), "Bearer secret".into()),
        ]);
        merge_beta_header(&mut headers, &["existing,computed".into(), "custom".into()]);
        assert_eq!(headers["anthropic-beta"], "oauth,existing,computed,custom");
        assert!(!headers.contains_key("Anthropic-Beta"));
        assert_eq!(headers["authorization"], "Bearer secret");
    }

    #[test]
    fn identity_policy_preserves_auth_identity_or_replaces_all_spellings() {
        let mut headers = BTreeMap::from([("USER-AGENT".into(), "auth-agent".into())]);
        apply_user_agent(&mut headers, UserAgentPolicy::IfAbsent("host-agent"));
        assert_eq!(headers["USER-AGENT"], "auth-agent");
        apply_user_agent(&mut headers, UserAgentPolicy::Replace("host-agent"));
        assert_eq!(
            headers,
            BTreeMap::from([("user-agent".into(), "host-agent".into())])
        );
    }

    #[test]
    fn main_policy_preserves_explicit_output_config_and_computed_speed() {
        let mut body = json!({"speed":"fast", "output_config":{"effort":"high"}, "messages":[]});
        let mut headers = BTreeMap::new();
        let mut overrides = BTreeMap::new();
        apply_value_policy(
            AnthropicRequestPolicy {
                extra_body: payload(
                    &json!({"speed":"slow", "output_config":{"effort":"low","unknown":true},
                "anthropic_beta":["custom","first"], "provider_extension":{"foo":1}}),
                ),
                body_betas: vec!["first".into(), "second".into()],
                ..Default::default()
            },
            &mut body,
            &mut headers,
            &mut overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
        )
        .unwrap();
        assert!(overrides.is_empty());
        assert_eq!(body["speed"], "fast");
        assert_eq!(
            body["output_config"],
            json!({"effort":"low","unknown":true})
        );
        assert_eq!(body["anthropic_beta"], json!(["custom", "first", "second"]));
        assert_eq!(body["provider_extension"], json!({"foo":1}));
    }

    #[test]
    fn fast_guard_runs_after_extra_and_covers_case_insensitive_headers() {
        let mut body = json!({"messages":[]});
        let mut headers = BTreeMap::from([("Anthropic-Beta".into(), "oauth,fast,other".into())]);
        let mut overrides = BTreeMap::new();
        apply_value_policy(
            AnthropicRequestPolicy {
                extra_body: payload(&json!({"speed":"fast"})),
                disallowed_fast_beta: Some("fast".into()),
                ..Default::default()
            },
            &mut body,
            &mut headers,
            &mut overrides,
            &mut "https://api.anthropic.com/v1/messages".to_owned(),
        )
        .unwrap();
        assert!(overrides.is_empty());
        assert!(body.get("speed").is_none());
        assert_eq!(headers["Anthropic-Beta"], "oauth,other");
    }
}

#[cfg(test)]
mod native_fetch_policy_tests {
    use super::*;

    #[test]
    fn controlled_native_293_client_metadata_adds_only_absent_defaults() {
        let metadata = AnthropicClientMetadata::claude_code_2_1_293("darwin", "aarch64");
        let mut headers = BTreeMap::new();
        apply_client_metadata(&mut headers, &metadata);
        let expected: BTreeMap<String, String> = [
            ("accept", "application/json"),
            ("anthropic-dangerous-direct-browser-access", "true"),
            ("x-app", "cli"),
            ("x-stainless-arch", "arm64"),
            ("x-stainless-lang", "js"),
            ("x-stainless-os", "MacOS"),
            ("x-stainless-package-version", "0.128.0"),
            ("x-stainless-retry-count", "0"),
            ("x-stainless-runtime", "node"),
            ("x-stainless-runtime-version", "v26.3.0"),
            ("x-stainless-timeout", "600"),
        ]
        .into_iter()
        .map(|(name, value)| (name.into(), value.into()))
        .collect();
        assert_eq!(headers, expected);

        let mut caller_headers = BTreeMap::from([("X-STAINLESS-RETRY-COUNT".into(), "2".into())]);
        apply_client_metadata(&mut caller_headers, &metadata);
        assert_eq!(caller_headers["X-STAINLESS-RETRY-COUNT"], "2");
        assert_eq!(
            caller_headers
                .keys()
                .filter(|name| name.eq_ignore_ascii_case("x-stainless-retry-count"))
                .count(),
            1
        );
    }

    #[test]
    fn native_metadata_uses_captured_session_and_preserves_explicit_header_values() {
        let mut metadata = AnthropicClientMetadata::claude_code_2_1_293("darwin", "arm64");
        metadata.session_id = Some("captured-session".into());
        let mut headers = BTreeMap::new();
        apply_client_metadata(&mut headers, &metadata);
        assert_eq!(headers["x-claude-code-session-id"], "captured-session");

        headers.insert("Accept".into(), "custom/type".into());
        headers.remove("accept");
        headers.insert("X-Claude-Code-Session-Id".into(), "explicit-session".into());
        headers.remove("x-claude-code-session-id");
        headers.insert(
            "X-Stainless-Package-Version".into(),
            "caller-version".into(),
        );
        headers.remove("x-stainless-package-version");
        apply_client_metadata(&mut headers, &metadata);
        assert_eq!(headers["Accept"], "custom/type");
        assert_eq!(headers["X-Claude-Code-Session-Id"], "explicit-session");
        assert_eq!(headers["X-Stainless-Package-Version"], "caller-version");
    }

    #[test]
    fn native_sdk_platform_mapping_matches_supported_host_names() {
        let linux = AnthropicClientMetadata::claude_code_2_1_293("linux", "x86_64");
        assert_eq!((linux.os.as_str(), linux.arch.as_str()), ("Linux", "x64"));
        let windows = AnthropicClientMetadata::claude_code_2_1_293("win32", "arm64");
        assert_eq!(
            (windows.os.as_str(), windows.arch.as_str()),
            ("Windows", "arm64")
        );
    }

    #[test]
    fn native_fetch_header_spellings_are_selective_and_preserve_unknown_names() {
        for (source, expected) in [
            ("accept", "Accept"),
            ("USER-AGENT", "User-Agent"),
            ("authorization", "Authorization"),
            ("x-stainless-os", "X-Stainless-OS"),
            ("x-claude-code-session-id", "X-Claude-Code-Session-Id"),
            ("anthropic-beta", "anthropic-beta"),
            ("X-Custom-header", "X-Custom-header"),
        ] {
            assert_eq!(native_fetch_header_name(source), expected);
        }
    }
}

#[cfg(test)]
mod native_context_management_tests {
    use super::*;

    fn source(text: &str) -> JsonPayload {
        JsonPayload::parse(text).expect("test source is valid JSON")
    }

    #[test]
    fn thinking_clear_precedes_tool_clear_and_requires_admitted_beta() {
        let context = AnthropicContextManagement {
            has_thinking: true,
            tool_clearing: Some(source(r#"{"type":"clear_tool_uses_20250919"}"#)),
        };
        let expected = source(
            r#"{"edits":[{"type":"clear_thinking_20251015","keep":"all"},{"type":"clear_tool_uses_20250919"}]}"#,
        );
        assert_eq!(context.to_wire().unwrap(), Some(expected.clone()));

        for admitted in [false, true] {
            let mut body = source(r#"{"model":"claude-sonnet-5-5","messages":[],"stream":true}"#);
            let mut headers = BTreeMap::new();
            if admitted {
                headers.insert(
                    "anthropic-beta".into(),
                    "context-management-2025-06-27".into(),
                );
            }
            AnthropicRequestPolicy {
                context_management: Some(context.clone()),
                ..Default::default()
            }
            .apply(
                &mut body,
                &mut headers,
                &mut BTreeMap::new(),
                &mut "http://configured-base/v1/messages".into(),
            )
            .unwrap();
            assert_eq!(
                body.raw_source_at("/context_management"),
                admitted.then(|| expected.as_json_source().to_owned())
            );
        }
        assert_eq!(
            AnthropicContextManagement::default().to_wire().unwrap(),
            None
        );
    }

    #[test]
    fn explicit_context_management_spread_keeps_caller_authority() {
        let mut body = source(r#"{"messages":[]}"#);
        let mut headers = [(
            "anthropic-beta".into(),
            "context-management-2025-06-27".into(),
        )]
        .into_iter()
        .collect();
        AnthropicRequestPolicy {
            context_management: Some(AnthropicContextManagement {
                has_thinking: true,
                ..Default::default()
            }),
            extra_body: source(r#"{"context_management":{"edits":[]}}"#),
            ..Default::default()
        }
        .apply(
            &mut body,
            &mut headers,
            &mut BTreeMap::new(),
            &mut "http://configured-base/v1/messages".into(),
        )
        .unwrap();
        assert_eq!(
            body.raw_source_at("/context_management"),
            Some(r#"{"edits":[]}"#.to_owned())
        );
    }

    #[test]
    fn empty_admitted_context_policy_uses_the_template_without_adding_context() {
        let input = r#"{"extension":{"opaque":true},"messages":[],"model":"m"}"#;
        let mut baseline = source(input);
        let mut baseline_headers = BTreeMap::from([(
            "anthropic-beta".into(),
            "context-management-2025-06-27".into(),
        )]);
        AnthropicRequestPolicy::default()
            .apply(
                &mut baseline,
                &mut baseline_headers,
                &mut BTreeMap::new(),
                &mut "http://configured-base/v1/messages".into(),
            )
            .unwrap();

        let mut body = source(input);
        let mut headers = BTreeMap::from([(
            "anthropic-beta".into(),
            "context-management-2025-06-27".into(),
        )]);
        AnthropicRequestPolicy {
            context_management: Some(AnthropicContextManagement::default()),
            ..Default::default()
        }
        .apply(
            &mut body,
            &mut headers,
            &mut BTreeMap::new(),
            &mut "http://configured-base/v1/messages".into(),
        )
        .unwrap();
        assert_eq!(
            body.as_json_source(),
            r#"{"model":"m","messages":[],"extension":{"opaque":true}}"#
        );
        assert_eq!(body, baseline);
        assert_eq!(
            body.raw_source_at("/extension"),
            Some(r#"{"opaque":true}"#.to_owned())
        );
        assert_eq!(body.raw_source_at("/context_management"), None);
    }
}
