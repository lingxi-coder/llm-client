//! Gemini Developer API cached-content resource lifecycle.
//!
//! This service implements the documented `cachedContents` REST collection.
//! The endpoint, profile identity, account identity, and API key are supplied
//! explicitly; this service never derives a route from a chat profile.

#![doc = concat!(
    include_str!("../../docs/gemini-context-cache.md"),
    "\n\n",
    include_str!("../../docs/gemini-context-cache.en.md")
)]

use crate::{
    files::provider_file_endpoint_fingerprint,
    protocol::{LlmError, ProviderId, Secret},
    transport::{HttpExecutor, HttpRequest, HttpResponse, Transport},
};
use bytes::Bytes;
use chrono::DateTime;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::time::Duration;
use url::Url;

const COLLECTION_PATH: &str = "/v1beta/cachedContents";
const API_HOST: &str = "generativelanguage.googleapis.com";
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_PAGE_SIZE: u32 = 1000;

/// Identity for one explicit Gemini Developer API cached-content route.
///
/// `endpoint` is the full `https://generativelanguage.googleapis.com/v1beta/cachedContents`
/// collection URL. It is intentionally independent of a chat profile's
/// `base_url`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiContextCacheScope {
    provider_id: ProviderId,
    profile_name: String,
    account_scope: String,
    endpoint: String,
    endpoint_fingerprint: String,
}

impl GeminiContextCacheScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        endpoint: impl AsRef<str>,
    ) -> Result<Self, GeminiContextCacheError> {
        let profile_name = profile_name.into();
        let account_scope = account_scope.into();
        if !valid_identity(&profile_name) || !valid_identity(&account_scope) {
            return Err(invalid(
                "profile name and account scope must be non-empty and contain no control characters",
            ));
        }
        let endpoint = normalize_endpoint(endpoint.as_ref())?;
        let endpoint_fingerprint = provider_file_endpoint_fingerprint(&endpoint);
        Ok(Self {
            provider_id: ProviderId::from("google"),
            profile_name,
            account_scope,
            endpoint,
            endpoint_fingerprint,
        })
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    fn validate(&self) -> Result<(), GeminiContextCacheError> {
        let endpoint = normalize_endpoint(&self.endpoint)?;
        if self.provider_id.as_str() != "google"
            || !valid_identity(&self.profile_name)
            || !valid_identity(&self.account_scope)
            || endpoint != self.endpoint
            || self.endpoint_fingerprint != provider_file_endpoint_fingerprint(&self.endpoint)
        {
            return Err(invalid("Gemini context-cache scope identity is invalid"));
        }
        Ok(())
    }
}

/// An opaque cache resource reference bound to one route and account scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiContextCacheRef {
    scope: GeminiContextCacheScope,
    cache_id: String,
}

impl GeminiContextCacheRef {
    /// Bind Google's `cachedContents/{id}` resource name to an explicit scope.
    pub fn from_resource_name(
        scope: &GeminiContextCacheScope,
        resource_name: impl AsRef<str>,
    ) -> Result<Self, GeminiContextCacheError> {
        scope.validate()?;
        let cache_id = resource_name
            .as_ref()
            .strip_prefix("cachedContents/")
            .filter(|id| valid_resource_id(id))
            .ok_or_else(|| invalid("cache name must be a valid cachedContents/{id} resource"))?;
        Ok(Self {
            scope: scope.clone(),
            cache_id: cache_id.to_owned(),
        })
    }

    pub fn cache_id(&self) -> &str {
        &self.cache_id
    }

    pub fn resource_name(&self) -> String {
        format!("cachedContents/{}", self.cache_id)
    }

    pub fn scope(&self) -> &GeminiContextCacheScope {
        &self.scope
    }
}

/// TTL or absolute expiration accepted by Google's `CachedContent` resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeminiContextCacheExpiration {
    /// Protobuf duration in seconds, such as `3600s` or `3.5s`.
    Ttl(String),
    /// RFC 3339 timestamp, such as `2030-05-01T12:00:00Z`.
    ExpireTime(String),
}

impl GeminiContextCacheExpiration {
    fn validate(&self) -> Result<(), GeminiContextCacheError> {
        match self {
            Self::Ttl(value) if valid_ttl(value) => Ok(()),
            Self::ExpireTime(value) if DateTime::parse_from_rfc3339(value).is_ok() => Ok(()),
            Self::Ttl(_) => Err(invalid(
                "TTL must be a non-negative seconds duration with at most nine fractional digits",
            )),
            Self::ExpireTime(_) => Err(invalid("expire time must be an RFC 3339 timestamp")),
        }
    }

    fn insert_wire(&self, object: &mut Map<String, Value>) {
        match self {
            Self::Ttl(value) => {
                object.insert("ttl".into(), Value::String(value.clone()));
            }
            Self::ExpireTime(value) => {
                object.insert("expireTime".into(), Value::String(value.clone()));
            }
        }
    }

    fn update_mask(&self) -> &'static str {
        match self {
            Self::Ttl(_) => "ttl",
            Self::ExpireTime(_) => "expireTime",
        }
    }
}

/// Native, immutable fields for `cachedContents.create`.
///
/// Content, Tool, and ToolConfig values are retained as provider-native JSON
/// objects so their documented nested fields do not need a second codec here.
#[derive(Clone, PartialEq)]
pub struct GeminiContextCacheCreateRequest {
    model: String,
    display_name: Option<String>,
    contents: Option<Vec<Value>>,
    system_instruction: Option<Value>,
    tools: Option<Vec<Value>>,
    tool_config: Option<Value>,
    expiration: Option<GeminiContextCacheExpiration>,
}

impl std::fmt::Debug for GeminiContextCacheCreateRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiContextCacheCreateRequest")
            .field("model", &self.model)
            .field("display_name", &self.display_name)
            .field("contents_count", &self.contents.as_ref().map(Vec::len))
            .field("has_system_instruction", &self.system_instruction.is_some())
            .field("tools_count", &self.tools.as_ref().map(Vec::len))
            .field("has_tool_config", &self.tool_config.is_some())
            .field("expiration", &self.expiration)
            .finish()
    }
}

impl GeminiContextCacheCreateRequest {
    pub fn new(model: impl Into<String>) -> Result<Self, GeminiContextCacheError> {
        let model = model.into();
        if !valid_model_resource(&model) {
            return Err(invalid("model must be a models/{model} resource name"));
        }
        Ok(Self {
            model,
            display_name: None,
            contents: None,
            system_instruction: None,
            tools: None,
            tool_config: None,
            expiration: None,
        })
    }

    pub fn with_display_name(
        mut self,
        display_name: impl Into<String>,
    ) -> Result<Self, GeminiContextCacheError> {
        let display_name = display_name.into();
        if display_name.chars().count() > 128 {
            return Err(invalid(
                "display name may contain at most 128 Unicode characters",
            ));
        }
        self.display_name = Some(display_name);
        Ok(self)
    }

    pub fn with_contents(mut self, contents: Vec<Value>) -> Result<Self, GeminiContextCacheError> {
        validate_object_array(&contents, "contents")?;
        self.contents = Some(contents);
        Ok(self)
    }

    pub fn with_system_instruction(
        mut self,
        system_instruction: Value,
    ) -> Result<Self, GeminiContextCacheError> {
        if !is_text_content(&system_instruction) {
            return Err(invalid(
                "system instruction must be a Content object with text-only parts",
            ));
        }
        self.system_instruction = Some(system_instruction);
        Ok(self)
    }

    pub fn with_tools(mut self, tools: Vec<Value>) -> Result<Self, GeminiContextCacheError> {
        validate_object_array(&tools, "tools")?;
        self.tools = Some(tools);
        Ok(self)
    }

    pub fn with_tool_config(mut self, tool_config: Value) -> Result<Self, GeminiContextCacheError> {
        if !tool_config.is_object() {
            return Err(invalid("tool config must be a ToolConfig object"));
        }
        self.tool_config = Some(tool_config);
        Ok(self)
    }

    pub fn with_expiration(
        mut self,
        expiration: GeminiContextCacheExpiration,
    ) -> Result<Self, GeminiContextCacheError> {
        expiration.validate()?;
        self.expiration = Some(expiration);
        Ok(self)
    }

    fn encode(&self) -> Result<Bytes, GeminiContextCacheError> {
        let mut object = Map::new();
        object.insert("model".into(), Value::String(self.model.clone()));
        if let Some(value) = &self.display_name {
            object.insert("displayName".into(), Value::String(value.clone()));
        }
        if let Some(value) = &self.contents {
            object.insert("contents".into(), json!(value));
        }
        if let Some(value) = &self.system_instruction {
            object.insert("systemInstruction".into(), value.clone());
        }
        if let Some(value) = &self.tools {
            object.insert("tools".into(), json!(value));
        }
        if let Some(value) = &self.tool_config {
            object.insert("toolConfig".into(), value.clone());
        }
        if let Some(expiration) = &self.expiration {
            expiration.validate()?;
            expiration.insert_wire(&mut object);
        }
        serde_json::to_vec(&Value::Object(object))
            .map(Bytes::from)
            .map_err(|_| invalid("Gemini context-cache create request could not be encoded"))
    }
}

/// A metadata snapshot. Google returns resource metadata from get/list; the
/// original cached input is input-only and is not exposed as a retrievable body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiContextCache {
    pub reference: GeminiContextCacheRef,
    pub model: String,
    pub display_name: Option<String>,
    pub create_time: Option<String>,
    pub update_time: Option<String>,
    pub expire_time: Option<String>,
    pub usage_metadata: Option<Value>,
    pub native: Value,
}

/// One provider-paginated cached-content page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiContextCachePage {
    pub items: Vec<GeminiContextCache>,
    pub next_page_token: Option<String>,
    pub native: Value,
}

#[derive(Debug, thiserror::Error)]
pub enum GeminiContextCacheError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("Gemini context-cache API returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("invalid Gemini context-cache response: {message}")]
    InvalidResponse { message: String, native: Value },
    #[error("outcome of Gemini context-cache {operation} is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
    },
    #[error("outcome of Gemini context-cache {operation} is unknown: {message}")]
    OutcomeUnknownResponse {
        operation: &'static str,
        message: String,
    },
}

/// Direct client for Google's explicit cached-content resource API.
///
/// Mutating requests are sent once. A transport failure after dispatch is
/// returned as [`GeminiContextCacheError::OutcomeUnknown`]; callers decide
/// whether to reconcile with `get`/`list` or take another action.
pub struct GeminiContextCacheService<'a> {
    http: &'a dyn Transport,
    scope: GeminiContextCacheScope,
}

impl<'a> GeminiContextCacheService<'a> {
    pub fn new(
        http: &'a dyn Transport,
        scope: GeminiContextCacheScope,
    ) -> Result<Self, GeminiContextCacheError> {
        scope.validate()?;
        Ok(Self { http, scope })
    }

    pub fn scope(&self) -> &GeminiContextCacheScope {
        &self.scope
    }

    /// Create one immutable cache resource. If the request was accepted but
    /// the response could not be decoded, the returned error marks that result
    /// as uncertain; this method never retries.
    pub async fn create(
        &self,
        request: &GeminiContextCacheCreateRequest,
        credential: &Secret<String>,
    ) -> Result<GeminiContextCache, GeminiContextCacheError> {
        let body = request.encode()?;
        let response = self
            .send("POST", self.collection_url()?, Some(body), credential)
            .await
            .map_err(|source| outcome_unknown("create", source))?;
        let native = success_json(response).map_err(|error| match error {
            GeminiContextCacheError::InvalidResponse { message, .. } => {
                GeminiContextCacheError::OutcomeUnknownResponse {
                    operation: "create",
                    message,
                }
            }
            other => other,
        })?;
        self.decode_cache(native, None)
            .map_err(|error| match error {
                GeminiContextCacheError::InvalidResponse { message, .. } => {
                    GeminiContextCacheError::OutcomeUnknownResponse {
                        operation: "create",
                        message,
                    }
                }
                other => other,
            })
    }

    /// Fetch one page from `cachedContents.list`. Page tokens are returned
    /// unchanged for an explicit subsequent call.
    pub async fn list(
        &self,
        options: &GeminiContextCacheListOptions,
        credential: &Secret<String>,
    ) -> Result<GeminiContextCachePage, GeminiContextCacheError> {
        validate_page(options)?;
        let mut url = self.collection_url()?;
        {
            let mut query = url.query_pairs_mut();
            if let Some(page_size) = options.page_size {
                query.append_pair("pageSize", &page_size.to_string());
            }
            if let Some(page_token) = options.page_token.as_deref() {
                query.append_pair("pageToken", page_token);
            }
        }
        let response = self.send("GET", url, None, credential).await?;
        let native = success_json(response)?;
        let rows = native
            .get("cachedContents")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("list response omitted cachedContents", native.clone()))?;
        let items = rows
            .iter()
            .cloned()
            .map(|row| self.decode_cache(row, None))
            .collect::<Result<Vec<_>, _>>()?;
        let next_page_token = native
            .get("nextPageToken")
            .and_then(Value::as_str)
            .filter(|token| !token.is_empty())
            .map(str::to_owned);
        Ok(GeminiContextCachePage {
            items,
            next_page_token,
            native,
        })
    }

    pub async fn get(
        &self,
        reference: &GeminiContextCacheRef,
        credential: &Secret<String>,
    ) -> Result<GeminiContextCache, GeminiContextCacheError> {
        self.validate_reference(reference)?;
        let response = self
            .send("GET", self.resource_url(reference)?, None, credential)
            .await?;
        let native = success_json(response)?;
        self.decode_cache(native, Some(&reference.cache_id))
    }

    /// Change only the cache's expiration through Google's documented PATCH
    /// method. The update mask names the selected member of the expiration
    /// union (`ttl` or `expireTime`).
    pub async fn update_expiration(
        &self,
        reference: &GeminiContextCacheRef,
        expiration: GeminiContextCacheExpiration,
        credential: &Secret<String>,
    ) -> Result<GeminiContextCache, GeminiContextCacheError> {
        self.validate_reference(reference)?;
        expiration.validate()?;
        let mut url = self.resource_url(reference)?;
        url.query_pairs_mut()
            .append_pair("updateMask", expiration.update_mask());
        let mut body = Map::new();
        expiration.insert_wire(&mut body);
        let body = serde_json::to_vec(&Value::Object(body))
            .map(Bytes::from)
            .map_err(|_| invalid("Gemini context-cache update request could not be encoded"))?;
        let response = self
            .send("PATCH", url, Some(body), credential)
            .await
            .map_err(|source| outcome_unknown("update_expiration", source))?;
        let native = success_json(response).map_err(|error| match error {
            GeminiContextCacheError::InvalidResponse { message, .. } => {
                GeminiContextCacheError::OutcomeUnknownResponse {
                    operation: "update_expiration",
                    message,
                }
            }
            other => other,
        })?;
        self.decode_cache(native, Some(&reference.cache_id))
            .map_err(|error| match error {
                GeminiContextCacheError::InvalidResponse { message, .. } => {
                    GeminiContextCacheError::OutcomeUnknownResponse {
                        operation: "update_expiration",
                        message,
                    }
                }
                other => other,
            })
    }

    pub async fn delete(
        &self,
        reference: &GeminiContextCacheRef,
        credential: &Secret<String>,
    ) -> Result<(), GeminiContextCacheError> {
        self.validate_reference(reference)?;
        let response = self
            .send("DELETE", self.resource_url(reference)?, None, credential)
            .await
            .map_err(|source| outcome_unknown("delete", source))?;
        ensure_success(response)
    }

    async fn send(
        &self,
        method: &str,
        url: Url,
        body: Option<Bytes>,
        credential: &Secret<String>,
    ) -> Result<HttpResponse, LlmError> {
        if credential.expose_secret().trim().is_empty() {
            return Err(LlmError::Authentication {
                message: "Gemini API key must be non-empty".into(),
            });
        }
        let body = body.unwrap_or_default();
        let mut headers = vec![("x-goog-api-key".into(), credential.expose_secret().clone())];
        if !body.is_empty() {
            headers.push(("content-type".into(), "application/json".into()));
        }
        HttpExecutor::new(self.http)
            .execute_bounded(
                HttpRequest {
                    method: method.into(),
                    url: url.into(),
                    headers,
                    body,
                    timeout: Some(Duration::from_secs(120)),
                },
                MAX_RESPONSE_BYTES,
            )
            .await
    }

    fn collection_url(&self) -> Result<Url, GeminiContextCacheError> {
        Url::parse(&self.scope.endpoint)
            .map_err(|_| invalid("Gemini context-cache endpoint could not be parsed"))
    }

    fn resource_url(
        &self,
        reference: &GeminiContextCacheRef,
    ) -> Result<Url, GeminiContextCacheError> {
        let mut url = self.collection_url()?;
        url.path_segments_mut()
            .map_err(|_| invalid("Gemini context-cache URL cannot accept a resource name"))?
            .push(&reference.cache_id);
        Ok(url)
    }

    fn validate_reference(
        &self,
        reference: &GeminiContextCacheRef,
    ) -> Result<(), GeminiContextCacheError> {
        if reference.scope != self.scope || !valid_resource_id(&reference.cache_id) {
            return Err(LlmError::PermissionDenied {
                message: "Gemini cached content belongs to another endpoint, profile, or account"
                    .into(),
            }
            .into());
        }
        Ok(())
    }

    fn decode_cache(
        &self,
        native: Value,
        expected_id: Option<&str>,
    ) -> Result<GeminiContextCache, GeminiContextCacheError> {
        let name = native
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("cache response omitted resource name", native.clone()))?;
        let cache_id = name
            .strip_prefix("cachedContents/")
            .filter(|id| valid_resource_id(id))
            .ok_or_else(|| {
                bad(
                    "cache response name is not a cachedContents resource",
                    native.clone(),
                )
            })?;
        if expected_id.is_some_and(|expected| expected != cache_id) {
            return Err(bad(
                "cache response name differs from the requested resource",
                native,
            ));
        }
        let model = native
            .get("model")
            .and_then(Value::as_str)
            .filter(|value| valid_model_resource(value))
            .ok_or_else(|| {
                bad(
                    "cache response omitted a valid model resource",
                    native.clone(),
                )
            })?
            .to_owned();
        Ok(GeminiContextCache {
            reference: GeminiContextCacheRef {
                scope: self.scope.clone(),
                cache_id: cache_id.into(),
            },
            model,
            display_name: string_field(&native, "displayName"),
            create_time: string_field(&native, "createTime"),
            update_time: string_field(&native, "updateTime"),
            expire_time: string_field(&native, "expireTime"),
            usage_metadata: native.get("usageMetadata").cloned(),
            native,
        })
    }
}

/// Parameters for one explicit `cachedContents.list` call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GeminiContextCacheListOptions {
    pub page_size: Option<u32>,
    pub page_token: Option<String>,
}

impl GeminiContextCacheListOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_page_size(mut self, page_size: u32) -> Result<Self, GeminiContextCacheError> {
        if !(1..=MAX_PAGE_SIZE).contains(&page_size) {
            return Err(invalid("page size must be between 1 and 1000"));
        }
        self.page_size = Some(page_size);
        Ok(self)
    }

    pub fn with_page_token(
        mut self,
        page_token: impl Into<String>,
    ) -> Result<Self, GeminiContextCacheError> {
        let page_token = page_token.into();
        if page_token.trim().is_empty() || page_token.chars().any(char::is_control) {
            return Err(invalid(
                "page token must be non-empty and contain no controls",
            ));
        }
        self.page_token = Some(page_token);
        Ok(self)
    }
}

fn validate_page(options: &GeminiContextCacheListOptions) -> Result<(), GeminiContextCacheError> {
    if options
        .page_size
        .is_some_and(|size| !(1..=MAX_PAGE_SIZE).contains(&size))
    {
        return Err(invalid("page size must be between 1 and 1000"));
    }
    if options
        .page_token
        .as_deref()
        .is_some_and(|token| token.trim().is_empty() || token.chars().any(char::is_control))
    {
        return Err(invalid(
            "page token must be non-empty and contain no controls",
        ));
    }
    Ok(())
}

fn normalize_endpoint(endpoint: &str) -> Result<String, GeminiContextCacheError> {
    let mut url = Url::parse(endpoint)
        .map_err(|_| invalid("Gemini context-cache endpoint must be an absolute HTTPS URL"))?;
    if url.scheme() != "https"
        || url.host_str() != Some(API_HOST)
        || url.port().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path().trim_end_matches('/') != COLLECTION_PATH
    {
        return Err(invalid(
            "endpoint must be Google's HTTPS /v1beta/cachedContents collection URL without query or fragment",
        ));
    }
    url.set_path(COLLECTION_PATH);
    Ok(url.into())
}

fn valid_identity(value: &str) -> bool {
    !value.trim().is_empty() && !value.chars().any(char::is_control)
}

fn valid_resource_id(value: &str) -> bool {
    !value.is_empty() && !value.contains('/') && !value.chars().any(char::is_control)
}

fn valid_model_resource(value: &str) -> bool {
    value
        .strip_prefix("models/")
        .is_some_and(|id| !id.is_empty() && !id.contains('/') && !id.chars().any(char::is_control))
}

fn valid_ttl(value: &str) -> bool {
    let Some(number) = value.strip_suffix('s') else {
        return false;
    };
    let Some((whole, fraction)) = number.split_once('.') else {
        return !number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit());
    };
    !whole.is_empty()
        && whole.bytes().all(|byte| byte.is_ascii_digit())
        && (1..=9).contains(&fraction.len())
        && fraction.bytes().all(|byte| byte.is_ascii_digit())
}

fn validate_object_array(values: &[Value], field: &str) -> Result<(), GeminiContextCacheError> {
    if values.iter().any(|value| !value.is_object()) {
        return Err(invalid(&format!("{field} must contain only JSON objects")));
    }
    Ok(())
}

fn is_text_content(content: &Value) -> bool {
    let Some(parts) = content.get("parts").and_then(Value::as_array) else {
        return false;
    };
    !parts.is_empty()
        && parts.iter().all(|part| {
            part.as_object().is_some_and(|object| {
                object.get("text").and_then(Value::as_str).is_some()
                    && object.keys().all(|key| key == "text")
            })
        })
}

fn string_field(native: &Value, name: &str) -> Option<String> {
    native.get(name).and_then(Value::as_str).map(str::to_owned)
}

fn success_json(response: HttpResponse) -> Result<Value, GeminiContextCacheError> {
    let native = parse_body(&response.body);
    if !(200..300).contains(&response.status) {
        return Err(provider_error(response.status, &response.headers, native));
    }
    if !native.is_object() {
        return Err(bad("success response is not a JSON object", native));
    }
    Ok(native)
}

fn ensure_success(response: HttpResponse) -> Result<(), GeminiContextCacheError> {
    if !(200..300).contains(&response.status) {
        return Err(provider_error(
            response.status,
            &response.headers,
            parse_body(&response.body),
        ));
    }
    Ok(())
}

fn provider_error(
    status: u16,
    headers: &[(String, String)],
    body: Value,
) -> GeminiContextCacheError {
    GeminiContextCacheError::Provider {
        status,
        request_id: headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("x-goog-request-id"))
            .map(|(_, value)| value.clone()),
        body,
    }
}

fn parse_body(body: &[u8]) -> Value {
    serde_json::from_slice(body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(body).into_owned()))
}

fn bad(message: &str, native: Value) -> GeminiContextCacheError {
    GeminiContextCacheError::InvalidResponse {
        message: message.into(),
        native,
    }
}

fn invalid(message: &str) -> GeminiContextCacheError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
    .into()
}

fn outcome_unknown(operation: &'static str, source: LlmError) -> GeminiContextCacheError {
    if matches!(
        &source,
        LlmError::Transport { .. }
            | LlmError::TransportTimeout { .. }
            | LlmError::StreamInterrupted { .. }
    ) {
        GeminiContextCacheError::OutcomeUnknown { operation, source }
    } else {
        GeminiContextCacheError::Llm(source)
    }
}
