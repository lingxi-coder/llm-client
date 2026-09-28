//! Gemini Interactions API: independent model/agent turns and stored references.
mod stream;
use crate::{
    client::RequestOptions,
    files::provider_file_endpoint_fingerprint,
    protocol::{LlmError, ProviderProfile, ServiceAuth, ServiceSetting},
    runtime::Deadline,
    runtime::{ClientSnapshot, ClientSource},
    transport::{HttpExecutor, HttpRequest, HttpResponse},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
pub use stream::{InteractionEvent, InteractionEventStream, InteractionStreamError};

const MAX_RESPONSE: usize = 16 * 1024 * 1024;
// Google recommends the Files API when a request exceeds 100 MB. Keep the
// encoded request at or below that boundary, including base64 expansion.
const MAX_REQUEST_BODY: usize = 100 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionRoute {
    /// Full `/v1beta/interactions` collection endpoint.
    pub endpoint: String,
    pub auth: ServiceAuth,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "id")]
pub enum InteractionTarget {
    Model(String),
    Agent(String),
}

/// The input forms accepted by Google's Interactions API.
///
/// `Text` preserves the compact string form. `Content` carries multimodal
/// content blocks, while `Steps` carries an interaction timeline such as a
/// function result or a caller-managed stateless history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InteractionInput {
    Text(String),
    ContentBlock(InteractionContent),
    Content(Vec<InteractionContent>),
    Steps(Vec<Value>),
}

impl InteractionInput {
    pub fn content(content: impl IntoIterator<Item = InteractionContent>) -> Self {
        Self::Content(content.into_iter().collect())
    }

    pub fn content_block(content: InteractionContent) -> Self {
        Self::ContentBlock(content)
    }

    /// Supply native Interactions API steps, including `function_result`
    /// steps or model steps copied unchanged from a prior response.
    pub fn steps(steps: impl IntoIterator<Item = Value>) -> Self {
        Self::Steps(steps.into_iter().collect())
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::ContentBlock(_) => false,
            Self::Content(content) => content.is_empty(),
            Self::Steps(steps) => steps.is_empty(),
        }
    }
}

impl From<String> for InteractionInput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for InteractionInput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<Vec<InteractionContent>> for InteractionInput {
    fn from(value: Vec<InteractionContent>) -> Self {
        Self::Content(value)
    }
}

impl From<Vec<Value>> for InteractionInput {
    fn from(value: Vec<Value>) -> Self {
        Self::Steps(value)
    }
}

/// A native Interactions API content block for multimodal input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum InteractionContent {
    Text {
        text: String,
    },
    Image {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resolution: Option<String>,
    },
    Audio {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        channels: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sample_rate: Option<u32>,
    },
    Document {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
    Video {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resolution: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        processing: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
}

impl InteractionContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image_data(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Image {
            data: Some(data.into()),
            uri: None,
            mime_type: Some(mime_type.into()),
            resolution: None,
        }
    }

    pub fn image_uri(uri: impl Into<String>, mime_type: Option<String>) -> Self {
        Self::Image {
            data: None,
            uri: Some(uri.into()),
            mime_type,
            resolution: None,
        }
    }

    pub fn audio_data(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Audio {
            data: Some(data.into()),
            uri: None,
            mime_type: Some(mime_type.into()),
            channels: None,
            sample_rate: None,
        }
    }

    pub fn audio_uri(uri: impl Into<String>, mime_type: Option<String>) -> Self {
        Self::Audio {
            data: None,
            uri: Some(uri.into()),
            mime_type,
            channels: None,
            sample_rate: None,
        }
    }

    pub fn document_data(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Document {
            data: Some(data.into()),
            uri: None,
            mime_type: Some(mime_type.into()),
        }
    }

    pub fn document_uri(uri: impl Into<String>, mime_type: Option<String>) -> Self {
        Self::Document {
            data: None,
            uri: Some(uri.into()),
            mime_type,
        }
    }

    pub fn video_uri(uri: impl Into<String>) -> Self {
        Self::Video {
            data: None,
            uri: Some(uri.into()),
            mime_type: None,
            resolution: None,
            processing: None,
            name: None,
        }
    }
}

/// A provider-native Interactions API tool declaration.
///
/// The transparent JSON representation keeps built-in tool declarations
/// forward-compatible. Use [`InteractionTool::function`] for the common
/// client-executed function tool shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InteractionTool(Value);

impl InteractionTool {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        let name = name.into();
        let description = description.into();
        Self(json!({
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters,
        }))
    }

    pub fn google_search() -> Self {
        Self(json!({ "type": "google_search" }))
    }

    pub fn url_context() -> Self {
        Self(json!({ "type": "url_context" }))
    }

    pub fn code_execution() -> Self {
        Self(json!({ "type": "code_execution" }))
    }

    /// Create a tool from a native JSON declaration, such as Google Maps,
    /// File Search, or an MCP server tool.
    pub fn from_value(value: Value) -> Result<Self, LlmError> {
        if value
            .as_object()
            .and_then(|object| object.get("type"))
            .and_then(Value::as_str)
            .is_some_and(|kind| !kind.is_empty())
        {
            Ok(Self(value))
        } else {
            Err(invalid("an interaction tool must be an object with a type"))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionRequest {
    pub target: InteractionTarget,
    pub input: InteractionInput,
    #[serde(default)]
    pub previous: Option<InteractionRef>,
    /// Stored interactions can be retrieved and used as subsequent context.
    #[serde(default = "default_store")]
    pub store: bool,
    #[serde(default)]
    pub background: bool,
    #[serde(default)]
    pub tools: Vec<InteractionTool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<Value>,
}
fn default_store() -> bool {
    true
}
impl InteractionRequest {
    pub fn model(model: impl Into<String>, input: impl Into<InteractionInput>) -> Self {
        Self {
            target: InteractionTarget::Model(model.into()),
            input: input.into(),
            previous: None,
            store: true,
            background: false,
            tools: Vec::new(),
            generation_config: None,
        }
    }
    pub fn agent(agent: impl Into<String>, input: impl Into<InteractionInput>) -> Self {
        Self {
            target: InteractionTarget::Agent(agent.into()),
            input: input.into(),
            previous: None,
            store: true,
            background: true,
            tools: Vec::new(),
            generation_config: None,
        }
    }

    pub fn with_tool(mut self, tool: InteractionTool) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn with_generation_config(mut self, generation_config: Value) -> Self {
        self.generation_config = Some(generation_config);
        self
    }
}

/// A stored interaction scoped to one provider profile, endpoint, and account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionRef {
    pub provider_id: String,
    pub profile_name: String,
    pub endpoint_fingerprint: String,
    pub account_scope: String,
    pub id: String,
}

#[derive(Debug, Clone)]
pub struct InteractionResult {
    pub id: String,
    pub status: String,
    pub reference: Option<InteractionRef>,
    pub native: Value,
    pub request_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum InteractionError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("Gemini Interactions returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("invalid Gemini interaction: {message}")]
    InvalidResponse { message: String, native: Value },
    #[error("interaction submission outcome is unknown: {source}")]
    SubmitOutcomeUnknown {
        #[source]
        source: LlmError,
    },
}

#[derive(Clone, Copy)]
pub struct InteractionService<'a> {
    source: ClientSource<'a>,
}
impl<'a> InteractionService<'a> {
    pub(crate) fn new(source: ClientSource<'a>) -> Self {
        Self { source }
    }

    /// Create a model or managed-agent interaction. No automatic retry/failover.
    pub async fn create(
        self,
        request: &InteractionRequest,
        options: &RequestOptions,
    ) -> Result<InteractionResult, InteractionError> {
        let profile_name = self.source.profile_name()?;
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .create(profile_name, request, options)
            .await
    }

    /// Retrieve a stored interaction by its account-bound reference.
    pub async fn get(
        self,
        reference: &InteractionRef,
        options: &RequestOptions,
    ) -> Result<InteractionResult, InteractionError> {
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }.get(reference, options).await
    }

    /// Cancel a stored interaction that is still running in the background.
    /// Google only supports cancellation for background interactions.
    pub async fn cancel(
        self,
        reference: &InteractionRef,
        options: &RequestOptions,
    ) -> Result<InteractionResult, InteractionError> {
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .cancel(reference, options)
            .await
    }

    /// Delete a stored interaction and its server-side record.
    pub async fn delete(
        self,
        reference: &InteractionRef,
        options: &RequestOptions,
    ) -> Result<(), InteractionError> {
        let snapshot = self.source.pin()?;
        Pinned { client: &snapshot }
            .delete(reference, options)
            .await
    }
}

struct Pinned<'a> {
    client: &'a ClientSnapshot,
}
impl Pinned<'_> {
    fn route<'s>(
        &'s self,
        profile_name: &str,
        options: &'s RequestOptions,
    ) -> Result<(&'s ProviderProfile, &'s InteractionRoute, &'s str), InteractionError> {
        let profile = self
            .client
            .native_profile(profile_name)
            .ok_or_else(|| invalid("unknown interaction profile"))?;
        if !profile.supports_region(self.client.region()) {
            return Err(invalid("interaction profile is unavailable in this region").into());
        }
        let ServiceSetting::Enabled(route) = &profile.interactions else {
            return Err(LlmError::UnsupportedCapability {
                message: "profile has no interactions route".into(),
            }
            .into());
        };
        let scope = options
            .account_scope
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| invalid("interactions require a non-secret account_scope"))?;
        Ok((profile, route, scope))
    }

    async fn create(
        &self,
        profile_name: &str,
        request: &InteractionRequest,
        options: &RequestOptions,
    ) -> Result<InteractionResult, InteractionError> {
        let (profile, route, scope) = self.route(profile_name, options)?;
        if let Some(previous) = &request.previous {
            check_reference(profile, route, scope, previous)?;
        }
        let body = encode_create_body(request, false)?;
        let response = self
            .send(route, options, "POST", route.endpoint.clone(), body, true)
            .await?;
        decode(profile, route, scope, response, request.store)
    }

    async fn get(
        &self,
        reference: &InteractionRef,
        options: &RequestOptions,
    ) -> Result<InteractionResult, InteractionError> {
        let (profile, route, scope) = self.route(&reference.profile_name, options)?;
        check_reference(profile, route, scope, reference)?;
        let mut url = url::Url::parse(&route.endpoint)
            .map_err(|_| invalid("invalid interactions endpoint"))?;
        url.path_segments_mut()
            .map_err(|_| invalid("invalid interactions endpoint"))?
            .push(&reference.id);
        let response = self
            .send(route, options, "GET", url.into(), vec![], false)
            .await?;
        let result = decode(profile, route, scope, response, true)?;
        if result.id != reference.id {
            return Err(InteractionError::InvalidResponse {
                message: "interaction id differs from reference".into(),
                native: result.native,
            });
        }
        Ok(result)
    }

    async fn cancel(
        &self,
        reference: &InteractionRef,
        options: &RequestOptions,
    ) -> Result<InteractionResult, InteractionError> {
        let (profile, route, scope) = self.route(&reference.profile_name, options)?;
        check_reference(profile, route, scope, reference)?;
        let mut url = interaction_url(route, reference)?;
        url.path_segments_mut()
            .map_err(|_| invalid("invalid interactions endpoint"))?
            .push("cancel");
        let response = self
            .send(route, options, "POST", url.into(), vec![], false)
            .await?;
        let result = decode(profile, route, scope, response, true)?;
        ensure_same_interaction(reference, result)
    }

    async fn delete(
        &self,
        reference: &InteractionRef,
        options: &RequestOptions,
    ) -> Result<(), InteractionError> {
        let (profile, route, scope) = self.route(&reference.profile_name, options)?;
        check_reference(profile, route, scope, reference)?;
        let url = interaction_url(route, reference)?;
        let response = self
            .send(route, options, "DELETE", url.into(), vec![], false)
            .await?;
        if !(200..300).contains(&response.status) {
            return Err(provider_error(response));
        }
        Ok(())
    }

    async fn send(
        &self,
        route: &InteractionRoute,
        options: &RequestOptions,
        method: &str,
        url: String,
        body: Vec<u8>,
        submitting: bool,
    ) -> Result<HttpResponse, InteractionError> {
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "Gemini Interactions requires a credential".into(),
            })?;
        let ServiceAuth::ApiKey { header } = &route.auth else {
            return Err(invalid("Interactions requires API key authentication").into());
        };
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        let request = HttpRequest {
            method: method.into(),
            url,
            headers: vec![
                (header.clone(), credential.expose_secret().into()),
                ("content-type".into(), "application/json".into()),
            ],
            body: body.into(),
            timeout: deadline.remaining()?,
        };
        HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(request, MAX_RESPONSE)
            .await
            .map_err(|source| {
                if submitting
                    && matches!(
                        source,
                        LlmError::Transport { .. }
                            | LlmError::TransportTimeout { .. }
                            | LlmError::StreamInterrupted { .. }
                    )
                {
                    InteractionError::SubmitOutcomeUnknown { source }
                } else {
                    InteractionError::Llm(source)
                }
            })
    }
}

fn decode(
    profile: &ProviderProfile,
    route: &InteractionRoute,
    scope: &str,
    response: HttpResponse,
    stored: bool,
) -> Result<InteractionResult, InteractionError> {
    let request_id = response
        .header("x-request-id")
        .or_else(|| response.header("request-id"))
        .map(str::to_owned);
    let native: Value = serde_json::from_slice(&response.body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&response.body).into_owned()));
    if !(200..300).contains(&response.status) {
        return Err(InteractionError::Provider {
            status: response.status,
            request_id,
            body: native,
        });
    }
    let id = native
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| valid_id(id))
        .ok_or_else(|| InteractionError::InvalidResponse {
            message: "missing interaction id".into(),
            native: native.clone(),
        })?
        .to_owned();
    let status = native
        .get("status")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| InteractionError::InvalidResponse {
            message: "missing interaction status".into(),
            native: native.clone(),
        })?
        .to_owned();
    let reference = stored.then(|| InteractionRef {
        provider_id: profile.provider_id.as_str().into(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&route.endpoint),
        account_scope: scope.into(),
        id: id.clone(),
    });
    Ok(InteractionResult {
        id,
        status,
        reference,
        native,
        request_id,
    })
}

fn interaction_url(
    route: &InteractionRoute,
    reference: &InteractionRef,
) -> Result<url::Url, InteractionError> {
    let mut url =
        url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid interactions endpoint"))?;
    url.path_segments_mut()
        .map_err(|_| invalid("invalid interactions endpoint"))?
        .push(&reference.id);
    Ok(url)
}

fn ensure_same_interaction(
    reference: &InteractionRef,
    result: InteractionResult,
) -> Result<InteractionResult, InteractionError> {
    if result.id != reference.id {
        return Err(InteractionError::InvalidResponse {
            message: "interaction id differs from reference".into(),
            native: result.native,
        });
    }
    Ok(result)
}

fn provider_error(response: HttpResponse) -> InteractionError {
    let request_id = response
        .header("x-request-id")
        .or_else(|| response.header("request-id"))
        .map(str::to_owned);
    let body = serde_json::from_slice(&response.body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&response.body).into_owned()));
    InteractionError::Provider {
        status: response.status,
        request_id,
        body,
    }
}

fn check_reference(
    profile: &ProviderProfile,
    route: &InteractionRoute,
    scope: &str,
    reference: &InteractionRef,
) -> Result<(), InteractionError> {
    if reference.provider_id != profile.provider_id.as_str()
        || reference.profile_name != profile.profile_name
        || reference.endpoint_fingerprint != provider_file_endpoint_fingerprint(&route.endpoint)
        || reference.account_scope != scope
        || !valid_id(&reference.id)
    {
        return Err(LlmError::PermissionDenied {
            message:
                "interaction reference belongs to another provider, profile, endpoint or account"
                    .into(),
        }
        .into());
    }
    Ok(())
}
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 256
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}
fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}

pub(super) fn encode_create_body(
    request: &InteractionRequest,
    stream: bool,
) -> Result<Vec<u8>, InteractionError> {
    if request.input.is_empty() {
        return Err(invalid("interaction input cannot be empty").into());
    }
    if !request.store && request.background {
        return Err(invalid("background interactions require store=true").into());
    }
    let (target_field, target_id) = match &request.target {
        InteractionTarget::Model(id) => ("model", id.as_str()),
        InteractionTarget::Agent(id) => ("agent", id.as_str()),
    };
    if target_id.is_empty()
        || target_id.len() > 128
        || !target_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        return Err(invalid("invalid interaction model or agent id").into());
    }
    if matches!(request.target, InteractionTarget::Agent(_)) && !request.background {
        return Err(invalid("managed agent interactions require background=true").into());
    }
    if request.tools.iter().any(|tool| {
        tool.0
            .as_object()
            .and_then(|object| object.get("type"))
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
    }) {
        return Err(invalid("every interaction tool must be an object with a type").into());
    }
    // The Interactions API currently excludes remote MCP for Gemini 3 models.
    // Keep agent and other model targets distinct: the published restriction
    // does not establish their support either way.
    if matches!(request.target, InteractionTarget::Model(_))
        && target_id.starts_with("gemini-3")
        && request
            .tools
            .iter()
            .any(|tool| tool.0.get("type").and_then(Value::as_str) == Some("mcp_server"))
    {
        return Err(invalid("Gemini 3 Interactions do not support remote MCP").into());
    }

    let input = serde_json::to_value(&request.input)
        .map_err(|_| invalid("interaction input cannot be serialized"))?;
    let mut body = json!({
        "input": input,
        "store": request.store,
        "background": request.background,
        "stream": stream,
    });
    body[target_field] = Value::String(target_id.into());
    if let Some(previous) = &request.previous {
        body["previous_interaction_id"] = Value::String(previous.id.clone());
    }
    if !request.tools.is_empty() {
        body["tools"] = serde_json::to_value(&request.tools)
            .map_err(|_| invalid("interaction tools cannot be serialized"))?;
    }
    if let Some(generation_config) = &request.generation_config {
        body["generation_config"] = generation_config.clone();
    }
    let body =
        serde_json::to_vec(&body).map_err(|_| invalid("interaction body cannot be serialized"))?;
    if body.len() > MAX_REQUEST_BODY {
        return Err(invalid("encoded interaction request exceeds 100 MiB").into());
    }
    Ok(body)
}

pub fn validate_route(profile: &ProviderProfile, route: &InteractionRoute) -> Result<(), LlmError> {
    if profile.provider_id.as_str() != "google"
        || route.auth
            != (ServiceAuth::ApiKey {
                header: "x-goog-api-key".into(),
            })
    {
        return Err(invalid(
            "Gemini Interactions requires Google identity and x-goog-api-key authentication",
        ));
    }
    let url =
        url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid interactions endpoint"))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !url.path().ends_with("/v1beta/interactions")
    {
        return Err(invalid(
            "interactions endpoint must be a full v1beta collection URL",
        ));
    }
    Ok(())
}
