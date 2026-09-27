//! Scoped lifecycle operations for Anthropic Skills.
//!
//! Upload inputs are supplied as in-memory or caller-owned streams. This
//! module never reads local paths, extracts archives, executes Skill content,
//! retries mutations, or writes downloaded content to disk.

use crate::{
    auth::Authenticator,
    files::{exact_upload_stream, multipart_boundary},
    protocol::{
        AnthropicSkillRef, AnthropicSkillScope, AuthStrategy, FoundryHosting, LlmError,
        ProviderProfile, Secret,
    },
    transport::{
        HttpExecutor, HttpRequest, HttpResponse, HttpStreamRequest, StreamResponse, Transport,
    },
};
use bytes::{Bytes, BytesMut};
use futures::{stream, stream::BoxStream, Stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

const API_VERSION: &str = "2023-06-01";
const SKILLS_PATH: &str = "/v1/skills";
const MAX_UPLOAD_BYTES: u64 = 30_000_000;
const MAX_CONTROL_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const MAX_CONTENT_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_OPERATION_TIMEOUT: Duration = Duration::from_secs(120);

/// A caller-owned Skill file. The stream is consumed once during create or
/// version creation. `relative_path` is encoded as the multipart filename so
/// the provider receives the Skill directory layout.
pub struct AnthropicSkillFile {
    relative_path: String,
    size_bytes: u64,
    body: BoxStream<'static, Result<Bytes, LlmError>>,
}

impl AnthropicSkillFile {
    /// Create a one-shot streamed file input with a declared exact byte size.
    pub fn new<S>(relative_path: impl Into<String>, size_bytes: u64, body: S) -> Self
    where
        S: Stream<Item = Result<Bytes, LlmError>> + Send + 'static,
    {
        Self {
            relative_path: relative_path.into(),
            size_bytes,
            body: body.boxed(),
        }
    }

    /// Build a one-chunk file input from bytes already held by the caller.
    pub fn from_bytes(relative_path: impl Into<String>, bytes: impl Into<Bytes>) -> Self {
        let bytes = bytes.into();
        let size_bytes = bytes.len() as u64;
        Self::new(
            relative_path,
            size_bytes,
            stream::once(async move { Ok(bytes) }),
        )
    }

    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    fn into_parts(self) -> (String, u64, BoxStream<'static, Result<Bytes, LlmError>>) {
        (self.relative_path, self.size_bytes, self.body)
    }
}

/// Optional list filters supported by the Skills endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicSkillSourceFilter {
    Custom,
    Anthropic,
}

/// Resource identity for any returned Skill source, including plugin and
/// example Skills that cannot be projected into `container.skills`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicSkillResourceRef {
    skill_id: String,
    source_type: String,
    scope: AnthropicSkillScope,
}

impl AnthropicSkillResourceRef {
    pub fn new(
        skill_id: impl Into<String>,
        source_type: impl Into<String>,
        scope: AnthropicSkillScope,
    ) -> Result<Self, AnthropicSkillsError> {
        let skill_id = skill_id.into();
        let source_type = source_type.into();
        validate_identifier(&skill_id, "skill ID")?;
        validate_identifier(&source_type, "source type")?;
        scope.validate()?;
        Ok(Self {
            skill_id,
            source_type,
            scope,
        })
    }

    pub fn skill_id(&self) -> &str {
        &self.skill_id
    }

    pub fn source_type(&self) -> &str {
        &self.source_type
    }

    pub fn scope(&self) -> &AnthropicSkillScope {
        &self.scope
    }

    /// Project only executable source types accepted by the Messages
    /// `container.skills` contract.
    pub fn messages_reference(&self) -> Option<AnthropicSkillRef> {
        match self.source_type.as_str() {
            "custom" => Some(AnthropicSkillRef::custom(
                self.skill_id.clone(),
                self.scope.clone(),
            )),
            "anthropic" => Some(AnthropicSkillRef::anthropic(self.skill_id.clone())),
            _ => None,
        }
    }
}

/// One cursor page of Skills. Source strings and the full original object are
/// preserved so newer source types such as `plugin` remain visible.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicSkillPage {
    pub skills: Vec<AnthropicSkill>,
    pub next_page: Option<String>,
    pub native: Value,
}

/// A provider Skill record. Only `custom` and `anthropic` source types can be
/// converted to the currently documented Messages `container.skills` form.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicSkill {
    pub id: String,
    pub created_at: String,
    pub display_name: String,
    pub latest_version_id: String,
    pub updated_at: String,
    pub source_type: String,
    pub reference: AnthropicSkillResourceRef,
    pub native: Value,
}

impl AnthropicSkill {
    /// Preserve unsupported/new source kinds as native data instead of
    /// pretending they can be attached to a Messages container.
    pub fn messages_reference(&self) -> Option<AnthropicSkillRef> {
        self.reference.messages_reference()
    }
}

/// One cursor page of versions belonging to a scoped Skill.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicSkillVersionPage {
    pub versions: Vec<AnthropicSkillVersion>,
    pub next_page: Option<String>,
    pub native: Value,
}

/// Provider Skill version metadata plus the scoped Skill reference it belongs
/// to. The native response remains available for fields added by Anthropic.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicSkillVersion {
    pub id: String,
    pub skill_id: String,
    pub created_at: String,
    pub description: String,
    pub name: String,
    pub reference: AnthropicSkillResourceRef,
    pub native: Value,
}

impl AnthropicSkillVersion {
    pub fn pinned_skill(&self) -> Option<AnthropicSkillRef> {
        Some(
            self.reference
                .messages_reference()?
                .with_version(self.id.clone()),
        )
    }
}

/// Successful Skill deletion result.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicSkillDeleted {
    pub id: String,
    pub native: Value,
}

/// Successful Skill-version deletion result.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicSkillVersionDeleted {
    pub id: String,
    pub native: Value,
}

/// Cursor and source options for `list`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnthropicSkillListOptions {
    pub limit: Option<u16>,
    pub page: Option<String>,
    pub source: Option<AnthropicSkillSourceFilter>,
}

/// Cursor options for `list_versions`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnthropicSkillVersionListOptions {
    pub limit: Option<u16>,
    pub page: Option<String>,
}

/// Errors from Anthropic Skills lifecycle calls.
#[derive(Debug, thiserror::Error)]
pub enum AnthropicSkillsError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid Anthropic Skills input: {0}")]
    InvalidInput(String),
    #[error("invalid Anthropic Skills response: {0}")]
    InvalidResponse(String),
    #[error("Anthropic Skills endpoint returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    #[error("outcome of Anthropic Skills {operation} is unknown for {identity:?}: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        identity: Option<String>,
        #[source]
        source: LlmError,
    },
    #[error("outcome of Anthropic Skills {operation} is unknown for {identity:?}: {reason}")]
    OutcomeUnknownResponse {
        operation: &'static str,
        identity: Option<String>,
        reason: String,
    },
}

/// Anthropic Skills CRUD and version lifecycle client. Each method makes one
/// request and leaves pagination, retries, and uncertain-mutation recovery to
/// the caller.
pub struct AnthropicSkillsService<'a> {
    http: &'a dyn Transport,
    credential: Secret<String>,
    scope: AnthropicSkillScope,
    foundry_profile: Option<&'a ProviderProfile>,
    foundry_authenticator: Option<&'a dyn Authenticator>,
    timeout: Duration,
}

impl<'a> AnthropicSkillsService<'a> {
    pub fn new(
        http: &'a dyn Transport,
        credential: Secret<String>,
        scope: AnthropicSkillScope,
    ) -> Result<Self, AnthropicSkillsError> {
        scope.validate()?;
        if scope.is_foundry() {
            return Err(invalid_input(
                "Foundry Skills require the explicit new_foundry constructor",
            ));
        }
        if credential.expose_secret().trim().is_empty()
            || credential.expose_secret().chars().any(char::is_control)
        {
            return Err(invalid_input(
                "Anthropic API key must be non-empty and contain no control characters",
            ));
        }
        Ok(Self {
            http,
            credential,
            scope,
            foundry_profile: None,
            foundry_authenticator: None,
            timeout: DEFAULT_OPERATION_TIMEOUT,
        })
    }

    /// Construct a Skills service for a Foundry resource explicitly marked as
    /// Anthropic-hosted. This service is resource/account-scoped and does not
    /// require a chat model row. The supplied authenticator applies Foundry's
    /// API-key or Entra bearer credential using the actual profile settings.
    pub fn new_foundry(
        http: &'a dyn Transport,
        profile: &'a ProviderProfile,
        account_scope: &str,
        hosting: FoundryHosting,
        authenticator: &'a dyn Authenticator,
        credential: Secret<String>,
    ) -> Result<Self, AnthropicSkillsError> {
        if profile.protocol != crate::protocol::ProtocolFamily::FoundryClaude {
            return Err(LlmError::UnsupportedCapability {
                message: "Foundry Skills require the Foundry Claude protocol".into(),
            }
            .into());
        }
        if profile.auth == AuthStrategy::None {
            return Err(LlmError::UnsupportedCapability {
                message: "Foundry Skills require an API-key or bearer authenticator".into(),
            }
            .into());
        }
        if credential.expose_secret().trim().is_empty()
            || credential.expose_secret().chars().any(char::is_control)
        {
            return Err(invalid_input(
                "Foundry Skills credential must be non-empty and contain no control characters",
            ));
        }
        let scope = AnthropicSkillScope::new_foundry(
            profile.profile_name.clone(),
            profile.base_url.clone(),
            account_scope,
            hosting,
        )?;
        Ok(Self {
            http,
            credential,
            scope,
            foundry_profile: Some(profile),
            foundry_authenticator: Some(authenticator),
            timeout: DEFAULT_OPERATION_TIMEOUT,
        })
    }

    /// Set the single deadline for each request, including uploads and response reads.
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self, AnthropicSkillsError> {
        if timeout.is_zero() {
            return Err(invalid_input("operation timeout must be positive"));
        }
        self.timeout = timeout;
        Ok(self)
    }

    pub fn scope(&self) -> &AnthropicSkillScope {
        &self.scope
    }

    /// Create a custom Skill from one directory's files. `display_name` is an
    /// optional single-line label; callers own all file content and paths.
    pub async fn create(
        &self,
        files: Vec<AnthropicSkillFile>,
        display_name: Option<&str>,
    ) -> Result<AnthropicSkill, AnthropicSkillsError> {
        let display_name = validate_display_name(display_name)?;
        let boundary = multipart_boundary();
        let (body, content_length) = multipart_upload(files, display_name.as_deref(), &boundary)?;
        let url = self.url(SKILLS_PATH)?;
        let request = self
            .request(
                "POST",
                url,
                Some(format!("multipart/form-data; boundary={boundary}")),
            )
            .await?;
        let stream_request = HttpStreamRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body,
            content_length,
            timeout: request.timeout,
        };
        let response = HttpExecutor::new(self.http)
            .with_deadline(crate::runtime::Deadline::after(Some(self.timeout)))
            .execute_stream_bounded(stream_request, MAX_CONTROL_RESPONSE_BYTES)
            .await
            .map_err(|source| mutation_transport_error("create", None, source))?;
        ensure_mutation_success(&response, "create", None)?;
        let skill = self.decode_skill(response.body.as_ref()).map_err(|error| {
            AnthropicSkillsError::OutcomeUnknownResponse {
                operation: "create",
                identity: None,
                reason: error.to_string(),
            }
        })?;
        if skill.source_type != "custom" {
            return Err(AnthropicSkillsError::OutcomeUnknownResponse {
                operation: "create",
                identity: Some(skill.id),
                reason: "create response did not identify a custom Skill".into(),
            });
        }
        Ok(skill)
    }

    /// Return exactly one cursor page. The caller passes `next_page` back to
    /// continue; this method never auto-paginates.
    pub async fn list(
        &self,
        options: &AnthropicSkillListOptions,
    ) -> Result<AnthropicSkillPage, AnthropicSkillsError> {
        let limit = validate_limit(options.limit)?;
        let mut query = url::form_urlencoded::Serializer::new(String::new());
        query.append_pair("limit", &limit.to_string());
        if let Some(page) = options.page.as_deref() {
            validate_cursor(page)?;
            query.append_pair("page", page);
        }
        if let Some(source) = options.source {
            query.append_pair(
                "source",
                match source {
                    AnthropicSkillSourceFilter::Custom => "custom",
                    AnthropicSkillSourceFilter::Anthropic => "anthropic",
                },
            );
        }
        let response = self
            .send_control("GET", format!("{SKILLS_PATH}?{}", query.finish()))
            .await?;
        ensure_success(&response)?;
        let native = parse_json(&response.body, "list")?;
        let page = PageWire::decode(&native)?;
        validate_next_cursor(&page.next_page, options.page.as_deref())?;
        let skills = page
            .data
            .into_iter()
            .map(|value| self.decode_skill_value(value))
            .collect::<Result<Vec<_>, _>>()?;
        ensure_unique_ids(skills.iter().map(|skill| skill.id.as_str()), "Skill")?;
        Ok(AnthropicSkillPage {
            skills,
            next_page: page.next_page,
            native,
        })
    }

    pub async fn get(
        &self,
        reference: &AnthropicSkillResourceRef,
    ) -> Result<AnthropicSkill, AnthropicSkillsError> {
        self.validate_reference(reference)?;
        let path = format!(
            "{SKILLS_PATH}/{}",
            encode_path_segment(reference.skill_id())
        );
        let response = self.send_control("GET", path).await?;
        ensure_success(&response)?;
        let skill = self.decode_skill(&response.body)?;
        ensure_skill_id(reference.skill_id(), &skill.id)?;
        Ok(skill)
    }

    /// Delete a custom Skill. Non-custom source types are read-only through
    /// this mutation API.
    pub async fn delete(
        &self,
        reference: &AnthropicSkillResourceRef,
    ) -> Result<AnthropicSkillDeleted, AnthropicSkillsError> {
        self.validate_custom_reference(reference)?;
        let identity = Some(reference.skill_id().to_owned());
        let path = format!(
            "{SKILLS_PATH}/{}",
            encode_path_segment(reference.skill_id())
        );
        let request = self.request("DELETE", self.url(&path)?, None).await?;
        let response = HttpExecutor::new(self.http)
            .with_deadline(crate::runtime::Deadline::after(Some(self.timeout)))
            .execute_bounded(request, MAX_CONTROL_RESPONSE_BYTES)
            .await
            .map_err(|source| mutation_transport_error("delete", identity.clone(), source))?;
        ensure_mutation_success(&response, "delete", identity.clone())?;
        let native = parse_json(&response.body, "delete")
            .map_err(|error| unknown_response("delete", identity.clone(), error))?;
        let wire: DeletedWire = serde_json::from_value(native.clone()).map_err(|error| {
            unknown_response(
                "delete",
                identity.clone(),
                invalid_response(error.to_string()),
            )
        })?;
        if wire.object_type != "skill_deleted" || wire.id != reference.skill_id() {
            return Err(AnthropicSkillsError::OutcomeUnknownResponse {
                operation: "delete",
                identity,
                reason: "Anthropic returned a different or invalid deleted Skill identity".into(),
            });
        }
        Ok(AnthropicSkillDeleted {
            id: wire.id,
            native,
        })
    }

    pub async fn create_version(
        &self,
        reference: &AnthropicSkillResourceRef,
        files: Vec<AnthropicSkillFile>,
    ) -> Result<AnthropicSkillVersion, AnthropicSkillsError> {
        self.validate_custom_reference(reference)?;
        let identity = Some(reference.skill_id().to_owned());
        let boundary = multipart_boundary();
        let (body, content_length) = multipart_upload(files, None, &boundary)?;
        let path = format!(
            "{SKILLS_PATH}/{}/versions",
            encode_path_segment(reference.skill_id())
        );
        let request = self
            .request(
                "POST",
                self.url(&path)?,
                Some(format!("multipart/form-data; boundary={boundary}")),
            )
            .await?;
        let response = HttpExecutor::new(self.http)
            .with_deadline(crate::runtime::Deadline::after(Some(self.timeout)))
            .execute_stream_bounded(
                HttpStreamRequest {
                    method: request.method,
                    url: request.url,
                    headers: request.headers,
                    body,
                    content_length,
                    timeout: request.timeout,
                },
                MAX_CONTROL_RESPONSE_BYTES,
            )
            .await
            .map_err(|source| {
                mutation_transport_error("create version", identity.clone(), source)
            })?;
        ensure_mutation_success(&response, "create version", identity.clone())?;
        let version = self
            .decode_version(&response.body, reference)
            .map_err(|error| unknown_response("create version", identity.clone(), error))?;
        Ok(version)
    }

    pub async fn list_versions(
        &self,
        reference: &AnthropicSkillResourceRef,
        options: &AnthropicSkillVersionListOptions,
    ) -> Result<AnthropicSkillVersionPage, AnthropicSkillsError> {
        self.validate_reference(reference)?;
        let limit = validate_limit(options.limit)?;
        let mut query = url::form_urlencoded::Serializer::new(String::new());
        query.append_pair("limit", &limit.to_string());
        if let Some(page) = options.page.as_deref() {
            validate_cursor(page)?;
            query.append_pair("page", page);
        }
        let path = format!(
            "{SKILLS_PATH}/{}/versions?{}",
            encode_path_segment(reference.skill_id()),
            query.finish()
        );
        let response = self.send_control("GET", path).await?;
        ensure_success(&response)?;
        let native = parse_json(&response.body, "list versions")?;
        let page = PageWire::decode(&native)?;
        validate_next_cursor(&page.next_page, options.page.as_deref())?;
        let versions = page
            .data
            .into_iter()
            .map(|value| decode_version_value(value, reference))
            .collect::<Result<Vec<_>, _>>()?;
        ensure_unique_ids(
            versions.iter().map(|version| version.id.as_str()),
            "Skill version",
        )?;
        Ok(AnthropicSkillVersionPage {
            versions,
            next_page: page.next_page,
            native,
        })
    }

    pub async fn get_version(
        &self,
        reference: &AnthropicSkillResourceRef,
        version_id: &str,
    ) -> Result<AnthropicSkillVersion, AnthropicSkillsError> {
        self.validate_reference(reference)?;
        validate_identifier(version_id, "version ID")?;
        let path = version_path(reference.skill_id(), version_id);
        let response = self.send_control("GET", path).await?;
        ensure_success(&response)?;
        let version = self.decode_version(&response.body, reference)?;
        if version.id != version_id {
            return Err(invalid_response(
                "Anthropic returned a different version ID than requested",
            ));
        }
        Ok(version)
    }

    pub async fn delete_version(
        &self,
        reference: &AnthropicSkillResourceRef,
        version_id: &str,
    ) -> Result<AnthropicSkillVersionDeleted, AnthropicSkillsError> {
        self.validate_custom_reference(reference)?;
        validate_identifier(version_id, "version ID")?;
        let identity = Some(format!("{}/{}", reference.skill_id(), version_id));
        let request = self
            .request(
                "DELETE",
                self.url(&version_path(reference.skill_id(), version_id))?,
                None,
            )
            .await?;
        let response = HttpExecutor::new(self.http)
            .with_deadline(crate::runtime::Deadline::after(Some(self.timeout)))
            .execute_bounded(request, MAX_CONTROL_RESPONSE_BYTES)
            .await
            .map_err(|source| {
                mutation_transport_error("delete version", identity.clone(), source)
            })?;
        ensure_mutation_success(&response, "delete version", identity.clone())?;
        let native = parse_json(&response.body, "delete version")
            .map_err(|error| unknown_response("delete version", identity.clone(), error))?;
        let wire: DeletedWire = serde_json::from_value(native.clone()).map_err(|error| {
            unknown_response(
                "delete version",
                identity.clone(),
                invalid_response(error.to_string()),
            )
        })?;
        if wire.object_type != "skill_version_deleted" || wire.id != version_id {
            return Err(AnthropicSkillsError::OutcomeUnknownResponse {
                operation: "delete version",
                identity,
                reason: "Anthropic returned a different or invalid deleted version identity".into(),
            });
        }
        Ok(AnthropicSkillVersionDeleted {
            id: wire.id,
            native,
        })
    }

    /// Download a version's ZIP bytes as a stream. The route is currently
    /// documented in Anthropic's Beta API reference; its example uses version
    /// IDs and does not require an `anthropic-beta` header. No archive is
    /// extracted and no local file is created.
    pub async fn download_version_content(
        &self,
        reference: &AnthropicSkillResourceRef,
        version_id: &str,
    ) -> Result<AnthropicSkillContentStream, AnthropicSkillsError> {
        self.validate_reference(reference)?;
        validate_identifier(version_id, "version ID")?;
        if self.scope.is_foundry() {
            return Err(LlmError::UnsupportedCapability {
                message: "Foundry does not support downloading Skill version content".into(),
            }
            .into());
        }
        let request = self
            .request(
                "GET",
                self.url(&format!(
                    "{}/content",
                    version_path(reference.skill_id(), version_id)
                ))?,
                None,
            )
            .await?;
        let response = HttpExecutor::new(self.http)
            .with_deadline(crate::runtime::Deadline::after(Some(self.timeout)))
            .send(request)
            .await?;
        if !(200..300).contains(&response.status) {
            let response = HttpExecutor::collect_response(response, Some(1024 * 1024)).await?;
            ensure_success(&response)?;
            return Err(invalid_response(
                "non-success content response was not rejected",
            ));
        }
        Ok(AnthropicSkillContentStream::new(response))
    }

    async fn send_control(
        &self,
        method: &str,
        path: String,
    ) -> Result<HttpResponse, AnthropicSkillsError> {
        let request = self.request(method, self.url(&path)?, None).await?;
        Ok(HttpExecutor::new(self.http)
            .with_deadline(crate::runtime::Deadline::after(Some(self.timeout)))
            .execute_bounded(request, MAX_CONTROL_RESPONSE_BYTES)
            .await?)
    }

    async fn request(
        &self,
        method: &str,
        url: String,
        content_type: Option<String>,
    ) -> Result<HttpRequest, AnthropicSkillsError> {
        let mut headers = vec![("anthropic-version".into(), API_VERSION.into())];
        if let Some(workspace_id) = self.scope.workspace_id() {
            headers.push(("anthropic-workspace-id".into(), workspace_id.into()));
        }
        if let Some(content_type) = content_type {
            headers.push(("Content-Type".into(), content_type));
        }
        let mut request = HttpRequest {
            method: method.into(),
            url,
            headers,
            body: Bytes::new(),
            timeout: None,
        };
        if let (Some(profile), Some(authenticator)) =
            (self.foundry_profile, self.foundry_authenticator)
        {
            authenticator
                .apply(&mut request, profile, Some(&self.credential))
                .await?;
        } else {
            request
                .headers
                .push(("X-Api-Key".into(), self.credential.expose_secret().clone()));
        }
        Ok(request)
    }

    fn url(&self, path: &str) -> Result<String, AnthropicSkillsError> {
        if !path.starts_with('/') || path.chars().any(char::is_control) || path.contains('#') {
            return Err(invalid_input("invalid internal Anthropic Skills route"));
        }
        Ok(format!(
            "{}{path}",
            self.scope.endpoint().trim_end_matches('/')
        ))
    }

    fn validate_reference(
        &self,
        reference: &AnthropicSkillResourceRef,
    ) -> Result<(), AnthropicSkillsError> {
        validate_identifier(reference.skill_id(), "skill ID")?;
        reference.scope().validate()?;
        if reference.scope() != &self.scope {
            return Err(scope_mismatch());
        }
        Ok(())
    }

    fn validate_custom_reference(
        &self,
        reference: &AnthropicSkillResourceRef,
    ) -> Result<(), AnthropicSkillsError> {
        self.validate_reference(reference)?;
        if reference.source_type() != "custom" {
            return Err(invalid_input("Anthropic-managed Skills are read-only"));
        }
        Ok(())
    }

    fn decode_skill(&self, body: &[u8]) -> Result<AnthropicSkill, AnthropicSkillsError> {
        let native = parse_json(body, "Skill")?;
        self.decode_skill_value(native)
    }

    fn decode_skill_value(&self, native: Value) -> Result<AnthropicSkill, AnthropicSkillsError> {
        let wire: SkillWire = serde_json::from_value(native.clone())
            .map_err(|error| invalid_response(error.to_string()))?;
        if wire.object_type != "skill" {
            return Err(invalid_response("response type was not skill"));
        }
        validate_identifier(&wire.id, "provider skill ID")
            .map_err(|_| invalid_response("provider returned an invalid skill ID"))?;
        let source_type = wire.source.kind;
        let reference = AnthropicSkillResourceRef::new(
            wire.id.clone(),
            source_type.clone(),
            self.scope.clone(),
        )?;
        Ok(AnthropicSkill {
            id: wire.id,
            created_at: wire.created_at,
            display_name: wire.display_name,
            latest_version_id: wire.latest_version_id,
            updated_at: wire.updated_at,
            source_type,
            reference,
            native,
        })
    }

    fn decode_version(
        &self,
        body: &[u8],
        reference: &AnthropicSkillResourceRef,
    ) -> Result<AnthropicSkillVersion, AnthropicSkillsError> {
        let native = parse_json(body, "Skill version")?;
        decode_version_value(native, reference)
    }
}

/// Stream of ZIP bytes from the Skills version content endpoint.
pub struct AnthropicSkillContentStream {
    body: BoxStream<'static, Result<Bytes, LlmError>>,
    bytes_seen: usize,
    terminal: bool,
}

struct StopAfterUploadError {
    source: BoxStream<'static, Result<Bytes, LlmError>>,
    terminal: bool,
}

impl Stream for StopAfterUploadError {
    type Item = Result<Bytes, LlmError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.terminal {
            return Poll::Ready(None);
        }
        match self.source.as_mut().poll_next(cx) {
            Poll::Ready(Some(Err(error))) => {
                self.terminal = true;
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                self.terminal = true;
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

impl AnthropicSkillContentStream {
    fn new(response: StreamResponse) -> Self {
        Self {
            body: response.body,
            bytes_seen: 0,
            terminal: false,
        }
    }
}

impl Stream for AnthropicSkillContentStream {
    type Item = Result<Bytes, AnthropicSkillsError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.terminal {
            return Poll::Ready(None);
        }
        match this.body.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                let Some(total) = this.bytes_seen.checked_add(chunk.len()) else {
                    this.terminal = true;
                    return Poll::Ready(Some(Err(LlmError::RequestTooLarge {
                        message: "downloaded Skill content size overflowed".into(),
                    }
                    .into())));
                };
                if total > MAX_CONTENT_BYTES {
                    this.terminal = true;
                    return Poll::Ready(Some(Err(LlmError::RequestTooLarge {
                        message: "downloaded Skill content exceeds the 64 MiB client limit".into(),
                    }
                    .into())));
                }
                this.bytes_seen = total;
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(error))) => {
                this.terminal = true;
                Poll::Ready(Some(Err(error.into())))
            }
            Poll::Ready(None) => {
                this.terminal = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[derive(Deserialize)]
struct PageWire {
    data: Vec<Value>,
    #[serde(default)]
    next_page: Option<String>,
}

impl PageWire {
    fn decode(native: &Value) -> Result<Self, AnthropicSkillsError> {
        let page: Self = serde_json::from_value(native.clone())
            .map_err(|error| invalid_response(error.to_string()))?;
        if let Some(cursor) = page.next_page.as_deref() {
            validate_cursor(cursor)?;
        }
        Ok(page)
    }
}

#[derive(Deserialize)]
struct SkillWire {
    id: String,
    #[serde(rename = "type")]
    object_type: String,
    created_at: String,
    display_name: String,
    latest_version_id: String,
    source: SkillSourceWire,
    updated_at: String,
}

#[derive(Deserialize)]
struct SkillSourceWire {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct SkillVersionWire {
    id: String,
    skill_id: String,
    #[serde(rename = "type")]
    object_type: String,
    created_at: String,
    description: String,
    name: String,
}

#[derive(Deserialize)]
struct DeletedWire {
    id: String,
    #[serde(rename = "type")]
    object_type: String,
}

fn decode_version_value(
    native: Value,
    reference: &AnthropicSkillResourceRef,
) -> Result<AnthropicSkillVersion, AnthropicSkillsError> {
    let wire: SkillVersionWire = serde_json::from_value(native.clone())
        .map_err(|error| invalid_response(error.to_string()))?;
    if wire.object_type != "skill_version" || wire.skill_id != reference.skill_id() {
        return Err(invalid_response(
            "version response has a different Skill identity or object type",
        ));
    }
    validate_identifier(&wire.id, "provider version ID")
        .map_err(|_| invalid_response("provider returned an invalid version ID"))?;
    Ok(AnthropicSkillVersion {
        id: wire.id,
        skill_id: wire.skill_id,
        created_at: wire.created_at,
        description: wire.description,
        name: wire.name,
        reference: reference.clone(),
        native,
    })
}

fn multipart_upload(
    files: Vec<AnthropicSkillFile>,
    display_name: Option<&str>,
    boundary: &str,
) -> Result<(BoxStream<'static, Result<Bytes, LlmError>>, u64), AnthropicSkillsError> {
    if files.is_empty() {
        return Err(invalid_input("at least one Skill file is required"));
    }
    let mut paths = BTreeSet::new();
    let mut root: Option<String> = None;
    let mut total = 0_u64;
    for file in &files {
        validate_relative_path(&file.relative_path)?;
        let components = file.relative_path.split('/').collect::<Vec<_>>();
        let top = components[0];
        if components.len() < 2 {
            return Err(invalid_input(
                "Skill files must be under one top-level directory",
            ));
        }
        if root.as_deref().is_some_and(|known| known != top) {
            return Err(invalid_input(
                "all Skill files must share the same top-level directory",
            ));
        }
        root.get_or_insert_with(|| top.to_owned());
        if !paths.insert(file.relative_path.clone()) {
            return Err(invalid_input("Skill file paths must be unique"));
        }
        total = total.checked_add(file.size_bytes).ok_or_else(|| {
            AnthropicSkillsError::Llm(LlmError::RequestTooLarge {
                message: "combined Skill upload size overflowed".into(),
            })
        })?;
    }
    let root = root.expect("non-empty upload was checked");
    if !paths.contains(&format!("{root}/SKILL.md")) {
        return Err(invalid_input(
            "Skill files must include SKILL.md at the top-level directory root",
        ));
    }
    if total > MAX_UPLOAD_BYTES {
        return Err(AnthropicSkillsError::Llm(LlmError::RequestTooLarge {
            message: "combined uncompressed Skill upload exceeds 30,000,000 bytes".into(),
        }));
    }

    let mut prefix = BytesMut::new();
    if let Some(display_name) = display_name {
        prefix.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"display_name\"\r\n\r\n{display_name}\r\n"
            )
            .as_bytes(),
        );
    }
    let mut content_length = prefix.len() as u64;
    let mut parts = vec![stream::once(async move { Ok(prefix.freeze()) }).boxed()];
    for file in files {
        let (path, size_bytes, body) = file.into_parts();
        let filename = path.replace('"', "\\\"");
        let part_prefix = Bytes::from(format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"files[]\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        ));
        let part_suffix = Bytes::from("\r\n");
        content_length = content_length
            .checked_add(part_prefix.len() as u64)
            .and_then(|value| value.checked_add(size_bytes))
            .and_then(|value| value.checked_add(part_suffix.len() as u64))
            .ok_or_else(|| {
                AnthropicSkillsError::Llm(LlmError::RequestTooLarge {
                    message: "multipart Skill upload size overflowed".into(),
                })
            })?;
        parts.push(stream::once(async move { Ok(part_prefix) }).boxed());
        let file_body = exact_upload_stream(body, size_bytes)
            .map(|item| {
                item.map_err(|error| LlmError::StreamInterrupted {
                    message: format!("Skill file stream failed during upload: {error}"),
                })
            })
            .boxed();
        parts.push(file_body);
        parts.push(stream::once(async move { Ok(part_suffix) }).boxed());
    }
    let suffix = Bytes::from(format!("--{boundary}--\r\n"));
    content_length = content_length
        .checked_add(suffix.len() as u64)
        .ok_or_else(|| {
            AnthropicSkillsError::Llm(LlmError::RequestTooLarge {
                message: "multipart Skill upload size overflowed".into(),
            })
        })?;
    parts.push(stream::once(async move { Ok(suffix) }).boxed());
    let body = stream::iter(parts.into_iter().map(Ok::<_, LlmError>))
        .try_flatten()
        .boxed();
    Ok((
        Box::pin(StopAfterUploadError {
            source: body,
            terminal: false,
        }),
        content_length,
    ))
}

fn validate_display_name(
    display_name: Option<&str>,
) -> Result<Option<String>, AnthropicSkillsError> {
    let Some(display_name) = display_name else {
        return Ok(None);
    };
    if display_name.trim().is_empty()
        || display_name.chars().count() > 255
        || display_name.chars().any(char::is_control)
    {
        return Err(invalid_input(
            "display_name must be a nonempty single-line value of at most 255 characters",
        ));
    }
    Ok(Some(display_name.to_owned()))
}

fn validate_relative_path(path: &str) -> Result<(), AnthropicSkillsError> {
    if path.trim().is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.chars().any(char::is_control)
        || path
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."))
    {
        return Err(invalid_input(
            "Skill file paths must be safe relative paths using forward slashes",
        ));
    }
    Ok(())
}

fn validate_limit(limit: Option<u16>) -> Result<u16, AnthropicSkillsError> {
    let limit = limit.unwrap_or(20);
    if !(1..=1000).contains(&limit) {
        return Err(invalid_input("page limit must be between 1 and 1000"));
    }
    Ok(limit)
}

fn validate_cursor(cursor: &str) -> Result<(), AnthropicSkillsError> {
    if cursor.trim().is_empty() || cursor.chars().any(char::is_control) {
        return Err(invalid_input("page cursor must be nonempty and valid"));
    }
    Ok(())
}

fn validate_next_cursor(
    next_page: &Option<String>,
    requested_page: Option<&str>,
) -> Result<(), AnthropicSkillsError> {
    if next_page
        .as_deref()
        .is_some_and(|next| Some(next) == requested_page)
    {
        return Err(invalid_response(
            "pagination returned the same cursor again",
        ));
    }
    Ok(())
}

fn ensure_unique_ids<'a>(
    ids: impl IntoIterator<Item = &'a str>,
    kind: &str,
) -> Result<(), AnthropicSkillsError> {
    let mut seen = BTreeSet::new();
    if ids.into_iter().all(|id| seen.insert(id)) {
        Ok(())
    } else {
        Err(invalid_response(format!(
            "page contains duplicate {kind} IDs"
        )))
    }
}

fn validate_identifier(value: &str, name: &str) -> Result<(), AnthropicSkillsError> {
    if value.trim().is_empty() || value.len() > 1024 || value.chars().any(char::is_control) {
        return Err(invalid_input(format!("{name} is empty or invalid")));
    }
    Ok(())
}

fn version_path(skill_id: &str, version_id: &str) -> String {
    format!(
        "{SKILLS_PATH}/{}/versions/{}",
        encode_path_segment(skill_id),
        encode_path_segment(version_id)
    )
}

fn encode_path_segment(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                output.push(char::from(byte));
            }
            _ => output.push_str(&format!("%{byte:02X}")),
        }
    }
    output
}

fn parse_json(body: &[u8], operation: &str) -> Result<Value, AnthropicSkillsError> {
    serde_json::from_slice(body)
        .map_err(|error| invalid_response(format!("{operation} response was not JSON: {error}")))
}

fn ensure_success(response: &HttpResponse) -> Result<(), AnthropicSkillsError> {
    if (200..300).contains(&response.status) {
        return Ok(());
    }
    let request_id = response
        .headers
        .iter()
        .find(|(name, _)| {
            name.eq_ignore_ascii_case("request-id") || name.eq_ignore_ascii_case("x-request-id")
        })
        .map(|(_, value)| value.clone());
    let body = serde_json::from_slice(&response.body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&response.body).into_owned()));
    Err(AnthropicSkillsError::Provider {
        status: response.status,
        request_id,
        body,
    })
}

fn ensure_mutation_success(
    response: &HttpResponse,
    operation: &'static str,
    identity: Option<String>,
) -> Result<(), AnthropicSkillsError> {
    match ensure_success(response) {
        Ok(()) => Ok(()),
        Err(AnthropicSkillsError::Provider {
            status,
            request_id,
            body,
        }) if status == 408 || status >= 500 => {
            Err(AnthropicSkillsError::OutcomeUnknownResponse {
                operation,
                identity,
                reason: format!(
                    "HTTP {status} response after mutation dispatch; request_id={request_id:?}; body={body}"
                ),
            })
        }
        Err(error) => Err(error),
    }
}

fn ensure_skill_id(expected: &str, actual: &str) -> Result<(), AnthropicSkillsError> {
    if expected != actual {
        return Err(invalid_response("Anthropic returned a different Skill ID"));
    }
    Ok(())
}

fn scope_mismatch() -> AnthropicSkillsError {
    LlmError::PermissionDenied {
        message:
            "Anthropic Skill reference belongs to another profile, endpoint, account, or workspace"
                .into(),
    }
    .into()
}

fn invalid_input(message: impl Into<String>) -> AnthropicSkillsError {
    AnthropicSkillsError::InvalidInput(message.into())
}

fn invalid_response(message: impl Into<String>) -> AnthropicSkillsError {
    AnthropicSkillsError::InvalidResponse(message.into())
}

fn unknown_response(
    operation: &'static str,
    identity: Option<String>,
    error: AnthropicSkillsError,
) -> AnthropicSkillsError {
    AnthropicSkillsError::OutcomeUnknownResponse {
        operation,
        identity,
        reason: error.to_string(),
    }
}

fn mutation_transport_error(
    operation: &'static str,
    identity: Option<String>,
    source: LlmError,
) -> AnthropicSkillsError {
    if matches!(&source, LlmError::UnsupportedCapability { .. }) {
        AnthropicSkillsError::Llm(source)
    } else {
        AnthropicSkillsError::OutcomeUnknown {
            operation,
            identity,
            source,
        }
    }
}
