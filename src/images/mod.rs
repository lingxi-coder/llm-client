//! Independent image generation and native task service.

mod wire;

use crate::client::LlmClient;
use crate::protocol::{
    ImageApi, ImageCapabilities, ImageEditRequest, ImageGenerationRequest, ImageInput,
    ImageModelListing, ImageModelProfile, ImageRequest, ImageRequestOptions, ImageResponse,
    ImageRouteConfig, ImageTaskRef, ImageTaskSnapshot, LlmError, ProviderProfile, Secret,
};
use crate::runtime::Deadline;
use crate::transport::{HttpExecutor, HttpRequest};
use async_trait::async_trait;
use base64::Engine;
use ring::digest::{digest, SHA256};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ImageError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid image response: {0}")]
    InvalidResponse(String),
    #[error("image task cannot be used with this connection: {0}")]
    TaskScopeMismatch(String),
    #[error("image request failed after dispatch ({dispatch:?}): {source}")]
    Dispatched {
        source: LlmError,
        dispatch: ImageDispatch,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageDispatch {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

impl ImageError {
    /// Whether a failed generation could already have reached the provider.
    pub fn dispatch(&self) -> ImageDispatch {
        match self {
            Self::Llm(_) | Self::TaskScopeMismatch(_) => ImageDispatch::NotSent,
            Self::InvalidResponse(_) => ImageDispatch::Accepted,
            Self::Dispatched { dispatch, .. } => *dispatch,
        }
    }
}

/// One image API's wire contract. Transport, authentication and routing stay in ImageService.
pub trait ImageAdapter: Send + Sync + 'static {
    fn api(&self) -> ImageApi;
    fn validate(
        &self,
        request: &ImageRequest,
        model: &ImageModelProfile,
        route: &ImageRouteConfig,
        asynchronous: bool,
    ) -> Result<(), ImageError>;
    fn encode(
        &self,
        request: &ImageRequest,
        model: &ImageModelProfile,
        route: &ImageRouteConfig,
        asynchronous: bool,
    ) -> Result<HttpRequest, ImageError>;
    fn decode(
        &self,
        body: &Value,
        model: &ImageModelProfile,
        profile: &ProviderProfile,
    ) -> Result<ImageResponse, ImageError>;
    fn task_id(&self, body: &Value) -> Result<String, ImageError>;
    fn task_query(
        &self,
        route: &ImageRouteConfig,
        task_id: &str,
    ) -> Result<HttpRequest, ImageError>;
    fn decode_task(
        &self,
        body: &Value,
        task: &ImageTaskRef,
        profile: &ProviderProfile,
    ) -> Result<ImageTaskSnapshot, ImageError>;
}

#[async_trait]
pub trait ImageAuthenticator: Send + Sync + 'static {
    async fn apply(
        &self,
        request: &mut HttpRequest,
        route: &ImageRouteConfig,
        credential: Option<&Secret<String>>,
    ) -> Result<(), LlmError>;
}

struct BuiltinImageAdapter(ImageApi);
impl ImageAdapter for BuiltinImageAdapter {
    fn api(&self) -> ImageApi {
        self.0
    }
    fn validate(
        &self,
        request: &ImageRequest,
        model: &ImageModelProfile,
        route: &ImageRouteConfig,
        asynchronous: bool,
    ) -> Result<(), ImageError> {
        wire::validate(request, model, route, asynchronous)
    }
    fn encode(
        &self,
        request: &ImageRequest,
        model: &ImageModelProfile,
        route: &ImageRouteConfig,
        asynchronous: bool,
    ) -> Result<HttpRequest, ImageError> {
        wire::encode(request, model, route, asynchronous)
    }
    fn decode(
        &self,
        body: &Value,
        model: &ImageModelProfile,
        profile: &ProviderProfile,
    ) -> Result<ImageResponse, ImageError> {
        wire::decode(self.0, body, model, profile)
    }
    fn task_id(&self, body: &Value) -> Result<String, ImageError> {
        wire::task_id(self.0, body)
    }
    fn task_query(
        &self,
        route: &ImageRouteConfig,
        task_id: &str,
    ) -> Result<HttpRequest, ImageError> {
        wire::task_query(route, task_id)
    }
    fn decode_task(
        &self,
        body: &Value,
        task: &ImageTaskRef,
        profile: &ProviderProfile,
    ) -> Result<ImageTaskSnapshot, ImageError> {
        wire::decode_task(self.0, body, task, profile)
    }
}

pub(crate) fn builtin_adapters() -> BTreeMap<ImageApi, Arc<dyn ImageAdapter>> {
    [
        ImageApi::OpenAi,
        ImageApi::Gemini,
        ImageApi::Qwen,
        ImageApi::Wan,
        ImageApi::Xai,
        ImageApi::Minimax,
        ImageApi::Zai,
        ImageApi::OpenRouter,
    ]
    .into_iter()
    .map(|api| {
        (
            api,
            Arc::new(BuiltinImageAdapter(api)) as Arc<dyn ImageAdapter>,
        )
    })
    .collect()
}

pub struct ImageService<'a> {
    client: &'a LlmClient,
}

impl<'a> ImageService<'a> {
    pub(crate) fn new(client: &'a LlmClient) -> Self {
        Self { client }
    }

    pub fn models(&self) -> Vec<ImageModelListing> {
        self.client
            .snapshot
            .profiles
            .iter()
            .filter(|p| p.supports_region(self.client.region) && !p.connection.hidden)
            .flat_map(|p| {
                p.images
                    .models
                    .iter()
                    .filter(|m| !m.hidden)
                    .map(move |m| ImageModelListing {
                        profile_name: p.profile_name.clone(),
                        provider_id: p.provider_id.as_str().to_owned(),
                        display_model: m.display_model.clone(),
                        request_model: m.request_model.clone(),
                        capabilities: m.capabilities.clone(),
                    })
            })
            .collect()
    }

    pub fn capabilities(&self, model: &str) -> Result<ImageCapabilities, ImageError> {
        Ok(self.resolve(model, None)?.1.capabilities.clone())
    }
    pub fn capabilities_in(
        &self,
        profile: &str,
        model: &str,
    ) -> Result<ImageCapabilities, ImageError> {
        Ok(self.resolve(model, Some(profile))?.1.capabilities.clone())
    }
    pub async fn generate(
        &self,
        req: &ImageGenerationRequest,
        opts: &ImageRequestOptions,
    ) -> Result<ImageResponse, ImageError> {
        self.execute(ImageRequest::Generate(req.clone()), None, opts, false)
            .await
            .map(|result| result.0.expect("sync image call returns a response"))
    }
    pub async fn generate_in(
        &self,
        profile: &str,
        req: &ImageGenerationRequest,
        opts: &ImageRequestOptions,
    ) -> Result<ImageResponse, ImageError> {
        self.execute(
            ImageRequest::Generate(req.clone()),
            Some(profile),
            opts,
            false,
        )
        .await
        .map(|result| result.0.expect("sync image call returns a response"))
    }
    pub async fn edit(
        &self,
        req: &ImageEditRequest,
        opts: &ImageRequestOptions,
    ) -> Result<ImageResponse, ImageError> {
        self.execute(ImageRequest::Edit(req.clone()), None, opts, false)
            .await
            .map(|result| result.0.expect("sync image call returns a response"))
    }
    pub async fn edit_in(
        &self,
        profile: &str,
        req: &ImageEditRequest,
        opts: &ImageRequestOptions,
    ) -> Result<ImageResponse, ImageError> {
        self.execute(ImageRequest::Edit(req.clone()), Some(profile), opts, false)
            .await
            .map(|result| result.0.expect("sync image call returns a response"))
    }
    pub async fn submit(
        &self,
        req: &ImageRequest,
        opts: &ImageRequestOptions,
    ) -> Result<ImageTaskRef, ImageError> {
        self.execute(req.clone(), None, opts, true)
            .await
            .map(|result| result.1.expect("async image call returns a task"))
    }
    pub async fn submit_in(
        &self,
        profile: &str,
        req: &ImageRequest,
        opts: &ImageRequestOptions,
    ) -> Result<ImageTaskRef, ImageError> {
        self.execute(req.clone(), Some(profile), opts, true)
            .await
            .map(|result| result.1.expect("async image call returns a task"))
    }

    pub async fn get_task(
        &self,
        task: &ImageTaskRef,
        opts: &ImageRequestOptions,
    ) -> Result<ImageTaskSnapshot, ImageError> {
        let deadline = Deadline::after(Some(opts.total_timeout.unwrap_or(Duration::from_secs(60))));
        let profile = self
            .client
            .snapshot
            .profile(&task.profile_name)
            .filter(|p| p.supports_region(self.client.region))
            .ok_or_else(|| {
                ImageError::TaskScopeMismatch("profile is unavailable in this region".into())
            })?;
        let route =
            profile.images.routes.get(&task.route).ok_or_else(|| {
                ImageError::TaskScopeMismatch("image route no longer exists".into())
            })?;
        if task.version != 1
            || task.provider_id != profile.provider_id.as_str()
            || task.endpoint_fingerprint != fingerprint(route)
            || opts.account_scope.as_deref() != Some(task.account_scope.as_str())
        {
            return Err(ImageError::TaskScopeMismatch(
                "task identity, endpoint, or account changed".into(),
            ));
        }
        let adapter = self.adapter(route.api)?;
        let mut request = adapter.task_query(route, &task.provider_task_id)?;
        deadline
            .run(authenticate(self.client, &mut request, route, opts))
            .await??;
        request.timeout = deadline.remaining()?;
        let response = HttpExecutor::new(self.client.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(request, opts.max_response_bytes.unwrap_or(64 * 1024 * 1024))
            .await
            .map_err(|source| ImageError::Dispatched {
                source,
                dispatch: ImageDispatch::Unknown,
            })?;
        let body = wire::parse_json(&response)
            .map_err(|error| classify_response_error(error, response.status))?;
        adapter.decode_task(&body, task, profile)
    }

    async fn execute(
        &self,
        mut req: ImageRequest,
        scope: Option<&str>,
        opts: &ImageRequestOptions,
        asynchronous: bool,
    ) -> Result<(Option<ImageResponse>, Option<ImageTaskRef>), ImageError> {
        let deadline = Deadline::after(Some(
            opts.total_timeout
                .unwrap_or(Duration::from_secs(if asynchronous { 60 } else { 300 })),
        ));
        let (profile, model, route) = self.resolve(req.model(), scope)?;
        let adapter = self.adapter(route.api)?;
        adapter.validate(&req, model, route, asynchronous)?;
        if asynchronous && opts.account_scope.as_deref().is_none_or(str::is_empty) {
            return Err(LlmError::InvalidRequest {
                message: "native image tasks require a non-empty account_scope".into(),
            }
            .into());
        }
        deadline.run(self.prepare(&mut req)).await??;
        let mut request = adapter.encode(&req, model, route, asynchronous)?;
        deadline
            .run(authenticate(self.client, &mut request, route, opts))
            .await??;
        request.timeout = deadline.remaining()?;
        let response = HttpExecutor::new(self.client.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(request, opts.max_response_bytes.unwrap_or(64 * 1024 * 1024))
            .await
            .map_err(|source| ImageError::Dispatched {
                source,
                dispatch: ImageDispatch::Unknown,
            })?;
        let body = wire::parse_json(&response)
            .map_err(|error| classify_response_error(error, response.status))?;
        if asynchronous {
            let task_id = adapter.task_id(&body)?;
            return Ok((
                None,
                Some(ImageTaskRef {
                    version: 1,
                    provider_id: profile.provider_id.as_str().to_owned(),
                    profile_name: profile.profile_name.clone(),
                    route: model.route.clone(),
                    endpoint_fingerprint: fingerprint(route),
                    account_scope: opts.account_scope.clone().expect("validated above"),
                    request_model: model.request_model.clone(),
                    provider_task_id: task_id,
                }),
            ));
        }
        let result = adapter.decode(&body, model, profile)?;
        Ok((Some(result), None))
    }

    fn resolve(
        &self,
        requested: &str,
        scoped: Option<&str>,
    ) -> Result<(&ProviderProfile, &ImageModelProfile, &ImageRouteConfig), ImageError> {
        // A concrete connection wins over an identically named group, even
        // when only a sibling declares the requested model.
        let matches_scope = |profile: &ProviderProfile, scope: &str| {
            if self
                .client
                .snapshot
                .profile(scope)
                .is_some_and(|p| p.supports_region(self.client.region))
            {
                profile.profile_name == scope
            } else {
                profile.group() == scope
            }
        };
        let mut candidates = Vec::new();
        for profile in &self.client.snapshot.profiles {
            if !profile.supports_region(self.client.region) {
                continue;
            }
            if scoped.is_some_and(|s| !matches_scope(profile, s)) {
                continue;
            }
            for model in &profile.images.models {
                let qualified = format!("{}/{}", profile.profile_name, model.display_model);
                let group_qualified = format!("{}/{}", profile.group(), model.display_model);
                if requested != model.display_model
                    && requested != model.request_model
                    && requested != qualified
                    && !(requested == group_qualified && matches_scope(profile, profile.group()))
                    && !model.aliases.iter().any(|alias| alias == requested)
                {
                    continue;
                }
                let route = profile.images.routes.get(&model.route).ok_or_else(|| {
                    LlmError::InvalidRequest {
                        message: format!(
                            "image route {:?} missing on profile {:?}",
                            model.route, profile.profile_name
                        ),
                    }
                })?;
                candidates.push((profile, model, route));
            }
        }
        if candidates.len() == 1 {
            return Ok(candidates[0]);
        }
        if candidates.is_empty() {
            return Err(LlmError::ModelUnavailable {
                message: format!("no image model matches {requested:?}"),
            }
            .into());
        }
        Err(LlmError::InvalidRequest {
            message: format!("image model {requested:?} is ambiguous; qualify the profile"),
        }
        .into())
    }

    fn adapter(&self, api: ImageApi) -> Result<&dyn ImageAdapter, ImageError> {
        self.client
            .image_adapters
            .get(&api)
            .map(Arc::as_ref)
            .ok_or_else(|| {
                LlmError::UnsupportedCapability {
                    message: format!("no image adapter registered for {api:?}"),
                }
                .into()
            })
    }

    async fn prepare(&self, req: &mut ImageRequest) -> Result<(), ImageError> {
        match req {
            ImageRequest::Generate(request) => {
                for reference in &mut request.references {
                    self.prepare_input(&mut reference.image).await?;
                }
            }
            ImageRequest::Edit(request) => {
                for image in &mut request.images {
                    self.prepare_input(image).await?;
                }
                if let Some(mask) = &mut request.mask {
                    self.prepare_input(&mut mask.source).await?;
                }
            }
        }
        Ok(())
    }

    async fn prepare_input(&self, input: &mut ImageInput) -> Result<(), ImageError> {
        if let ImageInput::Attachment { attachment } = input {
            if !attachment.media_type.starts_with("image/") {
                return Err(LlmError::InvalidRequest {
                    message: "image attachment has a non-image media type".into(),
                }
                .into());
            }
            let bytes = self.client.attachments.resolve_image(attachment).await?;
            *input = ImageInput::Base64 {
                media_type: attachment.media_type.clone(),
                data: base64::engine::general_purpose::STANDARD.encode(bytes),
            };
        }
        Ok(())
    }
}

async fn authenticate(
    client: &LlmClient,
    request: &mut HttpRequest,
    route: &ImageRouteConfig,
    opts: &ImageRequestOptions,
) -> Result<(), ImageError> {
    if let Some(name) = &route.authenticator {
        let authenticator = client.image_authenticators.get(name).ok_or_else(|| {
            LlmError::UnsupportedCapability {
                message: format!("image authenticator {name:?} is not registered"),
            }
        })?;
        return authenticator
            .apply(request, route, opts.credential.as_ref())
            .await
            .map_err(Into::into);
    }
    let credential = opts
        .credential
        .as_ref()
        .ok_or_else(|| LlmError::Authentication {
            message: "image request needs a credential".into(),
        })?;
    let header = route.api_key_header.as_deref().unwrap_or("authorization");
    let value = if header.eq_ignore_ascii_case("authorization") {
        format!("Bearer {}", credential.expose_secret())
    } else {
        credential.expose_secret().clone()
    };
    request
        .headers
        .retain(|(name, _)| !name.eq_ignore_ascii_case(header));
    request.headers.push((header.into(), value));
    Ok(())
}

fn fingerprint(route: &ImageRouteConfig) -> String {
    let bytes = serde_json::to_vec(route).expect("image route serializes");
    digest(&SHA256, &bytes)
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn classify_response_error(error: ImageError, status: u16) -> ImageError {
    match error {
        ImageError::Llm(source) => ImageError::Dispatched {
            source,
            dispatch: if (400..500).contains(&status) {
                ImageDispatch::Rejected
            } else if (200..300).contains(&status) {
                ImageDispatch::Accepted
            } else {
                ImageDispatch::Unknown
            },
        },
        other => other,
    }
}
