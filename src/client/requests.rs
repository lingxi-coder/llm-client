//! Public request entry points.
use super::executor::{DecisionExecutionError, RequestExecutor, RequestOutput};
use super::resolve::RequestRoute;
use super::*;
use crate::codecs::RequestMode;
use crate::protocol::{ChatResponse, WebSearchConfig};
impl ClientSnapshot {
    pub(crate) async fn web_search(
        &self,
        req: &ChatRequest,
        search: WebSearchConfig,
        opts: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        let mut req = req.clone();
        req.set_hosted_web_search(Some(search));
        self.complete(&req, opts).await
    }
    pub(crate) async fn web_search_in(
        &self,
        profile: &str,
        req: &ChatRequest,
        search: WebSearchConfig,
        opts: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        let mut req = req.clone();
        req.set_hosted_web_search(Some(search));
        self.complete_in(profile, &req, opts).await
    }
    pub(crate) async fn web_search_stream(
        &self,
        req: &ChatRequest,
        search: WebSearchConfig,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let mut req = req.clone();
        req.set_hosted_web_search(Some(search));
        self.stream(&req, opts).await
    }
    pub(crate) async fn web_search_stream_in(
        &self,
        profile: &str,
        req: &ChatRequest,
        search: WebSearchConfig,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let mut req = req.clone();
        req.set_hosted_web_search(Some(search));
        self.stream_in(profile, &req, opts).await
    }
    pub(crate) async fn complete(
        &self,
        req: &ChatRequest,
        opts: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        let route = self.state.config.resolve_request(&req.model, None)?;
        self.complete_route(route, req, opts).await
    }
    pub(crate) async fn complete_in(
        &self,
        profile: &str,
        req: &ChatRequest,
        opts: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        let route = self
            .state
            .config
            .resolve_request(&req.model, Some(profile))?;
        self.complete_route(route, req, opts).await
    }
    pub(crate) async fn complete_bound_in(
        &self,
        profile: &str,
        provider_id: &str,
        req: &ChatRequest,
        opts: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        let mut route = self
            .state
            .config
            .resolve_request(&req.model, Some(profile))?;
        route.retain_provider(provider_id)?;
        self.complete_route(route, req, opts).await
    }
    pub(crate) async fn stream(
        &self,
        req: &ChatRequest,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let route = self.state.config.resolve_request(&req.model, None)?;
        self.stream_route(route, req, opts).await
    }
    pub(crate) async fn stream_in(
        &self,
        profile: &str,
        req: &ChatRequest,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let route = self
            .state
            .config
            .resolve_request(&req.model, Some(profile))?;
        self.stream_route(route, req, opts).await
    }
    pub(crate) async fn stream_bound_in(
        &self,
        profile: &str,
        provider_id: &str,
        req: &ChatRequest,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let mut route = self
            .state
            .config
            .resolve_request(&req.model, Some(profile))?;
        route.retain_provider(provider_id)?;
        self.stream_route(route, req, opts).await
    }
    pub(super) async fn complete_route(
        &self,
        route: RequestRoute<'_>,
        req: &ChatRequest,
        opts: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        reject_image_output(&route)?;
        match RequestExecutor::new(self)
            .run(route, req, opts, RequestMode::Complete)
            .await?
        {
            RequestOutput::Complete(response) => Ok(*response),
            RequestOutput::Stream(_) => unreachable!(),
        }
    }
    pub(super) async fn complete_decision_route(
        &self,
        route: RequestRoute<'_>,
        req: &ChatRequest,
        opts: &RequestOptions,
    ) -> Result<(ChatResponse, Vec<crate::protocol::DecisionAttemptReport>), DecisionExecutionError>
    {
        reject_image_output(&route).map_err(DecisionExecutionError::Provider)?;
        match RequestExecutor::new(self)
            .run_decision(route, req, opts)
            .await?
        {
            (RequestOutput::Complete(response), prior_attempts) => Ok((*response, prior_attempts)),
            (RequestOutput::Stream(_), _) => unreachable!(),
        }
    }
    async fn stream_route(
        &self,
        route: RequestRoute<'_>,
        req: &ChatRequest,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        reject_image_output(&route)?;
        match RequestExecutor::new(self)
            .run(route, req, opts, RequestMode::Stream)
            .await?
        {
            RequestOutput::Stream(stream) => Ok(*stream),
            RequestOutput::Complete(_) => unreachable!(),
        }
    }
}

fn reject_image_output(route: &RequestRoute<'_>) -> Result<(), LlmError> {
    if route.connections.first().is_some_and(|connection| {
        connection
            .model
            .metadata
            .output_modalities
            .iter()
            .any(|modality| modality == "image")
    }) {
        return Err(LlmError::UnsupportedCapability {
            message: "image-output model requires client.images()".into(),
        });
    }
    Ok(())
}
