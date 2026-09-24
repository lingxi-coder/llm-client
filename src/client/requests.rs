//! Public request entry points.
use super::executor::{RequestExecutor, RequestOutput};
use super::resolve::RequestRoute;
use super::*;
use crate::codecs::RequestMode;
use crate::protocol::{CompletionResponse, WebSearchConfig};
impl LlmClient {
    pub async fn web_search(
        &self,
        req: &CompletionRequest,
        search: WebSearchConfig,
        opts: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        let mut req = req.clone();
        req.web_search = Some(search);
        self.complete(&req, opts).await
    }
    pub async fn web_search_in(
        &self,
        profile: &str,
        req: &CompletionRequest,
        search: WebSearchConfig,
        opts: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        let mut req = req.clone();
        req.web_search = Some(search);
        self.complete_in(profile, &req, opts).await
    }
    pub async fn web_search_stream(
        &self,
        req: &CompletionRequest,
        search: WebSearchConfig,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let mut req = req.clone();
        req.web_search = Some(search);
        self.stream(&req, opts).await
    }
    pub async fn web_search_stream_in(
        &self,
        profile: &str,
        req: &CompletionRequest,
        search: WebSearchConfig,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let mut req = req.clone();
        req.web_search = Some(search);
        self.stream_in(profile, &req, opts).await
    }
    pub async fn complete(
        &self,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        let route = self.snapshot.resolve_request(&req.model, None)?;
        self.complete_route(route, req, opts).await
    }
    pub async fn complete_in(
        &self,
        profile: &str,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        let route = self.snapshot.resolve_request(&req.model, Some(profile))?;
        self.complete_route(route, req, opts).await
    }
    pub async fn stream(
        &self,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let route = self.snapshot.resolve_request(&req.model, None)?;
        self.stream_route(route, req, opts).await
    }
    pub async fn stream_in(
        &self,
        profile: &str,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        let route = self.snapshot.resolve_request(&req.model, Some(profile))?;
        self.stream_route(route, req, opts).await
    }
    async fn complete_route(
        &self,
        route: RequestRoute<'_>,
        req: &CompletionRequest,
        opts: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        reject_image_output(&route)?;
        match RequestExecutor::new(self)
            .run(route, req, opts, RequestMode::Complete)
            .await?
        {
            RequestOutput::Complete(response) => Ok(*response),
            RequestOutput::Stream(_) => unreachable!(),
        }
    }
    async fn stream_route(
        &self,
        route: RequestRoute<'_>,
        req: &CompletionRequest,
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
