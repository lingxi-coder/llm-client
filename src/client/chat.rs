//! Conversation-facing service facade.
use super::{ClientSource, ModelStream, RequestOptions};
use crate::protocol::{ChatRequest, ChatResponse, LlmError, ModelListing};

#[derive(Clone, Copy)]
pub struct ChatService<'a> {
    client: ClientSource<'a>,
}

impl<'a> ChatService<'a> {
    pub(crate) fn new(client: ClientSource<'a>) -> Self {
        Self { client }
    }
    pub fn models(&self) -> Vec<ModelListing> {
        self.client.snapshot().models_matching(|model| {
            !model
                .metadata
                .output_modalities
                .iter()
                .any(|modality| modality == "image")
        })
    }
    pub async fn complete(
        self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        self.client.snapshot().complete(request, options).await
    }
    pub async fn complete_in(
        self,
        profile: &str,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        self.client
            .snapshot()
            .complete_in(profile, request, options)
            .await
    }
    pub async fn stream(
        self,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        self.client.snapshot().stream(request, options).await
    }
    pub async fn stream_in(
        self,
        profile: &str,
        request: &ChatRequest,
        options: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        self.client
            .snapshot()
            .stream_in(profile, request, options)
            .await
    }
    pub async fn web_search(
        self,
        request: &ChatRequest,
        search: crate::protocol::WebSearchConfig,
        options: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        self.client
            .snapshot()
            .web_search(request, search, options)
            .await
    }

    pub async fn web_search_in(
        self,
        profile: &str,
        request: &ChatRequest,
        search: crate::protocol::WebSearchConfig,
        options: &RequestOptions,
    ) -> Result<ChatResponse, LlmError> {
        self.client
            .snapshot()
            .web_search_in(profile, request, search, options)
            .await
    }

    pub async fn web_search_stream(
        self,
        request: &ChatRequest,
        search: crate::protocol::WebSearchConfig,
        options: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        self.client
            .snapshot()
            .web_search_stream(request, search, options)
            .await
    }

    pub async fn web_search_stream_in(
        self,
        profile: &str,
        request: &ChatRequest,
        search: crate::protocol::WebSearchConfig,
        options: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        self.client
            .snapshot()
            .web_search_stream_in(profile, request, search, options)
            .await
    }
}
