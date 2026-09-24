//! Conversation-facing service facade.
use super::{LlmClient, ModelStream, RequestOptions};
use crate::protocol::{CompletionRequest, CompletionResponse, LlmError, ModelListing};

pub struct ChatService<'a> {
    client: &'a LlmClient,
}

impl<'a> ChatService<'a> {
    pub(crate) fn new(client: &'a LlmClient) -> Self {
        Self { client }
    }
    pub fn models(&self) -> Vec<ModelListing> {
        self.client
            .models()
            .into_iter()
            .filter(|listed| {
                self.client
                    .profiles()
                    .iter()
                    .find(|p| p.profile_name == listed.profile_name)
                    .and_then(|p| {
                        p.models
                            .iter()
                            .find(|m| m.request_model == listed.request_model)
                    })
                    .is_none_or(|m| {
                        !m.metadata
                            .output_modalities
                            .iter()
                            .any(|modality| modality == "image")
                    })
            })
            .collect()
    }
    pub async fn complete(
        &self,
        request: &CompletionRequest,
        options: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        self.client.complete(request, options).await
    }
    pub async fn complete_in(
        &self,
        profile: &str,
        request: &CompletionRequest,
        options: &RequestOptions,
    ) -> Result<CompletionResponse, LlmError> {
        self.client.complete_in(profile, request, options).await
    }
    pub async fn stream(
        &self,
        request: &CompletionRequest,
        options: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        self.client.stream(request, options).await
    }
    pub async fn stream_in(
        &self,
        profile: &str,
        request: &CompletionRequest,
        options: &RequestOptions,
    ) -> Result<ModelStream, LlmError> {
        self.client.stream_in(profile, request, options).await
    }
}
