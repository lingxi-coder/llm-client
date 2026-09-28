use crate::client::{ClientSource, RequestOptions};
use crate::embeddings::{self, backend::*, *};
use crate::protocol::{LlmError, ProviderProfile};
use crate::transport::HttpRequest;
use serde_json::Value;

pub(crate) struct ZhipuEmbeddings;
impl EmbeddingBackend for ZhipuEmbeddings {
    fn encode(
        &self,
        _profile: &ProviderProfile,
        route: &EmbeddingRoute,
        req: &EmbeddingRequest,
    ) -> Result<HttpRequest, LlmError> {
        validate_input(route, req)?;
        if route.endpoint == "https://open.bigmodel.cn/api/paas/v4/embeddings"
            && req.model == "embedding-3"
        {
            validate_limits(
                req,
                Some(DocumentedModelLimits {
                    max_inputs: 64,
                    dimensions: Some(&[2048, 1024, 512, 256]),
                }),
            )?;
        }
        crate::codecs::openai::embeddings::reject_task(req)?;
        post(
            route.endpoint.clone(),
            crate::codecs::openai::embeddings::body(req),
        )
    }
    fn vectors(
        &self,
        body: &Value,
        req: &EmbeddingRequest,
    ) -> Result<Vec<EmbeddingVector>, EmbeddingError> {
        decode_vectors(&body["data"], Some("index"), "embedding", req)
    }
    fn usage(&self, body: &Value) -> EmbeddingUsage {
        usage(body["usage"].clone(), "prompt_tokens", "total_tokens")
    }
}
#[derive(Clone, Copy)]
pub struct Embeddings<'a> {
    source: ClientSource<'a>,
    profile: &'a str,
}
impl<'a> Embeddings<'a> {
    pub(crate) fn new(source: ClientSource<'a>, profile: &'a str) -> Self {
        Self { source, profile }
    }
    pub async fn embed(
        &self,
        input: &EmbeddingRequest,
        options: &RequestOptions,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let snapshot = self.source.pin()?;
        embeddings::execute(&snapshot, self.profile, input, options).await
    }
}
