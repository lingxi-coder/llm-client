use crate::client::{ClientSource, RequestOptions};
use crate::embeddings::{self, backend::*, *};
use crate::protocol::{LlmError, ProviderProfile};
use crate::transport::HttpRequest;
use serde_json::{json, Value};

pub(crate) struct QwenEmbeddings;
impl EmbeddingBackend for QwenEmbeddings {
    fn encode(
        &self,
        profile: &ProviderProfile,
        route: &EmbeddingRoute,
        req: &EmbeddingRequest,
    ) -> Result<HttpRequest, LlmError> {
        validate_input(route, req)?;
        if profile.provider_id.as_str() == "qwen" {
            validate_limits(req, model_limits(&req.model))?;
        }
        let body = {
            let mut parameters = json!({});
            if let Some(dim) = req.dimensions {
                parameters["dimension"] = json!(dim);
            }
            if let Some(task) = req.task {
                parameters["text_type"] = json!(match task {
                    EmbeddingTask::RetrievalQuery => "query",
                    EmbeddingTask::RetrievalDocument => "document",
                    _ =>
                        return Err(LlmError::UnsupportedCapability {
                            message: "Qwen text embedding task must be retrieval query or document"
                                .into()
                        }),
                });
            }
            json!({"model":req.model,"input":{"texts":req.input},"parameters":parameters})
        };
        post(route.endpoint.clone(), body)
    }
    fn vectors(
        &self,
        body: &Value,
        req: &EmbeddingRequest,
    ) -> Result<Vec<EmbeddingVector>, EmbeddingError> {
        decode_vectors(
            &body["output"]["embeddings"],
            Some("text_index"),
            "embedding",
            req,
        )
    }
    fn usage(&self, body: &Value) -> EmbeddingUsage {
        usage(body["usage"].clone(), "total_tokens", "total_tokens")
    }
}
fn model_limits(model: &str) -> Option<DocumentedModelLimits> {
    const QWEN_37: &[usize] = &[2560, 2048, 1536, 1024, 768, 512, 256];
    const QWEN_37_FLASH: &[usize] = &[1024, 768, 512, 256];
    const QWEN_V4: &[usize] = &[2048, 1536, 1024, 768, 512, 256, 128, 64];
    const QWEN_V3: &[usize] = &[1024, 768, 512, 256, 128, 64];
    match model {
        "qwen3.7-text-embedding" => Some(DocumentedModelLimits {
            max_inputs: 20,
            dimensions: Some(QWEN_37),
        }),
        "qwen3.7-text-embedding-flash" => Some(DocumentedModelLimits {
            max_inputs: 20,
            dimensions: Some(QWEN_37_FLASH),
        }),
        "text-embedding-v4" => Some(DocumentedModelLimits {
            max_inputs: 10,
            dimensions: Some(QWEN_V4),
        }),
        "text-embedding-v3" => Some(DocumentedModelLimits {
            max_inputs: 10,
            dimensions: Some(QWEN_V3),
        }),
        "text-embedding-v1" | "text-embedding-v2" => Some(DocumentedModelLimits {
            max_inputs: 25,
            dimensions: None,
        }),
        _ => None,
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
