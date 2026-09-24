use async_trait::async_trait;
use lingxi_llm_client::{protocol::*, *};
use serde_json::json;
use std::sync::Arc;
struct NoHttp;
#[async_trait]
impl Transport for NoHttp {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("local estimation never sends HTTP")
    }
}
fn estimate(provider: &str, model: &str) -> Result<LocalTokenEstimate, LocalTokenCountError> {
    let p:ProviderProfile=serde_json::from_value(json!({"profile_name":"p","provider_id":provider,"base_url":"https://example.test","protocol":"open_ai_chat","auth":"none","models":[{"display_model":model,"request_model":model,"billing_model":model}]})).unwrap();
    let c = LlmClientBuilder::with_transport(Arc::new(NoHttp), &[p])
        .with_region(Region::International)
        .build()
        .unwrap();
    let request:CompletionRequest=serde_json::from_value(json!({"model":model,"messages":[{"role":"user","content":[{"type":"text","text":"hello 世界"}]}]})).unwrap();
    c.estimate_local_tokens(&request)
}
#[test]
fn each_known_backend_is_explicitly_enabled_or_disabled() {
    for (provider, model, feature, enabled) in [
        (
            "openai",
            "gpt-4o",
            "tokenizer-openai",
            cfg!(feature = "tokenizer-openai"),
        ),
        (
            "deepseek",
            "deepseek-v4-pro",
            "tokenizer-deepseek",
            cfg!(feature = "tokenizer-deepseek"),
        ),
        (
            "qwen",
            "qwen3.8-max",
            "tokenizer-qwen",
            cfg!(feature = "tokenizer-qwen"),
        ),
        (
            "kimi",
            "kimi-k3",
            "tokenizer-kimi",
            cfg!(feature = "tokenizer-kimi"),
        ),
        (
            "zhipu",
            "glm-5",
            "tokenizer-glm",
            cfg!(feature = "tokenizer-glm"),
        ),
    ] {
        let result = estimate(provider, model);
        if enabled {
            assert!(result.unwrap().input_tokens > 0);
        } else {
            assert_eq!(
                result.unwrap_err(),
                LocalTokenCountError::FeatureDisabled {
                    feature: feature.into()
                }
            );
        }
        assert!(matches!(
            estimate(provider, "unknown-model"),
            Err(LocalTokenCountError::UnsupportedModel { .. })
        ));
    }
}
