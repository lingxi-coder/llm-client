//! Asset-free supported-model mapping.
#[cfg(feature = "tokenizer-openai")]
use super::backends::openai_encoder;
use super::{backends::Encoder, LocalTokenCountError};
use crate::protocol::ProviderId;
// Membership matches the pinned tiktoken-rs 0.12 mapping; kept asset-free so
// disabled features can be distinguished from unsupported model IDs.
pub(super) fn known_openai_model(model: &str) -> bool {
    let model = model
        .strip_prefix("ft:")
        .map_or(model, |model| model.split(':').next().unwrap_or(model));
    const EXACT: &[&str] = &[
        "o1",
        "o3",
        "o4-mini",
        "gpt-5",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5",
        "gpt-35-turbo",
        "davinci-002",
        "babbage-002",
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-davinci-003",
        "text-davinci-002",
        "text-davinci-001",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
        "davinci",
        "curie",
        "babbage",
        "ada",
        "code-davinci-002",
        "code-davinci-001",
        "code-cushman-002",
        "code-cushman-001",
        "davinci-codex",
        "cushman-codex",
        "text-davinci-edit-001",
        "code-davinci-edit-001",
        "text-similarity-davinci-001",
        "text-similarity-curie-001",
        "text-similarity-babbage-001",
        "text-similarity-ada-001",
        "text-search-davinci-doc-001",
        "text-search-curie-doc-001",
        "text-search-babbage-doc-001",
        "text-search-ada-doc-001",
        "code-search-babbage-code-001",
        "code-search-ada-code-001",
        "gpt2",
        "gpt-2",
    ];
    const PREFIX: &[&str] = &[
        "o1-",
        "o3-",
        "o4-mini-",
        "gpt-5-",
        "gpt-4.5-",
        "gpt-4.1-",
        "chatgpt-4o-",
        "gpt-4o-",
        "gpt-4-",
        "gpt-3.5-turbo-",
        "gpt-35-turbo-",
        "gpt-oss-",
        "gpt-5.",
        "codex-mini",
    ];
    EXACT.contains(&model) || PREFIX.iter().any(|prefix| model.starts_with(prefix))
}
pub(super) fn encoder_for(
    provider: &ProviderId,
    model: &str,
) -> Result<Option<Encoder>, LocalTokenCountError> {
    match provider.as_str() {
        "openai" if known_openai_model(model) => {
            #[cfg(feature = "tokenizer-openai")]
            {
                Ok(openai_encoder(model))
            }
            #[cfg(not(feature = "tokenizer-openai"))]
            {
                Err(LocalTokenCountError::FeatureDisabled {
                    feature: "tokenizer-openai".into(),
                })
            }
        }
        "deepseek" => match model {
            "deepseek-v4-pro" => {
                #[cfg(feature = "tokenizer-deepseek")]
                {
                    super::assets::deepseek_v4()
                }
                #[cfg(not(feature = "tokenizer-deepseek"))]
                {
                    Err(LocalTokenCountError::FeatureDisabled {
                        feature: "tokenizer-deepseek".into(),
                    })
                }
            }
            "deepseek-flash" => {
                #[cfg(feature = "tokenizer-deepseek")]
                {
                    super::assets::deepseek_v41()
                }
                #[cfg(not(feature = "tokenizer-deepseek"))]
                {
                    Err(LocalTokenCountError::FeatureDisabled {
                        feature: "tokenizer-deepseek".into(),
                    })
                }
            }
            _ => Ok(None),
        },
        "qwen" => match model {
            "qwen3.8-flash" | "qwen3.8-max" => {
                #[cfg(feature = "tokenizer-qwen")]
                {
                    super::assets::qwen38()
                }
                #[cfg(not(feature = "tokenizer-qwen"))]
                {
                    Err(LocalTokenCountError::FeatureDisabled {
                        feature: "tokenizer-qwen".into(),
                    })
                }
            }
            _ => Ok(None),
        },
        "kimi" if model == "kimi-k3" => {
            #[cfg(feature = "tokenizer-kimi")]
            {
                super::assets::kimi_k3()
            }
            #[cfg(not(feature = "tokenizer-kimi"))]
            {
                Err(LocalTokenCountError::FeatureDisabled {
                    feature: "tokenizer-kimi".into(),
                })
            }
        }
        "zhipu" if model == "glm-5" => {
            #[cfg(feature = "tokenizer-glm")]
            {
                super::assets::glm5()
            }
            #[cfg(not(feature = "tokenizer-glm"))]
            {
                Err(LocalTokenCountError::FeatureDisabled {
                    feature: "tokenizer-glm".into(),
                })
            }
        }
        // MiniMax M3's current upstream tokenizer asset is under a
        // non-commercial license and is intentionally not bundled.
        _ => Ok(None),
    }
}
