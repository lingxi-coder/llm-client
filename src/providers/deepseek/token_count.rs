//! Optional provider tokenizer mappings and bundled assets.
#[cfg(feature = "tokenizer-deepseek")]
use crate::token_count::{assets::load, backends::bundled_encoder};
use crate::token_count::{backends::Encoder, LocalTokenCountError};
#[cfg(feature = "tokenizer-deepseek")]
use std::sync::OnceLock;
#[cfg(feature = "tokenizer-deepseek")]
use tokenizers::Tokenizer;
#[cfg(feature = "tokenizer-deepseek")]
pub(crate) static DEEPSEEK_V4: OnceLock<Result<Tokenizer, String>> = OnceLock::new();
#[cfg(feature = "tokenizer-deepseek")]
pub(crate) static DEEPSEEK_V41: OnceLock<Result<Tokenizer, String>> = OnceLock::new();
#[cfg(feature = "tokenizer-deepseek")]
pub(crate) fn deepseek_v4() -> Result<Option<Encoder>, LocalTokenCountError> {
    let name = "DeepSeek V4 Pro (2026-09)";
    Ok(Some(bundled_encoder(
        load(
            &DEEPSEEK_V4,
            include_bytes!("../../../data/tokenizers/deepseek/v4.json.xz"),
            name,
        )?,
        name,
    )))
}
#[cfg(feature = "tokenizer-deepseek")]
pub(crate) fn deepseek_v41() -> Result<Option<Encoder>, LocalTokenCountError> {
    let name = "DeepSeek V4.1 Flash (2026-09)";
    Ok(Some(bundled_encoder(
        load(
            &DEEPSEEK_V41,
            include_bytes!("../../../data/tokenizers/deepseek/v41.json.xz"),
            name,
        )?,
        name,
    )))
}
pub(crate) fn encoder_for(model: &str) -> Result<Option<Encoder>, LocalTokenCountError> {
    match model {
        "deepseek-v4-pro" => {
            #[cfg(feature = "tokenizer-deepseek")]
            {
                deepseek_v4()
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
                deepseek_v41()
            }
            #[cfg(not(feature = "tokenizer-deepseek"))]
            {
                Err(LocalTokenCountError::FeatureDisabled {
                    feature: "tokenizer-deepseek".into(),
                })
            }
        }
        _ => Ok(None),
    }
}
