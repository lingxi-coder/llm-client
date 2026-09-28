//! Optional provider tokenizer mappings and bundled assets.
#[cfg(feature = "tokenizer-qwen")]
use crate::token_count::{assets::load, backends::bundled_encoder};
use crate::token_count::{backends::Encoder, LocalTokenCountError};
#[cfg(feature = "tokenizer-qwen")]
use std::sync::OnceLock;
#[cfg(feature = "tokenizer-qwen")]
use tokenizers::Tokenizer;
#[cfg(feature = "tokenizer-qwen")]
pub(crate) static QWEN38: OnceLock<Result<Tokenizer, String>> = OnceLock::new();
#[cfg(feature = "tokenizer-qwen")]
pub(crate) fn qwen38() -> Result<Option<Encoder>, LocalTokenCountError> {
    let name = "Qwen 3.8 tokenizer (Qwen3.8-27B asset)";
    Ok(Some(bundled_encoder(
        load(
            &QWEN38,
            include_bytes!("../../../data/tokenizers/qwen/qwen3.8.json.xz"),
            name,
        )?,
        name,
    )))
}
pub(crate) fn encoder_for(model: &str) -> Result<Option<Encoder>, LocalTokenCountError> {
    match model {
        "qwen3.8-flash" => {
            #[cfg(feature = "tokenizer-qwen")]
            {
                qwen38()
            }
            #[cfg(not(feature = "tokenizer-qwen"))]
            {
                Err(LocalTokenCountError::FeatureDisabled {
                    feature: "tokenizer-qwen".into(),
                })
            }
        }
        "qwen3.8-max" => {
            #[cfg(feature = "tokenizer-qwen")]
            {
                qwen38()
            }
            #[cfg(not(feature = "tokenizer-qwen"))]
            {
                Err(LocalTokenCountError::FeatureDisabled {
                    feature: "tokenizer-qwen".into(),
                })
            }
        }
        _ => Ok(None),
    }
}
