//! Optional provider tokenizer mappings and bundled assets.
#[cfg(feature = "tokenizer-glm")]
use crate::token_count::{assets::load, backends::bundled_encoder};
use crate::token_count::{backends::Encoder, LocalTokenCountError};
#[cfg(feature = "tokenizer-glm")]
use std::sync::OnceLock;
#[cfg(feature = "tokenizer-glm")]
use tokenizers::Tokenizer;
#[cfg(feature = "tokenizer-glm")]
pub(crate) static GLM5: OnceLock<Result<Tokenizer, String>> = OnceLock::new();
#[cfg(feature = "tokenizer-glm")]
pub(crate) fn glm5() -> Result<Option<Encoder>, LocalTokenCountError> {
    let name = "GLM 5 tokenizer (2026-09)";
    Ok(Some(bundled_encoder(
        load(
            &GLM5,
            include_bytes!("../../../data/tokenizers/glm/glm5.json.xz"),
            name,
        )?,
        name,
    )))
}
pub(crate) fn encoder_for(model: &str) -> Result<Option<Encoder>, LocalTokenCountError> {
    match model {
        "glm-5" => {
            #[cfg(feature = "tokenizer-glm")]
            {
                glm5()
            }
            #[cfg(not(feature = "tokenizer-glm"))]
            {
                Err(LocalTokenCountError::FeatureDisabled {
                    feature: "tokenizer-glm".into(),
                })
            }
        }
        _ => Ok(None),
    }
}
