//! Optional provider tokenizer mappings and bundled assets.
#[cfg(feature = "tokenizer-kimi")]
use crate::token_count::{assets::load, backends::bundled_encoder};
use crate::token_count::{backends::Encoder, LocalTokenCountError};
#[cfg(feature = "tokenizer-kimi")]
use std::sync::OnceLock;
#[cfg(feature = "tokenizer-kimi")]
use tokenizers::Tokenizer;
#[cfg(feature = "tokenizer-kimi")]
pub(crate) static KIMI_K3: OnceLock<Result<Tokenizer, String>> = OnceLock::new();
#[cfg(feature = "tokenizer-kimi")]
pub(crate) fn kimi_k3() -> Result<Option<Encoder>, LocalTokenCountError> {
    let name = "Kimi K3 tokenizer (verified fast-tokenizer conversion)";
    Ok(Some(bundled_encoder(
        load(
            &KIMI_K3,
            include_bytes!("../../../data/tokenizers/kimi/k3.json.xz"),
            name,
        )?,
        name,
    )))
}
pub(crate) fn encoder_for(model: &str) -> Result<Option<Encoder>, LocalTokenCountError> {
    match model {
        "kimi-k3" => {
            #[cfg(feature = "tokenizer-kimi")]
            {
                kimi_k3()
            }
            #[cfg(not(feature = "tokenizer-kimi"))]
            {
                Err(LocalTokenCountError::FeatureDisabled {
                    feature: "tokenizer-kimi".into(),
                })
            }
        }
        _ => Ok(None),
    }
}
