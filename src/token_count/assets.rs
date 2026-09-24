//! Feature-gated tokenizer assets and their one-time decompression.
use super::{
    backends::{bundled_encoder, Encoder},
    LocalTokenCountError,
};
use std::{io::Read, sync::OnceLock};
use tokenizers::Tokenizer;
use xz2::read::XzDecoder;
#[cfg(feature = "tokenizer-deepseek")]
pub(super) fn deepseek_v4() -> Result<Option<Encoder>, LocalTokenCountError> {
    static SLOT: OnceLock<Result<Tokenizer, String>> = OnceLock::new();
    let name = "DeepSeek V4 Pro (2026-09)";
    Ok(Some(bundled_encoder(
        load(
            &SLOT,
            include_bytes!("../../data/tokenizers/deepseek/v4.json.xz"),
            name,
        )?,
        name,
    )))
}
#[cfg(feature = "tokenizer-deepseek")]
pub(super) fn deepseek_v41() -> Result<Option<Encoder>, LocalTokenCountError> {
    static SLOT: OnceLock<Result<Tokenizer, String>> = OnceLock::new();
    let name = "DeepSeek V4.1 Flash (2026-09)";
    Ok(Some(bundled_encoder(
        load(
            &SLOT,
            include_bytes!("../../data/tokenizers/deepseek/v41.json.xz"),
            name,
        )?,
        name,
    )))
}
#[cfg(feature = "tokenizer-qwen")]
pub(super) fn qwen38() -> Result<Option<Encoder>, LocalTokenCountError> {
    static SLOT: OnceLock<Result<Tokenizer, String>> = OnceLock::new();
    let name = "Qwen 3.8 tokenizer (Qwen3.8-27B asset)";
    Ok(Some(bundled_encoder(
        load(
            &SLOT,
            include_bytes!("../../data/tokenizers/qwen/qwen3.8.json.xz"),
            name,
        )?,
        name,
    )))
}
#[cfg(feature = "tokenizer-kimi")]
pub(super) fn kimi_k3() -> Result<Option<Encoder>, LocalTokenCountError> {
    static SLOT: OnceLock<Result<Tokenizer, String>> = OnceLock::new();
    let name = "Kimi K3 tokenizer (verified fast-tokenizer conversion)";
    Ok(Some(bundled_encoder(
        load(
            &SLOT,
            include_bytes!("../../data/tokenizers/kimi/k3.json.xz"),
            name,
        )?,
        name,
    )))
}
#[cfg(feature = "tokenizer-glm")]
pub(super) fn glm5() -> Result<Option<Encoder>, LocalTokenCountError> {
    static SLOT: OnceLock<Result<Tokenizer, String>> = OnceLock::new();
    let name = "GLM 5 tokenizer (2026-09)";
    Ok(Some(bundled_encoder(
        load(
            &SLOT,
            include_bytes!("../../data/tokenizers/glm/glm5.json.xz"),
            name,
        )?,
        name,
    )))
}
fn load(
    slot: &'static OnceLock<Result<Tokenizer, String>>,
    bytes: &[u8],
    name: &str,
) -> Result<&'static Tokenizer, LocalTokenCountError> {
    slot.get_or_init(|| {
        let mut decoder = XzDecoder::new(bytes);
        let mut json = Vec::new();
        decoder.read_to_end(&mut json).map_err(|e| e.to_string())?;
        Tokenizer::from_bytes(&json).map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|message| LocalTokenCountError::TokenizerInitialization {
        tokenizer: name.into(),
        message: message.clone(),
    })
}
