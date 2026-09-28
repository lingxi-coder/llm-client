//! Optional offline token counting. Model support remains queryable without assets.
#[cfg(any(
    feature = "tokenizer-deepseek",
    feature = "tokenizer-qwen",
    feature = "tokenizer-kimi",
    feature = "tokenizer-glm"
))]
pub(crate) mod assets;
pub(crate) mod backends;
mod model;
mod request;
mod types;
pub use types::*;
