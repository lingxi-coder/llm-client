//! OpenAI model mapping and optional tiktoken backend.
use crate::token_count::{backends::Encoder, LocalTokenCountError};
// Membership matches the pinned tiktoken-rs 0.12 mapping; kept asset-free so
// disabled features can be distinguished from unsupported model IDs.
pub(crate) fn known_openai_model(model: &str) -> bool {
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
#[cfg(feature = "tokenizer-openai")]
pub(crate) fn openai_encoder(model: &str) -> Option<Encoder> {
    use tiktoken_rs::tokenizer::{get_tokenizer, Tokenizer as OpenAiTokenizer};

    let tokenizer = get_tokenizer(model)?;
    let (bpe, name) = match tokenizer {
        OpenAiTokenizer::O200kBase => (tiktoken_rs::o200k_base_singleton(), "o200k_base"),
        OpenAiTokenizer::O200kHarmony => (tiktoken_rs::o200k_harmony_singleton(), "o200k_harmony"),
        OpenAiTokenizer::Cl100kBase => (tiktoken_rs::cl100k_base_singleton(), "cl100k_base"),
        OpenAiTokenizer::P50kBase => (tiktoken_rs::p50k_base_singleton(), "p50k_base"),
        OpenAiTokenizer::P50kEdit => (tiktoken_rs::p50k_edit_singleton(), "p50k_edit"),
        OpenAiTokenizer::R50kBase | OpenAiTokenizer::Gpt2 => {
            (tiktoken_rs::r50k_base_singleton(), "r50k_base")
        }
    };
    Some(Encoder {
        name,
        count: Box::new(move |text| Ok(bpe.count_ordinary(text) as u64)),
    })
}

pub(crate) fn encoder_for(model: &str) -> Result<Option<Encoder>, LocalTokenCountError> {
    if !known_openai_model(model) {
        return Ok(None);
    }
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
