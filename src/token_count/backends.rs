use super::LocalTokenCountError;

type CountFn = dyn Fn(&str) -> Result<u64, LocalTokenCountError>;
pub(super) struct Encoder {
    name: &'static str,
    count: Box<CountFn>,
}
impl Encoder {
    pub(super) fn count(&self, text: &str) -> Result<u64, LocalTokenCountError> {
        (self.count)(text)
    }
    pub(super) fn name(&self) -> &'static str {
        self.name
    }
}
#[cfg(feature = "tokenizer-openai")]
pub(super) fn openai_encoder(model: &str) -> Option<Encoder> {
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

#[cfg(any(
    feature = "tokenizer-deepseek",
    feature = "tokenizer-qwen",
    feature = "tokenizer-kimi",
    feature = "tokenizer-glm"
))]
pub(super) fn bundled_encoder(
    tokenizer: &'static tokenizers::Tokenizer,
    name: &'static str,
) -> Encoder {
    Encoder {
        name,
        count: Box::new(move |text| {
            tokenizer
                .encode(text, false)
                .map(|encoding| encoding.get_ids().len() as u64)
                .map_err(|e| LocalTokenCountError::Tokenization(e.to_string()))
        }),
    }
}
