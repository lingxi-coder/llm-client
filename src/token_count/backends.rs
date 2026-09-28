use super::LocalTokenCountError;

type CountFn = dyn Fn(&str) -> Result<u64, LocalTokenCountError>;
pub(crate) struct Encoder {
    pub(crate) name: &'static str,
    pub(crate) count: Box<CountFn>,
}
impl Encoder {
    pub(crate) fn count(&self, text: &str) -> Result<u64, LocalTokenCountError> {
        (self.count)(text)
    }
    pub(crate) fn name(&self) -> &'static str {
        self.name
    }
}
#[cfg(any(
    feature = "tokenizer-deepseek",
    feature = "tokenizer-qwen",
    feature = "tokenizer-kimi",
    feature = "tokenizer-glm"
))]
pub(crate) fn bundled_encoder(
    tokenizer: &'static tokenizers::Tokenizer,
    name: &'static str,
) -> Encoder {
    Encoder {
        name,
        count: Box::new(move |text| {
            tokenizer
                // Counting only needs token IDs, not token text or offsets.
                .encode_fast(text, false)
                .map(|encoding| encoding.get_ids().len() as u64)
                .map_err(|e| LocalTokenCountError::Tokenization(e.to_string()))
        }),
    }
}
