//! Feature-gated tokenizer assets and their one-time decompression.
#[cfg(test)]
use super::backends::bundled_encoder;
use super::LocalTokenCountError;
use std::{io::Read, sync::OnceLock};
use tokenizers::Tokenizer;
use xz2::read::XzDecoder;

pub(crate) fn load(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_fast_count_matches(
        slot: &'static OnceLock<Result<Tokenizer, String>>,
        asset: &[u8],
        name: &'static str,
    ) {
        // Reuse the production slot so these comparisons do not retain a second
        // copy of a large tokenizer when other token-count tests also use it.
        let tokenizer = load(slot, asset, name).unwrap();
        let encoder = bundled_encoder(tokenizer, name);
        for text in [
            "",
            "hello world",
            "中文分词测试。日本語の文章です。한국어 문장입니다. العربية हिन्दी",
            "fn main() {\n    let json = r#\"{\"key\": [1, true, null]}\"#;\n    println!(\"{json}\");\n}\n",
            r#"{"type":"object","properties":{"query":{"type":"string","description":"搜索内容"},"limit":{"type":"integer","minimum":1}},"required":["query"],"additionalProperties":false}"#,
            "<|endoftext|><|im_start|>assistant<|im_end|>[CLS][SEP]<think>思考</think>",
            "\u{feff}e\u{301}é👩\u{200d}💻🇨🇳\u{200b}\0\r\n\t a\u{a0}b\u{2028}c\u{10ffff}",
        ] {
            let ordinary = tokenizer.encode(text, false).unwrap();
            let fast = tokenizer.encode_fast(text, false).unwrap();
            assert_eq!(fast.get_ids(), ordinary.get_ids(), "{name}: {text:?}");
            assert_eq!(
                encoder.count(text).unwrap(),
                ordinary.get_ids().len() as u64,
                "count {name}: {text:?}"
            );
        }
    }

    #[cfg(feature = "tokenizer-deepseek")]
    #[test]
    fn deepseek_v4_fast_count_matches() {
        assert_fast_count_matches(
            &crate::providers::deepseek::token_count::DEEPSEEK_V4,
            include_bytes!("../../data/tokenizers/deepseek/v4.json.xz"),
            "deepseek-v4",
        );
    }

    #[cfg(feature = "tokenizer-deepseek")]
    #[test]
    fn deepseek_v41_fast_count_matches() {
        assert_fast_count_matches(
            &crate::providers::deepseek::token_count::DEEPSEEK_V41,
            include_bytes!("../../data/tokenizers/deepseek/v41.json.xz"),
            "deepseek-v41",
        );
    }

    #[cfg(feature = "tokenizer-qwen")]
    #[test]
    fn qwen38_fast_count_matches() {
        assert_fast_count_matches(
            &crate::providers::qwen::token_count::QWEN38,
            include_bytes!("../../data/tokenizers/qwen/qwen3.8.json.xz"),
            "qwen3.8",
        );
    }

    #[cfg(feature = "tokenizer-kimi")]
    #[test]
    fn kimi_k3_fast_count_matches() {
        assert_fast_count_matches(
            &crate::providers::kimi::token_count::KIMI_K3,
            include_bytes!("../../data/tokenizers/kimi/k3.json.xz"),
            "kimi-k3",
        );
    }

    #[cfg(feature = "tokenizer-glm")]
    #[test]
    fn glm5_fast_count_matches() {
        assert_fast_count_matches(
            &crate::providers::zhipu::token_count::GLM5,
            include_bytes!("../../data/tokenizers/glm/glm5.json.xz"),
            "glm5",
        );
    }
}
