//! Compare the bundled tokenizer's full encoding with its count-only fast path.
//!
//! `cargo run --release --all-features --example tokenizer_count_bench -- 11`
//! Asset loading, input construction, equality checks and warmup are untimed.
//! Each round alternates which encoder runs first; output is the median of the
//! individual warm calls. These measurements exclude client request preparation.

#[cfg(any(
    feature = "tokenizer-deepseek",
    feature = "tokenizer-qwen",
    feature = "tokenizer-kimi",
    feature = "tokenizer-glm"
))]
mod benchmark {
    use std::{hint::black_box, io::Read, time::Instant};
    use tokenizers::Tokenizer;
    use xz2::read::XzDecoder;

    type Error = Box<dyn std::error::Error + Send + Sync>;

    pub fn run() -> Result<(), Error> {
        let rounds = std::env::args()
            .nth(1)
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(11);
        if rounds == 0 {
            return Err("round count must be positive".into());
        }
        let inputs = [
            (
                "multilingual",
                "并行请求应共享不变配置，同时保留每次请求的模型与推理设置。日本語と한국어。\n".repeat(512),
            ),
            (
                "code",
                "fn update(state: &mut HashMap<String, Value>, key: &str, value: Value) {\n    state.insert(key.to_owned(), value);\n}\n".repeat(512),
            ),
            (
                "tool_schema",
                serde_json::to_string(
                    &(0..256)
                        .map(|index| {
                            serde_json::json!({
                                "name": format!("search_{index}"),
                                "description": "Search indexed documents by query and return matching excerpts.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "搜索内容"},
                                        "limit": {"type": "integer", "minimum": 1, "maximum": 100}
                                    },
                                    "required": ["query"],
                                    "additionalProperties": false
                                }
                            })
                        })
                        .collect::<Vec<_>>(),
                )?,
            ),
        ];
        println!("tokenizer,input,bytes,tokens,rounds,encode_p50_ns,fast_p50_ns");
        for (name, asset) in [
            #[cfg(feature = "tokenizer-deepseek")]
            ("deepseek-v4", "data/tokenizers/deepseek/v4.json.xz"),
            #[cfg(feature = "tokenizer-deepseek")]
            ("deepseek-v41", "data/tokenizers/deepseek/v41.json.xz"),
            #[cfg(feature = "tokenizer-qwen")]
            ("qwen3.8", "data/tokenizers/qwen/qwen3.8.json.xz"),
            #[cfg(feature = "tokenizer-kimi")]
            ("kimi-k3", "data/tokenizers/kimi/k3.json.xz"),
            #[cfg(feature = "tokenizer-glm")]
            ("glm5", "data/tokenizers/glm/glm5.json.xz"),
        ] {
            let file =
                std::fs::File::open(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(asset))?;
            let mut json = Vec::new();
            XzDecoder::new(file).read_to_end(&mut json)?;
            let tokenizer = Tokenizer::from_bytes(&json)?;
            drop(json);
            for (input_name, input) in &inputs {
                let ordinary = tokenizer.encode(input.as_str(), false)?;
                let fast = tokenizer.encode_fast(input.as_str(), false)?;
                assert_eq!(ordinary.get_ids(), fast.get_ids(), "{name}/{input_name}");
                let tokens = ordinary.get_ids().len();
                drop((ordinary, fast));
                for _ in 0..2 {
                    timed_count(&tokenizer, input, false)?;
                    timed_count(&tokenizer, input, true)?;
                }
                let mut ordinary = Vec::with_capacity(rounds);
                let mut fast = Vec::with_capacity(rounds);
                for round in 0..rounds {
                    for use_fast in if round % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let elapsed = timed_count(&tokenizer, input, use_fast)?;
                        if use_fast {
                            fast.push(elapsed);
                        } else {
                            ordinary.push(elapsed);
                        }
                    }
                }
                ordinary.sort_unstable();
                fast.sort_unstable();
                println!(
                    "{name},{input_name},{},{tokens},{rounds},{},{}",
                    input.len(),
                    ordinary[rounds / 2],
                    fast[rounds / 2]
                );
            }
        }
        Ok(())
    }

    fn timed_count(tokenizer: &Tokenizer, input: &str, use_fast: bool) -> Result<u128, Error> {
        let start = Instant::now();
        let tokens = if use_fast {
            tokenizer.encode_fast(black_box(input), false)?
        } else {
            tokenizer.encode(black_box(input), false)?
        }
        .get_ids()
        .len();
        black_box(tokens);
        Ok(start.elapsed().as_nanos())
    }
}

#[cfg(any(
    feature = "tokenizer-deepseek",
    feature = "tokenizer-qwen",
    feature = "tokenizer-kimi",
    feature = "tokenizer-glm"
))]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    benchmark::run()
}

#[cfg(not(any(
    feature = "tokenizer-deepseek",
    feature = "tokenizer-qwen",
    feature = "tokenizer-kimi",
    feature = "tokenizer-glm"
)))]
fn main() {
    eprintln!("Enable a bundled tokenizer feature, for example --all-features.");
}
