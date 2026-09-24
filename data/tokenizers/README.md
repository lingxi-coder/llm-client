# Bundled local tokenizers

These assets are embedded at compile time by `src/client/token_count.rs` and
decoded from XZ locally when first used. The supported mapping is deliberately
narrow: only model IDs checked against an available tokenizer asset are enabled.
Each row records both the SHA-256 of the decoded tokenizer JSON and the SHA-256
of the compressed file distributed in this directory.

| Provider / model ID | Bundled tokenizer | Source and license |
| --- | --- | --- |
| `openai` / IDs recognized by `tiktoken-rs` | `tiktoken-rs` 0.12.0 embedded encodings (`o200k_base`, `o200k_harmony`, `cl100k_base`, `p50k_base`, `p50k_edit`, `r50k_base`) | [OpenAI tiktoken](https://github.com/openai/tiktoken); crate includes the encoding assets |
| `deepseek` / `deepseek-v4-pro` | `deepseek/v4.json.xz` | [DeepSeek tokenizer assets](https://github.com/deepseek-ai/deepseek-recipe); MIT, see `licenses/deepseek.txt`; decoded JSON SHA-256 `97d2f31b020d18b5aee5c9b3d5b4efb10ea210f3fe3f7dffe3f1cd90542d6b19`; XZ SHA-256 `fe71b248b5f34fbc81edfe9edbf3f5978b33758bb5c470fa3742571efdefd150` |
| `deepseek` / `deepseek-flash` | `deepseek/v41.json.xz` | [DeepSeek tokenizer assets](https://github.com/deepseek-ai/deepseek-recipe); MIT, see `licenses/deepseek.txt`; decoded JSON SHA-256 `81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99`; XZ SHA-256 `c17cf537900df5a0750659b1ef3974c5ad59e72c2feeb0314a697a74f147f08b` |
| `qwen` / `qwen3.8-flash`, `qwen3.8-max` | `qwen/qwen3.8.json.xz` (Qwen3.8-27B tokenizer) | [Qwen3.8-27B tokenizer](https://huggingface.co/Qwen/Qwen3.8-27B); Apache-2.0, see `licenses/qwen3.8-27b.txt`; decoded JSON SHA-256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`; XZ SHA-256 `3a76bc61d464c2d7da149dfd347b7023410cc4fb9813b0c321cc4a21d0cb696a` |
| `kimi` / `kimi-k3` | `kimi/k3.json.xz` (fast-tokenizer JSON conversion) | [Kimi K3 tokenizer conversion](https://huggingface.co/Xenova/Kimi-K3-tokenizer), based on the [official Moonshot Kimi K3 tokenizer](https://huggingface.co/moonshotai/Kimi-K3); Kimi K3 license, see `licenses/kimi-k3.txt`; decoded JSON SHA-256 `b55c4532c501114da9a8891b77d244fa32eee2ace2bd51abb5dc4fb156f60eb9`; XZ SHA-256 `a34e8b3afa6427819ef7577d829ca612e68ba4a50aac12e7fccc591c15149a94` |
| `zhipu` / `glm-5` | `glm/glm5.json.xz` | [Official GLM-5 repository](https://github.com/zai-org/GLM-5); Apache-2.0 per its `LICENSE` file, see `licenses/glm5.txt` (the Hugging Face model card metadata labels the model MIT); decoded JSON SHA-256 `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`; XZ SHA-256 `973c8f362503ea3236d3b51d585b0a302a13375c55e309efce8e4e08a4ef1427` |

MiniMax M3 is intentionally unsupported. Its upstream tokenizer asset is
licensed for non-commercial use, so it is not included in this library.

The decoded JSON files total about 65 MB; the XZ files total about 8.7 MiB.
They remain bundled so estimates work offline. The decoder is `xz2` with its
statically built liblzma backend. Tokenizer outputs cover text segmentation
only; request framing is estimated and service-side or multimodal content may
be omitted. Provider-reported usage remains authoritative for billing.
