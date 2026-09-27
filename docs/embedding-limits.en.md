# Embedding Model Dimensions and Batch Limits

The client checks these documented model limits before sending a request. An over-limit request fails without automatic splitting; callers should split inputs into batches themselves.

| Route and model | Maximum inputs per request | Supported requested dimensions |
| --- | ---: | --- |
| First-party OpenAI `text-embedding-3-small` | 2048 | `dimensions` minimum is 1; defaults to 1536 when omitted |
| First-party OpenAI `text-embedding-3-large` | 2048 | `dimensions` minimum is 1; defaults to 3072 when omitted |
| First-party OpenAI `text-embedding-ada-002` | 2048 | The `dimensions` parameter is unsupported |
| Qwen `qwen3.7-text-embedding` | 20 | 256, 512, 768, 1024, 1536, 2048, 2560 |
| Qwen `qwen3.7-text-embedding-flash` | 20 | 256, 512, 768, 1024 |
| Qwen `text-embedding-v4` | 10 | 64, 128, 256, 512, 768, 1024, 1536, 2048 |
| Qwen `text-embedding-v3` | 10 | 64, 128, 256, 512, 768, 1024 |
| Qwen `text-embedding-v1`, `text-embedding-v2` | 25 | Custom dimensions are not supported by this client parameter |
| GLM `embedding-3` | 64 | 256, 512, 1024, 2048 |

Qwen limits apply to the native Qwen Embeddings adapter for `provider_id = "qwen"` and the exact model IDs above. GLM limits apply to `embedding-3` for `provider_id = "zhipu"` on the OpenAI-format route `https://open.bigmodel.cn/api/paas/v4/embeddings`. Other OpenAI-compatible services do not inherit these model limits or generic OpenAI assumptions; they follow only an explicitly configured route `max_inputs`.

OpenAI limits apply only when `provider_id = "openai"` uses a first-party OpenAI HTTPS `/v1/embeddings` host. The API also limits each input to 8,192 tokens and the total request to 300,000 tokens. This string-based client path does not tokenize inputs locally, so OpenAI validates those token limits. The `dimensions` parameter is supported for `text-embedding-3` and later models; the API reference specifies a minimum of 1, while the provider validates each model's upper bound. The listed default dimensions describe the response size when the parameter is omitted; the client does not fill it in.

`list_openai_models` uses the separate `GET /v1/models` route and includes only the three IDs explicitly named by the Embeddings API reference in its typed `models` list. OpenAI's general directory does not label model capabilities; its full raw response, including other and unknown IDs, remains in `native`, so those IDs are not thereby classified as lacking embedding support. OpenAI, OpenRouter, and Gemini directories remain separate from the Chat model directory.

Qwen v1/v2 do not accept the client's `dimension` parameter: the upstream API documents that parameter only for qwen3.7-text-embedding, v3, and v4. The dimension lists describe output settings documented for each model. The client does not infer a default output dimension when one was not requested. Providers remain responsible for validating per-input token lengths; the local batch check limits only the number of input items.

References: [OpenAI Embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create), [OpenAI embedding guide](https://developers.openai.com/api/docs/guides/embeddings), [OpenAI Models API](https://developers.openai.com/api/reference/resources/models), [OpenAI regional endpoint guide](https://developers.openai.com/api/docs/guides/your-data), [Alibaba Cloud Model Studio embedding models](https://help.aliyun.com/en/model-studio/embedding), [Alibaba Cloud Model Studio synchronous text embedding API](https://help.aliyun.com/en/model-studio/text-embedding-synchronous-api), [Zhipu Embedding-3 documentation](https://docs.bigmodel.cn/cn/guide/models/embedding/embedding-3).
