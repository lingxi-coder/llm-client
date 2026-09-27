# Embedding 模型维度与批次限制

客户端会在发送请求前检查以下已公开的模型限制。超限请求会直接报错，客户端不会自动拆分；调用方应按表格拆分批次。

| 路由和模型 | 每次请求最多输入 | 可请求的输出维度 |
| --- | ---: | --- |
| OpenAI 第一方 `text-embedding-3-small` | 2048 | `dimensions` 最小值为 1；省略时为 1536 |
| OpenAI 第一方 `text-embedding-3-large` | 2048 | `dimensions` 最小值为 1；省略时为 3072 |
| OpenAI 第一方 `text-embedding-ada-002` | 2048 | 不支持 `dimensions` 参数 |
| Qwen `qwen3.7-text-embedding` | 20 | 256、512、768、1024、1536、2048、2560 |
| Qwen `qwen3.7-text-embedding-flash` | 20 | 256、512、768、1024 |
| Qwen `text-embedding-v4` | 10 | 64、128、256、512、768、1024、1536、2048 |
| Qwen `text-embedding-v3` | 10 | 64、128、256、512、768、1024 |
| Qwen `text-embedding-v1`、`text-embedding-v2` | 25 | 不支持此客户端参数中的自定义维度 |
| GLM `embedding-3` | 64 | 256、512、1024、2048 |

Qwen 限制适用于 `provider_id = "qwen"` 的 Qwen 原生 Embeddings 路由和上述精确模型 ID。GLM 限制适用于 `provider_id = "zhipu"`、OpenAI 格式路由 `https://open.bigmodel.cn/api/paas/v4/embeddings` 上的 `embedding-3`。其他 OpenAI 兼容服务不会继承这些模型限制，也不会自动套用 OpenAI 限制；它们只遵循路由中显式配置的 `max_inputs`。

OpenAI 限制只适用于 `provider_id = "openai"` 且使用 OpenAI 第一方 HTTPS `/v1/embeddings` 主机的路由。API 文档还限制每个输入最多 8192 tokens、单次请求的总输入最多 300,000 tokens；此接口接收字符串，客户端不在这里做 tokenization，超出这些 token 限制时由 OpenAI 校验。`dimensions` 只适用于 `text-embedding-3` 及后续模型；API 参考规定最小值为 1，具体模型的上限仍由提供方校验。表中的默认维度仅描述省略 `dimensions` 时的输出宽度，客户端不会自行填充该参数。

`list_openai_models` 使用独立的 `GET /v1/models`，只把 Embeddings API 参考明确列出的三个 ID 放入类型化 `models` 列表。OpenAI 通用目录本身不返回能力标记；完整原始响应（包括其他和未知 ID）保留在 `native`，因此它们不会因此被判断为不支持 Embeddings。OpenAI、OpenRouter 和 Gemini 的目录操作彼此独立，不会写入 Chat 模型目录。

Qwen v1/v2 不接受客户端发送的 `dimension` 参数；上游接口将该参数限定为 qwen3.7-text-embedding、v3 和 v4。维度列表描述模型文档支持的输出设置，不代表客户端会对未请求维度的响应推断默认维度。文本 token 长度仍由提供方校验；此处的本地批次检查只限制输入条目数。

参考：[OpenAI Embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create)、[OpenAI Embedding 指南](https://developers.openai.com/api/docs/guides/embeddings)、[OpenAI Models API](https://developers.openai.com/api/reference/resources/models)、[OpenAI 区域 endpoint 文档](https://developers.openai.com/api/docs/guides/your-data)、[阿里云百炼文本向量化与模型规格](https://help.aliyun.com/zh/model-studio/embedding)、[阿里云百炼同步文本向量 API](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api/)、[智谱 Embedding-3 文档](https://docs.bigmodel.cn/cn/guide/models/embedding/embedding-3)。
