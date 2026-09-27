# Qwen Chat 提示缓存

[English](qwen-prompt-cache.en.md)

Qwen 的 Chat Completions 可使用 `ChatRequest.prompt_cache` 设置显式消息断点。依据 2026-09-25 核实的[百炼 Context Cache 契约](https://help.aliyun.com/en/model-studio/context-cache)，标记位于消息内容块，格式为 `cache_control: {"type":"ephemeral"}`，有效期固定为 5 分钟，命中续期。自动缓存由服务端管理，无启用参数、关闭开关或调用方 TTL；两种模式不叠加。

```rust
use lingxi_llm_client::protocol::{CacheBreakpoint, CachePosition, CacheTtl};

# fn configure(request: &mut lingxi_llm_client::protocol::ChatRequest) {
request.prompt_cache.breakpoints = vec![
    CacheBreakpoint {
        position: CachePosition::System { index: 0 },
        ttl: CacheTtl::FiveMinutes,
    },
    CacheBreakpoint {
        position: CachePosition::Message { index: 0, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    },
];
# }
```

只需自动缓存时，保留默认空策略。`automatic: Some(...)`、一小时 TTL 和独立 Tool 定义断点会被拒绝；工具定义由供应商纳入系统前缀，不能独立标记。请求最多四个唯一断点；位置必须指向非空 system/text/tool-result 或 image 内容块，不能指向 thinking、tool-use、文档或越界索引。

编码保留原始 system 分块与分隔符，以及 user/assistant/tool result 的位置。单条文本被标记时转换为 content 数组；工具结果拆成独立 wire 消息后仍保留正确的 call ID 和断点。调用方请求不会被修改。

## 地域、部署范围与预检

显式策略只接受已核实的百炼 HTTPS Chat 路由，使用实际 request model ID 的明确列表。北京默认 `china_mainland`，新加坡默认 `international`；其他地域必须在 profile 的 `extra.qwen_cache_deployment_scope` 指定实际部署范围：

| 地域 | 可指定的部署范围 |
| --- | --- |
| 美国 Virginia | `global`、`us` |
| 中国香港 | `global`、`hong_kong` |
| 德国 Frankfurt | `global`、`eu` |
| 日本 Tokyo | `global`、`japan` |

例如美国 Global 部署可设置 `extra = { qwen_cache_deployment_scope = "global" }`。这是本库的校验配置，不会发送到供应商。不同范围的模型列表不会合并；未知模型、别名、代理路由与未确认组合在发送前报错。配置不能授予账户权限，调用方仍需使用对应区域和部署范围的凭证。

校验在附件解析、上传、认证和发送之前执行。未经类型化策略管理的顶层 `extra.body.cache_control`、`prompt_cache_options`、`prompt_cache_key`、`prompt_cache_retention` 会被拒绝。这里没有接入 Responses session cache 或其他厂商的同名字段。

供应商的 1,024 token 门槛和 20 内容块回溯窗口影响命中率。客户端不以字符数估计门槛，也不保证创建或命中缓存。

## 用量与验证

Chat 的 `prompt_tokens` 包含读取与创建缓存 token。`prompt_tokens_details.cached_tokens` 映射为读取，`cache_creation_input_tokens` 映射为写入，普通输入减去这两项。例如总输入 2,000、读取 1,200、写入 500，普通输入为 300。写入一小时计数始终为零；缺失价格仍未知。

Qwen 流式请求自动携带 `stream_options.include_usage=true`；与之冲突的配置在预检失败。同步与流式共享用量归一化；后续明确的零写入计数覆盖早期值，超界、非数值或冲突计数不会成为完整可计费报告。

`tests/qwen_prompt_cache.rs` 使用离线 fixture、分片 SSE 和禁止副作用的 mock 验证上述行为，并检查编码长度与实际字节数一致。没有调用真实账户，文档支持不等于线上验收。
