# 推理控制、能力与服务档位价格

[English](inference.en.md)

`effort` 控制模型投入的推理量，可能改变实际 token 用量；它不是单价维度。价格按模型、标准／fast 服务档位及官方声明的上下文、提交方式或时段条件选择。Fast 倍率只应用到对应模型明确列出的计费桶，不存在全局统一倍率。

[完整可编译示例](../examples/inference_pricing.rs) 展示查询、参数选择、请求及费用读取；运行它会使用 `OPENAI_API_KEY` 发送一次实际付费请求。

## 查询能力和价格

`providers().info.features` 描述当前连接与协议；`models().info.features` 是该连接上具体模型的有效能力。`Supported / Unsupported / Unknown` 区分支持、不支持和信息缺失。预算范围、动态预算、thinking 模式、允许的 effort、默认值、访问条件和文档来源均可查询。已支持的 API 仍可能要求账号获得预览权限。

`mode_support` 保存按模式报告的支持状态、effort 和强制工具调用限制；`effort.level_support` 保存按等级报告的支持状态。使用 `features.supports_mode(mode)`、`features.supports_effort(Some(mode), effort)` 和 `features.forced_tool_choice(mode)`，即可与请求校验共用有效能力。目录缺失字段保留原有事实。Anthropic Models API 会同步这些局部观测；OpenRouter 显式返回 `supported_efforts: null` 时，将旧列表更新为全部可用的 gateway effort。

`effort.with_disabled_thinking` 表示显式关闭 thinking 时能否同时指定非 `none` 的 effort，包含当前连接所配置协议的限制：OpenAI 不允许该组合，Anthropic 则按模型限制判断。显式 enabled 或 adaptive thinking 不能与 `effort: none` 共用。能力查询与请求校验共用这些组合规则；不指定 effort 时保留服务端在所选模式下的行为。清除模型覆盖后，连接和协议限制仍然有效。

例如，Claude Opus 5 关闭 thinking 时仅允许 low／medium／high effort；Fable 5 和 Fable 5.1 不能关闭 thinking。手动 thinking 不允许强制工具调用，adaptive 模式按模型限制判断，Fable 5.1 属于不允许的情况。这些规则均公开于能力信息并用于请求校验。[Claude thinking 文档](https://platform.claude.com/docs/en/build-with-claude/thinking)。

`ModelProfile.pricing` 是配置价格的唯一来源，运行时 `info.pricing` 是它的只读投影。使用 `ModelField::Pricing` 修改价格，使用 `ModelField::InferenceFeatures` 修改能力。目录观测只覆盖明确提供的字段，缺省字段不会抹去内置事实；用户覆盖优先。

```rust,no_run
use lingxi_llm_client::{LlmClient, protocol::*};

fn inspect(client: &LlmClient) -> Result<(), LlmError> {
    for model in client.models() {
        println!("{}: {:?}", model.request_model, model.info.features);
    }
    let quote = client.price_quote("claude-opus-5", Some("anthropic"),
        &PricingContext {
            service_tier: Some(ServiceTier::Fast),
            input_tokens: Some(2000),
            ..Default::default()
        })?;
    println!("{:?} {} {:?}", quote.status, quote.currency, quote.rates);
    println!("multiplier: {:?}; source: {:?}", quote.multiplier, quote.source);
    Ok(())
}
```

`PriceQuote` 返回 `Priced / Unknown / Unsupported`。每个费率仍是可选值：实际消耗的计费桶缺少价格时，费用估算失败。区间价格需要完整提示长度，包括缓存输入。金额单位是报价币种／百万 token；例如 MiniMax 国内站使用 CNY，国际站使用 USD，不进行隐式汇率转换。订阅配额倍率在 `quota` 中返回，不能解释为美元金额或免费 token。

规则的 `valid_from`（含）和 `valid_until`（不含）保存 `PriceBoundary { local, time_zone }`，例如 `local = "2027-01-01T00:00:00"`、`time_zone = "America/Los_Angeles"`。使用 IANA 规则处理夏令时，并转换为 UTC 与 `PricingContext.unix_seconds` 比较，不使用运行机器的时区。重复或不存在的当地时间需提供有效的 RFC 3339 offset；非法时区在配置时拒绝。官方未声明时区且当地时间未带 offset 时，在 UTC−12 到 UTC+14 可能产生的切换窗口内返回价格未知。报价保留来源时区、匹配规则及核验日期。`PricingContext` 未指定档位时按标准档位预估。

## 设置请求

```rust,no_run
use lingxi_llm_client::protocol::*;

fn request() -> CompletionRequest {
    let mut request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "claude-opus-5",
        "messages": [{"role":"user", "content":[{"type":"text", "text":"Explain this design."}]}],
        "max_tokens": 8192
    })).unwrap();
    request.thinking = Some(ThinkingConfig {
        mode: Some(ThinkingMode::Adaptive),
        effort: Some(ReasoningEffort::High),
        ..Default::default()
    });
    request.service_tier = Some(ServiceTier::Fast);
    request
}
```

数值预算使用 `ThinkingBudget::Tokens(n)`，动态预算使用 `ThinkingBudget::Dynamic`，关闭思考使用 `ThinkingMode::Disabled`。这些设置必须分别被模型支持。`max_tokens` 是独立的输出限制；思考预算通常只是目标，不代表确定费用。

| 连接 | 控制映射 |
| --- | --- |
| OpenAI | Chat 的 `reasoning_effort`、Responses 的 `reasoning.effort`；fast 服务档位 |
| Anthropic | 按模型区分 manual budget 与 adaptive/effort；fast speed 和 beta header |
| Gemini | `thinkingBudget`（动态为 -1，关闭为 0）、`thinkingLevel`；`serviceTier: "priority"` |
| Qwen | Chat 的 `enable_thinking/thinking_budget`；Responses 的 effort |
| DeepSeek | thinking 和 effort；不会发送会被忽略的数值预算 |
| Kimi、GLM/Z.AI | 模型／协议声明的 thinking 与 effort |
| MiniMax | 模型声明的 thinking；priority 准入档位 |
| Grok | 模型声明的 effort；Chat／Responses 的 priority |
| OpenRouter | `reasoning` 对象；priority 独立于吞吐量路由 |
| Copilot | 未核实的直接 HTTP 控制保持未知，不能从产品 UI 推定支持 |

Highspeed／Turbo 是独立模型，客户端不自动替换模型。明确不支持的控制返回 `UnsupportedCapability`；预算越界、矛盾设置及 `extra.body` 冲突返回 `InvalidRequest`。同一字段不能在 typed 参数和 `extra.body` 中设置不同值。每个 failover 尝试都保留控制参数并重新验证，不静默降级。自定义 codec 可通过 `WireCodec::validate_request` 在上传前验证自己的控制。

Gemini 的档位请求使用 `serviceTier`。原生 `serviceTier` 与 protobuf 的 `service_tier` 拼写会先校验冲突，最终只发送一个 `serviceTier`。完整和流式响应均保留 `usageMetadata.serviceTier` 中的实际档位及原始值。

## 实际档位和费用

完整响应的 `inference`、流的 `inference_report()`、`StreamEvent::Inference` 及结束事件保留请求 effort、请求档位、服务端回报及原始字符串。请求值不等于已确认值。Gemini 的 `usageMetadata.serviceTier`、档位响应头和其他协议的 JSON/SSE 回报均被保留；服务端未报告的值保持未知。

```rust,no_run
use lingxi_llm_client::{LlmClient, ResolvedRoute, ModelStream, protocol::*};

fn report(client: &LlmClient, route: &ResolvedRoute, response: &CompletionResponse,
          stream: &ModelStream) -> Result<(), LlmError> {
    if let Some(cost) = response.usage.usage.and_then(|usage| usage.cost) {
        println!("provider reported nano-USD: {}", cost.nano_usd);
    }
    let cost = client.estimate_actual_cost(route, response, Submission::Interactive)?;
    println!("token estimate: {cost:?}; inference: {:?}", response.inference);
    let streamed_cost = client.estimate_stream_cost(route, stream, Submission::Interactive)?;
    println!("stream token estimate: {streamed_cost:?}");
    Ok(())
}
```

`estimate_cost` 用于请求前按所选档位预估；`PricingContext.service_tier` 未设置时，报价和预估采用已声明的模型或连接默认档位（包括 Fast），显式档位优先。`estimate_actual_cost` 和 `estimate_stream_cost` 使用实际执行连接、完整用量及实际档位。Fast 请求缺少实际档位，或实际档位的价格未知，返回 `CostUnavailable`，不套用标准价。推理 token 作为输出子集计算一次；一小时缓存写入从总缓存写入中拆出，使用独立费率。估算覆盖 token 费用，不包括未计入费率表的托管工具费、缓存存储费等；provider 实报费用独立保留。

费用计算使用成功请求的本地发送时间 `inference.executed_at`，避免稍后查询时跨越价格时段；它不是服务端确认的时间。时段价格缺少发送时间时返回 `CostUnavailable`。

实际费用估算遇到配置默认 Fast 且服务端未回报档位时返回 `CostUnavailable`。Fast 倍率继承 Standard 限时价格时，同样必须保留执行时间。OpenAI Responses 的缓存写入 token 从普通输入中拆出，按写入单价计算，并校验缓存读取与写入之和不超过输入总量。

## API 迁移

- `ThinkingConfig::budget_tokens` 改为 `budget: Option<ThinkingBudget>`，并增加 `mode/effort`；请求增加 `service_tier`。
- 响应和流结束事件增加 `inference`；流增加非文本的 `Inference` 事件，穷尽匹配需处理它。
- `CostEstimate` 使用 `currency` 和 `input_cost/output_cost/cache_read_cost/cache_write_cost/reasoning_cost/total_cost`，不再把所有币种标为 USD。
- `estimate_cost` 必须接收 `PricingContext`，返回 `CostEstimate`；未知价格返回 `CostUnavailable`。`estimate_cost_for_profile` 必须提供 `InferenceReport`。
- 配置使用 v2 的显式覆盖机制；不提供旧版本字段别名、自动迁移或旧 API 包装。数值预算 JSON 必须写为 `{"budget":{"tokens":2048}}`，不接受 `budget_tokens` 或裸整数。
- 旧布尔 `capabilities` 已删除；模型、列表和路由统一使用三态 `capability_support`。

- `TokenPricing::at` 和公开的 `pricing::estimate` 已移除。请使用客户端的价格查询和估算方法，通过 `PricingContext` 统一选择上下文、档位、提交方式与时段规则。
