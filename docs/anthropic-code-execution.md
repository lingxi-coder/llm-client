# Anthropic Code Execution 与容器续用

[English](anthropic-code-execution.en.md)

`HostedTool::AnthropicCodeExecution` 通过第一方 Messages API，或显式声明为 Anthropic 托管的 Microsoft Foundry deployment，调用 Anthropic 托管容器。适配器发送固定工具定义 `{"type":"code_execution_20260521","name":"code_execution"}`。Code Execution 本身不需要 beta header。参见 Anthropic [Code Execution 指南](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)和[Claude on Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry)。

## 创建与续用容器

`AnthropicCodeExecutionConfig::default()` 不设置 `container`，由供应商按需创建容器。响应的 `ChatResponse.anthropic_container` 在公开字段 `envelope` 中保留完整原生容器对象。续用前，通过 `reference_for(scope)` 将 ID 明确绑定到原连接与账号；应用导入自行保存的 ID 时，也可以使用 `AnthropicContainerRef::new(id, scope)`。

以下示例专门假设客户端把 `claude-opus-5-5` 路由到已配置的第一方 Anthropic profile，并由调用方提供该账号的有效凭证。Foundry 路由与独立的容器 scope 见下文。若响应暂停、截断或要求执行客户端工具，应用应先处理该状态，再开始新的用户轮次。

```rust,no_run
use lingxi_llm_client::{
    protocol::{
        AnthropicCodeExecutionConfig, AnthropicContainerScope, ChatRequest, ChatResponse,
        ConversationMessage, HostedTool, Secret, StopReason,
    },
    LlmClient, RequestOptions,
};

# async fn example(
#     client: &LlmClient,
#     credential: Secret<String>,
# ) -> Result<ChatResponse, Box<dyn std::error::Error>> {
const MODEL: &str = "claude-opus-5-5";
const ACCOUNT: &str = "anthropic-workspace-main";
let options = RequestOptions {
    credential: Some(credential),
    account_scope: Some(ACCOUNT.into()),
    ..Default::default()
};
let mut request: ChatRequest = serde_json::from_value(serde_json::json!({
    "model": MODEL,
    "max_tokens": 4096,
    "messages": []
}))?;
request.messages.push(ConversationMessage::user_text(
    "Use code execution to write the number 37 to /tmp/number.txt.",
));
request.hosted_tools.push(HostedTool::AnthropicCodeExecution(
    AnthropicCodeExecutionConfig::default(),
));

let response = client.chat().complete(&request, &options).await?;
if response.stop_reason != StopReason::EndTurn {
    // 将该响应交给应用的轮次处理循环。
    return Ok(response);
}
let profile = response.executed_profile.as_deref()
    .ok_or("response has no executed profile")?;
let scope = AnthropicContainerScope::new(
    profile, "https://api.anthropic.com", ACCOUNT, MODEL,
)?;
let reference = response.anthropic_container.as_ref()
    .ok_or("provider returned no execution container")?
    .reference_for(scope)?;

// 保留原始 assistant 内容，包括原生执行 block。
request.messages.push(response.message);
request.messages.push(ConversationMessage::user_text(
    "Read /tmp/number.txt and use code execution to calculate its square.",
));
request.hosted_tools = vec![HostedTool::AnthropicCodeExecution(
    AnthropicCodeExecutionConfig {
        container: Some(reference),
        ..Default::default()
    },
)];
let follow_up = client.chat().complete(&request, &options).await?;
println!("{}", follow_up.message.text());
# Ok(follow_up)
# }
```

续用引用在发送前核对 profile 名称、规范化 endpoint、稳定且不含秘密的账号标识，以及精确匹配的实际请求模型。scope 应使用规范 wire model ID；目录别名会先解析再校验。不带 Skills 时，请求顶层 `container` 只发送容器 ID；带 Skills 时发送包含 `id` 和 `skills` 的对象。所有本地 scope 字段都不会进入请求。应用负责确保账号标识与实际凭证一致。`file_account_scope` 是独立选项，不能代替该账号检查。

续用必须提供 `RequestOptions.account_scope`。直接调用 codec 编码请求时，需通过 `CodecContext::with_account_scope(Some(account))` 提供同一账号。新建容器的请求可以省略账号 scope。响应 ID 或 `ChatRequest.continuation` 均不能代替 Anthropic 容器引用，也不能代替调用方需要保存的消息历史。

## Microsoft Foundry

Anthropic 当前文档允许在 Microsoft Foundry 使用 Code Execution 和程序化工具调用，但要求 deployment **Hosted on Anthropic**。Foundry resource endpoint 为 `https://{resource}.services.ai.azure.com/anthropic`，Messages 请求路径为 `/v1/messages`；body 中的 `model` 仍是自定义 deployment 名称。所选 model row 必须显式填写 `FoundryDeployment`，由客户端按 hosting 和底层 model ID 校验。`claude-mythos-preview` 文档支持 Code Execution，但这不代表它支持程序化工具调用。Azure 托管的 Foundry deployment 会在发送前被拒绝。

容器引用绑定到 Foundry resource、所选 deployment、底层 model 和调用方提供的同一账号标识。只有容器 ID 会通过顶层 `container` 字段发送：

```rust,no_run
use lingxi_llm_client::protocol::{
    AnthropicContainerRef, AnthropicContainerScope, ChatResponse, FoundryDeployment,
    FoundryHosting,
};

# fn bind(response: &ChatResponse) -> Result<AnthropicContainerRef, Box<dyn std::error::Error>> {
let scope = AnthropicContainerScope::new_foundry(
    "foundry-claude",
    "https://example-resource.services.ai.azure.com/anthropic",
    "foundry-resource-account",
    "my-opus-deployment",
    FoundryDeployment {
        hosting: FoundryHosting::Anthropic,
        model_id: "claude-opus-5-5".into(),
    },
)?;
let reference = response
    .anthropic_container
    .as_ref()
    .ok_or("provider returned no execution container")?
    .reference_for(scope)?;
# Ok(reference)
# }
```

`new_foundry` 会保留 Foundry resource endpoint，不会把它改成 `api.anthropic.com`；它会拒绝 Azure hosting。后续若更换 profile、endpoint、`account_scope`、deployment 名称、hosting 或底层 model，原引用不能续用。

## 使用 Agent Skills

Agent Skills 与 Code Execution 共用顶层 `container` 参数。Anthropic API 每个请求最多接受 20 个 Skill；每条引用包含来源（`anthropic` 或 `custom`）、`skill_id` 和可选版本。Skills API 已正式发布，客户端不会附加旧版 `skills-2025-10-02` beta header。Skills 仍要求启用 Code Execution，并使用其文档列出的兼容模型。预置 Skills 和自定义 Skills 均支持符合条件的 Anthropic-hosted Foundry deployment；自定义 Skill 通过独立 Skills 服务上传和管理，并绑定 Foundry resource 与 account。Foundry 不支持下载 Skill version content。

```rust,no_run
use lingxi_llm_client::protocol::{
    AnthropicCodeExecutionConfig, AnthropicSkillRef, AnthropicSkillScope, ChatRequest, HostedTool,
};

# fn add_skills(request: &mut ChatRequest) -> Result<(), Box<dyn std::error::Error>> {
let workspace = AnthropicSkillScope::new(
    "anthropic",
    "https://api.anthropic.com",
    "anthropic-workspace-main",
)?;
let mut execution = AnthropicCodeExecutionConfig::default();
execution.skills = vec![
    AnthropicSkillRef::anthropic("pptx").with_version("latest"),
    AnthropicSkillRef::custom("skill_01AbCdEfGhIjKlMnOpQrStUv", workspace)
        .with_version("skver_01AbCdEfGhIjKlMnOpQrStUv"),
];
request
    .hosted_tools
    .push(HostedTool::AnthropicCodeExecution(execution));
# Ok(())
# }
```

Anthropic 托管 Skill ID 包括 `pptx`、`xlsx`、`docx` 和 `pdf`；可指定 8 位目录版本或 `latest`。自定义 ID 必须先通过独立 [Skills 资源服务](anthropic-skills.md) 上传。第一方 API 使用 `AnthropicSkillScope::new`，Foundry 则使用 `AnthropicSkillScope::new_foundry`，且只接受显式 Anthropic hosting；自定义引用会校验对应 profile 和 account scope。Foundry 引用绑定 resource 与 account，不绑定单个 deployment。每个 resource 或 workspace 应使用不同且稳定的 `account_scope`，避免把自定义 ID 错发到其他身份。自定义版本接受当前 `skver_…` ID、旧版 `skill_version_…` ID 或 `latest`；省略 `version` 也受支持。Foundry 支持 Skills CRUD 和版本元数据，但不支持下载 version content。

续用已有容器并继续使用 Skills 时，每次请求都应再次提供需要的 Skills。带 scope 的 Skill 身份只在本地检查，不会序列化到提供方请求。参见 Anthropic [Agent Skills API 指南](https://platform.claude.com/docs/en/build-with-claude/skills-guide)、[Skills 快速入门](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/quickstart)和 [Code Execution 模型兼容性](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#compatibility)。

## 提供已上传文件

通过已有 `FileService` 上传输入文件，再把各个 `ProviderFileRef::model_reference()` 放入 `AnthropicCodeExecutionConfig.files`。构造首次执行请求时，可用以下配置替换前面示例中的默认配置：

```rust,no_run
use lingxi_llm_client::{
    files::ProviderFileRef,
    protocol::{AnthropicCodeExecutionConfig, HostedTool},
    RequestOptions,
};

# fn file_input(uploaded: &ProviderFileRef) -> (HostedTool, RequestOptions) {
let execution = HostedTool::AnthropicCodeExecution(AnthropicCodeExecutionConfig {
    files: vec![uploaded.model_reference()],
    ..Default::default()
});
let options = RequestOptions {
    // 与本次上传传给 FileService 的非秘密账号标识一致。
    file_account_scope: Some("anthropic-workspace-main".into()),
    ..Default::default()
};
# (execution, options)
# }
```

编码器会把原生 `container_upload` block 追加到最后一条 user 消息。因此，`files` 非空时，消息列表必须以 user 消息结束。每次发送都会核对文件的 provider、profile、endpoint 指纹、协议和账号绑定；`RequestOptions.file_account_scope`，或直接编码时的 `CodecContext::with_file_scope`，必须与文件一致。`files` 中重复的 ID，以及已存在于原生 `container_upload` block 中的类型化文件 ID，都会在发送前被拒绝。容器续用仍需前文说明的独立 `account_scope`。请求凭证由调用方照常提供。

`ProviderFileRef::model_reference()` 会把文件原始 `expires_at` 保留到 `ProviderFileSource`。共享文件预检会拒绝格式错误的到期值，以及小于或等于当前时间的已知到期值，客户端也会在发送前重新检查；时间戳不会进入模型输入 wire。允许缺少到期信息，但这不保证文件可用。引用也会保留原始处理状态；当前只有 Gemini/Qwen 的明确状态会被解释，不能用它们的状态规则推断 Anthropic 文件可用性。已知到期校验失败不会自动刷新或重新上传文件。时间格式与检查时机详见[文件附件校验](file-attachments.zh.md)。Anthropic 的[文件存储到期时间](https://platform.claude.com/docs/en/build-with-claude/files#file-expiration) 与执行容器的滚动 `expires_at` 不同，后者仍不会触发本地容器到期拒绝。

chat 请求不会自动上传或下载文件。文件上传与生成文件获取沿用 `FileService`，调用方应保留其带作用域的引用。历史中已有的原生 `container_upload` block 仍可原样回放。参见[文件附件与生命周期](file-attachments.zh.md)。

Foundry 路由应通过 `FileService::new_foundry_for_model(http, profile, selected_model, authenticator, credential, account_scope)` 创建。它要求所选模型行属于该 profile、显式声明 Anthropic hosting 的 `FoundryDeployment`，且 profile 使用受支持的 Foundry resource endpoint。文件操作使用 `/anthropic/v1/*` 下的资源路径，包括 `/anthropic/v1/files`。返回的引用记录 `FoundryClaude`、规范化 resource endpoint 和相同的非秘密 `account_scope`；可以像上文一样把其 `model_reference()` 放入 `AnthropicCodeExecutionConfig.files`。

Foundry 文件引用绑定 resource 和 account，不绑定 model 或 deployment；同一 resource、同一 account 下的其他 deployment 也可使用该文件，但所选模型本身仍须支持 Code Execution。容器引用仍使用前文更严格的 deployment/model scope。Azure hosting、第一方 `api.anthropic.com` 文件引用，以及来自其他 resource 或 account 的引用都会在发送前被拒绝。客户端会在本地拒绝已过期的 `expires_at`，但不会刷新或重新上传文件，也不能保证推理时文件仍可用。Foundry 指南仅为 Anthropic hosting 列出 Files API；Files API reference 列出 `/v1/files` 操作。

## 流式响应与原生内容

`ModelStream::anthropic_container()` 暴露从 `message_start.message.container` 和 `message_delta.delta.container` 观测到的当前 metadata。需持续消费流才能接收更新；该 accessor 不会消费 frame 或等待完成。原生对象及未知字段保留在 `envelope` 中。[Messages reference](https://platform.claude.com/docs/en/api/typescript/messages) 定义了响应与流中的容器字段。

```rust,no_run
use lingxi_llm_client::{protocol::ChatRequest, LlmClient, RequestOptions};

# async fn inspect_stream(
#     client: &LlmClient,
#     request: &ChatRequest,
#     options: &RequestOptions,
# ) -> Result<(), Box<dyn std::error::Error>> {
let mut stream = client.chat().stream(request, options).await?;
while let Some(event) = stream.next().await {
    let event = event?;
    // 将各事件交给应用的消息记录或流式展示逻辑。
    println!("{event:?}");
    if let Some(container) = stream.anthropic_container() {
        // 宿主可将该观测值与请求的实际 scope 一起保存。
        println!("{}", container.envelope);
    }
}
# Ok(())
# }
```

流中断前观测到容器 metadata，不代表执行已经完成。是否继续或检查已有状态由应用决定；客户端不会自动重复请求。

`ChatResponse.anthropic_usage: Option<serde_json::Value>` 保留原始 usage 对象。`ModelStream::anthropic_usage()` 暴露按字段合并已接收 frame 更新后的原生 usage，原始 frame 仍保留在 `ProviderEvent` 中。供应商报告的 `usage.server_tool_use.code_execution_requests` 映射到跨供应商的 `ServerToolUsage.code_interpreter_requests` 调用次数。该次数不能用于确定容器运行费用或组织免费额度，客户端也不会据此推断执行免费。

服务端执行调用及结果保留为 `ContentBlock::ProviderContent`，保留 ID、输入、输出及生成文件 metadata，以便原生回放。这些内容不会进入 `ConversationMessage::tool_uses()`，宿主不应执行它们。普通客户端工具仍通过 `ToolUse` 交给应用执行。原始流事件、输入分片与回放规则参见[保留 Anthropic 原生内容](anthropic-native-content.md)。

`pause_turn` 保持为 `StopReason::Other("pause_turn")`。调用方决定是否原样追加返回的 assistant 消息，并续用容器引用以继续执行。这是调用方明确发起的下一次请求；客户端不会自动循环，也不会把暂停改成 `EndTurn`。

## 路由与失败处理

- 第一方 profile 必须声明 `provider_id = "anthropic"`、使用 `ProtocolFamily::AnthropicMessages`，且 endpoint 规范化后为 `https://api.anthropic.com`。Foundry 则需要 `ProtocolFamily::FoundryClaude`、Foundry resource URL，以及所选 model row 中显式的 `FoundryDeployment`。兼容协议或任意代理 URL 本身不会启用这两种路由。
- Code Execution 精确模型范围包括 Fable 5/5.1、Mythos 5/5.1、Opus 4.5/4.6/4.7/4.8/5/5.5、Sonnet 4.5/4.6/5、Haiku 4.5，以及文档所列平台上的 Mythos Preview。程序化工具调用不支持 Haiku 4.5 和 Mythos Preview。Foundry 还要求底层 ID 属于该 hosting 的 Foundry 模型范围，且 deployment 为 Anthropic hosting。未知身份在发送前失败；账号和地区可用性由供应商决定。
- 执行请求无论新建还是续用容器，都不会自动重试或切换 failover profile。传输失败可能发生在服务端已经执行命令之后，因此是否重试由应用明确决定。
- 容器到期错误交给调用方处理。客户端不会静默创建替代容器，也不会依据 `expires_at` 提前拒绝续用：Anthropic 说明它是滚动时间戳，与容器实际 30 天寿命不同。参见[容器续用说明](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#container-reuse)。

程序化工具调用（`allowed_callers`）的请求约束、caller 元数据和续传方式见[Anthropic Programmatic Tool Calling](anthropic-programmatic-tools.md)。此切片支持第一方 API 和 Anthropic-hosted Foundry 上的预置与自定义 Skills；独立 Skills CRUD、版本上传和列举见 [Skills 资源服务](anthropic-skills.md)。Foundry Skill 引用绑定 resource 和 account；Foundry version-content 下载仍不支持。原生工具错误 block 仍可从响应内容检查。mock 与 wire 验证不能证明真实账号可用性，也不能证明 Anthropic 服务上的实际执行成功。
