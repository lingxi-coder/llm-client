# Anthropic Skills 资源服务

[English](anthropic-skills.en.md)

`AnthropicSkillsService` 独立管理 Anthropic 自定义 Skill 及其版本元数据，不属于 Chat 请求。它在第一方 Claude API 上支持创建、单页列表、获取、删除、版本创建/列表/获取/删除和版本内容下载。Foundry 支持相同的 Skills API 生命周期，但不支持下载版本内容。服务只负责 Skills HTTP 协议；凭据、上传内容、分页、重试决策和下载 ZIP 的后续处理都由调用方负责。

Skill 的创建、列表、获取、删除及版本创建、列表、获取、删除都列在 Anthropic 的 [GA Skills API 参考](https://platform.claude.com/docs/en/api/http/skills)中。`GET /v1/skills/{skill_id}/versions/{version}/content` 目前列在[Beta API 参考](https://platform.claude.com/docs/en/api/http/beta/skills/versions/download)中。本服务按该页面规定的版本 ID 路径和示例调用，不添加 `anthropic-beta` 请求头。下载结果以字节流返回，不会解压 ZIP 或写入本地文件。

## 作用域与 Messages 引用

每个服务和自定义 Skill 引用都绑定一个 `AnthropicSkillScope`。第一方 Claude API scope 包含 profile、`https://api.anthropic.com`、稳定账号标识和可选 Anthropic Workspace ID。如果 API key 可以访问多个 Workspace，应显式设置 Workspace ID；服务会发送 `anthropic-workspace-id`，Messages profile 也必须包含相同的 header。

```rust,no_run
use lingxi_llm_client::{
    providers::anthropic::{
        skills::{AnthropicSkillFile, AnthropicSkillsService},
        types::{AnthropicCodeExecutionConfig, AnthropicSkillScope},
        native::AnthropicHostedTool,
    },
    protocol::{ChatRequest, Secret},
    Transport,
};

# async fn example(
#     http: &dyn Transport,
#     credential: Secret<String>,
#     request: &mut ChatRequest,
# ) -> Result<(), Box<dyn std::error::Error>> {
let scope = AnthropicSkillScope::new(
    "anthropic-prod",
    "https://api.anthropic.com",
    "account-workspace-a",
)?
.with_workspace_id("wrkspc_01Example")?;
let request_options = lingxi_llm_client::RequestOptions {
    credential: Some(credential),
    ..Default::default()
};
let service = AnthropicSkillsService::new(http, scope)?;
let skill = service
    .create(
        vec![AnthropicSkillFile::from_bytes(
            "review/SKILL.md",
            b"---\nname: review\ndescription: Review documents.\n---\nUse the review checklist.".to_vec(),
        )],
        Some("Review documents"),
        &request_options,
    )
    .await?;
let messages_ref = skill
    .messages_reference()
    .ok_or("this Skill source cannot be attached to Messages")?;
request.hosted_tools.push(AnthropicHostedTool::CodeExecution(
    AnthropicCodeExecutionConfig {
        skills: vec![messages_ref],
        ..Default::default()
    },
).into());
# Ok(())
# }
```

`AnthropicSkillResourceRef` 为所有 API source 保存带作用域的资源身份，包括 `plugin` 和 `anthropic_example`。`AnthropicSkill::messages_reference()` 只会把当前 Messages `container.skills` 合约支持的 `custom` 和 `anthropic` source 转换为可执行引用；其他 source 的类型和完整 `native` 响应仍会保留。

多 Workspace API key 还需要为 Messages 配置相同的 Workspace 请求头：

```json
{
  "headers": {
    "anthropic-workspace-id": "wrkspc_01Example"
  }
}
```

只绑定单个 Workspace 的凭据可以不设置两个 Workspace 字段，由调用方提供的独立 `account_scope` 进行绑定。

## Microsoft Foundry

Anthropic 文档允许在 Microsoft Foundry 上使用 Skills API 上传自定义 Skill，但 deployment 必须 **Hosted on Anthropic**。服务从 Foundry resource base 构造路径，因此 `/v1/skills` 会对应到 `https://{resource}.services.ai.azure.com/anthropic/v1/skills`。Foundry scope 绑定 resource 和 account，不捕获 chat deployment 或 model，也不会发送 `anthropic-workspace-id`。

使用 Foundry `ProviderProfile`、显式的 `FoundryHosting::Anthropic`、稳定且非秘密的 `account_scope`、profile authenticator 创建服务；每次操作通过 `RequestOptions` 传入 credential：

```rust,no_run
use lingxi_llm_client::{
    providers::anthropic::skills::AnthropicSkillsService,
    protocol::{FoundryHosting, ProviderProfile},
    Authenticator, Transport,
};

# fn create_service<'a>(
#     http: &'a dyn Transport,
#     profile: &'a ProviderProfile,
#     auth: &'a dyn Authenticator,
# ) -> Result<AnthropicSkillsService<'a>, Box<dyn std::error::Error>> {
let service = AnthropicSkillsService::new_foundry(
    http,
    profile,
    "foundry-resource-account-a",
    FoundryHosting::Anthropic,
    auth,
)?;
# Ok(service)
# }
```

Profile 必须使用 `ProtocolFamily::FoundryClaude` 和受支持的 HTTPS Foundry Anthropic resource endpoint；显式 hosting 参数会拒绝 Azure-hosted deployment。构造 service 不需要 chat model row，因为 Skills 存储不绑定 chat model。若在 Code Execution 中使用自定义 Skill，请求仍须选择 typed、Anthropic-hosted deployment，且 Skill 引用的 resource、profile 和 `account_scope` 必须匹配。Foundry API key 可配合 `ApiKeyAuthenticator`，Entra token 可配合 `BearerAuthenticator`。

Foundry 通过相同的 `/v1/skills` 路由创建、列出、读取、删除自定义 Skill 并管理版本。Foundry 不支持 `GET /v1/skills/{skill_id}/versions/{version}/content`，所以 `download_version_content()` 会在发送前返回 `UnsupportedCapability`。Foundry Skill 引用不能用于第一方 API 或其他 Foundry resource/account。参见 Anthropic 的[Foundry Skills 说明](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)和[Claude on Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry)。

## 上传与生命周期行为

创建 Skill 和新建版本时，调用方提供的文件流会按 multipart `files[]` 上传。文件路径必须是安全的相对路径；所有文件必须位于同一个顶层目录，并且目录根部必须包含 `SKILL.md`。客户端按较保守的 30,000,000 字节上限校验合并后的未压缩内容。服务不会读取本地路径、解析 Skill YAML、推断 Skill 名称或执行上传内容。`AnthropicSkillFile::new` 接收一次性文件流和声明大小，发送时会校验实际字节数。

创建时可选的 `display_name` 必须是非空单行文本，最多 255 个字符。省略时由 Anthropic 根据 Skill 元数据生成。列表方法只返回一页；下一页请求由调用方把 `next_page` 传回 `page`。官方页大小范围为 1–1000，默认值是 20。

变更请求只发送一次，不会自动重试。派发后的传输中断、超时、响应读取失败、成功响应格式错误、上传流错误，以及 HTTP 408/5xx 响应会返回 `OutcomeUnknown` 或 `OutcomeUnknownResponse`；response variant 会保留操作名、已知资源身份、HTTP 状态码、request ID 和响应体。调用方应先用 list/get 核对状态，再决定是否再次变更。其他非 2xx 响应仍作为 provider error 返回状态码，以及存在时的 request ID。每个操作默认使用 120 秒总截止时间；可用 `with_timeout` 设置正数超时。版本内容流另有 64 MiB 客户端上限。

版本接口使用服务端返回的版本 ID。若 source 支持 `container.skills`，可以调用 `AnthropicSkillVersion::pinned_skill()` 创建固定到该版本 ID 的 Messages 引用。

## 合约来源

- [Skills API](https://platform.claude.com/docs/en/api/http/skills)
- [创建 Skill](https://platform.claude.com/docs/en/api/http/skills/create)
- [列出 Skills](https://platform.claude.com/docs/en/api/http/skills/list)
- [创建 Skill 版本](https://platform.claude.com/docs/en/api/http/skills/versions/create)
- [列出 Skill 版本](https://platform.claude.com/docs/en/api/typescript/skills/versions/list)
- [Skills 指南和限制](https://platform.claude.com/docs/en/build-with-claude/skills-guide)
- [Beta 版本内容下载参考](https://platform.claude.com/docs/en/api/http/beta/skills/versions/download)
