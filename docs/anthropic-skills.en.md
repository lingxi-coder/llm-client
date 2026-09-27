# Anthropic Skills resource service

[简体中文](anthropic-skills.md)

`AnthropicSkillsService` manages custom Anthropic Skills and version metadata independently of Chat. It supports create, one-page list, get, delete, version create/list/get/delete, and version-content download on the first-party Claude API. Foundry supports the same Skills API lifecycle except for version-content download. The service owns the Skills HTTP wire; callers own credentials, uploaded content, pagination decisions, retry policy, and downloaded ZIP handling.

Skill create/list/get/delete and version create/list/get/delete are in Anthropic's [GA Skills API reference](https://platform.claude.com/docs/en/api/http/skills). `GET /v1/skills/{skill_id}/versions/{version}/content` is currently documented in the [Beta API reference](https://platform.claude.com/docs/en/api/http/beta/skills/versions/download). The method follows that page's version-ID path and example, which does not add an `anthropic-beta` header. It returns a byte stream and never extracts the ZIP or writes a local file.

## Scope and Messages references

Each service and custom Skill reference is bound to an `AnthropicSkillScope`. A direct Claude API scope records the profile, `https://api.anthropic.com`, a stable account identity, and an optional Anthropic Workspace ID. If an API key can access multiple Workspaces, set the Workspace ID explicitly; the service sends `anthropic-workspace-id`, and the Messages profile must carry the same header.

```rust,no_run
use lingxi_llm_client::{
    anthropic_skills::{AnthropicSkillFile, AnthropicSkillsService},
    protocol::{
        AnthropicCodeExecutionConfig, AnthropicSkillScope, ChatRequest, HostedTool, Secret,
    },
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
let service = AnthropicSkillsService::new(http, credential, scope)?;
let skill = service
    .create(
        vec![AnthropicSkillFile::from_bytes(
            "review/SKILL.md",
            b"---\nname: review\ndescription: Review documents.\n---\nUse the review checklist.".to_vec(),
        )],
        Some("Review documents"),
    )
    .await?;
let messages_ref = skill
    .messages_reference()
    .ok_or("this Skill source cannot be attached to Messages")?;
request.hosted_tools.push(HostedTool::AnthropicCodeExecution(
    AnthropicCodeExecutionConfig {
        skills: vec![messages_ref],
        ..Default::default()
    },
));
# Ok(())
# }
```

`AnthropicSkillResourceRef` preserves a scoped identity for every API source, including `plugin` and `anthropic_example`. `AnthropicSkill::messages_reference()` projects only the documented executable `custom` and `anthropic` source types into `container.skills`; the complete `native` response and source type remain available for all records.

For a multi-Workspace key, configure the Messages profile header to the same Workspace ID:

```json
{
  "headers": {
    "anthropic-workspace-id": "wrkspc_01Example"
  }
}
```

A single-Workspace credential can omit both Workspace values and bind through the caller's distinct `account_scope`.

## Microsoft Foundry

Anthropic documents custom Skills upload through the Skills API on Microsoft Foundry when the deployment is **Hosted on Anthropic**. The service builds its routes from the Foundry resource base, so `/v1/skills` becomes `https://{resource}.services.ai.azure.com/anthropic/v1/skills`. Foundry scope is resource/account scoped and does not capture a chat deployment or model. It never sends `anthropic-workspace-id`.

Construct the service with the Foundry `ProviderProfile`, an explicit `FoundryHosting::Anthropic`, a stable non-secret `account_scope`, the profile's authenticator, and its credential:

```rust,no_run
use lingxi_llm_client::{
    anthropic_skills::AnthropicSkillsService,
    protocol::{FoundryHosting, ProviderProfile, Secret},
    Authenticator, Transport,
};

# fn create_service<'a>(
#     http: &'a dyn Transport,
#     profile: &'a ProviderProfile,
#     auth: &'a dyn Authenticator,
#     credential: Secret<String>,
# ) -> Result<AnthropicSkillsService<'a>, Box<dyn std::error::Error>> {
let service = AnthropicSkillsService::new_foundry(
    http,
    profile,
    "foundry-resource-account-a",
    FoundryHosting::Anthropic,
    auth,
    credential,
)?;
# Ok(service)
# }
```

The profile must use `ProtocolFamily::FoundryClaude` and the HTTPS Foundry Anthropic resource endpoint; the explicit hosting argument rejects Azure-hosted deployments. The constructor does not require a model row, because Skills storage is independent of chat model identity. When attaching a custom Skill to Code Execution, the request must still have a typed selected deployment hosted on Anthropic, and its resource, profile, and `account_scope` must match the Skill reference. Foundry API keys can use `ApiKeyAuthenticator`; Entra tokens can use `BearerAuthenticator`.

Custom Skills on Foundry are created, listed, fetched, deleted, and versioned through the same `/v1/skills` routes as the Claude API. `download_version_content()` returns `UnsupportedCapability` before transport because Foundry does not support `GET /v1/skills/{skill_id}/versions/{version}/content`. A Foundry Skill reference cannot be replayed on the first-party API or a different Foundry resource/account. See Anthropic's [Foundry Skills guidance](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) and [Claude in Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry).

## Uploads and lifecycle behavior

Create and version-create accept caller-provided file streams as multipart `files[]`. Paths must be safe and relative; all files must share one top-level directory containing `SKILL.md` at its root. The client conservatively caps the combined uncompressed upload at 30,000,000 bytes. It does not read local paths, parse Skill YAML, infer the Skill name, or execute uploaded content. `AnthropicSkillFile::new` accepts a one-shot stream and declared size; the service checks the actual byte count while sending.

On create, optional `display_name` must be a nonempty single-line value with at most 255 characters. When omitted, Anthropic derives it from Skill metadata. List methods return one page only; pass `next_page` as the next request's `page`. The documented page limit is 1–1000 and defaults to 20.

Mutations are sent once and never retried. A transport interruption, timeout, response-read failure, invalid success response, upload-stream error, or HTTP 408/5xx response after dispatch returns `OutcomeUnknown` or `OutcomeUnknownResponse`; the response variant retains the operation, resource identity when known, HTTP status, request ID, and response body. Reconcile using list/get before deciding whether to issue another mutation. Other non-2xx responses remain provider errors with status and request ID when present. Each operation has a 120-second default deadline; `with_timeout` sets a positive caller-owned override. The version-content stream is subject to a 64 MiB client-side cap.

Version methods use provider version IDs. `AnthropicSkillVersion::pinned_skill()` creates a Messages reference pinned to the exact returned version ID when that source type supports `container.skills`.

## Contract references

- [Skills API](https://platform.claude.com/docs/en/api/http/skills)
- [Create Skill](https://platform.claude.com/docs/en/api/http/skills/create)
- [List Skills](https://platform.claude.com/docs/en/api/http/skills/list)
- [Create Skill Version](https://platform.claude.com/docs/en/api/http/skills/versions/create)
- [List Skill Versions](https://platform.claude.com/docs/en/api/typescript/skills/versions/list)
- [Skills guide and limits](https://platform.claude.com/docs/en/build-with-claude/skills-guide)
- [Beta version-content download reference](https://platform.claude.com/docs/en/api/http/beta/skills/versions/download)
