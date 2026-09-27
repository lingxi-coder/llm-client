use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    anthropic_skills::{
        AnthropicSkillFile, AnthropicSkillListOptions, AnthropicSkillVersionListOptions,
        AnthropicSkillsError, AnthropicSkillsService,
    },
    protocol::{AnthropicSkillScope, FoundryHosting, LlmError, ProviderProfile, Secret},
    ApiKeyAuthenticator, BearerAuthenticator, CodecContext, EncodeRequest, FoundryClaudeCodec,
    HttpRequest, HttpStreamRequest, RequestMode, StreamResponse, Transport, WireCodec,
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

const ENDPOINT: &str = "https://resource-a.services.ai.azure.com/anthropic";
const PROFILE: &str = "foundry-skills-resource-a";
const ACCOUNT: &str = "foundry-skills-account-a";

#[derive(Clone)]
struct Reply {
    status: u16,
    body: Bytes,
}

#[derive(Debug, Clone)]
struct SeenRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Bytes,
}

struct MockTransport {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<SeenRequest>>,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn take_reply(&self) -> Reply {
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("mock reply")
    }

    fn requests(&self) -> Vec<SeenRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn response(&self, reply: Reply) -> StreamResponse {
        StreamResponse {
            status: reply.status,
            headers: vec![("request-id".into(), "req-foundry-skills".into())],
            body: stream::once(async move { Ok(reply.body) }).boxed(),
        }
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(SeenRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body,
        });
        Ok(self.response(self.take_reply()))
    }

    async fn send_stream(
        &self,
        mut request: HttpStreamRequest,
    ) -> Result<StreamResponse, LlmError> {
        let mut body = BytesMut::new();
        while let Some(chunk) = request.body.next().await {
            body.extend_from_slice(&chunk?);
        }
        self.requests.lock().unwrap().push(SeenRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: body.freeze(),
        });
        Ok(self.response(self.take_reply()))
    }
}

fn reply(body: Value) -> Reply {
    Reply {
        status: 200,
        body: serde_json::to_vec(&body).unwrap().into(),
    }
}

fn failed_reply(status: u16) -> Reply {
    Reply {
        status,
        body: Bytes::from_static(b"{\"error\":\"uncertain mutation\"}"),
    }
}

fn profile(auth: &str, endpoint: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"custom-foundry",
        "profile_name":PROFILE,
        "base_url":endpoint,
        "protocol":"foundry_claude",
        "auth":auth,
        "chat_enabled":false,
        "models":[]
    }))
    .unwrap()
}

fn chat_profile(hosting: FoundryHosting, request_model: &str, model_id: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"custom-foundry",
        "profile_name":PROFILE,
        "base_url":ENDPOINT,
        "protocol":"foundry_claude",
        "auth":"none",
        "models":[{
            "display_model":"Claude deployment",
            "request_model":request_model,
            "billing_model":model_id,
            "foundry":{
                "hosting":match hosting {
                    FoundryHosting::Anthropic => "anthropic",
                    FoundryHosting::Azure => "azure",
                },
                "model_id":model_id
            }
        }]
    }))
    .unwrap()
}

fn skill(id: &str) -> Value {
    json!({
        "id":id,
        "created_at":"2026-09-27T00:00:00Z",
        "display_name":"Quarterly report",
        "latest_version_id":"skver_latest",
        "source":{"type":"custom"},
        "type":"skill",
        "updated_at":"2026-09-27T00:00:00Z"
    })
}

fn version(skill_id: &str, id: &str) -> Value {
    json!({
        "id":id,
        "skill_id":skill_id,
        "type":"skill_version",
        "created_at":"2026-09-27T00:00:00Z",
        "description":"Does useful work",
        "name":"quarterly-report"
    })
}

#[tokio::test]
async fn foundry_custom_skills_use_resource_routes_for_crud_and_versions() {
    let transport = MockTransport::new([
        reply(skill("skill_foundry")),
        reply(json!({"data":[skill("skill_foundry")],"next_page":null})),
        reply(skill("skill_foundry")),
        reply(version("skill_foundry", "skver_one")),
        reply(json!({"data":[version("skill_foundry", "skver_one")],"next_page":null})),
        reply(version("skill_foundry", "skver_one")),
        reply(json!({"id":"skver_one","type":"skill_version_deleted"})),
        reply(json!({"id":"skill_foundry","type":"skill_deleted"})),
    ]);
    let profile = profile("api_key", &format!("{ENDPOINT}/"));
    let auth = ApiKeyAuthenticator;
    let service = AnthropicSkillsService::new_foundry(
        &transport,
        &profile,
        ACCOUNT,
        FoundryHosting::Anthropic,
        &auth,
        Secret::new("foundry-api-key".into()),
    )
    .unwrap();

    let created = service
        .create(
            vec![AnthropicSkillFile::from_bytes(
                "quarterly-report/SKILL.md",
                Bytes::from_static(b"---\nname: quarterly-report\n---"),
            )],
            Some("Quarterly report"),
        )
        .await
        .unwrap();
    assert_eq!(created.source_type, "custom");
    let scope = service.scope().clone();
    assert_eq!(scope.profile_name(), PROFILE);
    assert_eq!(scope.endpoint(), ENDPOINT);
    assert_eq!(scope.account_scope(), ACCOUNT);
    assert_eq!(scope.workspace_id(), None);
    let reference = created.reference.clone();

    let listed = service
        .list(&AnthropicSkillListOptions::default())
        .await
        .unwrap();
    assert_eq!(listed.skills[0].id, "skill_foundry");
    assert_eq!(service.get(&reference).await.unwrap().id, "skill_foundry");
    let version = service
        .create_version(
            &reference,
            vec![AnthropicSkillFile::from_bytes(
                "quarterly-report/SKILL.md",
                Bytes::from_static(b"updated"),
            )],
        )
        .await
        .unwrap();
    let versions = service
        .list_versions(&reference, &AnthropicSkillVersionListOptions::default())
        .await
        .unwrap();
    assert_eq!(versions.versions[0].id, version.id);
    assert_eq!(
        service
            .get_version(&reference, "skver_one")
            .await
            .unwrap()
            .id,
        "skver_one"
    );
    service
        .delete_version(&reference, "skver_one")
        .await
        .unwrap();
    service.delete(&reference).await.unwrap();

    assert!(matches!(
        service
            .download_version_content(&reference, "skver_one")
            .await,
        Err(AnthropicSkillsError::Llm(
            LlmError::UnsupportedCapability { .. }
        ))
    ));
    let requests = transport.requests();
    assert_eq!(
        requests.len(),
        8,
        "Foundry version-content is rejected locally"
    );
    assert_eq!(requests[0].url, format!("{ENDPOINT}/v1/skills"));
    assert_eq!(requests[1].method, "GET");
    assert_eq!(requests[1].url, format!("{ENDPOINT}/v1/skills?limit=20"));
    assert_eq!(
        requests[2].url,
        format!("{ENDPOINT}/v1/skills/skill_foundry")
    );
    assert_eq!(
        requests[3].url,
        format!("{ENDPOINT}/v1/skills/skill_foundry/versions")
    );
    assert!(requests[4]
        .url
        .ends_with("/v1/skills/skill_foundry/versions?limit=20"));
    assert_eq!(
        requests[5].url,
        format!("{ENDPOINT}/v1/skills/skill_foundry/versions/skver_one")
    );
    assert_eq!(requests[6].method, "DELETE");
    assert_eq!(requests[7].method, "DELETE");
    for request in &requests {
        assert!(request.headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("x-api-key") && value == "foundry-api-key"
        }));
        assert!(request.headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("anthropic-version") && value == "2023-06-01"
        }));
        assert!(!request
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("anthropic-workspace-id")));
    }
    assert!(String::from_utf8_lossy(&requests[0].body)
        .contains("name=\"files[]\"; filename=\"quarterly-report/SKILL.md\""));
}

#[tokio::test]
async fn foundry_skills_support_bearer_auth_and_reject_azure_or_wrong_protocol() {
    let transport = MockTransport::new([reply(json!({"data":[],"next_page":null}))]);
    let foundry_profile = profile("bearer", ENDPOINT);
    let auth = BearerAuthenticator;
    let service = AnthropicSkillsService::new_foundry(
        &transport,
        &foundry_profile,
        ACCOUNT,
        FoundryHosting::Anthropic,
        &auth,
        Secret::new("entra-token".into()),
    )
    .unwrap();
    service
        .list(&AnthropicSkillListOptions::default())
        .await
        .unwrap();
    let requests = transport.requests();
    let request = &requests[0];
    assert!(request.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("authorization") && value == "Bearer entra-token"
    }));
    assert!(!request
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("x-api-key")));

    let no_requests = MockTransport::new([]);
    assert!(matches!(
        AnthropicSkillsService::new_foundry(
            &no_requests,
            &foundry_profile,
            ACCOUNT,
            FoundryHosting::Azure,
            &auth,
            Secret::new("token".into()),
        ),
        Err(AnthropicSkillsError::Llm(
            LlmError::UnsupportedCapability { .. }
        ))
    ));
    let mut wrong_protocol = foundry_profile.clone();
    wrong_protocol.protocol = lingxi_llm_client::protocol::ProtocolFamily::AnthropicMessages;
    assert!(AnthropicSkillsService::new_foundry(
        &no_requests,
        &wrong_protocol,
        ACCOUNT,
        FoundryHosting::Anthropic,
        &auth,
        Secret::new("token".into()),
    )
    .is_err());
    assert!(no_requests.requests().is_empty());
}

#[tokio::test]
async fn foundry_mutation_server_errors_report_unknown_outcome_after_one_dispatch() {
    let profile = profile("api_key", ENDPOINT);
    let auth = ApiKeyAuthenticator;
    let skill_scope =
        AnthropicSkillScope::new_foundry(PROFILE, ENDPOINT, ACCOUNT, FoundryHosting::Anthropic)
            .unwrap();
    let reference = lingxi_llm_client::anthropic_skills::AnthropicSkillResourceRef::new(
        "skill_foundry",
        "custom",
        skill_scope,
    )
    .unwrap();

    let cases = [
        ("create", 500u16, None),
        ("create version", 408u16, Some("skill_foundry")),
        ("delete", 500u16, Some("skill_foundry")),
        ("delete version", 408u16, Some("skill_foundry/skver_one")),
    ];
    for (operation, status, expected_identity) in cases {
        let transport = MockTransport::new([failed_reply(status)]);
        let service = AnthropicSkillsService::new_foundry(
            &transport,
            &profile,
            ACCOUNT,
            FoundryHosting::Anthropic,
            &auth,
            Secret::new("foundry-api-key".into()),
        )
        .unwrap();
        let error = match operation {
            "create" => service
                .create(
                    vec![AnthropicSkillFile::from_bytes(
                        "quarterly-report/SKILL.md",
                        Bytes::from_static(b"skill"),
                    )],
                    None,
                )
                .await
                .unwrap_err(),
            "create version" => service
                .create_version(
                    &reference,
                    vec![AnthropicSkillFile::from_bytes(
                        "quarterly-report/SKILL.md",
                        Bytes::from_static(b"skill"),
                    )],
                )
                .await
                .unwrap_err(),
            "delete" => service.delete(&reference).await.unwrap_err(),
            "delete version" => service
                .delete_version(&reference, "skver_one")
                .await
                .unwrap_err(),
            _ => unreachable!(),
        };
        assert!(matches!(
            error,
            AnthropicSkillsError::OutcomeUnknownResponse {
                operation: actual,
                identity,
                reason,
            } if actual == operation
                && identity.as_deref() == expected_identity
                && reason.contains(&format!("HTTP {status}"))
                && reason.contains("req-foundry-skills")
                && reason.contains("uncertain mutation")
        ));
        assert_eq!(
            transport.requests().len(),
            1,
            "{operation} is never retried"
        );
    }
}

#[tokio::test]
async fn foundry_read_server_errors_remain_provider_errors() {
    let profile = profile("api_key", ENDPOINT);
    let auth = ApiKeyAuthenticator;
    let transport = MockTransport::new([failed_reply(500)]);
    let service = AnthropicSkillsService::new_foundry(
        &transport,
        &profile,
        ACCOUNT,
        FoundryHosting::Anthropic,
        &auth,
        Secret::new("foundry-api-key".into()),
    )
    .unwrap();

    assert!(matches!(
        service.list(&AnthropicSkillListOptions::default()).await,
        Err(AnthropicSkillsError::Provider { status: 500, .. })
    ));
    assert_eq!(transport.requests().len(), 1);
}

#[test]
fn foundry_skill_scope_is_explicit_and_rejects_azure_or_first_party_reuse() {
    assert!(matches!(
        AnthropicSkillScope::new_foundry(PROFILE, ENDPOINT, ACCOUNT, FoundryHosting::Azure),
        Err(LlmError::UnsupportedCapability { .. })
    ));
    for endpoint in [
        "http://resource-a.services.ai.azure.com/anthropic",
        "https://resource-a.services.ai.azure.com/anthropic/v1",
        "https://resource-b.services.ai.azure.com/anthropic?x=1",
        "https://api.anthropic.com",
    ] {
        assert!(AnthropicSkillScope::new_foundry(
            PROFILE,
            endpoint,
            ACCOUNT,
            FoundryHosting::Anthropic,
        )
        .is_err());
    }
    let foundry_scope =
        AnthropicSkillScope::new_foundry(PROFILE, ENDPOINT, ACCOUNT, FoundryHosting::Anthropic)
            .unwrap();
    assert!(foundry_scope
        .clone()
        .with_workspace_id("wrkspc_not_supported_on_foundry")
        .is_err());
    let transport = MockTransport::new([]);
    assert!(AnthropicSkillsService::new(
        &transport,
        Secret::new("foundry-key".into()),
        foundry_scope,
    )
    .is_err());
}

#[test]
fn foundry_custom_skill_reference_matches_resource_and_account_not_deployment_model() {
    let scope =
        AnthropicSkillScope::new_foundry(PROFILE, ENDPOINT, ACCOUNT, FoundryHosting::Anthropic)
            .unwrap();
    let skill = lingxi_llm_client::protocol::AnthropicSkillRef::custom(
        "skill_foundry_custom",
        scope.clone(),
    );

    let encode_with = |profile: &ProviderProfile, account_scope: &str| {
        let deployment = &profile.models[0];
        let mut request: lingxi_llm_client::protocol::ChatRequest = serde_json::from_value(json!({
            "model":deployment.request_model,
            "max_tokens":1024,
            "messages":[{"role":"user","content":[{"type":"text","text":"Use the skill."}]}]
        }))
        .unwrap();
        request.hosted_tools.push(
            lingxi_llm_client::protocol::HostedTool::AnthropicCodeExecution(
                lingxi_llm_client::protocol::AnthropicCodeExecutionConfig {
                    skills: vec![skill.clone()],
                    ..Default::default()
                },
            ),
        );
        FoundryClaudeCodec.encode_request(
            EncodeRequest::new(&request),
            &CodecContext::new(profile, &deployment.request_model, RequestMode::Complete)
                .with_account_scope(Some(account_scope)),
        )
    };

    let first_model = chat_profile(
        FoundryHosting::Anthropic,
        "custom-opus-deployment",
        "claude-opus-5-5",
    );
    let wire = encode_with(&first_model, ACCOUNT).unwrap();
    let body: Value = serde_json::from_slice(&wire.body).unwrap();
    assert_eq!(
        body["container"]["skills"],
        json!([{"type":"custom","skill_id":"skill_foundry_custom"}])
    );

    // Skill IDs are resource/account scoped; containers and chat model rows
    // carry the narrower deployment/model identity.
    let second_model = chat_profile(
        FoundryHosting::Anthropic,
        "custom-sonnet-deployment",
        "claude-sonnet-5",
    );
    assert!(encode_with(&second_model, ACCOUNT).is_ok());
    assert!(encode_with(&first_model, "another-account").is_err());

    let azure = chat_profile(
        FoundryHosting::Azure,
        "custom-azure-deployment",
        "claude-opus-5-5",
    );
    assert!(encode_with(&azure, ACCOUNT).is_err());
}
