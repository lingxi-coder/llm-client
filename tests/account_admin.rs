use async_trait::async_trait;
use bytes::Bytes;
use lingxi_agent_api::protocol::{LlmError, ProtocolFamily, ProviderProfile, Secret};
use lingxi_llm_client::{
    builtin_providers, AccountFailure, AccountIdentity, AccountMetric, AccountQuery,
    AccountScopeKind, HttpRequest, HttpResponse, LlmClientBuilder, StreamResponse, Transport,
    WebSocketSession,
};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct ScriptedTransport {
    replies: Mutex<VecDeque<(u16, Value)>>,
    seen: Mutex<Vec<HttpRequest>>,
}

impl ScriptedTransport {
    fn replies(replies: Vec<(u16, Value)>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into()),
            ..Self::default()
        })
    }
    fn requests(&self) -> Vec<HttpRequest> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for ScriptedTransport {
    async fn execute(&self, req: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.seen.lock().unwrap().push(req);
        let (status, body) = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected account request");
        Ok(HttpResponse {
            status,
            headers: vec![],
            body: Bytes::from(body.to_string()),
        })
    }
    async fn execute_no_follow(&self, req: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.execute(req).await
    }
    async fn open_stream(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("account API must not stream")
    }
    async fn open_responses_websocket_session(
        &self,
        _: HttpRequest,
    ) -> Result<Box<dyn WebSocketSession>, LlmError> {
        panic!("account API must not use websocket")
    }
}

fn profile(id: &str) -> ProviderProfile {
    builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.provider_id.as_str() == id)
        .unwrap()
}

fn client(profile: ProviderProfile, http: Arc<ScriptedTransport>) -> lingxi_llm_client::LlmClient {
    LlmClientBuilder::with_transport(http, &[profile])
        .with_region(lingxi_agent_api::protocol::Region::International)
        .build()
        .unwrap()
}

fn query() -> AccountQuery {
    let mut q = AccountQuery::new(AccountIdentity::ApiKey);
    q.credential = Some(Secret::new("ordinary-key".into()));
    q.management_credential = Some(Secret::new("admin-key".into()));
    q.since_unix = Some(1_735_689_600); // 2025-01-01T00:00:00Z
    q.until_unix = Some(1_735_776_000);
    q
}

fn param(url: &str, key: &str) -> Option<String> {
    reqwest::Url::parse(url)
        .unwrap()
        .query_pairs()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.into_owned())
}

#[tokio::test]
async fn openai_admin_usage_pages_and_filters() {
    let mut p = profile("openai");
    p.base_url = "https://untrusted.invalid/v1".into();
    let http = ScriptedTransport::replies(vec![
        (
            200,
            json!({"data":[{"start_time":1735689600,"end_time":1735776000,"results":[{"model":"gpt-4o","input_tokens":100,"output_tokens":20,"input_cached_tokens":40}]}],"has_more":true,"next_page":"cursor+1"}),
        ),
        (
            200,
            json!({"data":[{"start_time":1735776000,"end_time":1735862400,"results":[{"model":"gpt-4o","input_tokens":30,"output_tokens":10,"input_cached_tokens":0}]}],"has_more":false,"next_page":null}),
        ),
        (
            200,
            json!({"data":[{"start_time":1735689600,"end_time":1735776000,"results":[{"amount":{"value":0.0625,"currency":"usd"},"api_key_id":"key_123","project_id":"proj_123"}]}],"has_more":false,"next_page":null}),
        ),
    ]);
    let client = client(p.clone(), http.clone());
    let mut q = query();
    q.selector.project_id = Some("proj_123".into());
    q.selector.api_key_id = Some("key_123".into());
    q.selector.organization_id = Some("org_123".into());
    let result = client.account_usage(&p.profile_name, &q).await.unwrap();
    let AccountMetric::Available { scope, value, .. } = result.token_usage else {
        panic!("usage unavailable")
    };
    assert_eq!(scope.kind, AccountScopeKind::ApiKey);
    assert_eq!(value.buckets.len(), 2);
    assert_eq!(value.buckets[0].total_tokens, Some(120));
    assert_eq!(value.buckets[0].cached_input_tokens, Some(40));
    let AccountMetric::Available { scope, value, .. } = result.cost_usage else {
        panic!("cost unavailable")
    };
    assert_eq!(scope.kind, AccountScopeKind::ApiKey);
    assert_eq!(value.buckets[0].amount, "0.0625");
    assert_eq!(value.buckets[0].currency, "USD");
    let seen = http.requests();
    assert_eq!(seen.len(), 3);
    assert!(seen[..2].iter().all(|r| r
        .url
        .starts_with("https://api.openai.com/v1/organization/usage/completions?")));
    assert!(seen[2]
        .url
        .starts_with("https://api.openai.com/v1/organization/costs?"));
    assert_eq!(param(&seen[2].url, "api_key_ids[]"), Some("key_123".into()));
    assert_eq!(
        param(&seen[0].url, "project_ids[]"),
        Some("proj_123".into())
    );
    assert_eq!(param(&seen[0].url, "api_key_ids[]"), Some("key_123".into()));
    assert_eq!(param(&seen[1].url, "page"), Some("cursor+1".into()));
    assert!(seen[0]
        .headers
        .contains(&("authorization".into(), "Bearer admin-key".into())));
    assert!(seen[0]
        .headers
        .contains(&("openai-organization".into(), "org_123".into())));
}

#[tokio::test]
async fn anthropic_admin_usage_includes_cache_creation_and_read() {
    let p = profile("anthropic");
    let http = ScriptedTransport::replies(vec![
        (
            200,
            json!({"data":[{"starting_at":"2025-01-01T00:00:00Z","ending_at":"2025-01-02T00:00:00Z","results":[{"model":"claude-sonnet-4","uncached_input_tokens":100,"cache_creation":{"ephemeral_5m_input_tokens":30,"ephemeral_1h_input_tokens":20},"cache_read_input_tokens":50,"output_tokens":40}]}],"has_more":false,"next_page":null}),
        ),
        (
            200,
            json!({"data":[{"starting_at":"2025-01-01T00:00:00Z","ending_at":"2025-01-02T00:00:00Z","results":[{"workspace_id":"wrkspc_123","model":"claude-sonnet-4","amount":"123.78912","currency":"USD"},{"workspace_id":"other","model":"claude-sonnet-4","amount":"100","currency":"USD"}]}],"has_more":false,"next_page":null}),
        ),
    ]);
    let client = client(p.clone(), http.clone());
    let mut q = query();
    q.selector.workspace_id = Some("wrkspc_123".into());
    let result = client.account_usage(&p.profile_name, &q).await.unwrap();
    let AccountMetric::Available { scope, value, .. } = result.token_usage else {
        panic!("usage unavailable")
    };
    assert_eq!(scope.kind, AccountScopeKind::Workspace);
    assert_eq!(value.buckets[0].input_tokens, Some(200));
    assert_eq!(value.buckets[0].cached_input_tokens, Some(50));
    assert_eq!(value.buckets[0].cache_write_tokens, Some(50));
    assert_eq!(value.buckets[0].total_tokens, Some(240));
    let AccountMetric::Available { scope, value, .. } = result.cost_usage else {
        panic!("cost unavailable")
    };
    assert_eq!(scope.kind, AccountScopeKind::Workspace);
    assert_eq!(value.buckets.len(), 1);
    assert_eq!(value.buckets[0].amount, "1.2378912");
    assert_eq!(value.buckets[0].model.as_deref(), Some("claude-sonnet-4"));
    let seen = http.requests();
    assert_eq!(seen.len(), 2);
    assert!(seen[1]
        .url
        .starts_with("https://api.anthropic.com/v1/organizations/cost_report?"));
    assert_eq!(
        param(&seen[1].url, "group_by[]"),
        Some("description".into())
    );
    assert_eq!(
        param(&seen[0].url, "starting_at"),
        Some("2025-01-01T00:00:00Z".into())
    );
    assert_eq!(
        param(&seen[0].url, "workspace_ids[]"),
        Some("wrkspc_123".into())
    );
    assert!(seen[0]
        .headers
        .contains(&("x-api-key".into(), "admin-key".into())));
}

#[tokio::test]
async fn anthropic_default_workspace_cost_rows_use_null_workspace_id() {
    let p = profile("anthropic");
    let http = ScriptedTransport::replies(vec![
        (200, json!({"data":[],"has_more":false,"next_page":null})),
        (
            200,
            json!({"data":[{"starting_at":"2025-01-01T00:00:00Z","ending_at":"2025-01-02T00:00:00Z","results":[{"workspace_id":null,"amount":"125","currency":"USD"},{"workspace_id":"wrkspc_other","amount":"500","currency":"USD"}]}],"has_more":false,"next_page":null}),
        ),
        (
            200,
            json!({"id":"wrkspc_default","name":"Default","type":"workspace"}),
        ),
    ]);
    let mut q = query();
    q.selector.workspace_id = Some("wrkspc_default".into());
    let result = client(p.clone(), http.clone())
        .account_usage(&p.profile_name, &q)
        .await
        .unwrap();
    assert!(
        matches!(result.cost_usage, AccountMetric::Available { value, .. }
        if value.buckets.len() == 1 && value.buckets[0].amount == "1.25")
    );
    assert!(http.requests()[2]
        .url
        .starts_with("https://api.anthropic.com/v1/organizations/workspaces/wrkspc_default"));
}

#[tokio::test]
async fn xai_balance_requires_team_and_converts_cents_exactly() {
    let p = profile("xai");
    let http = ScriptedTransport::replies(vec![
        (200, json!({"total":{"val":"-12345"},"changes":[]})),
        (
            200,
            json!({"timeSeries":[{"group":["Chat grok-4"],"dataPoints":[{"timestamp":"2025-01-01T00:00:00Z","values":[0.75973725]}]}],"limitReached":false}),
        ),
    ]);
    let client = client(p.clone(), http.clone());
    let mut q = query();
    let missing = client.account_usage(&p.profile_name, &q).await.unwrap();
    assert!(matches!(missing.balance, AccountMetric::NotReported));
    assert!(http.requests().is_empty());
    q.selector.team_id = Some("team/unsafe".into());
    let result = client.account_usage(&p.profile_name, &q).await.unwrap();
    let AccountMetric::Available { value, scope, .. } = result.balance else {
        panic!("balance unavailable")
    };
    assert_eq!(scope.kind, AccountScopeKind::Team);
    assert_eq!(value[0].remaining, "123.45");
    assert_eq!(value[0].unit, "USD");
    assert!(matches!(result.token_usage, AccountMetric::Unsupported));
    let AccountMetric::Available { value, scope, .. } = result.cost_usage else {
        panic!("cost unavailable")
    };
    assert_eq!(scope.kind, AccountScopeKind::Team);
    assert_eq!(value.buckets[0].amount, "0.75973725");
    assert_eq!(value.buckets[0].currency, "USD");
    assert_eq!(value.buckets[0].model, None);
    let seen = http.requests();
    assert_eq!(seen.len(), 2);
    assert_eq!(
        seen[0].url,
        "https://management-api.x.ai/v1/billing/teams/team%2Funsafe/prepaid/balance"
    );
    assert_eq!(seen[1].method, "POST");
    assert_eq!(
        seen[1].url,
        "https://management-api.x.ai/v1/billing/teams/team%2Funsafe/usage"
    );
    let body: Value = serde_json::from_slice(&seen[1].body).unwrap();
    assert_eq!(
        body["analyticsRequest"]["timeRange"]["startTime"],
        "2025-01-01 00:00:00"
    );
    assert_eq!(
        body["analyticsRequest"]["timeRange"]["endTime"],
        "2025-01-01 23:59:59"
    );
    assert_eq!(body["analyticsRequest"]["values"][0]["name"], "usd");
}

#[tokio::test]
async fn xai_empty_usage_is_a_known_empty_report() {
    let p = profile("xai");
    let http = ScriptedTransport::replies(vec![
        (200, json!({"total":{"val":"-1000"}})),
        (200, json!({"limitReached":false})),
    ]);
    let client = client(p.clone(), http);
    let mut q = query();
    q.selector.team_id = Some("team-1".into());
    let result = client.account_usage(&p.profile_name, &q).await.unwrap();
    assert!(matches!(
        result.balance,
        AccountMetric::Available { value, .. } if value[0].remaining == "10.00"
    ));
    assert!(matches!(
        result.cost_usage,
        AccountMetric::Available { value, .. } if value.buckets.is_empty()
    ));
}

#[tokio::test]
async fn gemini_developer_metric_reports_output_only() {
    let p = profile("google");
    assert_eq!(p.protocol, ProtocolFamily::GeminiGenerateContent);
    let http = ScriptedTransport::replies(vec![(
        200,
        json!({"timeSeries":[{"metric":{"type":"generativelanguage.googleapis.com/generate_content_usage_output_token_count","labels":{"model":"gemini-2.5-pro","output_modality":"TEXT"}},"resource":{"type":"generativelanguage.googleapis.com/Location","labels":{"resource_container":"project-1"}},"points":[{"interval":{"startTime":"2025-01-01T00:00:00Z","endTime":"2025-01-01T01:00:00Z"},"value":{"int64Value":"42"}}]}]}),
    )]);
    let client = client(p.clone(), http.clone());
    let mut q = query();
    q.selector.project_id = Some("project-1".into());
    let result = client.account_usage(&p.profile_name, &q).await.unwrap();
    let AccountMetric::Available { value, .. } = result.token_usage else {
        panic!("usage unavailable")
    };
    assert_eq!(value.buckets[0].input_tokens, None);
    assert_eq!(value.buckets[0].output_tokens, Some(42));
    assert_eq!(value.buckets[0].model.as_deref(), Some("gemini-2.5-pro"));
    let seen = http.requests();
    assert!(param(&seen[0].url, "filter")
        .unwrap()
        .contains("generativelanguage.googleapis.com/"));
    assert!(seen[0]
        .headers
        .contains(&("authorization".into(), "Bearer admin-key".into())));
}

#[tokio::test]
async fn vertex_gemini_metric_merges_input_and_output() {
    let mut p = profile("google");
    p.protocol = ProtocolFamily::VertexGemini;
    let http = ScriptedTransport::replies(vec![(
        200,
        json!({"timeSeries":[
          {"metric":{"type":"aiplatform.googleapis.com/publisher/online_serving/token_count","labels":{"type":"input"}},"resource":{"labels":{"model_user_id":"gemini-2.5-pro"}},"points":[{"interval":{"startTime":"2025-01-01T00:00:00Z","endTime":"2025-01-01T01:00:00Z"},"value":{"int64Value":"100"}}]},
          {"metric":{"type":"aiplatform.googleapis.com/publisher/online_serving/token_count","labels":{"type":"output"}},"resource":{"labels":{"model_user_id":"gemini-2.5-pro"}},"points":[{"interval":{"startTime":"2025-01-01T00:00:00Z","endTime":"2025-01-01T01:00:00Z"},"value":{"int64Value":"25"}}]}
        ]}),
    )]);
    let client = client(p.clone(), http.clone());
    let mut q = query();
    q.selector.project_id = Some("project-1".into());
    let result = client.account_usage(&p.profile_name, &q).await.unwrap();
    let AccountMetric::Available { value, .. } = result.token_usage else {
        panic!("usage unavailable")
    };
    assert_eq!(value.buckets[0].input_tokens, Some(100));
    assert_eq!(value.buckets[0].output_tokens, Some(25));
    assert_eq!(value.buckets[0].total_tokens, Some(125));
    let seen = http.requests();
    assert!(param(&seen[0].url, "filter")
        .unwrap()
        .contains("aiplatform.googleapis.com/"));
}

#[tokio::test]
async fn malformed_admin_response_fails_without_partial_usage() {
    let p = profile("openai");
    let http = ScriptedTransport::replies(vec![
        (
            200,
            json!({"data":[{"start_time":1,"end_time":2,"results":[{"input_tokens":10,"output_tokens":3}]}],"has_more":true,"next_page":null}),
        ),
        (200, json!({"data":[],"has_more":false,"next_page":null})),
    ]);
    let client = client(p.clone(), http);
    let result = client
        .account_usage(&p.profile_name, &query())
        .await
        .unwrap();
    assert_eq!(
        result.token_usage,
        AccountMetric::Failed {
            reason: AccountFailure::InvalidResponse
        }
    );
    assert!(matches!(result.cost_usage, AccountMetric::Available { .. }));
}

#[tokio::test]
async fn anthropic_oauth_uses_bearer_and_cost_failure_keeps_tokens() {
    let p = profile("anthropic");
    let http = ScriptedTransport::replies(vec![
        (
            200,
            json!({"data":[{"starting_at":"2025-01-01T00:00:00Z","ending_at":"2025-01-02T00:00:00Z","results":[{"model":"claude-sonnet-4","uncached_input_tokens":9,"cache_creation":null,"cache_read_input_tokens":0,"output_tokens":3}]}],"has_more":false,"next_page":null}),
        ),
        (403, json!({"error":"forbidden"})),
    ]);
    let client = client(p.clone(), http.clone());
    let mut q = query();
    q.identity = AccountIdentity::AuthUser;
    let result = client.account_usage(&p.profile_name, &q).await.unwrap();
    let AccountMetric::Available { value, .. } = result.token_usage else {
        panic!("token usage unavailable")
    };
    assert_eq!(value.buckets[0].input_tokens, Some(9));
    assert_eq!(
        result.cost_usage,
        AccountMetric::Failed {
            reason: AccountFailure::PermissionDenied
        }
    );
    let seen = http.requests();
    assert_eq!(seen.len(), 2);
    assert!(seen.iter().all(|r| r
        .headers
        .contains(&("authorization".into(), "Bearer admin-key".into()))));
    assert!(seen
        .iter()
        .all(|r| !r.headers.iter().any(|(name, _)| name == "x-api-key")));
}

#[tokio::test]
async fn google_oauth_requires_project_before_request() {
    let p = profile("google");
    let http = ScriptedTransport::replies(vec![]);
    let client = client(p.clone(), http.clone());
    let mut q = query();
    q.identity = AccountIdentity::AuthUser;
    let result = client.account_usage(&p.profile_name, &q).await.unwrap();
    assert!(matches!(result.token_usage, AccountMetric::NotReported));
    assert!(http.requests().is_empty());
    q.management_credential = None;
    let result = client.account_usage(&p.profile_name, &q).await.unwrap();
    assert!(matches!(
        result.token_usage,
        AccountMetric::CredentialRequired
    ));
    assert!(http.requests().is_empty());
}
