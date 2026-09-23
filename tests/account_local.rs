use async_trait::async_trait;
use lingxi_llm_client::protocol::{LlmError, ProviderProfile, Secret};
use lingxi_llm_client::{
    AccountFailure, AccountIdentity, AccountMetric, AccountQuery, AccountRpc, AccountUsageSource,
    CodexAccountSource, CopilotAccountSource, HttpRequest, HttpResponse, KimiCodeAccountSource,
    LlmClientBuilder, StreamResponse, Transport, WebSocketSession,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

fn profile(name: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": name,
        "profile_name": name,
        "base_url": "https://example.invalid/v1",
        "protocol": "open_ai_chat",
        "auth": "none",
        "models": [],
    }))
    .unwrap()
}

struct MockRpc {
    values: BTreeMap<String, Result<Value, AccountFailure>>,
    calls: Mutex<Vec<(String, Value)>>,
}

impl MockRpc {
    fn new(
        values: impl IntoIterator<Item = (&'static str, Result<Value, AccountFailure>)>,
    ) -> Self {
        Self {
            values: values
                .into_iter()
                .map(|(method, value)| (method.to_owned(), value))
                .collect(),
            calls: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl AccountRpc for MockRpc {
    async fn call(&self, method: &str, params: &Value) -> Result<Value, AccountFailure> {
        self.calls
            .lock()
            .unwrap()
            .push((method.to_owned(), params.clone()));
        self.values
            .get(method)
            .cloned()
            .unwrap_or(Err(AccountFailure::InvalidResponse))
    }
}

#[derive(Default)]
struct MockHttp {
    responses: BTreeMap<String, Value>,
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for MockHttp {
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
        let response = self.responses.get(&request.url).cloned();
        self.requests.lock().unwrap().push(request);
        Ok(HttpResponse {
            status: if response.is_some() { 200 } else { 404 },
            headers: vec![],
            body: serde_json::to_vec(&response.unwrap_or(Value::Null))
                .unwrap()
                .into(),
        })
    }

    async fn execute_no_follow(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.execute(request).await
    }

    async fn open_stream(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        unreachable!()
    }

    async fn open_responses_websocket_session(
        &self,
        _: HttpRequest,
    ) -> Result<Box<dyn WebSocketSession>, LlmError> {
        unreachable!()
    }
}

#[tokio::test]
async fn codex_reads_plan_all_quota_windows_and_daily_tokens() {
    let rpc = Arc::new(MockRpc::new([
        (
            "account/read",
            Ok(json!({"account":{"type":"chatgpt","planType":"plus","email":null}})),
        ),
        (
            "account/rateLimits/read",
            Ok(json!({
                "rateLimits": {"primary": {"usedPercent": 20, "windowDurationMins": 300}},
                "rateLimitsByLimitId": {
                    "codex": {
                        "primary": {"usedPercent": 25, "windowDurationMins": 300, "resetsAt": 1_800_000_000},
                        "secondary": {"usedPercent": 40, "windowDurationMins": 10_080, "resetsAt": 1_800_100_000},
                        "credits": {"balance": "12.5", "hasCredits": true, "unlimited": false}
                    },
                    "gpt-reserve": {
                        "primary": {"usedPercent": 10, "windowDurationMins": 300}
                    }
                }
            })),
        ),
        (
            "account/usage/read",
            Ok(json!({
                "summary": {"lifetimeTokens": 123_456},
                "dailyUsageBuckets": [
                    {"startDate": "2026-09-22", "tokens": 111},
                    {"startDate": "2026-09-23", "tokens": 789}
                ]
            })),
        ),
    ]));
    let source = CodexAccountSource::new(rpc.clone());
    let query = AccountQuery::new(AccountIdentity::AuthUser);
    let result = source
        .fetch(
            &profile("openai"),
            &query,
            (1_790_121_600, 1_790_208_000),
            100,
            &MockHttp::default(),
        )
        .await;
    assert!(
        matches!(result.subscription, AccountMetric::Available { value, .. } if value.plan_name.as_deref() == Some("plus") && value.status == lingxi_llm_client::SubscriptionStatus::VerifiedActive)
    );
    assert!(
        matches!(result.quota_windows, AccountMetric::Available { value, .. } if value.len() == 3 && value[0].name == "codex/primary" && value[0].remaining_percent == Some(75.0) && value[2].name == "gpt-reserve/primary")
    );
    assert!(
        matches!(result.token_usage, AccountMetric::Available { value, .. } if value.lifetime_tokens == Some(123_456) && value.buckets.len() == 1 && value.buckets[0].total_tokens == Some(789) && value.buckets[0].end_unix - value.buckets[0].start_unix == 86_400)
    );
    assert_eq!(
        rpc.calls
            .lock()
            .unwrap()
            .iter()
            .map(|(method, _)| method.as_str())
            .collect::<Vec<_>>(),
        vec![
            "account/read",
            "account/rateLimits/read",
            "account/usage/read"
        ]
    );
    assert_eq!(
        rpc.calls.lock().unwrap()[0].1,
        json!({"refreshToken": false})
    );
}

#[tokio::test]
async fn codex_one_failed_rpc_does_not_hide_other_metrics() {
    let rpc = Arc::new(MockRpc::new([
        (
            "account/read",
            Ok(json!({"account":{"type":"chatgpt","planType":"pro"}})),
        ),
        ("account/rateLimits/read", Err(AccountFailure::RateLimited)),
        (
            "account/usage/read",
            Ok(json!({"summary":{"lifetimeTokens":11}})),
        ),
    ]));
    let result = CodexAccountSource::new(rpc)
        .fetch(
            &profile("openai"),
            &AccountQuery::new(AccountIdentity::AuthUser),
            (0, 1),
            100,
            &MockHttp::default(),
        )
        .await;
    assert!(matches!(
        result.quota_windows,
        AccountMetric::Failed {
            reason: AccountFailure::RateLimited
        }
    ));
    assert!(matches!(
        result.subscription,
        AccountMetric::Available { .. }
    ));
    assert!(matches!(
        result.token_usage,
        AccountMetric::Available { .. }
    ));
}

#[tokio::test]
async fn codex_uses_limit_plan_when_account_read_fails_and_keeps_overlapping_day() {
    let rpc = Arc::new(MockRpc::new([
        ("account/read", Err(AccountFailure::Transport)),
        (
            "account/rateLimits/read",
            Ok(
                json!({"rateLimits":{"planType":"plus","primary":{"usedPercent":25,"windowDurationMins":300}}}),
            ),
        ),
        (
            "account/usage/read",
            Ok(
                json!({"summary":{"lifetimeTokens":20},"dailyUsageBuckets":[{"startDate":"2026-09-23","tokens":20}]}),
            ),
        ),
    ]));
    let start = 1_790_121_600 + 12 * 3_600;
    let result = CodexAccountSource::new(rpc)
        .fetch(
            &profile("openai"),
            &AccountQuery::new(AccountIdentity::AuthUser),
            (start, start + 3_600),
            start,
            &MockHttp::default(),
        )
        .await;
    assert!(matches!(
        result.subscription,
        AccountMetric::Available { value, .. }
            if value.plan_name.as_deref() == Some("plus")
                && value.status == lingxi_llm_client::SubscriptionStatus::VerifiedActive
    ));
    assert!(matches!(
        result.token_usage,
        AccountMetric::Available { value, .. }
            if value.buckets.len() == 1
                && value.buckets[0].start_unix == 1_790_121_600
                && value.buckets[0].total_tokens == Some(20)
    ));
}

#[tokio::test]
async fn copilot_reports_account_quota_without_inventing_token_history() {
    let rpc = Arc::new(MockRpc::new([(
        "account.getQuota",
        Ok(json!({
            "quotaSnapshots": {
                "premium_interactions": {
                    "entitlementRequests": 300,
                    "usedRequests": 90,
                    "remainingPercentage": 70,
                    "resetDate": "2026-10-01T00:00:00Z"
                },
                "chat": {
                    "entitlementRequests": -1,
                    "usedRequests": 12,
                    "remainingPercentage": 100,
                    "resetDate": null
                }
            }
        })),
    )]));
    let result = CopilotAccountSource::new(rpc.clone())
        .fetch(
            &profile("copilot"),
            &AccountQuery::new(AccountIdentity::AuthUser),
            (0, 1),
            100,
            &MockHttp::default(),
        )
        .await;
    assert!(
        matches!(result.quota_windows, AccountMetric::Available { value, .. }
        if value.len() == 2
            && value.iter().any(|window| window.name == "chat" && window.limit.is_none())
            && value.iter().any(|window| window.name == "premium_interactions"
                && window.remaining == Some(210)
                && window.resets_at_unix.is_some()))
    );
    assert_eq!(result.subscription, AccountMetric::NotReported);
    assert_eq!(result.token_usage, AccountMetric::Unsupported);
    assert_eq!(result.balance, AccountMetric::Unsupported);
    assert_eq!(
        rpc.calls.lock().unwrap().as_slice(),
        &[("account.getQuota".into(), json!({}))]
    );
}

#[tokio::test]
async fn copilot_uses_the_queried_users_token_for_quota() {
    let rpc = Arc::new(MockRpc::new([(
        "account.getQuota",
        Ok(json!({"quotaSnapshots": {}})),
    )]));
    let mut query = AccountQuery::new(AccountIdentity::AuthUser);
    query.credential = Some(Secret::new("github-user-token".into()));
    CopilotAccountSource::new(rpc.clone())
        .fetch(
            &profile("copilot"),
            &query,
            (0, 1),
            100,
            &MockHttp::default(),
        )
        .await;
    assert_eq!(
        rpc.calls.lock().unwrap().as_slice(),
        &[(
            "account.getQuota".into(),
            json!({"gitHubToken":"github-user-token"})
        )]
    );
}

struct UserQuotaRpc;

#[async_trait]
impl AccountRpc for UserQuotaRpc {
    async fn call(&self, method: &str, params: &Value) -> Result<Value, AccountFailure> {
        assert_eq!(method, "account.getQuota");
        let used = match params.get("gitHubToken").and_then(Value::as_str) {
            Some("alice-token") => 10,
            Some("bob-token") => 20,
            _ => return Err(AccountFailure::Unauthorized),
        };
        Ok(json!({"quotaSnapshots":{"premium_interactions":{
            "entitlementRequests":100,"usedRequests":used,"remainingPercentage":100-used
        }}}))
    }
}

#[tokio::test]
async fn one_copilot_source_can_query_distinct_users_with_their_tokens() {
    let mut alice = profile("copilot");
    alice.profile_name = "copilot-alice".into();
    let mut bob = alice.clone();
    bob.profile_name = "copilot-bob".into();
    let mut builder =
        LlmClientBuilder::with_transport(Arc::new(MockHttp::default()), &[alice, bob]);
    builder.register_account_source(
        "copilot",
        AccountIdentity::AuthUser,
        Arc::new(CopilotAccountSource::new(Arc::new(UserQuotaRpc))),
    );
    let client = builder.build().unwrap();
    for (profile_name, token, expected) in [
        ("copilot-alice", "alice-token", 10),
        ("copilot-bob", "bob-token", 20),
    ] {
        let mut query = AccountQuery::new(AccountIdentity::AuthUser);
        query.credential = Some(Secret::new(token.into()));
        let snapshot = client.account_usage(profile_name, &query).await.unwrap();
        assert!(
            matches!(snapshot.quota_windows, AccountMetric::Available { value, .. }
            if value[0].used == Some(expected))
        );
    }
}

#[tokio::test]
async fn kimi_uses_authenticated_loopback_service_and_parses_wallet() {
    let mut http = MockHttp::default();
    http.responses.insert(
        "http://127.0.0.1:58627/api/v1/oauth/usage".into(),
        json!({
            "code": 0,
            "data": {"kind":"ok", "quota": {
                "usages": {
                    "limit5h":{"usedRatio":0.25,"resetAt":"2026-09-23T12:00:00Z"},
                    "limit7d":{"usedRatio":0.5,"resetAt":"2026-09-28T00:00:00Z"}
                },
                "extraUsage":{"balanceCents":1234,"totalCents":5000,"currency":"USD"}
            }}
        }),
    );
    http.responses.insert("http://127.0.0.1:58627/api/v1/oauth/userinfo".into(), json!({
        "code": 0,
        "data": {"kind":"ok", "userInfo":{"userId":"user-1","status":"active","userLevelName":"Allegro"}}
    }));
    let mut query = AccountQuery::new(AccountIdentity::AuthUser);
    query.service_url = Some("http://127.0.0.1:58627".into());
    query.service_credential = Some(Secret::new("service-token".into()));
    let result = KimiCodeAccountSource::new()
        .fetch(&profile("kimi-code"), &query, (0, 1), 100, &http)
        .await;
    assert!(
        matches!(result.balance, AccountMetric::Available { value, .. } if value[0].remaining == "12.34" && value[0].total.as_deref() == Some("50.00"))
    );
    assert!(
        matches!(result.quota_windows, AccountMetric::Available { value, .. } if value.len() == 2 && value[0].duration_mins == Some(300) && value[1].duration_mins == Some(10_080))
    );
    assert!(
        matches!(result.subscription, AccountMetric::Available { value, .. } if value.plan_name.as_deref() == Some("Allegro") && value.status == lingxi_llm_client::SubscriptionStatus::Unknown)
    );
    assert_eq!(result.token_usage, AccountMetric::Unsupported);
    let requests = http.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(|request| request.method == "GET"
        && request
            .headers
            .contains(&("authorization".into(), "Bearer service-token".into()))));
}

#[tokio::test]
async fn kimi_rejects_non_loopback_url_before_sending_credential() {
    let http = MockHttp::default();
    let mut query = AccountQuery::new(AccountIdentity::AuthUser);
    query.service_url = Some("https://example.com".into());
    query.service_credential = Some(Secret::new("service-token".into()));
    let result = KimiCodeAccountSource::new()
        .fetch(&profile("kimi-code"), &query, (0, 1), 100, &http)
        .await;
    assert!(matches!(
        result.quota_windows,
        AccountMetric::Failed {
            reason: AccountFailure::InvalidResponse
        }
    ));
    assert!(http.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn malformed_kimi_reset_fails_quota_without_hiding_wallet_or_plan() {
    let mut http = MockHttp::default();
    http.responses.insert(
        "http://127.0.0.1:58627/api/v1/oauth/usage".into(),
        json!({"code":0,"data":{"kind":"ok","quota":{
            "usages":{"limit5h":{"usedRatio":0.2,"resetAt":"not-a-date"}},
            "extraUsage":{"balanceCents":500,"totalCents":1000,"currency":"USD"}
        }}}),
    );
    http.responses.insert(
        "http://127.0.0.1:58627/api/v1/oauth/userinfo".into(),
        json!({"code":0,"data":{"kind":"ok","userInfo":{
            "userId":"user-1","status":"active","userLevelName":"Allegro"
        }}}),
    );
    let mut query = AccountQuery::new(AccountIdentity::AuthUser);
    query.service_url = Some("http://127.0.0.1:58627".into());
    query.service_credential = Some(Secret::new("service-token".into()));
    let result = KimiCodeAccountSource::new()
        .fetch(&profile("kimi-code"), &query, (0, 1), 100, &http)
        .await;
    assert!(matches!(
        result.quota_windows,
        AccountMetric::Failed {
            reason: AccountFailure::InvalidResponse
        }
    ));
    assert!(
        matches!(result.balance, AccountMetric::Available { value, .. } if value[0].remaining == "5.00")
    );
    assert!(matches!(
        result.subscription,
        AccountMetric::Available { .. }
    ));
}
