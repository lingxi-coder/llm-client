use async_trait::async_trait;
use lingxi_agent_api::protocol::{LlmError, ProviderProfile, Secret};
use lingxi_llm_client::{
    builtin_providers, AccountFailure, AccountIdentity, AccountMetric, AccountQuery,
    AccountScopeKind, HttpRequest, HttpResponse, LlmClientBuilder, StreamResponse, Transport,
    WebSocketSession,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct ScriptedHttp {
    replies: Mutex<Vec<(String, u16, String)>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl ScriptedHttp {
    fn reply(&self, url: &str, status: u16, body: Value) {
        self.replies
            .lock()
            .unwrap()
            .push((url.into(), status, body.to_string()));
    }

    fn requests(&self) -> Vec<HttpRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for ScriptedHttp {
    async fn execute(&self, _request: HttpRequest) -> Result<HttpResponse, LlmError> {
        panic!("account adapters must use the no-redirect transport path")
    }

    async fn execute_no_follow(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
        let reply = {
            let mut replies = self.replies.lock().unwrap();
            let position = replies
                .iter()
                .position(|(url, _, _)| url == &request.url)
                .expect("unexpected account URL");
            replies.remove(position)
        };
        self.requests.lock().unwrap().push(request);
        Ok(HttpResponse {
            status: reply.1,
            headers: vec![],
            body: reply.2.into(),
        })
    }

    async fn open_stream(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        unreachable!()
    }

    async fn open_responses_websocket_session(
        &self,
        _request: HttpRequest,
    ) -> Result<Box<dyn WebSocketSession>, LlmError> {
        unreachable!()
    }
}

struct RedirectFollowingHttp;

#[async_trait]
impl Transport for RedirectFollowingHttp {
    async fn execute(&self, _: HttpRequest) -> Result<HttpResponse, LlmError> {
        panic!("a credential-bearing request reached a redirect-following transport")
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
async fn account_credentials_require_explicit_no_redirect_transport_support() {
    let client =
        LlmClientBuilder::with_transport(Arc::new(RedirectFollowingHttp), &[profile("deepseek")])
            .build()
            .unwrap();
    let result = client
        .account_usage("deepseek", &ordinary_query())
        .await
        .unwrap();
    assert!(matches!(
        result.balance,
        AccountMetric::Failed {
            reason: AccountFailure::Transport
        }
    ));
}

fn profile(name: &str) -> ProviderProfile {
    builtin_providers()
        .unwrap()
        .into_iter()
        .find(|profile| profile.profile_name == name)
        .unwrap()
}

fn client(http: Arc<ScriptedHttp>, profile: ProviderProfile) -> lingxi_llm_client::LlmClient {
    LlmClientBuilder::with_transport(http, &[profile])
        .build()
        .unwrap()
}

fn ordinary_query() -> AccountQuery {
    let mut query = AccountQuery::new(AccountIdentity::ApiKey);
    query.credential = Some(Secret::new("ordinary-key".into()));
    query
}

fn assert_request(request: &HttpRequest, url: &str, bearer: &str) {
    assert_eq!(request.method, "GET");
    assert_eq!(request.url, url);
    assert!(request.body.is_empty());
    assert_eq!(
        request.headers,
        vec![("authorization".into(), format!("Bearer {bearer}"))]
    );
}

#[tokio::test]
async fn openrouter_ordinary_key_reports_only_its_own_spend_limit() {
    let http = Arc::new(ScriptedHttp::default());
    http.reply(
        "https://openrouter.ai/api/v1/key",
        200,
        json!({"data": {
            "limit": 100.50,
            "limit_remaining": 74.75,
            "usage": 25.75,
            "limit_reset": "daily"
        }}),
    );
    let client = client(http.clone(), profile("openrouter"));
    let mut query = ordinary_query();
    query.selector.api_key_id = Some("key-123".into());
    let result = client.account_usage("openrouter", &query).await.unwrap();

    assert_eq!(result.balance, AccountMetric::CredentialRequired);
    match result.quota_windows {
        AccountMetric::Available { scope, value, .. } => {
            assert_eq!(scope.kind, AccountScopeKind::ApiKey);
            assert_eq!(scope.id, None);
            assert_eq!(value[0].duration_mins, Some(1_440));
            assert_eq!(value[0].limit, None);
            assert_eq!(value[0].limit_decimal.as_deref(), Some("100.5"));
            assert_eq!(value[0].remaining_decimal.as_deref(), Some("74.75"));
            assert_eq!(value[0].unit.as_deref(), Some("USD"));
        }
        other => panic!("unexpected quota: {other:?}"),
    }
    assert_eq!(result.token_usage, AccountMetric::Unsupported);
    assert_eq!(result.subscription, AccountMetric::Unsupported);
    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    assert_request(
        &requests[0],
        "https://openrouter.ai/api/v1/key",
        "ordinary-key",
    );
}

#[tokio::test]
async fn openrouter_management_balance_survives_key_failure() {
    let http = Arc::new(ScriptedHttp::default());
    http.reply(
        "https://openrouter.ai/api/v1/key",
        401,
        json!({"error": "bad key"}),
    );
    http.reply(
        "https://openrouter.ai/api/v1/credits",
        200,
        json!({"data": {"total_credits": 100.5, "total_usage": 25.75}}),
    );
    let client = client(http.clone(), profile("openrouter"));
    let mut query = ordinary_query();
    query.management_credential = Some(Secret::new("management-key".into()));
    let result = client.account_usage("openrouter", &query).await.unwrap();

    assert_eq!(
        result.quota_windows,
        AccountMetric::Failed {
            reason: AccountFailure::Unauthorized
        }
    );
    match result.balance {
        AccountMetric::Available {
            scope,
            source,
            value,
        } => {
            assert_eq!(scope.kind, AccountScopeKind::User);
            assert_eq!(source, "https://openrouter.ai/api/v1/credits");
            assert_eq!(value[0].remaining, "74.75");
            assert_eq!(value[0].total.as_deref(), Some("100.5"));
        }
        other => panic!("unexpected balance: {other:?}"),
    }
    let requests = http.requests();
    assert_eq!(requests.len(), 2);
    assert_request(
        &requests[0],
        "https://openrouter.ai/api/v1/key",
        "ordinary-key",
    );
    assert_request(
        &requests[1],
        "https://openrouter.ai/api/v1/credits",
        "management-key",
    );
}

#[tokio::test]
async fn openrouter_key_quota_survives_management_permission_failure() {
    let http = Arc::new(ScriptedHttp::default());
    http.reply(
        "https://openrouter.ai/api/v1/key",
        200,
        json!({"data": {"limit": 10, "limit_remaining": 7.5, "limit_reset": "weekly"}}),
    );
    http.reply(
        "https://openrouter.ai/api/v1/credits",
        403,
        json!({"error": "management access required"}),
    );
    let client = client(http, profile("openrouter"));
    let mut query = ordinary_query();
    query.management_credential = Some(Secret::new("wrong-management-key".into()));
    let result = client.account_usage("openrouter", &query).await.unwrap();
    assert_eq!(
        result.balance,
        AccountMetric::Failed {
            reason: AccountFailure::PermissionDenied
        }
    );
    match result.quota_windows {
        AccountMetric::Available { scope, value, .. } => {
            assert_eq!(scope.kind, AccountScopeKind::ApiKey);
            assert_eq!(value[0].duration_mins, Some(10_080));
            assert_eq!(value[0].remaining_percent, Some(75.0));
        }
        other => panic!("unexpected quota: {other:?}"),
    }
}

#[tokio::test]
async fn openrouter_management_credential_can_report_balance_alone() {
    let http = Arc::new(ScriptedHttp::default());
    http.reply(
        "https://openrouter.ai/api/v1/credits",
        200,
        json!({"data": {"total_credits": "100.00", "total_usage": "25.005"}}),
    );
    let client = client(http.clone(), profile("openrouter"));
    let mut query = AccountQuery::new(AccountIdentity::ApiKey);
    query.management_credential = Some(Secret::new("management-key".into()));
    let result = client.account_usage("openrouter", &query).await.unwrap();
    match result.balance {
        AccountMetric::Available { scope, value, .. } => {
            assert_eq!(scope.kind, AccountScopeKind::User);
            assert_eq!(value[0].remaining, "74.995");
        }
        other => panic!("unexpected balance: {other:?}"),
    }
    assert_eq!(result.quota_windows, AccountMetric::CredentialRequired);
    assert_eq!(http.requests().len(), 1);
}

#[tokio::test]
async fn openrouter_uncapped_key_does_not_invent_a_balance() {
    let http = Arc::new(ScriptedHttp::default());
    http.reply(
        "https://openrouter.ai/api/v1/key",
        200,
        json!({"data": {"limit": null, "limit_remaining": null, "usage": 12.3}}),
    );
    let client = client(http, profile("openrouter"));
    let result = client
        .account_usage("openrouter", &ordinary_query())
        .await
        .unwrap();
    assert_eq!(result.balance, AccountMetric::CredentialRequired);
    assert_eq!(result.quota_windows, AccountMetric::NotReported);
    assert_eq!(result.token_usage, AccountMetric::Unsupported);
}

#[tokio::test]
async fn openrouter_out_of_range_remaining_has_no_percentage() {
    for remaining in [json!(11), json!(-1)] {
        let http = Arc::new(ScriptedHttp::default());
        http.reply(
            "https://openrouter.ai/api/v1/key",
            200,
            json!({"data": {"limit": 10, "limit_remaining": remaining}}),
        );
        let client = client(http, profile("openrouter"));
        let result = client
            .account_usage("openrouter", &ordinary_query())
            .await
            .unwrap();
        assert_eq!(result.balance, AccountMetric::CredentialRequired);
        let expected = remaining.to_string();
        match result.quota_windows {
            AccountMetric::Available { value, .. } => {
                assert_eq!(value[0].used_percent, None);
                assert_eq!(value[0].remaining_percent, None);
                assert_eq!(
                    value[0].remaining_decimal.as_deref(),
                    Some(expected.as_str())
                );
            }
            other => panic!("unexpected quota: {other:?}"),
        }
    }
}

#[tokio::test]
async fn deepseek_preserves_each_currency_and_decimal_string() {
    let http = Arc::new(ScriptedHttp::default());
    http.reply(
        "https://api.deepseek.com/user/balance",
        200,
        json!({"is_available": true, "balance_infos": [
            {"currency": "CNY", "total_balance": "110.00", "granted_balance": "10.00", "topped_up_balance": "100.00"},
            {"currency": "USD", "total_balance": "0.0040", "granted_balance": "0", "topped_up_balance": "0.0040"}
        ]}),
    );
    let client = client(http.clone(), profile("deepseek"));
    let result = client
        .account_usage("deepseek", &ordinary_query())
        .await
        .unwrap();
    match result.balance {
        AccountMetric::Available { scope, value, .. } => {
            assert_eq!(scope.kind, AccountScopeKind::User);
            assert_eq!(value.len(), 2);
            assert_eq!(value[0].unit, "CNY");
            assert_eq!(value[0].remaining, "110.00");
            assert_eq!(value[1].unit, "USD");
            assert_eq!(value[1].remaining, "0.0040");
            assert_eq!(value[0].total, None);
            assert_eq!(value[0].is_available, Some(true));
        }
        other => panic!("unexpected balance: {other:?}"),
    }
    assert_eq!(result.token_usage, AccountMetric::Unsupported);
    assert_eq!(result.quota_windows, AccountMetric::Unsupported);
    assert_request(
        &http.requests()[0],
        "https://api.deepseek.com/user/balance",
        "ordinary-key",
    );
}

#[tokio::test]
async fn deepseek_missing_key_and_malformed_data_are_field_local() {
    let http = Arc::new(ScriptedHttp::default());
    let client = client(http.clone(), profile("deepseek"));
    let missing = client
        .account_usage("deepseek", &AccountQuery::new(AccountIdentity::ApiKey))
        .await
        .unwrap();
    assert_eq!(missing.balance, AccountMetric::CredentialRequired);
    assert!(http.requests().is_empty());

    http.reply(
        "https://api.deepseek.com/user/balance",
        200,
        json!({"balance_infos": []}),
    );
    let malformed = client
        .account_usage("deepseek", &ordinary_query())
        .await
        .unwrap();
    assert_eq!(
        malformed.balance,
        AccountMetric::Failed {
            reason: AccountFailure::InvalidResponse
        }
    );
    assert_eq!(malformed.token_usage, AccountMetric::Unsupported);
}

#[tokio::test]
async fn deepseek_retains_a_nonspendable_balance_without_calling_it_zero() {
    let http = Arc::new(ScriptedHttp::default());
    http.reply(
        "https://api.deepseek.com/user/balance",
        200,
        json!({"is_available":false,"balance_infos":[{"currency":"USD","total_balance":"5.00"}]}),
    );
    let result = client(http, profile("deepseek"))
        .account_usage("deepseek", &ordinary_query())
        .await
        .unwrap();
    assert!(matches!(
        result.balance,
        AccountMetric::Available { value, .. }
            if value[0].remaining == "5.00" && value[0].is_available == Some(false)
    ));
}

#[tokio::test]
async fn moonshot_uses_the_region_matched_official_endpoint() {
    for (base, url, currency) in [
        (
            "https://api.moonshot.cn/v1",
            "https://api.moonshot.cn/v1/users/me/balance",
            "CNY",
        ),
        (
            "https://api.moonshot.ai/v1",
            "https://api.moonshot.ai/v1/users/me/balance",
            "USD",
        ),
    ] {
        let http = Arc::new(ScriptedHttp::default());
        http.reply(
            url,
            200,
            json!({"code": 0, "status": true, "scode": "0x0", "data": {
                "available_balance": 49.58894,
                "voucher_balance": 46.58893,
                "cash_balance": 3.00001
            }}),
        );
        let mut p = profile("kimi");
        p.base_url = base.into();
        let client = client(http.clone(), p);
        let result = client
            .account_usage("kimi", &ordinary_query())
            .await
            .unwrap();
        match result.balance {
            AccountMetric::Available { scope, value, .. } => {
                assert_eq!(scope.kind, AccountScopeKind::User);
                assert_eq!(value[0].unit, currency);
                assert_eq!(value[0].remaining, "49.58894");
                assert_eq!(value[0].total, None);
            }
            other => panic!("unexpected balance: {other:?}"),
        }
        assert_eq!(result.token_usage, AccountMetric::Unsupported);
        assert_request(&http.requests()[0], url, "ordinary-key");
    }
}

#[tokio::test]
async fn kimi_code_and_nonofficial_hosts_never_use_moonshot_balance() {
    let http = Arc::new(ScriptedHttp::default());
    let kimi_code = client(http.clone(), profile("kimi-code"));
    let result = kimi_code
        .account_usage("kimi-code", &ordinary_query())
        .await
        .unwrap();
    assert_eq!(result.balance, AccountMetric::Unsupported);

    let mut custom = profile("kimi");
    custom.base_url = "https://api.moonshot.cn.evil.test/v1".into();
    let custom = client(http.clone(), custom);
    let result = custom
        .account_usage("kimi", &ordinary_query())
        .await
        .unwrap();
    assert_eq!(result.balance, AccountMetric::Unsupported);
    assert!(http.requests().is_empty());
}

#[tokio::test]
async fn moonshot_provider_error_does_not_claim_balance_or_token_usage() {
    let http = Arc::new(ScriptedHttp::default());
    http.reply(
        "https://api.moonshot.cn/v1/users/me/balance",
        200,
        json!({"code": 1001, "status": false, "data": {"available_balance": 10}}),
    );
    let client = client(http, profile("kimi"));
    let result = client
        .account_usage("kimi", &ordinary_query())
        .await
        .unwrap();
    assert_eq!(
        result.balance,
        AccountMetric::Failed {
            reason: AccountFailure::ProviderError
        }
    );
    assert_eq!(result.token_usage, AccountMetric::Unsupported);
}

#[tokio::test]
async fn moonshot_requires_the_documented_code_status_and_data_envelope() {
    let http = Arc::new(ScriptedHttp::default());
    http.reply(
        "https://api.moonshot.cn/v1/users/me/balance",
        200,
        json!({"code": 0, "data": {"available_balance": 10}}),
    );
    let client = client(http, profile("kimi"));
    let result = client
        .account_usage("kimi", &ordinary_query())
        .await
        .unwrap();
    assert_eq!(
        result.balance,
        AccountMetric::Failed {
            reason: AccountFailure::InvalidResponse
        }
    );
}
