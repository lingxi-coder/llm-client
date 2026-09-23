use async_trait::async_trait;
use lingxi_agent_api::protocol::{LlmError, ProviderProfile, Secret};
use lingxi_llm_client::{
    builtin_providers, AccountIdentity, AccountMetric, AccountQuery, AlibabaAccessKey, HttpRequest,
    HttpResponse, LlmClientBuilder, StreamResponse, Transport, WebSocketSession,
};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct QueueHttp {
    requests: Mutex<Vec<HttpRequest>>,
    replies: Mutex<VecDeque<Value>>,
}

impl QueueHttp {
    fn with_json(replies: Vec<Value>) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            replies: Mutex::new(replies.into()),
        }
    }

    fn requests(&self) -> Vec<HttpRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for QueueHttp {
    async fn execute(&self, _: HttpRequest) -> Result<HttpResponse, LlmError> {
        panic!("account calls must use no-redirect transport")
    }

    async fn execute_no_follow(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let body = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("scripted account response")
            .to_string();
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: body.into(),
        })
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

fn profile(name: &str) -> ProviderProfile {
    builtin_providers()
        .unwrap()
        .into_iter()
        .find(|profile| profile.profile_name == name)
        .unwrap()
}

fn json_response(body: Value) -> Value {
    body
}

#[tokio::test]
async fn qwen_usage_reads_workspace_limits_and_signed_per_key_daily_costs() {
    let http = Arc::new(QueueHttp::with_json(vec![
        json_response(json!({
            "success": true,
            "output": {"total":1,"page_no":1,"page_size":100,"quotas": [{
                "model": "qwen3.8-max",
                "workspace_id": "ws-demo",
                "model_limit": {"request_limit": 50, "request_limit_period": 60, "usage_limit": 200000, "usage_limit_field": "total_tokens", "usage_limit_period": 60},
                "workspace_limit": {"request_limit": 10, "request_limit_period": 60}
            }]}
        })),
        json_response(json!({
            "success": true,
            "code": "200",
            "data": {
                "costTotals": {"amount":"2.50","currency":"CNY"},
                "resultByTime": [{
                    "period":"20250101",
                    "total":{"amount":"2.50","currency":"CNY"},
                    "periodDetails":[{"key":"qwen3.8-max","amount":"2.50"}]
                }]
            }
        })),
    ]));
    let client = LlmClientBuilder::with_transport(http.clone(), &[profile("qwen")])
        .build()
        .unwrap();
    let mut query = AccountQuery::new(AccountIdentity::ApiKey);
    query.credential = Some(Secret::new("dashscope-key".into()));
    query.selector.workspace_id = Some("ws-demo".into());
    query.selector.api_key_id = Some("key-123".into());
    query.since_unix = Some(1_735_689_600); // 2025-01-01 UTC
    query.until_unix = Some(1_735_776_000); // 2025-01-02 UTC
    query.alibaba_access_key = Some(AlibabaAccessKey {
        id: "ram-id".into(),
        secret: Secret::new("ram-secret-never-send".into()),
        security_token: Some(Secret::new("sts-token".into())),
    });

    let snapshot = client.account_usage("qwen", &query).await.unwrap();
    match snapshot.quota_windows {
        AccountMetric::Available { value, .. } => {
            assert_eq!(value.len(), 3);
            assert_eq!(value[0].limit, Some(50));
            assert_eq!(value[0].used, None);
            assert_eq!(value[0].remaining, None);
            assert_eq!(value[1].unit.as_deref(), Some("total_tokens"));
        }
        other => panic!("expected Qwen quota limits, got {other:?}"),
    }
    match snapshot.cost_usage {
        AccountMetric::Available { value, .. } => {
            assert_eq!(value.buckets.len(), 1);
            assert_eq!(value.buckets[0].amount, "2.50");
            assert_eq!(value.buckets[0].model.as_deref(), Some("qwen3.8-max"));
            assert_eq!(value.buckets[0].currency, "CNY");
        }
        other => panic!("expected Qwen billing history, got {other:?}"),
    }

    let requests = http.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].url,
        "https://ws-demo.cn-beijing.maas.aliyuncs.com/api/v1/quotas?page_no=1&page_size=100"
    );
    assert_eq!(
        requests[0].headers[0],
        ("authorization".into(), "Bearer dashscope-key".into())
    );
    let billing_url = reqwest::Url::parse(&requests[1].url).unwrap();
    assert_eq!(
        billing_url.host_str(),
        Some("modelstudio.cn-beijing.aliyuncs.com")
    );
    let query_filter = billing_url
        .query_pairs()
        .find(|(key, _)| key == "filter")
        .unwrap()
        .1;
    assert!(query_filter.contains("key-123"));
    assert!(query_filter.contains("cn-beijing"));
    let authorization = requests[1]
        .headers
        .iter()
        .find(|(name, _)| name == "authorization")
        .unwrap()
        .1
        .clone();
    assert!(authorization.starts_with("ACS3-HMAC-SHA256 Credential=ram-id,"));
    assert!(!authorization.contains("ram-secret-never-send"));
    assert!(requests[1]
        .headers
        .iter()
        .any(|(name, value)| name == "x-acs-security-token" && value == "sts-token"));
    assert!(!requests[1].url.contains("dashscope-key"));
    verify_billing_signature(&requests[1], "ram-secret-never-send");
}

fn verify_billing_signature(request: &HttpRequest, secret: &str) {
    let url = reqwest::Url::parse(&request.url).unwrap();
    let header = |name: &str| {
        request
            .headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
            .unwrap()
    };
    let authorization = header("authorization");
    let signed_headers = authorization
        .split("SignedHeaders=")
        .nth(1)
        .unwrap()
        .split(',')
        .next()
        .unwrap();
    let signature = authorization.split("Signature=").nth(1).unwrap();
    let canonical_headers = signed_headers
        .split(';')
        .map(|name| {
            let value = if name == "host" {
                url.host_str().unwrap()
            } else {
                header(name)
            };
            format!("{name}:{}\n", value.trim())
        })
        .collect::<String>();
    let payload_hash = hex(ring::digest::digest(&ring::digest::SHA256, b"").as_ref());
    let canonical_request = format!(
        "GET\n{}\n{}\n{}\n{}\n{}",
        url.path(),
        url.query().unwrap_or_default(),
        canonical_headers,
        signed_headers,
        payload_hash
    );
    let canonical_hash =
        hex(ring::digest::digest(&ring::digest::SHA256, canonical_request.as_bytes()).as_ref());
    let string_to_sign = format!("ACS3-HMAC-SHA256\n{canonical_hash}");
    let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, secret.as_bytes());
    let expected = hex(ring::hmac::sign(&key, string_to_sign.as_bytes()).as_ref());
    assert_eq!(signature, expected);
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|byte| {
            [
                char::from(HEX[(byte >> 4) as usize]),
                char::from(HEX[(byte & 0xf) as usize]),
            ]
        })
        .collect()
}

#[tokio::test]
async fn qwen_workspace_quota_reads_all_pages() {
    let make_row = |index: usize| {
        json!({
            "model": format!("qwen-model-{index}"),
            "model_limit": {"request_limit": 10, "request_limit_period": 60},
            "workspace_limit": null
        })
    };
    let first_page: Vec<_> = (0..100).map(make_row).collect();
    let second_page = vec![make_row(100)];
    let http = Arc::new(QueueHttp::with_json(vec![
        json!({"success":true,"output":{"total":101,"page_no":1,"page_size":100,"quotas":first_page}}),
        json!({"success":true,"output":{"total":101,"page_no":2,"page_size":100,"quotas":second_page}}),
    ]));
    let client = LlmClientBuilder::with_transport(http.clone(), &[profile("qwen")])
        .build()
        .unwrap();
    let mut query = AccountQuery::new(AccountIdentity::ApiKey);
    query.credential = Some(Secret::new("dashscope-key".into()));
    query.selector.workspace_id = Some("ws-demo".into());

    let snapshot = client.account_usage("qwen", &query).await.unwrap();
    match snapshot.quota_windows {
        AccountMetric::Available { value, .. } => assert_eq!(value.len(), 101),
        other => panic!("expected all Qwen quota pages, got {other:?}"),
    }
    let requests = http.requests();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].url.ends_with("?page_no=1&page_size=100"));
    assert!(requests[1].url.ends_with("?page_no=2&page_size=100"));
}

#[tokio::test]
async fn minimax_usage_maps_only_explicit_quota_fields() {
    let http = Arc::new(QueueHttp::with_json(vec![json!({
        "base_resp":{"status_code":0,"status_msg":"success"},
        "model_remains":[{
            "model_name":"MiniMax-M3",
            "start_time":1782381600000_u64,
            "end_time":1782399600000_u64,
            "current_interval_total_count":100,
            "current_interval_usage_count":80,
            "current_interval_remaining_percent":20,
            "current_weekly_total_count":500,
            "current_weekly_remaining_count":400
        }]
    })]));
    let client = LlmClientBuilder::with_transport(http.clone(), &[profile("minimax-intl")])
        .build()
        .unwrap();
    let mut query = AccountQuery::new(AccountIdentity::ApiKey);
    query.credential = Some(Secret::new("token-plan-key".into()));
    let snapshot = client.account_usage("minimax-intl", &query).await.unwrap();
    match snapshot.quota_windows {
        AccountMetric::Available { value, .. } => {
            assert_eq!(value.len(), 2);
            assert_eq!(value[0].limit, Some(100));
            assert_eq!(value[0].remaining_percent, Some(20.0));
            assert_eq!(value[0].unit.as_deref(), Some("tokens"));
            assert_eq!(
                value[0].used, None,
                "ambiguous usage_count is not reinterpreted"
            );
            assert_eq!(value[1].remaining, Some(400));
            assert_eq!(value[1].used, Some(100));
        }
        other => panic!("expected MiniMax token-plan quotas, got {other:?}"),
    }
    assert_eq!(
        http.requests()[0].url,
        "https://api.minimax.io/v1/token_plan/remains"
    );
    assert_eq!(
        http.requests()[0].headers[0],
        ("authorization".into(), "Bearer token-plan-key".into())
    );
}

#[tokio::test]
async fn minimax_china_quota_uses_the_china_api_host() {
    let http = Arc::new(QueueHttp::with_json(vec![json!({
        "base_resp":{"status_code":0,"status_msg":"success"},
        "model_remains":[]
    })]));
    let client = LlmClientBuilder::with_transport(http.clone(), &[profile("minimax")])
        .build()
        .unwrap();
    let mut query = AccountQuery::new(AccountIdentity::ApiKey);
    query.credential = Some(Secret::new("token-plan-key".into()));

    let _snapshot = client.account_usage("minimax", &query).await.unwrap();
    assert_eq!(
        http.requests()[0].url,
        "https://api.minimaxi.com/v1/token_plan/remains"
    );
}
