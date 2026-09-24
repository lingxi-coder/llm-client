use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{account::*, protocol::*, *};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
fn profile(name: &str) -> ProviderProfile {
    serde_json::from_value(json!({"profile_name":name,"provider_id":"test","base_url":"https://example.test","protocol":"open_ai_chat","auth":"none","models":[]})).unwrap()
}
fn query() -> AccountQuery {
    let mut q = AccountQuery::new(AccountIdentity::ApiKey);
    q.execution = AccountExecutionOptions {
        total_timeout: Duration::from_secs(60),
        operation_timeout: Duration::from_secs(30),
    };
    q
}
fn req() -> HttpRequest {
    HttpRequest {
        method: "GET".into(),
        url: "https://example.test".into(),
        body: Bytes::new(),
        headers: vec![],
        timeout: None,
    }
}
struct HungBody;
#[async_trait]
impl Transport for HungBody {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: stream::pending().boxed(),
        })
    }
}
struct HungRpc;
#[async_trait]
impl AccountRpc for HungRpc {
    async fn call(&self, _: &str, _: &Value) -> Result<Value, AccountFailure> {
        std::future::pending().await
    }
}
struct PartialSource;
#[async_trait]
impl AccountUsageSource for PartialSource {
    async fn fetch(
        &self,
        c: &AccountFetchContext<'_>,
        r: &mut AccountReport,
    ) -> Result<(), AccountFailure> {
        r.balance = Some(AccountMetric::NotReported);
        r.subscription = Some(AccountMetric::CredentialRequired);
        c.rpc(&HungRpc, "pending", &json!({})).await?;
        Ok(())
    }
}
#[tokio::test(start_paused = true)]
async fn account_rpc_timeout_preserves_completed_and_static_fields() {
    let p = profile("p");
    let q = query();
    let c = AccountFetchContext::new(&p, &q, (0, 1), 1, Arc::new(HungBody));
    let start = tokio::time::Instant::now();
    let result = c.collect(&PartialSource).await;
    assert_eq!(start.elapsed(), Duration::from_secs(30));
    assert_eq!(result.balance, AccountMetric::NotReported);
    assert_eq!(result.subscription, AccountMetric::CredentialRequired);
    assert_eq!(
        result.token_usage,
        AccountMetric::Failed {
            reason: AccountFailure::Timeout
        }
    );
}
struct HttpSource;
#[async_trait]
impl AccountUsageSource for HttpSource {
    async fn fetch(
        &self,
        c: &AccountFetchContext<'_>,
        r: &mut AccountReport,
    ) -> Result<(), AccountFailure> {
        r.balance = Some(AccountMetric::NotReported);
        lingxi_llm_client::transport::HttpExecutor::new(c.http())
            .execute(req())
            .await
            .map_err(|e| {
                if matches!(e, LlmError::TransportTimeout { .. }) {
                    AccountFailure::Timeout
                } else {
                    AccountFailure::Transport
                }
            })?;
        Ok(())
    }
}
#[tokio::test(start_paused = true)]
async fn account_http_deadline_includes_body_even_when_transport_ignores_timeout() {
    let p = profile("p");
    let mut q = query();
    q.execution.total_timeout = Duration::from_secs(10);
    let c = AccountFetchContext::new(&p, &q, (0, 1), 1, Arc::new(HungBody));
    let start = tokio::time::Instant::now();
    let result = c.collect(&HttpSource).await;
    assert_eq!(start.elapsed(), Duration::from_secs(10));
    assert_eq!(result.balance, AccountMetric::NotReported);
    assert_eq!(
        result.token_usage,
        AccountMetric::Failed {
            reason: AccountFailure::Timeout
        }
    );
}
struct PageRpc(AtomicUsize);
#[async_trait]
impl AccountRpc for PageRpc {
    async fn call(&self, _: &str, _: &Value) -> Result<Value, AccountFailure> {
        self.0.fetch_add(1, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_secs(25)).await;
        Ok(json!({}))
    }
}
struct Pages(PageRpc);
#[async_trait]
impl AccountUsageSource for Pages {
    async fn fetch(
        &self,
        c: &AccountFetchContext<'_>,
        r: &mut AccountReport,
    ) -> Result<(), AccountFailure> {
        for _ in 0..3 {
            c.rpc(&self.0, "page", &json!({})).await?;
            r.balance = Some(AccountMetric::NotReported);
        }
        Ok(())
    }
}
#[tokio::test(start_paused = true)]
async fn subsequent_rpcs_share_remaining_total_budget() {
    let p = profile("p");
    let q = query();
    let c = AccountFetchContext::new(&p, &q, (0, 1), 1, Arc::new(HungBody));
    let source = Pages(PageRpc(AtomicUsize::new(0)));
    let start = tokio::time::Instant::now();
    let result = c.collect(&source).await;
    assert_eq!(start.elapsed(), Duration::from_secs(60));
    assert_eq!(source.0 .0.load(Ordering::Relaxed), 3);
    assert_eq!(result.balance, AccountMetric::NotReported);
    assert_eq!(
        result.token_usage,
        AccountMetric::Failed {
            reason: AccountFailure::Timeout
        }
    );
}
struct Scheduling {
    active: AtomicUsize,
    max: AtomicUsize,
    started: Mutex<Vec<(String, tokio::time::Instant)>>,
}
struct Slot<'a>(&'a AtomicUsize);
impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}
#[async_trait]
impl AccountUsageSource for Scheduling {
    async fn fetch(
        &self,
        c: &AccountFetchContext<'_>,
        r: &mut AccountReport,
    ) -> Result<(), AccountFailure> {
        let count = self.active.fetch_add(1, Ordering::Relaxed) + 1;
        self.max.fetch_max(count, Ordering::Relaxed);
        let _slot = Slot(&self.active);
        self.started
            .lock()
            .unwrap()
            .push((c.profile.profile_name.clone(), tokio::time::Instant::now()));
        tokio::time::sleep(Duration::from_secs(if c.profile.profile_name == "0" {
            50
        } else {
            1
        }))
        .await;
        r.balance = Some(AccountMetric::NotReported);
        Ok(())
    }
}
#[tokio::test(start_paused = true)]
async fn account_batch_fills_free_slots_and_preserves_input_order() {
    let profiles = (0..6).map(|i| profile(&i.to_string())).collect::<Vec<_>>();
    let source = Arc::new(Scheduling {
        active: AtomicUsize::new(0),
        max: AtomicUsize::new(0),
        started: Mutex::new(vec![]),
    });
    let mut builder = LlmClientBuilder::with_transport(Arc::new(HungBody), &profiles);
    builder.register_account_source("test", AccountIdentity::ApiKey, source.clone());
    let client = builder.with_region(Region::International).build().unwrap();
    let queries: BTreeMap<_, _> = profiles
        .iter()
        .map(|p| (p.profile_name.clone(), query()))
        .collect();
    let start = tokio::time::Instant::now();
    let results = client.accounts_usage(&queries).await;
    assert_eq!(
        results.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
        profiles
            .iter()
            .map(|p| p.profile_name.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(source.max.load(Ordering::Relaxed), 4);
    assert_eq!(source.active.load(Ordering::Relaxed), 0);
    assert!(
        source
            .started
            .lock()
            .unwrap()
            .iter()
            .find(|(n, _)| n == "5")
            .unwrap()
            .1
            .duration_since(start)
            < Duration::from_secs(50)
    );
    assert!(results.into_iter().all(|(_, result)| result.is_ok()));
}
#[tokio::test(start_paused = true)]
async fn unrepresentable_timeout_does_not_disable_account_budget() {
    let p = profile("p");
    let mut q = query();
    q.execution.total_timeout = Duration::MAX;
    q.execution.operation_timeout = Duration::MAX;
    let c = AccountFetchContext::new(&p, &q, (0, 1), 1, Arc::new(HungBody));
    assert_eq!(
        c.collect(&PartialSource).await.token_usage,
        AccountMetric::Failed {
            reason: AccountFailure::Timeout
        }
    );
}
struct BrokenBody;
#[async_trait]
impl Transport for BrokenBody {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        Ok(StreamResponse {
            status: 200,
            headers: vec![],
            body: stream::iter(vec![Err(LlmError::StreamInterrupted {
                message: "socket lost".into(),
            })])
            .boxed(),
        })
    }
}
#[tokio::test]
async fn buffered_body_failure_is_classified_for_network_failover() {
    let error = lingxi_llm_client::transport::HttpExecutor::new(&BrokenBody)
        .execute(req())
        .await
        .unwrap_err();
    assert!(FailoverTriggers::DEFAULT.matches(&error));
    assert!(matches!(error, LlmError::Transport { .. }));
}
struct FailedStatus;
#[async_trait]
impl Transport for FailedStatus {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        Ok(StreamResponse {
            status: 429,
            headers: vec![("retry-after".into(), "2".into())],
            body: stream::iter(vec![
                Ok(Bytes::from(vec![b'x'; 100_000])),
                Err(LlmError::StreamInterrupted {
                    message: "closed".into(),
                }),
            ])
            .boxed(),
        })
    }
}
#[tokio::test]
async fn errors_keep_status_headers_and_only_a_bounded_prefix() {
    let response = lingxi_llm_client::transport::HttpExecutor::new(&FailedStatus)
        .execute(req())
        .await
        .unwrap();
    assert_eq!(response.status, 429);
    assert_eq!(response.headers[0].1, "2");
    assert_eq!(response.body.len(), 64 * 1024);
}
#[test]
fn actual_cost_requires_the_responses_complete_usage_report() {
    let mut p = profile("p");
    p.models = vec![serde_json::from_value(json!({"display_model":"m","request_model":"m","billing_model":"m","pricing":{"input_per_million":1.0}})).unwrap()];
    let context = CodecContext::new(&p, "m", RequestMode::Complete);
    let client = LlmClientBuilder::with_transport(Arc::new(HungBody), &[p])
        .with_region(Region::International)
        .build()
        .unwrap();
    let route = client.resolve("m").unwrap();
    let mut response = OpenAiChatCodec.decode_response(&HttpResponse {status:200,headers:vec![],body:json!({"model":"m","choices":[{"message":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000000,"completion_tokens":0}}).to_string().into()}, &context).unwrap();
    response.executed_profile = Some("p".into());
    for state in [
        UsageState::Missing,
        UsageState::Partial,
        UsageState::Invalid,
    ] {
        response.usage.state = state;
        assert!(matches!(
            client.estimate_actual_cost(&route, &response, Submission::Interactive),
            Err(LlmError::CostUnavailable { .. })
        ));
    }
    response.usage.state = UsageState::Complete;
    assert_eq!(
        client
            .estimate_actual_cost(&route, &response, Submission::Interactive)
            .unwrap()
            .total_cost,
        1.0
    );
}
