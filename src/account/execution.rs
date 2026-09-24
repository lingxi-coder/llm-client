//! Query budgets and incremental account reports.
use super::*;
use crate::{
    runtime::Deadline,
    transport::{HttpExecutor, StreamResponse},
};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccountExecutionOptions {
    pub total_timeout: Duration,
    pub operation_timeout: Duration,
}
impl Default for AccountExecutionOptions {
    fn default() -> Self {
        Self {
            total_timeout: Duration::from_secs(60),
            operation_timeout: Duration::from_secs(30),
        }
    }
}

/// Each completed field is committed independently of the source's future.
/// Unset fields become NotReported on success, or Failed on cancellation by a
/// deadline. Account identity is assigned only by the client.
#[derive(Debug, Default)]
pub struct AccountReport {
    pub balance: Option<AccountMetric<Vec<AccountBalance>>>,
    pub token_usage: Option<AccountMetric<AccountTokenUsage>>,
    pub cost_usage: Option<AccountMetric<AccountCostUsage>>,
    pub quota_windows: Option<AccountMetric<Vec<AccountQuotaWindow>>>,
    pub subscription: Option<AccountMetric<AccountSubscription>>,
}
impl AccountReport {
    pub(crate) fn unsupported(&mut self) {
        self.balance = Some(AccountMetric::Unsupported);
        self.token_usage = Some(AccountMetric::Unsupported);
        self.cost_usage = Some(AccountMetric::Unsupported);
        self.quota_windows = Some(AccountMetric::Unsupported);
        self.subscription = Some(AccountMetric::Unsupported);
    }
    pub(crate) fn finish(
        self,
        profile: &ProviderProfile,
        query: &AccountQuery,
        now: u64,
        outcome: Result<(), AccountFailure>,
    ) -> AccountSnapshot {
        fn metric<T>(
            value: Option<AccountMetric<T>>,
            outcome: Result<(), AccountFailure>,
        ) -> AccountMetric<T> {
            value.unwrap_or_else(|| match outcome {
                Ok(()) => AccountMetric::NotReported,
                Err(reason) => AccountMetric::Failed { reason },
            })
        }
        AccountSnapshot {
            profile_name: profile.profile_name.clone(),
            provider_id: profile.provider_id.clone(),
            identity: query.identity,
            fetched_at_unix: now,
            balance: metric(self.balance, outcome),
            token_usage: metric(self.token_usage, outcome),
            cost_usage: metric(self.cost_usage, outcome),
            quota_windows: metric(self.quota_windows, outcome),
            subscription: metric(self.subscription, outcome),
        }
    }
}

pub struct AccountFetchContext<'a> {
    pub profile: &'a ProviderProfile,
    pub query: &'a AccountQuery,
    pub range: (u64, u64),
    pub fetched_at_unix: u64,
    pub(crate) deadline: Deadline,
    http: AccountTransport,
}
impl<'a> AccountFetchContext<'a> {
    pub fn new(
        profile: &'a ProviderProfile,
        query: &'a AccountQuery,
        range: (u64, u64),
        now: u64,
        http: Arc<dyn Transport>,
    ) -> Self {
        let deadline = Deadline::after(Some(query.execution.total_timeout));
        Self {
            profile,
            query,
            range,
            fetched_at_unix: now,
            deadline,
            http: AccountTransport {
                inner: http,
                deadline,
                operation_timeout: query.execution.operation_timeout,
            },
        }
    }
    /// Run a source with this context's deadline, preserving fields completed
    /// before a timeout. The returned identity always comes from the context.
    pub async fn collect(&self, source: &dyn AccountUsageSource) -> AccountSnapshot {
        let mut report = AccountReport::default();
        let outcome = self
            .deadline
            .run(source.fetch(self, &mut report))
            .await
            .unwrap_or(Err(AccountFailure::Timeout));
        report.finish(self.profile, self.query, self.fetched_at_unix, outcome)
    }
    /// All calls through this transport share the query budget. It enforces
    /// per-operation timeouts even if the injected transport ignores them.
    pub fn http(&self) -> &dyn Transport {
        &self.http
    }
    pub async fn rpc(
        &self,
        rpc: &dyn AccountRpc,
        method: &str,
        params: &Value,
    ) -> Result<Value, AccountFailure> {
        self.deadline
            .cap(Some(self.query.execution.operation_timeout))
            .run(rpc.call(method, params))
            .await
            .map_err(map_transport_error)?
    }
}
struct AccountTransport {
    inner: Arc<dyn Transport>,
    deadline: Deadline,
    operation_timeout: Duration,
}
#[async_trait]
impl Transport for AccountTransport {
    async fn send(&self, req: HttpRequest) -> Result<StreamResponse, crate::protocol::LlmError> {
        HttpExecutor::new(self.inner.as_ref())
            .with_deadline(self.deadline.cap(Some(self.operation_timeout)))
            .send(req)
            .await
    }
}
pub(crate) fn map_transport_error(error: crate::protocol::LlmError) -> AccountFailure {
    if matches!(error, crate::protocol::LlmError::TransportTimeout { .. }) {
        AccountFailure::Timeout
    } else {
        AccountFailure::Transport
    }
}
