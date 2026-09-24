use super::*;
pub struct CopilotAccountSource {
    rpc: Arc<dyn AccountRpc>,
}

impl CopilotAccountSource {
    pub fn new(rpc: Arc<dyn AccountRpc>) -> Self {
        Self { rpc }
    }
}

#[async_trait]
impl AccountUsageSource for CopilotAccountSource {
    fn requires_profile_binding(&self, query: &AccountQuery) -> bool {
        query.credential.is_none()
    }

    async fn fetch(
        &self,
        context: &crate::account::AccountFetchContext<'_>,
        report: &mut crate::account::AccountReport,
    ) -> Result<(), AccountFailure> {
        let query = context.query;
        let _now = context.fetched_at_unix;
        let _http = context.http();
        let snapshot = report;
        snapshot.unsupported();
        snapshot.quota_windows = None;
        snapshot.subscription = Some(AccountMetric::NotReported);
        let params = query.credential.as_ref().map_or_else(
            || json!({}),
            |token| json!({"gitHubToken": token.expose_secret()}),
        );
        let quota = context
            .rpc(self.rpc.as_ref(), "account.getQuota", &params)
            .await
            .and_then(|value| copilot_windows(&value));
        // Quota exists for free-tier users too; this RPC does not attest a paid plan.
        snapshot.subscription = Some(AccountMetric::NotReported);
        snapshot.quota_windows = Some(field_from_result(
            quota,
            user_scope(None),
            "copilot account.getQuota",
        ));
        Ok(())
    }
}

fn copilot_windows(value: &Value) -> Result<Vec<AccountQuotaWindow>, AccountFailure> {
    let snapshots = value
        .get("quotaSnapshots")
        .and_then(Value::as_object)
        .ok_or(AccountFailure::InvalidResponse)?;
    snapshots
        .iter()
        .map(|(name, quota)| {
            let raw_limit = quota
                .get("entitlementRequests")
                .and_then(Value::as_i64)
                .ok_or(AccountFailure::InvalidResponse)?;
            let limit = match raw_limit {
                -1 => None, // Documented unlimited sentinel.
                0.. => Some(raw_limit as u64),
                _ => return Err(AccountFailure::InvalidResponse),
            };
            let used = quota.get("usedRequests").and_then(Value::as_u64);
            let remaining_percent = quota
                .get("remainingPercentage")
                .and_then(Value::as_f64)
                .filter(|number| (0.0..=100.0).contains(number));
            let resets_at_unix = optional_time(quota.get("resetDate"))?;
            Ok(AccountQuotaWindow {
                name: name.clone(),
                duration_mins: None,
                used_percent: remaining_percent.map(|value| 100.0 - value),
                remaining_percent,
                resets_at_unix,
                limit,
                used,
                remaining: limit
                    .zip(used)
                    .map(|(limit, used)| limit.saturating_sub(used)),
                limit_decimal: None,
                remaining_decimal: None,
                unit: Some("requests".into()),
            })
        })
        .collect()
}
