use super::*;
#[async_trait]
impl AccountUsageSource for DeepSeek {
    async fn fetch(
        &self,
        context: &crate::account::AccountFetchContext<'_>,
        report: &mut crate::account::AccountReport,
    ) -> Result<(), AccountFailure> {
        let query = context.query;
        let _now = context.fetched_at_unix;
        let http = context.http();
        let result = report;
        result.unsupported();
        result.balance = None;
        result.balance = Some(match &query.credential {
            Some(key) => {
                let balance = get_json(http, DEEPSEEK_BALANCE.into(), header_bearer(key))
                    .await
                    .and_then(parse_deepseek_balance);
                field_from_result(balance, user_scope(), DEEPSEEK_BALANCE)
            }
            None => AccountMetric::CredentialRequired,
        });
        Ok(())
    }
}

fn parse_deepseek_balance(body: Value) -> Result<Vec<AccountBalance>, AccountFailure> {
    let is_available = body
        .get("is_available")
        .and_then(Value::as_bool)
        .ok_or(AccountFailure::InvalidResponse)?;
    let rows = body
        .get("balance_infos")
        .and_then(Value::as_array)
        .ok_or(AccountFailure::InvalidResponse)?;
    rows.iter()
        .map(|row| {
            let unit = row
                .get("currency")
                .and_then(Value::as_str)
                .filter(|unit| matches!(*unit, "USD" | "CNY"))
                .ok_or(AccountFailure::InvalidResponse)?;
            let remaining = amount(
                row.get("total_balance")
                    .ok_or(AccountFailure::InvalidResponse)?,
            )?;
            Ok(AccountBalance {
                unit: unit.into(),
                remaining,
                total: None,
                is_available: Some(is_available),
            })
        })
        .collect()
}
