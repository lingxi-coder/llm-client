use super::*;
#[async_trait]
impl AccountUsageSource for OpenRouter {
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
        result.balance = if query.management_credential.is_none() {
            Some(AccountMetric::CredentialRequired)
        } else {
            None
        };
        result.quota_windows = None;
        let key_details = match &query.credential {
            Some(key) => Some(
                get_json(http, OPENROUTER_KEY.into(), header_bearer(key))
                    .await
                    .and_then(parse_openrouter_key),
            ),
            None => None,
        };

        result.quota_windows = Some(match &key_details {
            Some(Ok(details)) => details
                .quota
                .clone()
                .map(|quota| AccountMetric::available(key_scope(), OPENROUTER_KEY, vec![quota]))
                .unwrap_or(AccountMetric::NotReported),
            Some(Err(reason)) => AccountMetric::Failed { reason: *reason },
            None => AccountMetric::CredentialRequired,
        });

        result.balance = Some(if let Some(key) = &query.management_credential {
            let balance = get_json(http, OPENROUTER_CREDITS.into(), header_bearer(key))
                .await
                .and_then(parse_openrouter_credits);
            field_from_result(
                balance.map(|value| vec![value]),
                user_scope(),
                OPENROUTER_CREDITS,
            )
        } else {
            AccountMetric::CredentialRequired
        });
        Ok(())
    }
}

struct OpenRouterKeyDetails {
    quota: Option<AccountQuotaWindow>,
}

fn parse_openrouter_key(body: Value) -> Result<OpenRouterKeyDetails, AccountFailure> {
    let data = body
        .get("data")
        .and_then(Value::as_object)
        .ok_or(AccountFailure::InvalidResponse)?;
    let Some(limit) = data.get("limit").filter(|value| !value.is_null()) else {
        return Ok(OpenRouterKeyDetails { quota: None });
    };
    let limit = decimal(limit)?;
    let remaining = decimal(
        data.get("limit_remaining")
            .ok_or(AccountFailure::InvalidResponse)?,
    )?;
    let limit_text = limit.to_text();
    let remaining_text = remaining.to_text();
    let duration_mins = match data.get("limit_reset").and_then(Value::as_str) {
        Some("daily") => Some(1_440),
        Some("weekly") => Some(10_080),
        Some("monthly") | None => None,
        Some(_) => None,
    };
    let remaining_percent =
        if limit.units > 0 && remaining.units >= 0 && limit.subtract(remaining)?.units >= 0 {
            let fraction = remaining.to_f64() / limit.to_f64();
            Some(fraction * 100.0)
        } else {
            None
        };
    let quota = AccountQuotaWindow {
        name: "api_key_spend_limit".into(),
        duration_mins,
        used_percent: remaining_percent.map(|value| 100.0 - value),
        remaining_percent,
        resets_at_unix: None,
        // Money can have fractional units; these integer slots must stay empty.
        limit: None,
        used: None,
        remaining: None,
        limit_decimal: Some(limit_text),
        remaining_decimal: Some(remaining_text),
        unit: Some("USD".into()),
    };
    Ok(OpenRouterKeyDetails { quota: Some(quota) })
}

fn parse_openrouter_credits(body: Value) -> Result<AccountBalance, AccountFailure> {
    let data = body.get("data").ok_or(AccountFailure::InvalidResponse)?;
    let total = decimal(
        data.get("total_credits")
            .ok_or(AccountFailure::InvalidResponse)?,
    )?;
    let used = decimal(
        data.get("total_usage")
            .ok_or(AccountFailure::InvalidResponse)?,
    )?;
    Ok(AccountBalance {
        unit: "USD".into(),
        remaining: total.subtract(used)?.to_text(),
        total: Some(total.to_text()),
        is_available: None,
    })
}
