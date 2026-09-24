use super::*;
#[async_trait]
impl AccountUsageSource for Moonshot {
    async fn fetch(
        &self,
        context: &crate::account::AccountFetchContext<'_>,
        report: &mut crate::account::AccountReport,
    ) -> Result<(), AccountFailure> {
        let profile = context.profile;
        let query = context.query;
        let _now = context.fetched_at_unix;
        let http = context.http();
        let result = report;
        result.unsupported();
        if profile.profile_name == "kimi-code" {
            return Ok(());
        }
        let (url, unit) = if official_origin(&profile.base_url, "https://api.moonshot.cn") {
            (MOONSHOT_CN_BALANCE, "CNY")
        } else if official_origin(&profile.base_url, "https://api.moonshot.ai") {
            (MOONSHOT_GLOBAL_BALANCE, "USD")
        } else {
            // Kimi Code has the same provider ID but a separate subscription API.
            return Ok(());
        };
        result.balance = None;
        result.balance = Some(match &query.credential {
            Some(key) => {
                let balance = get_json(http, url.into(), header_bearer(key))
                    .await
                    .and_then(|body| parse_moonshot_balance(body, unit));
                field_from_result(balance.map(|row| vec![row]), user_scope(), url)
            }
            None => AccountMetric::CredentialRequired,
        });
        Ok(())
    }
}

fn official_origin(base: &str, origin: &str) -> bool {
    base == origin
        || base
            .strip_prefix(origin)
            .is_some_and(|tail| tail.starts_with('/'))
}

fn parse_moonshot_balance(body: Value, unit: &str) -> Result<AccountBalance, AccountFailure> {
    let code = body
        .get("code")
        .and_then(Value::as_i64)
        .ok_or(AccountFailure::InvalidResponse)?;
    let status = body
        .get("status")
        .and_then(Value::as_bool)
        .ok_or(AccountFailure::InvalidResponse)?;
    if code != 0 || !status {
        return Err(AccountFailure::ProviderError);
    }
    let remaining = amount(
        body.get("data")
            .and_then(|data| data.get("available_balance"))
            .ok_or(AccountFailure::InvalidResponse)?,
    )?;
    Ok(AccountBalance {
        unit: unit.into(),
        remaining,
        total: None,
        is_available: None,
    })
}
