use super::*;
pub(super) async fn openai_usage(
    query: &AccountQuery,
    range: (u64, u64),
    key: &str,
    http: &dyn Transport,
) -> Result<AccountTokenUsage, AccountFailure> {
    let mut params = vec![
        ("start_time", range.0.to_string()),
        ("end_time", range.1.to_string()),
        ("bucket_width", "1d".into()),
        ("limit", "31".into()),
        ("group_by[]", "model".into()),
    ];
    if let Some(id) = &query.selector.project_id {
        params.push(("project_ids[]", id.clone()));
    }
    if let Some(id) = &query.selector.api_key_id {
        params.push(("api_key_ids[]", id.clone()));
    }
    let mut headers = vec![("authorization".into(), format!("Bearer {key}"))];
    if let Some(id) = &query.selector.organization_id {
        headers.push(("openai-organization".into(), id.clone()));
    }
    let mut buckets = Vec::new();
    let mut page = None::<String>;
    for _ in 0..MAX_PAGES {
        let mut request_params = params.clone();
        if let Some(cursor) = &page {
            request_params.push(("page", cursor.clone()));
        }
        let body = get_json(
            http,
            url(
                "https://api.openai.com/v1/organization/usage/completions",
                &request_params,
            )?,
            headers.clone(),
        )
        .await?;
        let data = body
            .get("data")
            .and_then(Value::as_array)
            .ok_or(AccountFailure::InvalidResponse)?;
        for bucket in data {
            let start = number(bucket, "start_time")?;
            let end = number(bucket, "end_time")?;
            if start >= end {
                return Err(AccountFailure::InvalidResponse);
            }
            let results = bucket
                .get("results")
                .and_then(Value::as_array)
                .ok_or(AccountFailure::InvalidResponse)?;
            for result in results {
                let input = number(result, "input_tokens")?;
                let output = number(result, "output_tokens")?;
                let cached = optional_number(result, "input_cached_tokens")?;
                if cached.is_some_and(|cached| cached > input) {
                    return Err(AccountFailure::InvalidResponse);
                }
                let model = result
                    .get("model")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                buckets.push(AccountTokenBucket {
                    start_unix: start,
                    end_unix: end,
                    model,
                    input_tokens: Some(input),
                    output_tokens: Some(output),
                    cached_input_tokens: cached,
                    cache_write_tokens: None,
                    total_tokens: input.checked_add(output),
                });
            }
        }
        match next_page(&body, "has_more", "next_page")? {
            Some(cursor) if page.as_deref() != Some(cursor) => page = Some(cursor.into()),
            Some(_) => return Err(AccountFailure::InvalidResponse),
            None => {
                return Ok(AccountTokenUsage {
                    lifetime_tokens: None,
                    buckets,
                });
            }
        }
    }
    Err(AccountFailure::InvalidResponse)
}

pub(super) async fn openai_costs(
    query: &AccountQuery,
    range: (u64, u64),
    key: &str,
    http: &dyn Transport,
) -> Result<AccountCostUsage, AccountFailure> {
    let mut params = vec![
        ("start_time", range.0.to_string()),
        ("end_time", range.1.to_string()),
        ("bucket_width", "1d".into()),
        ("limit", "180".into()),
    ];
    if let Some(id) = &query.selector.project_id {
        params.push(("project_ids[]", id.clone()));
        params.push(("group_by[]", "project_id".into()));
    }
    if let Some(id) = &query.selector.api_key_id {
        params.push(("api_key_ids[]", id.clone()));
        params.push(("group_by[]", "api_key_id".into()));
    }
    let mut headers = vec![("authorization".into(), format!("Bearer {key}"))];
    if let Some(id) = &query.selector.organization_id {
        headers.push(("openai-organization".into(), id.clone()));
    }
    let mut buckets = Vec::new();
    let mut page = None::<String>;
    for _ in 0..MAX_PAGES {
        let mut request_params = params.clone();
        if let Some(cursor) = &page {
            request_params.push(("page", cursor.clone()));
        }
        let body = get_json(
            http,
            url(
                "https://api.openai.com/v1/organization/costs",
                &request_params,
            )?,
            headers.clone(),
        )
        .await?;
        let data = body
            .get("data")
            .and_then(Value::as_array)
            .ok_or(AccountFailure::InvalidResponse)?;
        for bucket in data {
            let start = number(bucket, "start_time")?;
            let end = number(bucket, "end_time")?;
            if start >= end {
                return Err(AccountFailure::InvalidResponse);
            }
            let results = bucket
                .get("results")
                .and_then(Value::as_array)
                .ok_or(AccountFailure::InvalidResponse)?;
            for result in results {
                let amount = result
                    .get("amount")
                    .ok_or(AccountFailure::InvalidResponse)?;
                let value = amount
                    .get("value")
                    .and_then(Value::as_number)
                    .ok_or(AccountFailure::InvalidResponse)?;
                let currency = amount
                    .get("currency")
                    .and_then(Value::as_str)
                    .ok_or(AccountFailure::InvalidResponse)?;
                if !valid_currency(currency) {
                    return Err(AccountFailure::InvalidResponse);
                }
                buckets.push(AccountCostBucket {
                    start_unix: start,
                    end_unix: end,
                    currency: currency.to_ascii_uppercase(),
                    amount: value.to_string(),
                    model: None,
                });
            }
        }
        match next_page(&body, "has_more", "next_page")? {
            Some(cursor) if page.as_deref() != Some(cursor) => page = Some(cursor.into()),
            Some(_) => return Err(AccountFailure::InvalidResponse),
            None => return Ok(AccountCostUsage { buckets }),
        }
    }
    Err(AccountFailure::InvalidResponse)
}
