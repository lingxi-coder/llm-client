use super::*;
pub(super) fn anthropic_headers(identity: AccountIdentity, key: &str) -> Vec<(String, String)> {
    let credential = match identity {
        AccountIdentity::ApiKey => ("x-api-key".into(), key.into()),
        AccountIdentity::AuthUser => ("authorization".into(), format!("Bearer {key}")),
    };
    vec![
        credential,
        ("anthropic-version".into(), "2023-06-01".into()),
    ]
}

pub(super) async fn anthropic_usage(
    query: &AccountQuery,
    range: (u64, u64),
    key: &str,
    http: &dyn Transport,
) -> Result<AccountTokenUsage, AccountFailure> {
    let mut params = vec![
        ("starting_at", format_rfc3339(range.0)?),
        ("ending_at", format_rfc3339(range.1)?),
        ("bucket_width", "1d".into()),
        ("limit", "31".into()),
        ("group_by[]", "model".into()),
    ];
    if let Some(id) = &query.selector.workspace_id {
        params.push(("workspace_ids[]", id.clone()));
    }
    if let Some(id) = &query.selector.api_key_id {
        params.push(("api_key_ids[]", id.clone()));
    }
    let headers = anthropic_headers(query.identity, key);
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
                "https://api.anthropic.com/v1/organizations/usage_report/messages",
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
            let start = parse_rfc3339(
                bucket
                    .get("starting_at")
                    .and_then(Value::as_str)
                    .ok_or(AccountFailure::InvalidResponse)?,
            )?;
            let end = parse_rfc3339(
                bucket
                    .get("ending_at")
                    .and_then(Value::as_str)
                    .ok_or(AccountFailure::InvalidResponse)?,
            )?;
            if start >= end {
                return Err(AccountFailure::InvalidResponse);
            }
            let results = bucket
                .get("results")
                .and_then(Value::as_array)
                .ok_or(AccountFailure::InvalidResponse)?;
            for result in results {
                let uncached = number(result, "uncached_input_tokens")?;
                let read = number(result, "cache_read_input_tokens")?;
                let creation = result.get("cache_creation");
                let created_5m = creation
                    .map(|value| optional_number(value, "ephemeral_5m_input_tokens"))
                    .transpose()?
                    .flatten()
                    .unwrap_or(0);
                let created_1h = creation
                    .map(|value| optional_number(value, "ephemeral_1h_input_tokens"))
                    .transpose()?
                    .flatten()
                    .unwrap_or(0);
                let input = uncached
                    .checked_add(read)
                    .and_then(|n| n.checked_add(created_5m))
                    .and_then(|n| n.checked_add(created_1h))
                    .ok_or(AccountFailure::InvalidResponse)?;
                let output = number(result, "output_tokens")?;
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
                    cached_input_tokens: Some(read),
                    cache_write_tokens: created_5m.checked_add(created_1h),
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

pub(super) async fn anthropic_costs(
    query: &AccountQuery,
    range: (u64, u64),
    key: &str,
    http: &dyn Transport,
) -> Result<AccountCostUsage, AccountFailure> {
    let mut params = vec![
        ("starting_at", format_rfc3339(range.0)?),
        ("ending_at", format_rfc3339(range.1)?),
        ("bucket_width", "1d".into()),
        ("limit", "31".into()),
        ("group_by[]", "description".into()),
    ];
    if query.selector.workspace_id.is_some() {
        params.push(("group_by[]", "workspace_id".into()));
    }
    let headers = anthropic_headers(query.identity, key);
    let mut buckets = Vec::new();
    let mut page = None::<String>;
    let mut selected_workspace_is_default = None;
    for _ in 0..MAX_PAGES {
        let mut request_params = params.clone();
        if let Some(cursor) = &page {
            request_params.push(("page", cursor.clone()));
        }
        let body = get_json(
            http,
            url(
                "https://api.anthropic.com/v1/organizations/cost_report",
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
            let start = parse_rfc3339(
                bucket
                    .get("starting_at")
                    .and_then(Value::as_str)
                    .ok_or(AccountFailure::InvalidResponse)?,
            )?;
            let end = parse_rfc3339(
                bucket
                    .get("ending_at")
                    .and_then(Value::as_str)
                    .ok_or(AccountFailure::InvalidResponse)?,
            )?;
            if start >= end {
                return Err(AccountFailure::InvalidResponse);
            }
            let results = bucket
                .get("results")
                .and_then(Value::as_array)
                .ok_or(AccountFailure::InvalidResponse)?;
            for result in results {
                if let Some(workspace) = &query.selector.workspace_id {
                    match result.get("workspace_id") {
                        Some(Value::String(reported)) if reported == workspace => {}
                        Some(Value::String(_)) => continue,
                        Some(Value::Null) => {
                            let is_default = match selected_workspace_is_default {
                                Some(value) => value,
                                None => {
                                    let value = anthropic_default_workspace(
                                        workspace,
                                        query.identity,
                                        key,
                                        http,
                                    )
                                    .await?;
                                    selected_workspace_is_default = Some(value);
                                    value
                                }
                            };
                            if !is_default {
                                continue;
                            }
                        }
                        _ => return Err(AccountFailure::InvalidResponse),
                    }
                }
                let minor = result
                    .get("amount")
                    .and_then(Value::as_str)
                    .ok_or(AccountFailure::InvalidResponse)?;
                let currency = result
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
                    amount: cents_to_major(minor)?,
                    model: result
                        .get("model")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
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

pub(super) async fn anthropic_default_workspace(
    workspace_id: &str,
    identity: AccountIdentity,
    key: &str,
    http: &dyn Transport,
) -> Result<bool, AccountFailure> {
    let mut endpoint = Url::parse("https://api.anthropic.com/v1/organizations/workspaces/")
        .map_err(|_| AccountFailure::InvalidResponse)?;
    endpoint
        .path_segments_mut()
        .map_err(|_| AccountFailure::InvalidResponse)?
        .pop_if_empty()
        .push(workspace_id);
    let body = get_json(http, endpoint.into(), anthropic_headers(identity, key)).await?;
    if body.get("id").and_then(Value::as_str) != Some(workspace_id) {
        return Err(AccountFailure::InvalidResponse);
    }
    Ok(body.get("name").and_then(Value::as_str) == Some("Default"))
}

pub(super) fn cents_to_major(value: &str) -> Result<String, AccountFailure> {
    let (negative, unsigned) = value
        .strip_prefix('-')
        .map_or((false, value), |rest| (true, rest));
    let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty()
        || !whole.bytes().all(|b| b.is_ascii_digit())
        || (!fraction.is_empty() && !fraction.bytes().all(|b| b.is_ascii_digit()))
        || (unsigned.contains('.') && fraction.is_empty())
    {
        return Err(AccountFailure::InvalidResponse);
    }
    let digits = format!("{whole}{fraction}");
    let scale = fraction.len() + 2;
    let padded = format!(
        "{}{}",
        "0".repeat(scale.saturating_add(1).saturating_sub(digits.len())),
        digits
    );
    let split = padded.len() - scale;
    let whole = padded[..split].trim_start_matches('0');
    let whole = if whole.is_empty() { "0" } else { whole };
    Ok(format!(
        "{}{}.{}",
        if negative { "-" } else { "" },
        whole,
        &padded[split..]
    ))
}
