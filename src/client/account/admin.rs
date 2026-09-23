//! Official read-only organization and management account APIs.
//!
//! These endpoints are deliberately fixed to vendor hosts. A profile's model
//! `base_url` is not an authority for account data or an admin credential.

use super::{
    field_from_result, get_json, parse_iso_utc, AccountBalance, AccountCostBucket,
    AccountCostUsage, AccountFailure, AccountIdentity, AccountMetric, AccountQuery, AccountScope,
    AccountScopeKind, AccountSnapshot, AccountTokenBucket, AccountTokenUsage, AccountUsageSource,
};
use crate::transport::{HttpRequest, Transport};
use async_trait::async_trait;
use bytes::Bytes;
use lingxi_agent_api::protocol::{ProtocolFamily, ProviderProfile};
use reqwest::Url;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

const MAX_PAGES: usize = 100;

pub(super) fn register(
    sources: &mut BTreeMap<(String, AccountIdentity), Arc<dyn AccountUsageSource>>,
) {
    for (provider, source) in [
        ("openai", AdminSource::OpenAi),
        ("anthropic", AdminSource::Anthropic),
        ("xai", AdminSource::Xai),
        ("google", AdminSource::Gemini),
    ] {
        sources.insert((provider.into(), AccountIdentity::ApiKey), Arc::new(source));
    }
    sources.insert(
        ("anthropic".into(), AccountIdentity::AuthUser),
        Arc::new(AdminSource::Anthropic),
    );
    sources.insert(
        ("google".into(), AccountIdentity::AuthUser),
        Arc::new(AdminSource::Gemini),
    );
}

#[derive(Clone, Copy)]
enum AdminSource {
    OpenAi,
    Anthropic,
    Xai,
    Gemini,
}

#[async_trait]
impl AccountUsageSource for AdminSource {
    async fn fetch(
        &self,
        profile: &ProviderProfile,
        query: &AccountQuery,
        range: (u64, u64),
        now: u64,
        http: &dyn Transport,
    ) -> AccountSnapshot {
        let mut snapshot = AccountSnapshot::unsupported(profile, query.identity, now);
        let Some(key) = &query.management_credential else {
            if matches!(self, Self::Xai) {
                snapshot.balance = AccountMetric::CredentialRequired;
                snapshot.cost_usage = AccountMetric::CredentialRequired;
            } else {
                snapshot.token_usage = AccountMetric::CredentialRequired;
                if matches!(self, Self::OpenAi | Self::Anthropic) {
                    snapshot.cost_usage = AccountMetric::CredentialRequired;
                }
            }
            return snapshot;
        };
        match self {
            Self::OpenAi => {
                let scope = query.selector.api_key_id.as_ref().map_or_else(
                    || {
                        query.selector.project_id.as_ref().map_or_else(
                            || {
                                AccountScope::new(
                                    AccountScopeKind::Organization,
                                    query.selector.organization_id.clone(),
                                )
                            },
                            |id| AccountScope::new(AccountScopeKind::Project, Some(id.clone())),
                        )
                    },
                    |id| AccountScope::new(AccountScopeKind::ApiKey, Some(id.clone())),
                );
                snapshot.token_usage = field_from_result(
                    openai_usage(query, range, key.expose_secret(), http).await,
                    scope.clone(),
                    "https://api.openai.com/v1/organization/usage/completions",
                );
                snapshot.cost_usage = field_from_result(
                    openai_costs(query, range, key.expose_secret(), http).await,
                    scope,
                    "https://api.openai.com/v1/organization/costs",
                );
            }
            Self::Anthropic => {
                let scope = query.selector.api_key_id.as_ref().map_or_else(
                    || {
                        query.selector.workspace_id.as_ref().map_or_else(
                            || {
                                AccountScope::new(
                                    AccountScopeKind::Organization,
                                    query.selector.organization_id.clone(),
                                )
                            },
                            |id| AccountScope::new(AccountScopeKind::Workspace, Some(id.clone())),
                        )
                    },
                    |id| AccountScope::new(AccountScopeKind::ApiKey, Some(id.clone())),
                );
                snapshot.token_usage = field_from_result(
                    anthropic_usage(query, range, key.expose_secret(), http).await,
                    scope,
                    "https://api.anthropic.com/v1/organizations/usage_report/messages",
                );
                let cost_scope = query.selector.workspace_id.as_ref().map_or_else(
                    || {
                        AccountScope::new(
                            AccountScopeKind::Organization,
                            query.selector.organization_id.clone(),
                        )
                    },
                    |id| AccountScope::new(AccountScopeKind::Workspace, Some(id.clone())),
                );
                snapshot.cost_usage = field_from_result(
                    anthropic_costs(query, range, key.expose_secret(), http).await,
                    cost_scope,
                    "https://api.anthropic.com/v1/organizations/cost_report",
                );
            }
            Self::Xai => {
                if let Some(team_id) = query.selector.team_id.as_deref().filter(|s| !s.is_empty()) {
                    let scope = AccountScope::new(AccountScopeKind::Team, Some(team_id.into()));
                    snapshot.balance = field_from_result(
                        xai_balance(team_id, key.expose_secret(), http).await,
                        scope.clone(),
                        "https://management-api.x.ai/v1/billing/teams/{team_id}/prepaid/balance",
                    );
                    snapshot.cost_usage = field_from_result(
                        xai_costs(team_id, range, key.expose_secret(), http).await,
                        scope,
                        "https://management-api.x.ai/v1/billing/teams/{team_id}/usage",
                    );
                } else {
                    snapshot.balance = AccountMetric::NotReported;
                    snapshot.cost_usage = AccountMetric::NotReported;
                }
            }
            Self::Gemini => {
                if let Some(project_id) = query
                    .selector
                    .project_id
                    .as_deref()
                    .filter(|s| !s.is_empty())
                {
                    let scope =
                        AccountScope::new(AccountScopeKind::Project, Some(project_id.into()));
                    let metric_type = match profile.protocol {
                        ProtocolFamily::GeminiGenerateContent => Some(
                            "generativelanguage.googleapis.com/generate_content_usage_output_token_count",
                        ),
                        ProtocolFamily::VertexGemini => {
                            Some("aiplatform.googleapis.com/publisher/online_serving/token_count")
                        }
                        _ => None,
                    };
                    if let Some(metric_type) = metric_type {
                        snapshot.token_usage = field_from_result(
                            gemini_usage(project_id, range, key.expose_secret(), metric_type, http)
                                .await,
                            scope,
                            "https://monitoring.googleapis.com/v3/projects/{project_id}/timeSeries",
                        );
                    }
                } else {
                    snapshot.token_usage = AccountMetric::NotReported;
                }
            }
        }
        snapshot
    }
}

fn url(base: &str, params: &[(&str, String)]) -> Result<String, AccountFailure> {
    let mut url = Url::parse(base).map_err(|_| AccountFailure::InvalidResponse)?;
    url.query_pairs_mut()
        .extend_pairs(params.iter().map(|(key, value)| (*key, value.as_str())));
    Ok(url.to_string())
}

fn next_page<'a>(
    body: &'a Value,
    has_more_name: &str,
    page_name: &str,
) -> Result<Option<&'a str>, AccountFailure> {
    let more = body
        .get(has_more_name)
        .and_then(Value::as_bool)
        .ok_or(AccountFailure::InvalidResponse)?;
    if more {
        body.get(page_name)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(Some)
            .ok_or(AccountFailure::InvalidResponse)
    } else {
        Ok(None)
    }
}

fn number(value: &Value, key: &str) -> Result<u64, AccountFailure> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or(AccountFailure::InvalidResponse)
}

fn optional_number(value: &Value, key: &str) -> Result<Option<u64>, AccountFailure> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or(AccountFailure::InvalidResponse),
    }
}

fn add(current: &mut Option<u64>, value: u64) -> Result<(), AccountFailure> {
    *current = Some(
        current
            .unwrap_or(0)
            .checked_add(value)
            .ok_or(AccountFailure::InvalidResponse)?,
    );
    Ok(())
}

fn anthropic_headers(identity: AccountIdentity, key: &str) -> Vec<(String, String)> {
    let credential = match identity {
        AccountIdentity::ApiKey => ("x-api-key".into(), key.into()),
        AccountIdentity::AuthUser => ("authorization".into(), format!("Bearer {key}")),
    };
    vec![
        credential,
        ("anthropic-version".into(), "2023-06-01".into()),
    ]
}

async fn openai_usage(
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

async fn anthropic_usage(
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

async fn openai_costs(
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

async fn anthropic_costs(
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

async fn anthropic_default_workspace(
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

fn valid_currency(value: &str) -> bool {
    value.len() == 3 && value.bytes().all(|b| b.is_ascii_alphabetic())
}

// Anthropic documents decimal cents; shift by two places without a float.
fn cents_to_major(value: &str) -> Result<String, AccountFailure> {
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

fn xai_endpoint(team_id: &str, tail: &[&str]) -> Result<String, AccountFailure> {
    let mut endpoint = Url::parse("https://management-api.x.ai/v1/billing/teams/")
        .map_err(|_| AccountFailure::InvalidResponse)?;
    let mut path = endpoint
        .path_segments_mut()
        .map_err(|_| AccountFailure::InvalidResponse)?;
    path.pop_if_empty().push(team_id);
    for segment in tail {
        path.push(segment);
    }
    drop(path);
    Ok(endpoint.into())
}

async fn post_json(
    http: &dyn Transport,
    url: String,
    key: &str,
    value: &Value,
) -> Result<Value, AccountFailure> {
    let body = serde_json::to_vec(value).map_err(|_| AccountFailure::InvalidResponse)?;
    let response = http
        .execute_no_follow_bounded(
            HttpRequest {
                method: "POST".into(),
                url,
                headers: vec![
                    ("authorization".into(), format!("Bearer {key}")),
                    ("content-type".into(), "application/json".into()),
                ],
                body: Bytes::from(body),
                timeout: Some(super::ACCOUNT_REQUEST_TIMEOUT),
            },
            super::MAX_ACCOUNT_BODY,
        )
        .await
        .map_err(|_| AccountFailure::Transport)?;
    match response.status {
        200..=299 => {}
        401 => return Err(AccountFailure::Unauthorized),
        403 => return Err(AccountFailure::PermissionDenied),
        429 => return Err(AccountFailure::RateLimited),
        _ => return Err(AccountFailure::ProviderError),
    }
    if response.body.len() > super::MAX_ACCOUNT_BODY {
        return Err(AccountFailure::InvalidResponse);
    }
    serde_json::from_slice(&response.body).map_err(|_| AccountFailure::InvalidResponse)
}

async fn xai_costs(
    team_id: &str,
    range: (u64, u64),
    key: &str,
    http: &dyn Transport,
) -> Result<AccountCostUsage, AccountFailure> {
    // xAI's timeRange end is inclusive, whereas AccountQuery.until_unix is exclusive.
    let start = format_rfc3339(range.0)?
        .replace('T', " ")
        .trim_end_matches('Z')
        .to_owned();
    let end = format_rfc3339(range.1 - 1)?
        .replace('T', " ")
        .trim_end_matches('Z')
        .to_owned();
    let request = serde_json::json!({"analyticsRequest": {
        "timeRange": {"startTime": start, "endTime": end, "timezone": "Etc/GMT"},
        "timeUnit": "TIME_UNIT_DAY",
        "values": [{"name": "usd", "aggregation": "AGGREGATION_SUM"}],
        "groupBy": ["description"], "filters": []
    }});
    let body = post_json(http, xai_endpoint(team_id, &["usage"])?, key, &request).await?;
    if body.get("limitReached").and_then(Value::as_bool) == Some(true) {
        return Err(AccountFailure::ProviderError);
    }
    let series = match body.get("timeSeries") {
        None => return Ok(AccountCostUsage::default()),
        Some(Value::Array(series)) => series,
        Some(_) => return Err(AccountFailure::InvalidResponse),
    };
    let mut buckets = Vec::new();
    for item in series {
        let points = match item.get("dataPoints") {
            None => continue,
            Some(Value::Array(points)) => points,
            Some(_) => return Err(AccountFailure::InvalidResponse),
        };
        for point in points {
            let start = parse_rfc3339(
                point
                    .get("timestamp")
                    .and_then(Value::as_str)
                    .ok_or(AccountFailure::InvalidResponse)?,
            )?;
            let end = start
                .checked_add(86_400)
                .ok_or(AccountFailure::InvalidResponse)?;
            let amount = point
                .pointer("/values/0")
                .and_then(Value::as_number)
                .ok_or(AccountFailure::InvalidResponse)?;
            buckets.push(AccountCostBucket {
                start_unix: start,
                end_unix: end,
                currency: "USD".into(),
                amount: amount.to_string(),
                model: None,
            });
        }
    }
    Ok(AccountCostUsage { buckets })
}

async fn xai_balance(
    team_id: &str,
    key: &str,
    http: &dyn Transport,
) -> Result<Vec<AccountBalance>, AccountFailure> {
    let body = get_json(
        http,
        xai_endpoint(team_id, &["prepaid", "balance"])?,
        vec![("authorization".into(), format!("Bearer {key}"))],
    )
    .await?;
    let cents = body
        .pointer("/total/val")
        .and_then(Value::as_str)
        .ok_or(AccountFailure::InvalidResponse)?
        .parse::<i128>()
        .map_err(|_| AccountFailure::InvalidResponse)?;
    // xAI's prepaid ledger records purchased credit as a negative liability.
    // Negating that signed ledger total yields the remaining usable credit.
    let available = cents.checked_neg().ok_or(AccountFailure::InvalidResponse)?;
    let abs = available.unsigned_abs();
    let remaining = format!(
        "{}{}.{:02}",
        if available < 0 { "-" } else { "" },
        abs / 100,
        abs % 100
    );
    Ok(vec![AccountBalance {
        unit: "USD".into(),
        remaining,
        total: None,
        is_available: None,
    }])
}

async fn gemini_usage(
    project_id: &str,
    range: (u64, u64),
    key: &str,
    metric_type: &str,
    http: &dyn Transport,
) -> Result<AccountTokenUsage, AccountFailure> {
    let mut endpoint = Url::parse("https://monitoring.googleapis.com/v3/projects/")
        .map_err(|_| AccountFailure::InvalidResponse)?;
    endpoint
        .path_segments_mut()
        .map_err(|_| AccountFailure::InvalidResponse)?
        .pop_if_empty()
        .push(project_id)
        .push("timeSeries");
    let base = endpoint.to_string();
    // A Monitoring metrics scope can include other projects. Constrain the
    // monitored resource as well as the project in the request path.
    let resource_type = if metric_type.starts_with("aiplatform.googleapis.com/") {
        "aiplatform.googleapis.com/PublisherModel"
    } else {
        "generativelanguage.googleapis.com/Location"
    };
    let project_literal =
        serde_json::to_string(project_id).map_err(|_| AccountFailure::InvalidResponse)?;
    let params = vec![
        (
            "filter",
            format!(
                "metric.type = \"{metric_type}\" AND resource.type = \"{resource_type}\" AND resource.labels.resource_container = {project_literal}"
            ),
        ),
        ("interval.startTime", format_rfc3339(range.0)?),
        ("interval.endTime", format_rfc3339(range.1)?),
        ("view", "FULL".into()),
        ("pageSize", "1000".into()),
    ];
    let headers = vec![("authorization".into(), format!("Bearer {key}"))];
    let mut buckets: BTreeMap<(u64, u64, Option<String>), AccountTokenBucket> = BTreeMap::new();
    let mut page = None::<String>;
    for _ in 0..MAX_PAGES {
        let mut request_params = params.clone();
        if let Some(cursor) = &page {
            request_params.push(("pageToken", cursor.clone()));
        }
        let body = get_json(http, url(&base, &request_params)?, headers.clone()).await?;
        if body
            .get("executionErrors")
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty())
            || body
                .get("unreachable")
                .and_then(Value::as_array)
                .is_some_and(|a| !a.is_empty())
        {
            return Err(AccountFailure::ProviderError);
        }
        let empty = Vec::new();
        let series = match body.get("timeSeries") {
            Some(value) => value.as_array().ok_or(AccountFailure::InvalidResponse)?,
            None => &empty,
        };
        for item in series {
            let metric = item.get("metric").ok_or(AccountFailure::InvalidResponse)?;
            if metric.get("type").and_then(Value::as_str) != Some(metric_type) {
                return Err(AccountFailure::InvalidResponse);
            }
            let vertex = metric_type.starts_with("aiplatform.googleapis.com/");
            let direction = if vertex {
                metric
                    .pointer("/labels/type")
                    .and_then(Value::as_str)
                    .ok_or(AccountFailure::InvalidResponse)?
            } else {
                "output"
            };
            let model = if vertex {
                item.pointer("/resource/labels/model_user_id")
                    .and_then(Value::as_str)
            } else {
                metric.pointer("/labels/model").and_then(Value::as_str)
            }
            .map(str::to_owned);
            let points = item
                .get("points")
                .and_then(Value::as_array)
                .ok_or(AccountFailure::InvalidResponse)?;
            for point in points {
                let interval = point
                    .get("interval")
                    .ok_or(AccountFailure::InvalidResponse)?;
                let start = parse_rfc3339(
                    interval
                        .get("startTime")
                        .and_then(Value::as_str)
                        .ok_or(AccountFailure::InvalidResponse)?,
                )?;
                let end = parse_rfc3339(
                    interval
                        .get("endTime")
                        .and_then(Value::as_str)
                        .ok_or(AccountFailure::InvalidResponse)?,
                )?;
                if start >= end {
                    return Err(AccountFailure::InvalidResponse);
                }
                let value = point
                    .pointer("/value/int64Value")
                    .and_then(Value::as_str)
                    .ok_or(AccountFailure::InvalidResponse)?
                    .parse::<u64>()
                    .map_err(|_| AccountFailure::InvalidResponse)?;
                let entry = buckets
                    .entry((start, end, model.clone()))
                    .or_insert_with(|| AccountTokenBucket {
                        start_unix: start,
                        end_unix: end,
                        model: model.clone(),
                        input_tokens: None,
                        output_tokens: None,
                        cached_input_tokens: None,
                        cache_write_tokens: None,
                        total_tokens: None,
                    });
                match direction {
                    "input" => add(&mut entry.input_tokens, value)?,
                    "output" => add(&mut entry.output_tokens, value)?,
                    _ => return Err(AccountFailure::InvalidResponse),
                }
            }
        }
        let cursor = body
            .get("nextPageToken")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        match cursor {
            Some(cursor) if page.as_deref() != Some(cursor) => page = Some(cursor.into()),
            Some(_) => return Err(AccountFailure::InvalidResponse),
            None => {
                for bucket in buckets.values_mut() {
                    bucket.total_tokens = match (bucket.input_tokens, bucket.output_tokens) {
                        (Some(input), Some(output)) => input.checked_add(output),
                        _ => None,
                    };
                }
                return Ok(AccountTokenUsage {
                    lifetime_tokens: None,
                    buckets: buckets.into_values().collect(),
                });
            }
        }
    }
    Err(AccountFailure::InvalidResponse)
}

// The report APIs accept UTC RFC 3339 timestamps.
fn format_rfc3339(seconds: u64) -> Result<String, AccountFailure> {
    let days = i64::try_from(seconds / 86_400).map_err(|_| AccountFailure::InvalidResponse)?;
    let (year, month, day) = civil_from_days(days).ok_or(AccountFailure::InvalidResponse)?;
    let day_seconds = seconds % 86_400;
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        day_seconds / 3_600,
        day_seconds % 3_600 / 60,
        day_seconds % 60
    ))
}

fn parse_rfc3339(value: &str) -> Result<u64, AccountFailure> {
    parse_iso_utc(value).ok_or(AccountFailure::InvalidResponse)
}

fn civil_from_days(days: i64) -> Option<(i64, u32, u32)> {
    let z = days.checked_add(719_468)?;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    Some((year, month as u32, day as u32))
}
