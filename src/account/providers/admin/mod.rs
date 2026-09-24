//! Official read-only organization and management account APIs.
//!
//! These endpoints are deliberately fixed to vendor hosts. A profile's model
//! `base_url` is not an authority for account data or an admin credential.

use super::{
    field_from_result, get_json, parse_iso_utc, AccountBalance, AccountCostBucket,
    AccountCostUsage, AccountFailure, AccountIdentity, AccountMetric, AccountQuery, AccountScope,
    AccountScopeKind, AccountTokenBucket, AccountTokenUsage, AccountUsageSource,
};
use crate::protocol::ProtocolFamily;
use crate::transport::{HttpRequest, Transport};
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use url::Url;

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
        context: &super::AccountFetchContext<'_>,
        report: &mut super::AccountReport,
    ) -> Result<(), AccountFailure> {
        let profile = context.profile;
        let query = context.query;
        let range = context.range;
        let _now = context.fetched_at_unix;
        let http = context.http();
        let snapshot = report;
        snapshot.unsupported();
        match self {
            Self::Xai => {
                snapshot.balance = None;
                snapshot.cost_usage = None;
            }
            Self::OpenAi | Self::Anthropic => {
                snapshot.token_usage = None;
                snapshot.cost_usage = None;
            }
            Self::Gemini => {
                if matches!(
                    profile.protocol,
                    ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini
                ) {
                    snapshot.token_usage = None;
                }
            }
        }
        let Some(key) = &query.management_credential else {
            if matches!(self, Self::Xai) {
                snapshot.balance = Some(AccountMetric::CredentialRequired);
                snapshot.cost_usage = Some(AccountMetric::CredentialRequired);
            } else {
                snapshot.token_usage = Some(AccountMetric::CredentialRequired);
                if matches!(self, Self::OpenAi | Self::Anthropic) {
                    snapshot.cost_usage = Some(AccountMetric::CredentialRequired);
                }
            }
            return Ok(());
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
                snapshot.token_usage = Some(field_from_result(
                    openai_usage(query, range, key.expose_secret(), http).await,
                    scope.clone(),
                    "https://api.openai.com/v1/organization/usage/completions",
                ));
                snapshot.cost_usage = Some(field_from_result(
                    openai_costs(query, range, key.expose_secret(), http).await,
                    scope,
                    "https://api.openai.com/v1/organization/costs",
                ));
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
                snapshot.token_usage = Some(field_from_result(
                    anthropic_usage(query, range, key.expose_secret(), http).await,
                    scope,
                    "https://api.anthropic.com/v1/organizations/usage_report/messages",
                ));
                let cost_scope = query.selector.workspace_id.as_ref().map_or_else(
                    || {
                        AccountScope::new(
                            AccountScopeKind::Organization,
                            query.selector.organization_id.clone(),
                        )
                    },
                    |id| AccountScope::new(AccountScopeKind::Workspace, Some(id.clone())),
                );
                snapshot.cost_usage = Some(field_from_result(
                    anthropic_costs(query, range, key.expose_secret(), http).await,
                    cost_scope,
                    "https://api.anthropic.com/v1/organizations/cost_report",
                ));
            }
            Self::Xai => {
                if let Some(team_id) = query.selector.team_id.as_deref().filter(|s| !s.is_empty()) {
                    let scope = AccountScope::new(AccountScopeKind::Team, Some(team_id.into()));
                    snapshot.balance = Some(field_from_result(
                        xai_balance(team_id, key.expose_secret(), http).await,
                        scope.clone(),
                        "https://management-api.x.ai/v1/billing/teams/{team_id}/prepaid/balance",
                    ));
                    snapshot.cost_usage = Some(field_from_result(
                        xai_costs(team_id, range, key.expose_secret(), http).await,
                        scope,
                        "https://management-api.x.ai/v1/billing/teams/{team_id}/usage",
                    ));
                } else {
                    snapshot.balance = Some(AccountMetric::NotReported);
                    snapshot.cost_usage = Some(AccountMetric::NotReported);
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
                        snapshot.token_usage = Some(field_from_result(
                            gemini_usage(project_id, range, key.expose_secret(), metric_type, http)
                                .await,
                            scope,
                            "https://monitoring.googleapis.com/v3/projects/{project_id}/timeSeries",
                        ));
                    }
                } else {
                    snapshot.token_usage = Some(AccountMetric::NotReported);
                }
            }
        }
        Ok(())
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

fn valid_currency(value: &str) -> bool {
    value.len() == 3 && value.bytes().all(|b| b.is_ascii_alphabetic())
}

// Anthropic documents decimal cents; shift by two places without a float.

async fn post_json(
    http: &dyn Transport,
    url: String,
    key: &str,
    value: &Value,
) -> Result<Value, AccountFailure> {
    let body = serde_json::to_vec(value).map_err(|_| AccountFailure::InvalidResponse)?;
    let response = crate::transport::HttpExecutor::new(http)
        .execute_bounded(
            HttpRequest {
                method: "POST".into(),
                url,
                headers: vec![
                    ("authorization".into(), format!("Bearer {key}")),
                    ("content-type".into(), "application/json".into()),
                ],
                body: Bytes::from(body),
                timeout: None,
            },
            super::MAX_ACCOUNT_BODY,
        )
        .await
        .map_err(super::execution::map_transport_error)?;
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

// The report APIs accept UTC RFC 3339 timestamps.

mod openai;
use openai::*;

mod anthropic;
use anthropic::*;

mod xai;
use xai::*;

mod gemini;
use gemini::*;
