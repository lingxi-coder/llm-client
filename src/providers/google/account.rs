//! Provider-owned organization account reporting.
use crate::account::support::*;
use crate::account::*;
use crate::protocol::ProtocolFamily;
use crate::transport::Transport;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use url::Url;

pub(crate) struct AccountSource;

pub(crate) fn register(
    sources: &mut BTreeMap<(String, AccountIdentity), Arc<dyn AccountUsageSource>>,
) {
    sources.insert(
        ("google".into(), AccountIdentity::ApiKey),
        Arc::new(AccountSource),
    );
    sources.insert(
        ("google".into(), AccountIdentity::AuthUser),
        Arc::new(AccountSource),
    );
}

#[async_trait]
impl AccountUsageSource for AccountSource {
    async fn fetch(
        &self,
        context: &AccountFetchContext<'_>,
        snapshot: &mut AccountReport,
    ) -> Result<(), AccountFailure> {
        let query = context.query;
        let range = context.range;
        let http = context.http();
        let profile = context.profile;
        snapshot.unsupported();
        if matches!(
            context.profile.protocol,
            ProtocolFamily::GeminiGenerateContent | ProtocolFamily::VertexGemini
        ) {
            snapshot.token_usage = None;
        }
        let Some(key) = &query.management_credential else {
            snapshot.token_usage = Some(AccountMetric::CredentialRequired);
            return Ok(());
        };

        if let Some(project_id) = query
            .selector
            .project_id
            .as_deref()
            .filter(|s| !s.is_empty())
        {
            let scope = AccountScope::new(AccountScopeKind::Project, Some(project_id.into()));
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
                    gemini_usage(project_id, range, key.expose_secret(), metric_type, http).await,
                    scope,
                    "https://monitoring.googleapis.com/v3/projects/{project_id}/timeSeries",
                ));
            }
        } else {
            snapshot.token_usage = Some(AccountMetric::NotReported);
        }

        Ok(())
    }
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
