use super::*;
pub(super) async fn gemini_usage(
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

pub(super) fn format_rfc3339(seconds: u64) -> Result<String, AccountFailure> {
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

pub(super) fn parse_rfc3339(value: &str) -> Result<u64, AccountFailure> {
    parse_iso_utc(value).ok_or(AccountFailure::InvalidResponse)
}

pub(super) fn civil_from_days(days: i64) -> Option<(i64, u32, u32)> {
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
