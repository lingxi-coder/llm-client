//! Shared file REST shapes; vendor exceptions dispatch to their adapter.
use super::super::*;
pub(crate) fn files_url(profile: &ProviderProfile, adapter: Adapter) -> String {
    match adapter {
        Adapter::Anthropic => crate::providers::anthropic::files::files_url(profile),
        Adapter::Gemini => crate::providers::google::files::files_url(profile),
        Adapter::MiniMax => crate::providers::minimax::files::minimax_files_root(profile),
        _ => format!("{}/files", profile.base_url.trim_end_matches('/')),
    }
}

pub(crate) fn content_url(profile: &ProviderProfile, adapter: Adapter, file_id: &str) -> String {
    match adapter {
        Adapter::Gemini => crate::providers::google::files::content_url(profile, file_id),
        Adapter::MiniMax => format!("{}/retrieve_content", files_url(profile, adapter)),
        _ => format!(
            "{}/{}/content",
            files_url(profile, adapter),
            path_segment(file_id)
        ),
    }
}

pub(crate) fn file_url(profile: &ProviderProfile, adapter: Adapter, file_id: &str) -> String {
    match adapter {
        Adapter::Gemini => crate::providers::google::files::file_url(profile, file_id),
        Adapter::MiniMax => format!("{}/retrieve", files_url(profile, adapter)),
        _ => format!("{}/{}", files_url(profile, adapter), path_segment(file_id)),
    }
}

pub(crate) fn list_url(
    profile: &ProviderProfile,
    adapter: Adapter,
    purpose: Option<FilePurpose>,
    cursor: Option<&str>,
) -> (String, CursorField) {
    let base = files_url(profile, adapter);
    if adapter == Adapter::MiniMax {
        let Some(purpose) = purpose.and_then(minimax_list_purpose) else {
            return (format!("{base}/list"), CursorField::MiniMax);
        };
        let params = vec![("purpose".to_owned(), purpose.to_owned())];
        return (
            format!(
                "{base}/list?{}",
                params
                    .into_iter()
                    .map(|(key, value)| format!("{key}={}", query_value(&value)))
                    .collect::<Vec<_>>()
                    .join("&")
            ),
            CursorField::MiniMax,
        );
    }
    let (field, limit_field, limit) = match adapter {
        Adapter::OpenAi => ("after", "limit", "100"),
        Adapter::Qwen => ("after", "limit", "100"),
        Adapter::Anthropic => ("page", "limit", "1000"),
        Adapter::Gemini => ("pageToken", "pageSize", "100"),
        Adapter::Xai => ("pagination_token", "limit", "100"),
        Adapter::OpenRouter => ("cursor", "limit", "100"),
        _ => return (base, CursorField::OpenAi),
    };
    let cursor_field = match adapter {
        Adapter::OpenAi => CursorField::OpenAi,
        Adapter::Anthropic => CursorField::Anthropic,
        Adapter::Gemini => CursorField::Gemini,
        Adapter::Xai => CursorField::Xai,
        Adapter::OpenRouter => CursorField::OpenRouter,
        Adapter::Qwen => CursorField::Qwen,
        _ => CursorField::OpenAi,
    };
    let mut params = vec![(limit_field.to_owned(), limit.to_owned())];
    if let Some(cursor) = cursor {
        params.push((field.to_owned(), cursor.to_owned()));
    }
    let query = params
        .into_iter()
        .map(|(key, value)| format!("{key}={}", query_value(&value)))
        .collect::<Vec<_>>()
        .join("&");
    let mut query = query;
    if matches!(adapter, Adapter::OpenAi | Adapter::Qwen) {
        if let Some(purpose) = purpose.and_then(|purpose| purpose_name(adapter, purpose)) {
            query.push_str("&purpose=");
            query.push_str(&query_value(purpose));
        }
    }
    (format!("{base}?{query}"), cursor_field)
}

pub(crate) fn adapter_json_success(
    adapter: Adapter,
    response: &HttpResponse,
    operation: &str,
) -> Result<Value, LlmError> {
    let value = json_success(response, operation)?;
    if adapter == Adapter::MiniMax {
        check_minimax_base_response(&value, operation)?;
    }
    Ok(value)
}

pub(crate) fn adapter_status_result(
    adapter: Adapter,
    response: &HttpResponse,
    operation: &str,
) -> Result<(), LlmError> {
    status_result(response, operation)?;
    if adapter == Adapter::MiniMax {
        let value: Value = serde_json::from_slice(&response.body).map_err(|error| {
            provider_shape(&format!("{operation} response was not JSON: {error}"))
        })?;
        check_minimax_base_response(&value, operation)?;
    }
    Ok(())
}

pub(crate) fn decode_metadata(
    profile: &ProviderProfile,
    account_scope: Option<&str>,
    value: &Value,
) -> Result<ProviderFileMetadata, LlmError> {
    let adapter = adapter(profile).ok_or_else(|| unsupported("file metadata decoding"))?;
    decode_metadata_for_adapter(profile, account_scope, value, adapter, &profile.base_url)
}

pub(crate) fn decode_metadata_for_adapter(
    profile: &ProviderProfile,
    account_scope: Option<&str>,
    value: &Value,
    adapter: Adapter,
    endpoint_identity: &str,
) -> Result<ProviderFileMetadata, LlmError> {
    let value = value.get("file").unwrap_or(value);
    let file_id = value
        .get("id")
        .or_else(|| value.get("file_id"))
        .or_else(|| value.get("name"))
        .and_then(json_optional_scalar_string)
        .ok_or_else(|| provider_shape("file metadata has no id or name"))?
        .to_owned();
    let protocol = profile.protocol;
    let purpose = value
        .get("purpose")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .or_else(|| value.get("state").and_then(Value::as_str))
        .map(str::to_owned);
    let uri = value
        .get("uri")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| match adapter {
            Adapter::Qwen => {
                crate::providers::qwen::files::metadata_uri(&file_id, purpose.as_deref())
            }
            Adapter::MiniMax => {
                crate::providers::minimax::files::metadata_uri(&file_id, purpose.as_deref())
            }
            _ => None,
        });
    let file = ProviderFileRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(endpoint_identity),
        account_scope: account_scope.map(str::to_owned),
        protocol,
        file_id,
        uri,
        filename: value
            .get("filename")
            .or_else(|| value.get("display_name"))
            .or_else(|| value.get("displayName"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        media_type: value
            .get("mime_type")
            .or_else(|| value.get("mimeType"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        size_bytes: value
            .get("bytes")
            .or_else(|| value.get("size_bytes"))
            .or_else(|| value.get("sizeBytes"))
            .and_then(json_u64),
        expires_at: value
            .get("expires_at")
            .or_else(|| value.get("expirationTime"))
            .and_then(json_optional_scalar_string),
        processing_status: status.clone(),
        downloadable: match adapter {
            Adapter::Gemini => crate::providers::google::files::metadata_downloadable(value),
            Adapter::MiniMax => crate::providers::minimax::files::metadata_downloadable(value),
            _ => value.get("downloadable").and_then(Value::as_bool),
        },
        purpose,
    };
    Ok(ProviderFileMetadata {
        file,
        created_at: value
            .get("created_at")
            .or_else(|| value.get("createTime"))
            .and_then(json_optional_scalar_string),
        status,
    })
}
