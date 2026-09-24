//! Shared file REST shapes; vendor exceptions dispatch to their adapter.
use super::super::*;
pub(crate) fn files_url(profile: &ProviderProfile, adapter: Adapter) -> String {
    let base = profile.base_url.trim_end_matches('/');
    match adapter {
        Adapter::Anthropic if base.ends_with("/v1") => format!("{base}/files"),
        Adapter::Anthropic => format!("{base}/v1/files"),
        Adapter::Gemini if base.ends_with("/v1beta") => format!("{base}/files"),
        Adapter::Gemini => format!("{base}/v1beta/files"),
        Adapter::MiniMax => minimax_files_root(profile),
        _ => format!("{base}/files"),
    }
}

pub(crate) fn content_url(profile: &ProviderProfile, adapter: Adapter, file_id: &str) -> String {
    let encoded = path_segment(file_id);
    match adapter {
        Adapter::OpenRouter | Adapter::Xai | Adapter::OpenAi => {
            format!("{}/{encoded}/content", files_url(profile, adapter))
        }
        Adapter::Anthropic => format!("{}/{encoded}/content", files_url(profile, adapter)),
        Adapter::Gemini => format!("{}:download?alt=media", file_url(profile, adapter, file_id)),
        Adapter::Moonshot => format!("{}/{encoded}/content", files_url(profile, adapter)),
        Adapter::MiniMax => format!("{}/retrieve_content", files_url(profile, adapter)),
        Adapter::Qwen => format!("{}/{encoded}/content", files_url(profile, adapter)),
        Adapter::Zai | Adapter::Zhipu => {
            format!("{}/{encoded}/content", files_url(profile, adapter))
        }
    }
}

pub(crate) fn file_url(profile: &ProviderProfile, adapter: Adapter, file_id: &str) -> String {
    if adapter == Adapter::MiniMax {
        let _ = file_id;
        return format!("{}/retrieve", files_url(profile, adapter));
    }
    if adapter == Adapter::Gemini {
        let resource_name = if file_id.starts_with("files/") {
            file_id.to_owned()
        } else {
            format!("files/{file_id}")
        };
        let base = profile.base_url.trim_end_matches('/');
        let base = if base.ends_with("/v1beta") {
            base.to_owned()
        } else {
            format!("{base}/v1beta")
        };
        return format!("{base}/{}", encoded_path(&resource_name));
    }
    format!("{}/{}", files_url(profile, adapter), path_segment(file_id))
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
    let value = value.get("file").unwrap_or(value);
    let adapter = adapter(profile).ok_or_else(|| unsupported("file metadata decoding"))?;
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
    let uri = value
        .get("uri")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| match (adapter, purpose.as_deref()) {
            (Adapter::Qwen, Some("file-extract")) => Some(format!("fileid://{file_id}")),
            (Adapter::MiniMax, Some("video_understanding")) => Some(format!("mm_file://{file_id}")),
            _ => None,
        });
    if adapter == Adapter::Gemini && uri.is_none() {
        // Some metadata/list responses omit URI, but uploaded model refs must
        // retain the URI from the upload response.
    }
    let file = ProviderFileRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&profile.base_url),
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
        // Gemini only returns content for generated files with a download URI.
        // `download` uses the canonical API endpoint, so credentials are never
        // forwarded to the metadata-provided URI.
        downloadable: if adapter == Adapter::Gemini {
            Some(
                value.get("source").and_then(Value::as_str) == Some("GENERATED")
                    && value
                        .get("downloadUri")
                        .and_then(nonempty_string)
                        .is_some_and(|uri| valid_download_uri(&uri)),
            )
        } else if adapter == Adapter::MiniMax {
            value
                .get("download_url")
                .or_else(|| value.get("downloadable"))
                .and_then(|v| v.as_bool().or_else(|| nonempty_string(v).map(|_| true)))
        } else {
            value.get("downloadable").and_then(Value::as_bool)
        },
        purpose,
    };
    Ok(ProviderFileMetadata {
        file,
        created_at: value
            .get("created_at")
            .or_else(|| value.get("createTime"))
            .and_then(json_optional_scalar_string),
        status: value
            .get("status")
            .or_else(|| value.get("state"))
            .and_then(Value::as_str)
            .map(str::to_owned),
    })
}
