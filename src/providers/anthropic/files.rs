//! Provider-owned file workflow and capability policy.
use crate::files::*;

pub(crate) fn capabilities() -> FileCapabilities {
    FileCapabilities {
        upload: true,
        retrieve_metadata: true,
        list: true,
        delete: true,
        download: DownloadSupport::GeneratedFilesOnly,
        extract_text: false,
        model_input: ModelFileReference::Unsupported,
        max_upload_bytes: Some(500 * 1024 * 1024),
        retention: None,
    }
}

impl FileService<'_> {
    /// Retrieve metadata for up to 100 known Anthropic Files IDs in one
    /// request. Pass references produced in this exact endpoint/account scope;
    /// IDs that are missing or inaccessible are omitted by the provider and
    /// remain absent from the returned page.
    pub async fn list_by_ids(
        &self,
        files: &[ProviderFileRef],
    ) -> Result<ProviderFilePage, LlmError> {
        let service = self.scoped_operation();
        let adapter = service
            .adapter()
            .ok_or_else(|| unsupported("file metadata lookup by IDs"))?;
        if adapter != Adapter::Anthropic {
            return Err(unsupported("file metadata lookup by IDs"));
        }
        if service
            .account_scope
            .is_none_or(|scope| scope.trim().is_empty())
        {
            return Err(LlmError::InvalidRequest {
                message: "file metadata lookup by IDs requires a nonempty account scope".into(),
            });
        }
        if files.len() > 100 {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic file metadata lookup accepts at most 100 IDs".into(),
            });
        }
        let mut requested = std::collections::BTreeSet::new();
        for file in files {
            service.check_ref(file)?;
            if !requested.insert(file.file_id.as_str()) {
                return Err(LlmError::InvalidRequest {
                    message: "file metadata lookup IDs must be distinct".into(),
                });
            }
        }
        if files.is_empty() {
            return Ok(ProviderFilePage {
                files: Vec::new(),
                next_cursor: None,
            });
        }

        let query = files
            .iter()
            .map(|file| format!("ids%5B%5D={}", query_value(&file.file_id)))
            .collect::<Vec<_>>()
            .join("&");
        let url = format!("{}?{query}", files_url(service.profile));
        let req = service.request("GET", url, Bytes::new(), None).await?;
        let response = service.executor().execute(req).await?;
        let value = adapter_json_success(adapter, &response, "file metadata lookup")?;
        let rows = value
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| provider_shape("Anthropic file ID lookup has no data array"))?;
        if value
            .get("next_page")
            .is_some_and(|next_page| !next_page.is_null())
        {
            return Err(provider_shape(
                "Anthropic file ID lookup unexpectedly returned a pagination cursor",
            ));
        }
        let mut returned = std::collections::BTreeSet::new();
        let mut metadata = Vec::with_capacity(rows.len());
        for row in rows {
            let metadata_row = service.decode_metadata(row)?;
            if !requested.contains(metadata_row.file.file_id.as_str())
                || !returned.insert(metadata_row.file.file_id.clone())
            {
                return Err(provider_shape(
                    "Anthropic file ID lookup returned an unrequested or duplicate ID",
                ));
            }
            metadata.push(metadata_row);
        }
        Ok(ProviderFilePage {
            files: metadata,
            next_cursor: None,
        })
    }
}

pub(crate) const MAX_AUTOMATIC_ANTHROPIC_FILE_ID_JSON_BYTES: usize = 512;

pub(crate) fn purpose_capabilities(purpose: FilePurpose, _media_type: &str) -> FileCapabilities {
    if purpose == FilePurpose::ModelInput {
        capabilities()
    } else {
        FileCapabilities::unsupported()
    }
}

pub(crate) fn model_reference(
    profile: &ProviderProfile,
    model_profile: &ModelProfile,
    _model: &str,
    media_type: &str,
) -> ModelFileReference {
    if profile.protocol == ProtocolFamily::AnthropicMessages
        && ((matches!(
            media_type,
            "image/jpeg" | "image/png" | "image/gif" | "image/webp"
        ) && model_declares_image_input(model_profile))
            || (matches!(media_type, "application/pdf" | "text/plain")
                && model_declares_file_input(model_profile)))
    {
        ModelFileReference::FileId
    } else {
        ModelFileReference::Unsupported
    }
}

pub(crate) fn files_url(profile: &ProviderProfile) -> String {
    let base = profile.base_url.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{base}/files")
    } else {
        format!("{base}/v1/files")
    }
}

pub(crate) fn adapter(_profile: &ProviderProfile, host: &str) -> Option<Adapter> {
    (host == "api.anthropic.com").then_some(Adapter::Anthropic)
}
