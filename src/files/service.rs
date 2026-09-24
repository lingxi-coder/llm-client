//! Connection-bound provider file operations.
use super::*;

/// File API adapter bound to one profile and one credential scope.
///
/// Construct this from the same profile, authenticator and per-attempt
/// credential used for a model request. File IDs are therefore never silently
/// reused across failover connections.
pub struct FileService<'a> {
    pub(in crate::files) http: &'a dyn Transport,
    pub(in crate::files) profile: &'a ProviderProfile,
    pub(in crate::files) authenticator: Option<&'a dyn Authenticator>,
    pub(in crate::files) credential: Option<&'a Secret<String>>,
    pub(in crate::files) account_scope: Option<&'a str>,
    pub(in crate::files) qwen_rate_limiter: Option<Arc<QwenFileRateLimiter>>,
    pub(in crate::files) gemini_upload_timeout: Option<Duration>,
    pub(in crate::files) gemini_processing_timeout: Option<Duration>,
    pub(in crate::files) gemini_request_deadline: Option<Instant>,
}

impl<'a> FileService<'a> {
    #[must_use]
    pub fn new(
        http: &'a dyn Transport,
        profile: &'a ProviderProfile,
        authenticator: Option<&'a dyn Authenticator>,
        credential: Option<&'a Secret<String>>,
        account_scope: Option<&'a str>,
    ) -> Self {
        Self {
            http,
            profile,
            authenticator,
            credential,
            account_scope,
            qwen_rate_limiter: None,
            gemini_upload_timeout: None,
            gemini_processing_timeout: None,
            gemini_request_deadline: None,
        }
    }

    /// Bound the Gemini upload transfer. Video files default to and are capped
    /// at two hours; other files default to two minutes.
    #[must_use]
    pub fn with_gemini_upload_timeout(mut self, timeout: Duration) -> Self {
        self.gemini_upload_timeout =
            Some(timeout.clamp(GEMINI_FILE_POLL_INTERVAL, GEMINI_FILE_RETENTION));
        self
    }

    /// Bound the wait for Gemini Files to become ACTIVE after upload.
    /// Files default to ten minutes of active polling before returning a
    /// recoverable pending reference.
    /// The upload transfer has its own deadline.
    /// Values are clamped between one second and the Files API's 48-hour retention;
    /// the initial video operation remains capped at two hours.
    #[must_use]
    pub fn with_gemini_processing_timeout(mut self, timeout: Duration) -> Self {
        self.gemini_processing_timeout =
            Some(timeout.clamp(GEMINI_FILE_POLL_INTERVAL, GEMINI_FILE_RETENTION));
        self
    }

    pub(crate) fn with_gemini_request_deadline(mut self, deadline: Instant) -> Self {
        self.gemini_request_deadline = Some(deadline);
        self
    }

    pub(crate) fn with_qwen_rate_limiter(mut self, limiter: Arc<QwenFileRateLimiter>) -> Self {
        self.qwen_rate_limiter = Some(limiter);
        self
    }

    pub(in crate::files) async fn pace_qwen_upload(&self) {
        if let Some(limiter) = &self.qwen_rate_limiter {
            limiter.wait_upload().await;
        }
    }

    pub(in crate::files) async fn pace_qwen_metadata(&self) {
        if let Some(limiter) = &self.qwen_rate_limiter {
            limiter.wait_metadata().await;
        }
    }

    /// Return capabilities for this concrete profile, model and media type.
    /// The caller must still check model vision/file modality when present.
    #[must_use]
    pub fn capabilities(&self, model: &str, media_type: &str) -> FileCapabilities {
        capabilities(self.profile, model, media_type)
    }

    /// Return upload and model-input support for a specific purpose. This
    /// separates extraction or agent auxiliary files from chat references.
    #[must_use]
    pub fn capabilities_for_purpose(
        &self,
        model: &str,
        media_type: &str,
        purpose: FilePurpose,
    ) -> FileCapabilities {
        capabilities_for_purpose(self.profile, model, media_type, purpose)
    }

    /// Upload bytes using a provider-supported purpose.
    pub async fn upload(
        &self,
        file: &UploadFile,
        purpose: FilePurpose,
    ) -> Result<ProviderFileRef, LlmError> {
        self.upload_with_expiration(file, purpose, None).await
    }

    /// Automatic uploads receive a bounded lifetime where supported. Explicit
    /// FileService uploads retain their existing caller-managed lifetime.
    pub(crate) async fn upload_automatic(
        &self,
        file: &UploadFile,
        purpose: FilePurpose,
    ) -> Result<ProviderFileRef, LlmError> {
        let expiration = matches!(
            adapter(self.profile),
            Some(Adapter::Anthropic | Adapter::OpenAi | Adapter::Xai)
        )
        .then_some(AUTOMATIC_FILE_TTL.as_secs());
        let mut retries = 0;
        let uploaded = loop {
            match self.upload_with_expiration(file, purpose, expiration).await {
                Err(LlmError::RateLimited { .. })
                    if adapter(self.profile) == Some(Adapter::Qwen) && retries < 3 =>
                {
                    async_delay(Duration::from_millis(500 * (1 << retries))).await;
                    retries += 1;
                }
                result => break result?,
            }
        };
        if adapter(self.profile) == Some(Adapter::Anthropic)
            && serde_json::to_string(&uploaded.file_id).map_or(true, |encoded| {
                encoded.len() > MAX_AUTOMATIC_ANTHROPIC_FILE_ID_JSON_BYTES
            })
        {
            // A provider ID longer than the preflight bound cannot safely be
            // substituted into a request that was checked before upload.
            let _ = self.delete(&uploaded).await;
            return Err(LlmError::UnsupportedCapability {
                message: "Anthropic returned a file ID too long for automatic request preparation"
                    .into(),
            });
        }
        Ok(uploaded)
    }

    pub(in crate::files) async fn upload_with_expiration(
        &self,
        file: &UploadFile,
        purpose: FilePurpose,
        expires_in_seconds: Option<u64>,
    ) -> Result<ProviderFileRef, LlmError> {
        validate_media_type(&file.media_type)?;
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("upload"))?;
        let caps = capabilities_for_media_type(
            purpose_capabilities(adapter, purpose, &file.media_type),
            adapter,
            &file.media_type,
        );
        if !caps.upload {
            return Err(unsupported("upload for this purpose"));
        }
        if matches!(
            purpose,
            FilePurpose::ModelInput | FilePurpose::VideoUnderstanding
        ) && self
            .account_scope
            .is_none_or(|scope| scope.trim().is_empty())
        {
            return Err(LlmError::InvalidRequest {
                message: "model-input file uploads require a nonempty account scope".into(),
            });
        }
        if let Some(limit) = caps.max_upload_bytes {
            if u64::try_from(file.bytes.len()).unwrap_or(u64::MAX) > limit {
                return Err(LlmError::RequestTooLarge {
                    message: format!("file exceeds the provider limit of {limit} bytes"),
                });
            }
        }

        if adapter == Adapter::Gemini {
            return self.upload_gemini(file).await;
        }

        let boundary = multipart_boundary();
        let body = multipart_body(adapter, purpose, file, &boundary, expires_in_seconds)?;
        let url = if adapter == Adapter::MiniMax {
            format!("{}/upload", files_url(self.profile, adapter))
        } else {
            files_url(self.profile, adapter)
        };
        let req = self
            .request(
                "POST",
                url,
                body,
                Some(format!("multipart/form-data; boundary={boundary}")),
            )
            .await?;
        if adapter == Adapter::Qwen {
            self.pace_qwen_upload().await;
        }
        let response = crate::transport::HttpExecutor::new(self.http)
            .execute(req)
            .await?;
        let value = adapter_json_success(adapter, &response, "file upload")?;
        let mut metadata = decode_metadata(self.profile, self.account_scope, &value)?;
        // The API may omit MIME metadata on upload, but the input is known here.
        if metadata.file.media_type.is_none() {
            metadata.file.media_type = Some(file.media_type.clone());
        }
        if metadata.file.filename.is_none() {
            metadata.file.filename = Some(file.filename.clone());
        }
        if metadata.file.purpose.is_none() {
            metadata.file.purpose = purpose_name(adapter, purpose).map(str::to_owned);
        }
        Ok(metadata.file)
    }

    /// Retrieve metadata for a provider-owned file.
    pub async fn get(&self, file: &ProviderFileRef) -> Result<ProviderFileMetadata, LlmError> {
        self.check_ref(file)?;
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("metadata retrieval"))?;
        if !capabilities_for_adapter(adapter).retrieve_metadata {
            return Err(unsupported("metadata retrieval"));
        }
        let url = if adapter == Adapter::MiniMax {
            format!(
                "{}?file_id={}",
                file_url(self.profile, adapter, &file.file_id),
                query_value(&file.file_id)
            )
        } else {
            file_url(self.profile, adapter, &file.file_id)
        };
        let req = self.request("GET", url, Bytes::new(), None).await?;
        if adapter == Adapter::Qwen {
            self.pace_qwen_metadata().await;
        }
        let response = crate::transport::HttpExecutor::new(self.http)
            .execute(req)
            .await?;
        let value = adapter_json_success(adapter, &response, "file metadata")?;
        let mut metadata = decode_metadata(self.profile, self.account_scope, &value)?;
        // Some file APIs omit MIME in metadata responses. Preserve caller-known
        // metadata only for the same file; never replace a server-stated value.
        if metadata.file.file_id == file.file_id && metadata.file.media_type.is_none() {
            metadata.file.media_type.clone_from(&file.media_type);
        }
        Ok(metadata)
    }

    /// List provider-owned files. The cursor is opaque.
    pub async fn list(&self, cursor: Option<&str>) -> Result<ProviderFilePage, LlmError> {
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("file listing"))?;
        if adapter == Adapter::MiniMax {
            return Err(LlmError::InvalidRequest {
                message: "MiniMax file listing requires list_for_purpose with an explicit purpose"
                    .into(),
            });
        }
        self.list_inner(adapter, None, cursor).await
    }

    /// List files for an API that requires a purpose filter (MiniMax), or
    /// constrain OpenAI and Qwen listings to a documented provider purpose.
    /// Other adapters return `UnsupportedCapability` rather than an unfiltered page.
    pub async fn list_for_purpose(
        &self,
        purpose: FilePurpose,
        cursor: Option<&str>,
    ) -> Result<ProviderFilePage, LlmError> {
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("file listing"))?;
        if !matches!(adapter, Adapter::MiniMax | Adapter::OpenAi | Adapter::Qwen) {
            return Err(unsupported("file listing filtered by purpose"));
        }
        if adapter != Adapter::MiniMax && purpose_name(adapter, purpose).is_none() {
            return Err(unsupported("file listing for this purpose"));
        }
        if adapter == Adapter::MiniMax && minimax_list_purpose(purpose).is_none() {
            return Err(unsupported("MiniMax file listing for this purpose"));
        }
        if adapter == Adapter::MiniMax && cursor.is_some() {
            return Err(unsupported("MiniMax file-list pagination"));
        }
        self.list_inner(adapter, Some(purpose), cursor).await
    }

    pub(in crate::files) async fn list_inner(
        &self,
        adapter: Adapter,
        purpose: Option<FilePurpose>,
        cursor: Option<&str>,
    ) -> Result<ProviderFilePage, LlmError> {
        if !capabilities_for_adapter(adapter).list {
            return Err(unsupported("file listing"));
        }
        let (url, cursor_field) = list_url(self.profile, adapter, purpose, cursor);
        let req = self.request("GET", url, Bytes::new(), None).await?;
        if adapter == Adapter::Qwen {
            self.pace_qwen_metadata().await;
        }
        let response = crate::transport::HttpExecutor::new(self.http)
            .execute(req)
            .await?;
        let value = adapter_json_success(adapter, &response, "file listing")?;
        if !value.is_object() {
            return Err(provider_shape("file list response is not an object"));
        }
        let rows = if adapter == Adapter::Gemini {
            value.get("files")
        } else {
            value
                .pointer("/data/files")
                .or_else(|| value.pointer("/data/file_list"))
                .or_else(|| value.pointer("/data/data"))
                .or_else(|| value.get("data").filter(|data| data.is_array()))
                .or_else(|| value.get("files"))
                .or_else(|| value.get("file_list"))
        };
        let rows = match rows {
            // Gemini's ProtoJSON omits an empty repeated field. Null is also
            // the JSON representation of an unset protobuf field.
            None | Some(Value::Null) if adapter == Adapter::Gemini => &[][..],
            Some(Value::Array(rows)) => rows.as_slice(),
            _ => return Err(provider_shape("file list has no array of files")),
        };
        let files = rows
            .iter()
            .map(|row| decode_metadata(self.profile, self.account_scope, row))
            .collect::<Result<Vec<_>, _>>()?;
        let next_cursor = match cursor_field {
            CursorField::OpenAi => value
                .get("has_more")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                .then(|| value.get("last_id").and_then(nonempty_string))
                .flatten(),
            CursorField::Qwen => {
                if value.get("has_more").and_then(Value::as_bool) == Some(true) {
                    Some(
                        rows.last()
                            .and_then(|row| row.get("id"))
                            .and_then(nonempty_string)
                            .ok_or_else(|| provider_shape("Qwen file list has no next-page ID"))?,
                    )
                } else {
                    None
                }
            }
            CursorField::Anthropic => value.get("next_page").and_then(nonempty_string),
            CursorField::Gemini => value.get("nextPageToken").and_then(nonempty_string),
            CursorField::Xai if rows.len() >= 100 => {
                value.get("pagination_token").and_then(nonempty_string)
            }
            CursorField::Xai => None,
            CursorField::OpenRouter => value
                .get("next_cursor")
                .or_else(|| value.get("cursor"))
                .and_then(nonempty_string),
            CursorField::MiniMax => None,
        };
        Ok(ProviderFilePage { files, next_cursor })
    }

    /// Delete provider-owned storage. This does not delete the app's original
    /// attachment.
    pub async fn delete(&self, file: &ProviderFileRef) -> Result<(), LlmError> {
        self.check_ref(file)?;
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("file deletion"))?;
        if !capabilities_for_adapter(adapter).delete {
            return Err(unsupported("file deletion"));
        }
        let (method, url, body, content_type) = if adapter == Adapter::MiniMax {
            let purpose = file
                .purpose
                .as_deref()
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "MiniMax file deletion requires the original provider purpose".into(),
                })?;
            let purpose = minimax_delete_purpose(purpose)
                .ok_or_else(|| unsupported("MiniMax deletion for this purpose"))?;
            (
                "POST",
                format!("{}/delete", files_url(self.profile, adapter)),
                Bytes::from(
                    serde_json::to_vec(
                        &serde_json::json!({"file_id": file.file_id, "purpose": purpose}),
                    )
                    .map_err(|error| provider_shape(&error.to_string()))?,
                ),
                Some("application/json".into()),
            )
        } else {
            (
                "DELETE",
                file_url(self.profile, adapter, &file.file_id),
                Bytes::new(),
                None,
            )
        };
        let req = self.request(method, url, body, content_type).await?;
        if adapter == Adapter::Qwen {
            self.pace_qwen_metadata().await;
        }
        let response = crate::transport::HttpExecutor::new(self.http)
            .execute(req)
            .await?;
        adapter_status_result(adapter, &response, "file deletion")?;
        if adapter == Adapter::Qwen {
            let value: Value = serde_json::from_slice(&response.body).map_err(|error| {
                provider_shape(&format!("file deletion response was not JSON: {error}"))
            })?;
            if value.get("deleted").and_then(Value::as_bool) != Some(true)
                || value.get("id").and_then(Value::as_str) != Some(file.file_id.as_str())
            {
                return Err(provider_shape(
                    "Qwen did not confirm deletion of the requested file",
                ));
            }
        }
        Ok(())
    }

    /// Download original provider bytes where the provider documents that
    /// uploaded files are retrievable. The accumulated body is capped at 64 MiB.
    pub async fn download(&self, file: &ProviderFileRef) -> Result<ProviderFileContent, LlmError> {
        self.check_ref(file)?;
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("file download"))?;
        let support = capabilities_for_adapter(adapter).download;
        if support == DownloadSupport::Unsupported {
            return Err(unsupported("file download"));
        }
        if matches!(
            support,
            DownloadSupport::GeneratedFilesOnly | DownloadSupport::ProviderMarkedDownloadable
        ) {
            let metadata = self.get(file).await?;
            if metadata.file.downloadable != Some(true) {
                return Err(unsupported("download of this uploaded file"));
            }
        }
        if adapter == Adapter::Moonshot {
            return Err(unsupported(
                "raw download; use extract_text for Moonshot files",
            ));
        }
        let url = if adapter == Adapter::MiniMax {
            format!(
                "{}?file_id={}",
                content_url(self.profile, adapter, &file.file_id),
                query_value(&file.file_id)
            )
        } else {
            content_url(self.profile, adapter, &file.file_id)
        };
        let req = self.request("GET", url, Bytes::new(), None).await?;
        let mut response = crate::transport::HttpExecutor::new(self.http)
            .send(req)
            .await?;
        if !(200..300).contains(&response.status) {
            let mut body = BytesMut::new();
            while let Some(frame) = response.body.next().await {
                if let Ok(frame) = frame {
                    let remaining =
                        crate::transport::MAX_ERROR_BODY_SIZE.saturating_sub(body.len());
                    if remaining == 0 {
                        break;
                    }
                    body.extend_from_slice(&frame[..frame.len().min(remaining)]);
                }
            }
            return Err(status_error(
                response.status,
                &String::from_utf8_lossy(&body),
                "file download",
            ));
        }
        let media_type = response
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
            .map(|(_, value)| value.split(';').next().unwrap_or(value).trim().to_owned());
        let mut body = BytesMut::new();
        while let Some(frame) = response.body.next().await {
            let frame = frame?;
            if body.len().saturating_add(frame.len()) > MAX_PROVIDER_FILE_DOWNLOAD_BYTES {
                return Err(LlmError::RequestTooLarge {
                    message: format!(
                        "provider file exceeds the {} byte download cap",
                        MAX_PROVIDER_FILE_DOWNLOAD_BYTES
                    ),
                });
            }
            body.extend_from_slice(&frame);
        }
        Ok(ProviderFileContent {
            media_type,
            bytes: body.freeze(),
        })
    }

    /// Extract the provider's text representation of an uploaded file. This is
    /// distinct from downloading the original binary; currently only
    /// Moonshot/Kimi documents this operation in the supported adapter set.
    pub async fn extract_text(&self, file: &ProviderFileRef) -> Result<String, LlmError> {
        self.check_ref(file)?;
        let adapter = adapter(self.profile).ok_or_else(|| unsupported("text extraction"))?;
        if !capabilities_for_adapter(adapter).extract_text {
            return Err(unsupported("text extraction"));
        }
        let url = format!(
            "{}/{}/content",
            files_url(self.profile, adapter),
            path_segment(&file.file_id)
        );
        let req = self.request("GET", url, Bytes::new(), None).await?;
        let response = crate::transport::HttpExecutor::new(self.http)
            .execute(req)
            .await?;
        status_result(&response, "file text extraction")?;
        String::from_utf8(response.body.to_vec())
            .map_err(|error| provider_shape(&format!("extracted file text is not UTF-8: {error}")))
    }

    pub(in crate::files) async fn request(
        &self,
        method: &str,
        url: String,
        body: Bytes,
        content_type: Option<String>,
    ) -> Result<HttpRequest, LlmError> {
        let mut headers = vec![("accept".into(), "application/json".into())];
        if let Some(content_type) = content_type {
            headers.push(("content-type".into(), content_type));
        }
        crate::wire_options::merge_headers(self.profile, &mut headers);
        if adapter(self.profile) == Some(Adapter::Anthropic) {
            headers.retain(|(name, _)| !name.eq_ignore_ascii_case("anthropic-version"));
            let version = self
                .profile
                .extra
                .get("api_version")
                .and_then(Value::as_str)
                .unwrap_or(crate::wire_options::DEFAULT_ANTHROPIC_API_VERSION);
            headers.push(("anthropic-version".into(), version.to_owned()));
        }
        let mut request = HttpRequest {
            method: method.to_owned(),
            url,
            headers,
            body,
            timeout: Some(FILE_TIMEOUT),
        };
        if self.profile.auth != AuthStrategy::None {
            let authenticator =
                self.authenticator
                    .ok_or_else(|| LlmError::UnsupportedCapability {
                        message: format!(
                            "no authenticator is registered for file operations on profile {:?}",
                            self.profile.profile_name
                        ),
                    })?;
            let deadline =
                crate::runtime::Deadline::at(self.gemini_request_deadline).cap(request.timeout);
            deadline
                .run(authenticator.apply(&mut request, self.profile, self.credential))
                .await??;
            request.timeout = deadline.remaining()?;
        }
        Ok(request)
    }

    pub(in crate::files) fn check_ref(&self, file: &ProviderFileRef) -> Result<(), LlmError> {
        if file.profile_name != self.profile.profile_name
            || file.provider_id != self.profile.provider_id
            || file.protocol != self.profile.protocol
            || file.endpoint_fingerprint
                != provider_file_endpoint_fingerprint(&self.profile.base_url)
            || file.account_scope.as_deref() != self.account_scope
        {
            return Err(LlmError::PermissionDenied {
                message:
                    "provider file reference belongs to another profile, endpoint, or account scope"
                        .into(),
            });
        }
        Ok(())
    }
}
