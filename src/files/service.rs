//! Connection-bound provider file operations.
use super::*;

/// File API adapter bound to one profile and one credential scope.
///
/// Construct this from the same profile, authenticator and per-attempt
/// credential used for a model request. File IDs are therefore never silently
/// reused across failover connections.
pub struct FileService<'a> {
    pub(crate) http: &'a dyn Transport,
    pub(crate) profile: &'a ProviderProfile,
    pub(crate) authenticator: Option<&'a dyn Authenticator>,
    pub(crate) credential: Option<&'a Secret<String>>,
    pub(crate) account_scope: Option<&'a str>,
    /// An explicit service route for APIs whose availability is a typed
    /// hosting choice rather than a fact inferable from provider ID and URL.
    adapter_override: Option<Adapter>,
    /// Canonical identity endpoint for an explicitly selected hosted route.
    endpoint_identity: Option<String>,
    pub(crate) qwen_rate_limiter: Option<Arc<QwenFileRateLimiter>>,
    pub(crate) gemini_upload_timeout: Option<Duration>,
    pub(crate) gemini_processing_timeout: Option<Duration>,
    pub(crate) request_deadline: Option<Instant>,
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
            adapter_override: None,
            endpoint_identity: None,
            qwen_rate_limiter: None,
            gemini_upload_timeout: None,
            gemini_processing_timeout: None,
            request_deadline: None,
        }
    }

    /// Bind the Anthropic Files API to a Microsoft Foundry resource explicitly
    /// marked as Anthropic-hosted. Files are resource/account scoped; this
    /// constructor intentionally does not require a chat model catalog row.
    /// The endpoint must be the HTTPS `https://{resource}.services.ai.azure.com/anthropic`
    /// base, and `account_scope` must be a stable, non-secret identifier for
    /// the account/resource credentials used by this service.
    pub fn new_foundry(
        http: &'a dyn Transport,
        profile: &'a ProviderProfile,
        hosting: crate::protocol::FoundryHosting,
        authenticator: Option<&'a dyn Authenticator>,
        credential: Option<&'a Secret<String>>,
        account_scope: &'a str,
    ) -> Result<Self, LlmError> {
        if profile.protocol != ProtocolFamily::FoundryClaude {
            return Err(unsupported(
                "Foundry Files API requires the Foundry Claude protocol",
            ));
        }
        if hosting != crate::protocol::FoundryHosting::Anthropic {
            return Err(unsupported(
                "Foundry Files API requires an Anthropic-hosted deployment",
            ));
        }
        if account_scope.trim().is_empty() || account_scope.chars().any(char::is_control) {
            return Err(LlmError::InvalidRequest {
                message: "Foundry file operations require a stable, nonempty account scope".into(),
            });
        }
        let endpoint_identity =
            crate::providers::anthropic::types::AnthropicContainerScope::normalize_foundry_endpoint(&profile.base_url)
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "Foundry Files API requires the HTTPS Anthropic resource endpoint"
                        .into(),
                })?;
        let mut service = Self::new(
            http,
            profile,
            authenticator,
            credential,
            Some(account_scope),
        );
        service.adapter_override = Some(Adapter::Anthropic);
        service.endpoint_identity = Some(endpoint_identity);
        Ok(service)
    }

    /// Construct a Foundry Files service from an actual model row in the
    /// profile. Services-only profiles should use [`Self::new_foundry`], which
    /// takes an explicit hosting assertion and does not invent chat models.
    pub fn new_foundry_for_model(
        http: &'a dyn Transport,
        profile: &'a ProviderProfile,
        selected_model: &ModelProfile,
        authenticator: Option<&'a dyn Authenticator>,
        credential: Option<&'a Secret<String>>,
        account_scope: &'a str,
    ) -> Result<Self, LlmError> {
        if !profile.models.iter().any(|model| model == selected_model) {
            return Err(unsupported(
                "selected Foundry model row does not belong to this profile",
            ));
        }
        let deployment = selected_model.foundry.as_ref().ok_or_else(|| {
            unsupported("Foundry Files API requires explicit hosting on the selected model row")
        })?;
        Self::new_foundry(
            http,
            profile,
            deployment.hosting,
            authenticator,
            credential,
            account_scope,
        )
    }

    pub(crate) fn adapter(&self) -> Option<Adapter> {
        self.adapter_override.or_else(|| adapter(self.profile))
    }

    pub(crate) fn endpoint_identity(&self) -> String {
        self.endpoint_identity
            .clone()
            .or_else(|| provider_file_endpoint_identity(self.profile))
            .unwrap_or_else(|| self.profile.base_url.clone())
    }

    pub(crate) fn decode_metadata(&self, value: &Value) -> Result<ProviderFileMetadata, LlmError> {
        let adapter = self
            .adapter()
            .ok_or_else(|| unsupported("file metadata decoding"))?;
        let endpoint_identity = self.endpoint_identity();
        decode_metadata_for_adapter(
            self.profile,
            self.account_scope,
            value,
            adapter,
            &endpoint_identity,
        )
    }

    pub(crate) fn with_request_deadline(mut self, deadline: Instant) -> Self {
        self.request_deadline = Some(deadline);
        self
    }

    /// Return capabilities for this concrete profile, model and media type.
    /// The caller must still check model vision/file modality when present.
    #[must_use]
    pub fn capabilities(&self, model: &str, media_type: &str) -> FileCapabilities {
        let Some(adapter) = self.adapter() else {
            return FileCapabilities::unsupported();
        };
        let mut result =
            capabilities_for_media_type(capabilities_for_adapter(adapter), adapter, media_type);
        result.model_input = model_reference_for(self.profile, model, media_type, adapter);
        result
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
        let Some(adapter) = self.adapter() else {
            return FileCapabilities::unsupported();
        };
        let mut result = capabilities_for_media_type(
            purpose_capabilities(adapter, purpose, media_type),
            adapter,
            media_type,
        );
        result.model_input = if matches!(
            purpose,
            FilePurpose::ModelInput | FilePurpose::VideoUnderstanding
        ) {
            model_reference_for(self.profile, model, media_type, adapter)
        } else {
            ModelFileReference::Unsupported
        };
        result
    }

    /// Upload bytes using a provider-supported purpose.
    pub async fn upload(
        &self,
        file: &UploadFile,
        purpose: FilePurpose,
    ) -> Result<ProviderFileRef, LlmError> {
        self.upload_with_expiration(file, purpose, None).await
    }

    /// Upload a one-shot byte stream using a provider-supported purpose.
    /// The declared size is validated before authentication or network work;
    /// the stream must end at exactly that size. This method never retries or
    /// polls. A Gemini file that is still processing is returned in a
    /// `ProviderFileProcessing` error so the caller can resume it explicitly.
    pub async fn upload_stream(
        &self,
        file: UploadFileStream,
        purpose: FilePurpose,
    ) -> Result<ProviderFileRef, FileUploadError> {
        let (filename, media_type, size_bytes, body) = file.into_parts();
        let (adapter, _) =
            self.preflight_upload(&filename, &media_type, size_bytes, purpose, true)?;
        if adapter == Adapter::Gemini {
            let is_video = media_type.to_ascii_lowercase().starts_with("video/");
            let timeout = self.gemini_upload_timeout.unwrap_or(if is_video {
                GEMINI_VIDEO_FILE_TIMEOUT
            } else {
                FILE_TIMEOUT
            });
            let timeout = if is_video {
                timeout.min(GEMINI_VIDEO_FILE_TIMEOUT)
            } else {
                timeout
            };
            return self
                .upload_gemini_stream(filename, media_type, size_bytes, body, timeout)
                .await;
        }
        self.upload_multipart_stream(
            filename,
            media_type,
            size_bytes,
            purpose,
            body,
            FILE_TIMEOUT,
        )
        .await
    }

    /// Stream a caller-managed OpenAI Batch JSONL file without buffering its
    /// multipart body. The declared size must match the input stream exactly.
    /// A connection failure can leave an accepted upload with an unknown ID;
    /// this method never retries it.
    pub async fn upload_batch_stream(
        &self,
        filename: &str,
        media_type: &str,
        size_bytes: u64,
        body: BoxStream<'static, Result<Bytes, LlmError>>,
        timeout: Option<Duration>,
    ) -> Result<ProviderFileRef, LlmError> {
        validate_media_type(media_type)?;
        let adapter = self.adapter().ok_or_else(|| unsupported("batch upload"))?;
        if adapter != Adapter::OpenAi {
            return Err(unsupported("batch streaming upload"));
        }
        if !filename.ends_with(".jsonl")
            || !matches!(
                media_type,
                "application/x-ndjson" | "application/jsonl" | "application/json"
            )
        {
            return Err(LlmError::InvalidRequest {
                message: "batch input must be a .jsonl file with a JSONL media type".into(),
            });
        }
        if self
            .account_scope
            .is_none_or(|scope| scope.trim().is_empty())
        {
            return Err(LlmError::InvalidRequest {
                message: "batch file upload requires a nonempty account scope".into(),
            });
        }
        if size_bytes == 0 || size_bytes > 200_000_000 {
            return Err(LlmError::RequestTooLarge {
                message: "batch input must contain 1–200000000 bytes".into(),
            });
        }
        let timeout = timeout.unwrap_or(FILE_TIMEOUT);
        if timeout.is_zero() {
            return Err(LlmError::InvalidRequest {
                message: "batch upload timeout must be positive".into(),
            });
        }
        let boundary = multipart_boundary();
        let (prefix, suffix) = multipart_batch_parts(filename, media_type, &boundary);
        let content_length = size_bytes
            .checked_add(prefix.len() as u64)
            .and_then(|len| len.checked_add(suffix.len() as u64))
            .ok_or_else(|| LlmError::RequestTooLarge {
                message: "batch multipart size overflows".into(),
            })?;
        let deadline = crate::runtime::Deadline::at(self.request_deadline).cap(Some(timeout));
        let request = deadline
            .run(self.request(
                "POST",
                files_url(self.profile, adapter),
                Bytes::new(),
                Some(format!("multipart/form-data; boundary={boundary}")),
            ))
            .await??;
        if !request.body.is_empty() {
            return Err(LlmError::UnsupportedCapability {
                message: "batch streaming authenticator must not replace the request body".into(),
            });
        }
        let parts = stream_batch_body(prefix, body, size_bytes, suffix);
        let request = crate::transport::HttpStreamRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: parts,
            content_length,
            timeout: deadline.remaining()?,
        };
        let response = crate::transport::HttpExecutor::new(self.http)
            .with_deadline(deadline)
            .send_stream(request)
            .await?;
        let response =
            crate::transport::HttpExecutor::collect_response(response, Some(1024 * 1024)).await?;
        let value = adapter_json_success(adapter, &response, "batch file upload")?;
        let mut metadata = self.decode_metadata(&value)?;
        if metadata
            .file
            .purpose
            .as_deref()
            .is_some_and(|purpose| purpose != "batch")
        {
            return Err(provider_shape(
                "batch upload returned a different file purpose",
            ));
        }
        if metadata.file.media_type.is_none() {
            metadata.file.media_type = Some(media_type.into());
        }
        if metadata.file.filename.is_none() {
            metadata.file.filename = Some(filename.into());
        }
        if metadata.file.purpose.is_none() {
            metadata.file.purpose = Some("batch".into());
        }
        if metadata.file.size_bytes.is_none() {
            metadata.file.size_bytes = Some(size_bytes);
        }
        Ok(metadata.file)
    }

    /// Automatic uploads receive a bounded lifetime where supported. Explicit
    /// FileService uploads retain their existing caller-managed lifetime.
    pub(crate) async fn upload_automatic(
        &self,
        file: &UploadFile,
        purpose: FilePurpose,
    ) -> Result<ProviderFileRef, LlmError> {
        let expiration = matches!(
            self.adapter(),
            Some(Adapter::Anthropic | Adapter::OpenAi | Adapter::Xai)
        )
        .then_some(AUTOMATIC_FILE_TTL.as_secs());
        let mut retries = 0;
        let uploaded = loop {
            match self.upload_with_expiration(file, purpose, expiration).await {
                Err(LlmError::RateLimited { .. })
                    if self.adapter() == Some(Adapter::Qwen) && retries < 3 =>
                {
                    async_delay(Duration::from_millis(500 * (1 << retries))).await;
                    retries += 1;
                }
                result => break result?,
            }
        };
        if self.adapter() == Some(Adapter::Anthropic)
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

    pub(crate) async fn upload_with_expiration(
        &self,
        file: &UploadFile,
        purpose: FilePurpose,
        expires_in_seconds: Option<u64>,
    ) -> Result<ProviderFileRef, LlmError> {
        let deadline = crate::runtime::Deadline::at(self.request_deadline);
        let (adapter, _) = self.preflight_upload(
            &file.filename,
            &file.media_type,
            u64::try_from(file.bytes.len()).unwrap_or(u64::MAX),
            purpose,
            false,
        )?;

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
        let req = deadline
            .run(self.request(
                "POST",
                url,
                body,
                Some(format!("multipart/form-data; boundary={boundary}")),
            ))
            .await??;
        if adapter == Adapter::Qwen {
            deadline.run(self.pace_qwen_upload()).await?;
        }
        deadline.remaining()?;
        let response = crate::transport::HttpExecutor::new(self.http)
            .with_deadline(deadline)
            .execute(req)
            .await
            .map_err(|error| buffered_upload_error("multipart upload", error))?;
        if !(200..300).contains(&response.status) {
            let error = status_error(
                response.status,
                &String::from_utf8_lossy(&response.body),
                "file upload",
            );
            return Err(if response.status >= 500 {
                unknown_buffered_upload_outcome("multipart upload response", error)
            } else {
                error
            });
        }
        let value: Value = serde_json::from_slice(&response.body).map_err(|error| {
            unknown_buffered_upload_outcome(
                "multipart upload response",
                provider_shape(&format!("file upload response was not JSON: {error}")),
            )
        })?;
        if adapter == Adapter::MiniMax {
            if let Err(error) = check_minimax_base_response(&value, "file upload") {
                return Err(
                    if minimax_base_response_code(&value).is_some_and(|code| code != 0) {
                        error
                    } else {
                        unknown_buffered_upload_outcome("multipart upload response", error)
                    },
                );
            }
        }
        let mut metadata = self
            .decode_metadata(&value)
            .map_err(|error| unknown_buffered_upload_outcome("multipart upload response", error))?;
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

    pub(crate) fn preflight_upload(
        &self,
        filename: &str,
        media_type: &str,
        size_bytes: u64,
        purpose: FilePurpose,
        require_nonempty_batch: bool,
    ) -> Result<(Adapter, FileCapabilities), LlmError> {
        validate_media_type(media_type)?;
        let adapter = self.adapter().ok_or_else(|| unsupported("upload"))?;
        let caps = capabilities_for_media_type(
            purpose_capabilities(adapter, purpose, media_type),
            adapter,
            media_type,
        );
        if !caps.upload {
            return Err(unsupported("upload for this purpose"));
        }
        if purpose == FilePurpose::Batch
            && (!filename.ends_with(".jsonl")
                || !matches!(
                    media_type,
                    "application/x-ndjson" | "application/jsonl" | "application/json"
                ))
        {
            return Err(LlmError::InvalidRequest {
                message: "batch input must be a .jsonl file with a JSONL media type".into(),
            });
        }
        if require_nonempty_batch && purpose == FilePurpose::Batch && size_bytes == 0 {
            return Err(LlmError::RequestTooLarge {
                message: "batch input must contain at least one byte".into(),
            });
        }
        if matches!(
            purpose,
            FilePurpose::ModelInput
                | FilePurpose::VideoUnderstanding
                | FilePurpose::Batch
                | FilePurpose::AsyncTtsInput
        ) && self
            .account_scope
            .is_none_or(|scope| scope.trim().is_empty())
        {
            return Err(LlmError::InvalidRequest {
                message: "model-input, batch, video-understanding, and async TTS uploads require a nonempty account scope".into(),
            });
        }
        if let Some(limit) = caps.max_upload_bytes {
            if size_bytes > limit {
                return Err(LlmError::RequestTooLarge {
                    message: format!("file exceeds the provider limit of {limit} bytes"),
                });
            }
        }
        Ok((adapter, caps))
    }

    async fn upload_multipart_stream(
        &self,
        filename: String,
        media_type: String,
        size_bytes: u64,
        purpose: FilePurpose,
        body: BoxStream<'static, Result<Bytes, LlmError>>,
        timeout: Duration,
    ) -> Result<ProviderFileRef, FileUploadError> {
        let adapter = self.adapter().ok_or_else(|| unsupported("upload"))?;
        let boundary = multipart_boundary();
        let (prefix, suffix) =
            multipart_parts(adapter, purpose, &filename, &media_type, &boundary, None);
        let content_length = size_bytes
            .checked_add(prefix.len() as u64)
            .and_then(|length| length.checked_add(suffix.len() as u64))
            .ok_or_else(|| LlmError::RequestTooLarge {
                message: "file multipart size overflows".into(),
            })?;
        let url = if adapter == Adapter::MiniMax {
            format!("{}/upload", files_url(self.profile, adapter))
        } else {
            files_url(self.profile, adapter)
        };
        let deadline = crate::runtime::Deadline::at(self.request_deadline).cap(Some(timeout));
        let mut request = deadline
            .run(self.request(
                "POST",
                url,
                Bytes::new(),
                Some(format!("multipart/form-data; boundary={boundary}")),
            ))
            .await??;
        if !request.body.is_empty() {
            return Err(unsupported(
                "stream upload authenticator must not replace the request body",
            )
            .into());
        }
        if adapter == Adapter::Qwen {
            deadline.run(self.pace_qwen_upload()).await?;
        }
        request.timeout = deadline.remaining()?;
        let request = crate::transport::HttpStreamRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: multipart_upload_stream(prefix, body, size_bytes, suffix),
            content_length,
            timeout: deadline.remaining()?,
        };
        let response = crate::transport::HttpExecutor::new(self.http)
            .with_deadline(deadline)
            .send_stream(request)
            .await
            .map_err(|error| match error {
                LlmError::UnsupportedCapability { .. } => FileUploadError::Llm(error),
                source => unknown_upload_outcome("multipart upload", source, None),
            })?;
        let response =
            crate::transport::HttpExecutor::collect_response(response, Some(1024 * 1024))
                .await
                .map_err(|source| {
                    unknown_upload_outcome("multipart upload response", source, None)
                })?;
        self.decode_stream_upload_response(adapter, &response, &filename, &media_type, purpose)
    }

    pub(crate) fn decode_stream_upload_response(
        &self,
        adapter: Adapter,
        response: &HttpResponse,
        filename: &str,
        media_type: &str,
        purpose: FilePurpose,
    ) -> Result<ProviderFileRef, FileUploadError> {
        if !(200..300).contains(&response.status) {
            let source = status_error(
                response.status,
                &String::from_utf8_lossy(&response.body),
                "file upload",
            );
            return Err(if response.status >= 500 {
                unknown_upload_outcome("multipart upload response", source, None)
            } else {
                FileUploadError::Llm(source)
            });
        }
        let value: Value = serde_json::from_slice(&response.body).map_err(|error| {
            unknown_upload_outcome(
                "multipart upload response",
                provider_shape(&format!("file upload response was not JSON: {error}")),
                None,
            )
        })?;
        if adapter == Adapter::MiniMax {
            if let Err(source) = check_minimax_base_response(&value, "file upload") {
                return Err(
                    if minimax_base_response_code(&value).is_some_and(|code| code != 0) {
                        FileUploadError::Llm(source)
                    } else {
                        unknown_upload_outcome("multipart upload response", source, None)
                    },
                );
            }
        }
        let mut metadata = self
            .decode_metadata(&value)
            .map_err(|source| unknown_upload_outcome("multipart upload response", source, None))?;
        if let Some(expected_purpose) = purpose_name(adapter, purpose) {
            if metadata
                .file
                .purpose
                .as_deref()
                .is_some_and(|actual_purpose| actual_purpose != expected_purpose)
            {
                return Err(unknown_upload_outcome(
                    "multipart upload response",
                    provider_shape("file upload response returned a different purpose"),
                    Some(metadata.file),
                ));
            }
        }
        if metadata.file.media_type.is_none() {
            metadata.file.media_type = Some(media_type.into());
        }
        if metadata.file.filename.is_none() {
            metadata.file.filename = Some(filename.into());
        }
        if metadata.file.purpose.is_none() {
            metadata.file.purpose = purpose_name(adapter, purpose).map(str::to_owned);
        }
        Ok(metadata.file)
    }

    /// Retrieve metadata for a provider-owned file.
    pub async fn get(&self, file: &ProviderFileRef) -> Result<ProviderFileMetadata, LlmError> {
        self.check_ref(file)?;
        let adapter = self
            .adapter()
            .ok_or_else(|| unsupported("metadata retrieval"))?;
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
        let mut metadata = self.decode_metadata(&value)?;
        if adapter == Adapter::Anthropic && metadata.file.file_id != file.file_id {
            return Err(provider_shape(
                "Anthropic file metadata returned a different file ID than requested",
            ));
        }
        // Some file APIs omit MIME in metadata responses. Preserve caller-known
        // metadata only for the same file; never replace a server-stated value.
        if metadata.file.file_id == file.file_id && metadata.file.media_type.is_none() {
            metadata.file.media_type.clone_from(&file.media_type);
        }
        let native_file = value.get("file").unwrap_or(&value);
        if metadata.file.file_id == file.file_id
            && native_file.get("expires_at").is_none()
            && native_file.get("expirationTime").is_none()
        {
            // An omitted field provides no new lifetime information. An
            // explicit null, however, is the provider clearing expiration.
            metadata.file.expires_at.clone_from(&file.expires_at);
        }
        if metadata.file.file_id == file.file_id
            && native_file.get("status").is_none()
            && native_file.get("state").is_none()
        {
            // An omitted status does not erase the last provider-reported
            // state. Explicit null clears it because decode_metadata returns
            // None and this fallback is skipped.
            metadata
                .file
                .processing_status
                .clone_from(&file.processing_status);
            metadata.status.clone_from(&file.processing_status);
        }
        Ok(metadata)
    }

    /// List provider-owned files. The cursor is opaque.
    pub async fn list(&self, cursor: Option<&str>) -> Result<ProviderFilePage, LlmError> {
        let adapter = self.adapter().ok_or_else(|| unsupported("file listing"))?;
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
        let adapter = self.adapter().ok_or_else(|| unsupported("file listing"))?;
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

    pub(crate) async fn list_inner(
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
            .map(|row| self.decode_metadata(row))
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
        let adapter = self.adapter().ok_or_else(|| unsupported("file deletion"))?;
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
        let adapter = self.adapter().ok_or_else(|| unsupported("file download"))?;
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
        let adapter = self
            .adapter()
            .ok_or_else(|| unsupported("text extraction"))?;
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

    pub(crate) async fn request(
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
        if self.adapter() == Some(Adapter::Anthropic) {
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
            let deadline = crate::runtime::Deadline::at(self.request_deadline).cap(request.timeout);
            deadline
                .run(authenticator.apply(&mut request, self.profile, self.credential))
                .await??;
            request.timeout = deadline.remaining()?;
        }
        Ok(request)
    }

    pub(crate) fn check_ref(&self, file: &ProviderFileRef) -> Result<(), LlmError> {
        if file.profile_name != self.profile.profile_name
            || file.provider_id != self.profile.provider_id
            || file.protocol != self.profile.protocol
            || file.endpoint_fingerprint
                != provider_file_endpoint_fingerprint(&self.endpoint_identity())
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

pub(crate) fn unknown_upload_outcome(
    operation: &'static str,
    source: LlmError,
    reference: Option<ProviderFileRef>,
) -> FileUploadError {
    FileUploadError::OutcomeUnknown {
        operation,
        source: Box::new(source),
        reference: reference.map(Box::new),
    }
}

/// Buffered uploads have no file ID if the transport fails after dispatch.
/// Keep deterministic local/provider validation failures in their original
/// class, while marking transport uncertainty so callers do not retry blind.
pub(crate) fn buffered_upload_error(operation: &str, error: LlmError) -> LlmError {
    match error {
        LlmError::Transport { .. }
        | LlmError::TransportTimeout { .. }
        | LlmError::StreamInterrupted { .. } => unknown_buffered_upload_outcome(operation, error),
        other => other,
    }
}

pub(crate) fn unknown_buffered_upload_outcome(operation: &str, error: LlmError) -> LlmError {
    LlmError::FileUploadOutcomeUnknown {
        message: format!("{operation}: {error}"),
    }
}

struct BatchStreamParts {
    prefix: Option<Bytes>,
    file: BoxStream<'static, Result<Bytes, LlmError>>,
    remaining: u64,
    file_finished: bool,
    suffix: Option<Bytes>,
}

fn stream_batch_body(
    prefix: Bytes,
    file: BoxStream<'static, Result<Bytes, LlmError>>,
    size_bytes: u64,
    suffix: Bytes,
) -> BoxStream<'static, Result<Bytes, LlmError>> {
    stream::try_unfold(
        BatchStreamParts {
            prefix: Some(prefix),
            file,
            remaining: size_bytes,
            file_finished: false,
            suffix: Some(suffix),
        },
        |mut state| async move {
            if let Some(prefix) = state.prefix.take() {
                return Ok(Some((prefix, state)));
            }
            if !state.file_finished {
                match state.file.next().await {
                    Some(Ok(chunk)) => {
                        if chunk.len() as u64 > state.remaining {
                            return Err(LlmError::InvalidRequest {
                                message: "batch upload stream exceeds declared size".into(),
                            });
                        }
                        state.remaining -= chunk.len() as u64;
                        return Ok(Some((chunk, state)));
                    }
                    Some(Err(error)) => return Err(error),
                    None => {
                        state.file_finished = true;
                        if state.remaining != 0 {
                            return Err(LlmError::InvalidRequest {
                                message: "batch upload stream is shorter than declared size".into(),
                            });
                        }
                    }
                }
            }
            Ok(state.suffix.take().map(|suffix| (suffix, state)))
        },
    )
    .boxed()
}
