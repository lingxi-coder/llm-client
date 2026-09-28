//! Provider-owned file workflow and capability policy.
use crate::files::*;

impl FileService<'_> {
    pub async fn resume_gemini_processing(
        &self,
        file: &ProviderFileSource,
    ) -> Result<ProviderFileRef, LlmError> {
        let service = self.scoped_operation();
        if crate::files::adapter(service.profile) != Some(Adapter::Gemini) {
            return Err(unsupported("Gemini file processing"));
        }
        if service
            .account_scope
            .is_none_or(|scope| scope.trim().is_empty())
            || file.file_id.trim().is_empty()
        {
            return Err(LlmError::InvalidRequest {
                message: "Gemini processing resume requires an account scope and file ID".into(),
            });
        }
        let pending = ProviderFileRef {
            provider_id: file.provider_id.clone(),
            profile_name: file.profile_name.clone(),
            endpoint_fingerprint: file.endpoint_fingerprint.clone(),
            account_scope: file.account_scope.clone(),
            protocol: file.protocol,
            file_id: file.file_id.clone(),
            uri: file.uri.clone(),
            filename: None,
            media_type: file.media_type.clone(),
            size_bytes: None,
            expires_at: file.expires_at.clone(),
            processing_status: file.processing_status.clone(),
            downloadable: None,
            purpose: file.purpose.clone(),
        };
        service.check_ref(&pending)?;
        if pending.processing_status.as_deref() == Some("FAILED") {
            return Err(LlmError::InvalidRequest {
                message: "Gemini file processing failed and cannot be resumed".into(),
            });
        }
        let timeout = service
            .gemini_processing_timeout
            .unwrap_or(DEFAULT_GEMINI_PROCESSING_TIMEOUT);
        let deadline =
            Instant::now()
                .checked_add(timeout)
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "Gemini processing timeout is too large".into(),
                })?;
        let deadline = service
            .request_deadline
            .map_or(deadline, |limit| deadline.min(limit));
        service
            .wait_for_gemini_file_active(&pending, deadline, timeout)
            .await
    }

    pub(crate) async fn upload_gemini(
        &self,
        file: &UploadFile,
    ) -> Result<ProviderFileRef, LlmError> {
        let (value, started, processing_timeout, is_video) =
            self.upload_gemini_response(file).await?;
        let file_value = value.get("file").unwrap_or(&value);
        let mut metadata =
            decode_metadata(self.profile, self.account_scope, file_value).map_err(|error| {
                unknown_buffered_upload_outcome("Gemini file upload response", error)
            })?;
        metadata.file.media_type = Some(file.media_type.clone());
        metadata.file.filename = Some(file.filename.clone());
        match file_value.get("state").and_then(Value::as_str) {
            Some("FAILED") => {
                return Err(provider_shape("Gemini file processing failed"));
            }
            Some("PROCESSING") => {
                let processing_deadline = Instant::now()
                    .checked_add(processing_timeout)
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: "Gemini processing timeout is too large".into(),
                    })?;
                let processing_deadline = if is_video {
                    processing_deadline.min(started + GEMINI_VIDEO_FILE_TIMEOUT)
                } else {
                    processing_deadline
                };
                let processing_deadline = self
                    .request_deadline
                    .map_or(processing_deadline, |deadline| {
                        processing_deadline.min(deadline)
                    });
                let effective_processing_timeout =
                    processing_deadline.saturating_duration_since(Instant::now());
                metadata.file = self
                    .wait_for_gemini_file_active(
                        &metadata.file,
                        processing_deadline,
                        effective_processing_timeout,
                    )
                    .await?;
                metadata.file.media_type = Some(file.media_type.clone());
                metadata.file.filename = Some(file.filename.clone());
            }
            Some("ACTIVE") | None => {} // Older APIs and transports may omit the state.
            Some(_) => {
                return Err(gemini_processing_unresolved(
                    &metadata.file,
                    provider_shape("Gemini file upload returned an unknown processing state"),
                ));
            }
        }
        if metadata.file.uri.as_deref().is_none_or(str::is_empty) {
            return Err(gemini_processing_unresolved(
                &metadata.file,
                provider_shape("Gemini file upload omitted the model-input URI"),
            ));
        }
        Ok(metadata.file)
    }

    async fn upload_gemini_response(
        &self,
        file: &UploadFile,
    ) -> Result<(Value, Instant, Duration, bool), LlmError> {
        let started = Instant::now();
        let is_video = file.media_type.to_ascii_lowercase().starts_with("video/");
        let upload_timeout = self.gemini_upload_timeout.unwrap_or(if is_video {
            GEMINI_VIDEO_FILE_TIMEOUT
        } else {
            FILE_TIMEOUT
        });
        let upload_timeout = if is_video {
            upload_timeout.min(GEMINI_VIDEO_FILE_TIMEOUT)
        } else {
            upload_timeout
        };
        let processing_timeout = self
            .gemini_processing_timeout
            .unwrap_or(DEFAULT_GEMINI_PROCESSING_TIMEOUT);
        let processing_timeout = if is_video {
            processing_timeout.min(GEMINI_VIDEO_FILE_TIMEOUT)
        } else {
            processing_timeout
        };
        let upload_deadline =
            started
                .checked_add(upload_timeout)
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "Gemini upload timeout is too large".into(),
                })?;
        let upload_deadline = self
            .request_deadline
            .map_or(upload_deadline, |deadline| upload_deadline.min(deadline));
        let start_url = gemini_upload_url(self.profile);
        let body = serde_json::json!({"file": {"display_name": file.filename}});
        let mut req = crate::runtime::Deadline::at(Some(upload_deadline))
            .run(
                self.request(
                    "POST",
                    start_url,
                    Bytes::from(
                        serde_json::to_vec(&body)
                            .map_err(|error| provider_shape(&error.to_string()))?,
                    ),
                    Some("application/json".to_owned()),
                ),
            )
            .await??;
        req.timeout = Some(gemini_timeout_remaining(
            upload_deadline,
            upload_timeout,
            "upload",
        )?);
        req.headers.extend([
            ("x-goog-upload-protocol".to_owned(), "resumable".to_owned()),
            ("x-goog-upload-command".to_owned(), "start".to_owned()),
            (
                "x-goog-upload-header-content-length".to_owned(),
                file.bytes.len().to_string(),
            ),
            (
                "x-goog-upload-header-content-type".to_owned(),
                file.media_type.clone(),
            ),
        ]);
        let start =
            self.executor().execute(req).await.map_err(|error| {
                buffered_upload_error("Gemini resumable upload initiation", error)
            })?;
        status_result(&start, "Gemini file upload start")?;
        let upload_url =
            start
                .header("x-goog-upload-url")
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "Gemini upload start response missing x-goog-upload-url header".into(),
                })?;
        if !same_origin(upload_url, &self.profile.base_url) {
            return Err(LlmError::PermissionDenied {
                message: "Gemini returned an upload URL outside the configured API origin".into(),
            });
        }
        // The resumable upload URL is a temporary bearer capability; Google's
        // documented second request does not send the API key again.
        let upload_req = HttpRequest {
            method: "POST".into(),
            url: upload_url.to_owned(),
            headers: vec![
                ("content-length".into(), file.bytes.len().to_string()),
                ("x-goog-upload-offset".into(), "0".into()),
                ("x-goog-upload-command".into(), "upload, finalize".into()),
            ],
            body: file.bytes.clone(),
            timeout: Some(gemini_timeout_remaining(
                upload_deadline,
                upload_timeout,
                "upload",
            )?),
        };
        let response = self
            .executor()
            .execute(upload_req)
            .await
            .map_err(|error| buffered_upload_error("Gemini resumable upload body", error))?;
        if !(200..300).contains(&response.status) {
            let error = status_error(
                response.status,
                &String::from_utf8_lossy(&response.body),
                "Gemini file upload",
            );
            return Err(if response.status >= 500 {
                unknown_buffered_upload_outcome("Gemini resumable upload response", error)
            } else {
                error
            });
        }
        let value = serde_json::from_slice(&response.body).map_err(|error| {
            unknown_buffered_upload_outcome(
                "Gemini resumable upload response",
                provider_shape(&format!(
                    "Gemini file upload response was not JSON: {error}"
                )),
            )
        })?;
        Ok((value, started, processing_timeout, is_video))
    }

    /// Upload without waiting for processing. FAILED/PROCESSING remain data;
    /// applications may use `poll_gemini_active` separately.
    pub async fn upload_gemini_unpolled(
        &self,
        file: &UploadFile,
    ) -> Result<crate::providers::google::files_wire::GeminiFile, LlmError> {
        let service = self.scoped_operation();
        if service.profile.protocol != ProtocolFamily::GeminiGenerateContent {
            return Err(LlmError::InvalidRequest {
                message: "file upload requires a gemini provider profile".into(),
            });
        }
        let (value, _, _, _) = service.upload_gemini_response(file).await?;
        crate::providers::google::files_wire::parse_upload_response(&value)
            .map_err(|error| unknown_buffered_upload_outcome("Gemini file upload response", error))
    }

    /// Explicit readiness polling, using the caller's cadence and total budget.
    pub async fn poll_gemini_active(
        &self,
        name: &str,
        interval: Duration,
        max_wait: Duration,
    ) -> Result<crate::providers::google::files_wire::GeminiFile, LlmError> {
        let service = self.scoped_operation();
        if service.profile.protocol != ProtocolFamily::GeminiGenerateContent {
            return Err(LlmError::InvalidRequest {
                message: "file upload requires a gemini provider profile".into(),
            });
        }
        let deadline = tokio::time::Instant::now()
            .checked_add(max_wait)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "file processing timeout too large".into(),
            })?;
        loop {
            let mut request = crate::providers::google::files_wire::file_status_request(
                &service.profile.base_url,
                name,
            );
            if let Some(auth) = service.authenticator {
                service
                    .deadline()
                    .run(auth.apply(&mut request, service.profile, service.credential))
                    .await??;
            }
            // max_wait bounds poll scheduling; allow a poll exactly at the
            // deadline, matching explicit split-phase lifecycle semantics.
            request.timeout = Some(FILE_TIMEOUT);
            let response = service.executor().execute(request).await?;
            status_result(&response, "Gemini file status")?;
            let value: Value = serde_json::from_slice(&response.body)
                .map_err(|e| provider_shape(&e.to_string()))?;
            let file = crate::providers::google::files_wire::parse_file_status(&value)?;
            match file.state.as_str() {
                "ACTIVE" => return Ok(file),
                "FAILED" => {
                    return Err(LlmError::InvalidRequest {
                        message: format!("gemini file processing failed: {name}"),
                    })
                }
                _ => {}
            }
            if tokio::time::Instant::now() + interval > deadline {
                return Err(LlmError::Transport {
                    message: format!(
                        "gemini file did not become ACTIVE within {}s",
                        max_wait.as_secs()
                    ),
                });
            }
            service.deadline().run(async_delay(interval)).await?;
        }
    }

    pub(crate) async fn wait_for_gemini_file_active(
        &self,
        pending: &ProviderFileRef,
        deadline: Instant,
        processing_timeout: Duration,
    ) -> Result<ProviderFileRef, LlmError> {
        loop {
            let mut request = gemini_processing_call(
                self.request(
                    "GET",
                    file_url(self.profile, &pending.file_id),
                    Bytes::new(),
                    None,
                ),
                deadline,
                processing_timeout,
                pending,
            )
            .await?;
            request.timeout = Some(
                gemini_processing_remaining(deadline, processing_timeout, pending)?
                    .min(FILE_TIMEOUT),
            );
            let response = gemini_processing_call(
                self.executor().execute(request),
                deadline,
                processing_timeout,
                pending,
            )
            .await?;
            let value = json_success(&response, "Gemini file processing status")
                .map_err(|error| gemini_processing_unresolved(pending, error))?;
            let file_value = value.get("file").unwrap_or(&value);
            match file_value.get("state").and_then(Value::as_str) {
                Some("ACTIVE") => {
                    let mut metadata =
                        decode_metadata(self.profile, self.account_scope, file_value)
                            .map_err(|error| gemini_processing_unresolved(pending, error))?
                            .file;
                    if metadata.file_id != pending.file_id {
                        return Err(gemini_processing_unresolved(
                            pending,
                            provider_shape("Gemini file status returned another file ID"),
                        ));
                    }
                    if metadata.uri.is_none() {
                        metadata.uri.clone_from(&pending.uri);
                    }
                    if metadata.media_type.is_none() {
                        metadata.media_type.clone_from(&pending.media_type);
                    }
                    if metadata.filename.is_none() {
                        metadata.filename.clone_from(&pending.filename);
                    }
                    if file_value.get("expirationTime").is_none()
                        && file_value.get("expires_at").is_none()
                    {
                        metadata.expires_at.clone_from(&pending.expires_at);
                    }
                    if metadata.uri.as_deref().is_none_or(str::is_empty) {
                        return Err(gemini_processing_unresolved(
                            pending,
                            provider_shape("Gemini active file omitted the model-input URI"),
                        ));
                    }
                    return Ok(metadata);
                }
                Some("FAILED") => {
                    return Err(provider_shape("Gemini file processing failed"));
                }
                Some("PROCESSING") => {
                    let remaining =
                        gemini_processing_remaining(deadline, processing_timeout, pending)?;
                    async_delay(GEMINI_FILE_POLL_INTERVAL.min(remaining)).await;
                }
                _ => {
                    return Err(gemini_processing_unresolved(
                        pending,
                        provider_shape("Gemini file processing returned an unknown state"),
                    ));
                }
            }
        }
    }
}

pub(crate) fn is_gemini_video_type(media_type: &str) -> bool {
    let media_type = media_type.to_ascii_lowercase();
    matches!(
        media_type.as_str(),
        "video/mp4"
            | "video/mpeg"
            | "video/mov"
            | "video/quicktime"
            | "video/avi"
            | "video/x-flv"
            | "video/mpg"
            | "video/webm"
            | "video/wmv"
            | "video/3gpp"
    )
}

pub(crate) fn is_gemini_audio_type(media_type: &str) -> bool {
    matches!(
        media_type,
        "audio/wav"
            | "audio/mp3"
            | "audio/aiff"
            | "audio/aac"
            | "audio/ogg"
            | "audio/flac"
            | "audio/mpeg"
            | "audio/m4a"
            | "audio/l16"
            | "audio/opus"
            | "audio/alaw"
            | "audio/mulaw"
            | "audio/webm"
    )
}

pub(crate) fn is_gemini_document_type(media_type: &str) -> bool {
    media_type.starts_with("text/")
        || matches!(
            media_type,
            "application/pdf" | "application/json" | "application/rtf"
        )
}

pub(crate) fn model_declares_gemini_document_input(model: &ModelProfile) -> bool {
    if model.metadata.input_modalities.is_empty() {
        return model_declares_file_input(model);
    }
    model.metadata.input_modalities.iter().any(|modality| {
        matches!(
            modality.to_ascii_lowercase().as_str(),
            "file" | "files" | "document" | "pdf"
        )
    })
}

pub(crate) fn capabilities() -> FileCapabilities {
    FileCapabilities {
        upload: true,
        retrieve_metadata: true,
        list: true,
        delete: true,
        download: DownloadSupport::GeneratedFilesOnly,
        extract_text: false,
        model_input: ModelFileReference::Unsupported,
        max_upload_bytes: Some(2 * 1024 * 1024 * 1024),
        retention: Some(Duration::from_secs(48 * 60 * 60)),
    }
}

pub(crate) fn gemini_upload_url(profile: &ProviderProfile) -> String {
    let base = profile.base_url.trim_end_matches('/');
    if base.ends_with("/v1beta") {
        format!("{}/upload/v1beta/files", base.trim_end_matches("/v1beta"))
    } else {
        format!("{base}/upload/v1beta/files")
    }
}

pub(crate) fn gemini_timeout_remaining(
    deadline: Instant,
    timeout: Duration,
    phase: &str,
) -> Result<Duration, LlmError> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        Err(LlmError::TransportTimeout {
            message: format!("Gemini file {phase} timed out after {timeout:?}"),
        })
    } else {
        Ok(remaining)
    }
}

pub(crate) fn gemini_processing_remaining(
    deadline: Instant,
    timeout: Duration,
    pending: &ProviderFileRef,
) -> Result<Duration, LlmError> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        Err(LlmError::ProviderFileProcessing {
            message: format!("Gemini file remains PROCESSING after {timeout:?}"),
            file: Box::new(pending.model_reference()),
        })
    } else {
        Ok(remaining)
    }
}

pub(crate) async fn gemini_processing_call<T>(
    operation: impl Future<Output = Result<T, LlmError>>,
    deadline: Instant,
    timeout: Duration,
    pending: &ProviderFileRef,
) -> Result<T, LlmError> {
    let remaining = gemini_processing_remaining(deadline, timeout, pending)?;
    match futures::future::select(Box::pin(operation), Box::pin(async_delay(remaining))).await {
        futures::future::Either::Left((result, _)) => {
            result.map_err(|error| gemini_processing_unresolved(pending, error))
        }
        futures::future::Either::Right((_, _)) => Err(LlmError::ProviderFileProcessing {
            message: format!("Gemini file processing timed out after {timeout:?}"),
            file: Box::new(pending.model_reference()),
        }),
    }
}

pub(crate) fn gemini_processing_unresolved(pending: &ProviderFileRef, error: LlmError) -> LlmError {
    LlmError::ProviderFileProcessing {
        message: error.to_string(),
        file: Box::new(pending.model_reference()),
    }
}

impl FileService<'_> {
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

    pub(crate) async fn upload_gemini_stream(
        &self,
        filename: String,
        media_type: String,
        size_bytes: u64,
        body: BoxStream<'static, Result<Bytes, LlmError>>,
        timeout: Duration,
    ) -> Result<ProviderFileRef, FileUploadError> {
        let started = Instant::now();
        let upload_deadline =
            started
                .checked_add(timeout)
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "Gemini upload timeout is too large".into(),
                })?;
        let upload_deadline = self
            .request_deadline
            .map_or(upload_deadline, |deadline| upload_deadline.min(deadline));
        let start_url = gemini_upload_url(self.profile);
        let start_body = serde_json::json!({"file": {"display_name": filename}});
        let mut start_request = crate::runtime::Deadline::at(Some(upload_deadline))
            .run(
                self.request(
                    "POST",
                    start_url,
                    Bytes::from(
                        serde_json::to_vec(&start_body)
                            .map_err(|error| provider_shape(&error.to_string()))?,
                    ),
                    Some("application/json".into()),
                ),
            )
            .await??;
        start_request.timeout = Some(gemini_timeout_remaining(
            upload_deadline,
            timeout,
            "upload",
        )?);
        start_request.headers.extend([
            ("x-goog-upload-protocol".into(), "resumable".into()),
            ("x-goog-upload-command".into(), "start".into()),
            (
                "x-goog-upload-header-content-length".into(),
                size_bytes.to_string(),
            ),
            (
                "x-goog-upload-header-content-type".into(),
                media_type.clone(),
            ),
        ]);
        let start = self
            .executor()
            .execute(start_request)
            .await
            .map_err(|source| {
                unknown_upload_outcome("Gemini resumable upload initiation", source, None)
            })?;
        status_result(&start, "Gemini file upload start").map_err(|source| {
            if start.status >= 500 {
                unknown_upload_outcome("Gemini resumable upload initiation", source, None)
            } else {
                FileUploadError::Llm(source)
            }
        })?;
        let upload_url = start.header("x-goog-upload-url").ok_or_else(|| {
            unknown_upload_outcome(
                "Gemini resumable upload initiation",
                provider_shape("Gemini upload start omitted x-goog-upload-url"),
                None,
            )
        })?;
        if !same_origin(upload_url, &self.profile.base_url) {
            return Err(LlmError::PermissionDenied {
                message: "Gemini returned an upload URL outside the configured API origin".into(),
            }
            .into());
        }
        let request = crate::transport::HttpStreamRequest {
            method: "POST".into(),
            url: upload_url.to_owned(),
            headers: vec![
                ("content-length".into(), size_bytes.to_string()),
                ("x-goog-upload-offset".into(), "0".into()),
                ("x-goog-upload-command".into(), "upload, finalize".into()),
            ],
            body: exact_upload_stream(body, size_bytes),
            content_length: size_bytes,
            timeout: Some(
                gemini_timeout_remaining(upload_deadline, timeout, "upload").map_err(|source| {
                    unknown_upload_outcome("Gemini resumable upload body", source, None)
                })?,
            ),
        };
        let response = self
            .executor()
            .with_deadline(crate::runtime::Deadline::at(Some(upload_deadline)))
            .send_stream(request)
            .await
            .map_err(|source| {
                unknown_upload_outcome("Gemini resumable upload body", source, None)
            })?;
        let response =
            crate::transport::HttpExecutor::collect_response(response, Some(1024 * 1024))
                .await
                .map_err(|source| {
                    unknown_upload_outcome("Gemini resumable upload response", source, None)
                })?;
        if !(200..300).contains(&response.status) {
            let source = status_error(
                response.status,
                &String::from_utf8_lossy(&response.body),
                "Gemini file upload",
            );
            return Err(if response.status >= 500 {
                unknown_upload_outcome("Gemini resumable upload", source, None)
            } else {
                FileUploadError::Llm(source)
            });
        }
        let value: Value = serde_json::from_slice(&response.body).map_err(|error| {
            unknown_upload_outcome(
                "Gemini file upload response",
                provider_shape(&format!(
                    "Gemini file upload response was not JSON: {error}"
                )),
                None,
            )
        })?;
        let file_value = value.get("file").unwrap_or(&value);
        let mut metadata = self.decode_metadata(file_value).map_err(|source| {
            unknown_upload_outcome("Gemini file upload response", source, None)
        })?;
        metadata.file.media_type = Some(media_type.clone());
        metadata.file.filename = Some(filename);
        metadata.file.size_bytes.get_or_insert(size_bytes);
        match file_value.get("state").and_then(Value::as_str) {
            Some("FAILED") => {
                return Err(FileUploadError::Llm(provider_shape(
                    "Gemini file processing failed",
                )));
            }
            Some("PROCESSING") => {
                return Err(FileUploadError::Llm(LlmError::ProviderFileProcessing {
                    message: "Gemini file is still processing; resume it explicitly".into(),
                    file: Box::new(metadata.file.model_reference()),
                }));
            }
            Some("ACTIVE") | None => {}
            Some(_) => {
                return Err(FileUploadError::Llm(gemini_processing_unresolved(
                    &metadata.file,
                    provider_shape("Gemini file upload returned an unknown processing state"),
                )));
            }
        }
        if metadata.file.uri.as_deref().is_none_or(str::is_empty) {
            return Err(FileUploadError::Llm(gemini_processing_unresolved(
                &metadata.file,
                provider_shape("Gemini file upload omitted the model-input URI"),
            )));
        }
        Ok(metadata.file)
    }
}

pub(crate) const GEMINI_FILE_POLL_INTERVAL: Duration = Duration::from_secs(1);

pub(crate) const DEFAULT_GEMINI_PROCESSING_TIMEOUT: Duration = Duration::from_secs(10 * 60);

pub(crate) const GEMINI_VIDEO_FILE_TIMEOUT: Duration = Duration::from_secs(2 * 60 * 60);

pub(crate) const GEMINI_FILE_RETENTION: Duration = Duration::from_secs(48 * 60 * 60);

pub(crate) const GEMINI_PDF_MAX_UPLOAD_BYTES: u64 = 50_000_000;

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
    let supports_modality = |name: &str| {
        model_profile
            .metadata
            .input_modalities
            .iter()
            .any(|m| m.eq_ignore_ascii_case(name))
    };
    if profile.protocol == ProtocolFamily::GeminiGenerateContent
        && ((matches!(
            media_type,
            "image/jpeg" | "image/png" | "image/webp" | "image/heic" | "image/heif"
        ) && model_declares_image_input(model_profile))
            || (is_gemini_video_type(media_type) && supports_modality("video"))
            || (is_gemini_audio_type(media_type) && supports_modality("audio"))
            || (is_gemini_document_type(media_type)
                && model_declares_gemini_document_input(model_profile)))
    {
        ModelFileReference::FileUri
    } else {
        ModelFileReference::Unsupported
    }
}

pub(crate) fn capabilities_for_media_type(
    mut result: FileCapabilities,
    media_type: &str,
) -> FileCapabilities {
    if result.upload && media_type.eq_ignore_ascii_case("application/pdf") {
        result.max_upload_bytes = Some(
            result
                .max_upload_bytes
                .map_or(GEMINI_PDF_MAX_UPLOAD_BYTES, |limit| {
                    limit.min(GEMINI_PDF_MAX_UPLOAD_BYTES)
                }),
        );
    }
    result
}

pub(crate) fn files_url(profile: &ProviderProfile) -> String {
    let base = profile.base_url.trim_end_matches('/');
    if base.ends_with("/v1beta") {
        format!("{base}/files")
    } else {
        format!("{base}/v1beta/files")
    }
}

pub(crate) fn file_url(profile: &ProviderProfile, file_id: &str) -> String {
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
    format!("{base}/{}", encoded_path(&resource_name))
}

pub(crate) fn content_url(profile: &ProviderProfile, file_id: &str) -> String {
    format!("{}:download?alt=media", file_url(profile, file_id))
}

pub(crate) fn metadata_downloadable(value: &Value) -> Option<bool> {
    // The canonical API download endpoint is used; metadata URLs are never an authentication authority.
    Some(
        value.get("source").and_then(Value::as_str) == Some("GENERATED")
            && value
                .get("downloadUri")
                .and_then(nonempty_string)
                .is_some_and(|uri| valid_download_uri(&uri)),
    )
}

pub(crate) fn adapter(profile: &ProviderProfile, host: &str) -> Option<Adapter> {
    (host == "generativelanguage.googleapis.com"
        && profile.protocol == ProtocolFamily::GeminiGenerateContent)
        .then_some(Adapter::Gemini)
}
