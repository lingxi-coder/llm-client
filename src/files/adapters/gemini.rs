//! Provider-specific file workflow.
use crate::files::*;

impl FileService<'_> {
    pub async fn resume_gemini_processing(
        &self,
        file: &ProviderFileSource,
    ) -> Result<ProviderFileRef, LlmError> {
        if adapter(self.profile) != Some(Adapter::Gemini) {
            return Err(unsupported("Gemini file processing"));
        }
        if self
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
            expires_at: None,
            downloadable: None,
            purpose: file.purpose.clone(),
        };
        self.check_ref(&pending)?;
        let timeout = self
            .gemini_processing_timeout
            .unwrap_or(DEFAULT_GEMINI_PROCESSING_TIMEOUT);
        let deadline =
            Instant::now()
                .checked_add(timeout)
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "Gemini processing timeout is too large".into(),
                })?;
        self.wait_for_gemini_file_active(&pending, deadline, timeout)
            .await
    }

    pub(in crate::files) async fn upload_gemini(
        &self,
        file: &UploadFile,
    ) -> Result<ProviderFileRef, LlmError> {
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
            .gemini_request_deadline
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
        let start = crate::transport::HttpExecutor::new(self.http)
            .execute(req)
            .await?;
        status_result(&start, "Gemini file upload start")?;
        let upload_url = start
            .header("x-goog-upload-url")
            .ok_or_else(|| provider_shape("Gemini upload start omitted x-goog-upload-url"))?;
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
        let response = crate::transport::HttpExecutor::new(self.http)
            .execute(upload_req)
            .await?;
        let value = json_success(&response, "Gemini file upload")?;
        let file_value = value.get("file").unwrap_or(&value);
        let mut metadata = decode_metadata(self.profile, self.account_scope, file_value)?;
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
                    .gemini_request_deadline
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

    pub(in crate::files) async fn wait_for_gemini_file_active(
        &self,
        pending: &ProviderFileRef,
        deadline: Instant,
        processing_timeout: Duration,
    ) -> Result<ProviderFileRef, LlmError> {
        loop {
            let mut request = gemini_processing_call(
                self.request(
                    "GET",
                    file_url(self.profile, Adapter::Gemini, &pending.file_id),
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
                crate::transport::HttpExecutor::new(self.http).execute(request),
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
