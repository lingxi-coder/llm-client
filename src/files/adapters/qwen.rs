//! Provider-specific file workflow.
use crate::files::*;

impl FileService<'_> {
    pub(crate) async fn wait_for_qwen_file_ready(
        &self,
        file: &ProviderFileRef,
        timeout: Option<Duration>,
    ) -> Result<ProviderFileMetadata, LlmError> {
        if adapter(self.profile) != Some(Adapter::Qwen) {
            return Err(unsupported("Qwen file processing"));
        }
        if !valid_qwen_file_id(&file.file_id) {
            return Err(provider_shape("Qwen upload returned an invalid file ID"));
        }
        let deadline = Instant::now()
            + timeout
                .unwrap_or(QWEN_FILE_PROCESSING_TIMEOUT)
                .min(QWEN_FILE_PROCESSING_TIMEOUT);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(LlmError::TransportTimeout {
                    message: "Qwen file parsing timed out".into(),
                });
            }
            let mut request = self
                .request(
                    "GET",
                    file_url(self.profile, Adapter::Qwen, &file.file_id),
                    Bytes::new(),
                    None,
                )
                .await?;
            self.pace_qwen_metadata().await;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(LlmError::TransportTimeout {
                    message: "Qwen file parsing timed out".into(),
                });
            }
            request.timeout = Some(remaining.min(FILE_TIMEOUT));
            let response = crate::transport::HttpExecutor::new(self.http)
                .execute(request)
                .await?;
            let value = match adapter_json_success(Adapter::Qwen, &response, "file metadata") {
                Err(LlmError::RateLimited { .. }) => {
                    async_delay(QWEN_FILE_POLL_INTERVAL.min(remaining)).await;
                    continue;
                }
                result => result?,
            };
            let mut metadata = decode_metadata(self.profile, self.account_scope, &value)?;
            if metadata.file.file_id != file.file_id {
                return Err(provider_shape(
                    "Qwen file metadata returned another file ID",
                ));
            }
            let native_file = value.get("file").unwrap_or(&value);
            if metadata.file.filename.is_none() {
                metadata.file.filename.clone_from(&file.filename);
            }
            if metadata.file.media_type.is_none() {
                metadata.file.media_type.clone_from(&file.media_type);
            }
            if metadata.file.uri.is_none() {
                metadata.file.uri.clone_from(&file.uri);
            }
            if metadata.file.purpose.is_none() {
                metadata.file.purpose.clone_from(&file.purpose);
            }
            if metadata.file.size_bytes.is_none() {
                metadata.file.size_bytes = file.size_bytes;
            }
            if native_file.get("expires_at").is_none()
                && native_file.get("expirationTime").is_none()
            {
                metadata.file.expires_at.clone_from(&file.expires_at);
            }
            if native_file.get("status").is_none() && native_file.get("state").is_none() {
                metadata
                    .file
                    .processing_status
                    .clone_from(&file.processing_status);
                metadata.status.clone_from(&file.processing_status);
            }
            match metadata.status.as_deref() {
                Some("processed") => return Ok(metadata),
                Some("error") => return Err(provider_shape("Qwen file parsing failed")),
                Some("uploaded" | "processing") => {
                    async_delay(QWEN_FILE_POLL_INTERVAL.min(remaining)).await;
                }
                _ => return Err(provider_shape("Qwen file metadata has an unknown status")),
            }
        }
    }
}
