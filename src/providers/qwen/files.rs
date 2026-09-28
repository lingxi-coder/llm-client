//! Provider-owned file workflow and capability policy.
use crate::files::*;

impl FileService<'_> {
    pub(crate) async fn wait_for_qwen_file_ready(
        &self,
        file: &ProviderFileRef,
        timeout: Option<Duration>,
    ) -> Result<ProviderFileMetadata, LlmError> {
        if crate::files::adapter(self.profile) != Some(Adapter::Qwen) {
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

pub(crate) fn valid_qwen_file_id(file_id: &str) -> bool {
    !file_id.is_empty()
        && file_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

pub(crate) fn is_qwen_long_model(model: &str) -> bool {
    model.eq_ignore_ascii_case("qwen-long") || model.to_ascii_lowercase().starts_with("qwen-long-")
}

pub(crate) fn validate_qwen_long_inputs(
    request: &ChatRequest,
    resolved: &[(usize, usize)],
) -> Result<(), LlmError> {
    validate_qwen_long_blocks(request.messages.iter().enumerate().flat_map(|(mi, m)| {
        m.content
            .iter()
            .enumerate()
            .filter_map(move |(bi, b)| (!resolved.contains(&(mi, bi))).then_some(b))
    }))
}

pub(crate) fn validate_qwen_long_blocks<'a>(
    blocks: impl IntoIterator<Item = &'a ContentBlock>,
) -> Result<(), LlmError> {
    for block in blocks {
        if matches!(
            block,
            ContentBlock::Image {
                source: ImageSource::Base64 { .. } | ImageSource::Url { .. }
            } | ContentBlock::Document {
                source: DocumentSource::Base64 { .. }
                    | DocumentSource::Text { .. }
                    | DocumentSource::Url { .. },
                ..
            }
        ) {
            return Err(LlmError::UnsupportedCapability{message:"Qwen-Long image and document inputs require an app attachment or a Qwen provider file reference; inline data and URLs are unsupported".into()});
        }
    }
    Ok(())
}

pub(crate) fn qwen_long_region_supported(profile: &ProviderProfile) -> bool {
    url::Url::parse(&profile.base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| {
            host == "dashscope.aliyuncs.com"
                || host
                    .strip_suffix(".cn-beijing.maas.aliyuncs.com")
                    .is_some_and(valid_workspace_id)
        })
}

pub(crate) fn is_qwen_files_host(host: &str) -> bool {
    if matches!(
        host,
        "dashscope.aliyuncs.com" | "dashscope-intl.aliyuncs.com"
    ) {
        return true;
    }
    [
        ".cn-beijing.maas.aliyuncs.com",
        ".ap-southeast-1.maas.aliyuncs.com",
    ]
    .iter()
    .find_map(|suffix| host.strip_suffix(suffix))
    .is_some_and(valid_workspace_id)
}

pub(crate) fn valid_workspace_id(workspace: &str) -> bool {
    !workspace.is_empty()
        && workspace
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

pub(crate) fn is_qwen_file_type(media_type: &str) -> bool {
    media_type.starts_with("text/")
        || matches!(
            media_type,
            "application/pdf"
                | "application/json"
                | "application/epub+zip"
                | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                | "application/vnd.oasis.opendocument.text"
                | "application/msword"
                | "application/vnd.ms-excel"
                | "image/bmp"
                | "image/png"
                | "image/jpeg"
                | "image/gif"
        )
}

pub(crate) fn capabilities() -> FileCapabilities {
    FileCapabilities {
        upload: true,
        retrieve_metadata: true,
        list: true,
        delete: true,
        download: DownloadSupport::Unsupported,
        extract_text: false,
        model_input: ModelFileReference::Unsupported,
        max_upload_bytes: Some(150_000_000),
        retention: None,
    }
}

/// Pace automatic file operations for one configured Qwen connection. The
/// upload and metadata/delete buckets have separate documented QPS limits.
pub(crate) struct QwenFileRateLimiter {
    next_upload: futures::lock::Mutex<Instant>,
    next_metadata: futures::lock::Mutex<Instant>,
}

impl QwenFileRateLimiter {
    pub(crate) fn new() -> Self {
        let now = Instant::now();
        Self {
            next_upload: futures::lock::Mutex::new(now),
            next_metadata: futures::lock::Mutex::new(now),
        }
    }

    pub(crate) async fn wait_upload(&self) {
        Self::wait(&self.next_upload, QWEN_UPLOAD_INTERVAL).await;
    }

    pub(crate) async fn wait_metadata(&self) {
        Self::wait(&self.next_metadata, QWEN_METADATA_INTERVAL).await;
    }

    async fn wait(next: &futures::lock::Mutex<Instant>, interval: Duration) {
        let mut next = next.lock().await;
        let delay = next.saturating_duration_since(Instant::now());
        if !delay.is_zero() {
            async_delay(delay).await;
        }
        *next = Instant::now() + interval;
    }
}

impl FileService<'_> {
    pub(crate) fn with_qwen_rate_limiter(mut self, limiter: Arc<QwenFileRateLimiter>) -> Self {
        self.qwen_rate_limiter = Some(limiter);
        self
    }
    pub(crate) async fn pace_qwen_upload(&self) {
        if let Some(limiter) = &self.qwen_rate_limiter {
            limiter.wait_upload().await;
        }
    }
    pub(crate) async fn pace_qwen_metadata(&self) {
        if let Some(limiter) = &self.qwen_rate_limiter {
            limiter.wait_metadata().await;
        }
    }
}

pub(crate) const QWEN_FILE_POLL_INTERVAL: Duration = Duration::from_secs(1);

pub(crate) const QWEN_FILE_PROCESSING_TIMEOUT: Duration = Duration::from_secs(10 * 60);

pub(crate) const QWEN_CLEANUP_DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

pub(crate) const QWEN_UPLOAD_INTERVAL: Duration = Duration::from_millis(350);

pub(crate) const QWEN_METADATA_INTERVAL: Duration = Duration::from_millis(110);

pub(crate) const QWEN_IMAGE_MAX_UPLOAD_BYTES: u64 = 20_000_000;

pub(crate) const QWEN_LONG_MAX_FILE_REFERENCES: usize = 100;

pub(crate) fn purpose_name(purpose: FilePurpose) -> Option<&'static str> {
    match purpose {
        FilePurpose::Extraction | FilePurpose::ModelInput => Some("file-extract"),
        _ => None,
    }
}

pub(crate) fn purpose_capabilities(purpose: FilePurpose, _media_type: &str) -> FileCapabilities {
    if matches!(purpose, FilePurpose::ModelInput | FilePurpose::Extraction) {
        capabilities()
    } else {
        FileCapabilities::unsupported()
    }
}

pub(crate) fn model_reference(
    profile: &ProviderProfile,
    _model_profile: &ModelProfile,
    model: &str,
    media_type: &str,
) -> ModelFileReference {
    if qwen_long_region_supported(profile)
        && profile.protocol == ProtocolFamily::OpenAiChat
        && is_qwen_long_model(model)
        && is_qwen_file_type(media_type)
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
    if result.upload && media_type.to_ascii_lowercase().starts_with("image/") {
        result.max_upload_bytes = Some(QWEN_IMAGE_MAX_UPLOAD_BYTES);
    }
    result
}

pub(crate) fn metadata_uri(file_id: &str, purpose: Option<&str>) -> Option<String> {
    (purpose == Some("file-extract")).then(|| format!("fileid://{file_id}"))
}

pub(crate) fn adapter(_profile: &ProviderProfile, host: &str) -> Option<Adapter> {
    (is_qwen_files_host(host)).then_some(Adapter::Qwen)
}
