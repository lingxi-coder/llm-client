//! File operations bound to one provider profile and one operation snapshot.
use super::binding::ProviderBinding;
use crate::{
    client::RequestOptions,
    files::{
        FileCapabilities, FilePurpose, FileService, FileUploadError, ProviderFileContent,
        ProviderFileMetadata, ProviderFilePage, ProviderFileRef, UploadFile, UploadFileStream,
    },
    protocol::{FoundryHosting, LlmError, ProviderFileSource},
    runtime::ClientSnapshot,
};
use std::{marker::PhantomData, time::Duration};

/// A credential-free file resource. Each operation captures one configuration revision.
/// Provider-specific operations are available only on the corresponding client resource.
#[derive(Clone)]
pub struct ProviderFiles<'a, P> {
    binding: &'a ProviderBinding,
    hosting: Option<FoundryHosting>,
    upload_timeout: Option<Duration>,
    processing_timeout: Option<Duration>,
    provider: PhantomData<fn() -> P>,
}
impl<'a, P> ProviderFiles<'a, P> {
    fn new(binding: &'a ProviderBinding, hosting: Option<FoundryHosting>) -> Self {
        Self {
            binding,
            hosting,
            upload_timeout: None,
            processing_timeout: None,
            provider: PhantomData,
        }
    }

    fn service<'s>(
        &self,
        snapshot: &'s ClientSnapshot,
        options: &'s RequestOptions,
    ) -> Result<FileService<'s>, LlmError> {
        let profile = snapshot
            .profile(self.binding.profile_name())
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "file profile is unavailable".into(),
            })?;
        let scope = options
            .file_account_scope
            .as_deref()
            .or(options.account_scope.as_deref())
            .filter(|scope| !scope.trim().is_empty())
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "file operations require an explicit nonempty account scope".into(),
            })?;
        let authenticator = options
            .authenticator
            .as_ref()
            .map(|value| value.0.as_ref())
            .or_else(|| {
                snapshot
                    .runtime
                    .authenticators
                    .get(&profile.auth)
                    .map(|value| value.as_ref())
            });
        let mut service = match self.hosting {
            Some(hosting) => FileService::new_foundry(
                snapshot.runtime.http.as_ref(),
                profile,
                hosting,
                authenticator,
                options.credential.as_ref(),
                scope,
            )?,
            None => FileService::new(
                snapshot.runtime.http.as_ref(),
                profile,
                authenticator,
                options.credential.as_ref(),
                Some(scope),
            ),
        };
        if let Some(timeout) = self.upload_timeout {
            service = service.with_gemini_upload_timeout(timeout);
        }
        if let Some(timeout) = self.processing_timeout {
            service = service.with_gemini_processing_timeout(timeout);
        }
        if let Some(timeout) = options.total_timeout {
            // A zero RequestOptions budget denotes an already expired operation.
            if timeout.is_zero() {
                return Err(crate::runtime::timeout_error());
            }
            service = service.with_timeout(timeout)?;
        }
        Ok(service)
    }

    pub fn capabilities(
        &self,
        model: &str,
        media_type: &str,
        purpose: FilePurpose,
        options: &RequestOptions,
    ) -> Result<FileCapabilities, LlmError> {
        let snapshot = self.binding.pin()?;
        Ok(self
            .service(&snapshot, options)?
            .capabilities_for_purpose(model, media_type, purpose))
    }

    pub async fn upload(
        &self,
        file: &UploadFile,
        purpose: FilePurpose,
        options: &RequestOptions,
    ) -> Result<ProviderFileRef, LlmError> {
        let snapshot = self.binding.pin()?;
        let service = self.service(&snapshot, options)?;
        service.upload(file, purpose).await
    }

    pub async fn upload_stream(
        &self,
        file: UploadFileStream,
        purpose: FilePurpose,
        options: &RequestOptions,
    ) -> Result<ProviderFileRef, FileUploadError> {
        let snapshot = self.binding.pin()?;
        // The upload core enforces the deadline and preserves an unknown
        // outcome if the deadline elapses after upload dispatch.
        let service = self.service(&snapshot, options)?;
        service.upload_stream(file, purpose).await
    }

    pub async fn get(
        &self,
        file: &ProviderFileRef,
        options: &RequestOptions,
    ) -> Result<ProviderFileMetadata, LlmError> {
        let snapshot = self.binding.pin()?;
        self.service(&snapshot, options)?.get(file).await
    }

    pub async fn list(
        &self,
        cursor: Option<&str>,
        options: &RequestOptions,
    ) -> Result<ProviderFilePage, LlmError> {
        let snapshot = self.binding.pin()?;
        self.service(&snapshot, options)?.list(cursor).await
    }

    pub async fn list_for_purpose(
        &self,
        purpose: FilePurpose,
        cursor: Option<&str>,
        options: &RequestOptions,
    ) -> Result<ProviderFilePage, LlmError> {
        let snapshot = self.binding.pin()?;
        self.service(&snapshot, options)?
            .list_for_purpose(purpose, cursor)
            .await
    }

    pub async fn delete(
        &self,
        file: &ProviderFileRef,
        options: &RequestOptions,
    ) -> Result<(), LlmError> {
        let snapshot = self.binding.pin()?;
        self.service(&snapshot, options)?.delete(file).await
    }

    pub async fn download(
        &self,
        file: &ProviderFileRef,
        options: &RequestOptions,
    ) -> Result<ProviderFileContent, LlmError> {
        let snapshot = self.binding.pin()?;
        self.service(&snapshot, options)?.download(file).await
    }

    pub async fn extract_text(
        &self,
        file: &ProviderFileRef,
        options: &RequestOptions,
    ) -> Result<String, LlmError> {
        let snapshot = self.binding.pin()?;
        self.service(&snapshot, options)?.extract_text(file).await
    }
}

impl ProviderFiles<'_, super::OpenAiClient> {
    /// Stream one Batch JSONL upload without retrying an unknown outcome.
    pub async fn upload_batch_stream(
        &self,
        file: UploadFileStream,
        options: &RequestOptions,
    ) -> Result<ProviderFileRef, LlmError> {
        let snapshot = self.binding.pin()?;
        let (filename, media_type, size_bytes, body) = file.into_parts();
        self.service(&snapshot, options)?
            .upload_batch_stream(
                &filename,
                &media_type,
                size_bytes,
                body,
                options.total_timeout,
            )
            .await
    }
}

impl ProviderFiles<'_, super::AnthropicClient> {
    pub async fn list_by_ids(
        &self,
        files: &[ProviderFileRef],
        options: &RequestOptions,
    ) -> Result<ProviderFilePage, LlmError> {
        let snapshot = self.binding.pin()?;
        self.service(&snapshot, options)?.list_by_ids(files).await
    }
}

impl ProviderFiles<'_, super::GoogleClient> {
    /// Set the upload budget, bounded by Google's file lifecycle limits.
    pub fn with_upload_timeout(mut self, timeout: Duration) -> Self {
        self.upload_timeout = Some(timeout);
        self
    }

    /// Set the readiness budget for uploads and explicit processing resumption.
    pub fn with_processing_timeout(mut self, timeout: Duration) -> Self {
        self.processing_timeout = Some(timeout);
        self
    }

    pub async fn upload_unpolled(
        &self,
        file: &UploadFile,
        options: &RequestOptions,
    ) -> Result<super::google::files_wire::GeminiFile, LlmError> {
        let snapshot = self.binding.pin()?;
        let service = self.service(&snapshot, options)?;
        service.upload_gemini_unpolled(file).await
    }

    pub async fn poll_active(
        &self,
        name: &str,
        interval: Duration,
        max_wait: Duration,
        options: &RequestOptions,
    ) -> Result<super::google::files_wire::GeminiFile, LlmError> {
        let snapshot = self.binding.pin()?;
        self.service(&snapshot, options)?
            .poll_gemini_active(name, interval, max_wait)
            .await
    }

    pub async fn resume_processing(
        &self,
        file: &ProviderFileSource,
        options: &RequestOptions,
    ) -> Result<ProviderFileRef, LlmError> {
        let snapshot = self.binding.pin()?;
        let service = self.service(&snapshot, options)?;
        service.resume_gemini_processing(file).await
    }
}

macro_rules! files {
    ($($provider:ident),+ $(,)?) => {$(impl super::$provider {
        pub fn files(&self) -> ProviderFiles<'_, Self> { ProviderFiles::new(&self.binding, None) }
    })+};
}
files!(
    OpenAiClient,
    AnthropicClient,
    GoogleClient,
    QwenClient,
    MiniMaxClient,
    ZhipuClient,
    KimiClient,
    XaiClient,
    OpenRouterClient
);
impl super::AnthropicClient {
    pub fn foundry_files(&self, hosting: FoundryHosting) -> ProviderFiles<'_, Self> {
        ProviderFiles::new(&self.binding, Some(hosting))
    }
}
