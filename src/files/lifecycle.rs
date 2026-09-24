//! Automatic upload ownership and cleanup.
use super::*;

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

/// Owns automatic Qwen uploads until the model response or stream is finished.
/// A dropped preparation/stream still schedules deletion, including when a
/// request is cancelled while file parsing is in progress.
pub(crate) struct AutomaticFileCleanup {
    http: Arc<dyn Transport>,
    profile: ProviderProfile,
    authenticator: Option<Arc<dyn Authenticator>>,
    credential: Option<Secret<String>>,
    account_scope: Option<String>,
    rate_limiter: Arc<QwenFileRateLimiter>,
    pending: Mutex<Vec<ProviderFileRef>>,
}

impl AutomaticFileCleanup {
    pub(crate) fn new(
        http: Arc<dyn Transport>,
        profile: ProviderProfile,
        authenticator: Option<Arc<dyn Authenticator>>,
        credential: Option<Secret<String>>,
        account_scope: Option<String>,
        rate_limiter: Arc<QwenFileRateLimiter>,
    ) -> Self {
        Self {
            http,
            profile,
            authenticator,
            credential,
            account_scope,
            rate_limiter,
            pending: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn rate_limiter(&self) -> Arc<QwenFileRateLimiter> {
        Arc::clone(&self.rate_limiter)
    }

    pub(crate) fn track(&self, file: ProviderFileRef) {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(file);
    }

    /// Delete after the provider has returned the complete response, using
    /// only the caller's remaining deadline. Unfinished deletions remain
    /// pending for the drop-time retry.
    pub(crate) async fn finish(&self, deadline: Option<Instant>) {
        let deadline = deadline.unwrap_or_else(|| Instant::now() + QWEN_CLEANUP_DEFAULT_TIMEOUT);
        let files = self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let service = FileService::new(
            self.http.as_ref(),
            &self.profile,
            self.authenticator.as_deref(),
            self.credential.as_ref(),
            self.account_scope.as_deref(),
        )
        .with_qwen_rate_limiter(Arc::clone(&self.rate_limiter));
        for file in files {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let deleted = if tokio::runtime::Handle::try_current().is_ok() {
                tokio::time::timeout(remaining, service.delete(&file))
                    .await
                    .is_ok_and(|result| result.is_ok())
            } else {
                matches!(
                    futures::future::select(
                        Box::pin(service.delete(&file)),
                        Box::pin(async_delay(remaining)),
                    )
                    .await,
                    futures::future::Either::Left((Ok(()), _))
                )
            };
            if deleted {
                self.pending
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .retain(|pending| pending.file_id != file.file_id);
            }
        }
    }
}

impl Drop for AutomaticFileCleanup {
    fn drop(&mut self) {
        let files = std::mem::take(
            self.pending
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        if files.is_empty() {
            return;
        }
        let http = Arc::clone(&self.http);
        let profile = self.profile.clone();
        let authenticator = self.authenticator.clone();
        let credential = self.credential.clone();
        let account_scope = self.account_scope.clone();
        let rate_limiter = Arc::clone(&self.rate_limiter);
        let cleanup = async move {
            // A stream may be dropped while the provider is still unwinding
            // its request. Do not delete the file immediately at that point.
            async_delay(Duration::from_secs(2)).await;
            let service = FileService::new(
                http.as_ref(),
                &profile,
                authenticator.as_deref(),
                credential.as_ref(),
                account_scope.as_deref(),
            )
            .with_qwen_rate_limiter(rate_limiter);
            for file in files {
                for attempt in 0..3 {
                    if service.delete(&file).await.is_ok() {
                        break;
                    }
                    if attempt < 2 {
                        async_delay(Duration::from_secs(1 << attempt)).await;
                    }
                }
            }
        };
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(cleanup);
        } else {
            std::thread::spawn(move || {
                if let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    runtime.block_on(cleanup);
                }
            });
        }
    }
}
