//! Automatic upload ownership and cleanup.
use super::*;

static BACKGROUND_FILE_CLEANUP: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(16);

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
        for (index, file) in files.iter().enumerate() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            // Give later files part of the cleanup window even if an earlier
            // deletion stalls. A successful deletion frees its unused time.
            let remaining_files = u32::try_from(files.len() - index).unwrap_or(u32::MAX);
            let timeout = remaining / remaining_files;
            let deleted = if tokio::runtime::Handle::try_current().is_ok() {
                tokio::time::timeout(timeout, service.delete(file))
                    .await
                    .is_ok_and(|result| result.is_ok())
            } else {
                matches!(
                    futures::future::select(
                        Box::pin(service.delete(file)),
                        Box::pin(async_delay(timeout)),
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
            let _permit = BACKGROUND_FILE_CLEANUP
                .acquire()
                .await
                .expect("cleanup semaphore is open");
            let service = FileService::new(
                http.as_ref(),
                &profile,
                authenticator.as_deref(),
                credential.as_ref(),
                account_scope.as_deref(),
            )
            .with_qwen_rate_limiter(rate_limiter);
            for file in files {
                let deadline = tokio::time::Instant::now() + QWEN_CLEANUP_DEFAULT_TIMEOUT;
                for attempt in 0..3 {
                    match tokio::time::timeout_at(deadline, service.delete(&file)).await {
                        Ok(Ok(())) | Err(_) => break,
                        Ok(Err(_)) if attempt < 2 => {
                            if tokio::time::timeout_at(
                                deadline,
                                async_delay(Duration::from_secs(1 << attempt)),
                            )
                            .await
                            .is_err()
                            {
                                break;
                            }
                        }
                        Ok(Err(_)) => break,
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
