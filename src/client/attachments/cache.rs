//! Scoped cache expiry, capacity, single-flight invalidation and pacing.
use super::*;
pub(super) fn prune_expired_provider_files(
    cache: &mut BTreeMap<FileCacheKey, CachedProviderFile>,
    now: Instant,
) {
    cache.retain(|_, entry| entry.expires_at.is_none_or(|expiry| expiry > now));
}

pub(super) fn cap_provider_file_cache(cache: &mut BTreeMap<FileCacheKey, CachedProviderFile>) {
    while cache.len() > MAX_PROVIDER_FILE_CACHE_ENTRIES {
        let oldest_key = cache
            .iter()
            .min_by(|(left_key, left), (right_key, right)| {
                left.cached_at
                    .cmp(&right.cached_at)
                    .then_with(|| left_key.cmp(right_key))
            })
            .map(|(key, _)| key.clone());
        let Some(oldest_key) = oldest_key else {
            break;
        };
        cache.remove(&oldest_key);
    }
}
impl AttachmentManager {
    pub(crate) fn qwen_file_rate_limiter(
        &self,
        profile: &ProviderProfile,
        stable_account_scope: Option<&str>,
    ) -> Arc<files::QwenFileRateLimiter> {
        let key = (
            profile.base_url.clone(),
            stable_account_scope
                .unwrap_or(&profile.profile_name)
                .to_owned(),
        );
        let mut limiters = self
            .qwen_file_rate_limiters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(
            limiters
                .entry(key)
                .or_insert_with(|| Arc::new(files::QwenFileRateLimiter::new())),
        )
    }
    pub(crate) async fn invalidate_provider_file_cache(
        &self,
        prepared_file_uses: &[PreparedProviderFileUse],
    ) {
        for used_file in prepared_file_uses {
            let Some(key) = used_file.key.as_ref() else {
                continue;
            };
            let lock = {
                let mut locks = self
                    .provider_file_upload_locks
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                locks.retain(|_, weak| weak.strong_count() > 0);
                if let Some(lock) = locks.get(key).and_then(std::sync::Weak::upgrade) {
                    lock
                } else {
                    let lock = Arc::new(futures::lock::Mutex::new(()));
                    locks.insert(key.clone(), Arc::downgrade(&lock));
                    lock
                }
            };
            let _guard = lock.lock().await;
            let now = Instant::now();
            let mut cache = self
                .provider_file_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            prune_expired_provider_files(&mut cache, now);
            let still_same_file = cache
                .get(key)
                .is_some_and(|entry| entry.file.file_id == used_file.file_id);
            if still_same_file {
                cache.remove(key);
            }
        }
    }
}

impl AttachmentManager {
    pub(super) fn content_read_lock(&self, key: &ContentCacheKey) -> Arc<futures::lock::Mutex<()>> {
        let mut locks = self
            .content_read_locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        locks.retain(|_, weak| weak.strong_count() > 0);
        locks
            .get(key)
            .and_then(std::sync::Weak::upgrade)
            .unwrap_or_else(|| {
                let lock = Arc::new(futures::lock::Mutex::new(()));
                locks.insert(key.clone(), Arc::downgrade(&lock));
                lock
            })
    }

    pub(super) fn cached_content(&self, key: &ContentCacheKey) -> Option<Bytes> {
        self.content_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entries
            .get(key)
            .map(|entry| entry.bytes.clone())
    }

    pub(super) fn cache_content(&self, key: ContentCacheKey, bytes: Bytes) -> Bytes {
        if bytes.len() > MAX_CONTENT_CACHE_BYTES {
            return bytes;
        }
        // A Bytes slice can own a much larger allocation than len(). Cache a
        // compact allocation, and release the resolver's owner outside the lock.
        // Return the same compact storage to this request as well as cache hits.
        let compact = Bytes::copy_from_slice(&bytes);
        drop(bytes);
        let bytes = compact;
        let mut cache = self
            .content_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(previous) = cache.entries.remove(&key) {
            cache.total_bytes -= previous.bytes.len();
        }
        while cache.entries.len() >= MAX_CONTENT_CACHE_ENTRIES
            || cache.total_bytes + bytes.len() > MAX_CONTENT_CACHE_BYTES
        {
            let oldest_key = cache
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.cached_at)
                .map(|(key, _)| key.clone());
            let Some(oldest_key) = oldest_key else {
                break;
            };
            if let Some(removed) = cache.entries.remove(&oldest_key) {
                cache.total_bytes -= removed.bytes.len();
            }
        }
        cache.total_bytes += bytes.len();
        cache.entries.insert(
            key,
            CachedContent {
                bytes: bytes.clone(),
                cached_at: Instant::now(),
            },
        );
        bytes
    }

    pub(super) fn upload_lock(&self, key: &FileCacheKey) -> Arc<futures::lock::Mutex<()>> {
        let mut locks = self
            .provider_file_upload_locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        locks.retain(|_, weak| weak.strong_count() > 0);
        locks
            .get(key)
            .and_then(std::sync::Weak::upgrade)
            .unwrap_or_else(|| {
                let lock = Arc::new(futures::lock::Mutex::new(()));
                locks.insert(key.clone(), Arc::downgrade(&lock));
                lock
            })
    }
    pub(super) fn cached_file(
        &self,
        key: &FileCacheKey,
        now: std::time::SystemTime,
    ) -> Option<files::ProviderFileRef> {
        let mut cache = self
            .provider_file_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        prune_expired_provider_files(&mut cache, Instant::now());
        // Provider expiry can precede the retention TTL measured after upload
        // and processing. Evict here so automatic attachments can be reuploaded
        // instead of repeatedly failing the final request preflight.
        if cache.get(key).is_some_and(|entry| {
            files::validate_file_expiration_at(entry.file.expires_at.as_deref(), now).is_err()
        }) {
            cache.remove(key);
        }
        cache.get(key).map(|entry| entry.file.clone())
    }
    pub(super) fn cache_file(
        &self,
        key: FileCacheKey,
        file: files::ProviderFileRef,
        retention: Option<std::time::Duration>,
    ) {
        let now = Instant::now();
        let mut cache = self
            .provider_file_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        prune_expired_provider_files(&mut cache, now);
        cache.insert(
            key,
            CachedProviderFile {
                file,
                expires_at: retention.and_then(|ttl| now.checked_add(ttl)),
                cached_at: now,
            },
        );
        cap_provider_file_cache(&mut cache);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct NoIo;

    #[async_trait]
    impl Transport for NoIo {
        async fn send(
            &self,
            _: crate::transport::HttpRequest,
        ) -> Result<crate::transport::StreamResponse, LlmError> {
            panic!("rate limiter identity test must not perform I/O")
        }
    }

    fn content_key(id: &str, size_bytes: u64) -> ContentCacheKey {
        ContentCacheKey::from(&AttachmentRef {
            attachment_id: id.into(),
            revision: "1".into(),
            filename: "file.pdf".into(),
            media_type: "application/pdf".into(),
            size_bytes,
        })
    }

    #[test]
    fn content_cache_enforces_both_entry_and_byte_limits() {
        let manager = AttachmentManager::new(Arc::new(NoIo), None);
        for index in 0..=MAX_CONTENT_CACHE_ENTRIES {
            manager.cache_content(content_key(&format!("{index:04}"), 0), Bytes::new());
        }
        assert!(manager.cached_content(&content_key("0000", 0)).is_none());
        assert_eq!(
            manager.content_cache.lock().unwrap().entries.len(),
            MAX_CONTENT_CACHE_ENTRIES
        );
        let bytes = Bytes::from(vec![0; MAX_CONTENT_CACHE_BYTES / 2 + 1]);
        let first = content_key("large-first", bytes.len() as u64);
        let second = content_key("large-second", bytes.len() as u64);
        manager.cache_content(first.clone(), bytes.clone());
        manager.cache_content(second.clone(), bytes.clone());
        assert!(manager.cached_content(&first).is_none());
        assert_eq!(manager.cached_content(&second).unwrap(), bytes);
        let cache = manager.content_cache.lock().unwrap();
        assert_eq!(cache.total_bytes, bytes.len());
        assert!(cache.total_bytes <= MAX_CONTENT_CACHE_BYTES);
        assert!(cache.entries.len() <= MAX_CONTENT_CACHE_ENTRIES);
    }

    #[test]
    fn content_cache_identity_includes_all_reference_metadata() {
        let manager = AttachmentManager::new(Arc::new(NoIo), None);
        let key = content_key("doc", 3);
        manager.cache_content(key.clone(), Bytes::from_static(b"pdf"));
        assert!(manager.cached_content(&key).is_some());
        let mut changed = key.clone();
        changed.filename = "other.pdf".into();
        assert!(manager.cached_content(&changed).is_none());
        let mut changed = key.clone();
        changed.media_type = "text/plain".into();
        assert!(manager.cached_content(&changed).is_none());
        let mut changed = key;
        changed.size_bytes = 4;
        assert!(manager.cached_content(&changed).is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn qwen_profile_revisions_keep_the_same_account_rate_limit_budget() {
        let mut profile: ProviderProfile = serde_json::from_value(serde_json::json!({
            "profile_name": "qwen-account", "provider_id": "qwen",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "protocol": "open_ai_chat", "auth": "none", "models": []
        }))
        .unwrap();
        let (client, config) =
            crate::LlmClientBuilder::with_transport(Arc::new(NoIo), &[profile.clone()])
                .with_region(crate::protocol::Region::ChinaMainland)
                .build_managed()
                .unwrap();
        let directory = std::env::temp_dir().join(format!(
            "llm-qwen-limiter-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        config.set_config_dir(&directory).await.unwrap();
        let old_snapshot = client.snapshot();
        let before = old_snapshot
            .runtime
            .attachments
            .qwen_file_rate_limiter(&profile, Some("account"));
        profile.extra = serde_json::json!({"headers": {"x-client-version": "new"}});
        config.add_provider(profile.clone()).await.unwrap();
        let current_snapshot = client.snapshot();
        let manager = &current_snapshot.runtime.attachments;
        let after = manager.qwen_file_rate_limiter(&profile, Some("account"));
        assert!(Arc::ptr_eq(&before, &after));
        before.wait_upload().await;
        let next_upload = after.wait_upload();
        futures::pin_mut!(next_upload);
        assert!(futures::poll!(next_upload.as_mut()).is_pending());

        let other_account = manager.qwen_file_rate_limiter(&profile, Some("other-account"));
        assert!(!Arc::ptr_eq(&after, &other_account));
        profile.base_url =
            "https://newworkspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1".into();
        let other_endpoint = manager.qwen_file_rate_limiter(&profile, Some("account"));
        assert!(!Arc::ptr_eq(&after, &other_endpoint));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
