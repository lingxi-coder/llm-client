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
    pub(super) fn cached_file(&self, key: &FileCacheKey) -> Option<files::ProviderFileRef> {
        let mut cache = self
            .provider_file_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        prune_expired_provider_files(&mut cache, Instant::now());
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
