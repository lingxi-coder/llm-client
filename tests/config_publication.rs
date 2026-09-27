use async_trait::async_trait;
use lingxi_llm_client::{
    configuration::{FieldOverride, ModelField},
    protocol::{LlmError, ProviderProfile, Region},
    AccountFailure, AccountFetchContext, AccountIdentity, AccountQuery, AccountReport,
    AccountUsageError, AccountUsageSource, ClientConfigManager, HttpRequest, HttpResponse,
    LlmClient, LlmClientBuilder, ProviderStoreError, StreamResponse, Transport,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::sync::{Barrier, Notify};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
struct ConfigDirectory(PathBuf);
impl ConfigDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "llm-publication-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn bytes(&self) -> Vec<u8> {
        std::fs::read(self.0.join("providers.json")).unwrap()
    }
}
impl Drop for ConfigDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn profile(name: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme", "profile_name": name,
        "protocol": "open_ai_chat", "auth": "none",
        "base_url": format!("https://{name}.example/v1"),
        "connection": {"group": name, "connection_id": name},
        "models": [{"request_model": "old", "display_model": "old", "billing_model": "old"}]
    }))
    .unwrap()
}

struct NoNetwork;
#[async_trait]
impl Transport for NoNetwork {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("this configuration test must not send HTTP")
    }
}
fn client(profiles: &[ProviderProfile]) -> (LlmClient, ClientConfigManager) {
    LlmClientBuilder::with_transport(Arc::new(NoNetwork), profiles)
        .with_region(Region::International)
        .build_managed()
        .unwrap()
}

struct BoundAccount {
    base_url: String,
    entered: Option<Arc<Notify>>,
    release: Option<Arc<Notify>>,
}
#[async_trait]
impl AccountUsageSource for BoundAccount {
    fn requires_profile_binding(&self, _: &AccountQuery) -> bool {
        true
    }
    async fn fetch(
        &self,
        context: &AccountFetchContext<'_>,
        _: &mut AccountReport,
    ) -> Result<(), AccountFailure> {
        assert_eq!(context.profile.base_url, self.base_url);
        if let Some(entered) = &self.entered {
            entered.notify_one();
        }
        if let Some(release) = &self.release {
            release.notified().await;
        }
        assert_eq!(context.profile.base_url, self.base_url);
        Ok(())
    }
}

#[tokio::test]
async fn account_binding_and_connection_are_published_together_during_inflight_query() {
    let directory = ConfigDirectory::new();
    let original = profile("p");
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let old_source = Arc::new(BoundAccount {
        base_url: original.base_url.clone(),
        entered: Some(entered.clone()),
        release: Some(release.clone()),
    });
    let mut builder =
        LlmClientBuilder::with_transport(Arc::new(NoNetwork), std::slice::from_ref(&original));
    builder.register_account_source("acme", AccountIdentity::AuthUser, old_source.clone());
    builder.register_profile_account_source("p", AccountIdentity::AuthUser, old_source);
    let (client, config) = builder
        .with_region(Region::International)
        .build_managed()
        .unwrap();
    config.set_config_dir(&directory.0).await.unwrap();
    let pinned = client.snapshot();
    let concurrent = client.clone();
    let query = tokio::spawn(async move {
        concurrent
            .account_usage("p", &AccountQuery::new(AccountIdentity::AuthUser))
            .await
    });
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .unwrap();

    let mut replacement = original.clone();
    replacement.base_url = "https://replacement.example/v1".into();
    config.add_provider(replacement.clone()).await.unwrap();
    assert!(matches!(
        client
            .account_usage("p", &AccountQuery::new(AccountIdentity::AuthUser))
            .await,
        Err(AccountUsageError::AmbiguousAccountSource(_))
    ));
    config
        .register_profile_account_source(
            "p",
            AccountIdentity::AuthUser,
            Arc::new(BoundAccount {
                base_url: replacement.base_url.clone(),
                entered: None,
                release: None,
            }),
        )
        .await
        .unwrap();
    client
        .account_usage("p", &AccountQuery::new(AccountIdentity::AuthUser))
        .await
        .unwrap();
    assert_eq!(pinned.provider("p").unwrap().base_url, original.base_url);
    assert_eq!(
        client.snapshot().provider("p").unwrap().base_url,
        replacement.base_url
    );
    release.notify_one();
    tokio::time::timeout(Duration::from_secs(5), query)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

struct CountingAccount {
    base_url: String,
    calls: Arc<AtomicU64>,
}
#[async_trait]
impl AccountUsageSource for CountingAccount {
    async fn fetch(
        &self,
        context: &AccountFetchContext<'_>,
        _: &mut AccountReport,
    ) -> Result<(), AccountFailure> {
        assert_eq!(context.profile.base_url, self.base_url);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::test]
async fn queued_accounts_keep_the_batch_snapshot_across_connection_and_binding_updates() {
    let directory = ConfigDirectory::new();
    let first = profile("p");
    let second = profile("q");
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let old_calls = Arc::new(AtomicU64::new(0));
    let new_calls = Arc::new(AtomicU64::new(0));
    let mut builder =
        LlmClientBuilder::with_transport(Arc::new(NoNetwork), &[first.clone(), second.clone()]);
    builder.with_account_concurrency(NonZeroUsize::new(1).unwrap());
    builder.register_profile_account_source(
        "p",
        AccountIdentity::AuthUser,
        Arc::new(BoundAccount {
            base_url: first.base_url.clone(),
            entered: Some(entered.clone()),
            release: Some(release.clone()),
        }),
    );
    builder.register_profile_account_source(
        "q",
        AccountIdentity::AuthUser,
        Arc::new(CountingAccount {
            base_url: second.base_url.clone(),
            calls: old_calls.clone(),
        }),
    );
    let (client, config) = builder
        .with_region(Region::International)
        .build_managed()
        .unwrap();
    config.set_config_dir(&directory.0).await.unwrap();
    let queries = BTreeMap::from([
        ("p".into(), AccountQuery::new(AccountIdentity::AuthUser)),
        ("q".into(), AccountQuery::new(AccountIdentity::AuthUser)),
    ]);
    let concurrent = client.clone();
    let batch_queries = queries.clone();
    let batch = tokio::spawn(async move { concurrent.accounts_usage(&batch_queries).await });
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .unwrap();
    // A concurrency limit of one leaves q queued while p is inside its fetch.
    assert_eq!(old_calls.load(Ordering::Relaxed), 0);
    let mut replacement = second.clone();
    replacement.base_url = "https://new-q.example/v1".into();
    tokio::time::timeout(Duration::from_secs(5), async {
        config.add_provider(replacement.clone()).await.unwrap();
        config
            .register_profile_account_source(
                "q",
                AccountIdentity::AuthUser,
                Arc::new(CountingAccount {
                    base_url: replacement.base_url.clone(),
                    calls: new_calls.clone(),
                }),
            )
            .await
            .unwrap();
    })
    .await
    .unwrap();
    release.notify_one();
    let results = tokio::time::timeout(Duration::from_secs(5), batch)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        results
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["p", "q"]
    );
    assert!(results.iter().all(|(_, result)| result.is_ok()));
    assert_eq!(old_calls.load(Ordering::Relaxed), 1);
    assert_eq!(new_calls.load(Ordering::Relaxed), 0);

    // The next whole batch uses the newly published q connection and binding.
    release.notify_one();
    let results = tokio::time::timeout(Duration::from_secs(5), client.accounts_usage(&queries))
        .await
        .unwrap();
    assert!(results.iter().all(|(_, result)| result.is_ok()));
    assert_eq!(old_calls.load(Ordering::Relaxed), 1);
    assert_eq!(new_calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn failed_validation_and_io_leave_all_clones_on_the_previous_revision() {
    let directory = ConfigDirectory::new();
    let (client, config) = client(&[profile("p")]);
    config.set_config_dir(&directory.0).await.unwrap();
    config
        .set_tracked_models("acme", ["old".to_owned()])
        .await
        .unwrap();
    let clone = client.clone();
    let before = client.snapshot();
    let bytes = directory.bytes();
    assert!(matches!(
        config
            .set_model_override(
                "p",
                "missing-row",
                ModelField::Hidden,
                FieldOverride::Set(json!(true))
            )
            .await,
        Err(ProviderStoreError::UnknownModel { .. })
    ));
    assert_eq!(directory.bytes(), bytes);
    assert_eq!(client.snapshot().revision(), before.revision());
    assert_eq!(clone.snapshot().revision(), before.revision());

    let lock_path = directory.0.join(".providers.json.lock");
    std::fs::remove_file(&lock_path).unwrap();
    std::fs::create_dir(&lock_path).unwrap();
    assert!(matches!(
        config.set_model_visibility("p", "old", false).await,
        Err(ProviderStoreError::Io(_))
    ));
    assert_eq!(directory.bytes(), bytes);
    assert_eq!(client.snapshot().revision(), before.revision());
    assert_eq!(clone.snapshot().revision(), before.revision());
    assert!(!client.snapshot().provider("p").unwrap().models[0].hidden);
}

struct ConcurrentDirectory {
    entered: Arc<Barrier>,
    release: Arc<Notify>,
}
#[async_trait]
impl Transport for ConcurrentDirectory {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let released = self.release.notified();
        tokio::pin!(released);
        released.as_mut().enable();
        self.entered.wait().await;
        released.await;
        let model = if request.url.contains("p.example") {
            "p-live"
        } else {
            "q-live"
        };
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: serde_json::to_vec(&json!({"data": [{"id": model}]}))
                .unwrap()
                .into(),
        }
        .into())
    }
}

#[tokio::test]
async fn parallel_directory_fetches_merge_with_configuration_changes_before_apply() {
    let directory = ConfigDirectory::new();
    let entered = Arc::new(Barrier::new(3));
    let release = Arc::new(Notify::new());
    let transport = Arc::new(ConcurrentDirectory {
        entered: entered.clone(),
        release: release.clone(),
    });
    let (client, config) =
        LlmClientBuilder::with_transport(transport, &[profile("p"), profile("q")])
            .with_region(Region::International)
            .build_managed()
            .unwrap();
    config.set_config_dir(&directory.0).await.unwrap();
    config
        .set_tracked_models("acme", ["old".into(), "p-live".into(), "q-live".into()])
        .await
        .unwrap();
    let p = config.prepare_provider_sync("p", None).await.unwrap();
    let q = config.prepare_provider_sync("q", None).await.unwrap();
    let p = tokio::spawn(p.fetch());
    let q = tokio::spawn(q.fetch());
    tokio::time::timeout(Duration::from_secs(5), entered.wait())
        .await
        .unwrap();
    config
        .set_model_visibility("p", "old", false)
        .await
        .unwrap();
    config
        .set_model_visibility("q", "old", false)
        .await
        .unwrap();
    release.notify_waiters();
    let (p, q) = tokio::join!(p, q);
    let p = p.unwrap().unwrap();
    let q = q.unwrap().unwrap();
    let (p, q) = tokio::join!(config.apply_provider_sync(p), config.apply_provider_sync(q));
    assert_eq!(p.unwrap(), 1);
    assert_eq!(q.unwrap(), 1);
    let snapshot = client.snapshot();
    for (name, added) in [("p", "p-live"), ("q", "q-live")] {
        let profile = snapshot.provider(name).unwrap();
        assert!(
            profile
                .models
                .iter()
                .find(|model| model.request_model == "old")
                .unwrap()
                .hidden
        );
        assert!(profile
            .models
            .iter()
            .any(|model| model.request_model == added));
    }
    // Reloading disk must produce exactly the same merged result.
    config.set_config_dir(&directory.0).await.unwrap();
    assert_eq!(client.snapshot().profiles(), snapshot.profiles());
}
