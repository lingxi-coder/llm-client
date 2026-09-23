use async_trait::async_trait;
use lingxi_agent_api::protocol::{LlmError, ProviderProfile, Secret};
use lingxi_llm_client::{
    HttpRequest, HttpResponse, LlmClientBuilder, ProviderStoreError, StreamResponse, Transport,
    WebSocketSession,
};
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

struct AccountDirectory;

#[async_trait]
impl Transport for AccountDirectory {
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
        let key = request
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            .map(|(_, value)| value.as_str());
        let model = match key {
            Some("Bearer primary-key") => "primary-only",
            Some("Bearer spare-key") => "spare-only",
            _ => {
                return Err(LlmError::Authentication {
                    message: "wrong account credential".into(),
                });
            }
        };
        Ok(HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: serde_json::to_vec(&json!({"data": [{"id": model}, {"id": "untracked"}]}))
                .unwrap()
                .into(),
        })
    }

    async fn open_stream(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        unreachable!()
    }

    async fn open_responses_websocket_session(
        &self,
        _request: HttpRequest,
    ) -> Result<Box<dyn WebSocketSession>, LlmError> {
        unreachable!()
    }
}

fn profile(name: &str, order: u32, hidden: bool) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme",
        "profile_name": name,
        "base_url": "https://acme.example/v1",
        "protocol": "open_ai_chat",
        "auth": "api_key",
        "credential": {"source": "env", "var": format!("{}_KEY", name.to_uppercase())},
        "models": [{
            "display_model": "shared", "request_model": "shared", "billing_model": "shared"
        }],
        "connection": {
            "group": "acme", "connection_id": name, "order": order, "hidden": hidden,
            "failover": {"rateLimit": true, "overloaded": true}
        }
    }))
    .unwrap()
}

fn temp_dir() -> PathBuf {
    std::env::temp_dir().join(format!(
        "lingxi-provider-store-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ))
}

#[tokio::test]
async fn accounts_sync_independently_and_visibility_survives_restart() {
    let dir = temp_dir();
    let http = Arc::new(AccountDirectory);
    let mut client = LlmClientBuilder::with_transport(http.clone(), &[])
        .build()
        .unwrap();
    client.set_config_dir(&dir).unwrap();
    client.add_provider(profile("primary", 0, false)).unwrap();
    assert!(
        client.models().is_empty(),
        "an unset allowlist tracks nothing"
    );
    client
        .set_tracked_models(
            "acme",
            ["shared", "primary-only", "spare-only"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
    client.add_provider(profile("spare", 1, true)).unwrap();

    client
        .sync_provider("primary", Some(&Secret::new("primary-key".into())))
        .await
        .unwrap();
    assert!(client
        .profiles()
        .iter()
        .find(|p| p.profile_name == "spare")
        .unwrap()
        .models
        .iter()
        .all(|m| m.request_model != "primary-only"));
    client
        .sync_provider("spare", Some(&Secret::new("spare-key".into())))
        .await
        .unwrap();
    client
        .set_model_visibility("primary", "primary-only", false)
        .unwrap();
    assert!(!client.models().iter().any(|m| m.id == "primary-only"));
    assert!(client.resolve_in("primary-only", Some("primary")).is_ok());

    let text = std::fs::read_to_string(dir.join("providers.json")).unwrap();
    assert!(!text.contains("primary-key"));
    assert!(!text.contains("spare-key"));
    assert!(!text.contains("untracked"));
    let mut restored = LlmClientBuilder::with_transport(http, &[]).build().unwrap();
    restored.set_config_dir(&dir).unwrap();
    assert!(restored
        .tracked_models("acme")
        .unwrap()
        .contains("primary-only"));
    assert_eq!(restored.profiles().len(), 2);
    assert!(
        restored
            .profiles()
            .iter()
            .find(|p| p.profile_name == "primary")
            .unwrap()
            .models
            .iter()
            .find(|m| m.request_model == "primary-only")
            .unwrap()
            .hidden
    );
    assert!(restored
        .profiles()
        .iter()
        .find(|p| p.profile_name == "spare")
        .unwrap()
        .models
        .iter()
        .any(|m| m.request_model == "spare-only"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn failed_sync_and_static_secret_leave_saved_profiles_unchanged() {
    let dir = temp_dir();
    let mut client = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
        .build()
        .unwrap();
    client.set_config_dir(&dir).unwrap();
    client
        .set_tracked_models("acme", ["shared".to_owned()])
        .unwrap();
    client.add_provider(profile("primary", 0, false)).unwrap();
    let original = std::fs::read(dir.join("providers.json")).unwrap();

    assert!(client
        .sync_provider("primary", Some(&Secret::new("wrong".into())))
        .await
        .is_err());
    assert_eq!(std::fs::read(dir.join("providers.json")).unwrap(), original);
    let mut static_profile = profile("spare", 1, true);
    static_profile.credential = lingxi_agent_api::protocol::CredentialConfig::Static {
        value: Secret::new("secret".into()),
    };
    assert!(client.add_provider(static_profile).is_err());
    assert_eq!(client.profiles().len(), 1);
    assert_eq!(std::fs::read(dir.join("providers.json")).unwrap(), original);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn provider_crud_and_model_untracking_survive_restart() {
    let dir = temp_dir();
    let base = profile("primary", 0, false);
    let http = Arc::new(AccountDirectory);
    let mut client = LlmClientBuilder::with_transport(http.clone(), &[])
        .build()
        .unwrap();
    client.set_config_dir(&dir).unwrap();
    client.add_provider(base.clone()).unwrap();
    client
        .set_tracked_models("acme", ["shared".to_owned()])
        .unwrap();
    assert!(client.provider("primary").is_some());

    let mut updated = base.clone();
    updated.base_url = "https://new.example/v1".into();
    client.add_provider(updated).unwrap();
    assert_eq!(
        client.provider("primary").unwrap().base_url,
        "https://new.example/v1"
    );
    client.untrack_model("acme", "shared").unwrap();
    assert!(client.models().is_empty());
    assert!(!std::fs::read_to_string(dir.join("providers.json"))
        .unwrap()
        .contains("\"request_model\": \"shared\""));

    client.remove_provider("primary").unwrap();
    assert!(client.provider("primary").is_none());
    let mut restored = LlmClientBuilder::with_transport(http, &[]).build().unwrap();
    restored.set_config_dir(&dir).unwrap();
    assert!(restored.provider("primary").is_none());
    assert!(restored.tracked_models("acme").unwrap().is_empty());

    restored.add_provider(profile("primary", 0, false)).unwrap();
    assert!(restored.provider("primary").is_some());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn built_in_profile_is_soft_deleted_and_can_be_restored() {
    let dir = temp_dir();
    let builtin = lingxi_llm_client::builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "openai")
        .unwrap();
    let http = Arc::new(AccountDirectory);
    let mut client = LlmClientBuilder::with_transport(http.clone(), std::slice::from_ref(&builtin))
        .build()
        .unwrap();
    client.set_config_dir(&dir).unwrap();
    client.remove_provider("openai").unwrap();
    assert!(client.provider("openai").is_none());
    assert!(client.deleted_builtin_profiles().contains("openai"));

    let mut restored = LlmClientBuilder::with_transport(http, &[builtin])
        .build()
        .unwrap();
    restored.set_config_dir(&dir).unwrap();
    assert!(restored.provider("openai").is_none());
    assert!(restored.deleted_builtin_profiles().contains("openai"));
    restored.restore_builtin("openai").unwrap();
    assert!(restored.provider("openai").is_some());
    assert!(!restored.deleted_builtin_profiles().contains("openai"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn clients_sharing_a_directory_preserve_each_others_changes() {
    let dir = temp_dir();
    let http = Arc::new(AccountDirectory);
    let mut first = LlmClientBuilder::with_transport(http.clone(), &[])
        .build()
        .unwrap();
    let mut second = LlmClientBuilder::with_transport(http.clone(), &[])
        .build()
        .unwrap();
    first.set_config_dir(&dir).unwrap();
    second.set_config_dir(&dir).unwrap();

    first
        .set_tracked_models("acme", ["shared".to_owned()])
        .unwrap();
    first.add_provider(profile("primary", 0, false)).unwrap();
    second.add_provider(profile("spare", 1, true)).unwrap();
    second
        .set_model_visibility("primary", "shared", false)
        .unwrap();

    let mut restored = LlmClientBuilder::with_transport(http, &[]).build().unwrap();
    restored.set_config_dir(&dir).unwrap();
    assert_eq!(restored.profiles().len(), 2);
    assert!(restored.provider("primary").unwrap().models[0].hidden);
    assert!(restored.tracked_models("acme").unwrap().contains("shared"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn changing_config_directory_discards_previous_directory_state() {
    let first_dir = temp_dir();
    let second_dir = temp_dir();
    let http = Arc::new(AccountDirectory);
    let mut client = LlmClientBuilder::with_transport(http.clone(), &[])
        .build()
        .unwrap();
    client.set_config_dir(&first_dir).unwrap();
    client.add_provider(profile("primary", 0, false)).unwrap();
    client.set_config_dir(&second_dir).unwrap();
    assert!(client.provider("primary").is_none());
    client.add_provider(profile("spare", 1, true)).unwrap();

    let mut first = LlmClientBuilder::with_transport(http.clone(), &[])
        .build()
        .unwrap();
    first.set_config_dir(&first_dir).unwrap();
    assert!(first.provider("primary").is_some());
    assert!(first.provider("spare").is_none());
    let mut second = LlmClientBuilder::with_transport(http, &[]).build().unwrap();
    second.set_config_dir(&second_dir).unwrap();
    assert!(second.provider("primary").is_none());
    assert!(second.provider("spare").is_some());
    std::fs::remove_dir_all(first_dir).unwrap();
    std::fs::remove_dir_all(second_dir).unwrap();
}

#[test]
fn relative_config_dir_remains_fixed_after_cwd_change() {
    const CHILD_ROOT: &str = "LINGXI_RELATIVE_CONFIG_TEST_ROOT";
    if let Some(root) = std::env::var_os(CHILD_ROOT) {
        let root = PathBuf::from(root);
        let mut client = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
            .build()
            .unwrap();
        client.set_config_dir("first").unwrap();
        std::env::set_current_dir(root.join("second")).unwrap();
        client
            .set_tracked_models("acme", ["shared".to_owned()])
            .unwrap();
        client.add_provider(profile("primary", 0, false)).unwrap();
        assert!(root.join("first/providers.json").exists());
        assert!(!root.join("second/first/providers.json").exists());
        return;
    }

    let root = temp_dir();
    std::fs::create_dir_all(root.join("second")).unwrap();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("relative_config_dir_remains_fixed_after_cwd_change")
        .env(CHILD_ROOT, &root)
        .current_dir(&root)
        .status()
        .unwrap();
    assert!(status.success());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn removing_builder_supplied_custom_profile_stays_removed_during_session() {
    let dir = temp_dir();
    let supplied = profile("primary", 0, false);
    let mut client = LlmClientBuilder::with_transport(
        Arc::new(AccountDirectory),
        std::slice::from_ref(&supplied),
    )
    .build()
    .unwrap();
    client.set_config_dir(&dir).unwrap();
    client.remove_provider("primary").unwrap();
    client
        .set_tracked_models("acme", ["shared".to_owned()])
        .unwrap();
    assert!(client.provider("primary").is_none());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn concurrent_clients_keep_both_accounts() {
    let dir = temp_dir();
    let barrier = Arc::new(std::sync::Barrier::new(2));
    std::thread::scope(|scope| {
        for (name, order) in [("primary", 0), ("spare", 1)] {
            let dir = dir.clone();
            let barrier = barrier.clone();
            scope.spawn(move || {
                let mut client = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
                    .build()
                    .unwrap();
                client.set_config_dir(&dir).unwrap();
                barrier.wait();
                client
                    .add_provider(profile(name, order, order != 0))
                    .unwrap();
            });
        }
    });

    let mut restored = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
        .build()
        .unwrap();
    restored.set_config_dir(&dir).unwrap();
    assert!(restored.provider("primary").is_some());
    assert!(restored.provider("spare").is_some());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tracking_a_model_after_adding_a_profile_restores_its_model() {
    let dir = temp_dir();
    let mut client = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
        .build()
        .unwrap();
    client.set_config_dir(&dir).unwrap();
    client.add_provider(profile("primary", 0, false)).unwrap();
    client
        .set_tracked_models("acme", ["shared".to_owned()])
        .unwrap();
    assert!(client
        .models()
        .iter()
        .any(|model| model.request_model == "shared"));
    client
        .set_model_visibility("primary", "shared", false)
        .unwrap();

    let mut restored = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
        .build()
        .unwrap();
    restored.set_config_dir(&dir).unwrap();
    assert!(restored.provider("primary").unwrap().models[0].hidden);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn expanding_tracking_keeps_builder_models_after_visibility_change() {
    let dir = temp_dir();
    let mut supplied = profile("primary", 0, false);
    let mut second = supplied.models[0].clone();
    second.display_model = "another".into();
    second.request_model = "another".into();
    second.billing_model = "another".into();
    supplied.models.push(second);
    let mut client = LlmClientBuilder::with_transport(
        Arc::new(AccountDirectory),
        std::slice::from_ref(&supplied),
    )
    .build()
    .unwrap();
    client.set_config_dir(&dir).unwrap();
    client
        .set_tracked_models("acme", ["shared".to_owned()])
        .unwrap();
    client
        .set_model_visibility("primary", "shared", false)
        .unwrap();
    let mut restored = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[supplied])
        .build()
        .unwrap();
    restored.set_config_dir(&dir).unwrap();
    restored
        .set_model_visibility("primary", "shared", true)
        .unwrap();
    restored
        .set_tracked_models("acme", ["shared".to_owned(), "another".to_owned()])
        .unwrap();
    assert!(restored
        .models()
        .iter()
        .any(|model| model.request_model == "another"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn restoring_a_builtin_without_builder_preset_survives_another_write() {
    let dir = temp_dir();
    let mut client = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
        .build()
        .unwrap();
    client.set_config_dir(&dir).unwrap();
    client.restore_builtin("openai").unwrap();
    client
        .set_tracked_models("openai", ["gpt-4o".to_owned()])
        .unwrap();
    assert!(client.provider("openai").is_some());

    let mut restored = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
        .build()
        .unwrap();
    restored.set_config_dir(&dir).unwrap();
    assert!(restored.provider("openai").is_some());
    std::fs::remove_dir_all(dir).unwrap();
}

struct PausedDirectory {
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl Transport for PausedDirectory {
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.started.notify_one();
        self.release.notified().await;
        AccountDirectory.execute(request).await
    }

    async fn open_stream(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        unreachable!()
    }

    async fn open_responses_websocket_session(
        &self,
        _request: HttpRequest,
    ) -> Result<Box<dyn WebSocketSession>, LlmError> {
        unreachable!()
    }
}

#[tokio::test]
async fn sync_rejects_a_connection_changed_during_the_request() {
    let dir = temp_dir();
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let http = Arc::new(PausedDirectory {
        started: started.clone(),
        release: release.clone(),
    });
    let mut syncing = LlmClientBuilder::with_transport(http, &[]).build().unwrap();
    syncing.set_config_dir(&dir).unwrap();
    syncing
        .set_tracked_models("acme", ["primary-only".to_owned()])
        .unwrap();
    syncing.add_provider(profile("primary", 0, false)).unwrap();
    let mut editing = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
        .build()
        .unwrap();
    editing.set_config_dir(&dir).unwrap();

    let task = tokio::spawn(async move {
        syncing
            .sync_provider("primary", Some(&Secret::new("primary-key".into())))
            .await
    });
    started.notified().await;
    let mut changed = profile("primary", 0, false);
    changed.base_url = "https://another-account.example/v1".into();
    editing.add_provider(changed).unwrap();
    release.notify_one();
    assert!(matches!(
        task.await.unwrap(),
        Err(ProviderStoreError::ProfileChanged(name)) if name == "primary"
    ));

    let mut restored = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
        .build()
        .unwrap();
    restored.set_config_dir(&dir).unwrap();
    assert_eq!(
        restored.provider("primary").unwrap().base_url,
        "https://another-account.example/v1"
    );
    assert!(restored
        .provider("primary")
        .unwrap()
        .models
        .iter()
        .all(|model| model.request_model != "primary-only"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn whitelist_update_does_not_restore_another_clients_removed_model() {
    let dir = temp_dir();
    let http = Arc::new(AccountDirectory);
    let mut first = LlmClientBuilder::with_transport(http.clone(), &[])
        .build()
        .unwrap();
    first.set_config_dir(&dir).unwrap();
    first
        .set_tracked_models("acme", ["shared".to_owned(), "another".to_owned()])
        .unwrap();
    let mut both = profile("primary", 0, false);
    let mut another = both.models[0].clone();
    another.display_model = "another".into();
    another.request_model = "another".into();
    another.billing_model = "another".into();
    both.models.push(another);
    first.add_provider(both).unwrap();

    let mut second = LlmClientBuilder::with_transport(http.clone(), &[])
        .build()
        .unwrap();
    second.set_config_dir(&dir).unwrap();
    second.add_provider(profile("primary", 0, false)).unwrap();
    first
        .set_tracked_models("acme", ["shared".to_owned(), "another".to_owned()])
        .unwrap();
    first
        .set_tracked_models("acme", ["shared".to_owned()])
        .unwrap();
    first
        .set_tracked_models("acme", ["shared".to_owned(), "another".to_owned()])
        .unwrap();

    let mut restored = LlmClientBuilder::with_transport(http, &[]).build().unwrap();
    restored.set_config_dir(&dir).unwrap();
    assert!(restored
        .provider("primary")
        .unwrap()
        .models
        .iter()
        .all(|model| model.request_model != "another"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn restoring_builtin_overrides_a_same_named_builder_profile() {
    let dir = temp_dir();
    let preset = lingxi_llm_client::builtin_providers()
        .unwrap()
        .into_iter()
        .find(|profile| profile.profile_name == "openai")
        .unwrap();
    let mut custom = preset.clone();
    custom.base_url = "https://custom.example/v1".into();
    let http = Arc::new(AccountDirectory);
    let mut client = LlmClientBuilder::with_transport(http.clone(), std::slice::from_ref(&custom))
        .build()
        .unwrap();
    client.set_config_dir(&dir).unwrap();
    client.remove_provider("openai").unwrap();
    client.restore_builtin("openai").unwrap();
    assert_eq!(client.provider("openai").unwrap().base_url, preset.base_url);
    client
        .set_tracked_models("openai", ["gpt-4o".to_owned()])
        .unwrap();
    assert_eq!(client.provider("openai").unwrap().base_url, preset.base_url);

    let mut restored = LlmClientBuilder::with_transport(http, &[custom])
        .build()
        .unwrap();
    restored.set_config_dir(&dir).unwrap();
    assert_eq!(
        restored.provider("openai").unwrap().base_url,
        preset.base_url
    );
    std::fs::remove_dir_all(dir).unwrap();
}
