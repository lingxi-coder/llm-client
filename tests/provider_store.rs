use async_trait::async_trait;
use lingxi_llm_client::protocol::{
    CapabilitySupport, LlmError, ModelCapability, ProviderProfile, Secret,
};
use lingxi_llm_client::{
    HttpRequest, HttpResponse, LlmClientBuilder, ProviderStoreError, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

struct AccountDirectory;

#[async_trait]
impl Transport for AccountDirectory {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.response(request).await.map(Into::into)
    }
}
impl AccountDirectory {
    async fn response(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
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
}

struct MutableDirectory {
    body: Mutex<Value>,
    requests: AtomicU64,
}

impl MutableDirectory {
    fn new(body: Value) -> Self {
        Self {
            body: Mutex::new(body),
            requests: AtomicU64::new(0),
        }
    }

    fn set_body(&self, body: Value) {
        *self.body.lock().unwrap() = body;
    }
}

#[async_trait]
impl Transport for MutableDirectory {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.response(request).await.map(Into::into)
    }
}
impl MutableDirectory {
    async fn response(&self, _request: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let body = self.body.lock().unwrap().clone();
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: serde_json::to_vec(&body).unwrap().into(),
        })
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

fn gemini_profile(name: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "google",
        "profile_name": name,
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "protocol": "gemini_generate_content",
        "model_list": "gemini_generate_content",
        "auth": "none",
        "models": [
            {
                "display_model": "Gemini Embedding 001",
                "request_model": "gemini-embedding-001",
                "billing_model": "gemini-embedding-001"
            },
            {
                "display_model": "Gemini 2.5 Pro",
                "request_model": "gemini-2.5-pro",
                "billing_model": "gemini-2.5-pro"
            }
        ]
    }))
    .unwrap()
}

fn gemini_page(methods: &[&str]) -> Value {
    json!({
        "models": [{
            "name": "models/gemini-embedding-001",
            "supportedGenerationMethods": methods
        }]
    })
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
    let (client, client_config) = LlmClientBuilder::with_transport(http.clone(), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();
    assert!(
        client.models().is_empty(),
        "an unset allowlist tracks nothing"
    );
    client_config
        .set_tracked_models(
            "acme",
            ["shared", "primary-only", "spare-only"]
                .into_iter()
                .map(str::to_owned),
        )
        .await
        .unwrap();
    client_config
        .add_provider(profile("spare", 1, true))
        .await
        .unwrap();

    client_config
        .sync_provider("primary", Some(&Secret::new("primary-key".into())))
        .await
        .unwrap();
    let client_view = client.snapshot();
    let imported = client_view
        .provider("primary")
        .unwrap()
        .models
        .iter()
        .find(|model| model.request_model == "primary-only")
        .unwrap();
    assert_eq!(
        imported.capability_support.unwrap_or_default(),
        lingxi_llm_client::protocol::ModelCapabilitySupport::default()
    );
    assert_eq!(imported.capability_support, None);
    assert_eq!(
        imported.capability_support_for(ModelCapability::Tools),
        CapabilitySupport::Unknown
    );
    assert!(client
        .snapshot()
        .profiles()
        .iter()
        .find(|p| p.profile_name == "spare")
        .unwrap()
        .models
        .iter()
        .all(|m| m.request_model != "primary-only"));
    client_config
        .sync_provider("spare", Some(&Secret::new("spare-key".into())))
        .await
        .unwrap();
    client_config
        .set_model_visibility("primary", "primary-only", false)
        .await
        .unwrap();
    assert!(!client.models().iter().any(|m| m.id == "primary-only"));
    assert!(client.resolve_in("primary-only", Some("primary")).is_ok());

    let text = std::fs::read_to_string(dir.join("providers.json")).unwrap();
    assert!(!text.contains("primary-key"));
    assert!(!text.contains("spare-key"));
    assert!(!text.contains("untracked"));
    let (restored, restored_config) = LlmClientBuilder::with_transport(http, &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    assert!(restored_config
        .tracked_models("acme")
        .await
        .unwrap()
        .unwrap()
        .contains("primary-only"));
    assert_eq!(restored.snapshot().profiles().len(), 2);
    assert!(
        restored
            .snapshot()
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
        .snapshot()
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
    let (client, client_config) = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config
        .set_tracked_models("acme", ["shared".to_owned()])
        .await
        .unwrap();
    client_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();
    let original = std::fs::read(dir.join("providers.json")).unwrap();

    assert!(client_config
        .sync_provider("primary", Some(&Secret::new("wrong".into())))
        .await
        .is_err());
    assert_eq!(std::fs::read(dir.join("providers.json")).unwrap(), original);
    let mut static_profile = profile("spare", 1, true);
    static_profile.credential = lingxi_llm_client::protocol::CredentialConfig::Static {
        value: Secret::new("secret".into()),
    };
    assert!(client_config.add_provider(static_profile).await.is_err());
    assert_eq!(client.snapshot().profiles().len(), 1);
    assert_eq!(std::fs::read(dir.join("providers.json")).unwrap(), original);
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn incompatible_gemini_models_stay_excluded_across_whitelist_and_reload() {
    let dir = temp_dir();
    let http = Arc::new(MutableDirectory::new(gemini_page(&["embedContent"])));
    let stale_profile = gemini_profile("gemini");
    let (client, client_config) =
        LlmClientBuilder::with_transport(http.clone(), std::slice::from_ref(&stale_profile))
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config
        .set_tracked_models("google", ["gemini-2.5-pro".to_owned()])
        .await
        .unwrap();

    assert_eq!(
        client_config.sync_provider("gemini", None).await.unwrap(),
        0
    );
    assert!(!client
        .snapshot()
        .provider("gemini")
        .unwrap()
        .models
        .iter()
        .any(|model| model.request_model == "gemini-embedding-001"));

    client_config
        .set_tracked_models(
            "google",
            [
                "gemini-2.5-pro".to_owned(),
                "gemini-embedding-001".to_owned(),
            ],
        )
        .await
        .unwrap();
    assert!(!client
        .snapshot()
        .provider("gemini")
        .unwrap()
        .models
        .iter()
        .any(|model| model.request_model == "gemini-embedding-001"));

    http.set_body(json!({
        "models": [{"name": "models/gemini-embedding-001"}]
    }));
    assert_eq!(
        client_config.sync_provider("gemini", None).await.unwrap(),
        0,
        "missing method metadata is unknown and cannot clear a known exclusion"
    );
    assert!(!client
        .snapshot()
        .provider("gemini")
        .unwrap()
        .models
        .iter()
        .any(|model| model.request_model == "gemini-embedding-001"));

    let (restored, restored_config) =
        LlmClientBuilder::with_transport(http.clone(), std::slice::from_ref(&stale_profile))
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    assert!(!restored
        .snapshot()
        .provider("gemini")
        .unwrap()
        .models
        .iter()
        .any(|model| model.request_model == "gemini-embedding-001"));

    http.set_body(gemini_page(&["embedContent", "generateContent"]));
    assert_eq!(
        client_config.sync_provider("gemini", None).await.unwrap(),
        1
    );
    assert!(client
        .snapshot()
        .provider("gemini")
        .unwrap()
        .models
        .iter()
        .any(|model| model.request_model == "gemini-embedding-001"));

    let (after_generation_support, after_generation_support_config) =
        LlmClientBuilder::with_transport(http, &[stale_profile])
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    after_generation_support_config
        .set_config_dir(&dir)
        .await
        .unwrap();
    assert!(after_generation_support
        .snapshot()
        .provider("gemini")
        .unwrap()
        .models
        .iter()
        .any(|model| model.request_model == "gemini-embedding-001"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn malformed_gemini_page_token_fails_sync_without_committing_partial_models() {
    let dir = temp_dir();
    let http = Arc::new(MutableDirectory::new(json!({
        "models": [{
            "name": "models/new-generation-model",
            "supportedGenerationMethods": ["generateContent"]
        }],
        "nextPageToken": 42
    })));
    let profile = gemini_profile("gemini-malformed-token");
    let (client, client_config) = LlmClientBuilder::with_transport(http.clone(), &[profile])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config
        .set_tracked_models("google", ["new-generation-model".to_owned()])
        .await
        .unwrap();
    let before = std::fs::read(dir.join("providers.json")).unwrap();

    let error = client_config
        .sync_provider("gemini-malformed-token", None)
        .await
        .expect_err("a numeric page token cannot complete a directory sync");
    assert!(
        matches!(
            error,
            ProviderStoreError::Directory(LlmError::ProviderInternal { .. })
        ),
        "{error}"
    );
    assert_eq!(http.requests.load(Ordering::Relaxed), 1);
    assert!(!client
        .snapshot()
        .provider("gemini-malformed-token")
        .unwrap()
        .models
        .iter()
        .any(|model| model.request_model == "new-generation-model"));
    assert_eq!(std::fs::read(dir.join("providers.json")).unwrap(), before);
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn replacing_a_gemini_profile_clears_saved_incompatible_model_ids() {
    let dir = temp_dir();
    let http = Arc::new(MutableDirectory::new(gemini_page(&["embedContent"])));
    let stale_profile = gemini_profile("gemini-replaced");
    let (client, client_config) =
        LlmClientBuilder::with_transport(http, std::slice::from_ref(&stale_profile))
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config
        .set_tracked_models("google", ["gemini-2.5-pro".to_owned()])
        .await
        .unwrap();
    client_config
        .sync_provider("gemini-replaced", None)
        .await
        .unwrap();

    let mut replacement = stale_profile;
    replacement.base_url = "https://replacement.example/v1beta".to_owned();
    client_config.add_provider(replacement).await.unwrap();
    client_config
        .set_tracked_models(
            "google",
            [
                "gemini-2.5-pro".to_owned(),
                "gemini-embedding-001".to_owned(),
            ],
        )
        .await
        .unwrap();
    assert!(client
        .snapshot()
        .provider("gemini-replaced")
        .unwrap()
        .models
        .iter()
        .any(|model| model.request_model == "gemini-embedding-001"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn provider_crud_and_model_untracking_survive_restart() {
    let dir = temp_dir();
    let base = profile("primary", 0, false);
    let http = Arc::new(AccountDirectory);
    let (client, client_config) = LlmClientBuilder::with_transport(http.clone(), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config.add_provider(base.clone()).await.unwrap();
    client_config
        .set_tracked_models("acme", ["shared".to_owned()])
        .await
        .unwrap();
    assert!(client.snapshot().provider("primary").is_some());

    let mut updated = base.clone();
    updated.base_url = "https://new.example/v1".into();
    client_config.add_provider(updated).await.unwrap();
    assert_eq!(
        client.snapshot().provider("primary").unwrap().base_url,
        "https://new.example/v1"
    );
    client_config.untrack_model("acme", "shared").await.unwrap();
    assert!(client.models().is_empty());
    assert!(!std::fs::read_to_string(dir.join("providers.json"))
        .unwrap()
        .contains("\"request_model\": \"shared\""));

    client_config.remove_provider("primary").await.unwrap();
    assert!(client.snapshot().provider("primary").is_none());
    let (restored, restored_config) = LlmClientBuilder::with_transport(http, &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    assert!(restored.snapshot().provider("primary").is_none());
    assert!(restored_config
        .tracked_models("acme")
        .await
        .unwrap()
        .unwrap()
        .is_empty());

    restored_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();
    assert!(restored.snapshot().provider("primary").is_some());
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn provider_add_and_update_cannot_persist_credential_bearing_extra_headers() {
    let dir = temp_dir();
    let (client, client_config) = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    let original = profile("primary", 0, false);
    client_config.add_provider(original.clone()).await.unwrap();
    let client_view = client.snapshot();
    let original = client_view.provider("primary").unwrap().clone();
    let saved_before = std::fs::read(dir.join("providers.json")).unwrap();

    let mut added = profile("secret-add", 1, false);
    added.extra = json!({"headers": {"Authorization": "Bearer should-not-persist"}});
    let error = client_config
        .add_provider(added)
        .await
        .expect_err("credential headers must be rejected on add");
    assert!(!error.to_string().contains("should-not-persist"));
    assert_eq!(
        std::fs::read(dir.join("providers.json")).unwrap(),
        saved_before
    );
    assert!(client.snapshot().provider("secret-add").is_none());

    for header in [
        "authorization",
        "Proxy-Authorization",
        "x-api-key",
        "api-key",
        "Cookie",
        "X-Goog-Api-Key",
    ] {
        let mut updated = original.clone();
        updated.extra = json!({"headers": {header: "credential-value"}});
        let error = match client_config.add_provider(updated).await {
            Ok(()) => panic!("{header} must be rejected on update"),
            Err(error) => error,
        };
        assert!(!error.to_string().contains("credential-value"));
        assert_eq!(
            std::fs::read(dir.join("providers.json")).unwrap(),
            saved_before
        );
        assert_eq!(client.snapshot().provider("primary").unwrap(), &original);
    }

    let mut custom = original.clone();
    custom.extra = json!({
        "credential_header": "X-House-Token",
        "headers": {"x-house-token": "custom-secret"}
    });
    let error = client_config
        .add_provider(custom)
        .await
        .expect_err("an explicitly configured credential header is also sensitive");
    assert!(!error.to_string().contains("custom-secret"));
    assert_eq!(
        std::fs::read(dir.join("providers.json")).unwrap(),
        saved_before
    );
    assert_eq!(client.snapshot().provider("primary").unwrap(), &original);

    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn built_in_profile_is_soft_deleted_and_can_be_restored() {
    let dir = temp_dir();
    let builtin = lingxi_llm_client::builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "openai")
        .unwrap();
    let http = Arc::new(AccountDirectory);
    let (client, client_config) =
        LlmClientBuilder::with_transport(http.clone(), std::slice::from_ref(&builtin))
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config.remove_provider("openai").await.unwrap();
    assert!(client.snapshot().provider("openai").is_none());
    assert!(client_config
        .deleted_builtin_profiles()
        .await
        .unwrap()
        .contains("openai"));

    let (restored, restored_config) = LlmClientBuilder::with_transport(http, &[builtin])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    assert!(restored.snapshot().provider("openai").is_none());
    assert!(restored_config
        .deleted_builtin_profiles()
        .await
        .unwrap()
        .contains("openai"));
    restored_config.restore_builtin("openai").await.unwrap();
    assert!(restored.snapshot().provider("openai").is_some());
    assert!(!restored_config
        .deleted_builtin_profiles()
        .await
        .unwrap()
        .contains("openai"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn clients_sharing_a_directory_preserve_each_others_changes() {
    let dir = temp_dir();
    let http = Arc::new(AccountDirectory);
    let (_first, first_config) = LlmClientBuilder::with_transport(http.clone(), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    let (_second, second_config) = LlmClientBuilder::with_transport(http.clone(), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    first_config.set_config_dir(&dir).await.unwrap();
    second_config.set_config_dir(&dir).await.unwrap();

    first_config
        .set_tracked_models("acme", ["shared".to_owned()])
        .await
        .unwrap();
    first_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();
    second_config
        .add_provider(profile("spare", 1, true))
        .await
        .unwrap();
    second_config
        .set_model_visibility("primary", "shared", false)
        .await
        .unwrap();

    let (restored, restored_config) = LlmClientBuilder::with_transport(http, &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    assert_eq!(restored.snapshot().profiles().len(), 2);
    assert!(restored.snapshot().provider("primary").unwrap().models[0].hidden);
    assert!(restored_config
        .tracked_models("acme")
        .await
        .unwrap()
        .unwrap()
        .contains("shared"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn changing_config_directory_discards_previous_directory_state() {
    let first_dir = temp_dir();
    let second_dir = temp_dir();
    let http = Arc::new(AccountDirectory);
    let (client, client_config) = LlmClientBuilder::with_transport(http.clone(), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&first_dir).await.unwrap();
    client_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();
    client_config.set_config_dir(&second_dir).await.unwrap();
    assert!(client.snapshot().provider("primary").is_none());
    client_config
        .add_provider(profile("spare", 1, true))
        .await
        .unwrap();

    let (first, first_config) = LlmClientBuilder::with_transport(http.clone(), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    first_config.set_config_dir(&first_dir).await.unwrap();
    assert!(first.snapshot().provider("primary").is_some());
    assert!(first.snapshot().provider("spare").is_none());
    let (second, second_config) = LlmClientBuilder::with_transport(http, &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    second_config.set_config_dir(&second_dir).await.unwrap();
    assert!(second.snapshot().provider("primary").is_none());
    assert!(second.snapshot().provider("spare").is_some());
    std::fs::remove_dir_all(first_dir).unwrap();
    std::fs::remove_dir_all(second_dir).unwrap();
}

#[tokio::test]
async fn relative_config_dir_remains_fixed_after_cwd_change() {
    const CHILD_ROOT: &str = "LINGXI_RELATIVE_CONFIG_TEST_ROOT";
    if let Some(root) = std::env::var_os(CHILD_ROOT) {
        let root = PathBuf::from(root);
        let (_client, client_config) =
            LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
                .with_region(lingxi_llm_client::protocol::Region::International)
                .build_managed()
                .unwrap();
        client_config.set_config_dir("first").await.unwrap();
        std::env::set_current_dir(root.join("second")).unwrap();
        client_config
            .set_tracked_models("acme", ["shared".to_owned()])
            .await
            .unwrap();
        client_config
            .add_provider(profile("primary", 0, false))
            .await
            .unwrap();
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

#[tokio::test]
async fn removing_builder_supplied_custom_profile_stays_removed_during_session() {
    let dir = temp_dir();
    let supplied = profile("primary", 0, false);
    let (client, client_config) = LlmClientBuilder::with_transport(
        Arc::new(AccountDirectory),
        std::slice::from_ref(&supplied),
    )
    .with_region(lingxi_llm_client::protocol::Region::International)
    .build_managed()
    .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config.remove_provider("primary").await.unwrap();
    client_config
        .set_tracked_models("acme", ["shared".to_owned()])
        .await
        .unwrap();
    assert!(client.snapshot().provider("primary").is_none());
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn concurrent_clients_keep_both_accounts() {
    let dir = temp_dir();
    let barrier = Arc::new(std::sync::Barrier::new(2));
    std::thread::scope(|scope| {
        for (name, order) in [("primary", 0), ("spare", 1)] {
            let dir = dir.clone();
            let barrier = barrier.clone();
            scope.spawn(move || {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async move {
                        let (_client, client_config) =
                            LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
                                .with_region(lingxi_llm_client::protocol::Region::International)
                                .build_managed()
                                .unwrap();
                        client_config.set_config_dir(&dir).await.unwrap();
                        barrier.wait();
                        client_config
                            .add_provider(profile(name, order, order != 0))
                            .await
                            .unwrap();
                    });
            });
        }
    });

    let (restored, restored_config) =
        LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    assert!(restored.snapshot().provider("primary").is_some());
    assert!(restored.snapshot().provider("spare").is_some());
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tracking_a_model_after_adding_a_profile_restores_its_model() {
    let dir = temp_dir();
    let (client, client_config) = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();
    client_config
        .set_tracked_models("acme", ["shared".to_owned()])
        .await
        .unwrap();
    assert!(client
        .models()
        .iter()
        .any(|model| model.request_model == "shared"));
    client_config
        .set_model_visibility("primary", "shared", false)
        .await
        .unwrap();

    let (restored, restored_config) =
        LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    assert!(restored.snapshot().provider("primary").unwrap().models[0].hidden);
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn expanding_tracking_keeps_builder_models_after_visibility_change() {
    let dir = temp_dir();
    let mut supplied = profile("primary", 0, false);
    let mut second = supplied.models[0].clone();
    second.display_model = "another".into();
    second.request_model = "another".into();
    second.billing_model = "another".into();
    supplied.models.push(second);
    let (_client, client_config) = LlmClientBuilder::with_transport(
        Arc::new(AccountDirectory),
        std::slice::from_ref(&supplied),
    )
    .with_region(lingxi_llm_client::protocol::Region::International)
    .build_managed()
    .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config
        .set_tracked_models("acme", ["shared".to_owned()])
        .await
        .unwrap();
    client_config
        .set_model_visibility("primary", "shared", false)
        .await
        .unwrap();
    let (restored, restored_config) =
        LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[supplied])
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    restored_config
        .set_model_visibility("primary", "shared", true)
        .await
        .unwrap();
    restored_config
        .set_tracked_models("acme", ["shared".to_owned(), "another".to_owned()])
        .await
        .unwrap();
    assert!(restored
        .models()
        .iter()
        .any(|model| model.request_model == "another"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn restoring_a_builtin_without_builder_preset_survives_another_write() {
    let dir = temp_dir();
    let (client, client_config) = LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config.restore_builtin("openai").await.unwrap();
    client_config
        .set_tracked_models("openai", ["gpt-4o".to_owned()])
        .await
        .unwrap();
    assert!(client.snapshot().provider("openai").is_some());

    let (restored, restored_config) =
        LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    assert!(restored.snapshot().provider("openai").is_some());
    std::fs::remove_dir_all(dir).unwrap();
}

struct PausedDirectory {
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

struct ConcurrentDirectory {
    both_started: Arc<tokio::sync::Barrier>,
}

#[async_trait]
impl Transport for PausedDirectory {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.response(request).await.map(Into::into)
    }
}
impl PausedDirectory {
    async fn response(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.started.notify_one();
        self.release.notified().await;
        AccountDirectory.response(request).await
    }
}

#[async_trait]
impl Transport for ConcurrentDirectory {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.response(request).await.map(Into::into)
    }
}
impl ConcurrentDirectory {
    async fn response(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.both_started.wait().await;
        AccountDirectory.response(request).await
    }
}

#[tokio::test]
async fn prepared_provider_syncs_fetch_concurrently_for_one_client() {
    let dir = temp_dir();
    let http = Arc::new(ConcurrentDirectory {
        both_started: Arc::new(tokio::sync::Barrier::new(2)),
    });
    let (client, client_config) = LlmClientBuilder::with_transport(http, &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config
        .set_tracked_models("acme", ["primary-only".to_owned(), "spare-only".to_owned()])
        .await
        .unwrap();
    client_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();
    client_config
        .add_provider(profile("spare", 1, true))
        .await
        .unwrap();

    let primary = client_config
        .prepare_provider_sync("primary", Some(&Secret::new("primary-key".into())))
        .await
        .unwrap();
    let spare = client_config
        .prepare_provider_sync("spare", Some(&Secret::new("spare-key".into())))
        .await
        .unwrap();
    let (primary, spare) = tokio::join!(primary.fetch(), spare.fetch());
    client_config
        .apply_provider_sync(primary.unwrap())
        .await
        .unwrap();
    client_config
        .apply_provider_sync(spare.unwrap())
        .await
        .unwrap();

    assert!(client
        .snapshot()
        .provider("primary")
        .unwrap()
        .models
        .iter()
        .any(|model| model.request_model == "primary-only"));
    assert!(client
        .snapshot()
        .provider("spare")
        .unwrap()
        .models
        .iter()
        .any(|model| model.request_model == "spare-only"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn prepared_sync_rejects_a_profile_changed_during_fetch() {
    let dir = temp_dir();
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let http = Arc::new(PausedDirectory {
        started: started.clone(),
        release: release.clone(),
    });
    let (client, client_config) = LlmClientBuilder::with_transport(http, &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config
        .set_tracked_models("acme", ["primary-only".to_owned()])
        .await
        .unwrap();
    client_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();

    let operation = client_config
        .prepare_provider_sync("primary", Some(&Secret::new("primary-key".into())))
        .await
        .unwrap();
    let fetch = tokio::spawn(operation.fetch());
    started.notified().await;
    let mut changed = profile("primary", 0, false);
    changed.base_url = "https://changed-during-fetch.example/v1".into();
    changed.credential = lingxi_llm_client::protocol::CredentialConfig::Env {
        var: "CHANGED_PRIMARY_KEY".into(),
    };
    client_config.add_provider(changed).await.unwrap();
    release.notify_one();
    let result = fetch.await.unwrap().unwrap();

    assert!(matches!(
        client_config.apply_provider_sync(result).await,
        Err(ProviderStoreError::ProfileChanged(name)) if name == "primary"
    ));
    assert_eq!(
        client.snapshot().provider("primary").unwrap().base_url,
        "https://changed-during-fetch.example/v1"
    );
    assert!(client
        .snapshot()
        .provider("primary")
        .unwrap()
        .models
        .iter()
        .all(|model| model.request_model != "primary-only"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn prepared_sync_cannot_write_into_a_new_config_directory() {
    let first_dir = temp_dir();
    let second_dir = temp_dir();
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let http = Arc::new(PausedDirectory {
        started: started.clone(),
        release: release.clone(),
    });
    let (client, client_config) = LlmClientBuilder::with_transport(http, &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&first_dir).await.unwrap();
    client_config
        .set_tracked_models("acme", ["primary-only".to_owned()])
        .await
        .unwrap();
    client_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();

    let operation = client_config
        .prepare_provider_sync("primary", Some(&Secret::new("primary-key".into())))
        .await
        .unwrap();
    let fetch = tokio::spawn(operation.fetch());
    started.notified().await;
    client_config.set_config_dir(&second_dir).await.unwrap();
    release.notify_one();
    let result = fetch.await.unwrap().unwrap();

    assert!(matches!(
        client_config.apply_provider_sync(result).await,
        Err(ProviderStoreError::ProfileChanged(name)) if name == "primary"
    ));
    assert!(client.snapshot().provider("primary").is_none());
    assert!(!second_dir.join("providers.json").exists());
    std::fs::remove_dir_all(first_dir).unwrap();
    std::fs::remove_dir_all(second_dir).unwrap();
}

#[tokio::test]
async fn prepared_sync_rejects_a_config_directory_round_trip() {
    let first_dir = temp_dir();
    let second_dir = temp_dir();
    let http = Arc::new(AccountDirectory);
    let (client, client_config) = LlmClientBuilder::with_transport(http, &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&first_dir).await.unwrap();
    client_config
        .set_tracked_models("acme", ["primary-only".to_owned()])
        .await
        .unwrap();
    client_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();
    let operation = client_config
        .prepare_provider_sync("primary", Some(&Secret::new("primary-key".into())))
        .await
        .unwrap();
    let result = operation.fetch().await.unwrap();

    client_config.set_config_dir(&second_dir).await.unwrap();
    client_config.set_config_dir(&first_dir).await.unwrap();
    assert!(matches!(
        client_config.apply_provider_sync(result).await,
        Err(ProviderStoreError::ProfileChanged(name)) if name == "primary"
    ));
    assert!(client
        .snapshot()
        .provider("primary")
        .unwrap()
        .models
        .iter()
        .all(|model| model.request_model != "primary-only"));
    std::fs::remove_dir_all(first_dir).unwrap();
    std::fs::remove_dir_all(second_dir).unwrap();
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
    let (_syncing, syncing_config) = LlmClientBuilder::with_transport(http, &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    syncing_config.set_config_dir(&dir).await.unwrap();
    syncing_config
        .set_tracked_models("acme", ["primary-only".to_owned()])
        .await
        .unwrap();
    syncing_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();
    let (_editing, editing_config) =
        LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    editing_config.set_config_dir(&dir).await.unwrap();

    let task = tokio::spawn(async move {
        syncing_config
            .sync_provider("primary", Some(&Secret::new("primary-key".into())))
            .await
    });
    started.notified().await;
    let mut changed = profile("primary", 0, false);
    changed.base_url = "https://another-account.example/v1".into();
    editing_config.add_provider(changed).await.unwrap();
    release.notify_one();
    assert!(matches!(
        task.await.unwrap(),
        Err(ProviderStoreError::ProfileChanged(name)) if name == "primary"
    ));

    let (restored, restored_config) =
        LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    assert_eq!(
        restored.snapshot().provider("primary").unwrap().base_url,
        "https://another-account.example/v1"
    );
    assert!(restored
        .snapshot()
        .provider("primary")
        .unwrap()
        .models
        .iter()
        .all(|model| model.request_model != "primary-only"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn sync_provider_does_not_block_runtime_while_waiting_for_store_lock() {
    let dir = temp_dir();
    let (_client, client_config) =
        LlmClientBuilder::with_transport(Arc::new(AccountDirectory), &[])
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config
        .set_tracked_models("acme", ["primary-only".to_owned()])
        .await
        .unwrap();
    client_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();

    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join(".providers.json.lock"))
        .unwrap();
    lock.lock().unwrap();

    let (heartbeat_tx, heartbeat_rx) = std::sync::mpsc::channel();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        let _ = heartbeat_tx.send(());
    });

    let release_lock = tokio::task::spawn_blocking(move || {
        let heartbeat_ran_while_locked = heartbeat_rx.recv_timeout(Duration::from_secs(2)).is_ok();
        drop(lock);
        heartbeat_ran_while_locked
    });
    let sync = tokio::spawn(async move {
        client_config
            .sync_provider("primary", Some(&Secret::new("primary-key".into())))
            .await
    });

    let heartbeat_ran_while_locked = release_lock.await.unwrap();
    sync.await.unwrap().unwrap();
    assert!(
        heartbeat_ran_while_locked,
        "the Tokio heartbeat must progress while the provider-store lock is held"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn whitelist_update_does_not_restore_another_clients_removed_model() {
    let dir = temp_dir();
    let http = Arc::new(AccountDirectory);
    let (_first, first_config) = LlmClientBuilder::with_transport(http.clone(), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    first_config.set_config_dir(&dir).await.unwrap();
    first_config
        .set_tracked_models("acme", ["shared".to_owned(), "another".to_owned()])
        .await
        .unwrap();
    let mut both = profile("primary", 0, false);
    let mut another = both.models[0].clone();
    another.display_model = "another".into();
    another.request_model = "another".into();
    another.billing_model = "another".into();
    both.models.push(another);
    first_config.add_provider(both).await.unwrap();

    let (_second, second_config) = LlmClientBuilder::with_transport(http.clone(), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    second_config.set_config_dir(&dir).await.unwrap();
    second_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();
    first_config
        .set_tracked_models("acme", ["shared".to_owned(), "another".to_owned()])
        .await
        .unwrap();
    first_config
        .set_tracked_models("acme", ["shared".to_owned()])
        .await
        .unwrap();
    first_config
        .set_tracked_models("acme", ["shared".to_owned(), "another".to_owned()])
        .await
        .unwrap();

    let (restored, restored_config) = LlmClientBuilder::with_transport(http, &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    assert!(restored
        .snapshot()
        .provider("primary")
        .unwrap()
        .models
        .iter()
        .all(|model| model.request_model != "another"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn restoring_builtin_overrides_a_same_named_builder_profile() {
    let dir = temp_dir();
    let preset = lingxi_llm_client::builtin_providers()
        .unwrap()
        .into_iter()
        .find(|profile| profile.profile_name == "openai")
        .unwrap();
    let mut custom = preset.clone();
    custom.base_url = "https://custom.example/v1".into();
    custom.background = lingxi_llm_client::protocol::ServiceSetting::Disabled;
    let http = Arc::new(AccountDirectory);
    let (client, client_config) =
        LlmClientBuilder::with_transport(http.clone(), std::slice::from_ref(&custom))
            .with_region(lingxi_llm_client::protocol::Region::International)
            .build_managed()
            .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config.remove_provider("openai").await.unwrap();
    client_config.restore_builtin("openai").await.unwrap();
    assert_eq!(
        client.snapshot().provider("openai").unwrap().base_url,
        preset.base_url
    );
    client_config
        .set_tracked_models("openai", ["gpt-4o".to_owned()])
        .await
        .unwrap();
    assert_eq!(
        client.snapshot().provider("openai").unwrap().base_url,
        preset.base_url
    );

    let (restored, restored_config) = LlmClientBuilder::with_transport(http, &[custom])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    assert_eq!(
        restored.snapshot().provider("openai").unwrap().base_url,
        preset.base_url
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn region_filtering_preserves_other_accounts_across_sync_updates_and_restart() {
    use lingxi_llm_client::protocol::Region;
    let dir = temp_dir();
    let http = Arc::new(AccountDirectory);
    let mut primary = profile("primary", 0, false);
    primary.regions = vec![Region::ChinaMainland];
    let mut spare = profile("spare", 1, true);
    spare.regions = vec![Region::International];
    let (cn, cn_config) = LlmClientBuilder::with_transport(http.clone(), &[])
        .with_region(Region::ChinaMainland)
        .build_managed()
        .unwrap();
    cn_config.set_config_dir(&dir).await.unwrap();
    cn_config.add_provider(primary).await.unwrap();
    cn_config
        .set_tracked_models(
            "acme",
            ["shared", "primary-only", "spare-only"].map(str::to_owned),
        )
        .await
        .unwrap();
    cn_config.add_provider(spare).await.unwrap();
    for (name, key) in [("primary", "primary-key"), ("spare", "spare-key")] {
        cn_config
            .sync_provider(name, Some(&Secret::new(key.into())))
            .await
            .unwrap();
    }
    assert_eq!(
        cn.snapshot().provider("primary").unwrap().regions,
        vec![Region::ChinaMainland]
    );
    assert_eq!(
        cn.snapshot().provider("spare").unwrap().regions,
        vec![Region::International]
    );
    assert_eq!(cn.providers().len(), 1);
    assert!(cn
        .models()
        .iter()
        .all(|m| m.profile_name == "primary" && m.request_model != "untracked"));
    assert!(cn.resolve_in("spare-only", Some("spare")).is_err());
    cn_config
        .set_model_visibility("primary", "primary-only", false)
        .await
        .unwrap();
    assert!(!cn
        .models()
        .iter()
        .any(|m| m.request_model == "primary-only"));
    let (intl, intl_config) = LlmClientBuilder::with_transport(http.clone(), &[])
        .with_region(Region::International)
        .build_managed()
        .unwrap();
    intl_config.set_config_dir(&dir).await.unwrap();
    assert_eq!(intl.snapshot().profiles().len(), 2);
    assert_eq!(intl.providers().len(), 1);
    assert!(
        intl.models().is_empty(),
        "hidden regional spare remains hidden"
    );
    let intl_view = intl.snapshot();
    let mut spare = intl_view.provider("spare").unwrap().clone();
    spare.connection.hidden = false;
    intl_config.add_provider(spare).await.unwrap();
    assert!(intl
        .models()
        .iter()
        .any(|m| m.request_model == "spare-only"));
    assert!(intl.models().iter().all(|m| m.profile_name == "spare"
        && m.request_model != "primary-only"
        && m.request_model != "untracked"));
    let (restarted, restarted_config) = LlmClientBuilder::with_transport(http, &[])
        .with_region(Region::ChinaMainland)
        .build_managed()
        .unwrap();
    restarted_config.set_config_dir(&dir).await.unwrap();
    assert_eq!(restarted.snapshot().profiles().len(), 2);
    assert_eq!(restarted.providers()[0].profile_name, "primary");
    assert!(!restarted
        .models()
        .iter()
        .any(|m| m.request_model == "primary-only"));
    let saved: Value =
        serde_json::from_slice(&std::fs::read(dir.join("providers.json")).unwrap()).unwrap();
    assert!(
        saved.get("region").is_none(),
        "the client region must not become shared state"
    );
    assert_eq!(saved["providers"].as_array().unwrap().len(), 2);
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn regional_changes_in_another_client_invalidate_pending_sync() {
    use lingxi_llm_client::protocol::Region;
    let dir = temp_dir();
    let http = Arc::new(MutableDirectory::new(json!({"data":[{"id":"shared"}]})));
    let (c, c_config) = LlmClientBuilder::with_transport(http.clone(), &[])
        .with_region(Region::International)
        .build_managed()
        .unwrap();
    c_config.set_config_dir(&dir).await.unwrap();
    c_config
        .add_provider(profile("primary", 0, false))
        .await
        .unwrap();
    c_config
        .set_tracked_models("acme", ["shared".to_owned()])
        .await
        .unwrap();
    let fetched = c_config
        .prepare_provider_sync("primary", Some(&Secret::new("key".into())))
        .await
        .unwrap()
        .fetch()
        .await
        .unwrap();
    let (other, other_config) = LlmClientBuilder::with_transport(http, &[])
        .with_region(Region::ChinaMainland)
        .build_managed()
        .unwrap();
    other_config.set_config_dir(&dir).await.unwrap();
    let other_view = other.snapshot();
    let mut p = other_view.provider("primary").unwrap().clone();
    p.regions = vec![Region::ChinaMainland];
    other_config.add_provider(p).await.unwrap();
    assert!(matches!(
        c_config.apply_provider_sync(fetched).await,
        Err(ProviderStoreError::ProfileChanged(_))
    ));
    c_config.set_config_dir(&dir).await.unwrap();
    assert!(c.providers().is_empty());
    assert!(c.resolve("shared").is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn temporarily_incompatible_models_keep_metadata_through_reload_and_recovery() {
    for reload in [false, true] {
        let dir = temp_dir();
        let http = Arc::new(MutableDirectory::new(gemini_page(&["embedContent"])));
        let mut original = gemini_profile("gemini");
        let model = &mut original.models[0];
        model.hidden = true;
        model.aliases = vec!["saved-alias".into()];
        model.billing_model = "billing-alias".into();
        model.capability_support.get_or_insert_default().tools =
            lingxi_llm_client::protocol::CapabilitySupport::Supported;
        model.metadata.context_window_tokens = Some(8192);
        model.pricing = Some(
            serde_json::from_value(json!({"input_per_million":2.0,"output_per_million":8.0}))
                .unwrap(),
        );
        model.info.pricing = model.pricing.clone();
        let build = |profiles: &[ProviderProfile]| {
            LlmClientBuilder::with_transport(http.clone(), profiles)
                .with_region(lingxi_llm_client::protocol::Region::International)
                .build_managed()
                .unwrap()
        };
        let (mut client, mut client_config) = build(&[original]);
        let client_view = client.snapshot();
        let expected = client_view.provider("gemini").unwrap().models[0].clone();
        client_config.set_config_dir(&dir).await.unwrap();
        client_config
            .set_tracked_models(
                "google",
                [expected.request_model.clone(), "gemini-2.5-pro".into()],
            )
            .await
            .unwrap();
        client_config.sync_provider("gemini", None).await.unwrap();
        assert!(client
            .resolve_in(&expected.request_model, Some("gemini"))
            .is_err());
        assert!(matches!(
            client_config
                .set_model_visibility("gemini", &expected.request_model, true)
                .await,
            Err(ProviderStoreError::UnknownModel { .. })
        ));
        client_config
            .set_model_visibility("gemini", "gemini-2.5-pro", false)
            .await
            .unwrap();
        let persisted: Value =
            serde_json::from_slice(&std::fs::read(dir.join("providers.json")).unwrap()).unwrap();
        assert_eq!(persisted["version"], 3);
        let saved = client_config
            .configured_models("gemini")
            .await
            .unwrap()
            .into_iter()
            .find(|m| m.model.request_model == expected.request_model)
            .unwrap();
        assert!(!saved.compatible);
        assert_eq!(saved.model, expected);
        if reload {
            (client, client_config) = build(&[]);
            client_config.set_config_dir(&dir).await.unwrap();
        }
        http.set_body(json!({"models":[{"name":format!("models/{}", expected.request_model)}]}));
        client_config.sync_provider("gemini", None).await.unwrap();
        assert!(client
            .resolve_in(&expected.request_model, Some("gemini"))
            .is_err());
        http.set_body(gemini_page(&["generateContent"]));
        client_config.sync_provider("gemini", None).await.unwrap();
        let client_view = client.snapshot();
        let restored = client_view
            .provider("gemini")
            .unwrap()
            .models
            .iter()
            .find(|m| m.request_model == expected.request_model)
            .unwrap();
        assert_eq!(
            restored, &expected,
            "metadata must survive availability changes (reload={reload})"
        );
        assert!(client.resolve_in("saved-alias", Some("gemini")).is_ok());
        assert!(!client
            .models()
            .iter()
            .any(|m| m.request_model == expected.request_model));
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[tokio::test]
async fn untracking_an_unavailable_model_removes_its_persisted_metadata() {
    let dir = temp_dir();
    let http = Arc::new(MutableDirectory::new(gemini_page(&["embedContent"])));
    let p = gemini_profile("gemini");
    let (_client, client_config) = LlmClientBuilder::with_transport(http.clone(), &[p])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config
        .set_tracked_models("google", ["gemini-embedding-001".into()])
        .await
        .unwrap();
    client_config.sync_provider("gemini", None).await.unwrap();
    client_config
        .untrack_model("google", "gemini-embedding-001")
        .await
        .unwrap();
    let persisted: Value =
        serde_json::from_slice(&std::fs::read(dir.join("providers.json")).unwrap()).unwrap();
    assert!(persisted["providers"][0]["models"]
        .as_array()
        .unwrap()
        .is_empty());
    let (restored, restored_config) = LlmClientBuilder::with_transport(http.clone(), &[])
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    restored_config.set_config_dir(&dir).await.unwrap();
    restored_config
        .set_tracked_models("google", ["gemini-embedding-001".into()])
        .await
        .unwrap();
    assert!(restored
        .resolve_in("gemini-embedding-001", Some("gemini"))
        .is_err());
    std::fs::remove_dir_all(dir).unwrap();
}
