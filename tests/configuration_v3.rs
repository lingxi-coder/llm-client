use async_trait::async_trait;
use lingxi_llm_client::{
    configuration::{FieldOverride, ModelField},
    protocol::*,
    *,
};
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "llm-v3-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn bytes(&self) -> Vec<u8> {
        std::fs::read(self.0.join("providers.json")).unwrap()
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[derive(Default)]
struct Http(Mutex<Value>);
#[async_trait]
impl Transport for Http {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: serde_json::to_vec(&*self.0.lock().unwrap()).unwrap().into(),
        }
        .into())
    }
}
fn model(id: &str, price: f64) -> ModelProfile {
    let mut model: ModelProfile = serde_json::from_value(json!({"display_model":id,"request_model":id,"billing_model":id,"description":"catalog","pricing":{"input_per_million":price},"metadata":{"contextWindowTokens":100}})).unwrap();
    model.info.pricing = model.pricing.clone();
    model
}
fn profile(models: Vec<ModelProfile>) -> ProviderProfile {
    serde_json::from_value(json!({"profile_name":"p","provider_id":"acme","protocol":"open_ai_chat","base_url":"https://example.test/v1","auth":"none","models":models})).unwrap()
}
fn client(profiles: &[ProviderProfile], http: Arc<Http>) -> (LlmClient, ClientConfigManager) {
    LlmClientBuilder::with_transport(http, profiles)
        .with_region(Region::International)
        .build_managed()
        .unwrap()
}
async fn row(c_config: &ClientConfigManager, wire: &str) -> String {
    c_config
        .configured_models("p")
        .await
        .unwrap()
        .into_iter()
        .find(|r| r.model.request_model == wire)
        .unwrap()
        .row_id
}

#[tokio::test]
async fn inference_facts_and_prices_preserve_override_provenance_across_sync_and_reload() {
    let d = Dir::new();
    let http = Arc::new(Http(Mutex::new(
        json!({"data":[{"id":"m","reasoning":{"supported_efforts":["low","high"]}}]}),
    )));
    let mut m = model("m", 2.0);
    m.info.features = InferenceFeatures {
        thinking: CapabilitySupport::Supported,
        fast: CapabilitySupport::Supported,
        budget: BudgetSupport {
            support: CapabilitySupport::Supported,
            min_tokens: Some(1024),
            max_tokens: Some(8192),
            ..Default::default()
        },
        ..Default::default()
    };
    let p = profile(vec![m]);
    let (c, c_config) = client(std::slice::from_ref(&p), http.clone());
    c_config.set_config_dir(&d.0).await.unwrap();
    c_config
        .set_tracked_models("acme", ["m".into()])
        .await
        .unwrap();
    c_config.sync_provider("p", None).await.unwrap();
    let id = row(&c_config, "m").await;
    let c_view = c.snapshot();
    let features = &c_view.profile("p").unwrap().models[0].info.features;
    assert_eq!(features.budget.min_tokens, Some(1024));
    assert_eq!(features.fast, CapabilitySupport::Supported);
    assert_eq!(
        features.effort.levels,
        Some(vec![ReasoningEffort::Low, ReasoningEffort::High])
    );
    let mut override_features = features.clone();
    override_features.fast = CapabilitySupport::Unsupported;
    c_config
        .set_model_override(
            "p",
            &id,
            ModelField::InferenceFeatures,
            FieldOverride::Set(serde_json::to_value(&override_features).unwrap()),
        )
        .await
        .unwrap();
    c_config.set_model_override("p", &id, ModelField::Pricing, FieldOverride::Set(json!({"currency":"CNY","input_per_million":7.0,"rules":[{"service_tier":"fast","multiplier":{"factor":1.5,"buckets":["input"]}}]}))).await.unwrap();
    *http.0.lock().unwrap() = json!({"data":[{"id":"m","reasoning":{}}]});
    c_config.sync_provider("p", None).await.unwrap();
    let (reloaded, reloaded_config) = client(&[p], http);
    reloaded_config.set_config_dir(&d.0).await.unwrap();
    let reloaded_view = reloaded.snapshot();
    let row = &reloaded_view.profile("p").unwrap().models[0];
    assert_eq!(row.info.features, override_features);
    assert_eq!(row.info.pricing, row.pricing);
    assert_eq!(
        row.info.pricing.as_ref().unwrap().currency.as_deref(),
        Some("CNY")
    );
    reloaded_config
        .set_model_override(
            "p",
            &id,
            ModelField::InferenceFeatures,
            FieldOverride::Clear,
        )
        .await
        .unwrap();
    assert_eq!(
        reloaded.snapshot().profile("p").unwrap().models[0]
            .info
            .features,
        InferenceFeatures {
            effort: EffortSupport {
                with_disabled_thinking: CapabilitySupport::Unsupported,
                ..Default::default()
            },
            ..Default::default()
        }
    );
    reloaded_config
        .set_model_override(
            "p",
            &id,
            ModelField::InferenceFeatures,
            FieldOverride::Inherit,
        )
        .await
        .unwrap();
    let reloaded_view = reloaded.snapshot();
    let features = &reloaded_view.profile("p").unwrap().models[0].info.features;
    assert_eq!(features.fast, CapabilitySupport::Supported);
    assert_eq!(
        features.effort.levels,
        Some(vec![ReasoningEffort::Low, ReasoningEffort::High])
    );
    reloaded_config
        .set_model_override("p", &id, ModelField::Pricing, FieldOverride::Clear)
        .await
        .unwrap();
    assert!(reloaded.snapshot().profile("p").unwrap().models[0]
        .info
        .pricing
        .is_none());
    reloaded_config
        .set_model_override("p", &id, ModelField::Pricing, FieldOverride::Inherit)
        .await
        .unwrap();
    assert_eq!(
        reloaded.snapshot().profile("p").unwrap().models[0]
            .info
            .pricing
            .as_ref()
            .unwrap()
            .input_per_million,
        Some(2.0)
    );
}

#[tokio::test]
async fn inherited_fields_follow_catalog_and_reset_is_field_specific() {
    let d = Dir::new();
    let http = Arc::new(Http::default());
    let (_a, a_config) = client(&[profile(vec![model("m", 1.)])], http.clone());
    a_config.set_config_dir(&d.0).await.unwrap();
    a_config
        .set_tracked_models("acme", ["m".into()])
        .await
        .unwrap();
    let id = row(&a_config, "m").await;
    a_config
        .set_model_override(
            "p",
            &id,
            ModelField::Description,
            FieldOverride::Set(json!("mine")),
        )
        .await
        .unwrap();
    let mut newer = model("m", 4.);
    newer.description = Some("new catalog".into());
    newer.metadata.context_window_tokens = Some(300);
    let (b, b_config) = client(&[profile(vec![newer.clone()])], http.clone());
    let b_view = b.snapshot();
    let newer = b_view.profile("p").unwrap().models[0].clone();
    b_config.set_config_dir(&d.0).await.unwrap();
    let b_view = b.snapshot();
    let current = &b_view.profile("p").unwrap().models[0];
    assert_eq!(current.description.as_deref(), Some("mine"));
    assert_eq!(current.pricing, newer.pricing);
    assert_eq!(current.metadata.context_window_tokens, Some(300));
    b_config
        .clear_model_override("p", &id, ModelField::Description)
        .await
        .unwrap();
    assert_eq!(b.snapshot().profile("p").unwrap().models[0], newer);
    let before = d.bytes();
    assert!(b_config
        .set_model_override(
            "p",
            &id,
            ModelField::Hidden,
            FieldOverride::Set(json!("wrong type"))
        )
        .await
        .is_err());
    assert_eq!(d.bytes(), before);
    assert_eq!(b.snapshot().profile("p").unwrap().models[0], newer);
    let (empty, empty_config) = client(&[], http);
    empty_config.set_config_dir(&d.0).await.unwrap();
    assert_eq!(
        empty.snapshot().profile("p").unwrap().models[0],
        newer,
        "successful write refreshes fallback"
    );
}

#[tokio::test]
async fn fallback_is_used_only_if_the_whole_definition_is_missing() {
    let d = Dir::new();
    let http = Arc::new(Http::default());
    let (_c, c_config) = client(
        &[profile(vec![model("a", 1.), model("b", 2.)])],
        http.clone(),
    );
    c_config.set_config_dir(&d.0).await.unwrap();
    c_config
        .set_tracked_models("acme", ["a".into()])
        .await
        .unwrap();
    let (absent, absent_config) = client(&[], http.clone());
    absent_config.set_config_dir(&d.0).await.unwrap();
    assert_eq!(absent.snapshot().profile("p").unwrap().models.len(), 1);
    assert!(absent.resolve("a").is_ok());
    let (changed, changed_config) = client(&[profile(vec![model("b", 3.)])], http);
    changed_config.set_config_dir(&d.0).await.unwrap();
    assert!(changed.resolve("a").is_err());
    assert!(
        changed.resolve("b").is_ok(),
        "allowlist does not change explicit routing"
    );
}
#[tokio::test]
async fn duplicate_wire_rows_keep_order_identity_and_independent_settings() {
    let d = Dir::new();
    let mut second = model("m", 2.);
    second.display_model = "second".into();
    let http = Arc::new(Http::default());
    let (_c, c_config) = client(
        &[profile(vec![model("m", 1.), second.clone()])],
        http.clone(),
    );
    c_config.set_config_dir(&d.0).await.unwrap();
    c_config
        .set_tracked_models("acme", ["m".into()])
        .await
        .unwrap();
    let rows = c_config.configured_models("p").await.unwrap();
    assert_ne!(rows[0].row_id, rows[1].row_id);
    let second = rows[1].model.clone();
    let before = d.bytes();
    assert!(c_config
        .set_model_visibility("p", "m", false)
        .await
        .is_err());
    assert_eq!(before, d.bytes());
    c_config
        .set_model_override(
            "p",
            &rows[0].row_id,
            ModelField::Hidden,
            FieldOverride::Set(json!(true)),
        )
        .await
        .unwrap();
    let (restored, restored_config) = client(&[], http);
    restored_config.set_config_dir(&d.0).await.unwrap();
    assert!(restored.snapshot().profile("p").unwrap().models[0].hidden);
    assert_eq!(restored.snapshot().profile("p").unwrap().models[1], second);
}
#[tokio::test]
async fn replacing_an_inherited_row_does_not_restore_the_original_or_misapply_allowlist() {
    for tracked in ["old", "new"] {
        let d = Dir::new();
        let base = profile(vec![model("old", 1.)]);
        let http = Arc::new(Http::default());
        let (c, c_config) = client(std::slice::from_ref(&base), http.clone());
        c_config.set_config_dir(&d.0).await.unwrap();
        c_config
            .set_tracked_models("acme", [tracked.into()])
            .await
            .unwrap();
        let id = row(&c_config, "old").await;
        c_config
            .replace_model("p", &id, model("new", 2.))
            .await
            .unwrap();
        assert_eq!(c.snapshot().profile("p").unwrap().models.len(), 1);
        assert!(c.resolve("old").is_err());
        assert!(c.resolve("new").is_ok());
        let (restored, restored_config) = client(&[base], http);
        restored_config.set_config_dir(&d.0).await.unwrap();
        assert!(restored.resolve("old").is_err());
        assert_eq!(restored.resolve("new").is_ok(), tracked == "new");
    }
}
#[tokio::test]
async fn stale_model_replacement_cannot_revive_another_clients_deleted_row() {
    let d = Dir::new();
    let base = profile(vec![model("m", 1.)]);
    let http = Arc::new(Http::default());
    let (_a, a_config) = client(std::slice::from_ref(&base), http.clone());
    a_config.set_config_dir(&d.0).await.unwrap();
    a_config
        .set_tracked_models("acme", ["m".into(), "n".into()])
        .await
        .unwrap();
    let id = row(&a_config, "m").await;
    let (_b, b_config) = client(&[base], http);
    b_config.set_config_dir(&d.0).await.unwrap();
    b_config
        .add_provider(profile(vec![model("n", 2.)]))
        .await
        .unwrap();
    let before = d.bytes();
    assert!(a_config
        .replace_model("p", &id, model("m", 3.))
        .await
        .is_err());
    assert_eq!(d.bytes(), before);
}
#[tokio::test]
async fn observed_rows_join_later_static_definitions_without_duplicates() {
    let d = Dir::new();
    let http = Arc::new(Http(Mutex::new(
        json!({"data":[{"id":"m","description":"observed"}]}),
    )));
    let (_a, a_config) = client(&[profile(vec![])], http.clone());
    a_config.set_config_dir(&d.0).await.unwrap();
    a_config
        .set_tracked_models("acme", ["m".into(), "n".into()])
        .await
        .unwrap();
    a_config.sync_provider("p", None).await.unwrap();
    let id = row(&a_config, "m").await;
    let (b, b_config) = client(&[profile(vec![model("m", 7.)])], http.clone());
    b_config.set_config_dir(&d.0).await.unwrap();
    assert_eq!(b.snapshot().profile("p").unwrap().models.len(), 1);
    assert_eq!(row(&b_config, "m").await, id);
    assert_eq!(
        b.snapshot().profile("p").unwrap().models[0].pricing,
        model("m", 7.).pricing
    );
    b_config
        .replace_model("p", &id, model("n", 1.))
        .await
        .unwrap();
    assert!(b.resolve("m").is_err());
    b_config.sync_provider("p", None).await.unwrap();
    assert!(b.resolve("m").is_ok());
    assert!(b.resolve("n").is_ok());
}
#[tokio::test]
async fn unsupported_configuration_versions_are_rejected_without_writes() {
    let d = Dir::new();
    let bytes = br#"{"version":2,"providers":[]}"#;
    std::fs::write(d.0.join("providers.json"), bytes).unwrap();
    let (c, c_config) = client(&[profile(vec![model("m", 1.)])], Arc::new(Http::default()));
    assert!(matches!(
        c_config.set_config_dir(&d.0).await,
        Err(ProviderStoreError::UnsupportedVersion(2))
    ));
    assert_eq!(d.bytes(), bytes);
    assert!(c.resolve("m").is_ok());
    assert!(!d.0.join("providers.v2.json.bak").exists());
}
#[tokio::test]
async fn explicit_overrides_outrank_observations_and_reset_keeps_observations() {
    let d = Dir::new();
    let http = Arc::new(Http(Mutex::new(
        json!({"data":[{"id":"m","description":"observed"}]}),
    )));
    let (c, c_config) = client(&[profile(vec![model("m", 1.)])], http);
    c_config.set_config_dir(&d.0).await.unwrap();
    c_config
        .set_tracked_models("acme", ["m".into()])
        .await
        .unwrap();
    let id = row(&c_config, "m").await;
    c_config
        .set_model_override(
            "p",
            &id,
            ModelField::Description,
            FieldOverride::Set(json!("user")),
        )
        .await
        .unwrap();
    c_config.sync_provider("p", None).await.unwrap();
    assert_eq!(
        c.snapshot().profile("p").unwrap().models[0]
            .description
            .as_deref(),
        Some("user")
    );
    c_config
        .clear_model_override("p", &id, ModelField::Description)
        .await
        .unwrap();
    assert_eq!(
        c.snapshot().profile("p").unwrap().models[0]
            .description
            .as_deref(),
        Some("observed")
    );
}
#[tokio::test]
async fn clearing_a_full_replacement_field_inherits_the_catalog() {
    let d = Dir::new();
    let (c, c_config) = client(&[profile(vec![model("m", 1.)])], Arc::new(Http::default()));
    c_config.set_config_dir(&d.0).await.unwrap();
    c_config
        .set_tracked_models("acme", ["m".into()])
        .await
        .unwrap();
    c_config
        .add_provider(profile(vec![model("m", 9.)]))
        .await
        .unwrap();
    let id = row(&c_config, "m").await;
    c_config
        .clear_model_override("p", &id, ModelField::Pricing)
        .await
        .unwrap();
    assert_eq!(
        c.snapshot().profile("p").unwrap().models[0].pricing,
        model("m", 1.).pricing
    );
}
#[tokio::test]
async fn removed_then_observed_then_restored_catalog_row_keeps_its_identity() {
    let d = Dir::new();
    let http = Arc::new(Http(Mutex::new(
        json!({"data":[{"id":"m","description":"observed"}]}),
    )));
    let base = profile(vec![model("m", 1.)]);
    let (_a, a_config) = client(std::slice::from_ref(&base), http.clone());
    a_config.set_config_dir(&d.0).await.unwrap();
    a_config
        .set_tracked_models("acme", ["m".into()])
        .await
        .unwrap();
    a_config
        .set_model_visibility("p", "m", false)
        .await
        .unwrap();
    let id = row(&a_config, "m").await;
    let (b, b_config) = client(&[profile(vec![])], http.clone());
    b_config.set_config_dir(&d.0).await.unwrap();
    assert!(b.resolve("m").is_err());
    b_config.sync_provider("p", None).await.unwrap();
    let (c, c_config) = client(&[base], http);
    c_config.set_config_dir(&d.0).await.unwrap();
    assert!(c.resolve("m").is_ok());
    let rows = c_config.configured_models("p").await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].row_id, id);
    assert!(rows[0].model.hidden);
    assert_eq!(rows[0].model.description.as_deref(), Some("observed"));
}
#[tokio::test]
async fn catalog_reordering_does_not_move_duplicate_wire_overrides() {
    let d = Dir::new();
    let mut a_model = model("m", 1.);
    a_model.display_model = "first".into();
    let mut b_model = model("m", 2.);
    b_model.display_model = "second".into();
    let http = Arc::new(Http::default());
    let (_a, a_config) = client(
        &[profile(vec![a_model.clone(), b_model.clone()])],
        http.clone(),
    );
    a_config.set_config_dir(&d.0).await.unwrap();
    a_config
        .set_tracked_models("acme", ["m".into()])
        .await
        .unwrap();
    let id = a_config.configured_models("p").await.unwrap()[0]
        .row_id
        .clone();
    a_config
        .set_model_override(
            "p",
            &id,
            ModelField::Hidden,
            FieldOverride::Set(json!(true)),
        )
        .await
        .unwrap();
    let (_b, b_config) = client(&[profile(vec![b_model, a_model])], http);
    b_config.set_config_dir(&d.0).await.unwrap();
    let rows = b_config.configured_models("p").await.unwrap();
    assert_eq!(rows[0].row_id, id);
    assert_eq!(rows[0].model.display_model, "first");
    assert!(rows[0].model.hidden);
    assert!(!rows[1].model.hidden);
}
#[tokio::test]
async fn builtin_references_are_explicit_and_recover_with_an_empty_builder() {
    let d = Dir::new();
    let http = Arc::new(Http::default());
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[]);
    builder.add_builtin_profile("openai").unwrap();
    let (c, c_config) = builder
        .with_region(Region::International)
        .build_managed()
        .unwrap();
    let c_view = c.snapshot();
    let wire = c_view.profile("openai").unwrap().models[0]
        .request_model
        .clone();
    c_config.set_config_dir(&d.0).await.unwrap();
    c_config
        .set_tracked_models("openai", [wire.clone()])
        .await
        .unwrap();
    let saved: Value = serde_json::from_slice(&d.bytes()).unwrap();
    assert_eq!(saved["providers"][0]["definition"]["source"], "builtin");
    assert_eq!(
        saved["providers"][0]["fallback"]["models"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let (empty, empty_config) = client(&[], http);
    empty_config.set_config_dir(&d.0).await.unwrap();
    assert!(empty.resolve_in(&wire, Some("openai")).is_ok());
}

#[tokio::test]
async fn observed_model_survives_static_adoption_and_withdrawal() {
    let d = Dir::new();
    let http = Arc::new(Http(Mutex::new(
        json!({"data":[{"id":"m","description":"observed"}]}),
    )));
    let (_discovered, discovered_config) = client(&[profile(vec![])], http.clone());
    discovered_config.set_config_dir(&d.0).await.unwrap();
    discovered_config
        .set_tracked_models("acme", ["m".into()])
        .await
        .unwrap();
    discovered_config.sync_provider("p", None).await.unwrap();
    let id = row(&discovered_config, "m").await;

    let (_adopted, adopted_config) = client(&[profile(vec![model("m", 7.)])], http.clone());
    adopted_config.set_config_dir(&d.0).await.unwrap();
    adopted_config
        .set_model_visibility("p", "m", false)
        .await
        .unwrap();
    assert_eq!(
        adopted_config.configured_models("p").await.unwrap()[0]
            .model
            .pricing,
        model("m", 7.).pricing
    );

    let (withdrawn, withdrawn_config) = client(&[profile(vec![])], http);
    withdrawn_config.set_config_dir(&d.0).await.unwrap();
    let rows = withdrawn_config.configured_models("p").await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].row_id, id);
    assert!(rows[0].compatible);
    assert!(rows[0].model.hidden);
    assert_eq!(rows[0].model.description.as_deref(), Some("observed"));
    assert!(
        rows[0].model.pricing.is_none(),
        "withdrawn static prices are not retained"
    );
    assert!(
        withdrawn.resolve("m").is_ok(),
        "hidden observed models remain addressable"
    );
}

#[tokio::test]
async fn reasoning_observations_can_switch_between_mandatory_and_optional() {
    let d = Dir::new();
    let http = Arc::new(Http::default());
    let (c, c_config) = client(&[profile(vec![model("m", 1.)])], http.clone());
    c_config.set_config_dir(&d.0).await.unwrap();
    c_config
        .set_tracked_models("acme", ["m".into()])
        .await
        .unwrap();
    let request: ChatRequest = serde_json::from_value(json!({
        "model":"m", "messages":[], "thinking":{"mode":"disabled"}
    }))
    .unwrap();
    for mandatory in [true, false, true] {
        *http.0.lock().unwrap() = json!({"data":[{"id":"m","reasoning":{
            "mandatory":mandatory, "default_enabled":mandatory, "supported_efforts":["none","low","high"]
        }}]});
        c_config.sync_provider("p", None).await.unwrap();
        let c_view = c.snapshot();
        let p = c_view.profile("p").unwrap();
        let features = &p.models[0].info.features;
        assert_eq!(
            features
                .modes
                .as_ref()
                .unwrap()
                .contains(&ThinkingMode::Disabled),
            !mandatory
        );
        assert_eq!(
            features
                .effort
                .levels
                .as_ref()
                .unwrap()
                .contains(&ReasoningEffort::None),
            !mandatory
        );
        let encoded = OpenAiChatCodec.encode_request(
            EncodeRequest::new(&request),
            &CodecContext::new(p, "m", RequestMode::Complete),
        );
        if mandatory {
            assert!(matches!(
                encoded,
                Err(LlmError::UnsupportedCapability { .. })
            ));
        } else {
            assert!(encoded.is_ok(), "{encoded:?}");
        }
    }
}

#[tokio::test]
async fn sparse_mandatory_updates_remove_conflicting_older_defaults() {
    let d = Dir::new();
    let http = Arc::new(Http(Mutex::new(json!({"data":[{"id":"m","reasoning":{
        "mandatory":false, "default_enabled":false, "default_effort":"none", "supported_efforts":["none","low"]
    }}]}))));
    let (c, c_config) = client(&[profile(vec![model("m", 1.)])], http.clone());
    c_config.set_config_dir(&d.0).await.unwrap();
    c_config
        .set_tracked_models("acme", ["m".into()])
        .await
        .unwrap();
    c_config.sync_provider("p", None).await.unwrap();
    assert_eq!(
        c.snapshot().profile("p").unwrap().models[0]
            .info
            .features
            .default_mode,
        Some(ThinkingMode::Disabled)
    );
    *http.0.lock().unwrap() = json!({"data":[{"id":"m","reasoning":{"mandatory":true}}]});
    c_config.sync_provider("p", None).await.unwrap();
    let c_view = c.snapshot();
    let features = &c_view.profile("p").unwrap().models[0].info.features;
    assert_eq!(features.modes, Some(vec![ThinkingMode::Enabled]));
    assert_eq!(features.default_mode, None);
    assert_eq!(features.effort.default, None);
    assert_eq!(features.effort.levels, Some(vec![ReasoningEffort::Low]));
    let saved: Value = serde_json::from_slice(&d.bytes()).unwrap();
    let observed = &saved["providers"][0]["observations"]["m"]["inference_features"];
    assert!(observed["default_mode"].is_null());
    assert!(observed["effort"]["default"].is_null());
    assert_eq!(observed["effort"]["levels"], json!(["low"]));
}

#[tokio::test]
async fn explicit_unrestricted_efforts_replace_prior_limits_and_survive_reload() {
    let d = Dir::new();
    let http = Arc::new(Http::default());
    let p = profile(vec![model("m", 1.0)]);
    let (c, c_config) = client(std::slice::from_ref(&p), http.clone());
    c_config.set_config_dir(&d.0).await.unwrap();
    c_config
        .set_tracked_models("acme", ["m".into()])
        .await
        .unwrap();
    for (levels, expected) in [
        (json!(["high"]), vec![ReasoningEffort::High]),
        (Value::Null, ReasoningEffort::ALL.to_vec()),
    ] {
        *http.0.lock().unwrap() =
            json!({"data":[{"id":"m","reasoning":{"supported_efforts":levels}}]});
        c_config.sync_provider("p", None).await.unwrap();
        assert_eq!(c.models()[0].info.features.effort.levels, Some(expected));
    }
    *http.0.lock().unwrap() = json!({"data":[{"id":"m","reasoning":{}}]});
    c_config.sync_provider("p", None).await.unwrap();
    let (restored, restored_config) = client(&[p], http);
    restored_config.set_config_dir(&d.0).await.unwrap();
    assert_eq!(
        restored.models()[0].info.features.effort.levels,
        Some(ReasoningEffort::ALL.to_vec())
    );
    let req: ChatRequest =
        serde_json::from_value(json!({"model":"m","messages":[],"thinking":{"effort":"low"}}))
            .unwrap();
    let restored_view = restored.snapshot();
    let p = restored_view.profile("p").unwrap();
    OpenAiChatCodec
        .validate_request(&req, &CodecContext::new(p, "m", RequestMode::Complete))
        .unwrap();
}

#[tokio::test]
async fn anthropic_partial_capabilities_merge_without_erasing_missing_facts() {
    let d = Dir::new();
    let http = Arc::new(Http::default());
    let mut m = model("m", 1.0);
    m.info.features = serde_json::from_value(json!({
        "thinking":"supported","modes":["enabled","disabled"],"fast":"supported",
        "effort":{"support":"supported","levels":["low","high","max"],"default":"high"}
    }))
    .unwrap();
    let mut p = profile(vec![m]);
    p.protocol = ProtocolFamily::AnthropicMessages;
    let (c, c_config) = client(std::slice::from_ref(&p), http.clone());
    c_config.set_config_dir(&d.0).await.unwrap();
    c_config
        .set_tracked_models("acme", ["m".into(), "new".into()])
        .await
        .unwrap();
    let caps = json!({"thinking":{"supported":true,"types":{"enabled":{"supported":false},"adaptive":{"supported":true}}},"effort":{"supported":true,"max":{"supported":false},"high":{"supported":true}}});
    *http.0.lock().unwrap() = json!({"data":[{"id":"m","capabilities":caps,"max_input_tokens":200000,"max_tokens":8192},{"id":"new","capabilities":caps}],"has_more":false});
    c_config.sync_provider("p", None).await.unwrap();
    let rows = c.models();
    let known = &rows.iter().find(|r| r.request_model == "m").unwrap();
    assert_eq!(known.context_window, Some(200000));
    assert_eq!(known.max_output_tokens, Some(8192));
    let f = &known.info.features;
    assert_eq!(
        f.supports_mode(ThinkingMode::Disabled),
        CapabilitySupport::Supported
    );
    assert_eq!(
        f.supports_mode(ThinkingMode::Adaptive),
        CapabilitySupport::Supported
    );
    assert_eq!(
        f.supports_mode(ThinkingMode::Enabled),
        CapabilitySupport::Unsupported
    );
    assert_eq!(
        f.supports_effort(Some(ThinkingMode::Adaptive), ReasoningEffort::Low),
        CapabilitySupport::Supported
    );
    assert_eq!(
        f.supports_effort(Some(ThinkingMode::Adaptive), ReasoningEffort::Max),
        CapabilitySupport::Unsupported
    );
    assert_eq!(f.fast, CapabilitySupport::Supported);
    let new = &rows
        .iter()
        .find(|r| r.request_model == "new")
        .unwrap()
        .info
        .features;
    assert_eq!(
        new.supports_mode(ThinkingMode::Disabled),
        CapabilitySupport::Unknown
    );
    assert_eq!(
        new.supports_mode(ThinkingMode::Enabled),
        CapabilitySupport::Unsupported
    );
    assert_eq!(
        new.supports_mode(ThinkingMode::Adaptive),
        CapabilitySupport::Supported
    );
    assert_eq!(
        new.supports_effort(Some(ThinkingMode::Adaptive), ReasoningEffort::Low),
        CapabilitySupport::Unknown
    );
    *http.0.lock().unwrap() = json!({"data":[{"id":"m","capabilities":{"effort":{"high":{"supported":false}}}},{"id":"new","capabilities":null}],"has_more":false});
    c_config.sync_provider("p", None).await.unwrap();
    let (restored, restored_config) = client(&[p], http);
    restored_config.set_config_dir(&d.0).await.unwrap();
    let features = restored
        .models()
        .into_iter()
        .find(|r| r.request_model == "m")
        .unwrap()
        .info
        .features;
    assert_eq!(features.effort.default, None);
    assert_eq!(features.effort.levels, Some(vec![ReasoningEffort::Low]));
    assert_eq!(
        features.supports_mode(ThinkingMode::Disabled),
        CapabilitySupport::Supported
    );
    assert_eq!(
        features.supports_mode(ThinkingMode::Adaptive),
        CapabilitySupport::Supported
    );
    let req: ChatRequest = serde_json::from_value(
        json!({"model":"m","messages":[],"thinking":{"mode":"adaptive","effort":"max"}}),
    )
    .unwrap();
    let restored_view = restored.snapshot();
    let profile = restored_view.profile("p").unwrap();
    assert!(matches!(
        AnthropicMessagesCodec.validate_request(
            &req,
            &CodecContext::new(profile, "m", RequestMode::Complete)
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));
    let id = row(&restored_config, "m").await;
    let mut user_features = features.clone();
    user_features.effort.with_disabled_thinking = CapabilitySupport::Unsupported;
    user_features
        .mode_support
        .entry(ThinkingMode::Adaptive)
        .or_default()
        .forced_tool_choice = CapabilitySupport::Unsupported;
    restored_config
        .set_model_override(
            "p",
            &id,
            ModelField::InferenceFeatures,
            FieldOverride::Set(serde_json::to_value(&user_features).unwrap()),
        )
        .await
        .unwrap();
    restored_config.set_config_dir(&d.0).await.unwrap();
    assert_eq!(
        restored
            .models()
            .into_iter()
            .find(|r| r.request_model == "m")
            .unwrap()
            .info
            .features,
        user_features
    );
    restored_config
        .set_model_override(
            "p",
            &id,
            ModelField::InferenceFeatures,
            FieldOverride::Clear,
        )
        .await
        .unwrap();
    assert_eq!(
        restored
            .models()
            .into_iter()
            .find(|r| r.request_model == "m")
            .unwrap()
            .info
            .features,
        InferenceFeatures {
            effort: EffortSupport {
                with_disabled_thinking: CapabilitySupport::Supported,
                ..Default::default()
            },
            ..Default::default()
        }
    );
    restored_config
        .set_model_override(
            "p",
            &id,
            ModelField::InferenceFeatures,
            FieldOverride::Inherit,
        )
        .await
        .unwrap();
    assert_eq!(
        restored
            .models()
            .into_iter()
            .find(|r| r.request_model == "m")
            .unwrap()
            .info
            .features,
        features
    );
}

#[tokio::test]
async fn service_routes_inherit_replace_and_disable_across_reload() {
    use lingxi_llm_client::embeddings::{
        EmbeddingApi, EmbeddingRoute, ServiceAuth, ServiceSetting,
    };
    use lingxi_llm_client::providers::openai::batches::{BatchApi, BatchRoute};
    use lingxi_llm_client::providers::openai::retrieval::{RetrievalApi, RetrievalRoute};
    let d = Dir::new();
    let mut definition = profile(vec![model("m", 1.)]);
    definition.embeddings = ServiceSetting::Enabled(EmbeddingRoute {
        api: EmbeddingApi::OpenAi,
        endpoint: "https://embedding.test/first".into(),
        auth: ServiceAuth::Bearer,
        models_endpoint: None,
        max_inputs: Some(10),
    });
    definition.retrieval = ServiceSetting::Enabled(RetrievalRoute {
        api: RetrievalApi::OpenAi,
        endpoint: "https://retrieval.test/first/vector_stores".into(),
        auth: ServiceAuth::Bearer,
    });
    definition.batches = ServiceSetting::Enabled(BatchRoute {
        api: BatchApi::OpenAi,
        endpoint: "https://batch.test/first/batches".into(),
        files_endpoint: "https://batch.test/first/files".into(),
        auth: ServiceAuth::Bearer,
    });
    let (_c, c_config) = client(&[definition.clone()], Arc::new(Http::default()));
    c_config.set_config_dir(&d.0).await.unwrap();
    c_config
        .set_model_override(
            "p",
            &c_config.configured_models("p").await.unwrap()[0].row_id,
            ModelField::Hidden,
            FieldOverride::Set(json!(true)),
        )
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&d.bytes()).unwrap()["version"],
        3
    );
    if let ServiceSetting::Enabled(route) = &mut definition.embeddings {
        route.endpoint = "https://embedding.test/second".into();
    }
    if let ServiceSetting::Enabled(route) = &mut definition.retrieval {
        route.endpoint = "https://retrieval.test/second/vector_stores".into();
    }
    if let ServiceSetting::Enabled(route) = &mut definition.batches {
        route.endpoint = "https://batch.test/second/batches".into();
        route.files_endpoint = "https://batch.test/second/files".into();
    }
    let (reloaded, reloaded_config) = client(&[definition.clone()], Arc::new(Http::default()));
    reloaded_config.set_config_dir(&d.0).await.unwrap();
    assert_eq!(
        reloaded.snapshot().profiles()[0].embeddings,
        definition.embeddings
    );
    assert_eq!(
        reloaded.snapshot().profiles()[0].retrieval,
        definition.retrieval
    );
    assert_eq!(
        reloaded.snapshot().profiles()[0].batches,
        definition.batches
    );
    let mut disabled = definition.clone();
    disabled.embeddings = ServiceSetting::Disabled;
    disabled.retrieval = ServiceSetting::Disabled;
    disabled.batches = ServiceSetting::Disabled;
    reloaded_config.add_provider(disabled).await.unwrap();
    let (reloaded, reloaded_config) = client(&[definition], Arc::new(Http::default()));
    reloaded_config.set_config_dir(&d.0).await.unwrap();
    assert!(matches!(
        reloaded.snapshot().profiles()[0].embeddings,
        ServiceSetting::Disabled
    ));
    assert!(matches!(
        reloaded.snapshot().profiles()[0].retrieval,
        ServiceSetting::Disabled
    ));
    assert!(matches!(
        reloaded.snapshot().profiles()[0].batches,
        ServiceSetting::Disabled
    ));
}

#[tokio::test]
async fn deferred_route_inherits_and_explicit_disable_survives_reload() {
    use lingxi_llm_client::embeddings::{ServiceAuth, ServiceSetting};
    use lingxi_llm_client::providers::xai::deferred::{DeferredApi, DeferredRoute};
    let d = Dir::new();
    let mut definition = profile(vec![model("m", 1.)]);
    definition.provider_id = "xai".into();
    definition.base_url = "https://api.x.ai/v1".into();
    definition.deferred = ServiceSetting::Enabled(DeferredRoute {
        api: DeferredApi::XaiChat,
        endpoint: "https://api.x.ai/v1/chat/completions".into(),
        results_endpoint: "https://api.x.ai/v1/chat/deferred-completion".into(),
        auth: ServiceAuth::Bearer,
    });
    let (_client, config) = client(&[definition.clone()], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    config
        .set_model_override(
            "p",
            &config.configured_models("p").await.unwrap()[0].row_id,
            ModelField::Hidden,
            FieldOverride::Set(json!(true)),
        )
        .await
        .unwrap();
    let (reloaded, config) = client(&[definition.clone()], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    assert_eq!(
        reloaded.snapshot().profiles()[0].deferred,
        definition.deferred
    );
    let mut disabled = definition.clone();
    disabled.deferred = ServiceSetting::Disabled;
    config.add_provider(disabled).await.unwrap();
    let (reloaded, config) = client(&[definition], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    assert!(matches!(
        reloaded.snapshot().profiles()[0].deferred,
        ServiceSetting::Disabled
    ));
}

#[tokio::test]
async fn background_route_inherits_and_explicit_disable_survives_reload() {
    use lingxi_llm_client::embeddings::{ServiceAuth, ServiceSetting};
    use lingxi_llm_client::providers::openai::background::BackgroundRoute;
    let d = Dir::new();
    let mut definition = profile(vec![model("m", 1.)]);
    definition.provider_id = "openai".into();
    definition.protocol = ProtocolFamily::OpenAiResponses;
    definition.base_url = "https://api.openai.com/v1".into();
    definition.background = ServiceSetting::Enabled(BackgroundRoute {
        endpoint: "https://api.openai.com/v1/responses".into(),
        auth: ServiceAuth::Bearer,
    });
    let (_client, config) = client(&[definition.clone()], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    let (reloaded, config) = client(&[definition.clone()], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    assert_eq!(
        reloaded.snapshot().profiles()[0].background,
        definition.background
    );
    let mut disabled = definition.clone();
    disabled.background = ServiceSetting::Disabled;
    config.add_provider(disabled).await.unwrap();
    let (reloaded, config) = client(&[definition], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    assert!(matches!(
        reloaded.snapshot().profiles()[0].background,
        ServiceSetting::Disabled
    ));
}

#[tokio::test]
async fn audio_route_inherits_and_explicit_disable_survives_reload() {
    use lingxi_llm_client::embeddings::{ServiceAuth, ServiceSetting};
    use lingxi_llm_client::providers::openai::audio::AudioRoute;
    let d = Dir::new();
    let mut definition = profile(vec![model("m", 1.)]);
    definition.provider_id = "openai".into();
    definition.audio = ServiceSetting::Enabled(AudioRoute {
        transcriptions_endpoint: "https://api.openai.com/v1/audio/transcriptions".into(),
        translations_endpoint: "https://api.openai.com/v1/audio/translations".into(),
        speech_endpoint: Some("https://api.openai.com/v1/audio/speech".into()),
        auth: ServiceAuth::Bearer,
    });
    let (_client, config) = client(&[definition.clone()], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    let (reloaded, config) = client(&[definition.clone()], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    assert_eq!(reloaded.snapshot().profiles()[0].audio, definition.audio);
    let mut disabled = definition.clone();
    disabled.audio = ServiceSetting::Disabled;
    config.add_provider(disabled).await.unwrap();
    let (reloaded, config) = client(&[definition], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    assert!(matches!(
        reloaded.snapshot().profiles()[0].audio,
        ServiceSetting::Disabled
    ));
}

#[tokio::test]
async fn interactions_route_inherits_and_explicit_disable_survives_reload() {
    use lingxi_llm_client::embeddings::{ServiceAuth, ServiceSetting};
    use lingxi_llm_client::providers::google::interactions::InteractionRoute;
    let d = Dir::new();
    let mut definition = profile(vec![model("m", 1.)]);
    definition.provider_id = "google".into();
    definition.interactions = ServiceSetting::Enabled(InteractionRoute {
        endpoint: "https://generativelanguage.googleapis.com/v1beta/interactions".into(),
        auth: ServiceAuth::ApiKey {
            header: "x-goog-api-key".into(),
        },
    });
    let (_client, config) = client(&[definition.clone()], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    let (reloaded, config) = client(&[definition.clone()], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    assert_eq!(
        reloaded.snapshot().profiles()[0].interactions,
        definition.interactions
    );
    let mut disabled = definition.clone();
    disabled.interactions = ServiceSetting::Disabled;
    config.add_provider(disabled).await.unwrap();
    let (reloaded, config) = client(&[definition], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    assert!(matches!(
        reloaded.snapshot().profiles()[0].interactions,
        ServiceSetting::Disabled
    ));
}

#[tokio::test]
async fn gemini_file_search_route_inherits_and_disable_survives_reload() {
    use lingxi_llm_client::embeddings::{ServiceAuth, ServiceSetting};
    use lingxi_llm_client::providers::google::file_search::GeminiFileSearchRoute;
    let d = Dir::new();
    let mut definition = profile(vec![model("m", 1.)]);
    definition.provider_id = "google".into();
    definition.protocol = ProtocolFamily::GeminiGenerateContent;
    definition.gemini_file_search = ServiceSetting::Enabled(GeminiFileSearchRoute {
        endpoint: "https://generativelanguage.googleapis.com/v1beta/fileSearchStores".into(),
        auth: ServiceAuth::ApiKey {
            header: "x-goog-api-key".into(),
        },
    });
    let (_client, config) = client(&[definition.clone()], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    let (reloaded, config) = client(&[definition.clone()], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    assert_eq!(
        reloaded.snapshot().profiles()[0].gemini_file_search,
        definition.gemini_file_search
    );
    let mut disabled = definition.clone();
    disabled.gemini_file_search = ServiceSetting::Disabled;
    config.add_provider(disabled).await.unwrap();
    let (reloaded, config) = client(&[definition], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    assert!(matches!(
        reloaded.snapshot().profiles()[0].gemini_file_search,
        ServiceSetting::Disabled
    ));
}

#[tokio::test]
async fn glm_knowledge_route_inherits_and_disable_survives_reload() {
    use lingxi_llm_client::embeddings::{ServiceAuth, ServiceSetting};
    use lingxi_llm_client::providers::zhipu::knowledge::GlmKnowledgeRoute;
    let d = Dir::new();
    let mut definition = profile(vec![model("m", 1.)]);
    definition.provider_id = "zhipu".into();
    definition.glm_knowledge = ServiceSetting::Enabled(GlmKnowledgeRoute {
        endpoint: "https://open.bigmodel.cn/api/llm-application/open".into(),
        auth: ServiceAuth::Bearer,
    });
    let (_client, config) = client(&[definition.clone()], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    let (reloaded, config) = client(&[definition.clone()], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    assert_eq!(
        reloaded.snapshot().profiles()[0].glm_knowledge,
        definition.glm_knowledge
    );
    let mut disabled = definition.clone();
    disabled.glm_knowledge = ServiceSetting::Disabled;
    config.add_provider(disabled).await.unwrap();
    let (reloaded, config) = client(&[definition], Arc::new(Http::default()));
    config.set_config_dir(&d.0).await.unwrap();
    assert!(matches!(
        reloaded.snapshot().profiles()[0].glm_knowledge,
        ServiceSetting::Disabled
    ));
}

#[tokio::test]
async fn foundry_deployment_identity_preserves_overrides_clear_and_inheritance_on_reload() {
    let d = Dir::new();
    let mut m = model("custom-deployment", 2.0);
    let original = FoundryDeployment {
        hosting: FoundryHosting::Azure,
        model_id: "claude-opus-5-5".into(),
    };
    m.foundry = Some(original.clone());
    let mut p = profile(vec![m]);
    p.protocol = ProtocolFamily::FoundryClaude;
    p.base_url = "https://example.services.ai.azure.com/anthropic".into();
    let http = Arc::new(Http::default());
    let (_, manager) = client(std::slice::from_ref(&p), http.clone());
    manager.set_config_dir(&d.0).await.unwrap();
    manager
        .set_tracked_models("acme", ["custom-deployment".into()])
        .await
        .unwrap();
    let id = row(&manager, "custom-deployment").await;
    let changed = FoundryDeployment {
        hosting: FoundryHosting::Anthropic,
        model_id: "claude-fable-5-1".into(),
    };
    manager
        .set_model_override(
            "p",
            &id,
            ModelField::Foundry,
            FieldOverride::Set(serde_json::to_value(&changed).unwrap()),
        )
        .await
        .unwrap();
    let configured = manager.configured_models("p").await.unwrap();
    assert_eq!(
        configured[0].model.foundry,
        Some(changed.clone()),
        "immediate configuration"
    );
    let saved: Value = serde_json::from_slice(&d.bytes()).unwrap();
    assert_eq!(
        saved["providers"][0]["models"][0]["overrides"]["foundry"],
        serde_json::to_value(&changed).unwrap(),
        "persisted override"
    );
    let (reloaded, manager) = client(&[p], http);
    manager.set_config_dir(&d.0).await.unwrap();
    let configured = manager.configured_models("p").await.unwrap();
    assert_eq!(
        configured[0].model.foundry,
        Some(changed.clone()),
        "reloaded configuration"
    );
    let snapshot = reloaded.snapshot();
    let selected = &snapshot.profile("p").unwrap().models[0];
    assert_eq!(selected.foundry, Some(changed));
    assert_eq!(selected.request_model, "custom-deployment");
    assert_eq!(selected.billing_model, "custom-deployment");
    manager
        .set_model_override("p", &id, ModelField::Foundry, FieldOverride::Clear)
        .await
        .unwrap();
    assert!(reloaded.snapshot().profile("p").unwrap().models[0]
        .foundry
        .is_none());
    manager
        .set_model_override("p", &id, ModelField::Foundry, FieldOverride::Inherit)
        .await
        .unwrap();
    assert_eq!(
        reloaded.snapshot().profile("p").unwrap().models[0].foundry,
        Some(original)
    );
}
