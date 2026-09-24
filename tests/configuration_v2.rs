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
            "llm-v2-{}-{}",
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
fn client(profiles: &[ProviderProfile], http: Arc<Http>) -> LlmClient {
    LlmClientBuilder::with_transport(http, profiles)
        .with_region(Region::International)
        .build()
        .unwrap()
}
fn row(c: &LlmClient, wire: &str) -> String {
    c.configured_models("p")
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
    let mut c = client(std::slice::from_ref(&p), http.clone());
    c.set_config_dir(&d.0).unwrap();
    c.set_tracked_models("acme", ["m".into()]).unwrap();
    c.sync_provider("p", None).await.unwrap();
    let id = row(&c, "m");
    let features = &c.provider("p").unwrap().models[0].info.features;
    assert_eq!(features.budget.min_tokens, Some(1024));
    assert_eq!(features.fast, CapabilitySupport::Supported);
    assert_eq!(
        features.effort.levels,
        Some(vec![ReasoningEffort::Low, ReasoningEffort::High])
    );
    let mut override_features = features.clone();
    override_features.fast = CapabilitySupport::Unsupported;
    c.set_model_override(
        "p",
        &id,
        ModelField::InferenceFeatures,
        FieldOverride::Set(serde_json::to_value(&override_features).unwrap()),
    )
    .unwrap();
    c.set_model_override("p", &id, ModelField::Pricing, FieldOverride::Set(json!({"currency":"CNY","input_per_million":7.0,"rules":[{"service_tier":"fast","multiplier":{"factor":1.5,"buckets":["input"]}}]}))).unwrap();
    *http.0.lock().unwrap() = json!({"data":[{"id":"m","reasoning":{}}]});
    c.sync_provider("p", None).await.unwrap();
    let mut reloaded = client(&[p], http);
    reloaded.set_config_dir(&d.0).unwrap();
    let row = &reloaded.provider("p").unwrap().models[0];
    assert_eq!(row.info.features, override_features);
    assert_eq!(row.info.pricing, row.pricing);
    assert_eq!(
        row.info.pricing.as_ref().unwrap().currency.as_deref(),
        Some("CNY")
    );
    reloaded
        .set_model_override(
            "p",
            &id,
            ModelField::InferenceFeatures,
            FieldOverride::Clear,
        )
        .unwrap();
    assert_eq!(
        reloaded.provider("p").unwrap().models[0].info.features,
        InferenceFeatures {
            effort: EffortSupport {
                with_disabled_thinking: CapabilitySupport::Unsupported,
                ..Default::default()
            },
            ..Default::default()
        }
    );
    reloaded
        .set_model_override(
            "p",
            &id,
            ModelField::InferenceFeatures,
            FieldOverride::Inherit,
        )
        .unwrap();
    let features = &reloaded.provider("p").unwrap().models[0].info.features;
    assert_eq!(features.fast, CapabilitySupport::Supported);
    assert_eq!(
        features.effort.levels,
        Some(vec![ReasoningEffort::Low, ReasoningEffort::High])
    );
    reloaded
        .set_model_override("p", &id, ModelField::Pricing, FieldOverride::Clear)
        .unwrap();
    assert!(reloaded.provider("p").unwrap().models[0]
        .info
        .pricing
        .is_none());
    reloaded
        .set_model_override("p", &id, ModelField::Pricing, FieldOverride::Inherit)
        .unwrap();
    assert_eq!(
        reloaded.provider("p").unwrap().models[0]
            .info
            .pricing
            .as_ref()
            .unwrap()
            .input_per_million,
        Some(2.0)
    );
}

#[test]
fn inherited_fields_follow_catalog_and_reset_is_field_specific() {
    let d = Dir::new();
    let http = Arc::new(Http::default());
    let mut a = client(&[profile(vec![model("m", 1.)])], http.clone());
    a.set_config_dir(&d.0).unwrap();
    a.set_tracked_models("acme", ["m".into()]).unwrap();
    let id = row(&a, "m");
    a.set_model_override(
        "p",
        &id,
        ModelField::Description,
        FieldOverride::Set(json!("mine")),
    )
    .unwrap();
    let mut newer = model("m", 4.);
    newer.description = Some("new catalog".into());
    newer.metadata.context_window_tokens = Some(300);
    let mut b = client(&[profile(vec![newer.clone()])], http.clone());
    let newer = b.provider("p").unwrap().models[0].clone();
    b.set_config_dir(&d.0).unwrap();
    let current = &b.provider("p").unwrap().models[0];
    assert_eq!(current.description.as_deref(), Some("mine"));
    assert_eq!(current.pricing, newer.pricing);
    assert_eq!(current.metadata.context_window_tokens, Some(300));
    b.clear_model_override("p", &id, ModelField::Description)
        .unwrap();
    assert_eq!(b.provider("p").unwrap().models[0], newer);
    let before = d.bytes();
    assert!(b
        .set_model_override(
            "p",
            &id,
            ModelField::Hidden,
            FieldOverride::Set(json!("wrong type"))
        )
        .is_err());
    assert_eq!(d.bytes(), before);
    assert_eq!(b.provider("p").unwrap().models[0], newer);
    let mut empty = client(&[], http);
    empty.set_config_dir(&d.0).unwrap();
    assert_eq!(
        empty.provider("p").unwrap().models[0],
        newer,
        "successful write refreshes fallback"
    );
}

#[test]
fn fallback_is_used_only_if_the_whole_definition_is_missing() {
    let d = Dir::new();
    let http = Arc::new(Http::default());
    let mut c = client(
        &[profile(vec![model("a", 1.), model("b", 2.)])],
        http.clone(),
    );
    c.set_config_dir(&d.0).unwrap();
    c.set_tracked_models("acme", ["a".into()]).unwrap();
    let mut absent = client(&[], http.clone());
    absent.set_config_dir(&d.0).unwrap();
    assert_eq!(absent.provider("p").unwrap().models.len(), 1);
    assert!(absent.resolve("a").is_ok());
    let mut changed = client(&[profile(vec![model("b", 3.)])], http);
    changed.set_config_dir(&d.0).unwrap();
    assert!(changed.resolve("a").is_err());
    assert!(
        changed.resolve("b").is_ok(),
        "allowlist does not change explicit routing"
    );
}
#[test]
fn duplicate_wire_rows_keep_order_identity_and_independent_settings() {
    let d = Dir::new();
    let mut second = model("m", 2.);
    second.display_model = "second".into();
    let http = Arc::new(Http::default());
    let mut c = client(
        &[profile(vec![model("m", 1.), second.clone()])],
        http.clone(),
    );
    c.set_config_dir(&d.0).unwrap();
    c.set_tracked_models("acme", ["m".into()]).unwrap();
    let rows = c.configured_models("p").unwrap();
    assert_ne!(rows[0].row_id, rows[1].row_id);
    let second = rows[1].model.clone();
    let before = d.bytes();
    assert!(c.set_model_visibility("p", "m", false).is_err());
    assert_eq!(before, d.bytes());
    c.set_model_override(
        "p",
        &rows[0].row_id,
        ModelField::Hidden,
        FieldOverride::Set(json!(true)),
    )
    .unwrap();
    let mut restored = client(&[], http);
    restored.set_config_dir(&d.0).unwrap();
    assert!(restored.provider("p").unwrap().models[0].hidden);
    assert_eq!(restored.provider("p").unwrap().models[1], second);
}
#[test]
fn replacing_an_inherited_row_does_not_restore_the_original_or_misapply_allowlist() {
    for tracked in ["old", "new"] {
        let d = Dir::new();
        let base = profile(vec![model("old", 1.)]);
        let http = Arc::new(Http::default());
        let mut c = client(std::slice::from_ref(&base), http.clone());
        c.set_config_dir(&d.0).unwrap();
        c.set_tracked_models("acme", [tracked.into()]).unwrap();
        let id = row(&c, "old");
        c.replace_model("p", &id, model("new", 2.)).unwrap();
        assert_eq!(c.provider("p").unwrap().models.len(), 1);
        assert!(c.resolve("old").is_err());
        assert!(c.resolve("new").is_ok());
        let mut restored = client(&[base], http);
        restored.set_config_dir(&d.0).unwrap();
        assert!(restored.resolve("old").is_err());
        assert_eq!(restored.resolve("new").is_ok(), tracked == "new");
    }
}
#[test]
fn stale_model_replacement_cannot_revive_another_clients_deleted_row() {
    let d = Dir::new();
    let base = profile(vec![model("m", 1.)]);
    let http = Arc::new(Http::default());
    let mut a = client(std::slice::from_ref(&base), http.clone());
    a.set_config_dir(&d.0).unwrap();
    a.set_tracked_models("acme", ["m".into(), "n".into()])
        .unwrap();
    let id = row(&a, "m");
    let mut b = client(&[base], http);
    b.set_config_dir(&d.0).unwrap();
    b.add_provider(profile(vec![model("n", 2.)])).unwrap();
    let before = d.bytes();
    assert!(a.replace_model("p", &id, model("m", 3.)).is_err());
    assert_eq!(d.bytes(), before);
}
#[tokio::test]
async fn observed_rows_join_later_static_definitions_without_duplicates() {
    let d = Dir::new();
    let http = Arc::new(Http(Mutex::new(
        json!({"data":[{"id":"m","description":"observed"}]}),
    )));
    let mut a = client(&[profile(vec![])], http.clone());
    a.set_config_dir(&d.0).unwrap();
    a.set_tracked_models("acme", ["m".into(), "n".into()])
        .unwrap();
    a.sync_provider("p", None).await.unwrap();
    let id = row(&a, "m");
    let mut b = client(&[profile(vec![model("m", 7.)])], http.clone());
    b.set_config_dir(&d.0).unwrap();
    assert_eq!(b.provider("p").unwrap().models.len(), 1);
    assert_eq!(row(&b, "m"), id);
    assert_eq!(
        b.provider("p").unwrap().models[0].pricing,
        model("m", 7.).pricing
    );
    b.replace_model("p", &id, model("n", 1.)).unwrap();
    assert!(b.resolve("m").is_err());
    b.sync_provider("p", None).await.unwrap();
    assert!(b.resolve("m").is_ok());
    assert!(b.resolve("n").is_ok());
}
#[test]
fn unsupported_configuration_versions_are_rejected_without_writes() {
    let d = Dir::new();
    let bytes = br#"{"version":1,"providers":[]}"#;
    std::fs::write(d.0.join("providers.json"), bytes).unwrap();
    let mut c = client(&[profile(vec![model("m", 1.)])], Arc::new(Http::default()));
    assert!(matches!(
        c.set_config_dir(&d.0),
        Err(ProviderStoreError::UnsupportedVersion(1))
    ));
    assert_eq!(d.bytes(), bytes);
    assert!(c.resolve("m").is_ok());
    assert!(!d.0.join("providers.v1.json.bak").exists());
}
#[tokio::test]
async fn explicit_overrides_outrank_observations_and_reset_keeps_observations() {
    let d = Dir::new();
    let http = Arc::new(Http(Mutex::new(
        json!({"data":[{"id":"m","description":"observed"}]}),
    )));
    let mut c = client(&[profile(vec![model("m", 1.)])], http);
    c.set_config_dir(&d.0).unwrap();
    c.set_tracked_models("acme", ["m".into()]).unwrap();
    let id = row(&c, "m");
    c.set_model_override(
        "p",
        &id,
        ModelField::Description,
        FieldOverride::Set(json!("user")),
    )
    .unwrap();
    c.sync_provider("p", None).await.unwrap();
    assert_eq!(
        c.provider("p").unwrap().models[0].description.as_deref(),
        Some("user")
    );
    c.clear_model_override("p", &id, ModelField::Description)
        .unwrap();
    assert_eq!(
        c.provider("p").unwrap().models[0].description.as_deref(),
        Some("observed")
    );
}
#[test]
fn clearing_a_full_replacement_field_inherits_the_catalog() {
    let d = Dir::new();
    let mut c = client(&[profile(vec![model("m", 1.)])], Arc::new(Http::default()));
    c.set_config_dir(&d.0).unwrap();
    c.set_tracked_models("acme", ["m".into()]).unwrap();
    c.add_provider(profile(vec![model("m", 9.)])).unwrap();
    let id = row(&c, "m");
    c.clear_model_override("p", &id, ModelField::Pricing)
        .unwrap();
    assert_eq!(
        c.provider("p").unwrap().models[0].pricing,
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
    let mut a = client(std::slice::from_ref(&base), http.clone());
    a.set_config_dir(&d.0).unwrap();
    a.set_tracked_models("acme", ["m".into()]).unwrap();
    a.set_model_visibility("p", "m", false).unwrap();
    let id = row(&a, "m");
    let mut b = client(&[profile(vec![])], http.clone());
    b.set_config_dir(&d.0).unwrap();
    assert!(b.resolve("m").is_err());
    b.sync_provider("p", None).await.unwrap();
    let mut c = client(&[base], http);
    c.set_config_dir(&d.0).unwrap();
    assert!(c.resolve("m").is_ok());
    let rows = c.configured_models("p").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].row_id, id);
    assert!(rows[0].model.hidden);
    assert_eq!(rows[0].model.description.as_deref(), Some("observed"));
}
#[test]
fn catalog_reordering_does_not_move_duplicate_wire_overrides() {
    let d = Dir::new();
    let mut a_model = model("m", 1.);
    a_model.display_model = "first".into();
    let mut b_model = model("m", 2.);
    b_model.display_model = "second".into();
    let http = Arc::new(Http::default());
    let mut a = client(
        &[profile(vec![a_model.clone(), b_model.clone()])],
        http.clone(),
    );
    a.set_config_dir(&d.0).unwrap();
    a.set_tracked_models("acme", ["m".into()]).unwrap();
    let id = a.configured_models("p").unwrap()[0].row_id.clone();
    a.set_model_override(
        "p",
        &id,
        ModelField::Hidden,
        FieldOverride::Set(json!(true)),
    )
    .unwrap();
    let mut b = client(&[profile(vec![b_model, a_model])], http);
    b.set_config_dir(&d.0).unwrap();
    let rows = b.configured_models("p").unwrap();
    assert_eq!(rows[0].row_id, id);
    assert_eq!(rows[0].model.display_model, "first");
    assert!(rows[0].model.hidden);
    assert!(!rows[1].model.hidden);
}
#[test]
fn builtin_references_are_explicit_and_recover_with_an_empty_builder() {
    let d = Dir::new();
    let http = Arc::new(Http::default());
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[]);
    builder.add_builtin_profile("openai").unwrap();
    let mut c = builder.with_region(Region::International).build().unwrap();
    let wire = c.provider("openai").unwrap().models[0]
        .request_model
        .clone();
    c.set_config_dir(&d.0).unwrap();
    c.set_tracked_models("openai", [wire.clone()]).unwrap();
    let saved: Value = serde_json::from_slice(&d.bytes()).unwrap();
    assert_eq!(saved["providers"][0]["definition"]["source"], "builtin");
    assert_eq!(
        saved["providers"][0]["fallback"]["models"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let mut empty = client(&[], http);
    empty.set_config_dir(&d.0).unwrap();
    assert!(empty.resolve_in(&wire, Some("openai")).is_ok());
}

#[tokio::test]
async fn observed_model_survives_static_adoption_and_withdrawal() {
    let d = Dir::new();
    let http = Arc::new(Http(Mutex::new(
        json!({"data":[{"id":"m","description":"observed"}]}),
    )));
    let mut discovered = client(&[profile(vec![])], http.clone());
    discovered.set_config_dir(&d.0).unwrap();
    discovered.set_tracked_models("acme", ["m".into()]).unwrap();
    discovered.sync_provider("p", None).await.unwrap();
    let id = row(&discovered, "m");

    let mut adopted = client(&[profile(vec![model("m", 7.)])], http.clone());
    adopted.set_config_dir(&d.0).unwrap();
    adopted.set_model_visibility("p", "m", false).unwrap();
    assert_eq!(
        adopted.configured_models("p").unwrap()[0].model.pricing,
        model("m", 7.).pricing
    );

    let mut withdrawn = client(&[profile(vec![])], http);
    withdrawn.set_config_dir(&d.0).unwrap();
    let rows = withdrawn.configured_models("p").unwrap();
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
    let mut c = client(&[profile(vec![model("m", 1.)])], http.clone());
    c.set_config_dir(&d.0).unwrap();
    c.set_tracked_models("acme", ["m".into()]).unwrap();
    let request: CompletionRequest = serde_json::from_value(json!({
        "model":"m", "messages":[], "thinking":{"mode":"disabled"}
    }))
    .unwrap();
    for mandatory in [true, false, true] {
        *http.0.lock().unwrap() = json!({"data":[{"id":"m","reasoning":{
            "mandatory":mandatory, "default_enabled":mandatory, "supported_efforts":["none","low","high"]
        }}]});
        c.sync_provider("p", None).await.unwrap();
        let p = c.provider("p").unwrap();
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
    let mut c = client(&[profile(vec![model("m", 1.)])], http.clone());
    c.set_config_dir(&d.0).unwrap();
    c.set_tracked_models("acme", ["m".into()]).unwrap();
    c.sync_provider("p", None).await.unwrap();
    assert_eq!(
        c.provider("p").unwrap().models[0]
            .info
            .features
            .default_mode,
        Some(ThinkingMode::Disabled)
    );
    *http.0.lock().unwrap() = json!({"data":[{"id":"m","reasoning":{"mandatory":true}}]});
    c.sync_provider("p", None).await.unwrap();
    let features = &c.provider("p").unwrap().models[0].info.features;
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
    let mut c = client(std::slice::from_ref(&p), http.clone());
    c.set_config_dir(&d.0).unwrap();
    c.set_tracked_models("acme", ["m".into()]).unwrap();
    for (levels, expected) in [
        (json!(["high"]), vec![ReasoningEffort::High]),
        (Value::Null, ReasoningEffort::ALL.to_vec()),
    ] {
        *http.0.lock().unwrap() =
            json!({"data":[{"id":"m","reasoning":{"supported_efforts":levels}}]});
        c.sync_provider("p", None).await.unwrap();
        assert_eq!(c.models()[0].info.features.effort.levels, Some(expected));
    }
    *http.0.lock().unwrap() = json!({"data":[{"id":"m","reasoning":{}}]});
    c.sync_provider("p", None).await.unwrap();
    let mut restored = client(&[p], http);
    restored.set_config_dir(&d.0).unwrap();
    assert_eq!(
        restored.models()[0].info.features.effort.levels,
        Some(ReasoningEffort::ALL.to_vec())
    );
    let req: CompletionRequest =
        serde_json::from_value(json!({"model":"m","messages":[],"thinking":{"effort":"low"}}))
            .unwrap();
    let p = restored.provider("p").unwrap();
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
    let mut c = client(std::slice::from_ref(&p), http.clone());
    c.set_config_dir(&d.0).unwrap();
    c.set_tracked_models("acme", ["m".into(), "new".into()])
        .unwrap();
    let caps = json!({"thinking":{"supported":true,"types":{"enabled":{"supported":false},"adaptive":{"supported":true}}},"effort":{"supported":true,"max":{"supported":false},"high":{"supported":true}}});
    *http.0.lock().unwrap() = json!({"data":[{"id":"m","capabilities":caps,"max_input_tokens":200000,"max_tokens":8192},{"id":"new","capabilities":caps}],"has_more":false});
    c.sync_provider("p", None).await.unwrap();
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
    c.sync_provider("p", None).await.unwrap();
    let mut restored = client(&[p], http);
    restored.set_config_dir(&d.0).unwrap();
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
    let req: CompletionRequest = serde_json::from_value(
        json!({"model":"m","messages":[],"thinking":{"mode":"adaptive","effort":"max"}}),
    )
    .unwrap();
    let profile = restored.provider("p").unwrap();
    assert!(matches!(
        AnthropicMessagesCodec.validate_request(
            &req,
            &CodecContext::new(profile, "m", RequestMode::Complete)
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));
    let id = row(&restored, "m");
    let mut user_features = features.clone();
    user_features.effort.with_disabled_thinking = CapabilitySupport::Unsupported;
    user_features
        .mode_support
        .entry(ThinkingMode::Adaptive)
        .or_default()
        .forced_tool_choice = CapabilitySupport::Unsupported;
    restored
        .set_model_override(
            "p",
            &id,
            ModelField::InferenceFeatures,
            FieldOverride::Set(serde_json::to_value(&user_features).unwrap()),
        )
        .unwrap();
    restored.set_config_dir(&d.0).unwrap();
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
    restored
        .set_model_override(
            "p",
            &id,
            ModelField::InferenceFeatures,
            FieldOverride::Clear,
        )
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
    restored
        .set_model_override(
            "p",
            &id,
            ModelField::InferenceFeatures,
            FieldOverride::Inherit,
        )
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
