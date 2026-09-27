//! Catalog indexes must preserve row identities, visibility and failover policy.
use lingxi_llm_client::{
    configuration::{FieldOverride, ModelField, ProviderStoreError},
    protocol::{ModelProfile, ProviderProfile, Region},
    ClientConfigManager, LlmClient, LlmClientBuilder, ResolveError,
};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

mod support;

fn model(label: &str, wire: &str) -> ModelProfile {
    serde_json::from_value(json!({
        "display_model": label, "request_model": wire, "billing_model": wire
    }))
    .unwrap()
}

fn profile(name: &str, models: Vec<ModelProfile>) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme", "profile_name": name,
        "base_url": "https://example.test/v1", "protocol": "open_ai_chat",
        "auth": "none", "models": models
    }))
    .unwrap()
}

fn client(profiles: &[ProviderProfile]) -> (LlmClient, ClientConfigManager) {
    LlmClientBuilder::with_transport(Arc::new(support::NoHttp), profiles)
        .with_region(Region::International)
        .build_managed()
        .unwrap()
}

struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "llm-catalog-performance-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn chat_listing_filters_the_exact_row_when_wire_ids_repeat() {
    let text = model("text", "shared");
    let mut image = model("image", "shared");
    image.metadata.output_modalities = vec!["image".into()];
    for rows in [
        vec![image.clone(), text.clone()],
        vec![text.clone(), image.clone()],
    ] {
        let (client, _) = client(&[profile("p", rows)]);
        assert_eq!(client.models().len(), 2);
        let listed = client.chat().models();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "text");
        assert_eq!(
            client
                .resolve_in(&listed[0].id, Some("p"))
                .unwrap()
                .display_model,
            "text"
        );
    }
}

#[tokio::test]
async fn chat_listing_preserves_tracking_visibility_region_and_catalog_order() {
    let dir = Dir::new();
    let mut hidden = model("hidden", "hidden");
    hidden.hidden = true;
    let mut image = model("image", "image");
    image.metadata.output_modalities = vec!["image".into()];
    let visible = profile(
        "visible",
        vec![
            model("second", "b"),
            model("untracked", "untracked"),
            hidden,
            image,
            model("first", "a"),
        ],
    );
    let mut spare = profile("spare", vec![model("spare", "a")]);
    spare.connection.hidden = true;
    let mut regional = profile("china", vec![model("china", "a")]);
    regional.regions = vec![Region::ChinaMainland];
    let (client, config) = client(&[visible, spare, regional]);
    config.set_config_dir(&dir.0).await.unwrap();
    config
        .set_tracked_models("acme", ["a", "b", "hidden", "image"].map(str::to_owned))
        .await
        .unwrap();
    let listed = client.chat().models();
    assert_eq!(
        listed.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        ["second", "first"]
    );
    assert!(listed.iter().all(|m| m.profile_name == "visible"));
}

#[test]
fn indexed_failover_keeps_wire_billing_ambiguity_and_connection_order() {
    let connection = |name: &str, order: u32, models| {
        let mut p = profile(name, models);
        p.connection.group = Some("group".into());
        p.connection.order = order;
        p
    };
    let head = connection("head", 0, vec![model("main", "wire")]);
    let late = connection("late", 5, vec![model("late", "wire")]);
    let mut hidden = connection("hidden", 1, vec![model("hidden", "wire")]);
    hidden.connection.hidden = true;
    let ambiguous = connection(
        "ambiguous",
        2,
        vec![model("one", "wire"), model("two", "wire")],
    );
    let alias = connection("alias", 3, vec![model("wire", "other-wire")]);
    let mut billed_elsewhere = model("free", "wire");
    billed_elsewhere.billing_mode = Some(lingxi_llm_client::protocol::BillingMode::Free);
    let mixed = connection("mixed", 4, vec![billed_elsewhere, model("metered", "wire")]);
    let mut regional = connection("regional", 1, vec![model("regional", "wire")]);
    regional.regions = vec![Region::ChinaMainland];
    let other_group = profile("other-group", vec![model("other", "wire")]);
    let (client, _) = client(&[
        late,
        head,
        alias,
        ambiguous,
        mixed,
        hidden,
        regional,
        other_group,
    ]);
    let route = client.resolve_in("main", Some("head")).unwrap();
    assert_eq!(
        route
            .connection_chain
            .iter()
            .map(|hop| hop.profile_name.as_str())
            .collect::<Vec<_>>(),
        ["hidden", "mixed", "late"]
    );
    assert!(matches!(
        client.resolve_in("wire", Some("ambiguous")),
        Err(ResolveError::DuplicateOnProfile { .. })
    ));
}

#[tokio::test]
async fn configured_rows_keep_duplicate_occurrences_and_overrides_across_reload() {
    let dir = Dir::new();
    let mut duplicate = model("same", "wire");
    duplicate.description = Some("catalog metadata".into());
    let p = profile("p", vec![duplicate.clone(), duplicate]);
    let (_, config) = client(std::slice::from_ref(&p));
    let before = config.configured_models("p").await.unwrap();
    assert_eq!(before.len(), 2);
    assert_ne!(before[0].row_id, before[1].row_id);
    assert!(matches!(
        config.configured_models("missing").await,
        Err(ProviderStoreError::UnknownProfile(_))
    ));
    config.set_config_dir(&dir.0).await.unwrap();
    config
        .set_tracked_models("acme", ["wire".to_owned()])
        .await
        .unwrap();
    config
        .set_model_override(
            "p",
            &before[1].row_id,
            ModelField::Description,
            FieldOverride::Set(json!("custom metadata")),
        )
        .await
        .unwrap();
    let (_, reloaded) = client(&[p]);
    reloaded.set_config_dir(&dir.0).await.unwrap();
    let rows = reloaded.configured_models("p").await.unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].row_id, before[0].row_id);
    assert_eq!(rows[1].row_id, before[1].row_id);
    assert_eq!(
        rows[0].model.description.as_deref(),
        Some("catalog metadata")
    );
    assert_eq!(
        rows[1].model.description.as_deref(),
        Some("custom metadata")
    );
    reloaded.remove_provider("p").await.unwrap();
    assert!(matches!(
        reloaded.configured_models("p").await,
        Err(ProviderStoreError::UnknownProfile(_))
    ));
}
