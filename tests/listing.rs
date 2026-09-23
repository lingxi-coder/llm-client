//! What `models()` hands a picker.
//!
//! These name providers on purpose — they assert what the shipped data says.
//! Gate 30 scans `src/` only.

use lingxi_agent_api::protocol::{AuthStrategy, ProviderProfile};
use lingxi_llm_client::{builtin_providers, LlmClient, LlmClientBuilder};
use serde_json::json;
use std::sync::Arc;

mod support;

fn client_of(profiles: &[ProviderProfile]) -> LlmClient {
    let http = Arc::new(support::NoHttp);
    let mut b = LlmClientBuilder::with_transport(http, profiles);
    b.register_codec(Arc::new(
        lingxi_llm_client::codecs::openai::chat::OpenAiChatCodec,
    ));
    b.register_codec(Arc::new(lingxi_llm_client::AnthropicMessagesCodec));
    b.register_codec(Arc::new(lingxi_llm_client::GeminiCodec));
    b.register_codec(Arc::new(
        lingxi_llm_client::codecs::openai::responses::OpenAiResponsesCodec,
    ));
    for strategy in AuthStrategy::ALL {
        b.register_authenticator(strategy, Arc::new(support::NoAuth));
    }
    b.with_region(lingxi_agent_api::protocol::Region::International)
        .build()
        .expect("every protocol has a codec")
}

fn profile(models: serde_json::Value) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme",
        "profile_name": "acme",
        "base_url": "https://x.test",
        "protocol": "open_ai_chat",
        "auth": "none",
        "models": models,
    }))
    .expect("profile fixture parses")
}

/// A description is a paragraph of vendor prose. It was being handed out as the
/// model's `display_name`, so a picker showing names showed sentences.
#[test]
fn the_display_name_is_a_name_and_the_blurb_keeps_its_own_field() {
    let c = client_of(&[profile(json!([{
        "display_model": "acme-large",
        "request_model": "acme-large-2026-01-01",
        "billing_model": "acme-large",
        "description": "Acme Large is a sparse mixture-of-experts model optimized for agentic coding, with a 200k context window and strong tool use.",
    }]))]);
    let listed = c.models();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].display_name, "acme-large");
    assert_eq!(listed[0].id, "acme-large");
    assert!(
        listed[0]
            .description
            .as_deref()
            .is_some_and(|d| d.starts_with("Acme Large is")),
        "the blurb is still carried, just not as a name"
    );
}

/// A model whose window the catalog does not publish must say so. Zero is a
/// different claim: a compactor divides by this, and "no room left" and "we do
/// not know the size of the room" call for opposite behaviour.
#[test]
fn an_unpublished_context_window_is_unknown_rather_than_zero() {
    let c = client_of(&[profile(json!([
        {
            "display_model": "known",
            "request_model": "known",
            "billing_model": "known",
            "metadata": {"contextWindowTokens": 200_000},
        },
        {
            "display_model": "unknown",
            "request_model": "unknown",
            "billing_model": "unknown",
        },
    ]))]);
    let listed = c.models();
    let by = |id: &str| {
        listed
            .iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("{id} listed"))
    };
    assert_eq!(by("known").context_window, Some(200_000));
    assert_eq!(by("unknown").context_window, None);
}

/// Across the shipped catalog, a display name is short. This is the property
/// the description-as-name bug broke, stated where a refresh would trip it.
#[test]
fn no_shipped_model_is_listed_under_a_sentence() {
    let c = client_of(&builtin_providers().unwrap());
    let listed = c.models();
    assert!(listed.len() > 100, "{} models listed", listed.len());
    for m in &listed {
        assert!(
            m.display_name.len() <= 80 && !m.display_name.contains(". "),
            "{:?} reads as prose, not a name",
            m.display_name
        );
    }
    assert!(
        listed.iter().any(|m| m.description.is_some()),
        "and the blurbs did survive somewhere"
    );
}

/// A hidden connection exists only to be failed over onto, so offering it would
/// let a user pick the spare key directly.
#[test]
fn a_hidden_connection_is_not_offered() {
    let visible: ProviderProfile = serde_json::from_value(json!({
        "provider_id": "acme", "profile_name": "acme",
        "base_url": "https://one.test", "protocol": "open_ai_chat", "auth": "none",
        "models": [{"display_model": "m", "request_model": "m", "billing_model": "m"}],
        "connection": {"group": "g", "order": 0},
    }))
    .unwrap();
    let spare: ProviderProfile = serde_json::from_value(json!({
        "provider_id": "acme", "profile_name": "acme-spare",
        "base_url": "https://two.test", "protocol": "open_ai_chat", "auth": "none",
        "models": [{"display_model": "spare-only", "request_model": "m", "billing_model": "m"}],
        "connection": {"group": "g", "order": 1, "hidden": true},
    }))
    .unwrap();

    let listed = client_of(&[visible, spare]).models();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, "m");
    assert!(
        !listed.iter().any(|m| m.id == "spare-only"),
        "the spare key is reachable by failover, never by picking it"
    );
}

#[test]
fn an_unqualified_model_starts_on_a_visible_connection() {
    let hidden: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"acme", "profile_name":"hidden", "base_url":"https://hidden.test",
        "protocol":"open_ai_chat", "auth":"none",
        "models":[{"display_model":"m", "request_model":"m", "billing_model":"m"}],
        "connection":{"group":"g", "order":0, "hidden":true}
    }))
    .unwrap();
    let visible: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"acme", "profile_name":"visible", "base_url":"https://visible.test",
        "protocol":"open_ai_chat", "auth":"none",
        "models":[{"display_model":"m", "request_model":"m", "billing_model":"m"}],
        "connection":{"group":"g", "order":1}
    }))
    .unwrap();
    let client = client_of(&[hidden, visible]);
    assert_eq!(client.resolve("m").unwrap().profile_name, "visible");
    assert_eq!(
        client.resolve_in("m", Some("hidden")).unwrap().profile_name,
        "hidden"
    );
}

/// The half of a model ref that says *which connection*. Without it a caller
/// holding a listing row cannot name the connection it came from, and a group
/// with two connections serving one model is ambiguous.
#[test]
fn a_listed_model_round_trips_back_through_resolve() {
    let c = client_of(&builtin_providers().unwrap());
    for m in c.models() {
        let qualified = format!("{}/{}", m.profile_name, m.id);
        let route = c
            .resolve_in(&m.id, Some(&m.profile_name))
            .unwrap_or_else(|e| panic!("{qualified} does not resolve back: {e}"));
        assert_eq!(
            route.request_model, m.request_model,
            "{qualified} resolves to a different wire model than it listed"
        );
        assert_eq!(
            route.profile_name, m.profile_name,
            "{qualified} started on a different connection than it listed"
        );
    }
}

/// `providers()` is deliberately unfiltered where `models()` is filtered: an app
/// needs the providers a user has *not* configured, because those are the ones
/// it offers to set up.
#[test]
fn every_provider_is_listed_including_the_spares() {
    let profiles = builtin_providers().unwrap();
    let listed = client_of(&profiles).providers();
    assert_eq!(
        listed.len(),
        profiles
            .iter()
            .filter(|p| p.supports_region(lingxi_agent_api::protocol::Region::International))
            .count()
    );
    assert!(
        listed.iter().any(|p| p.hidden),
        "the spares are listed too, flagged rather than dropped"
    );
    assert!(
        listed.iter().all(|p| p.model_count > 0),
        "a provider with no models could never be routed to"
    );
}

/// This crate holds no credentials, so it cannot say whether one is set. It says
/// where one is expected and stops there; the host decides what counts as
/// configured.
#[test]
fn a_provider_says_where_its_credential_comes_from_not_whether_it_is_set() {
    let listed = client_of(&builtin_providers().unwrap()).providers();
    for p in &listed {
        assert!(
            p.credential_env.as_deref().is_some_and(|v| !v.is_empty()),
            "{} names no environment variable",
            p.profile_name
        );
    }
}

/// A picker showing prices needs them on the row it is showing.
#[test]
fn a_listed_model_carries_its_price_and_how_it_is_billed() {
    let listed = client_of(&builtin_providers().unwrap()).models();
    let priced = listed.iter().filter(|m| m.pricing.is_some()).count();
    assert!(priced > 100, "only {priced} of {} priced", listed.len());
    assert!(
        listed
            .iter()
            .any(|m| m.billing_mode == lingxi_agent_api::protocol::BillingMode::Free),
        "the free tier is visible as free on the row itself"
    );
}

/// One preset is its own group's namesake and has a sibling serving a different
/// wire model under the same display name. A ref naming that connection has to
/// mean it, or a picker row routes to the sibling's model — which on this pair
/// is also billed differently.
///
/// Stated over whatever the catalog actually contains rather than one hard-coded
/// pair, so a refresh that creates another such clash is covered too.
#[test]
fn a_connection_qualified_ref_means_that_connection_not_its_group() {
    let profiles = builtin_providers().unwrap();
    let c = client_of(&profiles);
    let rows = c.models();

    let mut clashes = 0;
    for row in &rows {
        let twin = rows.iter().any(|o| {
            o.id == row.id
                && o.profile_name != row.profile_name
                && o.request_model != row.request_model
        });
        if !twin {
            continue;
        }
        clashes += 1;
        let route = c
            .resolve_in(&row.id, Some(&row.profile_name))
            .unwrap_or_else(|e| panic!("{}/{}: {e}", row.profile_name, row.id));
        assert_eq!(route.profile_name, row.profile_name);
        assert_eq!(
            route.request_model, row.request_model,
            "{}/{} routed to another connection's model",
            row.profile_name, row.id
        );
    }
    assert!(
        clashes >= 2,
        "the shipped catalog has such a clash to test with; found {clashes}"
    );

    // A bare group name still reaches the group, which is what a stored
    // group-qualified ref relies on.
    let group_only = profiles
        .iter()
        .find(|p| p.supports_region(c.region()) && p.group() != p.profile_name)
        .expect("some connection belongs to a group named otherwise");
    let group = group_only.group().to_owned();
    let model = group_only.models[0].display_model.clone();
    let by_group = c
        .resolve_in(&model, Some(&group))
        .expect("a group-qualified ref still resolves");
    let landed = profiles
        .iter()
        .find(|p| p.profile_name == by_group.profile_name)
        .expect("it resolved to a known connection");
    assert_eq!(
        landed.group(),
        group,
        "a group name still reaches the group, just not one connection in it"
    );
}
