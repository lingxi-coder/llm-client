use lingxi_llm_client::{client::RoutingCatalog, codecs::RequestMode, protocol::*, *};
use serde_json::json;
use std::sync::Arc;
mod support;
fn profiles() -> Vec<ProviderProfile> {
    [ ("account", "open_ai_chat", "https://chat.example.test/v1"),
      ("messages", "anthropic_messages", "https://messages.example.test") ]
    .into_iter().map(|(name, format, url)| serde_json::from_value(json!({
        "provider_id":"multi", "profile_name":name,"protocol":format,"base_url":url,
        "auth":"none","credential":{"source":"host_managed","key":"same-account"},
        "connection":{"protocol_profiles":{"anthropic_messages":"messages","open_ai_chat":"account"}},
        "models":[{"request_model":"m","display_model":"m","billing_model":"m"}]
    })).unwrap()).collect()
}
fn request() -> ChatRequest {
    serde_json::from_value(json!({"model":"account/m","max_tokens":32,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]})).unwrap()
}
#[tokio::test]
async fn explicit_format_changes_endpoint_and_codec_without_changing_default() {
    let client = LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &profiles())
        .with_region(Region::International)
        .build()
        .unwrap();
    let call = client
        .snapshot()
        .prepare_on(
            "account",
            &request(),
            &RequestOptions {
                protocol: Some(ProtocolFamily::AnthropicMessages),
                ..Default::default()
            },
            RequestMode::Stream,
        )
        .await
        .unwrap();
    assert_eq!(
        call.request().url,
        "https://messages.example.test/v1/messages"
    );
    let body: serde_json::Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(body["model"], "m");
    assert_eq!(body["max_tokens"], 32);
    assert_eq!(
        client
            .resolve_in("m", Some("account"))
            .unwrap()
            .profile_name,
        "account"
    );
    let call = client
        .snapshot()
        .prepare_on(
            "account",
            &request(),
            &RequestOptions::default(),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    assert_eq!(
        call.request().url,
        "https://chat.example.test/v1/chat/completions"
    );
}
#[test]
fn explicit_selection_rejects_other_accounts_providers_and_unavailable_formats() {
    for other in ["credential", "provider", "billing", "missing"] {
        let mut ps = profiles();
        match other {
            "credential" => {
                ps[1].credential = CredentialConfig::HostManaged {
                    key: "other-account".into(),
                }
            }
            "provider" => ps[1].provider_id = "other-provider".into(),
            "billing" => ps[1].pricing.billing_mode = BillingMode::Subscription,
            _ => ps[1].regions.clear(),
        }
        let catalog = RoutingCatalog::new(ps, Region::International);
        assert!(catalog
            .resolve_in_with_protocol(
                "account/m",
                Some("account"),
                Some(ProtocolFamily::AnthropicMessages)
            )
            .is_err());
    }
    assert!(RoutingCatalog::new(profiles(), Region::International)
        .resolve_in_with_protocol("m", Some("account"), Some(ProtocolFamily::OpenAiResponses))
        .is_err());
}
#[tokio::test]
async fn configured_default_survives_reload_and_explicit_override_wins() {
    let dir = std::env::temp_dir().join(format!("lingxi-format-setting-{}", std::process::id()));
    let (client, settings) =
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &profiles())
            .with_region(Region::International)
            .build_managed()
            .unwrap();
    settings.set_config_dir(&dir).await.unwrap();
    settings
        .set_default_protocol("account", Some(ProtocolFamily::AnthropicMessages))
        .await
        .unwrap();
    assert_eq!(
        client
            .resolve_in("m", Some("account"))
            .unwrap()
            .profile_name,
        "messages"
    );
    let (restored, settings) =
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &profiles())
            .with_region(Region::International)
            .build_managed()
            .unwrap();
    settings.set_config_dir(&dir).await.unwrap();
    assert_eq!(
        restored
            .resolve_in("m", Some("account"))
            .unwrap()
            .profile_name,
        "messages"
    );
    let call = restored
        .snapshot()
        .prepare_on(
            "account",
            &request(),
            &RequestOptions {
                protocol: Some(ProtocolFamily::OpenAiChat),
                ..Default::default()
            },
            RequestMode::Stream,
        )
        .await
        .unwrap();
    assert!(call.request().url.ends_with("/chat/completions"));
    assert!(settings
        .set_default_protocol("account", Some(ProtocolFamily::GeminiInteractions))
        .await
        .is_err());
    settings
        .set_default_protocol("account", None)
        .await
        .unwrap();
    assert_eq!(
        restored
            .resolve_in("m", Some("account"))
            .unwrap()
            .profile_name,
        "account"
    );
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn deepseek_declares_both_native_formats_with_distinct_capabilities() {
    let ps = builtin_providers().unwrap();
    let chat = ps.iter().find(|p| p.profile_name == "deepseek").unwrap();
    let catalog = RoutingCatalog::new(ps.clone(), Region::International);
    let selected = catalog
        .resolve_in_with_protocol(
            "deepseek-flash",
            Some("deepseek"),
            Some(ProtocolFamily::AnthropicMessages),
        )
        .unwrap();
    let messages = ps
        .iter()
        .find(|p| p.profile_name == selected.profile_name)
        .unwrap();
    assert!(!hosted_search::supports(chat));
    assert!(hosted_search::supports(messages));
}
