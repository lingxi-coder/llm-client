//! Region policy is shared by listings and execution, while configuration stays complete.
use async_trait::async_trait;
use lingxi_llm_client::protocol::{
    AuthStrategy, CompletionRequest, LlmError, ProviderProfile, Region, WebSearchConfig,
};
use lingxi_llm_client::{
    builtin_providers, BuildError, HttpRequest, HttpResponse, LlmClient, LlmClientBuilder,
    RequestOptions, StreamResponse, Transport, WebSocketSession,
};
use serde_json::json;
use std::sync::{Arc, Mutex};

mod support;

fn profile(name: &str, regions: &[Region]) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme", "profile_name": name, "regions": regions,
        "base_url": format!("https://{name}.test/v1"), "protocol": "open_ai_chat", "auth": "none",
        "connection": {"group": name, "failover": {"network": true}},
        "models": [{"display_model":"model", "request_model":"wire", "billing_model":"wire", "aliases":["alias"]}]
    })).unwrap()
}

fn client(profiles: &[ProviderProfile], region: Region) -> LlmClient {
    let mut builder = LlmClientBuilder::with_transport(Arc::new(support::NoHttp), profiles);
    for auth in AuthStrategy::ALL {
        builder.register_authenticator(auth, Arc::new(support::NoAuth));
    }
    builder.with_region(region).build().unwrap()
}

#[test]
fn region_is_required_even_for_an_empty_client() {
    assert!(matches!(
        LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[]).build(),
        Err(BuildError::MissingRegion)
    ));
    for region in Region::ALL {
        assert_eq!(client(&[], region).region(), region);
    }
}

#[test]
fn declarations_roundtrip_and_missing_is_different_from_empty() {
    let p = profile("custom", &Region::ALL);
    let mut legacy = serde_json::to_value(&p).unwrap();
    legacy.as_object_mut().unwrap().remove("regions");
    let parsed: ProviderProfile = serde_json::from_value(legacy.clone()).unwrap();
    assert_eq!(parsed.regions, Region::all());
    for regions in [
        vec![],
        vec![Region::ChinaMainland],
        vec![Region::International],
        Region::all(),
    ] {
        let mut value = legacy.clone();
        value["regions"] = serde_json::to_value(&regions).unwrap();
        let parsed: ProviderProfile = serde_json::from_value(value).unwrap();
        assert_eq!(parsed.regions, regions);
        assert_eq!(
            serde_json::from_str::<ProviderProfile>(&serde_json::to_string(&parsed).unwrap())
                .unwrap(),
            parsed
        );
        for region in Region::ALL {
            let c = client(std::slice::from_ref(&parsed), region);
            assert_eq!(c.providers().len(), usize::from(regions.contains(&region)));
            assert_eq!(c.resolve("model").is_ok(), regions.contains(&region));
        }
    }
    legacy["regions"] = json!(["unknown"]);
    assert!(serde_json::from_value::<ProviderProfile>(legacy).is_err());
    assert_eq!(
        serde_json::to_value(Region::ALL).unwrap(),
        json!(["china_mainland", "international"])
    );
}

#[test]
fn every_builtin_has_the_declared_policy_and_listed_models_resolve() {
    let profiles = builtin_providers().unwrap();
    for p in &profiles {
        let expected = match p.profile_name.as_str() {
            "qwen" | "qwen-search" | "minimax" | "kimi" | "kimi-search" | "glm" | "glm-coding" => {
                vec![Region::ChinaMainland]
            }
            "deepseek" | "deepseek-search" | "kimi-code" => Region::all(),
            "qwen-intl" | "qwen-us" | "qwen-hk" | "qwen-search-intl" | "qwen-search-us"
            | "qwen-search-hk" | "minimax-intl" | "kimi-intl" | "kimi-search-intl" | "zai"
            | "zai-coding" | "openai" | "anthropic" | "gemini" | "openrouter"
            | "github-copilot" | "grok" | "grok-anthropic" | "grok-responses" => {
                vec![Region::International]
            }
            unknown => panic!("new preset needs a region-policy assertion: {unknown}"),
        };
        assert_eq!(p.regions, expected, "{}", p.profile_name);
    }
    for region in Region::ALL {
        let c = client(&profiles, region);
        assert_eq!(c.profiles().len(), profiles.len());
        let listed = c.providers();
        for p in &profiles {
            assert_eq!(
                listed.iter().any(|r| r.profile_name == p.profile_name),
                p.supports_region(region)
            );
            assert!(c.provider(&p.profile_name).is_some());
        }
        for row in c.models() {
            assert!(row.regions.contains(&region));
            assert_eq!(
                c.resolve_in(&row.id, Some(&row.profile_name))
                    .unwrap()
                    .profile_name,
                row.profile_name
            );
        }
    }
}

#[test]
fn region_filters_aliases_qualified_refs_and_ambiguity_before_matching() {
    let cn = profile("cn", &[Region::ChinaMainland]);
    let intl = profile("intl", &[Region::International]);
    for (region, allowed, denied) in [
        (Region::ChinaMainland, "cn", "intl"),
        (Region::International, "intl", "cn"),
    ] {
        let c = client(&[cn.clone(), intl.clone()], region);
        for id in ["model", "wire", "alias"] {
            assert_eq!(c.resolve(id).unwrap().profile_name, allowed);
            assert!(c.resolve(&format!("{denied}/{id}")).is_err());
            assert!(c.resolve_in(id, Some(denied)).is_err());
        }
    }
    let mut slash = cn;
    slash.models[0].request_model = "intl/model".into();
    let c = client(&[slash.clone(), intl.clone()], Region::ChinaMainland);
    assert_eq!(c.resolve("intl/model").unwrap().profile_name, "cn");
    let c = client(&[slash, intl], Region::International);
    assert_eq!(c.resolve("intl/model").unwrap().profile_name, "intl");
}

#[test]
fn excluded_profile_names_do_not_shadow_available_groups() {
    let mut cn = profile("cn", &[Region::ChinaMainland]);
    cn.connection.group = Some("intl".into());
    let intl = profile("intl", &[Region::International]);
    let c = client(&[cn, intl], Region::ChinaMainland);
    assert_eq!(
        c.resolve_in("model", Some("intl")).unwrap().profile_name,
        "cn"
    );
    assert_eq!(c.resolve("intl/model").unwrap().profile_name, "cn");
}

#[test]
fn glm_and_zai_shared_group_cannot_cross_regions() {
    let profiles = builtin_providers().unwrap();
    for (region, name) in [
        (Region::ChinaMainland, "glm-coding"),
        (Region::International, "zai"),
    ] {
        let c = client(&profiles, region);
        let p = c.provider(name).unwrap();
        let route = c
            .resolve_in(&p.models[0].request_model, Some(name))
            .unwrap();
        assert!(route.connection_chain.iter().all(|hop| c
            .provider(&hop.profile_name)
            .unwrap()
            .supports_region(region)));
    }
}

#[derive(Default)]
struct RecordingHttp(Mutex<Vec<String>>);
impl RecordingHttp {
    fn fail(&self, req: HttpRequest) -> LlmError {
        self.0.lock().unwrap().push(req.url);
        LlmError::Transport {
            message: "offline".into(),
        }
    }
}
#[async_trait]
impl Transport for RecordingHttp {
    async fn execute(&self, req: HttpRequest) -> Result<HttpResponse, LlmError> {
        Err(self.fail(req))
    }
    async fn open_stream(&self, req: HttpRequest) -> Result<StreamResponse, LlmError> {
        Err(self.fail(req))
    }
    async fn open_responses_websocket_session(
        &self,
        req: HttpRequest,
    ) -> Result<Box<dyn WebSocketSession>, LlmError> {
        Err(self.fail(req))
    }
}
fn request(model: &str) -> CompletionRequest {
    serde_json::from_value(json!({"model":model, "messages":[{"role":"user", "content":[{"type":"text","text":"hello"}]}]})).unwrap()
}

#[tokio::test]
async fn blocked_completion_stream_search_and_media_never_reach_transport() {
    let mut denied = profile("intl", &[Region::International]);
    denied.vision_delegate = Some("vision".into());
    let http = Arc::new(RecordingHttp::default());
    let c = LlmClientBuilder::with_transport(http.clone(), &[denied])
        .with_region(Region::ChinaMainland)
        .build()
        .unwrap();
    let req = request("intl/model");
    let opts = RequestOptions::default();
    assert!(matches!(
        c.complete(&req, &opts).await,
        Err(LlmError::ModelUnavailable { .. })
    ));
    assert!(matches!(
        c.stream(&req, &opts).await,
        Err(LlmError::ModelUnavailable { .. })
    ));
    assert!(matches!(
        c.complete_in("intl", &request("model"), &opts).await,
        Err(LlmError::ModelUnavailable { .. })
    ));
    assert!(matches!(
        c.web_search(&req, WebSearchConfig::default(), &opts).await,
        Err(LlmError::ModelUnavailable { .. })
    ));
    assert!(matches!(
        c.web_search_stream_in("intl", &req, WebSearchConfig::default(), &opts)
            .await,
        Err(LlmError::ModelUnavailable { .. })
    ));
    let mut media = req;
    // Invalid content would trigger attachment/media handling if the region guard were bypassed.
    media.messages[0].content = vec![lingxi_llm_client::protocol::ContentBlock::Image {
        source: lingxi_llm_client::protocol::ImageSource::Url {
            url: "https://media.test/image.png".into(),
        },
    }];
    assert!(matches!(
        c.complete(&media, &opts).await,
        Err(LlmError::ModelUnavailable { .. })
    ));
    assert!(http.0.lock().unwrap().is_empty());
}

#[tokio::test]
async fn failover_only_sends_to_allowed_connections_even_when_spares_are_hidden() {
    let mut cn = profile("cn", &[Region::ChinaMainland]);
    let mut denied = profile("intl", &[Region::International]);
    let mut shared = profile("shared", &Region::ALL);
    cn.connection.group = Some("group".into());
    denied.connection.group = Some("group".into());
    shared.connection.group = Some("group".into());
    shared.connection.hidden = true;
    let http = Arc::new(RecordingHttp::default());
    let c = LlmClientBuilder::with_transport(http.clone(), &[cn, denied, shared])
        .with_region(Region::ChinaMainland)
        .build()
        .unwrap();
    let route = c.resolve_in("model", Some("cn")).unwrap();
    assert_eq!(route.connection_chain.len(), 1);
    assert_eq!(route.connection_chain[0].profile_name, "shared");
    assert_eq!(c.providers().len(), 2);
    assert_eq!(c.models().len(), 1);
    let req = request("cn/model");
    assert!(c.complete(&req, &RequestOptions::default()).await.is_err());
    assert!(c.stream(&req, &RequestOptions::default()).await.is_err());
    let seen = http.0.lock().unwrap();
    assert_eq!(seen.len(), 4);
    assert!(seen.iter().all(|url| !url.contains("intl.test")));
    assert!(seen.iter().any(|url| url.contains("shared.test")));
}
