use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

struct Fixture {
    requests: Mutex<Vec<HttpRequest>>,
    replies: Mutex<Vec<HttpResponse>>,
}
impl Fixture {
    fn new(replies: Vec<Value>) -> Arc<Self> {
        Arc::new(Self {
            requests: Mutex::new(Vec::new()),
            replies: Mutex::new(
                replies
                    .into_iter()
                    .rev()
                    .map(|body| HttpResponse {
                        status: 200,
                        headers: vec![],
                        body: Bytes::from(serde_json::to_vec(&body).unwrap()),
                    })
                    .collect(),
            ),
        })
    }
    fn requests(&self) -> Vec<HttpRequest> {
        self.requests.lock().unwrap().clone()
    }
}
#[async_trait]
impl Transport for Fixture {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        self.replies
            .lock()
            .unwrap()
            .pop()
            .map(Into::into)
            .ok_or_else(|| LlmError::Transport {
                message: "unexpected request".into(),
            })
    }
}
fn client(profile: &str, region: Region, fixture: Arc<Fixture>) -> LlmClient {
    let provider = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == profile)
        .unwrap();
    LlmClientBuilder::with_transport(fixture, &[provider])
        .with_region(region)
        .build()
        .unwrap()
}
fn options() -> ImageRequestOptions {
    ImageRequestOptions {
        credential: Some(Secret::new("test-secret".to_string())),
        ..Default::default()
    }
}
fn request(model: &str) -> ImageGenerationRequest {
    ImageGenerationRequest {
        model: model.into(),
        prompt: "a blue cat".into(),
        references: vec![],
        output: Default::default(),
        provider_options: Default::default(),
    }
}
fn body(request: &HttpRequest) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

#[tokio::test]
async fn openai_image_generation_uses_images_endpoint_and_base64_result() {
    let fixture = Fixture::new(vec![
        json!({"data":[{"b64_json":"AQID"}], "usage":{"total_tokens":3}}),
    ]);
    let client = client("openai", Region::International, fixture.clone());
    let response = client
        .images()
        .generate(&request("gpt-image-1.5"), &options())
        .await
        .unwrap();
    assert!(matches!(response.images[0].data, ImageData::Base64 { .. }));
    let sent = fixture.requests();
    assert_eq!(sent[0].url, "https://api.openai.com/v1/images/generations");
    assert_eq!(body(&sent[0])["model"], "gpt-image-1.5");
    assert!(sent[0]
        .headers
        .iter()
        .any(|(k, v)| k == "authorization" && v == "Bearer test-secret"));
}

#[tokio::test]
async fn openai_edit_uses_multipart_and_preserves_binary_input() {
    let fixture = Fixture::new(vec![json!({"data":[{"b64_json":"AQID"}]})]);
    let client = client("openai", Region::International, fixture.clone());
    let req = ImageEditRequest {
        model: "gpt-image-1.5".into(),
        prompt: "make it blue".into(),
        images: vec![ImageInput::Base64 {
            media_type: "image/png".into(),
            data: "AQID".into(),
        }],
        mask: None,
        output: Default::default(),
        provider_options: Default::default(),
    };
    client.images().edit(&req, &options()).await.unwrap();
    let sent = fixture.requests();
    assert_eq!(sent[0].url, "https://api.openai.com/v1/images/edits");
    assert!(sent[0]
        .headers
        .iter()
        .any(|(k, v)| k == "content-type" && v.starts_with("multipart/form-data")));
    assert!(sent[0].body.windows(3).any(|window| window == [1, 2, 3]));
}

#[tokio::test]
async fn gemini_generation_extracts_final_image_and_text_only() {
    let fixture = Fixture::new(vec![
        json!({"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"thought"},"thought":true},{"text":"Done"},{"inlineData":{"mimeType":"image/png","data":"final"}}]}}]}),
    ]);
    let client = client("gemini", Region::International, fixture.clone());
    let response = client
        .images()
        .generate(&request("gemini-3.1-flash-image"), &options())
        .await
        .unwrap();
    assert_eq!(response.images.len(), 1);
    assert_eq!(response.text.as_deref(), Some("Done"));
    let sent = fixture.requests();
    assert!(sent[0]
        .url
        .ends_with("/models/gemini-3.1-flash-image:generateContent"));
    assert_eq!(
        body(&sent[0])["generationConfig"]["responseModalities"],
        json!(["IMAGE"])
    );
    assert!(sent[0].headers.iter().any(|(k, _)| k == "x-goog-api-key"));
}

#[tokio::test]
async fn xai_edit_uses_json_endpoint() {
    let fixture = Fixture::new(vec![
        json!({"data":[{"url":"https://example.test/out.png"}]}),
    ]);
    let client = client("grok", Region::International, fixture.clone());
    let req = ImageEditRequest {
        model: "grok-imagine-image-2.0".into(),
        prompt: "make it blue".into(),
        images: vec![ImageInput::Url {
            url: "https://example.test/in.png".into(),
        }],
        mask: None,
        output: Default::default(),
        provider_options: Default::default(),
    };
    client.images().edit(&req, &options()).await.unwrap();
    let sent = fixture.requests();
    assert_eq!(sent[0].url, "https://api.x.ai/v1/images/edits");
    assert_eq!(
        body(&sent[0])["image"]["url"],
        "https://example.test/in.png"
    );
}

#[tokio::test]
async fn minimax_character_reference_and_openrouter_reference_have_distinct_shapes() {
    let mini = Fixture::new(vec![
        json!({"data":{"image_urls":["https://example.test/out.png"]},"base_resp":{"status_code":0}}),
    ]);
    let mini_client = client("minimax-intl", Region::International, mini.clone());
    let mut req = request("image-01");
    req.references.push(ImageReference {
        kind: ImageReferenceKind::Character,
        image: ImageInput::Url {
            url: "https://example.test/in.png".into(),
        },
    });
    mini_client
        .images()
        .generate(&req, &options())
        .await
        .unwrap();
    assert_eq!(
        body(&mini.requests()[0])["subject_reference"][0]["type"],
        "character"
    );

    let router = Fixture::new(vec![
        json!({"data":[{"b64_json":"AQID","media_type":"image/png"}]}),
    ]);
    let client = client("openrouter", Region::International, router.clone());
    let mut req = request("bytedance-seed/seedream-4.5");
    req.references.push(ImageReference {
        kind: ImageReferenceKind::General,
        image: ImageInput::Url {
            url: "https://example.test/in.png".into(),
        },
    });
    client.images().generate(&req, &options()).await.unwrap();
    assert_eq!(
        body(&router.requests()[0])["input_references"][0]["image_url"]["url"],
        "https://example.test/in.png"
    );
}

#[tokio::test]
async fn qwen_task_is_bound_to_its_profile_and_account() {
    let fixture = Fixture::new(vec![
        json!({"output":{"task_id":"task-123","task_status":"PENDING"}}),
        json!({"output":{"task_status":"SUCCEEDED","choices":[{"message":{"content":[{"type":"image","image":"https://example.test/out.png"}]}}]},"usage":{"output_image_count":1},"request_id":"req-1"}),
    ]);
    let client = client("qwen", Region::ChinaMainland, fixture.clone());
    let opts = ImageRequestOptions {
        account_scope: Some("account-a".into()),
        ..options()
    };
    let task = client
        .images()
        .submit(
            &ImageRequest::Generate(request("qwen-image-3.0-pro")),
            &opts,
        )
        .await
        .unwrap();
    assert_eq!(task.account_scope, "account-a");
    let result = client.images().get_task(&task, &opts).await.unwrap();
    let ImageTaskSnapshot::Succeeded { response } = result else {
        panic!("task did not succeed")
    };
    assert_eq!(response.images.len(), 1);
    assert_eq!(response.request_id.as_deref(), Some("req-1"));
    assert_eq!(response.usage.as_ref().unwrap()["output_image_count"], 1);
    let sent = fixture.requests();
    assert!(sent[0]
        .url
        .ends_with("/services/aigc/image-generation/generation"));
    assert_eq!(
        body(&sent[0])["input"]["messages"][0]["content"][0]["text"],
        "a blue cat"
    );
    assert!(sent[1].url.ends_with("/tasks/task-123"));
    let bad = ImageRequestOptions {
        account_scope: Some("account-b".into()),
        ..options()
    };
    assert!(client.images().get_task(&task, &bad).await.is_err());
    assert_eq!(fixture.requests().len(), 2);
}

#[tokio::test]
async fn unsupported_edit_fails_before_network() {
    let fixture = Fixture::new(vec![]);
    let client = client("zai", Region::International, fixture.clone());
    let req = ImageEditRequest {
        model: "glm-image".into(),
        prompt: "edit".into(),
        images: vec![ImageInput::Url {
            url: "https://example.test/in.png".into(),
        }],
        mask: None,
        output: Default::default(),
        provider_options: Default::default(),
    };
    assert!(client.images().edit(&req, &options()).await.is_err());
    assert!(fixture.requests().is_empty());
}

#[tokio::test]
async fn zai_native_task_uses_its_own_submit_and_query_endpoints() {
    let fixture = Fixture::new(vec![
        json!({"id":"task_123", "task_status":"PENDING"}),
        json!({"task_status":"SUCCESS", "image_result":[{"url":"https://example.test/out.png"}]}),
    ]);
    let client = client("zai", Region::International, fixture.clone());
    let opts = ImageRequestOptions {
        account_scope: Some("account-z".into()),
        ..options()
    };
    let task = client
        .images()
        .submit(&ImageRequest::Generate(request("glm-image")), &opts)
        .await
        .unwrap();
    let result = client.images().get_task(&task, &opts).await.unwrap();
    assert!(matches!(result, ImageTaskSnapshot::Succeeded { .. }));
    let sent = fixture.requests();
    assert_eq!(
        sent[0].url,
        "https://api.z.ai/api/paas/v4/async/images/generations"
    );
    assert_eq!(
        sent[1].url,
        "https://api.z.ai/api/paas/v4/async-result/task_123"
    );
}

#[tokio::test]
async fn zai_task_failure_statuses_are_terminal() {
    for status in ["FAIL", "FAILED"] {
        let fixture = Fixture::new(vec![
            json!({"id":"task_123", "task_status":"PROCESSING"}),
            json!({"task_status":status, "message":"generation failed"}),
        ]);
        let client = client("zai", Region::International, fixture);
        let opts = ImageRequestOptions {
            account_scope: Some("account-z".into()),
            ..options()
        };
        let task = client
            .images()
            .submit(&ImageRequest::Generate(request("glm-image")), &opts)
            .await
            .unwrap();
        assert_eq!(
            client.images().get_task(&task, &opts).await.unwrap(),
            ImageTaskSnapshot::Failed {
                message: "generation failed".into()
            }
        );
    }
}

#[tokio::test]
async fn zai_task_success_preserves_response_metadata() {
    let images = json!([{"url":"https://example.test/out.png"}]);
    for result in [
        json!({"image_result":images}),
        json!({"image_result":{"results":images}}),
        json!({"image_result":{"data":images}}),
        json!({"image_result":images[0]}),
        json!({"data":images}),
    ] {
        let mut reply = result;
        reply["task_status"] = json!("SUCCESS");
        reply["request_id"] = json!("req-123");
        reply["model"] = json!("reported-model");
        reply["usage"] = json!({"output_image_count":1});
        let fixture = Fixture::new(vec![
            json!({"id":"task_123", "task_status":"PROCESSING"}),
            reply,
        ]);
        let client = client("zai", Region::International, fixture);
        let opts = ImageRequestOptions {
            account_scope: Some("account-z".into()),
            ..options()
        };
        let task = client
            .images()
            .submit(&ImageRequest::Generate(request("glm-image")), &opts)
            .await
            .unwrap();
        let ImageTaskSnapshot::Succeeded { response } =
            client.images().get_task(&task, &opts).await.unwrap()
        else {
            panic!("task did not succeed")
        };
        assert_eq!(response.request_id.as_deref(), Some("req-123"));
        assert_eq!(response.reported_model.as_deref(), Some("reported-model"));
        assert_eq!(response.requested_model, "glm-image");
        assert_eq!(response.usage, Some(json!({"output_image_count":1})));
        assert_eq!(response.images.len(), 1);
        assert_eq!(
            response.images[0].data,
            ImageData::Url {
                url: "https://example.test/out.png".into()
            }
        );
    }
}

#[test]
fn image_routes_round_trip_independently_of_chat_models() {
    let profile = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "openai")
        .unwrap();
    let restored: ProviderProfile =
        serde_json::from_value(serde_json::to_value(&profile).unwrap()).unwrap();
    assert_eq!(restored.images, profile.images);
    assert!(restored
        .images
        .models
        .iter()
        .any(|model| model.request_model == "gpt-image-1.5"));
}

#[test]
fn builtin_image_routes_survive_configuration_reload() {
    let fixture = Fixture::new(vec![]);
    let path = std::env::temp_dir().join(format!(
        "llm-images-config-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&path).unwrap();
    let mut first_builder =
        LlmClientBuilder::with_transport(fixture.clone(), &[]).with_region(Region::International);
    first_builder.add_builtin_profile("openai").unwrap();
    let mut first = first_builder.build().unwrap();
    first.set_config_dir(&path).unwrap();
    first
        .set_tracked_models("openai", ["gpt-image-1.5".into()])
        .unwrap();
    let mut second_builder =
        LlmClientBuilder::with_transport(fixture, &[]).with_region(Region::International);
    second_builder.add_builtin_profile("openai").unwrap();
    let mut second = second_builder.build().unwrap();
    second.set_config_dir(&path).unwrap();
    assert!(second
        .images()
        .models()
        .iter()
        .any(|model| model.request_model == "gpt-image-1.5"));
    std::fs::remove_dir_all(path).unwrap();
}

#[tokio::test]
async fn custom_wan_route_uses_workspace_image_endpoint() {
    let profile: ProviderProfile = serde_json::from_value(json!({
        "profile_name":"wan-workspace", "provider_id":"qwen", "base_url":"https://example.test/compatible-mode/v1",
        "protocol":"open_ai_chat", "auth":"api_key", "regions":["international"],
        "images": {
            "routes": {"primary":{"api":"wan","base_url":"https://workspace.example.test/api/v1","task_base_url":"https://workspace.example.test/api/v1"}},
            "models":[{"display_model":"wan2.7-image-pro","request_model":"wan2.7-image-pro","route":"primary","capabilities":{"text_to_image":true,"editing":true,"async_generate":true,"async_edit":true,"max_inputs":9}}]
        }
    })).unwrap();
    let fixture = Fixture::new(vec![
        json!({"output":{"choices":[{"message":{"content":[{"type":"image","image":"https://example.test/wan.png"}]}}]}}),
    ]);
    let client = LlmClientBuilder::with_transport(fixture.clone(), std::slice::from_ref(&profile))
        .with_region(Region::International)
        .build()
        .unwrap();
    let response = client
        .images()
        .generate(&request("wan2.7-image-pro"), &options())
        .await
        .unwrap();
    assert_eq!(response.images.len(), 1);
    let sent = fixture.requests();
    assert_eq!(
        sent[0].url,
        "https://workspace.example.test/api/v1/services/aigc/multimodal-generation/generation"
    );
    assert_eq!(
        body(&sent[0])["input"]["messages"][0]["content"][0]["text"],
        "a blue cat"
    );

    let tasks = Fixture::new(vec![
        json!({"output":{"task_id":"wan-123","task_status":"PENDING"}}),
        json!({"output":{"task_status":"SUCCEEDED","choices":[{"message":{"content":[{"type":"image","image":"https://example.test/wan-task.png"}]}}]},"request_id":"wan-request"}),
    ]);
    let client = LlmClientBuilder::with_transport(tasks.clone(), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();
    let opts = ImageRequestOptions {
        account_scope: Some("workspace-account".into()),
        ..options()
    };
    let task = client
        .images()
        .submit(&ImageRequest::Generate(request("wan2.7-image-pro")), &opts)
        .await
        .unwrap();
    let ImageTaskSnapshot::Succeeded { response } =
        client.images().get_task(&task, &opts).await.unwrap()
    else {
        panic!("Wan task did not succeed")
    };
    assert_eq!(response.images.len(), 1);
    assert_eq!(response.request_id.as_deref(), Some("wan-request"));
    assert!(tasks.requests()[0]
        .url
        .ends_with("/services/aigc/image-generation/generation"));
}

#[tokio::test]
async fn invalid_output_options_fail_before_the_request_is_sent() {
    let fixture = Fixture::new(vec![]);
    let client = client("openrouter", Region::International, fixture.clone());
    let mut req = request("bytedance-seed/seedream-4.5");
    req.output.size = Some(ImageSize::Pixels {
        width: 1024,
        height: 1024,
    });
    req.output.aspect_ratio = Some((16, 9));
    let error = client
        .images()
        .generate(&req, &options())
        .await
        .unwrap_err();
    assert_eq!(error.dispatch(), ImageDispatch::NotSent);
    assert!(fixture.requests().is_empty());
}

#[tokio::test]
async fn openai_mask_rejects_nonfirst_target_before_dispatch() {
    let fixture = Fixture::new(vec![json!({"data":[{"b64_json":"AQID"}]})]);
    let client = client("openai", Region::International, fixture.clone());
    let image = ImageInput::Base64 {
        media_type: "image/png".into(),
        data: "AQID".into(),
    };
    let mut req = ImageEditRequest {
        model: "gpt-image-1.5".into(),
        prompt: "edit the masked area".into(),
        images: vec![image.clone(), image.clone()],
        mask: Some(ImageMask {
            image_index: 1,
            source: image,
        }),
        output: Default::default(),
        provider_options: Default::default(),
    };
    let error = client.images().edit(&req, &options()).await.unwrap_err();
    assert!(matches!(
        &error,
        ImageError::Llm(LlmError::UnsupportedCapability { .. })
    ));
    assert_eq!(error.dispatch(), ImageDispatch::NotSent);
    assert!(fixture.requests().is_empty());

    req.mask.as_mut().unwrap().image_index = 0;
    client.images().edit(&req, &options()).await.unwrap();
    let sent = fixture.requests();
    assert_eq!(sent.len(), 1);
    assert!(String::from_utf8_lossy(&sent[0].body).contains("name=\"mask\""));
}

#[tokio::test]
async fn image_response_preserves_media_type_and_legacy_mime_type() {
    let fixture = Fixture::new(vec![json!({"data":[
        {"b64_json":"AQID","media_type":"image/webp"},
        {"b64_json":"AQID","mime_type":"image/jpeg"},
        {"b64_json":"AQID"}
    ]})]);
    let client = client("openrouter", Region::International, fixture);
    let response = client
        .images()
        .generate(&request("bytedance-seed/seedream-4.5"), &options())
        .await
        .unwrap();
    assert_eq!(response.images[0].media_type.as_deref(), Some("image/webp"));
    assert_eq!(response.images[1].media_type.as_deref(), Some("image/jpeg"));
    assert_eq!(response.images[2].media_type, None);
}

#[test]
fn explicit_builtin_image_configuration_survives_replacement_and_reload() {
    let path = std::env::temp_dir().join(format!(
        "llm-images-override-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&path).unwrap();
    let build = || {
        let mut builder = LlmClientBuilder::with_transport(Fixture::new(vec![]), &[])
            .with_region(Region::International);
        builder.add_builtin_profile("openai").unwrap();
        let mut client = builder.build().unwrap();
        client.set_config_dir(&path).unwrap();
        client
    };
    let mut client = build();
    let mut profile = client.provider("openai").unwrap().clone();
    let route = profile.images.routes.get_mut("primary").unwrap();
    route.base_url = "https://proxy.example/v1".into();
    route.api_key_header = Some("x-image-key".into());
    profile.images.models.truncate(1);
    profile.images.models[0].request_model = "custom-image-model".into();
    let expected = profile.images.clone();
    client.add_provider(profile).unwrap();
    assert_eq!(client.provider("openai").unwrap().images, expected);
    let mut restored = build();
    assert_eq!(restored.provider("openai").unwrap().images, expected);

    // An explicitly empty replacement disables images instead of inheriting defaults.
    let mut profile = restored.provider("openai").unwrap().clone();
    profile.images = Default::default();
    restored.add_provider(profile).unwrap();
    assert!(restored.images().models().is_empty());
    assert!(build().images().models().is_empty());
    std::fs::remove_dir_all(path).unwrap();
}

fn image_group_profiles(primary_has_model: bool) -> Vec<ProviderProfile> {
    let mut primary = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "openai")
        .unwrap();
    primary.connection.group = Some("openai".into());
    primary.connection.connection_id = Some("primary".into());
    let mut backup = primary.clone();
    backup.profile_name = "openai-backup".into();
    backup.connection.connection_id = Some("backup".into());
    backup.images.routes.get_mut("primary").unwrap().base_url = "https://backup.example/v1".into();
    if !primary_has_model {
        primary
            .images
            .models
            .retain(|m| m.request_model != "gpt-image-1.5");
    }
    vec![primary, backup]
}

#[tokio::test]
async fn image_profile_scope_wins_over_same_named_group() {
    let fixture = Fixture::new(vec![
        json!({"data":[{"b64_json":"AQID"}]}),
        json!({"data":[{"b64_json":"AQID"}]}),
    ]);
    let client = LlmClientBuilder::with_transport(fixture.clone(), &image_group_profiles(true))
        .with_region(Region::International)
        .build()
        .unwrap();
    let scoped = client
        .images()
        .generate_in("openai", &request("gpt-image-1.5"), &options())
        .await
        .unwrap();
    let qualified = client
        .images()
        .generate(&request("openai/gpt-image-1.5"), &options())
        .await
        .unwrap();
    assert_eq!(scoped.executed_profile, "openai");
    assert_eq!(qualified.executed_profile, "openai");
    let sent = fixture.requests();
    assert_eq!(sent.len(), 2);
    assert!(sent
        .iter()
        .all(|req| req.url == "https://api.openai.com/v1/images/generations"));
}

#[tokio::test]
async fn image_profile_without_model_cannot_route_to_its_group_sibling() {
    let fixture = Fixture::new(vec![]);
    let client = LlmClientBuilder::with_transport(fixture.clone(), &image_group_profiles(false))
        .with_region(Region::International)
        .build()
        .unwrap();
    let scoped = client
        .images()
        .generate_in("openai", &request("gpt-image-1.5"), &options())
        .await
        .unwrap_err();
    let qualified = client
        .images()
        .generate(&request("openai/gpt-image-1.5"), &options())
        .await
        .unwrap_err();
    for error in [scoped, qualified] {
        assert_eq!(error.dispatch(), ImageDispatch::NotSent);
        assert!(matches!(
            error,
            ImageError::Llm(LlmError::ModelUnavailable { .. })
        ));
    }
    assert!(fixture.requests().is_empty());
}

#[tokio::test]
async fn image_group_scope_still_resolves_when_no_profile_has_that_name() {
    let fixture = Fixture::new(vec![
        json!({"data":[{"b64_json":"AQID"}]}),
        json!({"data":[{"b64_json":"AQID"}]}),
    ]);
    let mut profiles = image_group_profiles(false);
    profiles[0].profile_name = "openai-primary".into();
    let client = LlmClientBuilder::with_transport(fixture.clone(), &profiles)
        .with_region(Region::International)
        .build()
        .unwrap();
    let scoped = client
        .images()
        .generate_in("openai", &request("gpt-image-1.5"), &options())
        .await
        .unwrap();
    let qualified = client
        .images()
        .generate(&request("openai/gpt-image-1.5"), &options())
        .await
        .unwrap();
    assert_eq!(scoped.executed_profile, "openai-backup");
    assert_eq!(qualified.executed_profile, "openai-backup");
    let sent = fixture.requests();
    assert_eq!(sent.len(), 2);
    assert!(sent
        .iter()
        .all(|req| req.url == "https://backup.example/v1/images/generations"));
}
