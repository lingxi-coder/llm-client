use lingxi_llm_client::codecs::{
    openai::responses::OpenAiResponsesCodec, CodecContext, EncodeRequest, RequestMode, WireCodec,
};
use lingxi_llm_client::protocol::{
    CacheBreakpoint, CachePosition, CacheTtl, ChatRequest, LlmError, OpenAiPromptCacheMode,
    OpenAiPromptCacheOptions, OpenAiPromptCacheRetention, OpenAiPromptCacheTtl, ProviderProfile,
};
use serde_json::{json, Value};

fn profile(model: &str, extra: Value) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openai",
        "profile_name":"openai",
        "base_url":"https://api.openai.com/v1",
        "protocol":"open_ai_responses",
        "auth":"none",
        "models":[{"display_model":model,"request_model":model,"billing_model":model}],
        "extra":extra
    }))
    .unwrap()
}

fn request(model: &str) -> ChatRequest {
    serde_json::from_value(json!({
        "model":model,
        "system":[{"text":"Stable system rules"},{"text":"More stable rules"}],
        "messages":[{"role":"user","content":[
            {"type":"text","text":"First question"},
            {"type":"text","text":"Second question"}
        ]}]
    }))
    .unwrap()
}

fn encode_with(
    request: &ChatRequest,
    profile: &ProviderProfile,
    request_model: &str,
) -> Result<Value, LlmError> {
    let context = CodecContext::new(profile, request_model, RequestMode::Complete);
    OpenAiResponsesCodec
        .encode_request(EncodeRequest::new(request), &context)
        .map(|wire| serde_json::from_slice(&wire.body).unwrap())
}

fn encode(request: &ChatRequest) -> Result<Value, LlmError> {
    let profile = profile(&request.model, Value::Null);
    encode_with(request, &profile, &request.model)
}

fn breakpoint(position: CachePosition) -> CacheBreakpoint {
    CacheBreakpoint {
        position,
        ttl: CacheTtl::ThirtyMinutes,
    }
}

#[test]
fn key_options_and_existing_content_breakpoints_encode_on_responses() {
    let mut request = request("gpt-5.6-sol");
    request.prompt_cache.prompt_cache_key = Some("support:customer-17".into());
    request.prompt_cache.prompt_cache_options = Some(OpenAiPromptCacheOptions {
        mode: Some(OpenAiPromptCacheMode::Implicit),
        ttl: Some(OpenAiPromptCacheTtl::ThirtyMinutes),
    });
    request.prompt_cache.breakpoints = vec![
        breakpoint(CachePosition::System { index: 0 }),
        breakpoint(CachePosition::Message { index: 0, block: 1 }),
    ];

    let body = encode(&request).unwrap();
    assert_eq!(body["prompt_cache_key"], "support:customer-17");
    assert_eq!(
        body["prompt_cache_options"],
        json!({"mode":"implicit","ttl":"30m"})
    );
    assert!(body.get("instructions").is_none());
    assert_eq!(body["input"][0]["role"], "developer");
    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(
        body["input"][0]["content"][0]["prompt_cache_breakpoint"],
        json!({"mode":"explicit"})
    );
    assert_eq!(body["input"][0]["content"][1]["text"], "\n\n");
    assert!(body["input"][0]["content"][2]
        .get("prompt_cache_breakpoint")
        .is_none());
    assert_eq!(body["input"][1]["role"], "user");
    assert_eq!(
        body["input"][1]["content"][1]["prompt_cache_breakpoint"],
        json!({"mode":"explicit"})
    );
}

#[test]
fn explicit_mode_with_no_breakpoints_is_encoded_and_system_stays_in_instructions() {
    let mut request = request("gpt-5.6-sol");
    request.prompt_cache.prompt_cache_options = Some(OpenAiPromptCacheOptions {
        mode: Some(OpenAiPromptCacheMode::Explicit),
        ttl: None,
    });

    let body = encode(&request).unwrap();
    assert_eq!(body["prompt_cache_options"], json!({"mode":"explicit"}));
    assert_eq!(
        body["instructions"],
        "Stable system rules\n\nMore stable rules"
    );
    assert_eq!(body["input"][0]["role"], "user");
    assert!(body["input"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|item| item["content"].as_array().into_iter().flatten())
        .all(|part| part.get("prompt_cache_breakpoint").is_none()));
}

#[test]
fn implicit_and_explicit_modes_use_three_and_four_explicit_slots() {
    let mut request = request("gpt-5.6-sol");
    request.messages = (0..4)
        .map(|index| {
            serde_json::from_value(json!({
                "role":"user",
                "content":[{"type":"text","text":format!("Question {index}")}]
            }))
            .unwrap()
        })
        .collect();
    request.prompt_cache.breakpoints = (0..4)
        .map(|index| breakpoint(CachePosition::Message { index, block: 0 }))
        .collect();

    assert!(encode(&request).is_err());
    request.prompt_cache.prompt_cache_options = Some(OpenAiPromptCacheOptions {
        mode: Some(OpenAiPromptCacheMode::Implicit),
        ttl: None,
    });
    assert!(encode(&request).is_err());

    request.prompt_cache.prompt_cache_options = Some(OpenAiPromptCacheOptions {
        mode: Some(OpenAiPromptCacheMode::Explicit),
        ttl: None,
    });
    let body = encode(&request).unwrap();
    assert_eq!(
        body["input"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|message| message["content"][0]
                .get("prompt_cache_breakpoint")
                .is_some())
            .count(),
        4
    );
}

#[test]
fn retention_is_independent_from_ttl_and_never_a_fallback() {
    let mut request = request("gpt-5.5");
    request.prompt_cache.prompt_cache_key = Some("workspace-17".into());
    request.prompt_cache.prompt_cache_retention = Some(OpenAiPromptCacheRetention::TwentyFourHours);
    let body = encode(&request).unwrap();
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert_eq!(body["prompt_cache_key"], "workspace-17");
    assert!(body.get("prompt_cache_options").is_none());

    request.prompt_cache.prompt_cache_retention = None;
    request.prompt_cache.prompt_cache_options = Some(OpenAiPromptCacheOptions {
        mode: None,
        ttl: Some(OpenAiPromptCacheTtl::ThirtyMinutes),
    });
    assert!(encode(&request).is_err());

    request.model = "gpt-6-astra".into();
    request.prompt_cache.prompt_cache_options = Some(OpenAiPromptCacheOptions {
        mode: Some(OpenAiPromptCacheMode::Implicit),
        ttl: Some(OpenAiPromptCacheTtl::ThirtyMinutes),
    });
    request.prompt_cache.prompt_cache_retention = Some(OpenAiPromptCacheRetention::TwentyFourHours);
    request.prompt_cache.breakpoints =
        vec![breakpoint(CachePosition::Message { index: 0, block: 0 })];
    let body = encode(&request).unwrap();
    assert_eq!(
        body["prompt_cache_options"],
        json!({"mode":"implicit","ttl":"30m"})
    );
    assert_eq!(body["prompt_cache_retention"], "24h");

    request.prompt_cache.prompt_cache_retention = Some(OpenAiPromptCacheRetention::InMemory);
    assert!(encode(&request).is_err());

    request.prompt_cache.prompt_cache_retention = None;
    request.prompt_cache.prompt_cache_options = None;
    request.prompt_cache.breakpoints.clear();
    request.prompt_cache.automatic = Some(CacheTtl::ThirtyMinutes);
    assert!(encode(&request).is_err());
}

#[test]
fn profile_cache_fields_and_other_responses_adapters_are_rejected() {
    let request = request("gpt-5.6-sol");
    let conflicting_profile = profile(
        &request.model,
        json!({"body":{"prompt_cache_options":{"mode":"implicit"}}}),
    );
    assert!(encode_with(&request, &conflicting_profile, &request.model).is_err());

    let mut typed = request.clone();
    typed.prompt_cache.prompt_cache_key = Some("tenant-17".into());
    assert!(encode_with(&typed, &conflicting_profile, &typed.model).is_err());

    let compatible_profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"acme",
        "profile_name":"acme",
        "base_url":"https://api.acme.example/v1",
        "protocol":"open_ai_responses",
        "auth":"none"
    }))
    .unwrap();
    assert!(encode_with(&typed, &compatible_profile, &typed.model).is_err());
}

#[test]
fn unsupported_breakpoint_shapes_are_rejected() {
    let mut request = request("gpt-5.6-sol");
    request.prompt_cache.breakpoints = vec![CacheBreakpoint {
        position: CachePosition::Tool { index: 0 },
        ttl: CacheTtl::ThirtyMinutes,
    }];
    assert!(encode(&request).is_err());

    request.prompt_cache.breakpoints = vec![CacheBreakpoint {
        position: CachePosition::Message { index: 0, block: 0 },
        ttl: CacheTtl::FiveMinutes,
    }];
    assert!(encode(&request).is_err());

    request.prompt_cache.breakpoints =
        vec![breakpoint(CachePosition::Message { index: 0, block: 0 })];
    request.messages[0].role = lingxi_llm_client::protocol::MessageRole::Assistant;
    assert!(encode(&request).is_err());
}

#[test]
fn message_indices_follow_split_tool_and_native_items() {
    let mut request = request("gpt-5.6-sol");
    request.messages = serde_json::from_value(json!([
        {"role":"user","content":[
            {"type":"text","text":"Before"},
            {"type":"tool_result","tool_use_id":"call-1","content":"Tool output","is_error":false},
            {"type":"provider_content","protocol":"open_ai_responses","value":{
                "type":"message","role":"user","content":[{"type":"input_text","text":"Native text"}]
            }},
            {"type":"text","text":"After"}
        ]}
    ]))
    .unwrap();
    request.prompt_cache.breakpoints = vec![
        breakpoint(CachePosition::Message { index: 0, block: 1 }),
        breakpoint(CachePosition::Message { index: 0, block: 2 }),
    ];

    let body = encode(&request).unwrap();
    assert_eq!(body["input"][0]["role"], "user");
    assert_eq!(body["input"][1]["type"], "function_call_output");
    assert_eq!(
        body["input"][1]["output"][0]["prompt_cache_breakpoint"],
        json!({"mode":"explicit"})
    );
    assert_eq!(body["input"][2]["type"], "message");
    assert_eq!(
        body["input"][2]["content"][0]["prompt_cache_breakpoint"],
        json!({"mode":"explicit"})
    );
    assert_eq!(body["input"][3]["role"], "user");
}
