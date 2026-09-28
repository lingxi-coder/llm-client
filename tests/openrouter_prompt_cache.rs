use lingxi_llm_client::{
    codecs::{
        anthropic::AnthropicMessagesCodec, openai::chat::OpenAiChatCodec, EncodeRequest, WireCodec,
    },
    protocol::{
        CacheBreakpoint, CachePosition, CacheTtl, ChatRequest, ContentBlock, PromptCachePolicy,
        ProviderProfile,
    },
    RequestMode,
};
use serde_json::{json, Value};

fn profile(model: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openrouter",
        "profile_name":"openrouter",
        "base_url":"https://openrouter.ai/api/v1",
        "protocol":"open_ai_chat",
        "auth":"none",
        "models":[{"display_model":model,"request_model":model,"billing_model":model}]
    }))
    .unwrap()
}

fn sample_request(model: &str) -> ChatRequest {
    serde_json::from_value(json!({
        "model":model,
        "messages":[{"role":"user","content":[
            {"type":"text","text":"Stable reference"},
            {"type":"text","text":"Current question"}
        ]}]
    }))
    .unwrap()
}

fn context(profile: &ProviderProfile, model: &str) -> lingxi_llm_client::codecs::CodecContext {
    lingxi_llm_client::codecs::CodecContext::new(profile, model, RequestMode::Complete)
}

fn encode(
    request: &ChatRequest,
    profile: &ProviderProfile,
) -> Result<Value, lingxi_llm_client::protocol::LlmError> {
    let context = context(profile, &request.model);
    let wire = OpenAiChatCodec.encode_request(EncodeRequest::new(request), &context)?;
    assert!(wire.headers.iter().all(|(name, _)| {
        !name.eq_ignore_ascii_case("X-OpenRouter-Cache")
            && !name.eq_ignore_ascii_case("X-OpenRouter-Cache-TTL")
    }));
    Ok(serde_json::from_slice(&wire.body).unwrap())
}

fn breakpoint(position: CachePosition, ttl: CacheTtl) -> CacheBreakpoint {
    CacheBreakpoint {
        scope: None,
        position,
        ttl,
    }
}

#[test]
fn anthropic_route_encodes_automatic_and_explicit_cache_control_separately() {
    let model = "anthropic/claude-sonnet-5";
    let mut request = sample_request(model);
    request.system = vec![lingxi_llm_client::protocol::SystemBlock {
        text: "Stable system policy".into(),
    }];
    request.prompt_cache = PromptCachePolicy {
        automatic: Some(CacheTtl::FiveMinutes),
        breakpoints: vec![
            breakpoint(CachePosition::System { index: 0 }, CacheTtl::OneHour),
            breakpoint(
                CachePosition::Message { index: 0, block: 0 },
                CacheTtl::FiveMinutes,
            ),
        ],
        ..PromptCachePolicy::default()
    };
    let wire = encode(&request, &profile(model)).unwrap();
    assert_eq!(wire["cache_control"], json!({"type":"ephemeral"}));
    assert_eq!(
        wire["messages"][0]["content"][0]["cache_control"],
        json!({"type":"ephemeral","ttl":"1h"})
    );
    assert_eq!(
        wire["messages"][1]["content"][0]["cache_control"],
        json!({"type":"ephemeral"})
    );
    assert!(wire.get("prompt_cache_options").is_none());
}

#[test]
fn documented_alibaba_openrouter_models_use_a_five_minute_content_breakpoint() {
    let model = "qwen/qwen3.6-plus";
    let mut request = sample_request(model);
    request.prompt_cache.breakpoints.push(breakpoint(
        CachePosition::Message { index: 0, block: 0 },
        CacheTtl::FiveMinutes,
    ));
    let wire = encode(&request, &profile(model)).unwrap();
    assert_eq!(
        wire["messages"][0]["content"][0]["cache_control"],
        json!({"type":"ephemeral"})
    );
    assert!(wire.get("cache_control").is_none());
    assert!(wire.get("prompt_cache_options").is_none());
}

#[test]
fn openai_gpt_56_uses_thirty_minute_prompt_cache_options_and_explicit_breakpoints() {
    let model = "openai/gpt-5.6-sol";
    let mut request = sample_request(model);
    request.system = vec![lingxi_llm_client::protocol::SystemBlock {
        text: "Stable system policy".into(),
    }];
    request.prompt_cache.breakpoints = vec![
        breakpoint(CachePosition::System { index: 0 }, CacheTtl::ThirtyMinutes),
        breakpoint(
            CachePosition::Message { index: 0, block: 0 },
            CacheTtl::ThirtyMinutes,
        ),
    ];
    let wire = encode(&request, &profile(model)).unwrap();
    assert_eq!(
        wire["prompt_cache_options"],
        json!({"mode":"explicit","ttl":"30m"})
    );
    // OpenRouter accepts cache_control at the Chat API boundary and translates
    // it to the OpenAI provider's prompt_cache_breakpoint marker.
    assert_eq!(
        wire["messages"][0]["content"][0]["cache_control"],
        json!({"type":"ephemeral"})
    );
    assert_eq!(
        wire["messages"][1]["content"][0]["cache_control"],
        json!({"type":"ephemeral"})
    );
}

#[test]
fn openai_gpt_56_automatic_mode_is_distinct_from_explicit_only_mode() {
    let model = "openai/gpt-6-astra";
    let mut request = sample_request(model);
    request.prompt_cache.automatic = Some(CacheTtl::ThirtyMinutes);
    let wire = encode(&request, &profile(model)).unwrap();
    assert_eq!(
        wire["prompt_cache_options"],
        json!({"mode":"implicit","ttl":"30m"})
    );
    assert!(wire.get("cache_control").is_none());
    assert!(wire["messages"][0]["content"].is_string());
}

#[test]
fn unsupported_model_ttl_endpoint_and_breakpoint_shapes_fail_validation() {
    let model = "openai/gpt-5.5";
    let mut request = sample_request(model);
    request.prompt_cache.breakpoints.push(breakpoint(
        CachePosition::Message { index: 0, block: 0 },
        CacheTtl::ThirtyMinutes,
    ));
    assert!(OpenAiChatCodec
        .validate_request(&request, &context(&profile(model), model))
        .is_err());

    let model = "openai/gpt-5.6-sol";
    let mut request = sample_request(model);
    request.prompt_cache.automatic = Some(CacheTtl::OneHour);
    assert!(OpenAiChatCodec
        .validate_request(&request, &context(&profile(model), model))
        .is_err());

    let model = "anthropic/claude-sonnet-5";
    let mut request = sample_request(model);
    request.prompt_cache.automatic = Some(CacheTtl::ThirtyMinutes);
    assert!(OpenAiChatCodec
        .validate_request(&request, &context(&profile(model), model))
        .is_err());

    let model = "qwen/qwen3-max-thinking";
    let mut request = sample_request(model);
    request.prompt_cache.breakpoints.push(breakpoint(
        CachePosition::Message { index: 0, block: 0 },
        CacheTtl::FiveMinutes,
    ));
    assert!(OpenAiChatCodec
        .validate_request(&request, &context(&profile(model), model))
        .is_err());

    let model = "qwen/qwen3-max";
    let mut request = sample_request(model);
    request.prompt_cache.automatic = Some(CacheTtl::FiveMinutes);
    assert!(OpenAiChatCodec
        .validate_request(&request, &context(&profile(model), model))
        .is_err());

    let model = "anthropic/claude-sonnet-5";
    let mut request = sample_request(model);
    request.prompt_cache.breakpoints.push(breakpoint(
        CachePosition::Tool { index: 0 },
        CacheTtl::FiveMinutes,
    ));
    assert!(OpenAiChatCodec
        .validate_request(&request, &context(&profile(model), model))
        .is_err());

    let model = "openai/gpt-5.6-sol";
    let mut request = sample_request(model);
    request.prompt_cache.breakpoints.push(breakpoint(
        CachePosition::Message { index: 0, block: 0 },
        CacheTtl::ThirtyMinutes,
    ));
    let mut custom_profile = profile(model);
    custom_profile.base_url = "https://openrouter-proxy.example/api/v1".into();
    assert!(OpenAiChatCodec
        .validate_request(&request, &context(&custom_profile, model))
        .is_err());

    let model = "openai/gpt-5.6-sol";
    let mut request = sample_request(model);
    request.prompt_cache.breakpoints.push(breakpoint(
        CachePosition::Message { index: 0, block: 0 },
        CacheTtl::ThirtyMinutes,
    ));
    request.messages[0].content[0] = ContentBlock::Image {
        source: lingxi_llm_client::protocol::ImageSource::Url {
            url: "https://example.test/image.png".into(),
        },
    };
    assert!(OpenAiChatCodec
        .validate_request(&request, &context(&profile(model), model))
        .is_err());
}

#[test]
fn thirty_minute_ttl_is_rejected_on_direct_anthropic_routes() {
    let model = "claude-sonnet-4";
    let mut request = sample_request(model);
    request.prompt_cache.automatic = Some(CacheTtl::ThirtyMinutes);
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"anthropic",
        "profile_name":"anthropic",
        "base_url":"https://api.anthropic.com/v1",
        "protocol":"anthropic_messages",
        "auth":"none",
        "models":[{"display_model":model,"request_model":model,"billing_model":model}]
    }))
    .unwrap();
    let context = context(&profile, model);
    assert!(AnthropicMessagesCodec
        .encode_request(EncodeRequest::new(&request), &context)
        .is_err());
}

#[test]
fn provider_managed_caches_remain_automatic_without_explicit_controls() {
    let model = "x-ai/grok-4.7";
    let request = sample_request(model);
    let wire = encode(&request, &profile(model)).unwrap();
    assert!(wire.get("cache_control").is_none());
    assert!(wire.get("prompt_cache_options").is_none());
}
