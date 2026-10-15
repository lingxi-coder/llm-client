//! Official endpoint contracts, without provider credentials or network calls.
use lingxi_llm_client::{
    codecs::{CodecContext, EncodeRequest, RequestMode},
    protocol::*,
    providers::openai::chat_audio::{ChatAudioFormat, ChatAudioOutput},
    *,
};
use serde_json::{json, Value};
use std::sync::Arc;
mod support;
fn request(model: &str) -> ChatRequest {
    serde_json::from_value(json!({"model":model,"max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}]})).unwrap()
}
fn client() -> LlmClient {
    client_in_region(Region::International)
}
fn client_in_region(region: Region) -> LlmClient {
    LlmClientBuilder::with_transport(
        Arc::new(support::NoHttp),
        &builtin_providers()
            .unwrap()
            .into_iter()
            .filter(|p| p.provider_id.as_str() != "github-copilot")
            .collect::<Vec<_>>(),
    )
    .with_region(region)
    .build()
    .unwrap()
}
#[test]
fn adding_hidden_formats_preserves_unscoped_primary_resolution() {
    let c = client();
    assert_eq!(
        c.resolve("claude-sonnet-5").unwrap().profile_name,
        "anthropic"
    );
    assert_eq!(c.resolve("gpt-6.1-sol").unwrap().profile_name, "openai");
}
fn opts(protocol: ProtocolFamily) -> RequestOptions {
    RequestOptions {
        protocol: Some(protocol),
        credential: Some(Secret::new("fixture-key".into())),
        ..Default::default()
    }
}
#[tokio::test]
async fn selected_formats_use_the_official_path_and_body() {
    let client = client();
    for (profile, model, protocol, endpoint, field) in [
        (
            "deepseek",
            "deepseek-flash",
            ProtocolFamily::OpenAiResponses,
            "https://api.deepseek.com/responses",
            "input",
        ),
        (
            "minimax-intl",
            "MiniMax-M3",
            ProtocolFamily::OpenAiChat,
            "https://api.minimax.io/v1/chat/completions",
            "messages",
        ),
        (
            "minimax-intl",
            "MiniMax-M3",
            ProtocolFamily::OpenAiResponses,
            "https://api.minimax.io/v1/responses",
            "input",
        ),
        (
            "kimi-intl",
            "kimi-k3",
            ProtocolFamily::AnthropicMessages,
            "https://api.moonshot.ai/anthropic/v1/messages",
            "messages",
        ),
        (
            "kimi-code",
            "k3",
            ProtocolFamily::AnthropicMessages,
            "https://api.kimi.com/coding/v1/messages",
            "messages",
        ),
        (
            "kimi-code",
            "k3",
            ProtocolFamily::OpenAiResponses,
            "https://api.kimi.com/coding/v1/responses",
            "input",
        ),
        (
            "zai-coding",
            "glm-4.7",
            ProtocolFamily::OpenAiChat,
            "https://api.z.ai/api/coding/paas/v4/chat/completions",
            "messages",
        ),
        (
            "zai-coding",
            "glm-4.7",
            ProtocolFamily::OpenAiResponses,
            "https://api.z.ai/api/v1/responses",
            "input",
        ),
        (
            "qwen-intl",
            "qwen3.8-flash",
            ProtocolFamily::AnthropicMessages,
            "https://dashscope-intl.aliyuncs.com/apps/anthropic/v1/messages",
            "messages",
        ),
        (
            "openai",
            "gpt-6.1-sol",
            ProtocolFamily::OpenAiChat,
            "https://api.openai.com/v1/chat/completions",
            "messages",
        ),
        (
            "anthropic",
            "claude-sonnet-5",
            ProtocolFamily::OpenAiChat,
            "https://api.anthropic.com/v1/chat/completions",
            "messages",
        ),
        (
            "gemini",
            "gemini-3.7-flash",
            ProtocolFamily::OpenAiChat,
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "messages",
        ),
        (
            "openrouter",
            "openai/gpt-4o",
            ProtocolFamily::AnthropicMessages,
            "https://openrouter.ai/api/v1/messages",
            "messages",
        ),
        (
            "openrouter",
            "openai/gpt-4o",
            ProtocolFamily::OpenAiResponses,
            "https://openrouter.ai/api/v1/responses",
            "input",
        ),
    ] {
        let call = client
            .snapshot()
            .prepare_on(
                profile,
                &request(model),
                &opts(protocol),
                RequestMode::Stream,
            )
            .await
            .unwrap_or_else(|e| panic!("{profile}: {e}"));
        assert_eq!(call.request().url, endpoint, "{profile}");
        let body: Value = serde_json::from_slice(&call.request().body).unwrap();
        assert_eq!(body["model"], model);
        assert!(body[field].is_array());
        assert_eq!(body["stream"], true);
        assert!(call
            .request()
            .headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("authorization")
                || k.eq_ignore_ascii_case("x-api-key")));
    }
}
#[tokio::test]
async fn feature_controls_follow_the_selected_provider_protocol() {
    let c = client();
    let mut req = request("qwen3.8-flash");
    req.thinking = Some(ThinkingConfig {
        mode: Some(ThinkingMode::Enabled),
        effort: Some(ReasoningEffort::XHigh),
        ..Default::default()
    });
    let call = c
        .snapshot()
        .prepare_on(
            "qwen-intl",
            &req,
            &opts(ProtocolFamily::AnthropicMessages),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    let b: Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(b["thinking"]["type"], "enabled");
    assert_eq!(b["output_config"]["effort"], "xhigh");
    assert!(b.get("enable_thinking").is_none());
    let mut req = request("MiniMax-M3");
    req.thinking = Some(ThinkingConfig {
        mode: Some(ThinkingMode::Adaptive),
        ..Default::default()
    });
    let call = c
        .snapshot()
        .prepare_on(
            "minimax-intl",
            &req,
            &opts(ProtocolFamily::OpenAiChat),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    let b: Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(b["thinking"]["type"], "adaptive");
    assert_eq!(b["reasoning_split"], true);
    let mut req = request("claude-sonnet-5");
    req.output_format = OutputFormat::JsonObject;
    assert!(matches!(
        c.snapshot()
            .prepare_on(
                "anthropic",
                &req,
                &opts(ProtocolFamily::OpenAiChat),
                RequestMode::Stream
            )
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    req.output_format = OutputFormat::Text;
    req.thinking = Some(ThinkingConfig {
        effort: Some(ReasoningEffort::High),
        ..Default::default()
    });
    assert!(matches!(
        c.snapshot()
            .prepare_on(
                "anthropic",
                &req,
                &opts(ProtocolFamily::OpenAiChat),
                RequestMode::Stream
            )
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
}
#[tokio::test]
async fn search_stays_native_to_its_declared_format() {
    let c = client();
    let mut req = request("deepseek-flash");
    req.set_hosted_web_search(Some(WebSearchConfig::default()));
    assert!(matches!(
        c.snapshot()
            .prepare_on(
                "deepseek",
                &req,
                &opts(ProtocolFamily::OpenAiResponses),
                RequestMode::Stream
            )
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    let call = c
        .snapshot()
        .prepare_on(
            "deepseek",
            &req,
            &opts(ProtocolFamily::AnthropicMessages),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    let b: Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(b["tools"][0]["type"], "web_search_20250305");
    let mut req = request("grok-4.20");
    req.set_hosted_web_search(Some(WebSearchConfig::default()));
    let call = c
        .snapshot()
        .prepare_on(
            "grok",
            &req,
            &opts(ProtocolFamily::OpenAiResponses),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    let b: Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(b["tools"][0]["type"], "web_search");
    for protocol in [
        ProtocolFamily::AnthropicMessages,
        ProtocolFamily::OpenAiResponses,
    ] {
        let mut req = request("MiniMax-M3");
        req.set_hosted_web_search(Some(WebSearchConfig::default()));
        let call = c
            .snapshot()
            .prepare_on("minimax-intl", &req, &opts(protocol), RequestMode::Stream)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&call.request().body).unwrap();
        assert_eq!(
            body["tools"][0]["type"],
            if protocol == ProtocolFamily::AnthropicMessages {
                "web_search_20250305"
            } else {
                "web_search"
            }
        );
    }
    for protocol in [
        ProtocolFamily::OpenAiChat,
        ProtocolFamily::OpenAiResponses,
        ProtocolFamily::AnthropicMessages,
    ] {
        let mut req = request("openai/gpt-4o");
        req.set_hosted_web_search(Some(WebSearchConfig::default()));
        let call = c
            .snapshot()
            .prepare_on("openrouter", &req, &opts(protocol), RequestMode::Stream)
            .await
            .unwrap();
        let b: Value = serde_json::from_slice(&call.request().body).unwrap();
        assert_eq!(b["tools"][0]["type"], "openrouter:web_search");
    }
}
#[tokio::test]
async fn independent_audio_services_survive_conversation_format_changes() {
    let c = client();
    let snapshot = c.snapshot();
    for profile in ["openai", "openai-chat"] {
        let caps = snapshot
            .audio()
            .capabilities(&audio::AudioRoute::new(profile, "fixture-account"))
            .unwrap();
        assert!(caps.supports(audio::AudioOperation::FileTranscription));
        assert!(caps.supports(audio::AudioOperation::Synthesis));
    }
    let caps = snapshot
        .audio()
        .capabilities(&audio::AudioRoute::new(
            "deepseek-responses",
            "fixture-account",
        ))
        .unwrap();
    assert!(!caps.supports(audio::AudioOperation::Synthesis));
}
#[tokio::test]
async fn audio_input_output_and_history_use_the_chat_contract() {
    let c = client();
    let mut req = request("gpt-audio-1.5");
    req.messages[0].content.push(ContentBlock::Audio {
        format: "wav".into(),
        data: "AAECAw==".into(),
    });
    req.native_options.push(
        NativeExtension::from_typed(ChatAudioOutput {
            voice: "alloy".into(),
            format: ChatAudioFormat::Wav,
        })
        .unwrap(),
    );
    let call = c
        .snapshot()
        .prepare_on(
            "openai",
            &req,
            &opts(ProtocolFamily::OpenAiChat),
            RequestMode::Complete,
        )
        .await
        .unwrap();
    let b: Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(b["audio"], json!({"voice":"alloy","format":"wav"}));
    assert_eq!(b["modalities"], json!(["text", "audio"]));
    assert_eq!(
        b["messages"][0]["content"][1]["input_audio"]["format"],
        "wav"
    );
    assert!(c
        .snapshot()
        .prepare_on(
            "openai-chat",
            &req,
            &opts(ProtocolFamily::OpenAiChat),
            RequestMode::Stream
        )
        .await
        .is_err());
    let p = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "openai-chat")
        .unwrap();
    let context = CodecContext::new(&p, "gpt-audio-1.5", RequestMode::Complete);
    let response=OpenAiChatCodec.decode_response(&HttpResponse{status:200,headers:vec![],body:serde_json::to_vec(&json!({"id":"completion-1","model":"gpt-audio-1.5","choices":[{"message":{"role":"assistant","content":null,"audio":{"id":"audio-1","data":"AQIDBA==","transcript":"Hello","expires_at":12345}},"finish_reason":"stop"}]})).unwrap().into()},&context).unwrap();
    assert!(response.message.content.iter().any(|b| matches!(b,ContentBlock::ProviderContent{value,..} if value.string_at("/audio/data").as_deref()==Some("AQIDBA=="))));
    req.native_options.clear();
    req.messages.push(response.message);
    let encoded = OpenAiChatCodec
        .encode_request(EncodeRequest::new(&req), &context)
        .unwrap();
    let b: Value = serde_json::from_slice(&encoded.body).unwrap();
    assert_eq!(b["messages"][1]["audio"], json!({"id":"audio-1"}));
    let mut req = request("gemini-3.7-flash");
    req.messages[0].content.push(ContentBlock::Audio {
        format: "wav".into(),
        data: "AAECAw==".into(),
    });
    c.snapshot()
        .prepare_on(
            "gemini",
            &req,
            &opts(ProtocolFamily::OpenAiChat),
            RequestMode::Stream,
        )
        .await
        .unwrap();
}
#[test]
fn alternate_catalog_evidence_is_complete_and_keeps_unknowns() {
    let matrix: Value =
        serde_json::from_str(include_str!("../data/capability-matrix-protocols.json")).unwrap();
    assert_eq!(matrix["live_validation"], "not_run");
    for p in matrix["profiles"].as_array().unwrap() {
        let raw: toml::Value = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(p["catalog_path"].as_str().unwrap()),
        )
        .unwrap()
        .parse()
        .unwrap();
        assert_eq!(p["protocol"].as_str(), raw["protocol"].as_str());
        assert_eq!(p["base_url"].as_str(), raw["base_url"].as_str());
        assert_eq!(
            p["models"].as_array().unwrap().len(),
            raw["model"].as_array().unwrap().len()
        );
        for m in p["models"].as_array().unwrap() {
            assert_eq!(m["live_validation"], "not_run");
            assert_eq!(
                m["cells"].as_object().unwrap().len(),
                p["operation_ids"].as_array().unwrap().len()
            );
            for cell in m["cells"].as_object().unwrap().values() {
                for id in cell["source_ids"].as_array().unwrap() {
                    let source = &matrix["sources"][id.as_str().unwrap()];
                    assert!(source["url"].as_str().unwrap().starts_with("https://"));
                    assert_eq!(source["evidence_date"], "2026-10-12");
                }
            }
        }
    }
}

#[tokio::test]
async fn minimax_thinking_and_kimi_cache_keep_their_native_semantics() {
    let c = client();
    let mut req = request("MiniMax-M3");
    req.thinking = Some(ThinkingConfig {
        mode: Some(ThinkingMode::Enabled),
        ..Default::default()
    });
    let call = c
        .snapshot()
        .prepare_on(
            "minimax-intl",
            &req,
            &opts(ProtocolFamily::OpenAiResponses),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(body["reasoning"]["effort"], "high");
    req.thinking = Some(ThinkingConfig {
        mode: Some(ThinkingMode::Disabled),
        ..Default::default()
    });
    let call = c
        .snapshot()
        .prepare_on(
            "minimax-intl",
            &req,
            &opts(ProtocolFamily::OpenAiResponses),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(body["reasoning"]["effort"], "none");
    req.tool_choice = ToolChoice::Any;
    assert!(c
        .snapshot()
        .prepare_on(
            "minimax-intl",
            &req,
            &opts(ProtocolFamily::OpenAiResponses),
            RequestMode::Stream
        )
        .await
        .is_err());
    let mut req = request("kimi-k3");
    req.prompt_cache.automatic = Some(CacheTtl::OneHour);
    let call = c
        .snapshot()
        .prepare_on(
            "kimi-intl",
            &req,
            &opts(ProtocolFamily::AnthropicMessages),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(
        body["cache_control"],
        json!({"type":"ephemeral","ttl":"1h"})
    );
    req.prompt_cache.breakpoints.push(CacheBreakpoint {
        position: CachePosition::Message { index: 0, block: 0 },
        ttl: CacheTtl::FiveMinutes,
        scope: None,
    });
    assert!(c
        .snapshot()
        .prepare_on(
            "kimi-intl",
            &req,
            &opts(ProtocolFamily::AnthropicMessages),
            RequestMode::Stream
        )
        .await
        .is_err());
    req.prompt_cache.breakpoints.clear();
    req.model = "k3".into();
    assert!(c
        .snapshot()
        .prepare_on(
            "kimi-code",
            &req,
            &opts(ProtocolFamily::AnthropicMessages),
            RequestMode::Stream
        )
        .await
        .is_err());
}

#[test]
fn malformed_http_errors_preserve_status_classification_and_retry_after() {
    for (wire, codec) in [
        (
            ProtocolFamily::OpenAiChat,
            Box::new(OpenAiChatCodec) as Box<dyn WireCodec>,
        ),
        (
            ProtocolFamily::OpenAiResponses,
            Box::new(OpenAiResponsesCodec),
        ),
        (
            ProtocolFamily::AnthropicMessages,
            Box::new(AnthropicMessagesCodec),
        ),
    ] {
        let p: ProviderProfile = serde_json::from_value(json!({"profile_name":"fixture","provider_id":"fixture","protocol":wire,"auth":"none","base_url":"https://fixture.test","models":[]})).unwrap();
        let ctx = CodecContext::new(&p, "m", RequestMode::Complete);
        for (status, kind) in [
            (400, LlmErrorKind::InvalidRequest),
            (401, LlmErrorKind::Authentication),
            (429, LlmErrorKind::RateLimited),
            (500, LlmErrorKind::ProviderInternal),
        ] {
            let response = HttpResponse {
                status,
                headers: vec![("retry-after".into(), "5".into())],
                body: bytes::Bytes::from_static(b"<html>truncated"),
            };
            let error = codec.decode_error_response(&response, &ctx);
            assert_eq!(error.kind(), kind, "{wire:?} HTTP {status}: {error}");
            if status == 429 {
                match error.classification_source() {
                    LlmError::RateLimited { retry_after, .. }
                    | LlmError::ProviderResponse { retry_after, .. } => {
                        assert_eq!(*retry_after, Some(std::time::Duration::from_secs(5)))
                    }
                    _ => panic!("rate limit lost its retry metadata"),
                }
            }
        }
        assert!(codec
            .decode_response(
                &HttpResponse {
                    status: 200,
                    headers: vec![],
                    body: bytes::Bytes::from_static(b"invalid")
                },
                &ctx
            )
            .is_err());
    }
}

#[tokio::test]
async fn kimi_k3_chat_rejects_unsupported_vision_urls_and_fixed_parameter_changes() {
    let c = client_in_region(Region::ChinaMainland);
    let mut req = request("kimi-k3");
    req.temperature = Some(0.5);
    assert!(matches!(
        c.snapshot()
            .prepare_on(
                "kimi",
                &req,
                &opts(ProtocolFamily::OpenAiChat),
                RequestMode::Stream
            )
            .await,
        Err(LlmError::InvalidRequest { .. })
    ));
    req.temperature = None;
    req.messages[0].content.push(ContentBlock::Image {
        source: ImageSource::Url {
            url: "https://example.com/image.png".into(),
        },
    });
    assert!(matches!(
        c.snapshot()
            .prepare_on(
                "kimi",
                &req,
                &opts(ProtocolFamily::OpenAiChat),
                RequestMode::Stream
            )
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    req.messages[0].content.pop();
    req.thinking = Some(ThinkingConfig {
        effort: Some(ReasoningEffort::High),
        ..Default::default()
    });
    let call = c
        .snapshot()
        .prepare_on(
            "kimi",
            &req,
            &opts(ProtocolFamily::OpenAiChat),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(body["reasoning_effort"], "high");
    assert!(body.get("thinking").is_none());
}

#[tokio::test]
async fn minimax_chat_video_preserves_scopes_and_documented_request_fields() {
    let c = client();
    let mut req = request("MiniMax-M3");
    req.messages[0].content.push(ContentBlock::Video {
        source: VideoSource::Url {
            url: "https://example.com/movie.mp4".into(),
        },
    });
    let call = c
        .snapshot()
        .prepare_on(
            "minimax-intl",
            &req,
            &opts(ProtocolFamily::OpenAiChat),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(
        body["messages"][0]["content"][1],
        json!({"type":"video_url","video_url":{"url":"https://example.com/movie.mp4"}})
    );
    assert_eq!(body["max_completion_tokens"], 64);
    assert_eq!(body["stream_options"]["include_usage"], true);
    req.thinking = Some(ThinkingConfig {
        effort: Some(ReasoningEffort::High),
        ..Default::default()
    });
    assert!(matches!(
        c.snapshot()
            .prepare_on(
                "minimax-intl",
                &req,
                &opts(ProtocolFamily::OpenAiChat),
                RequestMode::Stream
            )
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    req.thinking = None;
    req.model = "MiniMax-M2.7".into();
    assert!(c
        .snapshot()
        .prepare_on(
            "minimax-intl",
            &req,
            &opts(ProtocolFamily::OpenAiChat),
            RequestMode::Stream
        )
        .await
        .is_err());
    req.model = "MiniMax-M3".into();
    req.messages[0].content[1] = ContentBlock::Video {
        source: VideoSource::Base64 {
            media_type: "video/mp4".into(),
            data: "AAECAw==".into(),
        },
    };
    let call = c
        .snapshot()
        .prepare_on(
            "minimax-intl",
            &req,
            &opts(ProtocolFamily::OpenAiChat),
            RequestMode::Stream,
        )
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(
        body["messages"][0]["content"][1]["video_url"]["url"],
        "data:video/mp4;base64,AAECAw=="
    );
    let p = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "minimax-chat-intl")
        .unwrap();
    let file:ProviderFileSource=serde_json::from_value(json!({"protocol":"open_ai_chat","provider_id":"minimax","profile_name":p.profile_name,"endpoint_fingerprint":files::provider_file_endpoint_fingerprint(&p.base_url),"account_scope":"fixture-account","file_id":"1234567890123456789","uri":"mm_file://1234567890123456789","media_type":"video/mp4","purpose":"video_understanding"})).unwrap();
    req.messages[0].content[1] = ContentBlock::Video {
        source: VideoSource::ProviderFile { file },
    };
    let mut options = opts(ProtocolFamily::OpenAiChat);
    options.file_account_scope = Some("fixture-account".into());
    let call = c
        .snapshot()
        .prepare_on("minimax-intl", &req, &options, RequestMode::Stream)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&call.request().body).unwrap();
    assert_eq!(
        body["messages"][0]["content"][1]["video_url"]["url"],
        "mm_file://1234567890123456789"
    );
    options.file_account_scope = Some("another-account".into());
    assert!(c
        .snapshot()
        .prepare_on("minimax-intl", &req, &options, RequestMode::Stream)
        .await
        .is_err());
    let attachment = AttachmentRef {
        attachment_id: "video-1".into(),
        revision: "1".into(),
        filename: "movie.mp4".into(),
        media_type: "video/mp4".into(),
        size_bytes: 4,
    };
    req.messages[0].content[1] = ContentBlock::Video {
        source: VideoSource::Attachment {
            attachment: attachment.clone(),
        },
    };
    let media = [lingxi_llm_client::codecs::PreparedMedia {
        attachment: &attachment,
        bytes: &[0, 1, 2, 3],
    }];
    let context = CodecContext::new(&p, "MiniMax-M3", RequestMode::Complete);
    let call = OpenAiChatCodec
        .encode_request(EncodeRequest::new(&req).with_media(&media), &context)
        .unwrap();
    let body: Value = serde_json::from_slice(&call.body).unwrap();
    assert_eq!(
        body["messages"][0]["content"][1]["video_url"]["url"],
        "data:video/mp4;base64,AAECAw=="
    );
}
