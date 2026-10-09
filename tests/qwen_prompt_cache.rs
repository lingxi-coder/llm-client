use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::{
    codecs::{CodecContext, EncodeRequest, RequestMode},
    protocol::*,
    *,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"qwen", "profile_name":"qwen-test", "protocol":"open_ai_chat",
        "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1", "auth":"bearer",
        "models":[{"display_model":"selected", "request_model":"qwen3.8-max", "billing_model":"qwen3.8-max",
            "capability_support":{"vision":"supported"}, "metadata":{"inputModalities":["text","image"]}}]
    })).unwrap()
}
fn request() -> ChatRequest {
    serde_json::from_value(
        json!({"model":"selected","system":[{"text":"first"},{"text":"second"}],
        "messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}),
    )
    .unwrap()
}
fn breakpoint(position: CachePosition) -> CacheBreakpoint {
    CacheBreakpoint {
        scope: None,
        position,
        ttl: CacheTtl::FiveMinutes,
    }
}
fn context(p: &ProviderProfile, model: &str, stream: bool) -> CodecContext {
    CodecContext::new(
        p,
        model,
        if stream {
            RequestMode::Stream
        } else {
            RequestMode::Complete
        },
    )
}
fn encode(
    req: &ChatRequest,
    p: &ProviderProfile,
    model: &str,
    stream: bool,
) -> Result<Value, LlmError> {
    let ctx = context(p, model, stream);
    let wire = OpenAiChatCodec.encode_request(EncodeRequest::new(req), &ctx)?;
    assert_eq!(
        wire.body.len(),
        OpenAiChatCodec.encoded_body_len(EncodeRequest::new(req), &ctx)?
    );
    Ok(serde_json::from_slice(&wire.body).unwrap())
}

#[test]
fn implicit_cache_needs_no_wire_control_and_streams_request_usage() {
    let req = request();
    let body = encode(&req, &profile(), "qwen3.8-max", false).unwrap();
    assert_eq!(body["messages"][0]["content"], "first\n\nsecond");
    assert_eq!(body["messages"][1]["content"], "hello");
    assert!(body.get("cache_control").is_none());
    assert!(body.get("prompt_cache_options").is_none());
    assert_eq!(
        encode(&req, &profile(), "qwen3.8-max", true).unwrap()["stream_options"]["include_usage"],
        true
    );
}

#[test]
fn original_system_message_and_tool_result_positions_survive_wire_splitting() {
    let mut req = request();
    req.messages = serde_json::from_value(json!([
        {"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},
        {"role":"assistant","content":[{"type":"text","text":"checking"},{"type":"tool_use","id":"call-1","name":"lookup","input":{}}]},
        {"role":"user","content":[{"type":"text","text":"before"},{"type":"tool_result","tool_use_id":"call-1","content":"answer","is_error":false},{"type":"text","text":"after"}]}
    ])).unwrap();
    req.prompt_cache.breakpoints = vec![
        breakpoint(CachePosition::System { index: 1 }),
        breakpoint(CachePosition::Message { index: 0, block: 1 }),
        breakpoint(CachePosition::Message { index: 1, block: 0 }),
        breakpoint(CachePosition::Message { index: 2, block: 1 }),
    ];
    let before = serde_json::to_value(&req).unwrap();
    let body = encode(&req, &profile(), "qwen3.8-max", false).unwrap();
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 6);
    assert_eq!(messages[0]["content"][0]["text"], "first");
    assert_eq!(messages[0]["content"][1]["text"], "\n\n");
    assert_eq!(
        messages[0]["content"][2],
        json!({"type":"text","text":"second","cache_control":{"type":"ephemeral"}})
    );
    assert_eq!(messages[1]["content"][1]["text"], "\n");
    assert_eq!(
        messages[1]["content"][2]["cache_control"],
        json!({"type":"ephemeral"})
    );
    assert_eq!(
        messages[2]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert_eq!(messages[2]["tool_calls"][0]["id"], "call-1");
    assert_eq!(messages[3]["content"], "before");
    assert_eq!(messages[4]["role"], "tool");
    assert_eq!(messages[4]["tool_call_id"], "call-1");
    assert_eq!(messages[4]["content"][0]["text"], "answer");
    assert_eq!(
        messages[4]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert_eq!(messages[5]["content"], "after");
    assert_eq!(serde_json::to_value(&req).unwrap(), before);
}

#[test]
fn text_and_multimodal_markers_are_not_flattened_away() {
    let mut req = request();
    req.prompt_cache.breakpoints = vec![breakpoint(CachePosition::Message { index: 0, block: 0 })];
    let body = encode(&req, &profile(), "qwen3.8-max", false).unwrap();
    assert_eq!(
        body["messages"][1]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    req.messages[0].content.insert(
        0,
        ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.com/image.png".into(),
            },
        },
    );
    let body = encode(&req, &profile(), "qwen3-vl-plus", false).unwrap();
    assert_eq!(body["messages"][1]["content"][0]["type"], "image_url");
    assert_eq!(
        body["messages"][1]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert_eq!(body["messages"][1]["content"][1]["text"], "hello");
}

#[test]
fn intermediate_text_and_consecutive_tool_results_keep_distinct_markers() {
    let mut req = request();
    req.system[0].text.clear();
    req.messages = serde_json::from_value(json!([
        {"role":"user","content":[{"type":"text","text":"cached"},{"type":"text","text":"new"}]},
        {"role":"user","content":[
            {"type":"tool_result","tool_use_id":"one","content":"first","is_error":false},
            {"type":"tool_result","tool_use_id":"two","content":"second","is_error":false}
        ]}
    ]))
    .unwrap();
    req.prompt_cache.breakpoints = vec![
        breakpoint(CachePosition::System { index: 1 }),
        breakpoint(CachePosition::Message { index: 0, block: 0 }),
        breakpoint(CachePosition::Message { index: 1, block: 0 }),
        breakpoint(CachePosition::Message { index: 1, block: 1 }),
    ];
    let body = encode(&req, &profile(), "qwen3.8-max", false).unwrap();
    assert_eq!(body["messages"][0]["content"][0]["text"], "\n\n");
    assert_eq!(body["messages"][0]["content"][1]["text"], "second");
    let content = &body["messages"][1]["content"];
    assert_eq!(content[0]["cache_control"]["type"], "ephemeral");
    assert!(content[2].get("cache_control").is_none());
    assert_eq!(content[2]["text"], "new");
    for (index, id, text) in [(2, "one", "first"), (3, "two", "second")] {
        let message = &body["messages"][index];
        assert_eq!(message["tool_call_id"], id);
        assert_eq!(message["content"][0]["text"], text);
        assert_eq!(message["content"][0]["cache_control"]["type"], "ephemeral");
    }
}

#[test]
fn invalid_ttl_positions_and_ignored_controls_fail_preflight_and_encoding() {
    let mut cases = Vec::new();
    let mut req = request();
    req.prompt_cache.automatic = Some(CacheTtl::FiveMinutes);
    cases.push(req);
    let mut req = request();
    req.prompt_cache.breakpoints = vec![CacheBreakpoint {
        scope: None,
        position: CachePosition::System { index: 0 },
        ttl: CacheTtl::OneHour,
    }];
    cases.push(req);
    let mut req = request();
    req.prompt_cache.breakpoints = vec![breakpoint(CachePosition::Tool { index: 0 })];
    cases.push(req);
    let mut req = request();
    req.prompt_cache.breakpoints = vec![breakpoint(CachePosition::System { index: 99 })];
    cases.push(req);
    let mut req = request();
    req.prompt_cache.breakpoints = vec![breakpoint(CachePosition::Message {
        index: 0,
        block: 99,
    })];
    cases.push(req);
    let mut req = request();
    req.prompt_cache.breakpoints = vec![breakpoint(CachePosition::System { index: 0 }); 2];
    cases.push(req);
    let mut req = request();
    req.prompt_cache.breakpoints = vec![breakpoint(CachePosition::System { index: 0 }); 5];
    cases.push(req);
    let mut req = request();
    req.system[0].text.clear();
    req.prompt_cache.breakpoints = vec![breakpoint(CachePosition::System { index: 0 })];
    cases.push(req);
    let p = profile();
    let ctx = context(&p, "qwen3.8-max", false);
    for req in cases {
        assert!(OpenAiChatCodec.validate_request(&req, &ctx).is_err());
        assert!(encode(&req, &p, "qwen3.8-max", false).is_err());
    }
    for field in [
        "cache_control",
        "prompt_cache_options",
        "prompt_cache_key",
        "prompt_cache_retention",
    ] {
        let mut p = profile();
        p.extra = json!({"body":{field:{}}});
        assert!(OpenAiChatCodec
            .validate_request(&request(), &context(&p, "qwen3.8-max", false))
            .is_err());
    }
    for options in [json!(false), json!({"include_usage":false})] {
        let mut p = profile();
        p.extra = json!({"body":{"stream_options":options}});
        assert!(OpenAiChatCodec
            .validate_request(&request(), &context(&p, "qwen3.8-max", true))
            .is_err());
    }
}

#[test]
fn exact_model_region_and_deployment_scope_are_required() {
    let mut req = request();
    req.prompt_cache.breakpoints = vec![breakpoint(CachePosition::System { index: 0 })];
    for (url, scope, model, ok) in [
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            None,
            "qwen3.8-max",
            true,
        ),
        (
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            None,
            "qwen3.8-max",
            true,
        ),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            None,
            "qwen-long",
            false,
        ),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            None,
            "qwen3.8-max-future",
            false,
        ),
        (
            "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            None,
            "qwen3.8-max",
            false,
        ),
        (
            "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            Some("global"),
            "qwen3.8-max",
            true,
        ),
        (
            "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            Some("global"),
            "qwen3.7-plus",
            false,
        ),
        (
            "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            Some("us"),
            "qwen3.7-plus",
            true,
        ),
        (
            "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
            Some("global"),
            "qwen3.8-max",
            true,
        ),
        (
            "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
            Some("hong_kong"),
            "qwen3.8-max",
            false,
        ),
        (
            "https://workspace.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
            Some("global"),
            "qwen3.6-flash",
            false,
        ),
        (
            "https://workspace.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
            Some("eu"),
            "qwen3.6-flash",
            true,
        ),
        (
            "https://workspace.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1",
            Some("japan"),
            "qwen3.8-max",
            false,
        ),
        (
            "https://workspace.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1",
            Some("global"),
            "qwen3.8-max",
            true,
        ),
        (
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            None,
            "qwen3.8-max",
            true,
        ),
        (
            "https://evil.dashscope.aliyuncs.com/compatible-mode/v1",
            None,
            "qwen3.8-max",
            false,
        ),
        (
            "https://dashscope.aliyuncs.com/apps/anthropic",
            None,
            "qwen3.8-max",
            false,
        ),
    ] {
        let mut p = profile();
        p.base_url = url.into();
        if let Some(scope) = scope {
            p.extra = json!({"qwen_cache_deployment_scope":scope});
        }
        assert_eq!(
            encode(&req, &p, model, false).is_ok(),
            ok,
            "{url} {scope:?} {model}"
        );
    }
}

fn raw_usage() -> Value {
    json!({"prompt_tokens":2000,"completion_tokens":100,"total_tokens":2100,
        "prompt_tokens_details":{"cached_tokens":1200,"cache_creation_input_tokens":500},
        "completion_tokens_details":{"reasoning_tokens":20}})
}
fn response(usage: Value) -> HttpResponse {
    HttpResponse {status:200,headers:vec![],body:Bytes::from(json!({"model":"qwen3.8-max", "choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":usage}).to_string())}
}

#[test]
fn cache_writes_partition_prompt_usage_only_for_qwen() {
    let mut p = profile();
    let report = OpenAiChatCodec
        .decode_response(&response(raw_usage()), &context(&p, "qwen3.8-max", false))
        .unwrap()
        .usage;
    let usage = report.complete().unwrap();
    assert_eq!(
        (
            usage.input_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens
        ),
        (300, 1200, 500)
    );
    assert_eq!(
        (
            usage.output_tokens,
            usage.reasoning_tokens,
            usage.cache_write_1h_tokens
        ),
        (100, 20, 0)
    );
    assert_eq!(usage.total(), 2100);
    p.provider_id = ProviderId::new("openai");
    let report = OpenAiChatCodec
        .decode_response(&response(raw_usage()), &context(&p, "qwen3.8-max", false))
        .unwrap()
        .usage;
    assert_eq!(report.complete().unwrap().cache_write_tokens, 0);
    assert_eq!(report.complete().unwrap().input_tokens, 800);
}

#[test]
fn fragmented_stream_usage_and_zero_corrections_match_complete_responses() {
    let p = profile();
    let ctx = context(&p, "qwen3.8-max", true);
    let mut decoder = OpenAiChatCodec.stream_decoder(&ctx);
    let mut u = raw_usage();
    let first = format!(
        "data: {}\n\n",
        json!({"model":"qwen3.8-max","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":u})
    );
    u["prompt_tokens_details"]["cache_creation_input_tokens"] = json!(0);
    let data = format!(
        "{first}data: {}\n\ndata: [DONE]\n\n",
        json!({"choices":[],"usage":u})
    );
    let mut terminal = None;
    for chunk in data.as_bytes().chunks(7) {
        for event in decoder.push_bytes(chunk) {
            if let StreamEvent::End { usage, .. } = event.unwrap() {
                terminal = Some(usage);
            }
        }
    }
    assert!(decoder.finish().is_empty());
    let expected = OpenAiChatCodec
        .decode_response(&response(u), &ctx)
        .unwrap()
        .usage;
    assert_eq!(terminal.unwrap(), expected);
    assert_eq!(decoder.usage_report(), expected);
    assert_eq!(expected.complete().unwrap().cache_write_tokens, 0);
}

#[test]
fn malformed_or_conflicting_cache_counters_are_not_billable_complete_reports() {
    for creation in [
        json!("500"),
        json!(-1),
        Value::Null,
        json!(1000),
        json!(u64::MAX),
    ] {
        let mut raw = raw_usage();
        raw["prompt_tokens_details"]["cache_creation_input_tokens"] = creation;
        let report = OpenAiChatCodec
            .decode_response(&response(raw), &context(&profile(), "qwen3.8-max", false))
            .unwrap()
            .usage;
        assert!(report.complete().is_none());
    }
    let mut raw = raw_usage();
    raw["prompt_tokens_details"]["cache_write_tokens"] = json!(499);
    assert!(OpenAiChatCodec
        .decode_response(&response(raw), &context(&profile(), "qwen3.8-max", false))
        .unwrap()
        .usage
        .complete()
        .is_none());
    for malformed in [Value::Null, json!("bad"), json!([])] {
        let mut raw = raw_usage();
        raw["prompt_tokens_details"] = malformed;
        assert!(OpenAiChatCodec
            .decode_response(&response(raw), &context(&profile(), "qwen3.8-max", false))
            .unwrap()
            .usage
            .complete()
            .is_none());
    }
}

#[derive(Default)]
struct NoSideEffects {
    resolves: AtomicUsize,
    auth: AtomicUsize,
    sends: AtomicUsize,
}
#[async_trait]
impl AttachmentResolver for NoSideEffects {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        self.resolves.fetch_add(1, Ordering::SeqCst);
        panic!("invalid cache must fail before resolving attachments")
    }
}
#[async_trait]
impl Authenticator for NoSideEffects {
    async fn apply(
        &self,
        _: &mut HttpRequest,
        _: &ProviderProfile,
        _: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        self.auth.fetch_add(1, Ordering::SeqCst);
        panic!("invalid cache must fail before auth")
    }
}
#[async_trait]
impl Transport for NoSideEffects {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sends.fetch_add(1, Ordering::SeqCst);
        panic!("invalid cache must fail before transport")
    }
}
#[tokio::test]
async fn invalid_cache_policy_fails_before_attachment_resolution_auth_or_send() {
    let io = Arc::new(NoSideEffects::default());
    let p = profile();
    let mut builder = LlmClientBuilder::with_transport(io.clone(), &[p]);
    builder
        .with_attachment_resolver(io.clone())
        .register_authenticator(AuthStrategy::Bearer, io.clone());
    let client = builder.with_region(Region::ChinaMainland).build().unwrap();
    let mut req = request();
    req.prompt_cache.automatic = Some(CacheTtl::OneHour);
    req.messages[0].content.push(ContentBlock::Image {
        source: ImageSource::Attachment {
            attachment: AttachmentRef {
                attachment_id: "image".into(),
                revision: "1".into(),
                filename: "image.png".into(),
                media_type: "image/png".into(),
                size_bytes: 4,
            },
        },
    });
    assert!(matches!(
        client
            .chat()
            .complete(&req, &RequestOptions::default())
            .await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert!(matches!(
        client.chat().stream(&req, &RequestOptions::default()).await,
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert_eq!(
        (
            io.resolves.load(Ordering::SeqCst),
            io.auth.load(Ordering::SeqCst),
            io.sends.load(Ordering::SeqCst)
        ),
        (0, 0, 0)
    );
}

#[test]
fn marked_text_only_tool_result_keeps_each_text_literal() {
    let raw = r#"[{"type":"text","text":"a \"x\""},{"type":"text","text":"b é"}]"#;
    let display = lingxi_llm_client::exact_json::parse_tool_output_json(raw).unwrap();
    let mut req = request();
    req.messages = serde_json::from_value(json!([
        {"role":"assistant","content":[{"type":"tool_use","id":"call-1","name":"lookup","input":{}}]},
        {"role":"user","content":[]}
    ]))
    .unwrap();
    req.messages[1].content.push(ContentBlock::ToolResult {
        cache_reference: None,
        tool_use_id: "call-1".into(),
        content: raw.into(),
        is_error: None,
        blocks: display.as_array().cloned(),
        output_json: Some(raw.into()),
        toolset_name: None,
    });
    req.prompt_cache.breakpoints = vec![breakpoint(CachePosition::Message { index: 1, block: 0 })];
    let p = profile();
    let ctx = context(&p, "qwen3.8-max", false);
    let wire = OpenAiChatCodec
        .encode_request(EncodeRequest::new(&req), &ctx)
        .unwrap();
    let bytes = String::from_utf8(wire.body.to_vec()).unwrap();
    assert!(
        bytes.contains(
            r#""content":[{"type":"text","text":"a \"x\"\nb é","cache_control":{"type":"ephemeral"}}]"#
        ),
        "{bytes}"
    );
    assert_eq!(
        wire.body.len(),
        OpenAiChatCodec
            .encoded_body_len(EncodeRequest::new(&req), &ctx)
            .unwrap()
    );
}
