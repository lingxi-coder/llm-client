use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn request(format: OutputFormat) -> ChatRequest {
    let mut r: ChatRequest = serde_json::from_value(json!({"model":"test","messages":[]})).unwrap();
    r.output_format = format;
    r
}
fn schema() -> OutputFormat {
    OutputFormat::JsonSchema {
        name: "answer".into(),
        strict: true,
        schema: json!({
            "type":"object","properties":{"answer":{"type":"string"}},
            "required":["answer"],"additionalProperties":false
        }),
    }
}
fn profile(protocol: ProtocolFamily) -> ProviderProfile {
    serde_json::from_value(json!({"provider_id":"test","profile_name":"test","base_url":"https://test.invalid/v1","protocol":protocol,"auth":"none"})).unwrap()
}
fn qwen_chat_profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"qwen",
        "profile_name":"qwen",
        "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1",
        "protocol":"open_ai_chat",
        "auth":"none"
    }))
    .unwrap()
}
fn deepseek_chat_profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"deepseek",
        "profile_name":"deepseek",
        "base_url":"https://api.deepseek.com",
        "protocol":"open_ai_chat",
        "auth":"none"
    }))
    .unwrap()
}
fn encode_model(
    codec: &dyn WireCodec,
    req: &ChatRequest,
    p: &ProviderProfile,
    model: &str,
) -> Result<Value, LlmError> {
    let context = CodecContext::new(p, model, RequestMode::Complete);
    codec
        .encode_request(EncodeRequest::new(req), &context)
        .map(|r| serde_json::from_slice(&r.body).unwrap())
}
fn encode(
    codec: &dyn WireCodec,
    req: &ChatRequest,
    p: &ProviderProfile,
    stream: bool,
) -> Result<Value, LlmError> {
    let context = CodecContext::new(
        p,
        "test",
        if stream {
            RequestMode::Stream
        } else {
            RequestMode::Complete
        },
    );
    codec
        .encode_request(EncodeRequest::new(req), &context)
        .map(|r| serde_json::from_slice(&r.body).unwrap())
}

#[test]
fn qwen_json_object_keyword_comes_only_from_explicit_system_or_user_text() {
    let p = qwen_chat_profile();
    let mut req = request(OutputFormat::JsonObject);
    assert!(matches!(
        encode_model(&OpenAiChatCodec, &req, &p, "qwen3.8-flash"),
        Err(LlmError::InvalidRequest { .. })
    ));

    req.messages.clear();
    req.tools.push(ToolSpec {
        input_schema_json: None,
        tool_type: None,
        extra: serde_json::Value::Null,
        name: "json_tool".into(),
        description: "Returns JSON".into(),
        input_schema: json!({"type":"object"}),
        strict: false,
        defer_loading: false,
        native_options: Vec::new(),
    });
    assert!(matches!(
        encode_model(&OpenAiChatCodec, &req, &p, "qwen3.8-flash"),
        Err(LlmError::InvalidRequest { .. })
    ));
    req.tools.clear();

    // Qwen documents the JSON keyword as case-insensitive. It may be adjacent
    // to non-ASCII prompt text in languages that do not separate words with spaces.
    req.messages = vec![ConversationMessage::user_text("请用jSoN格式回答")];
    assert_eq!(
        encode_model(&OpenAiChatCodec, &req, &p, "qwen3.8-flash").unwrap()["response_format"]
            ["type"],
        "json_object"
    );

    req.messages = vec![ConversationMessage::assistant(vec![ContentBlock::Text {
        text: "JSON".into(),
        thought_signature: None,
        citations: None,
    }])];
    assert!(matches!(
        encode_model(&OpenAiChatCodec, &req, &p, "qwen3.8-flash"),
        Err(LlmError::InvalidRequest { .. })
    ));

    req.messages = vec![ConversationMessage {
        native_options: Vec::new(),
        role: MessageRole::User,
        content: vec![ContentBlock::Document {
            source: DocumentSource::Text {
                media_type: "text/plain".into(),
                data: "JSON".into(),
            },
            title: None,
        }],
    }];
    assert!(matches!(
        encode_model(&OpenAiChatCodec, &req, &p, "qwen3.8-flash"),
        Err(LlmError::InvalidRequest { .. })
    ));

    req.messages.clear();
    req.system.push(SystemBlock {
        text: "Respond with JSON please.".into(),
    });
    assert!(encode_model(&OpenAiChatCodec, &req, &p, "qwen3.8-flash").is_ok());

    req.system.clear();
    req.messages = vec![ConversationMessage {
        native_options: Vec::new(),
        role: MessageRole::System,
        content: vec![ContentBlock::Text {
            text: "Respond with JSON please.".into(),
            thought_signature: None,
            citations: None,
        }],
    }];
    assert!(encode_model(&OpenAiChatCodec, &req, &p, "qwen3.8-flash").is_ok());

    // The same wire shape on a generic gateway does not inherit Qwen policy.
    assert!(encode_model(
        &OpenAiChatCodec,
        &request(OutputFormat::JsonObject),
        &profile(ProtocolFamily::OpenAiChat),
        "test"
    )
    .is_ok());
}

#[test]
fn deepseek_json_object_requires_json_in_system_or_user_text_on_first_party_chat() {
    let deepseek = deepseek_chat_profile();
    let mut req = request(OutputFormat::JsonObject);
    assert!(matches!(
        encode_model(&OpenAiChatCodec, &req, &deepseek, "deepseek-flash"),
        Err(LlmError::InvalidRequest { .. })
    ));

    req.messages = vec![ConversationMessage::user_text("请用 jSoN 格式回答")];
    let body = encode_model(&OpenAiChatCodec, &req, &deepseek, "deepseek-flash").unwrap();
    assert_eq!(body["response_format"]["type"], "json_object");

    req.messages = vec![ConversationMessage::assistant(vec![ContentBlock::Text {
        text: "JSON".into(),
        thought_signature: None,
        citations: None,
    }])];
    assert!(matches!(
        encode_model(&OpenAiChatCodec, &req, &deepseek, "deepseek-flash"),
        Err(LlmError::InvalidRequest { .. })
    ));

    req.messages.clear();
    req.system.push(SystemBlock {
        text: "Respond with JSON please.".into(),
    });
    assert!(encode_model(&OpenAiChatCodec, &req, &deepseek, "deepseek-flash").is_ok());

    // Do not apply DeepSeek's first-party prompt rule to a compatible proxy.
    req.system.clear();
    let mut proxy = deepseek.clone();
    proxy.base_url = "https://proxy.example/v1".into();
    assert!(encode_model(&OpenAiChatCodec, &req, &proxy, "deepseek-flash").is_ok());
}

#[test]
fn qwen_strict_schema_subset_exception_is_output_only_and_model_scoped() {
    let qwen = qwen_chat_profile();
    let optional_schema = OutputFormat::JsonSchema {
        name: "person".into(),
        strict: true,
        schema: json!({
            "type":"object",
            "properties":{"name":{"type":"string"},"email":{"type":"string"}},
            "required":["name"],
            "additionalProperties":true
        }),
    };
    let mut req = request(optional_schema.clone());
    let body = encode_model(&OpenAiChatCodec, &req, &qwen, "qwen3.8-flash").unwrap();
    assert_eq!(
        body.pointer("/response_format/json_schema/schema/additionalProperties"),
        Some(&json!(true))
    );
    assert_eq!(
        body.pointer("/response_format/json_schema/schema/required"),
        Some(&json!(["name"]))
    );
    assert!(encode_model(&OpenAiChatCodec, &req, &qwen, "qwen3.8-max").is_ok());

    // OpenAI's strict subset remains unchanged.
    let mut openai = qwen.clone();
    openai.provider_id = "openai".into();
    openai.profile_name = "openai".into();
    openai.base_url = "https://api.openai.com/v1".into();
    assert!(matches!(
        encode_model(&OpenAiChatCodec, &req, &openai, "gpt-4o-2024-08-06"),
        Err(LlmError::InvalidRequest { .. })
    ));

    // Only the two currently catalogued Qwen models are in the verified set.
    assert!(matches!(
        encode_model(&OpenAiChatCodec, &req, &qwen, "qwen-long"),
        Err(LlmError::UnsupportedCapability { .. })
    ));
    assert!(matches!(
        encode_model(&OpenAiChatCodec, &req, &qwen, "qwen3.7-max"),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    // Qwen Responses keeps the OpenAI strict-subset rule.
    let mut qwen_responses = qwen.clone();
    qwen_responses.protocol = ProtocolFamily::OpenAiResponses;
    assert!(matches!(
        encode_model(
            &OpenAiResponsesCodec,
            &req,
            &qwen_responses,
            "qwen3.8-flash"
        ),
        Err(LlmError::InvalidRequest { .. })
    ));

    // A strict function tool is outside this output-only relaxation and keeps
    // the existing OpenAI Chat encoding behavior.
    req.output_format = OutputFormat::Text;
    req.tools.push(ToolSpec {
        input_schema_json: None,
        tool_type: None,
        extra: serde_json::Value::Null,
        name: "capture_person".into(),
        description: String::new(),
        input_schema: json!({
            "type":"object",
            "properties":{"name":{"type":"string"},"email":{"type":"string"}},
            "required":["name"],
            "additionalProperties":true
        }),
        strict: true,
        defer_loading: false,
        native_options: Vec::new(),
    });
    let tool_body = encode_model(&OpenAiChatCodec, &req, &qwen, "qwen3.8-flash").unwrap();
    assert_eq!(
        tool_body.pointer("/tools/0/function/parameters/additionalProperties"),
        Some(&json!(true))
    );
}
#[test]
fn schema_contract_is_encoded_on_each_wire_for_complete_and_stream() {
    for (codec, family, pointer) in [
        (
            &OpenAiChatCodec as &dyn WireCodec,
            ProtocolFamily::OpenAiChat,
            "/response_format/json_schema/schema",
        ),
        (
            &OpenAiResponsesCodec,
            ProtocolFamily::OpenAiResponses,
            "/text/format/schema",
        ),
        (
            &AnthropicMessagesCodec,
            ProtocolFamily::AnthropicMessages,
            "/output_config/format/schema",
        ),
        (
            &GeminiCodec,
            ProtocolFamily::GeminiGenerateContent,
            "/generationConfig/responseFormat/text/schema",
        ),
    ] {
        for stream in [false, true] {
            let body = encode(codec, &request(schema()), &profile(family), stream).unwrap();
            assert_eq!(
                body.pointer(pointer).unwrap()["properties"]["answer"]["type"],
                "string"
            );
        }
    }
}

#[test]
fn old_gpt_4o_snapshot_allows_json_mode_but_rejects_json_schema() {
    let p: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"openai",
        "profile_name":"openai",
        "base_url":"https://api.openai.com/v1",
        "protocol":"open_ai_chat",
        "auth":"none"
    }))
    .unwrap();
    let context = CodecContext::new(&p, "gpt-4o-2024-05-13", RequestMode::Complete);
    let schema_request = request(schema());
    assert!(matches!(
        OpenAiChatCodec.encode_request(EncodeRequest::new(&schema_request), &context),
        Err(LlmError::UnsupportedCapability { .. })
    ));
    let json_request = request(OutputFormat::JsonObject);
    assert!(OpenAiChatCodec
        .encode_request(EncodeRequest::new(&json_request), &context)
        .is_ok());
}
#[test]
fn nested_siblings_survive_and_conflicting_schema_is_rejected() {
    let mut p = profile(ProtocolFamily::OpenAiResponses);
    p.extra = json!({"body":{"text":{"verbosity":"low"}}});
    let body = encode(&OpenAiResponsesCodec, &request(schema()), &p, false).unwrap();
    assert_eq!(body["text"]["verbosity"], "low");
    p.extra["body"]["text"]["format"] = json!({"type":"json_object"});
    assert!(matches!(
        encode(&OpenAiResponsesCodec, &request(schema()), &p, false),
        Err(LlmError::InvalidRequest { .. })
    ));
}
#[test]
fn malformed_and_unsupported_schema_is_not_silently_weakened() {
    for schema in [
        json!({"type":"object","properties":{"x":{"type":"string"}},"additionalProperties":false}),
        json!({"type":"object","properties":{},"required":[],"additionalProperties":false,"allOf":[{"type":"object"}]}),
        json!({"$ref":"https://test.invalid/schema"}),
    ] {
        let r = request(OutputFormat::JsonSchema {
            name: "answer".into(),
            schema,
            strict: true,
        });
        assert!(encode(
            &OpenAiResponsesCodec,
            &r,
            &profile(ProtocolFamily::OpenAiResponses),
            false
        )
        .is_err());
    }
    let mut r = request(schema());
    if let OutputFormat::JsonSchema { schema, .. } = &mut r.output_format {
        schema["properties"]["answer"] = json!({"type":"number","minimum":10});
    }
    assert!(matches!(
        encode(
            &AnthropicMessagesCodec,
            &r,
            &profile(ProtocolFamily::AnthropicMessages),
            false
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}

#[test]
fn claude_preflight_rejects_recursive_refs_allof_refs_and_unlisted_formats() {
    let mut p = profile(ProtocolFamily::AnthropicMessages);
    p.provider_id = "anthropic".into();
    let unsupported_schemas = [
        json!({
            "type":"object",
            "properties":{"root":{"$ref":"#/$defs/node"}},
            "required":["root"],
            "additionalProperties":false,
            "$defs":{"node":{
                "type":"object",
                "properties":{"next":{"$ref":"#/$defs/node"}},
                "required":["next"],
                "additionalProperties":false
            }}
        }),
        json!({
            "type":"object",
            "properties":{},
            "required":[],
            "additionalProperties":false,
            "allOf":[{"$ref":"#/$defs/answer"}],
            "$defs":{"answer":{
                "type":"object",
                "properties":{"answer":{"type":"string"}},
                "required":["answer"],
                "additionalProperties":false
            }}
        }),
        json!({
            "type":"object",
            "properties":{"email":{"type":"string","format":"custom-format"}},
            "required":["email"],
            "additionalProperties":false
        }),
    ];

    for schema in unsupported_schemas {
        let req = request(OutputFormat::JsonSchema {
            name: "answer".into(),
            schema,
            strict: true,
        });
        assert!(matches!(
            encode(&AnthropicMessagesCodec, &req, &p, false),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }

    let shared_ref = request(OutputFormat::JsonSchema {
        name: "shared".into(),
        strict: true,
        schema: json!({
            "type":"object",
            "properties":{
                "first":{"$ref":"#/$defs/shared"},
                "second":{"$ref":"#/$defs/shared"}
            },
            "required":["first","second"],
            "additionalProperties":false,
            "$defs":{"shared":{
                "type":"object",
                "properties":{"value":{"type":"string"}},
                "required":["value"],
                "additionalProperties":false
            }}
        }),
    });
    assert!(encode(&AnthropicMessagesCodec, &shared_ref, &p, false).is_ok());

    // Anthropic's documented format list remains accepted.
    let req = request(OutputFormat::JsonSchema {
        name: "answer".into(),
        schema: json!({
            "type":"object",
            "properties":{"email":{"type":"string","format":"email"}},
            "required":["email"],
            "additionalProperties":false
        }),
        strict: true,
    });
    assert!(encode(&AnthropicMessagesCodec, &req, &p, false).is_ok());
}

#[test]
fn claude_json_output_preflights_only_native_citations_and_a_final_assistant_prefill() {
    let mut p = profile(ProtocolFamily::AnthropicMessages);
    p.provider_id = "anthropic".into();
    let mut req = request(schema());
    req.messages = vec![ConversationMessage::assistant(vec![ContentBlock::Text {
        text: "{".into(),
        thought_signature: None,
        citations: None,
    }])];
    assert!(matches!(
        encode(&AnthropicMessagesCodec, &req, &p, false),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    // Earlier assistant turns remain valid when the current input ends with a
    // user message; a non-strict tool also stays outside Claude's strict-tool
    // schema limits.
    req.messages
        .push(ConversationMessage::user_text("continue"));
    req.tools = (0..21)
        .map(|index| ToolSpec {
            input_schema_json: None,
            tool_type: None,
            extra: serde_json::Value::Null,
            name: format!("tool_{index}"),
            description: String::new(),
            input_schema: json!({"type":"object","properties":{},"additionalProperties":false}),
            strict: false,
            defer_loading: false,
            native_options: Vec::new(),
        })
        .collect();
    assert!(encode(&AnthropicMessagesCodec, &req, &p, false).is_ok());

    req.messages = vec![ConversationMessage {
        native_options: Vec::new(),
        role: MessageRole::User,
        content: vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: json!({
                "type":"document",
                "source":{"type":"text","media_type":"text/plain","data":"source"},
                "citations":{"enabled":true}
            }),
        }],
    }];
    assert!(matches!(
        encode(&AnthropicMessagesCodec, &req, &p, false),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    req.messages = vec![ConversationMessage {
        native_options: Vec::new(),
        role: MessageRole::User,
        content: vec![ContentBlock::ProviderContent {
            protocol: ProtocolFamily::AnthropicMessages,
            value: json!({
                "type":"search_result",
                "source":"https://example.invalid/page",
                "title":"Source",
                "content":[{"type":"text","text":"source"}],
                "citations":{"enabled":true}
            }),
        }],
    }];
    assert!(matches!(
        encode(&AnthropicMessagesCodec, &req, &p, false),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    req.messages = vec![ConversationMessage {
        native_options: Vec::new(),
        role: MessageRole::User,
        content: vec![ContentBlock::Document {
            source: DocumentSource::Text {
                media_type: "text/plain".into(),
                data: "source without citations enabled".into(),
            },
            title: None,
        }],
    }];
    assert!(encode(&AnthropicMessagesCodec, &req, &p, false).is_ok());

    req.messages = vec![ConversationMessage {
        native_options: Vec::new(),
        role: MessageRole::User,
        content: vec![ContentBlock::Text {
            text: r#"{"type":"document","citations":{"enabled":true}}"#.into(),
            thought_signature: None,
            citations: None,
        }],
    }];
    assert!(encode(&AnthropicMessagesCodec, &req, &p, false).is_ok());

    // A generic Messages-compatible gateway does not inherit Claude-specific
    // combination limits from its wire shape alone.
    let mut gateway = p.clone();
    gateway.provider_id = "acme".into();
    req.messages = vec![ConversationMessage::assistant(vec![ContentBlock::Text {
        text: "{".into(),
        thought_signature: None,
        citations: None,
    }])];
    assert!(encode(&AnthropicMessagesCodec, &req, &gateway, false).is_ok());
}

#[test]
fn claude_strict_tool_schemas_are_checked_for_text_output_and_non_strict_tools_are_skipped() {
    let mut p = profile(ProtocolFamily::AnthropicMessages);
    p.provider_id = "anthropic".into();
    let mut req = request(OutputFormat::Text);
    req.tools = vec![ToolSpec {
        input_schema_json: None,
        tool_type: None,
        extra: serde_json::Value::Null,
        name: "strict_tool".into(),
        description: String::new(),
        input_schema: json!({
            "type":"object",
            "properties":{"count":{"type":"number","minimum":1}},
            "required":["count"],
            "additionalProperties":false
        }),
        strict: true,
        defer_loading: false,
        native_options: Vec::new(),
    }];
    assert!(matches!(
        encode(&AnthropicMessagesCodec, &req, &p, false),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    req.tools[0].strict = false;
    assert!(encode(&AnthropicMessagesCodec, &req, &p, false).is_ok());

    req.tools[0].strict = true;
    req.tools[0].input_schema = json!({
        "type":"object",
        "properties":{"next":{"$ref":"#/$defs/node"}},
        "required":["next"],
        "additionalProperties":false,
        "$defs":{"node":{
            "type":"object",
            "properties":{"next":{"$ref":"#/$defs/node"}},
            "required":["next"],
            "additionalProperties":false
        }}
    });
    assert!(matches!(
        encode(&AnthropicMessagesCodec, &req, &p, false),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    // The preflight keeps its documented schema restrictions scoped to
    // first-party and explicitly identified Claude profiles.
    p.provider_id = "acme".into();
    assert!(encode(&AnthropicMessagesCodec, &req, &p, false).is_ok());
}

fn response(text: &str, stop: StopReason) -> ChatResponse {
    serde_json::from_value(json!({"message":{"role":"assistant","content":[{"type":"text","text":text}]},"stop_reason":stop,"usage":{"state":"missing"},"model":"test"})).unwrap()
}
#[test]
fn failed_parse_retains_response_and_terminal_status_is_required() {
    let r = response("{\"answer\":3}", StopReason::EndTurn);
    let err = r.structured_json(&schema()).unwrap_err();
    assert!(matches!(
        err.kind,
        StructuredOutputErrorKind::SchemaMismatch(_)
    ));
    assert_eq!(*err.response, r);
    for stop in [
        StopReason::MaxTokens,
        StopReason::Refusal,
        StopReason::ToolUse,
        StopReason::Other("pause_turn".into()),
    ] {
        assert!(matches!(
            response("{\"answer\":\"ok\"}", stop)
                .structured_json(&schema())
                .unwrap_err()
                .kind,
            StructuredOutputErrorKind::Incomplete
        ));
    }
    assert_eq!(
        response("{\"answer\":\"ok\"}", StopReason::EndTurn)
            .structured_json(&schema())
            .unwrap()["answer"],
        "ok"
    );
}
#[test]
fn json_mode_is_not_schema_mode() {
    let r = request(OutputFormat::JsonObject);
    assert_eq!(
        encode(
            &OpenAiChatCodec,
            &r,
            &profile(ProtocolFamily::OpenAiChat),
            false
        )
        .unwrap()["response_format"],
        json!({"type":"json_object"})
    );
    assert!(encode(
        &AnthropicMessagesCodec,
        &r,
        &profile(ProtocolFamily::AnthropicMessages),
        false
    )
    .is_err());
    assert!(response("[]", StopReason::EndTurn)
        .structured_json(&OutputFormat::JsonObject)
        .is_err());
}

#[test]
fn schema_properties_named_like_keywords_are_not_misclassified() {
    let output = OutputFormat::JsonSchema {
        name: "answer".into(),
        strict: true,
        schema: json!({"type":"object","properties":{"not":{"type":"string"},"minimum":{"type":"number"}},"required":["not","minimum"],"additionalProperties":false}),
    };
    assert!(encode(
        &OpenAiResponsesCodec,
        &request(output),
        &profile(ProtocolFamily::OpenAiResponses),
        false
    )
    .is_ok());
}
#[test]
fn profile_cannot_inject_or_extend_an_output_contract() {
    let mut p = profile(ProtocolFamily::OpenAiResponses);
    p.extra = json!({"body":{"text":{"format":{"type":"json_object"}}}});
    assert!(encode(
        &OpenAiResponsesCodec,
        &request(OutputFormat::Text),
        &p,
        false
    )
    .is_err());
    let OutputFormat::JsonSchema {
        name,
        schema,
        strict,
    } = schema()
    else {
        unreachable!()
    };
    p.extra = json!({"body":{"text":{"format":{"type":"json_schema","name":name,"schema":schema,"strict":strict}}}});
    p.extra["body"]["text"]["format"]["schema"]["properties"]["extra"] = json!({"type":"string"});
    assert!(encode(
        &OpenAiResponsesCodec,
        &request(OutputFormat::JsonSchema {
            name,
            schema,
            strict
        }),
        &p,
        false
    )
    .is_err());
}

#[test]
fn external_schema_references_are_never_retrieved_even_without_strict_mode() {
    for reference in ["https://schema.invalid/object", "file:///tmp/schema.json"] {
        let format = OutputFormat::JsonSchema {
            name: "answer".into(),
            strict: false,
            schema: json!({"$ref":reference}),
        };
        assert!(encode(
            &OpenAiResponsesCodec,
            &request(format.clone()),
            &profile(ProtocolFamily::OpenAiResponses),
            false
        )
        .is_err());
        assert!(matches!(
            response("{}", StopReason::EndTurn)
                .structured_json(&format)
                .unwrap_err()
                .kind,
            StructuredOutputErrorKind::InvalidSchema(_)
        ));
    }
}

#[derive(Default)]
struct QwenPreflightCounts {
    resolver: AtomicUsize,
    authenticator: AtomicUsize,
    transport: AtomicUsize,
}

struct QwenPreflightResolver(Arc<QwenPreflightCounts>);

#[async_trait]
impl AttachmentResolver for QwenPreflightResolver {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        self.0.resolver.fetch_add(1, Ordering::SeqCst);
        Ok(Bytes::from_static(b"data"))
    }
}

struct QwenPreflightAuthenticator(Arc<QwenPreflightCounts>);

#[async_trait]
impl Authenticator for QwenPreflightAuthenticator {
    async fn apply(
        &self,
        _: &mut HttpRequest,
        _: &ProviderProfile,
        _: Option<&Secret<String>>,
    ) -> Result<(), LlmError> {
        self.0.authenticator.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct QwenPreflightTransport(Arc<QwenPreflightCounts>);

#[async_trait]
impl Transport for QwenPreflightTransport {
    async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.0.transport.fetch_add(1, Ordering::SeqCst);
        Err(LlmError::Transport {
            message: "unexpected dispatch in Qwen structured preflight test".into(),
        })
    }
}

#[tokio::test]
async fn qwen_json_object_preflight_precedes_attachment_resolution_auth_and_transport() {
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"qwen",
        "profile_name":"qwen",
        "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1",
        "protocol":"open_ai_chat",
        "auth":"api_key",
        "regions":["china_mainland"],
        "models":[{
            "display_model":"Qwen3.8 Flash",
            "request_model":"qwen3.8-flash",
            "billing_model":"qwen3.8-flash",
            "metadata":{"inputModalities":["text","file"]},
            "capability_support":{"documents":"supported"}
        }]
    }))
    .unwrap();
    let counts = Arc::new(QwenPreflightCounts::default());
    let transport = Arc::new(QwenPreflightTransport(counts.clone()));
    let mut builder = LlmClientBuilder::with_transport(transport, std::slice::from_ref(&profile));
    builder.with_attachment_resolver(Arc::new(QwenPreflightResolver(counts.clone())));
    builder.register_authenticator(
        AuthStrategy::ApiKey,
        Arc::new(QwenPreflightAuthenticator(counts.clone())),
    );
    let client = builder.with_region(Region::ChinaMainland).build().unwrap();
    let mut req: ChatRequest = serde_json::from_value(json!({
        "model":"qwen3.8-flash",
        "messages":[{
            "role":"user",
            "content":[{
                "type":"document",
                "source":{"type":"attachment","attachment":{
                    "attachment_id":"doc",
                    "revision":"1",
                    "filename":"doc.txt",
                    "media_type":"text/plain",
                    "size_bytes":4
                }}
            }]
        }]
    }))
    .unwrap();
    req.output_format = OutputFormat::JsonObject;

    assert!(matches!(
        client
            .chat()
            .complete(
                &req,
                &RequestOptions {
                    credential: Some(Secret::new("test-key".into())),
                    ..Default::default()
                }
            )
            .await,
        Err(LlmError::InvalidRequest { message }) if message.contains("Qwen JSON Object")
    ));
    assert_eq!(counts.resolver.load(Ordering::SeqCst), 0);
    assert_eq!(counts.authenticator.load(Ordering::SeqCst), 0);
    assert_eq!(counts.transport.load(Ordering::SeqCst), 0);
}

#[test]
fn cached_schema_still_checks_protocol_strictness_capabilities_and_native_fields() {
    let format = OutputFormat::JsonSchema {
        name: "answer".into(),
        strict: false,
        schema: json!({
            "type":"object",
            "properties":{"answer":{"type":"number","minimum":10}},
            "required":["answer"],
            "additionalProperties":false
        }),
    };
    let mut req = request(format);
    let mut p = profile(ProtocolFamily::OpenAiResponses);
    encode(&OpenAiResponsesCodec, &req, &p, false).unwrap();
    assert!(matches!(
        encode(
            &AnthropicMessagesCodec,
            &req,
            &profile(ProtocolFamily::AnthropicMessages),
            false
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));
    if let OutputFormat::JsonSchema { strict, .. } = &mut req.output_format {
        *strict = true;
    }
    encode(&OpenAiResponsesCodec, &req, &p, false).unwrap();
    assert!(matches!(
        encode(
            &AnthropicMessagesCodec,
            &req,
            &profile(ProtocolFamily::AnthropicMessages),
            false
        ),
        Err(LlmError::UnsupportedCapability { .. })
    ));
    p.models.push(
        serde_json::from_value(json!({
            "display_model":"test","request_model":"test","billing_model":"test",
            "capability_support":{"structured_output":"unsupported"}
        }))
        .unwrap(),
    );
    assert!(matches!(
        encode(&OpenAiResponsesCodec, &req, &p, false),
        Err(LlmError::UnsupportedCapability { .. })
    ));
    p.models.clear();
    p.extra = json!({"body":{"text":{"format":{"type":"json_object"}}}});
    assert!(matches!(
        encode(&OpenAiResponsesCodec, &req, &p, false),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn mutating_a_warmed_schema_changes_local_validation_and_wire_output() {
    let mut format = schema();
    let answer = response("{\"answer\":\"ok\"}", StopReason::EndTurn);
    answer.structured_json(&format).unwrap();
    if let OutputFormat::JsonSchema { schema, .. } = &mut format {
        schema["properties"]["answer"]["type"] = json!("integer");
    }
    assert!(matches!(
        answer.structured_json(&format).unwrap_err().kind,
        StructuredOutputErrorKind::SchemaMismatch(_)
    ));
    let body = encode(
        &OpenAiResponsesCodec,
        &request(format),
        &profile(ProtocolFamily::OpenAiResponses),
        false,
    )
    .unwrap();
    assert_eq!(
        body["text"]["format"]["schema"]["properties"]["answer"]["type"],
        "integer"
    );
}

#[test]
fn cached_compilation_does_not_cache_schema_names_or_strict_subset_validation() {
    let mut req = request(OutputFormat::JsonSchema {
        name: "answer".into(),
        strict: false,
        schema: json!({"type":"object","properties":{"answer":{"type":"string"}}}),
    });
    let p = profile(ProtocolFamily::OpenAiResponses);
    encode(&OpenAiResponsesCodec, &req, &p, false).unwrap();
    if let OutputFormat::JsonSchema { strict, .. } = &mut req.output_format {
        *strict = true;
    }
    assert!(matches!(
        encode(&OpenAiResponsesCodec, &req, &p, false),
        Err(LlmError::InvalidRequest { .. })
    ));
    if let OutputFormat::JsonSchema { strict, name, .. } = &mut req.output_format {
        *strict = false;
        *name = "invalid name".into();
    }
    assert!(matches!(
        encode(&OpenAiResponsesCodec, &req, &p, false),
        Err(LlmError::InvalidRequest { .. })
    ));
}

#[test]
fn large_local_response_contracts_remain_supported_but_cannot_be_sent() {
    let format = OutputFormat::JsonSchema {
        name: "answer".into(),
        strict: false,
        schema: json!({"type":"object","description":"x".repeat(1024 * 1024)}),
    };
    assert_eq!(
        response("{}", StopReason::EndTurn)
            .structured_json(&format)
            .unwrap(),
        json!({})
    );
    let error = encode(
        &OpenAiResponsesCodec,
        &request(format),
        &profile(ProtocolFamily::OpenAiResponses),
        false,
    )
    .unwrap_err();
    assert!(matches!(error, LlmError::InvalidRequest { .. }));
    assert!(error.to_string().contains("output schema exceeds 1 MiB"));
}

#[test]
fn review_regression_signed_zero_cannot_bypass_a_warm_schema_size_limit() {
    let positive = json!({"type":"object","examples":vec![0.0_f64; 240_000]});
    let negative = json!({"type":"object","examples":vec![-0.0_f64; 240_000]});
    assert_eq!(positive, negative);
    assert!(serde_json::to_vec(&positive).unwrap().len() < 1024 * 1024);
    assert!(serde_json::to_vec(&negative).unwrap().len() > 1024 * 1024);
    let format = |schema| OutputFormat::JsonSchema {
        name: "signed_zero".into(),
        strict: false,
        schema,
    };
    let p = profile(ProtocolFamily::OpenAiChat);
    encode(&OpenAiChatCodec, &request(format(positive)), &p, false).unwrap();
    let negative = format(negative);
    for stream in [false, true] {
        let error = encode(&OpenAiChatCodec, &request(negative.clone()), &p, stream)
            .expect_err("a warmed validator must not permit a larger schema representation");
        assert!(matches!(error, LlmError::InvalidRequest { .. }));
        assert!(error.to_string().contains("output schema exceeds 1 MiB"));
    }
    // The request cap still does not restrict local response validation.
    assert_eq!(
        response("{}", StopReason::EndTurn)
            .structured_json(&negative)
            .unwrap(),
        json!({})
    );
}
