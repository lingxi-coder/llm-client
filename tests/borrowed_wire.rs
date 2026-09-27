//! Borrowed request payloads preserve the provider wire and later metadata patches.
use lingxi_llm_client::{protocol::*, *};
use serde_json::{json, Value};

const TEXT: &str = "quoted \"text\"\n中文";

fn request() -> ChatRequest {
    serde_json::from_value(json!({
        "model":"test", "system":[{"text": TEXT}],
        "messages":[
            {"role":"user","content":[
                {"type":"text","text":TEXT},
                {"type":"image","source":{"type":"url","url":"https://test.invalid/image.png"}}
            ]},
            {"role":"assistant","content":[
                {"type":"text","text":"calling"},
                {"type":"tool_use","id":"call-1","name":"lookup","input":{"query":TEXT}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"call-1","content":TEXT,"is_error":false}
            ]}
        ],
        "tools":[{"name":"lookup","description":TEXT,"strict":false,
            "input_schema":{"type":"object","properties":{"query":{"type":"string","enum":[TEXT]}}}}]
    })).unwrap()
}

fn encode(codec: &dyn WireCodec, req: &ChatRequest, extra: Value) -> HttpRequest {
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"test", "profile_name":"test", "base_url":"https://test.invalid",
        "protocol":codec.family(), "auth":"none", "extra":extra,
    }))
    .unwrap();
    let context = CodecContext::new(&profile, "test", RequestMode::Complete);
    let input = EncodeRequest::new(req);
    let encoded = codec.encode_request(input, &context).unwrap();
    assert_eq!(
        codec.encoded_body_len(input, &context).unwrap(),
        encoded.body.len()
    );
    encoded
}

#[test]
fn text_tool_schema_results_media_and_additive_extras_keep_exact_wire_bytes() {
    let req = request();
    let schema = &req.tools[0].input_schema;
    let extra = json!({"body":{"audit_tag":"keep","tools":[{"wrong":true}],"model":"wrong"}});
    let cases: Vec<(&dyn WireCodec, Value)> = vec![
        (
            &OpenAiChatCodec,
            json!({
                "model":"test", "audit_tag":"keep", "tool_choice":"auto",
                "tools":[{"type":"function","function":{"name":"lookup","description":TEXT,"parameters":schema,"strict":false}}],
                "messages":[
                    {"role":"system","content":TEXT},
                    {"role":"user","content":[{"type":"text","text":TEXT},{"type":"image_url","image_url":{"url":"https://test.invalid/image.png"}}]},
                    {"role":"assistant","content":"calling","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":json!({"query":TEXT}).to_string()}}]},
                    {"role":"tool","tool_call_id":"call-1","content":TEXT}
                ],
            }),
        ),
        (
            &OpenAiResponsesCodec,
            json!({
                "model":"test", "audit_tag":"keep", "instructions":TEXT,"tool_choice":"auto",
                "tools":[{"type":"function","name":"lookup","description":TEXT,"parameters":schema,"strict":false}],
                "input":[
                    {"type":"message","role":"user","content":[{"type":"input_text","text":TEXT},{"type":"input_image","image_url":"https://test.invalid/image.png"}]},
                    {"type":"message","role":"assistant","content":[{"type":"output_text","text":"calling"}]},
                    {"type":"function_call","call_id":"call-1","name":"lookup","arguments":json!({"query":TEXT}).to_string()},
                    {"type":"function_call_output","call_id":"call-1","output":TEXT}
                ],
            }),
        ),
        (
            &AnthropicMessagesCodec,
            json!({
                "model":"test","max_tokens":4096,"audit_tag":"keep", "system":[{"type":"text","text":TEXT}],"tool_choice":{"type":"auto"},
                "tools":[{"name":"lookup","description":TEXT,"input_schema":schema}],
                "messages":[
                    {"role":"user","content":[{"type":"text","text":TEXT},{"type":"image","source":{"type":"url","url":"https://test.invalid/image.png"}}]},
                    {"role":"assistant","content":[{"type":"text","text":"calling"},{"type":"tool_use","id":"call-1","name":"lookup","input":{"query":TEXT}}]},
                    {"role":"user","content":[{"type":"tool_result","tool_use_id":"call-1","content":TEXT}]}
                ],
            }),
        ),
        (
            &GeminiCodec,
            json!({
                "audit_tag":"keep", "systemInstruction":{"parts":[{"text":TEXT}]},
                "toolConfig":{"functionCallingConfig":{"mode":"AUTO"}},
                "tools":[{"functionDeclarations":[{"name":"lookup","description":TEXT,"parameters":schema}]}],
                "contents":[
                    {"role":"user","parts":[{"text":TEXT},{"fileData":{"fileUri":"https://test.invalid/image.png"}}]},
                    {"role":"model","parts":[{"text":"calling"},{"functionCall":{"name":"lookup","args":{"query":TEXT}}}]},
                    {"role":"user","parts":[{"functionResponse":{"name":"lookup","response":{"result":TEXT}}}]}
                ],
            }),
        ),
    ];
    for (codec, mut expected) in cases {
        // Pin the existing top-level insertion order as well as JSON values.
        let keys: &[&str] = match codec.family() {
            ProtocolFamily::OpenAiChat => {
                &["model", "messages", "tools", "tool_choice", "audit_tag"]
            }
            ProtocolFamily::OpenAiResponses => &[
                "model",
                "instructions",
                "input",
                "tools",
                "tool_choice",
                "audit_tag",
            ],
            ProtocolFamily::AnthropicMessages => &[
                "model",
                "max_tokens",
                "system",
                "messages",
                "tools",
                "tool_choice",
                "audit_tag",
            ],
            ProtocolFamily::GeminiGenerateContent => &[
                "contents",
                "systemInstruction",
                "tools",
                "toolConfig",
                "audit_tag",
            ],
            _ => unreachable!(),
        };
        let expected = Value::Object(
            keys.iter()
                .map(|key| {
                    (
                        (*key).to_owned(),
                        expected.as_object_mut().unwrap().remove(*key).unwrap(),
                    )
                })
                .collect(),
        );
        let encoded = encode(codec, &req, extra.clone());
        assert_eq!(
            encoded.body.as_ref(),
            serde_json::to_vec(&expected).unwrap(),
            "{:?}",
            codec.family()
        );
    }
}

#[test]
fn cache_patches_preserve_borrowed_schema_system_and_structured_tool_results() {
    let mut req = request();
    let result = json!({"type":"text","text":TEXT});
    if let ContentBlock::ToolResult {
        blocks, is_error, ..
    } = &mut req.messages[2].content[0]
    {
        *blocks = Some(vec![result.clone()]);
        *is_error = true;
    }
    req.prompt_cache.breakpoints = vec![
        CacheBreakpoint {
            position: CachePosition::Tool { index: 0 },
            ttl: CacheTtl::FiveMinutes,
        },
        CacheBreakpoint {
            position: CachePosition::System { index: 0 },
            ttl: CacheTtl::FiveMinutes,
        },
        CacheBreakpoint {
            position: CachePosition::Message { index: 2, block: 0 },
            ttl: CacheTtl::FiveMinutes,
        },
    ];
    let encoded = encode(&AnthropicMessagesCodec, &req, Value::Null);
    let body: Value = serde_json::from_slice(&encoded.body).unwrap();
    assert_eq!(body["tools"][0]["input_schema"], req.tools[0].input_schema);
    assert_eq!(body["system"][0]["text"], TEXT);
    for value in [
        &body["tools"][0],
        &body["system"][0],
        &body["messages"][2]["content"][0],
    ] {
        assert_eq!(value["cache_control"], json!({"type":"ephemeral"}));
    }
    assert_eq!(
        body["messages"][2]["content"][0]["content"],
        json!([result])
    );
    assert_eq!(body["messages"][2]["content"][0]["is_error"], true);
    req.prompt_cache = Default::default();
    let encoded = encode(&OpenAiResponsesCodec, &req, Value::Null);
    let body: Value = serde_json::from_slice(&encoded.body).unwrap();
    assert_eq!(body["input"][3]["output"], json!([result]));
}

#[test]
fn native_tools_from_extras_are_unchanged_without_client_tools() {
    let mut req = request();
    req.tools.clear();
    let native = json!([{"type":"native","parameters":{"keep":[1,2,3]}}]);
    for codec in [
        &OpenAiChatCodec as &dyn WireCodec,
        &OpenAiResponsesCodec,
        &AnthropicMessagesCodec,
        &GeminiCodec,
    ] {
        let encoded = encode(codec, &req, json!({"body":{"tools":native}}));
        let body: Value = serde_json::from_slice(&encoded.body).unwrap();
        assert_eq!(body["tools"], native);
    }
}
