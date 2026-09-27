use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    openrouter_batch::*,
    protocol::{LlmError, Secret},
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

struct Reply {
    status: u16,
    body: Value,
}

struct MockTransport {
    replies: Mutex<VecDeque<Result<Reply, LlmError>>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Result<Reply, LlmError>>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<HttpRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn response(reply: Reply) -> StreamResponse {
        StreamResponse {
            status: reply.status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: futures::stream::once(async move {
                Ok(Bytes::from(serde_json::to_vec(&reply.body).unwrap()))
            })
            .boxed(),
        }
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let reply = self.replies.lock().unwrap().pop_front().unwrap()?;
        Ok(Self::response(reply))
    }

    async fn send_stream(&self, _request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        panic!("OpenRouter Batch must not upload a file or stream a request body")
    }
}

fn reply(status: u16, body: Value) -> Result<Reply, LlmError> {
    Ok(Reply { status, body })
}

fn scope(account: &str) -> OpenRouterBatchScope {
    OpenRouterBatchScope::new("router-prod", account).unwrap()
}

fn service<'a>(mock: &'a MockTransport, account: &str) -> OpenRouterBatchService<'a> {
    OpenRouterBatchService::new(mock, Secret::new("sk-or-test".into()), scope(account)).unwrap()
}

fn chat_line(custom_id: &str, content: &str) -> OpenRouterBatchLine {
    OpenRouterBatchLine {
        custom_id: custom_id.into(),
        body: OpenRouterBatchRequestBody::ChatCompletions(OpenRouterBatchChatBody {
            messages: vec![OpenRouterBatchChatMessage {
                role: OpenRouterBatchChatRole::User,
                content: content.into(),
            }],
            temperature: None,
            top_p: None,
            max_tokens: Some(64),
            max_completion_tokens: None,
        }),
    }
}

fn batch(id: &str, status: &str, results: Value) -> Value {
    json!({
        "id": id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "model": "openai/gpt-6-sol",
        "completion_window": "24h",
        "status": status,
        "created_at": 1780000000,
        "finalized_at": null,
        "request_counts": {"total": 2, "completed": 1, "failed": 1},
        "usage": {"prompt_tokens": 18, "completion_tokens": 7, "total_tokens": 25},
        "results": results,
        "error": null
    })
}

fn completed_results() -> Value {
    json!([
        {
            "id": "batch_req_1",
            "custom_id": "row-1",
            "response": {
                "status_code": 200,
                "request_id": "request-1",
                "body": {"choices": [{"message": {"content": "done"}}], "vendor_field": true}
            },
            "error": null
        },
        {
            "id": "batch_req_2",
            "custom_id": "row-2",
            "response": null,
            "error": {"code": "provider_error", "message": "one row failed"}
        }
    ])
}

#[test]
fn input_requires_unique_ids_and_matching_typed_endpoint_bodies() {
    let valid = OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::ChatCompletions,
        "openai/gpt-6-sol",
        vec![chat_line("row-1", "hello")],
    )
    .unwrap();
    assert_eq!(valid.endpoint(), OpenRouterBatchEndpoint::ChatCompletions);
    assert_eq!(valid.model(), "openai/gpt-6-sol");
    assert_eq!(valid.len(), 1);

    let duplicate = OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::ChatCompletions,
        "openai/gpt-6-sol",
        vec![chat_line("row-1", "a"), chat_line("row-1", "b")],
    );
    assert!(matches!(
        duplicate,
        Err(OpenRouterBatchError::InvalidInput(_))
    ));

    let mismatch = OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::Responses,
        "openai/gpt-6-sol",
        vec![chat_line("row-1", "hello")],
    );
    assert!(matches!(
        mismatch,
        Err(OpenRouterBatchError::InvalidInput(_))
    ));

    let invalid_provider = OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::ChatCompletions,
        "openai/gpt-6-sol",
        vec![chat_line("row-1", "hello")],
    )
    .unwrap()
    .with_provider_only(vec!["google-vertex".into(), "google-vertex".into()]);
    assert!(matches!(
        invalid_provider,
        Err(OpenRouterBatchError::InvalidInput(_))
    ));
}

#[test]
fn all_documented_text_endpoint_bodies_serialize_without_variant_tags() {
    let responses = OpenRouterBatchRequestBody::Responses(OpenRouterBatchResponsesBody {
        input: "summarize this".into(),
        instructions: Some("Be concise".into()),
        max_output_tokens: Some(64),
    });
    let messages = OpenRouterBatchRequestBody::Messages(OpenRouterBatchMessagesBody {
        messages: vec![OpenRouterBatchMessagesMessage {
            role: OpenRouterBatchMessagesRole::User,
            content: "summarize this".into(),
        }],
        max_tokens: 64,
        system: Some("Be concise".into()),
        temperature: None,
        top_p: None,
    });
    let embeddings = OpenRouterBatchRequestBody::Embeddings(OpenRouterBatchEmbeddingsBody {
        input: OpenRouterBatchEmbeddingInput::Texts(vec!["one".into(), "two".into()]),
        dimensions: Some(384),
        encoding_format: Some(OpenRouterBatchEncodingFormat::Float),
    });

    let responses_json = serde_json::to_value(responses).unwrap();
    assert_eq!(responses_json["input"], "summarize this");
    assert!(responses_json.get("Responses").is_none());
    let messages_json = serde_json::to_value(messages).unwrap();
    assert_eq!(messages_json["messages"][0]["role"], "user");
    assert_eq!(messages_json["max_tokens"], 64);
    let embeddings_json = serde_json::to_value(embeddings).unwrap();
    assert_eq!(embeddings_json["input"][0], "one");
    assert_eq!(embeddings_json["dimensions"], 384);
}

#[test]
fn url_multimodal_bodies_use_each_endpoint_wire_shape() {
    let chat = OpenRouterBatchChatBody {
        messages: vec![OpenRouterBatchChatMessage {
            role: OpenRouterBatchChatRole::User,
            content: OpenRouterBatchChatContent::Parts(vec![
                OpenRouterBatchChatContentPart::text("Describe these inputs."),
                OpenRouterBatchChatContentPart::image_url("https://example.com/image.png"),
                OpenRouterBatchChatContentPart::file_url(
                    "report.pdf",
                    "https://example.com/report.pdf",
                ),
            ]),
        }],
        temperature: None,
        top_p: None,
        max_tokens: Some(64),
        max_completion_tokens: None,
    };
    let chat_json = serde_json::to_value(&chat).unwrap();
    assert_eq!(chat_json["messages"][0]["content"][0]["type"], "text");
    assert_eq!(
        chat_json["messages"][0]["content"][1]["image_url"]["url"],
        "https://example.com/image.png"
    );
    assert_eq!(
        chat_json["messages"][0]["content"][2]["file"]["file_data"],
        "https://example.com/report.pdf"
    );
    assert!(chat_json["messages"][0]["content"][2]["file"]
        .get("file_url")
        .is_none());
    OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::ChatCompletions,
        "openai/gpt-6-sol",
        vec![OpenRouterBatchLine {
            custom_id: "chat-media".into(),
            body: OpenRouterBatchRequestBody::ChatCompletions(chat),
        }],
    )
    .unwrap()
    .with_provider_only(vec!["deepinfra".into()])
    .unwrap();

    let responses = OpenRouterBatchResponsesBody {
        input: OpenRouterBatchResponsesInput::Messages(vec![OpenRouterBatchResponsesMessage {
            role: OpenRouterBatchResponsesRole::User,
            content: OpenRouterBatchResponsesContent::Parts(vec![
                OpenRouterBatchResponsesContentPart::text("Describe these inputs."),
                OpenRouterBatchResponsesContentPart::image_url("https://example.com/image.png"),
                OpenRouterBatchResponsesContentPart::file_url("https://example.com/report.pdf"),
            ]),
        }]),
        instructions: None,
        max_output_tokens: Some(64),
    };
    let responses_json = serde_json::to_value(&responses).unwrap();
    assert_eq!(
        responses_json["input"][0]["content"][0]["type"],
        "input_text"
    );
    assert_eq!(
        responses_json["input"][0]["content"][1]["type"],
        "input_image"
    );
    assert_eq!(
        responses_json["input"][0]["content"][1]["image_url"],
        "https://example.com/image.png"
    );
    assert_eq!(
        responses_json["input"][0]["content"][2]["type"],
        "input_file"
    );
    assert_eq!(
        responses_json["input"][0]["content"][2]["file_url"],
        "https://example.com/report.pdf"
    );
    OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::Responses,
        "openai/gpt-6-sol",
        vec![OpenRouterBatchLine {
            custom_id: "responses-media".into(),
            body: OpenRouterBatchRequestBody::Responses(responses),
        }],
    )
    .unwrap()
    .with_provider_only(vec!["openai".into()])
    .unwrap();

    let messages = OpenRouterBatchMessagesBody {
        messages: vec![OpenRouterBatchMessagesMessage {
            role: OpenRouterBatchMessagesRole::User,
            content: OpenRouterBatchMessagesContent::Parts(vec![
                OpenRouterBatchMessagesContentPart::text("Describe these inputs."),
                OpenRouterBatchMessagesContentPart::image_url("https://example.com/image.png"),
                OpenRouterBatchMessagesContentPart::document_url("https://example.com/report.pdf"),
            ]),
        }],
        max_tokens: 64,
        system: None,
        temperature: None,
        top_p: None,
    };
    let messages_json = serde_json::to_value(&messages).unwrap();
    assert_eq!(messages_json["messages"][0]["content"][1]["type"], "image");
    assert_eq!(
        messages_json["messages"][0]["content"][1]["source"]["type"],
        "url"
    );
    assert_eq!(
        messages_json["messages"][0]["content"][2]["type"],
        "document"
    );
    assert_eq!(
        messages_json["messages"][0]["content"][2]["source"]["url"],
        "https://example.com/report.pdf"
    );
    OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::Messages,
        "anthropic/claude-sonnet-4",
        vec![OpenRouterBatchLine {
            custom_id: "messages-media".into(),
            body: OpenRouterBatchRequestBody::Messages(messages),
        }],
    )
    .unwrap()
    .with_provider_only(vec!["anthropic".into()])
    .unwrap();
}

#[test]
fn multimodal_inputs_reject_non_urls_wrong_roles_and_incompatible_pins() {
    let chat_line_with = |content| OpenRouterBatchLine {
        custom_id: "media".into(),
        body: OpenRouterBatchRequestBody::ChatCompletions(OpenRouterBatchChatBody {
            messages: vec![OpenRouterBatchChatMessage {
                role: OpenRouterBatchChatRole::User,
                content,
            }],
            temperature: None,
            top_p: None,
            max_tokens: Some(64),
            max_completion_tokens: None,
        }),
    };
    for content in [
        OpenRouterBatchChatContent::Parts(vec![OpenRouterBatchChatContentPart::image_url(
            "data:image/png;base64,AAAA",
        )]),
        OpenRouterBatchChatContent::Parts(vec![OpenRouterBatchChatContentPart::image_url(
            "file_123",
        )]),
        OpenRouterBatchChatContent::Parts(vec![OpenRouterBatchChatContentPart::image_url(
            "https://127.0.0.1/image.png",
        )]),
        OpenRouterBatchChatContent::Parts(vec![OpenRouterBatchChatContentPart::image_url(
            "https://[::1]/image.png",
        )]),
        OpenRouterBatchChatContent::Parts(vec![OpenRouterBatchChatContentPart::image_url(
            "https://[fc00::1]/image.png",
        )]),
        OpenRouterBatchChatContent::Parts(vec![OpenRouterBatchChatContentPart::image_url(
            "https://[fe80::1]/image.png",
        )]),
        OpenRouterBatchChatContent::Parts(vec![OpenRouterBatchChatContentPart::image_url(
            "https://[::ffff:127.0.0.1]/image.png",
        )]),
        OpenRouterBatchChatContent::Parts(vec![OpenRouterBatchChatContentPart::image_url(
            "https://[::ffff:192.168.1.1]/image.png",
        )]),
        OpenRouterBatchChatContent::Parts(vec![OpenRouterBatchChatContentPart::image_url(
            "https://user:secret@example.com/image.png",
        )]),
        OpenRouterBatchChatContent::Parts(vec![OpenRouterBatchChatContentPart::file_url(
            "report.pdf",
            "file_123",
        )]),
    ] {
        assert!(matches!(
            OpenRouterBatchInput::new(
                OpenRouterBatchEndpoint::ChatCompletions,
                "openai/gpt-6-sol",
                vec![chat_line_with(content)],
            ),
            Err(OpenRouterBatchError::InvalidInput(_))
        ));
    }

    for url in [
        "https://8.8.8.8/image.png",
        "https://[2001:4860:4860::8888]/image.png",
        "https://[::ffff:8.8.8.8]/image.png",
    ] {
        OpenRouterBatchInput::new(
            OpenRouterBatchEndpoint::ChatCompletions,
            "openai/gpt-6-sol",
            vec![chat_line_with(OpenRouterBatchChatContent::Parts(vec![
                OpenRouterBatchChatContentPart::image_url(url),
            ]))],
        )
        .unwrap_or_else(|_| panic!("global public IP literal should be accepted: {url}"));
    }

    let assistant_image = OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::ChatCompletions,
        "openai/gpt-6-sol",
        vec![OpenRouterBatchLine {
            custom_id: "assistant-media".into(),
            body: OpenRouterBatchRequestBody::ChatCompletions(OpenRouterBatchChatBody {
                messages: vec![OpenRouterBatchChatMessage {
                    role: OpenRouterBatchChatRole::Assistant,
                    content: OpenRouterBatchChatContent::Parts(vec![
                        OpenRouterBatchChatContentPart::image_url("https://example.com/image.png"),
                    ]),
                }],
                temperature: None,
                top_p: None,
                max_tokens: Some(64),
                max_completion_tokens: None,
            }),
        }],
    );
    assert!(matches!(
        assistant_image,
        Err(OpenRouterBatchError::InvalidInput(_))
    ));

    let chat_file = OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::ChatCompletions,
        "openai/gpt-6-sol",
        vec![chat_line_with(OpenRouterBatchChatContent::Parts(vec![
            OpenRouterBatchChatContentPart::file_url(
                "report.pdf",
                "https://example.com/report.pdf",
            ),
        ]))],
    )
    .unwrap();
    assert!(matches!(
        chat_file.with_provider_only(vec!["openai".into()]),
        Err(OpenRouterBatchError::InvalidInput(_))
    ));
    // Text routing accepts syntactically valid endpoint slugs; actual
    // model/provider availability remains the service's responsibility.
    assert!(OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::ChatCompletions,
        "openai/gpt-6-sol",
        vec![chat_line("row-1", "text")],
    )
    .unwrap()
    .with_provider_only(vec!["google-vertex".into()])
    .is_ok());
    assert!(matches!(
        OpenRouterBatchInput::new(
            OpenRouterBatchEndpoint::ChatCompletions,
            "openai/gpt-6-sol",
            vec![chat_line("row-1", "text")],
        )
        .unwrap()
        .with_provider_only(vec!["OpenAI".into()]),
        Err(OpenRouterBatchError::InvalidInput(_))
    ));
    for provider in [
        "openai/",
        "openai//turbo",
        "/openai",
        "openai/.",
        "openai/..",
        "openai/../deepinfra",
        "openai/\u{0007}turbo",
    ] {
        assert!(matches!(
            OpenRouterBatchInput::new(
                OpenRouterBatchEndpoint::ChatCompletions,
                "openai/gpt-6-sol",
                vec![chat_line("row-1", "text")],
            )
            .unwrap()
            .with_provider_only(vec![provider.into()]),
            Err(OpenRouterBatchError::InvalidInput(_))
        ));
    }
    let chat_image = || {
        OpenRouterBatchInput::new(
            OpenRouterBatchEndpoint::ChatCompletions,
            "openai/gpt-6-sol",
            vec![chat_line_with(OpenRouterBatchChatContent::Parts(vec![
                OpenRouterBatchChatContentPart::image_url("https://example.com/image.png"),
            ]))],
        )
        .unwrap()
    };
    assert!(chat_image().with_provider_only(vec!["xai".into()]).is_ok());
    assert!(chat_image()
        .with_provider_only(vec!["xai/zdr".into()])
        .is_ok());
    assert!(chat_image()
        .with_provider_only(vec!["deepinfra/turbo".into()])
        .is_ok());
    assert!(matches!(
        chat_image().with_provider_only(vec!["x-ai".into()]),
        Err(OpenRouterBatchError::InvalidInput(_))
    ));
}

#[tokio::test]
async fn submit_get_list_and_delete_use_inline_openrouter_contract() {
    let mock = MockTransport::new([
        reply(202, batch("batch_123", "validating", Value::Null)),
        reply(200, batch("batch_123", "completed", completed_results())),
        reply(
            200,
            json!({
                "object": "list",
                "data": [batch("batch_123", "completed", Value::Null)],
                "first_id": "batch_123",
                "last_id": "batch_123",
                "has_more": false
            }),
        ),
        reply(
            200,
            json!({
                "id": "batch_123",
                "object": "batch_deleted",
                "deletion": {"openrouter": "deleted", "upstream": {"status": "unsupported"}}
            }),
        ),
    ]);
    let client = service(&mock, "account-a");
    let input = OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::ChatCompletions,
        "openai/gpt-6-sol",
        vec![chat_line("row-1", "hello"), chat_line("row-2", "goodbye")],
    )
    .unwrap()
    .with_provider_only(vec!["deepinfra/turbo".into()])
    .unwrap();

    let submitted = client.submit(&input).await.unwrap();
    assert_eq!(submitted.status, OpenRouterBatchStatus::Validating);
    assert_eq!(submitted.reference.model(), "openai/gpt-6-sol");
    assert_eq!(
        submitted.reference.scope().provider_id().as_str(),
        "openrouter"
    );
    assert_eq!(submitted.reference.scope().account_scope(), "account-a");
    assert_eq!(
        submitted.reference.endpoint(),
        OpenRouterBatchEndpoint::ChatCompletions
    );

    let current = client.get(&submitted.reference).await.unwrap();
    assert_eq!(current.status, OpenRouterBatchStatus::Completed);
    let completed_reference = current.reference.clone();
    let results = current.results.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].custom_id, "row-1");
    assert_eq!(results[0].response.as_ref().unwrap().status_code, 200);
    assert_eq!(
        results[0].response.as_ref().unwrap().body["vendor_field"],
        true
    );
    assert_eq!(results[1].custom_id, "row-2");
    assert_eq!(results[1].error.as_ref().unwrap()["code"], "provider_error");
    assert_eq!(current.usage.unwrap()["total_tokens"], 25);

    let page = client
        .list(&OpenRouterBatchListOptions {
            limit: Some(5),
            after: Some("batch_000".into()),
            statuses: vec![OpenRouterBatchStatus::Completed],
            created_after: Some("2026-08-20T00:00:00Z".into()),
            created_before: Some("2026-08-21".into()),
        })
        .await
        .unwrap();
    assert_eq!(page.jobs.len(), 1);
    assert!(!page.has_more);
    assert_eq!(page.last_id.as_deref(), Some("batch_123"));

    let deleted = client.delete(&completed_reference).await.unwrap();
    assert_eq!(deleted.reference.batch_id(), "batch_123");
    assert_eq!(deleted.native["deletion"]["openrouter"], "deleted");

    let requests = mock.requests();
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].url, "https://openrouter.ai/api/v1/batches");
    assert_eq!(
        requests[0]
            .headers
            .iter()
            .find(|(name, _)| name == "Authorization")
            .unwrap()
            .1,
        "Bearer sk-or-test"
    );
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "Content-Type" && value == "application/json"));
    let request_text = String::from_utf8(requests[0].body.to_vec()).unwrap();
    assert!(request_text.starts_with(
        "{\"endpoint\":\"/v1/chat/completions\",\"model\":\"openai/gpt-6-sol\",\"provider\":{\"only\":[\"deepinfra/turbo\"]},\"completion_window\":\"24h\",\"requests\":"
    ));
    assert!(request_text.contains("\"custom_id\":\"row-1\""));
    assert!(request_text.contains("\"messages\":[{"));
    assert!(!request_text.contains("multipart/form-data"));
    assert_eq!(
        requests[1].url,
        "https://openrouter.ai/api/v1/batches/batch_123"
    );
    assert!(requests[2]
        .url
        .contains("limit=5&after=batch_000&status=completed"));
    assert!(requests[2]
        .url
        .contains("created_after=2026-08-20T00%3A00%3A00Z"));
    assert!(requests[2].url.contains("created_before=2026-08-21"));
    assert_eq!(requests[3].method, "DELETE");
}

#[tokio::test]
async fn refs_are_account_bound_and_unknown_submission_is_not_retried() {
    let mock = MockTransport::new([
        reply(202, batch("batch_123", "validating", Value::Null)),
        Err(LlmError::Transport {
            message: "connection closed".into(),
        }),
    ]);
    let client = service(&mock, "account-a");
    let input = OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::ChatCompletions,
        "openai/gpt-6-sol",
        vec![chat_line("row-1", "hello")],
    )
    .unwrap();
    let submitted = client.submit(&input).await.unwrap();

    let other_account = service(&mock, "account-b");
    let error = other_account.get(&submitted.reference).await.unwrap_err();
    assert!(matches!(
        error,
        OpenRouterBatchError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert_eq!(mock.requests().len(), 1);

    assert!(matches!(
        client.submit(&input).await,
        Err(OpenRouterBatchError::OutcomeUnknown {
            operation: "submit",
            ..
        })
    ));
    assert_eq!(mock.requests().len(), 2);
}

#[tokio::test]
async fn accepted_submit_with_an_unusable_response_has_an_unknown_outcome() {
    let mock = MockTransport::new([reply(202, json!({"id": "batch_123"}))]);
    let client = service(&mock, "account-a");
    let input = OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::ChatCompletions,
        "openai/gpt-6-sol",
        vec![chat_line("row-1", "hello")],
    )
    .unwrap();

    assert!(matches!(
        client.submit(&input).await,
        Err(OpenRouterBatchError::OutcomeUnknownResponse {
            operation: "submit",
            ..
        })
    ));
    assert_eq!(mock.requests().len(), 1);
}

#[tokio::test]
async fn list_validates_and_encodes_creation_time_filters_before_network_access() {
    let no_io = MockTransport::new([]);
    let client = service(&no_io, "account-a");
    for options in [
        OpenRouterBatchListOptions::default().created_after("not-a-date"),
        OpenRouterBatchListOptions::default().created_before("2026-02-30"),
        OpenRouterBatchListOptions::default()
            .created_after("2026-08-20")
            .created_before("2026-08-20T00:00:00Z"),
        OpenRouterBatchListOptions::default()
            .created_after("1787184000.25")
            .created_before("1787184000.25"),
        OpenRouterBatchListOptions::default()
            .created_after("1787184001")
            .created_before("1787184000"),
    ] {
        assert!(matches!(
            client.list(&options).await,
            Err(OpenRouterBatchError::InvalidInput(_))
        ));
    }
    assert!(no_io.requests().is_empty());

    let mock = MockTransport::new([reply(
        200,
        json!({
            "object": "list",
            "data": [],
            "first_id": null,
            "last_id": null,
            "has_more": false
        }),
    )]);
    service(&mock, "account-a")
        .list(
            &OpenRouterBatchListOptions::default()
                .created_after("1787184000.25")
                .created_before("2026-08-21"),
        )
        .await
        .unwrap();
    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].url.contains("created_after=1787184000.25"));
    assert!(requests[0].url.contains("created_before=2026-08-21"));
}

#[tokio::test]
async fn malformed_per_item_results_are_rejected_without_losing_native_error_data() {
    let malformed = batch(
        "batch_123",
        "completed",
        json!([{
            "custom_id": "row-1",
            "response": {"status_code": 200, "body": {}},
            "error": {"code": "also_set"}
        }]),
    );
    let mock = MockTransport::new([
        reply(202, batch("batch_123", "validating", Value::Null)),
        reply(200, malformed),
    ]);
    let client = service(&mock, "account-a");
    let input = OpenRouterBatchInput::new(
        OpenRouterBatchEndpoint::ChatCompletions,
        "openai/gpt-6-sol",
        vec![chat_line("row-1", "hello")],
    )
    .unwrap();
    let submitted = client.submit(&input).await.unwrap();
    assert!(matches!(
        client.get(&submitted.reference).await,
        Err(OpenRouterBatchError::InvalidResponse(_))
    ));
}
