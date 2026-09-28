use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{files::*, protocol::*, providers::openai::batches::*, *};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
};

struct Reply {
    method: &'static str,
    url: &'static str,
    body: Result<Vec<u8>, LlmError>,
}
struct Mock {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<HttpRequest>>,
}
#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected HTTP call");
        assert_eq!(request.method, reply.method);
        assert_eq!(request.url, reply.url);
        self.requests.lock().unwrap().push(request);
        let value = bytes::Bytes::from(reply.body?);
        let chunks = (0..value.len())
            .step_by(8192)
            .map(|start| Ok(value.slice(start..(start + 8192).min(value.len()))))
            .collect::<Vec<_>>();
        Ok(StreamResponse {
            status: 200,
            headers: vec![("x-request-id".into(), "req-1".into())],
            body: futures::stream::iter(chunks).boxed(),
        })
    }
}
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openai","profile_name":"openai","base_url":"https://files.test/v1",
        "protocol":"open_ai_responses","auth":"none",
        "batches":{"mode":"enabled","value":{
            "api":"open_ai","endpoint":"https://batch.test/v1/batches",
            "files_endpoint":"https://batch.test/v1/files","auth":{"type":"bearer"}
        }}
    }))
    .unwrap()
}
fn options(scope: &str) -> RequestOptions {
    RequestOptions {
        account_scope: Some(scope.into()),
        credential: Some(Secret::new("test-key".into())),
        ..Default::default()
    }
}
fn file() -> ProviderFileRef {
    ProviderFileRef {
        provider_id: "openai".into(),
        profile_name: "openai".into(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint("https://files.test/v1"),
        account_scope: Some("acct-a".into()),
        protocol: ProtocolFamily::OpenAiResponses,
        file_id: "file_1".into(),
        uri: None,
        filename: Some("input.jsonl".into()),
        media_type: Some("application/x-ndjson".into()),
        size_bytes: None,
        expires_at: None,
        processing_status: None,
        downloadable: None,
        purpose: Some("batch".into()),
    }
}
fn attachment_file(file_id: &str, purpose: &str, media_type: &str) -> ProviderFileRef {
    ProviderFileRef {
        provider_id: "openai".into(),
        profile_name: "openai".into(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint("https://files.test/v1"),
        account_scope: Some("acct-a".into()),
        protocol: ProtocolFamily::OpenAiResponses,
        file_id: file_id.into(),
        uri: None,
        filename: Some("attachment.pdf".into()),
        media_type: Some(media_type.into()),
        size_bytes: Some(42),
        expires_at: None,
        processing_status: None,
        downloadable: None,
        purpose: Some(purpose.into()),
    }
}
fn setup(replies: Vec<Reply>) -> (LlmClient, Arc<Mock>) {
    let mock = Arc::new(Mock {
        replies: Mutex::new(replies.into()),
        requests: Mutex::new(vec![]),
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile()])
        .with_region(Region::International)
        .build()
        .unwrap();
    (client, mock)
}
fn ok(method: &'static str, url: &'static str, body: Value) -> Reply {
    Reply {
        method,
        url,
        body: Ok(serde_json::to_vec(&body).unwrap()),
    }
}
fn raw(method: &'static str, url: &'static str, body: &'static str) -> Reply {
    Reply {
        method,
        url,
        body: Ok(body.as_bytes().to_vec()),
    }
}
fn job(status: &str) -> Value {
    json!({"id":"batch_1","endpoint":"/v1/responses","input_file_id":"file_1",
        "status":status,"output_file_id":"file_output","error_file_id":"file_errors",
        "request_counts":{"total":2,"completed":1,"failed":1},
        "usage":{"input_tokens":10},"errors":{"data":[{"code":"invalid_request"}]}})
}

fn result_ref() -> BatchResultRef {
    BatchResultRef {
        job: BatchJobRef {
            provider_id: "openai".into(),
            profile_name: "openai".into(),
            endpoint_fingerprint: provider_file_endpoint_fingerprint(
                "https://batch.test/v1/batches",
            ),
            account_scope: "acct-a".into(),
            job_id: "batch_1".into(),
        },
        result_endpoint_fingerprint: provider_file_endpoint_fingerprint(
            "https://batch.test/v1/files",
        ),
        file_id: "file_output".into(),
        kind: BatchResultKind::Output,
    }
}

#[tokio::test]
async fn result_stream_decodes_across_chunks_and_stops_on_duplicate_id() {
    let (client, mock) = setup(vec![raw(
        "GET",
        "https://batch.test/v1/files/file_output/content",
        "{\"custom_id\":\"first\",\"response\":{\"status_code\":200}}\r\n{\"custom_id\":\"second\",\"error\":{\"code\":\"bad\"}}\n{\"custom_id\":\"first\",\"error\":{\"code\":\"duplicate\"}}",
    )]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .batches()
        .stream_result(&result_ref(), &options("acct-a"))
        .await
        .unwrap();
    assert_eq!(stream.next().await.unwrap().unwrap().custom_id, "first");
    assert_eq!(stream.next().await.unwrap().unwrap().custom_id, "second");
    assert!(matches!(
        stream.next().await.unwrap(),
        Err(BatchError::InvalidResult(_))
    ));
    assert!(stream.next().await.is_none());
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn result_stream_reads_more_than_64_mib_without_collecting_the_file() {
    let padding = "x".repeat(1024 * 1024);
    let mut body = Vec::new();
    for index in 0..65 {
        serde_json::to_writer(
            &mut body,
            &json!({
                "custom_id": format!("item-{index}"),
                "response": {"status_code": 200, "body": padding}
            }),
        )
        .unwrap();
        body.push(b'\n');
    }
    assert!(body.len() > 64 * 1024 * 1024);
    let (client, _) = setup(vec![Reply {
        method: "GET",
        url: "https://batch.test/v1/files/file_output/content",
        body: Ok(body),
    }]);
    let mut stream = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .batches()
        .stream_result(&result_ref(), &options("acct-a"))
        .await
        .unwrap();
    for index in 0..65 {
        let row = stream.next().await.unwrap().unwrap();
        assert_eq!(row.custom_id, format!("item-{index}"));
    }
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn result_stream_checks_scope_before_http() {
    let (client, mock) = setup(vec![]);
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .batches()
            .stream_result(&result_ref(), &options("acct-b"))
            .await,
        Err(BatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[test]
fn jsonl_encoder_rejects_duplicate_ids_mixed_models_and_streaming() {
    let line = BatchLine {
        custom_id: "one".into(),
        body: json!({"model":"m","input":"hello"}),
    };
    let encoded = encode_jsonl(BatchEndpoint::Responses, std::slice::from_ref(&line)).unwrap();
    let mut written = Vec::new();
    let count = write_jsonl(
        BatchEndpoint::Responses,
        std::slice::from_ref(&line),
        &mut written,
    )
    .unwrap();
    assert_eq!(count as usize, encoded.len());
    assert_eq!(written, encoded);
    let row: Value = serde_json::from_slice(encoded.trim_ascii_end()).unwrap();
    assert_eq!(row["custom_id"], "one");
    assert_eq!(row["url"], "/v1/responses");
    assert_eq!(row["method"], "POST");
    assert!(encode_jsonl(BatchEndpoint::Responses, &[line.clone(), line.clone()]).is_err());
    let mut other = line.clone();
    other.custom_id = "two".into();
    other.body["model"] = json!("different");
    assert!(encode_jsonl(BatchEndpoint::Responses, &[line.clone(), other]).is_err());
    let mut untouched = b"sentinel".to_vec();
    let invalid = BatchLine {
        custom_id: "two".into(),
        body: json!({"model":"different","input":"hello"}),
    };
    assert!(write_jsonl(
        BatchEndpoint::Responses,
        &[line.clone(), invalid],
        &mut untouched
    )
    .is_err());
    assert_eq!(untouched, b"sentinel");
    let mut streaming = line;
    streaming.body["stream"] = json!(true);
    assert!(encode_jsonl(BatchEndpoint::Responses, &[streaming]).is_err());
}

#[test]
fn durable_attachment_manifest_is_required_and_exact() {
    let line = BatchLine {
        custom_id: "with-file".into(),
        body: json!({
            "model":"gpt-6-sol",
            "input":[{"role":"user","content":[
                {"type":"input_file","file_id":"file_doc"},
                {"type":"input_text","text":"summarize"}
            ]}]
        }),
    };
    assert!(encode_jsonl(BatchEndpoint::Responses, std::slice::from_ref(&line)).is_err());

    let attachment = BatchAttachmentRef::responses_file(attachment_file(
        "file_doc",
        "user_data",
        "application/pdf",
    ));
    let encoded = encode_jsonl_with_attachments(
        BatchEndpoint::Responses,
        std::slice::from_ref(&line),
        std::slice::from_ref(&attachment),
    )
    .unwrap();
    let mut output = Vec::new();
    let count = write_jsonl_with_attachments(
        BatchEndpoint::Responses,
        std::slice::from_ref(&line),
        std::slice::from_ref(&attachment),
        &mut output,
    )
    .unwrap();
    assert_eq!(output, encoded);
    assert_eq!(count as usize, encoded.len());

    let wrong_file = BatchAttachmentRef::responses_file(attachment_file(
        "file_other",
        "user_data",
        "application/pdf",
    ));
    assert!(encode_jsonl_with_attachments(
        BatchEndpoint::Responses,
        std::slice::from_ref(&line),
        &[wrong_file]
    )
    .is_err());
}

#[test]
fn durable_input_shapes_preflight_expiring_urls_and_chat_requires_pdf() {
    let remote_file_url = BatchLine {
        custom_id: "temporary-url".into(),
        body: json!({"model":"m","input":[{"type":"input_file","file_url":"https://signed.example/doc.pdf"}]}),
    };
    assert!(encode_jsonl(BatchEndpoint::Responses, &[remote_file_url]).is_err());

    let remote_image_url = BatchLine {
        custom_id: "temporary-image".into(),
        body: json!({"model":"m","input":[{"type":"input_image","image_url":"https://signed.example/image.png"}]}),
    };
    assert!(encode_jsonl(BatchEndpoint::Responses, &[remote_image_url]).is_err());

    let response_image = BatchLine {
        custom_id: "image".into(),
        body: json!({"model":"m","input":[{"type":"input_image","file_id":"file_img"}]}),
    };
    let image =
        BatchAttachmentRef::responses_image(attachment_file("file_img", "vision", "image/png"));
    assert!(
        encode_jsonl_with_attachments(BatchEndpoint::Responses, &[response_image], &[image])
            .is_ok()
    );

    let chat_pdf = BatchLine {
        custom_id: "pdf".into(),
        body: json!({"model":"m","messages":[{"role":"user","content":[
            {"type":"file","file":{"file_id":"file_pdf"}}
        ]}]}),
    };
    let pdf = BatchAttachmentRef::chat_completions_pdf(attachment_file(
        "file_pdf",
        "user_data",
        "application/pdf",
    ));
    assert!(encode_jsonl_with_attachments(
        BatchEndpoint::ChatCompletions,
        std::slice::from_ref(&chat_pdf),
        &[pdf]
    )
    .is_ok());

    let not_pdf = BatchAttachmentRef::chat_completions_pdf(attachment_file(
        "file_pdf",
        "user_data",
        "text/plain",
    ));
    assert!(
        encode_jsonl_with_attachments(BatchEndpoint::ChatCompletions, &[chat_pdf], &[not_pdf])
            .is_err()
    );

    let inline_html = BatchLine {
        custom_id: "wrong-inline-type".into(),
        body: json!({"model":"m","messages":[{"role":"user","content":[
            {"type":"file","file":{"filename":"doc.pdf","file_data":"data:text/plain;base64,SGVsbG8="}}
        ]}]}),
    };
    assert!(encode_jsonl(BatchEndpoint::ChatCompletions, &[inline_html]).is_err());
}

#[tokio::test]
async fn submit_get_list_cancel_preserve_scope_counts_and_output_ids() {
    let (client, mock) = setup(vec![
        ok("POST", "https://batch.test/v1/batches", job("validating")),
        ok(
            "GET",
            "https://batch.test/v1/batches/batch_1",
            job("completed"),
        ),
        ok(
            "GET",
            "https://batch.test/v1/batches?limit=1&after=batch_old",
            json!({"data":[job("completed")],"has_more":true,"last_id":"batch_1"}),
        ),
        ok(
            "POST",
            "https://batch.test/v1/batches/batch_1/cancel",
            job("cancelling"),
        ),
        raw(
            "GET",
            "https://batch.test/v1/files/file_output/content",
            "{\"custom_id\":\"two\",\"response\":{\"status_code\":200,\"body\":{\"ok\":true}},\"error\":null}\n{\"custom_id\":\"one\",\"response\":null,\"error\":{\"code\":\"invalid_request\"}}\n",
        ),
    ]);
    let opts = options("acct-a");
    let submitted = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .batches()
        .submit(
            &file(),
            BatchEndpoint::Responses,
            Some(&BTreeMap::from([("tag".into(), "nightly".into())])),
            &opts,
        )
        .await
        .unwrap();
    assert_eq!(submitted.status, BatchStatus::Validating);
    assert_eq!(submitted.reference.account_scope, "acct-a");
    let current = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .batches()
        .get(&submitted.reference, &opts)
        .await
        .unwrap();
    assert_eq!(current.status, BatchStatus::Completed);
    assert_eq!(current.output_file_id.as_deref(), Some("file_output"));
    assert_eq!(current.error_file_id.as_deref(), Some("file_errors"));
    assert_eq!(current.request_counts.as_ref().unwrap()["failed"], 1);
    let page = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .batches()
        .list(1, Some("batch_old"), &opts)
        .await
        .unwrap();
    assert!(page.has_more);
    assert_eq!(page.last_id.as_deref(), Some("batch_1"));
    let cancelled = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .batches()
        .cancel(&submitted.reference, &opts)
        .await
        .unwrap();
    assert_eq!(cancelled.status, BatchStatus::Cancelling);
    assert!(!cancelled.status.is_terminal());
    let rows = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .batches()
        .read_result(&current.output_ref().unwrap(), &opts)
        .await
        .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].custom_id, "two");
    assert_eq!(rows[0].response.as_ref().unwrap()["status_code"], 200);
    assert_eq!(rows[1].custom_id, "one");
    assert_eq!(rows[1].error.as_ref().unwrap()["code"], "invalid_request");
    let sent = mock.requests.lock().unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&sent[0].body).unwrap(),
        json!({"input_file_id":"file_1","endpoint":"/v1/responses",
            "completion_window":"24h","metadata":{"tag":"nightly"}})
    );
    assert!(sent.iter().all(|request| {
        request
            .headers
            .iter()
            .any(|(key, value)| key == "authorization" && value == "Bearer test-key")
    }));
}

#[tokio::test]
async fn submit_with_attachments_reconciles_files_before_batch_creation() {
    let (client, mock) = setup(vec![
        ok(
            "GET",
            "https://batch.test/v1/files/file_doc",
            json!({"id":"file_doc","purpose":"user_data","expires_at":null}),
        ),
        ok("POST", "https://batch.test/v1/batches", job("validating")),
    ]);
    let attachment = BatchAttachmentRef::responses_file(attachment_file(
        "file_doc",
        "user_data",
        "application/pdf",
    ));
    let submitted = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .batches()
        .submit_with_attachments(
            &file(),
            BatchEndpoint::Responses,
            None,
            &[attachment],
            &options("acct-a"),
        )
        .await
        .unwrap();
    assert_eq!(submitted.status, BatchStatus::Validating);
    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].method, "GET");
    assert_eq!(requests[1].method, "POST");
}

#[tokio::test]
async fn attachment_must_survive_completion_and_cancel_window() {
    let expires_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + 24 * 60 * 60
        + 10 * 60
        - 1;
    let (client, mock) = setup(vec![ok(
        "GET",
        "https://batch.test/v1/files/file_doc",
        json!({"id":"file_doc","purpose":"user_data","expires_at":expires_at}),
    )]);
    let attachment = BatchAttachmentRef::responses_file(attachment_file(
        "file_doc",
        "user_data",
        "application/pdf",
    ));
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .batches()
            .submit_with_attachments(
                &file(),
                BatchEndpoint::Responses,
                None,
                &[attachment],
                &options("acct-a"),
            )
            .await,
        Err(BatchError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn wrong_attachment_scope_or_remote_purpose_fails_before_batch_write() {
    let (client, mock) = setup(vec![]);
    let mut wrong_scope = attachment_file("file_doc", "user_data", "application/pdf");
    wrong_scope.account_scope = Some("acct-b".into());
    let wrong_scope = BatchAttachmentRef::responses_file(wrong_scope);
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .batches()
            .submit_with_attachments(
                &file(),
                BatchEndpoint::Responses,
                None,
                &[wrong_scope],
                &options("acct-a"),
            )
            .await,
        Err(BatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());

    let (client, mock) = setup(vec![ok(
        "GET",
        "https://batch.test/v1/files/file_doc",
        json!({"id":"file_doc","purpose":"batch","expires_at":null}),
    )]);
    let attachment = BatchAttachmentRef::responses_file(attachment_file(
        "file_doc",
        "user_data",
        "application/pdf",
    ));
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .batches()
            .submit_with_attachments(
                &file(),
                BatchEndpoint::Responses,
                None,
                &[attachment],
                &options("acct-a"),
            )
            .await,
        Err(BatchError::InvalidResponse(_))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn wrong_scope_and_wrong_purpose_fail_before_http() {
    let (client, mock) = setup(vec![]);
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .batches()
            .submit(&file(), BatchEndpoint::Responses, None, &options("acct-b"))
            .await,
        Err(BatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    let mut wrong = file();
    wrong.purpose = Some("user_data".into());
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .batches()
            .submit(&wrong, BatchEndpoint::Responses, None, &options("acct-a"))
            .await,
        Err(BatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn result_reference_scope_and_jsonl_shape_are_checked() {
    let (client, mock) = setup(vec![raw(
        "GET",
        "https://batch.test/v1/files/file_output/content",
        "{\"custom_id\":\"one\",\"response\":{\"status_code\":200}}\n{\"custom_id\":\"one\",\"error\":{\"code\":\"bad\"}}\n",
    )]);
    let job = BatchJobRef {
        provider_id: "openai".into(),
        profile_name: "openai".into(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint("https://batch.test/v1/batches"),
        account_scope: "acct-a".into(),
        job_id: "batch_1".into(),
    };
    let reference = BatchResultRef {
        job,
        result_endpoint_fingerprint: provider_file_endpoint_fingerprint(
            "https://batch.test/v1/files",
        ),
        file_id: "file_output".into(),
        kind: BatchResultKind::Output,
    };
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .batches()
            .read_result(&reference, &options("acct-b"))
            .await,
        Err(BatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    let mut wrong_endpoint = reference.clone();
    wrong_endpoint.result_endpoint_fingerprint = "other".into();
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .batches()
            .read_result(&wrong_endpoint, &options("acct-a"))
            .await,
        Err(BatchError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .batches()
            .read_result(&reference, &options("acct-a"))
            .await,
        Err(BatchError::InvalidResult(_))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn uncertain_submission_never_retries() {
    let (client, mock) = setup(vec![Reply {
        method: "POST",
        url: "https://batch.test/v1/batches",
        body: Err(LlmError::Transport {
            message: "connection reset".into(),
        }),
    }]);
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .batches()
            .submit(&file(), BatchEndpoint::Responses, None, &options("acct-a"))
            .await,
        Err(BatchError::OutcomeUnknown {
            operation: "submit",
            ..
        })
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[test]
fn invalid_batch_route_is_rejected_by_builder() {
    let mut p = profile();
    if let ServiceSetting::Enabled(route) = &mut p.batches {
        route.endpoint = "https://batch.test/v1/batches?leak=1".into();
    }
    assert!(LlmClientBuilder::with_transport(
        Arc::new(Mock {
            replies: Mutex::new(VecDeque::new()),
            requests: Mutex::new(vec![])
        }),
        &[p]
    )
    .with_region(Region::International)
    .build()
    .is_err());
}
