use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::files::{
    FilePurpose, FileService, FileUploadError, UploadFile, UploadFileStream,
};
use lingxi_llm_client::protocol::{LlmError, ProviderProfile};
use lingxi_llm_client::{HttpRequest, HttpResponse, HttpStreamRequest, StreamResponse, Transport};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

#[derive(Debug)]
struct CapturedRequest {
    url: String,
    headers: Vec<(String, String)>,
    content_length: u64,
    body: Vec<u8>,
}

#[derive(Default)]
struct Capture {
    buffered: Mutex<Vec<HttpRequest>>,
    streamed: Mutex<Vec<CapturedRequest>>,
    responses: Mutex<VecDeque<HttpResponse>>,
}

impl Capture {
    fn with_responses(responses: Vec<HttpResponse>) -> Self {
        Self {
            buffered: Mutex::new(Vec::new()),
            streamed: Mutex::new(Vec::new()),
            responses: Mutex::new(responses.into()),
        }
    }

    fn response(&self) -> Result<HttpResponse, LlmError> {
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| LlmError::Transport {
                message: "test transport has no response".into(),
            })
    }
}

#[async_trait]
impl Transport for Capture {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.buffered.lock().unwrap().push(request);
        let response = self.response()?;
        Ok(StreamResponse {
            status: response.status,
            headers: response.headers,
            body: stream::iter(vec![Ok(response.body)]).boxed(),
        })
    }

    async fn send_stream(
        &self,
        mut request: HttpStreamRequest,
    ) -> Result<StreamResponse, LlmError> {
        let mut captured = CapturedRequest {
            url: request.url,
            headers: request.headers,
            content_length: request.content_length,
            body: Vec::new(),
        };
        while let Some(chunk) = request.body.next().await {
            match chunk {
                Ok(chunk) => captured.body.extend_from_slice(&chunk),
                Err(error) => {
                    self.streamed.lock().unwrap().push(captured);
                    return Err(error);
                }
            }
        }
        self.streamed.lock().unwrap().push(captured);
        let response = self.response()?;
        Ok(StreamResponse {
            status: response.status,
            headers: response.headers,
            body: stream::iter(vec![Ok(response.body)]).boxed(),
        })
    }
}

fn make_profile(provider_id: &str, base_url: &str, protocol: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": provider_id,
        "profile_name": format!("{provider_id}-profile"),
        "base_url": base_url,
        "protocol": protocol,
        "auth": "none",
        "models": []
    }))
    .unwrap()
}

fn response(value: Value) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: vec![("content-type".into(), "application/json".into())],
        body: serde_json::to_vec(&value).unwrap().into(),
    }
}

fn upload(file: &[u8]) -> UploadFileStream {
    let chunk = Bytes::copy_from_slice(file);
    UploadFileStream::new(
        "input.bin",
        "application/pdf",
        file.len() as u64,
        stream::once(async move { Ok(chunk) }),
    )
}

fn purpose_value(body: &[u8]) -> Option<String> {
    let body = String::from_utf8_lossy(body);
    let (_, value) = body.split_once("name=\"purpose\"\r\n\r\n")?;
    Some(value.split_once("\r\n").unwrap().0.to_owned())
}

#[tokio::test]
async fn multipart_stream_uploads_preserve_each_supported_purpose_wire_form() {
    let cases = [
        (
            "openai",
            "https://api.openai.com/v1",
            "open_ai_responses",
            FilePurpose::ModelInput,
            Some("user_data"),
            Some("acct-a"),
        ),
        (
            "anthropic",
            "https://api.anthropic.com",
            "anthropic_messages",
            FilePurpose::ModelInput,
            None,
            Some("acct-a"),
        ),
        (
            "xai",
            "https://api.x.ai/v1",
            "open_ai_responses",
            FilePurpose::ModelInput,
            Some("assistants"),
            Some("acct-a"),
        ),
        (
            "openrouter",
            "https://openrouter.ai/api/v1",
            "open_ai_chat",
            FilePurpose::Workspace,
            None,
            None,
        ),
        (
            "moonshot",
            "https://api.moonshot.cn/v1",
            "open_ai_chat",
            FilePurpose::Extraction,
            Some("file-extract"),
            None,
        ),
        (
            "qwen",
            "https://dashscope.aliyuncs.com/api/v1",
            "open_ai_chat",
            FilePurpose::Extraction,
            Some("file-extract"),
            None,
        ),
        (
            "zhipu",
            "https://api.z.ai/api/paas/v4",
            "open_ai_chat",
            FilePurpose::Auxiliary,
            Some("agent"),
            None,
        ),
        (
            "zhipu",
            "https://open.bigmodel.cn/api/paas/v4",
            "open_ai_chat",
            FilePurpose::Auxiliary,
            Some("agent"),
            None,
        ),
        (
            "minimax",
            "https://api.minimaxi.com/v1",
            "open_ai_chat",
            FilePurpose::AsyncTtsInput,
            Some("t2a_async_input"),
            Some("acct-a"),
        ),
    ];

    for (provider, base, protocol, purpose, expected_purpose, scope) in cases {
        let body = if provider == "minimax" {
            response(json!({
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "file_id": "uploaded-file",
                "purpose": "t2a_async_input"
            }))
        } else {
            response(json!({"id": "uploaded-file"}))
        };
        let http = Capture::with_responses(vec![body]);
        let profile = make_profile(provider, base, protocol);
        let service = FileService::new(&http, &profile, None, None, scope);
        let uploaded = service
            .upload_stream(upload(b"pdf"), purpose)
            .await
            .unwrap_or_else(|error| panic!("{provider} upload failed: {error}"));
        assert_eq!(uploaded.file_id, "uploaded-file", "{provider}");
        let requests = http.streamed.lock().unwrap();
        assert_eq!(requests.len(), 1, "{provider}");
        assert_eq!(
            requests[0].content_length,
            requests[0].body.len() as u64,
            "{provider}"
        );
        assert_eq!(
            purpose_value(&requests[0].body).as_deref(),
            expected_purpose,
            "{provider}"
        );
        assert!(http.buffered.lock().unwrap().is_empty(), "{provider}");
    }
}

#[tokio::test]
async fn async_tts_scope_and_upload_preflight_happen_before_stream_polling() {
    let http = Capture::default();
    let minimax_profile = make_profile("minimax", "https://api.minimaxi.com/v1", "open_ai_chat");
    let missing_scope = FileService::new(&http, &minimax_profile, None, None, None);
    let polled = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&polled);
    let body = stream::once(async move {
        flag.store(true, Ordering::SeqCst);
        Ok(Bytes::from_static(b"text"))
    });
    assert!(matches!(
        missing_scope
            .upload_stream(
                UploadFileStream::new("input.txt", "text/plain", 4, body),
                FilePurpose::AsyncTtsInput
            )
            .await,
        Err(FileUploadError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(!polled.load(Ordering::SeqCst));
    assert!(http.buffered.lock().unwrap().is_empty());
    assert!(http.streamed.lock().unwrap().is_empty());

    let buffered_error = missing_scope
        .upload(
            &UploadFile {
                filename: "input.txt".into(),
                media_type: "text/plain".into(),
                bytes: Bytes::from_static(b"text"),
            },
            FilePurpose::AsyncTtsInput,
        )
        .await;
    assert!(matches!(
        buffered_error,
        Err(LlmError::InvalidRequest { .. })
    ));
    assert!(http.buffered.lock().unwrap().is_empty());

    let openai = make_profile("openai", "https://api.openai.com/v1", "open_ai_responses");
    let oversized = FileService::new(&http, &openai, None, None, Some("acct-a"));
    let polled = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&polled);
    let body = stream::once(async move {
        flag.store(true, Ordering::SeqCst);
        Ok(Bytes::from_static(b"x"))
    });
    assert!(matches!(
        oversized
            .upload_stream(
                UploadFileStream::new("large.pdf", "application/pdf", 50_000_001, body),
                FilePurpose::ModelInput
            )
            .await,
        Err(FileUploadError::Llm(LlmError::RequestTooLarge { .. }))
    ));
    assert!(!polled.load(Ordering::SeqCst));
    assert!(http.buffered.lock().unwrap().is_empty());
    assert!(http.streamed.lock().unwrap().is_empty());

    let minimax = FileService::new(&http, &minimax_profile, None, None, Some("acct-a"));
    let error = minimax
        .upload(
            &UploadFile {
                filename: "input.txt".into(),
                media_type: "text/plain".into(),
                bytes: Bytes::from_static(b"text"),
            },
            FilePurpose::AsyncTtsInput,
        )
        .await;
    assert!(error.is_err()); // no mock response is available; scope preflight succeeded.
    assert_eq!(http.buffered.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn multipart_length_failures_do_not_emit_the_closing_boundary_and_are_unknown() {
    for (declared, chunks) in [
        (4, vec![Bytes::from_static(b"abc")]),
        (2, vec![Bytes::from_static(b"abc")]),
    ] {
        let http = Capture::with_responses(vec![response(json!({"id": "unused"}))]);
        let profile = make_profile("openrouter", "https://openrouter.ai/api/v1", "open_ai_chat");
        let service = FileService::new(&http, &profile, None, None, None);
        let body = stream::iter(chunks.into_iter().map(Ok));
        let error = service
            .upload_stream(
                UploadFileStream::new("input.pdf", "application/pdf", declared, body),
                FilePurpose::Workspace,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, FileUploadError::OutcomeUnknown { .. }));
        let requests = http.streamed.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(!requests[0].body.ends_with(b"\r\n--\r\n"));
        assert!(!String::from_utf8_lossy(&requests[0].body).ends_with("--\r\n"));
    }
}

#[tokio::test]
async fn multipart_source_interruption_is_reported_as_an_unknown_mutation() {
    let http = Capture::with_responses(vec![response(json!({"id": "unused"}))]);
    let profile = make_profile("openrouter", "https://openrouter.ai/api/v1", "open_ai_chat");
    let service = FileService::new(&http, &profile, None, None, None);
    let body = stream::iter(vec![Err(LlmError::Transport {
        message: "source read failed".into(),
    })]);
    let error = service
        .upload_stream(
            UploadFileStream::new("input.pdf", "application/pdf", 3, body),
            FilePurpose::Workspace,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, FileUploadError::OutcomeUnknown { .. }));
    let requests = http.streamed.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let boundary = requests[0]
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .and_then(|(_, value)| value.split_once("boundary=").map(|(_, boundary)| boundary))
        .unwrap();
    let closing = format!("--{boundary}--\r\n");
    assert!(!requests[0]
        .body
        .windows(closing.len())
        .any(|window| window == closing.as_bytes()));
}

#[tokio::test]
async fn malformed_success_responses_are_unknown_but_explicit_minimax_rejections_are_known() {
    let cases = [
        (
            "openrouter",
            "https://openrouter.ai/api/v1",
            FilePurpose::Workspace,
            response(json!({"unexpected": "missing file id"})),
            true,
        ),
        (
            "minimax",
            "https://api.minimaxi.com/v1",
            FilePurpose::VoiceClone,
            response(json!({"file_id": "id-without-base-response"})),
            true,
        ),
        (
            "minimax",
            "https://api.minimaxi.com/v1",
            FilePurpose::VoiceClone,
            response(json!({
                "base_resp": {"status_code": 1004, "status_msg": "invalid file"}
            })),
            false,
        ),
    ];

    for (provider, base, purpose, response, unknown) in cases {
        let http = Capture::with_responses(vec![response]);
        let profile = make_profile(provider, base, "open_ai_chat");
        let service = FileService::new(&http, &profile, None, None, None);
        let result = service.upload_stream(upload(b"pdf"), purpose).await;
        if unknown {
            assert!(matches!(
                result,
                Err(FileUploadError::OutcomeUnknown { .. })
            ));
        } else {
            assert!(matches!(
                result,
                Err(FileUploadError::Llm(LlmError::ProviderInternal { .. }))
            ));
        }
    }
}

#[tokio::test]
async fn mismatched_success_purpose_is_unknown_and_retains_the_scoped_reference() {
    let http = Capture::with_responses(vec![response(json!({
        "id": "uploaded-with-wrong-purpose",
        "purpose": "batch"
    }))]);
    let profile = make_profile("openai", "https://api.openai.com/v1", "open_ai_responses");
    let service = FileService::new(&http, &profile, None, None, Some("acct-a"));
    match service
        .upload_stream(upload(b"pdf"), FilePurpose::ModelInput)
        .await
        .unwrap_err()
    {
        FileUploadError::OutcomeUnknown { reference, .. } => {
            let reference = reference.expect("decoded reference should be retained");
            assert_eq!(reference.file_id, "uploaded-with-wrong-purpose");
            assert_eq!(reference.purpose.as_deref(), Some("batch"));
            assert_eq!(reference.account_scope.as_deref(), Some("acct-a"));
        }
        other => panic!("expected an unknown result with retained reference, got {other}"),
    }
}

#[tokio::test]
async fn multipart_size_overflow_fails_before_auth_or_network() {
    let http = Capture::default();
    let profile = make_profile("minimax", "https://api.minimaxi.com/v1", "open_ai_chat");
    let service = FileService::new(&http, &profile, None, None, Some("acct-a"));
    let error = service
        .upload_stream(
            UploadFileStream::new(
                "large.bin",
                "application/octet-stream",
                u64::MAX,
                stream::empty::<Result<Bytes, LlmError>>(),
            ),
            FilePurpose::VoiceClone,
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        FileUploadError::Llm(LlmError::RequestTooLarge { .. })
    ));
    assert!(http.buffered.lock().unwrap().is_empty());
    assert!(http.streamed.lock().unwrap().is_empty());
}

#[tokio::test]
async fn gemini_stream_uses_resumable_start_then_one_exact_raw_body_without_polling() {
    let start = HttpResponse {
        status: 200,
        headers: vec![(
            "x-goog-upload-url".into(),
            "https://generativelanguage.googleapis.com/upload/session/abc".into(),
        )],
        body: Bytes::new(),
    };
    let finish = response(json!({
        "file": {
            "name": "files/gemini-file",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/gemini-file",
            "mimeType": "application/pdf",
            "state": "ACTIVE"
        }
    }));
    let http = Capture::with_responses(vec![start, finish]);
    let profile = make_profile(
        "google",
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini_generate_content",
    );
    let service = FileService::new(&http, &profile, None, None, Some("acct-a"));
    let uploaded = service
        .upload_stream(upload(b"pdf"), FilePurpose::ModelInput)
        .await
        .unwrap();
    assert_eq!(uploaded.file_id, "files/gemini-file");
    assert_eq!(uploaded.account_scope.as_deref(), Some("acct-a"));
    let starts = http.buffered.lock().unwrap();
    assert_eq!(starts.len(), 1, "stream upload must not poll Gemini");
    assert!(starts[0].headers.iter().any(|(name, value)| name
        .eq_ignore_ascii_case("x-goog-upload-command")
        && value == "start"));
    assert!(starts[0].headers.iter().any(|(name, value)| name
        .eq_ignore_ascii_case("x-goog-upload-header-content-length")
        && value == "3"));
    let requests = http.streamed.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].url,
        "https://generativelanguage.googleapis.com/upload/session/abc"
    );
    assert_eq!(requests[0].body, b"pdf");
    assert_eq!(requests[0].content_length, 3);
    assert!(requests[0].headers.iter().any(|(name, value)| name
        .eq_ignore_ascii_case("x-goog-upload-command")
        && value == "upload, finalize"));
}

#[tokio::test]
async fn gemini_processing_result_is_returned_for_explicit_caller_resume() {
    let start = HttpResponse {
        status: 200,
        headers: vec![(
            "x-goog-upload-url".into(),
            "https://generativelanguage.googleapis.com/upload/session/abc".into(),
        )],
        body: Bytes::new(),
    };
    let finish = response(json!({
        "file": {
            "name": "files/gemini-pending",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/gemini-pending",
            "mimeType": "application/pdf",
            "state": "PROCESSING"
        }
    }));
    let http = Capture::with_responses(vec![start, finish]);
    let profile = make_profile(
        "google",
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini_generate_content",
    );
    let service = FileService::new(&http, &profile, None, None, Some("acct-a"));
    let error = service
        .upload_stream(upload(b"pdf"), FilePurpose::ModelInput)
        .await
        .unwrap_err();
    match error {
        FileUploadError::Llm(LlmError::ProviderFileProcessing { file, .. }) => {
            assert_eq!(file.file_id, "files/gemini-pending");
            assert_eq!(file.account_scope.as_deref(), Some("acct-a"));
        }
        other => panic!("expected resumable processing error, got {other}"),
    }
    assert_eq!(http.buffered.lock().unwrap().len(), 1);
    assert_eq!(http.streamed.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn gemini_unsafe_upload_origin_is_rejected_before_body_streaming() {
    let start = HttpResponse {
        status: 200,
        headers: vec![(
            "x-goog-upload-url".into(),
            "https://elsewhere.test/upload".into(),
        )],
        body: Bytes::new(),
    };
    let http = Capture::with_responses(vec![start]);
    let profile = make_profile(
        "google",
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini_generate_content",
    );
    let service = FileService::new(&http, &profile, None, None, Some("acct-a"));
    let polled = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&polled);
    let body = stream::once(async move {
        flag.store(true, Ordering::SeqCst);
        Ok(Bytes::from_static(b"pdf"))
    });
    assert!(matches!(
        service
            .upload_stream(
                UploadFileStream::new("input.pdf", "application/pdf", 3, body),
                FilePurpose::ModelInput
            )
            .await,
        Err(FileUploadError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(!polled.load(Ordering::SeqCst));
    assert!(http.streamed.lock().unwrap().is_empty());
}

#[tokio::test]
async fn xai_multipart_limit_applies_before_reading_or_dispatch_for_both_upload_paths() {
    let http = Capture::default();
    let profile = make_profile("xai", "https://api.x.ai/v1", "open_ai_responses");
    let service = FileService::new(&http, &profile, None, None, Some("account-a"));
    let polled = Arc::new(AtomicBool::new(false));
    let read = polled.clone();
    let source = stream::once(async move {
        read.store(true, Ordering::SeqCst);
        Ok(Bytes::new())
    });
    let error = service
        .upload_stream(
            UploadFileStream::new("large.txt", "text/plain", 50_000_001, source),
            FilePurpose::ModelInput,
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        FileUploadError::Llm(LlmError::RequestTooLarge { .. })
    ));
    assert!(!polled.load(Ordering::SeqCst));
    let error = service
        .upload(
            &UploadFile {
                filename: "large.txt".into(),
                media_type: "text/plain".into(),
                bytes: Bytes::from(vec![b'x'; 50_000_001]),
            },
            FilePurpose::ModelInput,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, LlmError::RequestTooLarge { .. }));
    assert!(http.buffered.lock().unwrap().is_empty());
    assert!(http.streamed.lock().unwrap().is_empty());

    // At the exact boundary preflight succeeds and the source is read. Fail
    // the source deliberately so this boundary test needs no 50 MB payload.
    let read = polled.clone();
    let source = stream::once(async move {
        read.store(true, Ordering::SeqCst);
        Err(LlmError::Transport {
            message: "boundary source interrupted".into(),
        })
    });
    let error = service
        .upload_stream(
            UploadFileStream::new("large.txt", "text/plain", 50_000_000, source),
            FilePurpose::ModelInput,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, FileUploadError::OutcomeUnknown { .. }));
    assert!(polled.load(Ordering::SeqCst));
    assert_eq!(http.streamed.lock().unwrap().len(), 1);
}
