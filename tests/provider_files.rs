use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::files::{
    capabilities, capabilities_for_purpose, FilePurpose, FileService, ModelFileReference,
    UploadFile,
};
use lingxi_llm_client::protocol::{LlmError, ProviderProfile, Secret};
use lingxi_llm_client::{
    ApiKeyAuthenticator, HttpRequest, HttpResponse, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::Mutex;

#[derive(Default)]
struct MockTransport {
    requests: Mutex<Vec<HttpRequest>>,
    responses: Mutex<VecDeque<HttpResponse>>,
}

impl MockTransport {
    fn with_responses(responses: Vec<HttpResponse>) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            responses: Mutex::new(responses.into()),
        }
    }

    fn requests(&self) -> Vec<HttpRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, req: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(req);
        let response =
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| LlmError::Transport {
                    message: "mock has no response".into(),
                })?;
        Ok(StreamResponse {
            status: response.status,
            headers: response.headers,
            body: stream::iter(vec![Ok(response.body)]).boxed(),
        })
    }
}

fn profile(
    provider_id: &str,
    base_url: &str,
    protocol: &str,
    media_types: &[&str],
) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": provider_id,
        "profile_name": format!("{provider_id}-profile"),
        "base_url": base_url,
        "protocol": protocol,
        "auth": "api_key",
        "models": [{
            "display_model": "Test model",
            "request_model": "test-model",
            "billing_model": "test-model",
            "capability_support": {
                "vision": "supported",
                "documents": "supported",
                "tools": "unknown",
                "reasoning": "unknown",
                "signed_reasoning": "unknown",
                "streaming": "supported",
                "structured_output": "unknown"
            },
            "metadata": {"inputModalities": media_types, "attachments": true}
        }]
    }))
    .unwrap()
}

fn response(body: Value) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: vec![("content-type".into(), "application/json".into())],
        body: serde_json::to_vec(&body).unwrap().into(),
    }
}

fn service<'a>(
    http: &'a MockTransport,
    profile: &'a ProviderProfile,
    auth: &'a ApiKeyAuthenticator,
    key: &'a Secret<String>,
    scope: Option<&'a str>,
) -> FileService<'a> {
    FileService::new(http, profile, Some(auth), Some(key), scope)
}

#[tokio::test]
async fn openai_upload_builds_multipart_and_returns_a_profile_scoped_reference() {
    let http = MockTransport::with_responses(vec![
        response(json!({
            "id": "file-abc",
            "filename": "notes.pdf",
            "bytes": 4,
            "purpose": "user_data"
        })),
        response(json!({
            "id": "file-image",
            "filename": "photo.png",
            "bytes": 3,
            "purpose": "user_data"
        })),
    ]);
    let profile = profile(
        "openai",
        "https://api.openai.com/v1",
        "open_ai_responses",
        &["text", "file"],
    );
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("test-key".to_owned());
    let svc = service(&http, &profile, &auth, &key, Some("acct-a"));
    let uploaded = svc
        .upload(
            &UploadFile {
                filename: "notes.pdf".into(),
                media_type: "application/pdf".into(),
                bytes: Bytes::from_static(b"pdf!"),
            },
            FilePurpose::ModelInput,
        )
        .await
        .unwrap();

    assert_eq!(uploaded.file_id, "file-abc");
    assert_eq!(uploaded.profile_name, "openai-profile");
    assert_eq!(uploaded.account_scope.as_deref(), Some("acct-a"));
    assert_eq!(uploaded.model_reference().file_id, "file-abc");
    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url, "https://api.openai.com/v1/files");
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
            && value == "Bearer test-key"));
    let content_type = requests[0]
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .unwrap()
        .1
        .clone();
    assert!(content_type.starts_with("multipart/form-data; boundary="));
    let body = String::from_utf8_lossy(&requests[0].body);
    assert!(body.contains("name=\"purpose\"\r\n\r\nuser_data"));
    assert!(body.contains("filename=\"notes.pdf\""));
    assert!(body.contains("pdf!"));

    svc.upload(
        &UploadFile {
            filename: "photo.png".into(),
            media_type: "image/png".into(),
            bytes: Bytes::from_static(b"png"),
        },
        FilePurpose::ModelInput,
    )
    .await
    .unwrap();
    let requests = http.requests();
    let image_body = String::from_utf8_lossy(&requests[1].body);
    assert!(image_body.contains("name=\"purpose\"\r\n\r\nuser_data"));
}

#[tokio::test]
async fn openrouter_upload_requires_workspace_purpose_and_is_not_model_input() {
    let http = MockTransport::with_responses(vec![response(json!({
        "id": "file-workspace",
        "filename": "notes.pdf",
        "size_bytes": 4,
        "downloadable": false
    }))]);
    let profile = profile(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "open_ai_chat",
        &["text", "file"],
    );
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("openrouter-key".to_owned());
    let svc = service(&http, &profile, &auth, &key, Some("workspace-account"));
    let model_input =
        svc.capabilities_for_purpose("test-model", "application/pdf", FilePurpose::ModelInput);
    assert!(!model_input.upload);
    assert_eq!(model_input.model_input, ModelFileReference::Unsupported);

    let error = svc
        .upload(
            &UploadFile {
                filename: "notes.pdf".into(),
                media_type: "application/pdf".into(),
                bytes: Bytes::from_static(b"pdf!"),
            },
            FilePurpose::ModelInput,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, LlmError::UnsupportedCapability { .. }));
    assert!(http.requests().is_empty());

    let workspace_file = svc
        .upload(
            &UploadFile {
                filename: "notes.pdf".into(),
                media_type: "application/pdf".into(),
                bytes: Bytes::from_static(b"pdf!"),
            },
            FilePurpose::Workspace,
        )
        .await
        .unwrap();
    assert_eq!(workspace_file.file_id, "file-workspace");
    assert_eq!(
        svc.capabilities_for_purpose("test-model", "application/pdf", FilePurpose::Workspace)
            .model_input,
        ModelFileReference::Unsupported
    );
    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url, "https://openrouter.ai/api/v1/files");
    assert!(!String::from_utf8_lossy(&requests[0].body).contains("name=\"purpose\""));
}

#[tokio::test]
async fn upload_rejects_mime_header_injection_before_transport() {
    let http = MockTransport::default();
    let profile = profile(
        "openai",
        "https://api.openai.com/v1",
        "open_ai_responses",
        &["text", "file"],
    );
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("test-key".to_owned());
    let svc = service(&http, &profile, &auth, &key, None);

    for media_type in [
        "image/png\r\nX-Injected: true",
        "image/png; boundary=attacker",
        "image/with space",
        "image/",
    ] {
        let error = svc
            .upload(
                &UploadFile {
                    filename: "photo.png".into(),
                    media_type: media_type.into(),
                    bytes: Bytes::from_static(b"png"),
                },
                FilePurpose::ModelInput,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, LlmError::InvalidRequest { .. }));
    }
    assert!(http.requests().is_empty());
}

#[tokio::test]
async fn gemini_upload_uses_two_steps_and_never_sends_the_key_to_the_upload_url() {
    let http = MockTransport::with_responses(vec![
        HttpResponse {
            status: 200,
            headers: vec![(
                "x-goog-upload-url".into(),
                "https://generativelanguage.googleapis.com/upload/session/abc".into(),
            )],
            body: Bytes::new(),
        },
        response(json!({
            "file": {
                "name": "files/image-1",
                "uri": "https://generativelanguage.googleapis.com/files/image-1",
                "mimeType": "image/png"
            }
        })),
        response(json!({
            "name": "files/image-1",
            "uri": "https://generativelanguage.googleapis.com/files/image-1",
            "mimeType": "image/png"
        })),
    ]);
    let profile = profile(
        "google",
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini_generate_content",
        &["text", "image"],
    );
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("gemini-secret".to_owned());
    let svc = service(&http, &profile, &auth, &key, Some("google-account"));
    let uploaded = svc
        .upload(
            &UploadFile {
                filename: "photo.png".into(),
                media_type: "image/png".into(),
                bytes: Bytes::from_static(b"png"),
            },
            FilePurpose::ModelInput,
        )
        .await
        .unwrap();

    assert_eq!(uploaded.file_id, "files/image-1");
    assert_eq!(
        uploaded.uri.as_deref(),
        Some("https://generativelanguage.googleapis.com/files/image-1")
    );
    assert_eq!(
        uploaded.model_reference().media_type.as_deref(),
        Some("image/png")
    );
    svc.get(&uploaded).await.unwrap();
    let requests = http.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests[0].url,
        "https://generativelanguage.googleapis.com/upload/v1beta/files"
    );
    assert!(requests[0].headers.iter().any(|(name, value)| name
        .eq_ignore_ascii_case("x-goog-api-key")
        && value == "gemini-secret"));
    assert_eq!(
        requests[1].url,
        "https://generativelanguage.googleapis.com/upload/session/abc"
    );
    assert!(!requests[1]
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("x-goog-api-key")
            || name.eq_ignore_ascii_case("authorization")));
    assert_eq!(
        requests[2].url,
        "https://generativelanguage.googleapis.com/v1beta/files/image-1"
    );
}

#[tokio::test]
async fn anthropic_uses_ga_list_cursor_without_the_retired_beta_header() {
    let http = MockTransport::with_responses(vec![
        response(json!({
            "data": [{"id": "file-a", "filename": "a.pdf", "mime_type": "application/pdf"}],
            "next_page": "cursor/one+two"
        })),
        response(json!({
            "data": [{"id": "file-b", "filename": "b.pdf", "mime_type": "application/pdf"}],
            "next_page": null
        })),
    ]);
    let profile = profile(
        "anthropic",
        "https://api.anthropic.com",
        "anthropic_messages",
        &["text", "image", "file"],
    );
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("anthropic-secret".to_owned());
    let svc = service(&http, &profile, &auth, &key, Some("anthropic-account"));
    let first = svc.list(None).await.unwrap();
    assert_eq!(first.next_cursor.as_deref(), Some("cursor/one+two"));
    let second = svc.list(first.next_cursor.as_deref()).await.unwrap();
    assert_eq!(second.files[0].file.file_id, "file-b");
    let requests = http.requests();
    assert_eq!(
        requests[0].url,
        "https://api.anthropic.com/v1/files?limit=1000"
    );
    assert!(requests[1].url.contains("page=cursor%2Fone%2Btwo"));
    for request in requests {
        assert!(!request
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta")));
        assert!(request
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("anthropic-version")));
    }
}

#[tokio::test]
async fn provider_file_scope_mismatch_fails_before_network_access() {
    let http = MockTransport::default();
    let profile = profile(
        "openai",
        "https://api.openai.com/v1",
        "open_ai_responses",
        &["text", "file"],
    );
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("test-key".to_owned());
    let svc = service(&http, &profile, &auth, &key, Some("acct-b"));
    let foreign = lingxi_llm_client::files::ProviderFileRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
            &profile.base_url,
        ),
        account_scope: Some("acct-a".into()),
        protocol: profile.protocol,
        file_id: "file-abc".into(),
        uri: None,
        filename: None,
        media_type: None,
        size_bytes: None,
        expires_at: None,
        downloadable: None,
        purpose: None,
    };

    let error = svc.get(&foreign).await.unwrap_err();
    assert!(matches!(error, LlmError::PermissionDenied { .. }));
    assert!(http.requests().is_empty());
}

#[tokio::test]
async fn file_ids_are_encoded_as_one_path_segment() {
    let http = MockTransport::with_responses(vec![response(json!({
        "id": "opaque-id",
        "filename": "safe.pdf"
    }))]);
    let profile = profile(
        "openai",
        "https://api.openai.com/v1",
        "open_ai_responses",
        &["text", "file"],
    );
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("test-key".to_owned());
    let svc = service(&http, &profile, &auth, &key, None);
    let reference = lingxi_llm_client::files::ProviderFileRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
            &profile.base_url,
        ),
        account_scope: None,
        protocol: profile.protocol,
        file_id: "id /?x".into(),
        uri: None,
        filename: None,
        media_type: None,
        size_bytes: None,
        expires_at: None,
        downloadable: None,
        purpose: None,
    };

    svc.get(&reference).await.unwrap();
    assert_eq!(
        http.requests()[0].url,
        "https://api.openai.com/v1/files/id%20%2F%3Fx"
    );
}

#[tokio::test]
async fn moonshot_file_extraction_is_separate_from_chat_file_references() {
    let http = MockTransport::with_responses(vec![
        response(json!({
            "id": "file-moonshot",
            "filename": "notes.pdf",
            "bytes": 4,
            "status": "ready"
        })),
        HttpResponse {
            status: 200,
            headers: vec![("content-type".into(), "text/plain".into())],
            body: Bytes::from_static(b"extracted words"),
        },
    ]);
    let profile = profile(
        "kimi",
        "https://api.moonshot.cn/v1",
        "open_ai_chat",
        &["text", "file"],
    );
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("moonshot-secret".to_owned());
    let svc = service(&http, &profile, &auth, &key, Some("kimi-account"));
    let file = svc
        .upload(
            &UploadFile {
                filename: "notes.pdf".into(),
                media_type: "application/pdf".into(),
                bytes: Bytes::from_static(b"pdf!"),
            },
            FilePurpose::Extraction,
        )
        .await
        .unwrap();
    assert_eq!(svc.extract_text(&file).await.unwrap(), "extracted words");
    assert_eq!(file.model_reference().provider_id, profile.provider_id);
    assert_eq!(
        capabilities(&profile, "test-model", "application/pdf").model_input,
        ModelFileReference::Unsupported
    );
    let requests = http.requests();
    let upload_body = String::from_utf8_lossy(&requests[0].body);
    assert!(upload_body.contains("name=\"purpose\"\r\n\r\nfile-extract"));
    assert_eq!(
        requests[1].url,
        "https://api.moonshot.cn/v1/files/file-moonshot/content"
    );
}

#[tokio::test]
async fn uploaded_anthropic_file_marked_not_downloadable_is_not_fetched() {
    let http = MockTransport::with_responses(vec![response(json!({
        "id": "file-uploaded",
        "filename": "private.pdf",
        "mime_type": "application/pdf",
        "downloadable": false
    }))]);
    let profile = profile(
        "anthropic",
        "https://api.anthropic.com",
        "anthropic_messages",
        &["text", "image", "file"],
    );
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("anthropic-secret".to_owned());
    let svc = service(&http, &profile, &auth, &key, Some("anthropic-account"));
    let uploaded = lingxi_llm_client::files::ProviderFileRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
            &profile.base_url,
        ),
        account_scope: Some("anthropic-account".into()),
        protocol: profile.protocol,
        file_id: "file-uploaded".into(),
        uri: None,
        filename: Some("private.pdf".into()),
        media_type: Some("application/pdf".into()),
        size_bytes: None,
        expires_at: None,
        downloadable: Some(false),
        purpose: None,
    };

    let error = svc.download(&uploaded).await.unwrap_err();
    assert!(matches!(error, LlmError::UnsupportedCapability { .. }));
    let requests = http.requests();
    assert_eq!(
        requests.len(),
        1,
        "must inspect metadata before content fetch"
    );
    assert!(requests[0].url.ends_with("/v1/files/file-uploaded"));
}

#[test]
fn capability_matrix_keeps_management_and_chat_references_distinct() {
    let openai_chat = profile(
        "openai",
        "https://api.openai.com/v1",
        "open_ai_chat",
        &["text", "file"],
    );
    assert_eq!(
        capabilities(&openai_chat, "test-model", "application/pdf").model_input,
        ModelFileReference::FileId
    );
    assert_eq!(
        capabilities(&openai_chat, "test-model", "image/png").model_input,
        ModelFileReference::Unsupported
    );

    let xai_chat = profile(
        "xai",
        "https://api.x.ai/v1",
        "open_ai_chat",
        &["text", "file"],
    );
    let xai_files = capabilities(&xai_chat, "test-model", "application/pdf");
    assert!(xai_files.upload && xai_files.list && xai_files.delete);
    assert_eq!(xai_files.model_input, ModelFileReference::Unsupported);

    let xai_responses = profile(
        "xai",
        "https://api.x.ai/v1",
        "open_ai_responses",
        &["text", "file"],
    );
    assert_eq!(
        capabilities(&xai_responses, "test-model", "application/pdf").model_input,
        ModelFileReference::FileId
    );

    let kimi = profile(
        "kimi",
        "https://api.moonshot.cn/v1",
        "open_ai_chat",
        &["text", "file"],
    );
    assert_eq!(
        capabilities(&kimi, "test-model", "application/pdf").model_input,
        ModelFileReference::Unsupported
    );
    assert!(
        capabilities_for_purpose(
            &kimi,
            "test-model",
            "application/pdf",
            FilePurpose::Extraction
        )
        .upload
    );

    let kimi_code = profile(
        "kimi",
        "https://api.kimi.com/coding/v1",
        "open_ai_chat",
        &["text", "file"],
    );
    assert!(
        !capabilities_for_purpose(
            &kimi_code,
            "test-model",
            "application/pdf",
            FilePurpose::Extraction
        )
        .upload
    );
}

#[tokio::test]
async fn gemini_list_and_get_preserve_display_name() {
    let file = json!({"name":"files/report","displayName":"report.pdf","mimeType":"application/pdf","sizeBytes":"10"});
    let http =
        MockTransport::with_responses(vec![response(json!({"files":[file]})), response(file)]);
    let profile = profile(
        "google",
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini_generate_content",
        &["text", "file"],
    );
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("test-key".to_owned());
    let svc = service(&http, &profile, &auth, &key, None);
    let page = svc.list(None).await.unwrap();
    assert_eq!(page.files[0].file.filename.as_deref(), Some("report.pdf"));
    let metadata = svc.get(&page.files[0].file).await.unwrap();
    assert_eq!(metadata.file.filename.as_deref(), Some("report.pdf"));
}

#[tokio::test]
async fn gemini_short_processing_timeout_queries_before_waiting() {
    for (timeout_ms, state) in [(100, "ACTIVE"), (1000, "ACTIVE"), (1000, "PROCESSING")] {
        let file = |state| json!({"name":"files/video-1","uri":"https://generativelanguage.googleapis.com/v1beta/files/video-1","state":state});
        let http = MockTransport::with_responses(vec![
            HttpResponse {
                status: 200,
                headers: vec![(
                    "x-goog-upload-url".into(),
                    "https://generativelanguage.googleapis.com/upload/session/one".into(),
                )],
                body: Bytes::new(),
            },
            response(json!({"file":file("PROCESSING")})),
            response(file(state)),
        ]);
        let profile = profile(
            "google",
            "https://generativelanguage.googleapis.com/v1beta",
            "gemini_generate_content",
            &["text", "video"],
        );
        let auth = ApiKeyAuthenticator;
        let key = Secret::new("test-key".to_owned());
        let svc = service(&http, &profile, &auth, &key, Some("scope"))
            .with_gemini_processing_timeout(std::time::Duration::from_millis(timeout_ms));
        let result = svc
            .upload(
                &UploadFile {
                    filename: "clip.mp4".into(),
                    media_type: "video/mp4".into(),
                    bytes: Bytes::from_static(b"mp4"),
                },
                FilePurpose::ModelInput,
            )
            .await;
        if state == "ACTIVE" {
            assert!(result.is_ok(), "{result:?}");
        } else {
            let Err(LlmError::ProviderFileProcessing { file, .. }) = result else {
                panic!("expected a resumable processing error, got {result:?}");
            };
            assert_eq!(file.file_id, "files/video-1");
            assert_eq!(file.profile_name, profile.profile_name);
            assert_eq!(file.account_scope.as_deref(), Some("scope"));
        }
        assert_eq!(
            http.requests().iter().filter(|r| r.method == "GET").count(),
            1
        );
    }
}

#[tokio::test]
async fn gemini_file_lists_accept_omitted_or_null_empty_fields_but_reject_wrong_types() {
    let p = profile(
        "google",
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini_generate_content",
        &["text", "image"],
    );
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("test-key".into());
    for raw in [
        json!({}),
        json!({"files":null}),
        json!({"files":[]}),
        json!({"nextPageToken":"next"}),
    ] {
        let http = MockTransport::with_responses(vec![response(raw.clone())]);
        let page = service(&http, &p, &auth, &key, Some("account"))
            .list(None)
            .await
            .unwrap();
        assert!(page.files.is_empty());
        assert_eq!(
            page.next_cursor.as_deref(),
            raw.get("nextPageToken").and_then(Value::as_str)
        );
    }
    for raw in [
        json!({"files":{}}),
        json!({"files":false}),
        json!({"files":"invalid"}),
        json!({"files":false,"data":[]}),
        Value::Null,
        json!([]),
        json!(false),
        json!("invalid"),
    ] {
        let http = MockTransport::with_responses(vec![response(raw)]);
        assert!(matches!(
            service(&http, &p, &auth, &key, Some("account"))
                .list(None)
                .await,
            Err(LlmError::ProviderInternal { .. })
        ));
    }
    let p = profile(
        "openai",
        "https://api.openai.com/v1",
        "open_ai_responses",
        &["text", "file"],
    );
    let http = MockTransport::with_responses(vec![response(json!({}))]);
    assert!(matches!(
        service(&http, &p, &auth, &key, Some("account"))
            .list(None)
            .await,
        Err(LlmError::ProviderInternal { .. })
    ));
}

#[tokio::test]
async fn gemini_video_processing_keeps_its_own_budget_after_a_short_upload() {
    let file = |state| json!({"name":"files/video-1","uri":"https://generativelanguage.googleapis.com/v1beta/files/video-1","state":state});
    let http = MockTransport::with_responses(vec![
        HttpResponse {
            status: 200,
            headers: vec![(
                "x-goog-upload-url".into(),
                "https://generativelanguage.googleapis.com/upload/session/one".into(),
            )],
            body: Bytes::new(),
        },
        response(json!({"file":file("PROCESSING")})),
        response(file("PROCESSING")),
        response(file("ACTIVE")),
    ]);
    let p = profile(
        "google",
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini_generate_content",
        &["text", "video"],
    );
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("test-key".into());
    let svc = service(&http, &p, &auth, &key, Some("scope"))
        .with_gemini_upload_timeout(std::time::Duration::from_secs(1))
        .with_gemini_processing_timeout(std::time::Duration::from_secs(5));
    let file = svc
        .upload(
            &UploadFile {
                filename: "clip.mp4".into(),
                media_type: "video/mp4".into(),
                bytes: Bytes::from_static(b"mp4"),
            },
            FilePurpose::ModelInput,
        )
        .await
        .unwrap();
    assert_eq!(file.file_id, "files/video-1");
    assert_eq!(
        http.requests().iter().filter(|r| r.method == "GET").count(),
        2
    );
}

#[tokio::test]
async fn gemini_upload_deadline_cancels_pending_authentication() {
    struct PendingAuth;
    #[async_trait]
    impl lingxi_llm_client::Authenticator for PendingAuth {
        async fn apply(
            &self,
            _: &mut HttpRequest,
            _: &ProviderProfile,
            _: Option<&Secret<String>>,
        ) -> Result<(), LlmError> {
            std::future::pending().await
        }
    }
    let http = MockTransport::default();
    let p = profile(
        "google",
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini_generate_content",
        &["text", "video"],
    );
    let auth = PendingAuth;
    let key = Secret::new("test-key".into());
    let svc = FileService::new(&http, &p, Some(&auth), Some(&key), Some("scope"))
        .with_gemini_upload_timeout(std::time::Duration::from_secs(1));
    let upload = UploadFile {
        filename: "clip.mp4".into(),
        media_type: "video/mp4".into(),
        bytes: Bytes::from_static(b"mp4"),
    };
    let result = tokio::time::timeout(
        std::time::Duration::from_millis(1500),
        svc.upload(&upload, FilePurpose::ModelInput),
    )
    .await;
    assert!(
        matches!(result, Ok(Err(LlmError::TransportTimeout { .. }))),
        "{result:?}"
    );
    assert!(http.requests().is_empty());
}
