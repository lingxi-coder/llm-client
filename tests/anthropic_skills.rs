use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::providers::anthropic::types::AnthropicSkillScope;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::anthropic::skills::{
        AnthropicSkillFile, AnthropicSkillListOptions, AnthropicSkillResourceRef,
        AnthropicSkillSourceFilter, AnthropicSkillVersionListOptions, AnthropicSkillsError,
        AnthropicSkillsService,
    },
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    future,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    task::Poll,
    time::Duration,
};

#[derive(Clone)]
struct Reply {
    status: u16,
    body: Bytes,
}

#[derive(Debug, Clone)]
struct SeenRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Bytes,
    content_length: Option<u64>,
}

struct MockTransport {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<SeenRequest>>,
    upload_after_error_ended: AtomicBool,
    later_file_polled: Arc<AtomicBool>,
    send_count: AtomicUsize,
    hang: bool,
    pending_response_body: bool,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
            upload_after_error_ended: AtomicBool::new(false),
            later_file_polled: Arc::new(AtomicBool::new(false)),
            send_count: AtomicUsize::new(0),
            hang: false,
            pending_response_body: false,
        }
    }

    fn hanging() -> Self {
        Self {
            replies: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
            upload_after_error_ended: AtomicBool::new(false),
            later_file_polled: Arc::new(AtomicBool::new(false)),
            send_count: AtomicUsize::new(0),
            hang: true,
            pending_response_body: false,
        }
    }

    fn pending_body() -> Self {
        Self {
            replies: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
            upload_after_error_ended: AtomicBool::new(false),
            later_file_polled: Arc::new(AtomicBool::new(false)),
            send_count: AtomicUsize::new(0),
            hang: false,
            pending_response_body: true,
        }
    }

    fn response_with_pending_body(&self) -> StreamResponse {
        StreamResponse {
            status: 200,
            headers: vec![("request-id".into(), "req-skills-test".into())],
            body: stream::pending().boxed(),
        }
    }

    fn take_reply(&self) -> Reply {
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("mock reply")
    }

    fn response(&self, reply: Reply) -> StreamResponse {
        StreamResponse {
            status: reply.status,
            headers: vec![("request-id".into(), "req-skills-test".into())],
            body: stream::once(async move { Ok(reply.body) }).boxed(),
        }
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.send_count.fetch_add(1, Ordering::SeqCst);
        self.requests.lock().unwrap().push(SeenRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body,
            content_length: None,
        });
        if self.hang {
            return future::pending().await;
        }
        if self.pending_response_body {
            return Ok(self.response_with_pending_body());
        }
        Ok(self.response(self.take_reply()))
    }

    async fn send_stream(
        &self,
        mut request: HttpStreamRequest,
    ) -> Result<StreamResponse, LlmError> {
        self.send_count.fetch_add(1, Ordering::SeqCst);
        if self.hang {
            return future::pending().await;
        }
        let mut bytes = Vec::new();
        while let Some(item) = request.body.next().await {
            match item {
                Ok(chunk) => bytes.extend_from_slice(&chunk),
                Err(error) => {
                    // Poll again deliberately: a failed body must be fused so
                    // this transport cannot observe later parts or a close boundary.
                    self.upload_after_error_ended
                        .store(request.body.next().await.is_none(), Ordering::SeqCst);
                    self.requests.lock().unwrap().push(SeenRequest {
                        method: request.method,
                        url: request.url,
                        headers: request.headers,
                        body: Bytes::from(bytes),
                        content_length: Some(request.content_length),
                    });
                    return Err(error);
                }
            }
        }
        self.requests.lock().unwrap().push(SeenRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: Bytes::from(bytes),
            content_length: Some(request.content_length),
        });
        if self.pending_response_body {
            return Ok(self.response_with_pending_body());
        }
        Ok(self.response(self.take_reply()))
    }
}

fn reply_json(status: u16, value: Value) -> Reply {
    Reply {
        status,
        body: Bytes::from(serde_json::to_vec(&value).unwrap()),
    }
}

fn scope() -> AnthropicSkillScope {
    AnthropicSkillScope::new(
        "anthropic-prod",
        "https://api.anthropic.com",
        "account-workspace",
    )
    .unwrap()
    .with_workspace_id("wrkspc_skills_a")
    .unwrap()
}

fn service(transport: &MockTransport) -> AnthropicSkillsService<'_> {
    AnthropicSkillsService::new(transport, scope()).unwrap()
}

fn skill(id: &str, source: &str) -> Value {
    json!({
        "id": id,
        "created_at": "2026-01-02T03:04:05Z",
        "display_name": "Quarterly report",
        "latest_version_id": "skver_latest",
        "source": {"type": source, "future_source_metadata": {"retained": true}},
        "type": "skill",
        "updated_at": "2026-01-03T03:04:05Z",
        "future_skill_metadata": {"kept": true}
    })
}

fn version(skill_id: &str, id: &str) -> Value {
    json!({
        "id": id,
        "skill_id": skill_id,
        "type": "skill_version",
        "created_at": "2026-01-03T03:04:05Z",
        "description": "Does useful work",
        "name": "quarterly-report",
        "future_version_metadata": true
    })
}

fn skill_file(path: &str, bytes: &'static [u8]) -> AnthropicSkillFile {
    AnthropicSkillFile::from_bytes(path, Bytes::from_static(bytes))
}

#[tokio::test]
async fn create_streams_official_multipart_and_returns_scoped_custom_reference() {
    let transport = MockTransport::new([reply_json(200, skill("skill_alpha", "custom"))]);
    let service = service(&transport);

    let created = service
        .create(
            vec![
                skill_file(
                    "quarterly-report/SKILL.md",
                    b"---\nname: quarterly-report\n---",
                ),
                skill_file("quarterly-report/guide.md", b"Guide"),
            ],
            Some("Quarterly report"),
            &request_options(),
        )
        .await
        .unwrap();

    assert_eq!(created.id, "skill_alpha");
    assert_eq!(created.source_type, "custom");
    assert_eq!(
        created.messages_reference().unwrap().scope(),
        Some(&scope())
    );
    assert!(created.native["future_skill_metadata"]["kept"]
        .as_bool()
        .unwrap());
    let seen = transport.requests.lock().unwrap();
    let request = &seen[0];
    assert_eq!(request.method, "POST");
    assert_eq!(request.url, "https://api.anthropic.com/v1/skills");
    assert!(request
        .headers
        .iter()
        .any(|(name, value)| name.eq_ignore_ascii_case("x-api-key") && value == "sk-test"));
    assert!(request.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("anthropic-version") && value == "2023-06-01"
    }));
    assert!(request.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("anthropic-workspace-id") && value == "wrkspc_skills_a"
    }));
    let content_type = request
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .unwrap()
        .1
        .clone();
    assert!(content_type.starts_with("multipart/form-data; boundary="));
    let body = String::from_utf8_lossy(&request.body);
    assert!(body.contains("name=\"display_name\""));
    assert!(body.contains("Quarterly report"));
    assert!(body.contains("name=\"files[]\"; filename=\"quarterly-report/SKILL.md\""));
    assert!(body.contains("name=\"files[]\"; filename=\"quarterly-report/guide.md\""));
    assert!(request
        .content_length
        .is_some_and(|length| length == request.body.len() as u64));
}

#[tokio::test]
async fn list_preserves_plugin_sources_and_exposes_only_documented_messages_projection() {
    let transport = MockTransport::new([reply_json(
        200,
        json!({"data":[skill("skill_custom", "custom"), skill("skill_plugin", "plugin"), skill("skill_example", "anthropic_example")], "next_page":"cursor-two"}),
    )]);
    let service = service(&transport);
    let page = service
        .list(
            &AnthropicSkillListOptions {
                source: Some(AnthropicSkillSourceFilter::Custom),
                ..Default::default()
            },
            &request_options(),
        )
        .await
        .unwrap();

    assert_eq!(page.next_page.as_deref(), Some("cursor-two"));
    assert_eq!(
        page.skills[0].messages_reference().unwrap().skill_id(),
        "skill_custom"
    );
    assert!(page.skills[1].messages_reference().is_none());
    assert_eq!(page.skills[1].reference.source_type(), "plugin");
    assert!(
        page.skills[1].native["source"]["future_source_metadata"]["retained"]
            .as_bool()
            .unwrap()
    );
    let seen = transport.requests.lock().unwrap();
    assert!(seen[0].url.contains("limit=20"));
    assert!(seen[0].url.contains("source=custom"));
}

#[tokio::test]
async fn get_and_version_lifecycle_use_scoped_resources_and_preserve_native_fields() {
    let transport = MockTransport::new([
        reply_json(200, skill("skill_plugin", "plugin")),
        reply_json(
            200,
            json!({"data":[version("skill_plugin", "skver_one")], "next_page":null}),
        ),
        reply_json(200, version("skill_plugin", "skver_one")),
    ]);
    let service = service(&transport);
    let plugin = AnthropicSkillResourceRef::new("skill_plugin", "plugin", scope()).unwrap();

    let fetched = service.get(&plugin, &request_options()).await.unwrap();
    assert!(fetched.messages_reference().is_none());
    let page = service
        .list_versions(
            &plugin,
            &AnthropicSkillVersionListOptions::default(),
            &request_options(),
        )
        .await
        .unwrap();
    assert_eq!(page.versions[0].id, "skver_one");
    assert!(page.versions[0].native["future_version_metadata"]
        .as_bool()
        .unwrap());
    assert!(page.versions[0].pinned_skill().is_none());
    let got = service
        .get_version(&plugin, "skver_one", &request_options())
        .await
        .unwrap();
    assert_eq!(got.skill_id, "skill_plugin");
    assert!(got.native["future_version_metadata"].as_bool().unwrap());

    let seen = transport.requests.lock().unwrap();
    assert_eq!(seen.len(), 3);
    assert_eq!(
        seen[0].url,
        "https://api.anthropic.com/v1/skills/skill_plugin"
    );
    assert!(seen[1]
        .url
        .ends_with("/v1/skills/skill_plugin/versions?limit=20"));
    assert!(seen[2]
        .url
        .ends_with("/v1/skills/skill_plugin/versions/skver_one"));
}

#[tokio::test]
async fn create_and_delete_version_preserve_scoped_reference_and_check_response_ids() {
    let transport = MockTransport::new([
        reply_json(200, version("skill_alpha", "skver_two")),
        reply_json(
            200,
            json!({"id":"skver_two", "type":"skill_version_deleted"}),
        ),
    ]);
    let service = service(&transport);
    let custom = AnthropicSkillResourceRef::new("skill_alpha", "custom", scope()).unwrap();
    let created = service
        .create_version(
            &custom,
            vec![skill_file("quarterly-report/SKILL.md", b"contents")],
            &request_options(),
        )
        .await
        .unwrap();
    assert_eq!(created.pinned_skill().unwrap().version(), Some("skver_two"));
    let deleted = service
        .delete_version(&custom, "skver_two", &request_options())
        .await
        .unwrap();
    assert_eq!(deleted.id, "skver_two");
    let seen = transport.requests.lock().unwrap();
    assert_eq!(seen[0].method, "POST");
    assert!(seen[0].url.ends_with("/v1/skills/skill_alpha/versions"));
    assert_eq!(seen[1].method, "DELETE");
}

#[tokio::test]
async fn delete_custom_skill_validates_provider_acknowledgement() {
    let transport = MockTransport::new([reply_json(
        200,
        json!({"id":"skill_alpha", "type":"skill_deleted"}),
    )]);
    let service = service(&transport);
    let custom = AnthropicSkillResourceRef::new("skill_alpha", "custom", scope()).unwrap();
    let deleted = service.delete(&custom, &request_options()).await.unwrap();
    assert_eq!(deleted.id, "skill_alpha");
    assert_eq!(transport.requests.lock().unwrap()[0].method, "DELETE");
}

#[tokio::test]
async fn workspace_mismatch_and_non_custom_mutations_fail_before_dispatch() {
    let transport = MockTransport::new([]);
    let service = service(&transport);
    let other_workspace = AnthropicSkillResourceRef::new(
        "skill_alpha",
        "custom",
        AnthropicSkillScope::new(
            "anthropic-prod",
            "https://api.anthropic.com",
            "account-workspace",
        )
        .unwrap()
        .with_workspace_id("wrkspc_skills_b")
        .unwrap(),
    )
    .unwrap();
    let mismatch = service
        .get(&other_workspace, &request_options())
        .await
        .unwrap_err();
    assert!(matches!(
        mismatch,
        AnthropicSkillsError::Llm(LlmError::PermissionDenied { .. })
    ));
    let plugin = AnthropicSkillResourceRef::new("skill_plugin", "plugin", scope()).unwrap();
    let readonly = service
        .delete(&plugin, &request_options())
        .await
        .unwrap_err();
    assert!(matches!(readonly, AnthropicSkillsError::InvalidInput(_)));
    assert_eq!(transport.send_count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn malformed_upload_layout_and_page_options_fail_before_dispatch() {
    let transport = MockTransport::new([]);
    let service = service(&transport);
    assert!(service
        .create(vec![skill_file("SKILL.md", b"x")], None, &request_options())
        .await
        .is_err());
    assert!(service
        .create(
            vec![
                skill_file("one/SKILL.md", b"x"),
                skill_file("two/guide.md", b"y"),
            ],
            None,
            &request_options()
        )
        .await
        .is_err());
    assert!(service
        .create(
            vec![skill_file("one/SKILL.md", b"x")],
            Some("bad\nname"),
            &request_options()
        )
        .await
        .is_err());
    assert!(service
        .list(
            &AnthropicSkillListOptions {
                limit: Some(1001),
                ..Default::default()
            },
            &request_options()
        )
        .await
        .is_err());
    assert_eq!(transport.send_count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn multipart_source_error_is_uncertain_and_fused_before_later_file_or_boundary() {
    let transport = MockTransport::new([]);
    let later_file_polled = Arc::clone(&transport.later_file_polled);
    let later = stream::poll_fn(move |_cx| {
        later_file_polled.store(true, Ordering::SeqCst);
        Poll::Ready(Some(Ok(Bytes::from_static(b"must-not-send"))))
    });
    let first = stream::iter(vec![
        Ok(Bytes::from_static(b"x")),
        Err(LlmError::InvalidRequest {
            message: "caller stream failed".into(),
        }),
    ]);
    let service = service(&transport);

    let error = service
        .create(
            vec![
                AnthropicSkillFile::new("skill/SKILL.md", 1, first),
                AnthropicSkillFile::new("skill/later.md", 13, later),
            ],
            None,
            &request_options(),
        )
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        AnthropicSkillsError::OutcomeUnknown {
            operation: "create",
            ..
        }
    ));
    assert!(transport.upload_after_error_ended.load(Ordering::SeqCst));
    assert!(!transport.later_file_polled.load(Ordering::SeqCst));
    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let sent = String::from_utf8_lossy(&requests[0].body);
    assert!(sent.contains("filename=\"skill/SKILL.md\""));
    assert!(!sent.contains("filename=\"skill/later.md\""));
    assert!(!sent.contains("must-not-send"));
    let content_type = requests[0]
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .unwrap()
        .1
        .clone();
    let boundary = content_type.split("boundary=").nth(1).unwrap();
    assert!(!sent.contains(&format!("--{boundary}--")));
}

#[tokio::test]
async fn mutation_deadline_returns_uncertain_outcome_and_never_retries() {
    let transport = MockTransport::hanging();
    let service = AnthropicSkillsService::new(&transport, scope())
        .unwrap()
        .with_timeout(Duration::from_millis(2))
        .unwrap();

    let error = service
        .create(
            vec![skill_file("skill/SKILL.md", b"contents")],
            None,
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, AnthropicSkillsError::OutcomeUnknown { .. }));
    assert_eq!(transport.send_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn mutation_deadline_also_covers_response_body_reads() {
    let transport = MockTransport::pending_body();
    let service = AnthropicSkillsService::new(&transport, scope())
        .unwrap()
        .with_timeout(Duration::from_millis(2))
        .unwrap();

    let error = service
        .create(
            vec![skill_file("skill/SKILL.md", b"contents")],
            None,
            &request_options(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, AnthropicSkillsError::OutcomeUnknown { .. }));
    assert_eq!(transport.send_count.load(Ordering::SeqCst), 1);
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn malformed_empty_duplicate_and_stalled_pages_are_rejected() {
    let missing_data = MockTransport::new([reply_json(200, json!({"next_page":null}))]);
    assert!(service(&missing_data)
        .list(&AnthropicSkillListOptions::default(), &request_options())
        .await
        .is_err());

    let duplicate = MockTransport::new([reply_json(
        200,
        json!({"data":[skill("same", "custom"), skill("same", "plugin")], "next_page":null}),
    )]);
    assert!(service(&duplicate)
        .list(&AnthropicSkillListOptions::default(), &request_options())
        .await
        .is_err());

    let stalled = MockTransport::new([reply_json(
        200,
        json!({"data":[], "next_page":"cursor-one"}),
    )]);
    assert!(service(&stalled)
        .list(
            &AnthropicSkillListOptions {
                page: Some("cursor-one".into()),
                ..Default::default()
            },
            &request_options()
        )
        .await
        .is_err());
}

#[tokio::test]
async fn version_retrieval_rejects_a_different_returned_id() {
    let transport = MockTransport::new([reply_json(200, version("skill_alpha", "skver_other"))]);
    let service = service(&transport);
    let custom = AnthropicSkillResourceRef::new("skill_alpha", "custom", scope()).unwrap();
    let error = service
        .get_version(&custom, "skver_requested", &request_options())
        .await
        .unwrap_err();
    assert!(matches!(error, AnthropicSkillsError::InvalidResponse(_)));
}

#[tokio::test]
async fn content_download_streams_zip_bytes_without_beta_header_or_local_file() {
    let transport = MockTransport::new([Reply {
        status: 200,
        body: Bytes::from_static(b"PK\x03\x04zip"),
    }]);
    let service = service(&transport);
    let custom = AnthropicSkillResourceRef::new("skill_alpha", "custom", scope()).unwrap();
    let mut stream = service
        .download_version_content(&custom, "skver_one", &request_options())
        .await
        .unwrap();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        bytes.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(bytes, b"PK\x03\x04zip");
    let requests = transport.requests.lock().unwrap();
    assert_eq!(
        requests[0].url,
        "https://api.anthropic.com/v1/skills/skill_alpha/versions/skver_one/content"
    );
    assert!(!requests[0]
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta")));
}

#[tokio::test]
async fn downloaded_content_deadline_ends_the_stream_after_successful_headers() {
    let transport = MockTransport::pending_body();
    let service = service(&transport)
        .with_timeout(Duration::from_millis(2))
        .unwrap();
    let reference = AnthropicSkillResourceRef::new("skill_alpha", "custom", scope()).unwrap();
    let mut content = service
        .download_version_content(&reference, "skver_one", &request_options())
        .await
        .unwrap();
    assert!(matches!(
        content.next().await,
        Some(Err(AnthropicSkillsError::Llm(
            LlmError::TransportTimeout { .. }
        )))
    ));
    assert!(content.next().await.is_none());
    assert_eq!(transport.send_count.load(Ordering::SeqCst), 1);
}

fn request_options() -> lingxi_llm_client::RequestOptions {
    lingxi_llm_client::RequestOptions {
        credential: Some(Secret::new("sk-test".into())),
        ..Default::default()
    }
}

#[tokio::test]
async fn credentials_are_required_per_operation_and_can_rotate() {
    let mock = MockTransport::new([
        reply_json(401, json!({"error":"rejected"})),
        reply_json(401, json!({"error":"rejected"})),
    ]);
    let service = service(&mock);
    let query = AnthropicSkillListOptions::default();
    for credential in [
        None,
        Some(Secret::new(" ".into())),
        Some(Secret::new("key\nvalue".into())),
    ] {
        let options = lingxi_llm_client::RequestOptions {
            credential,
            ..Default::default()
        };
        assert!(service.list(&query, &options).await.is_err());
    }
    assert!(mock.requests.lock().unwrap().is_empty());
    for key in ["first-key", "rotated-key"] {
        let options = lingxi_llm_client::RequestOptions {
            credential: Some(Secret::new(key.into())),
            ..Default::default()
        };
        assert!(service.list(&query, &options).await.is_err());
    }
    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    for (request, key) in requests.iter().zip(["first-key", "rotated-key"]) {
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("x-api-key") && value == key));
    }
}
