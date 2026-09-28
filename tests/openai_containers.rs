use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{StreamExt, TryStreamExt};
use lingxi_llm_client::{
    files::UploadFile,
    protocol::{LlmError, Secret},
    providers::openai::containers::{
        OpenAiContainerCreateRequest, OpenAiContainerFileListOptions, OpenAiContainerListOptions,
        OpenAiContainerMemoryLimit, OpenAiContainerOrder, OpenAiContainerScope,
        OpenAiContainersError, OpenAiContainersService,
    },
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

struct Reply {
    status: u16,
    body: Vec<u8>,
    headers: Vec<(String, String)>,
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
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let reply = self.replies.lock().unwrap().pop_front().unwrap()?;
        Ok(StreamResponse {
            status: reply.status,
            headers: reply.headers,
            body: futures::stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
        })
    }

    async fn send_stream(&self, _request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        panic!("OpenAI Containers file upload uses a bounded multipart request body")
    }
}

fn reply(status: u16, value: Value) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body: serde_json::to_vec(&value).unwrap(),
        headers: vec![("content-type".into(), "application/json".into())],
    })
}

fn raw_reply(status: u16, body: Vec<u8>, content_type: &str) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body,
        headers: vec![("content-type".into(), content_type.into())],
    })
}

fn container(id: &str) -> Value {
    json!({
        "id": id,
        "object": "container",
        "name": "analysis",
        "status": "running",
        "created_at": 1_747_854_708_u64,
        "last_active_at": 1_747_854_800_u64,
        "memory_limit": "4g",
        "expires_after": {"anchor": "last_active_at", "minutes": 20}
    })
}

fn container_file(id: &str, container_id: &str) -> Value {
    json!({
        "id": id,
        "object": "container.file",
        "container_id": container_id,
        "bytes": 5,
        "created_at": 1_747_854_900_u64,
        "path": "/mnt/data/sample.txt",
        "source": "user"
    })
}

fn source_file(id: &str, expires_at: Value) -> Value {
    json!({"id": id, "object": "file", "expires_at": expires_at})
}

fn service<'a>(http: &'a MockTransport, account: &str) -> OpenAiContainersService<'a> {
    OpenAiContainersService::new(
        http,
        OpenAiContainerScope::new("openai-production", account).unwrap(),
    )
    .unwrap()
}

fn credential() -> Secret<String> {
    Secret::new("openai-test-key".to_owned())
}

#[tokio::test]
async fn create_encodes_documented_options_and_returns_a_scoped_reference() {
    let mock = MockTransport::new([
        reply(200, source_file("file-existing", Value::Null)),
        reply(200, container("cntr_123")),
    ]);
    let service = service(&mock, "account-a");
    let request = OpenAiContainerCreateRequest::new("analysis")
        .unwrap()
        .with_memory_limit(OpenAiContainerMemoryLimit::G4)
        .with_expiration_minutes(30)
        .unwrap()
        .with_file_ids(["file-existing".to_owned()])
        .unwrap();
    let created = service
        .create_container(&request, &credential())
        .await
        .unwrap();

    assert_eq!(created.reference.container_id(), "cntr_123");
    assert_eq!(created.status.as_deref(), Some("running"));
    assert_eq!(created.memory_limit.as_deref(), Some("4g"));
    let sent = mock.requests();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0].method, "GET");
    assert_eq!(sent[0].url, "https://api.openai.com/v1/files/file-existing");
    assert_eq!(sent[1].method, "POST");
    assert_eq!(sent[1].url, "https://api.openai.com/v1/containers");
    assert_eq!(
        sent[1]
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            .map(|(_, value)| value.as_str()),
        Some("Bearer openai-test-key")
    );
    let body: Value = serde_json::from_slice(&sent[1].body).unwrap();
    assert_eq!(body["name"], "analysis");
    assert_eq!(body["memory_limit"], "4g");
    assert_eq!(
        body["expires_after"],
        json!({"anchor": "last_active_at", "minutes": 30})
    );
    assert_eq!(body["file_ids"], json!(["file-existing"]));
}

#[tokio::test]
async fn create_rejects_source_file_that_expires_before_copy_preflight() {
    let expires_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + 30;
    let mock = MockTransport::new([reply(
        200,
        source_file("file-short-lived", json!(expires_at)),
    )]);
    let request = OpenAiContainerCreateRequest::new("analysis")
        .unwrap()
        .with_expiration_minutes(30)
        .unwrap()
        .with_file_ids(["file-short-lived"])
        .unwrap();

    let error = service(&mock, "account-a")
        .create_container(&request, &credential())
        .await
        .unwrap_err();
    assert!(matches!(error, OpenAiContainersError::InvalidInput(_)));
    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "GET");
}

#[tokio::test]
async fn attach_rejects_expiring_file_before_container_file_creation() {
    let expires_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + 30;
    let mock = MockTransport::new([
        reply(200, container("cntr_123")),
        reply(200, source_file("file-short-lived", json!(expires_at))),
    ]);
    let service = service(&mock, "account-a");
    let created = service
        .create_container(
            &OpenAiContainerCreateRequest::new("analysis").unwrap(),
            &credential(),
        )
        .await
        .unwrap();
    let error = service
        .attach_file(&created.reference, "file-short-lived", &credential())
        .await
        .unwrap_err();

    assert!(matches!(error, OpenAiContainersError::InvalidInput(_)));
    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].method, "GET");
    assert_eq!(
        requests[1].url,
        "https://api.openai.com/v1/files/file-short-lived"
    );
}

#[tokio::test]
async fn container_list_encodes_filters_and_returns_one_page_cursor() {
    let mock = MockTransport::new([reply(
        200,
        json!({
            "object": "list",
            "data": [container("cntr_123")],
            "first_id": "cntr_123",
            "last_id": "cntr_123",
            "has_more": true
        }),
    )]);
    let service = service(&mock, "account-a");
    let options = OpenAiContainerListOptions::new()
        .with_limit(3)
        .unwrap()
        .with_after("cntr/after")
        .unwrap_err();
    assert!(matches!(options, OpenAiContainersError::InvalidInput(_)));

    let options = OpenAiContainerListOptions::new()
        .with_limit(3)
        .unwrap()
        .with_after("cntr_after")
        .unwrap()
        .with_name("analysis / A")
        .unwrap()
        .with_order(OpenAiContainerOrder::Asc);
    let refreshed_credential = Secret::new("openai-refreshed-key".to_owned());
    let page = service
        .list_containers(&options, &refreshed_credential)
        .await
        .unwrap();
    assert_eq!(page.containers.len(), 1);
    assert!(page.has_more);
    assert_eq!(page.next_cursor.as_deref(), Some("cntr_123"));
    let url = &mock.requests()[0].url;
    assert!(url.starts_with("https://api.openai.com/v1/containers?"));
    assert!(url.contains("limit=3"));
    assert!(url.contains("after=cntr_after"));
    assert!(url.contains("order=asc"));
    assert!(url.contains("name=analysis+%2F+A"));
    assert_eq!(
        mock.requests()[0]
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            .map(|(_, value)| value.as_str()),
        Some("Bearer openai-refreshed-key")
    );
}

#[tokio::test]
async fn container_id_from_responses_can_be_bound_and_foreign_scope_is_rejected() {
    let seed = MockTransport::new([reply(200, container("cntr_auto"))]);
    let reference = service(&seed, "account-a")
        .create_container(
            &OpenAiContainerCreateRequest::new("analysis").unwrap(),
            &credential(),
        )
        .await
        .unwrap()
        .reference;
    let foreign = MockTransport::new([]);
    let error = service(&foreign, "account-b")
        .get_container(&reference, &credential())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        OpenAiContainersError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert!(foreign.requests().is_empty());

    let scope = OpenAiContainerScope::new("openai-production", "account-a").unwrap();
    let rebound = lingxi_llm_client::providers::openai::containers::OpenAiContainerRef::from_id(
        &scope,
        "cntr_from_response",
    )
    .unwrap();
    let retrieve = MockTransport::new([reply(200, container("cntr_from_response"))]);
    let found = service(&retrieve, "account-a")
        .get_container(&rebound, &credential())
        .await
        .unwrap();
    assert_eq!(found.reference.container_id(), "cntr_from_response");
}

#[tokio::test]
async fn file_upload_attach_list_get_download_and_delete_use_documented_routes() {
    let mock = MockTransport::new([
        reply(200, container("cntr_123")),
        reply(200, container_file("cfile_upload", "cntr_123")),
        reply(200, source_file("file-source-123", Value::Null)),
        reply(200, container_file("cfile_attached", "cntr_123")),
        reply(
            200,
            json!({
                "object": "list",
                "data": [
                    container_file("cfile_upload", "cntr_123"),
                    container_file("cfile_attached", "cntr_123")
                ],
                "first_id": "cfile_upload",
                "last_id": "cfile_attached",
                "has_more": false
            }),
        ),
        reply(200, container_file("cfile_upload", "cntr_123")),
        raw_reply(
            200,
            b"downloaded bytes".to_vec(),
            "text/plain; charset=utf-8",
        ),
        reply(
            200,
            json!({"id": "cfile_upload", "object": "container.file.deleted", "deleted": true}),
        ),
        reply(
            200,
            json!({"id": "cntr_123", "object": "container.deleted", "deleted": true}),
        ),
    ]);
    let service = service(&mock, "account-a");
    let container = service
        .create_container(
            &OpenAiContainerCreateRequest::new("analysis").unwrap(),
            &credential(),
        )
        .await
        .unwrap();
    let input = UploadFile {
        filename: "sample.txt".into(),
        media_type: "text/plain".into(),
        bytes: Bytes::from_static(b"hello"),
    };
    let uploaded = service
        .upload_file(&container.reference, &input, &credential())
        .await
        .unwrap();
    let attached = service
        .attach_file(&container.reference, "file-source-123", &credential())
        .await
        .unwrap();
    let page = service
        .list_files(
            &container.reference,
            &OpenAiContainerFileListOptions::new()
                .with_limit(2)
                .unwrap()
                .with_after("cfile_upload")
                .unwrap()
                .with_order(OpenAiContainerOrder::Desc),
            &credential(),
        )
        .await
        .unwrap();
    assert_eq!(page.files.len(), 2);
    assert!(!page.has_more);
    assert_eq!(
        page.files[0].reference.container().container_id(),
        "cntr_123"
    );
    assert_eq!(page.files[1].reference.file_id(), "cfile_attached");

    let found = service
        .get_file(&uploaded.reference, &credential())
        .await
        .unwrap();
    assert_eq!(found.path.as_deref(), Some("/mnt/data/sample.txt"));
    let content = service
        .download_file(&found.reference, &credential())
        .await
        .unwrap();
    assert_eq!(content.media_type.as_deref(), Some("text/plain"));
    let bytes = content
        .try_fold(BytesMut::new(), |mut collected, chunk| async move {
            collected.extend_from_slice(&chunk);
            Ok(collected)
        })
        .await
        .unwrap()
        .freeze();
    assert_eq!(bytes, Bytes::from_static(b"downloaded bytes"));
    assert!(
        service
            .delete_file(&uploaded.reference, &credential())
            .await
            .unwrap()
            .deleted
    );
    assert!(
        service
            .delete_container(&container.reference, &credential())
            .await
            .unwrap()
            .deleted
    );
    assert_eq!(attached.reference.file_id(), "cfile_attached");

    let requests = mock.requests();
    assert_eq!(requests.len(), 9);
    assert_eq!(requests[1].method, "POST");
    assert!(requests[1].url.ends_with("/containers/cntr_123/files"));
    assert!(String::from_utf8_lossy(&requests[1].body)
        .contains("name=\"file\"; filename=\"sample.txt\""));
    assert!(String::from_utf8_lossy(&requests[1].body).contains("hello"));
    assert_eq!(requests[2].method, "GET");
    assert_eq!(
        requests[2].url,
        "https://api.openai.com/v1/files/file-source-123"
    );
    assert_eq!(requests[3].method, "POST");
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[3].body).unwrap(),
        json!({"file_id": "file-source-123"})
    );
    assert!(requests[4].url.contains("limit=2"));
    assert!(requests[4].url.contains("after=cfile_upload"));
    assert!(requests[4].url.contains("order=desc"));
    assert_eq!(
        requests[6].url,
        "https://api.openai.com/v1/containers/cntr_123/files/cfile_upload/content"
    );
    assert_eq!(requests[7].method, "DELETE");
    assert_eq!(requests[8].method, "DELETE");
}

#[tokio::test]
async fn uncertain_container_create_is_not_retried() {
    let mock = MockTransport::new([Err(LlmError::Transport {
        message: "connection closed after dispatch".into(),
    })]);
    let error = service(&mock, "account-a")
        .create_container(
            &OpenAiContainerCreateRequest::new("analysis").unwrap(),
            &credential(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        OpenAiContainersError::OutcomeUnknown {
            operation: "create_container",
            container: None,
            ..
        }
    ));
    assert_eq!(mock.requests().len(), 1);
}

#[tokio::test]
async fn foreign_container_file_ref_is_rejected_before_network_access() {
    let seed = MockTransport::new([
        reply(200, container("cntr_123")),
        reply(200, source_file("file-source", Value::Null)),
        reply(200, container_file("cfile_123", "cntr_123")),
    ]);
    let service_a = service(&seed, "account-a");
    let container = service_a
        .create_container(
            &OpenAiContainerCreateRequest::new("analysis").unwrap(),
            &credential(),
        )
        .await
        .unwrap();
    let file = service_a
        .attach_file(&container.reference, "file-source", &credential())
        .await
        .unwrap();

    let foreign = MockTransport::new([]);
    let error = service(&foreign, "account-b")
        .delete_file(&file.reference, &credential())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        OpenAiContainersError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert!(foreign.requests().is_empty());
}
