use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    files::{provider_file_endpoint_fingerprint, ProviderFileRef},
    protocol::*,
    providers::google::file_search::*,
    transport::{HttpRequest, StreamResponse, Transport},
    *,
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

struct Reply {
    method: &'static str,
    url: &'static str,
    body: Value,
    headers: Vec<(String, String)>,
}

struct Mock {
    replies: Mutex<VecDeque<Reply>>,
    sent: Mutex<Vec<HttpRequest>>,
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
        self.sent.lock().unwrap().push(request);
        let bytes = serde_json::to_vec(&reply.body).unwrap();
        let mut headers = vec![("x-goog-request-id".into(), "req-file-search".into())];
        headers.extend(reply.headers);
        Ok(StreamResponse {
            status: 200,
            headers,
            body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
        })
    }
}

const BASE: &str = "https://generativelanguage.googleapis.com/v1beta";
const STORE_COLLECTION: &str = "https://generativelanguage.googleapis.com/v1beta/fileSearchStores";

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"google",
        "profile_name":"gemini",
        "base_url":BASE,
        "protocol":"gemini_generate_content",
        "auth":"api_key",
        "models":[],
        "gemini_file_search":{"mode":"enabled","value":{
            "endpoint":STORE_COLLECTION,
            "auth":{"type":"api_key","header":"x-goog-api-key"}
        }}
    }))
    .unwrap()
}

fn setup(replies: Vec<Reply>) -> (LlmClient, Arc<Mock>) {
    let mock = Arc::new(Mock {
        replies: Mutex::new(replies.into()),
        sent: Mutex::new(Vec::new()),
    });
    let mut profile = profile();
    profile.regions = vec![Region::International];
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();
    (client, mock)
}

fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("gemini-key".into())),
        account_scope: Some("account-a".into()),
        file_account_scope: Some("file-account-a".into()),
        ..Default::default()
    }
}

fn store_ref() -> GeminiFileSearchStoreRef {
    GeminiFileSearchStoreRef {
        provider_id: "google".into(),
        profile_name: "gemini".into(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(STORE_COLLECTION),
        account_scope: "account-a".into(),
        store_id: "store-alpha".into(),
    }
}

fn file() -> ProviderFileRef {
    ProviderFileRef {
        provider_id: "google".into(),
        profile_name: "gemini".into(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(BASE),
        account_scope: Some("file-account-a".into()),
        protocol: ProtocolFamily::GeminiGenerateContent,
        file_id: "files/file-alpha".into(),
        uri: Some("https://generativelanguage.googleapis.com/v1beta/files/file-alpha".into()),
        filename: Some("guide.txt".into()),
        media_type: Some("text/plain".into()),
        size_bytes: Some(128),
        expires_at: None,
        processing_status: None,
        downloadable: None,
        purpose: None,
    }
}

fn reply(method: &'static str, url: &'static str, body: Value) -> Reply {
    Reply {
        method,
        url,
        body,
        headers: Vec::new(),
    }
}

fn reply_with_headers(
    method: &'static str,
    url: &'static str,
    body: Value,
    headers: Vec<(String, String)>,
) -> Reply {
    Reply {
        method,
        url,
        body,
        headers,
    }
}

#[tokio::test]
async fn stores_use_documented_page_tokens_and_force_delete() {
    let (client, mock) = setup(vec![
        reply(
            "GET",
            "https://generativelanguage.googleapis.com/v1beta/fileSearchStores?pageSize=5&pageToken=older+page",
            json!({
                "fileSearchStores":[{"name":"fileSearchStores/store-alpha","displayName":"Docs"}],
                "nextPageToken":"next"
            }),
        ),
        reply(
            "GET",
            "https://generativelanguage.googleapis.com/v1beta/fileSearchStores/store-alpha",
            json!({"name":"fileSearchStores/store-alpha","displayName":"Docs"}),
        ),
        reply(
            "DELETE",
            "https://generativelanguage.googleapis.com/v1beta/fileSearchStores/store-alpha?force=true",
            json!({}),
        ),
    ]);
    let service_provider = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap();
    let service = service_provider.file_search();
    let page = service
        .list_stores(Some(5), Some("older page"), &options())
        .await
        .unwrap();
    assert_eq!(page.items[0].reference, store_ref());
    assert_eq!(page.next_page_token.as_deref(), Some("next"));
    let store = service
        .get_store(&page.items[0].reference, &options())
        .await
        .unwrap();
    assert_eq!(store.display_name.as_deref(), Some("Docs"));
    service
        .delete_store(&store.reference, true, &options())
        .await
        .unwrap();

    let requests = mock.sent.lock().unwrap();
    assert_eq!(requests[0].method, "GET");
    assert_eq!(requests[1].method, "GET");
    assert_eq!(requests[2].method, "DELETE");
}

#[tokio::test]
async fn create_store_import_and_get_operation_preserve_google_lro() {
    let (client, mock) = setup(vec![
        reply(
            "POST",
            STORE_COLLECTION,
            json!({
                "name":"fileSearchStores/store-alpha",
                "displayName":"Product docs",
                "embeddingModel":"models/gemini-embedding-2",
                "activeDocumentsCount":"0"
            }),
        ),
        reply(
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/fileSearchStores/store-alpha:importFile",
            json!({
                "name":"fileSearchStores/store-alpha/operations/import-01",
                "done":false,
                "metadata":{"progress":25}
            }),
        ),
        reply(
            "GET",
            "https://generativelanguage.googleapis.com/v1beta/fileSearchStores/store-alpha/operations/import-01",
            json!({
                "name":"fileSearchStores/store-alpha/operations/import-01",
                "done":true,
                "response":{"documentName":"fileSearchStores/store-alpha/documents/guide"}
            }),
        ),
    ]);
    let service_provider = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap();
    let service = service_provider.file_search();
    let store = service
        .create_store(
            Some("Product docs"),
            Some("models/gemini-embedding-2"),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(store.reference, store_ref());
    assert_eq!(store.active_documents_count, Some(0));

    let import = service
        .import_file(
            &store.reference,
            &file(),
            &[GeminiFileSearchMetadata {
                key: "department".into(),
                value: GeminiFileSearchMetadataValue::String("support".into()),
            }],
            Some(GeminiFileSearchWhiteSpaceChunking {
                max_tokens_per_chunk: 200,
                max_overlap_tokens: 20,
            }),
            &options(),
        )
        .await
        .unwrap();
    assert_eq!(import.state, GeminiFileSearchOperationState::Running);
    assert_eq!(import.metadata, Some(json!({"progress":25})));

    let done = service
        .get_operation(&import.reference, &options())
        .await
        .unwrap();
    assert_eq!(done.state, GeminiFileSearchOperationState::Succeeded);
    assert!(done.state.is_terminal());
    assert_eq!(done.request_id.as_deref(), Some("req-file-search"));

    let requests = mock.sent.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests[0].headers[0],
        ("x-goog-api-key".into(), "gemini-key".into())
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[0].body).unwrap(),
        json!({"displayName":"Product docs","embeddingModel":"models/gemini-embedding-2"})
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&requests[1].body).unwrap(),
        json!({
            "fileName":"files/file-alpha",
            "customMetadata":[{"key":"department","stringValue":"support"}],
            "chunkingConfig":{"whiteSpaceConfig":{"maxTokensPerChunk":200,"maxOverlapTokens":20}}
        })
    );
    assert_eq!(requests[2].method, "GET");
}

#[tokio::test]
async fn direct_upload_uses_resumable_session_and_scoped_upload_operation() {
    let start_url = "https://generativelanguage.googleapis.com/upload/v1beta/fileSearchStores/store-alpha:uploadToFileSearchStore";
    let session_url = "https://generativelanguage.googleapis.com/upload/v1beta/fileSearchStores/store-alpha:uploadToFileSearchStore?upload_id=session-1";
    let (client, mock) = setup(vec![
        reply_with_headers(
            "POST",
            start_url,
            Value::Null,
            vec![("X-Goog-Upload-URL".into(), session_url.into())],
        ),
        reply(
            "POST",
            session_url,
            json!({
                "name":"fileSearchStores/store-alpha/upload/operations/upload-01",
                "done":false,
                "metadata":{"progress":10}
            }),
        ),
        reply(
            "GET",
            "https://generativelanguage.googleapis.com/v1beta/fileSearchStores/store-alpha/upload/operations/upload-01",
            json!({
                "name":"fileSearchStores/store-alpha/upload/operations/upload-01",
                "done":true,
                "response":{"documentName":"fileSearchStores/store-alpha/documents/guide"}
            }),
        ),
    ]);
    let request =
        GeminiFileSearchUploadRequest::new(Bytes::from_static(b"hello Gemini"), "text/plain")
            .with_display_name("guide.txt")
            .with_custom_metadata([GeminiFileSearchMetadata {
                key: "department".into(),
                value: GeminiFileSearchMetadataValue::String("support".into()),
            }])
            .with_chunking_config(GeminiFileSearchWhiteSpaceChunking {
                max_tokens_per_chunk: 200,
                max_overlap_tokens: 20,
            });

    let service_provider = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap();
    let service = service_provider.file_search();
    let operation = service
        .upload_to_store(&store_ref(), &request, &options())
        .await
        .unwrap();
    assert_eq!(operation.state, GeminiFileSearchOperationState::Running);
    assert_eq!(operation.reference.store, store_ref());
    assert_eq!(operation.reference.operation_id, "upload-01");
    assert_eq!(operation.request_id.as_deref(), Some("req-file-search"));

    let completed = service
        .get_upload_operation(&operation.reference, &options())
        .await
        .unwrap();
    assert_eq!(completed.state, GeminiFileSearchOperationState::Succeeded);
    assert_eq!(
        completed.native["response"]["documentName"],
        "fileSearchStores/store-alpha/documents/guide"
    );

    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent.len(), 3);
    assert!(sent[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("x-goog-upload-protocol") && value == "resumable"
    }));
    assert!(sent[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("x-goog-upload-command") && value == "start"
    }));
    let expected_length = request.data.len().to_string();
    assert!(sent[0].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("x-goog-upload-header-content-length")
            && value == &expected_length
    }));
    let metadata: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(metadata["mimeType"], "text/plain");
    assert_eq!(metadata["displayName"], "guide.txt");
    assert_eq!(metadata["customMetadata"][0]["stringValue"], "support");
    assert_eq!(
        metadata["chunkingConfig"]["whiteSpaceConfig"]["maxTokensPerChunk"],
        200
    );

    assert_eq!(sent[1].body, request.data);
    assert!(sent[1]
        .headers
        .iter()
        .any(|(name, value)| name.eq_ignore_ascii_case("x-goog-upload-offset") && value == "0"));
    assert!(sent[1].headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("x-goog-upload-command") && value == "upload, finalize"
    }));
    assert!(!sent[1]
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("x-goog-api-key")));
    assert_eq!(sent[2].method, "GET");
}

#[tokio::test]
async fn direct_upload_rejects_wrong_account_and_untrusted_session_origin() {
    let start_url = "https://generativelanguage.googleapis.com/upload/v1beta/fileSearchStores/store-alpha:uploadToFileSearchStore";
    let (client, mock) = setup(vec![reply_with_headers(
        "POST",
        start_url,
        Value::Null,
        vec![(
            "x-goog-upload-url".into(),
            "https://attacker.example/upload?session=secret".into(),
        )],
    )]);
    let service_provider = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap();
    let service = service_provider.file_search();
    let request = GeminiFileSearchUploadRequest::new(Bytes::from_static(b"body"), "text/plain");
    let mut other = options();
    other.account_scope = Some("account-b".into());
    assert!(matches!(
        service
            .upload_to_store(&store_ref(), &request, &other)
            .await,
        Err(GeminiFileSearchError::Llm(
            LlmError::PermissionDenied { .. }
        ))
    ));
    assert!(mock.sent.lock().unwrap().is_empty());

    assert!(matches!(
        service
            .upload_to_store(&store_ref(), &request, &options())
            .await,
        Err(GeminiFileSearchError::InvalidResponse { .. })
    ));
    let sent = mock.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert!(!sent[0]
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("authorization")));
}

#[tokio::test]
async fn document_pages_keep_google_tokens_and_lifecycle_state() {
    let (client, mock) = setup(vec![
        reply(
            "GET",
            "https://generativelanguage.googleapis.com/v1beta/fileSearchStores/store-alpha/documents?pageSize=20&pageToken=next%2Fpage",
            json!({
                "documents":[{
                    "name":"fileSearchStores/store-alpha/documents/doc-alpha",
                    "displayName":"Guide",
                    "state":"STATE_ACTIVE",
                    "sizeBytes":"42",
                    "mimeType":"text/plain",
                    "customMetadata":[{"key":"department","stringValue":"support"}]
                }],
                "nextPageToken":"page 3"
            }),
        ),
        reply(
            "GET",
            "https://generativelanguage.googleapis.com/v1beta/fileSearchStores/store-alpha/documents/doc-alpha",
            json!({
                "name":"fileSearchStores/store-alpha/documents/doc-alpha",
                "state":"STATE_PENDING"
            }),
        ),
        reply(
            "DELETE",
            "https://generativelanguage.googleapis.com/v1beta/fileSearchStores/store-alpha/documents/doc-alpha?force=true",
            json!({}),
        ),
    ]);
    let service_provider = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap();
    let service = service_provider.file_search();
    let page = service
        .list_documents(&store_ref(), Some(20), Some("next/page"), &options())
        .await
        .unwrap();
    assert_eq!(page.next_page_token.as_deref(), Some("page 3"));
    assert!(page.items[0].state.is_ready());
    assert_eq!(page.items[0].size_bytes, Some(42));

    let pending = service
        .get_document(&page.items[0].reference, &options())
        .await
        .unwrap();
    assert_eq!(pending.state, GeminiFileSearchDocumentState::Pending);
    assert!(!pending.state.is_terminal());
    service
        .delete_document(&page.items[0].reference, true, &options())
        .await
        .unwrap();

    let requests = mock.sent.lock().unwrap();
    assert!(requests[0]
        .url
        .ends_with("pageSize=20&pageToken=next%2Fpage"));
    assert_eq!(requests[2].method, "DELETE");
}

#[tokio::test]
async fn resource_scope_and_page_size_are_checked_before_network() {
    let (client, mock) = setup(vec![]);
    let service_provider = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap();
    let service = service_provider.file_search();
    let mut other_account = options();
    other_account.account_scope = Some("account-b".into());
    assert!(matches!(
        service.get_store(&store_ref(), &other_account).await,
        Err(GeminiFileSearchError::Llm(
            LlmError::PermissionDenied { .. }
        ))
    ));
    assert!(matches!(
        service.list_stores(Some(21), None, &options()).await,
        Err(GeminiFileSearchError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(mock.sent.lock().unwrap().is_empty());
}

#[test]
fn route_requires_google_file_search_collection_and_api_key_header() {
    let profile = profile();
    let ServiceSetting::Enabled(route) = &profile.gemini_file_search else {
        panic!("expected route");
    };
    validate_route(&profile, route).unwrap();

    let mut invalid = route.clone();
    invalid.endpoint = "https://generativelanguage.googleapis.com/v1beta/files".into();
    assert!(validate_route(&profile, &invalid).is_err());
}
