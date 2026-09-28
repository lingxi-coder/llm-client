use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{StreamExt, TryStreamExt};
use lingxi_llm_client::{
    files::{UploadFile, UploadFileStream},
    protocol::{LlmError, Secret},
    providers::xai::collections::*,
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

struct Mock {
    replies: Mutex<VecDeque<(u16, Value)>>,
    sent: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        let (status, body) = self.replies.lock().unwrap().pop_front().unwrap();
        let bytes = serde_json::to_vec(&body).unwrap();
        Ok(StreamResponse {
            status,
            headers: vec![("x-request-id".into(), "xai-request-1".into())],
            body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
        })
    }
}

struct FailingTransport {
    sent: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for FailingTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(request);
        Err(LlmError::TransportTimeout {
            message: "mock update timeout".into(),
        })
    }
}

struct StreamRequestRecord {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Bytes,
    content_length: u64,
}

struct StreamMock {
    replies: Mutex<VecDeque<(u16, Value)>>,
    attempts: AtomicUsize,
    sent: Mutex<Vec<StreamRequestRecord>>,
}

#[async_trait]
impl Transport for StreamMock {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "stream-only test transport".into(),
        })
    }

    async fn send_stream(&self, request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        let HttpStreamRequest {
            method,
            url,
            headers,
            body,
            content_length,
            ..
        } = request;
        let body = body
            .try_fold(BytesMut::new(), |mut output, chunk| async move {
                output.extend_from_slice(&chunk);
                Ok(output)
            })
            .await?
            .freeze();
        self.sent.lock().unwrap().push(StreamRequestRecord {
            method,
            url,
            headers,
            body,
            content_length,
        });
        let (status, body) = self.replies.lock().unwrap().pop_front().unwrap();
        let bytes = serde_json::to_vec(&body).unwrap();
        Ok(StreamResponse {
            status,
            headers: vec![("x-request-id".into(), "xai-stream-request".into())],
            body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
        })
    }
}

struct FailingStreamTransport {
    attempts: AtomicUsize,
}

#[async_trait]
impl Transport for FailingStreamTransport {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "streaming only".into(),
        })
    }

    async fn send_stream(&self, _request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        Err(LlmError::TransportTimeout {
            message: "mock direct upload timeout".into(),
        })
    }
}

fn credentials() -> XaiCollectionsCredentials {
    XaiCollectionsCredentials::new(
        Secret::new("xai-api-secret".to_owned()),
        Secret::new("xai-management-secret".to_owned()),
    )
}

fn collection_json(id: &str, name: &str) -> Value {
    json!({
        "collection_id": id,
        "collection_name": name,
        "created_at": "2026-09-25T12:00:00Z",
        "documents_count": 0,
        "collection_description": "test collection"
    })
}

#[tokio::test]
async fn create_and_list_use_management_key_and_create_scoped_refs() {
    let mock = Mock {
        replies: Mutex::new(
            vec![
                (200, collection_json("collection_alpha", "Research")),
                (
                    200,
                    json!({"collections":[collection_json("collection_alpha", "Research")],"pagination_token":"next/1"}),
                ),
            ]
            .into(),
        ),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let creds = credentials();

    let created = client
        .create_collection(
            &XaiCreateCollectionRequest {
                name: "Research".into(),
                description: Some("test collection".into()),
                index_configuration: None,
                chunk_configuration: None,
                field_definitions: vec![XaiCollectionFieldDefinition {
                    key: "author".into(),
                    required: true,
                    unique: false,
                    inject_into_chunk: true,
                }],
            },
            &creds,
        )
        .await
        .unwrap();
    assert_eq!(created.name, "Research");
    assert_eq!(created.reference.scope.account_scope, "team/account-1");
    assert_eq!(created.reference.scope.provider_id.as_str(), "xai");

    let page = client
        .list_collections(
            &XaiCollectionListRequest {
                limit: 10,
                pagination_token: Some("older page".into()),
                filter: Some("collection_name:\"Research\"".into()),
                order: Some("Descending".into()),
                sort_by: Some("collection_name".into()),
            },
            &creds,
        )
        .await
        .unwrap();
    assert_eq!(page.collections[0].reference, created.reference);
    assert_eq!(page.pagination_token.as_deref(), Some("next/1"));

    let requests = mock.sent.lock().unwrap();
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].url, "https://management.test/v1/collections");
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer xai-management-secret"));
    assert!(!requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value.contains("xai-api-secret")));
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["collection_name"], "Research");
    assert_eq!(body["field_definitions"][0]["inject_into_chunk"], true);
    assert!(requests[1].url.contains("pagination_token=older+page"));
    assert!(requests[1]
        .url
        .contains("filter=collection_name%3A%22Research%22"));
    assert!(requests[1].url.contains("order=Descending"));
    assert!(requests[1].url.contains("sort_by=collection_name"));
    assert!(!format!("{creds:?}").contains("xai-management-secret"));
}

#[tokio::test]
async fn upload_then_attach_lists_and_removes_the_scoped_document() {
    let mock = Mock {
        replies: Mutex::new(
            vec![
                (
                    200,
                    json!({"id":"file_1","filename":"guide.md","bytes":5,"created_at":1780000000}),
                ),
                (200, json!({})),
                (
                    200,
                    json!({"documents":[{
                        "file_metadata":{"file_id":"file_1","name":"guide.md","size_bytes":"5","content_type":"text/markdown"},
                        "fields":{"author":"Mina"},"status":"DOCUMENT_STATUS_PROCESSED"
                    }],"pagination_token":null}),
                ),
                (
                    200,
                    json!({
                        "file_metadata":{"file_id":"file_1","name":"guide.md","size_bytes":"5","content_type":"text/markdown"},
                        "fields":{"author":"Mina"},"status":"DOCUMENT_STATUS_PROCESSED"
                    }),
                ),
                (200, json!({})),
            ]
            .into(),
        ),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let creds = credentials();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    let uploaded = client
        .upload_file(
            &UploadFile {
                filename: "guide.md".into(),
                media_type: "text/markdown".into(),
                bytes: Bytes::from_static(b"hello"),
            },
            &creds,
        )
        .await
        .unwrap();
    let document_ref = client
        .add_document(
            &collection,
            &uploaded.reference,
            Some(&json!({"author":"Mina"})),
            &creds,
        )
        .await
        .unwrap();
    let page = client
        .list_documents(
            &collection,
            &XaiDocumentListRequest {
                filter: Some("status:DOCUMENT_STATUS_PROCESSED".into()),
                order: Some("Descending".into()),
                sort_by: Some("name".into()),
                ..Default::default()
            },
            &creds,
        )
        .await
        .unwrap();
    assert_eq!(page.documents[0].reference, document_ref);
    assert_eq!(page.documents[0].status, Some(XaiDocumentStatus::Processed));
    assert_eq!(page.documents[0].fields["author"], "Mina");
    let fetched = client.get_document(&document_ref, &creds).await.unwrap();
    assert_eq!(fetched.filename.as_deref(), Some("guide.md"));
    client.remove_document(&document_ref, &creds).await.unwrap();

    let requests = mock.sent.lock().unwrap();
    assert_eq!(requests[0].url, "https://api.test/v1/files");
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer xai-api-secret"));
    let multipart = String::from_utf8_lossy(&requests[0].body);
    assert!(multipart.contains("name=\"purpose\"\r\n\r\nassistants"));
    assert!(multipart.contains("name=\"file\"; filename=\"guide.md\""));
    assert!(multipart.contains("\r\n\r\nhello\r\n"));
    assert_eq!(
        requests[1].url,
        "https://management.test/v1/collections/collection_alpha/documents/file_1"
    );
    assert!(requests[2]
        .url
        .contains("filter=status%3ADOCUMENT_STATUS_PROCESSED"));
    assert!(requests[2].url.contains("order=Descending"));
    assert!(requests[2].url.contains("sort_by=name"));
    let fields: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(fields, json!({"fields":{"author":"Mina"}}));
    assert_eq!(requests[2].method, "GET");
    assert_eq!(requests[3].method, "GET");
    assert_eq!(requests[4].method, "DELETE");
}

#[tokio::test]
async fn search_uses_api_key_and_rejects_refs_from_another_scope() {
    let mock = Mock {
        replies: Mutex::new(
            vec![(
                200,
                json!({"matches":[{
                    "file_id":"file_1",
                    "chunk_id":"chunk_1",
                    "chunk_content":"A useful passage",
                    "score":0.91,
                    "collection_ids":["collection_alpha"]
                }]}),
            )]
            .into(),
        ),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    let creds = credentials();
    let result = client
        .search(
            &XaiSearchRequest {
                query: "revenue trends".into(),
                collections: vec![collection.clone()],
                filter: Some("year >= 2024".into()),
                retrieval_mode: None,
            },
            &creds,
        )
        .await
        .unwrap();
    assert_eq!(result.matches[0].documents[0].collection, collection);
    assert_eq!(
        result.matches[0].content.as_deref(),
        Some("A useful passage")
    );
    assert_eq!(result.request_id.as_deref(), Some("xai-request-1"));
    {
        let requests = mock.sent.lock().unwrap();
        assert_eq!(requests[0].url, "https://api.test/v1/documents/search");
        assert!(requests[0]
            .headers
            .iter()
            .any(|(name, value)| name == "authorization" && value == "Bearer xai-api-secret"));
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["source"]["collection_ids"][0], "collection_alpha");
        assert_eq!(body["filter"], "year >= 2024");
        assert!(body.get("retrieval_mode").is_none());
    }

    let other_profile = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-personal", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    assert!(matches!(
        other_profile.delete_collection(&collection, &creds).await,
        Err(XaiCollectionsError::InvalidRequest(_))
    ));
    assert_eq!(mock.sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn search_can_select_each_documented_retrieval_mode() {
    let mock = Mock {
        replies: Mutex::new(vec![(200, json!({"matches":[]})); 3].into()),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let request = XaiSearchRequest {
        query: "quarterly revenue".into(),
        collections: vec![XaiCollectionRef {
            scope: client.scope().clone(),
            collection_id: "collection_alpha".into(),
        }],
        filter: None,
        retrieval_mode: None,
    };
    let credentials = credentials();

    for (mode, expected) in [
        (XaiRetrievalMode::Keyword, "keyword"),
        (XaiRetrievalMode::Semantic, "semantic"),
        (XaiRetrievalMode::Hybrid, "hybrid"),
    ] {
        let mut request_with_mode = request.clone();
        request_with_mode.retrieval_mode = Some(mode);
        client
            .search(&request_with_mode, &credentials)
            .await
            .unwrap();

        let sent = mock.sent.lock().unwrap();
        let body: Value = serde_json::from_slice(&sent.last().unwrap().body).unwrap();
        assert_eq!(body["retrieval_mode"]["type"], expected);
        assert_eq!(body["source"]["collection_ids"][0], "collection_alpha");
    }
    assert_eq!(mock.sent.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn list_query_options_reject_empty_or_control_only_values_before_dispatch() {
    let mock = Mock {
        replies: Mutex::new(VecDeque::new()),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };

    assert!(matches!(
        client
            .list_collections(
                &XaiCollectionListRequest {
                    filter: Some("  ".into()),
                    ..Default::default()
                },
                &credentials(),
            )
            .await,
        Err(XaiCollectionsError::InvalidRequest(_))
    ));
    assert!(matches!(
        client
            .list_documents(
                &collection,
                &XaiDocumentListRequest {
                    sort_by: Some("\n".into()),
                    ..Default::default()
                },
                &credentials(),
            )
            .await,
        Err(XaiCollectionsError::InvalidRequest(_))
    ));
    assert!(mock.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn search_rejects_provider_matches_outside_the_requested_collections() {
    let mock = Mock {
        replies: Mutex::new(
            vec![(
                200,
                json!({"matches":[{
                    "file_id":"file_1",
                    "collection_ids":["collection_other"]
                }]}),
            )]
            .into(),
        ),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    let result = client
        .search(
            &XaiSearchRequest {
                query: "query".into(),
                collections: vec![collection],
                filter: None,
                retrieval_mode: None,
            },
            &credentials(),
        )
        .await;
    assert!(matches!(
        result,
        Err(XaiCollectionsError::InvalidResponse(_))
    ));
}

#[tokio::test]
async fn delete_collection_uses_management_key_and_collection_scope() {
    let mock = Mock {
        replies: Mutex::new(vec![(200, json!({}))].into()),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    client
        .delete_collection(&collection, &credentials())
        .await
        .unwrap();
    let requests = mock.sent.lock().unwrap();
    assert_eq!(requests[0].method, "DELETE");
    assert_eq!(
        requests[0].url,
        "https://management.test/v1/collections/collection_alpha"
    );
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer xai-management-secret"));
}

#[tokio::test]
async fn update_collection_uses_put_returns_scoped_metadata_and_preflights() {
    let mock = Mock {
        replies: Mutex::new(vec![(200, collection_json("collection_alpha", "Updated"))].into()),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    let updated = client
        .update_collection(
            &collection,
            &XaiUpdateCollectionRequest {
                name: Some("Updated".into()),
                description: Some("new description".into()),
                chunk_configuration: Some(
                    json!({"tokens_configuration":{"max_chunk_size_tokens":512}}),
                ),
                field_definitions: Some(vec![XaiCollectionFieldDefinition {
                    key: "author".into(),
                    required: true,
                    unique: false,
                    inject_into_chunk: false,
                }]),
            },
            &credentials(),
        )
        .await
        .unwrap();
    assert_eq!(updated.reference, collection);
    assert_eq!(updated.name, "Updated");

    {
        let requests = mock.sent.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].method, "PUT");
        assert_eq!(
            requests[0].url,
            "https://management.test/v1/collections/collection_alpha"
        );
        assert!(requests[0].headers.iter().any(
            |(name, value)| name == "authorization" && value == "Bearer xai-management-secret"
        ));
        assert!(!requests[0]
            .headers
            .iter()
            .any(|(name, value)| name == "authorization" && value == "Bearer xai-api-secret"));
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["collection_name"], "Updated");
        assert_eq!(body["collection_description"], "new description");
        assert_eq!(
            body["chunk_configuration"]["tokens_configuration"]["max_chunk_size_tokens"],
            512
        );
        assert_eq!(body["field_definitions"][0]["key"], "author");
    }

    let other_account = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-2")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let foreign = XaiCollectionRef {
        scope: other_account.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    assert!(matches!(
        client
            .update_collection(
                &collection,
                &XaiUpdateCollectionRequest::default(),
                &credentials()
            )
            .await,
        Err(XaiCollectionsError::InvalidRequest(_))
    ));
    assert!(matches!(
        client
            .update_collection(
                &collection,
                &XaiUpdateCollectionRequest {
                    field_definitions: Some(vec![]),
                    ..Default::default()
                },
                &credentials()
            )
            .await,
        Err(XaiCollectionsError::InvalidRequest(_))
    ));
    assert!(matches!(
        client
            .update_collection(
                &foreign,
                &XaiUpdateCollectionRequest {
                    name: Some("Updated".into()),
                    ..Default::default()
                },
                &credentials()
            )
            .await,
        Err(XaiCollectionsError::InvalidRequest(_))
    ));
    assert_eq!(mock.sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn update_collection_rejects_malformed_and_mismatched_responses() {
    for body in [Value::Null, collection_json("collection_other", "Updated")] {
        let mock = Mock {
            replies: Mutex::new(vec![(200, body)].into()),
            sent: Mutex::new(vec![]),
        };
        let client = XaiCollectionsClient::new(
            &mock,
            XaiCollectionsConfig::new("xai-work", "team/account-1")
                .with_api_base_url("https://api.test/v1")
                .with_management_base_url("https://management.test/v1"),
        )
        .unwrap();
        let collection = XaiCollectionRef {
            scope: client.scope().clone(),
            collection_id: "collection_alpha".into(),
        };
        assert!(matches!(
            client
                .update_collection(
                    &collection,
                    &XaiUpdateCollectionRequest {
                        name: Some("Updated".into()),
                        ..Default::default()
                    },
                    &credentials()
                )
                .await,
            Err(XaiCollectionsError::InvalidResponse(_))
        ));
        assert_eq!(mock.sent.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn update_transport_failure_is_reported_without_retry() {
    let transport = FailingTransport {
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &transport,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    let result = client
        .update_collection(
            &collection,
            &XaiUpdateCollectionRequest {
                name: Some("Updated".into()),
                ..Default::default()
            },
            &credentials(),
        )
        .await;
    assert!(matches!(
        result,
        Err(XaiCollectionsError::Transport(
            LlmError::TransportTimeout { .. }
        ))
    ));
    assert_eq!(transport.sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn batch_get_documents_accepts_a_subset_and_preserves_native_status() {
    let mock = Mock {
        replies: Mutex::new(
            vec![(
                200,
                json!({"documents":[{
                    "file_metadata":{"file_id":"file_2","name":"second.md","size_bytes":"8","content_type":"text/markdown"},
                    "fields":{"author":"Mina"},
                    "status":"DOCUMENT_STATUS_FAILED",
                    "error_message":"indexing failed"
                }] }),
            )]
            .into(),
        ),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    let documents = ["file_1", "file_2"].map(|file_id| XaiDocumentRef {
        collection: collection.clone(),
        file_id: file_id.into(),
    });

    let result = client
        .batch_get_documents(&documents, &credentials())
        .await
        .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].reference, documents[1]);
    assert_eq!(result[0].status, Some(XaiDocumentStatus::Failed));
    assert_eq!(result[0].error_message.as_deref(), Some("indexing failed"));
    assert_eq!(result[0].native["status"], "DOCUMENT_STATUS_FAILED");

    let requests = mock.sent.lock().unwrap();
    assert_eq!(
        requests[0].url,
        "https://management.test/v1/collections/collection_alpha/documents:batchGet?file_ids=file_1&file_ids=file_2"
    );
    assert_eq!(requests[0].method, "GET");
    assert!(requests[0].headers.iter().any(|(name, value)| {
        name == "authorization" && value == "Bearer xai-management-secret"
    }));
}

#[tokio::test]
async fn batch_get_documents_preflights_refs_and_rejects_unrequested_response_ids() {
    let mock = Mock {
        replies: Mutex::new(
            vec![(
                200,
                json!({"documents":[{
                    "file_metadata":{"file_id":"file_other"},"fields":{}
                }]}),
            )]
            .into(),
        ),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    let document = XaiDocumentRef {
        collection: collection.clone(),
        file_id: "file_1".into(),
    };
    assert!(matches!(
        client.batch_get_documents(&[], &credentials()).await,
        Err(XaiCollectionsError::InvalidRequest(_))
    ));
    assert!(matches!(
        client
            .batch_get_documents(&[document.clone(), document.clone()], &credentials())
            .await,
        Err(XaiCollectionsError::InvalidRequest(_))
    ));
    let other_collection = XaiCollectionRef {
        collection_id: "collection_other".into(),
        ..collection.clone()
    };
    assert!(matches!(
        client
            .batch_get_documents(
                &[
                    document.clone(),
                    XaiDocumentRef {
                        collection: other_collection,
                        file_id: "file_2".into(),
                    },
                ],
                &credentials(),
            )
            .await,
        Err(XaiCollectionsError::InvalidRequest(_))
    ));
    assert_eq!(mock.sent.lock().unwrap().len(), 0);

    assert!(matches!(
        client
            .batch_get_documents(&[document], &credentials())
            .await,
        Err(XaiCollectionsError::InvalidResponse(_))
    ));
    assert_eq!(mock.sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn direct_collection_upload_streams_management_multipart_and_returns_document() {
    let mock = StreamMock {
        replies: Mutex::new(
            vec![(
                201,
                json!({
                    "file_metadata":{"file_id":"file_direct","name":"guide.md","size_bytes":"5","content_type":"text/markdown"},
                    "fields":{"author":"Mina"},
                    "status":"DOCUMENT_STATUS_PROCESSING"
                }),
            )]
            .into(),
        ),
        attempts: AtomicUsize::new(0),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    let source = futures::stream::iter(vec![
        Ok(Bytes::from_static(b"hel")),
        Ok(Bytes::from_static(b"lo")),
    ]);
    let document = client
        .upload_document_stream(
            &collection,
            UploadFileStream::new("guide.md", "text/markdown", 5, source),
            Some(&json!({"author":"Mina"})),
            &credentials(),
        )
        .await
        .unwrap();

    assert_eq!(document.reference.collection, collection);
    assert_eq!(document.reference.file_id, "file_direct");
    assert_eq!(document.status, Some(XaiDocumentStatus::Processing));
    assert_eq!(document.fields["author"], "Mina");
    assert_eq!(mock.attempts.load(Ordering::SeqCst), 1);
    let requests = mock.sent.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].url,
        "https://management.test/v1/collections/collection_alpha/documents"
    );
    assert_eq!(requests[0].content_length, requests[0].body.len() as u64);
    assert!(requests[0].headers.iter().any(|(name, value)| {
        name == "authorization" && value == "Bearer xai-management-secret"
    }));
    assert!(!requests[0]
        .headers
        .iter()
        .any(|(name, value)| { name == "authorization" && value == "Bearer xai-api-secret" }));
    let multipart = String::from_utf8_lossy(&requests[0].body);
    assert!(multipart.contains("name=\"name\"\r\n\r\nguide.md\r\n"));
    assert!(multipart.contains("name=\"data\"; filename=\"guide.md\""));
    assert!(multipart.contains("\r\n\r\nhello\r\n"));
    assert!(multipart.contains("name=\"content_type\"\r\n\r\ntext/markdown\r\n"));
    assert!(multipart.contains("name=\"fields\"\r\n\r\n{\"author\":\"Mina\"}\r\n"));
}

#[tokio::test]
async fn direct_collection_upload_source_invalid_request_has_unknown_outcome_once() {
    let mock = StreamMock {
        replies: Mutex::new(VecDeque::new()),
        attempts: AtomicUsize::new(0),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    let source = futures::stream::iter(vec![Err(LlmError::InvalidRequest {
        message: "source failed after dispatch".into(),
    })]);
    let result = client
        .upload_document_stream(
            &collection,
            UploadFileStream::new("guide.md", "text/markdown", 5, source),
            None,
            &credentials(),
        )
        .await;
    assert!(matches!(
        result,
        Err(XaiCollectionsError::OutcomeUnknown { .. })
    ));
    assert_eq!(mock.attempts.load(Ordering::SeqCst), 1);
    assert!(mock.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn direct_collection_upload_timeout_is_unknown_and_not_retried() {
    let transport = FailingStreamTransport {
        attempts: AtomicUsize::new(0),
    };
    let client = XaiCollectionsClient::new(
        &transport,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    let result = client
        .upload_document_stream(
            &collection,
            UploadFileStream::from_bytes("guide.md", "text/markdown", Bytes::from_static(b"hello")),
            None,
            &credentials(),
        )
        .await;
    assert!(matches!(
        result,
        Err(XaiCollectionsError::OutcomeUnknown { .. })
    ));
    assert_eq!(transport.attempts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn direct_collection_upload_preflights_document_size_before_dispatch() {
    let mock = StreamMock {
        replies: Mutex::new(VecDeque::new()),
        attempts: AtomicUsize::new(0),
        sent: Mutex::new(vec![]),
    };
    let client = XaiCollectionsClient::new(
        &mock,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let collection = XaiCollectionRef {
        scope: client.scope().clone(),
        collection_id: "collection_alpha".into(),
    };
    let stream = futures::stream::empty::<Result<Bytes, LlmError>>();
    let result = client
        .upload_document_stream(
            &collection,
            UploadFileStream::new(
                "oversized.pdf",
                "application/pdf",
                MAX_XAI_COLLECTION_DOCUMENT_BYTES + 1,
                stream,
            ),
            None,
            &credentials(),
        )
        .await;
    assert!(matches!(
        result,
        Err(XaiCollectionsError::InvalidRequest(_))
    ));
    assert_eq!(mock.attempts.load(Ordering::SeqCst), 0);
}
