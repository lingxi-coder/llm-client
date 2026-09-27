use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use lingxi_llm_client::{
    files::UploadFileStream,
    glm_knowledge::{
        GlmCreateKnowledgeRequest, GlmKnowledgeContextual, GlmKnowledgeDocumentListRequest,
        GlmKnowledgeEmbedding, GlmKnowledgeError, GlmKnowledgeListRequest,
        GlmKnowledgeRecallMethod, GlmKnowledgeRef, GlmKnowledgeRerankModel,
        GlmKnowledgeRetrieveRequest, GlmUpdateKnowledgeRequest, GlmUploadFileDocumentsRequest,
        GlmUploadUrlDocumentsRequest, GlmUrlDocumentInput,
    },
    protocol::{LlmError, ProviderProfile, Region, Secret},
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
    *,
};
use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

struct Reply {
    method: &'static str,
    url: &'static str,
    status: u16,
    body: Value,
}

struct Mock {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<HttpRequest>>,
    stream_lengths: Mutex<Vec<u64>>,
}

struct FailingTransport;

struct FailingUploadTransport {
    calls: AtomicUsize,
}

#[async_trait]
impl Transport for FailingTransport {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        Err(LlmError::TransportTimeout {
            message: "connection timed out after dispatch".into(),
        })
    }
}

#[async_trait]
impl Transport for FailingUploadTransport {
    async fn send(&self, _request: HttpRequest) -> Result<StreamResponse, LlmError> {
        panic!("unexpected buffered request")
    }

    async fn send_stream(
        &self,
        mut request: HttpStreamRequest,
    ) -> Result<StreamResponse, LlmError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let _ = request.body.next().await;
        let _ = request.body.next().await;
        Err(LlmError::TransportTimeout {
            message: "connection timed out after upload dispatch".into(),
        })
    }
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request");
        assert_eq!(request.method, reply.method);
        assert_eq!(request.url, reply.url);
        self.requests.lock().unwrap().push(request);
        Ok(response(reply))
    }

    async fn send_stream(&self, request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        let HttpStreamRequest {
            method,
            url,
            headers,
            mut body,
            content_length,
            timeout,
        } = request;
        self.stream_lengths.lock().unwrap().push(content_length);
        let mut collected = BytesMut::new();
        while let Some(chunk) = body.next().await {
            collected.extend_from_slice(&chunk?);
        }
        assert_eq!(collected.len() as u64, content_length);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request");
        assert_eq!(method, reply.method);
        assert_eq!(url, reply.url);
        self.requests.lock().unwrap().push(HttpRequest {
            method,
            url,
            headers,
            body: collected.freeze(),
            timeout,
        });
        Ok(response(reply))
    }
}

fn response(reply: Reply) -> StreamResponse {
    let bytes = serde_json::to_vec(&reply.body).unwrap();
    StreamResponse {
        status: reply.status,
        headers: vec![("x-request-id".into(), "trace-glm-1".into())],
        body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
    }
}

const ENDPOINT: &str = "https://open.bigmodel.cn/api/llm-application/open";

fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"zhipu",
        "profile_name":"glm-account",
        "base_url":"https://open.bigmodel.cn/api/paas/v4",
        "protocol":"open_ai_chat",
        "auth":"bearer",
        "regions":["china_mainland"],
        "glm_knowledge":{"mode":"enabled","value":{
            "endpoint":ENDPOINT,
            "auth":{"type":"bearer"}
        }}
    }))
    .unwrap()
}

fn options(scope: &str) -> RequestOptions {
    RequestOptions {
        account_scope: Some(scope.into()),
        credential: Some(Secret::new("zhipu-key-for-tests".into())),
        ..Default::default()
    }
}

fn setup(replies: Vec<Reply>) -> (LlmClient, Arc<Mock>) {
    let mock = Arc::new(Mock {
        replies: Mutex::new(replies.into()),
        requests: Mutex::new(Vec::new()),
        stream_lengths: Mutex::new(Vec::new()),
    });
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile()])
        .with_region(Region::ChinaMainland)
        .build()
        .unwrap();
    (client, mock)
}

fn reply(method: &'static str, url: &'static str, body: Value) -> Reply {
    Reply {
        method,
        url,
        status: 200,
        body,
    }
}

fn knowledge_ref(scope: &str) -> GlmKnowledgeRef {
    GlmKnowledgeRef {
        provider_id: "zhipu".into(),
        profile_name: "glm-account".into(),
        endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
            ENDPOINT,
        ),
        account_scope: scope.into(),
        knowledge_id: "kb_123".into(),
    }
}

#[tokio::test]
async fn submits_url_documents_and_retains_per_url_failures() {
    let (client, mock) = setup(vec![reply(
        "POST",
        "https://open.bigmodel.cn/api/llm-application/open/document/upload_url",
        json!({
            "data":{
                "successInfos":[{"documentId":"doc_8","url":"https://docs.example/guide"}],
                "failedInfos":[{"url":"https://docs.example/unsupported","failReason":"unsupported type"}]
            },
            "code":200,
            "message":"ok"
        }),
    )]);
    let knowledge = knowledge_ref("account-a");
    let result = client
        .glm_knowledge()
        .upload_url_documents(
            &knowledge,
            &GlmUploadUrlDocumentsRequest {
                documents: vec![GlmUrlDocumentInput {
                    url: "https://docs.example/guide".into(),
                    knowledge_type: None,
                    custom_separator: None,
                    sentence_size: Some(300),
                    callback_url: None,
                    callback_header: None,
                }],
            },
            &options("account-a"),
        )
        .await
        .unwrap();
    assert_eq!(result.succeeded.len(), 1);
    assert_eq!(result.succeeded[0].reference.knowledge, knowledge);
    assert_eq!(result.succeeded[0].reference.document_id, "doc_8");
    assert_eq!(result.failed.len(), 1);
    assert_eq!(
        result.failed[0].fail_reason.as_deref(),
        Some("unsupported type")
    );

    let sent = mock.requests.lock().unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&sent[0].body).unwrap(),
        json!({
            "knowledge_id":"kb_123",
            "upload_detail":[{"url":"https://docs.example/guide","sentence_size":300}]
        })
    );
}

#[tokio::test]
async fn streams_file_documents_and_preserves_per_file_partial_results() {
    let (client, mock) = setup(vec![reply(
        "POST",
        "https://open.bigmodel.cn/api/llm-application/open/document/upload_document/kb_123",
        json!({
            "data":{
                "successInfos":[
                    {"documentId":"doc_8","fileName":"manual.pdf","providerTag":"accepted"},
                    {"documentId":"doc_unrequested","fileName":"other.pdf"}
                ],
                "failedInfos":[{"fileName":"rows.csv","failReason":"unsupported format","providerCode":17}]
            },
            "code":200,
            "message":"ok"
        }),
    )]);
    let knowledge = knowledge_ref("account-a");
    let request = GlmUploadFileDocumentsRequest {
        knowledge_type: Some(5),
        custom_separator: Some(vec!["###".into(), "---".into()]),
        sentence_size: Some(512),
        parse_image: Some(true),
        callback_url: Some("https://hooks.example.test/glm?key=callback-secret".into()),
        callback_header: Some(json!({"authorization":"header-secret"})),
        word_num_limit: Some("50000".into()),
        request_id: Some("request-abc".into()),
    };
    let result = client
        .glm_knowledge()
        .upload_file_documents(
            &knowledge,
            vec![
                UploadFileStream::from_bytes(
                    "/private/manual.pdf",
                    "application/pdf",
                    Bytes::from_static(b"%PDF-1.7 manual"),
                ),
                UploadFileStream::from_bytes(
                    "rows.csv",
                    "text/csv",
                    Bytes::from_static(b"sku,count\r\n1,2"),
                ),
            ],
            &request,
            &options("account-a"),
        )
        .await
        .unwrap();

    assert_eq!(result.succeeded.len(), 1);
    assert_eq!(result.succeeded[0].reference.knowledge, knowledge);
    assert_eq!(result.succeeded[0].reference.document_id, "doc_8");
    assert_eq!(result.succeeded[0].file_name.as_deref(), Some("manual.pdf"));
    assert_eq!(result.succeeded[0].native["providerTag"], "accepted");
    assert_eq!(result.failed.len(), 1);
    assert_eq!(result.failed[0].file_name.as_deref(), Some("rows.csv"));
    assert_eq!(
        result.failed[0].fail_reason.as_deref(),
        Some("unsupported format")
    );
    assert_eq!(result.failed[0].native["providerCode"], 17);
    assert_eq!(result.unresolved.len(), 1);
    assert_eq!(result.unresolved[0]["fileName"], "other.pdf");
    assert!(result.missing_files.is_empty());

    let sent = mock.requests.lock().unwrap();
    let upload = &sent[0];
    assert_eq!(upload.method, "POST");
    assert!(upload
        .headers
        .iter()
        .any(|(name, value)| { name == "authorization" && value == "Bearer zhipu-key-for-tests" }));
    let content_type = upload
        .headers
        .iter()
        .find(|(name, _)| name == "content-type")
        .unwrap()
        .1
        .clone();
    assert!(content_type.starts_with("multipart/form-data; boundary="));
    let wire = String::from_utf8_lossy(&upload.body);
    assert!(wire.contains("name=\"files\"; filename=\"manual.pdf\""));
    assert!(wire.contains("name=\"files\"; filename=\"rows.csv\""));
    assert!(wire.contains("Content-Type: application/pdf"));
    assert!(wire.contains("Content-Type: text/csv"));
    assert!(wire.contains("%PDF-1.7 manual"));
    assert!(wire.contains("sku,count\r\n1,2"));
    assert_eq!(wire.matches("name=\"custom_separator\"").count(), 2);
    assert!(wire.contains("name=\"sentence_size\"\r\n\r\n512"));
    assert!(wire.contains("name=\"parse_image\"\r\n\r\ntrue"));
    assert!(wire.contains("name=\"callback_header\"\r\n\r\n{\"authorization\":\"header-secret\"}"));
    assert!(wire.contains("name=\"word_num_limit\"\r\n\r\n50000"));
    assert!(wire.contains("name=\"req_id\"\r\n\r\nrequest-abc"));
    assert_eq!(
        mock.stream_lengths.lock().unwrap()[0],
        upload.body.len() as u64
    );

    let debug = format!("{request:?}");
    assert!(debug.contains("<redacted>"));
    assert!(!debug.contains("callback-secret"));
    assert!(!debug.contains("header-secret"));
}

#[tokio::test]
async fn file_upload_preflights_scope_and_form_before_dispatch() {
    let (client, mock) = setup(vec![]);
    let file =
        || UploadFileStream::from_bytes("guide.txt", "text/plain", Bytes::from_static(b"guide"));
    let cross_account = client
        .glm_knowledge()
        .upload_file_documents(
            &knowledge_ref("account-a"),
            vec![file()],
            &GlmUploadFileDocumentsRequest::default(),
            &options("account-b"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        cross_account,
        GlmKnowledgeError::Llm(LlmError::PermissionDenied { .. })
    ));

    let invalid = client
        .glm_knowledge()
        .upload_file_documents(
            &knowledge_ref("account-a"),
            vec![file()],
            &GlmUploadFileDocumentsRequest {
                knowledge_type: Some(5),
                sentence_size: Some(19),
                ..Default::default()
            },
            &options("account-a"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        invalid,
        GlmKnowledgeError::Llm(LlmError::InvalidRequest { .. })
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
    assert!(mock.stream_lengths.lock().unwrap().is_empty());
}

#[tokio::test]
async fn interrupted_file_upload_is_unknown_and_is_not_retried() {
    let transport = Arc::new(FailingUploadTransport {
        calls: AtomicUsize::new(0),
    });
    let client = LlmClientBuilder::with_transport(transport.clone(), &[profile()])
        .with_region(Region::ChinaMainland)
        .build()
        .unwrap();
    let error = client
        .glm_knowledge()
        .upload_file_documents(
            &knowledge_ref("account-a"),
            vec![UploadFileStream::from_bytes(
                "guide.txt",
                "text/plain",
                Bytes::from_static(b"guide contents"),
            )],
            &GlmUploadFileDocumentsRequest::default(),
            &options("account-a"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        GlmKnowledgeError::OutcomeUnknown {
            operation: "upload_file_documents",
            source: LlmError::TransportTimeout { .. }
        }
    ));
    assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn incomplete_file_upload_response_reports_missing_input_files() {
    let (client, _) = setup(vec![reply(
        "POST",
        "https://open.bigmodel.cn/api/llm-application/open/document/upload_document/kb_123",
        json!({
            "data":{"successInfos":[{"documentId":"doc_1","fileName":"guide.txt"}]},
            "code":200,"message":"ok"
        }),
    )]);
    let result = client
        .glm_knowledge()
        .upload_file_documents(
            &knowledge_ref("account-a"),
            vec![
                UploadFileStream::from_bytes("guide.txt", "text/plain", Bytes::from_static(b"a")),
                UploadFileStream::from_bytes("missing.txt", "text/plain", Bytes::from_static(b"b")),
            ],
            &GlmUploadFileDocumentsRequest::default(),
            &options("account-a"),
        )
        .await
        .unwrap();
    assert_eq!(result.succeeded.len(), 1);
    assert_eq!(result.missing_files, vec!["missing.txt"]);
}

#[tokio::test]
async fn file_upload_timeout_and_malformed_success_responses_remain_uncertain() {
    let mut server_error = reply(
        "POST",
        "https://open.bigmodel.cn/api/llm-application/open/document/upload_document/kb_123",
        json!({"data":{"partial":"preserved"},"message":"gateway timeout"}),
    );
    server_error.status = 503;
    let (client, _) = setup(vec![server_error]);
    let result = client
        .glm_knowledge()
        .upload_file_documents(
            &knowledge_ref("account-a"),
            vec![UploadFileStream::from_bytes(
                "guide.txt",
                "text/plain",
                Bytes::from_static(b"guide"),
            )],
            &GlmUploadFileDocumentsRequest::default(),
            &options("account-a"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        result,
        GlmKnowledgeError::ResponseOutcomeUnknown {
            operation: "upload_file_documents",
            native,
            ..
        } if native["data"]["partial"] == "preserved"
    ));

    let (client, _) = setup(vec![reply(
        "POST",
        "https://open.bigmodel.cn/api/llm-application/open/document/upload_document/kb_123",
        json!({"code":200,"message":"ok","provider_field":"retained"}),
    )]);
    let result = client
        .glm_knowledge()
        .upload_file_documents(
            &knowledge_ref("account-a"),
            vec![UploadFileStream::from_bytes(
                "guide.txt",
                "text/plain",
                Bytes::from_static(b"guide"),
            )],
            &GlmUploadFileDocumentsRequest::default(),
            &options("account-a"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        result,
        GlmKnowledgeError::ResponseOutcomeUnknown {
            operation: "upload_file_documents",
            native,
            ..
        } if native["provider_field"] == "retained"
    ));
}

#[test]
fn callback_credentials_are_redacted_from_knowledge_request_debug() {
    let update = GlmUpdateKnowledgeRequest {
        callback_url: Some("https://hooks.example.test/?token=url-secret".into()),
        callback_header: Some(json!({"Authorization":"header-secret"})),
        ..Default::default()
    };
    let url_input = GlmUrlDocumentInput {
        url: "https://docs.example/manual".into(),
        knowledge_type: None,
        custom_separator: None,
        sentence_size: None,
        callback_url: Some("https://hooks.example.test/?token=url-secret".into()),
        callback_header: Some(json!({"Authorization":"header-secret"})),
    };
    for debug in [format!("{update:?}"), format!("{url_input:?}")] {
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("url-secret"));
        assert!(!debug.contains("header-secret"));
    }
}

#[tokio::test]
async fn creates_a_zhipu_knowledge_base_using_native_embedding_ids() {
    let (client, mock) = setup(vec![reply(
        "POST",
        "https://open.bigmodel.cn/api/llm-application/open/knowledge",
        json!({"data":{"id":"kb_123"},"code":200,"message":"ok"}),
    )]);
    let created = client
        .glm_knowledge()
        .create_knowledge(
            "glm-account",
            &GlmCreateKnowledgeRequest {
                embedding_id: GlmKnowledgeEmbedding::Embedding3,
                name: "Product guides".into(),
                embedding_model: None,
                contextual: Some(GlmKnowledgeContextual::Enabled),
                description: Some("Product documentation".into()),
                background: None,
                icon: None,
            },
            &options("account-a"),
        )
        .await
        .unwrap();
    assert_eq!(created.reference, knowledge_ref("account-a"));

    let sent = mock.requests.lock().unwrap();
    assert_eq!(
        sent[0]
            .headers
            .iter()
            .find(|(name, _)| name == "authorization")
            .unwrap()
            .1,
        "Bearer zhipu-key-for-tests"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&sent[0].body).unwrap(),
        json!({"embedding_id":11,"name":"Product guides","contextual":1,"description":"Product documentation"})
    );
}

#[tokio::test]
async fn lists_knowledge_bases_and_returns_account_scoped_references() {
    let (client, mock) = setup(vec![reply(
        "GET",
        "https://open.bigmodel.cn/api/llm-application/open/knowledge?page=2&size=3",
        json!({
            "data":{"list":[{"id":"kb_123","name":"Returns"}],"total":8},
            "code":200,"message":"ok"
        }),
    )]);
    let result = client
        .glm_knowledge()
        .list_knowledge(
            "glm-account",
            &GlmKnowledgeListRequest {
                page: Some(2),
                size: Some(3),
            },
            &options("account-a"),
        )
        .await
        .unwrap();
    assert_eq!(result.total, 8);
    assert_eq!(result.knowledge_bases.len(), 1);
    assert_eq!(
        result.knowledge_bases[0].reference,
        knowledge_ref("account-a")
    );
    assert_eq!(result.knowledge_bases[0].native["name"], "Returns");
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn reads_updates_and_deletes_a_scoped_knowledge_base() {
    let (client, mock) = setup(vec![
        reply(
            "GET",
            "https://open.bigmodel.cn/api/llm-application/open/knowledge/kb_123",
            json!({"data":{"id":"kb_123","name":"Returns"},"code":200,"message":"ok"}),
        ),
        reply(
            "PUT",
            "https://open.bigmodel.cn/api/llm-application/open/knowledge/kb_123",
            json!({"code":200,"message":"updated"}),
        ),
        reply(
            "DELETE",
            "https://open.bigmodel.cn/api/llm-application/open/knowledge/kb_123",
            json!({"code":200,"message":"deleted"}),
        ),
    ]);
    let knowledge = knowledge_ref("account-a");
    let detail = client
        .glm_knowledge()
        .get_knowledge(&knowledge, &options("account-a"))
        .await
        .unwrap();
    assert_eq!(detail.reference, knowledge);
    assert_eq!(detail.native["data"]["name"], "Returns");

    client
        .glm_knowledge()
        .update_knowledge(
            &knowledge,
            &GlmUpdateKnowledgeRequest {
                contextual: Some(GlmKnowledgeContextual::Disabled),
                name: Some("Updated returns".into()),
                callback_header: Some(json!({"x-trace":"enabled"})),
                ..Default::default()
            },
            &options("account-a"),
        )
        .await
        .unwrap();
    let deleted = client
        .glm_knowledge()
        .delete_knowledge(&knowledge, &options("account-a"))
        .await
        .unwrap();
    assert_eq!(deleted.native["message"], "deleted");

    let sent = mock.requests.lock().unwrap();
    assert_eq!(sent[1].method, "PUT");
    assert_eq!(
        serde_json::from_slice::<Value>(&sent[1].body).unwrap(),
        json!({
            "contextual":0,
            "name":"Updated returns",
            "callback_header":{"x-trace":"enabled"}
        })
    );
    assert_eq!(sent[2].method, "DELETE");
}

#[tokio::test]
async fn lists_documents_with_references_bound_to_the_knowledge_base() {
    let (client, mock) = setup(vec![reply(
        "GET",
        "https://open.bigmodel.cn/api/llm-application/open/document?knowledge_id=kb_123&page=2&size=5&word=returns",
        json!({
            "data":{"list":[{
                "id":"doc_9","knowledge_type":5,"custom_separator":["=="],
                "sentence_size":400,"length":900,"word_num":120,"name":"returns.md",
                "url":"https://docs.example/returns","embedding_stat":0
            }],"total":1},
            "code":200,"message":"ok"
        }),
    )]);
    let knowledge = knowledge_ref("account-a");
    let result = client
        .glm_knowledge()
        .list_documents(
            &knowledge,
            &GlmKnowledgeDocumentListRequest {
                page: Some(2),
                size: Some(5),
                word: Some("returns".into()),
            },
            &options("account-a"),
        )
        .await
        .unwrap();
    assert_eq!(result.total, 1);
    let document = &result.documents[0];
    assert_eq!(document.reference.knowledge, knowledge);
    assert_eq!(document.reference.document_id, "doc_9");
    assert_eq!(document.name.as_deref(), Some("returns.md"));
    assert_eq!(document.embedding_stat, Some(0));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn management_operations_reject_references_from_another_account_before_http() {
    let (client, mock) = setup(vec![]);
    let knowledge = knowledge_ref("account-a");
    let mismatched = options("account-b");
    let update = GlmUpdateKnowledgeRequest {
        name: Some("new name".into()),
        ..Default::default()
    };
    let documents = GlmKnowledgeDocumentListRequest::default();

    for result in [
        client
            .glm_knowledge()
            .get_knowledge(&knowledge, &mismatched)
            .await
            .map(|_| ()),
        client
            .glm_knowledge()
            .update_knowledge(&knowledge, &update, &mismatched)
            .await
            .map(|_| ()),
        client
            .glm_knowledge()
            .delete_knowledge(&knowledge, &mismatched)
            .await
            .map(|_| ()),
        client
            .glm_knowledge()
            .list_documents(&knowledge, &documents, &mismatched)
            .await
            .map(|_| ()),
    ] {
        assert!(matches!(
            result,
            Err(GlmKnowledgeError::Llm(LlmError::PermissionDenied { .. }))
        ));
    }
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn transport_failures_leave_update_and_delete_outcomes_unknown() {
    let client = LlmClientBuilder::with_transport(Arc::new(FailingTransport), &[profile()])
        .with_region(Region::ChinaMainland)
        .build()
        .unwrap();
    let knowledge = knowledge_ref("account-a");
    let update = GlmUpdateKnowledgeRequest {
        name: Some("new name".into()),
        ..Default::default()
    };

    let update_error = client
        .glm_knowledge()
        .update_knowledge(&knowledge, &update, &options("account-a"))
        .await
        .unwrap_err();
    assert!(matches!(
        update_error,
        GlmKnowledgeError::OutcomeUnknown {
            operation: "update_knowledge",
            source: LlmError::TransportTimeout { .. }
        }
    ));

    let delete_error = client
        .glm_knowledge()
        .delete_knowledge(&knowledge, &options("account-a"))
        .await
        .unwrap_err();
    assert!(matches!(
        delete_error,
        GlmKnowledgeError::OutcomeUnknown {
            operation: "delete_knowledge",
            source: LlmError::TransportTimeout { .. }
        }
    ));
}

#[tokio::test]
async fn retrieval_uses_scoped_native_knowledge_and_document_ids() {
    let (client, mock) = setup(vec![reply(
        "POST",
        "https://open.bigmodel.cn/api/llm-application/open/knowledge/retrieve",
        json!({"data":[{
            "text":"Keep receipts for 30 days.",
            "score":0.91,
            "metadata":{"knowledge_id":"kb_123","doc_id":"doc_7","doc_name":"returns.md","doc_url":"https://docs.example/returns","contextual_text":"Returns policy"}
        }],"code":200,"message":"ok"}),
    )]);
    let knowledge = knowledge_ref("account-a");
    let request = GlmKnowledgeRetrieveRequest {
        query: "What is the return window?".into(),
        request_id: Some("req-9".into()),
        documents: vec![lingxi_llm_client::glm_knowledge::GlmKnowledgeDocumentRef {
            knowledge: knowledge.clone(),
            document_id: "doc_7".into(),
        }],
        top_k: Some(5),
        top_n: Some(20),
        recall_method: Some(GlmKnowledgeRecallMethod::Mixed),
        recall_ratio: Some(80),
        rerank: Some(true),
        rerank_model: Some(GlmKnowledgeRerankModel::RerankPro),
        fractional_threshold: Some(0.7),
    };
    let result = client
        .glm_knowledge()
        .retrieve(&[knowledge], &request, &options("account-a"))
        .await
        .unwrap();
    assert_eq!(result.matches[0].text, "Keep receipts for 30 days.");
    assert_eq!(
        result.matches[0].metadata.document_id.as_deref(),
        Some("doc_7")
    );
    assert_eq!(
        result.matches[0].metadata.document_name.as_deref(),
        Some("returns.md")
    );

    let sent = mock.requests.lock().unwrap();
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["knowledge_ids"], json!(["kb_123"]));
    assert_eq!(body["document_ids"], json!(["doc_7"]));
    assert_eq!(body["rerank_status"], json!(1));
    assert_eq!(body["rerank_model"], json!("rerank-pro"));
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value.starts_with("Bearer ")));
}

#[tokio::test]
async fn rejects_cross_account_references_before_http() {
    let (client, mock) = setup(vec![]);
    let error = client
        .glm_knowledge()
        .retrieve(
            &[knowledge_ref("account-a")],
            &GlmKnowledgeRetrieveRequest {
                query: "question".into(),
                request_id: None,
                documents: Vec::new(),
                top_k: None,
                top_n: None,
                recall_method: None,
                recall_ratio: None,
                rerank: None,
                rerank_model: None,
                fractional_threshold: None,
            },
            &options("account-b"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        GlmKnowledgeError::Llm(LlmError::PermissionDenied { .. })
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn rejects_retrieval_thresholds_outside_the_open_unit_interval() {
    let (client, mock) = setup(vec![]);
    let error = client
        .glm_knowledge()
        .retrieve(
            &[knowledge_ref("account-a")],
            &GlmKnowledgeRetrieveRequest {
                query: "question".into(),
                request_id: None,
                documents: Vec::new(),
                top_k: None,
                top_n: None,
                recall_method: None,
                recall_ratio: None,
                rerank: None,
                rerank_model: None,
                fractional_threshold: Some(0.0),
            },
            &options("account-a"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        GlmKnowledgeError::Llm(LlmError::InvalidRequest { .. })
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn maps_zhipu_business_errors_even_when_http_is_successful() {
    let mut failure = reply(
        "POST",
        "https://open.bigmodel.cn/api/llm-application/open/knowledge/retrieve",
        json!({"code":400,"message":"unknown knowledge id"}),
    );
    failure.status = 200;
    let (client, _) = setup(vec![failure]);
    let error = client
        .glm_knowledge()
        .retrieve(
            &[knowledge_ref("account-a")],
            &GlmKnowledgeRetrieveRequest {
                query: "question".into(),
                request_id: None,
                documents: Vec::new(),
                top_k: None,
                top_n: None,
                recall_method: None,
                recall_ratio: None,
                rerank: None,
                rerank_model: None,
                fractional_threshold: None,
            },
            &options("account-a"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        GlmKnowledgeError::Provider {
            status: 200,
            code: Some(400),
            ..
        }
    ));
}

#[test]
fn embedding_and_contextual_values_serialize_as_native_integers() {
    let request = GlmCreateKnowledgeRequest {
        embedding_id: GlmKnowledgeEmbedding::Embedding2,
        name: "test".into(),
        embedding_model: None,
        contextual: Some(GlmKnowledgeContextual::Disabled),
        description: None,
        background: None,
        icon: None,
    };
    assert_eq!(
        serde_json::to_value(request).unwrap(),
        json!({"embedding_id":3,"name":"test","contextual":0})
    );
}
