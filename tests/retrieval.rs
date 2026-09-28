use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{files::*, protocol::*, providers::openai::retrieval::*, *};
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

struct Reply {
    method: &'static str,
    url: &'static str,
    body: Result<Value, LlmError>,
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
        let body = reply.body?;
        Ok(StreamResponse {
            status: 200,
            headers: vec![("x-request-id".into(), "request-1".into())],
            body: futures::stream::once(
                async move { Ok(serde_json::to_vec(&body).unwrap().into()) },
            )
            .boxed(),
        })
    }
}
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":"openai","profile_name":"openai",
        "base_url":"https://chat.test/v1","protocol":"open_ai_responses","auth":"none",
        "retrieval":{"mode":"enabled","value":{
            "api":"open_ai","endpoint":"https://retrieval.test/v1/vector_stores",
            "auth":{"type":"bearer"}
        }}
    }))
    .unwrap()
}
fn options(scope: &str) -> RequestOptions {
    RequestOptions {
        account_scope: Some(scope.into()),
        credential: Some(Secret::new("test-secret".into())),
        ..Default::default()
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
fn reply(method: &'static str, url: &'static str, body: Value) -> Reply {
    Reply {
        method,
        url,
        body: Ok(body),
    }
}
fn file() -> ProviderFileRef {
    ProviderFileRef {
        provider_id: "openai".into(),
        profile_name: "openai".into(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint("https://chat.test/v1"),
        account_scope: Some("account-a".into()),
        protocol: ProtocolFamily::OpenAiResponses,
        file_id: "file_123".into(),
        uri: None,
        filename: Some("faq.txt".into()),
        media_type: Some("text/plain".into()),
        size_bytes: None,
        expires_at: None,
        processing_status: None,
        downloadable: None,
        purpose: Some("assistants".into()),
    }
}
fn store_ref() -> RetrievalStoreRef {
    RetrievalStoreRef {
        provider_id: "openai".into(),
        profile_name: "openai".into(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(
            "https://retrieval.test/v1/vector_stores",
        ),
        account_scope: "account-a".into(),
        store_id: "vs_123".into(),
    }
}

#[test]
fn completed_batch_with_failed_files_is_not_ready() {
    let mut batch = IndexBatchTask {
        reference: IndexBatchRef {
            store: store_ref(),
            batch_id: "vsfb_1".into(),
        },
        status: IndexStatus::Completed,
        file_counts: IndexFileCounts {
            in_progress: 0,
            completed: 1,
            failed: 1,
            cancelled: 0,
            total: 2,
        },
        native: json!({}),
    };
    assert!(!batch.all_files_ready());
    batch.file_counts.failed = 0;
    batch.file_counts.completed = 2;
    assert!(batch.all_files_ready());
}

#[tokio::test]
async fn advanced_search_encodes_filter_ranking_and_rewrite() {
    let (client, mock) = setup(vec![reply(
        "POST",
        "https://retrieval.test/v1/vector_stores/vs_123/search",
        json!({"search_query":["refund deadline"],"has_more":false,"next_page":null,"data":[]}),
    )]);
    let mut request = SearchRequest::new("return policy", 20);
    request.filters = Some(RetrievalFilter::And {
        filters: vec![
            RetrievalFilter::Eq {
                key: "region".into(),
                value: json!("us"),
            },
            RetrievalFilter::Gte {
                key: "date".into(),
                value: json!(1704067200),
            },
        ],
    });
    request.ranking = Some(SearchRanking {
        ranker: Some(SearchRanker::Auto),
        score_threshold: Some(0.7),
    });
    request.rewrite_query = Some(true);
    let result = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .search_with(&store_ref(), &request, &options("account-a"))
        .await
        .unwrap();
    assert_eq!(result.search_query, json!(["refund deadline"]));
    let sent = mock.requests.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        serde_json::from_slice::<Value>(&sent[0].body).unwrap(),
        json!({
            "query":"return policy","max_num_results":20,"rewrite_query":true,
            "ranking_options":{"ranker":"auto","score_threshold":0.7},
            "filters":{"type":"and","filters":[
                {"type":"eq","key":"region","value":"us"},
                {"type":"gte","key":"date","value":1704067200}
            ]}
        })
    );
}

#[tokio::test]
async fn invalid_advanced_search_fails_before_http() {
    let (client, mock) = setup(vec![]);
    let mut request = SearchRequest::new("query", 10);
    request.filters = Some(RetrievalFilter::In {
        key: "region".into(),
        value: json!([]),
    });
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .retrieval()
            .search_with(&store_ref(), &request, &options("account-a"))
            .await,
        Err(RetrievalError::Llm(LlmError::InvalidRequest { .. }))
    ));
    request.filters = None;
    request.ranking = Some(SearchRanking {
        ranker: None,
        score_threshold: Some(f64::NAN),
    });
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .retrieval()
            .search_with(&store_ref(), &request, &options("account-a"))
            .await,
        Err(RetrievalError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn file_attributes_are_validated_and_sent_when_attaching() {
    let (client, mock) = setup(vec![reply(
        "POST",
        "https://retrieval.test/v1/vector_stores/vs_123/files",
        json!({"id":"file_123","vector_store_id":"vs_123","status":"in_progress"}),
    )]);
    let mut attrs = BTreeMap::from([("region".into(), json!("us"))]);
    client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .attach_file_with_attributes(&store_ref(), &file(), Some(&attrs), &options("account-a"))
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&mock.requests.lock().unwrap()[0].body).unwrap(),
        json!({"file_id":"file_123","attributes":{"region":"us"}})
    );
    attrs.insert("invalid".into(), json!({"nested":true}));
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .retrieval()
            .attach_file_with_attributes(&store_ref(), &file(), Some(&attrs), &options("account-a"))
            .await,
        Err(RetrievalError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn batch_index_lifecycle_preserves_status_and_file_pagination() {
    let batch = |status: &str| {
        json!({
            "id":"vsfb_1","vector_store_id":"vs_123","status":status,
            "file_counts":{"in_progress":1,"completed":1,"failed":0,"cancelled":0,"total":2}
        })
    };
    let (client, mock) = setup(vec![
        reply(
            "POST",
            "https://retrieval.test/v1/vector_stores/vs_123/file_batches",
            batch("in_progress"),
        ),
        reply(
            "GET",
            "https://retrieval.test/v1/vector_stores/vs_123/file_batches/vsfb_1",
            batch("in_progress"),
        ),
        reply(
            "GET",
            "https://retrieval.test/v1/vector_stores/vs_123/file_batches/vsfb_1/files?limit=1&after=file_old",
            json!({"has_more":true,"last_id":"file_123","data":[
                {"id":"file_123","vector_store_id":"vs_123","status":"completed"}
            ]}),
        ),
        reply(
            "POST",
            "https://retrieval.test/v1/vector_stores/vs_123/file_batches/vsfb_1/cancel",
            batch("cancelled"),
        ),
    ]);
    let mut other = file();
    other.file_id = "file_456".into();
    let entries = vec![
        BatchFileInput {
            file: file(),
            attributes: Some(BTreeMap::from([("region".into(), json!("us"))])),
            chunking: Some(FileChunking::Static {
                max_chunk_size_tokens: 1200,
                chunk_overlap_tokens: 200,
            }),
        },
        BatchFileInput {
            file: other,
            attributes: None,
            chunking: None,
        },
    ];
    let opts = options("account-a");
    let created = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .create_file_batch(&store_ref(), &entries, &opts)
        .await
        .unwrap();
    assert_eq!(created.status, IndexStatus::InProgress);
    let current = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .get_file_batch(&created.reference, &opts)
        .await
        .unwrap();
    assert_eq!(current.file_counts.total, 2);
    let page = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .list_batch_files(&created.reference, 1, Some("file_old"), &opts)
        .await
        .unwrap();
    assert!(page.has_more);
    assert_eq!(page.last_id.as_deref(), Some("file_123"));
    assert_eq!(page.files[0].status, IndexStatus::Completed);
    let cancelled = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .cancel_file_batch(&created.reference, &opts)
        .await
        .unwrap();
    assert_eq!(cancelled.status, IndexStatus::Cancelled);
    let sent = mock.requests.lock().unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&sent[0].body).unwrap(),
        json!({"files":[
            {"file_id":"file_123","attributes":{"region":"us"},
             "chunking_strategy":{"type":"static","static":{
                 "max_chunk_size_tokens":1200,"chunk_overlap_tokens":200}}},
            {"file_id":"file_456"}
        ]})
    );
}

#[tokio::test]
async fn batch_preflight_rejects_mixed_scope_and_bad_chunking_without_http() {
    let (client, mock) = setup(vec![]);
    let opts = options("account-a");
    let mut wrong = file();
    wrong.account_scope = Some("account-b".into());
    let inputs = [BatchFileInput {
        file: wrong,
        attributes: None,
        chunking: None,
    }];
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .retrieval()
            .create_file_batch(&store_ref(), &inputs, &opts)
            .await,
        Err(RetrievalError::Llm(LlmError::PermissionDenied { .. }))
    ));
    let invalid = [BatchFileInput {
        file: file(),
        attributes: None,
        chunking: Some(FileChunking::Static {
            max_chunk_size_tokens: 100,
            chunk_overlap_tokens: 51,
        }),
    }];
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .retrieval()
            .create_file_batch(&store_ref(), &invalid, &opts)
            .await,
        Err(RetrievalError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn batch_submission_transport_failure_has_unknown_outcome() {
    let (client, mock) = setup(vec![Reply {
        method: "POST",
        url: "https://retrieval.test/v1/vector_stores/vs_123/file_batches",
        body: Err(LlmError::Transport {
            message: "connection reset".into(),
        }),
    }]);
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .retrieval()
            .create_file_batch(
                &store_ref(),
                &[BatchFileInput {
                    file: file(),
                    attributes: None,
                    chunking: None,
                }],
                &options("account-a"),
            )
            .await,
        Err(RetrievalError::OutcomeUnknown {
            operation: "create_file_batch",
            ..
        })
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn store_file_listing_supports_reconciliation_and_cursor() {
    let (client, mock) = setup(vec![reply(
        "GET",
        "https://retrieval.test/v1/vector_stores/vs_123/files?limit=2&after=file_old",
        json!({"data":[
            {"id":"file_123","vector_store_id":"vs_123","status":"completed"},
            {"id":"file_456","vector_store_id":"vs_123","status":"failed",
             "last_error":{"code":"invalid_file","message":"bad format"}}
        ],"has_more":true,"last_id":"file_456"}),
    )]);
    let page = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .list_store_files(&store_ref(), 2, Some("file_old"), &options("account-a"))
        .await
        .unwrap();
    assert_eq!(page.files.len(), 2);
    assert_eq!(page.files[1].status, IndexStatus::Failed);
    assert_eq!(page.last_id.as_deref(), Some("file_456"));
    assert!(page.has_more);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn store_file_listing_encodes_status_sort_and_reverse_cursor() {
    let (client, mock) = setup(vec![reply(
        "GET",
        "https://retrieval.test/v1/vector_stores/vs_123/files?limit=10&before=file_new&filter=failed&order=asc",
        json!({"data":[],"has_more":false,"last_id":null}),
    )]);
    let request = IndexFileListRequest {
        limit: 10,
        before: Some("file_new".into()),
        status: Some(IndexStatus::Failed),
        order: Some(IndexFileOrder::Asc),
        ..Default::default()
    };
    let page = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .list_store_files_with(&store_ref(), &request, &options("account-a"))
        .await
        .unwrap();
    assert!(page.files.is_empty());
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn invalid_store_file_listing_fails_before_http() {
    let (client, mock) = setup(vec![]);
    let request = IndexFileListRequest {
        limit: 10,
        after: Some("file_old".into()),
        before: Some("file_new".into()),
        ..Default::default()
    };
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .retrieval()
            .list_store_files_with(&store_ref(), &request, &options("account-a"))
            .await,
        Err(RetrievalError::Llm(LlmError::InvalidRequest { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn file_attributes_can_be_replaced_without_reindexing() {
    let (client, mock) = setup(vec![reply(
        "POST",
        "https://retrieval.test/v1/vector_stores/vs_123/files/file_123",
        json!({"id":"file_123","vector_store_id":"vs_123","status":"completed",
            "attributes":{"region":"eu"}}),
    )]);
    let updated = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .update_file_attributes(
            &RetrievalFileRef {
                store: store_ref(),
                file_id: "file_123".into(),
            },
            &BTreeMap::from([("region".into(), json!("eu"))]),
            &options("account-a"),
        )
        .await
        .unwrap();
    assert_eq!(updated.native["attributes"]["region"], "eu");
    assert_eq!(
        serde_json::from_slice::<Value>(&mock.requests.lock().unwrap()[0].body).unwrap(),
        json!({"attributes":{"region":"eu"}})
    );
}

#[tokio::test]
async fn store_file_index_search_and_delete_are_separate_scoped_operations() {
    let base = "https://retrieval.test/v1/vector_stores";
    let (client, mock) = setup(vec![
        reply(
            "POST",
            base,
            json!({"id":"vs_123","name":"FAQ","status":"in_progress"}),
        ),
        reply(
            "POST",
            "https://retrieval.test/v1/vector_stores/vs_123/files",
            json!({
                "id":"file_123","vector_store_id":"vs_123","status":"in_progress"
            }),
        ),
        reply(
            "GET",
            "https://retrieval.test/v1/vector_stores/vs_123/files/file_123",
            json!({
                "id":"file_123","vector_store_id":"vs_123","status":"completed"
            }),
        ),
        reply(
            "POST",
            "https://retrieval.test/v1/vector_stores/vs_123/search",
            json!({
                "search_query":"return policy","has_more":false,"next_page":null,
                "data":[{"file_id":"file_123","filename":"faq.txt","score":0.91,
                    "attributes":{"kind":"faq"},"content":[{"type":"text","text":"30 days"}]}]
            }),
        ),
        reply(
            "DELETE",
            "https://retrieval.test/v1/vector_stores/vs_123/files/file_123",
            json!({
                "id":"file_123","deleted":true
            }),
        ),
        reply(
            "DELETE",
            "https://retrieval.test/v1/vector_stores/vs_123",
            json!({
                "id":"vs_123","deleted":true
            }),
        ),
    ]);
    let opts = options("account-a");
    let store = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .create_store("FAQ", &opts)
        .await
        .unwrap();
    assert_eq!(store.reference.store_id, "vs_123");
    let task = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .attach_file(&store.reference, &file(), &opts)
        .await
        .unwrap();
    assert_eq!(task.status, IndexStatus::InProgress);
    assert!(!task.status.is_ready());
    let ready = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .get_file(&task.reference, &opts)
        .await
        .unwrap();
    assert!(ready.status.is_ready());
    let result = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .search(&store.reference, "return policy", 10, &opts)
        .await
        .unwrap();
    assert_eq!(result.hits[0].file, task.reference);
    assert_eq!(result.hits[0].content[0]["text"], "30 days");
    assert_eq!(result.request_id.as_deref(), Some("request-1"));
    client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .delete_file(&ready.reference, &opts)
        .await
        .unwrap();
    client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .delete_store(&store.reference, &opts)
        .await
        .unwrap();
    let sent = mock.requests.lock().unwrap();
    assert_eq!(sent.len(), 6);
    assert!(sent.iter().all(|req| req
        .headers
        .iter()
        .any(|(key, value)| key == "authorization" && value == "Bearer test-secret")));
    assert_eq!(
        serde_json::from_slice::<Value>(&sent[1].body).unwrap(),
        json!({"file_id":"file_123"})
    );
}

#[tokio::test]
async fn resource_scope_and_uploaded_file_scope_are_checked_before_http() {
    let (client, mock) = setup(vec![]);
    let store = RetrievalStoreRef {
        provider_id: "openai".into(),
        profile_name: "openai".into(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(
            "https://retrieval.test/v1/vector_stores",
        ),
        account_scope: "account-a".into(),
        store_id: "vs_123".into(),
    };
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .retrieval()
            .search(&store, "query", 10, &options("account-b"))
            .await,
        Err(RetrievalError::Llm(LlmError::PermissionDenied { .. }))
    ));
    let mut wrong_file = file();
    wrong_file.account_scope = Some("account-b".into());
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .retrieval()
            .attach_file(&store, &wrong_file, &options("account-a"))
            .await,
        Err(RetrievalError::Llm(LlmError::PermissionDenied { .. }))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn unknown_create_outcome_is_explicit_and_never_resubmitted() {
    let (client, mock) = setup(vec![Reply {
        method: "POST",
        url: "https://retrieval.test/v1/vector_stores",
        body: Err(LlmError::Transport {
            message: "connection reset".into(),
        }),
    }]);
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .retrieval()
            .create_store("FAQ", &options("account-a"))
            .await,
        Err(RetrievalError::OutcomeUnknown {
            operation: "create_store",
            ..
        })
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn read_only_search_transport_failure_keeps_transport_classification() {
    let (client, mock) = setup(vec![Reply {
        method: "POST",
        url: "https://retrieval.test/v1/vector_stores/vs_123/search",
        body: Err(LlmError::Transport {
            message: "connection reset".into(),
        }),
    }]);
    let store = RetrievalStoreRef {
        provider_id: "openai".into(),
        profile_name: "openai".into(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(
            "https://retrieval.test/v1/vector_stores",
        ),
        account_scope: "account-a".into(),
        store_id: "vs_123".into(),
    };
    assert!(matches!(
        client
            .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
            .unwrap()
            .retrieval()
            .search(&store, "query", 10, &options("account-a"))
            .await,
        Err(RetrievalError::Llm(LlmError::Transport { .. }))
    ));
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn listing_can_reconcile_an_uncertain_create_and_retrieve_store_state() {
    let (client, _) = setup(vec![
        reply(
            "GET",
            "https://retrieval.test/v1/vector_stores?limit=2&after=vs_old",
            json!({"data":[{"id":"vs_123","name":"FAQ","status":"completed"}],
                "has_more":false,"last_id":"vs_123"}),
        ),
        reply(
            "GET",
            "https://retrieval.test/v1/vector_stores/vs_123",
            json!({"id":"vs_123","name":"FAQ","status":"completed",
                "file_counts":{"completed":1,"in_progress":0}}),
        ),
    ]);
    let opts = options("account-a");
    let (stores, has_more, last_id) = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .list_stores(2, Some("vs_old"), &opts)
        .await
        .unwrap();
    assert_eq!(stores.len(), 1);
    assert!(!has_more);
    assert_eq!(last_id.as_deref(), Some("vs_123"));
    let current = client
        .provider::<lingxi_llm_client::providers::OpenAiClient>("openai")
        .unwrap()
        .retrieval()
        .get_store(&stores[0].reference, &opts)
        .await
        .unwrap();
    assert_eq!(current.file_counts.unwrap()["completed"], 1);
}

#[test]
fn invalid_service_endpoints_are_rejected_when_building() {
    let mut p = profile();
    if let ServiceSetting::Enabled(route) = &mut p.retrieval {
        route.endpoint = "https://retrieval.test/v1/vector_stores?redirect=other".into();
    }
    assert!(matches!(
        LlmClientBuilder::with_transport(
            Arc::new(Mock {
                replies: Mutex::new(VecDeque::new()),
                requests: Mutex::new(vec![]),
            }),
            &[p]
        )
        .with_region(Region::International)
        .build(),
        Err(BuildError::InvalidService { .. })
    ));
}
