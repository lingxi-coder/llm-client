use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    transport::{HttpRequest, StreamResponse, Transport},
    xai_collections::{
        XaiCollectionRef, XaiCollectionsClient, XaiCollectionsConfig, XaiCollectionsCredentials,
        XaiDocumentRef,
    },
};
use std::sync::Mutex;

struct MockTransport {
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        Ok(StreamResponse {
            status: 200,
            headers: vec![("x-request-id".into(), "xai-reindex-1".into())],
            body: futures::stream::once(async { Ok(Bytes::from_static(b"{}")) }).boxed(),
        })
    }
}

fn credentials() -> XaiCollectionsCredentials {
    XaiCollectionsCredentials::new(
        Secret::new("xai-api-secret".to_owned()),
        Secret::new("xai-management-secret".to_owned()),
    )
}

#[tokio::test]
async fn reindex_document_uses_management_patch_contract() {
    let transport = MockTransport {
        requests: Mutex::new(Vec::new()),
    };
    let client = XaiCollectionsClient::new(
        &transport,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let document = XaiDocumentRef {
        collection: XaiCollectionRef {
            scope: client.scope().clone(),
            collection_id: "collection_alpha".into(),
        },
        file_id: "file_1".into(),
    };

    client
        .reindex_document(&document, &credentials())
        .await
        .unwrap();

    let requests = transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "PATCH");
    assert_eq!(
        requests[0].url,
        "https://management.test/v1/collections/collection_alpha/documents/file_1"
    );
    assert!(requests[0].body.is_empty());
    assert!(!requests[0]
        .headers
        .iter()
        .any(|(name, _)| name == "content-type"));
    assert!(requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value == "Bearer xai-management-secret"));
    assert!(!requests[0]
        .headers
        .iter()
        .any(|(name, value)| name == "authorization" && value.contains("xai-api-secret")));
}

#[tokio::test]
async fn reindex_rejects_document_from_another_account_before_dispatch() {
    let transport = MockTransport {
        requests: Mutex::new(Vec::new()),
    };
    let client = XaiCollectionsClient::new(
        &transport,
        XaiCollectionsConfig::new("xai-work", "team/account-1")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let foreign_client = XaiCollectionsClient::new(
        &transport,
        XaiCollectionsConfig::new("xai-work", "team/account-2")
            .with_api_base_url("https://api.test/v1")
            .with_management_base_url("https://management.test/v1"),
    )
    .unwrap();
    let foreign_document = XaiDocumentRef {
        collection: XaiCollectionRef {
            scope: foreign_client.scope().clone(),
            collection_id: "collection_alpha".into(),
        },
        file_id: "file_1".into(),
    };

    let result = client
        .reindex_document(&foreign_document, &credentials())
        .await;

    assert!(matches!(
        result,
        Err(lingxi_llm_client::xai_collections::XaiCollectionsError::InvalidRequest(_))
    ));
    assert!(transport.requests.lock().unwrap().is_empty());
}
