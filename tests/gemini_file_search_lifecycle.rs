use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    files::provider_file_endpoint_fingerprint,
    protocol::*,
    providers::google::file_search::{
        GeminiFileSearchDocumentRef, GeminiFileSearchError, GeminiFileSearchStoreRef,
    },
    transport::{HttpRequest, StreamResponse, Transport},
    *,
};
use serde_json::json;
use std::sync::{Arc, Mutex};

struct Mock {
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl Transport for Mock {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        assert_eq!(request.method, "DELETE");
        self.requests.lock().unwrap().push(request);
        let bytes = serde_json::to_vec(&json!({})).unwrap();
        Ok(StreamResponse {
            status: 200,
            headers: Vec::new(),
            body: futures::stream::once(async move { Ok(Bytes::from(bytes)) }).boxed(),
        })
    }
}

const STORE_COLLECTION: &str = "https://generativelanguage.googleapis.com/v1beta/fileSearchStores";

fn options(account_scope: &str) -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("gemini-key".into())),
        account_scope: Some(account_scope.into()),
        ..Default::default()
    }
}

fn setup() -> (LlmClient, Arc<Mock>) {
    let mock = Arc::new(Mock {
        requests: Mutex::new(Vec::new()),
    });
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"google",
        "profile_name":"gemini",
        "base_url":"https://generativelanguage.googleapis.com/v1beta",
        "protocol":"gemini_generate_content",
        "auth":"api_key",
        "models":[],
        "gemini_file_search":{"mode":"enabled","value":{
            "endpoint":STORE_COLLECTION,
            "auth":{"type":"api_key","header":"x-goog-api-key"}
        }}
    }))
    .unwrap();
    let mut profile = profile;
    profile.regions = vec![Region::International];
    let client = LlmClientBuilder::with_transport(mock.clone(), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap();
    (client, mock)
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

#[tokio::test]
async fn deletes_enforce_account_scope_and_follow_google_force_semantics() {
    let (client, mock) = setup();
    let service_provider = client
        .provider::<lingxi_llm_client::providers::GoogleClient>("gemini")
        .unwrap();
    let service = service_provider.file_search();
    let store = store_ref();
    let document = GeminiFileSearchDocumentRef {
        store: store.clone(),
        document_id: "document-alpha".into(),
    };

    // A resource reference cannot be used to delete another account's data.
    assert!(matches!(
        service
            .delete_store(&store, false, &options("account-b"))
            .await,
        Err(GeminiFileSearchError::Llm(
            LlmError::PermissionDenied { .. }
        ))
    ));
    assert!(matches!(
        service
            .delete_document(&document, true, &options("account-b"))
            .await,
        Err(GeminiFileSearchError::Llm(
            LlmError::PermissionDenied { .. }
        ))
    ));
    assert!(mock.requests.lock().unwrap().is_empty());

    // Google's default is force=false (no query parameter); force=true also
    // removes the document's associated chunks.
    service
        .delete_store(&store, false, &options("account-a"))
        .await
        .unwrap();
    service
        .delete_document(&document, true, &options("account-a"))
        .await
        .unwrap();

    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].url,
        "https://generativelanguage.googleapis.com/v1beta/fileSearchStores/store-alpha"
    );
    assert_eq!(
        requests[1].url,
        "https://generativelanguage.googleapis.com/v1beta/fileSearchStores/store-alpha/documents/document-alpha?force=true"
    );
    assert_eq!(requests[0].headers[0].0, "x-goog-api-key");
    assert_eq!(requests[0].headers[0].1, "gemini-key");
}
