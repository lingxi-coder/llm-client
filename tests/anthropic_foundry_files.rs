use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    files::{FilePurpose, FileService, ProviderFileRef, UploadFile},
    protocol::{FoundryHosting, LlmError, ProviderProfile, Secret},
    ApiKeyAuthenticator, HttpRequest, HttpResponse, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

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
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
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

fn response(body: Value) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: vec![("content-type".into(), "application/json".into())],
        body: serde_json::to_vec(&body).unwrap().into(),
    }
}

fn raw_response(body: &[u8]) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: vec![("content-type".into(), "application/octet-stream".into())],
        body: Bytes::copy_from_slice(body),
    }
}

fn profile(base_url: &str, protocol: &str, models: Value) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "custom-foundry",
        "profile_name": "foundry-resource-a",
        "base_url": base_url,
        "protocol": protocol,
        "auth": "api_key",
        "models": models
    }))
    .unwrap()
}

fn model(hosting: &str) -> Value {
    json!({
        "display_model": "Claude selected",
        "request_model": "my-arbitrary-deployment",
        "billing_model": "claude-opus-5-5",
        "foundry": {
            "hosting": hosting,
            "model_id": "claude-opus-5-5"
        }
    })
}

fn file_metadata(id: &str, downloadable: bool) -> Value {
    json!({
        "id": id,
        "type": "file",
        "filename": "input.csv",
        "mime_type": "text/csv",
        "size_bytes": 3,
        "created_at": "2026-09-27T00:00:00Z",
        "downloadable": downloadable,
        "expires_at": null
    })
}

const ENDPOINT: &str = "https://resource-a.services.ai.azure.com/anthropic";
const ACCOUNT: &str = "foundry-account-a";

#[tokio::test]
async fn explicit_foundry_service_uses_resource_scoped_anthropic_files_routes() {
    let uploaded = file_metadata("file-input", false);
    let service_transport = MockTransport::with_responses(vec![
        response(uploaded.clone()),
        response(json!({ "data": [uploaded.clone()], "next_page": null })),
        response(uploaded.clone()),
        response(file_metadata("file-input", false)),
        response(file_metadata("file-output", true)),
        raw_response(b"generated output"),
        response(json!({})),
    ]);
    // The trailing slash is accepted, then canonically bound in returned refs.
    let mut profile = profile(
        "https://resource-a.services.ai.azure.com/anthropic/",
        "foundry_claude",
        json!([]),
    );
    profile.chat_enabled = false;
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("test-key".to_owned());
    let service = FileService::new_foundry(
        &service_transport,
        &profile,
        FoundryHosting::Anthropic,
        Some(&auth),
        Some(&key),
        ACCOUNT,
    )
    .unwrap();
    assert!(
        service
            .capabilities("arbitrary-deployment", "text/csv")
            .upload
    );
    assert!(
        !FileService::new(
            &service_transport,
            &profile,
            Some(&auth),
            Some(&key),
            Some(ACCOUNT)
        )
        .capabilities("arbitrary-deployment", "text/csv")
        .upload
    );

    let input = service
        .upload(
            &UploadFile {
                filename: "input.csv".into(),
                media_type: "text/csv".into(),
                bytes: Bytes::from_static(b"a,b"),
            },
            FilePurpose::ModelInput,
        )
        .await
        .unwrap();
    assert_eq!(
        input.protocol,
        lingxi_llm_client::protocol::ProtocolFamily::FoundryClaude
    );
    assert_eq!(input.provider_id.as_str(), "custom-foundry");
    assert_eq!(input.profile_name, "foundry-resource-a");
    assert_eq!(input.account_scope.as_deref(), Some(ACCOUNT));
    assert_eq!(
        input.endpoint_fingerprint,
        lingxi_llm_client::files::provider_file_endpoint_fingerprint(ENDPOINT)
    );
    let model_reference = input.model_reference();
    assert_eq!(
        model_reference.protocol,
        lingxi_llm_client::protocol::ProtocolFamily::FoundryClaude
    );
    assert_eq!(model_reference.account_scope.as_deref(), Some(ACCOUNT));
    assert_eq!(model_reference.file_id, "file-input");

    let mut absent = input.clone();
    absent.file_id = "file-absent".into();
    let page = service.list_by_ids(&[input.clone(), absent]).await.unwrap();
    assert_eq!(page.files.len(), 1);
    assert_eq!(page.files[0].file.file_id, "file-input");
    assert_eq!(
        page.files[0].file.protocol,
        lingxi_llm_client::protocol::ProtocolFamily::FoundryClaude
    );
    assert_eq!(page.next_cursor, None);
    assert_eq!(
        service.get(&input).await.unwrap().file.file_id,
        "file-input"
    );
    assert!(service.download(&input).await.is_err());

    let mut generated = input.clone();
    generated.file_id = "file-output".into();
    let content = service.download(&generated).await.unwrap();
    assert_eq!(content.bytes, Bytes::from_static(b"generated output"));
    service.delete(&generated).await.unwrap();

    let requests = service_transport.requests();
    assert_eq!(requests.len(), 7);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].url, format!("{ENDPOINT}/v1/files"));
    assert_eq!(requests[1].method, "GET");
    assert_eq!(
        requests[1].url,
        format!("{ENDPOINT}/v1/files?ids%5B%5D=file-input&ids%5B%5D=file-absent")
    );
    assert_eq!(requests[2].url, format!("{ENDPOINT}/v1/files/file-input"));
    assert_eq!(requests[3].url, format!("{ENDPOINT}/v1/files/file-input"));
    assert_eq!(requests[4].url, format!("{ENDPOINT}/v1/files/file-output"));
    assert_eq!(
        requests[5].url,
        format!("{ENDPOINT}/v1/files/file-output/content")
    );
    assert_eq!(requests[6].method, "DELETE");
    assert_eq!(requests[6].url, format!("{ENDPOINT}/v1/files/file-output"));
    for request in &requests {
        assert!(request.headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("x-api-key") && value == "test-key"
        }));
        assert!(request.headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("anthropic-version") && !value.is_empty()
        }));
    }
}

#[tokio::test]
async fn foundry_route_is_explicit_and_model_row_convenience_checks_membership_and_hosting() {
    let transport = MockTransport::default();
    let profile_without_models = profile(ENDPOINT, "foundry_claude", json!([]));
    let explicit = FileService::new_foundry(
        &transport,
        &profile_without_models,
        FoundryHosting::Anthropic,
        None,
        None,
        ACCOUNT,
    );
    assert!(explicit.is_ok());

    let profile_with_model = profile(ENDPOINT, "foundry_claude", json!([model("anthropic")]));
    let selected_model = profile_with_model.models[0].clone();
    assert!(FileService::new_foundry_for_model(
        &transport,
        &profile_with_model,
        &selected_model,
        None,
        None,
        ACCOUNT,
    )
    .is_ok());

    let no_identity_profile = profile(
        ENDPOINT,
        "foundry_claude",
        json!([{
            "display_model": "No hosting identity",
            "request_model": "my-arbitrary-deployment",
            "billing_model": "claude-opus-5-5"
        }]),
    );
    assert!(FileService::new_foundry_for_model(
        &transport,
        &no_identity_profile,
        &no_identity_profile.models[0],
        None,
        None,
        ACCOUNT,
    )
    .is_err());

    let duplicate_profile = profile(
        ENDPOINT,
        "foundry_claude",
        json!([model("anthropic"), model("anthropic")]),
    );
    assert!(FileService::new_foundry_for_model(
        &transport,
        &duplicate_profile,
        &duplicate_profile.models[0],
        None,
        None,
        ACCOUNT,
    )
    .is_ok());

    let mut foreign_model = profile(ENDPOINT, "foundry_claude", json!([model("anthropic")]));
    foreign_model.models[0].display_model = "absent-row".into();
    assert!(FileService::new_foundry_for_model(
        &transport,
        &profile_with_model,
        &foreign_model.models[0],
        None,
        None,
        ACCOUNT,
    )
    .is_err());

    let azure_profile = profile(ENDPOINT, "foundry_claude", json!([model("azure")]));
    assert!(FileService::new_foundry_for_model(
        &transport,
        &azure_profile,
        &azure_profile.models[0],
        None,
        None,
        ACCOUNT,
    )
    .is_err());
    assert!(FileService::new_foundry(
        &transport,
        &profile_without_models,
        FoundryHosting::Azure,
        None,
        None,
        ACCOUNT,
    )
    .is_err());
    assert!(FileService::new_foundry(
        &transport,
        &profile_without_models,
        FoundryHosting::Anthropic,
        None,
        None,
        "  ",
    )
    .is_err());
    for bad_endpoint in [
        "http://resource-a.services.ai.azure.com/anthropic",
        "https://resource-a.services.ai.azure.com/anthropic/v1",
        "https://resource-a.services.ai.azure.com/anthropic?x=1",
        "https://other.example/anthropic",
    ] {
        assert!(FileService::new_foundry(
            &transport,
            &profile(bad_endpoint, "foundry_claude", json!([])),
            FoundryHosting::Anthropic,
            None,
            None,
            ACCOUNT,
        )
        .is_err());
    }

    let mut first_party_profile =
        profile("https://api.anthropic.com", "anthropic_messages", json!([]));
    first_party_profile.profile_name = "first-party".into();
    assert!(FileService::new_foundry(
        &transport,
        &first_party_profile,
        FoundryHosting::Anthropic,
        None,
        None,
        ACCOUNT,
    )
    .is_err());

    // The legacy profile-only path deliberately cannot infer Foundry Files
    // support from a resource URL or protocol.
    let profile = profile(ENDPOINT, "foundry_claude", json!([]));
    assert!(
        FileService::new(&transport, &profile, None, None, Some(ACCOUNT))
            .list(None)
            .await
            .is_err()
    );
    assert!(transport.requests().is_empty());
}

#[tokio::test]
async fn foundry_ids_lookup_rejects_other_scopes_and_unrequested_response_ids() {
    let transport = MockTransport::with_responses(vec![response(json!({
        "data": [file_metadata("file-sneaky", false)],
        "next_page": null
    }))]);
    let profile = profile(ENDPOINT, "foundry_claude", json!([]));
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("test-key".into());
    let service = FileService::new_foundry(
        &transport,
        &profile,
        FoundryHosting::Anthropic,
        Some(&auth),
        Some(&key),
        ACCOUNT,
    )
    .unwrap();
    let requested = scoped_ref(&profile, "file-known", ACCOUNT, ENDPOINT);
    let mut other_account = requested.clone();
    other_account.account_scope = Some("other-account".into());
    assert!(service.list_by_ids(&[other_account]).await.is_err());
    assert!(service
        .list_by_ids(&[requested.clone(), requested.clone()])
        .await
        .is_err());
    assert!(service
        .list_by_ids(&vec![requested.clone(); 101])
        .await
        .is_err());
    assert!(transport.requests().is_empty());

    assert!(service.list_by_ids(&[requested]).await.is_err());
    assert_eq!(transport.requests().len(), 1);
}

#[tokio::test]
async fn anthropic_get_rejects_metadata_for_a_different_requested_id() {
    let foundry_transport =
        MockTransport::with_responses(vec![response(file_metadata("file-other", false))]);
    let foundry_profile = profile(ENDPOINT, "foundry_claude", json!([]));
    let auth = ApiKeyAuthenticator;
    let key = Secret::new("test-key".into());
    let foundry_service = FileService::new_foundry(
        &foundry_transport,
        &foundry_profile,
        FoundryHosting::Anthropic,
        Some(&auth),
        Some(&key),
        ACCOUNT,
    )
    .unwrap();
    let foundry_ref = scoped_ref(&foundry_profile, "file-requested", ACCOUNT, ENDPOINT);
    assert!(foundry_service.get(&foundry_ref).await.is_err());
    assert_eq!(foundry_transport.requests().len(), 1);

    let first_party_transport =
        MockTransport::with_responses(vec![response(file_metadata("file-other", false))]);
    let first_party_profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id": "anthropic",
        "profile_name": "anthropic-profile",
        "base_url": "https://api.anthropic.com",
        "protocol": "anthropic_messages",
        "auth": "none",
        "models": []
    }))
    .unwrap();
    let first_party_service = FileService::new(
        &first_party_transport,
        &first_party_profile,
        None,
        None,
        Some(ACCOUNT),
    );
    let first_party_ref = scoped_ref(
        &first_party_profile,
        "file-requested",
        ACCOUNT,
        "https://api.anthropic.com",
    );
    assert!(first_party_service.get(&first_party_ref).await.is_err());
    assert_eq!(first_party_transport.requests().len(), 1);
}

fn scoped_ref(
    profile: &ProviderProfile,
    file_id: &str,
    account_scope: &str,
    endpoint: &str,
) -> ProviderFileRef {
    ProviderFileRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
            endpoint,
        ),
        account_scope: Some(account_scope.into()),
        protocol: profile.protocol,
        file_id: file_id.into(),
        uri: None,
        filename: None,
        media_type: Some("text/csv".into()),
        size_bytes: None,
        expires_at: None,
        processing_status: None,
        downloadable: None,
        purpose: None,
    }
}
