use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::xai_collections::{
    XaiCollectionsClient, XaiCollectionsConfig, XaiCollectionsCredentials, XaiCollectionsScope,
    XaiUploadedFile, XaiUploadedFileRef,
};
use lingxi_llm_client::{
    files::{provider_file_endpoint_fingerprint, FileService, UploadFile},
    protocol::{LlmError, ProtocolFamily, ProviderId, ProviderProfile, Secret},
    transport::{HttpRequest, StreamResponse, Transport},
    ApiKeyAuthenticator,
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

struct MockTransport {
    replies: Mutex<VecDeque<(u16, Value)>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl MockTransport {
    fn new(replies: Vec<(u16, Value)>) -> Self {
        Self {
            replies: Mutex::new(replies.into()),
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
        let (status, value) = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request");
        let body = if value.is_null() {
            Bytes::new()
        } else {
            Bytes::from(serde_json::to_vec(&value).unwrap())
        };
        Ok(StreamResponse {
            status,
            headers: vec![],
            body: futures::stream::once(async move { Ok(body) }).boxed(),
        })
    }
}

fn profile(
    provider_id: &str,
    profile_name: &str,
    base_url: &str,
    protocol: ProtocolFamily,
) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": provider_id,
        "profile_name": profile_name,
        "base_url": base_url,
        "protocol": serde_json::to_value(protocol).unwrap(),
        "auth": "api_key",
        "models": []
    }))
    .unwrap()
}

fn uploaded_file() -> XaiUploadedFile {
    let api_base_url = "https://api.x.ai/v1";
    XaiUploadedFile {
        reference: XaiUploadedFileRef {
            scope: XaiCollectionsScope {
                provider_id: ProviderId::new("xai"),
                profile_name: "grok-main".into(),
                api_endpoint_fingerprint: provider_file_endpoint_fingerprint(api_base_url),
                management_endpoint_fingerprint: provider_file_endpoint_fingerprint(
                    "https://management-api.x.ai/v1",
                ),
                account_scope: "team/account-1".into(),
            },
            file_id: "file_abc".into(),
        },
        filename: Some("guide.pdf".into()),
        size_bytes: Some(42),
        created_at_unix_seconds: Some(1_780_000_000),
        expires_at_unix_seconds: Some(1_900_000_000),
        native: json!({
            "filename": "guide.pdf",
            "bytes": 42,
            "expires_at": 1900000000,
            "status": "processing",
            "mime_type": "application/pdf",
            "purpose": "assistants",
            "downloadable": true
        }),
    }
}

#[tokio::test]
async fn uploaded_file_reference_bridges_to_file_service_get_and_delete() {
    let transport = MockTransport::new(vec![
        (
            200,
            json!({
                "id": "file_abc",
                "filename": "guide.pdf",
                "bytes": 42,
                "created_at": 1780000000,
                "expires_at": 1900000000,
                "status": "processing",
                "mime_type": "application/pdf",
                "purpose": "assistants",
                "downloadable": true
            }),
        ),
        (
            200,
            json!({
                "id": "file_abc",
                "filename": "guide.pdf",
                "bytes": 42,
                "status": "processing"
            }),
        ),
        (204, Value::Null),
    ]);
    let collections = XaiCollectionsClient::new(
        &transport,
        XaiCollectionsConfig::new("grok-main", "team/account-1")
            .with_api_base_url("https://api.x.ai/v1"),
    )
    .unwrap();
    let credentials = XaiCollectionsCredentials::new(
        Secret::new("xai-api-key".to_owned()),
        Secret::new("xai-management-key".to_owned()),
    );
    let uploaded = collections
        .upload_file(
            &UploadFile {
                filename: "guide.pdf".into(),
                media_type: "application/pdf".into(),
                bytes: Bytes::from_static(b"pdf"),
            },
            &credentials,
        )
        .await
        .unwrap();

    // The trailing slash is equivalent after Collections URL normalization;
    // the returned fingerprint still satisfies FileService's profile check.
    let profile = profile(
        "xai",
        "grok-main",
        "https://api.x.ai/v1/",
        ProtocolFamily::OpenAiChat,
    );
    let reference = uploaded
        .to_provider_file_ref(&profile, "team/account-1")
        .unwrap();
    assert_eq!(reference.provider_id.as_str(), "xai");
    assert_eq!(reference.profile_name, "grok-main");
    assert_eq!(
        reference.endpoint_fingerprint,
        provider_file_endpoint_fingerprint(&profile.base_url)
    );
    assert_eq!(reference.account_scope.as_deref(), Some("team/account-1"));
    assert_eq!(reference.protocol, ProtocolFamily::OpenAiChat);
    assert_eq!(reference.file_id, "file_abc");
    assert_eq!(reference.filename.as_deref(), Some("guide.pdf"));
    assert_eq!(reference.media_type.as_deref(), Some("application/pdf"));
    assert_eq!(reference.size_bytes, Some(42));
    assert_eq!(reference.expires_at.as_deref(), Some("1900000000"));
    assert_eq!(reference.processing_status.as_deref(), Some("processing"));
    assert_eq!(reference.downloadable, Some(true));
    assert_eq!(reference.purpose.as_deref(), Some("assistants"));

    let auth = ApiKeyAuthenticator;
    let api_key = Secret::new("xai-api-key".to_owned());
    let service = FileService::new(
        &transport,
        &profile,
        Some(&auth),
        Some(&api_key),
        Some("team/account-1"),
    );
    let metadata = service.get(&reference).await.unwrap();
    assert_eq!(metadata.file.file_id, "file_abc");
    assert_eq!(metadata.file.expires_at.as_deref(), Some("1900000000"));
    service.delete(&reference).await.unwrap();

    let requests = transport.requests();
    assert_eq!(requests[0].url, "https://api.x.ai/v1/files");
    assert_eq!(requests[1].url, "https://api.x.ai/v1/files/file_abc");
    assert_eq!(requests[2].method, "DELETE");
    assert_eq!(requests[2].url, "https://api.x.ai/v1/files/file_abc");
}

#[test]
fn conversion_rejects_cross_account_profile_and_endpoint_references() {
    let uploaded = uploaded_file();
    let matching_profile = profile(
        "xai",
        "grok-main",
        "https://api.x.ai/v1",
        ProtocolFamily::OpenAiChat,
    );

    assert!(matches!(
        uploaded.to_provider_file_ref(&matching_profile, "another-account"),
        Err(LlmError::PermissionDenied { .. })
    ));

    let other_profile = profile(
        "xai",
        "grok-other",
        "https://api.x.ai/v1",
        ProtocolFamily::OpenAiChat,
    );
    assert!(matches!(
        uploaded.to_provider_file_ref(&other_profile, "team/account-1"),
        Err(LlmError::PermissionDenied { .. })
    ));

    let other_endpoint = profile(
        "xai",
        "grok-main",
        "https://api.x.ai/v2",
        ProtocolFamily::OpenAiChat,
    );
    assert!(matches!(
        uploaded.to_provider_file_ref(&other_endpoint, "team/account-1"),
        Err(LlmError::PermissionDenied { .. })
    ));
}

#[test]
fn conversion_preserves_supported_xai_protocol_identity() {
    let uploaded = uploaded_file();
    for protocol in [
        ProtocolFamily::OpenAiChat,
        ProtocolFamily::OpenAiResponses,
        ProtocolFamily::AnthropicMessages,
    ] {
        let profile = profile("xai", "grok-main", "https://api.x.ai/v1", protocol);
        let reference = uploaded
            .to_provider_file_ref(&profile, "team/account-1")
            .unwrap();
        assert_eq!(reference.protocol, protocol);
    }
}

#[test]
fn conversion_uses_string_fallbacks_when_primary_native_metadata_is_null() {
    let mut uploaded = uploaded_file();
    uploaded.native = json!({
        "mime_type": null,
        "content_type": "application/x-pdf",
        "status": null,
        "state": "PROCESSING"
    });
    let profile = profile(
        "xai",
        "grok-main",
        "https://api.x.ai/v1",
        ProtocolFamily::OpenAiChat,
    );

    let reference = uploaded
        .to_provider_file_ref(&profile, "team/account-1")
        .unwrap();
    assert_eq!(reference.media_type.as_deref(), Some("application/x-pdf"));
    assert_eq!(reference.processing_status.as_deref(), Some("PROCESSING"));
}

#[test]
fn conversion_rejects_non_xai_and_unsupported_profiles() {
    let uploaded = uploaded_file();
    let non_xai = profile(
        "openai",
        "grok-main",
        "https://api.x.ai/v1",
        ProtocolFamily::OpenAiChat,
    );
    assert!(matches!(
        uploaded.to_provider_file_ref(&non_xai, "team/account-1"),
        Err(LlmError::PermissionDenied { .. })
    ));

    let unsupported_protocol = profile(
        "xai",
        "grok-main",
        "https://api.x.ai/v1",
        ProtocolFamily::GeminiGenerateContent,
    );
    assert!(matches!(
        uploaded.to_provider_file_ref(&unsupported_protocol, "team/account-1"),
        Err(LlmError::UnsupportedCapability { .. })
    ));

    let unsupported_host = profile(
        "xai",
        "grok-main",
        "https://files-gateway.example/v1",
        ProtocolFamily::OpenAiChat,
    );
    let gateway_uploaded = XaiUploadedFile {
        reference: XaiUploadedFileRef {
            scope: XaiCollectionsScope {
                api_endpoint_fingerprint: provider_file_endpoint_fingerprint(
                    "https://files-gateway.example/v1",
                ),
                ..uploaded.reference.scope.clone()
            },
            ..uploaded.reference.clone()
        },
        ..uploaded
    };
    assert!(matches!(
        gateway_uploaded.to_provider_file_ref(&unsupported_host, "team/account-1"),
        Err(LlmError::UnsupportedCapability { .. })
    ));
}
