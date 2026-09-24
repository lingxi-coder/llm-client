use async_trait::async_trait;
use lingxi_llm_client::files::{
    provider_file_endpoint_fingerprint, FilePurpose, FileService, UploadFile,
};
use lingxi_llm_client::protocol::{
    CompletionRequest, ContentBlock, ConversationMessage, DocumentSource, LlmError, MessageRole,
    ProviderFileSource, ProviderProfile, Region, Secret,
};
use lingxi_llm_client::{
    ApiKeyAuthenticator, HttpRequest, HttpResponse, LlmClientBuilder, RequestOptions,
    StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Http {
    replies: Mutex<VecDeque<Value>>,
    requests: Mutex<Vec<HttpRequest>>,
}
impl Http {
    fn new(replies: Vec<Value>) -> Self {
        Self {
            replies: Mutex::new(replies.into()),
            requests: Mutex::new(vec![]),
        }
    }
}
#[async_trait]
impl Transport for Http {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.response(request).await.map(Into::into)
    }
}
impl Http {
    async fn response(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        let body = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request");
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: body.to_string().into(),
        })
    }
}
fn profile() -> ProviderProfile {
    serde_json::from_value(json!({"provider_id":"openai","profile_name":"openai","base_url":"https://api.openai.com/v1","protocol":"open_ai_responses","auth":"api_key","models":[{"display_model":"test-model","request_model":"test-model","billing_model":"test-model","metadata":{"input_modalities":["text","image","file"]},"capability_support": {"documents":"supported","vision":"supported","tools":"unknown","reasoning":"unknown","signed_reasoning":"unknown","streaming":"supported","structured_output":"unknown"}}]})).unwrap()
}
fn file(p: &ProviderProfile, purpose: &str) -> ProviderFileSource {
    ProviderFileSource {
        protocol: p.protocol,
        provider_id: p.provider_id.clone(),
        profile_name: p.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&p.base_url),
        account_scope: Some("scope".to_owned()),
        file_id: "file-abc".to_owned(),
        uri: None,
        media_type: Some("application/pdf".to_owned()),
        purpose: Some(purpose.to_owned()),
    }
}
fn req(file: ProviderFileSource) -> CompletionRequest {
    let mut r: CompletionRequest =
        serde_json::from_value(json!({"model":"test-model","messages":[]})).unwrap();
    r.messages = vec![ConversationMessage {
        role: MessageRole::User,
        content: vec![ContentBlock::Document {
            source: DocumentSource::ProviderFile { file },
            title: None,
        }],
    }];
    r
}
fn response() -> Value {
    json!({"id":"resp-1","model":"test-model","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":3,"output_tokens":1,"total_tokens":4}})
}
#[tokio::test]
async fn openai_file_purposes_and_refreshed_mime_remain_usable() {
    let p = profile();
    let key = Secret::new("fake-key".to_owned());
    let opts = RequestOptions {
        credential: Some(key.clone()),
        file_account_scope: Some("scope".to_owned()),
        ..RequestOptions::default()
    };
    for purpose in ["user_data", "assistants"] {
        let http = Arc::new(Http::new(vec![response()]));
        let client = LlmClientBuilder::with_transport(http.clone(), std::slice::from_ref(&p))
            .with_region(Region::International)
            .build()
            .unwrap();
        let result = client.complete(&req(file(&p, purpose)), &opts).await;
        assert!(result.is_ok(), "purpose {purpose}: {result:?}");
        assert_eq!(http.requests.lock().unwrap().len(), 1);
    }
    let http = Arc::new(Http::new(vec![
        json!({"id":"file-abc","object":"file","bytes":3,"created_at":1730000000,"filename":"report.pdf","purpose":"user_data"}),
        json!({"id":"file-abc","object":"file","bytes":3,"created_at":1730000000,"filename":"report.pdf","purpose":"user_data"}),
        response(),
    ]));
    let auth = ApiKeyAuthenticator;
    let service = FileService::new(http.as_ref(), &p, Some(&auth), Some(&key), Some("scope"));
    let uploaded = service
        .upload(
            &UploadFile {
                filename: "report.pdf".to_owned(),
                media_type: "application/pdf".to_owned(),
                bytes: vec![1, 2, 3].into(),
            },
            FilePurpose::ModelInput,
        )
        .await
        .unwrap();
    let refreshed = service.get(&uploaded).await.unwrap().file;
    assert_eq!(refreshed.media_type.as_deref(), Some("application/pdf"));
    let client = LlmClientBuilder::with_transport(http.clone(), std::slice::from_ref(&p))
        .with_region(Region::International)
        .build()
        .unwrap();
    let result = client
        .complete(&req(refreshed.model_reference()), &opts)
        .await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(http.requests.lock().unwrap().len(), 3);
}
