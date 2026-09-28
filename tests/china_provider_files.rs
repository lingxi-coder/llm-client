#[path = "support/wire_api.rs"]
mod wire_api;
use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::files::{
    FilePurpose, FileService, ModelFileReference, UploadFile, UploadFileStream,
};
use lingxi_llm_client::protocol::{
    ChatRequest, ContentBlock, ConversationMessage, LlmError, MessageRole, ProtocolFamily,
    ProviderFileSource, ProviderId, ProviderProfile, Secret, ToolChoice, VideoSource,
};
use lingxi_llm_client::{
    AnthropicMessagesCodec, BearerAuthenticator, HttpRequest, HttpResponse, LlmClientBuilder,
    RequestOptions, StreamResponse, Transport, WireCodec,
};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::Mutex;

#[derive(Default)]
struct QueueHttp {
    requests: Mutex<Vec<HttpRequest>>,
    replies: Mutex<VecDeque<HttpResponse>>,
}

impl QueueHttp {
    fn with_json(replies: Vec<Value>) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            replies: Mutex::new(
                replies
                    .into_iter()
                    .map(|body| HttpResponse {
                        status: 200,
                        headers: vec![],
                        body: body.to_string().into(),
                    })
                    .collect(),
            ),
        }
    }

    fn requests(&self) -> Vec<HttpRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for QueueHttp {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.response(request).await.map(Into::into)
    }
}
impl QueueHttp {
    async fn response(&self, request: HttpRequest) -> Result<HttpResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| LlmError::Transport {
                message: "missing scripted response".into(),
            })
    }
}

fn profile(
    provider: &str,
    base_url: &str,
    protocol: &str,
    model: &str,
    modalities: &[&str],
) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": provider,
        "profile_name": format!("{provider}-test"),
        "base_url": base_url,
        "protocol": protocol,
        "auth": "bearer",
        "models": [{
            "display_model": model,
            "request_model": model,
            "billing_model": model,
            "metadata": {"inputModalities": modalities, "attachments": true},
            "capability_support": {"vision": "supported", "documents": "supported", "tools": "supported", "reasoning": "unknown", "signed_reasoning": "unknown", "streaming": "supported", "structured_output": "unknown"}
        }]
    }))
    .unwrap()
}

fn upload(filename: &str, media_type: &str, bytes: &'static [u8]) -> UploadFile {
    UploadFile {
        filename: filename.into(),
        media_type: media_type.into(),
        bytes: Bytes::from_static(bytes),
    }
}

#[tokio::test]
async fn qwen_long_files_use_file_extract_and_fileid_references() {
    let profile = profile(
        "qwen",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "open_ai_chat",
        "qwen-long",
        &["text", "file"],
    );
    let http = QueueHttp::with_json(vec![
        json!({"id":"file-fe-abc","filename":"report.pdf","purpose":"file-extract","bytes":3,"status":"processed"}),
        json!({"data":[{"id":"file-fe-abc","filename":"report.pdf","purpose":"file-extract"}],"has_more":false}),
        json!({"id":"file-fe-abc","filename":"report.pdf","purpose":"file-extract","status":"processed"}),
        json!({"id":"file-fe-abc","deleted":true}),
    ]);
    let auth = BearerAuthenticator;
    let key = Secret::new("dashscope-test-key".to_owned());
    let service = FileService::new(&http, &profile, Some(&auth), Some(&key), Some("account-1"));
    let caps = service.capabilities("qwen-long", "application/pdf");
    assert!(caps.upload);
    assert_eq!(caps.model_input, ModelFileReference::FileUri);

    let file = service
        .upload(
            &upload("report.pdf", "application/pdf", b"pdf"),
            FilePurpose::ModelInput,
        )
        .await
        .unwrap();
    assert_eq!(file.uri.as_deref(), Some("fileid://file-fe-abc"));
    assert_eq!(
        file.model_reference().purpose.as_deref(),
        Some("file-extract")
    );
    let page = service
        .list_for_purpose(FilePurpose::Extraction, None)
        .await
        .unwrap();
    assert_eq!(page.files[0].file.file_id, "file-fe-abc");
    service.get(&file).await.unwrap();
    service.delete(&file).await.unwrap();

    let requests = http.requests();
    assert_eq!(
        requests[0].url,
        "https://dashscope.aliyuncs.com/compatible-mode/v1/files"
    );
    assert!(String::from_utf8_lossy(&requests[0].body).contains("file-extract"));
    assert!(requests[1].url.contains("purpose=file-extract"));
    assert_eq!(
        requests[2].url,
        "https://dashscope.aliyuncs.com/compatible-mode/v1/files/file-fe-abc"
    );
    assert_eq!(requests[3].method, "DELETE");
}

#[tokio::test]
async fn minimax_video_file_ids_and_purpose_specific_management_are_preserved() {
    let profile = profile(
        "minimax",
        "https://api.minimaxi.com/anthropic",
        "anthropic_messages",
        "MiniMax-M3",
        &["text", "video"],
    );
    let http = QueueHttp::with_json(vec![
        json!({"file":{"file_id":9223372036854775808_u64,"filename":"clip.mp4","purpose":"video_understanding","bytes":3},"base_resp":{"status_code":0,"status_msg":"success"}}),
        json!({"files":[{"file_id":1234567890123456789_u64,"filename":"voice.wav","purpose":"voice_clone"}],"base_resp":{"status_code":0,"status_msg":"success"}}),
        json!({"base_resp":{"status_code":0,"status_msg":"success"}}),
    ]);
    let auth = BearerAuthenticator;
    let key = Secret::new("minimax-test-key".to_owned());
    let service = FileService::new(&http, &profile, Some(&auth), Some(&key), Some("account-2"));
    let caps = service.capabilities("MiniMax-M3", "video/mp4");
    assert!(caps.upload);
    assert_eq!(caps.model_input, ModelFileReference::FileUri);
    let video_caps = service.capabilities_for_purpose(
        "MiniMax-M3",
        "video/mp4",
        FilePurpose::VideoUnderstanding,
    );
    assert_eq!(video_caps.model_input, ModelFileReference::FileUri);
    assert_eq!(
        video_caps.retention,
        Some(std::time::Duration::from_secs(7 * 24 * 60 * 60))
    );
    let generation_caps = service.capabilities_for_purpose(
        "MiniMax-M3",
        "video/mp4",
        FilePurpose::VideoGenerationInput,
    );
    assert_eq!(
        generation_caps.retention,
        Some(std::time::Duration::from_secs(7 * 24 * 60 * 60))
    );

    let video = service
        .upload(
            &upload("clip.mp4", "video/mp4", b"mp4"),
            FilePurpose::VideoUnderstanding,
        )
        .await
        .unwrap();
    assert_eq!(video.file_id, "9223372036854775808");
    assert_eq!(video.uri.as_deref(), Some("mm_file://9223372036854775808"));

    let listed = service
        .list_for_purpose(FilePurpose::VoiceClone, None)
        .await
        .unwrap();
    assert_eq!(listed.files[0].file.file_id, "1234567890123456789");
    service.delete(&listed.files[0].file).await.unwrap();

    let requests = http.requests();
    assert_eq!(requests[0].url, "https://api.minimaxi.com/v1/files/upload");
    assert!(String::from_utf8_lossy(&requests[0].body).contains("video_understanding"));
    assert_eq!(
        requests[1].url,
        "https://api.minimaxi.com/v1/files/list?purpose=voice_clone"
    );
    assert_eq!(requests[2].method, "POST");
    assert_eq!(requests[2].url, "https://api.minimaxi.com/v1/files/delete");
    let delete_body: Value = serde_json::from_slice(&requests[2].body).unwrap();
    assert_eq!(delete_body["file_id"], "1234567890123456789");
    assert_eq!(delete_body["purpose"], "voice_clone");
}

#[tokio::test]
async fn minimax_mainland_voice_audio_uses_cn_route_and_documented_upload_bounds() {
    let profile = profile(
        "minimax",
        "https://api.minimax.cn/v1",
        "anthropic_messages",
        "MiniMax-M3",
        &["text"],
    );
    let http = QueueHttp::with_json(vec![
        json!({"file":{"file_id":"clone-file","filename":"voice.wav","purpose":"voice_clone"},"base_resp":{"status_code":0,"status_msg":"success"}}),
        json!({"file":{"file_id":"prompt-file","filename":"prompt.mp3","purpose":"prompt_audio"},"base_resp":{"status_code":0,"status_msg":"success"}}),
    ]);
    let auth = BearerAuthenticator;
    let key = Secret::new("minimax-cn-test-key".to_owned());
    let service = FileService::new(&http, &profile, Some(&auth), Some(&key), Some("account-cn"));

    for purpose in [FilePurpose::VoiceClone, FilePurpose::PromptAudio] {
        let caps = service.capabilities_for_purpose("MiniMax-M3", "audio/wav", purpose);
        assert!(caps.upload);
        assert_eq!(caps.max_upload_bytes, Some(20_000_000));
    }

    let clone_file = service
        .upload(
            &upload("voice.wav", "application/octet-stream", b"wav-audio"),
            FilePurpose::VoiceClone,
        )
        .await
        .unwrap();
    let prompt_file = service
        .upload(
            &upload("prompt.mp3", "audio/mpeg", b"mp3-audio"),
            FilePurpose::PromptAudio,
        )
        .await
        .unwrap();
    assert_eq!(clone_file.account_scope.as_deref(), Some("account-cn"));
    assert_eq!(clone_file.purpose.as_deref(), Some("voice_clone"));
    assert_eq!(prompt_file.account_scope.as_deref(), Some("account-cn"));
    assert_eq!(prompt_file.purpose.as_deref(), Some("prompt_audio"));

    let requests = http.requests();
    assert_eq!(requests.len(), 2);
    for request in &requests {
        assert_eq!(request.url, "https://api.minimax.cn/v1/files/upload");
        assert_eq!(request.method, "POST");
    }
    assert!(
        String::from_utf8_lossy(&requests[0].body).contains("name=\"purpose\"\r\n\r\nvoice_clone")
    );
    assert!(
        String::from_utf8_lossy(&requests[1].body).contains("name=\"purpose\"\r\n\r\nprompt_audio")
    );

    let no_send_http = QueueHttp::default();
    let no_send = FileService::new(
        &no_send_http,
        &profile,
        Some(&auth),
        Some(&key),
        Some("account-cn"),
    );
    assert!(matches!(
        no_send
            .upload_stream(
                UploadFileStream::new(
                    "reference.wav",
                    "audio/wav",
                    20_000_001,
                    futures::stream::empty::<Result<Bytes, LlmError>>(),
                ),
                FilePurpose::VoiceClone,
            )
            .await,
        Err(lingxi_llm_client::files::FileUploadError::Llm(
            LlmError::RequestTooLarge { .. }
        ))
    ));
    assert!(no_send_http.requests().is_empty());
}

#[test]
fn minimax_m3_encodes_only_uploaded_video_file_refs() {
    let p = profile(
        "minimax",
        "https://api.minimax.io/anthropic",
        "anthropic_messages",
        "MiniMax-M3",
        &["text", "video"],
    );
    let client = LlmClientBuilder::with_transport(
        std::sync::Arc::new(QueueHttp::default()),
        std::slice::from_ref(&p),
    )
    .with_region(lingxi_llm_client::protocol::Region::International)
    .build()
    .unwrap();
    let route = client.resolve("MiniMax-M3").unwrap();
    let file = ProviderFileSource {
        protocol: ProtocolFamily::AnthropicMessages,
        provider_id: ProviderId::new("minimax"),
        profile_name: p.profile_name.clone(),
        endpoint_fingerprint: lingxi_llm_client::files::provider_file_endpoint_fingerprint(
            &p.base_url,
        ),
        account_scope: Some("account-2".into()),
        expires_at: None,
        processing_status: None,
        file_id: "1234567890123456789".into(),
        uri: Some("mm_file://1234567890123456789".into()),
        media_type: Some("video/mp4".into()),
        purpose: Some("video_understanding".into()),
    };
    let request = ChatRequest {
        prompt_cache: Default::default(),
        output_format: Default::default(),
        controls: Default::default(),
        service_tier: None,
        model: "MiniMax-M3".into(),
        anthropic_client_toolsets: Vec::new(),
        hosted_tools: vec![],
        continuation: None,
        system: vec![],
        messages: vec![ConversationMessage {
            anthropic: None,
            role: MessageRole::User,
            content: vec![ContentBlock::Video {
                source: VideoSource::ProviderFile { file },
            }],
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_tokens: Some(32),
        temperature: None,
        thinking: None,
        stop_sequences: vec![],
        metadata: Value::Null,
    };
    let options = RequestOptions {
        file_account_scope: Some("account-2".into()),
        ..RequestOptions::default()
    };
    let encoded = AnthropicMessagesCodec
        .encode_request(
            lingxi_llm_client::EncodeRequest::new(&request),
            &wire_api::context(&p, &route.request_model, &options),
        )
        .unwrap();
    let body: Value = serde_json::from_slice(&encoded.body).unwrap();
    assert_eq!(body["messages"][0]["content"][0]["type"], "video");
    assert_eq!(
        body["messages"][0]["content"][0]["source"]["url"],
        "mm_file://1234567890123456789"
    );
}

#[test]
fn qwen_long_singapore_does_not_offer_model_file_references() {
    let profile = profile(
        "qwen",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "open_ai_chat",
        "qwen-long",
        &["text", "file"],
    );
    let http = QueueHttp::with_json(vec![]);
    let service = FileService::new(&http, &profile, None, None, None);
    assert_eq!(
        service
            .capabilities("qwen-long", "application/pdf")
            .model_input,
        ModelFileReference::Unsupported
    );
}
