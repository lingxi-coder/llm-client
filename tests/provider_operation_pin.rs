use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{LlmError, ProviderProfile, Region, Secret},
    providers::{google::batch::GeminiBatchScope, GoogleClient},
    transport::HttpStreamRequest,
    HttpRequest, HttpResponse, LlmClientBuilder, StreamResponse, Transport,
};
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::sync::Notify;

struct UploadTransport {
    started: Notify,
    resume: Notify,
    sends: AtomicUsize,
}
#[async_trait]
impl Transport for UploadTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sends.fetch_add(1, Ordering::SeqCst);
        assert!(request.url.ends_with("/upload/v1beta/files"));
        self.started.notify_one();
        self.resume.notified().await;
        Ok(HttpResponse {
            status: 200,
            headers: vec![(
                "x-goog-upload-url".into(),
                "https://generativelanguage.googleapis.com/upload/session-1".into(),
            )],
            body: Bytes::new(),
        }
        .into())
    }
    async fn send_stream(
        &self,
        mut request: HttpStreamRequest,
    ) -> Result<StreamResponse, LlmError> {
        self.sends.fetch_add(1, Ordering::SeqCst);
        assert_eq!(
            request.url,
            "https://generativelanguage.googleapis.com/upload/session-1"
        );
        while let Some(chunk) = request.body.next().await {
            chunk?;
        }
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: json!({"file":{"name":"files/input-1"}}).to_string().into(),
        }
        .into())
    }
}
fn profile(provider: &str) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id":provider,"profile_name":"native","chat_enabled":false,
        "base_url":"https://chat.example.test/v1","protocol":"open_ai_chat","auth":"none","models":[]
    })).unwrap()
}
struct ConfigDir(std::path::PathBuf);
impl Drop for ConfigDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn one_native_upload_keeps_its_revision_across_both_http_requests() {
    let http = Arc::new(UploadTransport {
        started: Notify::new(),
        resume: Notify::new(),
        sends: AtomicUsize::new(0),
    });
    let (client, manager) = LlmClientBuilder::with_transport(http.clone(), &[profile("google")])
        .with_region(Region::International)
        .build_managed()
        .unwrap();
    let config_dir = ConfigDir(
        std::env::temp_dir().join(format!("llm-native-operation-pin-{}", std::process::id())),
    );
    manager.set_config_dir(&config_dir.0).await.unwrap();
    let google = client.provider::<GoogleClient>("native").unwrap();
    let batches = google
        .batch(
            GeminiBatchScope::new(
                "native",
                "account",
                "https://generativelanguage.googleapis.com/v1beta",
            )
            .unwrap(),
        )
        .unwrap();
    let credential = Secret::new("request-key".into());
    let upload = batches.upload_input_stream(
        "input.jsonl",
        3,
        stream::once(async { Ok(Bytes::from_static(b"{}\n")) }).boxed(),
        &credential,
    );
    let change = async {
        http.started.notified().await;
        manager.add_provider(profile("openai")).await.unwrap();
        http.resume.notify_one();
    };
    let (result, ()) = tokio::join!(upload, change);
    assert!(
        result.is_ok(),
        "the admitted upload must finish under its original revision: {result:?}"
    );
    assert_eq!(http.sends.load(Ordering::SeqCst), 2);
    let next = batches
        .upload_input_stream(
            "input.jsonl",
            3,
            stream::once(async { Ok(Bytes::from_static(b"{}\n")) }).boxed(),
            &credential,
        )
        .await;
    assert!(
        next.is_err(),
        "the next operation must see the changed provider identity"
    );
    assert_eq!(
        http.sends.load(Ordering::SeqCst),
        2,
        "changed binding must fail before network dispatch"
    );
}

#[tokio::test]
async fn explicit_request_account_cannot_cross_a_native_resource_scope() {
    use lingxi_llm_client::providers::{
        qwen::batch::{QwenBatchListOptions, QwenBatchRegion, QwenBatchScope},
        QwenClient,
    };
    let http = Arc::new(UploadTransport {
        started: Notify::new(),
        resume: Notify::new(),
        sends: AtomicUsize::new(0),
    });
    let client = LlmClientBuilder::with_transport(http.clone(), &[profile("qwen")])
        .with_region(Region::International)
        .build()
        .unwrap();
    let qwen = client.provider::<QwenClient>("native").unwrap();
    let service = qwen
        .batch(QwenBatchScope::new("native", "account-a", QwenBatchRegion::Beijing, None).unwrap())
        .unwrap();
    let options = lingxi_llm_client::RequestOptions {
        credential: Some(Secret::new("account-b-key".into())),
        account_scope: Some("account-b".into()),
        ..Default::default()
    };
    assert!(service
        .list(&QwenBatchListOptions::new(), &options)
        .await
        .is_err());
    assert_eq!(http.sends.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn typed_account_queries_share_profile_source_and_query_scope() {
    use lingxi_llm_client::account::{
        AccountFailure, AccountFetchContext, AccountIdentity, AccountQuery, AccountReport,
        AccountUsageSource,
    };
    use lingxi_llm_client::providers::QwenClient;
    struct Source(AtomicUsize);
    #[async_trait]
    impl AccountUsageSource for Source {
        async fn fetch(
            &self,
            context: &AccountFetchContext<'_>,
            _report: &mut AccountReport,
        ) -> Result<(), AccountFailure> {
            assert_eq!(context.profile.profile_name, "native");
            assert_eq!(
                context.query.selector.workspace_id.as_deref(),
                Some("workspace-a")
            );
            assert_eq!(
                context.query.credential.as_ref().unwrap().expose_secret(),
                "account-key"
            );
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }
    let http = Arc::new(UploadTransport {
        started: Notify::new(),
        resume: Notify::new(),
        sends: AtomicUsize::new(0),
    });
    let source = Arc::new(Source(AtomicUsize::new(0)));
    let mut builder = LlmClientBuilder::with_transport(http.clone(), &[profile("qwen")])
        .with_region(Region::International);
    builder.register_profile_account_source("native", AccountIdentity::ApiKey, source.clone());
    let client = builder.build().unwrap();
    let mut query = AccountQuery::new(AccountIdentity::ApiKey);
    query.credential = Some(Secret::new("account-key".into()));
    query.selector.workspace_id = Some("workspace-a".into());
    let unified = client.account_usage("native", &query).await.unwrap();
    let typed = client
        .provider::<QwenClient>("native")
        .unwrap()
        .account_usage(&query)
        .await
        .unwrap();
    assert_eq!(unified.profile_name, typed.profile_name);
    assert_eq!(unified.provider_id, typed.provider_id);
    assert_eq!(unified.balance, typed.balance);
    assert_eq!(source.0.load(Ordering::SeqCst), 2);
    assert_eq!(http.sends.load(Ordering::SeqCst), 0);
}
