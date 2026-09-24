//! Deterministic 32 MiB attachment benchmark. All HTTP is mocked; no credentials or network.
use async_trait::async_trait;
use bytes::Bytes;
use lingxi_llm_client::{protocol::*, *};
use std::sync::Arc;
struct Source(Bytes);
#[async_trait]
impl AttachmentResolver for Source {
    async fn resolve(&self, _: &AttachmentRef) -> Result<Bytes, LlmError> {
        Ok(self.0.clone())
    }
}
struct Http;
#[async_trait]
impl Transport for Http {
    async fn send(&self, req: HttpRequest) -> Result<StreamResponse, LlmError> {
        let value = if req.url.ends_with("/files") {
            serde_json::json!({"id":"file-probe","object":"file","bytes":33554432,"filename":"probe.pdf","purpose":"user_data"})
        } else {
            serde_json::json!({"id":"resp-probe","model":"m","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}})
        };
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: value.to_string().into(),
        }
        .into())
    }
}

fn main() {
    let size = 32 * 1024 * 1024;
    let p:ProviderProfile=serde_json::from_value(serde_json::json!({"provider_id":"openai","profile_name":"probe","base_url":"https://api.openai.com/v1","protocol":"open_ai_responses","auth":"none","models":[{"display_model":"m","request_model":"m","billing_model":"m","metadata":{"inputModalities":["text","image","file"]}}]})).unwrap();
    let mut builder = LlmClientBuilder::with_transport(Arc::new(Http), &[p]);
    builder.with_attachment_resolver(Arc::new(Source(Bytes::from(vec![b'x'; size]))));
    let client = builder.with_region(Region::International).build().unwrap();
    let req:CompletionRequest=serde_json::from_value(serde_json::json!({"model":"m","messages":[{"role":"user","content":[{"type":"document","source":{"type":"attachment","attachment":{"attachment_id":"probe","revision":"1","filename":"probe.pdf","media_type":"application/pdf","size_bytes":size}}}]}]})).unwrap();
    let options = RequestOptions {
        file_account_scope: Some("probe-account".into()),
        ..Default::default()
    };
    futures::executor::block_on(async {
        for _ in 0..2 {
            client.complete(&req, &options).await.unwrap();
        }
    });
    println!("32 MiB PDF: upload then cached completion");
}
