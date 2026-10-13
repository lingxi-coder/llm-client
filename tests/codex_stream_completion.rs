use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{ChatRequest, LlmError, ProviderProfile, Region, WebSearchConfig},
    HttpRequest, LlmClient, LlmClientBuilder, RequestMode, RequestOptions, StreamResponse,
    Transport,
};
use serde_json::{json, Value};
use std::sync::Arc;

struct StreamOnlyTransport(String);

#[async_trait]
impl Transport for StreamOnlyTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["stream"], true, "Codex rejects nonstream generation");
        assert_eq!(
            request.url,
            "https://chatgpt.com/backend-api/codex/responses"
        );
        let chunks = self
            .0
            .as_bytes()
            .chunks(7)
            .map(Bytes::copy_from_slice)
            .map(Ok)
            .collect::<Vec<_>>();
        Ok(StreamResponse {
            status: 200,
            headers: vec![("content-type".into(), "text/event-stream".into())],
            body: stream::iter(chunks).boxed(),
        })
    }
}

fn client(event: &str) -> LlmClient {
    // Deferred host authentication retains the endpoint but replaces auth.
    let profile: ProviderProfile = serde_json::from_value(json!({
        "provider_id":"custom", "profile_name":"codex", "auth":"none",
        "base_url":"https://chatgpt.com/backend-api/codex", "protocol":"open_ai_responses",
        "extra":{"web_search":"openai_responses"},
        "models":[{"display_model":"m","request_model":"m","billing_model":"m"}]
    }))
    .unwrap();
    LlmClientBuilder::with_transport(Arc::new(StreamOnlyTransport(event.into())), &[profile])
        .with_region(Region::International)
        .build()
        .unwrap()
}

fn terminal() -> String {
    format!(
        "data: {}\r\n\r\n",
        json!({
            "type":"response.completed", "response":{
                "id":"resp-search", "model":"m", "status":"completed",
                "output":[
                    {"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"Wuhan weather"}},
                    {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Weather report","annotations":[{"type":"url_citation","url":"https://example.com/weather","title":"Weather","start_index":0,"end_index":7}]}]}
                ],
                "usage":{"input_tokens":12,"output_tokens":4}
            }
        })
    )
}

#[tokio::test]
async fn complete_and_host_collected_calls_decode_codex_sse_with_usage_and_sources() {
    for prepared in [false, true] {
        let client = client(&terminal());
        let request = ChatRequest::new("m");
        let options = RequestOptions::default();
        let response = if prepared {
            let call = client
                .prepare_draft_on("codex", &request, &options, RequestMode::Complete)
                .await
                .unwrap()
                .seal()
                .await
                .unwrap();
            let collected = call.dispatch_once().await.unwrap().collect().await.unwrap();
            assert_eq!(
                collected
                    .usage_report()
                    .usage
                    .as_ref()
                    .unwrap()
                    .input_tokens,
                12
            );
            let response = collected.decode().unwrap();
            collected.finish().await;
            response
        } else {
            client
                .chat()
                .complete_in("codex", &request, &options)
                .await
                .unwrap()
        };
        assert_eq!(response.message.text(), "Weather report");
        assert_eq!(response.response_id.unwrap().as_str(), "resp-search");
        assert_eq!(response.usage.usage.unwrap().output_tokens, 4);
        assert_eq!(
            response.web_search.unwrap().citations[0].url,
            "https://example.com/weather"
        );
    }
}

#[tokio::test]
async fn codex_search_requires_the_standalone_endpoint() {
    let error = client(&terminal())
        .chat()
        .web_search_in(
            "codex",
            &ChatRequest::new("m"),
            WebSearchConfig::default(),
            &RequestOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, LlmError::UnsupportedCapability { message } if message.contains("standalone"))
    );
}

#[tokio::test]
async fn interrupted_and_failed_codex_completions_are_not_successes() {
    for event in [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r\"}}\n\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"invalid_request_error\",\"message\":\"bad search\"}}}\n\n",
    ] {
        for prepared in [false, true] {
            let client = client(event);
            let request = ChatRequest::new("m");
            let options = RequestOptions::default();
            let error = if prepared {
                client.prepare_draft_on("codex", &request, &options, RequestMode::Complete)
                    .await.unwrap().seal().await.unwrap().dispatch_once().await.unwrap()
                    .collect().await.err().unwrap()
            } else {
                client.chat().complete_in("codex", &request, &options).await.unwrap_err()
            };
            if event.contains("response.failed") {
                assert!(matches!(error.classification_source(), LlmError::ProviderInternal { message } if message.contains("bad search")), "{error:?}");
            } else {
                assert!(matches!(error.classification_source(), LlmError::StreamInterrupted { .. }), "{error:?}");
            }
        }
    }
}
