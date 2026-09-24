#[path = "support/wire_api.rs"]
mod wire_api;
#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::{stream, StreamExt};
    use lingxi_llm_client::protocol::{
        CompletionRequest, LlmError, ProviderProfile, Region, StreamEvent,
    };
    use lingxi_llm_client::{
        HttpRequest, LlmClientBuilder, RequestOptions, StreamResponse, Transport,
    };
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct StreamHttp(Mutex<Option<StreamResponse>>);
    #[async_trait]
    impl Transport for StreamHttp {
        async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
            Ok(self.0.lock().unwrap().take().unwrap())
        }
    }

    async fn stream_for(
        body: futures::stream::BoxStream<'static, Result<Bytes, LlmError>>,
        protocol: &str,
    ) -> lingxi_llm_client::ModelStream {
        let profile: ProviderProfile = serde_json::from_value(json!({
            "provider_id":"test", "profile_name":"test", "protocol":protocol, "base_url":"https://test.invalid/v1", "auth":"none",
            "models":[{"display_model":"m","request_model":"m","billing_model":"m"}]
        })).unwrap();
        let http = Arc::new(StreamHttp(Mutex::new(Some(StreamResponse {
            status: 200,
            headers: vec![("content-type".into(), "text/event-stream".into())],
            body,
        }))));
        let client = LlmClientBuilder::with_transport(http, &[profile])
            .with_region(Region::International)
            .build()
            .unwrap();
        let req: CompletionRequest =
            serde_json::from_value(json!({"model":"m", "messages":[]})).unwrap();
        client
            .stream(&req, &RequestOptions::default())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn tail_text_should_arrive_before_interruption() {
        let payload = Bytes::from_static(
            br#"data: {"model":"m","choices":[{"delta":{"content":"last text"}}]}"#,
        );
        let mut s = stream_for(stream::iter(vec![Ok(payload)]).boxed(), "open_ai_chat").await;
        let first = s.next().await.unwrap();
        assert!(
            matches!(first, Ok(StreamEvent::Start { .. })),
            "expected decoded tail before interruption, got {first:?}"
        );
        let second = s.next().await.unwrap();
        assert!(matches!(second, Ok(StreamEvent::TextDelta { text, .. }) if text == "last text"));
        assert!(matches!(
            s.next().await,
            Some(Err(LlmError::StreamInterrupted { .. }))
        ));
        assert!(s.next().await.is_none());
        assert!(s.next().await.is_none());
    }

    struct Dropped(std::sync::Arc<std::sync::atomic::AtomicBool>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn terminal_markers_close_transport_and_keep_final_usage() {
        for (protocol, payload) in [
            ("open_ai_chat", concat!(
                "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5}}\n\n",
                "data: [DONE]\n\n")),
            ("open_ai_responses", "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":3,\"total_tokens\":5}}}\n\n"),
            ("anthropic_messages", concat!(
                "data: {\"type\":\"message_start\",\"message\":{\"model\":\"m\",\"usage\":{\"input_tokens\":2}}}\n\n",
                "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n",
                "data: {\"type\":\"message_stop\"}\n\n")),
        ] {
            let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let guard = Dropped(dropped.clone());
            let body = stream::iter(vec![Ok(Bytes::from(payload))]).chain(stream::pending()).inspect(move |_| { let _ = &guard; }).boxed();
            let mut s = stream_for(body, protocol).await;
            tokio::time::timeout(Duration::from_secs(1), async {
                loop {
                    if matches!(s.next().await.unwrap().unwrap(), StreamEvent::End { .. }) { break; }
                }
                assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
                assert!(s.next().await.is_none());
                assert!(s.next().await.is_none());
            }).await.expect("terminal marker must finish even while the socket stays open");
            let usage = crate::wire_api::observed_usage(&s).unwrap();
            assert_eq!(usage.input_tokens, 2);
            assert_eq!(usage.output_tokens, 3);
        }
    }
    #[tokio::test]
    async fn every_stream_failure_releases_transport_and_preserves_observed_usage() {
        let cases = vec![
            Ok(Bytes::from_static(b"data: {malformed}\n\n")),
            Ok(Bytes::from(vec![b'x'; 8 * 1024 * 1024 + 1])),
            Err(LlmError::Transport {
                message: "read failed".into(),
            }),
        ];
        for failure in cases {
            let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let guard = Dropped(dropped.clone());
            let body = stream::iter(vec![
                Ok(Bytes::from_static(b"data: {\"model\":\"m\",\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3}}\n\n")),
                failure,
            ]).chain(stream::pending()).inspect(move |_| { let _ = &guard; }).boxed();
            let mut response = stream_for(body, "open_ai_chat").await;
            assert!(matches!(
                response.next().await,
                Some(Ok(StreamEvent::Start { .. }))
            ));
            assert!(response.next().await.unwrap().is_err());
            assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
            assert!(response.next().await.is_none());
            assert!(response.next().await.is_none());
            assert_eq!(
                crate::wire_api::observed_usage(&response)
                    .unwrap()
                    .input_tokens,
                2
            );
        }
    }
    #[tokio::test]
    async fn a_terminal_batch_drops_the_body_before_the_first_queued_event_is_consumed() {
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let guard = Dropped(dropped.clone());
        let payload =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: {malformed}\n\n";
        let body = stream::iter(vec![Ok(Bytes::from_static(payload))])
            .chain(stream::pending())
            .inspect(move |_| {
                let _ = &guard;
            })
            .boxed();
        let mut response = stream_for(body, "open_ai_chat").await;
        assert!(response.next().await.unwrap().is_ok());
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
        let mut saw_error = false;
        while let Some(event) = response.next().await {
            if event.is_err() {
                saw_error = true;
            }
        }
        assert!(saw_error);
    }
}
