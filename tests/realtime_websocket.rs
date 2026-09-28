#![cfg(feature = "realtime-websocket")]

use futures::executor::block_on;
use lingxi_llm_client::realtime::{RealtimeConnectRequest, RealtimeError, RealtimeTransport};

fn request(endpoint: &str, headers: Vec<(String, String)>) -> RealtimeConnectRequest {
    RealtimeConnectRequest {
        endpoint: endpoint.into(),
        headers,
        max_frame_bytes: 4096,
    }
}

#[tokio::test]
async fn rustls_transport_rejects_plaintext_endpoints_before_connecting() {
    let error = lingxi_llm_client::HttpTransport::new()
        .unwrap()
        .connect(request(
            "ws://example.invalid/realtime?token=do-not-log",
            vec![],
        ))
        .await
        .err()
        .expect("plaintext WebSocket endpoints are rejected");

    assert!(matches!(error, RealtimeError::InvalidConfig { .. }));
    assert!(!error.to_string().contains("do-not-log"));
}

#[tokio::test]
async fn rustls_transport_rejects_endpoint_user_info_without_exposing_it() {
    let error = lingxi_llm_client::HttpTransport::new()
        .unwrap()
        .connect(request(
            "wss://user:secret@example.invalid/realtime",
            vec![],
        ))
        .await
        .err()
        .expect("credentials must be passed as headers, not URL user-info");

    assert!(matches!(error, RealtimeError::InvalidConfig { .. }));
    assert!(!error.to_string().contains("secret"));
}

#[tokio::test]
async fn rustls_transport_rejects_invalid_headers_without_exposing_values() {
    let error = lingxi_llm_client::HttpTransport::new()
        .unwrap()
        .connect(request(
            "wss://example.invalid/realtime",
            vec![(
                "Authorization".into(),
                "Bearer secret\r\nInjected: yes".into(),
            )],
        ))
        .await
        .err()
        .expect("invalid header values are rejected before connecting");

    assert!(matches!(error, RealtimeError::InvalidConfig { .. }));
    assert!(!error.to_string().contains("secret"));
}

#[test]
fn rustls_transport_requires_a_tokio_runtime() {
    let error = block_on(
        lingxi_llm_client::HttpTransport::new()
            .unwrap()
            .connect(request("wss://example.invalid/realtime", vec![])),
    )
    .err()
    .expect("transport should fail without a Tokio runtime");

    assert!(error.to_string().contains("requires a Tokio runtime"));
}
