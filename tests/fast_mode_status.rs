use async_trait::async_trait;
use lingxi_llm_client::protocol::{JsonPayload, LlmError};
use lingxi_llm_client::providers::anthropic::fast_mode::*;
use lingxi_llm_client::{HttpRequest, HttpResponse, StreamResponse, Transport};
use serde_json::json;
use std::sync::Mutex;
struct Capture {
    requests: Mutex<Vec<HttpRequest>>,
    status: u16,
    body: String,
}
#[test]
fn native_oauth_refresh_http_trigger() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/fast_oauth_2_1_287.json")).unwrap();
    for row in fixture["errors"].as_array().unwrap() {
        let error = FastModeStatusError::Http {
            status: row["input"]["status"].as_u64().unwrap() as u16,
            body: JsonPayload::from_generated_value(&row["input"]["body"]).unwrap(),
        };
        assert_eq!(
            error.requests_oauth_refresh(),
            row["expected"].as_bool().unwrap(),
            "{row}"
        );
    }
}
#[async_trait]
impl Transport for Capture {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.requests.lock().unwrap().push(request);
        Ok(HttpResponse {
            status: self.status,
            headers: vec![],
            body: self.body.as_bytes().to_vec().into(),
        }
        .into())
    }
}
#[tokio::test]
async fn native_status_headers_and_boolean_observations() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/fast_status_2_1_287.json")).unwrap();
    for (credential, oauth) in [
        (FastModeCredential::ApiKey("test-key"), false),
        (FastModeCredential::OAuth("test-oauth"), true),
    ] {
        for (enabled, expected) in [
            (json!(true), true),
            (json!(false), false),
            (json!("false"), true),
            (json!(0), false),
            (json!([]), true),
        ] {
            let capture = Capture {
                requests: Mutex::new(vec![]),
                status: 200,
                body: json!({"enabled":enabled,"disabled_reason":"unrecognized"}).to_string(),
            };
            let status = fetch_status(
                &capture,
                "https://api.anthropic.com",
                credential,
                "test-agent",
            )
            .await
            .unwrap();
            assert_eq!(status.enabled, expected);
            assert_eq!(
                status.disabled_reason.as_deref(),
                (!expected).then_some("unknown")
            );
            let requests = capture.requests.lock().unwrap();
            assert_eq!(requests.len(), 1);
            let req = &requests[0];
            let oracle = fixture["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["input"]["oauth"] == oauth && row["input"]["enabled"] == enabled)
                .unwrap();
            assert_eq!(req.url, oracle["expected"]["url"].as_str().unwrap());
            assert_eq!(
                status.enabled,
                oracle["expected"]["enabled"].as_bool().unwrap()
            );
            let native_headers = oracle["expected"]["headers"].as_object().unwrap();
            assert_eq!(req.headers.len(), native_headers.len());
            for (name, value) in &req.headers {
                assert_eq!(native_headers[name], *value);
            }
            assert_eq!(req.method, "GET");
            assert_eq!(
                req.url,
                "https://api.anthropic.com/api/claude_code_penguin_mode"
            );
            // HttpExecutor passes the remaining deadline to the transport.
            let timeout = req.timeout.unwrap();
            assert!(timeout <= std::time::Duration::from_secs(10));
            assert!(timeout > std::time::Duration::from_secs(9));
            assert!(req.body.is_empty());
            assert_eq!(req.headers[0], ("User-Agent".into(), "test-agent".into()));
            if oauth {
                assert_eq!(
                    req.headers[1],
                    ("Authorization".into(), "Bearer test-oauth".into())
                );
                assert_eq!(
                    req.headers[2],
                    ("anthropic-beta".into(), "oauth-2025-04-20".into())
                );
            } else {
                assert_eq!(req.headers[1], ("x-api-key".into(), "test-key".into()));
                assert_eq!(req.headers.len(), 2);
            }
        }
    }
}
#[tokio::test]
async fn native_disabled_reason_normalization() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/fast_status_2_1_287.json")).unwrap();
    for row in fixture["reasons"].as_array().unwrap() {
        let capture = Capture {
            requests: Mutex::new(vec![]),
            status: 200,
            body: json!({"enabled":false,"disabled_reason":row["input"]}).to_string(),
        };
        let status = fetch_status(
            &capture,
            "https://api.anthropic.com",
            FastModeCredential::ApiKey("test"),
            "test",
        )
        .await
        .unwrap();
        assert_eq!(status.disabled_reason.as_deref(), row["expected"].as_str());
    }
}
#[tokio::test]
async fn refusal_status_and_body_are_retained_without_implicit_retry() {
    let capture = Capture {
        requests: Mutex::new(vec![]),
        status: 403,
        body: json!({"error":{"type":"permission_error"}}).to_string(),
    };
    let result = fetch_status(
        &capture,
        "https://api.anthropic.com",
        FastModeCredential::OAuth("test"),
        "test",
    )
    .await;
    assert!(
        matches!(result,Err(FastModeStatusError::Http { status:403,body }) if serde_json::from_str::<serde_json::Value>(body.as_json_source()).unwrap()["error"]["type"]=="permission_error")
    );
    assert_eq!(capture.requests.lock().unwrap().len(), 1);
}
