use async_trait::async_trait;
use futures::StreamExt;
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::{
        openai::decisions::{CreateRequest, Decisions},
        typesafe::{SystemOne, SystemOneRequest},
    },
    transport::{HttpRequest, StreamResponse, Transport},
    RequestOptions,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, sync::Mutex};
struct Mock {
    calls: Mutex<Vec<HttpRequest>>,
    status: u16,
    body: &'static str,
}

#[tokio::test]
async fn self_hosted_resource_works_over_real_http_without_authentication() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/v1/systemone", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        loop {
            let mut buffer = [0; 4096];
            let count = socket.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0);
            bytes.extend_from_slice(&buffer[..count]);
            assert!(bytes.len() < 65536);
            if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                let headers = std::str::from_utf8(&bytes[..end]).unwrap().to_lowercase();
                let length: usize = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .unwrap()
                    .trim()
                    .parse()
                    .unwrap();
                if bytes.len() < end + 4 + length {
                    continue;
                }
                assert!(headers.starts_with("post /v1/systemone http/1.1"));
                assert!(!headers.contains("authorization:"));
                let body: Value =
                    serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
                assert_eq!(body["state"]["evidence"], "local evidence");
                assert_eq!(body["questions"]["q"]["type"], "noul");
                break;
            }
        }
        let body = r#"{"answers":{"q":{"type":"noul","noul":0.75}}}"#;
        let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        socket.write_all(response.as_bytes()).await.unwrap();
    });
    let transport =
        lingxi_llm_client::HttpTransport::with_client_configurator(|builder| builder.no_proxy())
            .unwrap();
    let result = SystemOne::new(&transport, endpoint)
        .evaluate(
            &SystemOneRequest {
                model: "self-hosted".into(),
                state: json!({"evidence":"local evidence"}),
                questions: BTreeMap::from([(
                    "q".into(),
                    json!({"type":"noul","instructions":"Is the evidence valid?"}),
                )]),
                assets: BTreeMap::new(),
            },
            &RequestOptions {
                total_timeout: Some(std::time::Duration::from_secs(3)),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(result["answers"]["q"]["noul"], 0.75);
    server.await.unwrap();
}
#[async_trait]
impl Transport for Mock {
    async fn send(&self, r: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.calls.lock().unwrap().push(r);
        let bytes = self.body.as_bytes().to_vec().into();
        Ok(StreamResponse {
            status: self.status,
            headers: vec![],
            body: futures::stream::once(async move { Ok(bytes) }).boxed(),
        })
    }
}
#[tokio::test]
async fn native_resources_use_the_exact_host_endpoint_and_request_scoped_auth() {
    let mock = Mock {
        calls: Mutex::new(vec![]),
        status: 200,
        body: r#"{"model":"actual","answers":[]}"#,
    };
    let options = RequestOptions {
        credential: Some(Secret::new("private-key".into())),
        ..Default::default()
    };
    Decisions::new(&mock, "http://localhost:8000/v1/decisions")
        .create(
            &CreateRequest {
                model: "gpt-6-luna".into(),
                input: json!("test"),
                questions: vec![json!({"type":"predicate","name":"q","instructions":"test"})],
            },
            &options,
        )
        .await
        .unwrap();
    let calls = mock.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].url, "http://localhost:8000/v1/decisions");
    assert!(calls[0]
        .headers
        .contains(&("authorization".into(), "Bearer private-key".into())));
    let body: Value = serde_json::from_slice(&calls[0].body).unwrap();
    assert!(body.get("response_format").is_none());
    assert!(body.get("messages").is_none());
}
#[tokio::test]
async fn typesafe_is_a_native_resource_without_chat_configuration() {
    let mock = Mock {
        calls: Mutex::new(vec![]),
        status: 200,
        body: r#"{"answers":{"q":{"type":"noul","noul":0.9}}}"#,
    };
    SystemOne::new(&mock, "http://localhost:8000/v1/systemone")
        .evaluate(
            &SystemOneRequest {
                assets: BTreeMap::new(),
                model: "laya".into(),
                state: json!({"text":"test"}),
                questions: BTreeMap::from([(
                    "q".into(),
                    json!({"type":"noul","instructions":"test"}),
                )]),
            },
            &Default::default(),
        )
        .await
        .unwrap();
    assert_eq!(mock.calls.lock().unwrap().len(), 1);
}
#[tokio::test]
async fn provider_failures_preserve_status_without_exposing_response_body_or_replaying() {
    let mock = Mock {
        calls: Mutex::new(vec![]),
        status: 401,
        body: "secret-bearing failure",
    };
    let error = Decisions::new(&mock, "http://localhost/v1/decisions")
        .create(
            &CreateRequest {
                model: "m".into(),
                input: json!("t"),
                questions: vec![json!({"type":"predicate"})],
            },
            &Default::default(),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("401"));
    assert!(!format!("{error:?}").contains("secret-bearing"));
    assert_eq!(mock.calls.lock().unwrap().len(), 1);
}
#[tokio::test]
async fn invalid_endpoints_and_credentials_never_reach_transport() {
    let mock = Mock {
        calls: Mutex::new(vec![]),
        status: 200,
        body: "{}",
    };
    let request = CreateRequest {
        model: "m".into(),
        input: json!("t"),
        questions: vec![json!({"type":"predicate"})],
    };
    for endpoint in [
        "file:///tmp/secret",
        "http://user:secret@localhost/api",
        "http://localhost/api?key=secret",
    ] {
        assert!(Decisions::new(&mock, endpoint)
            .create(&request, &Default::default())
            .await
            .is_err());
    }
    let options = RequestOptions {
        credential: Some(Secret::new("key\r\ninvalid".into())),
        ..Default::default()
    };
    assert!(Decisions::new(&mock, "http://localhost/api")
        .create(&request, &options)
        .await
        .is_err());
    assert!(mock.calls.lock().unwrap().is_empty());
}
