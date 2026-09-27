use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    transport::{HttpRequest, HttpStreamRequest, StreamResponse, Transport},
    xai_audio::{
        XaiAudioConfig, XaiAudioCredentials, XaiAudioError, XaiAudioService,
        XaiTranscriptionRequest, XaiTranscriptionUrl,
    },
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex, time::Duration};

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

enum Outcome {
    Response(Reply),
    Transport(LlmError),
}

struct SentRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    content_length: u64,
}

struct MockTransport {
    outcomes: Mutex<VecDeque<Outcome>>,
    sent: Mutex<Vec<SentRequest>>,
}

impl MockTransport {
    fn new(outcomes: impl IntoIterator<Item = Outcome>) -> Self {
        Self {
            outcomes: Mutex::new(outcomes.into_iter().collect()),
            sent: Mutex::new(Vec::new()),
        }
    }

    fn response(reply: Reply) -> StreamResponse {
        StreamResponse {
            status: reply.status,
            headers: reply.headers,
            body: stream::iter([Ok(Bytes::from(reply.body))]).boxed(),
        }
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(SentRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            content_length: request.body.len() as u64,
            body: request.body.to_vec(),
        });
        match self
            .outcomes
            .lock()
            .unwrap()
            .pop_front()
            .expect("an HTTP outcome is queued")
        {
            Outcome::Response(reply) => Ok(Self::response(reply)),
            Outcome::Transport(error) => Err(error),
        }
    }

    async fn send_stream(&self, _request: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            message: "URL transcription uses a bounded regular HTTP request".into(),
        })
    }
}

fn response(status: u16, headers: Vec<(String, String)>, body: Value) -> Outcome {
    Outcome::Response(Reply {
        status,
        headers,
        body: serde_json::to_vec(&body).unwrap(),
    })
}

fn credentials(key: &str) -> XaiAudioCredentials {
    XaiAudioCredentials::new(Secret::new(key.to_owned()))
}

fn service(transport: &MockTransport) -> XaiAudioService<'_> {
    XaiAudioService::new(
        transport,
        XaiAudioConfig::new("work", "team/account-7")
            .with_api_base_url("http://127.0.0.1:8391/v1")
            .with_request_timeout(Duration::from_secs(2)),
    )
    .unwrap()
}

#[tokio::test]
async fn transcribe_url_sends_the_exact_multipart_source_and_scopes_the_result() {
    let source_value = "https://media.example/audio.wav?signature=abc%2Fdef&expires=123";
    let source = XaiTranscriptionUrl::new(source_value).unwrap();
    assert!(!format!("{source:?}").contains("signature"));
    assert!(!format!("{source:?}").contains("abc%2Fdef"));

    let transport = MockTransport::new([response(
        200,
        vec![("x-request-id".into(), "url-stt-1".into())],
        json!({"text":"remote transcript","language":"en","duration":1.5}),
    )]);
    let service = service(&transport);
    let request = XaiTranscriptionRequest {
        language: Some("en".into()),
        format: true,
        keyterms: vec!["example".into()],
        ..XaiTranscriptionRequest::default()
    };
    let transcript = service
        .transcribe_url(&source, &request, &credentials("xai-test-key"))
        .await
        .unwrap();

    assert_eq!(transcript.text, "remote transcript");
    assert_eq!(transcript.duration_seconds, Some(1.5));
    assert_eq!(transcript.scope.account_scope, "team/account-7");
    assert_eq!(transcript.request_id.as_deref(), Some("url-stt-1"));

    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent.len(), 1);
    let call = &sent[0];
    assert_eq!(call.method, "POST");
    assert_eq!(call.url, "http://127.0.0.1:8391/v1/stt");
    assert_eq!(call.content_length, call.body.len() as u64);
    assert!(call.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("authorization") && value == "Bearer xai-test-key"
    }));
    let multipart = String::from_utf8_lossy(&call.body);
    let model = multipart.find("name=\"model\"").unwrap();
    let keyterm = multipart.find("name=\"keyterm\"").unwrap();
    let url_field = multipart.find("name=\"url\"").unwrap();
    assert!(model < keyterm && keyterm < url_field);
    let content_type = call
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .map(|(_, value)| value)
        .unwrap();
    let boundary = content_type.split_once("boundary=").unwrap().1;
    let url_value_start = url_field + "name=\"url\"\r\n\r\n".len();
    let closing = format!("\r\n--{boundary}--\r\n");
    let url_value_end = multipart.rfind(&closing).unwrap();
    assert_eq!(&multipart[url_value_start..url_value_end], source_value);
    assert!(multipart.ends_with(&closing));
    assert!(multipart.contains("name=\"format\"\r\n\r\ntrue"));
    assert!(!multipart.contains("name=\"file\""));
}

#[test]
fn transcription_url_accepts_http_and_https_but_rejects_ambiguous_or_unsafe_values() {
    assert!(XaiTranscriptionUrl::new("https://media.example/audio.wav?sig=a+b").is_ok());
    assert!(XaiTranscriptionUrl::new("http://media.example/audio.wav").is_ok());

    for value in [
        "file:///tmp/audio.wav",
        "ftp://media.example/audio.wav",
        "https:///audio.wav",
        "https://user:secret@media.example/audio.wav",
        "https://media.example/audio.wav#fragment",
        " https://media.example/audio.wav",
        "https://media.example/audio.wav\r\nfield",
    ] {
        let error = XaiTranscriptionUrl::new(value).unwrap_err();
        assert!(matches!(error, XaiAudioError::InvalidRequest(_)));
        assert!(!format!("{error:?}").contains(value));
    }
}

#[tokio::test]
async fn transcribe_url_preflight_and_provider_errors_do_not_echo_signed_urls() {
    let source_value = "https://media.example/audio.wav?signature=must-not-appear-in-error";
    let source = XaiTranscriptionUrl::new(source_value).unwrap();
    let transport = MockTransport::new([response(
        502,
        vec![("request-id".into(), "url-fetch-failed".into())],
        json!({
            "error": {
                "message": "URL download failed",
                "requested": source_value
            }
        }),
    )]);
    let service = service(&transport);
    let preflight = service
        .transcribe_url(
            &source,
            &XaiTranscriptionRequest::default(),
            &credentials("\n"),
        )
        .await
        .unwrap_err();
    assert!(matches!(preflight, XaiAudioError::InvalidRequest(_)));
    assert!(transport.sent.lock().unwrap().is_empty());

    let error = service
        .transcribe_url(
            &source,
            &XaiTranscriptionRequest::default(),
            &credentials("xai-test-key"),
        )
        .await
        .unwrap_err();
    let error_debug = format!("{error:?}");
    assert!(!error_debug.contains("must-not-appear-in-error"));
    match error {
        XaiAudioError::Provider {
            operation,
            status,
            request_id,
            body,
        } => {
            assert_eq!(operation, "speech-to-text");
            assert_eq!(status, 502);
            assert_eq!(request_id.as_deref(), Some("url-fetch-failed"));
            assert_eq!(body["error"]["requested"], "<redacted audio URL>");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn transcribe_url_reports_unknown_dispatch_on_transport_failure_without_retry() {
    let source_value = "https://media.example/audio.wav?transport_timeout";
    let source = XaiTranscriptionUrl::new(source_value).unwrap();
    let transport = MockTransport::new([Outcome::Transport(LlmError::TransportTimeout {
        message: format!("fetch failed for {source_value}"),
    })]);
    let error = service(&transport)
        .transcribe_url(
            &source,
            &XaiTranscriptionRequest::default(),
            &credentials("xai-test-key"),
        )
        .await
        .unwrap_err();

    match error {
        XaiAudioError::OutcomeUnknown {
            operation: "speech-to-text",
            source: LlmError::TransportTimeout { message },
            ..
        } => assert_eq!(message, "fetch failed for <redacted audio URL>"),
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(transport.sent.lock().unwrap().len(), 1);
}
