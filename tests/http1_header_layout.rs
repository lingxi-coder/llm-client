//! Capture actual HTTP/1 bytes so HeaderMap normalization cannot hide regressions.
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::protocol::LlmError;
use lingxi_llm_client::transport::http_backend;
use lingxi_llm_client::{
    Http1HeaderLayout, HttpRequest, HttpStreamRequest, HttpTransport, Transport,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const DEADLINE: Duration = Duration::from_secs(3);

fn transport() -> HttpTransport {
    HttpTransport::with_client_configurator(|builder| builder.no_proxy()).unwrap()
}

fn headers(values: &[(&str, &str)]) -> Vec<(String, String)> {
    values
        .iter()
        .map(|(name, value)| ((*name).into(), (*value).into()))
        .collect()
}

fn request(
    url: String,
    layout: Option<Http1HeaderLayout>,
    fields: &[(&str, &str)],
    body: &'static [u8],
) -> HttpRequest {
    HttpRequest {
        method: "POST".into(),
        url,
        headers: headers(fields),
        http1_header_layout: layout,
        body: Bytes::from_static(body),
        timeout: Some(DEADLINE),
    }
}

fn response(extra_headers: &str, body: &[u8]) -> Vec<u8> {
    let mut wire = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n{extra_headers}\r\n",
        body.len()
    )
    .into_bytes();
    wire.extend_from_slice(body);
    wire
}

async fn read_request(socket: &mut TcpStream) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let mut buffer = [0; 1024];
        let count = socket.read(&mut buffer).await.unwrap();
        assert_ne!(count, 0, "connection ended before the complete request");
        bytes.extend_from_slice(&buffer[..count]);
        assert!(bytes.len() < 64 * 1024, "unexpectedly large test request");
        if let Some(end) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
            let head = std::str::from_utf8(&bytes[..end]).unwrap();
            assert!(!head.to_ascii_lowercase().contains("transfer-encoding:"));
            let length: usize = head
                .split("\r\n")
                .skip(1)
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().unwrap())
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                assert_eq!(bytes.len(), end + 4 + length);
                return bytes;
            }
        }
    }
}

async fn capture(response: Vec<u8>) -> (String, String, tokio::task::JoinHandle<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let authority = listener.local_addr().unwrap().to_string();
    let url = format!("http://{authority}/capture?exact=1");
    let task = tokio::spawn(async move {
        tokio::time::timeout(DEADLINE, async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let request = read_request(&mut socket).await;
            socket.write_all(&response).await.unwrap();
            request
        })
        .await
        .expect("local request capture timed out")
    });
    (url, authority, task)
}

fn expected_request(fields: &str, body: &[u8]) -> Vec<u8> {
    let mut wire = format!("POST /capture?exact=1 HTTP/1.1\r\n{fields}\r\n").into_bytes();
    wire.extend_from_slice(body);
    wire
}

#[tokio::test]
async fn preserve_keeps_original_case_and_interleaved_duplicate_occurrences() {
    let (url, authority, captured) = capture(response("", b"ok")).await;
    let reply = transport()
        .send(request(
            url,
            Some(Http1HeaderLayout::Preserve),
            &[
                ("X-Repeat", "first"),
                ("Accept", "*/*"),
                ("x-between", "middle"),
                ("x-repeat", "second"),
                ("X-Repeat", "third"),
                ("X-Empty", ""),
            ],
            b"payload",
        ))
        .await
        .unwrap();
    assert_eq!(reply.status, 200);
    assert_eq!(
        captured.await.unwrap(),
        expected_request(
            &format!(
                "X-Repeat: first\r\nAccept: */*\r\nx-between: middle\r\nx-repeat: second\r\nX-Repeat: third\r\nX-Empty: \r\nhost: {authority}\r\ncontent-length: 7\r\n"
            ),
            b"payload"
        )
    );
}

#[tokio::test]
async fn native_fetch_sorts_original_ascii_spellings_stably_then_adds_missing_tail() {
    let (url, authority, captured) = capture(response("", b"ok")).await;
    transport()
        .send(request(
            url,
            Some(Http1HeaderLayout::NativeFetch),
            &[
                ("Z-Last", "z"),
                ("x-repeat", "lowercase"),
                ("X-Repeat", "first uppercase"),
                ("a-First", "a"),
                ("X-Repeat", "second uppercase"),
                ("Accept", "*/*"),
            ],
            b"payload",
        ))
        .await
        .unwrap();
    assert_eq!(
        captured.await.unwrap(),
        expected_request(
            &format!(
                "Accept: */*\r\nX-Repeat: first uppercase\r\nX-Repeat: second uppercase\r\nZ-Last: z\r\na-First: a\r\nx-repeat: lowercase\r\nConnection: keep-alive\r\nHost: {authority}\r\nAccept-Encoding: gzip, deflate, br, zstd\r\nContent-Length: 7\r\n"
            ),
            b"payload"
        )
    );
}

#[tokio::test]
async fn explicit_automatic_fields_keep_their_values_spelling_and_single_occurrence() {
    for layout in [Http1HeaderLayout::Preserve, Http1HeaderLayout::NativeFetch] {
        let (url, _, captured) = capture(response("", b"ok")).await;
        transport()
            .send(request(
                url,
                Some(layout),
                &[
                    ("Host", "explicit.example:8080"),
                    ("accept-encoding", "identity"),
                    ("connection", "close"),
                    ("CONTENT-LENGTH", "7"),
                    ("Accept", "*/*"),
                ],
                b"payload",
            ))
            .await
            .unwrap();
        let fields = match layout {
            Http1HeaderLayout::Preserve => "Host: explicit.example:8080\r\naccept-encoding: identity\r\nconnection: close\r\nCONTENT-LENGTH: 7\r\nAccept: */*\r\n",
            Http1HeaderLayout::NativeFetch => "Accept: */*\r\nCONTENT-LENGTH: 7\r\nHost: explicit.example:8080\r\naccept-encoding: identity\r\nconnection: close\r\n",
        };
        assert_eq!(
            captured.await.unwrap(),
            expected_request(fields, b"payload"),
            "{layout:?}"
        );
    }
}

#[tokio::test]
async fn native_fetch_empty_fixed_body_has_an_explicit_zero_length() {
    let (url, authority, captured) = capture(response("", b"ok")).await;
    transport()
        .send(request(
            url,
            Some(Http1HeaderLayout::NativeFetch),
            &[("Accept", "*/*")],
            b"",
        ))
        .await
        .unwrap();
    assert_eq!(
        captured.await.unwrap(),
        expected_request(
            &format!(
                "Accept: */*\r\nConnection: keep-alive\r\nHost: {authority}\r\nAccept-Encoding: gzip, deflate, br, zstd\r\nContent-Length: 0\r\n"
            ),
            b""
        )
    );
}

fn invalid_header_cases() -> Vec<(&'static str, Vec<(&'static str, &'static str)>)> {
    vec![
        ("invalid field name", vec![("not a token", "value")]),
        (
            "field name injection",
            vec![("X-Name\r\nInjected", "value")],
        ),
        (
            "field value injection",
            vec![("X-Name", "value\r\nInjected: yes")],
        ),
        ("nul field value", vec![("X-Name", "value\0")]),
        ("wrong byte count", vec![("Content-Length", "8")]),
        ("nondecimal length", vec![("Content-Length", "seven")]),
        ("combined lengths", vec![("Content-Length", "7, 7")]),
        (
            "duplicate matching lengths",
            vec![("Content-Length", "7"), ("content-length", "7")],
        ),
        ("transfer encoding", vec![("Transfer-Encoding", "chunked")]),
        (
            "conflicting framing",
            vec![("Content-Length", "7"), ("transfer-encoding", "chunked")],
        ),
        (
            "duplicate authority",
            vec![("Host", "one.example"), ("host", "two.example")],
        ),
    ]
}

async fn assert_no_connection(listener: &TcpListener, case: &str) {
    assert!(
        tokio::time::timeout(Duration::from_millis(25), listener.accept())
            .await
            .is_err(),
        "{case}: invalid headers reached the network"
    );
}

#[tokio::test]
async fn invalid_fixed_headers_and_framing_are_rejected_before_connecting() {
    let http = transport();
    for layout in [Http1HeaderLayout::Preserve, Http1HeaderLayout::NativeFetch] {
        for (case, fields) in invalid_header_cases() {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/invalid", listener.local_addr().unwrap());
            let result = tokio::time::timeout(
                DEADLINE,
                http.send(request(url, Some(layout), &fields, b"payload")),
            )
            .await
            .expect("invalid fixed request did not return");
            assert!(
                matches!(result, Err(LlmError::InvalidRequest { .. })),
                "{layout:?}: {case} should be rejected locally"
            );
            assert_no_connection(&listener, case).await;
        }
    }
}

#[tokio::test]
async fn invalid_stream_headers_reject_before_connecting_or_polling_single_use_body() {
    let http = transport();
    for layout in [Http1HeaderLayout::Preserve, Http1HeaderLayout::NativeFetch] {
        for (case, fields) in invalid_header_cases() {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/invalid", listener.local_addr().unwrap());
            let polls = Arc::new(AtomicUsize::new(0));
            let observed = polls.clone();
            let result = tokio::time::timeout(
                DEADLINE,
                http.send_stream(HttpStreamRequest {
                    method: "POST".into(),
                    url,
                    headers: headers(&fields),
                    http1_header_layout: Some(layout),
                    body: stream::once(async move {
                        observed.fetch_add(1, Ordering::SeqCst);
                        Ok(Bytes::from_static(b"payload"))
                    })
                    .boxed(),
                    content_length: 7,
                    timeout: Some(DEADLINE),
                }),
            )
            .await
            .expect("invalid streamed request did not return");
            assert!(
                matches!(result, Err(LlmError::InvalidRequest { .. })),
                "{layout:?}: {case} should be rejected locally"
            );
            assert_eq!(polls.load(Ordering::SeqCst), 0, "{layout:?}: {case}");
            assert_no_connection(&listener, case).await;
        }
    }
}

#[tokio::test]
async fn layouts_preserve_exact_streamed_binary_body_and_fixed_framing() {
    for layout in [Http1HeaderLayout::Preserve, Http1HeaderLayout::NativeFetch] {
        for explicit_length in [false, true] {
            let (url, authority, captured) = capture(response("", b"ok")).await;
            let mut request_headers = headers(&[
                ("X-Repeat", "first"),
                ("Accept", "*/*"),
                ("x-between", "middle"),
                ("X-Repeat", "second"),
            ]);
            if explicit_length {
                request_headers.insert(1, ("cOnTeNt-LeNgTh".into(), "6".into()));
            }
            transport()
                .send_stream(HttpStreamRequest {
                    method: "POST".into(),
                    url,
                    headers: request_headers,
                    http1_header_layout: Some(layout),
                    body: stream::iter([
                        Ok(Bytes::from_static(b"\0\xff")),
                        Ok(Bytes::from_static(b"\r\nab")),
                    ])
                    .boxed(),
                    content_length: 6,
                    timeout: Some(DEADLINE),
                })
                .await
                .unwrap();
            let fields = match (layout, explicit_length) {
            (Http1HeaderLayout::Preserve, false) => format!(
                "X-Repeat: first\r\nAccept: */*\r\nx-between: middle\r\nX-Repeat: second\r\nhost: {authority}\r\ncontent-length: 6\r\n"
            ),
            (Http1HeaderLayout::Preserve, true) => format!(
                "X-Repeat: first\r\ncOnTeNt-LeNgTh: 6\r\nAccept: */*\r\nx-between: middle\r\nX-Repeat: second\r\nhost: {authority}\r\n"
            ),
            (Http1HeaderLayout::NativeFetch, false) => format!(
                "Accept: */*\r\nX-Repeat: first\r\nX-Repeat: second\r\nx-between: middle\r\nConnection: keep-alive\r\nHost: {authority}\r\nAccept-Encoding: gzip, deflate, br, zstd\r\nContent-Length: 6\r\n"
            ),
            (Http1HeaderLayout::NativeFetch, true) => format!(
                "Accept: */*\r\nX-Repeat: first\r\nX-Repeat: second\r\ncOnTeNt-LeNgTh: 6\r\nx-between: middle\r\nConnection: keep-alive\r\nHost: {authority}\r\nAccept-Encoding: gzip, deflate, br, zstd\r\n"
            ),
        };
            assert_eq!(
                captured.await.unwrap(),
                expected_request(&fields, b"\0\xff\r\nab"),
                "{layout:?}, explicit_length={explicit_length}"
            );
        }
    }
}

#[tokio::test]
async fn layouts_do_not_follow_redirects_or_replay_streams_with_configurator_retry() {
    for layout in [Http1HeaderLayout::Preserve, Http1HeaderLayout::NativeFetch] {
        for (status, code) in [("307 Temporary Redirect", 307), ("503 Unavailable", 503)] {
            for streaming in [false, true] {
                let classified = Arc::new(AtomicUsize::new(0));
                let observed = classified.clone();
                let http = HttpTransport::with_client_configurator(|builder| {
                    builder
                        .no_proxy()
                        .redirect(http_backend::redirect::Policy::limited(4))
                        .retry(http_backend::retry::for_host("127.0.0.1").classify_fn(
                            move |result| {
                                observed.fetch_add(1, Ordering::SeqCst);
                                result.retryable()
                            },
                        ))
                })
                .unwrap();
                let response = format!(
                "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\nLocation: /replayed\r\n\r\n"
            );
                let (url, _, captured) = capture(response.into_bytes()).await;
                let consumed = Arc::new(AtomicUsize::new(0));
                let observed = consumed.clone();
                let reply = if streaming {
                    http.send_stream(HttpStreamRequest {
                        method: "POST".into(),
                        url,
                        headers: headers(&[("Accept", "*/*")]),
                        http1_header_layout: Some(layout),
                        body: stream::once(async move {
                            observed.fetch_add(1, Ordering::SeqCst);
                            Ok(Bytes::from_static(b"payload"))
                        })
                        .boxed(),
                        content_length: 7,
                        timeout: Some(DEADLINE),
                    })
                    .await
                    .unwrap()
                } else {
                    // A buffered body is replayable, so 307 must still be returned
                    // unchanged even when the caller tried enabling redirects.
                    http.send(request(url, Some(layout), &[("Accept", "*/*")], b"payload"))
                        .await
                        .unwrap()
                };
                assert_eq!(reply.status, code, "{layout:?}, streaming={streaming}");
                assert_eq!(consumed.load(Ordering::SeqCst), usize::from(streaming));
                assert_eq!(classified.load(Ordering::SeqCst), 0);
                assert!(captured.await.unwrap().ends_with(b"\r\n\r\npayload"));
            }
        }
    }
}

#[tokio::test]
async fn request_deadline_still_covers_streaming_response_reads_for_both_layouts() {
    let http = HttpTransport::with_read_timeout_and_client_configurator(None, |builder| {
        builder.no_proxy()
    })
    .unwrap();
    for layout in [Http1HeaderLayout::Preserve, Http1HeaderLayout::NativeFetch] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/slow", listener.local_addr().unwrap());
        let (release, wait) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            tokio::time::timeout(DEADLINE, async {
                let (mut socket, _) = listener.accept().await.unwrap();
                read_request(&mut socket).await;
                socket
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n")
                    .await
                    .unwrap();
                // Keep the body pending until the client proves its deadline.
                let _ = wait.await;
            })
            .await
            .expect("slow response fixture was not released");
        });
        let mut req = request(url, Some(layout), &[("Accept", "*/*")], b"payload");
        req.timeout = Some(Duration::from_millis(200));
        let mut reply = http.send(req).await.unwrap();
        assert_eq!(reply.status, 200);
        let next = tokio::time::timeout(DEADLINE, reply.body.next())
            .await
            .expect("response body ignored the request deadline");
        assert!(
            matches!(next, Some(Err(LlmError::TransportTimeout { .. }))),
            "{layout:?}: expected a response-body deadline error"
        );
        let _ = release.send(());
        server.await.unwrap();
    }
}

#[tokio::test]
async fn absent_layout_keeps_lowercase_headers_without_native_defaults() {
    let (url, _, captured) = capture(response("", b"ok")).await;
    transport()
        .send(request(
            url,
            None,
            &[
                ("Accept", "*/*"),
                ("User-Agent", "claude-cli/2.1.293"),
                ("X-Api-Key", "test-only"),
                ("X-Mixed-Case", "value"),
            ],
            b"payload",
        ))
        .await
        .unwrap();
    let wire = captured.await.unwrap();
    let end = wire
        .windows(4)
        .position(|value| value == b"\r\n\r\n")
        .unwrap();
    let head = std::str::from_utf8(&wire[..end]).unwrap();
    let names: Vec<_> = head
        .split("\r\n")
        .skip(1)
        .map(|line| line.split_once(':').unwrap().0)
        .collect();
    assert!(names.iter().all(|name| *name == name.to_ascii_lowercase()));
    assert!(head.contains("\r\nx-mixed-case: value"));
    assert!(!names.contains(&"accept-encoding"));
    assert!(!names.contains(&"connection"));
    assert_eq!(&wire[end + 4..], b"payload");
}

#[tokio::test]
async fn native_fetch_decodes_all_advertised_encodings_without_affecting_other_requests() {
    // Fixed fixtures for b"compressed response payload\n", each decoded locally
    // to verify its exact bytes. Tests need no compression tools or libraries.
    // Python gzip.compress(payload, mtime=0).
    let gzip = [
        0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x4b, 0xce, 0xcf, 0x2d, 0x28,
        0x4a, 0x2d, 0x2e, 0x4e, 0x4d, 0x51, 0x00, 0x52, 0x05, 0xf9, 0x79, 0xc5, 0xa9, 0x0a, 0x05,
        0x89, 0x95, 0x39, 0xf9, 0x89, 0x29, 0x5c, 0x00, 0xba, 0x5c, 0x98, 0xe1, 0x1c, 0x00, 0x00,
        0x00,
    ];
    // Python zlib.compress(payload): HTTP deflate uses a zlib wrapper.
    let deflate = [
        0x78, 0x9c, 0x4b, 0xce, 0xcf, 0x2d, 0x28, 0x4a, 0x2d, 0x2e, 0x4e, 0x4d, 0x51, 0x00, 0x52,
        0x05, 0xf9, 0x79, 0xc5, 0xa9, 0x0a, 0x05, 0x89, 0x95, 0x39, 0xf9, 0x89, 0x29, 0x5c, 0x00,
        0xa3, 0xa2, 0x0a, 0xd9,
    ];
    // brotli --quality=5 --stdout, verified with brotli --decompress --stdout.
    let brotli = [
        0x1f, 0x1b, 0x00, 0x00, 0xc4, 0x6d, 0xec, 0x7b, 0x96, 0xfb, 0x4a, 0xf6, 0x0a, 0xbb, 0xe5,
        0x26, 0x9e, 0xe8, 0x52, 0x58, 0xa6, 0xb2, 0x70, 0x6d, 0x26, 0xe7, 0xe9, 0x27, 0x32, 0x00,
    ];
    // zstd --quiet --stdout -3, verified with Python compression.zstd.decompress.
    let zstd = [
        0x28, 0xb5, 0x2f, 0xfd, 0x04, 0x58, 0xe1, 0x00, 0x00, 0x63, 0x6f, 0x6d, 0x70, 0x72, 0x65,
        0x73, 0x73, 0x65, 0x64, 0x20, 0x72, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x20, 0x70,
        0x61, 0x79, 0x6c, 0x6f, 0x61, 0x64, 0x0a, 0x4f, 0x41, 0x89, 0x6d,
    ];
    let fixtures: [(&str, &[u8]); 4] = [
        ("gzip", &gzip),
        ("deflate", &deflate),
        ("br", &brotli),
        ("zstd", &zstd),
    ];
    // Reuse one transport to catch decoder policy leaking between requests.
    let http = transport();
    for (encoding, encoded) in fixtures {
        for layout in [
            Some(Http1HeaderLayout::NativeFetch),
            None,
            Some(Http1HeaderLayout::Preserve),
        ] {
            let (url, _, captured) = capture(response(
                &format!("Content-Encoding: {encoding}\r\n"),
                encoded,
            ))
            .await;
            let mut reply = http
                .send(request(url, layout, &[("Accept", "*/*")], b"payload"))
                .await
                .unwrap();
            let mut decoded = Vec::new();
            while let Some(chunk) = reply.body.next().await {
                decoded.extend_from_slice(&chunk.unwrap());
            }
            let wire = String::from_utf8(captured.await.unwrap()).unwrap();
            if layout == Some(Http1HeaderLayout::NativeFetch) {
                assert_eq!(decoded, b"compressed response payload\n", "{encoding}");
                assert!(wire.contains("\r\nAccept-Encoding: gzip, deflate, br, zstd\r\n"));
            } else {
                assert_eq!(decoded, encoded, "{encoding}, {layout:?}");
                assert!(!wire.to_ascii_lowercase().contains("accept-encoding:"));
            }
        }
    }
}
