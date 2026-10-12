use super::{
    HttpRequest, HttpStreamRequest, NativeProxyResponse, ProxyAuthHelper, StreamResponse, Transport,
};
use crate::protocol::{LlmError, ProxyAuthHelperChallenge};
use async_trait::async_trait;
use futures::StreamExt;
use std::sync::Arc;
use std::time::Duration;

/// Pooled HTTP/HTTPS transport backed by reqwest and rustls.
///
/// Execute requests inside a Tokio runtime with I/O and time enabled. Clones
/// share the connection pool. Redirects and automatic retries are disabled so
/// a provider cannot forward credentials or silently replay a billed request.
/// Connections have a 30-second timeout and reads have a 60-second idle
/// timeout by default. [`HttpRequest::timeout`] controls the total deadline,
/// including streaming body reads.
/// Responses WebSocket sessions are available with `responses-websocket`.
#[derive(Clone)]
pub struct HttpTransport {
    pub(super) client: reqwest::Client,
    native_proxy_client: Option<reqwest::Client>,
    proxy_auth_helper: Option<Arc<dyn ProxyAuthHelper>>,
    #[cfg(feature = "responses-websocket")]
    pub(super) read_timeout: Option<Duration>,
}

impl HttpTransport {
    pub fn new() -> Result<Self, LlmError> {
        Self::with_read_timeout(Duration::from_secs(60))
    }

    /// Build with a custom timeout for each idle response-body read.
    pub fn with_read_timeout(read_timeout: Duration) -> Result<Self, LlmError> {
        Self::with_read_timeout_and_client_configurator(Some(read_timeout), |builder| builder)
    }

    /// Customize TLS/proxy settings with the default 60-second idle-read limit.
    /// Use `with_read_timeout_and_client_configurator` to select a different
    /// limit consistently for HTTP and Responses WebSocket.
    pub fn with_client_configurator(
        configure: impl FnOnce(reqwest::ClientBuilder) -> reqwest::ClientBuilder,
    ) -> Result<Self, LlmError> {
        Self::with_read_timeout_and_client_configurator(Some(Duration::from_secs(60)), configure)
    }

    /// Configure the HTTP and Responses idle-read policy together. `None`
    /// leaves read deadlines to the caller (for example a host watchdog).
    /// The configurator should only set connection/TLS/proxy settings; a
    /// supplied idle limit is applied after it, as are no-redirect/no-retry.
    pub fn with_read_timeout_and_client_configurator(
        read_timeout: Option<Duration>,
        configure: impl FnOnce(reqwest::ClientBuilder) -> reqwest::ClientBuilder,
    ) -> Result<Self, LlmError> {
        let client = build_http_client(read_timeout, configure(base_client_builder()))?;
        Ok(Self {
            client,
            native_proxy_client: None,
            proxy_auth_helper: None,
            #[cfg(feature = "responses-websocket")]
            read_timeout,
        })
    }

    /// Build separate default and Native first-party proxy pools. The host's
    /// native route configurator adds the source-selected process proxy and,
    /// when configured, installs the helper's dynamic header provider there.
    pub fn with_read_timeout_and_native_proxy(
        read_timeout: Option<Duration>,
        helper: Option<Arc<dyn ProxyAuthHelper>>,
        configure_default: impl FnOnce(reqwest::ClientBuilder) -> reqwest::ClientBuilder,
        configure_native: impl FnOnce(
            reqwest::ClientBuilder,
            Option<Arc<dyn ProxyAuthHelper>>,
        ) -> reqwest::ClientBuilder,
    ) -> Result<Self, LlmError> {
        let client = build_http_client(read_timeout, configure_default(base_client_builder()))?;
        let native_proxy_client = build_http_client(
            read_timeout,
            configure_native(base_client_builder(), helper.clone()),
        )?;
        Ok(Self {
            client,
            native_proxy_client: Some(native_proxy_client),
            proxy_auth_helper: helper,
            #[cfg(feature = "responses-websocket")]
            read_timeout,
        })
    }

    async fn send_request(
        &self,
        req: HttpRequest,
        native_proxy_route: bool,
    ) -> Result<(reqwest::Response, Option<ProxyAuthHelperChallenge>), LlmError> {
        let invalid = || LlmError::InvalidRequest {
            message: "invalid HTTP method, URL or headers".into(),
        };
        let method = reqwest::Method::from_bytes(req.method.as_bytes()).map_err(|_| invalid())?;
        let url = url::Url::parse(&req.url).map_err(|_| invalid())?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(invalid());
        }
        let prepared = prepare_request_headers(
            req.headers,
            req.http1_header_layout,
            req.body.len() as u64,
            false,
        )?;
        let client = if native_proxy_route {
            self.native_proxy_client.as_ref().unwrap_or(&self.client)
        } else {
            &self.client
        };
        let mut request = client
            .request(method, url)
            .headers(prepared.headers)
            .body(req.body);
        if let Some(layout) = prepared.layout {
            request = request.http1_header_layout(layout);
        }
        if prepared.decompress {
            request = request.http1_response_decompression();
        }
        if let Some(timeout) = req.timeout {
            request = request.timeout(timeout);
        }
        let response = request
            .send()
            .await
            .map_err(|err| self.request_error(err, native_proxy_route, false))?;
        let challenge = self.observe_http_response(&response, native_proxy_route);
        Ok((response, challenge))
    }

    async fn send_stream_request(
        &self,
        req: HttpStreamRequest,
        native_proxy_route: bool,
    ) -> Result<(reqwest::Response, Option<ProxyAuthHelperChallenge>), LlmError> {
        let invalid = || LlmError::InvalidRequest {
            message: "invalid streaming HTTP method, URL or headers".into(),
        };
        let method = reqwest::Method::from_bytes(req.method.as_bytes()).map_err(|_| invalid())?;
        let url = url::Url::parse(&req.url).map_err(|_| invalid())?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(invalid());
        }
        let prepared = prepare_request_headers(
            req.headers,
            req.http1_header_layout,
            req.content_length,
            true,
        )?;
        let client = if native_proxy_route {
            self.native_proxy_client.as_ref().unwrap_or(&self.client)
        } else {
            &self.client
        };
        let mut request = client
            .request(method, url)
            .headers(prepared.headers)
            .body(reqwest::Body::wrap_stream(req.body));
        if let Some(layout) = prepared.layout {
            request = request.http1_header_layout(layout);
        }
        if prepared.decompress {
            request = request.http1_response_decompression();
        }
        if let Some(timeout) = req.timeout {
            request = request.timeout(timeout);
        }
        let response = request
            .send()
            .await
            .map_err(|err| self.request_error(err, native_proxy_route, false))?;
        let challenge = self.observe_http_response(&response, native_proxy_route);
        Ok((response, challenge))
    }

    fn observe_http_response(
        &self,
        response: &reqwest::Response,
        native_proxy_route: bool,
    ) -> Option<ProxyAuthHelperChallenge> {
        if !native_proxy_route || response.status().as_u16() != 407 {
            return None;
        }
        let helper = self.proxy_auth_helper.as_ref()?;
        let mut values = response
            .headers()
            .get_all(reqwest::header::PROXY_AUTHENTICATE)
            .iter();
        let proxy_authenticate = values.next().map(|first| {
            let mut combined = first.as_bytes().to_vec();
            for value in values {
                // Fetch Headers.get() combines repeated values with comma-space.
                combined.extend_from_slice(b", ");
                combined.extend_from_slice(value.as_bytes());
            }
            combined
        });
        helper
            .on_proxy_auth_challenge(proxy_authenticate.as_deref())
            .then_some(ProxyAuthHelperChallenge {
                status: 407,
                proxy_authenticate,
            })
    }

    fn request_error(
        &self,
        error: reqwest::Error,
        native_proxy_route: bool,
        streaming_body: bool,
    ) -> LlmError {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some((status, proxy_authenticate)) = error.proxy_tunnel_response() {
            let tunnel = LlmError::ProxyTunnelResponse {
                status,
                proxy_authenticate: proxy_authenticate
                    .clone()
                    .map(crate::protocol::ProxyAuthHeader::new),
            };
            if status == 407 && native_proxy_route {
                if let Some(helper) = self.proxy_auth_helper.as_ref() {
                    if helper.on_proxy_auth_challenge(proxy_authenticate.as_deref()) {
                        return tunnel.with_proxy_auth_helper_challenge(ProxyAuthHelperChallenge {
                            status,
                            proxy_authenticate,
                        });
                    }
                }
            }
            return tunnel;
        }
        network_error(error, streaming_body)
    }

    async fn send_native_proxy_request(
        &self,
        req: HttpRequest,
    ) -> Result<NativeProxyResponse, LlmError> {
        let (response, proxy_auth_helper_challenge) = self.send_request(req, true).await?;
        Ok(NativeProxyResponse {
            response: stream_response(response),
            proxy_auth_helper_challenge,
        })
    }
}

fn base_client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .no_gzip()
        .no_brotli()
        .no_zstd()
        .no_deflate()
        .connect_timeout(Duration::from_secs(30))
}

fn build_http_client(
    read_timeout: Option<Duration>,
    mut builder: reqwest::ClientBuilder,
) -> Result<reqwest::Client, LlmError> {
    if let Some(timeout) = read_timeout {
        builder = builder.read_timeout(timeout);
    }
    builder
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .build()
        .map_err(|_| LlmError::Transport {
            message: "could not initialize HTTP client".into(),
        })
}

fn stream_response(response: reqwest::Response) -> StreamResponse {
    StreamResponse {
        status: response.status().as_u16(),
        headers: response_headers(&response),
        body: response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|err| network_error(err, true)))
            .boxed(),
    }
}

fn response_headers(response: &reqwest::Response) -> Vec<(String, String)> {
    response
        .headers()
        .iter()
        .map(|(name, value)| {
            (
                name.to_string(),
                String::from_utf8_lossy(value.as_bytes()).into_owned(),
            )
        })
        .collect()
}

fn certificate_error(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if matches!(
            error.downcast_ref::<rustls::Error>(),
            Some(rustls::Error::InvalidCertificate(_) | rustls::Error::NoCertificatesPresented)
        ) {
            return true;
        }
        // io::Error may expose the inner error's source rather than the inner
        // error itself; inspect its payload so a wrapped rustls error is retained.
        current = error
            .downcast_ref::<std::io::Error>()
            .and_then(|io| {
                io.get_ref()
                    .map(|inner| inner as &(dyn std::error::Error + 'static))
            })
            .or_else(|| error.source());
    }
    false
}

// Do not expose reqwest error strings: URLs, query credentials and underlying
// proxy errors may carry secrets. Preserve actionable categories instead.
pub(super) fn network_error(err: reqwest::Error, streaming_body: bool) -> LlmError {
    if err.is_timeout() {
        LlmError::TransportTimeout {
            message: "HTTP request timed out".into(),
        }
    } else if certificate_error(&err) {
        LlmError::TlsCert {
            message: "TLS certificate validation failed".into(),
        }
    } else if err.is_builder() {
        LlmError::InvalidRequest {
            message: "could not build HTTP request".into(),
        }
    } else {
        LlmError::Transport {
            message: if streaming_body {
                "HTTP response stream interrupted"
            } else {
                "HTTP connection or response read failed"
            }
            .into(),
        }
    }
}

struct PreparedHeaders {
    headers: reqwest::header::HeaderMap,
    layout: Option<reqwest::Http1HeaderLayout>,
    decompress: bool,
}

fn prepare_request_headers(
    mut fields: Vec<(String, String)>,
    policy: Option<super::Http1HeaderLayout>,
    content_length: u64,
    streamed: bool,
) -> Result<PreparedHeaders, LlmError> {
    use super::Http1HeaderLayout;
    use reqwest::header::{
        HeaderMap, HeaderName, HeaderValue, CONTENT_LENGTH, HOST, TRANSFER_ENCODING,
    };
    let invalid = || LlmError::InvalidRequest {
        message: "invalid HTTP headers or body framing".into(),
    };
    let native = policy == Some(Http1HeaderLayout::NativeFetch);
    if native {
        // Stable sorting uses the caller's original spelling, not HeaderName's
        // lowercased representation. Values move with their own occurrences.
        fields.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    }
    let mut headers = HeaderMap::new();
    let mut layout: Option<reqwest::Http1HeaderLayout> =
        policy.map(|_| reqwest::Http1HeaderLayout::new());
    let mut saw_content_length = false;
    let mut saw_host = false;
    for (spelling, value) in fields {
        let name = HeaderName::from_bytes(spelling.as_bytes()).map_err(|_| invalid())?;
        if streamed || policy.is_some() {
            if name == TRANSFER_ENCODING {
                // Both transport entrypoints have an exact length. Never let
                // Hyper repair conflicting caller framing or replay a body.
                return Err(invalid());
            }
            if name == CONTENT_LENGTH {
                if saw_content_length || value != content_length.to_string() {
                    return Err(invalid());
                }
                saw_content_length = true;
                if policy.is_none() {
                    // Preserve the original streaming transport's normalized
                    // single Content-Length when no layout was requested.
                    continue;
                }
            }
        }
        if policy.is_some() && name == HOST {
            if saw_host {
                return Err(invalid());
            }
            saw_host = true;
        }
        let mut value = HeaderValue::from_str(&value).map_err(|_| invalid())?;
        value.set_sensitive(true);
        headers.append(name, value);
        if let Some(layout) = &mut layout {
            layout.push(&spelling).map_err(|_| invalid())?;
        }
    }
    if let Some(layout) = &mut layout {
        if native && !headers.contains_key(reqwest::header::CONNECTION) {
            headers.insert(
                reqwest::header::CONNECTION,
                HeaderValue::from_static("keep-alive"),
            );
            layout.push("Connection").map_err(|_| invalid())?;
        }
        if !headers.contains_key(HOST) {
            // The connector supplies the authoritative host, including port
            // and IPv6 syntax, after proxy/request-target resolution.
            layout
                .push_optional(if native { "Host" } else { "host" })
                .map_err(|_| invalid())?;
        }
        if native && !headers.contains_key(reqwest::header::ACCEPT_ENCODING) {
            headers.insert(
                reqwest::header::ACCEPT_ENCODING,
                HeaderValue::from_static("gzip, deflate, br, zstd"),
            );
            layout.push("Accept-Encoding").map_err(|_| invalid())?;
        }
        if !headers.contains_key(CONTENT_LENGTH) {
            headers.insert(
                CONTENT_LENGTH,
                HeaderValue::from_str(&content_length.to_string()).map_err(|_| invalid())?,
            );
            layout
                .push(if native {
                    "Content-Length"
                } else {
                    "content-length"
                })
                .map_err(|_| invalid())?;
        }
    } else if streamed {
        headers.insert(
            CONTENT_LENGTH,
            HeaderValue::from_str(&content_length.to_string()).map_err(|_| invalid())?,
        );
    }
    Ok(PreparedHeaders {
        headers,
        layout,
        decompress: native,
    })
}

#[async_trait]
impl Transport for HttpTransport {
    #[cfg(feature = "responses-websocket")]
    async fn connect_websocket(
        &self,
        request: HttpRequest,
    ) -> Result<Box<dyn super::WebSocketConnection>, LlmError> {
        super::websocket::responses(self, request).await
    }

    async fn send(&self, req: HttpRequest) -> Result<StreamResponse, LlmError> {
        let (response, _) = self.send_request(req, false).await?;
        Ok(stream_response(response))
    }

    async fn send_native_proxy_request(
        &self,
        req: HttpRequest,
    ) -> Result<NativeProxyResponse, LlmError> {
        HttpTransport::send_native_proxy_request(self, req).await
    }

    async fn send_stream(&self, req: HttpStreamRequest) -> Result<StreamResponse, LlmError> {
        let (response, _) = self.send_stream_request(req, false).await?;
        Ok(stream_response(response))
    }
}

#[cfg(test)]
mod certificate_tests {
    use super::*;
    #[test]
    fn certificate_classification_requires_a_tls_error_not_matching_text() {
        let text = std::io::Error::other("certificate unknownissuer");
        assert!(!certificate_error(&text));
        let tls = rustls::Error::InvalidCertificate(rustls::CertificateError::UnknownIssuer);
        assert!(certificate_error(&tls));
        assert!(certificate_error(&std::io::Error::other(tls)));
    }
}

#[cfg(test)]
mod proxy_auth_helper_tests {
    use super::*;
    use crate::transport::ProxyAuthHelper;
    use bytes::Bytes;
    use futures::StreamExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    struct RecordingHelper {
        eligible: bool,
        calls: AtomicUsize,
        challenge: std::sync::Mutex<Option<Vec<u8>>>,
        cached: std::sync::Mutex<Option<String>>,
    }

    impl ProxyAuthHelper for RecordingHelper {
        fn cached_proxy_authorization(&self) -> Option<String> {
            self.cached.lock().unwrap().clone()
        }

        fn on_proxy_auth_challenge(&self, challenge: Option<&[u8]>) -> bool {
            self.calls.fetch_add(1, Ordering::Relaxed);
            *self.challenge.lock().unwrap() = challenge.map(<[u8]>::to_vec);
            if self.eligible {
                *self.cached.lock().unwrap() = Some("Basic helper-secret".into());
            }
            self.eligible
        }
    }

    fn request(url: String) -> HttpRequest {
        HttpRequest {
            http1_header_layout: None,
            method: "GET".into(),
            url,
            headers: vec![],
            body: Bytes::new(),
            timeout: None,
        }
    }

    async fn serve_407() -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).await;
            stream
                .write_all(
                    b"HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic realm=\"unit\"\x80\r\nContent-Length: 7\r\nConnection: close\r\n\r\nblocked",
                )
                .await
                .unwrap();
        });
        (format!("http://{address}/completion"), server)
    }

    #[tokio::test]
    async fn actual_407_response_remains_intact_and_returns_typed_challenge_when_enabled() {
        let helper = Arc::new(RecordingHelper {
            eligible: true,
            calls: AtomicUsize::new(0),
            challenge: std::sync::Mutex::new(None),
            cached: std::sync::Mutex::new(None),
        });
        let transport = HttpTransport::with_read_timeout_and_native_proxy(
            None,
            Some(helper.clone()),
            |builder| builder.no_proxy(),
            |builder, _| builder.no_proxy(),
        )
        .unwrap();
        let (url, server) = serve_407().await;

        let response = transport
            .send_native_proxy_request(request(url))
            .await
            .unwrap();
        assert_eq!(response.response.status, 407);
        let mut stream = response.response.body;
        let mut body = Vec::new();
        while let Some(chunk) = stream.next().await {
            body.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(body, b"blocked");
        let challenge = response.proxy_auth_helper_challenge.unwrap();
        assert_eq!(challenge.status, 407);
        assert_eq!(
            challenge.proxy_authenticate.as_deref(),
            Some(b"Basic realm=\"unit\"\x80".as_slice())
        );
        assert_eq!(helper.calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            helper.challenge.lock().unwrap().as_deref(),
            Some(b"Basic realm=\"unit\"\x80".as_slice())
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn actual_407_does_not_enable_retry_when_helper_gate_is_closed() {
        let helper = Arc::new(RecordingHelper {
            eligible: false,
            calls: AtomicUsize::new(0),
            challenge: std::sync::Mutex::new(None),
            cached: std::sync::Mutex::new(None),
        });
        let transport = HttpTransport::with_read_timeout_and_native_proxy(
            None,
            Some(helper.clone()),
            |builder| builder.no_proxy(),
            |builder, _| builder.no_proxy(),
        )
        .unwrap();
        let (url, server) = serve_407().await;

        let response = transport
            .send_native_proxy_request(request(url))
            .await
            .unwrap();
        assert!(response.proxy_auth_helper_challenge.is_none());
        assert_eq!(response.response.status, 407);
        assert_eq!(helper.calls.load(Ordering::Relaxed), 1);
        server.await.unwrap();
    }

    async fn read_headers(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut output = Vec::new();
        loop {
            let mut buffer = [0_u8; 1024];
            let count = stream.read(&mut buffer).await.unwrap();
            if count == 0 {
                break;
            }
            output.extend_from_slice(&buffer[..count]);
            if output.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
            assert!(
                output.len() <= 16 * 1024,
                "request headers must stay bounded"
            );
        }
        output
    }

    #[tokio::test]
    async fn https_connect_407_refreshes_proxy_header_for_next_connect_only() {
        use std::time::Duration;

        let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_address = proxy_listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut first, _) = proxy_listener.accept().await.unwrap();
            let first_connect = read_headers(&mut first).await;
            let first_text = String::from_utf8_lossy(&first_connect).to_ascii_lowercase();
            assert!(first_text.starts_with("connect origin.example:443 http/1.1"));
            assert!(!first_text.contains("proxy-authorization:"));
            first
                .write_all(
                    b"HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic realm=\"helper-test\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            drop(first);

            let (mut second, _) =
                tokio::time::timeout(Duration::from_secs(3), proxy_listener.accept())
                    .await
                    .expect("the next Native request should open a new CONNECT")
                    .unwrap();
            let second_connect = read_headers(&mut second).await;
            let second_text = String::from_utf8_lossy(&second_connect).to_ascii_lowercase();
            assert!(second_text.starts_with("connect origin.example:443 http/1.1"));
            assert!(second_text.contains("proxy-authorization: basic helper-secret"));
            second
                .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                .await
                .unwrap();

            // After CONNECT the only bytes visible to the proxy are the
            // encrypted origin TLS handshake. Proxy credentials must not be
            // copied into that origin stream.
            let mut tls = [0_u8; 4096];
            let count = tokio::time::timeout(Duration::from_secs(3), second.read(&mut tls))
                .await
                .expect("the client should start TLS after CONNECT")
                .unwrap();
            assert!(count > 0, "expected a TLS ClientHello");
            assert!(!tls[..count]
                .windows(b"helper-secret".len())
                .any(|window| window == b"helper-secret"));
            assert!(!tls[..count]
                .windows(b"Proxy-Authorization".len())
                .any(|window| window.eq_ignore_ascii_case(b"Proxy-Authorization")));
        });

        let helper = Arc::new(RecordingHelper {
            eligible: true,
            calls: AtomicUsize::new(0),
            challenge: std::sync::Mutex::new(None),
            cached: std::sync::Mutex::new(None),
        });
        let transport = HttpTransport::with_read_timeout_and_native_proxy(
            None,
            Some(helper.clone()),
            |builder| builder.no_proxy(),
            move |builder, helper| {
                let proxy = reqwest::Proxy::all(format!("http://{proxy_address}"))
                    .unwrap()
                    .custom_http_auth_provider(move || {
                        helper
                            .as_ref()?
                            .cached_proxy_authorization()?
                            .parse::<reqwest::header::HeaderValue>()
                            .ok()
                    });
                builder.no_proxy().proxy(proxy)
            },
        )
        .unwrap();

        let url = "https://origin.example/resource".to_owned();
        let first_error = match transport
            .send_native_proxy_request(request(url.clone()))
            .await
        {
            Err(error) => error,
            Ok(_) => panic!("CONNECT 407 must be returned as an error"),
        };
        assert!(matches!(
            first_error.classification_source(),
            LlmError::ProxyTunnelResponse { status: 407, .. }
        ));
        let challenge = first_error
            .proxy_auth_helper_challenge()
            .expect("eligible CONNECT 407 should retain helper evidence");
        assert_eq!(challenge.status, 407);
        assert_eq!(
            challenge.proxy_authenticate.as_deref(),
            Some(b"Basic realm=\"helper-test\"".as_slice())
        );
        assert_eq!(
            helper.cached_proxy_authorization().as_deref(),
            Some("Basic helper-secret")
        );

        assert!(transport
            .send_native_proxy_request(request(url))
            .await
            .is_err());
        assert_eq!(helper.calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            helper.challenge.lock().unwrap().as_deref(),
            Some(b"Basic realm=\"helper-test\"".as_slice())
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn ordinary_pool_http_407_never_invokes_native_helper_or_native_proxy() {
        use std::time::Duration;

        let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_address = proxy_listener.local_addr().unwrap();
        let proxy_server = tokio::spawn(async move {
            if let Ok(Ok((mut stream, _))) =
                tokio::time::timeout(Duration::from_millis(300), proxy_listener.accept()).await
            {
                let request = read_headers(&mut stream).await;
                stream
                    .write_all(
                        b"HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await
                    .unwrap();
                return Some(request);
            }
            None
        });

        let origin_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin_address = origin_listener.local_addr().unwrap();
        let origin_server = tokio::spawn(async move {
            let (mut stream, _) = origin_listener.accept().await.unwrap();
            let request = read_headers(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            request
        });

        let helper = Arc::new(RecordingHelper {
            eligible: true,
            calls: AtomicUsize::new(0),
            challenge: std::sync::Mutex::new(None),
            cached: std::sync::Mutex::new(None),
        });
        let transport = HttpTransport::with_read_timeout_and_native_proxy(
            None,
            Some(helper.clone()),
            |builder| builder.no_proxy(),
            move |builder, helper| {
                let proxy = reqwest::Proxy::all(format!("http://{proxy_address}"))
                    .unwrap()
                    .custom_http_auth_provider(move || {
                        helper
                            .as_ref()?
                            .cached_proxy_authorization()?
                            .parse::<reqwest::header::HeaderValue>()
                            .ok()
                    });
                builder.no_proxy().proxy(proxy)
            },
        )
        .unwrap();
        let response = transport
            .send(request(format!("http://{origin_address}/completion")))
            .await
            .unwrap();
        assert_eq!(response.status, 407);
        let _ = origin_server.await.unwrap();
        assert!(proxy_server.await.unwrap().is_none());
        assert_eq!(helper.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn native_no_proxy_origin_never_receives_cached_proxy_authorization() {
        let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_address = proxy_listener.local_addr().unwrap();
        let proxy_server = tokio::spawn(async move {
            tokio::time::timeout(
                std::time::Duration::from_millis(300),
                proxy_listener.accept(),
            )
            .await
            .is_ok()
        });

        let origin_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin_address = origin_listener.local_addr().unwrap();
        let origin_server = tokio::spawn(async move {
            let (mut stream, _) = origin_listener.accept().await.unwrap();
            let request = read_headers(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic realm=\"origin\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            request
        });

        let helper = Arc::new(RecordingHelper {
            eligible: true,
            calls: AtomicUsize::new(0),
            challenge: std::sync::Mutex::new(None),
            cached: std::sync::Mutex::new(Some("Basic helper-secret".into())),
        });
        let transport = HttpTransport::with_read_timeout_and_native_proxy(
            None,
            Some(helper.clone()),
            |builder| builder.no_proxy(),
            move |builder, helper| {
                let proxy = reqwest::Proxy::all(format!("http://{proxy_address}"))
                    .unwrap()
                    .no_proxy(reqwest::NoProxy::from_string("127.0.0.1"))
                    .custom_http_auth_provider(move || {
                        helper
                            .as_ref()?
                            .cached_proxy_authorization()?
                            .parse::<reqwest::header::HeaderValue>()
                            .ok()
                    });
                builder.no_proxy().proxy(proxy)
            },
        )
        .unwrap();

        let mut origin_request = request(format!("http://{origin_address}/completion"));
        origin_request
            .headers
            .push(("Authorization".into(), "Bearer origin-secret".into()));
        let response = transport
            .send_native_proxy_request(origin_request)
            .await
            .unwrap();
        assert_eq!(response.response.status, 407);
        assert!(response.proxy_auth_helper_challenge.is_some());
        let request = String::from_utf8_lossy(&origin_server.await.unwrap()).to_ascii_lowercase();
        assert!(request.contains("authorization: bearer origin-secret"));
        assert!(!request.contains("proxy-authorization:"));
        assert!(!request.contains("helper-secret"));
        assert!(
            !proxy_server.await.unwrap(),
            "NO_PROXY must connect directly"
        );
        assert_eq!(helper.calls.load(Ordering::Relaxed), 1);
    }
}
