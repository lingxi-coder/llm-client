use std::error::Error as StdError;
use std::future::Future;
use std::marker::{PhantomData, Unpin};
use std::pin::Pin;
use std::task::{self, ready, Poll};

use http::{HeaderMap, HeaderValue, Uri};
use hyper::rt::{Read, Write};
use pin_project_lite::pin_project;
use tower_service::Service;

/// Tunnel Proxy via HTTP CONNECT
///
/// This is a connector that can be used by the `legacy::Client`. It wraps
/// another connector, and after getting an underlying connection, it creates
/// an HTTP CONNECT tunnel over it.
#[derive(Debug, Clone)]
pub struct Tunnel<C> {
    headers: Headers,
    inner: C,
    proxy_dst: Uri,
}

#[derive(Clone, Debug)]
enum Headers {
    Empty,
    Auth(HeaderValue),
    Extra(HeaderMap),
}

/// Failure while opening an HTTP CONNECT proxy tunnel.
#[derive(Debug)]
pub enum TunnelError {
    /// The underlying connector failed before or while establishing the proxy connection.
    ConnectFailed(Box<dyn StdError + Send + Sync>),
    /// I/O failed while writing the CONNECT request or reading its response.
    Io(std::io::Error),
    /// The destination URI did not contain a host.
    MissingHost,
    /// The proxy returned HTTP 407 and its typed challenge fields.
    ProxyAuthRequired {
        /// Status code parsed from the proxy's CONNECT response.
        status: u16,
        /// Raw `Proxy-Authenticate` header bytes, when present.
        proxy_authenticate: Option<Vec<u8>>,
    },
    /// The proxy response headers exceeded the parser's fixed buffer.
    ProxyHeadersTooLong,
    /// The proxy closed the connection before completing its response headers.
    TunnelUnexpectedEof,
    /// The proxy returned a response other than HTTP 200 or HTTP 407.
    TunnelUnsuccessful,
}

pin_project! {
    // Not publicly exported (so missing_docs doesn't trigger).
    //
    // We return this `Future` instead of the `Pin<Box<dyn Future>>` directly
    // so that users don't rely on it fitting in a `Pin<Box<dyn Future>>` slot
    // (and thus we can change the type in the future).
    #[must_use = "futures do nothing unless polled"]
    #[allow(missing_debug_implementations)]
    pub struct Tunneling<F, T> {
        #[pin]
        fut: BoxTunneling<T>,
        _marker: PhantomData<F>,
    }
}

type BoxTunneling<T> = Pin<Box<dyn Future<Output = Result<T, TunnelError>> + Send>>;

impl<C> Tunnel<C> {
    /// Create a new Tunnel service.
    ///
    /// This wraps an underlying connector, and stores the address of a
    /// tunneling proxy server.
    ///
    /// A `Tunnel` can then be called with any destination. The `dst` passed to
    /// `call` will not be used to create the underlying connection, but will
    /// be used in an HTTP CONNECT request sent to the proxy destination.
    pub fn new(proxy_dst: Uri, connector: C) -> Self {
        Self {
            headers: Headers::Empty,
            inner: connector,
            proxy_dst,
        }
    }

    /// Add `proxy-authorization` header value to the CONNECT request.
    pub fn with_auth(mut self, mut auth: HeaderValue) -> Self {
        // just in case the user forgot
        auth.set_sensitive(true);
        match self.headers {
            Headers::Empty => {
                self.headers = Headers::Auth(auth);
            }
            Headers::Auth(ref mut existing) => {
                *existing = auth;
            }
            Headers::Extra(ref mut extra) => {
                extra.insert(http::header::PROXY_AUTHORIZATION, auth);
            }
        }

        self
    }

    /// Add extra headers to be sent with the CONNECT request.
    ///
    /// If existing headers have been set, these will be merged.
    pub fn with_headers(mut self, mut headers: HeaderMap) -> Self {
        match self.headers {
            Headers::Empty => {
                self.headers = Headers::Extra(headers);
            }
            Headers::Auth(auth) => {
                headers
                    .entry(http::header::PROXY_AUTHORIZATION)
                    .or_insert(auth);
                self.headers = Headers::Extra(headers);
            }
            Headers::Extra(ref mut extra) => {
                extra.extend(headers);
            }
        }

        self
    }
}

impl<C> Service<Uri> for Tunnel<C>
where
    C: Service<Uri>,
    C::Future: Send + 'static,
    C::Response: Read + Write + Unpin + Send + 'static,
    C::Error: Into<Box<dyn StdError + Send + Sync>>,
{
    type Response = C::Response;
    type Error = TunnelError;
    type Future = Tunneling<C::Future, C::Response>;

    fn poll_ready(&mut self, cx: &mut task::Context<'_>) -> Poll<Result<(), Self::Error>> {
        ready!(self.inner.poll_ready(cx)).map_err(|e| TunnelError::ConnectFailed(e.into()))?;
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, dst: Uri) -> Self::Future {
        let connecting = self.inner.call(self.proxy_dst.clone());
        let headers = self.headers.clone();

        Tunneling {
            fut: Box::pin(async move {
                let conn = connecting
                    .await
                    .map_err(|e| TunnelError::ConnectFailed(e.into()))?;
                tunnel(
                    conn,
                    dst.host().ok_or(TunnelError::MissingHost)?,
                    dst.port().map(|p| p.as_u16()).unwrap_or(443),
                    &headers,
                )
                .await
            }),
            _marker: PhantomData,
        }
    }
}

impl<F, T, E> Future for Tunneling<F, T>
where
    F: Future<Output = Result<T, E>>,
{
    type Output = Result<T, TunnelError>;

    fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
        self.project().fut.poll(cx)
    }
}

async fn tunnel<T>(mut conn: T, host: &str, port: u16, headers: &Headers) -> Result<T, TunnelError>
where
    T: Read + Write + Unpin,
{
    let mut buf = format!(
        "\
         CONNECT {host}:{port} HTTP/1.1\r\n\
         Host: {host}:{port}\r\n\
         "
    )
    .into_bytes();

    match headers {
        Headers::Auth(auth) => {
            buf.extend_from_slice(b"Proxy-Authorization: ");
            buf.extend_from_slice(auth.as_bytes());
            buf.extend_from_slice(b"\r\n");
        }
        Headers::Extra(extra) => {
            for (name, value) in extra {
                buf.extend_from_slice(name.as_str().as_bytes());
                buf.extend_from_slice(b": ");
                buf.extend_from_slice(value.as_bytes());
                buf.extend_from_slice(b"\r\n");
            }
        }
        Headers::Empty => (),
    }

    // headers end
    buf.extend_from_slice(b"\r\n");

    crate::rt::write_all(&mut conn, &buf)
        .await
        .map_err(TunnelError::Io)?;

    let mut buf = [0; 8192];
    let mut pos = 0;

    loop {
        let n = crate::rt::read(&mut conn, &mut buf[pos..])
            .await
            .map_err(TunnelError::Io)?;

        if n == 0 {
            return Err(TunnelError::TunnelUnexpectedEof);
        }
        pos += n;

        let recvd = &buf[..pos];
        let Some(header_end) = recvd.windows(4).position(|window| window == b"\r\n\r\n") else {
            if pos == buf.len() {
                return Err(TunnelError::ProxyHeadersTooLong);
            }
            continue;
        };
        let header_block = &recvd[..header_end + 2];
        let Some(status) = parse_status_code(header_block) else {
            return Err(TunnelError::TunnelUnsuccessful);
        };
        if status == 200 {
            return Ok(conn);
        }
        if status == 407 {
            return Err(TunnelError::ProxyAuthRequired {
                status,
                proxy_authenticate: proxy_authenticate(header_block),
            });
        }
        return Err(TunnelError::TunnelUnsuccessful);
    }
}

fn parse_status_code(header_block: &[u8]) -> Option<u16> {
    let line_end = header_block
        .windows(2)
        .position(|window| window == b"\r\n")?;
    let status_line = std::str::from_utf8(&header_block[..line_end]).ok()?;
    let mut fields = status_line.split_ascii_whitespace();
    let version = fields.next()?;
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1") {
        return None;
    }
    fields.next()?.parse().ok()
}

fn proxy_authenticate(header_block: &[u8]) -> Option<Vec<u8>> {
    let line_end = header_block
        .windows(2)
        .position(|window| window == b"\r\n")?;
    let mut combined: Option<Vec<u8>> = None;
    for line in header_block[line_end + 2..].split(|byte| *byte == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(colon) = line.iter().position(|byte| *byte == b':') else {
            continue;
        };
        if !std::str::from_utf8(&line[..colon])
            .ok()?
            .eq_ignore_ascii_case("proxy-authenticate")
        {
            continue;
        }
        let mut value = &line[colon + 1..];
        while value
            .first()
            .is_some_and(|byte| *byte == b' ' || *byte == b'\t')
        {
            value = &value[1..];
        }
        while value
            .last()
            .is_some_and(|byte| *byte == b' ' || *byte == b'\t')
        {
            value = &value[..value.len() - 1];
        }
        if let Some(combined) = &mut combined {
            combined.extend_from_slice(b", ");
            combined.extend_from_slice(value);
        } else {
            combined = Some(value.to_vec());
        }
    }
    combined
}

impl std::fmt::Display for TunnelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("tunnel error: ")?;

        f.write_str(match self {
            TunnelError::MissingHost => "missing destination host",
            TunnelError::ProxyAuthRequired { .. } => "proxy authorization required",
            TunnelError::ProxyHeadersTooLong => "proxy response headers too long",
            TunnelError::TunnelUnexpectedEof => "unexpected end of file",
            TunnelError::TunnelUnsuccessful => "unsuccessful",
            TunnelError::ConnectFailed(_) => "failed to create underlying connection",
            TunnelError::Io(_) => "io error establishing tunnel",
        })
    }
}

impl std::error::Error for TunnelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TunnelError::Io(ref e) => Some(e),
            TunnelError::ConnectFailed(ref e) => Some(&**e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_status_code, proxy_authenticate};

    #[test]
    fn typed_407_keeps_the_proxy_challenge_header_bytes() {
        let response = b"HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic realm=\"edge\"\x80\r\nProxy-Authenticate: Bearer\r\nContent-Length: 0\r\n\r\n";
        let block_end = response
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .unwrap();
        let block = &response[..block_end + 2];
        assert_eq!(parse_status_code(block), Some(407));
        assert_eq!(
            proxy_authenticate(block),
            Some(b"Basic realm=\"edge\"\x80, Bearer".to_vec())
        );
    }

    #[test]
    fn tunnel_parser_retains_the_actual_status_code() {
        let response = b"HTTP/1.1 403 Forbidden\r\nProxy-Authenticate: Basic\r\n\r\n";
        let block_end = response
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .unwrap();
        let block = &response[..block_end + 2];
        assert_eq!(parse_status_code(block), Some(403));
    }
}
