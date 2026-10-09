# HTTP/1 header layout dependency closure

These four crates are renamed, source-pinned upstream packages used transitively
by `lingxi-llm-client`. Consumers only depend on the SDK (or `harness-runtime`);
no consumer `[patch]`, registry replacement, local proxy, or custom global Cargo
configuration is required.

| Package | Upstream baseline | Local responsibility |
| --- | --- | --- |
| `lingxi-reqwest` | reqwest 0.12.28 | Forward the typed layout; select existing decompression middleware per request |
| `lingxi-hyper` | hyper 1.9.0 | Validate and emit ordered HTTP/1 field occurrences from the live HeaderMap |
| `lingxi-hyper-util` | hyper-util 0.1.20 | Manifest dependency aliases only |
| `lingxi-hyper-rustls` | hyper-rustls 0.27.5 | Manifest dependency aliases only |

`upstream.json` records the original crates.io archive SHA-256, release source
metadata and retained license files. The archive checksum is authoritative;
reqwest's published VCS metadata itself reports a dirty release tree. Its VCS SHA
must not be treated as an exact replacement for the published crate archive.
`http1-header-layout.patch` records every changed upstream file and the new
layout module, relative to this directory. `Cargo.toml.orig` remains the original
upstream manifest, for reference; Cargo uses the patched `Cargo.toml`.

The internal Cargo dependency keys retain `reqwest`, `hyper`, `hyper-util` and
`hyper-rustls`; `package` aliases select the renamed packages. All fork-internal
normal and test dependencies use sibling paths. Shared neutral crates (`http`,
`http-body`, `tokio`, `rustls`, Tower and compression codecs) remain upstream.
The SDK exposes its canonical backend as `transport::http_backend` so host TLS
configuration cannot accidentally mix a different reqwest `ClientBuilder` type.

## Supported feature closure

Production uses `default-features = false`: rustls, HTTP/1, stream, JSON, and the
opt-in `http1-header-layout` feature. The layout feature also enables actual gzip,
zlib/deflate, Brotli and Zstandard response decoding through upstream
`tower-http`/`async-compression`; no decompressor is reimplemented. The SDK and
internal general HTTP adapter disable implicit decompression by default to keep
the prior ordinary request wire unchanged. Only `NativeFetch` requests enable
all four existing decoders on a clone of the same pooled service. Explicit
Accept-Encoding values remain authoritative.

The original optional native-TLS manifests are retained for provenance, but are
not part of this four-package build contract: enabling `default-tls`/`native-tls`
would also require closing `hyper-tls` onto the renamed Hyper types. Do not enable
those features or claim vendor-wide `--all-features` support. Likewise the layout
is specifically HTTP/1; no mixed-case header promise is made for HTTP/2 or browser
Fetch. SDK features do not enable either alternate HTTP protocol on this backend.

## Wire and ownership rules

- Layout entries contain only validated field-name spelling, occurrence indices
  and whether a generated field may be absent. Values come exclusively from the
  current HeaderMap. Stale required occurrences and ambiguous multiplicity fail
  before any request bytes are appended to the encoder buffer.
- Hyper retains its request-target, framing, pooling and connection machinery.
  Additional connector-generated fields (for example proxy authentication) are
  emitted once using their current values. Removed values cannot be restored.
- The SDK validates fixed/streamed body framing before networking and never polls
  a rejected single-use body. Unselected requests use the upstream encoder.
- TLS, mTLS, custom CA roots, proxies, cancellation, redirects, retry and timeout
  owners remain in their original layers. SDK redirects/retries stay disabled.
- The NativeFetch persona is selected explicitly by typed request policy; URLs,
  header names, credentials and model identifiers never trigger it implicitly.

## Distribution and validation

A Git checkout carries the entire path dependency closure. Publishing to a
registry requires publishing these renamed packages first and retaining their
`package`/version dependency links; path dependencies are not bundled as ordinary
files inside the parent crate package. Never publish manifests that resolve back
to the unmodified upstream packages. No root patch table is a supported fallback.

Required validation includes SDK real TCP header capture tests, every advertised
compression decoder, existing streaming/cancellation/deadline tests, host custom
CA/mTLS/proxy regressions, and an external fixture whose only dependency is
`harness-runtime` with no patches. Source-level checks are not a substitute for
those builds or the Native293 differential acceptance run.
