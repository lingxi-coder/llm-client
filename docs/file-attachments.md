# File attachments

`lingxi-llm-client` separates an application's durable attachment from files a
model provider creates for its own account. A conversation should keep the
application's `AttachmentRef`; a provider file id or URI is only a request-time
model input and must not be used to render the attachment in another device.

```rust
use lingxi_llm_client::protocol::AttachmentRef;

let attachment = AttachmentRef {
    attachment_id: "att_...".into(), // stable id in the host application's store
    revision: "v1".into(),           // immutable content revision
    filename: "diagram.png".into(),
    media_type: "image/png".into(),
    size_bytes: 12345,
};
```

Use `ImageSource::Attachment`, `DocumentSource::Attachment`, or
`VideoSource::Attachment` in the message.
Register an `AttachmentResolver` on the client; it reads the exact bytes for
the requested id and revision from storage the host application controls.
The client checks the declared byte size, enforces a 64 MiB combined limit,
and resolves the reference into a request-local copy. The original message
continues to contain the stable reference.

For remote use, the application saves the original before sending the message
to the shared conversation. A receiving device displays it through the
application's authenticated attachment service, then passes the same reference
to `llm-client`. The resolver on the device running the model call reads from
that service. This keeps image display working independently of provider file
retention, including providers whose uploaded files cannot be downloaded. Any
authenticated preview URL and its lifetime belong to the application; a
provider file URI is not a preview URL. If the attachment service cannot
return a revision, the UI should show the attachment as unavailable and retry
when connectivity returns, while the resolver returns a clear `LlmError` to
the model call. The client does not invent a URL or substitute a provider file
id.

Provider upload and model input are separate capabilities. The client only
uses a provider-owned file reference when an adapter confirms the active
profile, model, media type and purpose support it. OpenAI Responses, Anthropic
Messages, and Gemini have native model input references; OpenAI Chat's native
file reference is limited to PDFs; xAI references are limited to documented
document-search inputs. Qwen's Beijing and Singapore Files APIs accept
documents and supported images with `file-extract`, but Qwen-Long model input
is available only on Beijing endpoints. Qwen-Long and its `qwen-long-*` versions
place `fileid://<id>` references in the second system message. The first system
message comes from `req.system` or a leading text `MessageRole::System`, with a
default role when neither is present. Images are limited to 20,000,000 bytes,
other files to 150,000,000 bytes, and each request to 100 file references.
Qwen-Long images and documents must use `AttachmentRef` or a same-account
`ProviderFile` reference. Direct Base64, text document, and URL sources return
`UnsupportedCapability` before a model request is sent. Qwen knowledge-base
retrieval is a separate Responses `file_search` feature. MiniMax M3 video
understanding requires upload
with `video_understanding` followed by a `mm_file://<id>` reference; M2.7 does
not accept video blocks. OpenRouter workspace files, Moonshot file management,
and GLM/Z.AI auxiliary file endpoints are not treated as general chat
attachments. Everything else remains inline when the wire supports it, or
returns `UnsupportedCapability`.

Gemini Files accepts PDFs up to 50,000,000 bytes (50 MB); the general 2 GiB
upload limit still applies to other Gemini file types.
Gemini video attachments use Files upload and a `fileData` URI when the model
declares video input and the MIME type is supported. Smaller video data can
also be sent inline. Direct provider file references must include a matching
media type and an input purpose compatible with the selected model.
Gemini file references follow the model's declared image, audio, video, or
document input modality. MOV videos may use `video/mov` or `video/quicktime`.
Gemini video file operations have a two-hour initial budget. The upload may
use that budget; processing polls for up to ten minutes and then returns
`LlmError::ProviderFileProcessing` with the scoped file reference when still
pending. Call `FileService::resume_gemini_processing()` with that reference
later, while the file remains within Gemini's 48-hour retention. Direct
`FileService` callers can configure upload and polling limits separately with
`with_gemini_upload_timeout()` and `with_gemini_processing_timeout()`.
The processing deadline includes authentication and each status request.
Polling failures, including HTTP errors and malformed status responses, also
return the scoped reference because readiness could not be confirmed. An
explicit `FAILED` status is a terminal processing error.
Completions containing video blocks default to a two-hour total deadline;
an explicit `RequestOptions::total_timeout` still takes precedence.
For first-party Anthropic Messages, the client measures the encoded request
body against the 32 MB limit and uploads app-owned images as file references
when several inline images would exceed it. If the body is still too large,
the client returns `RequestTooLarge` before uploading files. The preflight
encodes the planned file references with a bounded file ID size; the final
encoded body is checked again before sending. First-party OpenAI and Anthropic
accept JPEG, PNG, GIF, and WebP images; Gemini accepts JPEG, PNG, WebP, HEIC,
and HEIF. Unsupported image MIME types are rejected before inline or uploaded
image input is sent, including directly supplied Base64 images.

For explicit provider-side management, `client::files::FileService` exposes
capability checks, upload, metadata, listing, deletion, raw-byte download, and
text extraction where documented. MiniMax uploads take a `FilePurpose`; listing
requires the purpose through `list_for_purpose()`, and deletion uses the
purpose returned when the file was uploaded. MiniMax supports different
purpose sets for listing and deletion, so video-understanding uploads cannot
be treated as ordinary listable or deletable files. `download` returns original
bytes only when that provider marks them downloadable; text extraction is a
separate result. Each service instance is bound to one profile, authenticator,
credential and account scope.
Purpose-filtered listing is available for OpenAI, Qwen, and MiniMax; providers
without a documented purpose filter return `UnsupportedCapability`.
After a direct Qwen upload, use `get` to wait for `processed` before sending its
ID to the model; automatic app attachments perform that wait in the client.

FileService uses no-redirect transport operations for credentialed requests.
Custom `Transport` implementations provide one `send` method returning raw
bytes and must disable automatic redirects and retries. The shared executor
enforces its 64 MiB download cap while reading the stream. OpenAI model-input uploads for `input_file` documents are capped at
50,000,000 bytes. First-party OpenAI Responses and Chat requests also enforce
the 50,000,000-byte combined limit for file inputs with known sizes before
uploading app attachments. Callers supplying provider file IDs directly must
keep their combined file size within the provider limit.
Supported image inputs use the Files API's 512 MiB upload cap.
Attachment resolution remains limited to 64 MiB total per request.

For automatic `AttachmentRef` requests, set `RequestOptions::file_account_scope`
to a stable, non-secret identity for the provider account if uploads should be
reused across calls. Keep it different for different logins to the same
provider. If it is omitted, the client creates an internal per-attempt scope to
bind that request's upload; it is not reused across calls. Automatically
prepared files can be reuploaded once when a 404 identifies one of the files
used by that request, even without a stable scope. Automatic Anthropic,
OpenAI, and xAI uploads expire after 24 hours, and their cached references
stop being reused after 23 hours.
Qwen automatic uploads are not cached across requests. The client paces their
upload, status, and deletion calls, waits for parsing, and attempts deletion
after a complete response or stream. Foreground cleanup uses the remaining
request deadline, or at most 120 seconds at stream EOF without one. Cancellation,
early stream drop, deadline exhaustion, and deletion errors schedule background
retries. Qwen files do not expire server-side, so applications should periodically
list and delete any files left behind by process exit or persistent failures.
Explicit `FileService::upload` calls retain caller-managed lifetime and should
be deleted by the host when no longer needed.

Direct `FileService` uploads for `FilePurpose::ModelInput` or
`FilePurpose::VideoUnderstanding`, and directly supplied `ProviderFileSource`
references require an explicit, non-empty account scope.
Pass the same stable scope to `FileService` and as
`RequestOptions::file_account_scope` when sending the reference. The automatic
request-local scope is internal and cannot be used for a new direct model input;
it can be supplied to `resume_gemini_processing()` for the same pending upload.
Never pass an API key, OAuth token, or other credential as this scope.

Provider file references also carry a deterministic FNV-1a 128 fingerprint of
the profile's complete configured `base_url`. The reference stores the
fingerprint, not the URL, so userinfo and query parameters are not copied into
it. The binding includes the full URL and rejects use after the configured
endpoint changes; serialized references without a fingerprint are rejected.
This fingerprint is an endpoint identifier, not an authentication credential.
Normally obtain a `ProviderFileSource` from `ProviderFileRef::model_reference()`;
if constructing one manually, set `endpoint_fingerprint` with
`provider_file_endpoint_fingerprint(profile.base_url)`.

Attachments resolve once per revision into shared `Bytes`. Each attempt validates a complete transfer plan before uploading, and passes borrowed content bindings to the codec. Upload/cache paths never create Base64 strings. Inline Base64 writes directly into the final JSON serializer; Anthropic preflight uses the same serializer with a counting writer. One cleanup lease owns temporary files for each attempt, independently of cache-reference bookkeeping.
