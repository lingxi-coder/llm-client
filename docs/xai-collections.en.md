# xAI Collections

`xai_collections` exposes xAI's managed document collections as a service separate from chat and conversation attachments. It supports collection creation, listing, lookup, configuration updates and deletion; file upload followed by collection attachment; document listing, lookup, batch metadata retrieval, reindexing and removal; and semantic search that returns collection-scoped document references.

The service follows xAI's split API boundary. Collection management, direct upload through `upload_document_stream`, attaching files, and document management use an xAI **Management API key** on `https://management-api.x.ai/v1`. The two-step `upload_file` operation and semantic search use a regular xAI **API key** on `https://api.x.ai/v1`. The crate never looks up or persists either key. Supply both keys through `XaiCollectionsCredentials` at each operation; the credential type is not serializable and its debug output is redacted.

```rust,ignore
use lingxi_llm_client::{
    files::UploadFileStream,
    protocol::Secret,
    providers::xai::collections::{
        XaiCollectionListRequest, XaiCollectionsConfig, XaiCollectionsCredentials,
        XaiCreateCollectionRequest, XaiRetrievalMode, XaiSearchRequest,
        XaiUpdateCollectionRequest,
    },
};

let provider = client.provider::<lingxi_llm_client::providers::xai::XaiClient>("xai-primary")?;
let service = provider.collections(XaiCollectionsConfig::new(
    "xai-primary",
    "xai-team/account-42", // caller-chosen stable, non-secret scope
))?;
let credentials = XaiCollectionsCredentials::new(
    Secret::new(api_key),
    Secret::new(management_api_key),
);
let collection = service
    .create_collection(&XaiCreateCollectionRequest {
        name: "Research notes".into(),
        description: Some("Internal reports".into()),
        index_configuration: None,
        chunk_configuration: None,
        field_definitions: vec![],
    }, &credentials)
    .await?;
let page = service
    .list_collections(&XaiCollectionListRequest::default(), &credentials)
    .await?;
let renamed = service
    .update_collection(
        &collection.reference,
        &XaiUpdateCollectionRequest {
            name: Some("Research notes 2026".into()),
            ..Default::default()
        },
        &credentials,
    )
    .await?;
assert_eq!(renamed.name, "Research notes 2026");
```

`XaiCollectionListRequest` exposes the documented collection `filter`, `order`, and `sort_by` query parameters; `XaiDocumentListRequest` exposes document `filter`, `order`, and `sort_by`. The API reference types `order` and `sort_by` as strings without publishing an allowed-value enum, so the client rejects empty or control-character values but leaves provider-specific validation to xAI. Document listing uses `filter`; the older `name` query parameter is deprecated in the xAI reference.

`update_collection` sends xAI's `PUT /v1/collections/{collection_id}` with the Management API key and returns the updated collection object. Supply one or more documented fields: `name`, `description`, `chunk_configuration`, or nonempty `field_definitions`. The request must include at least one field; the client checks a supplied name, chunk-configuration object, and field-definition keys before dispatch, then verifies the response collection ID. The operation is sent once. If transport fails after dispatch, xAI may have applied the update; call `get_collection` to inspect current state before deciding whether to submit another update.

Upload is deliberately two explicit calls, matching xAI's documented flow. `upload_file` returns a scoped file reference first; `add_document` attaches it to a collection and accepts optional metadata fields. If attachment fails, the caller still has the file ID needed to retry the attach operation.

```rust,ignore
let uploaded = service.upload_file(&file, &credentials).await?;
let document = service
    .add_document(
        &collection.reference,
        &uploaded.reference,
        Some(&serde_json::json!({"author":"Mina","year":"2026"})),
        &credentials,
    )
    .await?;
let indexed = service.get_document(&document, &credentials).await?;
service.reindex_document(&document, &credentials).await?;
let indexed_again = service.get_document(&document, &credentials).await?;
```

You can also upload and attach a document in one call through the Management API multipart route. `upload_document_stream` accepts an `UploadFileStream`, optional JSON-object metadata fields, and returns the collection-scoped `XaiDocument`:

```rust,ignore
let document = service
    .upload_document_stream(
        &collection.reference,
        UploadFileStream::from_bytes("paper.pdf", "application/pdf", pdf_bytes),
        Some(&serde_json::json!({"author":"Mina"})),
        &credentials,
    )
    .await?;
```

`batch_get_documents` accepts one or more `XaiDocumentRef` values from the same collection and returns the metadata subset supplied by xAI. The API does not document how missing IDs are represented, so the client does not require the returned count to match the requested count. It rejects unrequested or duplicate IDs in the response.

Use `list_documents` or `get_document` to inspect provider indexing state. `reindex_document` submits xAI's documented `PATCH /v1/collections/{collection_id}/documents/{file_id}` operation with the Management API key. It returns after submission and does not wait for indexing; callers can read the document status later. The service does not poll automatically. Collection search accepts one or more `XaiCollectionRef` values, an optional xAI filter expression, and returns each match as `XaiDocumentRef` values bound to the collections reported by xAI. A response that cites a collection outside the requested set is rejected as malformed.

Search defaults to xAI's hybrid retrieval. Set `XaiSearchRequest::retrieval_mode` to `Some(XaiRetrievalMode::Keyword)`, `Semantic`, or `Hybrid` to select a documented mode explicitly. The wire request encodes it as `{"type":"keyword"}`, `{"type":"semantic"}`, or `{"type":"hybrid"}`; `None` omits the field so the provider applies its default.

```rust,ignore
let results = service
    .search(
        &XaiSearchRequest {
            query: "Find revenue guidance".into(),
            collections: vec![collection.reference.clone()],
            filter: None,
            retrieval_mode: Some(XaiRetrievalMode::Keyword),
        },
        &credentials,
    )
    .await?;
```

Collection, uploaded-file, and document references carry the provider ID, profile name, non-secret fingerprints of both API roots, and caller-supplied account scope. Reusing one against a different profile, endpoint, or account is rejected before sending a request. References can be serialized; credentials cannot.

The Collections guide documents a 100 MB file limit; `upload_document_stream` follows this direct Collections upload route and preflights at 100,000,000 bytes. In the two-step flow, `upload_file` calls the separate Files API `POST /v1/files`, whose reference documents a 50 MB limit; this implementation enforces 50,000,000 bytes there. xAI may also require account credits for uploads and collection indexing.

`upload_document_stream` is a one-shot mutation and is never retried automatically. If transport fails after dispatch or a successful response cannot be decoded, the outcome may be unknown; inspect the collection with `list_documents` or `get_document` before retrying. Global Files API deletion is available separately through the crate's file service. `reindex_document` submits xAI's documented `PATCH /v1/collections/{collection_id}/documents/{file_id}` operation with the Management API key and returns without waiting for indexing to finish; callers can inspect document status later. If transport is interrupted, xAI may already have accepted the request, so callers should read the document status before deciding whether to retry. The service does not automatically wait for indexing or make live provider calls. Its contract tests use an injected mock transport.

References: [xAI Collections overview](https://docs.x.ai/developers/files/collections), [Collections REST reference](https://docs.x.ai/developers/rest-api-reference/collections), [Collection management reference](https://docs.x.ai/developers/rest-api-reference/collections/collection), [Collections API guide](https://docs.x.ai/developers/files/collections/api), [Search reference](https://docs.x.ai/developers/rest-api-reference/collections/search), and [Files upload reference](https://docs.x.ai/developers/rest-api-reference/files/upload).

## Continue through the file service

An upload through `upload_file` can be queried, downloaded, or deleted through the ordinary `FileService` after converting its result. The conversion checks provider, profile, API root, protocol, and account scope, and preserves known file metadata, including expiry. It does not make a network call or confirm that the file is still available.

```rust,no_run
use lingxi_llm_client::{files::ProviderFileRef, protocol::{LlmError, ProviderProfile}, providers::xai::collections::XaiUploadedFile};
fn file_reference(uploaded: &XaiUploadedFile, profile: &ProviderProfile, account: &str) -> Result<ProviderFileRef, LlmError> {
    uploaded.to_provider_file_ref(profile, account)
}
```

Pass the returned reference to a `FileService` configured with the same profile and account scope. Removing collection membership and deleting the underlying global file are separate explicit operations.
