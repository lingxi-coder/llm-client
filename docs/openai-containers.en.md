# OpenAI Containers

`openai_containers` provides direct, scoped access to OpenAI's explicit Containers API: create, retrieve, list, and delete containers, plus create, list, retrieve, download, and delete container files. File creation supports both a raw multipart upload and attaching an existing OpenAI Files API `file_id`. Container creation can include both `file_ids` (IDs to copy into the container) and `expires_after`. Before copying any existing file ID, the service retrieves its Files API metadata with the same per-call credential, verifies the returned ID, and rejects an explicit expiry that is already past or within a 60-second request margin. An unavailable or malformed source file stops the operation before the container/file mutation. This is a preflight check, not a reservation; the file can still be deleted by another actor afterward. The service does not impose an undocumented file-purpose rule. It stores only the transport and non-secret scope; the host passes `&Secret<String>` to each operation so it retains credential refresh and rotation ownership. It does not paginate, retry mutations, or buffer downloaded file content; source-file create/attach operations issue metadata reads before the single write.

The routes follow the [Containers create reference](https://developers.openai.com/api/reference/resources/containers/methods/create), [container file reference](https://developers.openai.com/api/reference/resources/containers/subresources/files/methods/create), and [Files retrieve reference](https://developers.openai.com/api/reference/resources/files/methods/retrieve). The download method returns the raw byte stream from the documented container file content route.

```rust,ignore
use bytes::Bytes;
use futures::StreamExt;
use lingxi_llm_client::{
    files::UploadFile,
    providers::openai::containers::{
        OpenAiContainerCreateRequest, OpenAiContainerFileListOptions,
        OpenAiContainerListOptions, OpenAiContainerMemoryLimit,
        OpenAiContainerScope, OpenAiContainersError, OpenAiContainersService,
    },
    protocol::Secret,
    transport::Transport,
};

async fn run(
    transport: &dyn Transport,
    api_key: Secret<String>,
) -> Result<(), OpenAiContainersError> {
    let scope = OpenAiContainerScope::new("openai-production", "account-123")?;
    let service = OpenAiContainersService::new(transport, scope)?;
    let request = OpenAiContainerCreateRequest::new("analysis")?
        .with_memory_limit(OpenAiContainerMemoryLimit::G4)
        .with_expiration_minutes(30)?
        .with_file_ids(["file-from-files-api"])?;
    let container = service.create_container(&request, &api_key).await?;

    let _page = service
        .list_containers(&OpenAiContainerListOptions::new(), &api_key)
        .await?;
    let upload = UploadFile {
        filename: "notes.txt".into(),
        media_type: "text/plain".into(),
        bytes: Bytes::from_static(b"Container input"),
    };
    let file = service.upload_file(&container.reference, &upload, &api_key).await?;
    let _files = service
        .list_files(&container.reference, &OpenAiContainerFileListOptions::new(), &api_key)
        .await?;
    let mut content = service.download_file(&file.reference, &api_key).await?;
    while let Some(chunk) = content.next().await {
        let _bytes = chunk?;
    }

    // The host owns when to delete the container and its files.
    // service.delete_file(&file.reference, &api_key).await?;
    // service.delete_container(&container.reference, &api_key).await?;
    Ok(())
}
```

OpenAI documents memory tiers `1g`, `4g`, `16g`, and `64g`; omitting the tier leaves the documented `1g` default. The optional expiration setting is measured in minutes after `last_active_at`. Containers can expire after 20 minutes without activity, and file data is discarded when the container expires or is deleted. Download any generated files you need while the container is active. See the [Code Interpreter guide](https://developers.openai.com/api/docs/guides/tools-code-interpreter#containers) for the lifecycle details.

The source Files API object and the resulting `container.file` object have separate IDs. OpenAI describes the container operation as copying the source file, but its docs do not state how a source file's later expiry or deletion affects the copied bytes. The service only verifies the source is accessible and not about to expire at copy time; it does not extend or delete the source. Keep the source file available while copying, and download needed container files before the container expires. The original Files API object's `expires_at` is a Unix timestamp; omitted or `null` means there is no scheduled expiry in the metadata. See the [Files API](https://developers.openai.com/api/reference/typescript/resources/files/methods/create) for upload expiry policy and file retention.

List methods fetch one page. Container and file list limits are 1–100, with an API default of 20; pass each page's `next_cursor` as `after` in a later explicit call. A container reference returned by `code_interpreter_call` can be bound with `OpenAiContainerRef::from_id(&scope, id)`. A reference used with another profile, endpoint, account scope, container, or file ID is rejected before a request is sent.

Container creation, file upload or attachment, and deletion are writes. If a transport error or malformed success response makes the outcome uncertain, the service returns `OutcomeUnknown` or `OutcomeUnknownResponse`; it never repeats the operation automatically.
