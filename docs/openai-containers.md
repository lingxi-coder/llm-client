# OpenAI Containers

`openai_containers` 提供绑定作用域的 OpenAI Containers API：创建、读取、列表和删除容器，以及创建、列表、读取、下载和删除容器文件。创建容器文件支持原始 multipart 上传，也支持通过现有 OpenAI Files API `file_id` 挂载文件。创建容器时可以同时传入要复制到容器的 `file_ids` 和 `expires_after`。复制现有文件 ID 前，服务会使用本次调用的凭据读取 Files API 元数据，核对返回的 ID，并拒绝已经过期或将在 60 秒请求缓冲期内到期的文件。源文件不可用或元数据格式错误时，会在修改容器/文件前停止。这是提交前核对，并不锁定资源；其他调用者仍可能随后删除源文件。服务不会施加文档未规定的文件 purpose 限制。服务对象只保存 transport 和非密钥作用域；宿主在每次调用时传入 `&Secret<String>`，因此可以自行刷新和轮换凭据。服务不会自动翻页、重试写操作或缓冲下载文件内容；创建/挂载现有 Files API 文件时，会先读取元数据，再各发送一次写操作。

路由依据 OpenAI 的 [Containers 创建 API reference](https://developers.openai.com/api/reference/resources/containers/methods/create)、[容器文件 API reference](https://developers.openai.com/api/reference/resources/containers/subresources/files/methods/create) 和 [Files 查询 API reference](https://developers.openai.com/api/reference/resources/files/methods/retrieve)。下载方法从官方文档中的容器文件内容路由返回原始字节流。

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

    // 宿主决定何时删除容器及其文件。
    // service.delete_file(&file.reference, &api_key).await?;
    // service.delete_container(&container.reference, &api_key).await?;
    Ok(())
}
```

OpenAI 文档列出的内存级别为 `1g`、`4g`、`16g` 和 `64g`；省略时采用文档中的 `1g` 默认值。可选过期设置以 `last_active_at` 为锚点、单位为分钟。容器在 20 分钟没有活动后可能过期；过期或删除容器时，容器内文件也会被清除。需要保留的生成文件应在容器仍有效时下载。生命周期细节见 [Code Interpreter guide](https://developers.openai.com/api/docs/guides/tools-code-interpreter#containers)。

源 Files API 对象与复制后生成的 `container.file` 对象有不同 ID。OpenAI 将容器文件接口描述为复制源文件，但文档没有说明源文件之后到期或被删除时，复制到容器的字节会如何处理。服务只在复制前验证源文件可访问且没有即将到期；不会延长或删除源文件。复制期间请保持源文件可用，并在容器过期前下载需要保留的容器文件。Files API 元数据中的 `expires_at` 是 Unix 时间戳；字段缺失或为 `null` 表示元数据中没有计划到期时间。上传过期策略和文件保留细节见 [Files API](https://developers.openai.com/api/reference/typescript/resources/files/methods/create)。

列表方法每次只获取一页。容器和文件列表的 limit 范围是 1–100，API 默认值为 20；将每页返回的 `next_cursor` 作为后续显式调用的 `after` 参数。`code_interpreter_call` 返回的容器 ID 可以通过 `OpenAiContainerRef::from_id(&scope, id)` 绑定到当前作用域。若引用来自其他 profile、endpoint、账户范围、容器或文件 ID，服务会在发送请求前拒绝。

容器创建、文件上传或挂载、删除都属于写操作。传输错误或格式异常的成功响应导致结果不确定时，服务返回 `OutcomeUnknown` 或 `OutcomeUnknownResponse`，不会自动重复操作。
