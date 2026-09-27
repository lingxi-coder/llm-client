# 文件附件

`lingxi-llm-client` 将应用长期保存的附件与 provider 为自身账号创建的文件分开。会话应保存应用自己的 `AttachmentRef`。Provider 文件 ID 或 URI 只用于某次模型请求，不能用来在其他设备上显示原图。

```rust
use lingxi_llm_client::protocol::AttachmentRef;

let attachment = AttachmentRef {
    attachment_id: "att_...".into(), // 应用附件存储中的稳定 ID
    revision: "v1".into(),           // 内容不可变的版本
    filename: "diagram.png".into(),
    media_type: "image/png".into(),
    size_bytes: 12345,
};
```

在消息中使用 `ImageSource::Attachment`、`DocumentSource::Attachment` 或 `VideoSource::Attachment`，并在 client builder 上注册 `AttachmentResolver`。Resolver 从应用控制的存储中读取指定 ID 和版本的原始字节。客户端会检查声明大小和实际字节数，单次请求的附件总量上限为 64 MiB；解析只修改请求副本，调用方的会话消息仍保留稳定引用。

Remote 场景中，应用先保存原文件，再向共享会话发送引用。接收设备通过应用的鉴权附件服务显示文件，并把同一个引用交给 `llm-client`。执行模型请求的设备通过 resolver 从该服务读取字节，因此图片显示不依赖 provider 文件的保留时间，也不依赖 provider 是否允许下载用户上传内容。

鉴权预览 URL 由应用生成，URL 生命周期也由应用负责；Provider 文件 URI 不是预览 URL。如果附件服务暂时无法读取指定版本，界面应显示“附件暂不可用”，网络恢复后重试。Resolver 向模型请求返回清晰的 `LlmError`。客户端不会猜测一个 URL，也不会拿 Provider 文件 ID 代替应用附件 ID。

Provider 文件上传与模型输入是两种不同能力。客户端仅在适配器确认当前 profile、模型、媒体类型和用途都支持时才使用 Provider 文件引用：OpenAI Responses、Anthropic Messages 和 Gemini 有原生模型输入引用；OpenAI Chat 仅对 PDF 提供文件 ID 引用；xAI 只对文档搜索输入提供已确认的引用。Qwen 北京和新加坡的 Files 接口（包括 `{WorkspaceId}.cn-beijing.maas.aliyuncs.com` 与 `{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com`）可用 `file-extract` 上传文档和受支持的图片，但 Qwen-Long 及其 `qwen-long-*` 版本的模型输入仅支持北京端点。Qwen-Long 会将 `fileid://<id>` 引用放入第二条 system 消息；角色消息可来自 `req.system` 或首条文本 `MessageRole::System`，两者都没有时客户端会补上默认角色。Qwen-Long 的图片经 Files API 上传，单张上限 20,000,000 字节；其他文件上限 150,000,000 字节，单次请求最多引用 100 个文件。Qwen-Long 的图片和文档须使用 `AttachmentRef` 或同账号的 `ProviderFile` 引用；直接传入 Base64、文本型 Document 或 URL 会在请求发送前返回 `UnsupportedCapability`，避免生成不受支持的 Chat 请求。Qwen Search 的知识库检索则是独立的 Responses `file_search` 功能。MiniMax M3 视频理解需要先按 `video_understanding` 上传，再以 `mm_file://<id>` 引用；M2.7 不接受视频块。OpenRouter workspace 文件、Moonshot 文件管理及 GLM/Z.AI 辅助文件接口不会被当作普通聊天附件。其他情况在协议支持时以内联数据发送；无法表示时返回 `UnsupportedCapability`。

Gemini Files 的 PDF 上传上限为 50,000,000 字节（50 MB）；其他 Gemini 文件类型仍使用通用的 2 GiB 上传上限。
Gemini 视频附件在模型声明支持视频输入且 MIME 类型受支持时，会通过 Files 上传，并以 `fileData` URI 传给模型；较小的视频也可内联发送。直接传入的 provider 文件引用须携带与内容块匹配的媒体类型，且用途与所选模型的输入能力兼容。
Gemini 文件引用会按模型声明的图片、音频、视频或文档输入模态判断；MOV 视频可使用 `video/mov` 或 `video/quicktime`。视频文件的首次操作预算为两小时，上传可使用这段预算；处理先轮询最多十分钟，仍未完成时返回携带账号绑定文件引用的 `LlmError::ProviderFileProcessing`。处理期限包括认证和每次状态查询。轮询时遇到 HTTP 错误或异常状态响应，也会保留此引用，因为此时无法确认文件是否就绪；明确的 `FAILED` 状态仍是终止性处理错误。文件仍在 Gemini 的 48 小时保留期内时，可通过 `FileService::resume_gemini_processing()` 稍后继续轮询。直接使用 `FileService` 的调用方可分别通过 `with_gemini_upload_timeout()` 和 `with_gemini_processing_timeout()` 调整期限。含视频块的 completion 默认总期限为两小时，显式设置的 `RequestOptions::total_timeout` 仍优先。
对于 Anthropic 官方 Messages 接口，客户端会按编码后的请求体检查 32 MB 上限；多张内联图片可能超限时，会将应用附件上传并改用文件引用。上传前会用限定长度的文件 ID 模拟实际请求体，预测超限时直接返回 `RequestTooLarge`；发送前还会检查最终请求体。官方 OpenAI 和 Anthropic 接受 JPEG、PNG、GIF、WebP 图片；Gemini 接受 JPEG、PNG、WebP、HEIC、HEIF。包括直接传入的 Base64 图片在内，不支持的图片 MIME 类型会在发送或上传前被拒绝。

需要直接管理 provider 文件时，可使用 `client::files::FileService` 查询能力、上传、读取元数据、列举、删除、下载原始字节或提取文本（仅限服务文档明确支持的操作）。直接调用 `FileService::upload` 上传 Qwen 文件后，应通过 `get` 查询状态，确认 `processed` 后再把 ID 交给模型；自动处理的应用附件则由客户端完成这一步。MiniMax 文件上传由 `FilePurpose` 选择用途；其列表接口必须通过 `list_for_purpose()` 提供 purpose，删除则使用上传时返回的 purpose。MiniMax 列表与删除支持的 purpose 集合不同，视频理解文件不能用普通上传文件 purpose 假装可列举或删除。OpenAI、Qwen 和 MiniMax 支持按用途列举；未确认用途筛选接口的 provider 会返回 `UnsupportedCapability`。`download` 只在 provider 标记文件可下载时返回原始字节；文本提取是单独操作。每个 service 实例绑定一个 profile、认证器、凭证和账号范围。

Microsoft Foundry 的 Files API 仅适用于明确托管在 Anthropic 上的部署。可使用 `FileService::new_foundry(..., FoundryHosting::Anthropic, ...)` 显式启用，无需伪造聊天模型目录；也可用 `new_foundry_for_model(...)` 校验 profile 中实际选中的模型行。两种方式都要求严格的 `https://{resource}.services.ai.azure.com/anthropic` 路由和稳定、不含密钥的账号 scope。原有仅接收 profile 的 `new()` 不会推断 Foundry 文件支持。返回的引用保留 `FoundryClaude` 协议和规范化资源 endpoint 身份。文件按 workspace/resource 管理，不绑定部署名或底层模型，因此同一 resource 和账号 scope 下兼容的部署可复用文件。Foundry Code Execution 会在加入 `container_upload` 前检查该 scope。

Anthropic 的 `FileService::list_by_ids(&[ProviderFileRef])` 可在一次请求中查询最多 100 个已按 scope 绑定的引用。provider 会省略不存在或不可访问的 ID；如果调用方需要检测缺失项，应自行比较请求 ID 集合和返回 ID 集合。客户端会拒绝其他账号/resource 的引用，也会拒绝响应中未请求的 ID。Anthropic 说明用户上传的文件不可下载；只有元数据标记 `downloadable: true` 时，`download` 才会读取内容，例如 Code Execution 输出文件。Files API 指南写明单文件最大 500 MB；客户端本地上传上限用于安全预检，最终是否接收由 provider 决定（[Files API](https://platform.claude.com/docs/en/build-with-claude/files)，[Foundry 托管方式](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry)）。

FileService 会通过禁止自动跟随重定向的传输操作发送带凭证的请求。自定义 `Transport` 只实现返回原始字节流的 `send`，必须关闭自动重定向和自动重试。公共执行器在读取流时执行 64 MiB 的下载上限。OpenAI `input_file` 文档模型输入的上传上限为 50,000,000 字节；官方 OpenAI Responses 和 Chat 请求还会在上传应用附件前检查已知大小的文件输入合计不超过 50,000,000 字节。调用方直接传入 provider 文件 ID 时，仍需自行确保文件合计大小符合该限制。支持的 OpenAI 图像输入使用 Files API 的 512 MiB 上传上限。xAI 普通 multipart 文件上传上限为 50,000,000 字节，缓冲和流式上传均在发送前拒绝更大输入（[xAI 上传参考](https://docs.x.ai/developers/rest-api-reference/files/upload)）。应用附件解析的单次请求总量仍限制为 64 MiB。

对于自动处理的 `AttachmentRef` 请求，如果希望跨请求复用上传结果，请把 `RequestOptions::file_account_scope` 设置为稳定且不含密钥的 provider 账号标识。相同 provider 下的不同登录应使用不同值。未设置时，客户端会生成内部的单次尝试 scope 来绑定本次请求上传的文件，不会跨请求复用；若 404 错误明确指向本次使用的文件，自动准备的文件仍可重新上传并重试一次。自动上传到 Anthropic、OpenAI 和 xAI 的文件将在 24 小时后过期，缓存中的引用最多复用 23 小时。Qwen 自动上传不会跨请求缓存：客户端按连接限制自动上传、查询和删除的速率，等待文件解析为 `processed` 后才发送模型请求；普通响应和流结束后的删除工作受请求总截止时间约束，流请求未指定总期限时最多等待 120 秒。请求取消、流提前丢弃、截止时间耗尽及删除失败时会在后台重试。Qwen 服务端不提供文件自动过期，因此进程或运行时在后台清理完成前退出、或持续删除失败后，应用仍应通过 `FileService::list` 分页列出文件并用 `delete` 定期清理遗留文件。直接调用 `FileService::upload` 的文件生命周期由调用方管理，不再使用时应由应用删除。

如果输入来自一次性流，可使用
`FileService::upload_stream(UploadFileStream::new(filename, media_type,
size_bytes, stream))`。现有 REST 适配器会增量发送 multipart 上传；Gemini
Files 则通过 resumable start 请求和原始文件流请求完成上传。声明大小必须与
流实际产出的字节完全一致：长度不足、超出或流中断都会失败。multipart 适配器只有
在流精确到达 EOF 后才发送结束边界；Gemini 则按声明长度发送原始文件流。客户端会
在轮询输入流前完成本地预检。流只消费一次，不会自动
重试，也不会自动轮询 Gemini 处理状态。传输中断可能发生在 provider 已接受部分或
全部数据之后，因此 `FileUploadError::OutcomeUnknown` 表示调用方不应直接盲目重试；
可在 provider 支持时查询元数据或列举文件以核对结果。Gemini 返回 `PROCESSING` 时，
客户端以带 scope 的 `LlmError::ProviderFileProcessing` 返回引用，调用方可将其交给
`resume_gemini_processing()`。现有 `FileService::upload` 仍接受内存中的
`UploadFile`。自定义 `Transport` 必须实现 `send_stream` 才能使用
`upload_stream`；默认实现会在消费输入流前拒绝请求。

```rust,no_run
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    files::{FilePurpose, FileService, FileUploadError, UploadFileStream},
    protocol::{LlmError, ProviderProfile, Secret},
    Authenticator, Transport,
};

async fn upload_pdf(
    transport: &dyn Transport,
    profile: &ProviderProfile,
    authenticator: &dyn Authenticator,
    api_key: &Secret<String>,
) -> Result<(), FileUploadError> {
    let chunks = stream::iter([b"hello ".to_vec(), b"world".to_vec()])
        .map(|part| Ok::<_, LlmError>(part.into()));
    let files = FileService::new(
        transport,
        profile,
        Some(authenticator),
        Some(api_key),
        Some("account-42"),
    );
    let input = UploadFileStream::new("report.pdf", "application/pdf", 11, chunks);
    let _uploaded = files.upload_stream(input, FilePurpose::ModelInput).await?;
    Ok(())
}
```

直接通过 `FileService` 以 `FilePurpose::ModelInput`、`FilePurpose::VideoUnderstanding`、`FilePurpose::Batch` 或 `FilePurpose::AsyncTtsInput` 上传，以及直接传入 `ProviderFileSource`，都必须使用明确且非空的账号 scope。MiniMax 异步 TTS 文本文件也受此要求约束，使上传引用的账号绑定与异步 TTS 服务的检查一致。创建 `FileService` 和发送引用时，应使用同一个稳定 scope，并在请求选项中设置 `RequestOptions::file_account_scope`。自动请求生成的临时 scope 属于内部值，不能用于新的直接模型输入，但可用于同一待处理文件的 `resume_gemini_processing()`。不要把 API key、OAuth token 等凭证作为 scope。

Provider 文件引用还会携带配置文件 endpoint 的确定性 FNV-1a 128 指纹。大多数适配器绑定 profile 的完整 `base_url`；Foundry Files 绑定规范化后的 resource base，因此尾部斜杠不会改变身份。引用只保存指纹，不复制 URL，因此不会把 userinfo 或 query 参数写入引用。endpoint 变化后旧引用会被拒绝；缺少指纹的旧序列化引用也会被拒绝。该指纹用于标识 endpoint，不是认证凭证。

通常通过 `ProviderFileRef::model_reference()` 获取 `ProviderFileSource`；如果需要手动构造，普通 profile 应使用 `provider_file_endpoint_fingerprint(profile.base_url)` 设置 `endpoint_fingerprint`；Foundry 应对规范化的 `https://{resource}.services.ai.azure.com/anthropic` base 计算指纹。

`ProviderFileRef::model_reference()` 还会把原始 `expires_at` 时间戳复制到 `ProviderFileSource.expires_at: Option<String>`。它仅用于本地校验，不会进入供应商的模型输入 wire。存在的时间戳必须是 RFC 3339（支持小数秒和时区偏移），或以字符串表示的整数 Unix 秒；适配器会先把 JSON 数值到期时间转换为字符串。校验器不会猜测毫秒或秒，`"0"` 表示 Unix epoch，而不是永不过期。格式错误、无法表示的时间戳，以及小于或等于当前时间的到期值，都会返回 `LlmError::InvalidRequest`。允许缺少到期信息，但这不代表文件已处理就绪或仍然可用。

引用还会原样复制 provider 返回的 `processing_status`。它只用于本地校验，不会进入模型输入 wire。发送模型请求前，客户端会拒绝已知的 Gemini `PROCESSING`、Qwen `uploaded`/`processing` 状态；Gemini `FAILED` 和 Qwen `error` 会返回无效请求。Gemini `ACTIVE` 与 Qwen `processed` 是各自文档定义的就绪状态（[Gemini Files API](https://ai.google.dev/api/files)、[Qwen OpenAI-compatible File API](https://help.aliyun.com/en/model-studio/openai-file-interface)）。缺失或无法识别的状态仍按未知处理，不套用其他 provider 的状态含义。

高层客户端会在附件解析前检查已知到期时间。文件计划会在上传前检查直接引用及已知就绪状态；最终 codec 在编码前再次校验实际引用。到期时间还会在附件准备后，以及认证完成、即将写出模型请求前复查。直接调用 codec 也使用同一文件校验器。默认使用当前系统时间；直接调用方可通过 `CodecContext::with_file_validation_time` 指定自定义时钟值或进行确定性测试。已知到期校验失败不会触发自动刷新元数据或重新上传；已有上传缓存失效处理的其他行为保持不变。provider、profile、endpoint、协议及账号 scope 检查仍然适用。

`FileService::get()` 会在响应带有状态时刷新状态；若字段缺失，则保留最后一次已知状态。显式 `null` 会清除状态，使其恢复为未知。`resume_gemini_processing()` 返回最终元数据投影，因此 `ACTIVE` 会替换较早的 `PROCESSING`。聊天请求中的自动 Qwen 附件上传会等待 `processed`，并使用刷新后的引用。直接传入的模型输入引用不会隐式刷新或重试；Qwen 的直接 `FileService::upload()` 会返回 provider 报告的状态，调用方可先通过 `get()` 刷新后再使用。文件存储到期与 Anthropic 执行容器到期也是独立语义：Anthropic Files 支持[可选文件到期时间](https://platform.claude.com/docs/en/build-with-claude/files#file-expiration)，而执行容器的滚动 `expires_at` 只保留、不用于本地到期拒绝，详见 [Code Execution](anthropic-code-execution.md)。

附件每个 revision 只解析一次并共享 `Bytes`。每次 attempt 在上传前验证完整传输计划，通过借用的内容绑定交给 codec。上传／缓存路径不生成 Base64 字符串；inline Base64 直接写入最终 JSON，Anthropic 预检使用同一序列化实现和计数 writer。每次 attempt 的清理租约独立持有临时文件，与缓存引用记录分离。
