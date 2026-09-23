# 文件附件

`lingxi-llm-client` 将应用长期保存的附件与 provider 为自身账号创建的文件分开。会话应保存应用自己的 `AttachmentRef`。Provider 文件 ID 或 URI 只用于某次模型请求，不能用来在其他设备上显示原图。

```rust
AttachmentRef {
    attachment_id: "att_...", // 应用附件存储中的稳定 ID
    revision: "v1",           // 内容不可变的版本
    filename: "diagram.png",
    media_type: "image/png",
    size_bytes: 12345,
}
```

在消息中使用 `ImageSource::Attachment`、`DocumentSource::Attachment` 或 `VideoSource::Attachment`，并在 client builder 上注册 `AttachmentResolver`。Resolver 从应用控制的存储中读取指定 ID 和版本的原始字节。客户端会检查声明大小和实际字节数，单次请求的附件总量上限为 64 MiB；解析只修改请求副本，调用方的会话消息仍保留稳定引用。

Remote 场景中，应用先保存原文件，再向共享会话发送引用。接收设备通过应用的鉴权附件服务显示文件，并把同一个引用交给 `llm-client`。执行模型请求的设备通过 resolver 从该服务读取字节，因此图片显示不依赖 provider 文件的保留时间，也不依赖 provider 是否允许下载用户上传内容。

鉴权预览 URL 由应用生成，URL 生命周期也由应用负责；Provider 文件 URI 不是预览 URL。如果附件服务暂时无法读取指定版本，界面应显示“附件暂不可用”，网络恢复后重试。Resolver 向模型请求返回清晰的 `LlmError`。客户端不会猜测一个 URL，也不会拿 Provider 文件 ID 代替应用附件 ID。

Provider 文件上传与模型输入是两种不同能力。客户端仅在适配器确认当前 profile、模型、媒体类型和用途都支持时才使用 Provider 文件引用：OpenAI Responses、Anthropic Messages 和 Gemini 有原生模型输入引用；OpenAI Chat 仅对 PDF 提供文件 ID 引用；xAI 只对文档搜索输入提供已确认的引用。Qwen 北京和新加坡的 Files 接口（包括 `{WorkspaceId}.cn-beijing.maas.aliyuncs.com` 与 `{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com`）可用 `file-extract` 上传文档和受支持的图片，但 Qwen-Long 及其 `qwen-long-*` 版本的模型输入仅支持北京端点。Qwen-Long 会将 `fileid://<id>` 引用放入第二条 system 消息；角色消息可来自 `req.system` 或首条文本 `MessageRole::System`，两者都没有时客户端会补上默认角色。Qwen-Long 的图片经 Files API 上传，单张上限 20,000,000 字节；其他文件上限 150,000,000 字节，单次请求最多引用 100 个文件。Qwen-Long 的图片和文档须使用 `AttachmentRef` 或同账号的 `ProviderFile` 引用；直接传入 Base64、文本型 Document 或 URL 会在请求发送前返回 `UnsupportedCapability`，避免生成不受支持的 Chat 请求。Qwen Search 的知识库检索则是独立的 Responses `file_search` 功能。MiniMax M3 视频理解需要先按 `video_understanding` 上传，再以 `mm_file://<id>` 引用；M2.7 不接受视频块。OpenRouter workspace 文件、Moonshot 文件管理及 GLM/Z.AI 辅助文件接口不会被当作普通聊天附件。其他情况在协议支持时以内联数据发送；无法表示时返回 `UnsupportedCapability`。

Gemini Files 的 PDF 上传上限为 50,000,000 字节（50 MB）；其他 Gemini 文件类型仍使用通用的 2 GiB 上传上限。
对于 Anthropic 官方 Messages 接口，客户端会按编码后的请求体检查 32 MB 上限；多张内联图片可能超限时，会将应用附件上传并改用文件引用。上传前会用限定长度的文件 ID 模拟实际请求体，预测超限时直接返回 `RequestTooLarge`；发送前还会检查最终请求体。官方 OpenAI 和 Anthropic 接受 JPEG、PNG、GIF、WebP 图片；Gemini 接受 JPEG、PNG、WebP、HEIC、HEIF。包括直接传入的 Base64 图片在内，不支持的图片 MIME 类型会在发送或上传前被拒绝。

需要直接管理 provider 文件时，可使用 `client::files::FileService` 查询能力、上传、读取元数据、列举、删除、下载原始字节或提取文本（仅限服务文档明确支持的操作）。直接调用 `FileService::upload` 上传 Qwen 文件后，应通过 `get` 查询状态，确认 `processed` 后再把 ID 交给模型；自动处理的应用附件则由客户端完成这一步。MiniMax 文件上传由 `FilePurpose` 选择用途；其列表接口必须通过 `list_for_purpose()` 提供 purpose，删除则使用上传时返回的 purpose。MiniMax 列表与删除支持的 purpose 集合不同，视频理解文件不能用普通上传文件 purpose 假装可列举或删除。`download` 只在 provider 标记文件可下载时返回原始字节；文本提取是单独操作。每个 service 实例绑定一个 profile、认证器、凭证和账号范围。

FileService 会通过禁止自动跟随重定向的传输操作发送带凭证的请求。自定义 `Transport` 必须实现 `execute_no_follow`；文件下载还必须实现 `open_stream_no_follow`。不支持的操作会安全失败，客户端会在读取流时执行 64 MiB 的下载上限。OpenAI `input_file` 文档模型输入的上传上限为 50,000,000 字节；官方 OpenAI Responses 和 Chat 请求还会在上传应用附件前检查已知大小的文件输入合计不超过 50,000,000 字节。调用方直接传入 provider 文件 ID 时，仍需自行确保文件合计大小符合该限制。支持的图像输入使用 Files API 的 512 MiB 上传上限。应用附件解析的单次请求总量仍限制为 64 MiB。

对于自动处理的 `AttachmentRef` 请求，如果希望跨请求复用上传结果，请把 `RequestOptions::file_account_scope` 设置为稳定且不含密钥的 provider 账号标识。相同 provider 下的不同登录应使用不同值。未设置时，客户端会生成内部的单次尝试 scope 来绑定本次请求上传的文件，不会跨请求复用；若 404 错误明确指向本次使用的文件，自动准备的文件仍可重新上传并重试一次。自动上传到 Anthropic、OpenAI 和 xAI 的文件将在 24 小时后过期，缓存中的引用最多复用 23 小时。Qwen 自动上传不会跨请求缓存：客户端按连接限制自动上传、查询和删除的速率，等待文件解析为 `processed` 后才发送模型请求；普通响应和流结束后的删除工作受请求总截止时间约束，流请求未指定总期限时最多等待 120 秒。请求取消、流提前丢弃、截止时间耗尽及删除失败时会在后台重试。Qwen 服务端不提供文件自动过期，因此进程或运行时在后台清理完成前退出、或持续删除失败后，应用仍应通过 `FileService::list` 分页列出文件并用 `delete` 定期清理遗留文件。直接调用 `FileService::upload` 的文件生命周期由调用方管理，不再使用时应由应用删除。

直接通过 `FileService` 以 `FilePurpose::ModelInput` 或 `FilePurpose::VideoUnderstanding` 上传，以及直接传入 `ProviderFileSource`，都必须使用明确且非空的账号 scope。创建 `FileService` 和发送引用时，应使用同一个稳定 scope，并在请求选项中设置 `RequestOptions::file_account_scope`。自动请求生成的临时 scope 属于内部值，不能用于直接传入的文件引用。不要把 API key、OAuth token 等凭证作为 scope。

Provider 文件引用还会携带 profile 完整 `base_url` 的确定性 FNV-1a 128 指纹。引用中只保存指纹，不复制 URL，因此不会把 userinfo 或 query 参数写入引用。指纹覆盖完整 URL；配置 endpoint 变化后，旧引用会被拒绝。缺少指纹的旧序列化引用也会被拒绝。该指纹用于标识 endpoint，不是认证凭证。

通常通过 `ProviderFileRef::model_reference()` 获取 `ProviderFileSource`；如果需要手动构造，应使用 `provider_file_endpoint_fingerprint(profile.base_url)` 设置 `endpoint_fingerprint`。
