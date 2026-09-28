# Gemini 向量嵌入

常规 `EmbeddingRequest` 支持 Gemini 文本向量的批量路由。若要把一条文本和媒体混合内容编码成一个向量，请使用 `GeminiMultimodalEmbeddingRequest` 和 `GoogleClient::embeddings().embed_multimodal`；Gemini Embedding 2 会将传入的所有内容片段合并为一个向量。

```rust,no_run
use lingxi_llm_client::providers::google::embeddings::{GeminiEmbeddingMedia, GeminiEmbeddingPart, GeminiEmbeddingSource, GeminiMultimodalEmbeddingRequest};
use lingxi_llm_client::{LlmClient, RequestOptions};

async fn embed_image(
    client: &LlmClient,
    options: &RequestOptions,
    image_bytes: Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::google::GoogleClient>("gemini")?;
let input = GeminiMultimodalEmbeddingRequest {
    model: "gemini-embedding-2".into(),
    parts: vec![
        GeminiEmbeddingPart::Text("A red bicycle beside a tree".into()),
        GeminiEmbeddingPart::Media(GeminiEmbeddingMedia {
            mime_type: "image/jpeg".into(),
            source: GeminiEmbeddingSource::Inline(image_bytes),
            duration_seconds: None,
            page_count: None,
        }),
    ],
    dimensions: Some(768),
};

let result = provider
    .embeddings()
    .embed_multimodal(&input, &options)
    .await?;
assert_eq!(result.vectors.len(), 1);
Ok(())
}
```

`GeminiEmbeddingSource::Inline` 会将字节以 Base64 编码放入官方定义的 `inline_data` 字段。`FileUri` 直接传入调用方已上传到 Gemini Files API 的 URI；本客户端不会上传、下载、轮询或刷新媒体。请求使用 `batchEmbedContents`，其中只包含一个 `EmbedContentRequest`，因此 `Content.parts` 中的内容会合并为一个向量。

Google 文档说明 `gemini-embedding-2` 支持文本、PNG/JPEG 图片、MP3/WAV 音频、MP4/MOV 视频和 PDF。官方限制为：输入最多 8,192 个 token、每次请求最多 6 张图片、音频最长 180 秒、视频最长 120 秒、PDF 最多 1 个且不超过 6 页。输出维度范围为 128–3,072；Google 推荐 768、1,536 或 3,072。客户端会在发送前检查媒体类型、图片和 PDF 数量、调用方声明的音视频时长、PDF 页数和输出维度。时长和页数是调用方提供的本地校验信息，客户端不会解析媒体文件；媒体本身是否有效以及 8,192-token 限制仍由 Google 端判定。请求将 `autoTruncate` 设为 `false`，让服务端拒绝超长输入，避免静默截断。Google 最多从每个视频采样 32 帧，并且不会处理视频中的音轨。

`gemini-embedding-2` 不接受 `taskType`。纯文本任务应按 Google 的建议在文本片段中加入任务指令；多模态内容通常不建议添加任务前缀。`gemini-embedding-001` 仍只支持文本，输入限制为 2,048 个 token，并支持 `taskType`，不能用于多模态方法。Gemini Embedding 1 和 2 的向量空间互不兼容，不能直接比较。客户端原样返回向量，不做归一化。

## 发现嵌入模型

Gemini profile 显式配置自己的 `models_endpoint`，模型发现不会从 Chat 配置推导 URL。`GoogleClient::embeddings().list_models` 调用 Google `models.list`，只返回 `supportedGenerationMethods` 明确包含 `embedContent` 的模型。每个 `GeminiEmbeddingModel` 保留官方 `resource_name`、可直接用于 `EmbeddingRequest.model` 的裸资源名后缀（字段 `id`）、单独的服务端 `base_model_id`、版本、服务端报告的 token 上限和方法列表，以及完整原生对象。目录未提供的向量维度和输入模态不会根据模型名称推断。

`GeminiEmbeddingModelListQuery.page_size` 可不设置：`None` 会从 URL 中省略 `pageSize`，使用 Google 默认值 50。正数会按原值发送；即使请求更大的页，Google 每页最多仍返回 1000 条。页面的 `next_page_token` 是不透明、可序列化的 cursor，不是 offset。它绑定原始 `pageSize` 参数形状、路由、provider、profile、region 和可选的 `RequestOptions.account_scope`；换用其他范围，或显式修改 page size，都会在认证和 HTTP 前被拒绝。若调用方没有声明 account scope，cursor 也不绑定已声明的账户身份；客户端不会声称该标签能验证 API key 实际归属。`nextPageToken` 缺失或为空字符串表示分页结束。过滤不会自动读取更多页面；当前页没有嵌入模型时仍可能有下一页。

`GoogleClient::embeddings().get_model` 接收官方资源名，例如 `models/gemini-embedding-2`，会核对返回资源名，并拒绝 `supportedGenerationMethods` 未明确包含 `embedContent` 的资源。列表和详情查询沿用其他嵌入操作的 profile 启用状态、region、每次调用凭证和总超时规则。

```rust,no_run
use lingxi_llm_client::providers::google::embeddings::{GeminiEmbeddingModelListQuery};
use lingxi_llm_client::{
    embeddings::{EmbeddingError},
    LlmClient, RequestOptions,
};

async fn discover_gemini_models(
    client: &LlmClient,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let provider = client.provider::<lingxi_llm_client::providers::google::GoogleClient>("gemini")?;
    let first = provider
        .embeddings()
        .list_models(&GeminiEmbeddingModelListQuery::default(), options)
        .await?;

    if let Some(resource) = first.models.first() {
        let detail = provider
            .embeddings()
            .get_model(&resource.resource_name, options)
            .await?;
        assert_eq!(detail.id, resource.id);
    }

    if let Some(token) = first.next_page_token {
        let next = provider
            .embeddings()
            .list_models(
                &GeminiEmbeddingModelListQuery {
                    page_token: Some(token),
                    ..Default::default()
                },
                options,
            )
            .await?;
        let _ = next.models;
    }
    Ok(())
}
```

参考：[Google Gemini Embeddings 指南](https://ai.google.dev/gemini-api/docs/embeddings)、[Gemini Embeddings REST API](https://ai.google.dev/api/embeddings)、[Gemini Models API](https://ai.google.dev/api/models)、[Gemini Part/FileData 请求结构](https://ai.google.dev/api/generate-content)。
