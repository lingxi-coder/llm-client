# 图像生成与编辑

[English](images.en.md)

`client.images()` 是独立于 Chat 的图像服务。两项服务共享配置、HTTP 和按请求提供的凭据；图像模型及端点单独配置。

```rust,no_run
use lingxi_llm_client::{builtin_providers, LlmClientBuilder};
use lingxi_llm_client::protocol::{ImageGenerationRequest, ImageOutputOptions, ImageRequestOptions, Region, Secret};

async fn generate(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmClientBuilder::new(&builtin_providers()?)?
        .with_region(Region::International).build()?;
    let request = ImageGenerationRequest {
        model: "gpt-image-1.5".into(),
        prompt: "蓝色背景上的橘猫".into(),
        references: vec![],
        output: ImageOutputOptions::default(),
        provider_options: Default::default(),
    };
    let options = ImageRequestOptions { credential: Some(Secret::new(api_key)), ..Default::default() };
    let result = client.images().generate_in("openai", &request, &options).await?;
    println!("{} images", result.images.len());
    Ok(())
}
```

`generate` / `generate_in` 同步生成图片；`edit` / `edit_in` 编辑原图。`ImageGenerationRequest.references` 在模型支持时提供参考图，MiniMax 人物参考使用 `ImageReferenceKind::Character`。输入可为 HTTP(S) URL、带媒体类型的 Base64 或应用拥有的 `AttachmentRef`；附件需要在 builder 注册 `AttachmentResolver`。蒙版只在模型显式支持时可用；OpenAI 仅支持对第一张输入图片应用蒙版（`image_index = 0`），其他索引会在发送前报错。

`submit` / `submit_in` 提交 provider 原生异步任务，返回可序列化的 `ImageTaskRef`；`get_task` 每次只查询一次。提交与查询须提供相同的非秘密 `account_scope`，查询时仍需原连接的有效凭据。客户端不后台轮询或保存任务。

`images().models()` 列出当前区域可见的图像模型；`capabilities()` / `capabilities_in()` 返回生成、参考图、编辑和异步能力。显式请求不支持的参数会在发送前报错。`provider_options` 的键由适配器校验。

`ImageResponse` 保留原始 URL 或 Base64、实际执行 profile、请求 ID、可用用量及结果状态。临时 URL 由应用及时下载保存；图像用量不进入 Chat token 费用估算。图像请求不自动重试或切换连接。超时或网络中断后远端可能仍在生成，调用方决定是否再次提交。

| 服务 | 内置 profile | 能力 |
| --- | --- | --- |
| OpenAI | `openai` | 生成、编辑、蒙版 |
| Gemini | `gemini` | 生成、参考图、编辑 |
| Qwen | `qwen`、`qwen-intl`、`qwen-hk`、`qwen-us` | 生成、参考图、编辑、异步任务；各区域模型可用性依账号而定 |
| xAI | `grok` | 生成、编辑 |
| MiniMax | `minimax`、`minimax-intl` | 生成、人物参考 |
| 智谱 / Z.AI | `glm`、`zai` | 生成、异步任务 |
| OpenRouter | `openrouter` | 生成、参考图、编辑；实际能力取决于上游端点 |

其他内置 profile 未声明图像路由。自定义 `ProviderProfile.images` 可以明确指定图像端点、认证头、任务端点和模型。账号权限、区域及服务端模型发布状态仍需在实际环境核验。

接口依据：[OpenAI Images](https://developers.openai.com/api/docs/guides/image-generation)、[Gemini 图像生成](https://ai.google.dev/gemini-api/docs/generate-content/image-generation)、[Qwen Image](https://www.alibabacloud.com/help/en/model-studio/qwen-image-generation-and-editing-api-reference)、[xAI Image](https://docs.x.ai/developers/model-capabilities/images/generation)、[MiniMax Image](https://platform.minimax.io/docs/guides/image-generation)、[智谱图像生成](https://docs.bigmodel.cn/api-reference/模型-api/图像生成)、[OpenRouter Image](https://openrouter.ai/docs/guides/overview/multimodal/image-generation)。

Wan 2.7 使用 workspace 专属域名；配置 `api = "wan"` 的自定义图像路由、`base_url = "https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api/v1"`（或账号所在的北京域名）及对应模型，即可调用同步生成、编辑和原生任务。客户端不会推算 WorkspaceId。
