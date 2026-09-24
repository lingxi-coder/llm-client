# Image generation and editing

[简体中文](images.md)

`client.images()` is independent of Chat. Both services share configuration, HTTP transport, and request-supplied credentials; image models and endpoints have separate configuration.

```rust,no_run
use lingxi_llm_client::{builtin_providers, LlmClientBuilder};
use lingxi_llm_client::protocol::{ImageGenerationRequest, ImageOutputOptions, ImageRequestOptions, Region, Secret};

async fn generate(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmClientBuilder::new(&builtin_providers()?)?
        .with_region(Region::International).build()?;
    let request = ImageGenerationRequest {
        model: "gpt-image-1.5".into(),
        prompt: "An orange cat on a blue background".into(),
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

`generate` / `generate_in` return completed images. `edit` / `edit_in` edit source images. `ImageGenerationRequest.references` supplies references where supported; MiniMax character references use `ImageReferenceKind::Character`. Inputs can be HTTP(S) URLs, Base64 with a media type, or application-owned `AttachmentRef` values. Attachments require an `AttachmentResolver` on the builder. Masks require explicit model support. OpenAI masks apply only to the first input image (`image_index = 0`); other indices are rejected before sending.

`submit` / `submit_in` call a provider's native asynchronous endpoint and return a serializable `ImageTaskRef`. Each `get_task` call makes one query. Submission and lookup require the same nonsecret `account_scope`; lookup also requires a valid credential for the original connection. The client does not poll or persist tasks in the background.

`images().models()` lists visible image models in the selected region. `capabilities()` / `capabilities_in()` return generation, reference, editing, and task support. Unsupported explicit parameters fail before sending. Adapter-specific `provider_options` keys are validated.

`ImageResponse` preserves original URLs or Base64, the executed profile, request ID, available usage, and outcome. The application must download and save temporary URLs promptly. Image usage is separate from Chat token cost estimates. Image calls do not automatically retry or switch connections. After a timeout or network interruption, the provider may still be generating; the caller decides whether to submit again.

| Provider | Built-in profiles | Capabilities |
| --- | --- | --- |
| OpenAI | `openai` | Generation, editing, masks |
| Gemini | `gemini` | Generation, references, editing |
| Qwen | `qwen`, `qwen-intl`, `qwen-hk`, `qwen-us` | Generation, references, editing, native tasks; model availability depends on the regional account |
| xAI | `grok` | Generation, editing |
| MiniMax | `minimax`, `minimax-intl` | Generation, character references |
| Zhipu / Z.AI | `glm`, `zai` | Generation, native tasks |
| OpenRouter | `openrouter` | Generation, references, editing; actual support depends on the upstream endpoint |

Other built-in profiles have no image route. Custom `ProviderProfile.images` entries can declare image endpoints, authentication headers, task endpoints, and models. Account access, region, and server-side model availability still need verification in the target environment.

API references: [OpenAI Images](https://developers.openai.com/api/docs/guides/image-generation), [Gemini image generation](https://ai.google.dev/gemini-api/docs/generate-content/image-generation), [Qwen Image](https://www.alibabacloud.com/help/en/model-studio/qwen-image-generation-and-editing-api-reference), [xAI Image](https://docs.x.ai/developers/model-capabilities/images/generation), [MiniMax Image](https://platform.minimax.io/docs/guides/image-generation), [Zhipu image generation](https://docs.bigmodel.cn/api-reference/模型-api/图像生成), and [OpenRouter Image](https://openrouter.ai/docs/guides/overview/multimodal/image-generation).

Wan 2.7 requires a workspace-specific domain. Configure a custom image route with `api = "wan"`, a `base_url` such as `https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api/v1` (or the matching Beijing domain), and its models for synchronous generation, editing, and native tasks. The client never guesses the WorkspaceId.
