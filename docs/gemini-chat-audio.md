# Gemini Chat 音频输入

`ContentBlock::Audio { format, data }` 可用于 Gemini GenerateContent 与 Vertex Gemini 的用户消息，编码为 `inlineData`。`data` 必须是非空标准 Base64，不带 data URI；`format` 支持 wav、mp3、aiff、aac、ogg、flac、mpeg、m4a、l16、opus、alaw、mulaw、webm。适配器检查声明格式和编码，不解码音频内容或推断模型/账户权限。

```rust,no_run
use lingxi_llm_client::protocol::ContentBlock;
let audio = ContentBlock::Audio {
    format: "wav".into(),
    data: "UklGRg==".into(), // 示例占位；替换成完整音频的 Base64。
};
```

该入口只提交音频输入，输出仍按 Chat 请求处理。OpenRouter 专用输出配置不能用于 Gemini。带此内容块的完整请求采用 20 MB 上限；较大的输入可通过已有 Files/Document provider-file 路径传递。实际音频时长、模型支持范围和内容有效性由提供方校验。

来源：[Google 音频理解](https://ai.google.dev/gemini-api/docs/generate-content/audio)、[Vertex 音频理解](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/audio-understanding)。已通过本地编解码与无副作用预检回归，未进行在线验收。

类型化内联音频与自动附件混用时，客户端在任何附件上传前检查完整内联表示。若该表示超过 20 MB，请先显式上传较大文件并传入作用域文件引用。文件准备完成后的最终请求仍会再次校验。
