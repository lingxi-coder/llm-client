# 提供方真实验收

本地契约测试、官方文档证据和真实账户验收分别记录。通过某个只读接口，不代表同一账户能够生成内容、使用音频/Realtime、访问自定义音色或在其他区域调用服务。

## xAI 内置音色目录

`examples/xai_voice_acceptance.rs` 使用库的 `XaiAudioService` 和默认官方端点，检查 `list_voices()`，然后读取返回的第一个音色的 `get_voice()`。最多两个 GET，不创建资源、不生成音频、不自动重试、不访问自定义音色。详情请求核对返回 ID 与请求 ID 完全一致；目录为空时详情保持 `not_run`，整体验收不算通过。

默认只输出计划，不读取密钥或发出请求：

```sh
cargo run --locked --offline --example xai_voice_acceptance
```

在允许使用当前账户的凭证进行真实请求后，使用已配置的 `XAI_API_KEY`：

```sh
cargo run --locked --offline --example xai_voice_acceptance -- --run-read-only
```

不要把密钥作为命令行参数。输出只含检查状态、错误类别、HTTP 状态（若有）、音色数量和时间；不输出密钥、提供方错误体、音色正文或账户标识。`environment-xai-key` 仅是本地 scope 标签，不是已核实的账户身份。每个请求超时 20 秒，错误或未完成返回非零退出码。运行入口不会在普通 `cargo test` 中发出真实请求。

2026-09-27 本次状态：默认预览可运行。一次受限环境运行返回 transport 错误，未获得提供方响应；随后申请外部网络访问被自动审批拒绝，理由是没有明确授权向 xAI 发送环境凭证。没有绕过拒绝，也没有真实接口通过结果。完成本地实现后，真实执行等待用户明确授权这两个只读 GET。能力矩阵继续保留 `live_validation: not_run`，不能据本地 mock 改为已通过。

接口依据：[xAI Voice REST reference](https://docs.x.ai/developers/rest-api-reference/inference/voice)。

## 其余待验收范围

其他提供方需要对应账户、区域、服务权限及实际测试输入；生成、上传、异步作业和 Realtime 调用的测试范围应明确到具体操作。硬件录音/播放与端到端交互还需要宿主应用，不属于此库的设备层实现。

文档尚未建立的契约不靠账户试探来推断。OpenAI Speech SSE 的文本/音频对齐字段仍未由当前 [Speech API reference](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create) 和 [TTS guide](https://developers.openai.com/api/docs/guides/text-to-speech) 建立；不发送推测字段。Z.AI 国际 TTS 与 Anthropic 内联按值定义的搜索可发现性，以各服务指南的精确来源和限制为准。
