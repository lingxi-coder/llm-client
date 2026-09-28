# xAI 自定义音色

`XaiAudioService` 提供 xAI 自定义音色的创建、分页列表、读取、元数据更新、删除和参考音频流式读取。此资源属于团队级别；每个 [`XaiCustomVoiceRef`] 会绑定 profile、API endpoint 和 `account_scope`，因此不能拿另一个连接下的引用执行操作。每次调用都单独传入 `XaiAudioCredentials`。

```rust,no_run
use lingxi_llm_client::{
    providers::openai::audio::AudioInput,
    providers::xai::audio::{
        XaiAudioCredentials, XaiAudioService, XaiCustomVoiceAge,
        XaiCustomVoiceCreateRequest, XaiCustomVoiceGender,
        XaiCustomVoiceListRequest, XaiCustomVoicePatch, XaiCustomVoiceTone,
        XaiCustomVoiceUseCase,
    },
};

async fn manage_voice(
    service: &XaiAudioService<'_>,
    credentials: &XaiAudioCredentials,
) -> Result<(), Box<dyn std::error::Error>> {
    let audio = AudioInput::from_bytes(
        "reference.wav",
        "audio/wav",
        b"caller-owned audio bytes".to_vec(),
    );
    let request = XaiCustomVoiceCreateRequest::new(audio)
        .with_duration_seconds(90.0)
        .with_name("Friendly Narrator")
        .with_gender(XaiCustomVoiceGender::Female)
        .with_age(XaiCustomVoiceAge::Young)
        .with_language("en-US")
        .with_use_case(XaiCustomVoiceUseCase::Narration)
        .with_tone(XaiCustomVoiceTone::Warm);
    let created = service.create_custom_voice(request, credentials).await?;
    let reference = created.reference.clone();

    let first_page = service
        .list_custom_voices(&XaiCustomVoiceListRequest::new().with_limit(50), credentials)
        .await?;
    if let Some(cursor) = first_page.next_page {
        let _next_page = service
            .list_custom_voices(&XaiCustomVoiceListRequest::new().after(cursor), credentials)
            .await?;
    }

    let _current = service.get_custom_voice(&reference, credentials).await?;
    let patch = XaiCustomVoicePatch::new()
        .clear_description()
        .with_tone(XaiCustomVoiceTone::Calm);
    let _updated = service
        .update_custom_voice(&reference, &patch, credentials)
        .await?;

    // max_bytes is a caller-side bound. The stream is not written to disk.
    let mut audio = service
        .get_custom_voice_audio(&reference, 100_000_000, credentials)
        .await?;
    while let Some(chunk) = audio.next_chunk().await? {
        let _ = chunk;
    }

    let _receipt = service.delete_custom_voice(&reference, credentials).await?;
    Ok(())
}
```

创建通过 multipart 上传 `file`，可选字段是 `name`、`description`、`gender`、`accent`、`age`、`language`、`use_case` 和 `tone`。指南接受 WAV、MP3、FLAC、OGG、Opus、M4A、AAC、MKV 和 MP4，推荐 WAV；请求携带调用方提供的 MIME 类型。官方限制参考片段最长 120 秒，但未公布字节上限。`with_duration_seconds` 只验证调用方声明的时长，不会解码或测量媒体；未提供时由 xAI 校验。实现不会凭扩展名猜 MIME，也不会把 STT 上传上限套用到音色创建。

`XaiCustomVoicePatch::with_*` 写入非空值；`clear_*` 写入 JSON `null`，清除对应元数据。未设置的字段不发送，空字符串会在本地拒绝。响应保留类型化字段和完整 `native` JSON，以便使用未来新增的供应商字段。

`get_custom_voice_audio` 单独请求 `/audio` 并返回有界流，调用方必须给出正的 `max_bytes`；超过后续块上限时流会报错，已交付字节数仍可读取。它保留服务端实际 `Content-Type`，不缓存、不落盘，也不自动下载或重试。xAI 没有公布该下载路由的字节上限。

这些 API 的创建能力需要 Enterprise 计划；官方目前注明 Custom Voices 在美国开放，伊利诺伊州除外，并限制每个团队最多 30 个音色。客户端不根据本机区域或套餐推断权限，具体授权由服务端判定。指南没有公开同意/许可字段或 consent endpoint；调用方应在提交参考录音前取得必要权利。创建、更新、删除在 dispatch 后若连接失败，或成功 HTTP 响应中的回执无法识别，都会返回 `XaiAudioError::OutcomeUnknown`，不会自动重试；成功响应无法识别时可从 `response` 查看 request ID 与原始响应。

依据：[xAI Custom Voices 文档](https://docs.x.ai/developers/model-capabilities/audio/custom-voices)。

下载发生错误或到达 EOF 时，会立即释放底层 HTTP 响应流，即使调用方仍保留包装对象。
