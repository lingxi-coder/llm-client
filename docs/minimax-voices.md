# MiniMax 音色生命周期

`minimax_voices` 封装 MiniMax 原生音色接口：复刻 `POST /v1/voice_clone`、设计 `POST /v1/voice_design`、分类查询 `POST /v1/get_voice` 和删除 `POST /v1/delete_voice`。服务保留提供方原始 JSON、试听 URL/十六进制数据和请求 ID；不会下载试听内容、自动激活音色、轮询或重试写操作。

```rust,no_run
# async fn example(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    minimax_voices::{
        MiniMaxVoiceDesignRequest, MiniMaxVoiceListRequest, MiniMaxVoicesConfig,
        MiniMaxVoicesCredentials, MiniMaxVoicesRegion, MiniMaxVoicesService,
    },
    protocol::Secret,
    HttpTransport,
};

let transport = HttpTransport::new()?;
let service = MiniMaxVoicesService::new(
    &transport,
    MiniMaxVoicesConfig::new(
        "minimax-voice-profile",
        "team/account-17",
        MiniMaxVoicesRegion::ChinaMainland,
    ),
)?;
let credentials = MiniMaxVoicesCredentials::new(Secret::new(api_key));
let catalog = service
    .list_voices(&MiniMaxVoiceListRequest::default(), &credentials)
    .await?;

// Voice design uses the supplied text to generate a billed preview.
let paid_design = MiniMaxVoiceDesignRequest::new(
    "温和、清晰的中文讲述者",
    "这段文字会用于生成付费试听音频。",
)
.with_aigc_watermark(false);
let designed = service.design_voice(&paid_design, &credentials).await?;
let _ = (catalog, designed.reference, designed.trial_audio);
# Ok(())
# }
```

可以使用已上传的录音复刻音色而不请求试听，也可以显式删除选定的音色：

```rust,no_run
use lingxi_llm_client::{
    files::ProviderFileRef,
    minimax_voices::{
        MiniMaxVoiceCloneRequest, MiniMaxVoiceRef,
        MiniMaxVoicesCredentials, MiniMaxVoicesError, MiniMaxVoicesService,
    },
};

async fn clone_uploaded_audio(
    service: &MiniMaxVoicesService<'_>,
    credentials: &MiniMaxVoicesCredentials,
    audio: ProviderFileRef,
    new_voice_id: String,
) -> Result<MiniMaxVoiceRef, MiniMaxVoicesError> {
    let request = MiniMaxVoiceCloneRequest::new(audio, new_voice_id);
    Ok(service.clone_voice(&request, credentials).await?.reference)
}

async fn delete_selected_voice(
    service: &MiniMaxVoicesService<'_>,
    credentials: &MiniMaxVoicesCredentials,
    voice: &MiniMaxVoiceRef,
) -> Result<(), MiniMaxVoicesError> {
    service.delete_voice(voice, credentials).await?;
    Ok(())
}
```

`MiniMaxVoicesConfig` 要求稳定的 profile 名称、调用方提供的账号范围标签和明确的 `International` 或 `ChinaMainland` 区域。服务默认使用 `https://api.minimax.io/v1` 或 `https://api.minimax.cn/v1`。账号范围只是调用方路由标签，不证明 API key 的实际归属；每次操作都要传入对应账号的 `MiniMaxVoicesCredentials`。地区认证、套餐和权限由 MiniMax 服务端判断。

文件参数使用 `ProviderFileRef`，并在发请求前核对 provider、profile、账号范围、区域端点和上传用途。复刻音频用途为 `voice_clone`，提示音频用途为 `prompt_audio`。要使文件引用匹配，创建 `FileService` 时的 profile `base_url` 应与音色服务使用相同的 `/v1` 根，例如 `https://api.minimax.cn/v1`；服务不会替调用方改写或重新绑定文件引用。官方上传接口规定 MP3、M4A 或 WAV，文件不超过 20,000,000 字节；复刻音频为 10 秒至 5 分钟，提示音频短于 8 秒。服务不解码媒体文件；可选的时长参数是调用方声明值，仅用于本地范围校验。

克隆请求的 `voice_id` 须满足 MiniMax 对新克隆 ID 的长度和字符规则，且账号内唯一。可通过 `MiniMaxVoiceClonePrompt` 同时提交提示音频和逐字稿。克隆试听是可选项：只有显式同时设置文本与 `MiniMaxTtsModel` 时才会发送；试听文本最多 1,000 个字符，并可能产生费用。`accuracy` 只能与 `text_validation` 一起设置。成功响应若未返回 `voice_id`，服务使用请求中指定的 ID 生成 `MiniMaxVoiceRef`，并保留原始响应。新克隆音色不会由本模块激活；MiniMax 文档称，未在 7 天内使用的克隆音色可能被删除。

设计请求必须由调用方提供 `preview_text`，最多 500 个字符，试听合成会收费。返回的 `trial_audio` 保持为原始十六进制字符串，不会解码或下载。可选的 `voice_id` 不套用克隆接口的新 ID 规则。大陆官方接口还记录了 `aigc_watermark` 布尔字段，用于在试听末尾添加音频节奏标识，默认 `false`；只有显式调用 `.with_aigc_watermark(...)` 才会发送。国际版文档没有记录此字段，因此国际区域设置它会在发送前被拒绝。

列表支持系统、复刻、生成和全部四类。系统音色 ID 可以含空格、括号等字符，不能套用新克隆 ID 校验。结果带有按类别解析的音色列表和完整原始 JSON。`delete_voice` 只接受 `Cloned` 或 `Generated` 类型的同范围 `MiniMaxVoiceRef`；系统音色不可删除，删除的 ID 也不能复用。克隆和生成音色只有成功用于合成后才会出现在对应列表中。

克隆、设计和删除属于写操作。请求派发后遇到连接中断、超时或无法确认的成功响应时，错误会报告未知结果；服务不自动重试。调用方应检查 `MiniMaxVoicesError::dispatch()`，并避免盲目重复会产生副作用的操作。

官方契约：[国际版音色复刻](https://platform.minimax.io/docs/api-reference/voice-cloning-clone)、[国际版音色设计](https://platform.minimax.io/docs/api-reference/voice-design-design)、[国际版查询音色](https://platform.minimax.io/docs/api-reference/voice-management-get)、[国际版删除音色](https://platform.minimax.io/docs/api-reference/voice-management-delete)。大陆版音色设计字段见[大陆版音色设计](https://platform.minimax.cn/docs/api-reference/voice-design-design)；大陆版其余接口路径与字段见对应的 [复刻](https://platform.minimax.cn/docs/api-reference/voice-cloning-clone)、[查询](https://platform.minimax.cn/docs/api-reference/voice-management-get)和[删除](https://platform.minimax.cn/docs/api-reference/voice-management-delete)文档。
