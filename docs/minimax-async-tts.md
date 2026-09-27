# MiniMax 异步长文本 TTS

`minimax_async_tts` 实现 MiniMax 的原生异步长文本语音合成生命周期：提交 `POST /v1/t2a_async_v2`、按任务 ID 查询 `GET /v1/query/t2a_async_query_v2`，再按文件 ID 检索 `GET /v1/files/retrieve`。这三步由调用方分别触发；服务不会轮询、自动重试提交，也不会跟随临时下载 URL。

```rust,no_run
# async fn example(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    minimax_async_tts::{
        MiniMaxAsyncTtsConfig, MiniMaxAsyncTtsRegion, MiniMaxAsyncTtsRequest,
        MiniMaxAsyncTtsService, MiniMaxAsyncTtsStatus,
    },
    protocol::Secret,
    HttpTransport,
};

let transport = HttpTransport::new()?;
let service = MiniMaxAsyncTtsService::new(
    &transport,
    MiniMaxAsyncTtsConfig::new(
        "minimax-long-text",
        "team/account-17",
        MiniMaxAsyncTtsRegion::International,
    ),
)?;
let api_key = Secret::new(api_key);
let submitted = service
    .submit(
        &MiniMaxAsyncTtsRequest::new("A long passage to synthesize.", "English_expressive_narrator"),
        &api_key,
    )
    .await?;

// Query once when the caller chooses. Schedule later queries outside this service.
let task = service.query(&submitted.task, &api_key).await?;
if task.status == MiniMaxAsyncTtsStatus::Success {
    if let Some(file) = task.file.as_ref().or(submitted.file.as_ref()) {
        let result = service.get_result(file, &api_key).await?;
        let caller_managed_download_url = result.download_url.as_str();
        let _ = caller_managed_download_url;
    }
}
# Ok(())
# }
```

Choose `MiniMaxAsyncTtsRegion::International` for `https://api.minimax.io/v1` or `ChinaMainland` for `https://api.minimax.cn/v1`. The service enforces region/host agreement. A task and generated-file reference carry the provider, profile, region, endpoint fingerprint, and account scope; references from another connection are rejected before HTTP. Supply `&Secret<String>` to each operation so the host can refresh credentials without rebuilding or retaining a credential in the service.

通过 `minimax_voices` 获得的 `MiniMaxVoiceRef` 可传给 `submit_with_voice(&request, &voice, &api_key)`。服务会在 HTTP 提交前检查 reference 与当前 provider、profile、account scope、region 和 `/v1` API endpoint 一致，再使用 reference 中的 voice ID。内置声音 ID 仍可直接使用 `submit`。account scope 是调用方提供的路由标签，不验证 API key 的实际归属；调用方仍需传入该 MiniMax 账号对应的 key。

Inline text is limited to 50,000 characters. For larger inputs, create a MiniMax text-file reference with purpose `t2a_async_input` through the file service and pass it to `MiniMaxAsyncTtsRequest::from_text_file`; this service checks its scope and sends the numeric `text_file_id`. MiniMax documents TXT and ZIP inputs up to 1,000,000 characters. ZIP archives may contain same-type TXT or JSON files; their generated audio, sentence-level subtitles, and extra JSON outputs are provider-managed files. This module does not upload local files.

The request uses MiniMax's speech model IDs and native `voice_setting`, optional `audio_setting`, `pronunciation_dict`, `language_boost`, and `voice_modify` fields. The returned task ID is the handle for a single `query` call. Recognized task states are `Processing`, `Success`, `Failed`, and `Expired`; unknown future states are preserved. MiniMax documents a maximum of 10 status queries per second, so the caller controls pacing.

When a submission transport fails, returns HTTP 408/5xx, or returns a success response that cannot establish the task ID, the dispatch state is `Unknown`. Do not submit again automatically: MiniMax may have accepted the original request. A valid successful response returns the task token as `Secret<String>` and removes it from the retained native response. File metadata includes a download URL valid for 9 hours from generation. The result type redacts that URL in debug output and the service neither downloads nor persists it.

The contract follows MiniMax's official [international async create](https://platform.minimax.io/docs/api-reference/speech-t2a-async-create), [international task query](https://platform.minimax.io/docs/api-reference/speech-t2a-async-query), [international file retrieve](https://platform.minimax.io/docs/api-reference/file-management-retrieve), and corresponding [mainland create](https://platform.minimax.cn/docs/api-reference/speech-t2a-async-create), [mainland query](https://platform.minimax.cn/docs/api-reference/speech-t2a-async-query), and [mainland file retrieve](https://platform.minimax.cn/docs/api-reference/file-management-retrieve) references.
