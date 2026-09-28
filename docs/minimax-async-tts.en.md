# MiniMax Asynchronous Long-Text TTS

`minimax_async_tts` implements MiniMax's native asynchronous long-text speech lifecycle: submit with `POST /v1/t2a_async_v2`, query by task ID with `GET /v1/query/t2a_async_query_v2`, then retrieve file metadata by file ID with `GET /v1/files/retrieve`. The caller triggers each step separately. The service does not poll, automatically retry a submission, or follow the temporary download URL.

```rust,no_run
# async fn example(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    providers::minimax::async_tts::{
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

Choose `MiniMaxAsyncTtsRegion::International` for `https://api.minimax.io/v1` or `ChinaMainland` for `https://api.minimax.cn/v1`. The service checks that the selected region matches the host. Task and generated-file references carry the provider, profile, region, endpoint fingerprint, and account scope; a reference from another connection is rejected before HTTP. Pass `&Secret<String>` to each operation so the host can refresh credentials without rebuilding the service or leaving a credential in it.

A `MiniMaxVoiceRef` obtained from `minimax_voices` can be passed to `submit_with_voice(&request, &voice, &api_key)`. Before submitting HTTP, the service checks that the reference matches its provider, profile, account scope, region, and `/v1` API endpoint, then uses the referenced voice ID. Built-in voice IDs remain available through `submit`. The account scope is a caller-supplied routing label and does not verify API-key ownership; pass the key for the intended MiniMax account.

Inline text is limited to 50,000 characters. For larger input, create a MiniMax text-file reference with purpose `t2a_async_input` through the file service and pass it to `MiniMaxAsyncTtsRequest::from_text_file`; this service checks its scope and sends the numeric `text_file_id`. MiniMax documents TXT and ZIP input up to 1,000,000 characters. ZIP archives may contain same-type TXT or JSON files; audio, sentence-level subtitles, and extra JSON outputs are provider-managed files. This module does not upload local files.

The request uses MiniMax speech model IDs and its native `voice_setting`, optional `audio_setting`, `pronunciation_dict`, `language_boost`, and `voice_modify` fields. The returned task ID is the handle for one `query` call. Known states are `Processing`, `Success`, `Failed`, and `Expired`; unknown future values are preserved. MiniMax documents a maximum of 10 status queries per second, so the caller controls polling cadence.

If submission transport fails, returns HTTP 408/5xx, or returns a success response that does not establish the task ID, the dispatch state is `Unknown`. Do not automatically submit again because MiniMax may have accepted the original request. A valid success response returns the task token as `Secret<String>` and excludes it from the retained native response. File metadata includes a download URL that remains valid for 9 hours from generation. The result type redacts that URL from debug output; the service does not download or persist it.

The contract follows MiniMax's official [international async create](https://platform.minimax.io/docs/api-reference/speech-t2a-async-create), [international task query](https://platform.minimax.io/docs/api-reference/speech-t2a-async-query), [international file retrieve](https://platform.minimax.io/docs/api-reference/file-management-retrieve), and matching [mainland create](https://platform.minimax.cn/docs/api-reference/speech-t2a-async-create), [mainland query](https://platform.minimax.cn/docs/api-reference/speech-t2a-async-query), and [mainland file retrieve](https://platform.minimax.cn/docs/api-reference/file-management-retrieve) references.
