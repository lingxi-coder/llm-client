# Qwen 文件语音识别

`QwenAsrService` 实现 Model Studio 的 `qwen3-asr-flash-filetrans` 异步文件转写。它只提交任务、单次查询任务状态，并按调用方要求获取转写 JSON；不会自动轮询、重试、上传本地音频或切换区域。

```rust,no_run
use lingxi_llm_client::{
    HttpTransport, RequestOptions,
    protocol::Secret,
    providers::qwen::asr::{
        QwenAsrRegion, QwenAsrRequest, QwenAsrScope, QwenAsrService,
    },
};

async fn transcribe() -> Result<(), Box<dyn std::error::Error>> {
    let transport = HttpTransport::new()?;
    let scope = QwenAsrScope::new(
        "qwen-main",
        "account-42",
        QwenAsrRegion::Singapore,
        "my-workspace-id",
    )?;
    let service = QwenAsrService::new(&transport, scope)?;
    let options = RequestOptions {
        credential: Some(Secret::new("Singapore Model Studio API key".into())),
        account_scope: Some("account-42".into()),
        ..Default::default()
    };

    let submitted = service
        .submit(
            &QwenAsrRequest::new("https://storage.example.test/meeting.wav"),
            &options,
        )
        .await?;
    println!("task {}: {:?}", submitted.reference().task_id(), submitted.status);

    // Call again when the host's own scheduling policy decides. This performs
    // one GET and never sleeps or starts an implicit polling loop.
    let current = service.get_task(submitted.reference(), &options).await?;
    if let Some(result_ref) = current.result() {
        let transcript = service
            .fetch_transcription(result_ref, &RequestOptions::default())
            .await?;
        for track in transcript.transcripts {
            for sentence in track.sentences {
                println!("{}..{} ms: {}", sentence.start_ms, sentence.end_ms, sentence.text);
            }
        }
    }
    Ok(())
}
```

`QwenAsrScope` fixes the profile name, stable account identity, region, and workspace. Beijing uses `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1`; Singapore uses `https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api/v1`. The workspace ID is validated as one DNS label. Supply a Model Studio API key for that exact account and region on each submit/query call through `RequestOptions::credential`. If `RequestOptions::account_scope` is present, it must match the scope. The client does not store credentials or infer the region from Chat configuration.

`QwenAsrRequest` accepts one public HTTP(S) audio URL or an `oss://` URL supported by the REST API. Filetrans asks Model Studio to retrieve that object; this module does not upload local bytes. The default parameter object is sent even when empty because the current Singapore workspace endpoint requires it. Optional parameters include a documented language hint, ITN, word-level timestamps, and zero-based audio track indices. Each selected track may be billed separately. The provider documents a maximum 2 GB and 12 hours for this model; availability and media acceptance remain provider-side checks.

Qwen Audio 3.x Filetrans 使用另一套契约：通过 `submit_audio_filetrans` 和 `QwenAudioAsrRequest` 提交。支持 `qwen-audio-3.1-asr-flash-filetrans` 与 `qwen-audio-3.0-asr-flash-filetrans`；请求把单个 URL 放入 `input.file_urls`，并始终发送 `parameters`。原有 Qwen3-ASR `submit` 仍使用单数 `input.file_url`。

```rust,no_run
use lingxi_llm_client::{
    providers::qwen::asr::{
        QwenAsrService, QwenAudioAsrModel, QwenAudioAsrParameters,
        QwenAudioAsrRequest,
    },
    RequestOptions,
};

async fn transcribe_audio3(
    service: &QwenAsrService<'_>,
    options: &RequestOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut request = QwenAudioAsrRequest::new(
        QwenAudioAsrModel::QwenAudio31AsrFlashFiletrans,
        "https://storage.example.test/meeting.wav",
    );
    request.parameters = QwenAudioAsrParameters {
        keep_dialect: Some(true),
        ..Default::default()
    };
    let submitted = service.submit_audio_filetrans(&request, options).await?;
    let current = service.get_task(submitted.reference(), options).await?;
    if let Some(result) = current.result() {
        let _transcription = service
            .fetch_transcription(result, &RequestOptions::default())
            .await?;
    }
    Ok(())
}
```

Qwen Audio 3.x 的参数与 Qwen3-ASR 参数分开。可设置仅 3.1 支持的方言保留、预编译或即时热词、音轨、敏感词过滤、说话人分离及可选说话人数提示、最多四个语言提示，以及最多五轮上下文。每轮上下文可以包含 user 和 assistant 消息，合计最多 400 个字符。即时热词权重为 1–5 或 50，超强热词最多 50 个。如果即时热词与预编译词表合并后超过 2,000 个，Model Studio 会随机选取 2,000 个；客户端原样提交并保留服务端行为。客户端无法预先知道文件实际包含哪些音轨。

Qwen Audio 的查询响应使用 `output.results` 数组，每项有独立的 `subtask_status`。`QwenAsrTask.status` 表示整个任务状态；还应检查 `subtask_status` 与 `task_metrics`，因为某个文件识别失败时，服务端仍可能把整体任务标记为 `SUCCEEDED`。只有整体任务和子任务都成功且存在结果 URL 时，`result()` 才会返回值；每个文件的失败原因放在 `subtask_code` 和 `subtask_message` 中。Qwen Audio 转写文件把元数据称为 `properties`，把源音频格式称为 `audio_format`；这两种格式以及 Qwen3-ASR 的 `audio_info` 都映射到已有的类型。获取方法要求文档中的元数据对象和 `transcripts` 数组，但允许数组为空。

`submit` returns the observed initial task status and a `QwenAsrTaskRef`. `get_task` performs exactly one query. The status type represents `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `UNKNOWN`, and unrecognized future values. `UNKNOWN` means the provider cannot confirm the task state; it is not treated as success or terminal failure. The returned task has an optional `QwenAsrResultRef` only when Model Studio supplies a transcription URL.

The result URL is a temporary signed URL documented as valid for 24 hours. `fetch_transcription` upgrades the historical HTTP OSS URL form to HTTPS, checks that its host matches the task's region, and does not send the account API key to the storage host. The URL is redacted in `Debug`. The transcription response exposes track text, sentence segments and optional word segments with millisecond offsets, language, emotion, speaker ID, and punctuation. A 64 MiB response bound protects the client from unbounded result downloads.

No automatic retry follows task submission. `QwenAsrError::submission_outcome()` classifies failures from `submit`: local failures are `NotSent`, explicit HTTP 4xx responses are `Rejected`, transport/5xx ambiguity is `Unknown`, and an invalid 2xx response is `Accepted`. `Unknown` and `Accepted` may represent a billable task; retain the task/request diagnostics and do not blindly submit again. This slice has contract tests with mock transport and no live-account validation.

Official references: [Qwen-ASR API reference](https://help.aliyun.com/en/model-studio/qwen-asr-api-reference), [Qwen3 and Qwen Audio 3.x file-transcription HTTP API](https://help.aliyun.com/en/model-studio/fun-asr-recorded-speech-recognition-http-api), [non-real-time speech recognition guide](https://help.aliyun.com/en/model-studio/non-realtime-speech-recognition-user-guide), and [audio model specifications](https://help.aliyun.com/en/model-studio/asr-model).
