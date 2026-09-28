# Qwen file transcription

`QwenAsrService` implements Model Studio's asynchronous `qwen3-asr-flash-filetrans` file-transcription API. It submits a task, performs a single task lookup, and fetches the transcription JSON only when asked. It does not poll, retry, upload local audio, or switch regions automatically.

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

    // The host decides when to query again. This call performs one GET and
    // never sleeps or starts an implicit polling loop.
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

`QwenAsrScope` binds a profile name, stable account identity, region, and workspace. Beijing uses `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1`; Singapore uses `https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api/v1`. The workspace ID is validated as a single DNS label. Pass the Model Studio API key for that exact account and region on every submit/query call through `RequestOptions::credential`. If `RequestOptions::account_scope` is set, it must match the scope. Credentials are not stored, and the region is not inferred from Chat configuration.

`QwenAsrRequest` accepts one public HTTP(S) audio URL or an `oss://` URL supported by the REST API. Model Studio fetches the object; this module does not upload local bytes. The request always contains a `parameters` object because the current Singapore workspace endpoint requires it even when there are no options. Optional parameters include a documented language hint, ITN, word-level timestamps, and zero-based audio track indices. The provider may bill each selected track separately. Model Studio documents a 2 GB and 12-hour limit for this model; media availability and acceptance remain provider-side checks.

For the separate Qwen Audio 3.x Filetrans contract, call `submit_audio_filetrans` with a `QwenAudioAsrRequest`. It supports `qwen-audio-3.1-asr-flash-filetrans` and `qwen-audio-3.0-asr-flash-filetrans`; it sends one URL in `input.file_urls` and always includes `parameters`, while the original Qwen3-ASR `submit` continues to use its singular `input.file_url` schema.

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

Qwen Audio 3.x controls are separate from the Qwen3-ASR options. They include 3.1-only dialect preservation, precompiled or inline hotwords, track selection, sensitive-word filters, speaker diarization and an optional speaker-count hint, up to four language hints, and up to five context turns. A context turn may include a user message and an assistant message, with 400 combined characters at most. Inline hotword weights are 1–5 or 50, with at most 50 super-hotwords. If inline and precompiled lists combine to more than 2,000 entries, Model Studio randomly selects 2,000; the client forwards the list and leaves that provider behavior intact. It cannot determine the file's actual track count.

The Qwen Audio query response has an `output.results` array and separate `subtask_status` values. `QwenAsrTask.status` is the overall task status; inspect `subtask_status` and `task_metrics` as well, because the provider can report an overall `SUCCEEDED` task when its individual file failed. `result()` is available only when both the overall task and the subtask succeeded and a result URL is present; `subtask_code` and `subtask_message` carry a per-file failure reason. Its transcription artifact calls metadata `properties` and fields the source audio format `audio_format`; both that shape and Qwen3-ASR's `audio_info` shape decode into the existing typed view. The fetch method requires the documented metadata object and `transcripts` array, while allowing an empty array.

`submit` returns the observed initial task status and a `QwenAsrTaskRef`. `get_task` performs exactly one query. The status type represents `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `UNKNOWN`, and unrecognized future values. `UNKNOWN` means the provider cannot confirm the task state; it is not treated as success or terminal failure. A task includes a `QwenAsrResultRef` only when Model Studio supplies a transcription URL.

The result URL is a temporary signed URL documented as valid for 24 hours. `fetch_transcription` upgrades the historical HTTP OSS URL form to HTTPS, verifies that its host matches the task region, and never sends the account API key to the storage host. `Debug` output redacts the URL. The transcription response exposes track text, sentence and optional word segments with millisecond offsets, language, emotion, speaker ID, and punctuation. Downloads are bounded to 64 MiB.

Task submission is never retried automatically. `QwenAsrError::submission_outcome()` classifies `submit` failures: local failures are `NotSent`, explicit HTTP 4xx responses are `Rejected`, transport/5xx ambiguity is `Unknown`, and an invalid 2xx response is `Accepted`. `Unknown` and `Accepted` may mean that a billable task exists, so do not blindly submit again. This slice has mock-transport contract tests and no live-account validation.

Official references: [Qwen-ASR API reference](https://help.aliyun.com/en/model-studio/qwen-asr-api-reference), [Qwen3 and Qwen Audio 3.x file-transcription HTTP API](https://help.aliyun.com/en/model-studio/fun-asr-recorded-speech-recognition-http-api), [non-real-time speech recognition guide](https://help.aliyun.com/en/model-studio/non-realtime-speech-recognition-user-guide), and [audio model specifications](https://help.aliyun.com/en/model-studio/asr-model).
