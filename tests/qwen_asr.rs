use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, StreamExt};
use lingxi_llm_client::{
    protocol::{LlmError, Secret},
    providers::qwen::asr::{
        QwenAsrError, QwenAsrLanguage, QwenAsrParameters, QwenAsrRegion, QwenAsrRequest,
        QwenAsrScope, QwenAsrService, QwenAsrSubmissionOutcome, QwenAsrTaskRef, QwenAsrTaskStatus,
        QwenAudioAsrContextTurn, QwenAudioAsrLanguage, QwenAudioAsrModel, QwenAudioAsrParameters,
        QwenAudioAsrRequest, QwenAudioAsrSpecialWordFilter, QWEN3_ASR_FILETRANS_MODEL,
    },
    HttpRequest, RequestOptions, StreamResponse, Transport,
};
use serde_json::{json, Value};
use std::{collections::VecDeque, sync::Mutex};

struct Reply {
    status: u16,
    body: Vec<u8>,
}

#[derive(Debug)]
struct Sent {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

struct MockTransport {
    replies: Mutex<VecDeque<Result<Reply, LlmError>>>,
    sent: Mutex<Vec<Sent>>,
}

impl MockTransport {
    fn new(replies: impl IntoIterator<Item = Result<Reply, LlmError>>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            sent: Mutex::new(Vec::new()),
        }
    }

    fn sent(&self) -> Vec<Sent> {
        self.sent.lock().unwrap().clone()
    }
}

impl Clone for Sent {
    fn clone(&self) -> Self {
        Self {
            method: self.method.clone(),
            url: self.url.clone(),
            headers: self.headers.clone(),
            body: self.body.clone(),
        }
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send(&self, request: HttpRequest) -> Result<StreamResponse, LlmError> {
        self.sent.lock().unwrap().push(Sent {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body.to_vec(),
        });
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected Qwen ASR request")?;
        Ok(StreamResponse {
            status: reply.status,
            headers: Vec::new(),
            body: stream::once(async move { Ok(Bytes::from(reply.body)) }).boxed(),
        })
    }
}

fn scope(region: QwenAsrRegion) -> QwenAsrScope {
    QwenAsrScope::new("qwen-main", "account-42", region, "workspace-abc").unwrap()
}

fn options() -> RequestOptions {
    RequestOptions {
        credential: Some(Secret::new("region-specific-key".to_owned())),
        account_scope: Some("account-42".to_owned()),
        ..Default::default()
    }
}

fn reply(status: u16, value: Value) -> Result<Reply, LlmError> {
    Ok(Reply {
        status,
        body: serde_json::to_vec(&value).unwrap(),
    })
}

#[tokio::test]
async fn submit_uses_workspace_region_dashscope_task_contract() {
    let transport = MockTransport::new([reply(
        200,
        json!({
            "request_id": "submit-1",
            "output": { "task_id": "task_123", "task_status": "PENDING" }
        }),
    )]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Singapore)).unwrap();
    let request = QwenAsrRequest {
        file_url: "https://audio.example.test/mix%20one.wav?sig=private".into(),
        parameters: QwenAsrParameters {
            language: Some(QwenAsrLanguage::Chinese),
            enable_itn: Some(false),
            enable_words: Some(true),
            channel_ids: Some(vec![0, 1]),
        },
    };

    let task = service.submit(&request, &options()).await.unwrap();
    assert_eq!(task.status, QwenAsrTaskStatus::Pending);
    assert_eq!(task.request_id.as_deref(), Some("submit-1"));
    assert_eq!(task.reference().scope().account_scope(), "account-42");
    let sent = transport.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].method, "POST");
    assert_eq!(
        sent[0].url,
        "https://workspace-abc.ap-southeast-1.maas.aliyuncs.com/api/v1/services/audio/asr/transcription"
    );
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name == "x-dashscope-async" && value == "enable"));
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["model"], QWEN3_ASR_FILETRANS_MODEL);
    assert_eq!(body["input"]["file_url"], request.file_url);
    assert_eq!(body["parameters"]["language"], "zh");
    assert_eq!(body["parameters"]["enable_itn"], false);
    assert_eq!(body["parameters"]["enable_words"], true);
    assert_eq!(body["parameters"]["channel_id"], json!([0, 1]));
}

#[tokio::test]
async fn query_and_result_fetch_are_separate_and_segments_are_typed() {
    let url = "http://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/pre/result.json?Expires=1&Signature=secret";
    let transport = MockTransport::new([
        reply(
            200,
            json!({
                "request_id": "query-1",
                "output": {
                    "task_id": "task-123",
                    "task_status": "SUCCEEDED",
                    "result": { "transcription_url": url },
                    "submit_time": "2026-09-25 08:00:00.000",
                    "end_time": "2026-09-25 08:00:04.000"
                },
                "usage": { "seconds": 4 }
            }),
        ),
        reply(
            200,
            json!({
                "audio_info": { "format": "wav", "sample_rate": 16000 },
                "transcripts": [{
                    "channel_id": 0,
                    "text": "你好世界。",
                    "sentences": [{
                        "sentence_id": 0,
                        "begin_time": 100,
                        "end_time": 900,
                        "language": "zh",
                        "text": "你好世界。",
                        "words": [{
                            "begin_time": 100,
                            "end_time": 400,
                            "text": "你好",
                            "punctuation": ""
                        }]
                    }]
                }]
            }),
        ),
    ]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Beijing)).unwrap();
    let task_ref = lingxi_llm_client::providers::qwen::asr::QwenAsrTaskRef::new(
        scope(QwenAsrRegion::Beijing),
        "task-123",
    )
    .unwrap();
    let task = service.get_task(&task_ref, &options()).await.unwrap();
    assert_eq!(task.status, QwenAsrTaskStatus::Succeeded);
    assert_eq!(task.duration_seconds, Some(4));
    let result = task.result().unwrap();
    assert!(!format!("{result:?}").contains("Signature=secret"));
    let transcript = service
        .fetch_transcription(result, &RequestOptions::default())
        .await
        .unwrap();
    assert_eq!(
        transcript.audio_info.as_ref().unwrap().sample_rate,
        Some(16000)
    );
    let segment = &transcript.transcripts[0].sentences[0];
    assert_eq!((segment.start_ms, segment.end_ms), (100, 900));
    assert_eq!(segment.language.as_deref(), Some("zh"));
    assert_eq!(segment.words[0].text, "你好");

    let sent = transport.sent();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0].method, "GET");
    assert_eq!(
        sent[0].url,
        "https://workspace-abc.cn-beijing.maas.aliyuncs.com/api/v1/tasks/task-123"
    );
    assert!(sent[1]
        .url
        .starts_with("https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/"));
    assert!(!sent[1]
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("authorization")));
}

#[tokio::test]
async fn submission_transport_failure_is_unknown_and_never_retried() {
    let transport = MockTransport::new([Err(LlmError::Transport {
        message: "connection lost after send".into(),
    })]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Beijing)).unwrap();
    let error = service
        .submit(
            &QwenAsrRequest::new("https://audio.example.test/file.wav"),
            &options(),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.submission_outcome(),
        Some(QwenAsrSubmissionOutcome::Unknown)
    );
    assert_eq!(transport.sent().len(), 1);
}

#[tokio::test]
async fn explicit_rejection_is_safe_to_classify_but_not_automatically_retried() {
    let transport = MockTransport::new([reply(
        400,
        json!({ "code": "InvalidParameter", "message": "bad request", "request_id": "r1" }),
    )]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Beijing)).unwrap();
    let error = service
        .submit(
            &QwenAsrRequest::new("https://audio.example.test/file.wav"),
            &options(),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.submission_outcome(),
        Some(QwenAsrSubmissionOutcome::Rejected)
    );
    assert!(matches!(
        error,
        QwenAsrError::SubmissionRejected { status: 400, .. }
    ));
}

#[tokio::test]
async fn scope_and_url_validation_prevent_cross_account_routing() {
    assert!(QwenAsrScope::new("p", "a", QwenAsrRegion::Beijing, "workspace.evil").is_err());
    let transport = MockTransport::new([]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Beijing)).unwrap();
    let foreign = QwenAsrTaskRef::new(
        QwenAsrScope::new(
            "other",
            "other-account",
            QwenAsrRegion::Beijing,
            "workspace-abc",
        )
        .unwrap(),
        "task-123",
    )
    .unwrap();
    assert!(matches!(
        service.get_task(&foreign, &options()).await,
        Err(QwenAsrError::ScopeMismatch)
    ));
    assert!(matches!(
        service
            .submit(&QwenAsrRequest::new("file:///etc/passwd"), &options())
            .await,
        Err(QwenAsrError::InvalidInput(_))
    ));
    assert!(transport.sent().is_empty());
}

#[tokio::test]
async fn missing_submit_key_is_not_sent_but_query_errors_are_not_submission_outcomes() {
    let transport = MockTransport::new([]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Beijing)).unwrap();
    let task_ref = QwenAsrTaskRef::new(scope(QwenAsrRegion::Beijing), "task-123").unwrap();

    let submit_error = service
        .submit(
            &QwenAsrRequest::new("https://audio.example.test/file.wav"),
            &RequestOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(
        submit_error.submission_outcome(),
        Some(QwenAsrSubmissionOutcome::NotSent)
    );

    let query_error = service
        .get_task(&task_ref, &RequestOptions::default())
        .await
        .unwrap_err();
    assert_eq!(query_error.submission_outcome(), None);
    assert!(transport.sent().is_empty());
}

#[tokio::test]
async fn qwen_audio_filetrans_uses_plural_url_and_typed_controls() {
    let transport = MockTransport::new([reply(
        200,
        json!({
            "request_id": "audio-submit-1",
            "output": { "task_id": "audio-task-1", "task_status": "PENDING" }
        }),
    )]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Beijing)).unwrap();
    let mut vocabulary = std::collections::BTreeMap::new();
    vocabulary.insert("Lin Qiao".to_owned(), 50);
    vocabulary.insert("LingXi".to_owned(), 4);
    let request = QwenAudioAsrRequest {
        model: QwenAudioAsrModel::QwenAudio31AsrFlashFiletrans,
        file_url: "https://audio.example.test/meeting.wav?sig=private".into(),
        parameters: QwenAudioAsrParameters {
            keep_dialect: Some(true),
            vocabulary_id: Some("meeting-terms".into()),
            vocabulary: Some(vocabulary),
            channel_ids: Some(vec![0, 1]),
            special_word_filter: Some(QwenAudioAsrSpecialWordFilter {
                filter_with_signed: vec!["Alice".into()],
                filter_with_empty: vec!["secret".into()],
                system_reserved_filter: Some(false),
            }),
            diarization_enabled: Some(true),
            speaker_count: Some(3),
            language_hints: Some(vec![
                QwenAudioAsrLanguage::Chinese,
                QwenAudioAsrLanguage::English,
            ]),
        },
        context: vec![],
    }
    .with_context_turn(
        QwenAudioAsrContextTurn::new("Project names: LingXi and Qwen.")
            .with_assistant_text("Understood."),
    );

    let task = service
        .submit_audio_filetrans(&request, &options())
        .await
        .unwrap();
    assert_eq!(task.status, QwenAsrTaskStatus::Pending);
    assert_eq!(task.subtask_status, None);
    assert_eq!(task.task_metrics, None);
    let sent = transport.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].method, "POST");
    assert_eq!(
        sent[0].url,
        "https://workspace-abc.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/asr/transcription"
    );
    assert!(sent[0]
        .headers
        .iter()
        .any(|(name, value)| name == "x-dashscope-async" && value == "enable"));
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["model"], "qwen-audio-3.1-asr-flash-filetrans");
    assert_eq!(body["input"]["file_urls"], json!([request.file_url]));
    assert_eq!(body["input"]["context"][0]["role"], "user");
    assert_eq!(
        body["input"]["context"][0]["content"][0]["type"],
        "input_text"
    );
    assert_eq!(body["input"]["context"][1]["role"], "assistant");
    assert_eq!(body["parameters"]["keep_dialect"], true);
    assert_eq!(body["parameters"]["vocabulary_id"], "meeting-terms");
    assert_eq!(body["parameters"]["vocabulary"]["Lin Qiao"], 50);
    assert_eq!(body["parameters"]["channel_id"], json!([0, 1]));
    assert_eq!(
        body["parameters"]["special_word_filter"]["filter_with_signed"]["word_list"],
        json!(["Alice"])
    );
    assert_eq!(
        body["parameters"]["special_word_filter"]["filter_with_empty"]["word_list"],
        json!(["secret"])
    );
    assert_eq!(body["parameters"]["diarization_enabled"], true);
    assert_eq!(body["parameters"]["speaker_count"], 3);
    assert_eq!(body["parameters"]["language_hints"], json!(["zh", "en"]));
    assert!(!format!("{:?}", request).contains("Project names"));
    assert!(!format!("{:?}", request).contains("meeting.wav?sig=private"));
}

#[tokio::test]
async fn qwen_audio_filetrans_query_decodes_subtask_failure_and_properties() {
    let transcript_url = "http://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/result.json?Signature=artifact-secret";
    let transport = MockTransport::new([
        reply(
            200,
            json!({
                "request_id": "audio-query-1",
                "output": {
                    "task_id": "audio-task-1",
                    "task_status": "SUCCEEDED",
                    "results": [{
                        "file_url": "https://audio.example/meeting.wav",
                        "subtask_status": "SUCCEEDED",
                        "transcription_url": transcript_url
                    }],
                    "task_metrics": { "TOTAL": 1, "SUCCEEDED": 1, "FAILED": 0 }
                },
                "usage": { "duration": 9 }
            }),
        ),
        reply(
            200,
            json!({
                "properties": {
                    "audio_format": "wav",
                    "channels": [0, 1],
                    "original_sampling_rate": 48000,
                    "original_duration_in_milliseconds": 7500
                },
                "transcripts": [{
                    "channel_id": 0,
                    "content_duration_in_milliseconds": 3200,
                    "text": "hello",
                    "sentences": [{
                        "begin_time": 100,
                        "end_time": 500,
                        "text": "hello",
                        "words": [{
                            "begin_time": 100,
                            "end_time": 480,
                            "text": "hello",
                            "punctuation": "!"
                        }]
                    }]
                }]
            }),
        ),
        reply(
            200,
            json!({
                "request_id": "audio-query-2",
                "output": {
                    "task_id": "audio-task-2",
                    "task_status": "SUCCEEDED",
                    "results": [{
                        "subtask_status": "FAILED",
                        "code": "FILE_DOWNLOAD_FAILED",
                        "message": "file could not be read"
                    }],
                    "task_metrics": { "TOTAL": 1, "SUCCEEDED": 0, "FAILED": 1 }
                }
            }),
        ),
    ]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Beijing)).unwrap();
    let successful_ref =
        QwenAsrTaskRef::new(scope(QwenAsrRegion::Beijing), "audio-task-1").unwrap();
    let successful = service.get_task(&successful_ref, &options()).await.unwrap();
    assert_eq!(successful.status, QwenAsrTaskStatus::Succeeded);
    assert_eq!(
        successful.subtask_status,
        Some(QwenAsrTaskStatus::Succeeded)
    );
    assert_eq!(successful.duration_seconds, Some(9));
    let metrics = successful.task_metrics.as_ref().unwrap();
    assert_eq!(
        (metrics.total, metrics.succeeded, metrics.failed),
        (Some(1), Some(1), Some(0))
    );
    let artifact = successful.result().unwrap();
    assert!(!format!("{artifact:?}").contains("artifact-secret"));
    let transcript = service
        .fetch_transcription(artifact, &RequestOptions::default())
        .await
        .unwrap();
    let audio = transcript.audio_info.as_ref().unwrap();
    assert_eq!(audio.format.as_deref(), Some("wav"));
    assert_eq!(audio.channels, vec![0, 1]);
    assert_eq!(audio.original_sampling_rate, Some(48_000));
    assert_eq!(audio.original_duration_in_milliseconds, Some(7_500));
    assert_eq!(
        transcript.transcripts[0].content_duration_in_milliseconds,
        Some(3_200)
    );
    assert_eq!(
        transcript.transcripts[0].sentences[0].words[0]
            .punctuation
            .as_deref(),
        Some("!")
    );

    let failed_ref = QwenAsrTaskRef::new(scope(QwenAsrRegion::Beijing), "audio-task-2").unwrap();
    let failed = service.get_task(&failed_ref, &options()).await.unwrap();
    assert_eq!(failed.status, QwenAsrTaskStatus::Succeeded);
    assert_eq!(failed.subtask_status, Some(QwenAsrTaskStatus::Failed));
    assert_eq!(failed.subtask_code.as_deref(), Some("FILE_DOWNLOAD_FAILED"));
    assert_eq!(
        failed.subtask_message.as_deref(),
        Some("file could not be read")
    );
    assert!(failed.result().is_none());
    assert_eq!(failed.task_metrics.as_ref().unwrap().failed, Some(1));
}

#[tokio::test]
async fn audio_result_is_hidden_until_both_overall_and_subtask_succeed() {
    let url = "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/result.json?sig=secret";
    let transport = MockTransport::new([reply(
        200,
        json!({
            "output": {
                "task_id": "audio-task-contradictory",
                "task_status": "FAILED",
                "results": [{
                    "subtask_status": "SUCCEEDED",
                    "transcription_url": url
                }]
            }
        }),
    )]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Beijing)).unwrap();
    let reference =
        QwenAsrTaskRef::new(scope(QwenAsrRegion::Beijing), "audio-task-contradictory").unwrap();

    let task = service.get_task(&reference, &options()).await.unwrap();
    assert_eq!(task.status, QwenAsrTaskStatus::Failed);
    assert_eq!(task.subtask_status, Some(QwenAsrTaskStatus::Succeeded));
    assert!(task.result().is_none());
}

#[tokio::test]
async fn successful_task_with_empty_transcription_object_is_rejected() {
    let transcript_url =
        "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/result.json?sig=secret";
    let transport = MockTransport::new([
        reply(
            200,
            json!({
                "output": {
                    "task_id": "audio-task-empty-artifact",
                    "task_status": "SUCCEEDED",
                    "results": [{
                        "subtask_status": "SUCCEEDED",
                        "transcription_url": transcript_url
                    }]
                }
            }),
        ),
        reply(200, json!({})),
    ]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Beijing)).unwrap();
    let reference =
        QwenAsrTaskRef::new(scope(QwenAsrRegion::Beijing), "audio-task-empty-artifact").unwrap();
    let task = service.get_task(&reference, &options()).await.unwrap();

    assert!(matches!(
        service
            .fetch_transcription(task.result().unwrap(), &RequestOptions::default())
            .await,
        Err(QwenAsrError::InvalidResponse {
            operation:
                lingxi_llm_client::providers::qwen::asr::QwenAsrOperation::FetchTranscription,
            ..
        })
    ));
}

#[tokio::test]
async fn successful_audio_with_no_transcript_tracks_keeps_the_documented_empty_array() {
    let transcript_url =
        "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/result.json?sig=secret";
    let transport = MockTransport::new([
        reply(
            200,
            json!({
                "output": {
                    "task_id": "audio-task-silent",
                    "task_status": "SUCCEEDED",
                    "results": [{
                        "subtask_status": "SUCCEEDED",
                        "transcription_url": transcript_url
                    }]
                }
            }),
        ),
        reply(
            200,
            json!({
                "properties": { "audio_format": "wav" },
                "transcripts": []
            }),
        ),
    ]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Beijing)).unwrap();
    let reference =
        QwenAsrTaskRef::new(scope(QwenAsrRegion::Beijing), "audio-task-silent").unwrap();
    let task = service.get_task(&reference, &options()).await.unwrap();

    let transcription = service
        .fetch_transcription(task.result().unwrap(), &RequestOptions::default())
        .await
        .unwrap();
    assert!(transcription.transcripts.is_empty());
}

#[tokio::test]
async fn qwen_audio_filetrans_validates_documented_model_controls_before_http() {
    let transport = MockTransport::new([]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Beijing)).unwrap();

    let unsupported_keep_dialect = QwenAudioAsrRequest {
        model: QwenAudioAsrModel::QwenAudio30AsrFlashFiletrans,
        file_url: "https://audio.example.test/file.wav".into(),
        parameters: QwenAudioAsrParameters {
            keep_dialect: Some(false),
            ..Default::default()
        },
        context: vec![],
    };
    assert!(matches!(
        service
            .submit_audio_filetrans(&unsupported_keep_dialect, &options())
            .await,
        Err(QwenAsrError::InvalidInput(_))
    ));

    let invalid_controls = QwenAudioAsrRequest {
        model: QwenAudioAsrModel::QwenAudio31AsrFlashFiletrans,
        file_url: "https://audio.example.test/file.wav".into(),
        parameters: QwenAudioAsrParameters {
            speaker_count: Some(2),
            language_hints: Some(vec![
                QwenAudioAsrLanguage::Chinese,
                QwenAudioAsrLanguage::Chinese,
            ]),
            ..Default::default()
        },
        context: vec![],
    };
    assert!(matches!(
        service
            .submit_audio_filetrans(&invalid_controls, &options())
            .await,
        Err(QwenAsrError::InvalidInput(_))
    ));

    let too_much_context = (0..6).fold(
        QwenAudioAsrRequest::new(
            QwenAudioAsrModel::QwenAudio31AsrFlashFiletrans,
            "https://audio.example.test/file.wav",
        ),
        |request, _| request.with_context_turn(QwenAudioAsrContextTurn::new("prior")),
    );
    assert!(matches!(
        service
            .submit_audio_filetrans(&too_much_context, &options())
            .await,
        Err(QwenAsrError::InvalidInput(_))
    ));
    assert!(transport.sent().is_empty());
}

#[tokio::test]
async fn qwen_audio_filetrans_leaves_merged_vocabulary_truncation_to_provider() {
    let transport = MockTransport::new([reply(
        200,
        json!({
            "request_id": "audio-submit-large-vocabulary",
            "output": { "task_id": "audio-task-large-vocabulary", "task_status": "PENDING" }
        }),
    )]);
    let service = QwenAsrService::new(&transport, scope(QwenAsrRegion::Beijing)).unwrap();
    let vocabulary = (0..2_001)
        .map(|index| (format!("term-{index:04}"), 1))
        .collect();
    let request = QwenAudioAsrRequest {
        model: QwenAudioAsrModel::QwenAudio31AsrFlashFiletrans,
        file_url: "https://audio.example.test/file.wav".into(),
        parameters: QwenAudioAsrParameters {
            vocabulary: Some(vocabulary),
            ..Default::default()
        },
        context: vec![],
    };

    service
        .submit_audio_filetrans(&request, &options())
        .await
        .unwrap();
    let sent = transport.sent();
    assert_eq!(sent.len(), 1);
    let body: Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(
        body["parameters"]["vocabulary"].as_object().unwrap().len(),
        2_001
    );
}
