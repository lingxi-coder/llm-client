use bytes::{Bytes, BytesMut};
use futures::{stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use crate::providers::openai::audio::{AudioInput, TimedSegment, TimedWord};

/// An exact, host-selected profile and account. Session following stays in the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioRoute {
    pub profile_name: String,
    pub account_scope: String,
}
impl AudioRoute {
    pub fn new(profile_name: impl Into<String>, account_scope: impl Into<String>) -> Self {
        Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioOperation {
    FileTranscription,
    LiveAsr,
    Synthesis,
    IncrementalSynthesis,
    NativeRealtime,
}
/// Headerless encodings are distinct from their container representations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    #[default]
    Pcm16Le,
    Pcm16Be,
    Mulaw,
    Alaw,
}
impl AudioFormat {
    pub fn content_type(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Opus => "audio/opus",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Wav => "audio/wav",
            Self::Pcm16Le => "audio/pcm",
            Self::Pcm16Be => "audio/l16",
            Self::Mulaw => "audio/mulaw",
            Self::Alaw => "audio/alaw",
        }
    }
    pub fn is_raw(self) -> bool {
        matches!(
            self,
            Self::Pcm16Le | Self::Pcm16Be | Self::Mulaw | Self::Alaw
        )
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawAudioFormat {
    pub format: AudioFormat,
    pub sample_rate_hz: u32,
    pub channels: u8,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranscriptionRequest {
    pub model: Option<String>,
    pub language: Option<String>,
    pub raw_format: Option<RawAudioFormat>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SynthesisRequest {
    pub model: Option<String>,
    pub text: String,
    pub voice: Option<String>,
    #[serde(default)]
    pub format: AudioFormat,
    pub language: Option<String>,
}
impl SynthesisRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            model: None,
            text: text.into(),
            voice: None,
            format: AudioFormat::Pcm16Le,
            language: None,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioDispatch {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioErrorKind {
    InvalidRequest,
    Unsupported,
    Unavailable,
    MediaTooLarge,
    Provider,
}
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct AudioError {
    pub kind: AudioErrorKind,
    pub dispatch: AudioDispatch,
    pub message: String,
    /// Retain the original typed provider error for callers needing provider details.
    #[source]
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}
impl AudioError {
    pub fn dispatch(&self) -> AudioDispatch {
        self.dispatch
    }
    pub(crate) fn local(message: impl Into<String>) -> Self {
        Self {
            kind: AudioErrorKind::InvalidRequest,
            dispatch: AudioDispatch::NotSent,
            message: message.into(),
            source: None,
        }
    }
    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self {
            kind: AudioErrorKind::Unsupported,
            ..Self::local(message)
        }
    }
    pub(crate) fn limit(message: impl Into<String>) -> Self {
        Self {
            kind: AudioErrorKind::MediaTooLarge,
            ..Self::local(message)
        }
    }
    pub(crate) fn provider<E: std::error::Error + Send + Sync + 'static>(
        source: E,
        dispatch: AudioDispatch,
    ) -> Self {
        Self {
            kind: AudioErrorKind::Provider,
            message: source.to_string(),
            source: Some(Box::new(source)),
            dispatch,
        }
    }
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub audio_seconds: Option<f64>,
    pub characters: Option<u64>,
    pub cost_usd: Option<f64>,
    pub native: Option<Value>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub route: AudioRoute,
    pub provider_id: String,
    pub model: String,
    pub text: String,
    pub language: Option<String>,
    pub duration_seconds: Option<f64>,
    pub words: Vec<TimedWord>,
    pub segments: Vec<TimedSegment>,
    pub request_id: Option<String>,
    pub usage: AudioUsage,
    pub native: Option<Value>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioMetadata {
    pub route: AudioRoute,
    pub provider_id: String,
    pub model: String,
    pub format: AudioFormat,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u8>,
    pub bits_per_sample: Option<u8>,
    pub content_type: String,
    pub request_id: Option<String>,
}
/// Single-dispatch, pull-driven audio. Dropping the stream cancels the HTTP read.
/// A limit is mandatory when collecting; chunks remain encoded exactly as received.
pub struct AudioStream {
    pub metadata: AudioMetadata,
    pub usage: AudioUsage,
    pub(crate) body: BoxStream<'static, Result<Bytes, AudioError>>,
    pub(crate) shared_usage: Option<std::sync::Arc<std::sync::Mutex<AudioUsage>>>,
    delivered: usize,
    finished: bool,
}
impl AudioStream {
    pub(crate) fn new(
        metadata: AudioMetadata,
        usage: AudioUsage,
        body: BoxStream<'static, Result<Bytes, AudioError>>,
    ) -> Self {
        Self {
            metadata,
            usage,
            body,
            shared_usage: None,
            delivered: 0,
            finished: false,
        }
    }
    pub async fn next_chunk(&mut self) -> Result<Option<Bytes>, AudioError> {
        if self.finished {
            return Ok(None);
        }
        let chunk = self.body.next().await;
        if let Some(usage) = &self.shared_usage {
            self.usage = usage
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
        }
        match chunk {
            Some(Ok(bytes)) => {
                self.delivered = self.delivered.saturating_add(bytes.len());
                Ok(Some(bytes))
            }
            Some(Err(error)) => {
                self.finished = true;
                self.body = futures::stream::empty().boxed();
                Err(error)
            }
            None => {
                self.finished = true;
                if self.delivered == 0 {
                    Err(AudioError {
                        kind: AudioErrorKind::Provider,
                        dispatch: AudioDispatch::Accepted,
                        message: "audio response contained no audio bytes".into(),
                        source: None,
                    })
                } else {
                    Ok(None)
                }
            }
        }
    }
    pub async fn collect(mut self, max_bytes: usize) -> Result<CollectedAudio, AudioError> {
        if max_bytes == 0 {
            return Err(AudioError::local(
                "audio collection requires a positive byte limit",
            ));
        }
        let mut bytes = BytesMut::new();
        while let Some(chunk) = self.next_chunk().await? {
            if chunk.len() > max_bytes.saturating_sub(bytes.len()) {
                return Err(AudioError {
                    kind: AudioErrorKind::MediaTooLarge,
                    dispatch: AudioDispatch::Accepted,
                    message: "audio collection exceeded the caller's byte limit".into(),
                    source: None,
                });
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(CollectedAudio {
            bytes: bytes.freeze(),
            metadata: self.metadata,
            usage: self.usage,
        })
    }
}
impl std::fmt::Debug for AudioStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioStream")
            .field("metadata", &self.metadata)
            .field("delivered", &self.delivered)
            .finish_non_exhaustive()
    }
}
#[derive(Debug)]
pub struct CollectedAudio {
    pub bytes: Bytes,
    pub metadata: AudioMetadata,
    pub usage: AudioUsage,
}
/// A provider-owned URL is never downloaded by the facade.
pub struct AudioUrl {
    pub metadata: AudioMetadata,
    pub url: String,
    pub expires_at_unix_seconds: Option<u64>,
    pub usage: AudioUsage,
}
impl std::fmt::Debug for AudioUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioUrl")
            .field("metadata", &self.metadata)
            .field("url", &"<redacted provider URL>")
            .finish()
    }
}
#[derive(Debug)]
pub enum AudioOutput {
    Stream(AudioStream),
    Url(AudioUrl),
}
