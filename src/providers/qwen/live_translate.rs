//! Qwen LiveTranslate Realtime WebSocket adapter for Alibaba Cloud Model Studio.
//!
//! LiveTranslate has a wire contract distinct from Qwen-Omni Realtime. In
//! particular, Qwen 3.5 streams confirmed and tentative text in `text` and
//! `stash`, while Qwen 3.8 streams append-only `delta` events.

use crate::realtime::{
    validate_frame, validate_limits, RealtimeAudioFormat, RealtimeCodec, RealtimeConnectRequest,
    RealtimeConnection, RealtimeControl, RealtimeDriver, RealtimeError, RealtimeEvent,
    RealtimeFrame, RealtimeInput, RealtimeLimits, RealtimeSession, RealtimeTransport,
};
use crate::{files::provider_file_endpoint_fingerprint, protocol::Secret};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use bytes::Bytes;
use futures::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex},
};
use url::Url;

const MAX_SETUP_PREFACE_FRAMES: usize = 16;
const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;
const MAX_IMAGE_BYTES: usize = 500_000;
const QWEN35_OUTPUT_RATE_HZ: u32 = 24_000;
const QWEN38_INPUT_RATE_HZ: u32 = 16_000;
const QWEN38_OUTPUT_RATE_HZ: u32 = 24_000;

/// Workspace WebSocket path documented for Qwen LiveTranslate.
pub const QWEN_LIVETRANSLATE_WEBSOCKET_PATH: &str = "/api-ws/v1/realtime";

/// Model Studio region. The workspace and API key must belong to this region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenLiveTranslateRegion {
    Beijing,
    Singapore,
}

impl QwenLiveTranslateRegion {
    fn domain_suffix(self) -> &'static str {
        match self {
            Self::Beijing => "cn-beijing.maas.aliyuncs.com",
            Self::Singapore => "ap-southeast-1.maas.aliyuncs.com",
        }
    }
}

/// A regional workspace-bound LiveTranslate endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenLiveTranslateRoute {
    region: QwenLiveTranslateRegion,
    workspace_id: String,
    endpoint: String,
}

impl QwenLiveTranslateRoute {
    pub fn new(
        region: QwenLiveTranslateRegion,
        workspace_id: impl Into<String>,
    ) -> Result<Self, RealtimeError> {
        let workspace_id = workspace_id.into();
        validate_workspace_id(&workspace_id)?;
        let endpoint = format!(
            "wss://{}.{}/{}",
            workspace_id,
            region.domain_suffix(),
            QWEN_LIVETRANSLATE_WEBSOCKET_PATH.trim_start_matches('/')
        );
        let route = Self {
            region,
            workspace_id,
            endpoint,
        };
        route.validate()?;
        Ok(route)
    }

    pub fn region(&self) -> QwenLiveTranslateRegion {
        self.region
    }

    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn validate(&self) -> Result<(), RealtimeError> {
        validate_workspace_id(&self.workspace_id)?;
        let expected = format!(
            "wss://{}.{}/{}",
            self.workspace_id,
            self.region.domain_suffix(),
            QWEN_LIVETRANSLATE_WEBSOCKET_PATH.trim_start_matches('/')
        );
        if self.endpoint != expected {
            return Err(RealtimeError::InvalidConfig {
                message: "Qwen LiveTranslate endpoint does not match its workspace and region"
                    .into(),
            });
        }
        Ok(())
    }

    fn endpoint_fingerprint(&self) -> String {
        provider_file_endpoint_fingerprint(&self.endpoint)
    }

    fn model_endpoint(&self, model: QwenLiveTranslateModel) -> Result<String, RealtimeError> {
        self.validate()?;
        let mut url = Url::parse(&self.endpoint).map_err(|_| RealtimeError::InvalidConfig {
            message: "Qwen LiveTranslate endpoint is invalid".into(),
        })?;
        url.query_pairs_mut().append_pair("model", model.as_str());
        Ok(url.into())
    }
}

/// Non-secret profile/account identity bound to a regional workspace route.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenLiveTranslateScope {
    profile_name: String,
    account_scope: String,
    region: QwenLiveTranslateRegion,
    workspace_id: String,
    endpoint_fingerprint: String,
}

impl QwenLiveTranslateScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        route: &QwenLiveTranslateRoute,
    ) -> Result<Self, RealtimeError> {
        route.validate()?;
        let scope = Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region: route.region,
            workspace_id: route.workspace_id.clone(),
            endpoint_fingerprint: route.endpoint_fingerprint(),
        };
        scope.validate()?;
        Ok(scope)
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn account_scope(&self) -> &str {
        &self.account_scope
    }

    pub fn region(&self) -> QwenLiveTranslateRegion {
        self.region
    }

    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    fn validate(&self) -> Result<(), RealtimeError> {
        if self.profile_name.trim().is_empty()
            || self.account_scope.trim().is_empty()
            || self.endpoint_fingerprint.trim().is_empty()
        {
            return Err(RealtimeError::InvalidConfig {
                message: "Qwen LiveTranslate scope requires profile, account, region, workspace, and endpoint".into(),
            });
        }
        validate_workspace_id(&self.workspace_id)
    }

    fn validate_route(&self, route: &QwenLiveTranslateRoute) -> Result<(), RealtimeError> {
        route.validate()?;
        self.validate()?;
        if self.region != route.region
            || self.workspace_id != route.workspace_id
            || self.endpoint_fingerprint != route.endpoint_fingerprint()
        {
            return Err(RealtimeError::InvalidConfig {
                message: "Qwen LiveTranslate account scope belongs to a different region, workspace, or endpoint".into(),
            });
        }
        Ok(())
    }
}

/// First-party LiveTranslate models documented for WebSocket sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QwenLiveTranslateModel {
    #[serde(rename = "qwen3.5-livetranslate-flash-realtime")]
    Qwen35LiveTranslateFlashRealtime,
    #[serde(rename = "qwen3.5-livetranslate-flash-realtime-2026-05-19")]
    Qwen35LiveTranslateFlashRealtime20260519,
    #[serde(rename = "qwen3.8-livetranslate-flash-realtime")]
    Qwen38LiveTranslateFlashRealtime,
}

impl QwenLiveTranslateModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qwen35LiveTranslateFlashRealtime => "qwen3.5-livetranslate-flash-realtime",
            Self::Qwen35LiveTranslateFlashRealtime20260519 => {
                "qwen3.5-livetranslate-flash-realtime-2026-05-19"
            }
            Self::Qwen38LiveTranslateFlashRealtime => "qwen3.8-livetranslate-flash-realtime",
        }
    }

    fn is_qwen35(self) -> bool {
        matches!(
            self,
            Self::Qwen35LiveTranslateFlashRealtime | Self::Qwen35LiveTranslateFlashRealtime20260519
        )
    }
}

/// Whether translation produces text only or translated text and speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenLiveTranslateOutputMode {
    Text,
    TextAndAudio,
}

/// Input codec choices documented for the Qwen 3.5 session format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Qwen35LiveTranslateInputAudioFormat {
    #[default]
    Pcm,
    Opus,
}

impl Qwen35LiveTranslateInputAudioFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pcm => "pcm",
            Self::Opus => "opus",
        }
    }
}

/// 3.5 VAD supports manual commit or server-side VAD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Qwen35LiveTranslateTurnDetection {
    Manual,
    ServerVad {
        threshold: f32,
        silence_duration_ms: u32,
    },
}

impl Default for Qwen35LiveTranslateTurnDetection {
    fn default() -> Self {
        Self::ServerVad {
            threshold: 0.2,
            silence_duration_ms: 1000,
        }
    }
}

/// 3.8 supports speaker detection or server VAD in its nested audio schema.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Qwen38LiveTranslateTurnDetection {
    #[default]
    SpeakerDetection,
    ServerVad,
}

/// Same-language output behavior supported only by Qwen 3.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QwenLiveTranslateSameLanguageSkip {
    pub skip_text: bool,
    pub skip_audio: bool,
}

/// Voice cloning options supported by the Qwen 3.5 session schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenLiveTranslateVoiceCloneFrequency {
    Never,
    Once,
    Always,
}

impl QwenLiveTranslateVoiceCloneFrequency {
    fn as_str(self) -> &'static str {
        match self {
            Self::Never => "never",
            Self::Once => "once",
            Self::Always => "always",
        }
    }
}

/// Explicit model-specific session settings. Keeping separate structs avoids
/// emitting Qwen 3.5 fields in the nested Qwen 3.8 protocol or vice versa.
#[derive(Clone)]
pub enum QwenLiveTranslateConfig {
    Qwen35(Qwen35LiveTranslateConfig),
    Qwen38(Qwen38LiveTranslateConfig),
}

/// Flat session configuration for the Qwen 3.5 LiveTranslate wire format.
#[derive(Clone)]
pub struct Qwen35LiveTranslateConfig {
    pub model: QwenLiveTranslateModel,
    pub output_mode: QwenLiveTranslateOutputMode,
    pub target_language: String,
    pub source_language: Option<String>,
    pub enable_source_transcription: bool,
    pub input_audio_format: Qwen35LiveTranslateInputAudioFormat,
    pub input_sample_rate_hz: u32,
    pub turn_detection: Qwen35LiveTranslateTurnDetection,
    pub voice: Option<String>,
    pub voice_clone_frequency: Option<QwenLiveTranslateVoiceCloneFrequency>,
    pub same_language_skip: Option<QwenLiveTranslateSameLanguageSkip>,
    pub hotwords: BTreeMap<String, String>,
}

impl Qwen35LiveTranslateConfig {
    pub fn new(model: QwenLiveTranslateModel) -> Result<Self, RealtimeError> {
        if !model.is_qwen35() {
            return Err(RealtimeError::InvalidConfig {
                message: "Qwen 3.5 LiveTranslate config requires a Qwen 3.5 model ID".into(),
            });
        }
        Ok(Self {
            model,
            output_mode: QwenLiveTranslateOutputMode::TextAndAudio,
            target_language: "en".into(),
            source_language: None,
            enable_source_transcription: false,
            input_audio_format: Qwen35LiveTranslateInputAudioFormat::Pcm,
            input_sample_rate_hz: 16_000,
            turn_detection: Qwen35LiveTranslateTurnDetection::default(),
            voice: None,
            voice_clone_frequency: None,
            same_language_skip: None,
            hotwords: BTreeMap::new(),
        })
    }
}

impl Default for Qwen35LiveTranslateConfig {
    fn default() -> Self {
        Self::new(QwenLiveTranslateModel::Qwen35LiveTranslateFlashRealtime)
            .expect("default Qwen LiveTranslate model is a Qwen 3.5 model")
    }
}

/// Nested session configuration for the Qwen 3.8 LiveTranslate wire format.
#[derive(Clone)]
pub struct Qwen38LiveTranslateConfig {
    pub output_mode: QwenLiveTranslateOutputMode,
    pub target_language: String,
    pub turn_detection: Qwen38LiveTranslateTurnDetection,
    pub voice: Option<String>,
    pub hotwords: BTreeMap<String, String>,
}

impl Default for Qwen38LiveTranslateConfig {
    fn default() -> Self {
        Self {
            output_mode: QwenLiveTranslateOutputMode::TextAndAudio,
            target_language: "en".into(),
            turn_detection: Qwen38LiveTranslateTurnDetection::default(),
            voice: None,
            hotwords: BTreeMap::new(),
        }
    }
}

impl QwenLiveTranslateConfig {
    pub fn new_qwen35() -> Self {
        Self::Qwen35(Qwen35LiveTranslateConfig::default())
    }

    pub fn new_qwen38() -> Self {
        Self::Qwen38(Qwen38LiveTranslateConfig::default())
    }

    pub fn qwen35(config: Qwen35LiveTranslateConfig) -> Self {
        Self::Qwen35(config)
    }

    pub fn qwen38(config: Qwen38LiveTranslateConfig) -> Self {
        Self::Qwen38(config)
    }

    pub fn model(&self) -> QwenLiveTranslateModel {
        match self {
            Self::Qwen35(config) => config.model,
            Self::Qwen38(_) => QwenLiveTranslateModel::Qwen38LiveTranslateFlashRealtime,
        }
    }

    fn input_audio_format(&self) -> RealtimeAudioFormat {
        match self {
            Self::Qwen35(config) => match config.input_audio_format {
                Qwen35LiveTranslateInputAudioFormat::Pcm => RealtimeAudioFormat::Pcm16 {
                    sample_rate_hz: config.input_sample_rate_hz,
                },
                Qwen35LiveTranslateInputAudioFormat::Opus => RealtimeAudioFormat::Encoded {
                    mime_type: "audio/opus".into(),
                },
            },
            Self::Qwen38(_) => RealtimeAudioFormat::Pcm16 {
                sample_rate_hz: QWEN38_INPUT_RATE_HZ,
            },
        }
    }

    fn output_sample_rate_hz(&self) -> u32 {
        match self {
            Self::Qwen35(_) => QWEN35_OUTPUT_RATE_HZ,
            Self::Qwen38(_) => QWEN38_OUTPUT_RATE_HZ,
        }
    }
}

impl Default for QwenLiveTranslateConfig {
    fn default() -> Self {
        Self::new_qwen35()
    }
}

impl fmt::Debug for Qwen35LiveTranslateConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Qwen35LiveTranslateConfig")
            .field("model", &self.model)
            .field("output_mode", &self.output_mode)
            .field("target_language", &self.target_language)
            .field("source_language", &self.source_language)
            .field(
                "enable_source_transcription",
                &self.enable_source_transcription,
            )
            .field("input_audio_format", &self.input_audio_format)
            .field("input_sample_rate_hz", &self.input_sample_rate_hz)
            .field("turn_detection", &self.turn_detection)
            .field("voice", &self.voice)
            .field("voice_clone_frequency", &self.voice_clone_frequency)
            .field("same_language_skip", &self.same_language_skip)
            .field("hotwords", &self.hotwords.len())
            .finish()
    }
}

impl fmt::Debug for Qwen38LiveTranslateConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Qwen38LiveTranslateConfig")
            .field("output_mode", &self.output_mode)
            .field("target_language", &self.target_language)
            .field("turn_detection", &self.turn_detection)
            .field("voice", &self.voice)
            .field("hotwords", &self.hotwords.len())
            .finish()
    }
}

impl fmt::Debug for QwenLiveTranslateConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen35(config) => f.debug_tuple("Qwen35").field(config).finish(),
            Self::Qwen38(config) => f.debug_tuple("Qwen38").field(config).finish(),
        }
    }
}

/// Typed LiveTranslate events. Model-specific event variants preserve the
/// distinct 3.5 `text`/`stash` and 3.8 append-only `delta` semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum QwenLiveTranslateEvent {
    Realtime(RealtimeEvent),
    SessionCreated {
        session: Value,
        native: Value,
    },
    SessionUpdated {
        session: Value,
        native: Value,
    },
    SessionFinished {
        native: Value,
    },
    SpeechStarted {
        item_id: Option<String>,
        audio_start_ms: Option<u64>,
        speaker_id: Option<u64>,
        native: Value,
    },
    SpeechStopped {
        item_id: Option<String>,
        audio_end_ms: Option<u64>,
        native: Value,
    },
    InputAudioCommitted {
        native: Value,
    },
    InputAudioCleared {
        native: Value,
    },
    SourceTranscription35 {
        item_id: Option<String>,
        text: String,
        stash: String,
        language: Option<String>,
        emotion: Option<String>,
        native: Value,
    },
    SourceTranscriptionDelta38 {
        item_id: Option<String>,
        delta: String,
        native: Value,
    },
    SourceTranscriptionCompleted {
        item_id: Option<String>,
        transcript: String,
        language: Option<String>,
        emotion: Option<String>,
        native: Value,
    },
    SourceTranscriptionFailed {
        item_id: Option<String>,
        code: Option<String>,
        message: Option<String>,
        native: Value,
    },
    ResponseCreated {
        response: Value,
        native: Value,
    },
    TextProgress35 {
        response_id: Option<String>,
        item_id: Option<String>,
        text: String,
        stash: String,
        native: Value,
    },
    TextDelta38 {
        response_id: Option<String>,
        item_id: Option<String>,
        delta: String,
        native: Value,
    },
    TextDone {
        response_id: Option<String>,
        item_id: Option<String>,
        text: String,
        native: Value,
    },
    AudioTranscriptProgress35 {
        response_id: Option<String>,
        item_id: Option<String>,
        text: String,
        stash: String,
        native: Value,
    },
    AudioTranscriptDelta38 {
        response_id: Option<String>,
        item_id: Option<String>,
        delta: String,
        native: Value,
    },
    AudioTranscriptDone {
        response_id: Option<String>,
        item_id: Option<String>,
        transcript: String,
        native: Value,
    },
    AudioDelta {
        response_id: Option<String>,
        item_id: Option<String>,
        output_index: Option<u64>,
        content_index: Option<u64>,
        data: Bytes,
        format: RealtimeAudioFormat,
        native: Value,
    },
    AudioDone {
        response_id: Option<String>,
        item_id: Option<String>,
        native: Value,
    },
    ResponseDone {
        response: Value,
        native: Value,
    },
    ProviderError {
        error_type: Option<String>,
        code: Option<String>,
        message: String,
        param: Option<String>,
        native: Value,
    },
    Native {
        event_type: String,
        native: Value,
    },
}

/// Connected Qwen LiveTranslate session. Setup returns after `session.updated`.
pub struct QwenLiveTranslateSession {
    control: QwenLiveTranslateControl,
    events: QwenLiveTranslateEvents,
    scope: QwenLiveTranslateScope,
}

impl QwenLiveTranslateSession {
    pub async fn connect(
        transport: Arc<dyn RealtimeTransport>,
        route: QwenLiveTranslateRoute,
        scope: QwenLiveTranslateScope,
        credential: Secret<String>,
        config: QwenLiveTranslateConfig,
        limits: RealtimeLimits,
    ) -> Result<(Self, RealtimeDriver), RealtimeError> {
        validate_limits(limits)?;
        route.validate()?;
        scope.validate_route(&route)?;
        validate_credential(&credential)?;
        validate_config(&config)?;

        let setup = build_session_update(&config);
        let setup_bytes = serde_json::to_vec(&setup).map_err(|_| RealtimeError::Codec {
            message: "could not encode Qwen LiveTranslate session update".into(),
        })?;
        if setup_bytes.len() > limits.max_frame_bytes {
            return Err(RealtimeError::FrameTooLarge {
                actual: setup_bytes.len(),
                max: limits.max_frame_bytes,
            });
        }

        let request = RealtimeConnectRequest {
            endpoint: route.model_endpoint(config.model())?,
            headers: vec![(
                "Authorization".into(),
                format!("Bearer {}", credential.expose_secret()),
            )],
            max_frame_bytes: limits.max_frame_bytes,
        };
        let setup_transport = QwenLiveTranslateSetupTransport {
            inner: transport,
            setup_frame: RealtimeFrame::Text(Bytes::from(setup_bytes)),
            max_frame_bytes: limits.max_frame_bytes,
            requested_model: config.model(),
        };
        let codec = Arc::new(QwenLiveTranslateCodec {
            config: config.clone(),
        });
        let (session, driver) =
            RealtimeSession::connect(&setup_transport, request, codec, limits).await?;
        let (control, events) = session.into_parts();
        let state = Arc::new(Mutex::new(LiveTranslateState::default()));
        Ok((
            Self {
                control: QwenLiveTranslateControl {
                    inner: control,
                    state: state.clone(),
                    input_audio_format: config.input_audio_format(),
                    turn_detection: config.turn_detection(),
                },
                events: QwenLiveTranslateEvents {
                    inner: events,
                    state,
                    model: config.model(),
                    output_audio_format: RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: config.output_sample_rate_hz(),
                    },
                },
                scope,
            },
            driver,
        ))
    }

    pub fn into_parts(self) -> (QwenLiveTranslateControl, QwenLiveTranslateEvents) {
        (self.control, self.events)
    }

    pub fn scope(&self) -> &QwenLiveTranslateScope {
        &self.scope
    }
}

#[derive(Default)]
struct LiveTranslateState {
    audio_appended: bool,
    pending_audio: bool,
    finish_sent: bool,
    session_finished: bool,
}

/// Sender for audio, image, clear, commit, and finish events.
#[derive(Clone)]
pub struct QwenLiveTranslateControl {
    inner: RealtimeControl,
    state: Arc<Mutex<LiveTranslateState>>,
    input_audio_format: RealtimeAudioFormat,
    turn_detection: QwenLiveTranslateTurnMode,
}

#[derive(Clone, Copy)]
enum QwenLiveTranslateTurnMode {
    Qwen35Manual,
    Qwen35ServerVad,
    Qwen38,
}

impl QwenLiveTranslateControl {
    pub fn send(&self, input: RealtimeInput) -> Result<(), RealtimeError> {
        let mut state = self.state.lock().unwrap();
        if state.finish_sent {
            return Err(RealtimeError::InvalidInput {
                message: "Qwen LiveTranslate does not accept inputs after session.finish".into(),
            });
        }
        if matches!(&input, RealtimeInput::Image { .. }) && !state.audio_appended {
            return Err(RealtimeError::InvalidInput {
                message: "Qwen LiveTranslate requires an audio append before an image frame".into(),
            });
        }
        if matches!(&input, RealtimeInput::CommitAudio)
            && !matches!(self.turn_detection, QwenLiveTranslateTurnMode::Qwen35Manual)
        {
            return Err(RealtimeError::InvalidInput {
                message: "Qwen LiveTranslate audio commit is valid only in Qwen 3.5 manual mode"
                    .into(),
            });
        }
        if matches!(&input, RealtimeInput::CommitAudio) && !state.pending_audio {
            return Err(RealtimeError::InvalidInput {
                message: "Qwen LiveTranslate cannot commit an empty audio buffer".into(),
            });
        }
        let audio_appended = matches!(&input, RealtimeInput::Audio { .. });
        let clear_audio = matches!(&input, RealtimeInput::ClearAudio);
        let commit_audio = matches!(&input, RealtimeInput::CommitAudio);
        let finish = matches!(&input, RealtimeInput::FinishSession);
        self.inner.send(input)?;
        if audio_appended {
            state.audio_appended = true;
            state.pending_audio = true;
        }
        if clear_audio || commit_audio {
            state.pending_audio = false;
        }
        if finish {
            state.finish_sent = true;
        }
        Ok(())
    }

    pub fn clear_audio(&self) -> Result<(), RealtimeError> {
        self.send(RealtimeInput::ClearAudio)
    }

    /// Sends `session.finish`; keep consuming events until `SessionFinished`
    /// before explicitly closing if final transcripts are needed.
    pub fn finish(&self) -> Result<(), RealtimeError> {
        self.send(RealtimeInput::FinishSession)
    }

    pub fn commit_audio(&self) -> Result<(), RealtimeError> {
        self.send(RealtimeInput::CommitAudio)
    }

    /// Explicitly close the WebSocket. This does not wait for `session.finished`;
    /// callers may use it to abort a stalled or failed session.
    pub async fn close(&self, close: crate::realtime::RealtimeClose) -> Result<(), RealtimeError> {
        self.inner.close(close).await
    }

    pub async fn close_after_finished(
        &self,
        close: crate::realtime::RealtimeClose,
    ) -> Result<(), RealtimeError> {
        if !self.state.lock().unwrap().session_finished {
            return Err(RealtimeError::InvalidInput {
                message:
                    "Qwen LiveTranslate close_after_finished requires the session.finished event"
                        .into(),
            });
        }
        self.inner.close(close).await
    }

    pub fn audio_format(&self) -> RealtimeAudioFormat {
        self.input_audio_format.clone()
    }
}

/// Receiver that retains LiveTranslate event details and finish lifecycle.
pub struct QwenLiveTranslateEvents {
    inner: crate::realtime::RealtimeEvents,
    state: Arc<Mutex<LiveTranslateState>>,
    model: QwenLiveTranslateModel,
    output_audio_format: RealtimeAudioFormat,
}

impl QwenLiveTranslateEvents {
    pub async fn next(&mut self) -> Option<QwenLiveTranslateEvent> {
        let event = self.inner.next().await?;
        let RealtimeEvent::ProviderEvent { name, native } = &event else {
            return Some(QwenLiveTranslateEvent::Realtime(event));
        };
        let parsed = parse_native_event(name, native, self.model, &self.output_audio_format);
        if matches!(parsed, QwenLiveTranslateEvent::SessionFinished { .. }) {
            self.state.lock().unwrap().session_finished = true;
        }
        Some(parsed)
    }
}

struct QwenLiveTranslateSetupTransport {
    inner: Arc<dyn RealtimeTransport>,
    setup_frame: RealtimeFrame,
    max_frame_bytes: usize,
    requested_model: QwenLiveTranslateModel,
}

#[async_trait]
impl RealtimeTransport for QwenLiveTranslateSetupTransport {
    async fn connect(
        &self,
        request: RealtimeConnectRequest,
    ) -> Result<RealtimeConnection, RealtimeError> {
        let mut connection = self.inner.connect(request).await?;
        let mut preface = Vec::new();
        let mut created = false;
        let mut update_sent = false;
        let mut update_acknowledged = false;
        for _ in 0..MAX_SETUP_PREFACE_FRAMES {
            let frame = connection
                .inbound
                .next()
                .await
                .ok_or(RealtimeError::UnexpectedRemoteClose)??;
            validate_frame(&frame, self.max_frame_bytes)?;
            let native = parse_native_frame(&frame)?;
            let name = event_type(&native)?;
            if name == "error" {
                return Err(RealtimeError::Transport {
                    message: setup_error_message(&native),
                });
            }
            if name == "session.created" {
                if let Some(server_model) = native
                    .get("session")
                    .and_then(|session| session.get("model"))
                    .and_then(Value::as_str)
                {
                    if !same_model_identity(self.requested_model, server_model) {
                        return Err(RealtimeError::Transport {
                            message: "Qwen LiveTranslate session was created for a different model"
                                .into(),
                        });
                    }
                }
            }
            preface.push(Ok(frame));
            match name {
                "session.created" if !update_sent => {
                    created = true;
                    connection.outbound.send(self.setup_frame.clone()).await?;
                    update_sent = true;
                }
                "session.updated" if update_sent => {
                    update_acknowledged = true;
                    break;
                }
                _ => {}
            }
        }
        if !created || !update_acknowledged {
            return Err(RealtimeError::Transport {
                message:
                    "Qwen LiveTranslate did not acknowledge session.update within the setup limit"
                        .into(),
            });
        }
        connection.inbound = stream::iter(preface).chain(connection.inbound).boxed();
        Ok(connection)
    }
}

struct QwenLiveTranslateCodec {
    config: QwenLiveTranslateConfig,
}

impl RealtimeCodec for QwenLiveTranslateCodec {
    fn encode(&self, input: &RealtimeInput) -> Result<Vec<RealtimeFrame>, RealtimeError> {
        let message = match input {
            RealtimeInput::ImportHistory { .. } => return Err(RealtimeError::InvalidInput { message: "history import is unsupported by this adapter".into() }),
            RealtimeInput::Audio { data, format } => {
                validate_input_audio(data, format, &self.config.input_audio_format())?;
                json!({ "type": "input_audio_buffer.append", "audio": STANDARD.encode(data) })
            }
            RealtimeInput::Image { data, mime_type } => {
                validate_image(data, mime_type)?;
                json!({ "type": "input_image_buffer.append", "image": STANDARD.encode(data) })
            }
            RealtimeInput::CommitAudio => match &self.config {
                QwenLiveTranslateConfig::Qwen35(config)
                    if matches!(config.turn_detection, Qwen35LiveTranslateTurnDetection::Manual) =>
                {
                    json!({ "type": "input_audio_buffer.commit" })
                }
                _ => {
                    return Err(RealtimeError::InvalidInput {
                        message: "Qwen LiveTranslate audio commit is valid only in Qwen 3.5 manual mode".into(),
                    })
                }
            },
            RealtimeInput::ClearAudio => json!({ "type": "input_audio_buffer.clear" }),
            RealtimeInput::FinishSession => json!({ "type": "session.finish" }),
            RealtimeInput::Text(_)
            | RealtimeInput::RetrieveItem { .. }
            | RealtimeInput::DeleteItem { .. }
            | RealtimeInput::TruncateAudio { .. }
            | RealtimeInput::ToolResult { .. }
            | RealtimeInput::ToolResults { .. }
            | RealtimeInput::ContinueResponse
            | RealtimeInput::Interrupt => {
                return Err(RealtimeError::InvalidInput {
                    message: "Qwen LiveTranslate accepts audio, images, audio-buffer controls, and session.finish only".into(),
                })
            }
        };
        serde_json::to_vec(&message)
            .map(|bytes| vec![RealtimeFrame::Text(Bytes::from(bytes))])
            .map_err(|_| RealtimeError::Codec {
                message: "could not encode Qwen LiveTranslate client event".into(),
            })
    }

    fn decode(&self, frame: RealtimeFrame) -> Result<Vec<RealtimeEvent>, RealtimeError> {
        let native = parse_native_frame(&frame)?;
        let name = event_type(&native)?;
        if name == "response.audio.delta" {
            let encoded = native.get("delta").and_then(Value::as_str).ok_or_else(|| {
                RealtimeError::Codec {
                    message: "Qwen LiveTranslate audio delta is missing base64 `delta`".into(),
                }
            })?;
            STANDARD.decode(encoded).map_err(|_| RealtimeError::Codec {
                message: "Qwen LiveTranslate audio delta is not valid base64".into(),
            })?;
        }
        Ok(vec![RealtimeEvent::ProviderEvent {
            name: name.to_owned(),
            native,
        }])
    }
}

impl QwenLiveTranslateConfig {
    fn turn_detection(&self) -> QwenLiveTranslateTurnMode {
        match self {
            Self::Qwen35(config) => match config.turn_detection {
                Qwen35LiveTranslateTurnDetection::Manual => QwenLiveTranslateTurnMode::Qwen35Manual,
                Qwen35LiveTranslateTurnDetection::ServerVad { .. } => {
                    QwenLiveTranslateTurnMode::Qwen35ServerVad
                }
            },
            Self::Qwen38(_) => QwenLiveTranslateTurnMode::Qwen38,
        }
    }
}

fn validate_workspace_id(workspace_id: &str) -> Result<(), RealtimeError> {
    let valid = !workspace_id.is_empty()
        && workspace_id.len() <= 63
        && !workspace_id.starts_with('-')
        && !workspace_id.ends_with('-')
        && workspace_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-');
    if !valid {
        return Err(RealtimeError::InvalidConfig {
            message: "Qwen LiveTranslate workspace ID must be one valid DNS label".into(),
        });
    }
    Ok(())
}

fn validate_credential(credential: &Secret<String>) -> Result<(), RealtimeError> {
    let value = credential.expose_secret();
    if value.trim().is_empty()
        || value.len() > MAX_CREDENTIAL_BYTES
        || value
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte == b' ')
    {
        return Err(RealtimeError::InvalidConfig {
            message: "Qwen LiveTranslate bearer credential is empty or invalid".into(),
        });
    }
    Ok(())
}

fn validate_config(config: &QwenLiveTranslateConfig) -> Result<(), RealtimeError> {
    match config {
        QwenLiveTranslateConfig::Qwen35(config) => {
            if !config.model.is_qwen35() {
                return Err(RealtimeError::InvalidConfig {
                    message: "Qwen 3.5 LiveTranslate config requires a Qwen 3.5 model ID".into(),
                });
            }
            if !matches!(config.input_sample_rate_hz, 8_000 | 16_000) {
                return Err(RealtimeError::InvalidConfig {
                    message: "Qwen 3.5 LiveTranslate input rate must be 8 kHz or 16 kHz".into(),
                });
            }
            if let Qwen35LiveTranslateTurnDetection::ServerVad {
                threshold,
                silence_duration_ms,
            } = config.turn_detection
            {
                if !threshold.is_finite() || !(-1.0..=1.0).contains(&threshold) {
                    return Err(RealtimeError::InvalidConfig {
                        message: "Qwen 3.5 LiveTranslate VAD threshold must be between -1 and 1"
                            .into(),
                    });
                }
                if !(200..=6000).contains(&silence_duration_ms) {
                    return Err(RealtimeError::InvalidConfig {
                        message: "Qwen 3.5 LiveTranslate VAD silence duration must be 200–6000 ms"
                            .into(),
                    });
                }
            }
            if config
                .source_language
                .as_ref()
                .is_some_and(|language| !valid_language_tag(language))
            {
                return Err(RealtimeError::InvalidConfig {
                    message: "Qwen LiveTranslate source language must be a non-empty language tag"
                        .into(),
                });
            }
            if !valid_language_tag(&config.target_language) {
                return Err(RealtimeError::InvalidConfig {
                    message: "Qwen LiveTranslate target language must be a non-empty language tag"
                        .into(),
                });
            }
            if config.same_language_skip.is_some()
                && !matches!(config.target_language.as_str(), "zh" | "en")
            {
                return Err(RealtimeError::InvalidConfig {
                    message:
                        "Qwen 3.5 same-language skip options require target language `zh` or `en`"
                            .into(),
                });
            }
            if let Some(frequency) = config.voice_clone_frequency {
                match (frequency, config.voice.as_deref()) {
                    (QwenLiveTranslateVoiceCloneFrequency::Never, Some(voice))
                        if voice != "default" && !voice.trim().is_empty() => {}
                    (QwenLiveTranslateVoiceCloneFrequency::Once, Some("default"))
                    | (QwenLiveTranslateVoiceCloneFrequency::Always, Some("default")) => {}
                    _ => {
                        return Err(RealtimeError::InvalidConfig {
                            message: "Qwen 3.5 voice cloning requires `default` for once/always or a cloned voice ID for never".into(),
                        })
                    }
                }
            }
            validate_voice(config.voice.as_deref())?;
            validate_hotwords(&config.hotwords)?;
        }
        QwenLiveTranslateConfig::Qwen38(config) => {
            if !valid_language_tag(&config.target_language) {
                return Err(RealtimeError::InvalidConfig {
                    message: "Qwen LiveTranslate target language must be a non-empty language tag"
                        .into(),
                });
            }
            validate_voice(config.voice.as_deref())?;
            validate_hotwords(&config.hotwords)?;
        }
    }
    Ok(())
}

fn valid_language_tag(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 64 && !value.contains('\0')
}

fn validate_voice(voice: Option<&str>) -> Result<(), RealtimeError> {
    if voice
        .is_some_and(|value| value.trim().is_empty() || value.len() > 256 || value.contains('\0'))
    {
        return Err(RealtimeError::InvalidConfig {
            message: "Qwen LiveTranslate voice must be non-empty, NUL-free, and at most 256 bytes"
                .into(),
        });
    }
    Ok(())
}

fn validate_hotwords(hotwords: &BTreeMap<String, String>) -> Result<(), RealtimeError> {
    if hotwords.len() > 1000
        || hotwords
            .iter()
            .any(|(source, target)| source.trim().is_empty() || target.trim().is_empty())
    {
        return Err(RealtimeError::InvalidConfig {
            message:
                "Qwen LiveTranslate hotwords must be non-empty mappings with at most 1000 entries"
                    .into(),
        });
    }
    Ok(())
}

fn build_session_update(config: &QwenLiveTranslateConfig) -> Value {
    let modalities = |mode| match mode {
        QwenLiveTranslateOutputMode::Text => json!(["text"]),
        QwenLiveTranslateOutputMode::TextAndAudio => json!(["text", "audio"]),
    };
    match config {
        QwenLiveTranslateConfig::Qwen35(config) => {
            let turn_detection = match config.turn_detection {
                Qwen35LiveTranslateTurnDetection::Manual => Value::Null,
                Qwen35LiveTranslateTurnDetection::ServerVad {
                    threshold,
                    silence_duration_ms,
                } => json!({
                    "type": "server_vad",
                    "threshold": threshold,
                    "silence_duration_ms": silence_duration_ms,
                }),
            };
            let mut session = json!({
                "modalities": modalities(config.output_mode),
                "sample_rate": config.input_sample_rate_hz,
                "input_audio_format": config.input_audio_format.as_str(),
                "output_audio_format": "pcm",
                "turn_detection": turn_detection,
                "translation": { "language": config.target_language },
            });
            if let Some(voice) = &config.voice {
                session["voice"] = Value::String(voice.clone());
            }
            if let Some(frequency) = config.voice_clone_frequency {
                session["enable_voice_clone"] = Value::Bool(true);
                session["voice_clone_options"] = json!({ "frequency": frequency.as_str() });
            }
            if config.enable_source_transcription || config.source_language.is_some() {
                let mut transcription = serde_json::Map::new();
                if config.enable_source_transcription {
                    transcription.insert(
                        "model".into(),
                        Value::String("qwen3-asr-flash-realtime".into()),
                    );
                }
                if let Some(language) = &config.source_language {
                    transcription.insert("language".into(), Value::String(language.clone()));
                }
                session["input_audio_transcription"] = Value::Object(transcription);
            }
            if let Some(skip) = config.same_language_skip {
                session["translation"]["same_language_skip_options"] = json!({
                    "skip_text": skip.skip_text,
                    "skip_audio": skip.skip_audio,
                });
            }
            if !config.hotwords.is_empty() {
                session["translation"]["corpus"] = json!({ "phrases": config.hotwords });
            }
            json!({ "type": "session.update", "session": session })
        }
        QwenLiveTranslateConfig::Qwen38(config) => {
            let detection = match config.turn_detection {
                Qwen38LiveTranslateTurnDetection::SpeakerDetection => {
                    json!({ "type": "speaker_detection" })
                }
                Qwen38LiveTranslateTurnDetection::ServerVad => json!({ "type": "server_vad" }),
            };
            let mut session = json!({
                "output_modalities": modalities(config.output_mode),
                "audio": {
                    "input": {
                        "format": { "type": "pcm", "sample_rate": QWEN38_INPUT_RATE_HZ },
                        "turn_detection": detection,
                    },
                    "output": { "format": { "type": "pcm", "sample_rate": QWEN38_OUTPUT_RATE_HZ } },
                },
                "translation": { "language": config.target_language },
            });
            if let Some(voice) = &config.voice {
                session["audio"]["output"]["voice"] = Value::String(voice.clone());
            }
            if !config.hotwords.is_empty() {
                session["translation"]["corpus"] = json!({ "phrases": config.hotwords });
            }
            json!({ "type": "session.update", "session": session })
        }
    }
}

fn validate_input_audio(
    data: &Bytes,
    format: &RealtimeAudioFormat,
    expected_format: &RealtimeAudioFormat,
) -> Result<(), RealtimeError> {
    if data.is_empty() {
        return Err(RealtimeError::InvalidInput {
            message: "Qwen LiveTranslate audio input must not be empty".into(),
        });
    }
    if format != expected_format {
        return Err(RealtimeError::InvalidInput {
            message: format!("Qwen LiveTranslate audio format does not match the configured {expected_format:?} input"),
        });
    }
    if matches!(expected_format, RealtimeAudioFormat::Pcm16 { .. }) && !data.len().is_multiple_of(2)
    {
        return Err(RealtimeError::InvalidInput {
            message: "Qwen LiveTranslate PCM16 input must contain complete samples".into(),
        });
    }
    Ok(())
}

fn validate_image(data: &Bytes, mime_type: &str) -> Result<(), RealtimeError> {
    let valid_mime = matches!(mime_type, "image/jpeg" | "image/jpg");
    if !valid_mime || data.len() < 3 || !data.starts_with(&[0xff, 0xd8, 0xff]) {
        return Err(RealtimeError::InvalidInput {
            message: "Qwen LiveTranslate image frames must contain JPG/JPEG data".into(),
        });
    }
    if data.len() > MAX_IMAGE_BYTES {
        return Err(RealtimeError::InvalidInput {
            message: "Qwen LiveTranslate image frame exceeds the local 500,000-byte limit before Base64 encoding".into(),
        });
    }
    Ok(())
}

fn parse_native_frame(frame: &RealtimeFrame) -> Result<Value, RealtimeError> {
    let RealtimeFrame::Text(bytes) = frame else {
        return Err(RealtimeError::Codec {
            message: "Qwen LiveTranslate JSON/base64 protocol cannot decode binary frames".into(),
        });
    };
    serde_json::from_slice(bytes).map_err(|_| RealtimeError::Codec {
        message: "Qwen LiveTranslate event is not valid JSON".into(),
    })
}

fn event_type(native: &Value) -> Result<&str, RealtimeError> {
    native
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| RealtimeError::Codec {
            message: "Qwen LiveTranslate event is missing string `type`".into(),
        })
}

fn setup_error_message(native: &Value) -> String {
    let error = native.get("error").unwrap_or(&Value::Null);
    match error.get("code").and_then(Value::as_str) {
        Some(code) if code.len() <= 128 => {
            format!("Qwen LiveTranslate rejected session setup (code: {code})")
        }
        _ => "Qwen LiveTranslate rejected session setup".into(),
    }
}

fn same_model_identity(requested: QwenLiveTranslateModel, server_model: &str) -> bool {
    match server_model {
        "qwen3.5-livetranslate-flash-realtime"
        | "qwen3.5-livetranslate-flash-realtime-2026-05-19" => requested.is_qwen35(),
        "qwen3.8-livetranslate-flash-realtime" => {
            requested == QwenLiveTranslateModel::Qwen38LiveTranslateFlashRealtime
        }
        _ => false,
    }
}

fn parse_native_event(
    event_type: &str,
    native: &Value,
    model: QwenLiveTranslateModel,
    output_audio_format: &RealtimeAudioFormat,
) -> QwenLiveTranslateEvent {
    let string = |key: &str| native.get(key).and_then(Value::as_str).map(str::to_owned);
    let number = |key: &str| native.get(key).and_then(Value::as_u64);
    let qwen35 = model.is_qwen35();
    match event_type {
        "session.created" => QwenLiveTranslateEvent::SessionCreated {
            session: native.get("session").cloned().unwrap_or(Value::Null),
            native: native.clone(),
        },
        "session.updated" => QwenLiveTranslateEvent::SessionUpdated {
            session: native.get("session").cloned().unwrap_or(Value::Null),
            native: native.clone(),
        },
        "session.finished" => QwenLiveTranslateEvent::SessionFinished {
            native: native.clone(),
        },
        "input_audio_buffer.speech_started" => QwenLiveTranslateEvent::SpeechStarted {
            item_id: string("item_id"),
            audio_start_ms: number("audio_start_ms"),
            speaker_id: number("speaker_id"),
            native: native.clone(),
        },
        "input_audio_buffer.speech_stopped" => QwenLiveTranslateEvent::SpeechStopped {
            item_id: string("item_id"),
            audio_end_ms: number("audio_end_ms"),
            native: native.clone(),
        },
        "input_audio_buffer.committed" => QwenLiveTranslateEvent::InputAudioCommitted {
            native: native.clone(),
        },
        "input_audio_buffer.cleared" => QwenLiveTranslateEvent::InputAudioCleared {
            native: native.clone(),
        },
        "conversation.item.input_audio_transcription.text" if qwen35 => {
            QwenLiveTranslateEvent::SourceTranscription35 {
                item_id: string("item_id"),
                text: string("text").unwrap_or_default(),
                stash: string("stash").unwrap_or_default(),
                language: string("language"),
                emotion: string("emotion"),
                native: native.clone(),
            }
        }
        "conversation.item.input_audio_transcription.delta" if !qwen35 => {
            QwenLiveTranslateEvent::SourceTranscriptionDelta38 {
                item_id: string("item_id"),
                delta: string("delta").unwrap_or_default(),
                native: native.clone(),
            }
        }
        "conversation.item.input_audio_transcription.completed" => {
            QwenLiveTranslateEvent::SourceTranscriptionCompleted {
                item_id: string("item_id"),
                transcript: string("transcript").unwrap_or_default(),
                language: string("language"),
                emotion: string("emotion"),
                native: native.clone(),
            }
        }
        "conversation.item.input_audio_transcription.failed" => {
            let error = native.get("error").unwrap_or(&Value::Null);
            QwenLiveTranslateEvent::SourceTranscriptionFailed {
                item_id: string("item_id"),
                code: error.get("code").and_then(Value::as_str).map(str::to_owned),
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                native: native.clone(),
            }
        }
        "response.created" => QwenLiveTranslateEvent::ResponseCreated {
            response: native.get("response").cloned().unwrap_or(Value::Null),
            native: native.clone(),
        },
        "response.text.text" if qwen35 => QwenLiveTranslateEvent::TextProgress35 {
            response_id: string("response_id"),
            item_id: string("item_id"),
            text: string("text").unwrap_or_default(),
            stash: string("stash").unwrap_or_default(),
            native: native.clone(),
        },
        "response.text.delta" if !qwen35 => QwenLiveTranslateEvent::TextDelta38 {
            response_id: string("response_id"),
            item_id: string("item_id"),
            delta: string("delta").unwrap_or_default(),
            native: native.clone(),
        },
        "response.text.done" if qwen35 => QwenLiveTranslateEvent::TextDone {
            response_id: string("response_id"),
            item_id: string("item_id"),
            text: string("text").unwrap_or_default(),
            native: native.clone(),
        },
        "response.audio_transcript.text" if qwen35 => {
            QwenLiveTranslateEvent::AudioTranscriptProgress35 {
                response_id: string("response_id"),
                item_id: string("item_id"),
                text: string("text").unwrap_or_default(),
                stash: string("stash").unwrap_or_default(),
                native: native.clone(),
            }
        }
        "response.audio_transcript.delta" if !qwen35 => {
            QwenLiveTranslateEvent::AudioTranscriptDelta38 {
                response_id: string("response_id"),
                item_id: string("item_id"),
                delta: string("delta").unwrap_or_default(),
                native: native.clone(),
            }
        }
        "response.audio_transcript.done" => QwenLiveTranslateEvent::AudioTranscriptDone {
            response_id: string("response_id"),
            item_id: string("item_id"),
            transcript: string("transcript").unwrap_or_default(),
            native: native.clone(),
        },
        "response.audio.delta" => {
            let data = string("delta")
                .and_then(|delta| STANDARD.decode(delta).ok())
                .map(Bytes::from)
                .unwrap_or_default();
            QwenLiveTranslateEvent::AudioDelta {
                response_id: string("response_id"),
                item_id: string("item_id"),
                output_index: number("output_index"),
                content_index: number("content_index"),
                data,
                format: output_audio_format.clone(),
                native: native.clone(),
            }
        }
        "response.audio.done" => QwenLiveTranslateEvent::AudioDone {
            response_id: string("response_id"),
            item_id: string("item_id"),
            native: native.clone(),
        },
        "response.done" => QwenLiveTranslateEvent::ResponseDone {
            response: native.get("response").cloned().unwrap_or(Value::Null),
            native: native.clone(),
        },
        "error" => {
            let error = native.get("error").unwrap_or(&Value::Null);
            QwenLiveTranslateEvent::ProviderError {
                error_type: error.get("type").and_then(Value::as_str).map(str::to_owned),
                code: error.get("code").and_then(Value::as_str).map(str::to_owned),
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Qwen LiveTranslate provider error")
                    .to_owned(),
                param: error
                    .get("param")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                native: native.clone(),
            }
        }
        _ => QwenLiveTranslateEvent::Native {
            event_type: event_type.to_owned(),
            native: native.clone(),
        },
    }
}
