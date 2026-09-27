//! Qwen-Omni Realtime WebSocket adapter for Alibaba Cloud Model Studio.

use super::{
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
    fmt,
    sync::{Arc, Mutex},
};
use url::Url;

const MAX_SETUP_PREFACE_FRAMES: usize = 16;
const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;
const MAX_IMAGE_BASE64_BYTES: usize = 256 * 1024;
const INPUT_AUDIO_RATE_HZ: u32 = 16_000;
const OUTPUT_AUDIO_RATE_HZ: u32 = 24_000;

/// WebSocket path documented for Qwen-Omni Realtime.
pub const QWEN_REALTIME_WEBSOCKET_PATH: &str = "/api-ws/v1/realtime";

/// Model Studio region. Each region requires its own workspace and API key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenRealtimeRegion {
    Beijing,
    Singapore,
}

impl QwenRealtimeRegion {
    fn domain_suffix(self) -> &'static str {
        match self {
            Self::Beijing => "cn-beijing.maas.aliyuncs.com",
            Self::Singapore => "ap-southeast-1.maas.aliyuncs.com",
        }
    }
}

/// A workspace-bound regional Realtime endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenRealtimeRoute {
    region: QwenRealtimeRegion,
    workspace_id: String,
    endpoint: String,
}

impl QwenRealtimeRoute {
    pub fn new(
        region: QwenRealtimeRegion,
        workspace_id: impl Into<String>,
    ) -> Result<Self, RealtimeError> {
        let workspace_id = workspace_id.into();
        validate_workspace_id(&workspace_id)?;
        let endpoint = format!(
            "wss://{}.{}/{}",
            workspace_id,
            region.domain_suffix(),
            QWEN_REALTIME_WEBSOCKET_PATH.trim_start_matches('/')
        );
        let route = Self {
            region,
            workspace_id,
            endpoint,
        };
        route.validate()?;
        Ok(route)
    }

    pub fn region(&self) -> QwenRealtimeRegion {
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
            QWEN_REALTIME_WEBSOCKET_PATH.trim_start_matches('/')
        );
        if self.endpoint != expected {
            return Err(RealtimeError::InvalidConfig {
                message: "Qwen Realtime endpoint does not match its workspace and region".into(),
            });
        }
        Ok(())
    }

    fn endpoint_fingerprint(&self) -> String {
        provider_file_endpoint_fingerprint(&self.endpoint)
    }

    fn model_endpoint(&self, model: QwenRealtimeModel) -> Result<String, RealtimeError> {
        self.validate()?;
        let mut url = Url::parse(&self.endpoint).map_err(|_| RealtimeError::InvalidConfig {
            message: "Qwen Realtime endpoint is invalid".into(),
        })?;
        url.query_pairs_mut().append_pair("model", model.as_str());
        Ok(url.into())
    }
}

/// Non-secret identity bound to one Model Studio profile, account, workspace,
/// region, and WebSocket endpoint. Credentials are supplied per connection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenRealtimeScope {
    profile_name: String,
    account_scope: String,
    region: QwenRealtimeRegion,
    workspace_id: String,
    endpoint_fingerprint: String,
}

impl QwenRealtimeScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        route: &QwenRealtimeRoute,
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

    pub fn region(&self) -> QwenRealtimeRegion {
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
                message:
                    "Qwen Realtime scope requires profile, account, region, workspace, and endpoint"
                        .into(),
            });
        }
        validate_workspace_id(&self.workspace_id)
    }

    fn validate_route(&self, route: &QwenRealtimeRoute) -> Result<(), RealtimeError> {
        route.validate()?;
        self.validate()?;
        if self.region != route.region
            || self.workspace_id != route.workspace_id
            || self.endpoint_fingerprint != route.endpoint_fingerprint()
        {
            return Err(RealtimeError::InvalidConfig {
                message: "Qwen Realtime account scope belongs to a different region, workspace, or endpoint".into(),
            });
        }
        Ok(())
    }
}

/// Qwen-Omni Realtime models currently listed in the first-party WebSocket
/// connection guide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QwenRealtimeModel {
    #[serde(rename = "qwen3.8-omni-flash-realtime")]
    Qwen38OmniFlashRealtime,
    #[serde(rename = "qwen3.5-omni-plus-realtime")]
    Qwen35OmniPlusRealtime,
    #[serde(rename = "qwen3.5-omni-flash-realtime")]
    Qwen35OmniFlashRealtime,
}

impl QwenRealtimeModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qwen38OmniFlashRealtime => "qwen3.8-omni-flash-realtime",
            Self::Qwen35OmniPlusRealtime => "qwen3.5-omni-plus-realtime",
            Self::Qwen35OmniFlashRealtime => "qwen3.5-omni-flash-realtime",
        }
    }
}

/// Output modalities supported by the current Qwen Realtime contract: text
/// only, or text and audio together. Audio-only output is not accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenRealtimeOutputMode {
    Text,
    TextAndAudio,
}

/// Server-side, semantic, or host-controlled turn boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenRealtimeTurnDetection {
    ServerVad,
    SemanticVad,
    Manual,
}

/// Video-frame aggregation option available on Qwen3.8 Omni Realtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenRealtimeVideoMode {
    PreserveFrames,
    Compact,
}

/// Typed fields encoded in Qwen's `session.update` event.
#[derive(Clone)]
pub struct QwenRealtimeConfig {
    pub model: QwenRealtimeModel,
    pub instructions: Option<String>,
    pub output_mode: QwenRealtimeOutputMode,
    pub voice: Option<String>,
    pub turn_detection: QwenRealtimeTurnDetection,
    pub enable_input_audio_transcription: bool,
    pub video_mode: Option<QwenRealtimeVideoMode>,
}

impl QwenRealtimeConfig {
    pub fn new(model: QwenRealtimeModel) -> Self {
        Self {
            model,
            instructions: None,
            output_mode: QwenRealtimeOutputMode::TextAndAudio,
            voice: None,
            turn_detection: QwenRealtimeTurnDetection::ServerVad,
            enable_input_audio_transcription: true,
            video_mode: None,
        }
    }
}

impl Default for QwenRealtimeConfig {
    fn default() -> Self {
        Self::new(QwenRealtimeModel::Qwen38OmniFlashRealtime)
    }
}

impl fmt::Debug for QwenRealtimeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenRealtimeConfig")
            .field("model", &self.model)
            .field("instructions", &self.instructions.as_ref().map(|_| "<set>"))
            .field("output_mode", &self.output_mode)
            .field("voice", &self.voice)
            .field("turn_detection", &self.turn_detection)
            .field(
                "enable_input_audio_transcription",
                &self.enable_input_audio_transcription,
            )
            .field("video_mode", &self.video_mode)
            .finish()
    }
}

/// Typed Qwen Realtime events. Unknown server messages retain their original
/// JSON event name and payload.
#[derive(Debug, Clone, PartialEq)]
pub enum QwenRealtimeEvent {
    Realtime(RealtimeEvent),
    SessionCreated {
        session: Value,
        native: Value,
    },
    SessionUpdated {
        session: Value,
        native: Value,
    },
    SpeechStarted {
        item_id: Option<String>,
        audio_start_ms: Option<u64>,
        native: Value,
    },
    SpeechStopped {
        item_id: Option<String>,
        audio_end_ms: Option<u64>,
        native: Value,
    },
    InputAudioCommitted {
        item_id: Option<String>,
        native: Value,
    },
    InputTranscriptionDelta {
        item_id: Option<String>,
        text: String,
        stash: String,
        language: Option<String>,
        emotion: Option<String>,
        native: Value,
    },
    InputTranscriptionCompleted {
        item_id: Option<String>,
        transcript: String,
        native: Value,
    },
    InputTranscriptionFailed {
        item_id: Option<String>,
        code: Option<String>,
        message: Option<String>,
        native: Value,
    },
    ResponseCreated {
        response_id: Option<String>,
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
    AudioTranscriptDelta {
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
    TextDelta {
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
    ResponseDone {
        response_id: Option<String>,
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

/// Connected Qwen Realtime session. Setup completes only after
/// `session.updated` acknowledges the submitted configuration.
pub struct QwenRealtimeSession {
    control: QwenRealtimeControl,
    events: QwenRealtimeEvents,
    scope: QwenRealtimeScope,
}

impl QwenRealtimeSession {
    /// Connect to the route's workspace endpoint with one per-call API key.
    pub async fn connect(
        transport: Arc<dyn RealtimeTransport>,
        route: QwenRealtimeRoute,
        scope: QwenRealtimeScope,
        credential: Secret<String>,
        config: QwenRealtimeConfig,
        limits: RealtimeLimits,
    ) -> Result<(Self, RealtimeDriver), RealtimeError> {
        validate_limits(limits)?;
        route.validate()?;
        scope.validate_route(&route)?;
        validate_credential(&credential)?;
        validate_config(&config)?;

        let setup = build_session_update(&config)?;
        let setup_bytes = serde_json::to_vec(&setup).map_err(|_| RealtimeError::Codec {
            message: "could not encode Qwen Realtime session update".into(),
        })?;
        if setup_bytes.len() > limits.max_frame_bytes {
            return Err(RealtimeError::FrameTooLarge {
                actual: setup_bytes.len(),
                max: limits.max_frame_bytes,
            });
        }

        let request = RealtimeConnectRequest {
            endpoint: route.model_endpoint(config.model)?,
            headers: vec![(
                "Authorization".into(),
                format!("Bearer {}", credential.expose_secret()),
            )],
            max_frame_bytes: limits.max_frame_bytes,
        };
        let setup_transport = QwenSetupTransport {
            inner: transport,
            setup_frame: RealtimeFrame::Text(Bytes::from(setup_bytes)),
            max_frame_bytes: limits.max_frame_bytes,
        };
        let codec = Arc::new(QwenRealtimeCodec {
            turn_detection: config.turn_detection,
        });
        let (session, driver) =
            RealtimeSession::connect(&setup_transport, request, codec, limits).await?;
        let (control, events) = session.into_parts();
        Ok((
            Self {
                control: QwenRealtimeControl {
                    inner: control,
                    audio_appended: Arc::new(Mutex::new(false)),
                },
                events: QwenRealtimeEvents {
                    inner: events,
                    output_audio_format: RealtimeAudioFormat::Pcm16 {
                        sample_rate_hz: OUTPUT_AUDIO_RATE_HZ,
                    },
                },
                scope,
            },
            driver,
        ))
    }

    pub fn into_parts(self) -> (QwenRealtimeControl, QwenRealtimeEvents) {
        (self.control, self.events)
    }

    pub fn scope(&self) -> &QwenRealtimeScope {
        &self.scope
    }
}

/// Qwen session sender that enforces the documented audio-before-image rule.
#[derive(Clone)]
pub struct QwenRealtimeControl {
    inner: RealtimeControl,
    audio_appended: Arc<Mutex<bool>>,
}

impl QwenRealtimeControl {
    pub fn send(&self, input: RealtimeInput) -> Result<(), RealtimeError> {
        let mut audio_appended = self.audio_appended.lock().unwrap();
        if matches!(&input, RealtimeInput::Image { .. }) && !*audio_appended {
            return Err(RealtimeError::InvalidInput {
                message: "Qwen Realtime requires an audio append before an image frame".into(),
            });
        }
        let is_audio = matches!(&input, RealtimeInput::Audio { .. });
        self.inner.send(input)?;
        if is_audio {
            *audio_appended = true;
        }
        Ok(())
    }

    pub fn interrupt(&self) -> Result<(), RealtimeError> {
        self.inner.interrupt()
    }

    pub async fn close(&self, close: super::RealtimeClose) -> Result<(), RealtimeError> {
        self.inner.close(close).await
    }
}

/// Receiver that converts Qwen's native JSON events into typed outputs.
pub struct QwenRealtimeEvents {
    inner: super::RealtimeEvents,
    output_audio_format: RealtimeAudioFormat,
}

impl QwenRealtimeEvents {
    pub async fn next(&mut self) -> Option<QwenRealtimeEvent> {
        let event = self.inner.next().await?;
        let RealtimeEvent::ProviderEvent { name, native } = &event else {
            return Some(QwenRealtimeEvent::Realtime(event));
        };
        Some(parse_native_event(name, native, &self.output_audio_format))
    }
}

struct QwenSetupTransport {
    inner: Arc<dyn RealtimeTransport>,
    setup_frame: RealtimeFrame,
    max_frame_bytes: usize,
}

#[async_trait]
impl RealtimeTransport for QwenSetupTransport {
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
                message: "Qwen Realtime did not acknowledge session.update within the setup limit"
                    .into(),
            });
        }
        connection.inbound = stream::iter(preface).chain(connection.inbound).boxed();
        Ok(connection)
    }
}

struct QwenRealtimeCodec {
    turn_detection: QwenRealtimeTurnDetection,
}

impl RealtimeCodec for QwenRealtimeCodec {
    fn encode(&self, input: &RealtimeInput) -> Result<Vec<RealtimeFrame>, RealtimeError> {
        let messages = match input {
            RealtimeInput::RetrieveItem { .. }
            | RealtimeInput::DeleteItem { .. }
            | RealtimeInput::TruncateAudio { .. } => {
                return Err(RealtimeError::InvalidInput {
                    message: "conversation item retrieval, deletion and truncation are not supported by this adapter".into(),
                });
            }
            RealtimeInput::ClearAudio => {
                vec![json!({ "type": "input_audio_buffer.clear" })]
            }
            RealtimeInput::FinishSession => {
                return Err(RealtimeError::InvalidInput {
                    message: "explicit session finishing is not implemented by this adapter".into(),
                });
            }
            RealtimeInput::Text(text) => {
                validate_text(text)?;
                vec![
                    json!({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{ "type": "input_text", "text": text }]
                        }
                    }),
                    json!({ "type": "response.create" }),
                ]
            }
            RealtimeInput::Audio { data, format } => {
                validate_input_audio(data, format)?;
                vec![json!({
                    "type": "input_audio_buffer.append",
                    "audio": STANDARD.encode(data)
                })]
            }
            RealtimeInput::Image { data, mime_type } => {
                validate_image(data, mime_type)?;
                vec![json!({
                    "type": "input_image_buffer.append",
                    "image": STANDARD.encode(data)
                })]
            }
            RealtimeInput::CommitAudio
                if self.turn_detection == QwenRealtimeTurnDetection::Manual =>
            {
                vec![json!({ "type": "input_audio_buffer.commit" })]
            }
            RealtimeInput::CommitAudio => {
                return Err(RealtimeError::InvalidInput {
                    message: "Qwen Realtime audio commit is only valid in manual mode".into(),
                })
            }
            RealtimeInput::ContinueResponse
                if self.turn_detection == QwenRealtimeTurnDetection::Manual =>
            {
                vec![json!({ "type": "response.create" })]
            }
            RealtimeInput::ContinueResponse => {
                return Err(RealtimeError::InvalidInput {
                    message: "Qwen Realtime VAD mode creates responses automatically".into(),
                })
            }
            RealtimeInput::Interrupt => vec![json!({ "type": "response.cancel" })],
            RealtimeInput::ToolResult { .. } | RealtimeInput::ToolResults { .. } => {
                return Err(RealtimeError::InvalidInput {
                    message: "Qwen Realtime function result submission is outside this audio/video/text adapter".into(),
                })
            }
        };
        messages
            .into_iter()
            .map(|message| {
                serde_json::to_vec(&message)
                    .map(|bytes| RealtimeFrame::Text(Bytes::from(bytes)))
                    .map_err(|_| RealtimeError::Codec {
                        message: "could not encode Qwen Realtime client event".into(),
                    })
            })
            .collect()
    }

    fn decode(&self, frame: RealtimeFrame) -> Result<Vec<RealtimeEvent>, RealtimeError> {
        let native = parse_native_frame(&frame)?;
        let name = event_type(&native)?;
        if name == "response.audio.delta" {
            let audio = native.get("delta").and_then(Value::as_str).ok_or_else(|| {
                RealtimeError::Codec {
                    message: "Qwen Realtime audio delta is missing base64 `delta`".into(),
                }
            })?;
            STANDARD.decode(audio).map_err(|_| RealtimeError::Codec {
                message: "Qwen Realtime audio delta is not valid base64".into(),
            })?;
        }
        Ok(vec![RealtimeEvent::ProviderEvent {
            name: name.to_owned(),
            native,
        }])
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
            message: "Qwen Realtime workspace ID must be one valid DNS label".into(),
        });
    }
    Ok(())
}

fn validate_config(config: &QwenRealtimeConfig) -> Result<(), RealtimeError> {
    if config
        .instructions
        .as_ref()
        .is_some_and(|value| value.contains('\0') || value.len() > 64 * 1024)
    {
        return Err(RealtimeError::InvalidConfig {
            message: "Qwen Realtime instructions must be NUL-free and at most 64 KiB".into(),
        });
    }
    if config
        .voice
        .as_ref()
        .is_some_and(|value| value.trim().is_empty() || value.len() > 256)
    {
        return Err(RealtimeError::InvalidConfig {
            message: "Qwen Realtime voice must be non-empty and at most 256 bytes".into(),
        });
    }
    if config.video_mode.is_some() && config.model != QwenRealtimeModel::Qwen38OmniFlashRealtime {
        return Err(RealtimeError::InvalidConfig {
            message:
                "Qwen Realtime video aggregation is documented only for Qwen3.8 Omni Flash Realtime"
                    .into(),
        });
    }
    Ok(())
}

fn build_session_update(config: &QwenRealtimeConfig) -> Result<Value, RealtimeError> {
    let modalities = match config.output_mode {
        QwenRealtimeOutputMode::Text => json!(["text"]),
        QwenRealtimeOutputMode::TextAndAudio => json!(["text", "audio"]),
    };
    let turn_detection = match config.turn_detection {
        QwenRealtimeTurnDetection::ServerVad => json!({ "type": "server_vad" }),
        QwenRealtimeTurnDetection::SemanticVad => json!({ "type": "semantic_vad" }),
        QwenRealtimeTurnDetection::Manual => Value::Null,
    };
    let mut session = serde_json::Map::new();
    session.insert("model".into(), Value::String(config.model.as_str().into()));
    session.insert("modalities".into(), modalities);
    session.insert("turn_detection".into(), turn_detection);
    session.insert(
        "enable_input_audio_transcription".into(),
        Value::Bool(config.enable_input_audio_transcription),
    );
    if let Some(instructions) = &config.instructions {
        session.insert("instructions".into(), Value::String(instructions.clone()));
    }
    if let Some(voice) = &config.voice {
        if config.model == QwenRealtimeModel::Qwen38OmniFlashRealtime {
            session.insert("audio".into(), json!({ "output": { "voice": voice } }));
        } else {
            session.insert("voice".into(), Value::String(voice.clone()));
        }
    }
    if let Some(video_mode) = config.video_mode {
        let compact = match video_mode {
            QwenRealtimeVideoMode::PreserveFrames => "none",
            QwenRealtimeVideoMode::Compact => "normal",
        };
        session.insert(
            "video".into(),
            json!({ "input": { "representation_compact": compact } }),
        );
    }
    Ok(json!({ "type": "session.update", "session": session }))
}

fn validate_input_audio(data: &Bytes, format: &RealtimeAudioFormat) -> Result<(), RealtimeError> {
    if data.is_empty() {
        return Err(RealtimeError::InvalidInput {
            message: "Qwen Realtime audio input must not be empty".into(),
        });
    }
    if format
        != &(RealtimeAudioFormat::Pcm16 {
            sample_rate_hz: INPUT_AUDIO_RATE_HZ,
        })
    {
        return Err(RealtimeError::InvalidInput {
            message: "Qwen Realtime accepts 16 kHz PCM16 audio input in this adapter".into(),
        });
    }
    Ok(())
}

fn validate_image(data: &Bytes, mime_type: &str) -> Result<(), RealtimeError> {
    if mime_type != "image/jpeg" || data.len() < 2 || !data.starts_with(&[0xff, 0xd8]) {
        return Err(RealtimeError::InvalidInput {
            message: "Qwen Realtime video frames must contain JPEG data with image/jpeg MIME"
                .into(),
        });
    }
    let encoded_size = data.len().div_ceil(3).saturating_mul(4);
    if encoded_size > MAX_IMAGE_BASE64_BYTES {
        return Err(RealtimeError::InvalidInput {
            message: "Qwen Realtime JPEG frame exceeds the 256 KiB Base64 limit".into(),
        });
    }
    Ok(())
}

fn validate_text(text: &str) -> Result<(), RealtimeError> {
    if text.trim().is_empty() || text.contains('\0') {
        return Err(RealtimeError::InvalidInput {
            message: "Qwen Realtime text input must be non-empty and NUL-free".into(),
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
            message: "Qwen Realtime bearer credential is empty or invalid".into(),
        });
    }
    Ok(())
}

fn parse_native_frame(frame: &RealtimeFrame) -> Result<Value, RealtimeError> {
    let RealtimeFrame::Text(bytes) = frame else {
        return Err(RealtimeError::Codec {
            message: "Qwen Realtime JSON/base64 protocol cannot decode binary frames".into(),
        });
    };
    serde_json::from_slice(bytes).map_err(|_| RealtimeError::Codec {
        message: "Qwen Realtime event is not valid JSON".into(),
    })
}

fn event_type(native: &Value) -> Result<&str, RealtimeError> {
    native
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| RealtimeError::Codec {
            message: "Qwen Realtime event is missing string `type`".into(),
        })
}

fn setup_error_message(native: &Value) -> String {
    let error = native.get("error").unwrap_or(&Value::Null);
    let code = error.get("code").and_then(Value::as_str);
    match code {
        Some(code) if code.len() <= 128 => {
            format!("Qwen Realtime rejected session setup (code: {code})")
        }
        _ => "Qwen Realtime rejected session setup".into(),
    }
}

fn parse_native_event(
    event_type: &str,
    native: &Value,
    output_audio_format: &RealtimeAudioFormat,
) -> QwenRealtimeEvent {
    let string = |key: &str| native.get(key).and_then(Value::as_str).map(str::to_owned);
    let number = |key: &str| native.get(key).and_then(Value::as_u64);
    let nested_string = |outer: &str, key: &str| {
        native
            .get(outer)
            .and_then(|value| value.get(key))
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    match event_type {
        "session.created" => QwenRealtimeEvent::SessionCreated {
            session: native.get("session").cloned().unwrap_or(Value::Null),
            native: native.clone(),
        },
        "session.updated" => QwenRealtimeEvent::SessionUpdated {
            session: native.get("session").cloned().unwrap_or(Value::Null),
            native: native.clone(),
        },
        "input_audio_buffer.speech_started" => QwenRealtimeEvent::SpeechStarted {
            item_id: string("item_id"),
            audio_start_ms: number("audio_start_ms"),
            native: native.clone(),
        },
        "input_audio_buffer.speech_stopped" => QwenRealtimeEvent::SpeechStopped {
            item_id: string("item_id"),
            audio_end_ms: number("audio_end_ms"),
            native: native.clone(),
        },
        "input_audio_buffer.committed" => QwenRealtimeEvent::InputAudioCommitted {
            item_id: string("item_id"),
            native: native.clone(),
        },
        "conversation.item.input_audio_transcription.delta" => {
            QwenRealtimeEvent::InputTranscriptionDelta {
                item_id: string("item_id"),
                text: string("text").unwrap_or_default(),
                stash: string("stash").unwrap_or_default(),
                language: string("language"),
                emotion: string("emotion"),
                native: native.clone(),
            }
        }
        "conversation.item.input_audio_transcription.completed" => {
            QwenRealtimeEvent::InputTranscriptionCompleted {
                item_id: string("item_id"),
                transcript: string("transcript").unwrap_or_default(),
                native: native.clone(),
            }
        }
        "conversation.item.input_audio_transcription.failed" => {
            let error = native.get("error").unwrap_or(&Value::Null);
            QwenRealtimeEvent::InputTranscriptionFailed {
                item_id: string("item_id"),
                code: error.get("code").and_then(Value::as_str).map(str::to_owned),
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                native: native.clone(),
            }
        }
        "response.created" => QwenRealtimeEvent::ResponseCreated {
            response_id: nested_string("response", "id"),
            native: native.clone(),
        },
        "response.audio.delta" => {
            let data = string("delta")
                .and_then(|delta| STANDARD.decode(delta).ok())
                .map(Bytes::from)
                .unwrap_or_default();
            QwenRealtimeEvent::AudioDelta {
                response_id: string("response_id"),
                item_id: string("item_id"),
                output_index: number("output_index"),
                content_index: number("content_index"),
                data,
                format: output_audio_format.clone(),
                native: native.clone(),
            }
        }
        "response.audio.done" => QwenRealtimeEvent::AudioDone {
            response_id: string("response_id"),
            item_id: string("item_id"),
            native: native.clone(),
        },
        "response.audio_transcript.delta" => QwenRealtimeEvent::AudioTranscriptDelta {
            response_id: string("response_id"),
            item_id: string("item_id"),
            delta: string("delta").unwrap_or_default(),
            native: native.clone(),
        },
        "response.audio_transcript.done" => QwenRealtimeEvent::AudioTranscriptDone {
            response_id: string("response_id"),
            item_id: string("item_id"),
            transcript: string("transcript").unwrap_or_default(),
            native: native.clone(),
        },
        "response.text.delta" => QwenRealtimeEvent::TextDelta {
            response_id: string("response_id"),
            item_id: string("item_id"),
            delta: string("delta").unwrap_or_default(),
            native: native.clone(),
        },
        "response.text.done" => QwenRealtimeEvent::TextDone {
            response_id: string("response_id"),
            item_id: string("item_id"),
            text: string("text").unwrap_or_default(),
            native: native.clone(),
        },
        "response.done" => {
            let response = native.get("response").cloned().unwrap_or(Value::Null);
            QwenRealtimeEvent::ResponseDone {
                response_id: response
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                response,
                native: native.clone(),
            }
        }
        "error" => {
            let error = native.get("error").unwrap_or(&Value::Null);
            QwenRealtimeEvent::ProviderError {
                error_type: error.get("type").and_then(Value::as_str).map(str::to_owned),
                code: error.get("code").and_then(Value::as_str).map(str::to_owned),
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Qwen Realtime provider error")
                    .to_owned(),
                param: error
                    .get("param")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                native: native.clone(),
            }
        }
        _ => QwenRealtimeEvent::Native {
            event_type: event_type.to_owned(),
            native: native.clone(),
        },
    }
}
