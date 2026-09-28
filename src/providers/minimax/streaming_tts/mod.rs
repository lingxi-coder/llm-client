//! MiniMax's synchronous Text-to-Audio WebSocket protocol.
//!
//! This is an ordinary text-to-audio task stream, not the separate
//! speech-to-speech or bidirectional TTS protocols. It uses the official
//! international and mainland WSS routes selected explicitly by region.

use crate::{
    files::provider_file_endpoint_fingerprint,
    protocol::{ProviderId, Secret},
    providers::minimax::tts::MiniMaxTtsRegion,
    providers::minimax::voices::{MiniMaxVoiceLanguageBoost, MiniMaxVoiceRef},
    realtime::{
        RealtimeClose, RealtimeConnectRequest, RealtimeConnection, RealtimeError, RealtimeFrame,
        RealtimeSink, RealtimeTransport,
    },
};
use bytes::Bytes;
use futures::{lock::Mutex as AsyncMutex, stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fmt, sync::Arc};
use thiserror::Error;

/// Official international MiniMax T2A WebSocket route.
pub const MINIMAX_STREAMING_TTS_INTERNATIONAL_ENDPOINT: &str = "wss://api.minimax.io/ws/v1/t2a_v2";
/// Official mainland China MiniMax T2A WebSocket route.
pub const MINIMAX_STREAMING_TTS_CHINA_ENDPOINT: &str = "wss://api.minimax.cn/ws/v1/t2a_v2";

const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;
const MAX_SETUP_EVENTS: usize = 16;
const HARD_MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;
const HARD_MAX_TEXT_BYTES: usize = 256 * 1024;
const HARD_MAX_PENDING_SEGMENTS: usize = 64;
const HARD_MAX_AUDIO_BYTES: usize = 512 * 1024 * 1024;

/// Secret-free account and route selection for one streaming TTS service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxStreamingTtsConfig {
    pub profile_name: String,
    pub account_scope: String,
    pub region: MiniMaxTtsRegion,
    /// The service accepts only the exact first-party WSS endpoint for `region`.
    pub endpoint: String,
}

impl MiniMaxStreamingTtsConfig {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: MiniMaxTtsRegion,
    ) -> Self {
        let endpoint = match region {
            MiniMaxTtsRegion::International => MINIMAX_STREAMING_TTS_INTERNATIONAL_ENDPOINT,
            MiniMaxTtsRegion::ChinaMainland => MINIMAX_STREAMING_TTS_CHINA_ENDPOINT,
        };
        Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region,
            endpoint: endpoint.into(),
        }
    }

    /// Set an endpoint for validation or contract-test setup. `new` is the
    /// production route constructor; `MiniMaxStreamingTtsService::new` rejects
    /// any value other than the documented WSS URL for the selected region.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}

/// Non-secret provenance attached to a live stream and each returned event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxStreamingTtsScope {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub endpoint_fingerprint: String,
    pub account_scope: String,
    pub region: MiniMaxTtsRegion,
}

/// The model IDs listed by both ordinary MiniMax T2A WebSocket references.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MiniMaxStreamingTtsModel {
    #[default]
    #[serde(rename = "speech-2.8-hd")]
    Speech28Hd,
    #[serde(rename = "speech-2.8-turbo")]
    Speech28Turbo,
    #[serde(rename = "speech-2.6-hd")]
    Speech26Hd,
    #[serde(rename = "speech-2.6-turbo")]
    Speech26Turbo,
    #[serde(rename = "speech-02-hd")]
    Speech02Hd,
    #[serde(rename = "speech-02-turbo")]
    Speech02Turbo,
    #[serde(rename = "speech-01-hd")]
    Speech01Hd,
    #[serde(rename = "speech-01-turbo")]
    Speech01Turbo,
}

impl MiniMaxStreamingTtsModel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Speech28Hd => "speech-2.8-hd",
            Self::Speech28Turbo => "speech-2.8-turbo",
            Self::Speech26Hd => "speech-2.6-hd",
            Self::Speech26Turbo => "speech-2.6-turbo",
            Self::Speech02Hd => "speech-02-hd",
            Self::Speech02Turbo => "speech-02-turbo",
            Self::Speech01Hd => "speech-01-hd",
            Self::Speech01Turbo => "speech-01-turbo",
        }
    }

    const fn supports_continuous_sound(self) -> bool {
        matches!(self, Self::Speech28Hd | Self::Speech28Turbo)
    }

    const fn supports_fluent_whisper_emotion(self) -> bool {
        matches!(self, Self::Speech26Hd | Self::Speech26Turbo)
    }
}

/// Emotion values documented by MiniMax's ordinary T2A WebSocket reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxStreamingTtsEmotion {
    Happy,
    Sad,
    #[serde(rename = "angry")]
    Angry,
    Fearful,
    Disgusted,
    Surprised,
    Calm,
    Fluent,
    Whisper,
}

/// The documented T2A `language_boost` values, also used by MiniMax voice APIs.
pub type MiniMaxStreamingTtsLanguageBoost = MiniMaxVoiceLanguageBoost;

/// Voice controls in MiniMax's WebSocket `voice_setting` object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxStreamingTtsVoiceSetting {
    pub voice_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vol: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch: Option<i8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emotion: Option<MiniMaxStreamingTtsEmotion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub english_normalization: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latex_read: Option<bool>,
}

impl MiniMaxStreamingTtsVoiceSetting {
    pub fn new(voice_id: impl Into<String>) -> Self {
        Self {
            voice_id: voice_id.into(),
            speed: None,
            vol: None,
            pitch: None,
            emotion: None,
            english_normalization: None,
            latex_read: None,
        }
    }
}

/// One voice and its mix weight for `timbre_weights`. MiniMax supports up to
/// four entries; each weight is an integer from 1 through 100.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxStreamingTtsTimbreWeight {
    pub voice_id: String,
    pub weight: u8,
}

/// Sound-effect values accepted by `voice_modify.sound_effects`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxStreamingTtsSoundEffect {
    SpaciousEcho,
    AuditoriumEcho,
    LofiTelephone,
    Robotic,
}

/// Optional voice effects. This streaming T2A route documents these effects
/// for MP3 output only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxStreamingTtsVoiceModify {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch: Option<i16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intensity: Option<i16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timbre: Option<i16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sound_effects: Option<MiniMaxStreamingTtsSoundEffect>,
}

impl MiniMaxStreamingTtsVoiceModify {
    fn is_empty(&self) -> bool {
        self.pitch.is_none()
            && self.intensity.is_none()
            && self.timbre.is_none()
            && self.sound_effects.is_none()
    }
}

/// Subtitle granularity documented by the ordinary T2A WebSocket routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxStreamingTtsSubtitleType {
    Sentence,
    Word,
    WordStreaming,
}

/// Audio controls from MiniMax's ordinary T2A WebSocket `audio_setting` object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxStreamingTtsAudioSetting {
    pub sample_rate: u32,
    pub bitrate: u32,
    pub format: MiniMaxStreamingTtsAudioFormat,
    pub channel: u8,
}

impl Default for MiniMaxStreamingTtsAudioSetting {
    fn default() -> Self {
        Self {
            sample_rate: 32_000,
            bitrate: 128_000,
            format: MiniMaxStreamingTtsAudioFormat::Mp3,
            channel: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxStreamingTtsAudioFormat {
    #[default]
    Mp3,
    Wav,
    Flac,
    Pcm,
    PcmuRaw,
    PcmuWav,
    Opus,
}

/// Parameters sent once in the provider's `task_start` event. Text is sent
/// separately, in bounded `task_continue` messages.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxStreamingTtsRequest {
    pub model: MiniMaxStreamingTtsModel,
    pub voice_setting: MiniMaxStreamingTtsVoiceSetting,
    pub audio_setting: MiniMaxStreamingTtsAudioSetting,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language_boost: Option<MiniMaxStreamingTtsLanguageBoost>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pronunciation_dict: Option<crate::providers::minimax::tts::MiniMaxTtsPronunciationDict>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timbre_weights: Option<Vec<MiniMaxStreamingTtsTimbreWeight>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_modify: Option<MiniMaxStreamingTtsVoiceModify>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle_enable: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle_type: Option<MiniMaxStreamingTtsSubtitleType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuous_sound: Option<bool>,
}

impl fmt::Debug for MiniMaxStreamingTtsRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxStreamingTtsRequest")
            .field("model", &self.model)
            .field("voice_setting", &self.voice_setting)
            .field("audio_setting", &self.audio_setting)
            .field(
                "language_boost",
                &self.language_boost.as_ref().map(|_| "<set>"),
            )
            .field(
                "pronunciation_dict",
                &self.pronunciation_dict.as_ref().map(|_| "<set>"),
            )
            .field(
                "timbre_weights",
                &self.timbre_weights.as_ref().map(|_| "<set>"),
            )
            .field("voice_modify", &self.voice_modify.as_ref().map(|_| "<set>"))
            .field("subtitle_enable", &self.subtitle_enable)
            .field("subtitle_type", &self.subtitle_type)
            .field("continuous_sound", &self.continuous_sound)
            .finish()
    }
}

impl MiniMaxStreamingTtsRequest {
    pub fn new(voice_id: impl Into<String>) -> Self {
        Self {
            model: MiniMaxStreamingTtsModel::default(),
            voice_setting: MiniMaxStreamingTtsVoiceSetting::new(voice_id),
            audio_setting: MiniMaxStreamingTtsAudioSetting::default(),
            language_boost: None,
            pronunciation_dict: None,
            timbre_weights: None,
            voice_modify: None,
            subtitle_enable: None,
            subtitle_type: None,
            continuous_sound: None,
        }
    }

    pub(crate) fn validate_for_region(
        &self,
        region: MiniMaxTtsRegion,
    ) -> Result<(), MiniMaxStreamingTtsError> {
        self.validate_common()?;
        // The currently documented international and mainland ordinary WSS
        // schemas expose the same model IDs and task_start fields.
        match region {
            MiniMaxTtsRegion::International | MiniMaxTtsRegion::ChinaMainland => Ok(()),
        }
    }

    /// Encode the common, documented task_start fields for ordinary T2A and
    /// Bidi T2A. This performs structural/model-combination validation only;
    /// ordinary endpoint route checks belong to the ordinary service.
    pub(crate) fn start_event(&self) -> Result<Value, MiniMaxStreamingTtsError> {
        self.validate_common()?;

        let voice_setting = serde_json::to_value(&self.voice_setting)
            .map_err(|_| invalid("voice_setting could not be encoded"))?;
        let mut audio_setting = serde_json::to_value(self.audio_setting)
            .map_err(|_| invalid("audio_setting could not be encoded"))?;
        if self.audio_setting.format != MiniMaxStreamingTtsAudioFormat::Mp3 {
            if let Some(audio_setting) = audio_setting.as_object_mut() {
                audio_setting.remove("bitrate");
            }
        }
        let mut event = json!({
            "event": "task_start",
            "model": self.model.as_str(),
            "voice_setting": voice_setting,
            "audio_setting": audio_setting,
        });
        if let Some(pronunciation_dict) = &self.pronunciation_dict {
            event["pronunciation_dict"] = serde_json::to_value(pronunciation_dict)
                .map_err(|_| invalid("pronunciation_dict could not be encoded"))?;
        }
        if let Some(timbre_weights) = &self.timbre_weights {
            event["timbre_weights"] = serde_json::to_value(timbre_weights)
                .map_err(|_| invalid("timbre_weights could not be encoded"))?;
        }
        if let Some(voice_modify) = &self.voice_modify {
            if !voice_modify.is_empty() {
                event["voice_modify"] = serde_json::to_value(voice_modify)
                    .map_err(|_| invalid("voice_modify could not be encoded"))?;
            }
        }
        if let Some(subtitle_enable) = self.subtitle_enable {
            event["subtitle_enable"] = json!(subtitle_enable);
        }
        if let Some(subtitle_type) = self.subtitle_type {
            event["subtitle_type"] = json!(subtitle_type);
        }
        if let Some(continuous_sound) = self.continuous_sound {
            event["continuous_sound"] = json!(continuous_sound);
        }
        if let Some(language_boost) = self.language_boost {
            event["language_boost"] = serde_json::to_value(language_boost)
                .map_err(|_| invalid("language_boost could not be encoded"))?;
        }
        Ok(event)
    }

    fn validate_common(&self) -> Result<(), MiniMaxStreamingTtsError> {
        let mixed_voices = self
            .timbre_weights
            .as_ref()
            .is_some_and(|weights| !weights.is_empty());
        if self.timbre_weights.as_ref().is_some_and(Vec::is_empty) {
            return Err(invalid("timbre_weights must contain at least one voice"));
        }
        if mixed_voices {
            if !self.voice_setting.voice_id.is_empty() {
                return Err(invalid(
                    "voice_setting.voice_id must be empty when timbre_weights are provided",
                ));
            }
        } else if self.voice_setting.voice_id.trim().is_empty() {
            return Err(invalid("voice_id must not be empty unless mixing voices"));
        }
        if self.voice_setting.voice_id.chars().any(char::is_control) {
            return Err(invalid("voice_id must not contain control characters"));
        }
        if self
            .voice_setting
            .speed
            .is_some_and(|value| !value.is_finite() || !(0.5..=2.0).contains(&value))
        {
            return Err(invalid("voice speed must be between 0.5 and 2.0"));
        }
        if self
            .voice_setting
            .vol
            .is_some_and(|value| !value.is_finite() || value <= 0.0 || value > 10.0)
        {
            return Err(invalid(
                "voice volume must be greater than 0 and at most 10",
            ));
        }
        if self
            .voice_setting
            .pitch
            .is_some_and(|value| !(-12..=12).contains(&value))
        {
            return Err(invalid("voice pitch must be between -12 and 12"));
        }
        if self.voice_setting.emotion.is_some_and(|emotion| {
            matches!(
                emotion,
                MiniMaxStreamingTtsEmotion::Fluent | MiniMaxStreamingTtsEmotion::Whisper
            ) && !self.model.supports_fluent_whisper_emotion()
        }) {
            return Err(invalid(
                "fluent and whisper emotions are only supported by speech-2.6-hd and speech-2.6-turbo",
            ));
        }
        if self.voice_setting.latex_read == Some(true)
            && self
                .language_boost
                .is_some_and(|language| language != MiniMaxStreamingTtsLanguageBoost::Chinese)
        {
            return Err(invalid("latex_read requires Chinese language_boost"));
        }

        let audio = self.audio_setting;
        if !matches!(
            audio.sample_rate,
            8_000 | 16_000 | 22_050 | 24_000 | 32_000 | 44_100
        ) {
            return Err(invalid("audio sample rate is not documented by MiniMax"));
        }
        if !matches!(audio.bitrate, 32_000 | 64_000 | 128_000 | 256_000) {
            return Err(invalid("audio bitrate is not documented by MiniMax"));
        }
        if !matches!(audio.channel, 1 | 2) {
            return Err(invalid("audio channel count must be 1 or 2"));
        }
        if matches!(
            audio.format,
            MiniMaxStreamingTtsAudioFormat::PcmuRaw | MiniMaxStreamingTtsAudioFormat::PcmuWav
        ) && audio.sample_rate != 8_000
        {
            return Err(invalid(
                "pcmu_raw and pcmu_wav require an 8 kHz sample rate",
            ));
        }
        if self
            .voice_modify
            .as_ref()
            .is_some_and(|modify| !modify.is_empty())
        {
            if audio.format != MiniMaxStreamingTtsAudioFormat::Mp3 {
                return Err(invalid("voice_modify is only documented for streaming MP3"));
            }
            if self.voice_modify.as_ref().is_some_and(|modify| {
                modify
                    .pitch
                    .is_some_and(|value| !(-100..=100).contains(&value))
                    || modify
                        .intensity
                        .is_some_and(|value| !(-100..=100).contains(&value))
                    || modify
                        .timbre
                        .is_some_and(|value| !(-100..=100).contains(&value))
            }) {
                return Err(invalid("voice_modify values must be between -100 and 100"));
            }
        }
        if self.continuous_sound.is_some() && !self.model.supports_continuous_sound() {
            return Err(invalid(
                "continuous_sound is only supported by speech-2.8-hd and speech-2.8-turbo",
            ));
        }
        if let Some(weights) = &self.timbre_weights {
            if weights.len() > 4 {
                return Err(invalid("timbre_weights supports at most four voices"));
            }
            if weights.iter().any(|item| {
                item.voice_id.trim().is_empty()
                    || item.voice_id.chars().any(char::is_control)
                    || !(1..=100).contains(&item.weight)
            }) {
                return Err(invalid(
                    "each timbre weight needs a nonempty voice_id and a weight from 1 through 100",
                ));
            }
        }
        if let Some(language) = self.language_boost {
            if matches!(
                self.model,
                MiniMaxStreamingTtsModel::Speech01Hd
                    | MiniMaxStreamingTtsModel::Speech01Turbo
                    | MiniMaxStreamingTtsModel::Speech02Hd
                    | MiniMaxStreamingTtsModel::Speech02Turbo
            ) && matches!(
                language,
                MiniMaxStreamingTtsLanguageBoost::Persian
                    | MiniMaxStreamingTtsLanguageBoost::Filipino
                    | MiniMaxStreamingTtsLanguageBoost::Tamil
            ) {
                return Err(invalid(
                    "speech-01 and speech-02 models do not support Persian, Filipino, or Tamil language_boost",
                ));
            }
        }
        if let Some(dictionary) = &self.pronunciation_dict {
            if dictionary
                .tone
                .iter()
                .any(|entry| entry.trim().is_empty() || entry.chars().any(char::is_control))
            {
                return Err(invalid(
                    "pronunciation entries must be nonempty and contain no control characters",
                ));
            }
        }
        Ok(())
    }
}

/// Local memory and payload bounds for one WebSocket session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniMaxStreamingTtsLimits {
    pub max_frame_bytes: usize,
    pub max_text_bytes_per_segment: usize,
    pub max_pending_segments: usize,
    pub max_audio_bytes_per_session: usize,
}

impl Default for MiniMaxStreamingTtsLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 1024 * 1024,
            max_text_bytes_per_segment: 16 * 1024,
            max_pending_segments: 8,
            max_audio_bytes_per_session: 64 * 1024 * 1024,
        }
    }
}

impl MiniMaxStreamingTtsLimits {
    fn validate(self) -> Result<(), MiniMaxStreamingTtsError> {
        if self.max_frame_bytes == 0
            || self.max_frame_bytes > HARD_MAX_FRAME_BYTES
            || self.max_text_bytes_per_segment == 0
            || self.max_text_bytes_per_segment > HARD_MAX_TEXT_BYTES
            || self.max_pending_segments == 0
            || self.max_pending_segments > HARD_MAX_PENDING_SEGMENTS
            || self.max_audio_bytes_per_session == 0
            || self.max_audio_bytes_per_session > HARD_MAX_AUDIO_BYTES
        {
            return Err(invalid(
                "stream limits are outside the supported local bounds",
            ));
        }
        if self.max_text_bytes_per_segment > self.max_frame_bytes {
            return Err(invalid(
                "text limit cannot exceed the maximum WebSocket frame size",
            ));
        }
        Ok(())
    }
}

/// A MiniMax WebSocket T2A service bound to one account profile and route.
/// Credentials are borrowed for each connection and are never stored.
pub struct MiniMaxStreamingTtsService {
    transport: Arc<dyn RealtimeTransport>,
    scope: MiniMaxStreamingTtsScope,
    endpoint: String,
}

impl fmt::Debug for MiniMaxStreamingTtsService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxStreamingTtsService")
            .field("scope", &self.scope)
            .field("endpoint", &"<fixed>")
            .finish_non_exhaustive()
    }
}

impl MiniMaxStreamingTtsService {
    pub fn new(
        transport: Arc<dyn RealtimeTransport>,
        config: MiniMaxStreamingTtsConfig,
    ) -> Result<Self, MiniMaxStreamingTtsError> {
        let expected_endpoint = match config.region {
            MiniMaxTtsRegion::International => MINIMAX_STREAMING_TTS_INTERNATIONAL_ENDPOINT,
            MiniMaxTtsRegion::ChinaMainland => MINIMAX_STREAMING_TTS_CHINA_ENDPOINT,
        };
        if config.endpoint != expected_endpoint {
            return Err(invalid(
                "MiniMax T2A WSS endpoint does not match the selected region",
            ));
        }
        if config.profile_name.trim().is_empty() || config.account_scope.trim().is_empty() {
            return Err(invalid("profile_name and account_scope must not be empty"));
        }
        let scope = MiniMaxStreamingTtsScope {
            provider_id: ProviderId::new("minimax"),
            profile_name: config.profile_name,
            endpoint_fingerprint: provider_file_endpoint_fingerprint(&config.endpoint),
            account_scope: config.account_scope,
            region: config.region,
        };
        Ok(Self {
            transport,
            scope,
            endpoint: config.endpoint,
        })
    }

    pub fn scope(&self) -> &MiniMaxStreamingTtsScope {
        &self.scope
    }

    /// Connect using an explicitly scoped MiniMax voice resource. Only refs
    /// created under the matching regional HTTP `/v1` root can be used with
    /// this fixed regional WSS route. Built-in IDs remain available through
    /// [`Self::connect`].
    pub async fn connect_with_voice(
        &self,
        credential: &Secret<String>,
        request: &MiniMaxStreamingTtsRequest,
        voice: &MiniMaxVoiceRef,
        limits: MiniMaxStreamingTtsLimits,
    ) -> Result<MiniMaxStreamingTtsSession, MiniMaxStreamingTtsError> {
        if request.timbre_weights.is_some() {
            return Err(invalid(
                "connect_with_voice cannot be combined with timbre_weights; pass the complete mix through connect",
            ));
        }
        self.validate_voice_reference(voice)?;
        let mut request = request.clone();
        request.voice_setting.voice_id = voice.voice_id().to_owned();
        self.connect(credential, &request, limits).await
    }

    fn validate_voice_reference(
        &self,
        voice: &MiniMaxVoiceRef,
    ) -> Result<(), MiniMaxStreamingTtsError> {
        let reference_scope = voice.scope();
        if MiniMaxVoiceRef::new(
            reference_scope.clone(),
            voice.kind(),
            voice.voice_id().to_owned(),
        )
        .is_err()
        {
            return Err(invalid("MiniMax voice reference is malformed"));
        }
        let expected_voice_endpoint = match self.scope.region {
            MiniMaxTtsRegion::International => "https://api.minimax.io/v1",
            MiniMaxTtsRegion::ChinaMainland => "https://api.minimax.cn/v1",
        };
        let expected_endpoint = provider_file_endpoint_fingerprint(expected_voice_endpoint);
        if reference_scope.provider_id != self.scope.provider_id
            || reference_scope.profile_name != self.scope.profile_name
            || reference_scope.endpoint_fingerprint != expected_endpoint
            || reference_scope.account_scope != self.scope.account_scope
            || reference_scope.region != self.scope.region
        {
            return Err(invalid(
                "MiniMax voice reference does not match the verified regional WSS profile, account, region, or HTTP API endpoint",
            ));
        }
        Ok(())
    }

    /// Open one T2A WebSocket session and wait for both `connected_success`
    /// and `task_started`. Once `task_start` has been attempted, any missing
    /// acknowledgement is returned as an outcome-unknown error; this method
    /// never retries or reconnects.
    pub async fn connect(
        &self,
        credential: &Secret<String>,
        request: &MiniMaxStreamingTtsRequest,
        limits: MiniMaxStreamingTtsLimits,
    ) -> Result<MiniMaxStreamingTtsSession, MiniMaxStreamingTtsError> {
        validate_credential(credential)?;
        limits.validate()?;
        request.validate_for_region(self.scope.region)?;
        let start_event = request.start_event()?;
        let start_frame = json_frame(&start_event, limits.max_frame_bytes)?;

        let mut connection = self
            .transport
            .connect(RealtimeConnectRequest {
                endpoint: self.endpoint.clone(),
                headers: vec![(
                    "Authorization".into(),
                    format!("Bearer {}", credential.expose_secret()),
                )],
                max_frame_bytes: limits.max_frame_bytes,
            })
            .await?;

        let connected_native =
            receive_required_json(&mut connection, limits.max_frame_bytes).await?;
        validate_provider_status(&connected_native)?;
        if event_name(&connected_native) != Some("connected_success") {
            return Err(protocol(
                "the first server event was not connected_success",
                Some(connected_native),
            ));
        }

        if let Err(source) = connection.outbound.send(start_frame).await {
            return Err(MiniMaxStreamingTtsError::OutcomeUnknown {
                operation: "task_start",
                reason: source.to_string(),
            });
        }

        let mut setup_events = Vec::new();
        let mut task_started_native = None;
        for _ in 0..MAX_SETUP_EVENTS {
            let native = receive_required_json(&mut connection, limits.max_frame_bytes)
                .await
                .map_err(|error| error.with_operation_unknown("task_start"))?;
            if let Some(error) = provider_status_error(&native) {
                return Err(error);
            }
            match event_name(&native) {
                Some("task_started") => {
                    task_started_native = Some(native);
                    break;
                }
                Some("task_failed") => return Err(provider_failure(native)),
                _ => setup_events.push(native),
            }
        }
        let Some(task_started_native) = task_started_native else {
            return Err(MiniMaxStreamingTtsError::OutcomeUnknown {
                operation: "task_start",
                reason: "MiniMax did not acknowledge task_started within the setup event limit"
                    .into(),
            });
        };

        let metadata = Arc::new(MiniMaxStreamingTtsMetadata {
            scope: self.scope.clone(),
            connected_native,
            task_started_native,
            setup_events,
        });
        let last_session_id = string_field(&metadata.task_started_native, "session_id")
            .or_else(|| string_field(&metadata.connected_native, "session_id"));
        let last_trace_id = string_field(&metadata.task_started_native, "trace_id")
            .or_else(|| string_field(&metadata.connected_native, "trace_id"));
        let shared = Arc::new(AsyncMutex::new(SharedState {
            sink: Some(connection.outbound),
            pending_segments: 0,
            finishing: false,
            finished: false,
            failed: false,
            audio_bytes: 0,
        }));
        Ok(MiniMaxStreamingTtsSession {
            input: MiniMaxStreamingTtsInput {
                shared: shared.clone(),
                limits,
                metadata: metadata.clone(),
            },
            events: MiniMaxStreamingTtsEvents {
                inbound: connection.inbound,
                shared,
                limits,
                metadata,
                terminal_event_seen: false,
                last_session_id,
                last_trace_id,
            },
        })
    }
}

/// Raw provider-native responses consumed while establishing a session.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxStreamingTtsMetadata {
    pub scope: MiniMaxStreamingTtsScope,
    pub connected_native: Value,
    pub task_started_native: Value,
    /// Any unrecognized setup events received before `task_started`.
    pub setup_events: Vec<Value>,
}

impl fmt::Debug for MiniMaxStreamingTtsMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxStreamingTtsMetadata")
            .field("scope", &self.scope)
            .field("connected_native", &"<preserved>")
            .field("task_started_native", &"<preserved>")
            .field("setup_event_count", &self.setup_events.len())
            .finish()
    }
}

/// Connected duplex T2A task. Split the handles and poll `events.next()` while
/// sending more text through `input` to use the documented provider queue.
pub struct MiniMaxStreamingTtsSession {
    input: MiniMaxStreamingTtsInput,
    events: MiniMaxStreamingTtsEvents,
}

impl fmt::Debug for MiniMaxStreamingTtsSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxStreamingTtsSession")
            .field("metadata", &self.input.metadata)
            .finish_non_exhaustive()
    }
}

impl MiniMaxStreamingTtsSession {
    pub fn metadata(&self) -> &MiniMaxStreamingTtsMetadata {
        &self.input.metadata
    }

    pub fn into_parts(self) -> (MiniMaxStreamingTtsInput, MiniMaxStreamingTtsEvents) {
        (self.input, self.events)
    }
}

struct SharedState {
    sink: Option<Box<dyn RealtimeSink>>,
    pending_segments: usize,
    finishing: bool,
    finished: bool,
    failed: bool,
    audio_bytes: usize,
}

impl SharedState {
    async fn send_once(
        &mut self,
        frame: RealtimeFrame,
        operation: &'static str,
    ) -> Result<(), MiniMaxStreamingTtsError> {
        let mut sink = self.sink.take().ok_or(MiniMaxStreamingTtsError::Closed)?;
        // Keep the state failed until the write completes. Cancellation drops
        // the owned sink and leaves this flag set, just like an uncertain error.
        // The caller holds the shared lock through the write and state update.
        self.failed = true;
        sink.send(frame)
            .await
            .map_err(|source| MiniMaxStreamingTtsError::OutcomeUnknown {
                operation,
                reason: source.to_string(),
            })?;
        self.sink = Some(sink);
        self.failed = false;
        Ok(())
    }
}

/// Text sender half. `send_text` queues one bounded segment; the service does
/// not replay it if the send result becomes uncertain.
pub struct MiniMaxStreamingTtsInput {
    shared: Arc<AsyncMutex<SharedState>>,
    limits: MiniMaxStreamingTtsLimits,
    metadata: Arc<MiniMaxStreamingTtsMetadata>,
}

impl fmt::Debug for MiniMaxStreamingTtsInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxStreamingTtsInput")
            .field("metadata", &self.metadata)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl MiniMaxStreamingTtsInput {
    pub fn metadata(&self) -> &MiniMaxStreamingTtsMetadata {
        &self.metadata
    }

    /// Queue one text segment. Multiple segments may be queued in order up to
    /// `max_pending_segments`; audio results remain a provider-native stream.
    /// Cancelling an in-progress write invalidates the sender and drops its
    /// transport half because the provider may already have received the text.
    pub async fn send_text(&mut self, text: &str) -> Result<(), MiniMaxStreamingTtsError> {
        if text.trim().is_empty() {
            return Err(invalid("text segment must not be empty"));
        }
        if text.len() > self.limits.max_text_bytes_per_segment {
            return Err(invalid("text segment exceeds the configured byte limit"));
        }
        let frame = json_frame(
            &json!({ "event": "task_continue", "text": text }),
            self.limits.max_frame_bytes,
        )?;

        let mut shared = self.shared.lock().await;
        if shared.failed || shared.finished {
            return Err(MiniMaxStreamingTtsError::Closed);
        }
        if shared.finishing {
            return Err(invalid("task_finish was already sent"));
        }
        if shared.pending_segments >= self.limits.max_pending_segments {
            return Err(MiniMaxStreamingTtsError::PendingQueueFull {
                max: self.limits.max_pending_segments,
            });
        }
        shared.send_once(frame, "task_continue").await?;
        shared.pending_segments += 1;
        Ok(())
    }

    /// Tell MiniMax to drain all queued segments and close the task. Continue
    /// reading events until `TaskFinished`; a failed write is outcome-unknown.
    /// Cancelling an in-progress write also invalidates the sender.
    pub async fn finish(&mut self) -> Result<(), MiniMaxStreamingTtsError> {
        let mut shared = self.shared.lock().await;
        if shared.failed || shared.finished {
            return Err(MiniMaxStreamingTtsError::Closed);
        }
        if shared.finishing {
            return Err(invalid("task_finish was already sent"));
        }
        let frame = RealtimeFrame::text(Bytes::from_static(br#"{"event":"task_finish"}"#));
        if frame.len() > self.limits.max_frame_bytes {
            return Err(RealtimeError::FrameTooLarge {
                actual: frame.len(),
                max: self.limits.max_frame_bytes,
            }
            .into());
        }
        shared.send_once(frame, "task_finish").await?;
        shared.finishing = true;
        Ok(())
    }

    /// Abort the WebSocket explicitly. Dropping both session halves also
    /// releases the injected transport without sending a replay or retry.
    pub async fn abort(&mut self) -> Result<(), MiniMaxStreamingTtsError> {
        let mut shared = self.shared.lock().await;
        if shared.finished || shared.failed {
            return Ok(());
        }
        if let Some(mut sink) = shared.sink.take() {
            sink.close(RealtimeClose::normal("MiniMax TTS session cancelled"))
                .await?;
        }
        shared.failed = true;
        Ok(())
    }
}

/// A provider event. Native JSON is retained for forward compatibility.
#[derive(Clone, PartialEq)]
pub enum MiniMaxStreamingTtsEvent {
    AudioDelta {
        audio: Bytes,
        is_final: bool,
        extra_info: Option<Value>,
        session_id: Option<String>,
        trace_id: Option<String>,
        native: Value,
    },
    TaskFinished {
        session_id: Option<String>,
        trace_id: Option<String>,
        native: Value,
    },
    TaskFailed {
        status_code: Option<i64>,
        status_message: Option<String>,
        session_id: Option<String>,
        trace_id: Option<String>,
        native: Value,
    },
    ProviderError {
        status_code: Option<i64>,
        status_message: Option<String>,
        session_id: Option<String>,
        trace_id: Option<String>,
        native: Value,
    },
    Native {
        event: Option<String>,
        session_id: Option<String>,
        trace_id: Option<String>,
        native: Value,
    },
}

impl fmt::Debug for MiniMaxStreamingTtsEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AudioDelta {
                audio,
                is_final,
                session_id,
                trace_id,
                ..
            } => f
                .debug_struct("AudioDelta")
                .field("audio_bytes", &audio.len())
                .field("is_final", is_final)
                .field("session_id", session_id)
                .field("trace_id", trace_id)
                .field("native", &"<preserved>")
                .finish(),
            Self::TaskFinished {
                session_id,
                trace_id,
                ..
            } => f
                .debug_struct("TaskFinished")
                .field("session_id", session_id)
                .field("trace_id", trace_id)
                .field("native", &"<preserved>")
                .finish(),
            Self::TaskFailed {
                status_code,
                status_message,
                session_id,
                trace_id,
                ..
            } => f
                .debug_struct("TaskFailed")
                .field("status_code", status_code)
                .field("status_message", status_message)
                .field("session_id", session_id)
                .field("trace_id", trace_id)
                .field("native", &"<preserved>")
                .finish(),
            Self::ProviderError {
                status_code,
                status_message,
                session_id,
                trace_id,
                ..
            } => f
                .debug_struct("ProviderError")
                .field("status_code", status_code)
                .field("status_message", status_message)
                .field("session_id", session_id)
                .field("trace_id", trace_id)
                .field("native", &"<preserved>")
                .finish(),
            Self::Native {
                event,
                session_id,
                trace_id,
                ..
            } => f
                .debug_struct("Native")
                .field("event", event)
                .field("session_id", session_id)
                .field("trace_id", trace_id)
                .field("native", &"<preserved>")
                .finish(),
        }
    }
}

/// Receive half for incremental audio and terminal task events.
pub struct MiniMaxStreamingTtsEvents {
    inbound: BoxStream<'static, Result<RealtimeFrame, RealtimeError>>,
    shared: Arc<AsyncMutex<SharedState>>,
    limits: MiniMaxStreamingTtsLimits,
    metadata: Arc<MiniMaxStreamingTtsMetadata>,
    terminal_event_seen: bool,
    last_session_id: Option<String>,
    last_trace_id: Option<String>,
}

impl fmt::Debug for MiniMaxStreamingTtsEvents {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MiniMaxStreamingTtsEvents")
            .field("metadata", &self.metadata)
            .field("limits", &self.limits)
            .field("terminal_event_seen", &self.terminal_event_seen)
            .finish_non_exhaustive()
    }
}

impl MiniMaxStreamingTtsEvents {
    pub fn metadata(&self) -> &MiniMaxStreamingTtsMetadata {
        &self.metadata
    }

    pub async fn next(
        &mut self,
    ) -> Result<Option<MiniMaxStreamingTtsEvent>, MiniMaxStreamingTtsError> {
        if self.terminal_event_seen {
            return Ok(None);
        }
        let Some(frame) = self.inbound.next().await else {
            let shared = self.shared.lock().await;
            if shared.finished {
                return Ok(None);
            }
            return Err(MiniMaxStreamingTtsError::Interrupted {
                session_id: self.last_session_id.clone(),
                trace_id: self.last_trace_id.clone(),
                reason: "WebSocket closed before task_finished".into(),
            });
        };
        let frame = match frame {
            Ok(frame) => frame,
            Err(error) => {
                return Err(MiniMaxStreamingTtsError::Interrupted {
                    session_id: self.last_session_id.clone(),
                    trace_id: self.last_trace_id.clone(),
                    reason: error.to_string(),
                })
            }
        };
        if frame.len() > self.limits.max_frame_bytes {
            return Err(RealtimeError::FrameTooLarge {
                actual: frame.len(),
                max: self.limits.max_frame_bytes,
            }
            .into());
        }
        let RealtimeFrame::Text(bytes) = frame else {
            return Err(protocol(
                "MiniMax T2A events must be JSON text frames",
                None,
            ));
        };
        let native: Value = serde_json::from_slice(&bytes)
            .map_err(|_| protocol("MiniMax sent invalid JSON", None))?;
        if !native.is_object() {
            return Err(protocol(
                "MiniMax event must be a JSON object",
                Some(native),
            ));
        }
        let event = event_name(&native).map(str::to_owned);
        let (status_code, status_message) = provider_status(&native);
        let session_id = string_field(&native, "session_id");
        let trace_id = string_field(&native, "trace_id");
        if session_id.is_some() {
            self.last_session_id = session_id.clone();
        }
        if trace_id.is_some() {
            self.last_trace_id = trace_id.clone();
        }

        if event.as_deref() == Some("task_failed") {
            let mut shared = self.shared.lock().await;
            shared.failed = true;
            if let Some(sink) = shared.sink.as_mut() {
                let _ = sink
                    .close(RealtimeClose::normal("MiniMax TTS task failed"))
                    .await;
            }
            self.terminal_event_seen = true;
            return Ok(Some(MiniMaxStreamingTtsEvent::TaskFailed {
                status_code,
                status_message,
                session_id,
                trace_id,
                native,
            }));
        }

        if status_code.is_some_and(|code| code != 0) {
            let mut shared = self.shared.lock().await;
            shared.failed = true;
            if let Some(sink) = shared.sink.as_mut() {
                let _ = sink
                    .close(RealtimeClose::normal("MiniMax TTS provider error"))
                    .await;
            }
            self.terminal_event_seen = true;
            return Ok(Some(MiniMaxStreamingTtsEvent::ProviderError {
                status_code,
                status_message,
                session_id,
                trace_id,
                native,
            }));
        }

        if event.as_deref() == Some("task_finished") {
            let mut shared = self.shared.lock().await;
            if shared.pending_segments != 0 {
                return Err(protocol(
                    "task_finished arrived while text segments were still pending",
                    Some(native),
                ));
            }
            shared.finished = true;
            self.terminal_event_seen = true;
            return Ok(Some(MiniMaxStreamingTtsEvent::TaskFinished {
                session_id,
                trace_id,
                native,
            }));
        }

        if let Some(audio_hex) = native
            .get("data")
            .and_then(Value::as_object)
            .and_then(|data| data.get("audio"))
        {
            let Some(audio_hex) = audio_hex.as_str() else {
                return Err(protocol(
                    "data.audio must be a hexadecimal string",
                    Some(native),
                ));
            };
            let audio =
                decode_hex(audio_hex).map_err(|message| protocol(message, Some(native.clone())))?;
            let is_final = match native.get("is_final") {
                Some(Value::Bool(value)) => *value,
                _ => {
                    return Err(protocol(
                        "audio event is missing boolean is_final",
                        Some(native.clone()),
                    ))
                }
            };
            let mut shared = self.shared.lock().await;
            if shared.pending_segments == 0 {
                return Err(protocol(
                    "audio arrived without a pending task_continue segment",
                    Some(native),
                ));
            }
            let next_total = shared.audio_bytes.saturating_add(audio.len());
            if next_total > self.limits.max_audio_bytes_per_session {
                shared.failed = true;
                if let Some(sink) = shared.sink.as_mut() {
                    let _ = sink
                        .close(RealtimeClose::normal("MiniMax TTS audio limit reached"))
                        .await;
                }
                return Err(MiniMaxStreamingTtsError::AudioLimitExceeded {
                    max: self.limits.max_audio_bytes_per_session,
                });
            }
            shared.audio_bytes = next_total;
            if is_final {
                shared.pending_segments -= 1;
            }
            return Ok(Some(MiniMaxStreamingTtsEvent::AudioDelta {
                audio: Bytes::from(audio),
                is_final,
                extra_info: native.get("extra_info").cloned(),
                session_id,
                trace_id,
                native,
            }));
        }

        Ok(Some(MiniMaxStreamingTtsEvent::Native {
            event,
            session_id,
            trace_id,
            native,
        }))
    }
}

#[derive(Debug, Error)]
pub enum MiniMaxStreamingTtsError {
    #[error("invalid MiniMax streaming TTS input: {message}")]
    InvalidInput { message: String },
    #[error("MiniMax streaming TTS is not documented for region {region:?}")]
    UnsupportedRegion { region: MiniMaxTtsRegion },
    #[error("MiniMax API credential is empty or exceeds the local limit")]
    InvalidCredential,
    #[error("MiniMax streaming TTS session is closed")]
    Closed,
    #[error("MiniMax streaming TTS has {max} pending segments, the configured limit")]
    PendingQueueFull { max: usize },
    #[error("MiniMax streaming TTS output exceeded the configured {max}-byte session limit")]
    AudioLimitExceeded { max: usize },
    #[error("MiniMax streaming TTS connection was interrupted (session {session_id:?}, trace {trace_id:?}): {reason}")]
    Interrupted {
        session_id: Option<String>,
        trace_id: Option<String>,
        reason: String,
    },
    #[error("MiniMax streaming TTS protocol error: {message}")]
    Protocol {
        message: String,
        native: Option<Value>,
    },
    #[error("MiniMax T2A provider error {status_code:?}: {status_message:?}")]
    Provider {
        status_code: Option<i64>,
        status_message: Option<String>,
        native: Value,
    },
    #[error("the outcome of MiniMax {operation} is unknown: {reason}")]
    OutcomeUnknown {
        operation: &'static str,
        reason: String,
    },
    #[error(transparent)]
    Realtime(#[from] RealtimeError),
}

impl MiniMaxStreamingTtsError {
    fn with_operation_unknown(self, operation: &'static str) -> Self {
        match self {
            Self::Realtime(source) => Self::OutcomeUnknown {
                operation,
                reason: source.to_string(),
            },
            other => Self::OutcomeUnknown {
                operation,
                reason: other.to_string(),
            },
        }
    }
}

fn invalid(message: impl Into<String>) -> MiniMaxStreamingTtsError {
    MiniMaxStreamingTtsError::InvalidInput {
        message: message.into(),
    }
}

fn protocol(message: impl Into<String>, native: Option<Value>) -> MiniMaxStreamingTtsError {
    MiniMaxStreamingTtsError::Protocol {
        message: message.into(),
        native,
    }
}

fn validate_credential(credential: &Secret<String>) -> Result<(), MiniMaxStreamingTtsError> {
    let value = credential.expose_secret();
    if value.trim().is_empty() || value.len() > MAX_CREDENTIAL_BYTES {
        return Err(MiniMaxStreamingTtsError::InvalidCredential);
    }
    Ok(())
}

fn json_frame(
    value: &Value,
    max_frame_bytes: usize,
) -> Result<RealtimeFrame, MiniMaxStreamingTtsError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| protocol("could not encode MiniMax T2A event", None))?;
    if bytes.len() > max_frame_bytes {
        return Err(RealtimeError::FrameTooLarge {
            actual: bytes.len(),
            max: max_frame_bytes,
        }
        .into());
    }
    Ok(RealtimeFrame::text(Bytes::from(bytes)))
}

async fn receive_required_json(
    connection: &mut RealtimeConnection,
    max_frame_bytes: usize,
) -> Result<Value, MiniMaxStreamingTtsError> {
    let frame = connection.inbound.next().await.ok_or_else(|| {
        MiniMaxStreamingTtsError::Realtime(RealtimeError::UnexpectedRemoteClose)
    })??;
    if frame.len() > max_frame_bytes {
        return Err(RealtimeError::FrameTooLarge {
            actual: frame.len(),
            max: max_frame_bytes,
        }
        .into());
    }
    let RealtimeFrame::Text(bytes) = frame else {
        return Err(protocol(
            "MiniMax T2A events must be JSON text frames",
            None,
        ));
    };
    let native: Value = serde_json::from_slice(&bytes)
        .map_err(|_| protocol("MiniMax sent invalid JSON during setup", None))?;
    if !native.is_object() {
        return Err(protocol(
            "MiniMax setup event must be a JSON object",
            Some(native),
        ));
    }
    Ok(native)
}

fn event_name(native: &Value) -> Option<&str> {
    native.get("event").and_then(Value::as_str)
}

fn provider_status(native: &Value) -> (Option<i64>, Option<String>) {
    let response = native.get("base_resp");
    (
        response
            .and_then(|value| value.get("status_code"))
            .and_then(Value::as_i64),
        response
            .and_then(|value| value.get("status_msg"))
            .and_then(Value::as_str)
            .map(str::to_owned),
    )
}

fn provider_status_error(native: &Value) -> Option<MiniMaxStreamingTtsError> {
    let (status_code, status_message) = provider_status(native);
    status_code
        .filter(|code| *code != 0)
        .map(|status_code| MiniMaxStreamingTtsError::Provider {
            status_code: Some(status_code),
            status_message,
            native: native.clone(),
        })
}

fn validate_provider_status(native: &Value) -> Result<(), MiniMaxStreamingTtsError> {
    if let Some(error) = provider_status_error(native) {
        return Err(error);
    }
    Ok(())
}

fn provider_failure(native: Value) -> MiniMaxStreamingTtsError {
    let (status_code, status_message) = provider_status(&native);
    MiniMaxStreamingTtsError::Provider {
        status_code,
        status_message,
        native,
    }
}

fn string_field(native: &Value, field: &str) -> Option<String> {
    native.get(field).and_then(Value::as_str).map(str::to_owned)
}

fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    if !value.len().is_multiple_of(2) {
        return Err("data.audio has an odd number of hexadecimal digits".into());
    }
    let mut decoded = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let high =
            hex_nibble(pair[0]).ok_or_else(|| "data.audio is not valid hexadecimal".to_owned())?;
        let low =
            hex_nibble(pair[1]).ok_or_else(|| "data.audio is not valid hexadecimal".to_owned())?;
        decoded.push((high << 4) | low);
    }
    Ok(decoded)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}
