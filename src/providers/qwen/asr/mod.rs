//! Asynchronous file transcription through Alibaba Cloud Model Studio's
//! Qwen3-ASR-Flash-Filetrans and Qwen Audio 3.x Filetrans task APIs.
//!
//! The caller supplies a public file URL and a regional Model Studio API key.
//! The service submits one task, exposes one-shot task lookup, and can fetch
//! the short-lived transcription document returned by the task API. It never
//! polls, retries, uploads audio, or changes region on the caller's behalf.

use crate::{
    client::RequestOptions,
    protocol::LlmError,
    transport::{HttpExecutor, HttpRequest, Transport},
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::fmt;
use thiserror::Error;
use url::Url;

/// The only model in this slice. It accepts public audio URLs and returns an
/// asynchronous file-transcription task.
pub const QWEN3_ASR_FILETRANS_MODEL: &str = "qwen3-asr-flash-filetrans";

const MAX_CONTROL_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const MAX_TRANSCRIPTION_BYTES: usize = 64 * 1024 * 1024;
const SUBMIT_PATH: &str = "/services/audio/asr/transcription";
const TASK_PATH: &str = "/tasks/";

/// Model Studio region. Filetrans's workspace-specific REST routes are
/// documented for Beijing and Singapore; API keys are region-specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenAsrRegion {
    Beijing,
    Singapore,
}

impl QwenAsrRegion {
    fn domain_suffix(self) -> &'static str {
        match self {
            Self::Beijing => "cn-beijing.maas.aliyuncs.com",
            Self::Singapore => "ap-southeast-1.maas.aliyuncs.com",
        }
    }

    fn result_suffix(self) -> &'static str {
        match self {
            Self::Beijing => ".oss-cn-beijing.aliyuncs.com",
            Self::Singapore => ".oss-ap-southeast-1.aliyuncs.com",
        }
    }
}

/// Stable caller-owned identity for one Qwen ASR account, region, and
/// workspace. A task reference carries this scope so it cannot be queried
/// through a service bound to another account or workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenAsrScope {
    profile_name: String,
    account_scope: String,
    region: QwenAsrRegion,
    workspace_id: String,
}

impl QwenAsrScope {
    pub fn new(
        profile_name: impl Into<String>,
        account_scope: impl Into<String>,
        region: QwenAsrRegion,
        workspace_id: impl Into<String>,
    ) -> Result<Self, QwenAsrError> {
        let scope = Self {
            profile_name: profile_name.into(),
            account_scope: account_scope.into(),
            region,
            workspace_id: workspace_id.into(),
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

    pub fn region(&self) -> QwenAsrRegion {
        self.region
    }

    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    fn validate(&self) -> Result<(), QwenAsrError> {
        if self.profile_name.trim().is_empty() || self.account_scope.trim().is_empty() {
            return Err(QwenAsrError::InvalidInput(
                "profile name and account scope must be non-empty".into(),
            ));
        }
        // Workspace IDs occupy exactly one DNS label in the official
        // workspace-specific endpoint. Keep this check strict so the caller's
        // API key cannot be directed to an arbitrary host.
        let id = self.workspace_id.as_bytes();
        if id.is_empty()
            || id.len() > 63
            || !id[0].is_ascii_alphanumeric()
            || !id[id.len() - 1].is_ascii_alphanumeric()
            || !id
                .iter()
                .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'-')
        {
            return Err(QwenAsrError::InvalidInput(
                "workspace ID must be one DNS label containing only letters, digits, and internal hyphens".into(),
            ));
        }
        Ok(())
    }

    fn api_base(&self) -> String {
        format!(
            "https://{}.{}{}",
            self.workspace_id,
            self.region.domain_suffix(),
            "/api/v1"
        )
    }
}

/// Language hint accepted by the current Qwen3 file-transcription API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QwenAsrLanguage {
    #[serde(rename = "zh")]
    Chinese,
    #[serde(rename = "yue")]
    Cantonese,
    #[serde(rename = "en")]
    English,
    #[serde(rename = "ja")]
    Japanese,
    #[serde(rename = "de")]
    German,
    #[serde(rename = "ko")]
    Korean,
    #[serde(rename = "ru")]
    Russian,
    #[serde(rename = "fr")]
    French,
    #[serde(rename = "pt")]
    Portuguese,
    #[serde(rename = "ar")]
    Arabic,
    #[serde(rename = "it")]
    Italian,
    #[serde(rename = "es")]
    Spanish,
    #[serde(rename = "hi")]
    Hindi,
    #[serde(rename = "id")]
    Indonesian,
    #[serde(rename = "th")]
    Thai,
    #[serde(rename = "tr")]
    Turkish,
    #[serde(rename = "uk")]
    Ukrainian,
    #[serde(rename = "vi")]
    Vietnamese,
    #[serde(rename = "cs")]
    Czech,
    #[serde(rename = "da")]
    Danish,
    #[serde(rename = "fil")]
    Filipino,
    #[serde(rename = "fi")]
    Finnish,
    #[serde(rename = "is")]
    Icelandic,
    #[serde(rename = "ms")]
    Malay,
    #[serde(rename = "no")]
    Norwegian,
    #[serde(rename = "pl")]
    Polish,
    #[serde(rename = "sv")]
    Swedish,
}

impl QwenAsrLanguage {
    fn wire_value(self) -> &'static str {
        match self {
            Self::Chinese => "zh",
            Self::Cantonese => "yue",
            Self::English => "en",
            Self::Japanese => "ja",
            Self::German => "de",
            Self::Korean => "ko",
            Self::Russian => "ru",
            Self::French => "fr",
            Self::Portuguese => "pt",
            Self::Arabic => "ar",
            Self::Italian => "it",
            Self::Spanish => "es",
            Self::Hindi => "hi",
            Self::Indonesian => "id",
            Self::Thai => "th",
            Self::Turkish => "tr",
            Self::Ukrainian => "uk",
            Self::Vietnamese => "vi",
            Self::Czech => "cs",
            Self::Danish => "da",
            Self::Filipino => "fil",
            Self::Finnish => "fi",
            Self::Icelandic => "is",
            Self::Malay => "ms",
            Self::Norwegian => "no",
            Self::Polish => "pl",
            Self::Swedish => "sv",
        }
    }
}

/// Optional parameters for one Qwen3-ASR-Flash-Filetrans job.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QwenAsrParameters {
    pub language: Option<QwenAsrLanguage>,
    pub enable_itn: Option<bool>,
    pub enable_words: Option<bool>,
    /// Zero-based audio track indices. Each requested track is billed
    /// separately by Model Studio.
    pub channel_ids: Option<Vec<u32>>,
}

impl QwenAsrParameters {
    fn validate(&self) -> Result<(), QwenAsrError> {
        if self.channel_ids.as_ref().is_some_and(Vec::is_empty) {
            return Err(QwenAsrError::InvalidInput(
                "channel_ids must be omitted or contain at least one track".into(),
            ));
        }
        if let Some(ids) = &self.channel_ids {
            let mut unique = std::collections::BTreeSet::new();
            if ids.iter().any(|id| !unique.insert(id)) {
                return Err(QwenAsrError::InvalidInput(
                    "channel_ids must not contain duplicates".into(),
                ));
            }
        }
        Ok(())
    }

    fn to_wire(&self) -> Map<String, Value> {
        let mut parameters = Map::new();
        if let Some(language) = self.language {
            parameters.insert("language".into(), json!(language.wire_value()));
        }
        if let Some(enable_itn) = self.enable_itn {
            parameters.insert("enable_itn".into(), json!(enable_itn));
        }
        if let Some(enable_words) = self.enable_words {
            parameters.insert("enable_words".into(), json!(enable_words));
        }
        if let Some(channel_ids) = &self.channel_ids {
            parameters.insert("channel_id".into(), json!(channel_ids));
        }
        parameters
    }
}

/// Qwen Audio ASR Filetrans model snapshots supported by the official
/// asynchronous transcription API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QwenAudioAsrModel {
    #[serde(rename = "qwen-audio-3.1-asr-flash-filetrans")]
    QwenAudio31AsrFlashFiletrans,
    #[serde(rename = "qwen-audio-3.0-asr-flash-filetrans")]
    QwenAudio30AsrFlashFiletrans,
}

impl QwenAudioAsrModel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QwenAudio31AsrFlashFiletrans => "qwen-audio-3.1-asr-flash-filetrans",
            Self::QwenAudio30AsrFlashFiletrans => "qwen-audio-3.0-asr-flash-filetrans",
        }
    }
}

/// Language hint accepted by Qwen Audio 3.x ASR Filetrans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QwenAudioAsrLanguage {
    #[serde(rename = "zh")]
    Chinese,
    #[serde(rename = "en")]
    English,
    #[serde(rename = "ja")]
    Japanese,
    #[serde(rename = "ko")]
    Korean,
    #[serde(rename = "vi")]
    Vietnamese,
    #[serde(rename = "th")]
    Thai,
    #[serde(rename = "id")]
    Indonesian,
    #[serde(rename = "ms")]
    Malay,
    #[serde(rename = "tl")]
    Filipino,
    #[serde(rename = "hi")]
    Hindi,
    #[serde(rename = "ar")]
    Arabic,
    #[serde(rename = "fr")]
    French,
    #[serde(rename = "de")]
    German,
    #[serde(rename = "es")]
    Spanish,
    #[serde(rename = "pt")]
    Portuguese,
    #[serde(rename = "ru")]
    Russian,
    #[serde(rename = "it")]
    Italian,
    #[serde(rename = "nl")]
    Dutch,
    #[serde(rename = "sv")]
    Swedish,
    #[serde(rename = "da")]
    Danish,
    #[serde(rename = "fi")]
    Finnish,
    #[serde(rename = "no")]
    Norwegian,
    #[serde(rename = "el")]
    Greek,
    #[serde(rename = "pl")]
    Polish,
    #[serde(rename = "cs")]
    Czech,
    #[serde(rename = "hu")]
    Hungarian,
    #[serde(rename = "ro")]
    Romanian,
    #[serde(rename = "bg")]
    Bulgarian,
    #[serde(rename = "hr")]
    Croatian,
    #[serde(rename = "sk")]
    Slovak,
}

impl QwenAudioAsrLanguage {
    const fn wire_value(self) -> &'static str {
        match self {
            Self::Chinese => "zh",
            Self::English => "en",
            Self::Japanese => "ja",
            Self::Korean => "ko",
            Self::Vietnamese => "vi",
            Self::Thai => "th",
            Self::Indonesian => "id",
            Self::Malay => "ms",
            Self::Filipino => "tl",
            Self::Hindi => "hi",
            Self::Arabic => "ar",
            Self::French => "fr",
            Self::German => "de",
            Self::Spanish => "es",
            Self::Portuguese => "pt",
            Self::Russian => "ru",
            Self::Italian => "it",
            Self::Dutch => "nl",
            Self::Swedish => "sv",
            Self::Danish => "da",
            Self::Finnish => "fi",
            Self::Norwegian => "no",
            Self::Greek => "el",
            Self::Polish => "pl",
            Self::Czech => "cs",
            Self::Hungarian => "hu",
            Self::Romanian => "ro",
            Self::Bulgarian => "bg",
            Self::Croatian => "hr",
            Self::Slovak => "sk",
        }
    }
}

/// Sensitive-word replacement/removal settings supported by Qwen Audio ASR
/// Filetrans. The system list is used by default when this object is omitted.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct QwenAudioAsrSpecialWordFilter {
    pub filter_with_signed: Vec<String>,
    pub filter_with_empty: Vec<String>,
    pub system_reserved_filter: Option<bool>,
}

impl fmt::Debug for QwenAudioAsrSpecialWordFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenAudioAsrSpecialWordFilter")
            .field("filter_with_signed_count", &self.filter_with_signed.len())
            .field("filter_with_empty_count", &self.filter_with_empty.len())
            .field("system_reserved_filter", &self.system_reserved_filter)
            .finish()
    }
}

impl QwenAudioAsrSpecialWordFilter {
    fn to_wire(&self) -> Value {
        let mut filter = Map::new();
        if !self.filter_with_signed.is_empty() {
            filter.insert(
                "filter_with_signed".into(),
                json!({ "word_list": self.filter_with_signed }),
            );
        }
        if !self.filter_with_empty.is_empty() {
            filter.insert(
                "filter_with_empty".into(),
                json!({ "word_list": self.filter_with_empty }),
            );
        }
        if let Some(system_reserved_filter) = self.system_reserved_filter {
            filter.insert(
                "system_reserved_filter".into(),
                json!(system_reserved_filter),
            );
        }
        Value::Object(filter)
    }
}

/// One prior-conversation turn used as transcription context. User text maps
/// to `input_text`; an optional assistant reply maps to `text`.
#[derive(Clone, PartialEq, Eq)]
pub struct QwenAudioAsrContextTurn {
    user_text: String,
    assistant_text: Option<String>,
}

impl QwenAudioAsrContextTurn {
    pub fn new(user_text: impl Into<String>) -> Self {
        Self {
            user_text: user_text.into(),
            assistant_text: None,
        }
    }

    pub fn with_assistant_text(mut self, text: impl Into<String>) -> Self {
        self.assistant_text = Some(text.into());
        self
    }

    fn validate(&self) -> Result<(), QwenAsrError> {
        let user_len = self.user_text.chars().count();
        let assistant_len = self
            .assistant_text
            .as_ref()
            .map_or(0, |text| text.chars().count());
        if user_len + assistant_len > 400 {
            return Err(QwenAsrError::InvalidInput(
                "each Qwen Audio ASR context turn permits at most 400 combined characters".into(),
            ));
        }
        Ok(())
    }

    fn append_wire(&self, messages: &mut Vec<Value>) {
        messages.push(json!({
            "role": "user",
            "content": [{ "type": "input_text", "text": self.user_text }]
        }));
        if let Some(text) = &self.assistant_text {
            messages.push(json!({
                "role": "assistant",
                "content": [{ "type": "text", "text": text }]
            }));
        }
    }
}

impl fmt::Debug for QwenAudioAsrContextTurn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenAudioAsrContextTurn")
            .field("user_text", &"<redacted context>")
            .field(
                "assistant_text",
                &self.assistant_text.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// Controls for one Qwen Audio 3.x Filetrans request.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct QwenAudioAsrParameters {
    /// Preserve dialect expressions. Supported only by the 3.1 model.
    pub keep_dialect: Option<bool>,
    /// Reference a caller-managed precompiled hotword list.
    pub vocabulary_id: Option<String>,
    /// Inline hotwords and their weights (1–5 or 50).
    pub vocabulary: Option<std::collections::BTreeMap<String, u8>>,
    /// Zero-based audio track indices. Omission selects the first track.
    pub channel_ids: Option<Vec<u32>>,
    pub special_word_filter: Option<QwenAudioAsrSpecialWordFilter>,
    pub diarization_enabled: Option<bool>,
    pub speaker_count: Option<u8>,
    /// At most four language hints; omission leaves language detection automatic.
    pub language_hints: Option<Vec<QwenAudioAsrLanguage>>,
}

impl QwenAudioAsrParameters {
    fn validate(&self, model: QwenAudioAsrModel) -> Result<(), QwenAsrError> {
        if self.keep_dialect.is_some() && model != QwenAudioAsrModel::QwenAudio31AsrFlashFiletrans {
            return Err(QwenAsrError::InvalidInput(
                "keep_dialect is documented only for qwen-audio-3.1-asr-flash-filetrans".into(),
            ));
        }
        if let Some(vocabulary) = &self.vocabulary {
            if vocabulary
                .values()
                .any(|weight| !((1..=5).contains(weight) || *weight == 50))
                || vocabulary.values().filter(|weight| **weight == 50).count() > 50
            {
                return Err(QwenAsrError::InvalidInput(
                    "inline hotword weights must be 1-5 or 50, with at most 50 super hotwords"
                        .into(),
                ));
            }
        }
        if self.channel_ids.as_ref().is_some_and(Vec::is_empty) {
            return Err(QwenAsrError::InvalidInput(
                "channel_ids must be omitted or contain at least one track".into(),
            ));
        }
        if let Some(ids) = &self.channel_ids {
            let mut unique = std::collections::BTreeSet::new();
            if ids.iter().any(|id| !unique.insert(id)) {
                return Err(QwenAsrError::InvalidInput(
                    "channel_ids must not contain duplicates".into(),
                ));
            }
        }
        if let Some(speaker_count) = self.speaker_count {
            if self.diarization_enabled != Some(true) || !(2..=100).contains(&speaker_count) {
                return Err(QwenAsrError::InvalidInput(
                    "speaker_count requires diarization_enabled=true and must be between 2 and 100"
                        .into(),
                ));
            }
        }
        if self
            .language_hints
            .as_ref()
            .is_some_and(|languages| languages.is_empty() || languages.len() > 4)
        {
            return Err(QwenAsrError::InvalidInput(
                "language_hints must contain between one and four languages when set".into(),
            ));
        }
        if let Some(languages) = &self.language_hints {
            let mut unique = std::collections::BTreeSet::new();
            if languages.iter().any(|language| !unique.insert(*language)) {
                return Err(QwenAsrError::InvalidInput(
                    "language_hints must not contain duplicates".into(),
                ));
            }
        }
        Ok(())
    }

    fn to_wire(&self) -> Map<String, Value> {
        let mut parameters = Map::new();
        if let Some(keep_dialect) = self.keep_dialect {
            parameters.insert("keep_dialect".into(), json!(keep_dialect));
        }
        if let Some(vocabulary_id) = &self.vocabulary_id {
            parameters.insert("vocabulary_id".into(), json!(vocabulary_id));
        }
        if let Some(vocabulary) = &self.vocabulary {
            parameters.insert("vocabulary".into(), json!(vocabulary));
        }
        if let Some(channel_ids) = &self.channel_ids {
            parameters.insert("channel_id".into(), json!(channel_ids));
        }
        if let Some(special_word_filter) = &self.special_word_filter {
            parameters.insert("special_word_filter".into(), special_word_filter.to_wire());
        }
        if let Some(diarization_enabled) = self.diarization_enabled {
            parameters.insert("diarization_enabled".into(), json!(diarization_enabled));
        }
        if let Some(speaker_count) = self.speaker_count {
            parameters.insert("speaker_count".into(), json!(speaker_count));
        }
        if let Some(language_hints) = &self.language_hints {
            parameters.insert(
                "language_hints".into(),
                json!(language_hints
                    .iter()
                    .map(|language| language.wire_value())
                    .collect::<Vec<_>>()),
            );
        }
        parameters
    }
}

impl fmt::Debug for QwenAudioAsrParameters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenAudioAsrParameters")
            .field("keep_dialect", &self.keep_dialect)
            .field(
                "vocabulary_id",
                &self.vocabulary_id.as_ref().map(|_| "<set>"),
            )
            .field(
                "vocabulary_count",
                &self
                    .vocabulary
                    .as_ref()
                    .map(std::collections::BTreeMap::len),
            )
            .field("channel_ids", &self.channel_ids)
            .field(
                "special_word_filter",
                &self.special_word_filter.as_ref().map(|_| "<set>"),
            )
            .field("diarization_enabled", &self.diarization_enabled)
            .field("speaker_count", &self.speaker_count)
            .field("language_hints", &self.language_hints)
            .finish()
    }
}

/// One asynchronous Qwen Audio 3.x Filetrans request. It is deliberately
/// separate from [`QwenAsrRequest`], whose model and singular `file_url` wire
/// field remain specific to Qwen3-ASR.
#[derive(Clone, PartialEq, Eq)]
pub struct QwenAudioAsrRequest {
    pub model: QwenAudioAsrModel,
    pub file_url: String,
    pub parameters: QwenAudioAsrParameters,
    pub context: Vec<QwenAudioAsrContextTurn>,
}

impl QwenAudioAsrRequest {
    pub fn new(model: QwenAudioAsrModel, file_url: impl Into<String>) -> Self {
        Self {
            model,
            file_url: file_url.into(),
            parameters: QwenAudioAsrParameters::default(),
            context: Vec::new(),
        }
    }

    pub fn with_context_turn(mut self, turn: QwenAudioAsrContextTurn) -> Self {
        self.context.push(turn);
        self
    }

    fn validate(&self) -> Result<(), QwenAsrError> {
        validate_audio_url(&self.file_url)?;
        self.parameters.validate(self.model)?;
        if self.context.len() > 5 {
            return Err(QwenAsrError::InvalidInput(
                "Qwen Audio ASR accepts at most five context turns".into(),
            ));
        }
        for turn in &self.context {
            turn.validate()?;
        }
        Ok(())
    }

    fn to_wire(&self) -> Value {
        let mut input = Map::new();
        input.insert("file_urls".into(), json!([self.file_url]));
        if !self.context.is_empty() {
            let mut messages = Vec::new();
            for turn in &self.context {
                turn.append_wire(&mut messages);
            }
            input.insert("context".into(), Value::Array(messages));
        }
        json!({
            "model": self.model.as_str(),
            "input": input,
            "parameters": self.parameters.to_wire(),
        })
    }
}

impl fmt::Debug for QwenAudioAsrRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenAudioAsrRequest")
            .field("model", &self.model)
            .field("file_url", &"<redacted>")
            .field("parameters", &self.parameters)
            .field("context_turn_count", &self.context.len())
            .finish()
    }
}

/// A public, Model Studio-accessible audio URL. Filetrans does not upload local
/// bytes; the provider fetches this URL after the task is submitted.
#[derive(Clone, PartialEq, Eq)]
pub struct QwenAsrRequest {
    pub file_url: String,
    pub parameters: QwenAsrParameters,
}

impl QwenAsrRequest {
    pub fn new(file_url: impl Into<String>) -> Self {
        Self {
            file_url: file_url.into(),
            parameters: QwenAsrParameters::default(),
        }
    }

    fn validate(&self) -> Result<(), QwenAsrError> {
        validate_audio_url(&self.file_url)?;
        self.parameters.validate()
    }
}

impl fmt::Debug for QwenAsrRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenAsrRequest")
            .field("file_url", &"<redacted>")
            .field("parameters", &self.parameters)
            .finish()
    }
}

/// A task reference is bound to the exact account profile, region, and
/// workspace used at submission time. It contains no credential.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenAsrTaskRef {
    scope: QwenAsrScope,
    task_id: String,
}

impl QwenAsrTaskRef {
    pub fn new(scope: QwenAsrScope, task_id: impl Into<String>) -> Result<Self, QwenAsrError> {
        scope.validate()?;
        let task_id = task_id.into();
        if !valid_task_id(&task_id) {
            return Err(QwenAsrError::InvalidInput(
                "task ID must contain only ASCII letters, digits, hyphens, or underscores".into(),
            ));
        }
        Ok(Self { scope, task_id })
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn scope(&self) -> &QwenAsrScope {
        &self.scope
    }
}

/// Task lifecycle values documented by the Model Studio async task API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenAsrTaskStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    /// The API says the task does not exist or its status is unknown.
    Unknown,
    Other(String),
}

impl QwenAsrTaskStatus {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Pending => "PENDING",
            Self::Running => "RUNNING",
            Self::Succeeded => "SUCCEEDED",
            Self::Failed => "FAILED",
            Self::Unknown => "UNKNOWN",
            Self::Other(status) => status,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed)
    }

    fn parse(status: String) -> Self {
        match status.as_str() {
            "PENDING" => Self::Pending,
            "RUNNING" => Self::Running,
            "SUCCEEDED" => Self::Succeeded,
            "FAILED" => Self::Failed,
            "UNKNOWN" => Self::Unknown,
            _ => Self::Other(status),
        }
    }
}

/// Short-lived signed URL for the JSON transcription artifact. Its URL is
/// intentionally hidden from Debug because its query string grants access.
#[derive(Clone, PartialEq, Eq)]
pub struct QwenAsrResultRef {
    task: QwenAsrTaskRef,
    url: String,
}

impl QwenAsrResultRef {
    pub fn task(&self) -> &QwenAsrTaskRef {
        &self.task
    }
}

impl fmt::Debug for QwenAsrResultRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenAsrResultRef")
            .field("task", &self.task)
            .field("url", &"<redacted short-lived URL>")
            .finish()
    }
}

/// One observed state of an asynchronous task. `get_task` performs exactly
/// one GET; callers choose when and whether to query again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenAsrTask {
    reference: QwenAsrTaskRef,
    pub request_id: Option<String>,
    pub status: QwenAsrTaskStatus,
    pub submit_time: Option<String>,
    pub scheduled_time: Option<String>,
    pub end_time: Option<String>,
    pub duration_seconds: Option<u64>,
    pub code: Option<String>,
    pub message: Option<String>,
    pub result: Option<QwenAsrResultRef>,
    /// Qwen Audio Filetrans reports an individual result status separately
    /// from the overall task status. Qwen3-ASR leaves this unset.
    pub subtask_status: Option<QwenAsrTaskStatus>,
    /// Failure details for the single Qwen Audio Filetrans subtask.
    pub subtask_code: Option<String>,
    pub subtask_message: Option<String>,
    /// Per-task counts included by the Qwen Audio Filetrans query response.
    pub task_metrics: Option<QwenAsrTaskMetrics>,
}

impl QwenAsrTask {
    pub fn reference(&self) -> &QwenAsrTaskRef {
        &self.reference
    }

    pub fn result(&self) -> Option<&QwenAsrResultRef> {
        self.result.as_ref()
    }
}

/// Aggregate result counts returned by the Qwen Audio Filetrans task API.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QwenAsrTaskMetrics {
    pub total: Option<u64>,
    pub succeeded: Option<u64>,
    pub failed: Option<u64>,
}

/// Typed transcription document downloaded from the task's temporary result
/// URL. All times are provider millisecond offsets within the audio.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct QwenAsrTranscription {
    /// Qwen Audio Filetrans calls this metadata object `properties`; the
    /// Qwen3-ASR task response uses `audio_info`.
    #[serde(alias = "properties")]
    #[serde(default)]
    pub audio_info: Option<QwenAsrAudioInfo>,
    #[serde(default)]
    pub transcripts: Vec<QwenAsrTrackTranscript>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct QwenAsrAudioInfo {
    #[serde(alias = "audio_format")]
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub sample_rate: Option<u32>,
    #[serde(default)]
    pub original_sampling_rate: Option<u32>,
    #[serde(default)]
    pub original_duration_in_milliseconds: Option<u64>,
    #[serde(default)]
    pub channels: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct QwenAsrTrackTranscript {
    pub channel_id: u32,
    #[serde(default)]
    pub content_duration_in_milliseconds: Option<u64>,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub sentences: Vec<QwenAsrSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct QwenAsrSegment {
    #[serde(rename = "begin_time")]
    pub start_ms: u64,
    #[serde(rename = "end_time")]
    pub end_ms: u64,
    pub text: String,
    #[serde(default)]
    pub sentence_id: Option<u64>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub emotion: Option<String>,
    #[serde(default)]
    pub speaker_id: Option<u64>,
    #[serde(default)]
    pub words: Vec<QwenAsrWord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct QwenAsrWord {
    #[serde(rename = "begin_time")]
    pub start_ms: u64,
    #[serde(rename = "end_time")]
    pub end_ms: u64,
    pub text: String,
    #[serde(default)]
    pub punctuation: Option<String>,
}

/// How far a submission may have progressed at Model Studio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenAsrSubmissionOutcome {
    NotSent,
    Rejected,
    Unknown,
    Accepted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenAsrOperation {
    Submit,
    Query,
    FetchTranscription,
}

#[derive(Debug, Error)]
pub enum QwenAsrError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("Qwen ASR {operation:?} failed before receiving an HTTP response: {source}")]
    OperationError {
        operation: QwenAsrOperation,
        #[source]
        source: LlmError,
    },
    #[error("invalid Qwen ASR input: {0}")]
    InvalidInput(String),
    #[error("Qwen ASR task belongs to a different account, region, or workspace")]
    ScopeMismatch,
    #[error("Qwen ASR submission outcome is unknown: {source}")]
    SubmissionOutcomeUnknown {
        #[source]
        source: LlmError,
        request_id: Option<String>,
    },
    #[error("Qwen ASR submission was rejected with HTTP {status}: {message}")]
    SubmissionRejected {
        status: u16,
        code: Option<String>,
        message: String,
        request_id: Option<String>,
    },
    #[error("Qwen ASR accepted the submission but its response was invalid: {message}")]
    SubmissionAcceptedInvalidResponse {
        message: String,
        request_id: Option<String>,
    },
    #[error("Qwen ASR {operation:?} returned HTTP {status}: {message}")]
    Provider {
        operation: QwenAsrOperation,
        status: u16,
        code: Option<String>,
        message: String,
        request_id: Option<String>,
    },
    #[error("Qwen ASR {operation:?} returned an invalid response: {message}")]
    InvalidResponse {
        operation: QwenAsrOperation,
        message: String,
    },
}

impl QwenAsrError {
    /// Classify retry safety for a failed submit call. Query and artifact-fetch
    /// errors return `None` because they cannot create a duplicate task.
    pub fn submission_outcome(&self) -> Option<QwenAsrSubmissionOutcome> {
        match self {
            Self::InvalidInput(_) | Self::ScopeMismatch | Self::Llm(_) => {
                Some(QwenAsrSubmissionOutcome::NotSent)
            }
            Self::SubmissionOutcomeUnknown { .. } => Some(QwenAsrSubmissionOutcome::Unknown),
            Self::SubmissionRejected { .. } => Some(QwenAsrSubmissionOutcome::Rejected),
            Self::SubmissionAcceptedInvalidResponse { .. } => {
                Some(QwenAsrSubmissionOutcome::Accepted)
            }
            Self::OperationError { .. } | Self::Provider { .. } | Self::InvalidResponse { .. } => {
                None
            }
        }
    }
}

/// Qwen3-ASR-Flash-Filetrans over its regional Model Studio task API.
#[derive(Clone)]
pub struct QwenAsrService<'a> {
    binding: Option<crate::providers::binding::ProviderBinding>,
    transport: &'a dyn Transport,
    scope: QwenAsrScope,
}

impl<'a> QwenAsrService<'a> {
    fn pin(&self) -> Result<Self, crate::protocol::LlmError> {
        let mut pinned = self.clone();
        pinned.binding = self
            .binding
            .as_ref()
            .map(crate::providers::binding::ProviderBinding::pinned)
            .transpose()?;
        Ok(pinned)
    }

    pub(crate) fn with_binding(
        mut self,
        binding: &crate::providers::binding::ProviderBinding,
    ) -> Self {
        self.binding = Some(binding.clone());
        self
    }

    pub fn new(transport: &'a dyn Transport, scope: QwenAsrScope) -> Result<Self, QwenAsrError> {
        scope.validate()?;
        Ok(Self {
            binding: None,
            transport,
            scope,
        })
    }

    pub fn scope(&self) -> &QwenAsrScope {
        &self.scope
    }

    /// Submit exactly one Filetrans task. Network failures after dispatch are
    /// reported as unknown; callers must not blindly replay such an error.
    pub async fn submit(
        &self,
        request: &QwenAsrRequest,
        options: &RequestOptions,
    ) -> Result<QwenAsrTask, QwenAsrError> {
        let pinned_service = self.pin()?;
        pinned_service.scope.validate()?;
        request.validate()?;
        pinned_service.validate_options_scope(options)?;
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "Qwen ASR requires a Model Studio API key for the selected region".into(),
            })?;

        let body = json!({
            "model": QWEN3_ASR_FILETRANS_MODEL,
            "input": { "file_url": request.file_url },
            // Model Studio documents parameters as optional, but requires the
            // object for the newer Singapore workspace endpoint. Always send it.
            "parameters": request.parameters.to_wire(),
        });
        pinned_service
            .send_submission_body(body, options, credential.expose_secret())
            .await
    }

    /// Submit one Qwen Audio 3.x ASR Filetrans job using its distinct
    /// `file_urls` request contract. The shared task query/result lifecycle is
    /// the same, while Qwen3-ASR remains on [`Self::submit`].
    pub async fn submit_audio_filetrans(
        &self,
        request: &QwenAudioAsrRequest,
        options: &RequestOptions,
    ) -> Result<QwenAsrTask, QwenAsrError> {
        let pinned_service = self.pin()?;
        pinned_service.scope.validate()?;
        request.validate()?;
        pinned_service.validate_options_scope(options)?;
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "Qwen ASR requires a Model Studio API key for the selected region".into(),
            })?;
        pinned_service
            .send_submission_body(request.to_wire(), options, credential.expose_secret())
            .await
    }

    async fn send_submission_body(
        &self,
        body: Value,
        options: &RequestOptions,
        api_key: &str,
    ) -> Result<QwenAsrTask, QwenAsrError> {
        let body = serde_json::to_vec(&body).map_err(|error| {
            QwenAsrError::InvalidInput(format!("could not encode request: {error}"))
        })?;
        let response = HttpExecutor::new(self.transport)
            .execute_bounded(
                HttpRequest {
                    http1_header_layout: None,
                    method: "POST".into(),
                    url: format!("{}{}", self.scope.api_base(), SUBMIT_PATH),
                    headers: vec![
                        ("authorization".into(), format!("Bearer {api_key}")),
                        ("content-type".into(), "application/json".into()),
                        ("accept".into(), "application/json".into()),
                        ("x-dashscope-async".into(), "enable".into()),
                    ],
                    body: Bytes::from(body),
                    timeout: options.total_timeout,
                },
                MAX_CONTROL_RESPONSE_BYTES,
            )
            .await
            .map_err(|source| QwenAsrError::SubmissionOutcomeUnknown {
                source,
                request_id: None,
            })?;

        if !(200..300).contains(&response.status) {
            let (code, message, request_id) = error_details(&response.body);
            if (400..500).contains(&response.status) {
                return Err(QwenAsrError::SubmissionRejected {
                    status: response.status,
                    code,
                    message,
                    request_id,
                });
            }
            return Err(QwenAsrError::SubmissionOutcomeUnknown {
                source: LlmError::ProviderInternal {
                    message: format!(
                        "Model Studio returned HTTP {} after task submission: {}",
                        response.status, message
                    ),
                },
                request_id,
            });
        }

        let value: Value = serde_json::from_slice(&response.body).map_err(|error| {
            QwenAsrError::SubmissionAcceptedInvalidResponse {
                message: format!("response JSON could not be decoded: {error}"),
                request_id: None,
            }
        })?;
        let request_id = value
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::to_owned);
        decode_task(value, &self.scope, None, QwenAsrOperation::Submit).map_err(|error| match error
        {
            QwenAsrError::InvalidResponse { message, .. } => {
                QwenAsrError::SubmissionAcceptedInvalidResponse {
                    message,
                    request_id,
                }
            }
            other => other,
        })
    }

    /// Query a submitted task once. This method never waits or polls.
    pub async fn get_task(
        &self,
        reference: &QwenAsrTaskRef,
        options: &RequestOptions,
    ) -> Result<QwenAsrTask, QwenAsrError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(reference)?;
        pinned_service.validate_options_scope(options)?;
        let credential = options
            .credential
            .as_ref()
            .ok_or_else(|| LlmError::Authentication {
                message: "Qwen ASR requires a Model Studio API key for the selected region".into(),
            })
            .map_err(|source| QwenAsrError::OperationError {
                operation: QwenAsrOperation::Query,
                source,
            })?;
        let response = HttpExecutor::new(pinned_service.transport)
            .execute_bounded(
                HttpRequest {
                    http1_header_layout: None,
                    method: "GET".into(),
                    url: format!(
                        "{}{}{}",
                        pinned_service.scope.api_base(),
                        TASK_PATH,
                        reference.task_id
                    ),
                    headers: vec![
                        (
                            "authorization".into(),
                            format!("Bearer {}", credential.expose_secret()),
                        ),
                        ("accept".into(), "application/json".into()),
                    ],
                    body: Bytes::new(),
                    timeout: options.total_timeout,
                },
                MAX_CONTROL_RESPONSE_BYTES,
            )
            .await
            .map_err(|source| QwenAsrError::OperationError {
                operation: QwenAsrOperation::Query,
                source,
            })?;
        if !(200..300).contains(&response.status) {
            let (code, message, request_id) = error_details(&response.body);
            return Err(QwenAsrError::Provider {
                operation: QwenAsrOperation::Query,
                status: response.status,
                code,
                message,
                request_id,
            });
        }
        let value: Value = serde_json::from_slice(&response.body).map_err(|error| {
            QwenAsrError::InvalidResponse {
                operation: QwenAsrOperation::Query,
                message: format!("response JSON could not be decoded: {error}"),
            }
        })?;
        decode_task(
            value,
            &pinned_service.scope,
            Some(reference),
            QwenAsrOperation::Query,
        )
    }

    /// Download the result document for a successful task. Model Studio's URL
    /// is temporary (documented validity: 24 hours) and already signed, so the
    /// account API key is deliberately never sent to this storage host. The
    /// optional request timeout is still honored.
    pub async fn fetch_transcription(
        &self,
        result: &QwenAsrResultRef,
        options: &RequestOptions,
    ) -> Result<QwenAsrTranscription, QwenAsrError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_reference(&result.task)?;
        let response = HttpExecutor::new(pinned_service.transport)
            .execute_bounded(
                HttpRequest {
                    http1_header_layout: None,
                    method: "GET".into(),
                    url: result.url.clone(),
                    headers: vec![("accept".into(), "application/json".into())],
                    body: Bytes::new(),
                    timeout: options.total_timeout,
                },
                MAX_TRANSCRIPTION_BYTES,
            )
            .await
            .map_err(|source| QwenAsrError::OperationError {
                operation: QwenAsrOperation::FetchTranscription,
                source,
            })?;
        if !(200..300).contains(&response.status) {
            let (code, message, request_id) = error_details(&response.body);
            return Err(QwenAsrError::Provider {
                operation: QwenAsrOperation::FetchTranscription,
                status: response.status,
                code,
                message,
                request_id,
            });
        }
        let native: Value = serde_json::from_slice(&response.body).map_err(|error| {
            QwenAsrError::InvalidResponse {
                operation: QwenAsrOperation::FetchTranscription,
                message: format!("transcription JSON could not be decoded: {error}"),
            }
        })?;
        let has_audio_metadata_object = native
            .get("audio_info")
            .or_else(|| native.get("properties"))
            .is_some_and(Value::is_object);
        let has_transcripts_array = native.get("transcripts").is_some_and(Value::is_array);
        if !has_audio_metadata_object || !has_transcripts_array {
            return Err(invalid_response(
                QwenAsrOperation::FetchTranscription,
                "transcription result must contain an audio_info/properties object and a transcripts array",
            ));
        }
        let transcript: QwenAsrTranscription =
            serde_json::from_value(native).map_err(|error| QwenAsrError::InvalidResponse {
                operation: QwenAsrOperation::FetchTranscription,
                message: format!("transcription JSON could not be decoded: {error}"),
            })?;
        validate_transcription(&transcript)?;
        Ok(transcript)
    }

    fn validate_options_scope(&self, options: &RequestOptions) -> Result<(), QwenAsrError> {
        if options
            .account_scope
            .as_deref()
            .is_some_and(|scope| scope != self.scope.account_scope)
        {
            return Err(QwenAsrError::ScopeMismatch);
        }
        Ok(())
    }

    fn validate_reference(&self, reference: &QwenAsrTaskRef) -> Result<(), QwenAsrError> {
        self.scope.validate()?;
        if reference.scope != self.scope || !valid_task_id(&reference.task_id) {
            return Err(QwenAsrError::ScopeMismatch);
        }
        Ok(())
    }
}

fn decode_task(
    value: Value,
    scope: &QwenAsrScope,
    expected_reference: Option<&QwenAsrTaskRef>,
    operation: QwenAsrOperation,
) -> Result<QwenAsrTask, QwenAsrError> {
    let request_id = value
        .get("request_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let output = value
        .get("output")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_response(operation, "missing output object"))?;
    let task_id = output
        .get("task_id")
        .and_then(Value::as_str)
        .filter(|id| valid_task_id(id))
        .ok_or_else(|| invalid_response(operation, "missing or invalid task_id"))?;
    if expected_reference.is_some_and(|reference| reference.task_id != task_id) {
        return Err(invalid_response(
            operation,
            "query response task_id does not match the requested task",
        ));
    }
    let status = output
        .get("task_status")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_response(operation, "missing task_status"))?;
    let reference = match expected_reference {
        Some(reference) => reference.clone(),
        None => QwenAsrTaskRef::new(scope.clone(), task_id.to_owned())?,
    };
    let status = QwenAsrTaskStatus::parse(status.to_owned());
    let audio_result = output.get("results");
    let subtask = match audio_result {
        Some(Value::Array(results)) if results.len() <= 1 => results.first(),
        Some(Value::Array(_)) => {
            return Err(invalid_response(
                operation,
                "this Qwen Audio Filetrans adapter accepts one file and cannot decode multiple subtask results",
            ));
        }
        Some(_) => {
            return Err(invalid_response(
                operation,
                "output.results must be an array",
            ));
        }
        None => None,
    };
    if subtask.is_some_and(|item| !item.is_object()) {
        return Err(invalid_response(
            operation,
            "output.results entries must be objects",
        ));
    }
    let subtask_status = subtask
        .and_then(|item| item.get("subtask_status"))
        .and_then(Value::as_str)
        .map(|status| QwenAsrTaskStatus::parse(status.to_owned()));
    let subtask_code = subtask
        .and_then(|item| item.get("code"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let subtask_message = subtask
        .and_then(|item| item.get("message"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let result = if audio_result.is_some() {
        if status == QwenAsrTaskStatus::Succeeded
            && subtask_status == Some(QwenAsrTaskStatus::Succeeded)
        {
            subtask
                .and_then(|item| item.get("transcription_url"))
                .and_then(Value::as_str)
                .map(|url| {
                    validate_result_url(url, scope.region)
                        .map_err(|error| match error {
                            QwenAsrError::InvalidResponse { message, .. } => {
                                invalid_response(operation, message)
                            }
                            other => other,
                        })
                        .map(|url| QwenAsrResultRef {
                            task: reference.clone(),
                            url,
                        })
                })
                .transpose()?
        } else {
            None
        }
    } else if status == QwenAsrTaskStatus::Succeeded {
        output
            .get("result")
            .and_then(Value::as_object)
            .and_then(|result| result.get("transcription_url"))
            .and_then(Value::as_str)
            .map(|url| {
                validate_result_url(url, scope.region)
                    .map_err(|error| match error {
                        QwenAsrError::InvalidResponse { message, .. } => {
                            invalid_response(operation, message)
                        }
                        other => other,
                    })
                    .map(|url| QwenAsrResultRef {
                        task: reference.clone(),
                        url,
                    })
            })
            .transpose()?
    } else {
        None
    };
    Ok(QwenAsrTask {
        reference,
        request_id,
        status,
        submit_time: output
            .get("submit_time")
            .and_then(Value::as_str)
            .map(str::to_owned),
        scheduled_time: output
            .get("scheduled_time")
            .or_else(|| output.get("schedule_time"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        end_time: output
            .get("end_time")
            .and_then(Value::as_str)
            .map(str::to_owned),
        duration_seconds: value
            .get("usage")
            .and_then(|usage| usage.get("seconds").or_else(|| usage.get("duration")))
            .and_then(Value::as_u64),
        code: output
            .get("code")
            .and_then(Value::as_str)
            .map(str::to_owned),
        message: output
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_owned),
        result,
        subtask_status,
        subtask_code,
        subtask_message,
        task_metrics: output
            .get("task_metrics")
            .map(|metrics| QwenAsrTaskMetrics {
                total: metrics.get("TOTAL").and_then(Value::as_u64),
                succeeded: metrics.get("SUCCEEDED").and_then(Value::as_u64),
                failed: metrics.get("FAILED").and_then(Value::as_u64),
            }),
    })
}

fn validate_audio_url(value: &str) -> Result<(), QwenAsrError> {
    if value.trim() != value || value.is_empty() || value.chars().any(char::is_control) {
        return Err(QwenAsrError::InvalidInput(
            "audio file URL must be non-empty and contain no surrounding whitespace or control characters".into(),
        ));
    }
    let url = Url::parse(value).map_err(|_| {
        QwenAsrError::InvalidInput("audio file must use a public HTTP(S) or OSS URL".into())
    })?;
    let supported_scheme = matches!(url.scheme(), "http" | "https" | "oss");
    if !supported_scheme
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(QwenAsrError::InvalidInput(
            "audio file must use a public HTTP(S) or OSS URL without userinfo or a fragment".into(),
        ));
    }
    Ok(())
}

fn validate_result_url(value: &str, region: QwenAsrRegion) -> Result<String, QwenAsrError> {
    let mut url = Url::parse(value).map_err(|_| {
        invalid_response(
            QwenAsrOperation::Query,
            "transcription result URL is invalid",
        )
    })?;
    let host = url.host_str().unwrap_or_default();
    let expected_suffix = region.result_suffix();
    let bucket_label = host.strip_suffix(expected_suffix).unwrap_or_default();
    if !matches!(url.scheme(), "http" | "https")
        || !bucket_label.starts_with("dashscope-result-")
        || url.port_or_known_default() != Some(if url.scheme() == "http" { 80 } else { 443 })
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid_response(
            QwenAsrOperation::Query,
            "transcription result URL is outside the task's regional Model Studio result host",
        ));
    }
    // The API examples have historically returned HTTP OSS URLs. Upgrade the
    // same signed object URL to HTTPS before making a request; never send the
    // time-limited query signature over plaintext.
    if url.scheme() == "http" {
        url.set_scheme("https").map_err(|_| {
            invalid_response(QwenAsrOperation::Query, "could not secure result URL")
        })?;
    }
    Ok(url.into())
}

fn valid_task_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn validate_transcription(transcription: &QwenAsrTranscription) -> Result<(), QwenAsrError> {
    for track in &transcription.transcripts {
        for sentence in &track.sentences {
            if sentence.end_ms < sentence.start_ms {
                return Err(invalid_response(
                    QwenAsrOperation::FetchTranscription,
                    "sentence end_time precedes begin_time",
                ));
            }
            for word in &sentence.words {
                if word.end_ms < word.start_ms {
                    return Err(invalid_response(
                        QwenAsrOperation::FetchTranscription,
                        "word end_time precedes begin_time",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn error_details(body: &[u8]) -> (Option<String>, String, Option<String>) {
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        let code = value
            .get("code")
            .or_else(|| value.pointer("/output/code"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let message = value
            .get("message")
            .or_else(|| value.pointer("/output/message"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| String::from_utf8_lossy(body).into_owned());
        let request_id = value
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::to_owned);
        (code, message, request_id)
    } else {
        (None, String::from_utf8_lossy(body).into_owned(), None)
    }
}

fn invalid_response(operation: QwenAsrOperation, message: impl Into<String>) -> QwenAsrError {
    QwenAsrError::InvalidResponse {
        operation,
        message: message.into(),
    }
}
