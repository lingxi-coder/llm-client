use super::*;
use crate::protocol::{AuthStrategy, ProtocolFamily, ProviderProfile, Region, ServiceSetting};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioVoiceDescriptor {
    pub id: String,
    pub label: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioModelDescriptor {
    /// None means the native endpoint has no model selector (xAI REST TTS).
    pub id: Option<String>,
    pub voices: Vec<AudioVoiceDescriptor>,
    pub default_voice: Option<String>,
    pub voice_discovery_required: bool,
    pub formats: Vec<AudioFormat>,
    /// Declared default PCM output geometry; None means device playback must
    /// reject the raw route before dispatch rather than guess a rate/layout.
    pub pcm_sample_rate_hz: Option<u32>,
    pub pcm_channels: Option<u8>,
    pub pcm_bits_per_sample: Option<u8>,
    pub max_input_bytes: Option<u64>,
    pub max_text_characters: Option<usize>,
    pub max_duration_seconds: Option<u32>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioOperationDescriptor {
    pub operation: AudioOperation,
    /// True only when this facade implements the common operation.
    pub common_adapter: bool,
    pub default_model: Option<String>,
    pub models: Vec<AudioModelDescriptor>,
    /// Exact provider service entry point for advanced/provider-only operations.
    pub advanced_service: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioCapabilities {
    pub profile_name: String,
    pub provider_id: String,
    pub account_scope: String,
    pub region: Region,
    /// API account region, distinct from the client presentation region.
    pub service_region: String,
    pub required_scope_fields: Vec<String>,
    pub auth: AuthStrategy,
    pub operations: Vec<AudioOperationDescriptor>,
    /// Protocol normalization must support the complete Harness conversation contract.
    pub agent_conversation: bool,
    /// Contract enabled by the typed agent convenience connector; the connected
    /// control still reports its actual configuration-specific capabilities.
    pub native_realtime_contract: Option<crate::realtime::RealtimeCapabilities>,
}
impl AudioCapabilities {
    pub fn operation(&self, operation: AudioOperation) -> Option<&AudioOperationDescriptor> {
        self.operations
            .iter()
            .find(|item| item.operation == operation)
    }
    pub fn supports(&self, operation: AudioOperation) -> bool {
        self.operation(operation)
            .is_some_and(|item| item.common_adapter && !item.models.is_empty())
    }
    pub fn model(
        &self,
        operation: AudioOperation,
        requested: Option<&str>,
    ) -> Result<&AudioModelDescriptor, AudioError> {
        let descriptor = self.operation(operation).ok_or_else(|| {
            AudioError::unsupported(format!(
                "{} has no declared {operation:?} operation",
                self.profile_name
            ))
        })?;
        let id = requested.or(descriptor.default_model.as_deref());
        descriptor
            .models
            .iter()
            .find(|m| m.id.as_deref() == id)
            .ok_or_else(|| {
                AudioError::unsupported(format!(
                    "audio model {id:?} is not declared for {operation:?} on {}",
                    self.profile_name
                ))
            })
    }
}
fn model(
    id: &str,
    formats: &[AudioFormat],
    max_bytes: Option<u64>,
    max_chars: Option<usize>,
) -> AudioModelDescriptor {
    AudioModelDescriptor {
        id: (!id.is_empty()).then(|| id.into()),
        voices: vec![],
        default_voice: None,
        voice_discovery_required: false,
        formats: formats.into(),
        pcm_sample_rate_hz: formats.contains(&AudioFormat::Pcm16Le).then_some(24_000),
        pcm_channels: formats.contains(&AudioFormat::Pcm16Le).then_some(1),
        pcm_bits_per_sample: formats.contains(&AudioFormat::Pcm16Le).then_some(16),
        max_input_bytes: max_bytes,
        max_text_characters: max_chars,
        max_duration_seconds: None,
    }
}
fn operation(
    kind: AudioOperation,
    models: Vec<AudioModelDescriptor>,
    default: Option<&str>,
    common: bool,
    service: &str,
) -> AudioOperationDescriptor {
    AudioOperationDescriptor {
        operation: kind,
        models,
        default_model: default.map(str::to_owned),
        common_adapter: common,
        advanced_service: Some(service.into()),
    }
}
fn voices(
    mut model: AudioModelDescriptor,
    names: &[&str],
    default: Option<&str>,
    discovery: bool,
) -> AudioModelDescriptor {
    model.voices = names
        .iter()
        .map(|id| AudioVoiceDescriptor {
            id: (*id).into(),
            label: (*id).into(),
        })
        .collect();
    model.default_voice = default.map(str::to_owned);
    model.voice_discovery_required = discovery;
    model
}

pub(crate) fn catalog(
    profile: &ProviderProfile,
    route: &AudioRoute,
    region: Region,
) -> Result<AudioCapabilities, AudioError> {
    use AudioFormat::*;
    use AudioOperation::*;
    if route.profile_name.trim().is_empty()
        || route.account_scope.trim().is_empty()
        || route.account_scope.chars().any(char::is_control)
    {
        return Err(AudioError::local(
            "audio requires an exact profile and nonempty non-secret account scope",
        ));
    }
    if !profile.supports_region(region) {
        return Err(AudioError::local(
            "audio profile is unavailable in this client's region",
        ));
    }
    let mut c = AudioCapabilities {
        profile_name: profile.profile_name.clone(),
        provider_id: profile.provider_id.to_string(),
        account_scope: route.account_scope.clone(),
        region,
        service_region: String::new(),
        required_scope_fields: vec![],
        auth: profile.auth,
        operations: vec![],
        agent_conversation: false,
        native_realtime_contract: None,
    };
    // Restricted OAuth/chat subscriptions are never upgraded into paid API audio.
    if matches!(
        profile.auth,
        AuthStrategy::ChatGptPlan
            | AuthStrategy::ChatGptOAuth
            | AuthStrategy::CopilotBearer
            | AuthStrategy::AwsSigV4
            | AuthStrategy::AzureToken
    ) {
        return Ok(c);
    }
    let url = url::Url::parse(&profile.base_url)
        .map_err(|_| AudioError::local("invalid audio profile base URL"))?;
    if profile.provider_id.as_str() != "openai"
        && (url.scheme() != "https"
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some())
    {
        return Ok(c);
    }
    let host = url.host_str().unwrap_or_default().to_owned();
    c.service_region = match host.as_str() {
        "dashscope.aliyuncs.com" => "beijing",
        "dashscope-intl.aliyuncs.com" => "singapore",
        "api.minimax.cn" | "api.minimaxi.com" | "open.bigmodel.cn" => "mainland_china",
        "api.minimax.io" | "api.z.ai" => "international",
        _ => "global",
    }
    .into();
    match profile.provider_id.as_str() {
        "openai" => {
            c.auth = AuthStrategy::Bearer;
            if host == "api.openai.com"
                && matches!(profile.auth, AuthStrategy::ApiKey | AuthStrategy::Bearer)
            {
                c.operations.push(operation(
                    NativeRealtime,
                    vec![voices(
                        model("gpt-realtime-2.1", &[Pcm16Le, Mulaw, Alaw], None, None),
                        &[
                            "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse",
                            "marin", "cedar",
                        ],
                        Some("marin"),
                        false,
                    )],
                    Some("gpt-realtime-2.1"),
                    false,
                    "OpenAiClient::connect_agent_realtime",
                ));
                c.native_realtime_contract = Some(crate::realtime::RealtimeCapabilities {
                    history_import: true,
                    tools: true,
                    input_transcription: true,
                    output_transcription: true,
                    usage: true,
                    interruption: true,
                    audio_truncation: true,
                    session_resumption: false,
                });
                c.agent_conversation = true;
            }
            if !matches!(profile.audio, ServiceSetting::Enabled(_)) {
                return Ok(c);
            }
            c.operations.push(operation(
                FileTranscription,
                [
                    "gpt-transcribe",
                    "gpt-4o-transcribe",
                    "gpt-4o-mini-transcribe",
                    "gpt-4o-mini-transcribe-2025-12-15",
                    "gpt-4o-transcribe-diarize",
                    "whisper-1",
                ]
                .into_iter()
                .map(|id| model(id, &[Wav, Mp3, Flac], Some(25_000_000), None))
                .collect(),
                Some("gpt-4o-mini-transcribe"),
                true,
                "OpenAiClient::audio",
            ));
            if !matches!(&profile.audio,ServiceSetting::Enabled(route) if route.speech_endpoint.is_some())
            {
                return Ok(c);
            }
            let speech = [
                "gpt-4o-mini-tts",
                "gpt-4o-mini-tts-2025-12-15",
                "tts-1",
                "tts-1-hd",
            ]
            .into_iter()
            .map(|id| {
                let names: &[&str] = if id.starts_with("tts-") {
                    &[
                        "alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer",
                    ]
                } else {
                    &[
                        "alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage",
                        "shimmer", "verse", "marin", "cedar",
                    ]
                };
                voices(
                    model(id, &[Mp3, Opus, Aac, Flac, Wav, Pcm16Le], None, Some(4096)),
                    names,
                    Some("alloy"),
                    false,
                )
            })
            .collect();
            c.operations.push(operation(
                Synthesis,
                speech,
                Some("gpt-4o-mini-tts"),
                true,
                "OpenAiClient::audio",
            ));
        }
        "xai"
            if host == "api.x.ai"
                && matches!(profile.auth, AuthStrategy::ApiKey | AuthStrategy::Bearer) =>
        {
            c.operations.push(operation(
                FileTranscription,
                ["grok-voice-transcribe-1.0", "grok-voice-transcribe-2.0"]
                    .into_iter()
                    .map(|id| {
                        model(
                            id,
                            &[Wav, Mp3, Flac, Pcm16Le, Mulaw, Alaw],
                            Some(500_000_000),
                            None,
                        )
                    })
                    .collect(),
                Some("grok-voice-transcribe-2.0"),
                true,
                "XaiClient::audio",
            ));
            c.operations.push(operation(
                Synthesis,
                vec![voices(
                    model("", &[Mp3, Wav, Pcm16Le, Mulaw, Alaw], None, Some(60_000)),
                    &[],
                    None,
                    true,
                )],
                None,
                true,
                "XaiClient::audio",
            ));
            for (kind, service) in [
                (LiveAsr, "XaiClient::stt"),
                (IncrementalSynthesis, "XaiClient::streaming_tts"),
                (NativeRealtime, "XaiClient::realtime"),
            ] {
                c.operations
                    .push(operation(kind, vec![], None, false, service));
            }
        }
        "google"
            if profile.protocol == ProtocolFamily::VertexGemini
                && (host == "aiplatform.googleapis.com"
                    || host.ends_with("-aiplatform.googleapis.com"))
                && matches!(
                    profile.auth,
                    AuthStrategy::GcpToken | AuthStrategy::OAuthBearer | AuthStrategy::Bearer
                ) =>
        {
            c.required_scope_fields = vec!["signing.project".into(), "signing.region".into()];
            let location = profile
                .signing
                .as_ref()
                .and_then(|s| s.region.as_deref())
                .unwrap_or("global");
            c.service_region = location.into();
            if !matches!(
                location,
                "global"
                    | "northamerica-northeast1"
                    | "europe-central2"
                    | "europe-north1"
                    | "europe-southwest1"
                    | "europe-west1"
                    | "europe-west4"
                    | "us-central1"
                    | "us-east1"
                    | "us-east4"
                    | "us-east5"
                    | "us-south1"
                    | "us-west1"
                    | "us-west4"
            ) {
                return Ok(c);
            }
            let mut ids = vec![
                "gemini-2.5-flash-tts",
                "gemini-2.5-pro-tts",
                "gemini-2.5-flash-lite-preview-tts",
            ];
            if location == "global" {
                ids.push("gemini-3.1-flash-tts-preview");
            }
            if location == "northamerica-northeast1" {
                ids.retain(|id| *id != "gemini-2.5-pro-tts");
            }
            c.operations.push(operation(
                Synthesis,
                ids.into_iter()
                    .map(|id| {
                        voices(
                            model(id, &[Pcm16Le], None, None),
                            &["Kore"],
                            Some("Kore"),
                            true,
                        )
                    })
                    .collect(),
                Some("gemini-2.5-flash-tts"),
                true,
                "GoogleClient::vertex_speech",
            ));
        }
        "google"
            if host == "generativelanguage.googleapis.com"
                && profile.auth == AuthStrategy::ApiKey =>
        {
            c.operations.push(operation(
                Synthesis,
                ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"]
                    .into_iter()
                    .map(|id| {
                        voices(
                            model(id, &[Wav, Pcm16Le, Mulaw, Alaw], None, Some(500_000)),
                            &["Kore"],
                            Some("Kore"),
                            true,
                        )
                    })
                    .collect(),
                Some("gemini-3.8-flash-tts"),
                true,
                "GoogleClient::speech",
            ));
            c.operations.push(operation(
                NativeRealtime,
                vec![voices(
                    model("gemini-3.8-live", &[Pcm16Le], None, None),
                    &[
                        "Zephyr",
                        "Puck",
                        "Charon",
                        "Kore",
                        "Fenrir",
                        "Leda",
                        "Orus",
                        "Aoede",
                        "Callirrhoe",
                        "Autonoe",
                        "Enceladus",
                        "Iapetus",
                        "Umbriel",
                        "Algieba",
                        "Despina",
                        "Erinome",
                        "Algenib",
                        "Rasalgethi",
                        "Laomedeia",
                        "Achernar",
                        "Alnilam",
                        "Schedar",
                        "Gacrux",
                        "Pulcherrima",
                        "Achird",
                        "Zubenelgenubi",
                        "Vindemiatrix",
                        "Sadachbia",
                        "Sadaltager",
                        "Sulafat",
                    ],
                    Some("Kore"),
                    false,
                )],
                Some("gemini-3.8-live"),
                false,
                "GoogleClient::connect_agent_live",
            ));
            c.native_realtime_contract = Some(crate::realtime::RealtimeCapabilities {
                history_import: true,
                tools: true,
                input_transcription: true,
                output_transcription: true,
                usage: true,
                interruption: true,
                audio_truncation: false,
                session_resumption: false,
            });
            c.agent_conversation = true;
        }
        "minimax"
            if matches!(
                host.as_str(),
                "api.minimax.io" | "api.minimax.cn" | "api.minimaxi.com"
            ) && matches!(profile.auth, AuthStrategy::ApiKey | AuthStrategy::Bearer) =>
        {
            c.operations.push(operation(
                FileTranscription,
                vec![model("asr-1.0", &[Wav, Mp3, Flac], Some(50_000_000), None)],
                Some("asr-1.0"),
                true,
                "MiniMaxClient::audio",
            ));
            c.operations.push(operation(
                Synthesis,
                [
                    "speech-2.8-hd",
                    "speech-2.8-turbo",
                    "speech-2.6-hd",
                    "speech-2.6-turbo",
                    "speech-02-hd",
                    "speech-02-turbo",
                    "speech-01-hd",
                    "speech-01-turbo",
                ]
                .into_iter()
                .map(|id| {
                    voices(
                        model(id, &[Mp3, Wav, Pcm16Le, Flac], None, Some(9999)),
                        &[],
                        None,
                        true,
                    )
                })
                .collect(),
                Some("speech-2.8-hd"),
                true,
                "MiniMaxClient::tts",
            ));
            c.operations.push(operation(
                IncrementalSynthesis,
                vec![],
                None,
                false,
                "MiniMaxClient::bidi_tts",
            ));
        }
        "qwen"
            if matches!(
                host.as_str(),
                "dashscope.aliyuncs.com" | "dashscope-intl.aliyuncs.com"
            ) && matches!(profile.auth, AuthStrategy::ApiKey | AuthStrategy::Bearer) =>
        {
            c.required_scope_fields = vec!["extra.workspace_id".into()];
            c.operations.push(operation(
                Synthesis,
                vec![voices(
                    model("qwen3-tts-flash", &[Pcm16Le], None, Some(600)),
                    &["Cherry"],
                    Some("Cherry"),
                    true,
                )],
                Some("qwen3-tts-flash"),
                true,
                "QwenClient::tts",
            ));
            for (kind, service) in [
                (
                    FileTranscription,
                    "QwenClient::asr (URL input, asynchronous task)",
                ),
                (LiveAsr, "QwenClient::asr_realtime"),
                (IncrementalSynthesis, "QwenClient::tts_realtime"),
                (NativeRealtime, "QwenClient::realtime"),
            ] {
                c.operations
                    .push(operation(kind, vec![], None, false, service));
            }
        }
        "zhipu"
            if matches!(host.as_str(), "open.bigmodel.cn" | "api.z.ai")
                && matches!(profile.auth, AuthStrategy::ApiKey | AuthStrategy::Bearer) =>
        {
            let mut asr = model("glm-asr-2512", &[Wav, Mp3], Some(25_000_000), None);
            asr.max_duration_seconds = Some(30);
            c.operations.push(operation(
                FileTranscription,
                vec![asr],
                Some("glm-asr-2512"),
                true,
                "ZhipuClient::cloud_audio",
            ));
            // Hosted GLM encoding/voice values are account-dependent and remain opaque.
            if host == "open.bigmodel.cn" {
                c.operations.push(operation(
                    Synthesis,
                    vec![voices(
                        model("glm-tts", &[], None, Some(100_000)),
                        &[],
                        None,
                        true,
                    )],
                    Some("glm-tts"),
                    false,
                    "ZhipuClient::cloud_audio",
                ));
            }
            c.operations.push(operation(
                NativeRealtime,
                vec![],
                None,
                false,
                "ZhipuClient::realtime",
            ));
        }
        "openrouter"
            if host == "openrouter.ai"
                && matches!(profile.auth, AuthStrategy::ApiKey | AuthStrategy::Bearer) =>
        {
            for (kind, modality, formats) in [
                (
                    FileTranscription,
                    "transcription",
                    vec![Wav, Mp3, Flac, Aac],
                ),
                (Synthesis, "speech", vec![Mp3, Pcm16Le]),
            ] {
                let models = profile
                    .models
                    .iter()
                    .filter(|m| {
                        m.metadata
                            .output_modalities
                            .iter()
                            .any(|value| value == modality)
                    })
                    .map(|m| {
                        voices(
                            model(
                                &m.request_model,
                                &formats,
                                if kind == FileTranscription {
                                    Some(25_000_000)
                                } else {
                                    None
                                },
                                None,
                            ),
                            &[],
                            None,
                            true,
                        )
                    })
                    .collect();
                c.operations.push(operation(
                    kind,
                    models,
                    None,
                    true,
                    "OpenRouterClient::audio (models require explicit discovery)",
                ));
            }
        }
        _ => {}
    }
    for operation in &mut c.operations {
        if !matches!(
            operation.operation,
            AudioOperation::Synthesis | AudioOperation::NativeRealtime
        ) || c.provider_id == "openrouter"
        {
            for model in &mut operation.models {
                model.pcm_sample_rate_hz = None;
                model.pcm_channels = None;
                model.pcm_bits_per_sample = None;
            }
        }
    }
    Ok(c)
}
