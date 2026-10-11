//! Exact-profile cloud audio independent of chat models, retry and failover.
//! The host resolves session-following policy and supplies credentials per operation.
//! Capability descriptors distinguish common adapters from provider-only services.
mod catalog;
mod types;
pub use catalog::*;
pub use types::*;

use crate::{
    protocol::{LlmError, ProviderProfile, Secret},
    runtime::ClientSource,
    ClientSnapshot, LlmClient, RequestOptions,
};
use futures::{stream, StreamExt};
use serde_json::Value;

#[derive(Clone, Copy)]
pub struct AudioService<'a> {
    source: ClientSource<'a>,
}
impl LlmClient {
    pub fn audio(&self) -> AudioService<'_> {
        AudioService {
            source: ClientSource::live(&self.runtime, &self.published),
        }
    }
}
impl ClientSnapshot {
    pub fn audio(&self) -> AudioService<'_> {
        AudioService {
            source: ClientSource::fixed(self),
        }
    }
}
impl AudioService<'_> {
    fn pin(&self, route: &AudioRoute) -> Result<(ClientSnapshot, AudioCapabilities), AudioError> {
        let snapshot = self.source.pin().map_err(local_llm)?;
        let profile = snapshot
            .native_profile(&route.profile_name)
            .ok_or_else(|| {
                AudioError::local(format!(
                    "unknown exact audio profile {:?}; groups are not routes",
                    route.profile_name
                ))
            })?;
        let capabilities = catalog::catalog(profile, route, snapshot.region())?;
        Ok((snapshot, capabilities))
    }
    pub fn capabilities(&self, route: &AudioRoute) -> Result<AudioCapabilities, AudioError> {
        self.pin(route).map(|(_, capabilities)| capabilities)
    }
    pub async fn transcribe(
        &self,
        route: &AudioRoute,
        input: AudioInput,
        request: &TranscriptionRequest,
        options: &RequestOptions,
    ) -> Result<TranscriptionResult, AudioError> {
        with_timeout(
            self.transcribe_once(route, input, request, options),
            options.total_timeout,
        )
        .await
    }
    async fn transcribe_once(
        &self,
        route: &AudioRoute,
        input: AudioInput,
        request: &TranscriptionRequest,
        options: &RequestOptions,
    ) -> Result<TranscriptionResult, AudioError> {
        let (snapshot, caps) = self.pin(route)?;
        let descriptor = caps.model(AudioOperation::FileTranscription, request.model.as_deref())?;
        if descriptor
            .max_input_bytes
            .is_some_and(|max| input.size_bytes > max)
        {
            return Err(AudioError::limit(
                "audio input exceeds the selected model's byte limit",
            ));
        }
        if request.raw_format.is_some() && caps.provider_id != "xai" {
            return Err(AudioError::local(
                "this transcription adapter does not accept headerless audio",
            ));
        }
        let opts = scoped_options(route, options)?;
        let model = descriptor.id.clone().unwrap_or_default();
        let profile = snapshot
            .profile(&route.profile_name)
            .expect("validated profile");
        let mut result = TranscriptionResult {
            route: route.clone(),
            provider_id: caps.provider_id.clone(),
            model: model.clone(),
            text: String::new(),
            language: None,
            duration_seconds: None,
            words: vec![],
            segments: vec![],
            request_id: None,
            usage: AudioUsage::default(),
            native: None,
        };
        match caps.provider_id.as_str() {
            "openai" => {
                use crate::providers::openai::audio as p;
                let mut typed = p::TranscriptionRequest::new(openai_transcription_model(&model)?);
                typed.language = request.language.clone();
                if model == "gpt-4o-transcribe-diarize" {
                    typed.chunking_auto = true;
                    typed.format = p::AudioTextFormat::DiarizedJson;
                }
                let client = snapshot
                    .provider::<crate::providers::OpenAiClient>(&route.profile_name)
                    .map_err(binding_error)?;
                let output = client
                    .audio()
                    .transcribe(input, &typed, &opts)
                    .await
                    .map_err(openai_error)?;
                result.text = output.text;
                result.language = output.language;
                result.duration_seconds = output.duration_seconds;
                result.words = output.words;
                result.segments = output.segments;
                result.request_id = output.request_id;
                result.native = output.native;
            }
            "xai" => {
                use crate::providers::xai::audio as p;
                let mut typed = p::XaiTranscriptionRequest {
                    model: if model == "grok-voice-transcribe-1.0" {
                        p::XaiTranscriptionModel::GrokVoiceTranscribe1
                    } else {
                        p::XaiTranscriptionModel::GrokVoiceTranscribe2
                    },
                    language: request.language.clone(),
                    ..Default::default()
                };
                if let Some(raw) = &request.raw_format {
                    typed.audio_format = Some(match raw.format {
                        AudioFormat::Pcm16Le => p::XaiRawAudioFormat::Pcm,
                        AudioFormat::Mulaw => p::XaiRawAudioFormat::Mulaw,
                        AudioFormat::Alaw => p::XaiRawAudioFormat::Alaw,
                        _ => return Err(AudioError::local("unsupported raw xAI input encoding")),
                    });
                    typed.sample_rate = Some(raw.sample_rate_hz);
                    if raw.channels > 1 {
                        typed.channels = Some(raw.channels);
                        typed.multichannel = true;
                    } else if raw.channels != 1 {
                        return Err(AudioError::local("raw audio requires at least one channel"));
                    }
                }
                let client = snapshot
                    .provider::<crate::providers::XaiClient>(&route.profile_name)
                    .map_err(binding_error)?;
                let service = client
                    .audio(xai_config(route, profile, &opts))
                    .map_err(xai_error)?;
                let output = service
                    .transcribe(
                        input,
                        &typed,
                        &p::XaiAudioCredentials::new(credential(&opts)?.clone()),
                    )
                    .await
                    .map_err(xai_error)?;
                result.text = output.text;
                result.language = output.language;
                result.duration_seconds = output.duration_seconds;
                result.request_id = output.request_id;
                result.native = Some(output.native);
                result.words = output
                    .words
                    .into_iter()
                    .map(|word| TimedWord {
                        word: word.text,
                        start: word.start,
                        end: word.end,
                    })
                    .collect();
            }
            "minimax" => {
                use crate::providers::minimax::audio as p;
                if request.language.is_some() {
                    return Err(AudioError::local("common MiniMax transcription does not accept a language hint; use its typed service"));
                }
                let endpoint = if china_endpoint(profile) {
                    p::MINIMAX_ASR_CHINA_ENDPOINT
                } else {
                    p::MINIMAX_ASR_INTERNATIONAL_ENDPOINT
                };
                let client = snapshot
                    .provider::<crate::providers::MiniMaxClient>(&route.profile_name)
                    .map_err(binding_error)?;
                let output = client
                    .audio(endpoint)
                    .map_err(minimax_asr_error)?
                    .transcribe(input, &p::MiniMaxTranscriptionRequest::new(), &opts)
                    .await
                    .map_err(minimax_asr_error)?;
                result.text = output.text;
                result.duration_seconds = output.duration_seconds;
                result.request_id = output.request_id;
                result.native = output.native;
            }
            "zhipu" => {
                use crate::providers::zhipu::cloud_audio as p;
                if request.language.is_some() {
                    return Err(AudioError::local(
                        "hosted GLM transcription has no declared language hint",
                    ));
                }
                let client = snapshot
                    .provider::<crate::providers::ZhipuClient>(&route.profile_name)
                    .map_err(binding_error)?;
                let scope = p::GlmCloudAudioScope::new(
                    &route.profile_name,
                    &route.account_scope,
                    if china_endpoint(profile) {
                        p::GlmCloudAudioRegion::MainlandChina
                    } else {
                        p::GlmCloudAudioRegion::International
                    },
                )
                .map_err(glm_error)?;
                let output = client
                    .cloud_audio(scope)
                    .map_err(glm_error)?
                    .transcribe(
                        credential(&opts)?,
                        input,
                        &p::GlmCloudTranscriptionRequest::new(),
                    )
                    .await
                    .map_err(glm_error)?;
                match output {
                    p::GlmCloudTranscriptionOutput::Complete(output) => {
                        result.text = output.text;
                        result.request_id = output.request_id;
                        result.native = Some(output.native);
                    }
                    _ => unreachable!("nonstreaming request"),
                }
            }
            "openrouter" => {
                use crate::providers::openrouter::audio as p;
                let format = match input.media_type.as_str() {
                    "audio/wav" | "audio/x-wav" => p::OpenRouterInputAudioFormat::Wav,
                    "audio/mpeg" | "audio/mp3" => p::OpenRouterInputAudioFormat::Mp3,
                    "audio/flac" => p::OpenRouterInputAudioFormat::Flac,
                    "audio/mp4" => p::OpenRouterInputAudioFormat::M4a,
                    "audio/ogg" => p::OpenRouterInputAudioFormat::Ogg,
                    "audio/webm" => p::OpenRouterInputAudioFormat::Webm,
                    "audio/aac" => p::OpenRouterInputAudioFormat::Aac,
                    _ => return Err(AudioError::local("unsupported OpenRouter input media type")),
                };
                let mut typed = p::OpenRouterTranscriptionRequest::new(&model, format);
                typed.language = request.language.clone();
                let client = snapshot
                    .provider::<crate::providers::OpenRouterClient>(&route.profile_name)
                    .map_err(binding_error)?;
                let output = client
                    .audio()
                    .transcribe(input, &typed, &opts)
                    .await
                    .map_err(openrouter_error)?;
                result.text = output.text;
                result.request_id = output.generation_id;
                result.native = Some(output.native);
                if let Some(usage) = output.usage {
                    result.usage = AudioUsage {
                        input_tokens: usage.input_tokens,
                        output_tokens: usage.output_tokens,
                        total_tokens: usage.total_tokens,
                        audio_seconds: usage.seconds,
                        cost_usd: usage.cost,
                        ..Default::default()
                    };
                }
            }
            _ => {
                return Err(AudioError::local(
                    "audio transcription adapter is unavailable",
                ))
            }
        }
        if result.usage == AudioUsage::default() {
            result.usage = usage_from_native(result.native.as_ref());
        }
        result.usage.audio_seconds = result.usage.audio_seconds.or(result.duration_seconds);
        Ok(result)
    }
    pub async fn synthesize(
        &self,
        route: &AudioRoute,
        request: &SynthesisRequest,
        options: &RequestOptions,
    ) -> Result<AudioOutput, AudioError> {
        with_timeout(
            self.synthesize_once(route, request, options),
            options.total_timeout,
        )
        .await
    }
    async fn synthesize_once(
        &self,
        route: &AudioRoute,
        request: &SynthesisRequest,
        options: &RequestOptions,
    ) -> Result<AudioOutput, AudioError> {
        let (snapshot, caps) = self.pin(route)?;
        let descriptor = caps.model(AudioOperation::Synthesis, request.model.as_deref())?;
        if !descriptor.formats.contains(&request.format) {
            return Err(AudioError::unsupported(
                "the selected audio model does not declare this output encoding",
            ));
        }
        if request.text.trim().is_empty()
            || request.text.contains('\0')
            || descriptor
                .max_text_characters
                .is_some_and(|max| request.text.chars().count() > max)
        {
            return Err(AudioError::local("speech text is empty, contains NUL, or exceeds the selected model's character limit"));
        }
        let voice = request
            .voice
            .as_deref()
            .or(descriptor.default_voice.as_deref())
            .ok_or_else(|| {
                AudioError::local("this audio model requires an explicit discovered voice")
            })?;
        let opts = scoped_options(route, options)?;
        let model = descriptor.id.clone().unwrap_or_default();
        let profile = snapshot
            .profile(&route.profile_name)
            .expect("validated profile");
        let mut metadata = AudioMetadata {
            route: route.clone(),
            provider_id: caps.provider_id.clone(),
            model: model.clone(),
            format: request.format,
            sample_rate_hz: request.format.is_raw().then_some(24000),
            channels: request.format.is_raw().then_some(1),
            bits_per_sample: request.format.is_raw().then_some(
                if matches!(request.format, AudioFormat::Mulaw | AudioFormat::Alaw) {
                    8
                } else {
                    16
                },
            ),
            content_type: request.format.content_type().into(),
            request_id: None,
        };
        match caps.provider_id.as_str() {
            "openai" => {
                use crate::providers::openai::audio as p;
                if request.language.is_some() {
                    return Err(AudioError::local(
                        "OpenAI speech does not declare a language parameter",
                    ));
                }
                let typed = p::SpeechRequest {
                    model: openai_speech_model(&model)?,
                    input: request.text.clone(),
                    voice: openai_voice(voice)?,
                    format: openai_format(request.format)?,
                    instructions: None,
                    speed: None,
                };
                let client = snapshot
                    .provider::<crate::providers::OpenAiClient>(&route.profile_name)
                    .map_err(binding_error)?;
                let output = client
                    .audio()
                    .synthesize(&typed, &opts)
                    .await
                    .map_err(openai_error)?;
                metadata.request_id = output.request_id.clone();
                let body = stream::try_unfold(output, |mut s| async move {
                    s.next_chunk()
                        .await
                        .map(|chunk| chunk.map(|data| (data, s)))
                        .map_err(|e| AudioError::provider(e, AudioDispatch::Accepted))
                })
                .boxed();
                Ok(AudioOutput::Stream(AudioStream::new(
                    metadata,
                    AudioUsage::default(),
                    body,
                )))
            }
            "xai" => {
                use crate::providers::xai::audio as p;
                let mut typed = p::XaiSpeechRequest::new(
                    &request.text,
                    voice,
                    request.language.as_deref().unwrap_or("auto"),
                );
                typed.output_format = p::XaiSpeechFormat::new(match request.format {
                    AudioFormat::Mp3 => p::XaiSpeechCodec::Mp3,
                    AudioFormat::Wav => p::XaiSpeechCodec::Wav,
                    AudioFormat::Pcm16Le => p::XaiSpeechCodec::Pcm,
                    AudioFormat::Mulaw => p::XaiSpeechCodec::Mulaw,
                    AudioFormat::Alaw => p::XaiSpeechCodec::Alaw,
                    _ => unreachable!("validated encoding"),
                });
                let client = snapshot
                    .provider::<crate::providers::XaiClient>(&route.profile_name)
                    .map_err(binding_error)?;
                let output = client
                    .audio(xai_config(route, profile, &opts))
                    .map_err(xai_error)?
                    .synthesize(
                        &typed,
                        &p::XaiAudioCredentials::new(credential(&opts)?.clone()),
                    )
                    .await
                    .map_err(xai_error)?;
                let p::XaiSpeechOutput::Audio(output) = output else {
                    unreachable!("timestamp mode disabled")
                };
                metadata.content_type = output.content_type.clone();
                metadata.request_id = output.request_id.clone();
                let body = stream::try_unfold(output, |mut s| async move {
                    s.next_chunk()
                        .await
                        .map(|chunk| chunk.map(|data| (data, s)))
                        .map_err(|e| AudioError::provider(e, AudioDispatch::Accepted))
                })
                .boxed();
                Ok(AudioOutput::Stream(AudioStream::new(
                    metadata,
                    AudioUsage::default(),
                    body,
                )))
            }
            "google" if profile.protocol == crate::protocol::ProtocolFamily::VertexGemini => {
                use crate::hosting::vertex::speech as p;
                let signing = profile.signing.as_ref().ok_or_else(|| {
                    AudioError::local(
                        "Vertex speech requires the exact profile's project/location configuration",
                    )
                })?;
                let scope = p::VertexSpeechScope::new(
                    &route.profile_name,
                    &route.account_scope,
                    signing.project.as_deref().ok_or_else(|| {
                        AudioError::local("Vertex speech requires an explicit project")
                    })?,
                    signing.region.as_deref().unwrap_or("global"),
                )
                .map_err(vertex_error)?;
                let typed = p::VertexSpeechRequest::single(
                    vertex_model(&model)?,
                    &request.text,
                    request.language.as_deref().unwrap_or("en-US"),
                    voice,
                );
                let client = snapshot
                    .provider::<crate::providers::GoogleClient>(&route.profile_name)
                    .map_err(binding_error)?;
                let output = client
                    .vertex_speech(scope)
                    .map_err(vertex_error)?
                    .synthesize(&typed, &opts)
                    .await
                    .map_err(vertex_error)?;
                metadata.sample_rate_hz = Some(output.pcm_sample_rate_hz);
                metadata.channels = Some(output.pcm_channels);
                metadata.bits_per_sample = Some(output.pcm_bits_per_sample);
                metadata.content_type = output.mime_type.unwrap_or(metadata.content_type);
                Ok(bytes_output(
                    metadata,
                    output.audio,
                    usage_from_native(Some(&output.native)),
                ))
            }
            "google" => {
                use crate::providers::google::speech as p;
                if request.language.is_some() {
                    return Err(AudioError::local("Gemini Interactions speech has no common language parameter; use its typed style controls"));
                }
                let scope = p::GeminiSpeechScope::new(
                    &route.profile_name,
                    &route.account_scope,
                    p::GEMINI_SPEECH_ENDPOINT,
                )
                .map_err(google_error)?;
                let typed = p::GeminiSpeechRequest::new(
                    if model == "gemini-3.8-flash-tts" {
                        p::GeminiSpeechModel::Gemini38FlashTts
                    } else {
                        p::GeminiSpeechModel::Gemini38FlashLiteTts
                    },
                    &request.text,
                    voice,
                )
                .map_err(google_error)?
                .with_format(match request.format {
                    AudioFormat::Wav => p::GeminiSpeechFormat::Wav,
                    AudioFormat::Pcm16Le => p::GeminiSpeechFormat::LinearPcm16,
                    AudioFormat::Mulaw => p::GeminiSpeechFormat::MuLaw,
                    AudioFormat::Alaw => p::GeminiSpeechFormat::ALaw,
                    _ => unreachable!("validated encoding"),
                });
                let client = snapshot
                    .provider::<crate::providers::GoogleClient>(&route.profile_name)
                    .map_err(binding_error)?;
                let output = client
                    .speech(scope)
                    .map_err(google_error)?
                    .synthesize(credential(&opts)?, &typed)
                    .await
                    .map_err(google_error)?;
                metadata.sample_rate_hz = Some(output.response_sample_rate.unwrap_or(24000));
                metadata.content_type = output.response_mime_type.unwrap_or(metadata.content_type);
                metadata.request_id = output.interaction_id;
                Ok(bytes_output(
                    metadata,
                    output.audio,
                    usage_from_native(Some(&output.native)),
                ))
            }
            "minimax" => {
                use crate::providers::minimax::tts as p;
                let mut typed = p::MiniMaxTtsRequest::new(&request.text, voice);
                typed.model = minimax_model(&model)?;
                typed.audio_setting.format = match request.format {
                    AudioFormat::Mp3 => p::MiniMaxTtsAudioFormat::Mp3,
                    AudioFormat::Wav => p::MiniMaxTtsAudioFormat::Wav,
                    AudioFormat::Pcm16Le => p::MiniMaxTtsAudioFormat::Pcm,
                    AudioFormat::Flac => p::MiniMaxTtsAudioFormat::Flac,
                    _ => unreachable!("validated encoding"),
                };
                typed.audio_setting.sample_rate = 24000;
                typed.language_boost = request.language.clone();
                let mut config = p::MiniMaxTtsConfig::new(
                    &route.profile_name,
                    &route.account_scope,
                    if china_endpoint(profile) {
                        p::MiniMaxTtsRegion::ChinaMainland
                    } else {
                        p::MiniMaxTtsRegion::International
                    },
                );
                if let Some(timeout) = opts.total_timeout {
                    config = config.with_request_timeout(timeout);
                }
                let client = snapshot
                    .provider::<crate::providers::MiniMaxClient>(&route.profile_name)
                    .map_err(binding_error)?;
                let output = client
                    .tts(config)
                    .map_err(minimax_tts_error)?
                    .synthesize(
                        &typed,
                        &p::MiniMaxTtsCredentials::new(credential(&opts)?.clone()),
                    )
                    .await
                    .map_err(minimax_tts_error)?;
                match output {
                    p::MiniMaxTtsOutput::Audio(output) => {
                        metadata.request_id = output.request_id;
                        if let Some(extra) = &output.extra_info {
                            if request.format.is_raw() {
                                metadata.sample_rate_hz =
                                    extra.audio_sample_rate.or(metadata.sample_rate_hz);
                                metadata.channels = extra.audio_channel.or(metadata.channels);
                            }
                            let expected = match request.format {
                                AudioFormat::Pcm16Le => "pcm",
                                AudioFormat::Mp3 => "mp3",
                                AudioFormat::Wav => "wav",
                                AudioFormat::Flac => "flac",
                                _ => unreachable!("validated format"),
                            };
                            if extra
                                .audio_format
                                .as_deref()
                                .is_some_and(|actual| actual != expected)
                            {
                                return Err(AudioError {kind:AudioErrorKind::Provider,dispatch:AudioDispatch::Accepted,message:"MiniMax returned an encoding different from the requested encoding".into(),source:None});
                            }
                        }
                        Ok(bytes_output(
                            metadata,
                            output.bytes,
                            usage_from_native(Some(&output.native)),
                        ))
                    }
                    p::MiniMaxTtsOutput::Url(_) => unreachable!("hex output requested"),
                }
            }
            "qwen" => {
                use crate::providers::qwen::tts as p;
                if request.language.is_some() {
                    return Err(AudioError::local("common Qwen speech uses automatic language; use the typed language enum for an override"));
                }
                let workspace = profile
                    .extra
                    .get("workspace_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        AudioError::local(
                            "Qwen speech requires this profile's explicit extra.workspace_id",
                        )
                    })?;
                let scope = p::QwenTtsScope::new(
                    &route.profile_name,
                    &route.account_scope,
                    if china_endpoint(profile) {
                        p::QwenTtsRegion::Beijing
                    } else {
                        p::QwenTtsRegion::Singapore
                    },
                    workspace,
                )
                .map_err(qwen_error)?;
                let client = snapshot
                    .provider::<crate::providers::QwenClient>(&route.profile_name)
                    .map_err(binding_error)?;
                let output = client
                    .tts(scope)
                    .map_err(qwen_error)?
                    .synthesize_stream(&p::QwenTtsRequest::new(&request.text, voice), &opts)
                    .await
                    .map_err(qwen_error)?;
                metadata.request_id = output.request_id().map(str::to_owned);
                let shared_usage =
                    std::sync::Arc::new(std::sync::Mutex::new(AudioUsage::default()));
                let stream_usage = shared_usage.clone();
                let body =
                    stream::try_unfold((output, stream_usage), |(mut s, usage)| async move {
                        loop {
                            match s.next_event().await.map_err(qwen_error)? {
                                Some(event) => match event.kind {
                                    p::QwenTtsStreamEventKind::AudioDelta { data } => {
                                        return Ok(Some((data, (s, usage))))
                                    }
                                    p::QwenTtsStreamEventKind::Completed { synthesis } => {
                                        let mut normalized = usage_from_native(Some(&event.native));
                                        normalized.characters =
                                            normalized.characters.or(synthesis.characters);
                                        *usage
                                            .lock()
                                            .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                            normalized;
                                    }
                                    _ => continue,
                                },
                                None => return Ok(None),
                            }
                        }
                    })
                    .boxed();
                let mut stream = AudioStream::new(metadata, AudioUsage::default(), body);
                stream.shared_usage = Some(shared_usage);
                Ok(AudioOutput::Stream(stream))
            }
            "openrouter" => {
                use crate::providers::openrouter::audio as p;
                if request.language.is_some() {
                    return Err(AudioError::local(
                        "OpenRouter speech has no declared common language parameter",
                    ));
                }
                let mut typed = p::OpenRouterSpeechRequest::new(&model, &request.text);
                typed.voice = Some(voice.into());
                typed.response_format = if request.format == AudioFormat::Mp3 {
                    p::OpenRouterSpeechFormat::Mp3
                } else {
                    p::OpenRouterSpeechFormat::Pcm
                };
                let client = snapshot
                    .provider::<crate::providers::OpenRouterClient>(&route.profile_name)
                    .map_err(binding_error)?;
                let output = client
                    .audio()
                    .speak(&typed, &opts)
                    .await
                    .map_err(openrouter_error)?;
                metadata.content_type = output.content_type;
                metadata.request_id = output.generation_id;
                // OpenRouter PCM parameters are model-dependent, not inferred from OpenAI.
                metadata.sample_rate_hz = None;
                metadata.channels = None;
                metadata.bits_per_sample = None;
                Ok(bytes_output(metadata, output.bytes, AudioUsage::default()))
            }
            _ => Err(AudioError::local("speech adapter is unavailable")),
        }
    }
}
fn bytes_output(metadata: AudioMetadata, bytes: bytes::Bytes, usage: AudioUsage) -> AudioOutput {
    AudioOutput::Stream(AudioStream::new(
        metadata,
        usage,
        stream::once(async move { Ok(bytes) }).boxed(),
    ))
}
fn scoped_options(
    route: &AudioRoute,
    options: &RequestOptions,
) -> Result<RequestOptions, AudioError> {
    if options
        .account_scope
        .as_deref()
        .is_some_and(|scope| scope != route.account_scope)
    {
        return Err(AudioError::local(
            "audio route and credential account scope differ",
        ));
    }
    if options.authenticator.is_some() || options.finalizer.is_some() {
        return Err(AudioError::local("common audio adapters require a host-supplied credential; request authenticators/finalizers are not declared"));
    }
    let mut scoped = options.clone();
    scoped.account_scope = Some(route.account_scope.clone());
    Ok(scoped)
}
fn credential(options: &RequestOptions) -> Result<&Secret<String>, AudioError> {
    options
        .credential
        .as_ref()
        .filter(|value| !value.expose_secret().trim().is_empty())
        .ok_or_else(|| AudioError::local("audio requires a host-supplied credential"))
}
fn binding_error(error: crate::providers::ProviderBindingError) -> AudioError {
    AudioError::provider(error, AudioDispatch::NotSent)
}
fn local_llm(error: LlmError) -> AudioError {
    AudioError::provider(error, AudioDispatch::NotSent)
}
fn china_endpoint(profile: &ProviderProfile) -> bool {
    profile.base_url.contains("dashscope.aliyuncs.com")
        || profile.base_url.contains("minimax.cn")
        || profile.base_url.contains("minimaxi.com")
        || profile.base_url.contains("open.bigmodel.cn")
}
fn xai_config(
    route: &AudioRoute,
    profile: &ProviderProfile,
    options: &RequestOptions,
) -> crate::providers::xai::audio::XaiAudioConfig {
    let config = crate::providers::xai::audio::XaiAudioConfig::new(
        &route.profile_name,
        &route.account_scope,
    )
    .with_api_base_url(&profile.base_url);
    if let Some(timeout) = options.total_timeout {
        config.with_request_timeout(timeout)
    } else {
        config
    }
}
fn usage_from_native(native: Option<&Value>) -> AudioUsage {
    let value = native.and_then(|v| {
        v.get("usage")
            .or_else(|| v.get("usageMetadata"))
            .or_else(|| v.get("extra_info"))
    });
    let get = |keys: &[&str]| {
        value.and_then(|v| keys.iter().find_map(|k| v.get(*k).and_then(Value::as_u64)))
    };
    AudioUsage {
        input_tokens: get(&["input_tokens", "prompt_tokens", "promptTokenCount"]),
        output_tokens: get(&["output_tokens", "completion_tokens", "candidatesTokenCount"]),
        total_tokens: get(&["total_tokens", "totalTokenCount"]),
        characters: get(&["characters", "usage_characters"]),
        native: value.cloned(),
        ..Default::default()
    }
}
fn openai_error(error: crate::providers::openai::audio::AudioError) -> AudioError {
    use crate::providers::openai::audio::AudioError as E;
    let dispatch = match &error {
        E::Llm(_) => AudioDispatch::NotSent,
        E::Provider { status, .. } if (400..500).contains(status) && *status != 408 => {
            AudioDispatch::Rejected
        }
        E::Provider { .. } | E::OutcomeUnknown { .. } => AudioDispatch::Unknown,
        E::InvalidResponse { .. } => AudioDispatch::Accepted,
    };
    AudioError::provider(error, dispatch)
}
fn xai_error(error: crate::providers::xai::audio::XaiAudioError) -> AudioError {
    use crate::providers::xai::audio::XaiAudioError as E;
    let dispatch = match &error {
        E::Llm(_) | E::InvalidRequest(_) => AudioDispatch::NotSent,
        E::Provider { status, .. } if (400..500).contains(status) && *status != 408 => {
            AudioDispatch::Rejected
        }
        E::Provider { .. } | E::OutcomeUnknown { .. } => AudioDispatch::Unknown,
        E::InvalidResponse { .. } => AudioDispatch::Accepted,
    };
    AudioError::provider(error, dispatch)
}
macro_rules! mapped_error {
    ($fn:ident,$error:path,$dispatch:path) => {
        fn $fn(error: $error) -> AudioError {
            use $dispatch as D;
            let state = match error.dispatch() {
                D::NotSent => AudioDispatch::NotSent,
                D::Rejected => AudioDispatch::Rejected,
                D::Unknown => AudioDispatch::Unknown,
                D::Accepted => AudioDispatch::Accepted,
            };
            AudioError::provider(error, state)
        }
    };
}
mapped_error!(
    google_error,
    crate::providers::google::speech::GeminiSpeechError,
    crate::providers::google::speech::GeminiSpeechDispatch
);
mapped_error!(
    vertex_error,
    crate::hosting::vertex::speech::VertexSpeechError,
    crate::hosting::vertex::speech::VertexSpeechDispatch
);
mapped_error!(
    minimax_tts_error,
    crate::providers::minimax::tts::MiniMaxTtsError,
    crate::providers::minimax::tts::MiniMaxTtsDispatch
);
mapped_error!(
    minimax_asr_error,
    crate::providers::minimax::audio::MiniMaxAudioError,
    crate::providers::minimax::audio::MiniMaxAudioDispatch
);
mapped_error!(
    glm_error,
    crate::providers::zhipu::cloud_audio::GlmCloudAudioError,
    crate::providers::zhipu::cloud_audio::GlmCloudAudioDispatch
);
mapped_error!(
    openrouter_error,
    crate::providers::openrouter::audio::OpenRouterAudioError,
    crate::providers::openrouter::audio::OpenRouterAudioDispatch
);
fn qwen_error(error: crate::providers::qwen::tts::QwenTtsError) -> AudioError {
    use crate::providers::qwen::tts::QwenTtsDispatchOutcome as D;
    let dispatch = match error.dispatch_outcome() {
        D::NotSent => AudioDispatch::NotSent,
        D::Rejected => AudioDispatch::Rejected,
        D::Unknown => AudioDispatch::Unknown,
        D::Accepted => AudioDispatch::Accepted,
    };
    AudioError::provider(error, dispatch)
}
fn openai_transcription_model(
    model: &str,
) -> Result<crate::providers::openai::audio::TranscriptionModel, AudioError> {
    use crate::providers::openai::audio::TranscriptionModel as M;
    Ok(match model {
        "gpt-transcribe" => M::GptTranscribe,
        "gpt-4o-transcribe" => M::Gpt4oTranscribe,
        "gpt-4o-mini-transcribe" => M::Gpt4oMiniTranscribe,
        "gpt-4o-mini-transcribe-2025-12-15" => M::Gpt4oMiniTranscribe2025_12_15,
        "gpt-4o-transcribe-diarize" => M::Gpt4oTranscribeDiarize,
        "whisper-1" => M::Whisper1,
        _ => return Err(AudioError::local("unknown OpenAI transcription model")),
    })
}
fn openai_speech_model(
    model: &str,
) -> Result<crate::providers::openai::audio::SpeechModel, AudioError> {
    use crate::providers::openai::audio::SpeechModel as M;
    Ok(match model {
        "gpt-4o-mini-tts" => M::Gpt4oMiniTts,
        "gpt-4o-mini-tts-2025-12-15" => M::Gpt4oMiniTts2025_12_15,
        "tts-1" => M::Tts1,
        "tts-1-hd" => M::Tts1Hd,
        _ => return Err(AudioError::local("unknown OpenAI speech model")),
    })
}
fn openai_voice(voice: &str) -> Result<crate::providers::openai::audio::SpeechVoice, AudioError> {
    use crate::providers::openai::audio::SpeechVoice as V;
    Ok(match voice{"alloy"=>V::Alloy,"ash"=>V::Ash,"ballad"=>V::Ballad,"coral"=>V::Coral,"echo"=>V::Echo,"fable"=>V::Fable,"nova"=>V::Nova,"onyx"=>V::Onyx,"sage"=>V::Sage,"shimmer"=>V::Shimmer,"verse"=>V::Verse,"marin"=>V::Marin,"cedar"=>V::Cedar,_=>return Err(AudioError::local("common OpenAI speech requires a builtin voice; scoped custom voices use the typed provider service"))})
}
fn openai_format(
    format: AudioFormat,
) -> Result<crate::providers::openai::audio::SpeechFormat, AudioError> {
    use crate::providers::openai::audio::SpeechFormat as F;
    Ok(match format {
        AudioFormat::Mp3 => F::Mp3,
        AudioFormat::Opus => F::Opus,
        AudioFormat::Aac => F::Aac,
        AudioFormat::Flac => F::Flac,
        AudioFormat::Wav => F::Wav,
        AudioFormat::Pcm16Le => F::Pcm,
        _ => return Err(AudioError::local("unsupported OpenAI speech format")),
    })
}
fn minimax_model(
    model: &str,
) -> Result<crate::providers::minimax::tts::MiniMaxTtsModel, AudioError> {
    use crate::providers::minimax::tts::MiniMaxTtsModel as M;
    Ok(match model {
        "speech-2.8-hd" => M::Speech28Hd,
        "speech-2.8-turbo" => M::Speech28Turbo,
        "speech-2.6-hd" => M::Speech26Hd,
        "speech-2.6-turbo" => M::Speech26Turbo,
        "speech-02-hd" => M::Speech02Hd,
        "speech-02-turbo" => M::Speech02Turbo,
        "speech-01-hd" => M::Speech01Hd,
        "speech-01-turbo" => M::Speech01Turbo,
        _ => return Err(AudioError::local("unknown MiniMax speech model")),
    })
}
fn vertex_model(
    model: &str,
) -> Result<crate::hosting::vertex::speech::VertexSpeechModel, AudioError> {
    use crate::hosting::vertex::speech::VertexSpeechModel as M;
    Ok(match model {
        "gemini-2.5-flash-tts" => M::Gemini25FlashTts,
        "gemini-2.5-pro-tts" => M::Gemini25ProTts,
        "gemini-2.5-flash-lite-preview-tts" => M::Gemini25FlashLitePreviewTts,
        "gemini-3.1-flash-tts-preview" => M::Gemini31FlashTtsPreview,
        _ => return Err(AudioError::local("unknown Vertex speech model")),
    })
}

async fn with_timeout<T>(
    future: impl std::future::Future<Output = Result<T, AudioError>>,
    timeout: Option<std::time::Duration>,
) -> Result<T, AudioError> {
    match timeout {
        Some(value) if value.is_zero() => Err(AudioError::local("audio deadline must be positive")),
        Some(value) => tokio::time::timeout(value, future)
            .await
            .unwrap_or_else(|_| {
                Err(AudioError {
                    kind: AudioErrorKind::Unavailable,
                    dispatch: AudioDispatch::Unknown,
                    message: "audio deadline elapsed; provider acceptance is unknown".into(),
                    source: None,
                })
            }),
        None => future.await,
    }
}
