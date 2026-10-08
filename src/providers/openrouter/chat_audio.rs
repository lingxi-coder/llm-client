//! OpenRouter Chat audio capabilities and option validation.
use crate::codecs::CodecContext;
use crate::protocol::{ContentBlock, JsonPayload, LlmError, MessageRole, ProviderProfile};
use crate::response_json::FlatJsonKind;

#[derive(Debug)]
pub(crate) struct AudioOutputConfig {
    pub(crate) voice: String,
    pub(crate) format: String,
}

pub(crate) fn openrouter_audio_output(
    value: Option<&JsonPayload>,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<Option<AudioOutputConfig>, LlmError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if profile.provider_id.as_str() != "openrouter" {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenRouter Chat audio output configuration requires an OpenRouter profile"
                .into(),
        });
    }
    if !opts.stream {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenRouter Chat audio output requires streaming".into(),
        });
    }
    if value.kind_at("") != Some(FlatJsonKind::Object) {
        return Err(LlmError::InvalidRequest {
            message: "metadata.openrouter_chat_audio must be an object".into(),
        });
    }
    if !value.has_only_fields_at("", &["voice", "format"]) {
        return Err(LlmError::InvalidRequest {
            message: "metadata.openrouter_chat_audio accepts only voice and format".into(),
        });
    }
    let voice_units = value
        .string_units_at("/voice")
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "metadata.openrouter_chat_audio.voice must be a non-empty string".into(),
        })?;
    let voice = String::from_utf16(&voice_units).map_err(|_| LlmError::InvalidRequest {
        message: "metadata.openrouter_chat_audio.voice must be valid Unicode".into(),
    })?;
    if voice.trim().is_empty() {
        return Err(LlmError::InvalidRequest {
            message: "metadata.openrouter_chat_audio.voice must be a non-empty string".into(),
        });
    }
    let format_units =
        value
            .string_units_at("/format")
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "metadata.openrouter_chat_audio.format must be a string".into(),
            })?;
    let format = String::from_utf16(&format_units).map_err(|_| LlmError::InvalidRequest {
        message: "metadata.openrouter_chat_audio.format must be valid Unicode".into(),
    })?;
    if !["wav", "mp3", "flac", "opus", "pcm16"].contains(&format.as_str()) {
        return Err(LlmError::InvalidRequest {
            message: "OpenRouter Chat audio output format must be wav, mp3, flac, opus, or pcm16"
                .into(),
        });
    }
    validate_model_audio_modality(profile, &opts.request_model, false)?;
    Ok(Some(AudioOutputConfig { voice, format }))
}

pub(crate) fn validate_audio_request(
    req: &crate::protocol::ChatRequest,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<(), LlmError> {
    let mut has_audio = false;
    for message in &req.messages {
        for block in &message.content {
            let ContentBlock::Audio { format, data } = block else {
                continue;
            };
            has_audio = true;
            if profile.provider_id.as_str() != "openrouter" {
                return Err(LlmError::UnsupportedCapability {
                    message: "Chat audio input is currently supported only by OpenRouter profiles"
                        .into(),
                });
            }
            if message.role != MessageRole::User {
                return Err(LlmError::InvalidRequest {
                    message: "OpenRouter Chat audio input must be in a user message".into(),
                });
            }
            validate_audio_format(format)?;
            validate_base64_audio(data)?;
        }
    }
    if has_audio {
        validate_model_audio_modality(profile, &opts.request_model, true)?;
    }
    let audio = req.metadata.payload_at("/openrouter_chat_audio");
    if audio.is_some() {
        openrouter_audio_output(audio.as_ref(), profile, opts)?;
    }
    Ok(())
}

fn validate_model_audio_modality(
    profile: &ProviderProfile,
    request_model: &str,
    input: bool,
) -> Result<(), LlmError> {
    if let Some(model) = profile
        .models
        .iter()
        .find(|model| model.request_model == request_model)
    {
        let modalities = if input {
            &model.metadata.input_modalities
        } else {
            &model.metadata.output_modalities
        };
        if !modalities.is_empty() && !modalities.iter().any(|value| value == "audio") {
            return Err(LlmError::UnsupportedCapability {
                message: format!(
                    "selected OpenRouter model {request_model} does not advertise {} audio modality",
                    if input { "input" } else { "output" }
                ),
            });
        }
    }
    Ok(())
}

fn validate_audio_format(format: &str) -> Result<(), LlmError> {
    if [
        "wav", "mp3", "aiff", "aac", "ogg", "flac", "m4a", "pcm16", "pcm24",
    ]
    .contains(&format)
    {
        Ok(())
    } else {
        Err(LlmError::InvalidRequest {
            message: format!("unsupported OpenRouter Chat audio input format {format:?}"),
        })
    }
}

fn validate_base64_audio(data: &str) -> Result<(), LlmError> {
    let bytes = data.as_bytes();
    let padding = bytes.iter().rev().take_while(|&&byte| byte == b'=').count();
    let valid_alphabet = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/';
    let valid = !bytes.is_empty()
        && bytes.len().is_multiple_of(4)
        && padding <= 2
        && bytes[..bytes.len() - padding]
            .iter()
            .all(|&byte| valid_alphabet(byte))
        && bytes[bytes.len() - padding..]
            .iter()
            .all(|&byte| byte == b'=')
        && match padding {
            0 => true,
            1 => (bytes.len() - padding) % 4 == 3,
            2 => (bytes.len() - padding) % 4 == 2,
            _ => false,
        };
    if valid {
        Ok(())
    } else {
        Err(LlmError::InvalidRequest {
            message: "OpenRouter Chat audio data must be standard base64 without a data URI".into(),
        })
    }
}
