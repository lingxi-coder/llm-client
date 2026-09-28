//! OpenRouter Chat audio capabilities and option validation.
use crate::codecs::CodecContext;
use crate::protocol::{ContentBlock, LlmError, MessageRole, ProviderProfile};
use serde_json::Value;

#[derive(Debug)]
pub(crate) struct AudioOutputConfig<'a> {
    pub(crate) voice: &'a str,
    pub(crate) format: &'a str,
}

pub(crate) fn openrouter_audio_output<'a>(
    value: Option<&'a Value>,
    profile: &ProviderProfile,
    opts: &CodecContext,
) -> Result<Option<AudioOutputConfig<'a>>, LlmError> {
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
    let config = value.as_object().ok_or_else(|| LlmError::InvalidRequest {
        message: "metadata.openrouter_chat_audio must be an object".into(),
    })?;
    if config.keys().any(|key| key != "voice" && key != "format") {
        return Err(LlmError::InvalidRequest {
            message: "metadata.openrouter_chat_audio accepts only voice and format".into(),
        });
    }
    let voice = config
        .get("voice")
        .and_then(Value::as_str)
        .filter(|voice| !voice.trim().is_empty())
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "metadata.openrouter_chat_audio.voice must be a non-empty string".into(),
        })?;
    let format = config
        .get("format")
        .and_then(Value::as_str)
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "metadata.openrouter_chat_audio.format must be a string".into(),
        })?;
    if !["wav", "mp3", "flac", "opus", "pcm16"].contains(&format) {
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
    if let Some(audio) = req.metadata.get("openrouter_chat_audio") {
        openrouter_audio_output(Some(audio), profile, opts)?;
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
