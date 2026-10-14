//! OpenAI Chat audio has its own model, format and replay contract.
use crate::codecs::CodecContext;
use crate::protocol::{ChatRequest, ContentBlock, LlmError, MessageRole, ProtocolFamily};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatAudioFormat {
    Wav,
    Mp3,
    Flac,
    Opus,
    Pcm16,
}

/// Add with `NativeExtension::from_typed` to `ChatRequest.native_options`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatAudioOutput {
    pub voice: String,
    pub format: ChatAudioFormat,
}
impl crate::protocol::NativeType for ChatAudioOutput {
    const FORMAT: &'static str = "openai.chat.audio_output";
}

pub(crate) fn has_output(req: &ChatRequest) -> bool {
    req.native_options.iter().any(|v| v.is::<ChatAudioOutput>())
}

pub(crate) fn accepts(context: &CodecContext) -> bool {
    let p = context.profile();
    p.protocol == ProtocolFamily::OpenAiChat
        && matches!(p.provider_id.as_str(), "openai" | "google")
        && !matches!(
            p.auth,
            crate::protocol::AuthStrategy::ChatGptPlan
                | crate::protocol::AuthStrategy::ChatGptOAuth
                | crate::protocol::AuthStrategy::CopilotBearer
        )
}

pub(crate) fn validate(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    let configs = req
        .native_options
        .iter()
        .filter(|v| v.is::<ChatAudioOutput>())
        .collect::<Vec<_>>();
    if !configs.is_empty() {
        if configs.len() != 1 {
            return Err(invalid("one Chat audio output configuration is allowed"));
        }
        if !accepts(context) || context.profile().provider_id.as_str() != "openai" {
            return Err(unsupported(
                "OpenAI Chat audio output requires an API-key OpenAI Chat profile",
            ));
        }
        modality(context, false)?;
        let config = configs[0].decode::<ChatAudioOutput>()?;
        if config.voice.trim().is_empty() || config.voice.chars().any(char::is_control) {
            return Err(invalid("Chat audio requires an explicit nonempty voice"));
        }
        if context.stream && config.format != ChatAudioFormat::Pcm16 {
            return Err(unsupported(
                "streaming OpenAI Chat audio requires pcm16 output",
            ));
        }
    }
    if context.profile().provider_id.as_str() == "openrouter" {
        return Ok(());
    }
    for message in &req.messages {
        for block in &message.content {
            let ContentBlock::Audio { format, data } = block else {
                continue;
            };
            if !accepts(context) {
                return Err(unsupported(
                    "audio input requires an audio-capable Chat profile",
                ));
            }
            modality(context, true)?;
            if message.role != MessageRole::User {
                return Err(invalid("Chat audio input belongs to a user message"));
            }
            if !matches!(format.as_str(), "wav" | "mp3") {
                return Err(invalid("OpenAI-compatible audio input accepts wav or mp3"));
            }
            use base64::Engine;
            if data.is_empty()
                || base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .is_err()
            {
                return Err(invalid(
                    "Chat audio input requires nonempty standard Base64 bytes",
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn apply(
    req: &ChatRequest,
    context: &CodecContext,
    body: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<(), LlmError> {
    validate(req, context)?;
    if let Some(config) = req
        .native_options
        .iter()
        .find(|v| v.is::<ChatAudioOutput>())
    {
        let config = config.decode::<ChatAudioOutput>()?;
        body.insert("modalities".into(), serde_json::json!(["text", "audio"]));
        body.insert(
            "audio".into(),
            serde_json::to_value(config)
                .map_err(|_| invalid("invalid Chat audio configuration"))?,
        );
    }
    Ok(())
}

pub(crate) fn replay(
    protocol: ProtocolFamily,
    value: &serde_json::Value,
) -> Result<Option<serde_json::Map<String, serde_json::Value>>, LlmError> {
    if value.get("audio").is_none() {
        return Ok(None);
    }
    if protocol != ProtocolFamily::OpenAiChat {
        return Err(unsupported("Chat audio history cannot cross protocols"));
    }
    let id = value["audio"]["id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("Chat audio history requires its provider audio id"))?;
    Ok(Some(serde_json::Map::from_iter([(
        "audio".into(),
        serde_json::json!({"id": id}),
    )])))
}

fn modality(context: &CodecContext, input: bool) -> Result<(), LlmError> {
    let model = context
        .profile()
        .models
        .iter()
        .find(|m| m.request_model == context.request_model());
    if !model.is_some_and(|m| {
        (if input {
            &m.metadata.input_modalities
        } else {
            &m.metadata.output_modalities
        })
        .iter()
        .any(|v| v == "audio")
    }) {
        return Err(unsupported(
            "the selected Chat model does not declare the requested audio modality",
        ));
    }
    Ok(())
}
fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
fn unsupported(message: &str) -> LlmError {
    LlmError::UnsupportedCapability {
        message: message.into(),
    }
}
