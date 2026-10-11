//! Native realtime Agent conversation on a pinned exact audio route.
use super::*;
use crate::{
    audio::{AudioOperation, AudioRoute},
    protocol::{Secret, ToolSpec},
    ClientSnapshot,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AudioRealtimeConfig {
    pub model: Option<String>,
    pub voice: Option<String>,
    pub instructions: Option<String>,
}

/// Handles remain bounded. Hosts run `driver`, own capture/playback, and run
/// every ToolCall through their existing permission and sandbox path.
pub struct ConnectedAudioConversation {
    pub control: RealtimeControl,
    pub events: RealtimeEvents,
    pub driver: RealtimeDriver,
    pub model: String,
    pub input_format: RealtimeAudioFormat,
    pub output_format: RealtimeAudioFormat,
    pub capabilities: RealtimeCapabilities,
}

/// Builds native provider configuration internally; no endpoints or vendor
/// frames escape into the host. No chat model or fallback routing is used.
#[allow(clippy::too_many_arguments)]
pub async fn connect_audio_conversation(
    snapshot: &ClientSnapshot,
    route: &AudioRoute,
    config: AudioRealtimeConfig,
    tools: Vec<ToolSpec>,
    history: Vec<RealtimeHistoryItem>,
    credential: Secret<String>,
    transport: Arc<dyn RealtimeTransport>,
    limits: RealtimeLimits,
) -> Result<ConnectedAudioConversation, RealtimeError> {
    let invalid = |message: String| RealtimeError::InvalidConfig { message };
    validate_limits(limits)?;
    let caps = snapshot
        .audio()
        .capabilities(route)
        .map_err(|error| invalid(error.to_string()))?;
    if !caps.agent_conversation {
        return Err(invalid(format!(
            "{} does not implement native Agent conversation",
            route.profile_name
        )));
    }
    let descriptor = caps
        .operation(AudioOperation::NativeRealtime)
        .ok_or_else(|| invalid("native realtime operation is unavailable".into()))?;
    let model = config
        .model
        .as_deref()
        .or(descriptor.default_model.as_deref())
        .ok_or_else(|| invalid("native realtime requires an explicit audio model".into()))?
        .to_owned();
    if model.trim().is_empty()
        || !descriptor
            .models
            .iter()
            .any(|entry| entry.id.as_deref() == Some(&model))
            && descriptor.default_model.as_deref() != Some(&model)
    {
        return Err(invalid(format!(
            "native audio model {model:?} is not declared for {}",
            route.profile_name
        )));
    }
    if tools.len() > MAX_REALTIME_TOOL_RESULTS {
        return Err(invalid("native realtime tool limit exceeded".into()));
    }
    let mut names = std::collections::HashSet::new();
    for tool in &tools {
        if tool.name.trim().is_empty()
            || tool.name.contains('\0')
            || !names.insert(&tool.name)
            || !tool.input_schema.is_object()
            || tool.description.contains('\0')
        {
            return Err(invalid(
                "native realtime tools require unique names, NUL-free text and object schemas"
                    .into(),
            ));
        }
        if tool
            .tool_type
            .as_deref()
            .is_some_and(|kind| kind != "function")
            || !tool.native_options.is_empty()
            || !tool.extra.is_null()
            || tool.defer_loading
        {
            return Err(invalid(
                "native realtime supports plain local function tools only".into(),
            ));
        }
    }
    if config
        .voice
        .as_deref()
        .is_some_and(|voice| voice.trim().is_empty() || voice.contains('\0'))
        || config
            .instructions
            .as_deref()
            .is_some_and(|text| text.contains('\0'))
    {
        return Err(invalid(
            "native realtime voice/instructions are invalid".into(),
        ));
    }
    let (control, events, driver, input_rate) = match caps.provider_id.as_str() {
        "openai" => {
            use crate::providers::openai::realtime::*;
            let client = snapshot
                .provider::<crate::providers::OpenAiClient>(&route.profile_name)
                .map_err(|error| invalid(error.to_string()))?;
            let provider_config = OpenAiRealtimeConfig {
                instructions: config.instructions,
                voice: config.voice.map(OpenAiRealtimeVoice::BuiltIn),
                tools: tools
                    .into_iter()
                    .map(|tool| OpenAiRealtimeFunctionTool {
                        name: tool.name,
                        description: Some(tool.description),
                        parameters: tool.input_schema,
                    })
                    .collect(),
                ..Default::default()
            };
            let (session, driver) = client
                .connect_agent_realtime(
                    transport,
                    &route.account_scope,
                    credential,
                    &model,
                    provider_config,
                    history,
                    limits,
                )
                .await?;
            let (control, events) = session.into_parts();
            (control, events, driver, 24000)
        }
        "google" => {
            use crate::providers::google::live::GeminiLiveConfig;
            let client = snapshot
                .provider::<crate::providers::GoogleClient>(&route.profile_name)
                .map_err(|error| invalid(error.to_string()))?;
            let mut provider_config = GeminiLiveConfig::new(&model, &route.account_scope);
            provider_config.system_instruction = config.instructions;
            if let Some(voice) = config.voice {
                provider_config.generation_config["speechConfig"] =
                    json!({"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":voice}}});
            }
            if !tools.is_empty() {
                provider_config.tools = vec![
                    json!({"functionDeclarations":tools.into_iter().map(|tool|json!({"name":tool.name,"description":tool.description,"parametersJsonSchema":tool.input_schema})).collect::<Vec<_>>()}),
                ];
            }
            // System messages use Google's dedicated system-instruction field.
            let mut turns = Vec::new();
            for item in history {
                if let RealtimeHistoryItem::Message {
                    role: RealtimeRole::System,
                    text,
                    ..
                } = item
                {
                    let instruction = provider_config
                        .system_instruction
                        .get_or_insert_with(String::new);
                    if !instruction.is_empty() {
                        instruction.push('\n');
                    }
                    instruction.push_str(&text);
                } else {
                    turns.push(item);
                }
            }
            let (control, events, driver) = client
                .connect_agent_live(transport, credential, provider_config, turns, limits)
                .await?;
            (control, events, driver, 16000)
        }
        _ => return Err(invalid("provider has no native Agent adapter".into())),
    };
    let capabilities = control.capabilities();
    Ok(ConnectedAudioConversation {
        control,
        events,
        driver,
        model,
        input_format: RealtimeAudioFormat::Pcm16 {
            sample_rate_hz: input_rate,
        },
        output_format: RealtimeAudioFormat::Pcm16 {
            sample_rate_hz: 24000,
        },
        capabilities,
    })
}
