use super::*;

/// The shape whose list is a flat `data` array of `{id}` objects, with no
/// pagination: the whole catalog arrives in one response.
///
/// Several endpoints speaking this shape publish more than the id — a name, a
/// blurb, a context length — and those are read when present. The bare shape
/// carries only the id, which is still enough to say what exists.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAiChatDirectory;

impl ModelDirectory for OpenAiChatDirectory {
    fn shape(&self) -> ProtocolFamily {
        ProtocolFamily::OpenAiChat
    }

    fn list_request(&self, profile: &ProviderProfile, _cursor: Option<&str>) -> HttpRequest {
        // This shape has no cursor, so there is nothing to carry forward and a
        // caller can never have one to hand back.
        // Anthropic inference profiles use a bare origin, even when their
        // model directory publishes the OpenAI shape.
        let path = if profile.protocol == ProtocolFamily::AnthropicMessages {
            "/v1/models"
        } else {
            "/models"
        };
        get(endpoint(profile, path), profile)
    }

    fn decode_page(&self, resp: &HttpResponse) -> Result<ModelPage, LlmError> {
        let body = ok_or_classified(resp, crate::codecs::openai::chat::classify_error)?;
        let models = rows(&body, "data")?
            .iter()
            .map(|row| {
                Ok(LiveModel {
                    inference_features: reasoning_features(row),
                    request_model: required_id(row, "id")?,
                    display_name: text(row, "name"),
                    description: text(row, "description"),
                    context_window: number(row, "context_length"),
                    max_output_tokens: number(row, "max_output_tokens").or_else(|| {
                        row.get("top_provider")
                            .and_then(|t| number(t, "max_completion_tokens"))
                    }),
                })
            })
            .collect::<Result<Vec<_>, LlmError>>()?;
        Ok(ModelPage {
            models,
            next_cursor: None,
        })
    }
}

fn reasoning_features(row: &Value) -> Option<crate::protocol::InferenceFeatures> {
    use crate::protocol::*;
    let reasoning = row.get("reasoning")?.as_object()?;
    let mut features = InferenceFeatures::default();
    if let Some(levels) = reasoning.get("supported_efforts") {
        features.effort.support = CapabilitySupport::Supported;
        if levels.is_null() {
            // OpenRouter explicitly accepts every gateway effort here. An
            // absent key is different and must retain previous observations.
            features.effort.levels = Some(ReasoningEffort::ALL.to_vec());
        } else if let Some(levels) = levels.as_array() {
            let parsed = levels
                .iter()
                .cloned()
                .map(serde_json::from_value)
                .collect::<Result<Vec<_>, _>>()
                .ok();
            features.effort.levels = parsed;
        }
    }
    features.effort.default = reasoning
        .get("default_effort")
        .cloned()
        .and_then(|v| serde_json::from_value(v).ok());
    if let Some(value) = reasoning
        .get("supports_max_tokens")
        .and_then(Value::as_bool)
    {
        features.budget.support = if value {
            CapabilitySupport::Supported
        } else {
            CapabilitySupport::Unsupported
        };
    }
    if let Some(mandatory) = reasoning.get("mandatory").and_then(Value::as_bool) {
        features.modes = Some(if mandatory {
            vec![ThinkingMode::Enabled]
        } else {
            vec![ThinkingMode::Disabled, ThinkingMode::Enabled]
        });
        if mandatory {
            if let Some(levels) = &mut features.effort.levels {
                levels.retain(|e| *e != ReasoningEffort::None);
            }
        }
    }
    if let Some(enabled) = reasoning.get("default_enabled").and_then(Value::as_bool) {
        features.default_mode = Some(if enabled {
            ThinkingMode::Enabled
        } else {
            ThinkingMode::Disabled
        });
    }
    if features == InferenceFeatures::default() {
        return None;
    }
    features.thinking = CapabilitySupport::Supported;
    Some(features)
}
