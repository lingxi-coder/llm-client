use super::*;

/// The shape that paginates with an id cursor: `data`, `has_more`, `last_id`,
/// and `after_id` to continue.
#[derive(Debug, Default, Clone, Copy)]
pub struct AnthropicMessagesDirectory;

impl ModelDirectory for AnthropicMessagesDirectory {
    fn shape(&self) -> ProtocolFamily {
        ProtocolFamily::AnthropicMessages
    }

    fn list_request(&self, profile: &ProviderProfile, cursor: Option<&str>) -> HttpRequest {
        let url = endpoint(profile, "/v1/models");
        let url = match cursor {
            Some(after) => with_query(&url, &[("limit", PAGE_SIZE), ("after_id", after)]),
            None => with_query(&url, &[("limit", PAGE_SIZE)]),
        };
        let mut req = get(url, profile);
        // The Models API requires the same version header as Messages.
        req.headers
            .retain(|(name, _)| !name.eq_ignore_ascii_case("anthropic-version"));
        req.headers.push((
            "anthropic-version".to_owned(),
            profile
                .extra
                .get("api_version")
                .and_then(Value::as_str)
                .unwrap_or(crate::wire_options::DEFAULT_ANTHROPIC_API_VERSION)
                .to_owned(),
        ));
        req
    }

    fn decode_page(&self, resp: &HttpResponse) -> Result<ModelPage, LlmError> {
        let body = ok_or_classified(resp, crate::codecs::anthropic::classify_error)?;
        let models = rows(&body, "data")?
            .iter()
            .map(|row| {
                Ok(LiveModel {
                    inference_features: inference_features(row),
                    request_model: required_id(row, "id")?,
                    display_name: text(row, "display_name"),
                    description: None,
                    context_window: number(row, "max_input_tokens"),
                    max_output_tokens: number(row, "max_tokens"),
                })
            })
            .collect::<Result<Vec<_>, LlmError>>()?;
        // `has_more` is what ends the walk; `last_id` alone is sent on the
        // final page too, so following it would loop forever.
        let next_cursor = match body.get("has_more").and_then(Value::as_bool) {
            Some(true) => Some(cursor(&body, "last_id").ok_or_else(|| {
                LlmError::ProviderInternal {
                    message: "the Anthropic model directory says there is more data but has no usable \"last_id\" cursor".to_owned(),
                }
            })?),
            Some(false) => None,
            None => {
                return Err(LlmError::ProviderInternal {
                    message: "the Anthropic model directory page has no boolean \"has_more\" value".to_owned(),
                });
            }
        };
        Ok(ModelPage {
            models,
            next_cursor,
        })
    }
}

fn inference_features(row: &Value) -> Option<crate::protocol::InferenceFeatures> {
    use crate::protocol::*;
    let caps = row.get("capabilities")?.as_object()?;
    let support = |value: &Value| match value.get("supported").and_then(Value::as_bool) {
        Some(true) => CapabilitySupport::Supported,
        Some(false) => CapabilitySupport::Unsupported,
        None => CapabilitySupport::Unknown,
    };
    let mut features = InferenceFeatures::default();
    if let Some(thinking) = caps.get("thinking") {
        features.thinking = support(thinking);
        if let Some(types) = thinking.get("types").and_then(Value::as_object) {
            for (name, value) in types {
                let Ok(mode) = serde_json::from_value::<ThinkingMode>(Value::String(name.clone()))
                else {
                    continue;
                };
                let fact = support(value);
                if fact != CapabilitySupport::Unknown {
                    features.mode_support.insert(
                        mode,
                        ThinkingModeSupport {
                            support: fact,
                            ..Default::default()
                        },
                    );
                    if mode == ThinkingMode::Enabled {
                        features.budget.support = fact;
                    }
                }
            }
        }
    }
    if let Some(effort) = caps.get("effort") {
        features.effort.support = support(effort);
        for level in ReasoningEffort::ALL {
            if let Some(value) = effort.get(level.as_str()) {
                let fact = support(value);
                if fact != CapabilitySupport::Unknown {
                    features.effort.level_support.insert(level, fact);
                }
            }
        }
    }
    if features == InferenceFeatures::default() {
        None
    } else {
        features
            .sources
            .push("https://platform.claude.com/docs/en/api/models/list".into());
        Some(features)
    }
}
