use crate::codecs::{CodecContext, EventDecoder};
use crate::protocol::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

struct Step {
    value: Value,
    arguments: String,
    stopped: bool,
}
pub(super) struct InteractionsDecoder {
    context: CodecContext,
    steps: BTreeMap<usize, Step>,
    id: Option<String>,
    started: bool,
    ended: bool,
    usage: UsageReport,
    buffered: usize,
}
impl InteractionsDecoder {
    pub(super) fn new(context: &CodecContext) -> Self {
        Self {
            context: context.clone(),
            steps: BTreeMap::new(),
            id: None,
            started: false,
            ended: false,
            usage: Default::default(),
            buffered: 0,
        }
    }
    fn interrupted(message: impl Into<String>) -> LlmError {
        LlmError::StreamInterrupted {
            message: message.into(),
        }
    }
    fn index(event: &Value) -> Result<usize, LlmError> {
        event["index"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Self::interrupted("Interactions step event has no valid index"))
    }
}
impl EventDecoder for InteractionsDecoder {
    fn decode_frame(&mut self, frame: &[u8]) -> Result<Vec<StreamEvent>, LlmError> {
        if self.ended {
            return Ok(vec![]);
        }
        if frame == b"[DONE]" {
            return Err(Self::interrupted(
                "Interactions stream ended without interaction.completed",
            ));
        }
        let event: Value = serde_json::from_slice(frame).map_err(|error| {
            Self::interrupted(format!("invalid Interactions stream JSON: {error}"))
        })?;
        let kind = event["event_type"]
            .as_str()
            .ok_or_else(|| Self::interrupted("Interactions event is missing event_type"))?;
        self.buffered = self.buffered.saturating_add(frame.len());
        if self.buffered > 16 * 1024 * 1024 {
            return Err(Self::interrupted("Interactions response exceeds 16 MiB"));
        }
        match kind {
            "interaction.created" => {
                if self.started {
                    return Err(Self::interrupted("duplicate Interactions start"));
                }
                let id = event["interaction"]["id"]
                    .as_str()
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| Self::interrupted("Interactions start has no id"))?;
                self.id = Some(id.into());
                self.started = true;
                Ok(vec![StreamEvent::Start {
                    model: event["interaction"]["model"]
                        .as_str()
                        .unwrap_or(self.context.request_model())
                        .into(),
                    response_id: Some(id.into()),
                }])
            }
            "step.start" => {
                if !self.started {
                    return Err(Self::interrupted("Interactions step before start"));
                }
                let index = Self::index(&event)?;
                if self.steps.contains_key(&index) || !event["step"].is_object() {
                    return Err(Self::interrupted(
                        "duplicate or malformed Interactions step start",
                    ));
                }
                let mut value = event["step"].clone();
                if value["type"] == "model_output" && value.get("content").is_none() {
                    value["content"] = json!([]);
                }
                self.steps.insert(
                    index,
                    Step {
                        value,
                        arguments: String::new(),
                        stopped: false,
                    },
                );
                Ok(vec![])
            }
            "step.delta" => {
                let step = self
                    .steps
                    .get_mut(&Self::index(&event)?)
                    .filter(|step| !step.stopped)
                    .ok_or_else(|| Self::interrupted("Interactions delta has no open step"))?;
                let delta = &event["delta"];
                match delta["type"].as_str() {
                    Some("arguments_delta") if step.value["type"] == "function_call" => {
                        let fragment = delta["arguments"].as_str().ok_or_else(|| {
                            Self::interrupted("Interactions argument delta must be a string")
                        })?;
                        step.arguments.push_str(fragment);
                    }
                    Some("text") if step.value["type"] == "model_output" => {
                        let text = delta["text"].as_str().ok_or_else(|| {
                            Self::interrupted("Interactions text delta must contain text")
                        })?;
                        let parts = step.value["content"].as_array_mut().ok_or_else(|| {
                            Self::interrupted("Interactions model content is not an array")
                        })?;
                        if let Some(last) = parts.last_mut().filter(|part| part["type"] == "text") {
                            let current = last["text"].as_str().unwrap_or("").to_owned();
                            last["text"] = json!(current + text);
                        } else {
                            parts.push(delta.clone());
                        }
                    }
                    Some("thought_signature") => {
                        step.value["signature"] = delta["signature"].clone();
                    }
                    Some("thought_summary") => {
                        let list = step
                            .value
                            .as_object_mut()
                            .ok_or_else(|| Self::interrupted("malformed Interactions thought"))?
                            .entry("summary")
                            .or_insert_with(|| json!([]))
                            .as_array_mut()
                            .ok_or_else(|| {
                                Self::interrupted("malformed Interactions thought summary")
                            })?;
                        list.push(delta["content"].clone());
                    }
                    Some("image" | "audio") if step.value["type"] == "model_output" => {
                        step.value["content"]
                            .as_array_mut()
                            .ok_or_else(|| {
                                Self::interrupted("malformed Interactions multimodal content")
                            })?
                            .push(delta.clone());
                    }
                    _ => {
                        if !delta.is_object() {
                            return Err(Self::interrupted(
                                "Interactions step delta must be an object",
                            ));
                        }
                        for (key, value) in delta.as_object().unwrap() {
                            if key != "type" {
                                step.value[key] = value.clone();
                            }
                        }
                    }
                }
                Ok(vec![])
            }
            "step.stop" => {
                let step = self
                    .steps
                    .get_mut(&Self::index(&event)?)
                    .filter(|step| !step.stopped)
                    .ok_or_else(|| Self::interrupted("Interactions stop has no open step"))?;
                if step.value["type"] == "function_call" && !step.arguments.is_empty() {
                    step.value["arguments"] =
                        serde_json::from_str(&step.arguments).map_err(|error| {
                            Self::interrupted(format!(
                                "invalid completed Interactions function arguments: {error}"
                            ))
                        })?;
                }
                step.stopped = true;
                Ok(vec![])
            }
            "interaction.completed" => {
                if !self.started || self.steps.values().any(|step| !step.stopped) {
                    return Err(Self::interrupted(
                        "Interactions completion before all steps stopped",
                    ));
                }
                let mut body = event["interaction"].clone();
                self.usage = super::decode::usage_report(body.get("usage"));
                if body["id"].as_str() != self.id.as_deref() {
                    return Err(Self::interrupted(
                        "Interactions terminal id differs from start",
                    ));
                }
                let streamed: Vec<_> = self.steps.values().map(|step| step.value.clone()).collect();
                if let Some(terminal) = body.get("steps") {
                    let terminal = terminal.as_array().ok_or_else(|| {
                        Self::interrupted("Interactions terminal steps must be an array")
                    })?;
                    if terminal.len() != streamed.len()
                        || terminal
                            .iter()
                            .zip(&streamed)
                            .any(|(final_step, streamed_step)| {
                                final_step["type"] != streamed_step["type"]
                                    || (streamed_step["type"] == "function_call"
                                        && final_step != streamed_step)
                            })
                    {
                        return Err(Self::interrupted(
                            "Interactions terminal calls differ from completed streamed steps",
                        ));
                    }
                } else {
                    body["steps"] = Value::Array(streamed);
                }
                // Validation is atomic: no function/native call escapes if a later
                // member, duplicate ID or terminal status is malformed.
                let response = super::decode::response(&body, &self.context)?;
                let mut events = vec![];
                for (block, content) in response.message.content.into_iter().enumerate() {
                    events.push(match content {
                        ContentBlock::Text { text, .. } => StreamEvent::TextDelta { block, text },
                        ContentBlock::ToolUse {
                            id,
                            name,
                            input,
                            provider_id,
                            caller,
                            toolset_name,
                            ..
                        } => StreamEvent::ToolCallDelta {
                            block,
                            id,
                            name,
                            provider_id,
                            caller,
                            toolset_name,
                            arguments_fragment: input.to_string(),
                        },
                        ContentBlock::Native { value } => StreamEvent::Native { block, value },
                        ContentBlock::ProviderContent { protocol, value } => {
                            StreamEvent::ProviderContent {
                                block,
                                protocol,
                                value,
                            }
                        }
                        _ => {
                            return Err(Self::interrupted("unexpected Interactions decoded block"))
                        }
                    });
                    events.push(StreamEvent::BlockEnd { block });
                }
                self.ended = true;
                events.push(StreamEvent::End {
                    stop_reason: response.stop_reason,
                    usage: response.usage,
                    inference: response.inference,
                });
                Ok(events)
            }
            "error" => Err(Self::interrupted(format!(
                "Interactions error: {}",
                event["error"]
            ))),
            _ => Ok(vec![StreamEvent::ProviderEvent {
                protocol: ProtocolFamily::GeminiInteractions,
                payload: event,
            }]),
        }
    }
    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        if self.ended {
            Ok(vec![])
        } else {
            Err(Self::interrupted("Interactions stream interrupted before terminal; submission outcome may be unknown"))
        }
    }
    fn usage_report(&self) -> UsageReport {
        self.usage.clone()
    }
}
