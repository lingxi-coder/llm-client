use crate::codecs::{CodecContext, EventDecoder};
use crate::protocol::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

struct Step {
    arguments_decoder: crate::response_json::ArgumentJsonDecoder,
    value: Value,
    source: JsonPayload,
    arguments: Vec<u16>,
    stopped: bool,
}

fn replace_root_fields(
    source: &JsonPayload,
    updates: Vec<(Vec<u16>, String)>,
) -> Result<JsonPayload, LlmError> {
    use crate::response_json::FlatJsonKind;
    let scanner = source.scanner()?;
    let root = scanner.root_node();
    if root.kind() != FlatJsonKind::Object {
        return Err(LlmError::InvalidRequest {
            message: "Interactions source update requires an object".into(),
        });
    }
    let mut updates = updates.into_iter().collect::<BTreeMap<_, _>>();
    let mut fields = Vec::new();
    for entry in root
        .object_entries()
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "Interactions source object has no flat entries".into(),
        })?
    {
        let value = if let Some(replacement) = updates.remove(entry.key_units) {
            replacement
        } else {
            crate::exact_json::serialize_json_node(
                entry.value,
                crate::exact_json::JsonEncoding::Serde,
            )?
        };
        fields.push((entry.key_units.to_vec(), value));
    }
    fields.extend(updates);
    JsonPayload::object_from_fields(&fields).map_err(|error| LlmError::InvalidRequest {
        message: format!("Interactions source update is invalid: {error}"),
    })
}

fn set_root_field(
    source: &JsonPayload,
    key: &str,
    value: &JsonPayload,
) -> Result<JsonPayload, LlmError> {
    replace_root_fields(
        source,
        vec![(
            key.encode_utf16().collect(),
            value.as_json_source().to_owned(),
        )],
    )
}

fn append_content_part(step: &mut Step, part: JsonPayload) -> Result<(), LlmError> {
    let content =
        step.source
            .payload_at("/content")
            .ok_or_else(|| LlmError::StreamInterrupted {
                message: "Interactions model content is not an array".into(),
            })?;
    let mut parts = content
        .array_items()
        .ok_or_else(|| LlmError::StreamInterrupted {
            message: "Interactions model content is not an array".into(),
        })?;
    parts.push(part);
    let content = JsonPayload::array_from_items(&parts, crate::exact_json::JsonEncoding::Serde)?;
    step.source = set_root_field(&step.source, "content", &content)?;
    Ok(())
}

fn merge_delta_fields(step: &mut Step, delta: &JsonPayload) -> Result<(), LlmError> {
    let scanner = delta.scanner()?;
    let mut updates = Vec::new();
    for entry in
        scanner
            .root_node()
            .object_entries()
            .ok_or_else(|| LlmError::StreamInterrupted {
                message: "Interactions step delta must be an object".into(),
            })?
    {
        if entry.key_units == "type".encode_utf16().collect::<Vec<_>>() {
            continue;
        }
        let raw = crate::exact_json::serialize_json_node(
            entry.value,
            crate::exact_json::JsonEncoding::Serde,
        )?;
        updates.push((entry.key_units.to_vec(), raw));
    }
    step.source = replace_root_fields(&step.source, updates)?;
    Ok(())
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
        let event_text = std::str::from_utf8(frame).map_err(|error| {
            Self::interrupted(format!("invalid Interactions stream UTF-8: {error}"))
        })?;
        let event_source = JsonPayload::parse(event_text).map_err(|error| {
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
                let mut source = event_source.payload_at("/step").ok_or_else(|| {
                    Self::interrupted("Interactions step start has no source step")
                })?;
                if value["type"] == "model_output" && value.get("content").is_none() {
                    value["content"] = json!([]);
                    let defaults = JsonPayload::parse(r#"{"content":[]}"#)
                        .expect("content default is valid JSON");
                    source = source.with_missing_object_fields(
                        &defaults,
                        crate::exact_json::JsonEncoding::Serde,
                    )?;
                }
                self.steps.insert(
                    index,
                    Step {
                        arguments_decoder: Default::default(),
                        value,
                        source,
                        arguments: Vec::new(),
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
                let delta_source = event_source
                    .payload_at("/delta")
                    .ok_or_else(|| Self::interrupted("Interactions delta has no source object"))?;
                match delta["type"].as_str() {
                    Some("arguments_delta") if step.value["type"] == "function_call" => {
                        let units =
                            delta_source.string_units_at("/arguments").ok_or_else(|| {
                                Self::interrupted("Interactions argument delta must be a string")
                            })?;
                        let display = delta_source
                            .string_at("/arguments")
                            .unwrap_or_else(|| String::from_utf16_lossy(&units));
                        let fragment =
                            step.arguments_decoder
                                .push(crate::response_json::DecodedText {
                                    text: display,
                                    utf16_code_units: Some(units),
                                })?;
                        step.arguments
                            .extend(fragment.utf16_code_units().iter().copied());
                    }
                    Some("text") if step.value["type"] == "model_output" => {
                        let text_units =
                            delta_source.string_units_at("/text").ok_or_else(|| {
                                Self::interrupted("Interactions text delta must contain text")
                            })?;
                        let text = delta_source.string_at("/text").ok_or_else(|| {
                            Self::interrupted("Interactions text delta must contain text")
                        })?;
                        let extend_text = step.value["content"]
                            .as_array()
                            .and_then(|parts| parts.last())
                            .is_some_and(|part| part["type"] == "text");
                        if extend_text {
                            let last = step.value["content"]
                                .as_array_mut()
                                .and_then(|parts| parts.last_mut())
                                .expect("text content was checked above");
                            let current = last["text"].as_str().unwrap_or("").to_owned();
                            last["text"] = json!(current + &text);
                            let mut source_parts = step
                                .source
                                .payload_at("/content")
                                .and_then(|content| content.array_items())
                                .ok_or_else(|| {
                                    Self::interrupted("Interactions model content is not an array")
                                })?;
                            let source_last = source_parts.last().cloned().ok_or_else(|| {
                                Self::interrupted("Interactions model content is empty")
                            })?;
                            let mut units =
                                source_last.string_units_at("/text").unwrap_or_default();
                            units.extend(text_units);
                            let exact_text = JsonPayload::string_from_utf16_code_units(&units);
                            *source_parts.last_mut().expect("source item exists") =
                                set_root_field(&source_last, "text", &exact_text)?;
                            let content = JsonPayload::array_from_items(
                                &source_parts,
                                crate::exact_json::JsonEncoding::Serde,
                            )?;
                            step.source = set_root_field(&step.source, "content", &content)?;
                        } else {
                            step.value["content"]
                                .as_array_mut()
                                .ok_or_else(|| {
                                    Self::interrupted("Interactions model content is not an array")
                                })?
                                .push(delta.clone());
                            append_content_part(step, delta_source)?;
                        }
                    }
                    Some("thought_signature") => {
                        step.value["signature"] = delta["signature"].clone();
                        if let Some(signature) = delta_source.payload_at("/signature") {
                            step.source = set_root_field(&step.source, "signature", &signature)?;
                        }
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
                        let mut summary = step
                            .source
                            .payload_at("/summary")
                            .and_then(|summary| summary.array_items())
                            .unwrap_or_default();
                        if let Some(content) = delta_source.payload_at("/content") {
                            summary.push(content);
                            let summary = JsonPayload::array_from_items(
                                &summary,
                                crate::exact_json::JsonEncoding::Serde,
                            )?;
                            step.source = set_root_field(&step.source, "summary", &summary)?;
                        }
                    }
                    Some("image" | "audio") if step.value["type"] == "model_output" => {
                        step.value["content"]
                            .as_array_mut()
                            .ok_or_else(|| {
                                Self::interrupted("malformed Interactions multimodal content")
                            })?
                            .push(delta.clone());
                        append_content_part(step, delta_source)?;
                    }
                    _ => {
                        if !delta.is_object() {
                            return Err(Self::interrupted(
                                "Interactions step delta must be an object",
                            ));
                        }
                        if step.value["type"] == "function_call" {
                            for key in ["id", "name"] {
                                let pointer = format!("/{key}");
                                if step
                                    .source
                                    .string_units_at(&pointer)
                                    .is_some_and(|units| !units.is_empty())
                                {
                                    if let Some(delta_value) = delta_source.payload_at(&pointer) {
                                        let current = step
                                            .source
                                            .payload_at(&pointer)
                                            .expect("present function identity has source");
                                        if !current.semantically_eq(&delta_value)? {
                                            return Err(Self::interrupted(
                                                "Interactions delta changes function call identity",
                                            ));
                                        }
                                    }
                                }
                            }
                            if let Some(arguments) = delta_source.payload_at("/arguments") {
                                if arguments.is_root_string() {
                                    return Err(Self::interrupted(
                                        "Interactions argument snapshot must be a JSON value",
                                    ));
                                }
                                // An argument snapshot supersedes any fragments received
                                // before it. Later fragments start a fresh JSON value.
                                step.arguments.clear();
                                step.arguments_decoder = Default::default();
                            }
                        }
                        for (key, value) in delta.as_object().unwrap() {
                            if key != "type" {
                                step.value[key] = value.clone();
                            }
                        }
                        merge_delta_fields(step, &delta_source)?;
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
                    let fragment = JsonPayloadFragment::from_utf16_code_units(std::mem::take(
                        &mut step.arguments,
                    ));
                    let arguments = JsonPayload::parse_fragments([&fragment]).map_err(|error| {
                        Self::interrupted(format!(
                            "invalid completed Interactions function arguments: {error}"
                        ))
                    })?;
                    step.source = set_root_field(&step.source, "arguments", &arguments)?;
                    step.value["arguments"] =
                        serde_json::from_str(arguments.as_json_source()).unwrap_or(Value::Null);
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
                let mut body_source = event_source.payload_at("/interaction").ok_or_else(|| {
                    Self::interrupted("Interactions terminal event has no source interaction")
                })?;
                self.usage =
                    super::decode::usage_report_source(body_source.payload_at("/usage").as_ref());
                if body["id"].as_str() != self.id.as_deref() {
                    return Err(Self::interrupted(
                        "Interactions terminal id differs from start",
                    ));
                }
                let streamed_values: Vec<_> =
                    self.steps.values().map(|step| step.value.clone()).collect();
                let streamed_sources: Vec<_> = self
                    .steps
                    .values()
                    .map(|step| step.source.clone())
                    .collect();
                if let Some(terminal) = body.get("steps") {
                    terminal.as_array().ok_or_else(|| {
                        Self::interrupted("Interactions terminal steps must be an array")
                    })?;
                    let terminal_len = body_source.array_len("/steps").ok_or_else(|| {
                        Self::interrupted("Interactions terminal steps must be an array")
                    })?;
                    if terminal_len != streamed_values.len() {
                        return Err(Self::interrupted(
                            "Interactions terminal calls differ from completed streamed steps",
                        ));
                    }
                    let mut confirmed_ids = std::collections::BTreeSet::new();
                    for index in 0..terminal_len {
                        let final_step =
                            body_source.array_item_at("/steps", index).ok_or_else(|| {
                                Self::interrupted("Interactions terminal step source is missing")
                            })?;
                        if !final_step.string_is("/type", "function_call") {
                            if final_step.string_units_at("/type")
                                != streamed_sources[index].string_units_at("/type")
                            {
                                return Err(Self::interrupted(
                                    "Interactions terminal steps differ from completed streamed steps",
                                ));
                            }
                            continue;
                        }
                        let id = final_step.string_units_at("/id").ok_or_else(|| {
                            Self::interrupted("Interactions terminal call has no id")
                        })?;
                        if !confirmed_ids.insert(id.clone()) {
                            return Err(Self::interrupted(
                                "Interactions terminal repeated a function call id",
                            ));
                        }
                        let mut candidates = self.steps.values().filter(|step| {
                            step.source.string_is("/type", "function_call")
                                && step.source.string_units_at("/id").as_ref() == Some(&id)
                        });
                        let streamed_step = candidates.next().ok_or_else(|| {
                            Self::interrupted("Interactions terminal call has no streamed identity")
                        })?;
                        if candidates.next().is_some() {
                            return Err(Self::interrupted(
                                "Interactions stream repeated a function call id",
                            ));
                        }
                        let final_identity = final_step.without_object_field("arguments")?;
                        let streamed_identity =
                            streamed_step.source.without_object_field("arguments")?;
                        let null = JsonPayload::parse("null").expect("null is valid JSON");
                        let final_arguments = final_step
                            .payload_at("/arguments")
                            .unwrap_or_else(|| null.clone());
                        let streamed_arguments = streamed_step
                            .source
                            .payload_at("/arguments")
                            .unwrap_or(null);
                        if !final_identity.semantically_eq(&streamed_identity)?
                            || !final_arguments.semantically_eq(&streamed_arguments)?
                        {
                            return Err(Self::interrupted(
                                "Interactions terminal calls differ from completed streamed steps",
                            ));
                        }
                    }
                    if confirmed_ids.len()
                        != streamed_sources
                            .iter()
                            .filter(|step| step.string_is("/type", "function_call"))
                            .count()
                    {
                        return Err(Self::interrupted(
                            "Interactions terminal omitted a streamed function call",
                        ));
                    }
                } else {
                    body["steps"] = Value::Array(streamed_values);
                    let steps = JsonPayload::array_from_items(
                        &streamed_sources,
                        crate::exact_json::JsonEncoding::Serde,
                    )?;
                    body_source = set_root_field(&body_source, "steps", &steps)?;
                }
                // Validation is atomic: no function/native call escapes if a later
                // member, duplicate ID or terminal status is malformed.
                let response = super::decode::response(&body_source, &self.context)?;
                let mut events = vec![];
                for (block, content) in response.message.content.into_iter().enumerate() {
                    events.push(match content {
                        ContentBlock::Text { text, .. } => StreamEvent::TextDelta { block, text },
                        ContentBlock::TextJsUtf16 {
                            text,
                            utf16_code_units,
                            ..
                        } => StreamEvent::TextDeltaJsUtf16 {
                            block,
                            text,
                            utf16_code_units,
                        },
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
                            arguments_fragment: JsonPayloadFragment::from_text(
                                input.as_json_source(),
                            ),
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
                            return Err(Self::interrupted("unexpected Interactions decoded block"));
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
            "error" => {
                let scanner = event_source.scanner()?;
                let status = scanner
                    .number_source_at("/status")
                    .and_then(|status| status.parse::<u16>().ok())
                    .unwrap_or(500);
                let selected_text = scanner.provider_error_text_at("/error/message", None)?;
                let provider_text = match &selected_text {
                    Some(text) => text.prefixed(&format!("{status} ")),
                    None => {
                        scanner.provider_error_text_or_body_at("/error/message", Some(status))?
                    }
                };
                let body = scanner.provider_error_classification_body(
                    "/error",
                    &["message", "code", "status"],
                    selected_text.as_ref(),
                );
                Err(
                    crate::codecs::gemini::classify_error(status, &body, None, false)
                        .with_provider_error_text(provider_text),
                )
            }
            _ => Ok(vec![StreamEvent::ProviderEvent {
                protocol: ProtocolFamily::GeminiInteractions,
                payload: event_source,
            }]),
        }
    }
    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        if self.ended {
            Ok(vec![])
        } else {
            Err(Self::interrupted(
                "Interactions stream interrupted before terminal; submission outcome may be unknown",
            ))
        }
    }
    fn usage_report(&self) -> UsageReport {
        self.usage.clone()
    }
}
