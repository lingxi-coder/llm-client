use crate::codecs::{CodecContext, EventDecoder};
use crate::protocol::*;
use std::collections::BTreeMap;

struct Step {
    arguments_decoder: crate::response_json::ArgumentJsonDecoder,
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
            scanner
                .raw_source_for_node(entry.value)
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "Interactions source field has no raw JSON span".into(),
                })?
                .to_owned()
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

/// Assemble provider subtrees without applying outbound codec number semantics.
fn source_array(items: &[JsonPayload]) -> Result<JsonPayload, LlmError> {
    let invalid = |error: serde_json::Error| LlmError::InvalidRequest {
        message: format!("Interactions source array is invalid: {error}"),
    };
    let source = serde_json::to_string(items).map_err(invalid)?;
    JsonPayload::parse(&source).map_err(invalid)
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
    let content = source_array(&parts)?;
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
        let raw = scanner
            .raw_source_for_node(entry.value)
            .ok_or_else(|| LlmError::StreamInterrupted {
                message: "Interactions delta field has no raw JSON span".into(),
            })?
            .to_owned();
        updates.push((entry.key_units.to_vec(), raw));
    }
    step.source = replace_root_fields(&step.source, updates)?;
    Ok(())
}

fn decoded_string(source: &JsonPayload, pointer: &str) -> Option<String> {
    String::from_utf16(&source.string_units_at(pointer)?).ok()
}

fn string_is(source: &JsonPayload, pointer: &str, expected: &str) -> bool {
    let expected_units = expected.encode_utf16().collect::<Vec<_>>();
    source.string_units_at(pointer).as_deref() == Some(expected_units.as_slice())
}

fn append_text_delta(step: &mut Step, delta: &JsonPayload) -> Result<(), LlmError> {
    let text_units = delta.string_units_at("/text").ok_or_else(|| {
        InteractionsDecoder::interrupted("Interactions text delta must contain text")
    })?;
    let content = step
        .source
        .payload_at("/content")
        .and_then(|content| content.array_items())
        .ok_or_else(|| {
            InteractionsDecoder::interrupted("Interactions model content is not an array")
        })?;

    let mut content = content;
    if let Some(last) = content
        .last_mut()
        .filter(|part| string_is(part, "/type", "text"))
    {
        let mut units = last.string_units_at("/text").unwrap_or_default();
        units.extend(text_units);
        let text = JsonPayload::string_from_utf16_code_units(&units);
        *last = set_root_field(last, "text", &text)?;
        let content = source_array(&content)?;
        step.source = set_root_field(&step.source, "content", &content)?;
    } else {
        append_content_part(step, delta.clone())?;
    }
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

    fn index(event: &JsonPayload) -> Result<usize, LlmError> {
        event
            .unsigned_integer_at("/index")
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
        let event_text = std::str::from_utf8(frame).map_err(|error| {
            Self::interrupted(format!("invalid Interactions stream UTF-8: {error}"))
        })?;
        let event = JsonPayload::parse(event_text).map_err(|error| {
            Self::interrupted(format!("invalid Interactions stream JSON: {error}"))
        })?;
        if !event.is_root_object() {
            return Err(Self::interrupted("Interactions event must be an object"));
        }
        let kind = decoded_string(&event, "/event_type")
            .ok_or_else(|| Self::interrupted("Interactions event is missing event_type"))?;
        self.buffered = self.buffered.saturating_add(frame.len());
        if self.buffered > 16 * 1024 * 1024 {
            return Err(Self::interrupted("Interactions response exceeds 16 MiB"));
        }

        match kind.as_str() {
            "interaction.created" => {
                if self.started {
                    return Err(Self::interrupted("duplicate Interactions start"));
                }
                let interaction = event
                    .payload_at("/interaction")
                    .ok_or_else(|| Self::interrupted("Interactions start has no interaction"))?;
                let id = decoded_string(&interaction, "/id")
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| Self::interrupted("Interactions start has no id"))?;
                let model = decoded_string(&interaction, "/model")
                    .unwrap_or_else(|| self.context.request_model().to_owned());
                self.id = Some(id.clone());
                self.started = true;
                Ok(vec![StreamEvent::Start {
                    model,
                    response_id: Some(ResponseId(id)),
                }])
            }
            "step.start" => {
                if !self.started {
                    return Err(Self::interrupted("Interactions step before start"));
                }
                let index = Self::index(&event)?;
                let mut source = event.payload_at("/step").ok_or_else(|| {
                    Self::interrupted("Interactions step start has no source step")
                })?;
                if !source.is_root_object() || self.steps.contains_key(&index) {
                    return Err(Self::interrupted(
                        "duplicate or malformed Interactions step start",
                    ));
                }
                if string_is(&source, "/type", "model_output")
                    && source.payload_at("/content").is_none()
                {
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
                        source,
                        arguments: Vec::new(),
                        stopped: false,
                    },
                );
                Ok(vec![])
            }
            "step.delta" => {
                let index = Self::index(&event)?;
                let step = self
                    .steps
                    .get_mut(&index)
                    .filter(|step| !step.stopped)
                    .ok_or_else(|| Self::interrupted("Interactions delta has no open step"))?;
                let delta = event
                    .payload_at("/delta")
                    .ok_or_else(|| Self::interrupted("Interactions delta has no source object"))?;
                if !delta.is_root_object() {
                    return Err(Self::interrupted(
                        "Interactions step delta must be an object",
                    ));
                }
                let delta_type = decoded_string(&delta, "/type");
                let step_is_call = string_is(&step.source, "/type", "function_call");
                let step_is_output = string_is(&step.source, "/type", "model_output");
                match delta_type.as_deref() {
                    Some("arguments_delta") if step_is_call => {
                        let units = delta.string_units_at("/arguments").ok_or_else(|| {
                            Self::interrupted("Interactions argument delta must be a string")
                        })?;
                        let display = String::from_utf16_lossy(&units);
                        let fragment =
                            step.arguments_decoder
                                .push(crate::response_json::DecodedText {
                                    text: display,
                                    utf16_code_units: Some(units),
                                })?;
                        step.arguments
                            .extend(fragment.utf16_code_units().iter().copied());
                    }
                    Some("text") if step_is_output => append_text_delta(step, &delta)?,
                    Some("thought_signature") => {
                        if let Some(signature) = delta.payload_at("/signature") {
                            step.source = set_root_field(&step.source, "signature", &signature)?;
                        }
                    }
                    Some("thought_summary") => {
                        let content = delta.payload_at("/content").ok_or_else(|| {
                            Self::interrupted("Interactions thought summary has no content")
                        })?;
                        let mut summary = match step.source.payload_at("/summary") {
                            Some(summary) => summary.array_items().ok_or_else(|| {
                                Self::interrupted("malformed Interactions thought summary")
                            })?,
                            None => Vec::new(),
                        };
                        summary.push(content);
                        let summary = source_array(&summary)?;
                        step.source = set_root_field(&step.source, "summary", &summary)?;
                    }
                    Some("image" | "audio") if step_is_output => {
                        append_content_part(step, delta)?;
                    }
                    _ => {
                        if step_is_call {
                            for key in ["id", "name"] {
                                let pointer = format!("/{key}");
                                if step
                                    .source
                                    .string_units_at(&pointer)
                                    .is_some_and(|units| !units.is_empty())
                                {
                                    if let Some(delta_value) = delta.payload_at(&pointer) {
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
                            if let Some(arguments) = delta.payload_at("/arguments") {
                                if arguments.is_root_string() {
                                    return Err(Self::interrupted(
                                        "Interactions argument snapshot must be a JSON value",
                                    ));
                                }
                                // A complete snapshot supersedes fragments received before it.
                                step.arguments.clear();
                                step.arguments_decoder = Default::default();
                            }
                        }
                        merge_delta_fields(step, &delta)?;
                    }
                }
                Ok(vec![])
            }
            "step.stop" => {
                let index = Self::index(&event)?;
                let step = self
                    .steps
                    .get_mut(&index)
                    .filter(|step| !step.stopped)
                    .ok_or_else(|| Self::interrupted("Interactions stop has no open step"))?;
                if string_is(&step.source, "/type", "function_call") && !step.arguments.is_empty() {
                    let fragment = JsonPayloadFragment::from_utf16_code_units(std::mem::take(
                        &mut step.arguments,
                    ));
                    let arguments = JsonPayload::parse_fragments([&fragment]).map_err(|error| {
                        Self::interrupted(format!(
                            "invalid completed Interactions function arguments: {error}"
                        ))
                    })?;
                    step.source = set_root_field(&step.source, "arguments", &arguments)?;
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
                let mut body_source = event.payload_at("/interaction").ok_or_else(|| {
                    Self::interrupted("Interactions terminal event has no source interaction")
                })?;
                if !body_source.is_root_object() {
                    return Err(Self::interrupted(
                        "Interactions terminal interaction is malformed",
                    ));
                }
                self.usage =
                    super::decode::usage_report_source(body_source.payload_at("/usage").as_ref());
                if decoded_string(&body_source, "/id").as_deref() != self.id.as_deref() {
                    return Err(Self::interrupted(
                        "Interactions terminal id differs from start",
                    ));
                }
                let streamed_sources: Vec<_> = self
                    .steps
                    .values()
                    .map(|step| step.source.clone())
                    .collect();
                if let Some(terminal_steps) = body_source.payload_at("/steps") {
                    let terminal_len = terminal_steps.array_len("").ok_or_else(|| {
                        Self::interrupted("Interactions terminal steps must be an array")
                    })?;
                    if terminal_len != streamed_sources.len() {
                        return Err(Self::interrupted(
                            "Interactions terminal calls differ from completed streamed steps",
                        ));
                    }
                    let mut confirmed_ids = std::collections::BTreeSet::new();
                    for (index, streamed_source) in
                        streamed_sources.iter().enumerate().take(terminal_len)
                    {
                        let final_step = terminal_steps.array_item(index).ok_or_else(|| {
                            Self::interrupted("Interactions terminal step source is missing")
                        })?;
                        if !string_is(&final_step, "/type", "function_call") {
                            if final_step.string_units_at("/type")
                                != streamed_source.string_units_at("/type")
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
                            string_is(&step.source, "/type", "function_call")
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
                            .filter(|step| string_is(step, "/type", "function_call"))
                            .count()
                    {
                        return Err(Self::interrupted(
                            "Interactions terminal omitted a streamed function call",
                        ));
                    }
                } else {
                    let steps = source_array(&streamed_sources)?;
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
                let scanner = event.scanner()?;
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
                payload: event,
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

#[cfg(test)]
mod source_snapshot_tests {
    use super::InteractionsDecoder;
    use crate::codecs::{CodecContext, EventDecoder, RequestMode};
    use crate::protocol::{ProtocolFamily, ProviderProfile, StreamEvent};
    use serde_json::json;

    #[test]
    fn argument_snapshot_keeps_raw_numbers_and_utf16_source_through_terminal_event() {
        let profile: ProviderProfile = serde_json::from_value(json!({
            "provider_id": "test",
            "profile_name": "test",
            "base_url": "https://example.invalid",
            "protocol": "gemini_interactions",
            "auth": "none",
            "models": [{
                "display_model": "model",
                "request_model": "model",
                "billing_model": "model"
            }]
        }))
        .unwrap();
        let context = CodecContext::new(&profile, "model", RequestMode::Stream);
        assert_eq!(
            context.profile().protocol,
            ProtocolFamily::GeminiInteractions
        );
        let mut decoder = InteractionsDecoder::new(&context);

        decoder
            .decode_frame(
                br#"{"event_type":"interaction.created","interaction":{"id":"interaction-1","model":"model"}}"#,
            )
            .unwrap();
        decoder
            .decode_frame(
                br#"{"event_type":"step.start","index":0,"step":{"type":"function_call","id":"call-1","name":"run","arguments":{"initial":true}}}"#,
            )
            .unwrap();
        decoder
            .decode_frame(
                br#"{"event_type":"step.delta","index":0,"delta":{"type":"function_call","arguments":{"z":1e+03,"overflow":1e400,"\ud800":"value \udfff"}}}"#,
            )
            .unwrap();
        decoder
            .decode_frame(br#"{"event_type":"step.stop","index":0}"#)
            .unwrap();
        let events = decoder
            .decode_frame(
                br#"{"event_type":"interaction.completed","interaction":{"id":"interaction-1","status":"requires_action"}}"#,
            )
            .unwrap();

        let expected = r#"{"z":1e+03,"overflow":1e400,"\ud800":"value \udfff"}"#;
        let call = events
            .iter()
            .find_map(|event| match event {
                StreamEvent::ToolCallDelta {
                    id,
                    provider_id,
                    arguments_fragment,
                    ..
                } => Some((id, provider_id, arguments_fragment)),
                _ => None,
            })
            .expect("terminal response emits the streamed function call");
        assert_eq!(call.0.as_str(), "call-1");
        assert_eq!(call.1.as_deref(), Some("call-1"));
        assert_eq!(
            call.2.utf16_code_units(),
            expected.encode_utf16().collect::<Vec<_>>()
        );
    }
}
