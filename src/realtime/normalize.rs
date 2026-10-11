use super::*;
use base64::{engine::general_purpose::STANDARD, Engine as _};

pub(crate) fn decode_json_events(
    frame: RealtimeFrame,
    format: RealtimeAudioFormat,
) -> Result<Vec<RealtimeEvent>, RealtimeError> {
    let RealtimeFrame::Text(bytes) = frame else {
        return Err(RealtimeError::Codec {
            message: "realtime events require JSON text frames".into(),
        });
    };
    let native = serde_json::from_slice(&bytes).map_err(|error| RealtimeError::Codec {
        message: format!("invalid realtime JSON: {error}"),
    })?;
    normalize_json_events(native, format, false)
}

pub(crate) fn normalize_json_events(
    native: Value,
    format: RealtimeAudioFormat,
    arguments_done: bool,
) -> Result<Vec<RealtimeEvent>, RealtimeError> {
    let name = native
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| RealtimeError::Codec {
            message: "realtime event lacks string type".into(),
        })?;
    let string = |key: &str| native.get(key).and_then(Value::as_str).map(str::to_owned);
    let response = native.get("response");
    let turn_id = string("response_id").or_else(|| {
        response
            .and_then(|r| r.get("id"))
            .and_then(Value::as_str)
            .map(str::to_owned)
    });
    let item_id = string("item_id");
    let mut events = Vec::new();
    match name {
        "session.created" | "session.updated" => events.push(RealtimeEvent::SessionReady),
        "response.created" => events.push(RealtimeEvent::TurnStarted { turn_id }),
        "response.done" => {
            let status = response
                .and_then(|r| r.get("status"))
                .and_then(Value::as_str)
                .map(str::to_owned);
            events.push(RealtimeEvent::TurnCompleted {
                turn_id: turn_id.clone(),
                status: status.clone(),
            });
            if let Some(usage) = response
                .and_then(|r| r.get("usage"))
                .filter(|u| !u.is_null())
            {
                events.push(RealtimeEvent::Usage {
                    turn_id: turn_id.clone(),
                    input_tokens: usage.get("input_tokens").and_then(Value::as_u64),
                    output_tokens: usage.get("output_tokens").and_then(Value::as_u64),
                    total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
                    native: usage.clone(),
                });
            }
            if status.as_deref() == Some("cancelled") {
                let call_ids = response
                    .and_then(|r| r.get("output"))
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|item| item.get("call_id").and_then(Value::as_str))
                    .map(str::to_owned)
                    .collect();
                events.push(RealtimeEvent::ToolCancelled { call_ids });
                events.push(RealtimeEvent::Interrupted);
            }
        }
        "response.output_audio.delta" | "response.audio.delta" => {
            let raw = string("delta").ok_or_else(|| RealtimeError::Codec {
                message: "audio delta lacks base64 delta".into(),
            })?;
            let data = STANDARD.decode(raw).map_err(|_| RealtimeError::Codec {
                message: "audio delta is invalid base64".into(),
            })?;
            events.push(RealtimeEvent::AudioDelta {
                data: data.into(),
                format,
                item_id,
            });
        }
        "conversation.item.input_audio_transcription.delta"
        | "conversation.item.input_audio_transcription.updated"
        | "conversation.item.input_audio_transcription.completed"
        | "response.output_audio_transcript.delta"
        | "response.output_audio_transcript.done"
        | "response.audio_transcript.delta"
        | "response.audio_transcript.done" => {
            let direction = if name.starts_with("conversation.") {
                RealtimeTranscriptDirection::Input
            } else {
                RealtimeTranscriptDirection::Output
            };
            let final_chunk = name.ends_with(".done") || name.ends_with(".completed");
            let text = if final_chunk {
                string("transcript").or_else(|| string("text"))
            } else {
                string("delta").or_else(|| string("transcript"))
            }
            .ok_or_else(|| RealtimeError::Codec {
                message: "transcription lacks text".into(),
            })?;
            events.push(RealtimeEvent::Transcript {
                direction,
                update: if final_chunk || name.ends_with(".updated") {
                    RealtimeTranscriptUpdate::Replace
                } else {
                    RealtimeTranscriptUpdate::Delta
                },
                text,
                item_id,
                turn_id,
                final_chunk,
            });
        }
        "response.output_text.delta"
        | "response.text.delta"
        | "response.output_text.done"
        | "response.text.done" => {
            let final_chunk = name.ends_with(".done");
            let text = string(if final_chunk { "text" } else { "delta" }).ok_or_else(|| {
                RealtimeError::Codec {
                    message: "text event lacks text".into(),
                }
            })?;
            events.push(RealtimeEvent::TextDelta {
                text,
                item_id,
                final_chunk,
            });
        }
        "response.output_item.done" | "response.function_call_arguments.done" => {
            let item = if name == "response.output_item.done" {
                native.get("item").unwrap_or(&Value::Null)
            } else {
                &native
            };
            if (name == "response.function_call_arguments.done" && arguments_done)
                || (name == "response.output_item.done"
                    && !arguments_done
                    && item.get("type").and_then(Value::as_str) == Some("function_call"))
            {
                if let (Some(call_id), Some(name)) = (
                    item.get("call_id").and_then(Value::as_str),
                    item.get("name").and_then(Value::as_str),
                ) {
                    let arguments = match item.get("arguments") {
                        Some(Value::String(raw)) => {
                            serde_json::from_str(raw).map_err(|_| RealtimeError::Codec {
                                message: "function arguments are invalid JSON".into(),
                            })?
                        }
                        Some(value) => value.clone(),
                        None => serde_json::json!({}),
                    };
                    if call_id.trim().is_empty() || name.trim().is_empty() || !arguments.is_object()
                    {
                        return Err(RealtimeError::Codec {
                            message: "function call requires nonempty ID/name and object arguments"
                                .into(),
                        });
                    }
                    // argument-done and item-done may both be emitted. The generic
                    // adapter only publishes item-done where that event is used.
                    events.push(RealtimeEvent::ToolCall {
                        call_id: call_id.into(),
                        name: name.into(),
                        arguments,
                    });
                }
            }
        }
        "input_audio_buffer.speech_started" => events.push(RealtimeEvent::UserSpeechStarted),
        "error" => {
            let error = native.get("error").unwrap_or(&Value::Null);
            events.push(RealtimeEvent::ProviderError {
                code: error.get("code").and_then(Value::as_str).map(str::to_owned),
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("realtime provider error")
                    .into(),
            });
        }
        _ => {}
    }
    // Preserve every recognized provider field, including usage details,
    // content indices, timing, and provider-specific continuation state.
    events.push(RealtimeEvent::ProviderEvent {
        name: name.to_owned(),
        native,
    });
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn provider_transcript_snapshots_are_not_misreported_as_deltas() {
        let native = json!({"type":"conversation.item.input_audio_transcription.updated","item_id":"u1","transcript":"whole snapshot","audio_start_ms":10});
        let events = normalize_json_events(
            native.clone(),
            RealtimeAudioFormat::Pcm16 {
                sample_rate_hz: 24000,
            },
            true,
        )
        .unwrap();
        assert!(matches!(
            &events[0],
            RealtimeEvent::Transcript {
                direction: RealtimeTranscriptDirection::Input,
                update: RealtimeTranscriptUpdate::Replace,
                final_chunk: false,
                ..
            }
        ));
        assert!(
            matches!(&events[1],RealtimeEvent::ProviderEvent{native:value,..} if value==&native)
        );
    }
    #[test]
    fn provider_function_argument_done_and_item_done_are_not_double_executed() {
        let args = json!({"type":"response.function_call_arguments.done","call_id":"c1","name":"lookup","arguments":"{}"});
        let item = json!({"type":"response.output_item.done","item":{"type":"function_call","call_id":"c1","name":"lookup","arguments":"{}"}});
        for publish_args in [false, true] {
            let mut events =
                normalize_json_events(args.clone(), RealtimeAudioFormat::G711MuLaw, publish_args)
                    .unwrap();
            events.extend(
                normalize_json_events(item.clone(), RealtimeAudioFormat::G711MuLaw, publish_args)
                    .unwrap(),
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, RealtimeEvent::ToolCall { .. }))
                    .count(),
                1
            );
        }
    }
}
