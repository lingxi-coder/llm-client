//! The Responses API stream decoder.
//!
//! Every event names the output index it belongs to, so block indices come from
//! the wire rather than being synthesised.

use super::decode;
use crate::codecs::file_search_decode::FileSearchStream;
use crate::codecs::usage;
use crate::codecs::web_search_decode::{self, SearchStream};
use crate::codecs::EventDecoder;
use crate::protocol::{
    LlmError, LlmErrorKind, NativeExtension, ResponseId, StopReason, StreamEvent, ToolUseId,
};
use crate::providers::openai::computer::OpenAiComputerCall;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
struct StreamedFunctionCall {
    index: usize,
    id: ToolUseId,
    name: String,
    arguments: String,
}

#[derive(Debug, Default)]
pub struct ResponsesStreamDecoder {
    inference: crate::codecs::inference::StreamInference,
    started: bool,
    created_response_id: Option<ResponseId>,
    native_items: std::collections::BTreeSet<usize>,
    /// Function calls learned from `output_item.added` and argument deltas.
    calls: Vec<StreamedFunctionCall>,
    /// Final computer items observed before the terminal response.
    done_computer_calls: BTreeMap<usize, OpenAiComputerCall>,
    /// Stable computer identity learned when an output item is added.
    added_computer_calls: BTreeMap<usize, (String, String)>,
    /// A computer item was observed, but only a terminal response can publish it.
    observed_computer_call: bool,
    /// The provider's usage object as sent. Kept raw so the report can be
    /// checked for self-consistency: the buckets we publish are derived by
    /// subtraction, and a saturated subtraction must not read as a measurement.
    ///
    /// This wire restates the whole object each time it reports, so the latest
    /// one wins outright; only the Anthropic wire splits its report across
    /// frames and needs a field-by-field fold.
    usage_raw: Option<Value>,
    stop: Option<StopReason>,
    saw_tool_call: bool,
    saw_refusal: bool,
    requires_action: bool,
    openai_approval_semantics: bool,
    openai_tool_search_semantics: bool,
    qwen_code_interpreter: bool,
    chatgpt_plan: bool,
    http_status: Option<u16>,
    request_id: Option<String>,
    done: bool,
    search: SearchStream,
    file_search: FileSearchStream,
}

impl EventDecoder for ResponsesStreamDecoder {
    fn decode_frame(&mut self, frame: &[u8]) -> Result<Vec<StreamEvent>, LlmError> {
        if self.done {
            return Ok(Vec::new());
        }
        let text = std::str::from_utf8(frame).map_err(|_| LlmError::InvalidRequest {
            message: "stream frame is not valid UTF-8".to_owned(),
        })?;
        let data = text.trim();
        let mut out = Vec::new();

        // This wire ends at `response.completed` and sends no sentinel, but an
        // OpenAI-compatible gateway in front of it may append one.
        if data == "[DONE]" {
            if self.chatgpt_plan {
                return Err(LlmError::StreamInterrupted {
                    message: "ChatGPT plan stream ended without response.completed".into(),
                });
            }
            if self.observed_computer_call {
                return Err(LlmError::StreamInterrupted {
                    message: "Responses stream ended before its computer call was confirmed by a terminal response".into(),
                });
            }
            self.finish_into(&mut out);
            return Ok(out);
        }
        if data.is_empty() {
            return Ok(out);
        }
        let root: Value = serde_json::from_str(data).map_err(|_| LlmError::InvalidRequest {
            message: "stream frame is not valid JSON".to_owned(),
        })?;

        if let Some(u) = root.pointer("/response/usage").filter(|v| !v.is_null()) {
            self.usage_raw = Some(u.clone());
        }

        let index = |r: &Value| {
            r.get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or_default() as usize
        };
        let delta = |r: &Value| {
            r.get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };

        self.inference.observe(&root, &mut out);
        match root.get("type").and_then(Value::as_str) {
            Some("response.output_text.annotation.added") => {
                if root["annotation"]["type"].as_str() == Some("url_citation") {
                    self.search.emit(web_search_decode::result(serde_json::json!({"annotations":[{
                        "output_index": index(&root), "content_index":root.get("content_index").and_then(Value::as_u64).unwrap_or(0), "annotation":root["annotation"]
                    }]})), &mut out);
                }
            }
            Some("response.output_item.done") => {
                let item = &root["item"];
                if let Some((id, call_id)) = self.added_computer_calls.get(&index(&root)) {
                    if item.get("type").and_then(Value::as_str) != Some("computer_call")
                        || item.get("id").and_then(Value::as_str) != Some(id.as_str())
                        || item.get("call_id").and_then(Value::as_str) != Some(call_id.as_str())
                    {
                        return Err(LlmError::InvalidRequest {
                            message:
                                "Responses computer call identity changed before output_item.done"
                                    .into(),
                        });
                    }
                }
                if item.get("type").and_then(Value::as_str) == Some("computer_call") {
                    self.observed_computer_call = true;
                    let call = OpenAiComputerCall::from_response_item(item)?;
                    if self
                        .done_computer_calls
                        .insert(index(&root), call)
                        .is_some()
                    {
                        return Err(LlmError::InvalidRequest {
                            message:
                                "Responses stream repeated a completed computer call output index"
                                    .into(),
                        });
                    }
                }
                out.push(StreamEvent::BlockEnd {
                    block: index(&root),
                });
                if item.get("type").and_then(Value::as_str) != Some("computer_call") {
                    self.emit_native(index(&root), item, &mut out);
                }
                self.saw_refusal |= decode::has_refusal(item);
                if item.get("type").and_then(Value::as_str) == Some("file_search_call") {
                    self.file_search
                        .emit(&serde_json::json!({"output":[item]}), &mut out);
                }
                if item["type"].as_str() == Some("web_search_call") {
                    self.search.emit(
                        web_search_decode::result(serde_json::json!({"web_search_calls":[item]})),
                        &mut out,
                    );
                }
            }
            Some("response.created") => {
                if !self.started {
                    self.started = true;
                    self.created_response_id = root
                        .pointer("/response/id")
                        .and_then(Value::as_str)
                        .filter(|id| !id.is_empty())
                        .map(ResponseId::new);
                    out.push(StreamEvent::Start {
                        model: root
                            .get("response")
                            .and_then(|r| r.get("model"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        response_id: self.created_response_id.clone(),
                    });
                }
            }
            Some("response.output_item.added") => {
                let item = root.get("item").unwrap_or(&Value::Null);
                self.saw_refusal |= decode::has_refusal(item);
                if item.get("type").and_then(Value::as_str) == Some("computer_call") {
                    self.observed_computer_call = true;
                    let id = item.get("id").and_then(Value::as_str).ok_or_else(|| {
                        LlmError::InvalidRequest {
                            message: "streamed computer call has no item id".into(),
                        }
                    })?;
                    let call_id = item.get("call_id").and_then(Value::as_str).ok_or_else(|| {
                        LlmError::InvalidRequest {
                            message: "streamed computer call has no call_id".into(),
                        }
                    })?;
                    if self
                        .added_computer_calls
                        .insert(index(&root), (id.to_owned(), call_id.to_owned()))
                        .is_some()
                    {
                        return Err(LlmError::InvalidRequest {
                            message: "Responses stream repeated a computer call output index"
                                .into(),
                        });
                    }
                }
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    self.saw_tool_call = true;
                    let id = ToolUseId::new(
                        item.get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    );
                    let name = item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    let arguments = item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    // Only this event names the call; the argument deltas that
                    // follow carry the index alone.
                    self.calls.push(StreamedFunctionCall {
                        index: index(&root),
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    });
                    out.push(StreamEvent::ToolCallDelta {
                        block: index(&root),
                        id,
                        caller: None,
                        toolset_name: None,
                        name,
                        arguments_fragment: arguments,
                        provider_id: None,
                    });
                }
            }
            Some("response.output_text.delta") => out.push(StreamEvent::TextDelta {
                block: index(&root),
                text: delta(&root),
            }),
            Some("response.refusal.delta" | "response.refusal.done") => {
                self.saw_refusal = true;
            }
            Some("response.content_part.added" | "response.content_part.done") => {
                self.saw_refusal |= decode::has_refusal(root.get("part").unwrap_or(&Value::Null));
            }
            Some("response.reasoning_text.delta" | "response.reasoning_summary_text.delta") => out
                .push(StreamEvent::ReasoningDelta {
                    block: index(&root),
                    text: delta(&root),
                }),
            Some(
                "response.code_interpreter_call.in_progress"
                | "response.code_interpreter_call.interpreting"
                | "response.code_interpreter_call.completed",
            ) if self.qwen_code_interpreter => out.push(StreamEvent::ProviderContent {
                block: index(&root),
                protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
                value: root.clone(),
            }),
            Some("response.function_call_arguments.delta") => {
                let i = index(&root);
                if let Some(call) = self.calls.iter_mut().find(|call| call.index == i) {
                    let fragment = delta(&root);
                    call.arguments.push_str(&fragment);
                    out.push(StreamEvent::ToolCallDelta {
                        block: i,
                        id: call.id.clone(),
                        caller: None,
                        toolset_name: None,
                        name: call.name.clone(),
                        arguments_fragment: fragment,
                        provider_id: None,
                    });
                } else {
                    return Err(LlmError::InvalidRequest {
                        message: "tool argument delta has no preceding call identity".into(),
                    });
                }
            }
            Some("response.function_call_arguments.done") => {
                let i = index(&root);
                let arguments = root
                    .get("arguments")
                    .and_then(Value::as_str)
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: "finalized function call has no arguments string".into(),
                    })?;
                let call = self
                    .calls
                    .iter_mut()
                    .find(|call| call.index == i)
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: "finalized function call has no preceding call identity".into(),
                    })?;
                if let Some(remainder) = arguments.strip_prefix(call.arguments.as_str()) {
                    if !remainder.is_empty() {
                        out.push(StreamEvent::ToolCallDelta {
                            block: i,
                            id: call.id.clone(),
                            caller: None,
                            toolset_name: None,
                            name: call.name.clone(),
                            arguments_fragment: remainder.to_owned(),
                            provider_id: None,
                        });
                    }
                } else {
                    let streamed: Value = serde_json::from_str(&call.arguments).map_err(|_| {
                        LlmError::InvalidRequest {
                            message: "streamed function call arguments are malformed".into(),
                        }
                    })?;
                    let finalized: Value =
                        serde_json::from_str(arguments).map_err(|_| LlmError::InvalidRequest {
                            message: "finalized function call arguments are malformed".into(),
                        })?;
                    if streamed != finalized {
                        return Err(LlmError::InvalidRequest {
                            message:
                                "finalized function call arguments conflict with streamed fragments"
                                    .into(),
                        });
                    }
                }
                call.arguments.clear();
                call.arguments.push_str(arguments);
            }
            Some("response.completed" | "response.incomplete") => {
                let response = root.get("response").unwrap_or(&Value::Null);
                let completed_event = root["type"] == "response.completed";
                if self.chatgpt_plan && root["type"] == "response.incomplete" {
                    return Err(LlmError::ProviderResponse {
                        status: self.http_status.unwrap_or(200),
                        request_id: self.request_id.clone(),
                        body: response.clone(),
                        classification: LlmErrorKind::StreamInterrupted,
                        retry_after: None,
                    });
                }
                if self.chatgpt_plan
                    && response
                        .get("status")
                        .and_then(Value::as_str)
                        .is_some_and(|status| status != "completed")
                {
                    return Err(LlmError::ProviderResponse {
                        status: self.http_status.unwrap_or(200),
                        request_id: self.request_id.clone(),
                        body: response.clone(),
                        classification: LlmErrorKind::ProviderInternal,
                        retry_after: None,
                    });
                }
                let output = response.get("output").and_then(Value::as_array);
                if completed_event && self.observed_computer_call && output.is_none() {
                    return Err(LlmError::InvalidRequest {
                        message: "completed Responses computer call response has no output array"
                            .into(),
                    });
                }
                if let Some(items) = output {
                    let terminal_status = response.get("status").and_then(Value::as_str);
                    let has_computer_calls = items.iter().any(|item| {
                        item.get("type").and_then(Value::as_str) == Some("computer_call")
                    });
                    if has_computer_calls || completed_event && self.observed_computer_call {
                        let terminal_id = response
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|id| !id.is_empty());
                        if self.created_response_id.as_ref().map(ResponseId::as_str) != terminal_id
                            || terminal_id.is_none()
                        {
                            return Err(LlmError::InvalidRequest {
                                message: "Responses computer call terminal response ID does not match response.created".into(),
                            });
                        }
                        let expected_status = if completed_event {
                            "completed"
                        } else {
                            "incomplete"
                        };
                        if terminal_status != Some(expected_status) {
                            return Err(LlmError::InvalidRequest {
                                message: format!(
                                    "Responses {} event contains computer calls but response.status is {:?}",
                                    root["type"].as_str().unwrap_or("unknown terminal"),
                                    terminal_status,
                                ),
                            });
                        }
                    }
                    let completed_response =
                        completed_event && terminal_status == Some("completed");
                    let computer_calls =
                        self.decode_terminal_computer_calls(items, completed_response)?;
                    for (output_index, item) in items.iter().enumerate() {
                        if let Some(value) = computer_calls.get(&output_index) {
                            self.saw_tool_call = true;
                            out.push(StreamEvent::Native {
                                block: output_index,
                                value: value.clone(),
                            });
                        } else if item.get("type").and_then(Value::as_str) == Some("computer_call")
                        {
                            // An incomplete terminal response remains visible
                            // as raw provider content, never as an executable
                            // typed call.
                            out.push(StreamEvent::ProviderContent {
                                block: output_index,
                                protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
                                value: item.clone(),
                            });
                        } else {
                            self.emit_native(output_index, item, &mut out);
                        }
                    }
                }
                self.file_search.emit(response, &mut out);
                self.search.emit(
                    web_search_decode::with_usage(
                        web_search_decode::responses(response),
                        response.get("usage"),
                    ),
                    &mut out,
                );
                if let Some(u) = response.get("usage") {
                    self.usage_raw = Some(u.clone());
                }
                self.saw_refusal |= response
                    .get("output")
                    .and_then(Value::as_array)
                    .is_some_and(|items| items.iter().any(decode::has_refusal));
                self.stop = Some(decode::stop_reason_with_approval_support(
                    response,
                    self.openai_approval_semantics,
                    self.openai_tool_search_semantics,
                ));
                self.finish_into(&mut out);
            }
            Some("response.failed") => {
                let response = root.get("response").unwrap_or(&Value::Null);
                let classified = decode::classify_error(500, response, None);
                return Err(if self.chatgpt_plan {
                    LlmError::ProviderResponse {
                        status: self.http_status.unwrap_or(200),
                        request_id: self.request_id.clone(),
                        body: response.clone(),
                        classification: classified.kind(),
                        retry_after: None,
                    }
                } else {
                    classified
                });
            }
            Some("error") => {
                let envelope = if root.get("error").is_some() {
                    root
                } else {
                    serde_json::json!({"error": root})
                };
                let status = envelope
                    .get("status")
                    .and_then(Value::as_u64)
                    .and_then(|s| u16::try_from(s).ok())
                    .unwrap_or(500);
                let retry_after = envelope
                    .pointer("/headers/retry-after")
                    .and_then(Value::as_str)
                    .and_then(|s| s.parse::<f64>().ok())
                    .and_then(|s| std::time::Duration::try_from_secs_f64(s).ok());
                let classified = decode::classify_error(status, &envelope, retry_after);
                return Err(if self.chatgpt_plan {
                    LlmError::ProviderResponse {
                        status: self.http_status.unwrap_or(200),
                        request_id: self.request_id.clone(),
                        body: envelope.clone(),
                        classification: classified.kind(),
                        retry_after,
                    }
                } else {
                    classified
                });
            }
            // Unknown types (`response.in_progress`, `…output_text.done`, and
            // whatever is added next) are ignored.
            _ => {}
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        if !self.done {
            return Err(LlmError::StreamInterrupted {
                message: "provider stream ended before a terminal event".to_owned(),
            });
        }
        let mut out = Vec::new();
        self.finish_into(&mut out);
        Ok(out)
    }

    fn inference_report(&self) -> crate::protocol::InferenceReport {
        self.inference.report.clone()
    }
    fn set_response_headers(&mut self, headers: &[(String, String)]) {
        self.inference.headers(headers);
        self.request_id = headers
            .iter()
            .find(|(name, _)| {
                name.eq_ignore_ascii_case("x-request-id")
                    || name.eq_ignore_ascii_case("openai-request-id")
            })
            .map(|(_, value)| value.clone());
    }
    fn set_response_status(&mut self, status: u16) {
        self.http_status = Some(status);
    }
    fn usage_report(&self) -> crate::protocol::UsageReport {
        usage::report(
            self.usage_raw.as_ref(),
            &usage::OPENAI_RESPONSES,
            decode::usage,
            self.done,
        )
    }
}

impl ResponsesStreamDecoder {
    /// Validate the complete call batch before emitting any computer call.
    /// Earlier function-call observations must agree with the terminal items.
    fn decode_terminal_computer_calls(
        &self,
        items: &[Value],
        completed_response: bool,
    ) -> Result<BTreeMap<usize, NativeExtension>, LlmError> {
        let mut calls = BTreeMap::new();
        let mut call_ids = BTreeSet::new();
        for (index, item) in items.iter().enumerate() {
            match item.get("type").and_then(Value::as_str) {
                Some("function_call") => {
                    if completed_response {
                        for key in ["call_id", "name", "arguments"] {
                            if item.get(key).and_then(Value::as_str).is_none() {
                                return Err(LlmError::InvalidRequest {
                                    message: format!("provider function call has no {key} string"),
                                });
                            }
                        }
                        if serde_json::from_str::<Value>(item["arguments"].as_str().unwrap())
                            .is_err()
                        {
                            return Err(LlmError::InvalidRequest {
                                message: "provider returned malformed tool arguments".into(),
                            });
                        }
                    }
                    if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
                        if !call_ids.insert(call_id.to_owned()) {
                            return Err(LlmError::InvalidRequest {
                                message: "Responses output contains duplicate call_id values"
                                    .into(),
                            });
                        }
                    }
                }
                Some("computer_call") => {
                    if !self.openai_tool_search_semantics {
                        return Err(LlmError::UnsupportedCapability {
                            message: "OpenAI computer calls require the official OpenAI Responses profile".into(),
                        });
                    }
                    if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
                        if !call_ids.insert(call_id.to_owned()) {
                            return Err(LlmError::InvalidRequest {
                                message: "Responses output contains duplicate call_id values"
                                    .into(),
                            });
                        }
                    }
                    if completed_response {
                        let call = OpenAiComputerCall::from_response_item(item)?;
                        call.validate_completed_generation()?;
                        calls.insert(index, call.into_native_extension()?);
                    }
                }
                _ => {}
            }
        }
        if completed_response {
            for (index, (id, call_id)) in &self.added_computer_calls {
                let Some(item) = items.get(*index) else {
                    return Err(LlmError::InvalidRequest {
                        message: "added computer call is missing from terminal output".into(),
                    });
                };
                if item.get("type").and_then(Value::as_str) != Some("computer_call")
                    || item.get("id").and_then(Value::as_str) != Some(id.as_str())
                    || item.get("call_id").and_then(Value::as_str) != Some(call_id.as_str())
                {
                    return Err(LlmError::InvalidRequest {
                        message: "added computer call identity changed before terminal response"
                            .into(),
                    });
                }
            }
            for (index, observed) in &self.done_computer_calls {
                let Some(item) = items.get(*index) else {
                    return Err(LlmError::InvalidRequest {
                        message: "completed computer call is missing from terminal output".into(),
                    });
                };
                let terminal = OpenAiComputerCall::from_response_item(item)?;
                if observed != &terminal {
                    return Err(LlmError::InvalidRequest {
                        message: "completed computer call changed before terminal response".into(),
                    });
                }
            }
        }
        if completed_response && !calls.is_empty() {
            self.validate_streamed_function_calls(items)?;
        }
        Ok(calls)
    }

    fn validate_streamed_function_calls(&self, items: &[Value]) -> Result<(), LlmError> {
        let mut observed_indices = BTreeSet::new();
        for call in &self.calls {
            if !observed_indices.insert(call.index) {
                return Err(LlmError::InvalidRequest {
                    message: "Responses stream repeated a function call output index".into(),
                });
            }
            let Some(item) = items.get(call.index) else {
                return Err(LlmError::InvalidRequest {
                    message: "Responses stream function call is missing from terminal output"
                        .into(),
                });
            };
            if item["type"] != "function_call"
                || item["call_id"].as_str() != Some(call.id.as_str())
                || item["name"].as_str() != Some(call.name.as_str())
            {
                return Err(LlmError::InvalidRequest {
                    message: "Responses stream function call identity changed before completion"
                        .into(),
                });
            }
            let streamed_arguments = if call.arguments.is_empty() {
                Value::Object(Default::default())
            } else {
                serde_json::from_str(&call.arguments).map_err(|_| LlmError::InvalidRequest {
                    message: "Responses stream function call arguments are malformed".into(),
                })?
            };
            let terminal_arguments: Value =
                serde_json::from_str(item["arguments"].as_str().unwrap()).map_err(|_| {
                    LlmError::InvalidRequest {
                        message: "provider returned malformed tool arguments".into(),
                    }
                })?;
            if streamed_arguments != terminal_arguments {
                return Err(LlmError::InvalidRequest {
                    message: "Responses stream function call arguments changed before completion"
                        .into(),
                });
            }
        }
        if items.iter().enumerate().any(|(index, item)| {
            item["type"] == "function_call" && !observed_indices.contains(&index)
        }) {
            return Err(LlmError::InvalidRequest {
                message: "Responses terminal function call has no preceding stream identity".into(),
            });
        }
        Ok(())
    }

    fn emit_native(&mut self, block: usize, item: &Value, out: &mut Vec<StreamEvent>) {
        self.requires_action |= self.openai_approval_semantics
            && item["type"] == "mcp_approval_request"
            || self.openai_tool_search_semantics && decode::is_client_tool_search_call(item);
        if item.get("type").and_then(Value::as_str) == Some("computer_call") {
            return;
        }
        if item["type"]
            .as_str()
            .is_some_and(|kind| !matches!(kind, "message" | "function_call"))
            && self.native_items.insert(block)
        {
            out.push(StreamEvent::ProviderContent {
                block,
                protocol: crate::protocol::ProtocolFamily::OpenAiResponses,
                value: item.clone(),
            });
        }
    }

    fn finish_into(&mut self, out: &mut Vec<StreamEvent>) {
        if self.done {
            return;
        }
        self.done = true;
        out.push(StreamEvent::End {
            stop_reason: if self
                .stop
                .as_ref()
                .is_some_and(|reason| *reason != StopReason::EndTurn)
            {
                self.stop.clone().unwrap()
            } else if self.requires_action {
                StopReason::Other("requires_action".into())
            } else if self.saw_tool_call {
                StopReason::ToolUse
            } else if self.saw_refusal {
                StopReason::Refusal
            } else {
                self.stop.clone().unwrap_or(StopReason::EndTurn)
            },
            usage: self.usage_report(),
            inference: self.inference.report.clone(),
        });
    }
}

impl ResponsesStreamDecoder {
    pub(crate) fn configured(context: &crate::codecs::CodecContext) -> Self {
        Self {
            inference: crate::codecs::inference::StreamInference::new(context),
            openai_approval_semantics:
                crate::providers::xai::responses_policy::uses_openai_approval_semantics(
                    context.profile(),
                ),
            openai_tool_search_semantics:
                crate::providers::openai::responses_policy::is_official_openai_responses_profile(
                    context.profile(),
                ),
            qwen_code_interpreter: crate::providers::qwen::hosted::supports_code_interpreter(
                context.profile(),
                context.request_model(),
            ),
            chatgpt_plan: context.profile().auth == crate::protocol::AuthStrategy::ChatGptPlan,
            ..Self::default()
        }
    }
}
