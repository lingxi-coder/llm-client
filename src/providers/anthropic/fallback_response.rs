//! Messages fallback control observations. Lane admission and session ownership
//! belong to the caller. Applying an admitted hop changes the response model;
//! the caller still owns its query route and session selection.
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FallbackHop {
    pub from_model: String,
    pub model: String,
    pub reason: String,
    pub category: Option<String>,
}

impl FallbackHop {
    /// Native lNe materialization omits the trigger and all extra wire fields.
    pub fn materialized_block(&self) -> Value {
        serde_json::json!({"type":"fallback","from":{"model":self.from_model},"to":{"model":self.model}})
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FallbackStart {
    pub index: Number,
    #[serde(flatten)]
    pub hop: FallbackHop,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FallbackControl {
    Start {
        start: FallbackStart,
    },
    Boundary {
        stop_reason: Option<String>,
        iterations: UsageIterations,
        /// Distinguishes an omitted partial usage field from an explicit
        /// empty array that replaces the prior native iteration snapshot.
        #[serde(default)]
        iterations_present: bool,
    },
}
impl crate::protocol::NativeType for FallbackControl {
    const FORMAT: &'static str = "anthropic.fallback_control.v1";
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerFallbackEvent {
    pub from_model: String,
    pub to_model: String,
    pub reason: String,
    pub api_refusal_category: Option<String>,
    pub mid_stream: bool,
    pub request_id: Option<String>,
    pub discarded_blocks: Vec<usize>,
    pub retained_blocks: Vec<usize>,
    pub retained_text: String,
    pub final_stop_reason: Option<String>,
}

fn text(block: &crate::protocol::ContentBlock) -> Option<&str> {
    match block {
        crate::protocol::ContentBlock::Text { text, .. }
        | crate::protocol::ContentBlock::TextJsUtf16 { text, .. } => Some(text),
        crate::protocol::ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
            value,
        } if value["type"] == "text" => value["text"].as_str(),
        _ => None,
    }
}

/// Native FHt retention considers completed messages only, in block order.
pub fn stream_event(
    hop: &FallbackHop,
    completed: &[(usize, crate::protocol::ContentBlock)],
    request_id: Option<String>,
) -> ServerFallbackEvent {
    let mut event = ServerFallbackEvent {
        from_model: hop.from_model.clone(),
        to_model: hop.model.clone(),
        reason: hop.reason.clone(),
        api_refusal_category: hop.category.clone(),
        mid_stream: !completed.is_empty(),
        request_id,
        discarded_blocks: Vec::new(),
        retained_blocks: Vec::new(),
        retained_text: String::new(),
        final_stop_reason: None,
    };
    if matches!(hop.reason.as_str(), "refusal" | "sticky") {
        let mut retained = Vec::new();
        for (index, block) in completed {
            if let Some(text) = text(block) {
                retained.push((*index, text));
            } else {
                event.discarded_blocks.push(*index);
            }
        }
        retained.sort_by_key(|(index, _)| *index);
        for (index, text) in retained {
            event.retained_blocks.push(index);
            event.retained_text.push_str(text);
        }
    }
    event
}

pub fn terminal_event(
    request_model: &str,
    hop: Option<&FallbackHop>,
    iterations: Option<&UsageIterations>,
    request_id: Option<String>,
    stop_reason: Option<String>,
) -> Option<ServerFallbackEvent> {
    let to_model = hop
        .map(|hop| &hop.model)
        .or_else(|| iterations?.served_fallback_model.as_ref())?;
    Some(ServerFallbackEvent {
        from_model: hop.map_or(request_model, |hop| &hop.from_model).into(),
        to_model: to_model.clone(),
        reason: hop.map_or("sticky", |hop| hop.reason.as_str()).into(),
        api_refusal_category: hop.and_then(|hop| hop.category.clone()),
        mid_stream: false,
        request_id,
        discarded_blocks: Vec::new(),
        retained_blocks: Vec::new(),
        retained_text: String::new(),
        final_stop_reason: stop_reason,
    })
}

/// Native $cn entries retain arbitrary string types and empty model strings.
/// Counters are finite, nonnegative numbers, including fractional counts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageIteration {
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub input_tokens: f64,
    pub output_tokens: f64,
    pub cache_read_input_tokens: f64,
    pub cache_creation_input_tokens: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageIterations {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub served_fallback_model: Option<String>,
    pub entries: Vec<UsageIteration>,
    /// Native Mwe inputs observed beside `usage.iterations`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
    /// Native `$cn` passes aggregate server tool usage only to its final
    /// zero-token Mwe call. Fractional counters are retained as reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_requests: Option<f64>,
}

/// Observations survive partial streams without granting a server fallback lane.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FallbackResponse {
    pub hops: Vec<FallbackHop>,
    pub malformed_blocks: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iterations: Option<UsageIterations>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content_positions: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_control_position: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_stop_reason: Option<String>,
}

pub fn project_nonstream(
    response: &mut crate::protocol::ChatResponse,
    request_model: &str,
    request_id: Option<String>,
) -> Option<ServerFallbackEvent> {
    let facts = response.anthropic_fallback()?.clone();
    if let Some(last) = facts.last_control_position {
        let refused = response.stop_reason == crate::protocol::StopReason::Refusal;
        let mut index = 0;
        response.message.content.retain(|block| {
            let position = facts
                .content_positions
                .get(index)
                .copied()
                .unwrap_or(usize::MAX);
            index += 1;
            !refused && (position >= last || text(block).is_some())
        });
        // Sbn retains materialized valid controls before presentation removes
        // them. Only an empty native result with no valid control uses iu.
        if response.message.content.is_empty() && facts.hops.is_empty() {
            response
                .message
                .content
                .push(crate::protocol::ContentBlock::Text {
                    text: "(no content)".into(),
                    thought_signature: None,
                    citations: Some(Some(serde_json::json!([]))),
                });
        }
    }
    let event = terminal_event(
        request_model,
        facts.hops.last(),
        facts.iterations.as_ref(),
        request_id,
        facts.final_stop_reason,
    )?;
    response.model.clone_from(&event.to_model);
    Some(event)
}

pub fn is_fallback_block(value: &Value) -> bool {
    value.is_object() && value["type"] == "fallback"
}

fn model(value: &Value) -> Option<String> {
    value
        .as_object()?
        .get("model")?
        .as_str()
        .filter(|model| !model.is_empty())
        .map(str::to_owned)
}

/// Native BHt counts UTF-16 code units, rather than bytes or Unicode scalars.
pub fn category(trigger: &Value) -> Option<String> {
    if !trigger.is_object() || trigger["type"] != "refusal" {
        return None;
    }
    trigger["category"]
        .as_str()
        .filter(|category| matches!(category.encode_utf16().count(), 1..=64))
        .map(str::to_owned)
}

pub fn complete_block(value: &Value) -> Option<FallbackHop> {
    is_fallback_block(value).then_some(())?;
    Some(FallbackHop {
        from_model: model(&value["from"])?,
        model: model(&value["to"])?,
        reason: "refusal".into(),
        category: category(&value["trigger"]),
    })
}

/// Native dNe/HHt accept any JSON number as the control index. Do not cast it
/// into an ordinary content index: negative/fractional indices remain controls.
pub fn control_index(frame: &Value) -> Option<&Number> {
    if frame["type"] != "content_block_start" || !is_fallback_block(&frame["content_block"]) {
        return None;
    }
    frame["index"].as_number()
}

pub fn stream_start(frame: &Value) -> Option<FallbackStart> {
    Some(FallbackStart {
        index: control_index(frame)?.clone(),
        hop: complete_block(&frame["content_block"])?,
    })
}

pub fn malformed_start(frame: &Value) -> bool {
    control_index(frame).is_some() && stream_start(frame).is_none()
}

fn counter(value: &Value) -> f64 {
    value
        .as_f64()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.0)
}

pub fn usage_iterations(usage: &Value) -> UsageIterations {
    let mut result = UsageIterations {
        speed: usage
            .get("speed")
            .and_then(Value::as_str)
            .map(str::to_owned),
        inference_geo: usage
            .get("inference_geo")
            .and_then(Value::as_str)
            .map(str::to_owned),
        web_search_requests: usage
            .pointer("/server_tool_use/web_search_requests")
            .and_then(Value::as_f64)
            .filter(|count| count.is_finite() && *count >= 0.0),
        ..UsageIterations::default()
    };
    let Some(iterations) = usage.get("iterations").and_then(Value::as_array) else {
        return result;
    };
    for iteration in iterations {
        let Some(kind) = iteration.get("type").and_then(Value::as_str) else {
            continue;
        };
        let model = iteration
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned);
        if kind == "fallback_message" && model.is_some() {
            result.served_fallback_model.clone_from(&model);
        }
        result.entries.push(UsageIteration {
            r#type: kind.into(),
            model,
            input_tokens: counter(&iteration["input_tokens"]),
            output_tokens: counter(&iteration["output_tokens"]),
            cache_read_input_tokens: counter(&iteration["cache_read_input_tokens"]),
            cache_creation_input_tokens: counter(&iteration["cache_creation_input_tokens"]),
        });
    }
    result
}

/// Extract fallback control and usage facts without decoding assistant content.
/// This keeps server accounting observations available when a later content
/// block is malformed and the ordinary response projection fails.
pub fn from_nonstream_body(body: &Value) -> Option<FallbackResponse> {
    let mut fallback = FallbackResponse {
        final_stop_reason: body
            .get("stop_reason")
            .and_then(Value::as_str)
            .map(str::to_owned),
        ..Default::default()
    };
    if let Some(content) = body.get("content").and_then(Value::as_array) {
        for (position, block) in content.iter().enumerate() {
            if is_fallback_block(block) {
                fallback.last_control_position = Some(position);
            } else {
                fallback.content_positions.push(position);
            }
            fallback.observe_block(block);
        }
    }
    fallback.observe_usage(&body["usage"]);
    fallback.observed().then_some(fallback)
}

impl FallbackResponse {
    pub(crate) fn observe_block(&mut self, block: &Value) {
        if let Some(hop) = complete_block(block) {
            self.hops.push(hop);
        } else if is_fallback_block(block) {
            self.malformed_blocks += 1;
        }
    }
    pub(crate) fn observe_usage(&mut self, usage: &Value) {
        let iterations_present = usage.get("iterations").is_some_and(Value::is_array);
        self.merge_usage_iterations(usage_iterations(usage), iterations_present);
    }
    pub(crate) fn observe_boundary(
        &mut self,
        stop_reason: Option<&str>,
        iterations: &UsageIterations,
        iterations_present: bool,
    ) {
        self.observe_stop_reason(stop_reason);
        self.merge_usage_iterations(iterations.clone(), iterations_present);
    }
    pub(crate) fn observe_stop_reason(&mut self, stop_reason: Option<&str>) {
        if let Some(stop_reason) = stop_reason {
            self.final_stop_reason = Some(stop_reason.to_owned());
        }
    }
    fn merge_usage_iterations(&mut self, incoming: UsageIterations, iterations_present: bool) {
        let has_context = incoming.speed.is_some()
            || incoming.inference_geo.is_some()
            || incoming.web_search_requests.is_some();
        if !iterations_present && !has_context && self.iterations.is_none() {
            return;
        }
        let current = self.iterations.get_or_insert_with(UsageIterations::default);
        if iterations_present {
            current.entries = incoming.entries;
            current.served_fallback_model = incoming.served_fallback_model;
        }
        if incoming.speed.is_some() {
            current.speed = incoming.speed;
        }
        if incoming.inference_geo.is_some() {
            current.inference_geo = incoming.inference_geo;
        }
        if incoming.web_search_requests.is_some() {
            current.web_search_requests = incoming.web_search_requests;
        }
    }
    pub(crate) fn observed(&self) -> bool {
        !self.hops.is_empty() || self.malformed_blocks != 0 || self.iterations.is_some()
    }
}
