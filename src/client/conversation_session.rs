//! Opt-in, full-history conversation continuation. Recovery is bounded to one
//! rejected continuation; every physical dispatch still enters host admission.
use super::{
    ClientSnapshot, ModelStream, PreparedCall, ReceivedCall, RequestDraft, RequestOptions,
};
use crate::protocol::{
    ChatRequest, ChatResponse, JsonPayload, LlmError, ProtocolFamily, ProviderProfile, StreamEvent,
};
use crate::RequestMode;
use futures::StreamExt;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Stateful behavior belongs to an exact endpoint, not its compatible wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuationCapability {
    Unsupported,
    Responses,
    AnthropicThread,
    GeminiInteraction,
}
impl ContinuationCapability {
    pub fn for_profile(profile: &ProviderProfile) -> Self {
        match profile.protocol {
            ProtocolFamily::OpenAiResponses
                if profile
                    .extra
                    .get("supports_previous_response_id")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true) =>
            {
                Self::Responses
            }
            ProtocolFamily::GeminiInteractions => Self::GeminiInteraction,
            ProtocolFamily::AnthropicMessages
                if profile
                    .extra
                    .get("supports_message_threads")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
                    && profile.provider_id.as_str() == "anthropic"
                    && profile.base_url.trim_end_matches('/') == "https://api.anthropic.com" =>
            {
                Self::AnthropicThread
            }
            _ => Self::Unsupported,
        }
    }
    fn input(self) -> &'static str {
        if self == Self::AnthropicThread {
            "messages"
        } else {
            "input"
        }
    }
    fn previous(self) -> &'static str {
        if self == Self::GeminiInteraction {
            "previous_interaction_id"
        } else {
            "previous_response_id"
        }
    }
    fn items(self, body: &JsonPayload) -> Option<Vec<JsonPayload>> {
        body.payload_at(if self == Self::GeminiInteraction {
            "/input/steps"
        } else if self == Self::AnthropicThread {
            "/messages"
        } else {
            "/input"
        })?
        .array_items()
    }
    fn replace_input(
        self,
        body: &JsonPayload,
        items: &[JsonPayload],
    ) -> Result<JsonPayload, LlmError> {
        let items = array(items)?;
        let input = if self == Self::GeminiInteraction {
            replace(
                &body.payload_at("/input").ok_or_else(|| invalid("input"))?,
                "steps",
                Some(items),
            )?
        } else {
            items
        };
        replace(body, self.input(), Some(input))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ConversationRequestMode {
    #[default]
    Full,
    Create,
    Continue,
    Recreate,
}

/// Sizes and mode only; snapshots never expose credentials or conversation text.
#[derive(Debug, Clone, Default)]
pub struct ConversationRequestSnapshot {
    pub mode: ConversationRequestMode,
    pub logical_bytes: usize,
    pub wire_bytes: usize,
    pub new_items: usize,
}

struct Completed {
    identity: [u8; 32],
    logical: JsonPayload,
    output: Vec<JsonPayload>,
    id: String,
    at: Instant,
}
#[derive(Default)]
struct State {
    epoch: u64,
    completed: Option<Completed>,
}

/// Pass a complete, replayable history on every call. The session checks its
/// prefix and shrinks only the final wire request. It owns no durable history.
/// A separate session must be used for each conversation or fork.
#[derive(Default)]
pub struct ConversationSession {
    transport: Option<Arc<dyn crate::Transport>>,
    websocket: super::ResponsesSession,
    websocket_prepared: bool,
    state: Arc<Mutex<State>>,
    pending: Option<Observation>,
    binding: Option<Arc<()>>,
    snapshot: ConversationRequestSnapshot,
    recovery: bool,
    last_reference: Option<String>,
    last_capability: Option<ContinuationCapability>,
    disabled_thread: Option<[u8; 32]>,
    last_identity: Option<[u8; 32]>,
}
impl ConversationSession {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn last_request(&self) -> &ConversationRequestSnapshot {
        &self.snapshot
    }
    pub async fn close(&mut self) -> Result<(), LlmError> {
        let result = self.websocket.close().await;
        self.reset();
        result
    }
    pub fn reset(&mut self) {
        self.websocket.invalidate();
        self.websocket_prepared = false;
        self.last_reference = None;
        let mut state = self.state.lock().expect("conversation state");
        state.epoch = state.epoch.wrapping_add(1);
        state.completed = None;
        self.pending = None;
        self.binding = None;
    }

    /// Prepare before sealing/signing and host admission. Explicit caller
    /// continuations are left intact and never seed an implicit chain.
    pub async fn prepare(&mut self, draft: &mut RequestDraft) -> Result<(), LlmError> {
        if draft.request_mode() == RequestMode::CountTokens {
            return Err(LlmError::UnsupportedCapability {
                message: "conversation continuation is not a token-count operation".into(),
            });
        }
        self.binding = None;
        self.pending = None;
        self.websocket_prepared = false;
        self.last_reference = None;
        draft.settle_body()?;
        let transport = draft.transport();
        if self
            .transport
            .as_ref()
            .is_some_and(|old| !Arc::ptr_eq(old, &transport))
        {
            self.reset();
        }
        self.transport = Some(transport);
        let logical = draft.semantic_body_source()?;
        let capability = ContinuationCapability::for_profile(draft.profile());
        let authenticated = draft.websocket_handshake().await?;
        let mut headers: Vec<_> = authenticated
            .headers
            .iter()
            .filter(|(name, _)| {
                !name.eq_ignore_ascii_case("x-request-id")
                    && !name.eq_ignore_ascii_case("traceparent")
            })
            .map(|(name, value)| (name.to_ascii_lowercase(), value))
            .collect();
        headers.sort_unstable();
        let identity: [u8; 32] = Sha256::digest(
            serde_json::to_vec(&serde_json::json!([
                draft.account_scope(),
                draft.profile(),
                authenticated.url,
                headers
            ]))
            .map_err(invalid)?,
        )
        .into();
        let input = capability.items(&logical);
        let mut wire = logical.clone();
        let mut mode = ConversationRequestMode::Full;
        let explicit = logical.payload_at("/previous_response_id").is_some()
            || logical.payload_at("/previous_interaction_id").is_some()
            || logical.payload_at("/conversation").is_some()
            || logical.payload_at("/thread").is_some();
        let eligible = capability != ContinuationCapability::Unsupported
            && !(capability == ContinuationCapability::AnthropicThread
                && self.disabled_thread == Some(identity))
            && draft.account_scope().is_some_and(|s| !s.is_empty())
            && logical.raw_source_at("/store").as_deref() != Some("false")
            && (draft
                .profile()
                .extra
                .get("response_store_default")
                .and_then(serde_json::Value::as_bool)
                != Some(false)
                || logical.raw_source_at("/store").as_deref() == Some("true"))
            && !explicit
            && input.is_some()
            && (capability != ContinuationCapability::AnthropicThread
                || thread_supported(&logical));
        let mut new_items = input.as_ref().map_or(0, Vec::len);
        let epoch = {
            let mut state = self.state.lock().expect("conversation state");
            if eligible {
                let previous = state.completed.take();
                if let Some(previous) = previous.filter(|previous| {
                    previous.identity == identity
                        && (capability != ContinuationCapability::AnthropicThread
                            || previous.at.elapsed() < Duration::from_secs(300))
                }) {
                    if compatible(&previous.logical, &logical, capability)? {
                        let mut prefix = capability.items(&previous.logical).unwrap_or_default();
                        prefix.extend(previous.output);
                        let current = input.as_ref().expect("eligible input");
                        if prefix.len() <= current.len()
                            && prefix_matches(&prefix, current, capability)?
                        {
                            new_items = current.len() - prefix.len();
                            wire = capability.replace_input(&wire, &current[prefix.len()..])?;
                            self.last_reference = Some(previous.id.clone());
                            if capability == ContinuationCapability::AnthropicThread {
                                wire = replace(&wire, "system", None)?;
                                wire = replace(&wire, "tools", None)?;
                                wire = replace(
                                    &wire,
                                    "thread",
                                    Some(generated(
                                        serde_json::json!({"type":"continue","previous_message_id":previous.id}),
                                    )?),
                                )?;
                            } else {
                                wire = replace(
                                    &wire,
                                    capability.previous(),
                                    Some(JsonPayload::string(&previous.id)),
                                )?;
                            }
                            mode = ConversationRequestMode::Continue;
                        }
                    }
                }
                if mode != ConversationRequestMode::Continue
                    && capability == ContinuationCapability::AnthropicThread
                {
                    wire = replace(
                        &wire,
                        "thread",
                        Some(generated(serde_json::json!({"type":"create"}))?),
                    )?;
                    mode = if self.recovery {
                        ConversationRequestMode::Recreate
                    } else {
                        ConversationRequestMode::Create
                    };
                } else if self.recovery && mode != ConversationRequestMode::Continue {
                    mode = ConversationRequestMode::Recreate;
                }
                if capability == ContinuationCapability::AnthropicThread {
                    add_beta(draft, "message-threads-2026-08-12");
                }
            } else {
                state.completed = None;
            }
            if self.recovery && mode == ConversationRequestMode::Full {
                mode = ConversationRequestMode::Recreate;
            }
            state.epoch = state.epoch.wrapping_add(1);
            state.epoch
        };
        let body = wire.as_json_source().as_bytes().to_vec().into();
        draft.set_session_wire_body(body);
        let binding = Arc::new(());
        draft.bind_session(binding.clone());
        self.binding = Some(binding);
        self.snapshot = ConversationRequestSnapshot {
            mode,
            logical_bytes: logical.as_json_source().len(),
            wire_bytes: wire.as_json_source().len(),
            new_items,
        };
        if eligible {
            self.pending = Some(Observation {
                state: self.state.clone(),
                epoch,
                identity,
                logical,
                capability,
                wire: Arc::new(Mutex::new(WireObservation::default())),
            });
        }
        self.recovery = false;
        self.last_identity = Some(identity);
        self.last_capability = Some(capability);
        if draft.profile().supports_websockets
            && draft.profile().protocol == ProtocolFamily::OpenAiResponses
            && logical_streaming(&draft.semantic_body_source()?)
            && websocket_representable(&draft.semantic_body_source()?)
        {
            self.websocket.prepare(draft, false, true).await?;
            self.binding = self.websocket.preparation_binding();
            self.websocket_prepared = true;
            self.snapshot.wire_bytes = draft.request().body.len();
            self.snapshot.new_items =
                JsonPayload::parse(std::str::from_utf8(&draft.request().body).map_err(invalid)?)
                    .ok()
                    .and_then(|body| capability.items(&body))
                    .map_or(0, |items| items.len());
            if self
                .websocket
                .last_request_snapshot()
                .wire_used_previous_response_id
            {
                self.snapshot.mode = ConversationRequestMode::Continue;
                self.last_reference = self
                    .websocket
                    .last_request_snapshot()
                    .wire_request_body
                    .and_then(|v| {
                        v.get("previous_response_id")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_owned)
                    });
            }
        }
        Ok(())
    }

    /// Dispatch exactly once. A rejected admission sends nothing. Recovery
    /// requires a newly prepared and signed request and a fresh admission.
    pub async fn dispatch(
        &mut self,
        mut call: PreparedCall,
        on_dispatch: impl FnOnce() -> Result<(), LlmError> + Send,
    ) -> Result<ReceivedCall, LlmError> {
        if !self
            .binding
            .as_ref()
            .is_some_and(|binding| call.is_bound_to_session(binding))
        {
            return Err(LlmError::InvalidRequest {
                message: "call does not match conversation preparation".into(),
            });
        }
        self.binding = None;
        call.conversation_observation = self.pending.take();
        if self.websocket_prepared {
            self.websocket_prepared = false;
            self.websocket.dispatch(call, on_dispatch).await
        } else {
            call.dispatch_once_with(on_dispatch).await
        }
    }

    /// Only exact pre-inference rejection codes authorize a replay. Generic
    /// 404s, transport failures, truncation and timeouts never do.
    pub fn recover(&mut self, response: &crate::HttpResponse) -> bool {
        let Ok(source) = std::str::from_utf8(&response.body) else {
            return false;
        };
        let Ok(body) = JsonPayload::parse(source) else {
            return false;
        };
        let unsupported_thread = self.last_capability
            == Some(ContinuationCapability::AnthropicThread)
            && matches!(
                self.snapshot.mode,
                ConversationRequestMode::Continue
                    | ConversationRequestMode::Create
                    | ConversationRequestMode::Recreate
            )
            && response.status == 400
            && body.string_at("/error/details/error_code").as_deref()
                == Some("thread_unsupported_request");
        let named_missing = matches!(response.status, 400 | 404)
            && self.last_capability != Some(ContinuationCapability::AnthropicThread)
            && self.last_reference.as_deref().is_some_and(|id| {
                body.string_at("/error/message").is_some_and(|message| {
                    let lower = message.to_ascii_lowercase();
                    message.contains(id)
                        && lower.contains("not found")
                        && (lower.contains("response") || lower.contains("interaction"))
                })
            });
        if !unsupported_thread
            && !(self.snapshot.mode == ConversationRequestMode::Continue
                && (replayable_error(response.status, &body) || named_missing))
        {
            return false;
        }
        if unsupported_thread {
            self.disabled_thread = self.last_identity;
        }
        self.reset();
        self.recovery = true;
        true
    }

    /// Convenience completion with at most one continuation recovery. The
    /// callback observes accounting facts for every response, including a
    /// rejected first attempt, before decoding or retrying.
    pub async fn complete_in(
        &mut self,
        client: &ClientSnapshot,
        profile: &str,
        request: &ChatRequest,
        options: &RequestOptions,
        mut on_dispatch: impl FnMut() -> Result<(), LlmError> + Send,
        mut on_response: impl FnMut(&super::CollectedResponse) + Send,
    ) -> Result<ChatResponse, LlmError> {
        let mut options = options.clone();
        options
            .total_timeout
            .get_or_insert(super::options::DEFAULT_REQUEST_TIMEOUT);
        let started = Instant::now();
        for attempt in 0..2 {
            let options = remaining_options(&options, started)?;
            let mut draft = client
                .prepare_draft_on(profile, request, &options, RequestMode::Complete)
                .await?;
            self.prepare(&mut draft).await?;
            let call = draft.seal().await?;
            let collected = self
                .dispatch(call, &mut on_dispatch)
                .await?
                .collect()
                .await?;
            on_response(&collected);
            let recover = attempt == 0 && self.recover(collected.response());
            let result = collected.decode();
            collected.finish().await;
            if !recover {
                return result;
            }
        }
        unreachable!("bounded recovery")
    }

    /// Streaming recovery handles HTTP or first-event continuation rejections
    /// before generation opens. Once a stream is returned, no automatic replay is
    /// possible; use a new full-history turn after interrupted output.
    pub async fn stream_in(
        &mut self,
        client: &ClientSnapshot,
        profile: &str,
        request: &ChatRequest,
        options: &RequestOptions,
        mut on_dispatch: impl FnMut() -> Result<(), LlmError> + Send,
        mut on_response: impl FnMut(&super::CollectedResponse) + Send,
    ) -> Result<ModelStream, LlmError> {
        let started = Instant::now();
        for attempt in 0..2 {
            let options = remaining_options(options, started)?;
            let mut draft = client
                .prepare_draft_on(profile, request, &options, RequestMode::Stream)
                .await?;
            self.prepare(&mut draft).await?;
            let mut received = self.dispatch(draft.seal().await?, &mut on_dispatch).await?;
            if self.snapshot.mode == ConversationRequestMode::Continue {
                if let Some(rejection) = peek_rejection(&mut received).await? {
                    let recover = attempt == 0 && self.recover(&rejection);
                    let collected = received.collect().await?;
                    on_response(&collected);
                    collected.finish().await;
                    if recover {
                        continue;
                    }
                    return Err(LlmError::InvalidRequest {
                        message: "continuation rejected before generation; recovery exhausted"
                            .into(),
                    });
                }
            }
            match received.into_stream() {
                Ok(stream) => return Ok(stream),
                Err(received) => {
                    let collected = received.collect().await?;
                    on_response(&collected);
                    let recover = attempt == 0 && self.recover(collected.response());
                    let result = collected.decode();
                    collected.finish().await;
                    if !recover {
                        return Err(result.err().unwrap_or_else(|| LlmError::ProviderInternal {
                            message: "expected streaming response".into(),
                        }));
                    }
                }
            }
        }
        unreachable!("bounded recovery")
    }
}

fn remaining_options(
    options: &RequestOptions,
    started: Instant,
) -> Result<RequestOptions, LlmError> {
    let mut next = options.clone();
    if let Some(timeout) = options.total_timeout {
        next.total_timeout = Some(
            timeout
                .checked_sub(started.elapsed())
                .filter(|v| !v.is_zero())
                .ok_or_else(|| LlmError::TransportTimeout {
                    message: "conversation recovery deadline exceeded".into(),
                })?,
        );
    }
    Ok(next)
}
fn replayable_error(status: u16, body: &JsonPayload) -> bool {
    let code = body
        .string_at("/error/details/error_code")
        .or_else(|| body.string_at("/error/code"));
    matches!(
        (status, code.as_deref()),
        (
            404,
            Some("thread_not_found" | "previous_response_not_found" | "interaction_not_found")
        ) | (
            400,
            Some(
                "thread_already_continued"
                    | "thread_continue_in_flight"
                    | "thread_fingerprint_mismatch"
                    | "previous_response_not_found"
            )
        )
    )
}
fn invalid(error: impl std::fmt::Display) -> LlmError {
    LlmError::InvalidRequest {
        message: format!("conversation JSON: {error}"),
    }
}
fn generated(value: serde_json::Value) -> Result<JsonPayload, LlmError> {
    JsonPayload::from_generated_value(&value).map_err(invalid)
}
fn array(items: &[JsonPayload]) -> Result<JsonPayload, LlmError> {
    JsonPayload::parse(&serde_json::to_string(items).map_err(invalid)?).map_err(invalid)
}
fn replace(
    source: &JsonPayload,
    key: &str,
    value: Option<JsonPayload>,
) -> Result<JsonPayload, LlmError> {
    let scanner = source.scanner()?;
    let units: Vec<_> = key.encode_utf16().collect();
    let mut fields = Vec::new();
    for entry in scanner
        .root_node()
        .object_entries()
        .ok_or_else(|| invalid("expected object"))?
    {
        if entry.key_units != units {
            fields.push((
                entry.key_units.to_vec(),
                scanner
                    .raw_source_for_node(entry.value)
                    .ok_or_else(|| invalid("missing field"))?
                    .to_owned(),
            ));
        }
    }
    if let Some(value) = value {
        fields.push((units, value.as_json_source().to_owned()));
    }
    JsonPayload::object_from_fields(&fields).map_err(invalid)
}
fn compatible(
    previous: &JsonPayload,
    current: &JsonPayload,
    capability: ContinuationCapability,
) -> Result<bool, LlmError> {
    let mut previous = replace(previous, capability.input(), None)?;
    let mut current = replace(current, capability.input(), None)?;
    previous = replace(&previous, "stream", None)?;
    current = replace(&current, "stream", None)?;
    // These fields are explicitly free to vary in an Anthropic thread. Other
    // providers conservatively rebuild when any non-input parameter changes.
    if capability == ContinuationCapability::AnthropicThread {
        for key in ["max_tokens", "stop_sequences", "metadata", "service_tier"] {
            previous = replace(&previous, key, None)?;
            current = replace(&current, key, None)?;
        }
        previous = thread_fingerprint(&previous)?;
        current = thread_fingerprint(&current)?;
    }
    previous.semantically_eq(&current)
}
fn prefix_matches(
    prefix: &[JsonPayload],
    current: &[JsonPayload],
    capability: ContinuationCapability,
) -> Result<bool, LlmError> {
    for (a, b) in prefix.iter().zip(current) {
        let a = comparable(a, capability)?;
        let b = comparable(b, capability)?;
        if !a.semantically_eq(&b)? {
            return Ok(false);
        }
    }
    Ok(true)
}
fn comparable(
    item: &JsonPayload,
    capability: ContinuationCapability,
) -> Result<JsonPayload, LlmError> {
    let mut item = item.clone();
    if capability == ContinuationCapability::AnthropicThread {
        if let Some(blocks) = item.payload_at("/content").and_then(|v| v.array_items()) {
            let blocks = blocks
                .iter()
                .map(|block| replace(block, "cache_control", None))
                .collect::<Result<Vec<_>, _>>()?;
            item = replace(&item, "content", Some(array(&blocks)?))?;
        }
    }
    if capability == ContinuationCapability::Responses {
        let kind = item.string_at("/type");
        if matches!(kind.as_deref(), Some("message" | "function_call"))
            || kind.is_none() && item.string_at("/role").as_deref() == Some("assistant")
        {
            for key in ["id", "status"] {
                item = replace(&item, key, None)?;
            }
            if kind.as_deref() != Some("function_call") {
                item = replace(&item, "type", Some(JsonPayload::string("message")))?;
                if let Some(parts) = item.payload_at("/content").and_then(|v| v.array_items()) {
                    let parts = parts
                        .into_iter()
                        .map(|mut part| {
                            if part.string_at("/type").as_deref() == Some("output_text") {
                                for key in ["annotations", "logprobs"] {
                                    part = replace(&part, key, None)?;
                                }
                            }
                            Ok(part)
                        })
                        .collect::<Result<Vec<_>, LlmError>>()?;
                    item = replace(&item, "content", Some(array(&parts)?))?;
                } else if let Some(part) =
                    item.payload_at("/content").filter(|v| v.is_root_object())
                {
                    let part = replace(&replace(&part, "annotations", None)?, "logprobs", None)?;
                    item = replace(&item, "content", Some(array(&[part])?))?;
                }
            }
        }
    }
    Ok(item)
}
fn thread_supported(body: &JsonPayload) -> bool {
    body.payload_at("/compaction").is_none() && body.payload_at("/context_management").is_none()
}
fn thread_fingerprint(body: &JsonPayload) -> Result<JsonPayload, LlmError> {
    let mut result = replace(
        body,
        "fallbacks",
        Some(generated(serde_json::json!(body
            .payload_at("/fallbacks")
            .is_some()))?),
    )?;
    result = replace(&result, "container", body.payload_at("/container/skills"))?;
    if let Some(servers) = body
        .payload_at("/mcp_servers")
        .and_then(|v| v.array_items())
    {
        let servers = servers
            .iter()
            .map(|server| replace(server, "authorization_token", None))
            .collect::<Result<Vec<_>, _>>()?;
        result = replace(&result, "mcp_servers", Some(array(&servers)?))?;
    }
    Ok(result)
}

// Inspect only the first non-ping SSE event. A rejected continuation can be
// replayed before any generation event has escaped. All consumed bytes are
// restored, preserving status, usage and the normal decoder's observations.
async fn peek_rejection(
    received: &mut ReceivedCall,
) -> Result<Option<crate::HttpResponse>, LlmError> {
    if !(200..300).contains(&received.status()) {
        return Ok(None);
    }
    let mut buffered = Vec::new();
    let mut splitter = crate::SseFrameSplitter::new();
    let mut rejection = None;
    'read: while buffered.len() < 64 * 1024 {
        let Some(chunk) = received.response.body.next().await else {
            break;
        };
        let bytes = chunk?;
        buffered.extend_from_slice(&bytes);
        let frames = match splitter.push(&bytes) {
            Ok(frames) => frames,
            Err(_) => break,
        };
        for frame in frames {
            let Some(body) = std::str::from_utf8(&frame)
                .ok()
                .and_then(|s| JsonPayload::parse(s).ok())
            else {
                break 'read;
            };
            if body.string_at("/type").as_deref() == Some("ping") {
                continue;
            }
            if body.string_at("/type").as_deref() == Some("error") {
                let status = body
                    .unsigned_integer_at("/status")
                    .and_then(|v| u16::try_from(v).ok())
                    .unwrap_or_else(|| {
                        if body.string_at("/error/type").as_deref() == Some("not_found_error") {
                            404
                        } else {
                            400
                        }
                    });
                if replayable_error(status, &body) {
                    rejection = Some(crate::HttpResponse {
                        status,
                        headers: received.headers().to_vec(),
                        body: frame.into(),
                    });
                }
            }
            break 'read;
        }
    }
    if !buffered.is_empty() {
        let remaining = std::mem::replace(
            &mut received.response.body,
            futures::stream::empty().boxed(),
        );
        received.response.body = futures::stream::once(async move { Ok(buffered.into()) })
            .chain(remaining)
            .boxed();
    }
    Ok(rejection)
}
fn add_beta(draft: &mut RequestDraft, beta: &str) {
    let headers = draft.session_headers_mut();
    let mut values = Vec::<String>::new();
    headers.retain(|(key, value)| {
        if key.eq_ignore_ascii_case("anthropic-beta") {
            values.extend(value.split(',').map(|v| v.trim().to_owned()));
            false
        } else {
            true
        }
    });
    if !values.iter().any(|value| value == beta) {
        values.push(beta.to_owned());
    }
    headers.push(("anthropic-beta".into(), values.join(",")));
}

#[derive(Clone)]
pub(super) struct Observation {
    state: Arc<Mutex<State>>,
    epoch: u64,
    identity: [u8; 32],
    logical: JsonPayload,
    capability: ContinuationCapability,
    wire: Arc<Mutex<WireObservation>>,
}
#[derive(Default)]
struct WireObservation {
    splitter: crate::SseFrameSplitter,
    candidate: Option<JsonPayload>,
    blocks: BTreeMap<usize, JsonPayload>,
    partial: BTreeMap<usize, Vec<u16>>,
    message: Option<JsonPayload>,
    invalid: bool,
    committed: bool,
}
impl Observation {
    pub(super) fn complete(&self, response: &crate::HttpResponse) {
        let candidate = std::str::from_utf8(&response.body)
            .ok()
            .and_then(|s| JsonPayload::parse(s).ok());
        if let Some(candidate) = candidate {
            self.commit(&candidate);
        }
    }
    pub(super) fn frames(&self, response: &mut crate::StreamResponse) {
        // Responses and Interactions already expose a validated terminal
        // source from their decoder. Only Messages needs delta assembly here.
        if self.capability != ContinuationCapability::AnthropicThread {
            return;
        }
        let original = std::mem::replace(&mut response.body, futures::stream::empty().boxed());
        let observation = self.clone();
        response.body = original
            .map(move |chunk| {
                if let Ok(bytes) = &chunk {
                    observation.bytes(bytes);
                }
                chunk
            })
            .boxed();
    }
    fn bytes(&self, bytes: &[u8]) {
        let mut wire = self.wire.lock().expect("wire observation");
        if wire.invalid {
            return;
        }
        let frames = match wire.splitter.push(bytes) {
            Ok(frames) => frames,
            Err(_) => {
                wire.invalid = true;
                return;
            }
        };
        for frame in frames {
            if frame == b"[DONE]" {
                continue;
            }
            let Some(body) = std::str::from_utf8(&frame)
                .ok()
                .and_then(|s| JsonPayload::parse(s).ok())
            else {
                wire.invalid = true;
                continue;
            };
            if observe_wire(&mut wire, &body, self.capability).is_err() {
                wire.invalid = true;
            }
        }
    }
    pub(super) fn event(
        &self,
        event: &Result<StreamEvent, LlmError>,
        terminal: Option<&JsonPayload>,
    ) {
        if matches!(event, Err(_)) {
            self.invalidate();
        }
        if matches!(event, Ok(StreamEvent::End { .. })) {
            if let Some(terminal) = terminal {
                self.commit(terminal);
                return;
            }
            let wire = self.wire.lock().expect("wire observation");
            let candidate = (!wire.invalid).then(|| wire.candidate.clone()).flatten();
            drop(wire);
            if let Some(candidate) = candidate {
                self.commit(&candidate);
            }
        }
    }
    fn invalidate(&self) {
        let mut state = self.state.lock().expect("conversation state");
        if state.epoch == self.epoch {
            state.completed = None;
        }
    }
    fn commit(&self, body: &JsonPayload) {
        if body.raw_source_at("/store").as_deref() == Some("false") {
            self.invalidate();
            return;
        }
        if body
            .payload_at("/usage/iterations")
            .and_then(|v| v.array_items())
            .is_some_and(|iterations| {
                iterations
                    .iter()
                    .any(|v| v.string_at("/type").as_deref() == Some("fallback_message"))
            })
        {
            self.invalidate();
            return;
        }
        let id = body.string_at("/id").filter(|v| !v.is_empty());
        let output = match self.capability {
            ContinuationCapability::AnthropicThread
                if matches!(
                    body.string_at("/stop_reason").as_deref(),
                    Some("end_turn" | "tool_use" | "stop_sequence" | "pause_turn")
                ) =>
            {
                body.payload_at("/content")
                    .and_then(|content| {
                        replace(
                            &JsonPayload::parse("{\"role\":\"assistant\"}").expect("object"),
                            "content",
                            Some(content),
                        )
                        .ok()
                    })
                    .map(|v| vec![v])
            }
            ContinuationCapability::Responses
                if body.string_at("/status").as_deref() == Some("completed") =>
            {
                body.payload_at("/output").and_then(|v| v.array_items())
            }
            ContinuationCapability::GeminiInteraction
                if matches!(
                    body.string_at("/status").as_deref(),
                    Some("completed" | "requires_action")
                ) =>
            {
                body.payload_at("/steps").and_then(|v| v.array_items())
            }
            _ => None,
        };
        let mut wire = self.wire.lock().expect("wire observation");
        if wire.committed {
            return;
        }
        wire.committed = true;
        let mut state = self.state.lock().expect("conversation state");
        if state.epoch != self.epoch {
            return;
        }
        state.completed = id.zip(output).map(|(id, output)| Completed {
            identity: self.identity,
            logical: self.logical.clone(),
            output,
            id,
            at: Instant::now(),
        });
    }
}
fn observe_wire(
    wire: &mut WireObservation,
    body: &JsonPayload,
    capability: ContinuationCapability,
) -> Result<(), LlmError> {
    match capability {
        ContinuationCapability::Responses
            if body.string_at("/type").as_deref() == Some("response.completed") =>
        {
            wire.candidate = body.payload_at("/response")
        }
        ContinuationCapability::GeminiInteraction
            if body.string_at("/event_type").as_deref() == Some("interaction.completed") =>
        {
            wire.candidate = body.payload_at("/interaction")
        }
        ContinuationCapability::AnthropicThread => match body.string_at("/type").as_deref() {
            Some("message_start") => wire.message = body.payload_at("/message"),
            Some("content_block_start") => {
                if let (Some(index), Some(block)) = (
                    body.unsigned_integer_at("/index"),
                    body.payload_at("/content_block"),
                ) {
                    wire.blocks.insert(index as usize, block);
                }
            }
            Some("content_block_delta") => {
                let index = body
                    .unsigned_integer_at("/index")
                    .ok_or_else(|| invalid("block index"))? as usize;
                let delta = body.payload_at("/delta").ok_or_else(|| invalid("delta"))?;
                let kind = delta.string_at("/type");
                if kind.as_deref() == Some("citations_delta") {
                    let block = wire
                        .blocks
                        .get_mut(&index)
                        .ok_or_else(|| invalid("missing block"))?;
                    let mut citations = block
                        .payload_at("/citations")
                        .and_then(|v| v.array_items())
                        .unwrap_or_default();
                    citations.push(
                        delta
                            .payload_at("/citation")
                            .ok_or_else(|| invalid("citation"))?,
                    );
                    *block = replace(block, "citations", Some(array(&citations)?))?;
                    return Ok(());
                }
                let field = match kind.as_deref() {
                    Some("text_delta") => "text",
                    Some("thinking_delta") => "thinking",
                    Some("signature_delta") => "signature",
                    Some("input_json_delta") => "partial_json",
                    _ => return Err(invalid("unrecognized thread delta")),
                };
                let units = delta
                    .string_units_at(&format!("/{field}"))
                    .ok_or_else(|| invalid("delta string"))?;
                if field == "partial_json" {
                    wire.partial.entry(index).or_default().extend(units);
                } else {
                    let block = wire
                        .blocks
                        .get_mut(&index)
                        .ok_or_else(|| invalid("missing block"))?;
                    let mut text = block
                        .string_units_at(&format!("/{field}"))
                        .unwrap_or_default();
                    text.extend(units);
                    *block = replace(
                        block,
                        field,
                        Some(JsonPayload::string_from_utf16_code_units(&text)),
                    )?;
                }
            }
            Some("content_block_stop") => {
                let index = body
                    .unsigned_integer_at("/index")
                    .ok_or_else(|| invalid("block index"))? as usize;
                if let Some(units) = wire.partial.remove(&index) {
                    let fragment =
                        crate::protocol::JsonPayloadFragment::from_utf16_code_units(units);
                    let input = JsonPayload::parse_fragments([&fragment]).map_err(invalid)?;
                    let block = wire
                        .blocks
                        .get_mut(&index)
                        .ok_or_else(|| invalid("missing block"))?;
                    *block = replace(block, "input", Some(input))?;
                }
            }
            Some("message_delta") => {
                if let Some(message) = &mut wire.message {
                    if let Some(reason) = body.payload_at("/delta/stop_reason") {
                        *message = replace(message, "stop_reason", Some(reason))?;
                    }
                    if let Some(usage) = body.payload_at("/usage") {
                        *message = replace(message, "usage", Some(usage))?;
                    }
                }
            }
            Some("message_stop") => {
                if !wire.partial.is_empty() {
                    return Err(invalid("incomplete tool JSON"));
                }
                if let Some(message) = &wire.message {
                    wire.candidate = Some(replace(
                        message,
                        "content",
                        Some(array(&wire.blocks.values().cloned().collect::<Vec<_>>())?),
                    )?);
                }
            }
            _ => {}
        },
        _ => {}
    }
    Ok(())
}

fn logical_streaming(body: &JsonPayload) -> bool {
    body.raw_source_at("/stream").as_deref() == Some("true")
}

fn websocket_representable(body: &JsonPayload) -> bool {
    serde_json::from_str::<serde_json::Value>(body.as_json_source())
        .ok()
        .and_then(|value| JsonPayload::from_generated_value(&value).ok())
        .is_some_and(|roundtrip| body.semantically_eq(&roundtrip).unwrap_or(false))
}
