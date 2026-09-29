//! `ModelStream`: the transport's frames run through the codec's decoder.

use crate::codecs::StreamDecoder;
use crate::files::AutomaticFileCleanup;
use crate::protocol::{
    ChatResponse, ContinuationRef, LlmError, OutputFormat, ResponseCacheObservation, StreamEvent,
    StructuredOutputError, StructuredOutputErrorKind, Usage,
};
use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

const MAX_STRUCTURED_STREAM_BYTES: usize = 64 * 1024 * 1024;

/// Count the actual escaped JSON bytes without allocating an encoded event.
struct EventByteCounter {
    bytes: usize,
    limit: usize,
}

impl std::io::Write for EventByteCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let size = self
            .bytes
            .checked_add(bytes.len())
            .filter(|size| *size <= self.limit)
            .ok_or_else(|| std::io::Error::other("structured stream byte limit exceeded"))?;
        self.bytes = size;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A validated terminal stream value with the reconstructed response and all
/// provider-neutral events needed to inspect native output and usage.
#[derive(Debug)]
pub struct StructuredStreamResult<T> {
    pub value: T,
    pub response: ChatResponse,
    pub events: Vec<StreamEvent>,
}

#[derive(Debug, thiserror::Error)]
pub enum StructuredStreamError {
    #[error("structured stream failed: {source}")]
    Stream {
        #[source]
        source: LlmError,
        events: Vec<StreamEvent>,
    },
    #[error("structured stream has no terminal response")]
    MissingEnd { events: Vec<StreamEvent> },
    #[error("structured stream exceeds 64 MiB of decoded content")]
    TooLarge { events: Vec<StreamEvent> },
    #[error("structured stream validation failed: {source}")]
    Validation {
        #[source]
        source: StructuredOutputError,
        events: Vec<StreamEvent>,
    },
}

/// One synchronous decoder observation. Events may be empty while usage changed.
/// A terminal error is retained beside the last usage rather than replacing it.
pub struct StreamBatch {
    pub events: Vec<Result<StreamEvent, LlmError>>,
    pub usage: crate::protocol::UsageReport,
    pub inference: crate::protocol::InferenceReport,
    pub finished: bool,
}

/// A provider-neutral event stream: the transport's frames run through the
/// codec's decoder. Dropping it drops the underlying byte stream, which is how
/// a cancelled turn disconnects (§15).
pub struct ModelStream {
    frames: BoxStream<'static, Result<Bytes, LlmError>>,
    decoder: Box<dyn StreamDecoder>,
    requested_inference: crate::protocol::InferenceReport,
    ready: VecDeque<Result<StreamEvent, LlmError>>,
    finished: bool,
    yielded_event: bool,
    status: u16,
    headers: Vec<(String, String)>,
    executed_profile: String,
    response_cache: Option<ResponseCacheObservation>,
    provider_observation: crate::providers::dispatch::StreamObservation,
    continuation: Option<ContinuationRef>,
    continuation_completed: bool,
    automatic_file_cleanup: Option<(Arc<AutomaticFileCleanup>, Option<Instant>)>,
    pub(super) pricing: Option<super::FrozenPricing>,
}

impl ModelStream {
    pub(super) fn new(
        resp: crate::transport::StreamResponse,
        mut decoder: Box<dyn StreamDecoder>,
        executed_profile: String,
        response_cache: Option<ResponseCacheObservation>,
        automatic_file_cleanup: Option<(Arc<AutomaticFileCleanup>, Option<Instant>)>,
        requested_inference: crate::protocol::InferenceReport,
        continuation: Option<ContinuationRef>,
    ) -> Self {
        decoder.set_response_headers(&resp.headers);
        let mut ready = VecDeque::new();
        let initial = decoder.inference_report();
        if initial != crate::protocol::InferenceReport::default() {
            ready.push_back(Ok(StreamEvent::Inference { report: initial }));
        }
        Self {
            pricing: None,
            requested_inference,
            frames: resp.body,
            decoder,
            ready,
            finished: false,
            yielded_event: false,
            status: resp.status,
            headers: resp.headers,
            executed_profile,
            response_cache,
            provider_observation: Default::default(),
            continuation,
            continuation_completed: false,
            automatic_file_cleanup,
        }
    }

    pub(super) fn with_pricing(mut self, pricing: super::FrozenPricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    pub(super) fn with_provider_observation(
        mut self,
        observation: crate::providers::dispatch::StreamObservation,
    ) -> Self {
        self.provider_observation = observation;
        self
    }

    /// Latest first-party Anthropic container envelope observed in this stream.
    /// This metadata is available before completion and also after interruption;
    /// it does not assert that execution finished. The caller decides recovery.
    pub fn anthropic_container(
        &self,
    ) -> Option<&crate::providers::anthropic::types::AnthropicContainerMetadata> {
        self.provider_observation.anthropic_container()
    }

    /// Native Anthropic usage with reported fields folded across stream frames.
    /// Original JSON frames remain available through `ProviderEvent`. This
    /// observation may be partial until the terminal usage event arrives.
    pub fn anthropic_stop_details(&self) -> Option<&Value> {
        self.provider_observation.anthropic_stop_details()
    }

    pub fn anthropic_usage(&self) -> Option<&Value> {
        self.provider_observation.anthropic_usage()
    }

    /// The status the stream opened with.
    #[must_use]
    pub fn status(&self) -> u16 {
        self.status
    }

    /// The connection that accepted this streamed request.
    pub fn executed_profile(&self) -> &str {
        &self.executed_profile
    }

    /// State from a complete stream, suitable for the next Responses request.
    /// An interrupted stream never yields a continuation reference.
    pub fn continuation(&self) -> Option<&ContinuationRef> {
        self.continuation_completed
            .then_some(self.continuation.as_ref())
            .flatten()
            .filter(|reference| !reference.response_id.as_str().is_empty())
    }

    /// A response header of the streamed response, case-insensitively.
    ///
    /// These are what a provider reports rate-limit and quota state through,
    /// and they are only readable here because `open_stream` carries them; the
    /// frames alone never did.
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Every header the stream opened with, in the order the provider sent them.
    #[must_use]
    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    /// OpenRouter's documented response-cache observation from response headers.
    /// Absent or unrecognized cache-status headers remain `None`.
    pub fn response_cache(&self) -> Option<&ResponseCacheObservation> {
        self.response_cache.as_ref()
    }

    /// Frozen selected prices; ordinary client streams and prepared calls set this.
    pub fn pricing_snapshot(&self) -> Option<&super::FrozenPricing> {
        self.pricing.as_ref()
    }

    /// Read at most one transport chunk and immediately return its observation.
    /// No await occurs after decoding and before returning accounting facts.
    pub async fn next_batch(&mut self) -> Option<StreamBatch> {
        let mut events: Vec<_> = self.ready.drain(..).collect();
        if events.is_empty() {
            if self.finished {
                if let Some((cleanup, deadline)) = self.automatic_file_cleanup.take() {
                    cleanup.finish(deadline).await;
                }
                return None;
            }
            events = match self.frames.next().await {
                Some(Ok(chunk)) => self.decoder.push_bytes(&chunk),
                Some(Err(error)) => {
                    self.finish_transport();
                    vec![Err(error)]
                }
                None => {
                    let events = self.decoder.finish();
                    self.finish_transport();
                    events
                }
            };
        }
        for event in &mut events {
            self.observe_event(event);
        }
        Some(StreamBatch {
            events,
            usage: self.usage_report(),
            inference: self.inference_report(),
            finished: self.finished,
        })
    }

    /// The next event, or `None` at the end. One frame can decode to several
    /// events, so decoded events are buffered and drained before the next
    /// frame is pulled. After a terminal event or EOF, automatic Qwen file deletion may use the
    /// remaining request deadline, or up to 120 seconds without one.
    pub async fn next(&mut self) -> Option<Result<StreamEvent, LlmError>> {
        loop {
            if let Some(mut event) = self.ready.pop_front() {
                self.observe_event(&mut event);
                return Some(event);
            }
            if self.finished {
                if let Some((cleanup, deadline)) = self.automatic_file_cleanup.take() {
                    cleanup.finish(deadline).await;
                }
                return None;
            }
            match self.frames.next().await {
                Some(Ok(chunk)) => {
                    let events = self.decoder.push_bytes(&chunk);
                    if events
                        .iter()
                        .any(|event| matches!(event, Ok(StreamEvent::End { .. }) | Err(_)))
                    {
                        self.finish_transport();
                    }
                    self.ready.extend(events);
                }
                Some(Err(error)) => {
                    let mut event = Err(error);
                    self.observe_event(&mut event);
                    return Some(event);
                }
                None => {
                    self.ready.extend(self.decoder.finish());
                    self.finish_transport();
                }
            }
        }
    }

    /// A terminal outcome releases network resources even when the caller
    /// retains this handle to inspect usage and response headers.
    fn observe_event(&mut self, event: &mut Result<StreamEvent, LlmError>) {
        if self.yielded_event {
            if let Err(LlmError::Transport { message }) = event {
                *event = Err(LlmError::StreamInterrupted {
                    message: message.clone(),
                });
            }
        }
        if let Ok(event) = event {
            // Raw provider observations can be keepalives, and inference
            // settings may be seeded before any provider output arrives.
            self.yielded_event |= !matches!(
                event,
                StreamEvent::ProviderEvent { .. } | StreamEvent::Inference { .. }
            );
            self.provider_observation.observe(event);
        }
        match &mut *event {
            Ok(StreamEvent::Inference { report })
            | Ok(StreamEvent::End {
                inference: report, ..
            }) => self.add_requested(report),
            _ => {}
        }
        match &*event {
            Ok(StreamEvent::Start {
                response_id: Some(id),
                ..
            }) => {
                if let Some(reference) = &mut self.continuation {
                    reference.response_id = id.clone();
                }
            }
            Ok(StreamEvent::End { .. }) => self.continuation_completed = true,
            _ => {}
        }
        if matches!(&*event, Ok(StreamEvent::End { .. }) | Err(_)) {
            self.finish_transport();
        }
    }

    fn finish_transport(&mut self) {
        self.finished = true;
        self.frames = futures::stream::empty().boxed();
    }

    pub fn inference_report(&self) -> crate::protocol::InferenceReport {
        let mut report = self.decoder.inference_report();
        self.add_requested(&mut report);
        report
    }
    fn add_requested(&self, report: &mut crate::protocol::InferenceReport) {
        report.executed_at = self.requested_inference.executed_at;
        report.requested_effort = self.requested_inference.requested_effort;
        report.requested_service_tier = self.requested_inference.requested_service_tier;
        report
            .requested_raw_service_tier
            .clone_from(&self.requested_inference.requested_raw_service_tier);
    }

    pub fn usage_report(&self) -> crate::protocol::UsageReport {
        self.decoder.usage_report()
    }

    /// Usage the decoder observed, once the stream has ended.
    pub fn observed_usage(&self) -> Option<Usage> {
        self.decoder.usage_report().usage
    }

    /// Whether that usage is a complete, self-consistent report.
    ///
    /// `false` on a stream that ended before the provider restated its final
    /// counts, and on one whose subtotals do not reconcile. The counts are
    /// still worth showing; they are not worth billing from.
    #[must_use]
    pub fn usage_is_complete(&self) -> bool {
        self.decoder.usage_report().state == crate::protocol::UsageState::Complete
    }

    /// Consume a stream, then validate its final text against an output
    /// contract. Deltas are never parsed as standalone JSON.
    pub async fn collect_structured_json(
        mut self,
        format: &OutputFormat,
    ) -> Result<StructuredStreamResult<Value>, StructuredStreamError> {
        let mut accumulator = crate::StreamAccumulator::new();
        let mut event_bytes = EventByteCounter {
            bytes: 0,
            limit: MAX_STRUCTURED_STREAM_BYTES,
        };
        let mut saw_tool = false;
        while let Some(next) = self.next().await {
            let event = match next {
                Ok(event) => event,
                Err(source) => {
                    return Err(StructuredStreamError::Stream {
                        source,
                        events: accumulator.snapshot().events,
                    })
                }
            };
            if serde_json::to_writer(&mut event_bytes, &event).is_err() {
                return Err(StructuredStreamError::TooLarge {
                    events: accumulator.snapshot().events,
                });
            }
            saw_tool |= matches!(event, StreamEvent::ToolCallDelta { .. });
            accumulator.observe(&event);
        }
        let assembly = accumulator.snapshot();
        if !assembly.terminal {
            return Err(StructuredStreamError::MissingEnd {
                events: assembly.events,
            });
        }
        let events = assembly.events;
        let mut response = assembly.response;
        response.response_cache = self.response_cache.clone();
        response.continuation = self.continuation().cloned();
        response.executed_profile = Some(self.executed_profile.clone());
        response.set_anthropic_metadata(
            self.anthropic_container().cloned(),
            self.anthropic_usage().cloned(),
        );
        response.set_anthropic_stop_details(self.anthropic_stop_details().cloned());
        if saw_tool {
            return Err(StructuredStreamError::Validation {
                source: StructuredOutputError {
                    kind: StructuredOutputErrorKind::Incomplete,
                    response: Box::new(response),
                },
                events,
            });
        }
        let value = match response.structured_json(format) {
            Ok(value) => value,
            Err(source) => return Err(StructuredStreamError::Validation { source, events }),
        };
        Ok(StructuredStreamResult {
            value,
            response,
            events,
        })
    }

    /// Deserialize the validated final JSON while retaining its source data.
    pub async fn collect_structured<T: DeserializeOwned>(
        self,
        format: &OutputFormat,
    ) -> Result<StructuredStreamResult<T>, StructuredStreamError> {
        let result = self.collect_structured_json(format).await?;
        let value = match serde_json::from_value(result.value) {
            Ok(value) => value,
            Err(error) => {
                return Err(StructuredStreamError::Validation {
                    source: StructuredOutputError {
                        kind: StructuredOutputErrorKind::Deserialization(error.to_string()),
                        response: Box::new(result.response),
                    },
                    events: result.events,
                });
            }
        };
        Ok(StructuredStreamResult {
            value,
            response: result.response,
            events: result.events,
        })
    }
}

#[cfg(test)]
mod byte_budget_tests {
    use super::*;

    #[test]
    fn event_budget_counts_escaped_unicode_across_events_at_the_exact_limit() {
        let event = StreamEvent::TextDelta {
            block: 0,
            text: "quote\" slash\\ newline\n nul\0 雪🙂".into(),
        };
        let encoded = serde_json::to_vec(&event).unwrap();
        let mut counter = EventByteCounter {
            bytes: 0,
            limit: encoded.len() * 2,
        };
        serde_json::to_writer(&mut counter, &event).unwrap();
        assert_eq!(counter.bytes, encoded.len());
        serde_json::to_writer(&mut counter, &event).unwrap();
        assert_eq!(counter.bytes, counter.limit);
        assert!(serde_json::to_writer(&mut counter, &event).is_err());
        assert_eq!(counter.bytes, counter.limit);

        let mut too_small = EventByteCounter {
            bytes: 0,
            limit: encoded.len() - 1,
        };
        assert!(serde_json::to_writer(&mut too_small, &event).is_err());
        assert!(too_small.bytes <= too_small.limit);
    }
}
