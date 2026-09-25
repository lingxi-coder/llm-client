//! `ModelStream`: the transport's frames run through the codec's decoder.

use crate::codecs::StreamDecoder;
use crate::files::AutomaticFileCleanup;
use crate::protocol::{LlmError, StreamEvent, Usage};
use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

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
    status: u16,
    headers: Vec<(String, String)>,
    executed_profile: String,
    automatic_file_cleanup: Option<Arc<AutomaticFileCleanup>>,
    automatic_file_cleanup_deadline: Option<Instant>,
    pub(super) pricing: Option<super::FrozenPricing>,
}

impl ModelStream {
    pub(super) fn new(
        resp: crate::transport::StreamResponse,
        mut decoder: Box<dyn StreamDecoder>,
        executed_profile: String,
        automatic_file_cleanup: Option<Arc<AutomaticFileCleanup>>,
        automatic_file_cleanup_deadline: Option<Instant>,
        requested_inference: crate::protocol::InferenceReport,
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
            status: resp.status,
            headers: resp.headers,
            executed_profile,
            automatic_file_cleanup,
            automatic_file_cleanup_deadline,
        }
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
                if let Some(cleanup) = self.automatic_file_cleanup.take() {
                    cleanup.finish(self.automatic_file_cleanup_deadline).await;
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
            if let Ok(
                StreamEvent::Inference { report }
                | StreamEvent::End {
                    inference: report, ..
                },
            ) = event
            {
                self.add_requested(report);
            }
        }
        if events
            .iter()
            .any(|e| matches!(e, Ok(StreamEvent::End { .. }) | Err(_)))
        {
            self.finish_transport();
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
                match &mut event {
                    Ok(StreamEvent::Inference { report })
                    | Ok(StreamEvent::End {
                        inference: report, ..
                    }) => self.add_requested(report),
                    _ => {}
                }
                if matches!(&event, Ok(StreamEvent::End { .. }) | Err(_)) {
                    self.finish_transport();
                }
                return Some(event);
            }
            if self.finished {
                if let Some(cleanup) = self.automatic_file_cleanup.take() {
                    cleanup.finish(self.automatic_file_cleanup_deadline).await;
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
                    self.finish_transport();
                    return Some(Err(error));
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
}
