//! `ModelStream`: the transport's frames run through the codec's decoder.

use crate::codecs::StreamDecoder;
use crate::framing::sse::SseFrameSplitter;
use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
use lingxi_agent_api::protocol::{LlmError, StreamEvent, Usage};
use std::collections::VecDeque;

/// A provider-neutral event stream: the transport's frames run through the
/// codec's decoder. Dropping it drops the underlying byte stream, which is how
/// a cancelled turn disconnects (§15).
pub struct ModelStream {
    frames: BoxStream<'static, Result<Bytes, LlmError>>,
    decoder: Box<dyn StreamDecoder>,
    sse: Option<SseFrameSplitter>,
    pending_frames: VecDeque<Bytes>,
    ready: VecDeque<StreamEvent>,
    finished: bool,
    status: u16,
    headers: Vec<(String, String)>,
    executed_profile: String,
}

impl ModelStream {
    pub(super) fn new(
        resp: crate::transport::StreamResponse,
        decoder: Box<dyn StreamDecoder>,
        executed_profile: String,
    ) -> Self {
        let sse = resp
            .header("content-type")
            .is_some_and(|value| {
                value
                    .split(';')
                    .next()
                    .unwrap_or_default()
                    .trim()
                    .eq_ignore_ascii_case("text/event-stream")
            })
            .then(SseFrameSplitter::new);
        Self {
            sse,
            pending_frames: VecDeque::new(),
            frames: resp.body,
            decoder,
            ready: VecDeque::new(),
            finished: false,
            status: resp.status,
            headers: resp.headers,
            executed_profile,
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

    /// The next event, or `None` at the end. One frame can decode to several
    /// events, so decoded events are buffered and drained before the next
    /// frame is pulled.
    pub async fn next(&mut self) -> Option<Result<StreamEvent, LlmError>> {
        loop {
            if let Some(ev) = self.ready.pop_front() {
                return Some(Ok(ev));
            }
            if self.finished {
                return None;
            }
            if let Some(frame) = self.pending_frames.pop_front() {
                match self.decoder.decode_frame(&frame) {
                    Ok(evs) => self.ready.extend(evs),
                    Err(e) => {
                        self.finished = true;
                        return Some(Err(e));
                    }
                }
                continue;
            }
            match self.frames.next().await {
                Some(Ok(chunk)) => {
                    if let Some(sse) = &mut self.sse {
                        match sse.push(&chunk) {
                            Ok(frames) => self
                                .pending_frames
                                .extend(frames.into_iter().map(Bytes::from)),
                            Err(error) => {
                                self.finished = true;
                                return Some(Err(error));
                            }
                        }
                    } else {
                        self.pending_frames.push_back(chunk);
                    }
                }
                Some(Err(e)) => {
                    self.finished = true;
                    return Some(Err(e));
                }
                None => {
                    let tail = match self.sse.as_mut().map(SseFrameSplitter::finish) {
                        Some(Ok(frame)) => frame,
                        Some(Err(error)) => {
                            self.finished = true;
                            return Some(Err(error));
                        }
                        None => None,
                    };
                    if let Some(frame) = tail {
                        match self.decoder.decode_frame(&frame) {
                            Ok(evs) => self.ready.extend(evs),
                            Err(e) => {
                                self.finished = true;
                                return Some(Err(e));
                            }
                        }
                    }
                    self.finished = true;
                    match self.decoder.finish() {
                        Ok(evs) => self.ready.extend(evs),
                        Err(e) => return Some(Err(e)),
                    }
                }
            }
        }
    }

    /// Usage the decoder observed, once the stream has ended.
    pub fn observed_usage(&self) -> Option<Usage> {
        self.decoder.observed_usage()
    }

    /// Whether that usage is a complete, self-consistent report.
    ///
    /// `false` on a stream that ended before the provider restated its final
    /// counts, and on one whose subtotals do not reconcile. The counts are
    /// still worth showing; they are not worth billing from.
    #[must_use]
    pub fn usage_is_complete(&self) -> bool {
        self.decoder.usage_is_complete()
    }
}
