//! Raw-byte framing adapters. A batch preserves earlier events before errors.
use super::{EventDecoder, StreamDecoder};
use crate::{
    framing::sse::SseFrameSplitter,
    protocol::{LlmError, StreamEvent, UsageReport},
};

pub(crate) struct SseDecoder<D> {
    inner: D,
    framing: SseFrameSplitter,
    ended: bool,
}
impl<D: EventDecoder> SseDecoder<D> {
    pub(crate) fn new(inner: D) -> Self {
        Self {
            inner,
            framing: Default::default(),
            ended: false,
        }
    }
    fn append(
        &mut self,
        result: Result<Vec<StreamEvent>, LlmError>,
        out: &mut Vec<Result<StreamEvent, LlmError>>,
    ) {
        match result {
            Ok(events) => {
                for event in events {
                    let terminal = matches!(event, StreamEvent::End { .. });
                    out.push(Ok(event));
                    if terminal {
                        self.ended = true;
                        break;
                    }
                }
            }
            Err(error) => {
                self.ended = true;
                out.push(Err(error));
            }
        }
    }
}
impl<D: EventDecoder> StreamDecoder for SseDecoder<D> {
    fn inference_report(&self) -> crate::protocol::InferenceReport {
        self.inner.inference_report()
    }
    fn set_response_headers(&mut self, headers: &[(String, String)]) {
        self.inner.set_response_headers(headers);
    }
    fn push_bytes(&mut self, bytes: &[u8]) -> Vec<Result<StreamEvent, LlmError>> {
        if self.ended {
            return vec![];
        }
        let mut out = Vec::new();
        let (frames, error) = self.framing.push_batch(bytes);
        for frame in frames {
            let events = self.inner.decode_frame(&frame);
            self.append(events, &mut out);
            if self.ended {
                break;
            }
        }
        if !self.ended {
            if let Some(error) = error {
                self.append(Err(error), &mut out);
            }
        }
        out
    }
    fn finish(&mut self) -> Vec<Result<StreamEvent, LlmError>> {
        if self.ended {
            return Vec::new();
        }
        let mut out = Vec::new();
        match self.framing.finish() {
            Ok(Some(frame)) => {
                let result = self.inner.decode_frame(&frame);
                self.append(result, &mut out);
            }
            Ok(None) => {}
            Err(error) => self.append(Err(error), &mut out),
        }
        if !self.ended {
            let result = self.inner.finish();
            self.append(result, &mut out);
        }
        self.ended = true;
        out
    }
    fn usage_report(&self) -> UsageReport {
        self.inner.usage_report()
    }
}
