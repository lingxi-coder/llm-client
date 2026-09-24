//! Incremental SSE frame splitting for byte-stream transports.
//!
//! Ported from the previous project's `llm-client/src/sse.rs`. A host whose
//! HTTP layer already parses SSE can hand each event's data payload straight to
//! a decoder and skip this.

use crate::protocol::LlmError;

const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;

/// Splits an SSE byte stream into one frame per event.
///
/// A frame carries the event's joined `data:` payload without the field prefix
/// — exactly what a provider's stream decoder consumes. Comment lines and
/// `event:` / `id:` / `retry:` fields are ignored; an event with no data lines
/// produces no frame.
#[derive(Debug, Default)]
pub struct SseFrameSplitter {
    buffer: Vec<u8>,
    skip_lf: bool,
}

impl SseFrameSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes; return a frame for every event this chunk completed.
    ///
    /// A partial event stays buffered until a blank line terminates it, so a
    /// transport that splits mid-event loses nothing. Mixed line endings are
    /// accepted because providers mix them.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<Vec<u8>>, LlmError> {
        let (frames, error) = self.push_batch(bytes);
        match error {
            Some(error) => Err(error),
            None => Ok(frames),
        }
    }
    pub(crate) fn push_batch(&mut self, bytes: &[u8]) -> (Vec<Vec<u8>>, Option<LlmError>) {
        // Normalize CR, LF and CRLF while preserving a CRLF pair split
        // across transport chunks. A CR already terminates its line.
        let mut frames = Vec::new();
        for &byte in bytes {
            if std::mem::take(&mut self.skip_lf) && byte == b'\n' {
                continue;
            }
            if byte == b'\r' {
                self.buffer.push(b'\n');
                self.skip_lf = true;
            } else {
                self.buffer.push(byte);
            }
            if self.buffer.ends_with(b"\n\n") {
                if let Some(frame) = parse_event(&self.buffer[..self.buffer.len() - 1]) {
                    frames.push(frame);
                }
                self.buffer.clear();
            } else if self.buffer.len() > MAX_EVENT_BYTES {
                self.buffer.clear();
                return (
                    frames,
                    Some(LlmError::StreamInterrupted {
                        message: format!("SSE event exceeds the {MAX_EVENT_BYTES}-byte limit"),
                    }),
                );
            }
        }
        (frames, None)
    }

    /// Flush a trailing unterminated event at end of stream. A provider that
    /// closes without the final blank line still gets its last event decoded.
    pub fn finish(&mut self) -> Result<Option<Vec<u8>>, LlmError> {
        self.skip_lf = false;
        let event = std::mem::take(&mut self.buffer);
        Ok(parse_event(&event))
    }
}

fn parse_event(event: &[u8]) -> Option<Vec<u8>> {
    let mut data = Vec::new();
    let mut has_data = false;
    for line in event.split(|byte| *byte == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(mut value) = line
            .strip_prefix(b"data:")
            .or_else(|| (line == b"data").then_some(&b""[..]))
        else {
            continue;
        };
        // One optional space after the colon is part of the framing, not the
        // payload.
        if value.first() == Some(&b' ') {
            value = &value[1..];
        }
        if has_data {
            data.push(b'\n');
        }
        data.extend_from_slice(value);
        has_data = true;
    }
    has_data.then_some(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(frames: Vec<Vec<u8>>) -> Vec<String> {
        frames
            .into_iter()
            .map(|f| String::from_utf8(f).unwrap())
            .collect()
    }

    #[test]
    fn cr_and_mixed_endings_work_at_every_chunk_boundary() {
        let bytes = b"data: one\r\rdata: two\r\n\r\ndata: three\n\r";
        for split in 0..=bytes.len() {
            let mut s = SseFrameSplitter::new();
            let mut frames = s.push(&bytes[..split]).unwrap();
            frames.extend(s.push(&bytes[split..]).unwrap());
            assert_eq!(text(frames), vec!["one", "two", "three"], "split {split}");
            assert!(s.finish().unwrap().is_none());
        }
    }

    #[test]
    fn a_data_field_without_a_colon_is_an_empty_data_line() {
        let mut s = SseFrameSplitter::new();
        assert_eq!(text(s.push(b"data\n\n").unwrap()), vec![""]);
    }

    #[test]
    fn one_event_per_blank_line() {
        let mut s = SseFrameSplitter::new();
        assert_eq!(
            text(s.push(b"data: one\n\ndata: two\n\n").unwrap()),
            vec!["one", "two"]
        );
    }

    #[test]
    fn an_event_split_across_chunks_is_held_until_it_is_whole() {
        let mut s = SseFrameSplitter::new();
        assert!(
            s.push(b"data: par").unwrap().is_empty(),
            "nothing is emitted yet"
        );
        assert_eq!(text(s.push(b"tial\n\n").unwrap()), vec!["partial"]);
    }

    #[test]
    fn several_data_lines_join_with_newlines() {
        let mut s = SseFrameSplitter::new();
        assert_eq!(text(s.push(b"data: a\ndata: b\n\n").unwrap()), vec!["a\nb"]);
    }

    #[test]
    fn other_fields_and_comments_are_not_frames() {
        let mut s = SseFrameSplitter::new();
        assert!(s
            .push(b": keep-alive\nevent: ping\nid: 7\nretry: 100\n\n")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn crlf_and_mixed_endings_both_terminate() {
        let mut s = SseFrameSplitter::new();
        assert_eq!(
            text(s.push(b"data: one\r\n\r\ndata: two\n\n").unwrap()),
            vec!["one", "two"]
        );
    }

    #[test]
    fn a_stream_that_ends_without_a_blank_line_still_yields_its_last_event() {
        let mut s = SseFrameSplitter::new();
        assert!(s.push(b"data: last").unwrap().is_empty());
        assert_eq!(
            String::from_utf8(s.finish().unwrap().unwrap()).unwrap(),
            "last"
        );
    }

    #[test]
    fn an_unfinished_event_over_eight_mib_is_rejected() {
        let mut s = SseFrameSplitter::new();
        let chunk = vec![b'a'; MAX_EVENT_BYTES + 1];
        let err = s.push(&chunk).unwrap_err();
        assert!(
            matches!(err, LlmError::StreamInterrupted { message } if message.contains("SSE event") && message.contains("limit"))
        );
        assert!(s.buffer.len() <= MAX_EVENT_BYTES);
    }

    #[test]
    fn many_small_events_in_one_large_chunk_do_not_hit_the_limit() {
        let mut s = SseFrameSplitter::new();
        let event = [b"data: ".as_slice(), &vec![b'x'; 1024], b"\n\n"].concat();
        let count = MAX_EVENT_BYTES / event.len() + 1;
        let chunk = event.repeat(count);
        assert!(chunk.len() > MAX_EVENT_BYTES);
        assert_eq!(s.push(&chunk).unwrap().len(), count);
        assert!(s.buffer.is_empty());
    }
}
