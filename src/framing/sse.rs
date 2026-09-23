//! Incremental SSE frame splitting for byte-stream transports.
//!
//! Ported from the previous project's `llm-client/src/sse.rs`. A host whose
//! HTTP layer already parses SSE can hand each event's data payload straight to
//! a decoder and skip this.

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
    pub fn push(&mut self, bytes: &[u8]) -> Vec<Vec<u8>> {
        // Normalize CR, LF and CRLF while preserving a CRLF pair split
        // across transport chunks. A CR already terminates its line.
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
        }
        let mut frames = Vec::new();
        while let Some((content_len, consumed)) = find_event_boundary(&self.buffer) {
            if let Some(frame) = parse_event(&self.buffer[..content_len]) {
                frames.push(frame);
            }
            let remaining = self.buffer.len() - consumed;
            self.buffer.copy_within(consumed.., 0);
            self.buffer.truncate(remaining);
        }
        frames
    }

    /// Flush a trailing unterminated event at end of stream. A provider that
    /// closes without the final blank line still gets its last event decoded.
    pub fn finish(&mut self) -> Option<Vec<u8>> {
        self.skip_lf = false;
        let event = std::mem::take(&mut self.buffer);
        parse_event(&event)
    }
}

/// The first blank line: `(content_len, total_consumed)`.
fn find_event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    for (index, byte) in buffer.iter().enumerate() {
        if *byte != b'\n' {
            continue;
        }
        if buffer.get(index + 1) == Some(&b'\n') {
            return Some((index + 1, index + 2));
        }
        if buffer.get(index + 1) == Some(&b'\r') && buffer.get(index + 2) == Some(&b'\n') {
            return Some((index + 1, index + 3));
        }
    }
    None
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
            let mut frames = s.push(&bytes[..split]);
            frames.extend(s.push(&bytes[split..]));
            assert_eq!(text(frames), vec!["one", "two", "three"], "split {split}");
            assert!(s.finish().is_none());
        }
    }

    #[test]
    fn a_data_field_without_a_colon_is_an_empty_data_line() {
        let mut s = SseFrameSplitter::new();
        assert_eq!(text(s.push(b"data\n\n")), vec![""]);
    }

    #[test]
    fn one_event_per_blank_line() {
        let mut s = SseFrameSplitter::new();
        assert_eq!(
            text(s.push(b"data: one\n\ndata: two\n\n")),
            vec!["one", "two"]
        );
    }

    #[test]
    fn an_event_split_across_chunks_is_held_until_it_is_whole() {
        let mut s = SseFrameSplitter::new();
        assert!(s.push(b"data: par").is_empty(), "nothing is emitted yet");
        assert_eq!(text(s.push(b"tial\n\n")), vec!["partial"]);
    }

    #[test]
    fn several_data_lines_join_with_newlines() {
        let mut s = SseFrameSplitter::new();
        assert_eq!(text(s.push(b"data: a\ndata: b\n\n")), vec!["a\nb"]);
    }

    #[test]
    fn other_fields_and_comments_are_not_frames() {
        let mut s = SseFrameSplitter::new();
        assert!(s
            .push(b": keep-alive\nevent: ping\nid: 7\nretry: 100\n\n")
            .is_empty());
    }

    #[test]
    fn crlf_and_mixed_endings_both_terminate() {
        let mut s = SseFrameSplitter::new();
        assert_eq!(
            text(s.push(b"data: one\r\n\r\ndata: two\n\n")),
            vec!["one", "two"]
        );
    }

    #[test]
    fn a_stream_that_ends_without_a_blank_line_still_yields_its_last_event() {
        let mut s = SseFrameSplitter::new();
        assert!(s.push(b"data: last").is_empty());
        assert_eq!(String::from_utf8(s.finish().unwrap()).unwrap(), "last");
    }
}
