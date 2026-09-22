//! AWS event-stream binary framing.
//!
//! Ported from the previous project's `llm-client/src/eventstream.rs`. One
//! provider streams in this format rather than SSE:
//!
//! ```text
//! [u32 BE total_len][u32 BE headers_len][u32 BE prelude_crc][headers][payload][u32 BE message_crc]
//! ```
//!
//! The CRC is IEEE CRC32 (reflected, poly 0xEDB8_8320), computed from a
//! hand-rolled table so this crate keeps its dependency list.

use lingxi_agent_api::protocol::LlmError;

const fn make_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0usize;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

const CRC_TABLE: [u32; 256] = make_crc_table();

/// IEEE CRC32, matching the POSIX/zlib/gzip family.
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC_TABLE[idx];
    }
    !crc
}

/// A decoded frame. Only string-valued headers are kept — the others are parsed
/// to advance the cursor and then dropped, because nothing here reads them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventStreamMessage {
    pub headers: Vec<(String, String)>,
    pub payload: Vec<u8>,
}

impl EventStreamMessage {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Reassembles frames from an arbitrary byte stream.
#[derive(Debug, Default)]
pub struct EventStreamSplitter {
    buf: Vec<u8>,
}

const PRELUDE_BYTES: usize = 12;
const FRAME_OVERHEAD: usize = 16;
/// Defence in depth against a hostile length field; real frames are far smaller.
const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

const VALUE_TYPE_BOOL_TRUE: u8 = 0;
const VALUE_TYPE_BOOL_FALSE: u8 = 1;
const VALUE_TYPE_I8: u8 = 2;
const VALUE_TYPE_I16: u8 = 3;
const VALUE_TYPE_I32: u8 = 4;
const VALUE_TYPE_I64: u8 = 5;
const VALUE_TYPE_BYTES: u8 = 6;
const VALUE_TYPE_STRING: u8 = 7;
const VALUE_TYPE_TIMESTAMP: u8 = 8;
const VALUE_TYPE_UUID: u8 = 9;

impl EventStreamSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk; return every frame it completed. A partial frame is kept
    /// until a later chunk finishes it.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<EventStreamMessage>, LlmError> {
        self.buf.extend_from_slice(chunk);
        let mut messages = Vec::new();

        loop {
            if self.buf.len() < PRELUDE_BYTES {
                break;
            }
            let total_len = u32::from_be_bytes(self.buf[0..4].try_into().unwrap()) as usize;
            let headers_len = u32::from_be_bytes(self.buf[4..8].try_into().unwrap()) as usize;

            // The prelude CRC is checked *before* waiting for `total_len` bytes.
            // A fabricated length would otherwise make this buffer gigabytes
            // waiting for a frame that never arrives.
            let want = u32::from_be_bytes(self.buf[8..12].try_into().unwrap());
            let got = crc32(&self.buf[0..8]);
            if got != want {
                return Err(interrupted(format!(
                    "event-stream prelude CRC mismatch: expected 0x{want:08X}, got 0x{got:08X}"
                )));
            }
            if total_len < FRAME_OVERHEAD {
                return Err(interrupted(format!(
                    "event-stream frame total_len={total_len} is below the minimum {FRAME_OVERHEAD}"
                )));
            }
            if total_len > MAX_FRAME_BYTES {
                return Err(interrupted(format!(
                    "event-stream frame total_len={total_len} exceeds the {MAX_FRAME_BYTES}-byte limit"
                )));
            }
            if self.buf.len() < total_len {
                break;
            }

            let frame = &self.buf[..total_len];
            let want = u32::from_be_bytes(frame[total_len - 4..total_len].try_into().unwrap());
            let got = crc32(&frame[..total_len - 4]);
            if got != want {
                return Err(interrupted(format!(
                    "event-stream message CRC mismatch: expected 0x{want:08X}, got 0x{got:08X}"
                )));
            }

            let headers_end = PRELUDE_BYTES + headers_len;
            if headers_end > total_len - 4 {
                return Err(interrupted(format!(
                    "event-stream headers_len={headers_len} overflows total_len={total_len}"
                )));
            }
            let headers = decode_headers(&frame[PRELUDE_BYTES..headers_end])?;
            let payload = frame[headers_end..total_len - 4].to_vec();
            messages.push(EventStreamMessage { headers, payload });

            let remaining = self.buf.len() - total_len;
            self.buf.copy_within(total_len.., 0);
            self.buf.truncate(remaining);
        }
        Ok(messages)
    }

    /// Anything left at end of stream means the stream stopped mid-frame.
    pub fn finish(&self) -> Result<(), LlmError> {
        if self.buf.is_empty() {
            Ok(())
        } else {
            Err(interrupted(format!(
                "event-stream ended mid-frame with {} unconsumed bytes",
                self.buf.len()
            )))
        }
    }
}

fn interrupted(message: String) -> LlmError {
    LlmError::StreamInterrupted { message }
}

fn decode_headers(mut data: &[u8]) -> Result<Vec<(String, String)>, LlmError> {
    let mut headers = Vec::new();
    while !data.is_empty() {
        let name_len = take_u8(&mut data)? as usize;
        let name_bytes = take(&mut data, name_len, "header name")?;
        let name = String::from_utf8(name_bytes.to_vec())
            .map_err(|e| interrupted(format!("event-stream header name is not UTF-8: {e}")))?;

        let value_type = take_u8(&mut data)?;
        match value_type {
            VALUE_TYPE_BOOL_TRUE | VALUE_TYPE_BOOL_FALSE => {}
            VALUE_TYPE_I8 => {
                take(&mut data, 1, "i8 header value")?;
            }
            VALUE_TYPE_I16 => {
                take(&mut data, 2, "i16 header value")?;
            }
            VALUE_TYPE_I32 => {
                take(&mut data, 4, "i32 header value")?;
            }
            VALUE_TYPE_I64 | VALUE_TYPE_TIMESTAMP => {
                take(&mut data, 8, "8-byte header value")?;
            }
            VALUE_TYPE_BYTES => {
                let len = take_u16(&mut data, "bytes header length")? as usize;
                take(&mut data, len, "bytes header value")?;
            }
            VALUE_TYPE_STRING => {
                let len = take_u16(&mut data, "string header length")? as usize;
                let value_bytes = take(&mut data, len, "string header value")?;
                let value = String::from_utf8(value_bytes.to_vec()).map_err(|e| {
                    interrupted(format!("event-stream header {name:?} is not UTF-8: {e}"))
                })?;
                headers.push((name, value));
            }
            VALUE_TYPE_UUID => {
                take(&mut data, 16, "uuid header value")?;
            }
            other => {
                return Err(interrupted(format!(
                    "event-stream header {name:?} has unknown value type {other}"
                )))
            }
        }
    }
    Ok(headers)
}

fn take<'a>(data: &mut &'a [u8], n: usize, what: &str) -> Result<&'a [u8], LlmError> {
    if data.len() < n {
        return Err(interrupted(format!(
            "event-stream truncated while reading {what}"
        )));
    }
    let (head, tail) = data.split_at(n);
    *data = tail;
    Ok(head)
}

fn take_u8(data: &mut &[u8]) -> Result<u8, LlmError> {
    Ok(take(data, 1, "byte")?[0])
}

fn take_u16(data: &mut &[u8], what: &str) -> Result<u16, LlmError> {
    let b = take(data, 2, what)?;
    Ok(u16::from_be_bytes([b[0], b[1]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one frame the way the provider does, so a test can feed real bytes.
    fn frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        let mut h = Vec::new();
        for (name, value) in headers {
            h.push(name.len() as u8);
            h.extend_from_slice(name.as_bytes());
            h.push(VALUE_TYPE_STRING);
            h.extend_from_slice(&(value.len() as u16).to_be_bytes());
            h.extend_from_slice(value.as_bytes());
        }
        let total = (PRELUDE_BYTES + h.len() + payload.len() + 4) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(&total.to_be_bytes());
        out.extend_from_slice(&(h.len() as u32).to_be_bytes());
        out.extend_from_slice(&crc32(&out[0..8]).to_be_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(payload);
        let crc = crc32(&out);
        out.extend_from_slice(&crc.to_be_bytes());
        out
    }

    #[test]
    fn the_crc_matches_the_standard() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn a_frame_round_trips_with_its_headers_and_payload() {
        let mut s = EventStreamSplitter::new();
        let got = s
            .feed(&frame(&[(":message-type", "event")], b"{\"a\":1}"))
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].header(":message-type"), Some("event"));
        assert_eq!(got[0].payload, b"{\"a\":1}");
        s.finish().unwrap();
    }

    #[test]
    fn a_frame_split_across_chunks_is_held_until_it_is_whole() {
        let bytes = frame(&[(":message-type", "event")], b"payload");
        let mut s = EventStreamSplitter::new();
        assert!(s.feed(&bytes[..10]).unwrap().is_empty());
        assert!(s.feed(&bytes[10..20]).unwrap().is_empty());
        assert_eq!(s.feed(&bytes[20..]).unwrap().len(), 1);
    }

    #[test]
    fn a_fabricated_length_is_refused_before_anything_is_buffered_for_it() {
        let mut s = EventStreamSplitter::new();
        // A huge total_len with a prelude CRC that does not match it.
        let mut bytes = vec![0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0];
        bytes.extend_from_slice(&0u32.to_be_bytes());
        let err = s.feed(&bytes).unwrap_err();
        assert!(
            matches!(err, LlmError::StreamInterrupted { ref message } if message.contains("prelude CRC")),
            "checking the prelude CRC first is what stops a bad length making \
             this buffer gigabytes: {err:?}"
        );
    }

    #[test]
    fn a_corrupted_payload_fails_the_message_crc() {
        let mut bytes = frame(&[(":message-type", "event")], b"payload");
        let last = bytes.len() - 6;
        bytes[last] ^= 0xFF;
        let err = EventStreamSplitter::new().feed(&bytes).unwrap_err();
        assert!(
            matches!(err, LlmError::StreamInterrupted { ref message } if message.contains("message CRC"))
        );
    }

    #[test]
    fn a_stream_that_stops_mid_frame_is_an_interruption_not_a_clean_end() {
        let bytes = frame(&[(":message-type", "event")], b"payload");
        let mut s = EventStreamSplitter::new();
        s.feed(&bytes[..bytes.len() - 3]).unwrap();
        assert!(s.finish().is_err());
    }
}
