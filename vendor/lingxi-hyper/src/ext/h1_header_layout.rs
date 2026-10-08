use std::collections::HashMap;

use bytes::Bytes;
use http::header::{
    HeaderMap, HeaderName, InvalidHeaderName, CONTENT_LENGTH, HOST, TRANSFER_ENCODING,
};

/// Ordered HTTP/1 field-name occurrences. Values always come from the current
/// `HeaderMap`; this extension cannot restore removed credentials or raw bytes.
///
/// Fields inserted later by a connector, such as proxy authentication, are
/// appended once after the declared occurrences using their canonical names.
#[derive(Clone, Debug, Default)]
pub struct Http1HeaderLayout {
    entries: Vec<Entry>,
}

#[derive(Clone, Debug)]
struct Entry {
    name: HeaderName,
    spelling: Bytes,
    ordinal: usize,
    optional: bool,
}

impl Http1HeaderLayout {
    /// Create an empty layout.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a required occurrence, validating its original field-name bytes.
    /// Repeated case-insensitive names address successive current values.
    pub fn push(&mut self, name: &str) -> Result<(), InvalidHeaderName> {
        self.push_inner(name, false)
    }

    /// Reserve an occurrence for an automatically generated field. An absent
    /// field is skipped; its value is never synthesized by this extension.
    pub fn push_optional(&mut self, name: &str) -> Result<(), InvalidHeaderName> {
        self.push_inner(name, true)
    }

    fn push_inner(&mut self, spelling: &str, optional: bool) -> Result<(), InvalidHeaderName> {
        let name = HeaderName::from_bytes(spelling.as_bytes())?;
        let ordinal = self
            .entries
            .iter()
            .filter(|entry| entry.name == name)
            .count();
        self.entries.push(Entry {
            name,
            spelling: Bytes::copy_from_slice(spelling.as_bytes()),
            ordinal,
            optional,
        });
        Ok(())
    }

    pub(crate) fn validate(&self, headers: &HeaderMap) -> bool {
        let mut counts = HashMap::<&HeaderName, usize>::new();
        for entry in &self.entries {
            *counts.entry(&entry.name).or_default() += 1;
            if !entry.optional
                && headers
                    .get_all(&entry.name)
                    .iter()
                    .nth(entry.ordinal)
                    .is_none()
            {
                return false;
            }
        }
        // A changed multiplicity is ambiguous: never silently duplicate or
        // drop values when a caller mutates a request after binding its layout.
        if counts
            .iter()
            .any(|(name, count)| headers.get_all(*name).iter().count() > *count)
        {
            return false;
        }
        if headers.get_all(HOST).iter().count() > 1
            || headers.get_all(CONTENT_LENGTH).iter().count() > 1
            || (headers.contains_key(CONTENT_LENGTH) && headers.contains_key(TRANSFER_ENCODING))
        {
            return false;
        }
        if let Some(value) = headers.get(CONTENT_LENGTH) {
            let bytes = value.as_bytes();
            if bytes.is_empty()
                || !bytes.iter().all(u8::is_ascii_digit)
                || value
                    .to_str()
                    .ok()
                    .and_then(|text| text.parse::<u64>().ok())
                    .is_none()
            {
                return false;
            }
        }
        true
    }

    pub(crate) fn write(&self, headers: &HeaderMap, dst: &mut Vec<u8>) {
        for entry in &self.entries {
            if let Some(value) = headers.get_all(&entry.name).iter().nth(entry.ordinal) {
                dst.extend_from_slice(&entry.spelling);
                dst.extend_from_slice(b": ");
                dst.extend_from_slice(value.as_bytes());
                dst.extend_from_slice(b"\r\n");
            }
        }
        for (name, value) in headers {
            if !self.entries.iter().any(|entry| entry.name == name) {
                dst.extend_from_slice(name.as_str().as_bytes());
                dst.extend_from_slice(b": ");
                dst.extend_from_slice(value.as_bytes());
                dst.extend_from_slice(b"\r\n");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_duplicates_keep_live_values_and_spelling() {
        let mut layout = Http1HeaderLayout::new();
        for name in ["X-A", "x-B", "x-a"] {
            layout.push(name).unwrap();
        }
        let mut headers = HeaderMap::new();
        headers.append("x-a", "one".parse().unwrap());
        headers.append("x-b", "two".parse().unwrap());
        headers.append("x-a", "three".parse().unwrap());
        assert!(layout.validate(&headers));
        let mut bytes = Vec::new();
        layout.write(&headers, &mut bytes);
        assert_eq!(bytes, b"X-A: one\r\nx-B: two\r\nx-a: three\r\n");
        headers.remove("x-a");
        assert!(!layout.validate(&headers));
    }

    #[test]
    fn optional_generated_fields_do_not_restore_removed_values() {
        let mut layout = Http1HeaderLayout::new();
        layout.push_optional("Authorization").unwrap();
        let mut headers = HeaderMap::new();
        headers.insert("proxy-authorization", "current".parse().unwrap());
        assert!(layout.validate(&headers));
        let mut bytes = Vec::new();
        layout.write(&headers, &mut bytes);
        assert_eq!(bytes, b"proxy-authorization: current\r\n");
        assert!(layout.push("Bad\r\nHeader").is_err());
    }
}
