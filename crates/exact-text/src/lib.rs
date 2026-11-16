//! Exact text construction shared by wire codecs and host accumulators.
//!
//! Display text is never authoritative for a carried UTF-16 string. Explicit
//! units remain explicit even when well formed, so adopting this type does not
//! change a caller's message variant or persistence format. There is no JSON,
//! Unicode normalization, request policy, or execution state in this crate.
#![forbid(unsafe_code)]

use std::borrow::Cow;

#[derive(Debug, Clone, Default)]
pub struct ExactText {
    display: String,
    units: Option<Vec<u16>>,
}

/// Display and exact units describe different strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayMismatch;

impl std::fmt::Display for DisplayMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("exact UTF-16 units do not match display text")
    }
}
impl std::error::Error for DisplayMismatch {}

impl ExactText {
    pub fn from_text(text: String) -> Self {
        Self {
            display: text,
            units: None,
        }
    }

    pub fn from_units(units: Vec<u16>) -> Self {
        Self {
            display: String::from_utf16_lossy(&units),
            units: Some(units),
        }
    }

    pub fn from_parts(display: String, units: Option<Vec<u16>>) -> Result<Self, DisplayMismatch> {
        if units
            .as_ref()
            .is_some_and(|units| String::from_utf16_lossy(units) != display)
        {
            return Err(DisplayMismatch);
        }
        Ok(Self { display, units })
    }

    pub fn display(&self) -> &str {
        &self.display
    }

    pub fn carried_units(&self) -> Option<&[u16]> {
        self.units.as_deref()
    }

    pub fn reserve_display(&mut self, additional_bytes: usize) {
        self.display.reserve(additional_bytes);
    }

    pub fn code_units(&self) -> Cow<'_, [u16]> {
        self.units.as_deref().map_or_else(
            || Cow::Owned(self.display.encode_utf16().collect()),
            Cow::Borrowed,
        )
    }

    pub fn into_parts(self) -> (String, Option<Vec<u16>>) {
        (self.display, self.units)
    }

    pub fn push_str(&mut self, text: &str) {
        if let Some(units) = &mut self.units {
            units.extend(text.encode_utf16());
        }
        self.display.push_str(text);
    }

    pub fn push_units(&mut self, fragment: &[u16]) {
        let units = self
            .units
            .get_or_insert_with(|| self.display.encode_utf16().collect());
        // Only a pending high surrogate can join a future fragment. Re-decode
        // that one boundary; scanning the complete prefix on every frame is
        // quadratic and can also accidentally normalize historical units.
        let high = units
            .last()
            .copied()
            .filter(|unit| (0xd800..=0xdbff).contains(unit));
        if high.is_some() {
            debug_assert!(self.display.ends_with(char::REPLACEMENT_CHARACTER));
            self.display.pop();
        }
        self.display.extend(
            char::decode_utf16(high.into_iter().chain(fragment.iter().copied()))
                .map(|ch| ch.unwrap_or(char::REPLACEMENT_CHARACTER)),
        );
        units.extend_from_slice(fragment);
    }
}

impl PartialEq for ExactText {
    fn eq(&self, other: &Self) -> bool {
        match (&self.units, &other.units) {
            (Some(a), Some(b)) => a == b,
            (None, None) => self.display == other.display,
            (Some(a), None) => a.iter().copied().eq(other.display.encode_utf16()),
            (None, Some(b)) => self.display.encode_utf16().eq(b.iter().copied()),
        }
    }
}
impl Eq for ExactText {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_fragment_partitions_match_complete_decode_and_exact_input() {
        let units = [0x61, 0xd83d, 0xde00, 0xd800, 0xd801, 0xdc01, 0xdc00, 0x62];
        for mask in 0..1usize << (units.len() - 1) {
            let mut text = ExactText::default();
            let mut start = 0;
            for end in 1..=units.len() {
                if end == units.len() || mask & (1 << (end - 1)) != 0 {
                    text.push_units(&units[start..end]);
                    text.push_units(&[]);
                    assert_eq!(text.display(), String::from_utf16_lossy(&units[..end]));
                    assert_eq!(text.carried_units(), Some(&units[..end]));
                    start = end;
                }
            }
            assert_eq!(text, ExactText::from_units(units.to_vec()));
        }
    }

    #[test]
    fn ordinary_input_has_no_sidecar_and_explicit_carriers_keep_their_shape() {
        let mut text = ExactText::from_text("中文😀e\u{301}\r\n\\\"".into());
        text.push_str(" more");
        assert!(text.carried_units().is_none());
        let units = text.code_units().into_owned();
        let carried = ExactText::from_units(units.clone());
        assert_eq!(text, carried);
        assert_eq!(carried.into_parts().1, Some(units));
    }

    #[test]
    fn equal_displays_do_not_conflate_distinct_source_strings() {
        let a = ExactText::from_units(vec![0xd800]);
        let b = ExactText::from_units(vec![0xd801]);
        let replacement = ExactText::from_text("�".into());
        assert_eq!(a.display(), b.display());
        assert_ne!(a, b);
        assert_ne!(a, replacement);
        assert_eq!(
            ExactText::from_parts("wrong".into(), Some(vec![0xd800])),
            Err(DisplayMismatch)
        );
    }

    #[test]
    fn mixed_fragments_and_new_builders_never_join_separate_attempts() {
        let mut first = ExactText::from_text("prefix ".into());
        first.push_units(&[0xd83d]);
        let checkpoint = first.clone();
        first.push_str("x");
        first.push_units(&[0xde00]);
        assert_eq!(first.display(), "prefix �x�");
        assert_eq!(checkpoint.display(), "prefix �");
        let mut resumed = ExactText::default();
        resumed.push_units(&[0xde00]);
        assert_eq!(resumed.display(), "�");
        let mut continued = checkpoint;
        continued.push_units(&[0xde00]);
        assert_eq!(continued.display(), "prefix 😀");
    }
}
