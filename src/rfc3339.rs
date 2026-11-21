//! RFC 3339 `date-time` parsing.
//!
//! jiff's ISO 8601 parser also accepts forms RFC 3339 excludes (no seconds, a bracketed
//! time zone annotation, no offset), so the shape is checked first: separator `T`, `t` or a
//! space; an optional fraction; `Z`, `z` or a `±hh:mm` offset.

use jiff::fmt::temporal::Pieces;
use jiff::Timestamp;

fn is_date_time(value: &str) -> bool {
    let bytes = value.as_bytes();
    let digits = |range: std::ops::Range<usize>| {
        bytes
            .get(range)
            .is_some_and(|part| part.iter().all(u8::is_ascii_digit))
    };
    if !(digits(0..4)
        && bytes.get(4) == Some(&b'-')
        && digits(5..7)
        && bytes.get(7) == Some(&b'-')
        && digits(8..10)
        && matches!(bytes.get(10), Some(b'T' | b't' | b' '))
        && digits(11..13)
        && bytes.get(13) == Some(&b':')
        && digits(14..16)
        && bytes.get(16) == Some(&b':')
        && digits(17..19))
    {
        return false;
    }
    let mut rest = &bytes[19..];
    if let Some(fraction) = rest.strip_prefix(b".") {
        let count = fraction
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        if count == 0 {
            return false;
        }
        rest = &fraction[count..];
    }
    match rest {
        [b'Z' | b'z'] => true,
        [b'+' | b'-', hours @ .., b':', m1, m2] if hours.len() == 2 => {
            hours.iter().chain([m1, m2]).all(u8::is_ascii_digit)
        }
        _ => false,
    }
}

/// The instant an RFC 3339 `date-time` names.
pub(crate) fn timestamp(value: &str) -> Option<Timestamp> {
    is_date_time(value).then(|| value.parse().ok()).flatten()
}

/// The local date, time and offset an RFC 3339 `date-time` was written with.
pub(crate) fn pieces(value: &str) -> Option<Pieces<'_>> {
    is_date_time(value)
        .then(|| Pieces::parse(value).ok())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_rfc3339_date_times_only() {
        for valid in [
            "2026-11-20T08:30:00Z",
            "2026-11-20t08:30:00z",
            "2026-11-20 08:30:00.123456789+08:00",
            "2026-11-20T08:30:00-05:00",
        ] {
            assert!(timestamp(valid).is_some(), "{valid}");
        }
        for invalid in [
            "2026-11-20",
            "2026-11-20T08:30Z",
            "2026-11-20T08:30:00",
            "2026-11-20T08:30:00+0800",
            "2026-11-20T08:30:00.Z",
            "2026-11-20T08:30:00Z[UTC]",
            "2026-11-20T08:30:00+08",
        ] {
            assert!(timestamp(invalid).is_none(), "{invalid}");
        }
        assert_eq!(
            timestamp("2026-11-20T08:30:00+08:00").map(Timestamp::as_second),
            timestamp("2026-11-20T00:30:00Z").map(Timestamp::as_second)
        );
    }
}
