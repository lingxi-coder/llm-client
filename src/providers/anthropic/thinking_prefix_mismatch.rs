//! Parsing for Native's bounded Anthropic prefix-lock mismatch response header.

use crate::protocol::{NativeThinkingPrefixBlock, NativeThinkingPrefixMismatch};

const HEADER_NAME: &str = "anthropic-thinking-prefix-mismatch";
const MAX_HEADER_UTF16_UNITS: usize = 2048;

/// Combine repeated case-insensitive header names the way Native's `Headers.get`
/// exposes them, then parse the resulting value.
pub fn parse_response_headers(
    headers: &[(String, String)],
) -> Option<NativeThinkingPrefixMismatch> {
    let mut values = headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case(HEADER_NAME))
        .map(|(_, value)| value.as_str());
    let mut combined = values.next()?.to_owned();
    for value in values {
        combined.push_str(", ");
        combined.push_str(value);
    }
    parse(Some(&combined))
}

/// Parse the value with Native's `String.length`, first-key-wins,
/// semicolon-delimited grammar. Oversized values are rejected.
pub fn parse(value: Option<&str>) -> Option<NativeThinkingPrefixMismatch> {
    let value = value?;
    let length = value.encode_utf16().count();
    if length == 0 || length > MAX_HEADER_UTF16_UNITS {
        return None;
    }

    let (mut block_value, mut kind_value) = (None, None);
    let (mut block_seen, mut kind_seen) = (false, false);
    for part in value.split(';') {
        let Some(equal) = part.find('=') else {
            continue;
        };
        if equal == 0 {
            continue;
        }
        let key = javascript_trim(&part[..equal]);
        let value = javascript_trim(&part[equal + 1..]);
        match key {
            "block" if !block_seen => {
                block_seen = true;
                block_value = Some(value);
            }
            "kind" if !kind_seen => {
                kind_seen = true;
                kind_value = Some(value);
            }
            _ => {}
        }
    }

    let block = block_value.and_then(parse_block);
    let kind = kind_value
        .filter(|value| {
            (1..=40).contains(&value.len())
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
        })
        .map(str::to_owned);
    (block.is_some() || kind.is_some()).then_some(NativeThinkingPrefixMismatch { block, kind })
}

fn parse_block(value: &str) -> Option<NativeThinkingPrefixBlock> {
    let value = value.strip_prefix("messages.")?;
    let (message_index, block_index) = value.split_once(".content.")?;
    Some(NativeThinkingPrefixBlock {
        message_index: parse_index(message_index)?,
        block_index: parse_index(block_index)?,
    })
}

fn parse_index(value: &str) -> Option<u32> {
    if !(1..=6).contains(&value.len()) || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

fn javascript_trim(value: &str) -> &str {
    fn is_space(character: char) -> bool {
        matches!(
            character,
            '\u{0009}'
                | '\u{000A}'
                | '\u{000B}'
                | '\u{000C}'
                | '\u{000D}'
                | '\u{0020}'
                | '\u{00A0}'
                | '\u{1680}'
                | '\u{2000}'
                ..='\u{200A}'
                    | '\u{2028}'
                    | '\u{2029}'
                    | '\u{202F}'
                    | '\u{205F}'
                    | '\u{3000}'
                    | '\u{FEFF}'
        )
    }

    let start = value
        .char_indices()
        .find_map(|(index, character)| (!is_space(character)).then_some(index))
        .unwrap_or(value.len());
    let end = value
        .char_indices()
        .rev()
        .find_map(|(index, character)| {
            (!is_space(character)).then_some(index + character.len_utf8())
        })
        .unwrap_or(start);
    &value[start..end]
}

#[cfg(test)]
mod tests {
    use super::parse;
    use crate::protocol::{NativeThinkingPrefixBlock, NativeThinkingPrefixMismatch};

    #[test]
    fn first_duplicate_fields_win_and_values_follow_native_grammar() {
        assert_eq!(
            parse(Some("\u{FEFF}block\u{FEFF}=messages.000007.content.3;block=bad;kind=lower_case;kind=UPPER")),
            Some(NativeThinkingPrefixMismatch {
                block: Some(NativeThinkingPrefixBlock {
                    message_index: 7,
                    block_index: 3,
                }),
                kind: Some("lower_case".into()),
            })
        );
    }

    #[test]
    fn malformed_values_do_not_create_a_mismatch() {
        for header in [
            "",
            "block=messages.1234567.content.0",
            "block=messages.1.content.2.extra",
            "kind=Upper",
            "kind=way-too-long-kind-that-is-over-forty-characters",
            " =messages.0.content.0",
        ] {
            assert_eq!(parse(Some(header)), None, "header={header:?}");
        }
    }

    #[test]
    fn header_limit_counts_javascript_utf16_units() {
        let mut header = "block=messages.0.content.0;pad=".to_owned();
        while header.encode_utf16().count() + 2 <= 2048 {
            header.push('😀');
        }
        while header.encode_utf16().count() < 2048 {
            header.push('x');
        }
        assert_eq!(header.encode_utf16().count(), 2048);
        assert!(parse(Some(&header)).is_some());
        header.push('x');
        assert_eq!(header.encode_utf16().count(), 2049);
        assert_eq!(parse(Some(&header)), None);
    }
}
