//! Exact JSON serialization for callers retaining JavaScript UTF-16 text.
use crate::protocol::LlmError;
use serde_json::Value;
use std::{collections::BTreeMap, io::Write};

/// Serialize JSON, replacing selected string leaves with their exact UTF-16
/// code units. JSON pointers must identify existing strings. Lone surrogates
/// are emitted as escapes rather than replaced with U+FFFD.
pub fn serialize(
    value: &Value,
    overrides: &BTreeMap<String, Vec<u16>>,
) -> Result<Vec<u8>, LlmError> {
    for pointer in overrides.keys() {
        if !matches!(value.pointer(pointer), Some(Value::String(_))) {
            return Err(LlmError::InvalidRequest {
                message: format!("UTF-16 JSON override does not target a string leaf: {pointer}"),
            });
        }
    }
    let mut out = Vec::new();
    write_json_value_with_overrides(&mut out, value, overrides, "")?;
    Ok(out)
}

fn write_json_value_with_overrides(
    out: &mut Vec<u8>,
    value: &Value,
    overrides: &BTreeMap<String, Vec<u16>>,
    pointer: &str,
) -> Result<(), LlmError> {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => serde_json::to_writer(out, value)
            .map_err(|err| LlmError::InvalidRequest {
                message: format!("failed to encode JSON leaf: {err}"),
            }),
        Value::String(text) => {
            if let Some(units) = overrides.get(pointer) {
                write_json_string_from_utf16(out, units);
                Ok(())
            } else {
                serde_json::to_writer(out, text).map_err(|err| LlmError::InvalidRequest {
                    message: format!("failed to encode JSON string: {err}"),
                })
            }
        }
        Value::Array(items) => {
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                let child_pointer = format!("{pointer}/{index}");
                write_json_value_with_overrides(out, item, overrides, &child_pointer)?;
            }
            out.push(b']');
            Ok(())
        }
        Value::Object(map) => {
            out.push(b'{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut *out, key).map_err(|err| LlmError::InvalidRequest {
                    message: format!("failed to encode JSON object key: {err}"),
                })?;
                out.push(b':');
                let child_pointer = format!("{pointer}/{}", escape_json_pointer_token(key));
                write_json_value_with_overrides(out, item, overrides, &child_pointer)?;
            }
            out.push(b'}');
            Ok(())
        }
    }
}

fn escape_json_pointer_token(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

fn write_json_string_from_utf16(out: &mut Vec<u8>, utf16_code_units: &[u16]) {
    out.push(b'"');
    let mut idx = 0usize;
    while idx < utf16_code_units.len() {
        let unit = utf16_code_units[idx];
        if matches!(unit, 0xD800..=0xDBFF)
            && utf16_code_units
                .get(idx + 1)
                .is_some_and(|next| matches!(next, 0xDC00..=0xDFFF))
        {
            let high = unit;
            let low = utf16_code_units[idx + 1];
            let scalar =
                0x1_0000 + ((((u32::from(high)) - 0xD800) << 10) | ((u32::from(low)) - 0xDC00));
            write_json_char(
                out,
                char::from_u32(scalar).expect("valid surrogate pair yields scalar"),
            );
            idx += 2;
            continue;
        }
        if matches!(unit, 0xD800..=0xDFFF) {
            write!(out, "\\u{unit:04x}").expect("vec writes cannot fail");
            idx += 1;
            continue;
        }
        write_json_char(
            out,
            char::from_u32(u32::from(unit)).expect("BMP non-surrogate is scalar"),
        );
        idx += 1;
    }
    out.push(b'"');
}

/// Returns the exact JSON byte length of a UTF-16 string, including quotes.
pub fn json_string_len_from_utf16(utf16_code_units: &[u16]) -> u64 {
    let mut encoded = Vec::new();
    write_json_string_from_utf16(&mut encoded, utf16_code_units);
    u64::try_from(encoded.len()).unwrap_or(u64::MAX)
}

fn write_json_char(out: &mut Vec<u8>, ch: char) {
    match ch {
        '"' => out.extend_from_slice(br#"\""#),
        '\\' => out.extend_from_slice(br"\\"),
        '\u{08}' => out.extend_from_slice(br"\b"),
        '\u{0C}' => out.extend_from_slice(br"\f"),
        '\n' => out.extend_from_slice(br"\n"),
        '\r' => out.extend_from_slice(br"\r"),
        '\t' => out.extend_from_slice(br"\t"),
        ch if ch <= '\u{1F}' => {
            write!(out, "\\u{:04x}", u32::from(ch)).expect("vec writes cannot fail");
        }
        _ => {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
}
