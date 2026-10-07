//! The shallow string/presence view native session identity actually consumes.
use serde::{
    de::{IgnoredAny, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer,
};
use std::{collections::BTreeMap, fmt};

#[derive(Default)]
struct StringFacts {
    nonempty: bool,
    session_worker: bool,
}
impl<'de> Deserialize<'de> for StringFacts {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FactsVisitor;
        impl<'de> Visitor<'de> for FactsVisitor {
            type Value = StringFacts;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(StringFacts {
                    nonempty: !value.is_empty(),
                    session_worker: value == "session_worker",
                })
            }
            fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Self::Value, E> {
                Ok(StringFacts::default())
            }
            fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Self::Value, E> {
                Ok(StringFacts::default())
            }
            fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Self::Value, E> {
                Ok(StringFacts::default())
            }
            fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Self::Value, E> {
                Ok(StringFacts::default())
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(StringFacts::default())
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut values: A) -> Result<Self::Value, A::Error> {
                while values.next_element::<IgnoredAny>()?.is_some() {}
                Ok(StringFacts::default())
            }
            fn visit_map<A: MapAccess<'de>>(self, mut values: A) -> Result<Self::Value, A::Error> {
                while values.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(StringFacts::default())
            }
        }
        deserializer.deserialize_any(FactsVisitor)
    }
}
#[derive(Default)]
pub(super) struct Claims(BTreeMap<String, StringFacts>);
impl Claims {
    pub(super) fn contains(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
    pub(super) fn nonempty(&self, key: &str) -> bool {
        self.0.get(key).is_some_and(|value| value.nonempty)
    }
    pub(super) fn session_worker(&self) -> bool {
        self.0
            .get("ccr:role")
            .is_some_and(|value| value.session_worker)
    }
}
impl<'de> Deserialize<'de> for Claims {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ClaimsVisitor;
        impl<'de> Visitor<'de> for ClaimsVisitor {
            type Value = Claims;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut values: A) -> Result<Claims, A::Error> {
                let mut claims = Claims::default();
                while let Some(key) = values.next_key::<String>()? {
                    if matches!(
                        key.as_str(),
                        "account_uuid"
                            | "sub"
                            | "org_service_name"
                            | "code_agent_id"
                            | "ccr:role"
                            | "ccr:account_id"
                    ) {
                        claims.0.insert(key, values.next_value::<StringFacts>()?);
                    } else {
                        values.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(claims)
            }
        }
        deserializer.deserialize_map(ClaimsVisitor)
    }
}
fn hex(bytes: &[u8]) -> Option<u16> {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|value| u16::from_str_radix(value, 16).ok())
}

// Classification only needs property presence and string facts. Normalize
// numeric values to zero and lone UTF-16 surrogates to replacement characters
// without changing those facts. Strict lexing plus the final JSON parser rejects
// invalid grammar; unknown nested values are skipped instead of materialized.
fn normalized_json(raw: &str) -> Option<String> {
    let input = raw.as_bytes();
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        if input[i] == b'"' {
            out.push(b'"');
            i += 1;
            let mut closed = false;
            while i < input.len() {
                match input[i] {
                    b'"' => {
                        out.push(b'"');
                        i += 1;
                        closed = true;
                        break;
                    }
                    b'\\' => {
                        let escape = *input.get(i + 1)?;
                        if escape == b'u' {
                            let unit = hex(input.get(i + 2..i + 6)?)?;
                            let paired = (0xD800..=0xDBFF).contains(&unit)
                                && input.get(i + 6..i + 8) == Some(b"\\u")
                                && input
                                    .get(i + 8..i + 12)
                                    .and_then(hex)
                                    .is_some_and(|unit| (0xDC00..=0xDFFF).contains(&unit));
                            if paired {
                                out.extend_from_slice(&input[i..i + 12]);
                                i += 12;
                            } else if (0xD800..=0xDFFF).contains(&unit) {
                                out.extend_from_slice(b"\\ufffd");
                                i += 6;
                            } else {
                                out.extend_from_slice(&input[i..i + 6]);
                                i += 6;
                            }
                        } else {
                            if !matches!(
                                escape,
                                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't'
                            ) {
                                return None;
                            }
                            out.extend_from_slice(&input[i..i + 2]);
                            i += 2;
                        }
                    }
                    byte if byte < 0x20 => return None,
                    byte => {
                        out.push(byte);
                        i += 1;
                    }
                }
            }
            if !closed {
                return None;
            }
        } else if matches!(input[i], b'-' | b'0'..=b'9') {
            if input[i] == b'-' {
                i += 1;
            }
            match input.get(i)? {
                b'0' => i += 1,
                b'1'..=b'9' => {
                    i += 1;
                    while input.get(i).is_some_and(u8::is_ascii_digit) {
                        i += 1;
                    }
                }
                _ => return None,
            }
            if input.get(i) == Some(&b'.') {
                i += 1;
                let start = i;
                while input.get(i).is_some_and(u8::is_ascii_digit) {
                    i += 1;
                }
                if i == start {
                    return None;
                }
            }
            if matches!(input.get(i), Some(b'e' | b'E')) {
                i += 1;
                if matches!(input.get(i), Some(b'+' | b'-')) {
                    i += 1;
                }
                let start = i;
                while input.get(i).is_some_and(u8::is_ascii_digit) {
                    i += 1;
                }
                if i == start {
                    return None;
                }
            }
            if input.get(i).is_some_and(|byte| {
                !matches!(byte, b' ' | b'\t' | b'\r' | b'\n' | b',' | b']' | b'}')
            }) {
                return None;
            }
            out.push(b'0');
        } else {
            out.push(input[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}
pub(super) fn parse(raw: &str) -> Option<Claims> {
    let normalized = normalized_json(raw)?;
    let mut parser = serde_json::Deserializer::from_str(&normalized);
    let claims = Claims::deserialize(&mut parser).ok()?;
    parser.end().ok()?;
    Some(claims)
}
