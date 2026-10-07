//! Exact JSON serialization for callers retaining JavaScript UTF-16 text.
use crate::protocol::LlmError;
use serde::{
    ser::{SerializeMap, SerializeSeq},
    Deserialize, Serialize,
};
use serde_json::{Map, Value};
use std::{collections::BTreeMap, io::Write};

/// The route's JSON number and object-enumeration semantics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum JsonEncoding {
    #[default]
    Serde,
    JavaScript,
}
impl JsonEncoding {
    pub const fn for_protocol(protocol: crate::protocol::ProtocolFamily) -> Self {
        use crate::protocol::ProtocolFamily::*;
        match protocol {
            AnthropicMessages | FoundryClaude | BedrockClaude | VertexClaude => Self::JavaScript,
            _ => Self::Serde,
        }
    }
}

pub(crate) fn javascript_number(value: f64) -> String {
    ryu_js::Buffer::new().format_finite(value).to_owned()
}

pub(crate) fn ordered_entries(map: &Map<String, Value>) -> Vec<(&String, &Value)> {
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_by_key(|(key, _)| {
        key.parse::<u32>()
            .ok()
            .filter(|index| *index != u32::MAX && index.to_string() == **key)
            .map_or((1, 0), |index| (0, index))
    });
    entries
}

pub(crate) enum Entries<'a> {
    Plain(serde_json::map::Iter<'a>),
    Ordered(std::vec::IntoIter<(&'a String, &'a Value)>),
}
impl<'a> Iterator for Entries<'a> {
    type Item = (&'a String, &'a Value);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Plain(iter) => iter.next(),
            Self::Ordered(iter) => iter.next(),
        }
    }
}
pub(crate) fn entries(map: &Map<String, Value>, encoding: JsonEncoding) -> Entries<'_> {
    match encoding {
        JsonEncoding::Serde => Entries::Plain(map.iter()),
        JsonEncoding::JavaScript => Entries::Ordered(ordered_entries(map).into_iter()),
    }
}

pub(crate) struct JavaScriptValue<'a>(pub &'a Value);
impl Serialize for JavaScriptValue<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Value::Object(map) => {
                let mut object = serializer.serialize_map(Some(map.len()))?;
                for (key, value) in ordered_entries(map) {
                    object.serialize_entry(key, &JavaScriptValue(value))?;
                }
                object.end()
            }
            Value::Array(items) => {
                let mut array = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    array.serialize_element(&JavaScriptValue(item))?;
                }
                array.end()
            }
            other => other.serialize(serializer),
        }
    }
}

pub(crate) struct JavaScriptFormatter;
impl serde_json::ser::Formatter for JavaScriptFormatter {
    fn write_f64<W: ?Sized + Write>(&mut self, writer: &mut W, value: f64) -> std::io::Result<()> {
        writer.write_all(javascript_number(value).as_bytes())
    }
    fn write_f32<W: ?Sized + Write>(&mut self, writer: &mut W, value: f32) -> std::io::Result<()> {
        self.write_f64(writer, f64::from(value))
    }
    fn write_i64<W: ?Sized + Write>(&mut self, writer: &mut W, value: i64) -> std::io::Result<()> {
        self.write_f64(writer, value as f64)
    }
    fn write_u64<W: ?Sized + Write>(&mut self, writer: &mut W, value: u64) -> std::io::Result<()> {
        self.write_f64(writer, value as f64)
    }
    fn write_i128<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        value: i128,
    ) -> std::io::Result<()> {
        self.write_f64(writer, value as f64)
    }
    fn write_u128<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        value: u128,
    ) -> std::io::Result<()> {
        self.write_f64(writer, value as f64)
    }
}

/// Serialize JSON, replacing selected string leaves with their exact UTF-16
/// code units. JSON pointers must identify existing strings. Lone surrogates
/// are emitted as escapes rather than replaced with U+FFFD.
pub fn serialize(
    value: &Value,
    overrides: &BTreeMap<String, Vec<u16>>,
    encoding: JsonEncoding,
) -> Result<Vec<u8>, LlmError> {
    for pointer in overrides.keys() {
        if !canonical_string_pointer(value, pointer) {
            return Err(LlmError::InvalidRequest {
                message: format!("UTF-16 JSON override does not target a string leaf: {pointer}"),
            });
        }
    }
    let mut out = Vec::new();
    write_json_value_with_overrides(&mut out, value, overrides, "", encoding)?;
    Ok(out)
}

/// Serialize a prepared provider request at the final SDK body boundary.
/// Native's pinned `hook_prompt` path normalizes lone UTF-16 units in the
/// fully assembled Anthropic Messages body immediately before dispatch. Keep
/// the exact carrier intact until this call so intermediate host APIs retain
/// their original units.
pub fn serialize_for_request(
    value: &Value,
    overrides: &BTreeMap<String, Vec<u16>>,
    encoding: JsonEncoding,
    protocol: Option<crate::protocol::ProtocolFamily>,
    request_kind: crate::providers::anthropic::request_policy::AnthropicRequestKind,
) -> Result<Vec<u8>, LlmError> {
    use crate::providers::anthropic::request_policy::AnthropicRequestKind;
    if protocol != Some(crate::protocol::ProtocolFamily::AnthropicMessages)
        || request_kind != AnthropicRequestKind::HookPrompt
    {
        return serialize(value, overrides, encoding);
    }

    let normalized = overrides
        .iter()
        .map(|(pointer, units)| (pointer.clone(), to_well_formed_utf16(units)))
        .collect();
    serialize(value, &normalized, encoding)
}

fn to_well_formed_utf16(units: &[u16]) -> Vec<u16> {
    let mut normalized = Vec::with_capacity(units.len());
    let mut index = 0;
    while index < units.len() {
        let unit = units[index];
        if (0xD800..=0xDBFF).contains(&unit)
            && units
                .get(index + 1)
                .is_some_and(|next| (0xDC00..=0xDFFF).contains(next))
        {
            normalized.push(unit);
            normalized.push(units[index + 1]);
            index += 2;
        } else if (0xD800..=0xDFFF).contains(&unit) {
            normalized.push(0xFFFD);
            index += 1;
        } else {
            normalized.push(unit);
            index += 1;
        }
    }
    normalized
}

/// Reindex canonical ChatRequest text sidecars onto the selected codec's
/// encoded body. A missing or ambiguous target is an error so exact text can
/// never disappear silently during protocol projection.
pub fn map_message_text_overrides(
    request: &crate::protocol::ChatRequest,
    protocol: crate::protocol::ProtocolFamily,
    body: &Value,
    overrides: &BTreeMap<String, Vec<u16>>,
) -> Result<BTreeMap<String, Vec<u16>>, LlmError> {
    let family = crate::replay::native_family(protocol);
    let mut result = BTreeMap::new();
    let mut override_positions = BTreeMap::new();
    let mut passthrough = Vec::new();

    for (pointer, units) in overrides {
        if let Some((message_index, block_index)) = input_text_pointer(pointer) {
            let message =
                request
                    .messages
                    .get(message_index)
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: format!("UTF-16 message pointer is out of range: {pointer}"),
                    })?;
            let block =
                message
                    .content
                    .get(block_index)
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: format!("UTF-16 content pointer is out of range: {pointer}"),
                    })?;
            if !matches!(
                block,
                crate::protocol::ContentBlock::Text { .. }
                    | crate::protocol::ContentBlock::TextJsUtf16 { .. }
            ) {
                return Err(LlmError::InvalidRequest {
                    message: format!("UTF-16 text pointer does not target a text block: {pointer}"),
                });
            }
            override_positions.insert((message_index, block_index), units.clone());
        } else {
            passthrough.push((pointer.clone(), units.clone()));
        }
    }

    // Include every text-emitting source in each identity group, not only
    // overridden typed leaves. Gemini Thinking and Anthropic native text
    // ProviderContent become text-shaped wire fields too. Count them in the
    // source order so an identical display value is mapped by its stable
    // occurrence, while only typed Text blocks can carry an override.
    let mut source_texts: BTreeMap<(String, String), Vec<(usize, usize, Option<Vec<u16>>)>> =
        BTreeMap::new();
    for (message_index, message) in request.messages.iter().enumerate() {
        let role = input_text_role(message.role, family).to_owned();
        for (block_index, block) in message.content.iter().enumerate() {
            if let Some(text) = source_text_contributor_text(block, family) {
                let explicit_units = override_positions.remove(&(message_index, block_index));
                let carried_units = match block {
                    crate::protocol::ContentBlock::TextJsUtf16 {
                        utf16_code_units, ..
                    } => Some(utf16_code_units.clone()),
                    _ => None,
                };
                if let (Some(explicit), Some(carried)) = (&explicit_units, &carried_units) {
                    if explicit != carried {
                        return Err(LlmError::InvalidRequest {
                            message: "UTF-16 block carrier conflicts with its request override"
                                .into(),
                        });
                    }
                }
                let units = carried_units.or(explicit_units);
                if units
                    .as_ref()
                    .is_some_and(|units| String::from_utf16_lossy(units) != text)
                {
                    return Err(LlmError::InvalidRequest {
                        message: "UTF-16 block units do not match the text display string".into(),
                    });
                }
                source_texts
                    .entry((role.clone(), text.to_owned()))
                    .or_default()
                    .push((message_index, block_index, units));
            }
        }
    }
    if !override_positions.is_empty() {
        return Err(LlmError::InvalidRequest {
            message: "UTF-16 text override does not map to a source text block".into(),
        });
    }

    let targets = encoded_text_fields(body, family);
    for ((role, text), source_texts) in source_texts {
        if !source_texts.iter().any(|(_, _, units)| units.is_some()) {
            continue;
        }
        let matching: Vec<_> = targets
            .iter()
            .filter(|(candidate_role, candidate_text, _)| {
                candidate_role == &role && candidate_text == &text
            })
            .collect();
        if matching.len() != source_texts.len() {
            return Err(LlmError::InvalidRequest {
                message: format!(
                    "UTF-16 source text maps to {} encoded text fields, expected {}",
                    matching.len(),
                    source_texts.len()
                ),
            });
        }
        for ((_, _, units), (_, _, pointer)) in source_texts.into_iter().zip(matching) {
            if let Some(units) = units {
                result.insert(pointer.clone(), units);
            }
        }
    }

    for (pointer, units) in passthrough {
        if family != crate::protocol::ProtocolFamily::AnthropicMessages
            || body.pointer(&pointer).and_then(Value::as_str).is_none()
        {
            return Err(LlmError::InvalidRequest {
                message: format!(
                    "UTF-16 override does not map to a string in the encoded request: {pointer}"
                ),
            });
        }
        result.insert(pointer, units);
    }
    Ok(result)
}

/// Display text emitted as a top-level text-shaped field by the current codec
/// family. This is only a source-contributor projection for stable identity
/// mapping; overrides remain valid only for typed Text blocks.
fn source_text_contributor_text(
    block: &crate::protocol::ContentBlock,
    family: crate::protocol::ProtocolFamily,
) -> Option<&str> {
    use crate::protocol::{ContentBlock, ProtocolFamily};

    match block {
        ContentBlock::Text { text, .. } | ContentBlock::TextJsUtf16 { text, .. } => Some(text),
        ContentBlock::Thinking { text, .. } if family == ProtocolFamily::GeminiGenerateContent => {
            Some(text)
        }
        ContentBlock::ProviderContent { protocol, value }
            if family == ProtocolFamily::AnthropicMessages
                && *protocol == ProtocolFamily::AnthropicMessages
                && value["type"].as_str() == Some("text") =>
        {
            value["text"].as_str()
        }
        _ => None,
    }
}

fn input_text_pointer(pointer: &str) -> Option<(usize, usize)> {
    let tokens = pointer.strip_prefix('/')?.split('/').collect::<Vec<_>>();
    if tokens.len() != 5 || tokens[0] != "messages" || tokens[2] != "content" || tokens[4] != "text"
    {
        return None;
    }
    let message = tokens[1].parse::<usize>().ok()?;
    let block = tokens[3].parse::<usize>().ok()?;
    (message.to_string() == tokens[1] && block.to_string() == tokens[3]).then_some((message, block))
}

fn input_text_role(
    role: crate::protocol::MessageRole,
    family: crate::protocol::ProtocolFamily,
) -> &'static str {
    match (family, role) {
        (
            crate::protocol::ProtocolFamily::GeminiGenerateContent,
            crate::protocol::MessageRole::Assistant,
        ) => "model",
        (_, crate::protocol::MessageRole::Assistant) => "assistant",
        (_, crate::protocol::MessageRole::User) => "user",
        (_, crate::protocol::MessageRole::System) => "system",
    }
}

fn encoded_text_fields(
    body: &Value,
    family: crate::protocol::ProtocolFamily,
) -> Vec<(String, String, String)> {
    let mut fields = Vec::new();
    let mut add = |role: &str, text: &Value, pointer: String| {
        if let Some(text) = text.as_str() {
            fields.push((role.to_owned(), text.to_owned(), pointer));
        }
    };
    match family {
        crate::protocol::ProtocolFamily::AnthropicMessages => {
            for (message_index, message) in body["messages"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
                let role = message["role"].as_str().unwrap_or_default();
                for (block_index, block) in message["content"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    if block["type"].as_str() == Some("text") {
                        add(
                            role,
                            &block["text"],
                            format!("/messages/{message_index}/content/{block_index}/text"),
                        );
                    }
                }
            }
        }
        crate::protocol::ProtocolFamily::OpenAiChat => {
            for (message_index, message) in body["messages"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
                let role = message["role"].as_str().unwrap_or_default();
                match &message["content"] {
                    Value::String(_) => add(
                        role,
                        &message["content"],
                        format!("/messages/{message_index}/content"),
                    ),
                    Value::Array(parts) => {
                        for (part_index, part) in parts.iter().enumerate() {
                            if part["type"].as_str() == Some("text") {
                                add(
                                    role,
                                    &part["text"],
                                    format!("/messages/{message_index}/content/{part_index}/text"),
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        crate::protocol::ProtocolFamily::OpenAiResponses => {
            for (message_index, message) in
                body["input"].as_array().into_iter().flatten().enumerate()
            {
                if message["type"].as_str() != Some("message") {
                    continue;
                }
                let role = message["role"].as_str().unwrap_or_default();
                for (part_index, part) in message["content"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    if matches!(part["type"].as_str(), Some("input_text" | "output_text")) {
                        add(
                            role,
                            &part["text"],
                            format!("/input/{message_index}/content/{part_index}/text"),
                        );
                    }
                }
            }
        }
        crate::protocol::ProtocolFamily::GeminiGenerateContent => {
            for (message_index, message) in body["contents"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
                let role = message["role"].as_str().unwrap_or_default();
                for (part_index, part) in message["parts"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    add(
                        role,
                        &part["text"],
                        format!("/contents/{message_index}/parts/{part_index}/text"),
                    );
                }
            }
        }
        _ => unreachable!("native_family selects one of four current codec families"),
    }
    fields
}

pub(crate) fn canonical_string_pointer(mut value: &Value, pointer: &str) -> bool {
    if pointer.is_empty() {
        return value.is_string();
    }
    let Some(tokens) = pointer.strip_prefix('/') else {
        return false;
    };
    for token in tokens.split('/') {
        value = match value {
            Value::Object(map) => {
                let key = token.replace("~1", "/").replace("~0", "~");
                if escape_json_pointer_token(&key) != token {
                    return false;
                }
                let Some(child) = map.get(&key) else {
                    return false;
                };
                child
            }
            Value::Array(items) => {
                let Ok(index) = token.parse::<usize>() else {
                    return false;
                };
                if index.to_string() != token {
                    return false;
                }
                let Some(child) = items.get(index) else {
                    return false;
                };
                child
            }
            _ => return false,
        };
    }
    value.is_string()
}

fn write_json_value_with_overrides(
    out: &mut Vec<u8>,
    value: &Value,
    overrides: &BTreeMap<String, Vec<u16>>,
    pointer: &str,
    encoding: JsonEncoding,
) -> Result<(), LlmError> {
    match value {
        Value::Number(number) if encoding == JsonEncoding::JavaScript => {
            out.extend_from_slice(
                javascript_number(number.as_f64().expect("JSON number")).as_bytes(),
            );
            Ok(())
        }
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
                write_json_value_with_overrides(out, item, overrides, &child_pointer, encoding)?;
            }
            out.push(b']');
            Ok(())
        }
        Value::Object(map) => {
            out.push(b'{');
            for (index, (key, item)) in entries(map, encoding).enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut *out, key).map_err(|err| LlmError::InvalidRequest {
                    message: format!("failed to encode JSON object key: {err}"),
                })?;
                out.push(b':');
                let child_pointer = format!("{pointer}/{}", escape_json_pointer_token(key));
                write_json_value_with_overrides(out, item, overrides, &child_pointer, encoding)?;
            }
            out.push(b'}');
            Ok(())
        }
    }
}

fn escape_json_pointer_token(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

pub(crate) fn write_json_string_from_utf16(out: &mut Vec<u8>, utf16_code_units: &[u16]) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn native_ieee754_json_bytes_and_numeric_key_enumeration_match_reference() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/json_numbers_2_1_287.json"))
                .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let bits = u64::from_str_radix(case["bits"].as_str().unwrap(), 16).unwrap();
            let number = f64::from_bits(bits);
            let parsed: Value =
                serde_json::from_str(case["expected_json"].as_str().unwrap()).unwrap();
            assert_eq!(
                parsed.as_f64().unwrap().to_bits(),
                if number == 0.0 { 0 } else { bits },
                "decimal parser {case}"
            );
            assert_eq!(
                javascript_number(number),
                case["expected_string"].as_str().unwrap(),
                "{case}"
            );
            assert_eq!(
                String::from_utf8(
                    serialize(&json!(number), &BTreeMap::new(), JsonEncoding::JavaScript).unwrap()
                )
                .unwrap(),
                case["expected_json"].as_str().unwrap(),
                "{case}"
            );
            let body = json!({"number":number,"before":true,"2":number,"0":number,"01":number,"nested":[{"number":number},-0.0]});
            assert_eq!(
                String::from_utf8(
                    serialize(&body, &BTreeMap::new(), JsonEncoding::JavaScript).unwrap()
                )
                .unwrap(),
                case["expected_body_json"].as_str().unwrap(),
                "{case}"
            );
            assert_eq!(
                serialize(&body, &BTreeMap::new(), JsonEncoding::Serde).unwrap(),
                serde_json::to_vec(&body).unwrap()
            );
        }
    }

    #[test]
    fn overrides_require_canonical_pointers_and_keep_exact_unicode_with_native_numbers() {
        for path in ["/~2", "/items/00"] {
            assert!(serialize(
                &json!({"~2":"x","items":["x"]}),
                &BTreeMap::from([(path.into(), vec![0xD800])]),
                JsonEncoding::JavaScript
            )
            .is_err());
        }
        let body = json!({"2":1.0,"0":-0.0,"a/~":["display"]});
        let strings = BTreeMap::from([("/a~1~0/0".into(), vec![0xD800, 65, 0xDC00])]);
        assert_eq!(
            String::from_utf8(serialize(&body, &strings, JsonEncoding::JavaScript).unwrap())
                .unwrap(),
            r#"{"0":0,"2":1,"a/~":["\ud800A\udc00"]}"#
        );
        let body = json!({"integer":u64::MAX,"negative_zero":-0.0});
        assert_eq!(
            String::from_utf8(serialize(&body, &BTreeMap::new(), JsonEncoding::Serde).unwrap())
                .unwrap(),
            r#"{"integer":18446744073709551615,"negative_zero":-0.0}"#
        );
        assert_eq!(
            String::from_utf8(
                serialize(&body, &BTreeMap::new(), JsonEncoding::JavaScript).unwrap()
            )
            .unwrap(),
            r#"{"integer":18446744073709552000,"negative_zero":0}"#
        );
    }

    #[test]
    fn hook_prompt_policy_is_applied_only_by_the_final_anthropic_sdk_serializer() {
        use crate::protocol::ProtocolFamily;
        use crate::providers::anthropic::request_policy::AnthropicRequestKind as Kind;

        let body = json!({"messages":[{"content":[{"text":"display"}]}]});
        let units = vec![0xD800, b'x' as u16, 0xD83D, 0xDE03, 0xDC00];
        let overrides = BTreeMap::from([("/messages/0/content/0/text".into(), units)]);

        let native_hook = String::from_utf8(
            serialize_for_request(
                &body,
                &overrides,
                JsonEncoding::JavaScript,
                Some(ProtocolFamily::AnthropicMessages),
                Kind::HookPrompt,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(native_hook.contains("\u{fffd}x😃\u{fffd}"));
        assert!(!native_hook.contains("\\ud800"));

        let ordinary = String::from_utf8(
            serialize_for_request(
                &body,
                &overrides,
                JsonEncoding::JavaScript,
                Some(ProtocolFamily::AnthropicMessages),
                Kind::SideQuery,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(ordinary.contains("\\ud800x😃\\udc00"));

        let other_protocol = String::from_utf8(
            serialize_for_request(
                &body,
                &overrides,
                JsonEncoding::JavaScript,
                Some(ProtocolFamily::OpenAiChat),
                Kind::HookPrompt,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(other_protocol.contains("\\ud800x😃\\udc00"));
    }

    #[test]
    fn canonical_text_overrides_map_to_every_current_codec_body_shape() {
        use crate::protocol::{
            ChatRequest, ContentBlock, ConversationMessage, MessageRole, ProtocolFamily as P,
        };

        let mut request = ChatRequest::new("model");
        request.messages.push(ConversationMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: "\u{fffd}".into(),
                thought_signature: None,
                citations: None,
            }],
            native_options: Vec::new(),
        });
        let source = BTreeMap::from([("/messages/0/content/0/text".into(), vec![0xD800])]);
        let cases = [
            (
                P::AnthropicMessages,
                json!({"messages":[{"role":"user","content":[{"type":"text","text":"\u{fffd}"}]}]}),
                "/messages/0/content/0/text",
            ),
            (
                P::OpenAiChat,
                json!({"messages":[{"role":"user","content":"\u{fffd}"}]}),
                "/messages/0/content",
            ),
            (
                P::OpenAiResponses,
                json!({"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"\u{fffd}"}]}]}),
                "/input/0/content/0/text",
            ),
            (
                P::GeminiGenerateContent,
                json!({"contents":[{"role":"user","parts":[{"text":"\u{fffd}"}]}]}),
                "/contents/0/parts/0/text",
            ),
            (
                P::VertexClaude,
                json!({"messages":[{"role":"user","content":[{"type":"text","text":"\u{fffd}"}]}]}),
                "/messages/0/content/0/text",
            ),
            (
                P::VertexGemini,
                json!({"contents":[{"role":"user","parts":[{"text":"\u{fffd}"}]}]}),
                "/contents/0/parts/0/text",
            ),
            (
                P::BedrockClaude,
                json!({"messages":[{"role":"user","content":[{"type":"text","text":"\u{fffd}"}]}]}),
                "/messages/0/content/0/text",
            ),
            (
                P::AzureOpenAi,
                json!({"messages":[{"role":"user","content":"\u{fffd}"}]}),
                "/messages/0/content",
            ),
            (
                P::FoundryClaude,
                json!({"messages":[{"role":"user","content":[{"type":"text","text":"\u{fffd}"}]}]}),
                "/messages/0/content/0/text",
            ),
        ];
        for (protocol, body, expected) in cases {
            let mapped = map_message_text_overrides(&request, protocol, &body, &source)
                .unwrap_or_else(|error| panic!("{protocol:?}: {error}"));
            assert_eq!(mapped.get(expected), Some(&vec![0xD800]), "{protocol:?}");
        }
    }

    #[test]
    fn canonical_text_override_refuses_ambiguous_target_fields() {
        use crate::protocol::{
            ChatRequest, ContentBlock, ConversationMessage, MessageRole, ProtocolFamily,
        };

        let mut request = ChatRequest::new("model");
        request.messages.push(ConversationMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: "display".into(),
                thought_signature: None,
                citations: None,
            }],
            native_options: Vec::new(),
        });
        let source = BTreeMap::from([("/messages/0/content/0/text".into(), vec![0xD800])]);
        let body = json!({"messages":[
            {"role":"user","content":"display"},
            {"role":"user","content":"display"}
        ]});
        assert!(
            map_message_text_overrides(&request, ProtocolFamily::OpenAiChat, &body, &source,)
                .is_err()
        );
    }

    #[test]
    fn canonical_text_override_uses_source_position_when_display_text_collides() {
        use crate::protocol::{
            ChatRequest, ContentBlock, ConversationMessage, MessageRole, ProtocolFamily,
        };

        let mut request = ChatRequest::new("model");
        request.messages.push(ConversationMessage {
            role: MessageRole::User,
            content: vec![
                ContentBlock::Text {
                    text: "�".into(),
                    thought_signature: None,
                    citations: None,
                },
                ContentBlock::Text {
                    text: "�".into(),
                    thought_signature: None,
                    citations: None,
                },
            ],
            native_options: Vec::new(),
        });
        let body = json!({"messages":[{"role":"user","content":[
            {"type":"text","text":"�"},
            {"type":"text","text":"�"}
        ]}]});

        for source_index in [0, 1] {
            let source = BTreeMap::from([(
                format!("/messages/0/content/{source_index}/text"),
                vec![0xD800],
            )]);
            let mapped =
                map_message_text_overrides(&request, ProtocolFamily::OpenAiChat, &body, &source)
                    .unwrap();
            let expected = format!("/messages/0/content/{source_index}/text");
            assert_eq!(mapped.get(&expected), Some(&vec![0xD800]));
            assert_eq!(mapped.len(), 1);
        }
    }

    fn request_with_message(
        role: crate::protocol::MessageRole,
        content: Vec<crate::protocol::ContentBlock>,
    ) -> crate::protocol::ChatRequest {
        let mut request = crate::protocol::ChatRequest::new("text-contributor-model");
        request.messages.push(crate::protocol::ConversationMessage {
            role,
            content,
            native_options: Vec::new(),
        });
        request
    }

    fn actual_codec_body(
        request: &crate::protocol::ChatRequest,
        protocol: crate::protocol::ProtocolFamily,
    ) -> Value {
        use crate::codecs::{CodecContext, EncodeRequest, RequestMode, WireCodec};
        use crate::protocol::ProtocolFamily as P;

        let profile: crate::protocol::ProviderProfile = serde_json::from_value(json!({
            "provider_id": "text-contributor-test",
            "profile_name": "text-contributor-test",
            "base_url": "https://text-contributor.invalid",
            "protocol": protocol,
            "auth": "none",
            "models": [{
                "display_model": "text-contributor-model",
                "request_model": "text-contributor-model",
                "billing_model": "text-contributor-model"
            }]
        }))
        .unwrap();
        let context = CodecContext::new(&profile, "text-contributor-model", RequestMode::Complete);
        let wire = EncodeRequest::new(request);
        let encoded = match protocol {
            P::GeminiGenerateContent => {
                crate::codecs::gemini::GeminiCodec.encode_request(wire, &context)
            }
            P::VertexGemini => crate::hosting::VertexGeminiCodec.encode_request(wire, &context),
            P::AnthropicMessages => {
                crate::codecs::anthropic::AnthropicMessagesCodec.encode_request(wire, &context)
            }
            P::BedrockClaude => crate::hosting::BedrockClaudeCodec.encode_request(wire, &context),
            P::VertexClaude => crate::hosting::VertexClaudeCodec.encode_request(wire, &context),
            P::FoundryClaude => crate::hosting::FoundryClaudeCodec.encode_request(wire, &context),
            _ => panic!("test helper only accepts Gemini and Anthropic native families"),
        }
        .unwrap_or_else(|error| {
            panic!("current {protocol:?} codec rejected test request: {error}")
        });
        serde_json::from_slice(&encoded.body).expect("current codec emits JSON")
    }

    fn serialized_overrides(body: &Value, overrides: &BTreeMap<String, Vec<u16>>) -> String {
        String::from_utf8(serialize(body, overrides, JsonEncoding::JavaScript).unwrap()).unwrap()
    }

    fn typed_text(display: &str) -> crate::protocol::ContentBlock {
        crate::protocol::ContentBlock::Text {
            text: display.into(),
            thought_signature: None,
            citations: None,
        }
    }

    fn thinking_text(display: &str) -> crate::protocol::ContentBlock {
        crate::protocol::ContentBlock::Thinking {
            text: display.into(),
            signature: None,
        }
    }

    fn native_anthropic_text(display: &str, marker: &str) -> crate::protocol::ContentBlock {
        crate::protocol::ContentBlock::ProviderContent {
            protocol: crate::protocol::ProtocolFamily::AnthropicMessages,
            value: json!({
                "type": "text",
                "text": display,
                "provider_metadata": {"raw_marker": marker}
            }),
        }
    }

    #[test]
    fn gemini_thinking_text_collision_maps_typed_text_by_stable_source_order() {
        use crate::protocol::{MessageRole, ProtocolFamily as P};

        for protocol in [P::GeminiGenerateContent, P::VertexGemini] {
            for typed_first in [true, false] {
                let content = if typed_first {
                    vec![typed_text("�"), thinking_text("�")]
                } else {
                    vec![thinking_text("�"), typed_text("�")]
                };
                let typed_index = if typed_first { 0 } else { 1 };
                let thinking_index = if typed_first { 1 } else { 0 };
                let request = request_with_message(MessageRole::Assistant, content);
                let body = actual_codec_body(&request, protocol);

                // Evidence from the live codec path: Thinking is a text-shaped
                // Gemini Part and both parts share the same visible value.
                assert_eq!(body["contents"][0]["role"], "model", "{protocol:?}");
                let parts = body["contents"][0]["parts"].as_array().unwrap();
                assert_eq!(parts.len(), 2, "{protocol:?}");
                assert_eq!(parts[0]["text"], "�", "{protocol:?}");
                assert_eq!(parts[1]["text"], "�", "{protocol:?}");
                assert_eq!(parts[thinking_index]["thought"], true, "{protocol:?}");

                let source = BTreeMap::from([(
                    format!("/messages/0/content/{typed_index}/text"),
                    vec![0xD800],
                )]);
                let expected_pointer = format!("/contents/0/parts/{typed_index}/text");
                let mapped = map_message_text_overrides(&request, protocol, &body, &source)
                    .unwrap_or_else(|error| {
                        panic!("{protocol:?}, typed_first={typed_first}: {error}")
                    });
                assert_eq!(
                    mapped,
                    BTreeMap::from([(expected_pointer.clone(), vec![0xD800])])
                );

                let serialized = serialized_overrides(&body, &mapped);
                assert!(serialized.contains("\\ud800"), "{serialized}");
                assert_eq!(
                    serialized.matches('�').count(),
                    1,
                    "Thinking bytes stay plain: {serialized}"
                );

                // An unrelated same-display target remains ambiguous. The
                // mapper must fail closed instead of selecting the first two.
                let mut extra_target = body.clone();
                extra_target["contents"][0]["parts"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"text":"�"}));
                assert!(
                    map_message_text_overrides(&request, protocol, &extra_target, &source).is_err()
                );
            }
        }
    }

    #[test]
    fn gemini_duplicate_texts_count_thinking_as_a_stable_non_override_contributor() {
        use crate::protocol::{MessageRole, ProtocolFamily as P};

        let request = request_with_message(
            MessageRole::Assistant,
            vec![typed_text("�"), thinking_text("�"), typed_text("�")],
        );
        let source = BTreeMap::from([
            ("/messages/0/content/0/text".into(), vec![0xD800]),
            ("/messages/0/content/2/text".into(), vec![0xD801]),
        ]);

        for protocol in [P::GeminiGenerateContent, P::VertexGemini] {
            let body = actual_codec_body(&request, protocol);
            let parts = body["contents"][0]["parts"].as_array().unwrap();
            assert_eq!(parts.len(), 3, "{protocol:?}");
            assert_eq!(parts[1]["thought"], true, "{protocol:?}");
            let mapped = map_message_text_overrides(&request, protocol, &body, &source)
                .unwrap_or_else(|error| panic!("{protocol:?}: {error}"));
            assert_eq!(mapped.len(), 2, "{protocol:?}");
            assert_eq!(mapped.get("/contents/0/parts/0/text"), Some(&vec![0xD800]));
            assert_eq!(mapped.get("/contents/0/parts/2/text"), Some(&vec![0xD801]));
            assert!(!mapped.contains_key("/contents/0/parts/1/text"));

            let serialized = serialized_overrides(&body, &mapped);
            assert!(serialized.contains("\\ud800"), "{serialized}");
            assert!(serialized.contains("\\ud801"), "{serialized}");
            assert_eq!(serialized.matches('�').count(), 1, "{serialized}");

            let mut extra_target = body.clone();
            extra_target["contents"][0]["parts"]
                .as_array_mut()
                .unwrap()
                .push(json!({"text":"�"}));
            assert!(
                map_message_text_overrides(&request, protocol, &extra_target, &source).is_err()
            );
        }
    }

    #[test]
    fn anthropic_raw_text_provider_content_is_counted_without_receiving_an_override() {
        use crate::protocol::{MessageRole, ProtocolFamily as P};

        for protocol in [
            P::AnthropicMessages,
            P::BedrockClaude,
            P::VertexClaude,
            P::FoundryClaude,
        ] {
            for typed_first in [true, false] {
                let content = if typed_first {
                    vec![typed_text("�"), native_anthropic_text("�", "raw-a")]
                } else {
                    vec![native_anthropic_text("�", "raw-a"), typed_text("�")]
                };
                let typed_index = if typed_first { 0 } else { 1 };
                let request = request_with_message(MessageRole::User, content);
                let body = actual_codec_body(&request, protocol);

                // The active alias codec emits the ProviderContent value
                // unchanged as a normal Anthropic text block.
                let blocks = body["messages"][0]["content"].as_array().unwrap();
                assert_eq!(blocks.len(), 2, "{protocol:?}");
                assert_eq!(blocks[0]["text"], "�", "{protocol:?}");
                assert_eq!(blocks[1]["text"], "�", "{protocol:?}");
                let raw_index = if typed_first { 1 } else { 0 };
                assert_eq!(
                    blocks[raw_index]["provider_metadata"]["raw_marker"], "raw-a",
                    "{protocol:?}"
                );

                let source = BTreeMap::from([(
                    format!("/messages/0/content/{typed_index}/text"),
                    vec![0xD800],
                )]);
                let expected_pointer = format!("/messages/0/content/{typed_index}/text");
                let mapped = map_message_text_overrides(&request, protocol, &body, &source)
                    .unwrap_or_else(|error| {
                        panic!("{protocol:?}, typed_first={typed_first}: {error}")
                    });
                assert_eq!(mapped, BTreeMap::from([(expected_pointer, vec![0xD800])]));

                let serialized = serialized_overrides(&body, &mapped);
                assert!(serialized.contains("\\ud800"), "{serialized}");
                assert_eq!(
                    serialized.matches('�').count(),
                    1,
                    "raw ProviderContent stays intact: {serialized}"
                );

                // A same-display target not attributable to a source block is
                // still rejected for every native Anthropic family alias.
                let mut extra_target = body.clone();
                extra_target["messages"][0]["content"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"type":"text","text":"�","provider_metadata":{"raw_marker":"unattributed"}}));
                assert!(
                    map_message_text_overrides(&request, protocol, &extra_target, &source).is_err()
                );
            }
        }
    }

    #[test]
    fn anthropic_duplicate_raw_and_typed_text_contributors_keep_exact_typed_positions() {
        use crate::protocol::{MessageRole, ProtocolFamily};

        let request = request_with_message(
            MessageRole::User,
            vec![
                native_anthropic_text("�", "raw-a"),
                typed_text("�"),
                native_anthropic_text("�", "raw-b"),
                typed_text("�"),
            ],
        );
        let source = BTreeMap::from([
            ("/messages/0/content/1/text".into(), vec![0xD800]),
            ("/messages/0/content/3/text".into(), vec![0xD801]),
        ]);
        let body = actual_codec_body(&request, ProtocolFamily::AnthropicMessages);
        let blocks = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0]["provider_metadata"]["raw_marker"], "raw-a");
        assert_eq!(blocks[2]["provider_metadata"]["raw_marker"], "raw-b");

        let mapped =
            map_message_text_overrides(&request, ProtocolFamily::AnthropicMessages, &body, &source)
                .unwrap();
        assert_eq!(mapped.len(), 2);
        assert_eq!(
            mapped.get("/messages/0/content/1/text"),
            Some(&vec![0xD800])
        );
        assert_eq!(
            mapped.get("/messages/0/content/3/text"),
            Some(&vec![0xD801])
        );
        assert!(!mapped.contains_key("/messages/0/content/0/text"));
        assert!(!mapped.contains_key("/messages/0/content/2/text"));

        let serialized = serialized_overrides(&body, &mapped);
        assert!(serialized.contains("\\ud800"), "{serialized}");
        assert!(serialized.contains("\\ud801"), "{serialized}");
        assert_eq!(
            serialized.matches('�').count(),
            2,
            "raw ProviderContent values stay intact: {serialized}"
        );
    }
}
