//! Exact JSON serialization for callers retaining JavaScript UTF-16 text.
use crate::protocol::LlmError;
use serde::{
    ser::{SerializeMap, SerializeSeq},
    Deserialize, Serialize,
};
use serde_json::{Map, Value};
use std::{collections::BTreeMap, io::Write};

type SourceTextIdentity = (String, String);
type SourceTextOccurrence = (usize, usize, Option<Vec<u16>>);
type SourceTextOccurrences = BTreeMap<SourceTextIdentity, Vec<SourceTextOccurrence>>;

/// Canonicalize a validated flat source node using the selected SDK wire's current number,
/// object-key, and UTF-16 string semantics. The scanner owns the only parsed node graph; this
/// writer uses an explicit stack so deeply nested provider input does not recurse.
pub(crate) fn serialize_json_node<'a>(
    root: crate::response_json::JsonNodeRef<'a>,
    encoding: JsonEncoding,
) -> Result<String, LlmError> {
    serialize_json_node_with_fields(root, encoding, &[])
}

/// Serialize a flat JSON object while replacing/appending selected shallow fields.
/// The replacement source values must already be validated JSON strings.
pub(crate) fn serialize_json_node_with_fields<'a>(
    root: crate::response_json::JsonNodeRef<'a>,
    encoding: JsonEncoding,
    fields: &[(&str, &str)],
) -> Result<String, LlmError> {
    serialize_json_node_with_policy(root, encoding, fields, &BTreeMap::new(), false)
}

fn serialize_json_node_with_policy<'a>(
    root: crate::response_json::JsonNodeRef<'a>,
    encoding: JsonEncoding,
    fields: &[(&str, &str)],
    string_overrides: &BTreeMap<usize, Vec<u16>>,
    normalize_string_values: bool,
) -> Result<String, LlmError> {
    if !fields.is_empty() && root.kind() != crate::response_json::FlatJsonKind::Object {
        return Err(LlmError::InvalidRequest {
            message: "JSON field overlays require an object payload".into(),
        });
    }
    enum Work<'a> {
        Node(crate::response_json::JsonNodeRef<'a>),
        Byte(u8),
        String(Vec<u16>),
        Raw(Vec<u8>),
    }

    let mut overrides = fields
        .iter()
        .map(|(key, source)| (String::from(*key), source.as_bytes().to_vec()))
        .collect::<Vec<_>>();
    let mut out = Vec::new();
    let mut stack = vec![Work::Node(root)];
    while let Some(work) = stack.pop() {
        match work {
            Work::Byte(byte) => out.push(byte),
            Work::String(units) => write_json_string_from_utf16(&mut out, &units),
            Work::Raw(source) => out.extend_from_slice(&source),
            Work::Node(node) => match node.kind() {
                crate::response_json::FlatJsonKind::Null => out.extend_from_slice(b"null"),
                crate::response_json::FlatJsonKind::Boolean => {
                    let primitive =
                        node.primitive_source()
                            .ok_or_else(|| LlmError::InvalidRequest {
                                message: "flat JSON boolean has no source token".into(),
                            })?;
                    out.extend_from_slice(primitive.as_bytes());
                }
                crate::response_json::FlatJsonKind::Number => {
                    let source = node
                        .number_source()
                        .ok_or_else(|| LlmError::InvalidRequest {
                            message: "flat JSON number has no source token".into(),
                        })?;
                    match encoding {
                        JsonEncoding::Serde => {
                            let number: serde_json::Number = serde_json::from_str(source).map_err(
                                |error| LlmError::InvalidRequest {
                                    message: format!("JSON number cannot be represented by the SDK codec: {error}"),
                                },
                            )?;
                            serde_json::to_writer(&mut out, &number).map_err(|error| {
                                LlmError::InvalidRequest {
                                    message: format!("failed to encode JSON number: {error}"),
                                }
                            })?;
                        }
                        JsonEncoding::JavaScript => {
                            let number = source.parse::<f64>().map_err(|error| {
                                LlmError::InvalidRequest {
                                    message: format!("JSON number cannot be converted to a JavaScript number: {error}"),
                                }
                            })?;
                            if number.is_finite() {
                                out.extend_from_slice(javascript_number(number).as_bytes());
                            } else {
                                // JSON.parse accepts an overflowing exponent as Infinity; native
                                // JSON.stringify writes non-finite numbers as null.
                                out.extend_from_slice(b"null");
                            }
                        }
                    }
                }
                crate::response_json::FlatJsonKind::String => {
                    let units = node
                        .string_units()
                        .ok_or_else(|| LlmError::InvalidRequest {
                            message: "flat JSON string has no decoded UTF-16 units".into(),
                        })?;
                    let units = string_overrides.get(&node.identity()).unwrap_or(&units);
                    if normalize_string_values {
                        write_json_string_from_utf16(&mut out, &to_well_formed_utf16(units));
                    } else {
                        write_json_string_from_utf16(&mut out, units);
                    }
                }
                crate::response_json::FlatJsonKind::Array => {
                    let len = node.array_len().ok_or_else(|| LlmError::InvalidRequest {
                        message: "flat JSON array has no child list".into(),
                    })?;
                    stack.push(Work::Byte(b']'));
                    for index in (0..len).rev() {
                        let child =
                            node.array_item(index)
                                .ok_or_else(|| LlmError::InvalidRequest {
                                    message: "flat JSON array child index disappeared".into(),
                                })?;
                        stack.push(Work::Node(child));
                        if index > 0 {
                            stack.push(Work::Byte(b','));
                        }
                    }
                    stack.push(Work::Byte(b'['));
                }
                crate::response_json::FlatJsonKind::Object => {
                    let scanned_entries =
                        node.object_entries()
                            .ok_or_else(|| LlmError::InvalidRequest {
                                message: "flat JSON object has no child list".into(),
                            })?;
                    let original_len = scanned_entries.len();
                    let mut entries = scanned_entries
                        .into_iter()
                        .enumerate()
                        .map(|(position, entry)| {
                            let exact_key = String::from_utf16(entry.key_units).ok();
                            let replacement = exact_key.as_deref().and_then(|key| {
                                overrides
                                    .iter()
                                    .position(|(candidate, _)| candidate == key)
                                    .map(|index| {
                                        let (_, source) = overrides.remove(index);
                                        source
                                    })
                            });
                            (
                                position,
                                entry.key_units.to_vec(),
                                Some(entry.value),
                                replacement,
                            )
                        })
                        .collect::<Vec<_>>();
                    for (position, (key, source)) in overrides.drain(..).enumerate() {
                        entries.push((
                            original_len + position,
                            key.encode_utf16().collect(),
                            // Appended field has no scanner node. It is emitted as a raw token.
                            None,
                            Some(source),
                        ));
                    }
                    if encoding == JsonEncoding::JavaScript {
                        entries.sort_by_key(|(position, key, _, _)| {
                            javascript_array_index(key)
                                .map(|index| (0_u8, index, *position))
                                .unwrap_or((1_u8, 0, *position))
                        });
                    }
                    stack.push(Work::Byte(b'}'));
                    for (index, (_, key_units, value, replacement)) in
                        entries.into_iter().enumerate().rev()
                    {
                        if let Some(source) = replacement {
                            stack.push(Work::Raw(source));
                        } else {
                            stack.push(Work::Node(value.ok_or_else(|| {
                                LlmError::InvalidRequest {
                                    message: "raw JSON overlay lost its source value".into(),
                                }
                            })?));
                        }
                        stack.push(Work::Byte(b':'));
                        stack.push(Work::String(key_units));
                        if index > 0 {
                            stack.push(Work::Byte(b','));
                        }
                    }
                    stack.push(Work::Byte(b'{'));
                }
            },
        }
    }
    String::from_utf8(out).map_err(|error| LlmError::InvalidRequest {
        message: format!("canonical JSON output is not UTF-8: {error}"),
    })
}

pub(crate) fn javascript_array_index(key: &[u16]) -> Option<u32> {
    if key.is_empty() || key.len() > 10 || (key.len() > 1 && key[0] == u16::from(b'0')) {
        return None;
    }
    let mut index = 0_u64;
    for unit in key {
        if !(u16::from(b'0')..=u16::from(b'9')).contains(unit) {
            return None;
        }
        index = index
            .checked_mul(10)?
            .checked_add(u64::from(*unit - u16::from(b'0')))?;
    }
    (index < u64::from(u32::MAX)).then_some(index as u32)
}

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

/// Serialize a source-backed request, replacing selected string value leaves.
/// Provider source stays flat and exact through draft/host modifications. Native
/// request sanitation belongs to the final dispatch boundary, before signing.
pub fn serialize_json_payload_with_overrides(
    value: &crate::protocol::JsonPayload,
    overrides: &BTreeMap<String, Vec<u16>>,
    encoding: JsonEncoding,
) -> Result<Vec<u8>, LlmError> {
    let scanner = value.scanner()?;
    let mut by_node = BTreeMap::new();
    for (pointer, units) in overrides {
        let node = scanner
            .node_ref_at(pointer)
            .filter(|node| node.kind() == crate::response_json::FlatJsonKind::String)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: format!("UTF-16 JSON override does not target a string leaf: {pointer}"),
            })?;
        by_node.insert(node.identity(), units.clone());
    }
    serialize_json_node_with_policy(scanner.root_node(), encoding, &[], &by_node, false)
        .map(String::into_bytes)
}

/// Native VNe walks values in the assembled generation request. It ignores
/// object keys and leaves drafts and non-Claude provider protocols unchanged.
/// None means the native detector found no malformed string value, so the
/// current body byte image can remain untouched.
pub(crate) fn normalize_native_generation_json_body(
    body: &[u8],
    protocol: crate::protocol::ProtocolFamily,
) -> Result<Option<Vec<u8>>, LlmError> {
    use crate::protocol::ProtocolFamily;
    use crate::response_json::FlatJsonKind;
    if !matches!(
        protocol,
        ProtocolFamily::AnthropicMessages
            | ProtocolFamily::FoundryClaude
            | ProtocolFamily::BedrockClaude
            | ProtocolFamily::VertexClaude
    ) {
        return Ok(None);
    }
    let scanner = crate::response_json::ResponseJsonScanner::parse(
        body,
        "final request body is invalid JSON",
    )?;
    let root = scanner.root_node();
    if !matches!(root.kind(), FlatJsonKind::Array | FlatJsonKind::Object) {
        return Ok(None);
    }
    let mut pending = vec![root];
    let mut malformed = false;
    while let Some(node) = pending.pop() {
        match node.kind() {
            FlatJsonKind::String => {
                let units = node
                    .string_units()
                    .ok_or_else(|| LlmError::InvalidRequest {
                        message: "flat JSON string has no decoded UTF-16 units".into(),
                    })?;
                if String::from_utf16(&units).is_err() {
                    malformed = true;
                    break;
                }
            }
            FlatJsonKind::Array => {
                for index in 0..node.array_len().unwrap_or(0) {
                    if let Some(child) = node.array_item(index) {
                        pending.push(child);
                    }
                }
            }
            FlatJsonKind::Object => {
                for entry in node.object_entries().unwrap_or_default() {
                    pending.push(entry.value);
                }
            }
            _ => {}
        }
    }
    if !malformed {
        return Ok(None);
    }
    serialize_json_node_with_policy(root, JsonEncoding::JavaScript, &[], &BTreeMap::new(), true)
        .map(|source| Some(source.into_bytes()))
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
    body: &crate::protocol::JsonPayload,
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
    let mut source_texts: SourceTextOccurrences = BTreeMap::new();
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

    let body_scanner = body.scanner()?;
    let targets = encoded_text_fields(&body_scanner, family);
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
            || body_scanner.kind_at(&pointer) != Some(crate::response_json::FlatJsonKind::String)
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
) -> Option<String> {
    use crate::protocol::{ContentBlock, ProtocolFamily};

    match block {
        ContentBlock::Text { text, .. } | ContentBlock::TextJsUtf16 { text, .. } => {
            Some(text.clone())
        }
        ContentBlock::Thinking { text, .. } if family == ProtocolFamily::GeminiGenerateContent => {
            Some(text.clone())
        }
        ContentBlock::ProviderContent { protocol, value }
            if family == ProtocolFamily::AnthropicMessages
                && *protocol == ProtocolFamily::AnthropicMessages
                && value.string_at("/type").as_deref() == Some("text") =>
        {
            value.string_at("/text")
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
    body: &crate::response_json::ResponseJsonScanner,
    family: crate::protocol::ProtocolFamily,
) -> Vec<(String, String, String)> {
    use crate::response_json::FlatJsonKind;

    let mut fields = Vec::new();
    let mut add = |role: &str, pointer: String| {
        if let Some(text) = body.string_at(&pointer) {
            fields.push((role.to_owned(), text.to_owned(), pointer));
        }
    };
    match family {
        crate::protocol::ProtocolFamily::AnthropicMessages => {
            for message_index in 0..body.array_len("/messages").unwrap_or(0) {
                let message_pointer = format!("/messages/{message_index}");
                let role = body
                    .string_at(&format!("{message_pointer}/role"))
                    .unwrap_or_default();
                for block_index in 0..body
                    .array_len(&format!("{message_pointer}/content"))
                    .unwrap_or(0)
                {
                    let block_pointer = format!("{message_pointer}/content/{block_index}");
                    if body.string_at(&format!("{block_pointer}/type")) == Some("text") {
                        let text_pointer = format!("{block_pointer}/text");
                        add(role, text_pointer);
                    }
                }
            }
        }
        crate::protocol::ProtocolFamily::OpenAiChat => {
            for message_index in 0..body.array_len("/messages").unwrap_or(0) {
                let message_pointer = format!("/messages/{message_index}");
                let role = body
                    .string_at(&format!("{message_pointer}/role"))
                    .unwrap_or_default();
                let content_pointer = format!("{message_pointer}/content");
                match body.kind_at(&content_pointer) {
                    Some(FlatJsonKind::String) => add(role, content_pointer),
                    Some(FlatJsonKind::Array) => {
                        for part_index in 0..body.array_len(&content_pointer).unwrap_or(0) {
                            let part_pointer = format!("{content_pointer}/{part_index}");
                            if body.string_at(&format!("{part_pointer}/type")) == Some("text") {
                                let text_pointer = format!("{part_pointer}/text");
                                add(role, text_pointer);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        crate::protocol::ProtocolFamily::OpenAiResponses => {
            for message_index in 0..body.array_len("/input").unwrap_or(0) {
                let message_pointer = format!("/input/{message_index}");
                if body.string_at(&format!("{message_pointer}/type")) != Some("message") {
                    continue;
                }
                let role = body
                    .string_at(&format!("{message_pointer}/role"))
                    .unwrap_or_default();
                let content_pointer = format!("{message_pointer}/content");
                for part_index in 0..body.array_len(&content_pointer).unwrap_or(0) {
                    let part_pointer = format!("{content_pointer}/{part_index}");
                    if matches!(
                        body.string_at(&format!("{part_pointer}/type")),
                        Some("input_text" | "output_text")
                    ) {
                        let text_pointer = format!("{part_pointer}/text");
                        add(role, text_pointer);
                    }
                }
            }
        }
        crate::protocol::ProtocolFamily::GeminiGenerateContent => {
            for message_index in 0..body.array_len("/contents").unwrap_or(0) {
                let message_pointer = format!("/contents/{message_index}");
                let role = body
                    .string_at(&format!("{message_pointer}/role"))
                    .unwrap_or_default();
                let parts_pointer = format!("{message_pointer}/parts");
                for part_index in 0..body.array_len(&parts_pointer).unwrap_or(0) {
                    let text_pointer = format!("{parts_pointer}/{part_index}/text");
                    add(role, text_pointer);
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

    fn map_message_text_overrides_from_fixture(
        request: &crate::protocol::ChatRequest,
        protocol: crate::protocol::ProtocolFamily,
        body: &Value,
        overrides: &BTreeMap<String, Vec<u16>>,
    ) -> Result<BTreeMap<String, Vec<u16>>, LlmError> {
        let body = crate::protocol::JsonPayload::from_generated_value(body)
            .expect("bounded JSON test fixture converts to JsonPayload");
        super::map_message_text_overrides(request, protocol, &body, overrides)
    }

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
    fn native_final_body_normalizes_values_after_source_projection_and_keeps_keys() {
        use crate::protocol::{JsonPayload, ProtocolFamily};
        let body = JsonPayload::parse(r#"{"system":[{"type":"text","text":"\ud800Ω\udc00"}],"messages":[{"role":"user","content":[{"type":"text","text":"display"}]}],"\ud800":{"nested":["x\udc00","\ud83d\ude03",null]},"10":10,"2":2}"#).unwrap();
        let overrides = BTreeMap::from([(
            "/messages/0/content/0/text".into(),
            vec![0xd800, b'x' as u16, 0xd83d, 0xde03, 0xdc00],
        )]);
        let draft =
            serialize_json_payload_with_overrides(&body, &overrides, JsonEncoding::JavaScript)
                .unwrap();
        assert!(String::from_utf8(draft.clone())
            .unwrap()
            .contains("\\ud800Ω\\udc00"));
        for protocol in [
            ProtocolFamily::AnthropicMessages,
            ProtocolFamily::FoundryClaude,
            ProtocolFamily::BedrockClaude,
            ProtocolFamily::VertexClaude,
        ] {
            let wire = normalize_native_generation_json_body(&draft, protocol)
                .unwrap()
                .unwrap();
            assert_eq!(
                String::from_utf8(wire).unwrap(),
                r#"{"2":2,"10":10,"system":[{"type":"text","text":"�Ω�"}],"messages":[{"role":"user","content":[{"type":"text","text":"�x😃�"}]}],"\ud800":{"nested":["x�","😃",null]}}"#
            );
        }
        assert!(
            normalize_native_generation_json_body(&draft, ProtocolFamily::OpenAiChat)
                .unwrap()
                .is_none()
        );
        assert!(body.as_json_source().contains("\\ud800Ω\\udc00"));
    }

    #[test]
    fn native_final_body_normalizes_tool_result_text_after_anthropic_encoding() {
        use crate::codecs::{CodecContext, EncodeRequest, RequestMode, WireCodec};
        use crate::protocol::{
            ChatRequest, ContentBlock, ConversationMessage, JsonPayload, MessageRole,
            ProtocolFamily, ProviderProfile, ToolUseId,
        };

        let mut request = ChatRequest::new("text-contributor-model");
        request.messages.push(ConversationMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::ToolResult {
                cache_reference: None,
                tool_use_id: ToolUseId::new("toolu_utf16"),
                content: JsonPayload::string_from_utf16_code_units(&[0xD800]),
                is_error: Some(false),
                blocks: None,
                toolset_name: None,
            }],
            native_options: Vec::new(),
        });
        let profile: ProviderProfile = serde_json::from_value(json!({
            "provider_id": "tool-result-normalization-test",
            "profile_name": "tool-result-normalization-test",
            "base_url": "https://tool-result-normalization.invalid",
            "protocol": "anthropic_messages",
            "auth": "none",
            "models": [{
                "display_model": "text-contributor-model",
                "request_model": "text-contributor-model",
                "billing_model": "text-contributor-model", "metadata":{"applicationOutputTokens":4096}
            }]
        }))
        .unwrap();
        let context = CodecContext::new(&profile, "text-contributor-model", RequestMode::Complete);
        let draft = crate::codecs::anthropic::AnthropicMessagesCodec
            .encode_request(EncodeRequest::new(&request), &context)
            .unwrap();
        let draft_source = std::str::from_utf8(&draft.body).unwrap();
        assert!(draft_source.contains(r#""content":"\ud800""#));
        assert_eq!(
            JsonPayload::parse(draft_source)
                .unwrap()
                .string_units_at("/messages/0/content/0/content"),
            Some(vec![0xD800])
        );

        let wire =
            normalize_native_generation_json_body(&draft.body, ProtocolFamily::AnthropicMessages)
                .unwrap()
                .expect("native request sealing normalizes malformed string values");
        let wire_source = std::str::from_utf8(&wire).unwrap();
        assert!(wire_source.contains("\"content\":\"�\""));
        assert_eq!(
            JsonPayload::parse(wire_source)
                .unwrap()
                .string_units_at("/messages/0/content/0/content"),
            Some(vec![0xFFFD])
        );
    }

    #[test]
    fn native_final_detector_ignores_lone_keys_and_preserves_well_formed_body_bytes() {
        use crate::protocol::ProtocolFamily;
        let body = br#"{ "\ud800" : ["\ud83d\ude03", -0, 1.00] }"#;
        assert!(
            normalize_native_generation_json_body(body, ProtocolFamily::AnthropicMessages)
                .unwrap()
                .is_none()
        );
        assert!(normalize_native_generation_json_body(
            br#""\ud800""#,
            ProtocolFamily::AnthropicMessages
        )
        .unwrap()
        .is_none());
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
                text: "�".into(),
                thought_signature: None,
                citations: None,
            }],
            native_options: Vec::new(),
        });
        let source = BTreeMap::from([("/messages/0/content/0/text".into(), vec![0xD800])]);
        let cases = [
            (
                P::AnthropicMessages,
                json!({"messages":[{"role":"user","content":[{"type":"text","text":"�"}]}]}),
                "/messages/0/content/0/text",
            ),
            (
                P::OpenAiChat,
                json!({"messages":[{"role":"user","content":"�"}]}),
                "/messages/0/content",
            ),
            (
                P::OpenAiResponses,
                json!({"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"�"}]}]}),
                "/input/0/content/0/text",
            ),
            (
                P::GeminiGenerateContent,
                json!({"contents":[{"role":"user","parts":[{"text":"�"}]}]}),
                "/contents/0/parts/0/text",
            ),
            (
                P::VertexClaude,
                json!({"messages":[{"role":"user","content":[{"type":"text","text":"�"}]}]}),
                "/messages/0/content/0/text",
            ),
            (
                P::VertexGemini,
                json!({"contents":[{"role":"user","parts":[{"text":"�"}]}]}),
                "/contents/0/parts/0/text",
            ),
            (
                P::BedrockClaude,
                json!({"messages":[{"role":"user","content":[{"type":"text","text":"�"}]}]}),
                "/messages/0/content/0/text",
            ),
            (
                P::AzureOpenAi,
                json!({"messages":[{"role":"user","content":"�"}]}),
                "/messages/0/content",
            ),
            (
                P::FoundryClaude,
                json!({"messages":[{"role":"user","content":[{"type":"text","text":"�"}]}]}),
                "/messages/0/content/0/text",
            ),
        ];
        for (protocol, body, expected) in cases {
            let mapped =
                map_message_text_overrides_from_fixture(&request, protocol, &body, &source)
                    .unwrap_or_else(|error| panic!("{protocol:?}: {error}"));
            assert_eq!(mapped.get(expected), Some(&vec![0xD800]), "{protocol:?}");
        }
    }

    #[test]
    fn canonical_text_override_rejects_mismatched_display_text() {
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
        let overrides = BTreeMap::from([("/messages/0/content/0/text".into(), vec![0xD800])]);
        let body =
            json!({"messages":[{"role":"user","content":[{"type":"text","text":"display"}]}]});
        assert!(matches!(
            map_message_text_overrides_from_fixture(&request, ProtocolFamily::AnthropicMessages, &body, &overrides),
            Err(LlmError::InvalidRequest { message }) if message == "UTF-16 block units do not match the text display string"
        ));
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
        assert!(map_message_text_overrides_from_fixture(
            &request,
            ProtocolFamily::OpenAiChat,
            &body,
            &source,
        )
        .is_err());
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
            let mapped = map_message_text_overrides_from_fixture(
                &request,
                ProtocolFamily::OpenAiChat,
                &body,
                &source,
            )
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
                "billing_model": "text-contributor-model", "metadata":{"applicationOutputTokens":4096}
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
            value: crate::protocol::JsonPayload::parse(
                &json!({
                    "type": "text",
                    "text": display,
                    "provider_metadata": {"raw_marker": marker}
                })
                .to_string(),
            )
            .unwrap(),
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
                let mapped =
                    map_message_text_overrides_from_fixture(&request, protocol, &body, &source)
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
                assert!(map_message_text_overrides_from_fixture(
                    &request,
                    protocol,
                    &extra_target,
                    &source
                )
                .is_err());
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
            let mapped =
                map_message_text_overrides_from_fixture(&request, protocol, &body, &source)
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
            assert!(map_message_text_overrides_from_fixture(
                &request,
                protocol,
                &extra_target,
                &source
            )
            .is_err());
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
                let mapped =
                    map_message_text_overrides_from_fixture(&request, protocol, &body, &source)
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
                assert!(map_message_text_overrides_from_fixture(
                    &request,
                    protocol,
                    &extra_target,
                    &source
                )
                .is_err());
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

        let mapped = map_message_text_overrides_from_fixture(
            &request,
            ProtocolFamily::AnthropicMessages,
            &body,
            &source,
        )
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
