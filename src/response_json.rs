//! Response JSON parsing that can carry JavaScript lone-surrogate text leaves.
//!
//! Provider responses are JSON text. A valid escaped lone UTF-16 unit cannot
//! be represented by `serde_json::Value`'s `String`, so the fallback parser
//! replaces such units in the display tree and retains the original units by
//! JSON pointer. Codecs must consume every retained pointer as a typed text
//! leaf; lone units in keys or unsupported opaque values are rejected.

use crate::protocol::LlmError;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedText {
    pub(crate) text: String,
    pub(crate) utf16_code_units: Option<Vec<u16>>,
}

#[derive(Debug)]
pub(crate) struct ResponseJson {
    pub(crate) value: Value,
    exact_strings: BTreeMap<String, Vec<u16>>,
    exact_keys: BTreeMap<(String, String), Vec<u16>>,
    source: String,
    raw_ranges: BTreeMap<String, (usize, usize)>,
}

impl ResponseJson {
    pub(crate) fn parse(input: &[u8], invalid_message: &str) -> Result<Self, LlmError> {
        // Ordinary text/usage frames keep the existing single-parse path.
        // Only object-valued tool arguments need lexical subtree boundaries.
        if let Ok(value) = serde_json::from_slice::<Value>(input) {
            if !needs_tool_subtree_ranges(&value) {
                return Ok(Self {
                    value,
                    exact_strings: BTreeMap::new(),
                    exact_keys: BTreeMap::new(),
                    source: String::from_utf8(input.to_vec()).map_err(|_| {
                        LlmError::InvalidRequest {
                            message: invalid_message.to_owned(),
                        }
                    })?,
                    raw_ranges: BTreeMap::new(),
                });
            }
        }
        let source = std::str::from_utf8(input).map_err(|_| LlmError::InvalidRequest {
            message: invalid_message.to_owned(),
        })?;
        let parser = ResponseJsonParser::new(source);
        let parsed = parser
            .document()
            .map_err(|message| LlmError::InvalidRequest {
                message: format!("{invalid_message}: {message}"),
            })?;
        // The flat scanner accepts arbitrary nesting without recursive parsing,
        // writing, cloning, or destruction. Current provider codecs still consume
        // `serde_json::Value`, so materializing that tree remains a separate
        // library boundary and may reject valid but deeply nested JSON.
        let value = serde_json::from_slice(&parsed.sanitized).map_err(|error| {
            LlmError::UnsupportedCapability {
                message: format!(
                    "valid response JSON cannot be represented as serde_json::Value: {error}"
                ),
            }
        })?;
        Ok(Self {
            value,
            exact_strings: parsed.exact_strings,
            exact_keys: parsed.exact_keys,
            source: source.to_owned(),
            raw_ranges: parsed.raw_ranges,
        })
    }

    /// Read a recognized provider text field and consume its exact-unit entry.
    pub(crate) fn take_text(
        &mut self,
        pointer: &str,
        display: &str,
    ) -> Result<DecodedText, LlmError> {
        let Some(units) = self.exact_strings.remove(pointer) else {
            return Ok(DecodedText {
                text: display.to_owned(),
                utf16_code_units: None,
            });
        };
        if String::from_utf16_lossy(&units) != display {
            return Err(LlmError::InvalidRequest {
                message: "response text UTF-16 projection did not match its display string".into(),
            });
        }
        Ok(DecodedText {
            text: display.to_owned(),
            utf16_code_units: Some(units),
        })
    }

    /// Consume a tool-owned JSON subtree, preserving its original number,
    /// string and key tokens. The returned display has a standalone namespace.
    pub(crate) fn take_tool_input(
        &mut self,
        pointer: &str,
    ) -> Result<(Value, Option<String>), LlmError> {
        let Some(&(start, end)) = self.raw_ranges.get(pointer) else {
            return Ok((
                self.value.pointer(pointer).cloned().unwrap_or(Value::Null),
                None,
            ));
        };
        let raw = self.source[start..end].to_owned();
        let display = parse_tool_input_json(&raw)?;
        let prefix = format!("{pointer}/");
        self.exact_strings
            .retain(|path, _| path != pointer && !path.starts_with(&prefix));
        self.exact_keys
            .retain(|(path, _), _| path != pointer && !path.starts_with(&prefix));
        Ok((display, Some(raw)))
    }

    /// Reject strings in provider-owned or otherwise unsupported locations.
    pub(crate) fn finish(self) -> Result<Value, LlmError> {
        if self.exact_strings.is_empty() && self.exact_keys.is_empty() {
            Ok(self.value)
        } else {
            Err(LlmError::UnsupportedCapability {
                message:
                    "provider response contains lone UTF-16 text outside a supported text leaf"
                        .into(),
            })
        }
    }
}

fn needs_tool_subtree_ranges(value: &Value) -> bool {
    let mut pending = vec![value];
    while let Some(value) = pending.pop() {
        match value {
            Value::Object(object) => {
                if matches!(
                    object.get("type").and_then(Value::as_str),
                    Some("tool_result" | "function_call_output" | "function_result")
                ) || object.get("role").and_then(Value::as_str) == Some("tool")
                    || object.contains_key("functionResponse")
                    || (object.get("name").is_some_and(Value::is_string)
                        && ["input_schema", "parameters", "parametersJsonSchema"]
                            .iter()
                            .any(|field| object.contains_key(*field)))
                    || (object.get("type").and_then(Value::as_str) == Some("tool_use")
                        && object.contains_key("input"))
                    || (object.get("type").and_then(Value::as_str) == Some("function_call")
                        && object
                            .get("arguments")
                            .is_some_and(|value| !value.is_string()))
                    || (object
                        .get("arguments")
                        .is_some_and(|value| !value.is_string()))
                    || (object.get("name").is_some_and(Value::is_string)
                        && object.contains_key("args"))
                {
                    return true;
                }
                pending.extend(object.values());
            }
            Value::Array(array) => pending.extend(array),
            _ => {}
        }
    }
    false
}

pub(crate) fn combine_text_parts<'a>(
    parts: impl IntoIterator<Item = &'a DecodedText>,
) -> DecodedText {
    let mut text = String::new();
    let mut units: Option<Vec<u16>> = None;
    for part in parts {
        if units.is_none() && part.utf16_code_units.is_some() {
            units = Some(text.encode_utf16().collect());
        }
        if let Some(units) = &mut units {
            match &part.utf16_code_units {
                Some(part_units) => units.extend(part_units),
                None => units.extend(part.text.encode_utf16()),
            }
            text = String::from_utf16_lossy(units);
        } else {
            text.push_str(&part.text);
        }
    }
    DecodedText {
        text,
        utf16_code_units: units,
    }
}

pub(crate) fn content_block(
    text: DecodedText,
    thought_signature: Option<String>,
    citations: Option<Option<Value>>,
) -> crate::protocol::ContentBlock {
    match text.utf16_code_units {
        Some(utf16_code_units) => crate::protocol::ContentBlock::TextJsUtf16 {
            text: text.text,
            utf16_code_units,
            thought_signature,
            citations,
        },
        None => crate::protocol::ContentBlock::Text {
            text: text.text,
            thought_signature,
            citations,
        },
    }
}

pub(crate) fn text_delta(block: usize, text: DecodedText) -> crate::protocol::StreamEvent {
    match text.utf16_code_units {
        Some(utf16_code_units) => crate::protocol::StreamEvent::TextDeltaJsUtf16 {
            block,
            text: text.text,
            utf16_code_units,
        },
        None => crate::protocol::StreamEvent::TextDelta {
            block,
            text: text.text,
        },
    }
}

enum FlatNode {
    Primitive(String),
    String(String),
    Array(Vec<usize>),
    Object(Vec<(String, usize)>),
}

#[derive(Clone, Copy)]
enum FrameState {
    FirstArray,
    ArrayValue,
    ArrayComma,
    FirstObject,
    ObjectKey,
    ObjectColon,
    ObjectValue,
    ObjectComma,
}

struct Frame {
    node: usize,
    state: FrameState,
    key: Option<String>,
    path_component: Option<String>,
}

struct FlatDocument {
    sanitized: Vec<u8>,
    exact_strings: BTreeMap<String, Vec<u16>>,
    exact_keys: BTreeMap<(String, String), Vec<u16>>,
    raw_ranges: BTreeMap<String, (usize, usize)>,
}

struct ResponseJsonParser<'a> {
    source: &'a str,
    offset: usize,
    nodes: Vec<FlatNode>,
    path: Vec<String>,
    exact_strings: BTreeMap<String, Vec<u16>>,
    exact_keys: BTreeMap<(String, String), Vec<u16>>,
    raw_ranges: BTreeMap<String, (usize, usize)>,
    node_starts: Vec<usize>,
    next_key: u64,
}

enum WriteJson<'a> {
    Node(usize),
    Byte(u8),
    String(&'a str),
}

impl<'a> ResponseJsonParser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            offset: 0,
            nodes: Vec::new(),
            path: Vec::new(),
            exact_strings: BTreeMap::new(),
            exact_keys: BTreeMap::new(),
            raw_ranges: BTreeMap::new(),
            node_starts: Vec::new(),
            next_key: 0,
        }
    }

    fn document(mut self) -> Result<FlatDocument, String> {
        let (root, state) = self.value()?;
        let mut stack = Vec::new();
        if let Some(state) = state {
            stack.push(Frame {
                node: root,
                state,
                key: None,
                path_component: None,
            });
        }

        while !stack.is_empty() {
            self.whitespace();
            let frame = stack.last().ok_or("missing JSON parser frame")?;
            let node = frame.node;
            match frame.state {
                FrameState::FirstArray => {
                    if self.consume(b']') {
                        self.pop_frame(&mut stack)?;
                    } else {
                        self.array_value(node, &mut stack)?;
                    }
                }
                FrameState::ArrayValue => self.array_value(node, &mut stack)?,
                FrameState::ArrayComma => {
                    if self.consume(b']') {
                        self.pop_frame(&mut stack)?;
                    } else if self.consume(b',') {
                        stack.last_mut().ok_or("missing JSON parser frame")?.state =
                            FrameState::ArrayValue;
                    } else {
                        return Err("expected a comma or closing bracket in a JSON array".into());
                    }
                }
                FrameState::FirstObject => {
                    if self.consume(b'}') {
                        self.pop_frame(&mut stack)?;
                    } else {
                        self.object_key(&mut stack)?;
                    }
                }
                FrameState::ObjectKey => self.object_key(&mut stack)?,
                FrameState::ObjectColon => {
                    if !self.consume(b':') {
                        return Err("expected a colon after a JSON object key".into());
                    }
                    stack.last_mut().ok_or("missing JSON parser frame")?.state =
                        FrameState::ObjectValue;
                }
                FrameState::ObjectValue => self.object_value(node, &mut stack)?,
                FrameState::ObjectComma => {
                    if self.consume(b'}') {
                        self.pop_frame(&mut stack)?;
                    } else if self.consume(b',') {
                        stack.last_mut().ok_or("missing JSON parser frame")?.state =
                            FrameState::ObjectKey;
                    } else {
                        return Err("expected a comma or closing brace in a JSON object".into());
                    }
                }
            }
        }

        self.whitespace();
        if self.offset != self.source.len() {
            return Err("trailing data after the JSON value".into());
        }
        let sanitized = self.stringify(root)?;
        Ok(FlatDocument {
            sanitized,
            exact_strings: self.exact_strings,
            exact_keys: self.exact_keys,
            raw_ranges: self.raw_ranges,
        })
    }

    fn value(&mut self) -> Result<(usize, Option<FrameState>), String> {
        self.whitespace();
        let start = self.offset;
        let result = self.parse_value()?;
        self.node_starts[result.0] = start;
        if result.1.is_none() {
            self.record_raw(start);
        }
        Ok(result)
    }
    fn record_raw(&mut self, start: usize) {
        let pointer = self.current_pointer();
        if pointer.is_empty()
            || self.path.last().is_some_and(|key| {
                matches!(
                    key.as_str(),
                    "input"
                        | "args"
                        | "arguments"
                        | "content"
                        | "output"
                        | "result"
                        | "error"
                        | "input_schema"
                        | "parameters"
                        | "parametersJsonSchema"
                )
            })
        {
            self.raw_ranges.insert(pointer, (start, self.offset));
        }
    }
    fn parse_value(&mut self) -> Result<(usize, Option<FrameState>), String> {
        self.whitespace();
        match self.peek() {
            Some(b'{') => {
                self.offset += 1;
                Ok((
                    self.push_node(FlatNode::Object(Vec::new())),
                    Some(FrameState::FirstObject),
                ))
            }
            Some(b'[') => {
                self.offset += 1;
                Ok((
                    self.push_node(FlatNode::Array(Vec::new())),
                    Some(FrameState::FirstArray),
                ))
            }
            Some(b'"') => {
                let (text, exact) = self.string(false)?;
                if let Some(units) = exact {
                    self.exact_strings.insert(self.current_pointer(), units);
                }
                Ok((self.push_node(FlatNode::String(text)), None))
            }
            Some(_) => self.primitive(),
            None => Err("unexpected end of JSON".into()),
        }
    }

    fn array_value(&mut self, parent: usize, stack: &mut Vec<Frame>) -> Result<(), String> {
        let index = match self.nodes.get(parent) {
            Some(FlatNode::Array(items)) => items.len(),
            _ => return Err("array frame does not reference an array node".into()),
        };
        let component = index.to_string();
        self.path.push(component.clone());
        let (child, state) = self.value()?;
        match self.nodes.get_mut(parent) {
            Some(FlatNode::Array(items)) => items.push(child),
            _ => return Err("array frame does not reference an array node".into()),
        }
        stack.last_mut().ok_or("missing JSON parser frame")?.state = FrameState::ArrayComma;
        if let Some(state) = state {
            stack.push(Frame {
                node: child,
                state,
                key: None,
                path_component: Some(component),
            });
        } else {
            self.path.pop();
        }
        Ok(())
    }

    fn object_key(&mut self, stack: &mut [Frame]) -> Result<(), String> {
        if self.peek() != Some(b'"') {
            return Err("expected a JSON object key".into());
        }
        let (mut key, exact) = self.string(true)?;
        let parent = self.current_pointer();
        let node = stack.last().ok_or("missing JSON parser frame")?.node;
        if let Some(units) = exact {
            if let Some(((_, old), _)) = self
                .exact_keys
                .iter()
                .find(|((path, _), value)| path == &parent && **value == units)
            {
                key = old.clone();
            } else {
                key = self.fresh_key(node, None);
                self.exact_keys.insert((parent, key.clone()), units);
            }
        } else if self.exact_keys.contains_key(&(parent.clone(), key.clone())) {
            let replacement = self.fresh_key(node, Some(&key));
            let units = self
                .exact_keys
                .remove(&(parent.clone(), key.clone()))
                .unwrap();
            self.exact_keys
                .insert((parent.clone(), replacement.clone()), units);
            if let FlatNode::Object(fields) = &mut self.nodes[node] {
                for (old, _) in fields {
                    if old == &key {
                        *old = replacement.clone();
                    }
                }
            }
            let old = self.pointer_with_child(&key);
            let new = self.pointer_with_child(&replacement);
            self.rebase_sidecars(&old, &new);
        }
        let frame = stack.last_mut().ok_or("missing JSON parser frame")?;
        frame.key = Some(key);
        frame.state = FrameState::ObjectColon;
        Ok(())
    }

    fn object_value(&mut self, parent: usize, stack: &mut Vec<Frame>) -> Result<(), String> {
        let key = stack
            .last_mut()
            .and_then(|frame| frame.key.take())
            .ok_or("object value is missing its key")?;
        let duplicate = match self.nodes.get(parent) {
            Some(FlatNode::Object(fields)) => fields.iter().any(|(existing, _)| existing == &key),
            _ => return Err("object frame does not reference an object node".into()),
        };
        if duplicate {
            let pointer = self.pointer_with_child(&key);
            self.clear_replaced_subtree(&pointer);
        }
        self.path.push(key.clone());
        let (child, state) = self.value()?;
        match self.nodes.get_mut(parent) {
            Some(FlatNode::Object(fields)) => {
                if let Some((_, value)) = fields.iter_mut().find(|(existing, _)| *existing == key) {
                    *value = child;
                } else {
                    fields.push((key.clone(), child));
                }
            }
            _ => return Err("object frame does not reference an object node".into()),
        }
        stack.last_mut().ok_or("missing JSON parser frame")?.state = FrameState::ObjectComma;
        if let Some(state) = state {
            stack.push(Frame {
                node: child,
                state,
                key: None,
                path_component: Some(key),
            });
        } else {
            self.path.pop();
        }
        Ok(())
    }

    fn pop_frame(&mut self, stack: &mut Vec<Frame>) -> Result<(), String> {
        let frame = stack.pop().ok_or("missing JSON parser frame")?;
        self.record_raw(self.node_starts[frame.node]);
        if frame.path_component.is_some() {
            self.path.pop().ok_or("missing JSON path component")?;
        }
        Ok(())
    }

    fn primitive(&mut self) -> Result<(usize, Option<FrameState>), String> {
        let start = self.offset;
        match self.peek() {
            Some(b'n') => self.literal("null")?,
            Some(b't') => self.literal("true")?,
            Some(b'f') => self.literal("false")?,
            Some(b'-' | b'0'..=b'9') => self.number()?,
            _ => return Err("expected a JSON value".into()),
        }
        let primitive = self
            .source
            .get(start..self.offset)
            .ok_or_else(|| "invalid JSON primitive boundary".to_owned())?
            .to_owned();
        Ok((self.push_node(FlatNode::Primitive(primitive)), None))
    }

    fn literal(&mut self, literal: &str) -> Result<(), String> {
        if self
            .source
            .get(self.offset..)
            .is_some_and(|remaining| remaining.starts_with(literal))
        {
            self.offset += literal.len();
            Ok(())
        } else {
            Err("invalid JSON literal".into())
        }
    }

    fn number(&mut self) -> Result<(), String> {
        if self.peek() == Some(b'-') {
            self.offset += 1;
        }
        match self.peek() {
            Some(b'0') => self.offset += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err("invalid JSON number".into()),
        }
        if self.peek() == Some(b'.') {
            self.offset += 1;
            if !self.digits() {
                return Err("invalid JSON number fraction".into());
            }
        }
        if self.peek().is_some_and(|byte| matches!(byte, b'e' | b'E')) {
            self.offset += 1;
            if self.peek().is_some_and(|byte| matches!(byte, b'+' | b'-')) {
                self.offset += 1;
            }
            if !self.digits() {
                return Err("invalid JSON number exponent".into());
            }
        }
        Ok(())
    }

    fn digits(&mut self) -> bool {
        let start = self.offset;
        while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.offset += 1;
        }
        self.offset != start
    }

    fn string(&mut self, is_key: bool) -> Result<(String, Option<Vec<u16>>), String> {
        if !self.consume(b'"') {
            return Err("expected a JSON string".into());
        }
        let mut units = Vec::new();
        loop {
            let byte = self
                .peek()
                .ok_or_else(|| "unterminated JSON string".to_owned())?;
            match byte {
                b'"' => {
                    self.offset += 1;
                    break;
                }
                b'\\' => {
                    self.offset += 1;
                    let escape = self
                        .peek()
                        .ok_or_else(|| "unterminated JSON escape".to_owned())?;
                    self.offset += 1;
                    match escape {
                        b'"' => units.push(u16::from(b'"')),
                        b'\\' => units.push(u16::from(b'\\')),
                        b'/' => units.push(u16::from(b'/')),
                        b'b' => units.push(0x0008),
                        b'f' => units.push(0x000c),
                        b'n' => units.push(0x000a),
                        b'r' => units.push(0x000d),
                        b't' => units.push(0x0009),
                        b'u' => units.push(self.hex_unit()?),
                        _ => return Err("invalid JSON string escape".into()),
                    }
                }
                0x00..=0x1f => return Err("unescaped control character in JSON string".into()),
                _ => {
                    let character = self.source[self.offset..]
                        .chars()
                        .next()
                        .ok_or_else(|| "invalid UTF-8 in JSON string".to_owned())?;
                    let mut buffer = [0u16; 2];
                    units.extend(character.encode_utf16(&mut buffer).iter().copied());
                    self.offset += character.len_utf8();
                }
            }
        }
        if let Ok(display) = String::from_utf16(&units) {
            return Ok((display, None));
        }
        let _ = is_key;
        let display = normalize_lone_units(&units);
        Ok((display, Some(units)))
    }

    fn hex_unit(&mut self) -> Result<u16, String> {
        let end = self.offset.checked_add(4).ok_or("JSON offset overflow")?;
        let digits = self
            .source
            .as_bytes()
            .get(self.offset..end)
            .ok_or_else(|| "short JSON unicode escape".to_owned())?;
        let mut value = 0u16;
        for digit in digits {
            value = value
                .checked_mul(16)
                .and_then(|value| value.checked_add(hex_value(*digit)?))
                .ok_or_else(|| "invalid JSON unicode escape".to_owned())?;
        }
        self.offset = end;
        Ok(value)
    }

    fn push_node(&mut self, node: FlatNode) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        self.node_starts.push(0);
        id
    }

    fn fresh_key(&mut self, node: usize, forbidden: Option<&str>) -> String {
        loop {
            self.next_key += 1;
            let key = format!("__llmClientUtf16KeyV1_{}__", self.next_key);
            if forbidden != Some(key.as_str())
                && !matches!(&self.nodes[node],FlatNode::Object(fields) if fields.iter().any(|(old,_)| old == &key))
            {
                return key;
            }
        }
    }
    fn rebase_sidecars(&mut self, old: &str, new: &str) {
        let rebase = |path: String| {
            if path == old || path.starts_with(&format!("{old}/")) {
                format!("{new}{}", &path[old.len()..])
            } else {
                path
            }
        };
        self.exact_strings = std::mem::take(&mut self.exact_strings)
            .into_iter()
            .map(|(p, v)| (rebase(p), v))
            .collect();
        self.exact_keys = std::mem::take(&mut self.exact_keys)
            .into_iter()
            .map(|((p, k), v)| ((rebase(p), k), v))
            .collect();
        self.raw_ranges = std::mem::take(&mut self.raw_ranges)
            .into_iter()
            .map(|(p, v)| (rebase(p), v))
            .collect();
    }
    fn current_pointer(&self) -> String {
        let mut pointer = String::new();
        for segment in &self.path {
            push_pointer_segment(&mut pointer, segment);
        }
        pointer
    }

    fn pointer_with_child(&self, child: &str) -> String {
        let mut pointer = self.current_pointer();
        push_pointer_segment(&mut pointer, child);
        pointer
    }

    fn clear_replaced_subtree(&mut self, pointer: &str) {
        let prefix = format!("{pointer}/");
        self.exact_keys
            .retain(|(path, _), _| path != pointer && !path.starts_with(&prefix));
        self.raw_ranges
            .retain(|path, _| path != pointer && !path.starts_with(&prefix));
        let descendant_prefix = format!("{pointer}/");
        self.exact_strings
            .retain(|existing, _| existing != pointer && !existing.starts_with(&descendant_prefix));
    }

    fn stringify(&self, root: usize) -> Result<Vec<u8>, String> {
        let mut output = Vec::new();
        let mut stack = vec![WriteJson::Node(root)];
        while let Some(job) = stack.pop() {
            match job {
                WriteJson::Byte(byte) => output.push(byte),
                WriteJson::String(text) => {
                    serde_json::to_writer(&mut output, text).map_err(|error| error.to_string())?
                }
                WriteJson::Node(id) => match self.nodes.get(id).ok_or("invalid JSON node")? {
                    FlatNode::Primitive(raw) => output.extend_from_slice(raw.as_bytes()),
                    FlatNode::String(text) => serde_json::to_writer(&mut output, text)
                        .map_err(|error| error.to_string())?,
                    FlatNode::Array(items) => {
                        output.push(b'[');
                        stack.push(WriteJson::Byte(b']'));
                        for (index, child) in items.iter().copied().enumerate().rev() {
                            stack.push(WriteJson::Node(child));
                            if index != 0 {
                                stack.push(WriteJson::Byte(b','));
                            }
                        }
                    }
                    FlatNode::Object(fields) => {
                        output.push(b'{');
                        stack.push(WriteJson::Byte(b'}'));
                        for (index, (key, child)) in fields.iter().enumerate().rev() {
                            stack.push(WriteJson::Node(*child));
                            stack.push(WriteJson::Byte(b':'));
                            stack.push(WriteJson::String(key));
                            if index != 0 {
                                stack.push(WriteJson::Byte(b','));
                            }
                        }
                    }
                },
            }
        }
        Ok(output)
    }

    fn whitespace(&mut self) {
        while self
            .peek()
            .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
        {
            self.offset += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.source.as_bytes().get(self.offset).copied()
    }

    fn consume(&mut self, byte: u8) -> bool {
        self.whitespace();
        if self.peek() == Some(byte) {
            self.offset += 1;
            true
        } else {
            false
        }
    }
}

fn push_pointer_segment(pointer: &mut String, segment: &str) {
    pointer.push('/');
    pointer.push_str(&segment.replace('~', "~0").replace('/', "~1"));
}

fn normalize_lone_units(units: &[u16]) -> String {
    let mut display = Vec::with_capacity(units.len());
    let mut index = 0;
    while index < units.len() {
        let unit = units[index];
        if (0xd800..=0xdbff).contains(&unit)
            && units
                .get(index + 1)
                .is_some_and(|next| (0xdc00..=0xdfff).contains(next))
        {
            display.extend_from_slice(&[unit, units[index + 1]]);
            index += 2;
        } else if (0xd800..=0xdfff).contains(&unit) {
            display.push(0xfffd);
            index += 1;
        } else {
            display.push(unit);
            index += 1;
        }
    }
    String::from_utf16(&display).expect("normalization emits well-formed UTF-16")
}

fn hex_value(byte: u8) -> Option<u16> {
    match byte {
        b'0'..=b'9' => Some(u16::from(byte - b'0')),
        b'a'..=b'f' => Some(u16::from(byte - b'a' + 10)),
        b'A'..=b'F' => Some(u16::from(byte - b'A' + 10)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INVALID: &str = "response frame is not valid JSON";

    #[test]
    fn recognizes_normal_and_sse_text_leaves_without_private_json_fields() {
        let mut normal = ResponseJson::parse(
            br#"{"content":[{"type":"text","text":"A\ud800B"}]}"#,
            INVALID,
        )
        .unwrap();
        let text = normal.take_text("/content/0/text", "A�B").unwrap();
        assert_eq!(text.utf16_code_units, Some(vec![0x41, 0xd800, 0x42]));
        assert_eq!(normal.finish().unwrap()["content"][0]["text"], "A�B");

        let mut event = ResponseJson::parse(
            br#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"A\udfffB"}}"#,
            INVALID,
        )
        .unwrap();
        let text = event.take_text("/delta/text", "A�B").unwrap();
        assert_eq!(text.utf16_code_units, Some(vec![0x41, 0xdfff, 0x42]));
        assert_eq!(event.finish().unwrap()["delta"]["text"], "A�B");
    }

    #[test]
    fn replacement_and_paired_emoji_remain_plain_rust_strings() {
        for (source, expected) in [
            (
                br#"{"text":"A\ufffdB"}"#.as_slice(),
                vec![0x41, 0xfffd, 0x42],
            ),
            (
                br#"{"text":"A\ud83d\ude00B"}"#.as_slice(),
                vec![0x41, 0xd83d, 0xde00, 0x42],
            ),
        ] {
            let parsed = ResponseJson::parse(source, INVALID).unwrap();
            assert!(parsed.exact_strings.is_empty());
            let text = parsed.value["text"].as_str().unwrap();
            assert_eq!(text.encode_utf16().collect::<Vec<_>>(), expected);
        }
    }

    #[test]
    fn rejects_unpaired_keys_and_unconsumed_opaque_text() {
        assert!(ResponseJson::parse(br#"{"\ud800":"value"}"#, INVALID)
            .unwrap()
            .finish()
            .is_err());
        let parsed = ResponseJson::parse(br#"{"opaque":{"text":"\ud800"}}"#, INVALID).unwrap();
        assert!(matches!(
            parsed.finish(),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }

    fn nested_arrays(depth: usize, leaf: &str) -> String {
        format!("{}{}{}", "[".repeat(depth), leaf, "]".repeat(depth))
    }

    #[test]
    fn fallback_scanner_and_value_materialization_handle_valid_depth_126() {
        let source = nested_arrays(126, r#""\ud800""#);
        let pointer = "/0".repeat(126);
        let mut parsed = ResponseJson::parse(source.as_bytes(), INVALID).unwrap();
        let display = parsed
            .value
            .pointer(&pointer)
            .and_then(Value::as_str)
            .unwrap()
            .to_owned();
        assert_eq!(display, "�");
        let decoded = parsed.take_text(&pointer, &display).unwrap();
        assert_eq!(decoded.utf16_code_units, Some(vec![0xd800]));
        assert!(parsed.finish().is_ok());
    }

    #[test]
    fn flat_fallback_scanner_handles_valid_depth_384_before_value_boundary() {
        let source = nested_arrays(384, r#""\ud800""#);
        let scanned = ResponseJsonParser::new(&source).document().unwrap();
        let pointer = "/0".repeat(384);
        assert_eq!(scanned.exact_strings.get(&pointer), Some(&vec![0xd800]));
        assert!(serde_json::from_slice::<Value>(&scanned.sanitized).is_err());
        assert!(matches!(
            ResponseJson::parse(source.as_bytes(), INVALID),
            Err(LlmError::UnsupportedCapability { message })
                if message.contains("serde_json::Value")
        ));
    }

    #[test]
    fn flat_fallback_scanner_rejects_malformed_deep_and_shallow_controls() {
        let malformed_escape = nested_arrays(384, r#""\uD80Z""#);
        assert!(ResponseJsonParser::new(&malformed_escape)
            .document()
            .is_err());
        let unclosed = format!("{}{}{}", "[".repeat(384), r#""\ud800""#, "]".repeat(383));
        assert!(ResponseJsonParser::new(&unclosed).document().is_err());
        assert!(ResponseJsonParser::new(r#"[true,]"#).document().is_err());
        assert!(ResponseJsonParser::new("[NaN]").document().is_err());
        assert!(ResponseJsonParser::new("[01]").document().is_err());
        assert!(ResponseJsonParser::new(
            r#"{"x":"raw
control"}"#
        )
        .document()
        .is_err());
        assert!(ResponseJson::parse(malformed_escape.as_bytes(), INVALID).is_err());
    }

    #[test]
    fn duplicate_json_keys_clear_stale_utf16_sidecars() {
        for source in [
            br#"{"text":"\ud800","text":"plain"}"#.as_slice(),
            br#"{"a":{"text":"\ud800"},"a":{"text":"plain"}}"#.as_slice(),
        ] {
            let parsed = ResponseJson::parse(source, INVALID).unwrap();
            assert!(parsed.exact_strings.is_empty());
            assert!(parsed.finish().is_ok());
        }
    }
}

/// Validate raw argument JSON and produce a Rust-safe display projection.
/// This consumes exactly one complete JSON value; trailing data is rejected.
pub(crate) fn parse_tool_input_json(raw: &str) -> Result<Value, LlmError> {
    serde_json::from_str::<&serde_json::value::RawValue>(raw).map_err(|_| {
        LlmError::InvalidRequest {
            message: "tool input raw JSON is not one complete JSON value".into(),
        }
    })?;
    Ok(ResponseJson::parse(raw.as_bytes(), "tool input raw JSON is invalid")?.value)
}

pub(crate) fn request_projection(
    input: &[u8],
) -> Result<crate::exact_json::RequestJsonProjection, LlmError> {
    let mut parsed = ResponseJson::parse(input, "request body is not valid JSON")?;
    let candidates: Vec<_> = parsed
        .raw_ranges
        .keys()
        .filter(|pointer| !pointer.is_empty())
        .cloned()
        .collect();
    let mut raw_subtrees = BTreeMap::new();
    for pointer in candidates {
        let Some((parent, field)) = pointer.rsplit_once('/') else {
            continue;
        };
        let Some(object) = parsed.value.pointer(parent) else {
            continue;
        };
        let supported = (field == "input" && object["type"] == "tool_use")
            || (field == "args" && object.get("name").is_some())
            || (field == "arguments"
                && object["type"] == "function_call"
                && !object[field].is_string())
            || (field == "content"
                && (object["type"] == "tool_result" || object["role"] == "tool"))
            || (field == "output" && object["type"] == "function_call_output")
            || (field == "result" && object["type"] == "function_result")
            || (matches!(
                field,
                "input_schema" | "parameters" | "parametersJsonSchema"
            ) && pointer.starts_with("/tools/")
                && object.get("name").is_some())
            || (matches!(field, "result" | "error")
                && parent.ends_with("/functionResponse/response"));
        if !supported
            || raw_subtrees
                .keys()
                .any(|prior: &String| pointer.starts_with(&format!("{prior}/")))
        {
            continue;
        }
        let (value, raw) = parsed.take_tool_input(&pointer)?;
        if let Some(raw) = raw {
            if let Some(slot) = parsed.value.pointer_mut(&pointer) {
                *slot = value;
            }
            raw_subtrees.insert(pointer, raw);
        }
    }
    if !parsed.exact_keys.is_empty() {
        return Err(LlmError::UnsupportedCapability {
            message: "request contains exact object keys outside a supported tool input subtree"
                .into(),
        });
    }
    Ok(crate::exact_json::RequestJsonProjection {
        value: parsed.value,
        string_overrides: parsed.exact_strings,
        raw_subtrees,
    })
}

/// Syntax state belongs to one argument stream, across every SSE fragment.
/// Canonicalizing a literal UTF-16 unit is safe only inside an unescaped string.
#[derive(Debug, Default)]
pub(crate) struct ArgumentJsonDecoder {
    in_string: bool,
    escaped: bool,
    unicode_remaining: u8,
}
impl ArgumentJsonDecoder {
    pub(crate) fn push(&mut self, text: DecodedText) -> Result<String, LlmError> {
        let units = text
            .utf16_code_units
            .unwrap_or_else(|| text.text.encode_utf16().collect());
        let mut output = String::new();
        for decoded in char::decode_utf16(units) {
            let character = decoded.as_ref().ok().copied();
            let malformed = || LlmError::InvalidRequest {
                message: "tool argument JSON contains an invalid string escape".into(),
            };
            if self.unicode_remaining > 0 {
                if !character.is_some_and(|ch| ch.is_ascii_hexdigit()) {
                    return Err(malformed());
                }
                self.unicode_remaining -= 1;
            } else if self.escaped {
                match character {
                    Some('u') => self.unicode_remaining = 4,
                    Some('"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't') => {}
                    _ => return Err(malformed()),
                }
                self.escaped = false;
            } else if self.in_string {
                match character {
                    Some('"') => self.in_string = false,
                    Some('\\') => self.escaped = true,
                    Some(ch) if ch < '\u{20}' => return Err(malformed()),
                    _ => {}
                }
            } else {
                match character {
                    Some('"') => self.in_string = true,
                    Some(_) => {}
                    None => return Err(malformed()),
                }
            }
            match decoded {
                Ok(ch) => output.push(ch),
                Err(error) => output.push_str(&format!("\\u{:04x}", error.unpaired_surrogate())),
            }
        }
        Ok(output)
    }
}

pub(crate) fn argument_fragment(text: DecodedText) -> Result<String, LlmError> {
    ArgumentJsonDecoder::default().push(text)
}

pub(crate) fn tool_input_json_equal(left: &str, right: &str) -> Result<bool, LlmError> {
    let left = ResponseJson::parse(left.as_bytes(), "tool input JSON is invalid")?;
    let right = ResponseJson::parse(right.as_bytes(), "tool input JSON is invalid")?;
    let mut pending = vec![(&left.value, String::new(), &right.value, String::new())];
    let join =
        |path: &str, key: &str| format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"));
    while let Some((a, ap, b, bp)) = pending.pop() {
        match (a, b) {
            (Value::String(at), Value::String(bt)) => {
                let au = left
                    .exact_strings
                    .get(&ap)
                    .cloned()
                    .unwrap_or_else(|| at.encode_utf16().collect());
                let bu = right
                    .exact_strings
                    .get(&bp)
                    .cloned()
                    .unwrap_or_else(|| bt.encode_utf16().collect());
                if au != bu {
                    return Ok(false);
                }
            }
            (Value::Number(a), Value::Number(b)) => {
                if !crate::exact_json::json_numbers_equal(a, b) {
                    return Ok(false);
                }
            }
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Ok(false);
                }
                for (index, (a, b)) in a.iter().zip(b).enumerate() {
                    pending.push((a, format!("{ap}/{index}"), b, format!("{bp}/{index}")));
                }
            }
            (Value::Object(a), Value::Object(b)) => {
                if a.len() != b.len() {
                    return Ok(false);
                }
                let mut candidates: BTreeMap<Vec<u16>, (&Value, String)> = b
                    .iter()
                    .map(|(key, value)| {
                        let units = right
                            .exact_keys
                            .get(&(bp.clone(), key.clone()))
                            .cloned()
                            .unwrap_or_else(|| key.encode_utf16().collect());
                        (units, (value, join(&bp, key)))
                    })
                    .collect();
                for (key, value) in a {
                    let units = left
                        .exact_keys
                        .get(&(ap.clone(), key.clone()))
                        .cloned()
                        .unwrap_or_else(|| key.encode_utf16().collect());
                    let Some((other, other_path)) = candidates.remove(&units) else {
                        return Ok(false);
                    };
                    pending.push((value, join(&ap, key), other, other_path));
                }
            }
            _ => {
                if a != b {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tool_input_tests {
    use super::*;
    #[test]
    fn reconciliation_compares_exact_units_instead_of_lossy_display() {
        assert!(!tool_input_json_equal(r#"{"x":"\ud800"}"#, r#"{"x":"\ud801"}"#).unwrap());
        assert!(!tool_input_json_equal(r#"{"\ud800":1}"#, r#"{"\ud801":1}"#).unwrap());
        assert!(
            tool_input_json_equal(r#"{"\ud800":1,"\ud801":2}"#, r#"{"\ud801":2,"\ud800":1}"#)
                .unwrap()
        );
        assert!(tool_input_json_equal(r#"{"x":1.0}"#, r#"{"x":1}"#).unwrap());
    }
    #[test]
    fn integer_reconciliation_does_not_round_away_source_changes() {
        assert!(
            !tool_input_json_equal(r#"{"n":9007199254740993}"#, r#"{"n":9007199254740992}"#)
                .unwrap()
        );
        assert!(!tool_input_json_equal(
            r#"{"n":18446744073709551615}"#,
            r#"{"n":18446744073709551614}"#
        )
        .unwrap());
        assert!(
            !tool_input_json_equal(r#"{"n":9007199254740993}"#, r#"{"n":9007199254740992.0}"#)
                .unwrap()
        );
    }
    #[test]
    fn argument_escape_state_is_preserved_across_fragments() {
        let exact = |units: Vec<u16>| DecodedText {
            text: String::from_utf16_lossy(&units),
            utf16_code_units: Some(units),
        };
        let mut decoder = ArgumentJsonDecoder::default();
        decoder
            .push(DecodedText {
                text: "{\"x\":\"\\".into(),
                utf16_code_units: None,
            })
            .unwrap();
        assert!(decoder.push(exact(vec![0xd800])).is_err());
        assert!(
            argument_fragment(exact(vec![b'"' as u16, b'\\' as u16, 0xd800, b'"' as u16])).is_err()
        );
        let mut decoder = ArgumentJsonDecoder::default();
        let mut raw = decoder
            .push(DecodedText {
                text: "{\"x\":\"".into(),
                utf16_code_units: None,
            })
            .unwrap();
        raw.push_str(&decoder.push(exact(vec![0xd800])).unwrap());
        raw.push_str(
            &decoder
                .push(DecodedText {
                    text: "\"}".into(),
                    utf16_code_units: None,
                })
                .unwrap(),
        );
        assert_eq!(raw, r#"{"x":"\ud800"}"#);
        assert!(parse_tool_input_json(&raw).is_ok());
        // A split valid escaped backslash does not consume the following unit.
        let mut decoder = ArgumentJsonDecoder::default();
        decoder
            .push(DecodedText {
                text: "\"\\".into(),
                utf16_code_units: None,
            })
            .unwrap();
        assert!(decoder
            .push(exact(vec![b'\\' as u16, 0xd800, b'"' as u16]))
            .is_ok());
        let mut decoder = ArgumentJsonDecoder::default();
        decoder
            .push(DecodedText {
                text: "\"\\uD8".into(),
                utf16_code_units: None,
            })
            .unwrap();
        assert!(decoder.push(exact(vec![0xd800])).is_err());
    }
    #[test]
    fn supported_tool_input_consumes_keys_but_opaque_values_remain_rejected() {
        let mut parsed = ResponseJson::parse(
            br#"{"content":[{"type":"tool_use","input":{"\ud800":"\ud801"}}],"opaque":"\ud802"}"#,
            "invalid",
        )
        .unwrap();
        let (_, raw) = parsed.take_tool_input("/content/0/input").unwrap();
        assert_eq!(raw.as_deref(), Some(r#"{"\ud800":"\ud801"}"#));
        assert!(parsed.finish().is_err());
    }
}
