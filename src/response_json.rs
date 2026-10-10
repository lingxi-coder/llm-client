//! Response JSON parsing with source spans and JavaScript lone-surrogate text.
//!
//! Provider responses are JSON text. A valid escaped lone UTF-16 unit cannot
//! be represented by `serde_json::Value`'s `String`, so the fallback parser
//! replaces such units in the display tree and retains the original units by
//! JSON pointer. Codecs consume typed text leaves or hand opaque raw subtrees to
//! a typed source carrier; exact keys and unsupported leaves remain rejected by
//! `finish` unless that subtree was handed off.

use crate::protocol::{JsonPayload, LlmError, ProviderErrorText};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedText {
    pub(crate) text: String,
    pub(crate) utf16_code_units: Option<Vec<u16>>,
}

/// Validate string escapes across argument deltas without rewriting their source units.
/// Complete JSON grammar is checked when the assembled payload is consumed.
#[derive(Debug, Default)]
pub(crate) struct ArgumentJsonDecoder {
    in_string: bool,
    escaped: bool,
    unicode_remaining: u8,
}

impl ArgumentJsonDecoder {
    pub(crate) fn push(
        &mut self,
        text: DecodedText,
    ) -> Result<crate::protocol::JsonPayloadFragment, LlmError> {
        let fragment = crate::protocol::JsonPayloadFragment::from_decoded_text(
            &text.text,
            text.utf16_code_units,
        )
        .map_err(|message| LlmError::InvalidRequest {
            message: message.into(),
        })?;
        for decoded in char::decode_utf16(fragment.utf16_code_units().iter().copied()) {
            let character = decoded.ok();
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
        }
        Ok(fragment)
    }
}

#[derive(Debug)]
pub(crate) struct ResponseJson {
    pub(crate) value: Value,
    source: String,
    source_spans: BTreeMap<String, Range<usize>>,
    exact_strings: BTreeMap<String, Vec<u16>>,
    exact_keys: BTreeMap<String, Vec<u16>>,
    opaque_paths: BTreeSet<String>,
}

/// JSON value kind exposed by the flat provider-response scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlatJsonKind {
    Null,
    Boolean,
    Number,
    String,
    Array,
    Object,
}

/// Borrowed handle to one node in the validated flat document. It exposes no
/// recursive Rust value and can therefore be walked with an explicit stack.
#[derive(Clone, Copy)]
pub(crate) struct JsonNodeRef<'a> {
    document: &'a FlatDocument,
    id: usize,
}

pub(crate) struct JsonObjectEntryRef<'a> {
    #[cfg(test)]
    pub(crate) key: &'a str,
    pub(crate) key_units: &'a [u16],
    pub(crate) value: JsonNodeRef<'a>,
}

impl<'a> JsonNodeRef<'a> {
    fn node(&self) -> &'a FlatNode {
        &self.document.nodes[self.id]
    }

    pub(crate) fn identity(&self) -> usize {
        self.id
    }

    pub(crate) fn kind(&self) -> FlatJsonKind {
        match self.node() {
            FlatNode::Primitive(raw) if raw == "null" => FlatJsonKind::Null,
            FlatNode::Primitive(raw) if raw == "true" || raw == "false" => FlatJsonKind::Boolean,
            FlatNode::Primitive(_) => FlatJsonKind::Number,
            FlatNode::String(_) => FlatJsonKind::String,
            FlatNode::Array(_) => FlatJsonKind::Array,
            FlatNode::Object(_) => FlatJsonKind::Object,
        }
    }

    pub(crate) fn number_source(&self) -> Option<&'a str> {
        match self.node() {
            FlatNode::Primitive(raw) if self.kind() == FlatJsonKind::Number => Some(raw),
            _ => None,
        }
    }

    pub(crate) fn primitive_source(&self) -> Option<&'a str> {
        match self.node() {
            FlatNode::Primitive(raw) => Some(raw),
            _ => None,
        }
    }

    pub(crate) fn string_at(&self) -> Option<&'a str> {
        match self.node() {
            FlatNode::String(text) => Some(text),
            _ => None,
        }
    }

    pub(crate) fn string_units(&self) -> Option<Vec<u16>> {
        let text = self.string_at()?;
        Some(
            self.document
                .node_exact_strings
                .get(&self.id)
                .cloned()
                .unwrap_or_else(|| text.encode_utf16().collect()),
        )
    }

    pub(crate) fn array_len(&self) -> Option<usize> {
        match self.node() {
            FlatNode::Array(items) => Some(items.len()),
            _ => None,
        }
    }

    pub(crate) fn array_item(&self, index: usize) -> Option<JsonNodeRef<'a>> {
        let FlatNode::Array(items) = self.node() else {
            return None;
        };
        Some(JsonNodeRef {
            document: self.document,
            id: *items.get(index)?,
        })
    }

    pub(crate) fn object_entries(&self) -> Option<Vec<JsonObjectEntryRef<'a>>> {
        let FlatNode::Object(fields) = self.node() else {
            return None;
        };
        Some(
            fields
                .iter()
                .map(|(_key, key_units, id)| JsonObjectEntryRef {
                    #[cfg(test)]
                    key: _key,
                    key_units,
                    value: JsonNodeRef {
                        document: self.document,
                        id: *id,
                    },
                })
                .collect(),
        )
    }
}

/// Iterative provider JSON scan used to select opaque source-backed leaves
/// before constructing the bounded metadata `Value` view.
pub(crate) struct ResponseJsonScanner {
    source: String,
    document: FlatDocument,
}

impl ResponseJsonScanner {
    pub(crate) fn parse(input: &[u8], invalid_message: &str) -> Result<Self, LlmError> {
        let source = std::str::from_utf8(input)
            .map_err(|_| LlmError::InvalidRequest {
                message: invalid_message.to_owned(),
            })?
            .to_owned();
        let document = ResponseJsonParser::new(&source)
            .document()
            .map_err(|message| LlmError::InvalidRequest {
                message: format!("{invalid_message}: {message}"),
            })?;
        Ok(Self { source, document })
    }

    fn node_id_at(&self, pointer: &str) -> Option<usize> {
        if self.document.exact_keys.is_empty() {
            return self.document.pointer_nodes.get(pointer).copied();
        }
        if pointer.is_empty() {
            return Some(self.document.root);
        }
        let mut id = self.document.root;
        for segment in pointer.strip_prefix('/')?.split('/') {
            let segment = segment.replace("~1", "/").replace("~0", "~");
            id = match &self.document.nodes[id] {
                FlatNode::Object(fields) => {
                    let units = segment.encode_utf16().collect::<Vec<_>>();
                    fields.iter().find(|(_, key, _)| *key == units)?.2
                }
                FlatNode::Array(items) => {
                    if segment.is_empty()
                        || segment.starts_with('+')
                        || (segment.len() > 1 && segment.starts_with('0'))
                    {
                        return None;
                    }
                    *items.get(segment.parse::<usize>().ok()?)?
                }
                _ => return None,
            };
        }
        Some(id)
    }

    fn node_at(&self, pointer: &str) -> Option<&FlatNode> {
        self.document.nodes.get(self.node_id_at(pointer)?)
    }

    pub(crate) fn node_ref_at(&self, pointer: &str) -> Option<JsonNodeRef<'_>> {
        Some(JsonNodeRef {
            document: &self.document,
            id: self.node_id_at(pointer)?,
        })
    }

    pub(crate) fn root_node(&self) -> JsonNodeRef<'_> {
        JsonNodeRef {
            document: &self.document,
            id: self.document.root,
        }
    }

    pub(crate) fn number_source_at(&self, pointer: &str) -> Option<&str> {
        match self.node_at(pointer)? {
            FlatNode::Primitive(raw) if self.kind_at(pointer) == Some(FlatJsonKind::Number) => {
                Some(raw)
            }
            _ => None,
        }
    }

    pub(crate) fn canonical_json_at(&self, pointer: &str) -> Result<String, LlmError> {
        let node = self
            .node_ref_at(pointer)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: format!("provider JSON pointer did not exist: {pointer}"),
            })?;
        crate::exact_json::serialize_json_node(node, crate::exact_json::JsonEncoding::JavaScript)
    }

    pub(crate) fn kind_at(&self, pointer: &str) -> Option<FlatJsonKind> {
        match self.node_at(pointer)? {
            FlatNode::Primitive(raw) if raw == "null" => Some(FlatJsonKind::Null),
            FlatNode::Primitive(raw) if raw == "true" || raw == "false" => {
                Some(FlatJsonKind::Boolean)
            }
            FlatNode::Primitive(_) => Some(FlatJsonKind::Number),
            FlatNode::String(_) => Some(FlatJsonKind::String),
            FlatNode::Array(_) => Some(FlatJsonKind::Array),
            FlatNode::Object(_) => Some(FlatJsonKind::Object),
        }
    }

    pub(crate) fn string_at(&self, pointer: &str) -> Option<&str> {
        match self.node_at(pointer)? {
            FlatNode::String(text) => Some(text),
            _ => None,
        }
    }

    /// Exact decoded units for a string; ordinary Unicode strings are returned
    /// in their normal UTF-16 encoding too.
    pub(crate) fn string_units_at(&self, pointer: &str) -> Option<Vec<u16>> {
        self.node_ref_at(pointer)?.string_units()
    }

    pub(crate) fn array_len(&self, pointer: &str) -> Option<usize> {
        match self.node_at(pointer)? {
            FlatNode::Array(items) => Some(items.len()),
            _ => None,
        }
    }

    /// Object keys carry their authoritative UTF-16 units so a normalized
    /// display key can never be mistaken for a known provider field.
    pub(crate) fn object_entries_at(&self, pointer: &str) -> Option<Vec<(String, Vec<u16>)>> {
        match self.node_at(pointer)? {
            FlatNode::Object(fields) => Some(
                fields
                    .iter()
                    .map(|(display, units, _)| (display.clone(), units.clone()))
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Read a node by identity, so distinct UTF-16 keys with the same display
    /// spelling cannot replace one another's source spans.
    pub(crate) fn raw_source_for_node(&self, node: JsonNodeRef<'_>) -> Option<&str> {
        if !std::ptr::eq(node.document, &self.document) {
            return None;
        }
        self.source
            .get(self.document.node_source_spans.get(node.id)?.clone())
    }

    pub(crate) fn raw_source_at(&self, pointer: &str) -> Option<&str> {
        self.raw_source_for_node(self.node_ref_at(pointer)?)
    }

    /// Build the typed provider-error text directly from a selected source
    /// string before any part of the response is materialized as `Value`.
    /// Callers select the provider-specific pointer with `string_at`; this
    /// method only adds the HTTP status role when that codec supplies one.
    pub(crate) fn provider_error_text_at(
        &self,
        pointer: &str,
        status: Option<u16>,
    ) -> Result<Option<ProviderErrorText>, LlmError> {
        let Some(display) = self.string_at(pointer).filter(|text| !text.is_empty()) else {
            return Ok(None);
        };
        let mut units = self
            .string_units_at(pointer)
            .expect("a selected string has UTF-16 units");
        let mut text = String::new();
        if let Some(status) = status {
            let prefix = format!("{status} ");
            units.splice(0..0, prefix.encode_utf16());
            text.push_str(&prefix);
        }
        text.push_str(display);
        ProviderErrorText::new(text, units)
            .map(Some)
            .ok_or_else(|| LlmError::InvalidRequest {
                message: "provider error text UTF-16 projection did not match display text".into(),
            })
    }

    /// Preserve the provider-selected error message, or the complete canonical
    /// error body when the provider omitted that field. This read happens while
    /// the body is still a flat source document, before any recursive Value view.
    pub(crate) fn provider_error_text_or_body_at(
        &self,
        pointer: &str,
        status: Option<u16>,
    ) -> Result<ProviderErrorText, LlmError> {
        self.provider_error_text_or_source_at(pointer, status, "")
    }

    pub(crate) fn provider_error_text_or_source_at(
        &self,
        pointer: &str,
        status: Option<u16>,
        source_pointer: &str,
    ) -> Result<ProviderErrorText, LlmError> {
        if let Some(text) = self.provider_error_text_at(pointer, status)? {
            return Ok(text);
        }
        let body = self.canonical_json_at(source_pointer)?;
        let prefix = status
            .map(|status| format!("{status} "))
            .unwrap_or_default();
        let display = format!("{prefix}{body}");
        let units = prefix.encode_utf16().chain(body.encode_utf16()).collect();
        ProviderErrorText::new(display, units).ok_or_else(|| LlmError::InvalidRequest {
            message: "provider error body UTF-16 projection did not match display text".into(),
        })
    }

    /// A shallow classification DTO assembled only from the provider string
    /// fields the current codec classifier reads. Unrecognized/deep fields stay
    /// source-backed and never enter a recursive Value tree.
    pub(crate) fn provider_error_classification_body(
        &self,
        error_pointer: &str,
        fields: &[&str],
        selected_text: Option<&ProviderErrorText>,
    ) -> Value {
        let mut error = serde_json::Map::new();
        for field in fields {
            let pointer = if error_pointer.is_empty() {
                format!("/{field}")
            } else {
                format!("{error_pointer}/{field}")
            };
            if *field == "message" {
                if let Some(selected_text) = selected_text {
                    error.insert(
                        "message".into(),
                        Value::String(selected_text.display().to_owned()),
                    );
                }
            } else if let Some(value) = self.string_at(&pointer) {
                error.insert((*field).into(), Value::String(value.to_owned()));
            }
        }
        Value::Object(serde_json::Map::from_iter([(
            "error".into(),
            Value::Object(error),
        )]))
    }

    /// Construct the exact current Native `Rt.makeMessage(status, body, raw)`
    /// text for a valid JSON body. Native selects a truthy top-level `message`
    /// (string as-is, other JSON values as `JSON.stringify`), otherwise the
    /// whole truthy body as `JSON.stringify`, otherwise the original
    /// `Response.text()` string.
    pub(crate) fn native_anthropic_http_error_text(
        &self,
        status: Option<u16>,
    ) -> Result<ProviderErrorText, LlmError> {
        use FlatJsonKind::*;

        fn truthy(node: JsonNodeRef<'_>) -> bool {
            match node.kind() {
                Null => false,
                Boolean => node.primitive_source() == Some("true"),
                Number => node
                    .number_source()
                    .and_then(|source| source.parse::<f64>().ok())
                    .is_some_and(|number| number != 0.0 && !number.is_nan()),
                String => node.string_at().is_some_and(|value| !value.is_empty()),
                Array | Object => true,
            }
        }

        let root = self.root_node();
        let raw_message = if truthy(root) {
            let top_message = root.object_entries().and_then(|entries| {
                entries
                    .into_iter()
                    .rev()
                    .find(|entry| entry.key_units == [109, 101, 115, 115, 97, 103, 101])
                    .map(|entry| entry.value)
            });
            match top_message.filter(|message| truthy(*message)) {
                Some(message) if message.kind() == String => {
                    let display = message.string_at().unwrap_or_default().to_owned();
                    let units = message.string_units().unwrap_or_default();
                    ProviderErrorText::new(display, units).ok_or_else(|| {
                        LlmError::InvalidRequest {
                            message: "Native Anthropic error message UTF-16 projection is invalid"
                                .into(),
                        }
                    })?
                }
                Some(_) => ProviderErrorText::from(self.canonical_json_at("/message")?),
                None => ProviderErrorText::from(self.canonical_json_at("")?),
            }
        } else {
            ProviderErrorText::from(self.source.clone())
        };
        native_status_text(status, raw_message)
    }

    pub(crate) fn into_response_json_with_opaque_paths(
        self,
        opaque_paths: &[String],
    ) -> Result<ResponseJson, LlmError> {
        let opaque_paths = opaque_paths.iter().cloned().collect::<BTreeSet<_>>();
        if let Some(missing) = opaque_paths
            .iter()
            .find(|pointer| !self.document.source_spans.contains_key(*pointer))
        {
            return Err(LlmError::InvalidRequest {
                message: format!("opaque provider JSON pointer did not exist: {missing}"),
            });
        }
        let sanitized =
            self.document
                .stringify(&opaque_paths)
                .map_err(|message| LlmError::InvalidRequest {
                    message: format!("provider response JSON could not be projected: {message}"),
                })?;
        let value = serde_json::from_slice(&sanitized).map_err(|error| {
            LlmError::UnsupportedCapability {
                message: format!(
                    "provider response metadata cannot be represented as serde_json::Value: {error}"
                ),
            }
        })?;
        Ok(ResponseJson {
            value,
            source: self.source,
            source_spans: self.document.source_spans,
            exact_strings: self.document.exact_strings,
            exact_keys: self.document.exact_keys,
            opaque_paths,
        })
    }
}

impl ResponseJson {
    #[cfg(test)]
    pub(crate) fn parse(input: &[u8], invalid_message: &str) -> Result<Self, LlmError> {
        let source = std::str::from_utf8(input)
            .map_err(|_| LlmError::InvalidRequest {
                message: invalid_message.to_owned(),
            })?
            .to_owned();
        if let Ok(value) = serde_json::from_slice(input) {
            return Ok(Self {
                value,
                source,
                source_spans: BTreeMap::new(),
                exact_strings: BTreeMap::new(),
                exact_keys: BTreeMap::new(),
                opaque_paths: BTreeSet::new(),
            });
        }

        let parser = ResponseJsonParser::new(&source);
        let parsed = parser
            .document()
            .map_err(|message| LlmError::InvalidRequest {
                message: format!("{invalid_message}: {message}"),
            })?;
        // The flat scanner accepts arbitrary nesting without recursive parsing,
        // writing, cloning, or destruction. Current provider codecs still consume
        // `serde_json::Value`, so materializing that tree remains a separate
        // library boundary and may reject valid but deeply nested JSON.
        let sanitized =
            parsed
                .stringify(&BTreeSet::new())
                .map_err(|message| LlmError::InvalidRequest {
                    message: format!("provider response JSON could not be projected: {message}"),
                })?;
        let value = serde_json::from_slice(&sanitized).map_err(|error| {
            LlmError::UnsupportedCapability {
                message: format!(
                    "valid response JSON cannot be represented as serde_json::Value: {error}"
                ),
            }
        })?;
        Ok(Self {
            value,
            source,
            source_spans: parsed.source_spans,
            exact_strings: parsed.exact_strings,
            exact_keys: parsed.exact_keys,
            opaque_paths: BTreeSet::new(),
        })
    }

    /// Return and consume the exact JSON source for one validated pointer.
    /// This is the ingress for typed opaque provider leaves such as tool input,
    /// where reconstructing from `Value` would lose duplicate keys, number
    /// lexemes, or lone UTF-16 units in keys and strings.
    pub(crate) fn take_raw_json_source_at(
        &mut self,
        pointer: &str,
    ) -> Result<Option<&str>, LlmError> {
        if self.source_spans.is_empty() {
            let parsed = ResponseJsonParser::new(&self.source)
                .document()
                .map_err(|message| LlmError::InvalidRequest {
                    message: format!("provider response JSON is not valid: {message}"),
                })?;
            self.source_spans = parsed.source_spans;
            self.exact_strings = parsed.exact_strings;
            self.exact_keys = parsed.exact_keys;
        }
        let Some(span) = self.source_spans.get(pointer).cloned() else {
            return Ok(None);
        };
        self.exact_strings
            .retain(|candidate, _| !pointer_is_within(candidate, pointer));
        self.exact_keys
            .retain(|candidate, _| !pointer_is_within(candidate, pointer));
        self.opaque_paths
            .retain(|candidate| !pointer_is_within(candidate, pointer));
        Ok(Some(&self.source[span]))
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

    /// Decode `Response.text()` for the Native Anthropic HTTP error path.
    /// Fetch uses UTF-8 replacement decoding and consumes one leading BOM.
    pub(crate) fn response_text(input: &[u8]) -> String {
        let text = String::from_utf8_lossy(input);
        text.strip_prefix('\u{feff}')
            .unwrap_or(text.as_ref())
            .to_owned()
    }

    /// Construct the same Native message when `Response.text()` was not JSON.
    pub(crate) fn native_anthropic_raw_http_error(
        status: u16,
        input: &[u8],
    ) -> (Value, ProviderErrorText) {
        let body_text = Self::response_text(input);
        let text = native_status_text(Some(status), ProviderErrorText::from(body_text.clone()))
            .expect("UTF-8 response text has a matching UTF-16 projection");
        let body = serde_json::json!({"error": {"message": body_text}});
        (body, text)
    }

    /// Preserve a non-JSON HTTP error body as the provider's current message
    /// field. Successful response bodies still require valid provider JSON.
    pub(crate) fn raw_http_error(status: u16, input: &[u8]) -> (Value, ProviderErrorText) {
        let body_text = String::from_utf8_lossy(input).into_owned();
        let display = format!("{status} {body_text}");
        let text = ProviderErrorText::new(display, {
            let mut units = format!("{status} ").encode_utf16().collect::<Vec<_>>();
            units.extend(body_text.encode_utf16());
            units
        })
        .expect("lossy UTF-8 has a matching UTF-16 projection");
        let body = serde_json::json!({"error": {"message": body_text}});
        (body, text)
    }

    /// Reject strings in provider-owned or otherwise unsupported locations.
    pub(crate) fn finish(self) -> Result<Value, LlmError> {
        if self.exact_strings.is_empty()
            && self.exact_keys.is_empty()
            && self.opaque_paths.is_empty()
        {
            Ok(self.value)
        } else {
            Err(LlmError::UnsupportedCapability {
                message:
                    "provider response contains lone UTF-16 text or keys outside a supported typed leaf"
                        .into(),
            })
        }
    }
}

fn native_status_text(
    status: Option<u16>,
    raw_message: ProviderErrorText,
) -> Result<ProviderErrorText, LlmError> {
    let message = raw_message.display();
    let display = match status.filter(|status| *status != 0) {
        Some(status) if message.is_empty() => format!("{status} status code (no body)"),
        Some(status) => format!("{status} {message}"),
        None if message.is_empty() => "(no status code or body)".to_owned(),
        None => message.to_owned(),
    };
    let units = match status.filter(|status| *status != 0) {
        Some(status) if message.is_empty() => format!("{status} status code (no body)")
            .encode_utf16()
            .collect(),
        Some(status) => format!("{status} ")
            .encode_utf16()
            .chain(raw_message.utf16_code_units().iter().copied())
            .collect(),
        None if message.is_empty() => "(no status code or body)".encode_utf16().collect(),
        None => raw_message.utf16_code_units().to_vec(),
    };
    ProviderErrorText::new(display, units).ok_or_else(|| LlmError::InvalidRequest {
        message: "Native Anthropic error text UTF-16 projection did not match its display".into(),
    })
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
    citations: Option<Option<JsonPayload>>,
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
    Object(Vec<(String, Vec<u16>, usize)>),
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
    key: Option<(String, Vec<u16>)>,
    path_component: Option<String>,
}

struct FlatDocument {
    root: usize,
    nodes: Vec<FlatNode>,
    node_pointers: Vec<String>,
    node_source_spans: Vec<Range<usize>>,
    node_exact_strings: BTreeMap<usize, Vec<u16>>,
    pointer_nodes: BTreeMap<String, usize>,
    source_spans: BTreeMap<String, Range<usize>>,
    exact_strings: BTreeMap<String, Vec<u16>>,
    exact_keys: BTreeMap<String, Vec<u16>>,
}

impl FlatDocument {
    fn stringify(&self, opaque_paths: &BTreeSet<String>) -> Result<Vec<u8>, String> {
        stringify_flat_json(&self.nodes, &self.node_pointers, self.root, opaque_paths)
    }
}

struct ResponseJsonParser<'a> {
    source: &'a str,
    offset: usize,
    nodes: Vec<FlatNode>,
    starts: Vec<usize>,
    node_pointers: Vec<String>,
    node_source_spans: Vec<Range<usize>>,
    node_exact_strings: BTreeMap<usize, Vec<u16>>,
    pointer_nodes: BTreeMap<String, usize>,
    path_offsets: Vec<usize>,
    current_path: String,
    source_spans: BTreeMap<String, Range<usize>>,
    exact_strings: BTreeMap<String, Vec<u16>>,
    exact_keys: BTreeMap<String, Vec<u16>>,
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
            starts: Vec::new(),
            node_pointers: Vec::new(),
            node_source_spans: Vec::new(),
            node_exact_strings: BTreeMap::new(),
            pointer_nodes: BTreeMap::new(),
            path_offsets: Vec::new(),
            current_path: String::new(),
            source_spans: BTreeMap::new(),
            exact_strings: BTreeMap::new(),
            exact_keys: BTreeMap::new(),
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
        Ok(FlatDocument {
            root,
            nodes: self.nodes,
            node_pointers: self.node_pointers,
            node_source_spans: self.node_source_spans,
            node_exact_strings: self.node_exact_strings,
            pointer_nodes: self.pointer_nodes,
            source_spans: self.source_spans,
            exact_strings: self.exact_strings,
            exact_keys: self.exact_keys,
        })
    }

    fn value(&mut self) -> Result<(usize, Option<FrameState>), String> {
        self.whitespace();
        let start = self.offset;
        let pointer = self.current_pointer();
        match self.peek() {
            Some(b'{') => {
                self.offset += 1;
                Ok((
                    self.push_node(FlatNode::Object(Vec::new()), start, pointer.clone()),
                    Some(FrameState::FirstObject),
                ))
            }
            Some(b'[') => {
                self.offset += 1;
                Ok((
                    self.push_node(FlatNode::Array(Vec::new()), start, pointer.clone()),
                    Some(FrameState::FirstArray),
                ))
            }
            Some(b'"') => {
                let (text, exact) = self.string()?;
                let node = self.push_node(FlatNode::String(text), start, pointer.clone());
                if let Some(units) = exact {
                    self.node_exact_strings.insert(node, units.clone());
                    self.exact_strings.insert(self.current_pointer(), units);
                }
                self.source_spans.insert(pointer, start..self.offset);
                Ok((node, None))
            }
            Some(_) => {
                let (node, state) = self.primitive(start, pointer.clone())?;
                self.source_spans.insert(pointer, start..self.offset);
                Ok((node, state))
            }
            None => Err("unexpected end of JSON".into()),
        }
    }

    fn array_value(&mut self, parent: usize, stack: &mut Vec<Frame>) -> Result<(), String> {
        let index = match self.nodes.get(parent) {
            Some(FlatNode::Array(items)) => items.len(),
            _ => return Err("array frame does not reference an array node".into()),
        };
        let component = index.to_string();
        self.push_path(&component);
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
            self.pop_path()
                .expect("a scalar child has a path component");
        }
        Ok(())
    }

    fn object_key(&mut self, stack: &mut [Frame]) -> Result<(), String> {
        if self.peek() != Some(b'"') {
            return Err("expected a JSON object key".into());
        }
        let (key, exact) = self.string()?;
        let units = exact.unwrap_or_else(|| key.encode_utf16().collect());
        let frame = stack.last_mut().ok_or("missing JSON parser frame")?;
        frame.key = Some((key, units));
        frame.state = FrameState::ObjectColon;
        Ok(())
    }

    fn object_value(&mut self, parent: usize, stack: &mut Vec<Frame>) -> Result<(), String> {
        let (key, key_units) = stack
            .last_mut()
            .and_then(|frame| frame.key.take())
            .ok_or("object value is missing its key")?;
        let duplicate = match self.nodes.get(parent) {
            Some(FlatNode::Object(fields)) => fields
                .iter()
                .any(|(_, existing_units, _)| existing_units == &key_units),
            _ => return Err("object frame does not reference an object node".into()),
        };
        if duplicate {
            let pointer = self.pointer_with_child(&key);
            self.clear_replaced_subtree(&pointer);
        }
        let key_pointer = self.pointer_with_child(&key);
        if !String::from_utf16_lossy(&key_units)
            .encode_utf16()
            .eq(key_units.iter().copied())
        {
            self.exact_keys.insert(key_pointer, key_units.clone());
        }
        self.push_path(&key);
        let (child, state) = self.value()?;
        match self.nodes.get_mut(parent) {
            Some(FlatNode::Object(fields)) => {
                if let Some((_, _, value)) = fields
                    .iter_mut()
                    .find(|(_, existing_units, _)| *existing_units == key_units)
                {
                    *value = child;
                } else {
                    fields.push((key.clone(), key_units, child));
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
            self.pop_path()
                .expect("a scalar child has a path component");
        }
        Ok(())
    }

    fn pop_frame(&mut self, stack: &mut Vec<Frame>) -> Result<(), String> {
        let frame = stack.pop().ok_or("missing JSON parser frame")?;
        let start = *self
            .starts
            .get(frame.node)
            .ok_or("missing JSON source span start")?;
        self.source_spans
            .insert(self.current_pointer(), start..self.offset);
        self.node_source_spans[frame.node] = start..self.offset;
        if frame.path_component.is_some() {
            self.pop_path()?;
        }
        Ok(())
    }

    fn primitive(
        &mut self,
        start: usize,
        pointer: String,
    ) -> Result<(usize, Option<FrameState>), String> {
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
        Ok((
            self.push_node(FlatNode::Primitive(primitive), start, pointer),
            None,
        ))
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

    fn string(&mut self) -> Result<(String, Option<Vec<u16>>), String> {
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

    fn push_node(&mut self, node: FlatNode, start: usize, pointer: String) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        self.starts.push(start);
        self.node_pointers.push(pointer.clone());
        self.node_source_spans.push(start..self.offset);
        self.pointer_nodes.insert(pointer, id);
        id
    }

    fn push_path(&mut self, component: &str) {
        self.path_offsets.push(self.current_path.len());
        push_pointer_segment(&mut self.current_path, component);
    }

    fn pop_path(&mut self) -> Result<(), String> {
        let offset = self
            .path_offsets
            .pop()
            .ok_or("missing JSON path component")?;
        self.current_path.truncate(offset);
        Ok(())
    }

    fn current_pointer(&self) -> String {
        // Copy the already escaped path once. Rejoining every ancestor here
        // allocated two replacement strings per segment at every depth.
        self.current_path.clone()
    }

    fn pointer_with_child(&self, child: &str) -> String {
        let mut pointer = self.current_pointer();
        push_pointer_segment(&mut pointer, child);
        pointer
    }

    fn clear_replaced_subtree(&mut self, pointer: &str) {
        self.exact_strings
            .retain(|existing, _| !pointer_is_within(existing, pointer));
        self.exact_keys
            .retain(|existing, _| !pointer_is_within(existing, pointer));
        self.source_spans
            .retain(|existing, _| !pointer_is_within(existing, pointer));
        self.pointer_nodes
            .retain(|existing, _| !pointer_is_within(existing, pointer));
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

fn stringify_flat_json(
    nodes: &[FlatNode],
    node_pointers: &[String],
    root: usize,
    opaque_paths: &BTreeSet<String>,
) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let mut stack = vec![WriteJson::Node(root)];
    while let Some(job) = stack.pop() {
        match job {
            WriteJson::Byte(byte) => output.push(byte),
            WriteJson::String(text) => {
                serde_json::to_writer(&mut output, text).map_err(|error| error.to_string())?
            }
            WriteJson::Node(id) => {
                if node_pointers
                    .get(id)
                    .is_some_and(|pointer| opaque_paths.contains(pointer))
                {
                    output.extend_from_slice(b"null");
                    continue;
                }
                match nodes.get(id).ok_or("invalid JSON node")? {
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
                        for (index, (key, _, child)) in fields.iter().enumerate().rev() {
                            stack.push(WriteJson::Node(*child));
                            stack.push(WriteJson::Byte(b':'));
                            stack.push(WriteJson::String(key));
                            if index != 0 {
                                stack.push(WriteJson::Byte(b','));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

fn push_pointer_segment(pointer: &mut String, segment: &str) {
    pointer.push('/');
    if !segment.contains(['~', '/']) {
        pointer.push_str(segment);
        return;
    }
    for character in segment.chars() {
        match character {
            '~' => pointer.push_str("~0"),
            '/' => pointer.push_str("~1"),
            other => pointer.push(other),
        }
    }
}

fn pointer_is_within(candidate: &str, pointer: &str) -> bool {
    pointer.is_empty()
        || candidate == pointer
        || candidate
            .strip_prefix(pointer)
            .is_some_and(|rest| rest.starts_with('/'))
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
    fn argument_escape_state_preserves_source_across_fragments() {
        let exact = |units: Vec<u16>| DecodedText {
            text: String::from_utf16_lossy(&units),
            utf16_code_units: Some(units),
        };
        let text = |value: &str| DecodedText {
            text: value.into(),
            utf16_code_units: None,
        };
        let mut decoder = ArgumentJsonDecoder::default();
        decoder.push(text("{\"x\":\"\\")).unwrap();
        assert!(decoder.push(exact(vec![0xd800])).is_err());

        let mut decoder = ArgumentJsonDecoder::default();
        decoder.push(text("\"\\uD8")).unwrap();
        assert!(decoder.push(exact(vec![0xd800])).is_err());

        let mut decoder = ArgumentJsonDecoder::default();
        let prefix = decoder.push(text("{\"x\":\"")).unwrap();
        let high = decoder.push(exact(vec![0xd83d])).unwrap();
        let low = decoder.push(exact(vec![0xde00])).unwrap();
        let lone = decoder.push(exact(vec![0xd800])).unwrap();
        let suffix = decoder.push(text("\"}")).unwrap();
        assert_eq!(high.utf16_code_units(), &[0xd83d]);
        assert_eq!(lone.utf16_code_units(), &[0xd800]);
        let payload = JsonPayload::parse_fragments([&prefix, &high, &low, &lone, &suffix]).unwrap();
        assert_eq!(
            payload.string_units_at("/x"),
            Some(vec![0xd83d, 0xde00, 0xd800])
        );

        let mut decoder = ArgumentJsonDecoder::default();
        decoder.push(text("\"\\")).unwrap();
        assert!(decoder
            .push(exact(vec![b'\\' as u16, 0xd800, b'"' as u16]))
            .is_ok());
    }

    #[test]
    fn cached_pointer_restores_parent_and_escapes_each_component_once() {
        let scanner = ResponseJsonScanner::parse(
            br#"{"a/b":{"~x":[{"text":"\ud800"},17],"next":false},"a/b":{"~x":[{"text":"\udc00"},18],"next":true},"tail":[null,"end"]}"#,
            INVALID,
        ).unwrap();
        assert_eq!(
            scanner.raw_source_at("/a~1b/~0x/0/text"),
            Some(r#""\udc00""#)
        );
        assert_eq!(scanner.raw_source_at("/a~1b/~0x/1"), Some("18"));
        assert_eq!(scanner.raw_source_at("/a~1b/next"), Some("true"));
        assert_eq!(scanner.raw_source_at("/tail/0"), Some("null"));
        assert_eq!(scanner.raw_source_at("/tail/1"), Some(r#""end""#));
    }

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
    fn ordinary_provider_error_text_keeps_its_selected_source_role() {
        let scanner =
            ResponseJsonScanner::parse(br#"{"error":{"message":"bad request"}}"#, INVALID).unwrap();
        let selected = scanner
            .provider_error_text_at("/error/message", None)
            .unwrap()
            .expect("ordinary provider text retains its source role");
        assert_eq!(selected.display(), "bad request");
        let text = selected.prefixed("400 ");
        assert_eq!(text.display(), "400 bad request");
        assert_eq!(
            text.utf16_code_units(),
            "400 bad request".encode_utf16().collect::<Vec<_>>()
        );
    }

    #[test]
    fn ordinary_http_429_error_text_is_typed_for_current_host_copy() {
        let scanner =
            ResponseJsonScanner::parse(br#"{"error":{"message":"free tier busy"}}"#, INVALID)
                .unwrap();
        let selected = scanner
            .provider_error_text_at("/error/message", Some(429))
            .unwrap()
            .expect("HTTP 429 status and text are retained together");
        assert_eq!(selected.display(), "429 free tier busy");
        assert_eq!(
            selected.utf16_code_units(),
            "429 free tier busy".encode_utf16().collect::<Vec<_>>()
        );
    }

    #[test]
    fn flat_scanner_exposes_exact_selected_error_units_before_value_projection() {
        let scanner = ResponseJsonScanner::parse(
            br#"{"error":{"message":"bad\ud800"},"other":true}"#,
            INVALID,
        )
        .unwrap();
        assert_eq!(
            scanner.kind_at("/error/message"),
            Some(FlatJsonKind::String)
        );
        assert_eq!(scanner.string_at("/error/message"), Some("bad�"));
        assert_eq!(
            scanner.string_units_at("/error/message"),
            Some(vec![b'b' as u16, b'a' as u16, b'd' as u16, 0xd800])
        );
        assert_eq!(scanner.array_len("/error"), None);
        assert_eq!(
            scanner.object_entries_at("/error"),
            Some(vec![(
                "message".to_owned(),
                "message".encode_utf16().collect()
            )])
        );
        let selected = scanner
            .provider_error_text_at("/error/message", None)
            .unwrap()
            .unwrap();
        assert_eq!(selected.display(), "bad�");
        assert_eq!(selected.utf16_code_units(), &[98, 97, 100, 0xd800]);
        let error = selected.prefixed("400 ");
        assert_eq!(error.display(), "400 bad�");
        assert_eq!(
            error.utf16_code_units(),
            &[52, 48, 48, 32, 98, 97, 100, 0xd800]
        );
    }

    #[test]
    fn flat_scanner_pointer_access_tracks_last_duplicate_key_value() {
        let scanner =
            ResponseJsonScanner::parse(br#"{"error":{"message":"old"},"error":"new"}"#, INVALID)
                .unwrap();
        assert_eq!(scanner.string_at("/error"), Some("new"));
        assert_eq!(scanner.string_at("/error/message"), None);
        assert_eq!(scanner.raw_source_at("/error"), Some(r#""new""#));
        assert_eq!(scanner.raw_source_at("/error/message"), None);
    }

    #[test]
    fn flat_scanner_object_entries_keep_exact_surrogate_key_units() {
        let scanner =
            ResponseJsonScanner::parse(br#"{"payload":{"field\ud800":1}}"#, INVALID).unwrap();
        assert_eq!(
            scanner.object_entries_at("/payload"),
            Some(vec![(
                "field�".to_owned(),
                vec![102, 105, 101, 108, 100, 0xd800]
            )])
        );
        assert_eq!(
            scanner.raw_source_at("/payload"),
            Some(r#"{"field\ud800":1}"#)
        );
    }

    #[test]
    fn flat_node_handles_expose_canonicalizer_inputs_without_value_tree() {
        let scanner =
            ResponseJsonScanner::parse(br#"{"10":[1.00],"2":"\ud800","text":"ok"}"#, INVALID)
                .unwrap();
        let root = scanner.root_node();
        assert_eq!(root.kind(), FlatJsonKind::Object);
        let entries = root.object_entries().unwrap();
        assert_eq!(entries[0].key, "10");
        assert_eq!(entries[0].key_units, &[49, 48]);
        let array = &entries[0].value;
        assert_eq!(array.kind(), FlatJsonKind::Array);
        assert_eq!(array.array_len(), Some(1));
        assert_eq!(array.array_item(0).unwrap().number_source(), Some("1.00"));
        let truth = &entries[2].value;
        assert_eq!(truth.kind(), FlatJsonKind::String);
        assert_eq!(truth.primitive_source(), None);
        let surrogate = &entries[1].value;
        assert_eq!(surrogate.kind(), FlatJsonKind::String);
        assert_eq!(surrogate.string_at(), Some("�"));
        assert_eq!(surrogate.string_units(), Some(vec![0xd800]));

        let primitive_scanner =
            ResponseJsonScanner::parse(br#"{"truth":true,"empty":null}"#, INVALID).unwrap();
        let primitive_entries = primitive_scanner.root_node().object_entries().unwrap();
        assert_eq!(primitive_entries[0].value.primitive_source(), Some("true"));
        assert_eq!(primitive_entries[1].value.primitive_source(), Some("null"));
    }

    #[test]
    fn opaque_source_handoff_masks_deep_payload_before_value_materialization() {
        const DEPTH: usize = 16_384;
        let raw = format!("{}0{}", "[".repeat(DEPTH), "]".repeat(DEPTH));
        let source = format!(r#"{{"content":[{{"type":"tool_use","input":{raw}}}]}}"#);
        let scanner = ResponseJsonScanner::parse(source.as_bytes(), INVALID).unwrap();
        assert_eq!(
            scanner.kind_at("/content/0/input"),
            Some(FlatJsonKind::Array)
        );
        assert_eq!(
            scanner.raw_source_at("/content/0/input"),
            Some(raw.as_str())
        );
        let mut response = scanner
            .into_response_json_with_opaque_paths(&["/content/0/input".to_owned()])
            .unwrap();
        assert!(response
            .value
            .pointer("/content/0/input")
            .unwrap()
            .is_null());
        assert_eq!(
            response
                .take_raw_json_source_at("/content/0/input")
                .unwrap(),
            Some(raw.as_str())
        );
        assert!(response.finish().is_ok());
    }

    #[test]
    fn provider_error_leaf_is_not_reparsed_as_a_second_json_document() {
        let scanner = ResponseJsonScanner::parse(
            br#"{"error":{"message":"400 {\"error\":{\"message\":\"bad\"}}\ud800","type":"invalid_request_error","detail":[[[0]]]} }"#,
            INVALID,
        )
        .unwrap();
        let selected = scanner
            .provider_error_text_at("/error/message", None)
            .unwrap()
            .unwrap();
        assert_eq!(selected.display(), "400 {\"error\":{\"message\":\"bad\"}}�");
        let expected = "400 {\"error\":{\"message\":\"bad\"}}"
            .encode_utf16()
            .chain([0xd800])
            .collect::<Vec<_>>();
        assert_eq!(selected.utf16_code_units(), expected.as_slice());
        let final_text = selected.prefixed("400 ");
        assert_eq!(
            final_text.display(),
            "400 400 {\"error\":{\"message\":\"bad\"}}�"
        );
        let body = scanner.provider_error_classification_body(
            "/error",
            &["type", "message"],
            Some(&selected),
        );
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], selected.display());
        assert!(body["error"].get("detail").is_none());
    }

    #[test]
    fn missing_error_message_keeps_the_flat_canonical_body_as_the_text() {
        let nested = format!("{}0{}", "[".repeat(16_384), "]".repeat(16_384));
        let source = format!(r#"{{"error":{{"type":"api_error","detail":{nested}}}}}"#);
        let scanner = ResponseJsonScanner::parse(source.as_bytes(), INVALID).unwrap();
        let selected = scanner
            .provider_error_text_or_body_at("/error/message", Some(500))
            .unwrap();
        assert!(selected.display().starts_with("500 {\"error\":{"));
        assert!(selected.display().contains(&nested));
    }

    #[test]
    fn raw_http_error_body_keeps_exact_text_without_a_json_parse() {
        let html = "<!DOCTYPE html><html>proxy</html>";
        let (body, text) = ResponseJson::raw_http_error(400, html.as_bytes());
        assert_eq!(
            body.pointer("/error/message").and_then(Value::as_str),
            Some(html)
        );
        assert_eq!(text.display(), format!("400 {html}"));
        assert_eq!(
            text.utf16_code_units(),
            format!("400 {html}").encode_utf16().collect::<Vec<_>>()
        );
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
    fn rejects_unconsumed_unpaired_keys_and_opaque_text() {
        let parsed = ResponseJson::parse(br#"{"\ud800":"value"}"#, INVALID).unwrap();
        assert!(matches!(
            parsed.finish(),
            Err(LlmError::UnsupportedCapability { .. })
        ));
        let parsed = ResponseJson::parse(br#"{"opaque":{"text":"\ud800"}}"#, INVALID).unwrap();
        assert!(matches!(
            parsed.finish(),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }

    #[test]
    fn typed_raw_subtree_preserves_lexemes_duplicate_keys_and_surrogate_keys() {
        let source = br#"{"content":[{"type":"tool_use","input":{"n":1.00,"dup":1,"dup":2,"\ud800":"v","nested":{"s":"\udfff"}}}]}"#;
        let mut parsed = ResponseJson::parse(source, INVALID).unwrap();
        let raw = parsed
            .take_raw_json_source_at("/content/0/input")
            .unwrap()
            .expect("typed tool-input source");
        assert_eq!(
            raw,
            r#"{"n":1.00,"dup":1,"dup":2,"\ud800":"v","nested":{"s":"\udfff"}}"#
        );
        assert!(parsed.finish().is_ok());
    }

    #[test]
    fn taking_one_raw_subtree_does_not_consume_unrelated_surrogate_text() {
        let mut parsed =
            ResponseJson::parse(br#"{"input":{"text":"\ud800"},"other":"\udfff"}"#, INVALID)
                .unwrap();
        assert_eq!(
            parsed.take_raw_json_source_at("/input").unwrap(),
            Some(r#"{"text":"\ud800"}"#)
        );
        assert!(matches!(
            parsed.finish(),
            Err(LlmError::UnsupportedCapability { .. })
        ));
    }

    #[test]
    fn raw_pointer_uses_the_last_duplicate_provider_field() {
        let mut parsed = ResponseJson::parse(
            br#"{"error":{"message":"old"},"error":{"message":"new"}}"#,
            INVALID,
        )
        .unwrap();
        assert_eq!(
            parsed
                .take_raw_json_source_at("/error")
                .unwrap()
                .expect("last error field"),
            r#"{"message":"new"}"#
        );
        assert_eq!(parsed.value["error"]["message"], "new");
        assert!(parsed.finish().is_ok());
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
        let leaf_span = scanned
            .source_spans
            .get(&pointer)
            .expect("the flat scanner retains the deep leaf source span");
        assert_eq!(&source[leaf_span.clone()], r#""\ud800""#);
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
