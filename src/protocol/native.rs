//! Lossless, format-tagged provider extensions.
//!
//! The original JSON is the serialization authority. Decoding caches a typed
//! view without rewriting unknown fields or invalidating borrowed references.
use super::{JsonPayload, LlmError};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    any::Any,
    fmt,
    sync::{Arc, OnceLock},
};

/// A provider-owned typed view of one native format.
pub trait NativeType: Serialize + DeserializeOwned + Send + Sync + 'static {
    const FORMAT: &'static str;
}

type Decoded = Result<Box<dyn Any + Send + Sync>, LlmError>;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeExtension {
    format: String,
    data: JsonPayload,
    #[serde(skip)]
    decoded: Arc<OnceLock<Decoded>>,
}

impl NativeExtension {
    /// Preserve native JSON without interpreting it. The receiving provider
    /// must validate supported formats before executing a request.
    pub fn new(format: impl Into<String>, data: JsonPayload) -> Result<Self, LlmError> {
        let format = format.into();
        if format.is_empty() || format.chars().any(char::is_whitespace) {
            return Err(LlmError::InvalidRequest {
                message: "native extension format must be a nonempty identifier".into(),
            });
        }
        Ok(Self {
            format,
            data,
            decoded: Arc::default(),
        })
    }

    pub fn from_typed<T: NativeType>(value: T) -> Result<Self, LlmError> {
        let source = serde_json::to_string(&value).map_err(|error| LlmError::InvalidRequest {
            message: format!("cannot serialize {}: {error}", T::FORMAT),
        })?;
        let data = JsonPayload::parse(&source).map_err(|error| LlmError::InvalidRequest {
            message: format!("cannot serialize {}: {error}", T::FORMAT),
        })?;
        let extension = Self::new(T::FORMAT, data)?;
        let _ = extension.decoded.set(Ok(Box::new(value)));
        Ok(extension)
    }

    /// Edit a typed view while preserving unknown JSON fields. References
    /// captured by other cloned requests retain their original snapshot.
    /// Editing an array with unmodeled fields is rejected because element
    /// identity cannot be inferred safely. Failed edits leave this extension
    /// unchanged; callers may explicitly replace the extension instead.
    pub fn edit<T: NativeType + Clone, R>(
        &mut self,
        edit: impl FnOnce(&mut T) -> R,
    ) -> Result<R, LlmError> {
        // Decoding a cold extension for an edit must not publish a cache
        // entry until the entire edit succeeds.
        let mut typed = if self.decoded.get().is_some() {
            self.decode::<T>()?.clone()
        } else {
            if !self.is::<T>() {
                return Err(LlmError::InvalidRequest {
                    message: format!(
                        "native format {} cannot be read as {}",
                        self.format,
                        T::FORMAT
                    ),
                });
            }
            serde_json::from_str::<T>(self.data.as_json_source()).map_err(|error| {
                LlmError::InvalidRequest {
                    message: format!("invalid {} extension: {error}", T::FORMAT),
                }
            })?
        };
        let before_source =
            serde_json::to_string(&typed).map_err(|error| LlmError::InvalidRequest {
                message: error.to_string(),
            })?;
        let before =
            JsonPayload::parse(&before_source).map_err(|error| LlmError::InvalidRequest {
                message: error.to_string(),
            })?;
        let result = edit(&mut typed);
        let after_source =
            serde_json::to_string(&typed).map_err(|error| LlmError::InvalidRequest {
                message: error.to_string(),
            })?;
        let after =
            JsonPayload::parse(&after_source).map_err(|error| LlmError::InvalidRequest {
                message: error.to_string(),
            })?;
        let updated = merge_typed_changes(&self.data, &before, &after)?;
        self.data = updated;
        self.decoded = Arc::default();
        let _ = self.decoded.set(Ok(Box::new(typed)));
        Ok(result)
    }

    pub fn format(&self) -> &str {
        &self.format
    }
    pub fn data(&self) -> &JsonPayload {
        &self.data
    }
    pub fn is<T: NativeType>(&self) -> bool {
        self.format == T::FORMAT
    }

    pub fn decode<T: NativeType>(&self) -> Result<&T, LlmError> {
        if !self.is::<T>() {
            return Err(LlmError::InvalidRequest {
                message: format!(
                    "native format {} cannot be read as {}",
                    self.format,
                    T::FORMAT
                ),
            });
        }
        let decoded = self.decoded.get_or_init(|| {
            serde_json::from_str::<T>(self.data.as_json_source())
                .map(|value| Box::new(value) as Box<dyn Any + Send + Sync>)
                .map_err(|error| LlmError::InvalidRequest {
                    message: format!("invalid {} extension: {error}", T::FORMAT),
                })
        });
        match decoded {
            Ok(value) => value
                .downcast_ref::<T>()
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: format!(
                        "native format {} has conflicting Rust type registrations",
                        self.format
                    ),
                }),
            Err(error) => Err(error.clone()),
        }
    }
}

fn merge_typed_changes(
    raw: &JsonPayload,
    before: &JsonPayload,
    after: &JsonPayload,
) -> Result<JsonPayload, LlmError> {
    use crate::response_json::{FlatJsonKind, JsonNodeRef};

    enum Work<'r, 'b, 'a> {
        Merge(
            Option<JsonNodeRef<'r>>,
            Option<JsonNodeRef<'b>>,
            Option<JsonNodeRef<'a>>,
        ),
        Byte(u8),
        Key(Vec<u16>),
        Raw(Vec<u8>),
    }

    let raw_scanner = raw.scanner()?;
    let before_scanner = before.scanner()?;
    let after_scanner = after.scanner()?;
    let raw_root = raw_scanner.root_node();
    let before_root = before_scanner.root_node();
    let after_root = after_scanner.root_node();
    if has_unmodeled_array_changes(raw_root, before_root, after_root) {
        return Err(LlmError::InvalidRequest {
            message:
                "cannot edit a native array with unmodeled fields without an explicit replacement"
                    .into(),
        });
    }

    let mut output = Vec::new();
    let mut stack = vec![Work::Merge(
        Some(raw_root),
        Some(before_root),
        Some(after_root),
    )];
    while let Some(work) = stack.pop() {
        match work {
            Work::Byte(byte) => output.push(byte),
            Work::Key(key) => crate::exact_json::write_json_string_from_utf16(&mut output, &key),
            Work::Raw(source) => output.extend_from_slice(&source),
            Work::Merge(raw, before, after) => match (raw, before, after) {
                (Some(raw), Some(before), Some(after))
                    if raw.kind() == FlatJsonKind::Object
                        && before.kind() == FlatJsonKind::Object
                        && after.kind() == FlatJsonKind::Object =>
                {
                    let raw_entries =
                        raw.object_entries()
                            .ok_or_else(|| LlmError::InvalidRequest {
                                message: "native JSON object has no flat children".into(),
                            })?;
                    let before_entries =
                        before
                            .object_entries()
                            .ok_or_else(|| LlmError::InvalidRequest {
                                message: "typed native JSON object has no flat children".into(),
                            })?;
                    let after_entries =
                        after
                            .object_entries()
                            .ok_or_else(|| LlmError::InvalidRequest {
                                message: "edited native JSON object has no flat children".into(),
                            })?;
                    let mut fields = Vec::with_capacity(raw_entries.len() + after_entries.len());
                    for raw_entry in &raw_entries {
                        let before_value = before_entries
                            .iter()
                            .find(|entry| entry.key_units == raw_entry.key_units)
                            .map(|entry| entry.value);
                        let after_value = after_entries
                            .iter()
                            .find(|entry| entry.key_units == raw_entry.key_units)
                            .map(|entry| entry.value);
                        if before_value.is_some() && after_value.is_none() {
                            continue;
                        }
                        fields.push((
                            raw_entry.key_units.to_vec(),
                            Some(raw_entry.value),
                            before_value,
                            after_value,
                        ));
                    }
                    for after_entry in &after_entries {
                        if raw_entries
                            .iter()
                            .any(|entry| entry.key_units == after_entry.key_units)
                        {
                            continue;
                        }
                        let before_value = before_entries
                            .iter()
                            .find(|entry| entry.key_units == after_entry.key_units)
                            .map(|entry| entry.value);
                        fields.push((
                            after_entry.key_units.to_vec(),
                            None,
                            before_value,
                            Some(after_entry.value),
                        ));
                    }

                    stack.push(Work::Byte(b'}'));
                    for (index, (key, raw, before, after)) in fields.into_iter().enumerate().rev() {
                        stack.push(Work::Merge(raw, before, after));
                        stack.push(Work::Byte(b':'));
                        stack.push(Work::Key(key));
                        if index > 0 {
                            stack.push(Work::Byte(b','));
                        }
                    }
                    stack.push(Work::Byte(b'{'));
                }
                (Some(raw), Some(before), Some(after))
                    if raw.kind() == FlatJsonKind::Array
                        && before.kind() == FlatJsonKind::Array
                        && after.kind() == FlatJsonKind::Array
                        && raw.array_len() == before.array_len()
                        && before.array_len() == after.array_len() =>
                {
                    let len = raw.array_len().unwrap_or(0);
                    stack.push(Work::Byte(b']'));
                    for index in (0..len).rev() {
                        stack.push(Work::Merge(
                            raw.array_item(index),
                            before.array_item(index),
                            after.array_item(index),
                        ));
                        if index > 0 {
                            stack.push(Work::Byte(b','));
                        }
                    }
                    stack.push(Work::Byte(b'['));
                }
                (Some(raw), Some(before), Some(after)) => {
                    let selected = if same_json_representation(before, after) {
                        raw
                    } else {
                        after
                    };
                    stack.push(Work::Raw(
                        crate::exact_json::serialize_json_node(
                            selected,
                            crate::exact_json::JsonEncoding::Serde,
                        )?
                        .into_bytes(),
                    ));
                }
                (_, _, Some(after)) => stack.push(Work::Raw(
                    crate::exact_json::serialize_json_node(
                        after,
                        crate::exact_json::JsonEncoding::Serde,
                    )?
                    .into_bytes(),
                )),
                (Some(raw), _, None) => stack.push(Work::Raw(
                    crate::exact_json::serialize_json_node(
                        raw,
                        crate::exact_json::JsonEncoding::Serde,
                    )?
                    .into_bytes(),
                )),
                (None, _, None) => output.extend_from_slice(b"null"),
            },
        }
    }
    let source = String::from_utf8(output).map_err(|error| LlmError::InvalidRequest {
        message: format!("edited native JSON is not UTF-8: {error}"),
    })?;
    JsonPayload::parse(&source).map_err(|error| LlmError::InvalidRequest {
        message: format!("edited native JSON is invalid: {error}"),
    })
}

fn same_json_representation<'l, 'r>(
    left: crate::response_json::JsonNodeRef<'l>,
    right: crate::response_json::JsonNodeRef<'r>,
) -> bool {
    use crate::response_json::FlatJsonKind;
    let mut pending = vec![(left, right)];
    while let Some((left, right)) = pending.pop() {
        if left.kind() != right.kind() {
            return false;
        }
        match left.kind() {
            FlatJsonKind::Null => {}
            FlatJsonKind::Boolean => {
                if left.primitive_source() != right.primitive_source() {
                    return false;
                }
            }
            FlatJsonKind::Number => {
                let left = left
                    .number_source()
                    .and_then(|source| serde_json::from_str::<serde_json::Number>(source).ok());
                let right = right
                    .number_source()
                    .and_then(|source| serde_json::from_str::<serde_json::Number>(source).ok());
                let (Some(left), Some(right)) = (left, right) else {
                    return false;
                };
                if left != right
                    || left.as_f64().map(f64::to_bits) != right.as_f64().map(f64::to_bits)
                {
                    return false;
                }
            }
            FlatJsonKind::String => {
                if left.string_units() != right.string_units() {
                    return false;
                }
            }
            FlatJsonKind::Array => {
                let (Some(left_len), Some(right_len)) = (left.array_len(), right.array_len())
                else {
                    return false;
                };
                if left_len != right_len {
                    return false;
                }
                for index in 0..left_len {
                    let (Some(left), Some(right)) =
                        (left.array_item(index), right.array_item(index))
                    else {
                        return false;
                    };
                    pending.push((left, right));
                }
            }
            FlatJsonKind::Object => {
                let (Some(left), Some(right)) = (left.object_entries(), right.object_entries())
                else {
                    return false;
                };
                if left.len() != right.len() {
                    return false;
                }
                for (left, right) in left.into_iter().zip(right) {
                    if left.key_units != right.key_units {
                        return false;
                    }
                    pending.push((left.value, right.value));
                }
            }
        }
    }
    true
}

fn has_unmodeled_fields(
    raw: crate::response_json::JsonNodeRef<'_>,
    typed: crate::response_json::JsonNodeRef<'_>,
) -> bool {
    use crate::response_json::FlatJsonKind;
    let mut pending = vec![(raw, typed)];
    while let Some((raw, typed)) = pending.pop() {
        match (raw.kind(), typed.kind()) {
            (FlatJsonKind::Object, FlatJsonKind::Object) => {
                let (Some(raw), Some(typed)) = (raw.object_entries(), typed.object_entries())
                else {
                    return true;
                };
                for raw_entry in raw {
                    let Some(typed_entry) = typed
                        .iter()
                        .find(|entry| entry.key_units == raw_entry.key_units)
                    else {
                        return true;
                    };
                    pending.push((raw_entry.value, typed_entry.value));
                }
            }
            (FlatJsonKind::Array, FlatJsonKind::Array) => {
                let (Some(raw_len), Some(typed_len)) = (raw.array_len(), typed.array_len()) else {
                    return true;
                };
                if raw_len != typed_len {
                    return true;
                }
                for index in 0..raw_len {
                    let (Some(raw), Some(typed)) = (raw.array_item(index), typed.array_item(index))
                    else {
                        return true;
                    };
                    pending.push((raw, typed));
                }
            }
            _ => {}
        }
    }
    false
}

/// Find arrays at any depth whose raw elements contain unmodeled fields and
/// whose typed contents changed. A typed object can hide such an array below
/// ordinary fields, so checking only the root value would allow reordering
/// unknown fields by array position.
fn has_unmodeled_array_changes(
    raw: crate::response_json::JsonNodeRef<'_>,
    before: crate::response_json::JsonNodeRef<'_>,
    after: crate::response_json::JsonNodeRef<'_>,
) -> bool {
    use crate::response_json::FlatJsonKind;

    let mut pending = vec![(raw, before, Some(after))];
    while let Some((raw, before, after)) = pending.pop() {
        if raw.kind() == FlatJsonKind::Array
            && before.kind() == FlatJsonKind::Array
            && has_unmodeled_fields(raw, before)
            && after.is_none_or(|after| !same_json_representation(before, after))
        {
            return true;
        }

        let Some(after) = after else {
            continue;
        };
        match (raw.kind(), before.kind(), after.kind()) {
            (FlatJsonKind::Object, FlatJsonKind::Object, FlatJsonKind::Object) => {
                let (Some(raw_entries), Some(before_entries), Some(after_entries)) = (
                    raw.object_entries(),
                    before.object_entries(),
                    after.object_entries(),
                ) else {
                    return true;
                };
                for before_entry in before_entries {
                    let Some(raw_entry) = raw_entries
                        .iter()
                        .find(|entry| entry.key_units == before_entry.key_units)
                    else {
                        continue;
                    };
                    let after_value = after_entries
                        .iter()
                        .find(|entry| entry.key_units == before_entry.key_units)
                        .map(|entry| entry.value);
                    pending.push((raw_entry.value, before_entry.value, after_value));
                }
            }
            (FlatJsonKind::Array, FlatJsonKind::Array, FlatJsonKind::Array) => {
                let (Some(raw_len), Some(before_len), Some(after_len)) =
                    (raw.array_len(), before.array_len(), after.array_len())
                else {
                    return true;
                };
                for index in 0..raw_len.min(before_len).min(after_len) {
                    let (Some(raw_item), Some(before_item), Some(after_item)) = (
                        raw.array_item(index),
                        before.array_item(index),
                        after.array_item(index),
                    ) else {
                        return true;
                    };
                    pending.push((raw_item, before_item, Some(after_item)));
                }
            }
            _ => {}
        }
    }
    false
}

impl fmt::Debug for NativeExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeExtension")
            .field("format", &self.format)
            .field("data", &self.data)
            .finish()
    }
}
impl PartialEq for NativeExtension {
    fn eq(&self, other: &Self) -> bool {
        self.format == other.format && self.data == other.data
    }
}
impl Eq for NativeExtension {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn payload(value: Value) -> JsonPayload {
        JsonPayload::from_generated_value(&value).unwrap()
    }
    #[derive(Clone, Serialize, Deserialize)]
    struct Known {
        text: String,
    }
    impl NativeType for Known {
        const FORMAT: &'static str = "test.known.v1";
    }
    #[derive(Serialize, Deserialize)]
    struct Other {
        text: String,
    }
    impl NativeType for Other {
        const FORMAT: &'static str = "test.other.v1";
    }

    #[test]
    fn typed_view_preserves_unknown_fields_and_borrow_identity() {
        let extension = NativeExtension::new(
            Known::FORMAT,
            payload(serde_json::json!({"text":"hello", "signature":"opaque", "future":{"x":1}})),
        )
        .unwrap();
        let view = extension.decode::<Known>().unwrap();
        assert_eq!(view.text, "hello");
        assert!(std::ptr::eq(view, extension.decode::<Known>().unwrap()));
        let json = serde_json::to_value(&extension).unwrap();
        assert_eq!(json["data"]["signature"], "opaque");
        assert_eq!(json["data"]["future"]["x"], 1);
        assert!(extension.decode::<Other>().is_err());
        let decoded: NativeExtension = serde_json::from_value(json).unwrap();
        assert_eq!(decoded, extension);
        assert_eq!(decoded.decode::<Known>().unwrap().text, "hello");
    }

    #[test]
    fn editing_keeps_unknown_fields_and_cloned_snapshot_unchanged() {
        let mut extension = NativeExtension::new(
            Known::FORMAT,
            payload(serde_json::json!({"text":"before", "signature":"opaque", "future":{"x":1}})),
        )
        .unwrap();
        let snapshot = extension.clone();
        let captured = snapshot.decode::<Known>().unwrap();
        extension
            .edit::<Known, _>(|value| value.text = "after".into())
            .unwrap();
        assert_eq!(extension.decode::<Known>().unwrap().text, "after");
        assert_eq!(captured.text, "before");
        assert_eq!(
            extension.data().string_at("/signature").as_deref(),
            Some("opaque")
        );
        assert_eq!(extension.data().number_at("/future/x"), Some(1.0));
        assert_eq!(
            snapshot.data().string_at("/text").as_deref(),
            Some("before")
        );
    }

    #[test]
    fn typed_edits_preserve_explicit_signed_zero_changes() {
        #[derive(Clone, Serialize, Deserialize)]
        struct Coordinates {
            value: f64,
        }
        impl NativeType for Coordinates {
            const FORMAT: &'static str = "test.coordinates.v1";
        }
        let mut extension = NativeExtension::from_typed(Coordinates { value: -0.0 }).unwrap();
        extension
            .edit::<Coordinates, _>(|typed| typed.value = 0.0)
            .unwrap();
        assert!(!extension
            .data()
            .number_at("/value")
            .unwrap()
            .is_sign_negative());
        extension
            .edit::<Coordinates, _>(|typed| typed.value = -0.0)
            .unwrap();
        assert!(extension
            .data()
            .number_at("/value")
            .unwrap()
            .is_sign_negative());
    }

    #[test]
    fn unknown_array_edits_fail_without_changing_raw_or_typed_cache() {
        #[derive(Clone, Serialize, Deserialize)]
        struct List {
            title: String,
            items: Vec<Known>,
        }
        impl NativeType for List {
            const FORMAT: &'static str = "test.list.v1";
        }
        let mut extension = NativeExtension::new(
            List::FORMAT,
            payload(serde_json::json!({
                "title":"before", "items":[
                    {"text":"a", "signature":"signature-a"},
                    {"text":"b", "signature":"signature-b"}
                ]
            })),
        )
        .unwrap();
        let original_raw = extension.data().clone();
        assert!(extension.decoded.get().is_none());
        assert!(extension
            .edit::<List, _>(|value| value.items.swap(0, 1))
            .is_err());
        assert!(extension.decoded.get().is_none());
        assert_eq!(extension.data(), &original_raw);
        let original_view = extension.decode::<List>().unwrap() as *const List;
        let result = extension.edit::<List, _>(|value| {
            value.title = "after".into();
            value.items.swap(0, 1);
        });
        assert!(matches!(result, Err(LlmError::InvalidRequest { .. })));
        assert_eq!(extension.data(), &original_raw);
        assert_eq!(
            extension.decode::<List>().unwrap() as *const List,
            original_view
        );
        assert_eq!(extension.decode::<List>().unwrap().title, "before");
        assert_eq!(extension.decode::<List>().unwrap().items[0].text, "a");
        extension
            .edit::<List, _>(|value| value.title = "safe".into())
            .unwrap();
        assert_eq!(
            extension.data().string_at("/items/0/signature").as_deref(),
            Some("signature-a")
        );
        let mut fully_typed = NativeExtension::from_typed(List {
            title: "before".into(),
            items: vec![Known { text: "a".into() }, Known { text: "b".into() }],
        })
        .unwrap();
        fully_typed
            .edit::<List, _>(|value| value.items.swap(0, 1))
            .unwrap();
        assert_eq!(
            fully_typed.data().string_at("/items/0/text").as_deref(),
            Some("b")
        );
    }

    #[test]
    fn malformed_payload_and_conflicting_type_are_errors() {
        let extension = NativeExtension::new(Known::FORMAT, payload(Value::Null)).unwrap();
        assert!(extension.decode::<Known>().is_err());
        let extension = NativeExtension::from_typed(Known { text: "ok".into() }).unwrap();
        #[derive(Serialize, Deserialize)]
        struct Conflict {
            text: String,
        }
        impl NativeType for Conflict {
            const FORMAT: &'static str = Known::FORMAT;
        }
        assert!(extension.decode::<Conflict>().is_err());
    }

    #[test]
    fn native_extension_wire_keeps_opaque_deep_and_lone_data_source() {
        let depth = 16_384;
        let nested = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        let source = format!(r#"{{"text":"ok","future":{{"deep":{nested},"lone":"\ud800"}}}}"#);
        let extension =
            NativeExtension::new(Known::FORMAT, JsonPayload::parse(&source).unwrap()).unwrap();
        let encoded = serde_json::to_string(&extension).unwrap();
        assert!(encoded.contains(&format!("\"data\":{source}")));
        let decoded: NativeExtension = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.data().as_json_source(), source);
    }
}
