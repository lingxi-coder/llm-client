//! Lossless, format-tagged provider extensions.
//!
//! The original JSON is the serialization authority. Decoding caches a typed
//! view without rewriting unknown fields or invalidating borrowed references.
use super::LlmError;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
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
    data: Value,
    #[serde(skip)]
    decoded: Arc<OnceLock<Decoded>>,
}

impl NativeExtension {
    /// Preserve native JSON without interpreting it. The receiving provider
    /// must validate supported formats before executing a request.
    pub fn new(format: impl Into<String>, data: Value) -> Result<Self, LlmError> {
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
        let data = serde_json::to_value(&value).map_err(|error| LlmError::InvalidRequest {
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
            serde_json::from_value::<T>(self.data.clone()).map_err(|error| {
                LlmError::InvalidRequest {
                    message: format!("invalid {} extension: {error}", T::FORMAT),
                }
            })?
        };
        let before = serde_json::to_value(&typed).map_err(|error| LlmError::InvalidRequest {
            message: error.to_string(),
        })?;
        let result = edit(&mut typed);
        let after = serde_json::to_value(&typed).map_err(|error| LlmError::InvalidRequest {
            message: error.to_string(),
        })?;
        let mut updated = self.data.clone();
        merge_typed_changes(&mut updated, &before, &after)?;
        self.data = updated;
        self.decoded = Arc::default();
        let _ = self.decoded.set(Ok(Box::new(typed)));
        Ok(result)
    }

    pub fn format(&self) -> &str {
        &self.format
    }
    pub fn data(&self) -> &Value {
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
            serde_json::from_value::<T>(self.data.clone())
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

fn merge_typed_changes(raw: &mut Value, before: &Value, after: &Value) -> Result<(), LlmError> {
    if matches!(before, Value::Array(_))
        && has_unmodeled_fields(raw, before)
        && !same_json_representation(before, after)
    {
        return Err(LlmError::InvalidRequest {
            message:
                "cannot edit a native array with unmodeled fields without an explicit replacement"
                    .into(),
        });
    }
    match (raw, before, after) {
        (Value::Object(raw), Value::Object(before), Value::Object(after)) => {
            for key in before.keys() {
                if !after.contains_key(key) {
                    raw.remove(key);
                }
            }
            for (key, value) in after {
                match (raw.get_mut(key), before.get(key)) {
                    (Some(raw), Some(before)) => merge_typed_changes(raw, before, value)?,
                    _ => {
                        raw.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (Value::Array(raw), Value::Array(before), Value::Array(after))
            if raw.len() == before.len() && before.len() == after.len() =>
        {
            for ((raw, before), after) in raw.iter_mut().zip(before).zip(after) {
                merge_typed_changes(raw, before, after)?;
            }
        }
        (raw, before, after) => {
            // JSON value equality equates 0.0 and -0.0, but their wire
            // representations differ. Compare number spelling when deciding
            // whether a typed edit actually changed a known scalar.
            let changed = !same_json_representation(before, after);
            if changed {
                *raw = after.clone();
            }
        }
    }
    Ok(())
}

fn same_json_representation(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            left == right && left.as_f64().map(f64::to_bits) == right.as_f64().map(f64::to_bits)
        }
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| same_json_representation(left, right))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|((left_key, left), (right_key, right))| {
                        left_key == right_key && same_json_representation(left, right)
                    })
        }
        _ => left == right,
    }
}

fn has_unmodeled_fields(raw: &Value, typed: &Value) -> bool {
    match (raw, typed) {
        (Value::Object(raw), Value::Object(typed)) => raw.iter().any(|(key, value)| {
            typed
                .get(key)
                .is_none_or(|typed| has_unmodeled_fields(value, typed))
        }),
        (Value::Array(raw), Value::Array(typed)) => {
            raw.len() != typed.len()
                || raw
                    .iter()
                    .zip(typed)
                    .any(|(raw, typed)| has_unmodeled_fields(raw, typed))
        }
        _ => false,
    }
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
            serde_json::json!({"text":"hello", "signature":"opaque", "future":{"x":1}}),
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
            serde_json::json!({"text":"before", "signature":"opaque", "future":{"x":1}}),
        )
        .unwrap();
        let snapshot = extension.clone();
        let captured = snapshot.decode::<Known>().unwrap();
        extension
            .edit::<Known, _>(|value| value.text = "after".into())
            .unwrap();
        assert_eq!(extension.decode::<Known>().unwrap().text, "after");
        assert_eq!(captured.text, "before");
        assert_eq!(extension.data()["signature"], "opaque");
        assert_eq!(extension.data()["future"]["x"], 1);
        assert_eq!(snapshot.data()["text"], "before");
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
        assert!(!extension.data()["value"]
            .as_f64()
            .unwrap()
            .is_sign_negative());
        extension
            .edit::<Coordinates, _>(|typed| typed.value = -0.0)
            .unwrap();
        assert!(extension.data()["value"]
            .as_f64()
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
            serde_json::json!({
                "title":"before", "items":[
                    {"text":"a", "signature":"signature-a"},
                    {"text":"b", "signature":"signature-b"}
                ]
            }),
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
        assert_eq!(extension.data()["items"][0]["signature"], "signature-a");
        let mut fully_typed = NativeExtension::from_typed(List {
            title: "before".into(),
            items: vec![Known { text: "a".into() }, Known { text: "b".into() }],
        })
        .unwrap();
        fully_typed
            .edit::<List, _>(|value| value.items.swap(0, 1))
            .unwrap();
        assert_eq!(fully_typed.data()["items"][0]["text"], "b");
    }

    #[test]
    fn malformed_payload_and_conflicting_type_are_errors() {
        let extension = NativeExtension::new(Known::FORMAT, Value::Null).unwrap();
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
}
