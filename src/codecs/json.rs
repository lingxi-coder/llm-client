//! JSON values that borrow request text, schemas, and inline byte payloads.
use crate::{
    exact_json::{JavaScriptFormatter, JavaScriptValue, JsonEncoding},
    protocol::LlmError,
    transport::HttpRequest,
};
use serde::{
    ser::{SerializeMap, SerializeSeq},
    Serialize, Serializer,
};
use serde_json::Value;
use std::{
    borrow::Cow,
    fmt, io,
    ops::{Index, IndexMut},
    sync::OnceLock,
};

#[derive(Clone)]
pub(crate) struct WireValue<'a> {
    value: Cow<'a, Value>,
    fields: Vec<(&'static str, WireValue<'a>)>,
    payload: Option<Payload<'a>>,
    // Inspection through the Value API is uncommon for borrowed payloads.
    // Materialize only on demand; serialization itself never uses this cache.
    materialized: OnceLock<Value>,
}
#[derive(Clone)]
enum Payload<'a> {
    Text(&'a str),
    Raw(&'a serde_json::value::RawValue),
    Array(Vec<WireValue<'a>>),
    Data(&'a [u8], String),
}
impl<'a> From<Value> for WireValue<'a> {
    fn from(value: Value) -> Self {
        Self {
            value: Cow::Owned(value),
            fields: Default::default(),
            payload: None,
            materialized: OnceLock::new(),
        }
    }
}
impl<'a> WireValue<'a> {
    pub(crate) fn borrowed(value: &'a Value) -> Self {
        Self {
            value: Cow::Borrowed(value),
            ..Value::Null.into()
        }
    }
    pub(crate) fn text(text: &'a str) -> Self {
        Self {
            payload: Some(Payload::Text(text)),
            ..Value::Null.into()
        }
    }
    pub(crate) fn tool_input(input: &'a Value, raw: Option<&'a str>) -> Result<Self, LlmError> {
        match crate::exact_json::validated_tool_input_json(input, raw)? {
            Some(raw) => {
                let raw = serde_json::from_str::<&serde_json::value::RawValue>(raw)
                    .map_err(json_error)?;
                Ok(Self {
                    payload: Some(Payload::Raw(raw)),
                    value: Cow::Borrowed(input),
                    ..Value::Null.into()
                })
            }
            None => Ok(Self::borrowed(input)),
        }
    }
    pub(crate) fn tool_output(
        content: &'a str,
        blocks: Option<&'a [Value]>,
        raw: Option<&'a str>,
    ) -> Result<Self, LlmError> {
        match crate::exact_json::validated_tool_output_json(content, blocks, raw)? {
            Some(raw) => Ok(Self {
                payload: Some(Payload::Raw(
                    serde_json::from_str::<&serde_json::value::RawValue>(raw)
                        .map_err(json_error)?,
                )),
                value: Cow::Owned(crate::exact_json::parse_tool_output_json(raw)?),
                ..Value::Null.into()
            }),
            None => Ok(blocks.map_or_else(
                || Self::text(content),
                |blocks| Self::array(blocks.iter().map(Self::borrowed).collect()),
            )),
        }
    }
    pub(crate) fn array(values: Vec<Self>) -> Self {
        Self {
            payload: Some(Payload::Array(values)),
            ..Value::Null.into()
        }
    }
    pub(crate) fn base64(bytes: &'a [u8], prefix: String) -> Self {
        Self {
            payload: Some(Payload::Data(bytes, prefix)),
            ..Value::Null.into()
        }
    }
    fn as_value(&self) -> &Value {
        if self.payload.is_none() && self.fields.is_empty() {
            return &self.value;
        }
        self.materialized.get_or_init(|| {
            let mut value = match &self.payload {
                Some(Payload::Raw(_)) => self.value.as_ref().clone(),
                Some(Payload::Text(text)) => Value::String((*text).to_owned()),
                Some(Payload::Array(items)) => {
                    Value::Array(items.iter().map(|item| item.as_value().clone()).collect())
                }
                Some(Payload::Data(bytes, prefix)) => {
                    Value::String(Data(bytes, prefix).to_string())
                }
                None => self.value.as_ref().clone(),
            };
            for (key, child) in &self.fields {
                value[*key] = child.as_value().clone();
            }
            value
        })
    }
    fn materialize(&mut self) {
        if self.payload.is_some() || !self.fields.is_empty() {
            let value = self.as_value().clone();
            self.value = Cow::Owned(value);
            self.payload = None;
            self.fields.clear();
        }
        self.materialized.take();
    }
    fn into_value(mut self) -> Value {
        self.materialize();
        self.value.into_owned()
    }
    fn field(&self, key: &str) -> Option<&Self> {
        self.fields
            .iter()
            .find(|(field, _)| *field == key)
            .map(|(_, value)| value)
    }
    fn remove_field(&mut self, key: &str) -> Option<Self> {
        let index = self.fields.iter().position(|(field, _)| *field == key)?;
        Some(self.fields.swap_remove(index).1)
    }
    pub(crate) fn with(mut self, field: &'static str, value: Self) -> Self {
        if self.payload.is_some() {
            self.materialize();
        }
        self.materialized.take();
        self.value.to_mut()[field] = Value::Null;
        if let Some((_, old)) = self.fields.iter_mut().find(|(key, _)| *key == field) {
            *old = value;
        } else if self.fields.is_empty() {
            self.fields = vec![(field, value)];
        } else {
            self.fields.push((field, value));
        }
        self
    }
    /// Rewrite an owned JSON array before borrowed field overlays are attached.
    pub(crate) fn map_array_field(
        mut self,
        field: &'static str,
        mut map: impl FnMut(usize, Value) -> Self,
    ) -> Self {
        let Some(array) = self
            .value
            .to_mut()
            .get_mut(field)
            .and_then(Value::as_array_mut)
        else {
            return self;
        };
        let array = std::mem::take(array)
            .into_iter()
            .enumerate()
            .map(|(index, value)| map(index, value))
            .collect();
        self.with(field, Self::array(array))
    }
    pub(crate) fn array_field_mut(&mut self, field: &str) -> Option<&mut Vec<Self>> {
        self.materialized.take();
        let value = &mut self.fields.iter_mut().find(|(key, _)| *key == field)?.1;
        value.materialized.take();
        match value.payload.as_mut()? {
            Payload::Array(array) => Some(array),
            _ => None,
        }
    }
    pub(crate) fn get(&self, field: &str) -> Option<&Value> {
        self.field(field)
            .map(Self::as_value)
            .or_else(|| self.value.get(field))
    }
    pub(crate) fn get_str(&self, field: &str) -> Option<&str> {
        match self.field(field) {
            Some(Self {
                payload: Some(Payload::Text(text)),
                ..
            }) => Some(text),
            _ => self.get(field).and_then(Value::as_str),
        }
    }
    pub(crate) fn take_field(&mut self, field: &str) -> Option<Self> {
        self.materialized.take();
        let owned = self.value.to_mut().as_object_mut()?.remove(field)?;
        Some(self.remove_field(field).unwrap_or_else(|| owned.into()))
    }
    /// Update one owned field without materializing unrelated borrowed payloads.
    pub(crate) fn insert(&mut self, field: String, value: Value) {
        self.materialized.take();
        self.remove_field(&field);
        self.value
            .to_mut()
            .as_object_mut()
            .expect("wire object")
            .insert(field, value);
    }
    pub(crate) fn extend_fields(&mut self, fields: serde_json::Map<String, Value>) {
        for (field, value) in fields {
            self.insert(field, value);
        }
    }
    pub(crate) fn remove(&mut self, field: &str) {
        self.materialized.take();
        self.remove_field(field);
        self.value
            .to_mut()
            .as_object_mut()
            .expect("wire object")
            .remove(field);
    }
}
impl Index<&str> for WireValue<'_> {
    type Output = Value;
    fn index(&self, key: &str) -> &Value {
        self.get(key).unwrap_or(&Value::Null)
    }
}
impl IndexMut<&str> for WireValue<'_> {
    fn index_mut(&mut self, key: &str) -> &mut Value {
        if self.payload.is_some() {
            self.materialize();
        }
        self.materialized.take();
        if let Some(value) = self.remove_field(key) {
            self.value.to_mut()[key] = value.into_value();
        }
        &mut self.value.to_mut()[key]
    }
}
struct Data<'a>(&'a [u8], &'a str);
impl fmt::Display for Data<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.1)?;
        fmt::Display::fmt(
            &base64::display::Base64Display::new(
                self.0,
                &base64::engine::general_purpose::STANDARD,
            ),
            f,
        )
    }
}
struct EncodedWireValue<'a, 'b>(&'a WireValue<'b>, JsonEncoding);
impl Serialize for EncodedWireValue<'_, '_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize_mode(serializer, self.1)
    }
}
impl Serialize for WireValue<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.serialize_mode(serializer, JsonEncoding::Serde)
    }
}
impl WireValue<'_> {
    fn serialize_mode<S: Serializer>(
        &self,
        serializer: S,
        encoding: JsonEncoding,
    ) -> Result<S::Ok, S::Error> {
        if let Some(payload) = &self.payload {
            return match payload {
                Payload::Text(text) => serializer.serialize_str(text),
                Payload::Raw(raw) => raw.serialize(serializer),
                Payload::Data(bytes, prefix) => serializer.collect_str(&Data(bytes, prefix)),
                Payload::Array(array) => {
                    let mut seq = serializer.serialize_seq(Some(array.len()))?;
                    for value in array {
                        seq.serialize_element(&EncodedWireValue(value, encoding))?;
                    }
                    seq.end()
                }
            };
        }
        if self.fields.is_empty() {
            return if encoding == JsonEncoding::JavaScript {
                JavaScriptValue(&self.value).serialize(serializer)
            } else {
                self.value.serialize(serializer)
            };
        }
        let object = self.value.as_object().expect("overridden wire object");
        let mut map = serializer.serialize_map(Some(object.len()))?;
        for (key, value) in crate::exact_json::entries(object, encoding) {
            if let Some(value) = self.field(key) {
                map.serialize_entry(key, &EncodedWireValue(value, encoding))?;
            } else if encoding == JsonEncoding::JavaScript {
                map.serialize_entry(key, &JavaScriptValue(value))?;
            } else {
                map.serialize_entry(key, value)?;
            }
        }
        map.end()
    }
}
pub(crate) struct WireRequest<'a> {
    pub(crate) http: HttpRequest,
    pub(crate) body: WireValue<'a>,
    pub(crate) json_encoding: JsonEncoding,
}
impl<'a> WireRequest<'a> {
    pub(crate) fn new(url: String, headers: Vec<(String, String)>, body: WireValue<'a>) -> Self {
        Self {
            http: HttpRequest {
                http1_header_layout: None,
                method: "POST".into(),
                url,
                headers,
                body: Default::default(),
                timeout: None,
            },
            body,
            json_encoding: JsonEncoding::Serde,
        }
    }
    pub(crate) fn write_body<W: io::Write>(&self, writer: W) -> Result<(), LlmError> {
        match self.json_encoding {
            JsonEncoding::Serde => serde_json::to_writer(writer, &self.body).map_err(json_error),
            JsonEncoding::JavaScript => EncodedWireValue(&self.body, self.json_encoding)
                .serialize(&mut serde_json::Serializer::with_formatter(
                    writer,
                    JavaScriptFormatter,
                ))
                .map_err(json_error),
        }
    }
    pub(crate) fn encode(mut self) -> Result<HttpRequest, LlmError> {
        let mut bytes = Vec::new();
        self.write_body(&mut bytes)?;
        self.http.body = bytes.into();
        Ok(self.http)
    }
    pub(crate) fn body_len(&self) -> Result<usize, LlmError> {
        let mut writer = Counter(0);
        self.write_body(&mut writer)?;
        Ok(writer.0)
    }
}
fn json_error(error: serde_json::Error) -> LlmError {
    LlmError::InvalidRequest {
        message: format!("request body is not serializable: {error}"),
    }
}
struct Counter(usize);
impl io::Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::other("request size overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn native_numbers_and_order_keep_borrowed_payloads_and_exact_size() {
        let schema = json!({"2":1.0,"0":-0.0,"integer":u64::MAX});
        let text = "borrowed text";
        for encoding in [JsonEncoding::Serde, JsonEncoding::JavaScript] {
            let body = WireValue::from(json!({"schema":null,"text":null,"data":null}))
                .with("schema", WireValue::borrowed(&schema))
                .with("text", WireValue::text(text))
                .with("data", WireValue::base64(b"bytes", String::new()));
            let mut request = WireRequest::new("https://fixture".into(), vec![], body);
            request.json_encoding = encoding;
            let length = request.body_len().unwrap();
            assert!(request.body.materialized.get().is_none());
            for (_, field) in &request.body.fields {
                assert!(field.materialized.get().is_none());
            }
            let bytes = request.encode().unwrap().body;
            assert_eq!(length, bytes.len());
            let wire = std::str::from_utf8(&bytes).unwrap();
            if encoding == JsonEncoding::JavaScript {
                assert_eq!(
                    wire,
                    r#"{"schema":{"0":0,"2":1,"integer":18446744073709552000},"text":"borrowed text","data":"Ynl0ZXM="}"#
                );
            } else {
                assert_eq!(
                    wire,
                    r#"{"schema":{"2":1.0,"0":-0.0,"integer":18446744073709551615},"text":"borrowed text","data":"Ynl0ZXM="}"#
                );
            }
        }
    }

    #[test]
    fn borrowed_values_remain_coherent_when_inspected_and_mutated() {
        let schema = json!({"properties": {"message": {"type": "string"}}});
        let text = "quoted \"text\"\n中文";
        let mut value = WireValue::from(json!({"role": "user"}))
            .with("schema", WireValue::borrowed(&schema))
            .with("text", WireValue::text(text))
            .with("content", WireValue::array(vec![WireValue::text(text)]));
        assert_eq!(value.get_str("text"), Some(text));
        assert_eq!(value.get("text"), Some(&json!(text)));
        assert_eq!(value["schema"], schema);
        assert_eq!(value["content"], json!([text]));

        // A cached materialization must never hide later array or field edits.
        value
            .array_field_mut("content")
            .unwrap()
            .push(json!(7).into());
        assert_eq!(value["content"], json!([text, 7]));
        value["schema"]["additionalProperties"] = json!(false);
        value["text"] = json!("changed");
        assert_eq!(value.get_str("text"), Some("changed"));
        assert!(schema.get("additionalProperties").is_none());
        value.insert("extra".into(), json!(true));
        value.remove("role");
        // serde_json's insertion-ordered map swaps the last field into the
        // removed role's position, as it did before adding borrowed payloads.
        let expected = json!({
            "extra": true,
            "schema": {"properties": {"message": {"type": "string"}}, "additionalProperties": false},
            "text": "changed", "content": [text, 7],
        });
        assert_eq!(
            serde_json::to_vec(&value).unwrap(),
            serde_json::to_vec(&expected).unwrap()
        );
    }

    #[test]
    fn metadata_edits_keep_text_schemas_and_base64_payloads_borrowed() {
        let schema = json!({"type": "object"});
        let text = "long text";
        let mut body = WireValue::from(json!({"model": "test", "stream": true}))
            .with("schema", WireValue::borrowed(&schema))
            .with("text", WireValue::text(text))
            .with("data", WireValue::base64(b"hello", String::new()));
        // Hosted Anthropic wrappers change only metadata; Chat reasoning adds
        // native fields. Neither operation may allocate encoded media strings.
        body.remove("model");
        body.remove("stream");
        body.insert("anthropic_version".into(), json!("bedrock-2023-05-31"));
        body.extend_fields(json!({"reasoning": "native"}).as_object().unwrap().clone());
        assert!(body.materialized.get().is_none());
        assert!(matches!(
            body.field("schema").unwrap().value,
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            body.field("text").unwrap().payload,
            Some(Payload::Text(_))
        ));
        let data = body.field("data").unwrap();
        assert!(matches!(data.payload, Some(Payload::Data(_, _))));
        assert!(data.materialized.get().is_none());
        assert_eq!(serde_json::to_value(&body).unwrap()["data"], "aGVsbG8=");
    }

    #[test]
    fn borrowed_payloads_share_serialization_and_body_length() {
        let text = "\n\t\"\\中文";
        let schema = json!({"enum": [text]});
        let value = WireValue::from(json!({}))
            .with("schema", WireValue::borrowed(&schema))
            .with("text", WireValue::text(text))
            .with(
                "data",
                WireValue::base64(b"hello", "data:text/plain;base64,".into()),
            );
        let request = WireRequest::new("https://test.invalid".into(), vec![], value);
        let length = request.body_len().unwrap();
        let body = request.encode().unwrap().body;
        let expected =
            json!({"schema":schema,"text":text,"data":"data:text/plain;base64,aGVsbG8="});
        assert_eq!(length, body.len());
        assert_eq!(body.as_ref(), serde_json::to_vec(&expected).unwrap());
    }
}
