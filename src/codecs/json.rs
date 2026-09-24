//! JSON values with borrowed inline byte payloads. No encoded media strings.
use crate::{protocol::LlmError, transport::HttpRequest};
use serde::{
    ser::{SerializeMap, SerializeSeq},
    Serialize, Serializer,
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fmt, io,
    ops::{Index, IndexMut},
};

#[derive(Clone)]
pub(crate) struct WireValue<'a> {
    value: Value,
    fields: BTreeMap<String, WireValue<'a>>,
    array: Option<Vec<WireValue<'a>>>,
    data: Option<(&'a [u8], String)>,
}
impl<'a> From<Value> for WireValue<'a> {
    fn from(value: Value) -> Self {
        Self {
            value,
            fields: Default::default(),
            array: None,
            data: None,
        }
    }
}
impl<'a> WireValue<'a> {
    pub(crate) fn array(values: Vec<Self>) -> Self {
        Self {
            array: Some(values),
            ..Value::Null.into()
        }
    }
    pub(crate) fn base64(bytes: &'a [u8], prefix: String) -> Self {
        Self {
            data: Some((bytes, prefix)),
            ..Value::Null.into()
        }
    }
    pub(crate) fn with(mut self, field: &str, value: Self) -> Self {
        self.value[field] = Value::Null;
        self.fields.insert(field.into(), value);
        self
    }
    pub(crate) fn get(&self, field: &str) -> Option<&Value> {
        self.fields
            .get(field)
            .map(|value| &value.value)
            .or_else(|| self.value.get(field))
    }
    pub(crate) fn object_mut(&mut self) -> &mut serde_json::Map<String, Value> {
        self.value.as_object_mut().expect("wire object")
    }
    pub(crate) fn remove(&mut self, field: &str) {
        self.fields.remove(field);
        self.object_mut().remove(field);
    }
}
impl Index<&str> for WireValue<'_> {
    type Output = Value;
    fn index(&self, key: &str) -> &Value {
        &self.value[key]
    }
}
impl IndexMut<&str> for WireValue<'_> {
    fn index_mut(&mut self, key: &str) -> &mut Value {
        self.fields.remove(key);
        &mut self.value[key]
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
impl Serialize for WireValue<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let Some((bytes, prefix)) = &self.data {
            return serializer.collect_str(&Data(bytes, prefix));
        }
        if let Some(array) = &self.array {
            let mut seq = serializer.serialize_seq(Some(array.len()))?;
            for value in array {
                seq.serialize_element(value)?;
            }
            return seq.end();
        }
        if self.fields.is_empty() {
            return self.value.serialize(serializer);
        }
        let object = self.value.as_object().expect("overridden wire object");
        let mut map = serializer.serialize_map(Some(object.len()))?;
        for (key, value) in object {
            if let Some(value) = self.fields.get(key) {
                map.serialize_entry(key, value)?;
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
}
impl<'a> WireRequest<'a> {
    pub(crate) fn new(url: String, headers: Vec<(String, String)>, body: WireValue<'a>) -> Self {
        Self {
            http: HttpRequest {
                method: "POST".into(),
                url,
                headers,
                body: Default::default(),
                timeout: None,
            },
            body,
        }
    }
    pub(crate) fn encode(mut self) -> Result<HttpRequest, LlmError> {
        self.http.body = serde_json::to_vec(&self.body).map_err(json_error)?.into();
        Ok(self.http)
    }
    pub(crate) fn body_len(&self) -> Result<usize, LlmError> {
        let mut writer = Counter(0);
        serde_json::to_writer(&mut writer, &self.body).map_err(json_error)?;
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
