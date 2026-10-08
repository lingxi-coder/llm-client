//! Explicit per-request wire controls; never shared mutable profile defaults.
use crate::protocol::JsonPayload;
use serde::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RequestControls {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub anthropic: AnthropicControls,
    #[serde(default)]
    pub responses: ResponsesControls,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    JsonObject,
    JsonSchema {
        name: String,
        schema: crate::protocol::JsonPayload,
        #[serde(default)]
        strict: bool,
    },
}

impl<'de> Deserialize<'de> for ResponseFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let scanner = crate::response_json::ResponseJsonScanner::parse(
            raw.get().as_bytes(),
            "response format is not valid JSON",
        )
        .map_err(serde::de::Error::custom)?;
        if scanner.kind_at("") != Some(crate::response_json::FlatJsonKind::Object) {
            return Err(serde::de::Error::custom(
                "response format must be an object",
            ));
        }
        match scanner.string_at("/type") {
            Some("json_object") => Ok(Self::JsonObject),
            Some("json_schema") => {
                let name = scanner
                    .string_at("/name")
                    .ok_or_else(|| serde::de::Error::custom("JSON schema name must be a string"))?
                    .to_owned();
                let source = scanner
                    .raw_source_at("/schema")
                    .ok_or_else(|| serde::de::Error::custom("JSON schema source is required"))?;
                let schema = JsonPayload::parse(source).map_err(serde::de::Error::custom)?;
                let strict = match scanner.raw_source_at("/strict") {
                    None | Some("false") => false,
                    Some("true") => true,
                    _ => {
                        return Err(serde::de::Error::custom(
                            "JSON schema strict must be a boolean",
                        ))
                    }
                };
                Ok(Self::JsonSchema {
                    name,
                    schema,
                    strict,
                })
            }
            _ => Err(serde::de::Error::custom(
                "response format has an unknown type",
            )),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<CacheTtl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<CacheScope>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheScope {
    Global,
}
impl CacheControl {
    pub fn wire_value(&self) -> Value {
        let mut value = serde_json::to_value(self).expect("cache control is serializable");
        value["type"] = Value::String("ephemeral".into());
        value
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AnthropicControls {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_hint: Option<crate::protocol::JsonPayload>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponsesControls {
    /// Explicit legacy continuation. Prefer account-scoped ContinuationRef.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub include: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub client_metadata: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tagged_response_format_keeps_deep_and_lone_source() {
        let nested = format!("{}0{}", "[".repeat(512), "]".repeat(512));
        let source = format!(
            r#"{{"type":"json_schema","name":"answer","schema":{{"\ud800":"\ud800","deep":{nested}}},"strict":true}}"#
        );
        let decoded: ResponseFormat = serde_json::from_str(&source).unwrap();
        let ResponseFormat::JsonSchema { schema, .. } = decoded else {
            panic!("schema format expected")
        };
        assert!(schema.as_json_source().contains(&nested));
        assert!(schema.as_json_source().contains(r#""\ud800":"\ud800""#));
    }

    #[test]
    fn context_hint_is_a_source_carrying_json_value() {
        let controls: RequestControls =
            serde_json::from_str(r#"{"anthropic":{"context_hint":{"\ud800":"\ud800"}}}"#).unwrap();
        let hint = controls.anthropic.context_hint.unwrap();
        assert_eq!(hint.as_json_source(), r#"{"\ud800":"\ud800"}"#);
    }
}
