//! Strict JSON-schema conversion aligned with Claude Code 2.1.293's embedded
//! `ulr`/`C` converter source (structured-output strict mode).
//!
//! `ulr` walks a JSON schema and produces the strict variant the Anthropic
//! Messages API accepts for `strict: true` tools: an `object` root, every
//! object closed with `additionalProperties: false`, and only a conservative
//! keyword/type subset allowed. A schema that cannot be represented strictly is
//! rejected with a machine reason (surfaced in the `sending non-strict`
//! warning). This is an explicit opt-in transformation, separate from schema validation.
//! Callers decide whether a failed conversion should reject or use non-strict tools.

use crate::protocol::JsonPayload;
use crate::response_json::{FlatJsonKind, JsonNodeRef, JsonObjectEntryRef};
use std::collections::HashSet;

/// Recursion / size budget — claude `C(e, 32, {remaining: 1e5})`.
const MAX_DEPTH: i32 = 32;
const MAX_NODES: i64 = 100_000;

/// Keywords allowed on any node — claude `Be`.
const ALLOWED_KEYWORDS: &[&str] = &[
    "$schema",
    "type",
    "description",
    "title",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "anyOf",
];

/// Keywords allowed alongside `anyOf` — claude `Ye`.
const ANYOF_SIBLINGS: &[&str] = &["$schema", "description", "title"];

/// Types allowed — claude `le`.
const ALLOWED_TYPES: &[&str] = &[
    "object", "array", "string", "integer", "number", "boolean", "null",
];

/// Convert `schema` to its strict form (claude `ulr`). `Ok(strict_schema)` on
/// success; `Err(reason)` with claude's exact machine reason otherwise. The
/// root must resolve to an `object`.
pub fn to_strict_schema(schema: &JsonPayload) -> Result<JsonPayload, &'static str> {
    let scanner = schema.scanner().map_err(|_| "invalid_schema")?;
    let mut remaining = MAX_NODES;
    let node = convert(scanner.root_node(), MAX_DEPTH, &mut remaining)?;
    if node.string_units_at("/type") != Some("object".encode_utf16().collect()) {
        return Err("root_not_object");
    }
    Ok(node)
}

#[derive(Hash, PartialEq, Eq)]
enum PrimitiveKey {
    Null,
    Boolean(bool),
    Number(u64),
    String(Vec<u16>),
}

fn primitive_key(node: JsonNodeRef<'_>) -> Option<PrimitiveKey> {
    match node.kind() {
        FlatJsonKind::Null => Some(PrimitiveKey::Null),
        FlatJsonKind::Boolean => match node.primitive_source()? {
            "true" => Some(PrimitiveKey::Boolean(true)),
            "false" => Some(PrimitiveKey::Boolean(false)),
            _ => None,
        },
        FlatJsonKind::Number => {
            let number = node.number_source()?.parse::<f64>().ok()?;
            if !number.is_finite() {
                return None;
            }
            Some(PrimitiveKey::Number(if number == 0.0 {
                0.0_f64.to_bits()
            } else {
                number.to_bits()
            }))
        }
        FlatJsonKind::String => Some(PrimitiveKey::String(node.string_units()?)),
        FlatJsonKind::Array | FlatJsonKind::Object => None,
    }
}

fn is_primitive(node: JsonNodeRef<'_>) -> bool {
    primitive_key(node).is_some()
}

fn has_duplicates(nodes: &[JsonNodeRef<'_>]) -> bool {
    let mut seen = HashSet::with_capacity(nodes.len());
    nodes
        .iter()
        .any(|node| primitive_key(*node).is_none_or(|key| !seen.insert(key)))
}

fn key_is(units: &[u16], key: &str) -> bool {
    units.iter().copied().eq(key.encode_utf16())
}

fn field<'a>(node: JsonNodeRef<'a>, key: &str) -> Option<JsonNodeRef<'a>> {
    node.object_entries()?
        .into_iter()
        .rev()
        .find(|entry| key_is(entry.key_units, key))
        .map(|entry| entry.value)
}

/// JavaScript JSON.parse keeps the first insertion position of a property while
/// replacing its value on duplicate keys. Mirror that object enumeration shape.
fn entries_last_by_key<'a>(node: JsonNodeRef<'a>) -> Option<Vec<JsonObjectEntryRef<'a>>> {
    let mut entries: Vec<JsonObjectEntryRef<'a>> = Vec::new();
    for entry in node.object_entries()? {
        if let Some(existing) = entries
            .iter_mut()
            .find(|existing| existing.key_units == entry.key_units)
        {
            existing.value = entry.value;
        } else {
            entries.push(entry);
        }
    }
    // Object.keys/entries enumerate array-index keys first in ascending order;
    // stable sorting retains the insertion order of all other exact UTF-16 keys.
    entries.sort_by_key(|entry| {
        crate::exact_json::javascript_array_index(entry.key_units)
            .map(|index| (0_u8, index))
            .unwrap_or((1_u8, 0))
    });
    Some(entries)
}

fn payload_from_node(node: JsonNodeRef<'_>) -> Result<JsonPayload, &'static str> {
    let source =
        crate::exact_json::serialize_json_node(node, crate::exact_json::JsonEncoding::JavaScript)
            .map_err(|_| "invalid_schema")?;
    JsonPayload::parse(&source).map_err(|_| "invalid_schema")
}

fn object_payload(fields: &[(Vec<u16>, JsonPayload)]) -> Result<JsonPayload, &'static str> {
    let fields = fields
        .iter()
        .map(|(key, value)| (key.clone(), value.as_json_source().to_owned()))
        .collect::<Vec<_>>();
    JsonPayload::object_from_fields(&fields).map_err(|_| "invalid_schema")
}

fn array_payload(items: &[JsonPayload]) -> Result<JsonPayload, &'static str> {
    JsonPayload::array_from_items(items, crate::exact_json::JsonEncoding::JavaScript)
        .map_err(|_| "invalid_schema")
}

fn push_field(fields: &mut Vec<(Vec<u16>, JsonPayload)>, key: &str, value: JsonPayload) {
    fields.push((key.encode_utf16().collect(), value));
}

fn source_array<'a>(node: JsonNodeRef<'a>) -> Option<Vec<JsonNodeRef<'a>>> {
    let length = node.array_len()?;
    (0..length).map(|index| node.array_item(index)).collect()
}

fn type_is(units: &[u16], candidate: &str) -> bool {
    key_is(units, candidate)
}

/// claude `C` — the recursive strict-schema node converter, operating on
/// the source scanner and emitting JSON source without constructing a Value tree.
fn convert(
    e: JsonNodeRef<'_>,
    depth: i32,
    remaining: &mut i64,
) -> Result<JsonPayload, &'static str> {
    if depth <= 0 {
        return Err("max_depth");
    }
    *remaining -= 1;
    if *remaining < 0 {
        return Err("max_nodes");
    }
    if e.kind() != FlatJsonKind::Object {
        return Err("not_object");
    }
    let entries = entries_last_by_key(e).ok_or("not_object")?;
    for entry in &entries {
        if !ALLOWED_KEYWORDS
            .iter()
            .any(|keyword| key_is(entry.key_units, keyword))
        {
            return Err("unsupported_keyword");
        }
    }
    let mut fields = Vec::new();
    if let Some(description) = field(e, "description") {
        if description.kind() != FlatJsonKind::String {
            return Err("unsupported_keyword");
        }
        push_field(&mut fields, "description", payload_from_node(description)?);
    }
    if let Some(title) = field(e, "title") {
        if title.kind() != FlatJsonKind::String {
            return Err("unsupported_keyword");
        }
        push_field(&mut fields, "title", payload_from_node(title)?);
    }
    if let Some(any_of) = field(e, "anyOf") {
        for entry in &entries {
            if !key_is(entry.key_units, "anyOf")
                && !ANYOF_SIBLINGS
                    .iter()
                    .any(|keyword| key_is(entry.key_units, keyword))
            {
                return Err("unsupported_keyword");
            }
        }
        let Some(arr) = source_array(any_of) else {
            return Err("unsupported_keyword");
        };
        if arr.is_empty() {
            return Err("unsupported_keyword");
        }
        let mut out = Vec::with_capacity(arr.len());
        for s in arr {
            out.push(convert(s, depth - 1, remaining)?);
        }
        push_field(&mut fields, "anyOf", array_payload(&out)?);
        return object_payload(&fields);
    }
    if let Some(constant) = field(e, "const") {
        if !is_primitive(constant) {
            return Err("unsupported_const");
        }
        push_field(&mut fields, "const", payload_from_node(constant)?);
    }
    if let Some(enumeration) = field(e, "enum") {
        let Some(arr) = source_array(enumeration) else {
            return Err("unsupported_enum");
        };
        if arr.is_empty() || !arr.iter().all(|node| is_primitive(*node)) || has_duplicates(&arr) {
            return Err("unsupported_enum");
        }
        let values = arr
            .into_iter()
            .map(payload_from_node)
            .collect::<Result<Vec<_>, _>>()?;
        push_field(&mut fields, "enum", array_payload(&values)?);
    }
    // Type validation (claude's `o` handling).
    let o = field(e, "type");
    if let Some(ty) = o {
        match ty.kind() {
            FlatJsonKind::String => {
                let Some(units) = ty.string_units() else {
                    return Err("unsupported_type");
                };
                if !ALLOWED_TYPES.iter().any(|allowed| type_is(&units, allowed)) {
                    return Err("unsupported_type");
                }
                push_field(&mut fields, "type", payload_from_node(ty)?);
            }
            FlatJsonKind::Array => {
                let arr = source_array(ty).ok_or("unsupported_type")?;
                let mut seen = HashSet::with_capacity(arr.len());
                let ok = !arr.is_empty()
                    && arr.iter().all(|item| {
                        item.string_units().is_some_and(|units| {
                            ALLOWED_TYPES.iter().any(|allowed| type_is(&units, allowed))
                                && !type_is(&units, "object")
                                && !type_is(&units, "array")
                                && seen.insert(units)
                        })
                    });
                if !ok {
                    return Err("unsupported_type");
                }
                push_field(&mut fields, "type", payload_from_node(ty)?);
            }
            _ => return Err("unsupported_type"),
        }
    }
    // `o` as a plain string ("object"/"array"/…), or None (incl. a type-array).
    let o_is_object = o
        .and_then(|value| value.string_units())
        .is_some_and(|units| type_is(&units, "object"));
    let o_is_array = o
        .and_then(|value| value.string_units())
        .is_some_and(|units| type_is(&units, "array"));
    if !o_is_object
        && (field(e, "properties").is_some()
            || field(e, "required").is_some()
            || field(e, "additionalProperties").is_some())
    {
        return Err("mismatched_keywords");
    }
    if !o_is_array && field(e, "items").is_some() {
        return Err("mismatched_keywords");
    }
    if o_is_object {
        let Some(properties) = field(e, "properties") else {
            return Err("no_properties");
        };
        if properties.kind() != FlatJsonKind::Object {
            return Err("no_properties");
        }
        if let Some(additional) = field(e, "additionalProperties") {
            if additional.kind() != FlatJsonKind::Boolean
                || additional.primitive_source() != Some("false")
            {
                return Err("additional_properties");
            }
        }
        let property_entries = entries_last_by_key(properties).ok_or("no_properties")?;
        if let Some(required) = field(e, "required") {
            let Some(required_entries) = source_array(required) else {
                return Err("invalid_required");
            };
            let valid_strings = required_entries
                .iter()
                .map(|node| node.string_units())
                .collect::<Option<Vec<_>>>();
            let Some(required_names) = valid_strings else {
                return Err("invalid_required");
            };
            let duplicate = {
                let mut seen = HashSet::with_capacity(required_names.len());
                required_names.iter().any(|name| !seen.insert(name.clone()))
            };
            let all_present = required_names.iter().all(|name| {
                property_entries
                    .iter()
                    .any(|property| property.key_units == name.as_slice())
            });
            if duplicate || !all_present {
                return Err("invalid_required");
            }
            push_field(&mut fields, "required", payload_from_node(required)?);
        }
        let mut strict_properties = Vec::with_capacity(property_entries.len());
        for property in property_entries {
            strict_properties.push((
                property.key_units.to_vec(),
                convert(property.value, depth - 1, remaining)?,
            ));
        }
        push_field(
            &mut fields,
            "properties",
            object_payload(&strict_properties)?,
        );
        push_field(
            &mut fields,
            "additionalProperties",
            JsonPayload::parse("false").map_err(|_| "invalid_schema")?,
        );
    } else if o_is_array {
        let Some(items) = field(e, "items") else {
            return Err("unsupported_items");
        };
        if items.kind() == FlatJsonKind::Array {
            return Err("unsupported_items");
        }
        push_field(&mut fields, "items", convert(items, depth - 1, remaining)?);
    } else if o.is_none() && field(e, "enum").is_none() && field(e, "const").is_none() {
        return Err("missing_type");
    }
    object_payload(&fields)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Existing cases use bounded locally-generated fixtures; production callers
    /// use the source-only converter above.
    fn strict_schema(schema: &serde_json::Value) -> Result<serde_json::Value, &'static str> {
        let source = JsonPayload::from_generated_value(schema).map_err(|_| "invalid_schema")?;
        let strict = super::to_strict_schema(&source)?;
        serde_json::from_str(strict.as_json_source()).map_err(|_| "invalid_schema")
    }

    #[test]
    fn source_transform_preserves_lone_surrogate_property_and_description() {
        let schema = JsonPayload::parse(
            r#"{"type":"object","properties":{"\ud800":{"type":"string","description":"before\ud800after"}}}"#,
        )
        .unwrap();
        let strict = super::to_strict_schema(&schema).unwrap();
        let scanner = strict.scanner().unwrap();
        let properties = field(scanner.root_node(), "properties").unwrap();
        let property = entries_last_by_key(properties).unwrap().pop().unwrap();
        assert_eq!(property.key_units, &[0xd800]);
        let description = field(property.value, "description").unwrap();
        assert_eq!(
            description.string_units(),
            Some(
                "before"
                    .encode_utf16()
                    .chain([0xd800])
                    .chain("after".encode_utf16())
                    .collect()
            )
        );
    }

    #[test]
    fn source_enum_duplicate_check_uses_javascript_number_equality() {
        let schema =
            JsonPayload::parse(r#"{"type":"object","properties":{"x":{"enum":[1,1.0]}}}"#).unwrap();
        assert_eq!(
            super::to_strict_schema(&schema).unwrap_err(),
            "unsupported_enum"
        );
    }

    #[test]
    fn source_keywords_compare_exact_utf16_names() {
        let schema =
            JsonPayload::parse(r#"{"type":"object","properties":{},"\ud800":true}"#).unwrap();
        assert_eq!(
            super::to_strict_schema(&schema).unwrap_err(),
            "unsupported_keyword"
        );
    }

    #[test]
    fn source_non_finite_const_and_enum_are_rejected() {
        for (source, reason) in [
            (
                r#"{"type":"object","properties":{"x":{"const":1e400}}}"#,
                "unsupported_const",
            ),
            (
                r#"{"type":"object","properties":{"x":{"const":-1e400}}}"#,
                "unsupported_const",
            ),
            (
                r#"{"type":"object","properties":{"x":{"enum":[1e400]}}}"#,
                "unsupported_enum",
            ),
        ] {
            let schema = JsonPayload::parse(source).unwrap();
            assert_eq!(super::to_strict_schema(&schema).unwrap_err(), reason);
        }
    }

    #[test]
    fn source_property_traversal_uses_javascript_index_order() {
        let schema = JsonPayload::parse(
            r#"{"type":"object","properties":{"2":{"type":"bad"},"1":{"items":{}}}}"#,
        )
        .unwrap();
        assert_eq!(
            super::to_strict_schema(&schema).unwrap_err(),
            "mismatched_keywords"
        );
    }

    #[test]
    fn source_output_preserves_javascript_property_enumeration_bytes() {
        let schema = JsonPayload::parse(
            r#"{"type":"object","properties":{"2":{"type":"string"},"1":{"type":"number"},"01":{"type":"boolean"}}}"#,
        ).unwrap();
        let strict = super::to_strict_schema(&schema).unwrap();
        assert_eq!(
            strict.as_json_source(),
            r#"{"type":"object","properties":{"1":{"type":"number"},"2":{"type":"string"},"01":{"type":"boolean"}},"additionalProperties":false}"#
        );
    }

    #[test]
    fn simple_object_gets_additional_properties_false() {
        let schema = json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"]
        });
        let strict = strict_schema(&schema).unwrap();
        assert_eq!(
            strict,
            json!({
                "type": "object",
                "required": ["name"],
                "properties": { "name": { "type": "string" } },
                "additionalProperties": false
            })
        );
    }

    #[test]
    fn nested_objects_and_arrays_are_closed_recursively() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": { "type": "object", "properties": { "id": { "type": "integer" } } }
                }
            }
        });
        let strict = strict_schema(&schema).unwrap();
        // Every object node closed with additionalProperties:false.
        assert_eq!(strict["additionalProperties"], json!(false));
        assert_eq!(
            strict["properties"]["items"]["items"]["additionalProperties"],
            json!(false)
        );
    }

    #[test]
    fn strips_schema_keyword_but_permits_it() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {}
        });
        let strict = strict_schema(&schema).unwrap();
        assert!(strict.get("$schema").is_none());
    }

    #[test]
    fn root_must_be_object() {
        assert_eq!(
            strict_schema(&json!({ "type": "string" })).unwrap_err(),
            "root_not_object"
        );
    }

    #[test]
    fn rejects_unsupported_keyword_and_type() {
        assert_eq!(
            strict_schema(&json!({ "type": "object", "properties": {}, "minProperties": 1 }))
                .unwrap_err(),
            "unsupported_keyword"
        );
        assert_eq!(
            strict_schema(&json!({ "type": "tuple" })).unwrap_err(),
            "unsupported_type"
        );
    }

    #[test]
    fn rejects_open_additional_properties() {
        assert_eq!(
            strict_schema(
                &json!({ "type": "object", "properties": {}, "additionalProperties": true })
            )
            .unwrap_err(),
            "additional_properties"
        );
    }

    #[test]
    fn rejects_mismatched_keywords() {
        // items on a non-array.
        assert_eq!(
            strict_schema(&json!({ "type": "object", "properties": {}, "items": {} })).unwrap_err(),
            "mismatched_keywords"
        );
    }

    #[test]
    fn rejects_bad_required_and_missing_properties() {
        assert_eq!(
            strict_schema(&json!({ "type": "object", "required": ["x"], "properties": {} }))
                .unwrap_err(),
            "invalid_required"
        );
        assert_eq!(
            strict_schema(&json!({ "type": "object" })).unwrap_err(),
            "no_properties"
        );
    }

    #[test]
    fn accepts_enum_const_anyof() {
        // anyOf root is a valid node (though not an object root → root_not_object).
        assert_eq!(
            strict_schema(&json!({ "anyOf": [ { "type": "string" } ] })).unwrap_err(),
            "root_not_object"
        );
        // enum-only property node.
        let schema = json!({
            "type": "object",
            "properties": { "color": { "enum": ["r", "g", "b"] } }
        });
        let strict = strict_schema(&schema).unwrap();
        assert_eq!(
            strict["properties"]["color"]["enum"],
            json!(["r", "g", "b"])
        );
    }

    #[test]
    fn rejects_enum_with_duplicates_or_non_primitive() {
        assert_eq!(
            strict_schema(&json!({ "type": "object", "properties": { "x": { "enum": [1, 1] } } }))
                .unwrap_err(),
            "unsupported_enum"
        );
        assert_eq!(
            strict_schema(
                &json!({ "type": "object", "properties": { "x": { "enum": [{ "a": 1 }] } } })
            )
            .unwrap_err(),
            "unsupported_enum"
        );
    }

    #[test]
    fn rejects_multi_type_with_object_or_array() {
        // A type array containing "object" is disallowed.
        assert_eq!(
            strict_schema(&json!({ "type": ["object", "string"] })).unwrap_err(),
            "unsupported_type"
        );
        // A scalar multi-type is fine as a node (but not as the object root).
        assert_eq!(
            strict_schema(&json!({ "type": ["string", "integer"] })).unwrap_err(),
            "root_not_object"
        );
    }

    #[test]
    fn enforces_depth_budget() {
        // Build 40 nested single-property objects → exceeds MAX_DEPTH (32).
        let mut schema =
            json!({ "type": "object", "properties": { "leaf": { "type": "string" } } });
        for _ in 0..40 {
            schema = json!({ "type": "object", "properties": { "child": schema } });
        }
        assert_eq!(strict_schema(&schema).unwrap_err(), "max_depth");
    }
}
