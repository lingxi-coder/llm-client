//! Typed output contracts and local validation of completed responses.
use super::{ChatResponse, ContentBlock, StopReason};
use crate::protocol::JsonPayload;
use crate::protocol::JsonPayloadFragment;
use serde::Deserializer;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::Value;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, OnceLock},
};

/// Output constraints are independent of a function tool's `strict` setting.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: JsonPayload,
        /// Whether the provider must enforce its schema-constrained mode.
        strict: bool,
    },
}

impl<'de> Deserialize<'de> for OutputFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let scanner = crate::response_json::ResponseJsonScanner::parse(
            raw.get().as_bytes(),
            "output format is not valid JSON",
        )
        .map_err(serde::de::Error::custom)?;
        if scanner.kind_at("") != Some(crate::response_json::FlatJsonKind::Object) {
            return Err(serde::de::Error::custom("output format must be an object"));
        }
        match scanner.string_at("/type") {
            Some("text") => Ok(Self::Text),
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
                    Some("true") => true,
                    Some("false") => false,
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
                "output format has an unknown type",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StructuredOutputErrorKind {
    #[error("the response is not a completed final answer")]
    Incomplete,
    #[error("a JSON output contract is required")]
    NoContract,
    #[error("invalid JSON: {0}")]
    InvalidJson(String),
    #[error("invalid output schema: {0}")]
    InvalidSchema(String),
    #[error("output does not satisfy the schema: {0}")]
    SchemaMismatch(String),
    #[error("cannot deserialize the output: {0}")]
    Deserialization(String),
}

/// A failed validation never discards response text, usage, or provider metadata.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{kind}")]
pub struct StructuredOutputError {
    pub kind: StructuredOutputErrorKind,
    pub response: Box<ChatResponse>,
}

impl ChatResponse {
    /// Validate only a final response. Streaming deltas are not complete JSON.
    /// The returned payload keeps the response's JSON source without building a recursive Value.
    pub fn structured_json(
        &self,
        format: &OutputFormat,
    ) -> Result<JsonPayload, StructuredOutputError> {
        self.parse_structured(format)
            .map_err(|kind| StructuredOutputError {
                kind,
                response: Box::new(self.clone()),
            })
    }

    pub fn structured<T: DeserializeOwned>(
        &self,
        format: &OutputFormat,
    ) -> Result<T, StructuredOutputError> {
        let payload = self.structured_json(format)?;
        serde_json::from_str(payload.as_json_source()).map_err(|error| StructuredOutputError {
            kind: StructuredOutputErrorKind::Deserialization(error.to_string()),
            response: Box::new(self.clone()),
        })
    }

    fn parse_structured(
        &self,
        format: &OutputFormat,
    ) -> Result<JsonPayload, StructuredOutputErrorKind> {
        if !matches!(
            self.stop_reason,
            StopReason::EndTurn | StopReason::StopSequence
        ) || self
            .message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
        {
            return Err(StructuredOutputErrorKind::Incomplete);
        }
        if matches!(format, OutputFormat::Text) {
            return Err(StructuredOutputErrorKind::NoContract);
        }
        let fragments = self
            .message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. } => Some(JsonPayloadFragment::from_text(text)),
                ContentBlock::TextJsUtf16 {
                    utf16_code_units, ..
                } => Some(JsonPayloadFragment::from_utf16_code_units(
                    utf16_code_units.clone(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        let payload = JsonPayload::parse_fragments(fragments.iter())
            .map_err(|error| StructuredOutputErrorKind::InvalidJson(error.to_string()))?;
        match format {
            OutputFormat::JsonObject if !payload.is_root_object() => {
                return Err(StructuredOutputErrorKind::SchemaMismatch(
                    "expected a JSON object".into(),
                ));
            }
            OutputFormat::JsonSchema { schema, .. } => {
                let validator =
                    compile_schema(schema).map_err(StructuredOutputErrorKind::InvalidSchema)?;
                let value = materialize_jsonschema_value(&payload).map_err(|message| {
                    StructuredOutputErrorKind::InvalidSchema(format!(
                        "local JSON Schema backend cannot represent response JSON: {message}"
                    ))
                })?;
                validator.validate(&value).map_err(|error| {
                    StructuredOutputErrorKind::SchemaMismatch(error.to_string())
                })?;
            }
            _ => {}
        }
        Ok(payload)
    }
}

/// Reuse only compilation, never request- or provider-specific contract checks.
/// No remote reference fetching, file access, or implicit schema rewriting.
fn compile_schema(schema: &JsonPayload) -> Result<Arc<jsonschema::Validator>, String> {
    schema_cache().compile(schema)
}

/// Validate the source-level constraints for a schema before sending it to a
/// provider. Provider encoding keeps the schema as `JsonPayload`; it must not
/// materialize the source through the recursive local response validator.
/// Strict provider vocabularies are checked by their codec-specific flat
/// walkers. This check bounds the exact bytes, requires an object root, and
/// rejects remote references without following local references.
pub(crate) fn validate_request_schema(schema: &JsonPayload) -> Result<(), String> {
    if schema.as_json_source().len() > MAX_SCHEMA_BYTES {
        return Err("output schema exceeds 1 MiB".into());
    }
    use crate::response_json::FlatJsonKind;

    const OBJECT_SCHEMA_MAPS: &[&str] = &[
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    ];
    const SINGLE_SCHEMAS: &[&str] = &[
        "items",
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "propertyNames",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
    ];
    const SCHEMA_ARRAYS: &[&str] = &["anyOf", "oneOf", "allOf", "prefixItems"];
    const REFERENCE_KEYS: &[&str] = &["$ref", "$dynamicRef", "$recursiveRef"];

    let scanner = schema.scanner().map_err(|error| error.to_string())?;
    let root = scanner.root_node();
    if root.kind() != FlatJsonKind::Object {
        return Err("JSON schema must be an object".into());
    }

    let key_is =
        |units: &[u16], key: &str| units == key.encode_utf16().collect::<Vec<_>>().as_slice();
    let mut pending = vec![root];
    while let Some(node) = pending.pop() {
        let Some(entries) = node.object_entries() else {
            continue;
        };

        for entry in &entries {
            if REFERENCE_KEYS
                .iter()
                .any(|key| key_is(entry.key_units, key))
            {
                let units = entry
                    .value
                    .string_units()
                    .ok_or_else(|| "schema references must be strings".to_owned())?;
                let reference = String::from_utf16(&units)
                    .map_err(|_| "schema references must use valid Unicode".to_owned())?;
                if !reference.starts_with('#') {
                    return Err("only local schema references are permitted".into());
                }
            }

            if OBJECT_SCHEMA_MAPS
                .iter()
                .any(|key| key_is(entry.key_units, key))
            {
                if let Some(children) = entry.value.object_entries() {
                    pending.extend(children.into_iter().map(|child| child.value));
                }
            } else if SINGLE_SCHEMAS
                .iter()
                .any(|key| key_is(entry.key_units, key))
            {
                if entry.value.kind() == FlatJsonKind::Object {
                    pending.push(entry.value);
                }
            } else if SCHEMA_ARRAYS.iter().any(|key| key_is(entry.key_units, key)) {
                if let Some(length) = entry.value.array_len() {
                    pending.extend((0..length).filter_map(|index| entry.value.array_item(index)));
                }
            }
        }
    }

    Ok(())
}

fn materialize_jsonschema_value(payload: &JsonPayload) -> Result<Value, String> {
    serde_json::from_str(payload.as_json_source()).map_err(|error| {
        format!(
            "serde_json/jsonschema backend cannot materialize source (including its recursion and UTF-16 limits): {error}"
        )
    })
}

fn schema_cache() -> &'static SchemaCache {
    static CACHE: OnceLock<SchemaCache> = OnceLock::new();
    CACHE.get_or_init(SchemaCache::default)
}

const MAX_SCHEMA_BYTES: usize = 1024 * 1024;
const MAX_CACHED_SCHEMA_BYTES: usize = 4 * MAX_SCHEMA_BYTES;
const MAX_CACHED_SCHEMAS: usize = 16;

type CompiledSchema = Result<Arc<jsonschema::Validator>, String>;

struct SchemaEntry {
    schema: JsonPayload,
    bytes: usize,
    compiled: OnceLock<CompiledSchema>,
}

impl SchemaEntry {
    fn matches_schema(&self, schema: &JsonPayload) -> bool {
        self.schema.as_json_source() == schema.as_json_source()
    }
}

#[derive(Default)]
struct SchemaCache {
    entries: Mutex<VecDeque<Arc<SchemaEntry>>>,
}

impl SchemaCache {
    fn compile(&self, schema: &JsonPayload) -> CompiledSchema {
        self.compile_with(schema, build_schema)
    }

    fn compile_with(
        &self,
        schema: &JsonPayload,
        compile: impl FnOnce(&JsonPayload) -> CompiledSchema,
    ) -> CompiledSchema {
        let Some(entry) = self.entry(schema) else {
            // Large local schemas are compiled without retaining them in the shared cache.
            return compile(schema);
        };
        // Independent schemas compile concurrently. Same-schema callers share
        // initialization, including failures, without holding the cache lock.
        entry
            .compiled
            .get_or_init(|| compile(&entry.schema))
            .clone()
    }

    fn entry(&self, schema: &JsonPayload) -> Option<Arc<SchemaEntry>> {
        if let Some(entry) = self.find_cached(|entry| entry.matches_schema(schema)) {
            return Some(entry);
        }
        let bytes = schema.as_json_source().len();
        if bytes > MAX_SCHEMA_BYTES {
            return None;
        }
        let candidate = Arc::new(SchemaEntry {
            schema: schema.clone(),
            bytes,
            compiled: OnceLock::new(),
        });
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = find_schema(&mut entries, schema) {
            return Some(entry);
        }
        let mut cached_bytes: usize = entries.iter().map(|entry| entry.bytes).sum();
        let mut evicted_entries = Vec::new();
        while entries.len() >= MAX_CACHED_SCHEMAS || cached_bytes + bytes > MAX_CACHED_SCHEMA_BYTES
        {
            if let Some(evicted) = entries.pop_back() {
                cached_bytes -= evicted.bytes;
                evicted_entries.push(evicted);
            }
        }
        entries.push_front(Arc::clone(&candidate));
        drop(entries);
        // Dropping a compiled validator can be costly; unrelated cache hits
        // must not wait for destruction of evicted entries.
        drop(evicted_entries);
        Some(candidate)
    }

    fn find_cached(
        &self,
        mut matches: impl FnMut(&SchemaEntry) -> bool,
    ) -> Option<Arc<SchemaEntry>> {
        let candidates: [Option<Arc<SchemaEntry>>; MAX_CACHED_SCHEMAS] = {
            let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            std::array::from_fn(|index| entries.get(index).map(Arc::clone))
        };
        // Large contracts can take longer to compare than this cache's lock
        // bookkeeping. Compare immutable snapshots concurrently, not under the
        // global mutex; even an entry evicted meanwhile remains safe to use.
        let entry = candidates
            .into_iter()
            .flatten()
            .find(|entry| matches(entry))?;
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(index) = entries
            .iter()
            .position(|cached| Arc::ptr_eq(cached, &entry))
        {
            let cached = entries.remove(index).expect("entry index is in bounds");
            entries.push_front(cached);
        }
        Some(entry)
    }
}

fn find_schema(
    entries: &mut VecDeque<Arc<SchemaEntry>>,
    schema: &JsonPayload,
) -> Option<Arc<SchemaEntry>> {
    let index = entries
        .iter()
        .position(|entry| entry.matches_schema(schema))?;
    let entry = entries.remove(index)?;
    entries.push_front(Arc::clone(&entry));
    Some(entry)
}

fn build_schema(schema: &JsonPayload) -> CompiledSchema {
    let value = materialize_jsonschema_value(schema)?;
    if !value.is_object() {
        return Err("JSON schema must be an object".into());
    }
    jsonschema::options()
        .with_retriever(NoExternalSchemas)
        .with_draft(jsonschema::Draft::Draft202012)
        .should_validate_formats(true)
        .should_ignore_unknown_formats(false)
        .build(&value)
        .map(Arc::new)
        .map_err(|error| error.to_string())
}

struct NoExternalSchemas;
impl jsonschema::Retrieve for NoExternalSchemas {
    fn retrieve(
        &self,
        _uri: &jsonschema::Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err("external schema resolution is disabled".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Barrier,
    };

    fn payload(value: Value) -> JsonPayload {
        JsonPayload::from_generated_value(&value).unwrap()
    }

    #[test]
    fn concurrent_schema_compilation_is_shared() {
        let cache = SchemaCache::default();
        let schema = payload(json!({"type":"integer","minimum":3}));
        let compilations = AtomicUsize::new(0);
        let start = Barrier::new(8);
        let validators = std::thread::scope(|scope| {
            (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        cache
                            .compile_with(&schema, |schema| {
                                compilations.fetch_add(1, Ordering::SeqCst);
                                build_schema(schema)
                            })
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(compilations.load(Ordering::SeqCst), 1);
        assert!(validators
            .iter()
            .all(|validator| Arc::ptr_eq(validator, &validators[0])));
    }

    #[test]
    fn changed_schema_source_compiles_a_distinct_validator() {
        let cache = SchemaCache::default();
        let schema = payload(json!({"type":"integer","minimum":3}));
        let old = cache.compile(&schema).unwrap();
        let copied = cache.compile(&schema.clone()).unwrap();
        assert!(Arc::ptr_eq(&old, &copied));
        let changed = payload(json!({"type":"integer","minimum":5}));
        let new = cache.compile(&changed).unwrap();
        assert!(!Arc::ptr_eq(&old, &new));
        assert!(old.is_valid(&json!(4)));
        assert!(!new.is_valid(&json!(4)));
    }

    #[test]
    fn source_equality_keeps_signed_zero_encodings_distinct() {
        let cache = SchemaCache::default();
        let positive =
            JsonPayload::parse(r#"{"type":"object","examples":[{"nested":[0.0]}]}"#).unwrap();
        let negative =
            JsonPayload::parse(r#"{"type":"object","examples":[{"nested":[-0.0]}]}"#).unwrap();
        assert_ne!(positive.as_json_source(), negative.as_json_source());
        let first = cache.compile(&positive).unwrap();
        // This is also the lookup used to deduplicate concurrent cold misses.
        assert!(find_schema(&mut cache.entries.lock().unwrap(), &negative).is_none());
        let second = cache.compile(&negative).unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(Arc::ptr_eq(&first, &cache.compile(&positive).unwrap()));
        assert!(Arc::ptr_eq(&second, &cache.compile(&negative).unwrap()));
    }

    #[test]
    fn warm_lookup_compares_source_outside_the_cache_lock_and_survives_eviction() {
        let cache = SchemaCache::default();
        let schema = payload(json!({"type":"integer","minimum":3}));
        let validator = cache.compile(&schema).unwrap();
        let entry = cache
            .find_cached(|entry| {
                // Simulate eviction while the source comparison is in progress.
                // Acquiring the lock here also proves source equality cannot
                // serialize concurrent warm lookups behind the global mutex.
                cache
                    .entries
                    .try_lock()
                    .expect("warm source comparisons must run outside the cache lock")
                    .clear();
                entry.schema == schema
            })
            .unwrap();
        let cached = entry.compiled.get().unwrap().as_ref().unwrap();
        assert!(Arc::ptr_eq(cached, &validator));
        assert!(cached.is_valid(&json!(4)));
        assert!(cache.entries.lock().unwrap().is_empty());
    }

    #[test]
    fn compile_failures_are_shared_without_resolving_external_references() {
        let cache = SchemaCache::default();
        let schema = JsonPayload::parse(r#"{"$ref":"https://schema.invalid/example"}"#).unwrap();
        let compilations = AtomicUsize::new(0);
        for _ in 0..2 {
            let error = cache
                .compile_with(&schema, |schema| {
                    compilations.fetch_add(1, Ordering::SeqCst);
                    build_schema(schema)
                })
                .unwrap_err();
            assert!(error.contains("external schema resolution is disabled"));
        }
        assert_eq!(compilations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cache_evicts_old_entries_without_invalidating_live_validators() {
        let cache = SchemaCache::default();
        let first_schema = payload(json!({"const":0}));
        let first = cache.compile(&first_schema).unwrap();
        for n in 1..=MAX_CACHED_SCHEMAS {
            cache.compile(&payload(json!({"const":n}))).unwrap();
        }
        assert_eq!(cache.entries.lock().unwrap().len(), MAX_CACHED_SCHEMAS);
        assert!(first.is_valid(&json!(0)));
        assert!(!Arc::ptr_eq(&first, &cache.compile(&first_schema).unwrap()));
    }

    #[test]
    fn oversized_source_is_compiled_without_being_retained_in_the_cache() {
        let cache = SchemaCache::default();
        let schema = JsonPayload::parse(&format!(
            r#"{{"description":"{}"}}"#,
            "x".repeat(MAX_SCHEMA_BYTES)
        ))
        .unwrap();
        assert!(cache.entry(&schema).is_none());
    }

    #[test]
    fn tagged_output_format_deserializes_deep_source_without_value_buffering() {
        let nested = format!("{}0{}", "[".repeat(512), "]".repeat(512));
        let source = format!(
            r#"{{"type":"json_schema","name":"answer","schema":{{"examples":[{nested}]}},"strict":true}}"#
        );
        let decoded: OutputFormat = serde_json::from_str(&source).unwrap();
        let OutputFormat::JsonSchema { schema, .. } = decoded else {
            panic!("schema variant expected")
        };
        assert!(schema.as_json_source().contains(&nested));
    }

    #[test]
    fn tagged_output_format_keeps_escaped_lone_utf16_schema_source() {
        let source =
            r#"{"type":"json_schema","name":"answer","schema":{"\ud800":"\ud800"},"strict":true}"#;
        let decoded: OutputFormat = serde_json::from_str(source).unwrap();
        let OutputFormat::JsonSchema { schema, .. } = decoded else {
            panic!("schema variant expected")
        };
        assert_eq!(schema.as_json_source(), r#"{"\ud800":"\ud800"}"#);
        assert_eq!(
            serde_json::to_string(&OutputFormat::JsonSchema {
                name: "answer".into(),
                schema,
                strict: true,
            })
            .unwrap(),
            source
        );
    }
}
