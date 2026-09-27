//! Typed output contracts and local validation of completed responses.
use super::{ChatResponse, ContentBlock, StopReason};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::VecDeque,
    io,
    sync::{Arc, Mutex, OnceLock},
};

/// Output constraints are independent of a function tool's `strict` setting.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        /// Whether the provider must enforce its schema-constrained mode.
        strict: bool,
    },
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
    pub fn structured_json(&self, format: &OutputFormat) -> Result<Value, StructuredOutputError> {
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
        let value = self.structured_json(format)?;
        serde_json::from_value(value).map_err(|error| StructuredOutputError {
            kind: StructuredOutputErrorKind::Deserialization(error.to_string()),
            response: Box::new(self.clone()),
        })
    }

    fn parse_structured(&self, format: &OutputFormat) -> Result<Value, StructuredOutputErrorKind> {
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
        let text: String = self
            .message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| StructuredOutputErrorKind::InvalidJson(error.to_string()))?;
        match format {
            OutputFormat::JsonObject if !value.is_object() => {
                return Err(StructuredOutputErrorKind::SchemaMismatch(
                    "expected a JSON object".into(),
                ));
            }
            OutputFormat::JsonSchema { schema, .. } => {
                let validator =
                    compile_schema(schema).map_err(StructuredOutputErrorKind::InvalidSchema)?;
                validator.validate(&value).map_err(|error| {
                    StructuredOutputErrorKind::SchemaMismatch(error.to_string())
                })?;
            }
            _ => {}
        }
        Ok(value)
    }
}

/// Reuse only compilation, never request- or provider-specific contract checks.
/// No remote reference fetching, file access, or implicit schema rewriting.
pub(crate) fn compile_schema(schema: &Value) -> Result<Arc<jsonschema::Validator>, String> {
    schema_cache().compile(schema)
}

pub(crate) fn compile_request_schema(schema: &Value) -> Result<Arc<jsonschema::Validator>, String> {
    schema_cache().compile_with(schema, true, build_schema)
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
    schema: Value,
    bytes: usize,
    has_float_zero: bool,
    compiled: OnceLock<CompiledSchema>,
}

impl SchemaEntry {
    fn matches_schema(&self, schema: &Value) -> bool {
        if self.has_float_zero {
            cache_schema_matches(&self.schema, schema)
        } else {
            // Signed floating zero is the representation-sensitive case in
            // Number equality. Keep ordinary schema hits on the usual path.
            self.schema == *schema
        }
    }
}

#[derive(Default)]
struct SchemaCache {
    entries: Mutex<VecDeque<Arc<SchemaEntry>>>,
}

impl SchemaCache {
    fn compile(&self, schema: &Value) -> CompiledSchema {
        self.compile_with(schema, false, build_schema)
    }

    fn compile_with(
        &self,
        schema: &Value,
        require_size_limit: bool,
        compile: impl FnOnce(&Value) -> CompiledSchema,
    ) -> CompiledSchema {
        let Some(entry) = self.entry(schema) else {
            // The request wire has a size limit, while local response parsing
            // historically accepts arbitrary schemas. Keep large local-only
            // contracts supported without retaining them in the global cache.
            return if require_size_limit {
                Err("output schema exceeds 1 MiB".into())
            } else {
                compile(schema)
            };
        };
        // Independent schemas compile concurrently. Same-schema callers share
        // initialization, including failures, without holding the cache lock.
        entry
            .compiled
            .get_or_init(|| compile(&entry.schema))
            .clone()
    }

    fn entry(&self, schema: &Value) -> Option<Arc<SchemaEntry>> {
        if let Some(entry) = self.find_cached(|entry| entry.matches_schema(schema)) {
            return Some(entry);
        }
        // Warm hits compare source values directly: no serialization, hashing,
        // or source cloning. Cold work remains outside the shared cache lock.
        let bytes = schema_size(schema)?;
        let candidate = Arc::new(SchemaEntry {
            schema: schema.clone(),
            bytes,
            has_float_zero: contains_float_zero(schema),
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
    schema: &Value,
) -> Option<Arc<SchemaEntry>> {
    let index = entries
        .iter()
        .position(|entry| entry.matches_schema(schema))?;
    let entry = entries.remove(index)?;
    entries.push_front(Arc::clone(&entry));
    Some(entry)
}

fn contains_float_zero(schema: &Value) -> bool {
    match schema {
        Value::Number(number) => number.is_f64() && number.as_f64() == Some(0.0),
        Value::Array(values) => values.iter().any(contains_float_zero),
        Value::Object(values) => values.values().any(contains_float_zero),
        _ => false,
    }
}

/// A cache hit reuses both the validator and its checked source size. JSON
/// equality alone is insufficient: 0.0 == -0.0, but their wire sizes differ.
/// Object order can differ without changing size or schema semantics.
fn cache_schema_matches(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            left == right && left.as_f64().map(f64::to_bits) == right.as_f64().map(f64::to_bits)
        }
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| cache_schema_matches(left, right))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, left)| {
                    right
                        .get(key)
                        .is_some_and(|right| cache_schema_matches(left, right))
                })
        }
        _ => left == right,
    }
}

fn schema_size(schema: &Value) -> Option<usize> {
    struct SizeLimit(usize);
    impl io::Write for SizeLimit {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0 = self.0.saturating_add(bytes.len());
            if self.0 > MAX_SCHEMA_BYTES {
                return Err(io::Error::other("output schema exceeds 1 MiB"));
            }
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut size = SizeLimit(0);
    serde_json::to_writer(&mut size, schema).ok()?;
    Some(size.0)
}

fn build_schema(schema: &Value) -> CompiledSchema {
    jsonschema::options()
        .with_retriever(NoExternalSchemas)
        .with_draft(jsonschema::Draft::Draft202012)
        .should_validate_formats(true)
        .should_ignore_unknown_formats(false)
        .build(schema)
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

    #[test]
    fn concurrent_schema_compilation_is_shared() {
        let cache = SchemaCache::default();
        let schema = json!({"type":"integer","minimum":3});
        let compilations = AtomicUsize::new(0);
        let start = Barrier::new(8);
        let validators = std::thread::scope(|scope| {
            (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        cache
                            .compile_with(&schema, true, |schema| {
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
    fn changed_schema_compiles_a_distinct_validator() {
        let cache = SchemaCache::default();
        let mut schema = json!({"type":"integer","minimum":3});
        let old = cache.compile(&schema).unwrap();
        let copied = cache.compile(&schema.clone()).unwrap();
        assert!(Arc::ptr_eq(&old, &copied));
        schema["minimum"] = json!(5);
        let new = cache.compile(&schema).unwrap();
        assert!(!Arc::ptr_eq(&old, &new));
        assert!(old.is_valid(&json!(4)));
        assert!(!new.is_valid(&json!(4)));
    }

    #[test]
    fn signed_zero_representations_do_not_share_cached_source_sizes() {
        let cache = SchemaCache::default();
        let positive = json!({"type":"object","examples":[{"nested":[0.0]}]});
        let negative = json!({"type":"object","examples":[{"nested":[-0.0]}]});
        assert_eq!(positive, negative);
        assert_ne!(schema_size(&positive), schema_size(&negative));
        let first = cache.compile(&positive).unwrap();
        // This is also the lookup used to deduplicate concurrent cold misses.
        assert!(find_schema(&mut cache.entries.lock().unwrap(), &negative).is_none());
        let second = cache.compile(&negative).unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(Arc::ptr_eq(&first, &cache.compile(&positive).unwrap()));
        assert!(Arc::ptr_eq(&second, &cache.compile(&negative).unwrap()));
    }

    #[test]
    fn warm_lookup_compares_without_the_cache_lock_and_survives_eviction() {
        let cache = SchemaCache::default();
        let schema = json!({"type":"integer","minimum":3});
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
        let schema = json!({"$ref":"https://schema.invalid/example"});
        let compilations = AtomicUsize::new(0);
        for _ in 0..2 {
            let error = cache
                .compile_with(&schema, true, |schema| {
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
        let first_schema = json!({"const":0});
        let first = cache.compile(&first_schema).unwrap();
        for n in 1..=MAX_CACHED_SCHEMAS {
            cache.compile(&json!({"const":n})).unwrap();
        }
        assert_eq!(cache.entries.lock().unwrap().len(), MAX_CACHED_SCHEMAS);
        assert!(first.is_valid(&json!(0)));
        assert!(!Arc::ptr_eq(&first, &cache.compile(&first_schema).unwrap()));
    }

    #[test]
    fn cache_bounds_source_bytes_and_rejects_oversized_schemas() {
        let cache = SchemaCache::default();
        for n in 0..10 {
            cache
                .entry(&json!({"const":n,"description":"x".repeat(MAX_SCHEMA_BYTES / 2)}))
                .unwrap();
        }
        let entries = cache.entries.lock().unwrap();
        assert!(entries.len() < 10);
        assert!(entries.iter().map(|entry| entry.bytes).sum::<usize>() <= MAX_CACHED_SCHEMA_BYTES);
        drop(entries);
        let error = cache
            .compile_with(
                &json!({"description":"x".repeat(MAX_SCHEMA_BYTES)}),
                true,
                build_schema,
            )
            .unwrap_err();
        assert!(error.contains("output schema exceeds 1 MiB"));
    }
}
