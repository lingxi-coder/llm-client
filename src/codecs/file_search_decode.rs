//! Qwen Responses knowledge-base search decoding.

use crate::protocol::{FileSearchHit, FileSearchResult, JsonPayload, StreamEvent};
use std::collections::BTreeSet;

pub(crate) fn responses_payload(body: &JsonPayload) -> Option<FileSearchResult> {
    let len = body.array_len("/output")?;
    let mut calls = Vec::new();
    for index in 0..len {
        let pointer = format!("/output/{index}");
        if body.string_at(&format!("{pointer}/type")).as_deref() == Some("file_search_call") {
            calls.push(JsonPayload::parse(&body.raw_source_at(&pointer)?).ok()?);
        }
    }
    result_from_calls(calls)
}

fn scalar_string(value: &JsonPayload, pointer: &str) -> Option<String> {
    String::from_utf16(&value.string_units_at(pointer)?).ok()
}

fn result_from_calls(calls: Vec<JsonPayload>) -> Option<FileSearchResult> {
    if calls.is_empty() {
        return None;
    }
    let mut queries = Vec::new();
    let mut hits = Vec::new();
    for call in &calls {
        for index in 0..call.array_len("/queries").unwrap_or(0) {
            if let Some(query) = scalar_string(call, &format!("/queries/{index}")) {
                queries.push(query);
            }
        }
        for index in 0..call.array_len("/results").unwrap_or(0) {
            let pointer = format!("/results/{index}");
            let Some(hit) = decode_hit(call, &pointer) else {
                continue;
            };
            hits.push(hit);
        }
    }
    let source = format!(
        "{{\"file_search_calls\":[{}]}}",
        calls
            .iter()
            .map(JsonPayload::as_json_source)
            .collect::<Vec<_>>()
            .join(",")
    );
    Some(FileSearchResult {
        queries,
        hits,
        metadata: JsonPayload::parse(&source).ok()?,
    })
}

fn decode_hit(value: &JsonPayload, pointer: &str) -> Option<FileSearchHit> {
    let file_id = scalar_string(value, &format!("{pointer}/file_id"))
        .or_else(|| scalar_string(value, &format!("{pointer}/id")))?;
    Some(FileSearchHit {
        file_id,
        filename: scalar_string(value, &format!("{pointer}/filename")),
        score: value.number_at(&format!("{pointer}/score")),
        text: scalar_string(value, &format!("{pointer}/text")),
    })
}

#[derive(Debug, Default)]
pub(crate) struct FileSearchStream {
    emitted_calls: BTreeSet<String>,
}

impl FileSearchStream {
    pub(crate) fn emit_payload(&mut self, body: &JsonPayload, out: &mut Vec<StreamEvent>) {
        let Some(result) = responses_payload(body) else {
            return;
        };
        let Some(count) = result.metadata.array_len("/file_search_calls") else {
            return;
        };
        for index in 0..count {
            let Some(source) = result
                .metadata
                .raw_source_at(&format!("/file_search_calls/{index}"))
            else {
                continue;
            };
            let Ok(call) = JsonPayload::parse(&source) else {
                continue;
            };
            let Ok(fingerprint) = call.canonical_json(crate::exact_json::JsonEncoding::JavaScript)
            else {
                continue;
            };
            if !self.emitted_calls.insert(fingerprint) {
                continue;
            }
            let Some(event) = result_from_calls(vec![call]) else {
                continue;
            };
            out.push(StreamEvent::FileSearch { result: event });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn responses_results_keep_queries_and_provider_metadata() {
        let body = JsonPayload::parse(
            r#"{"output":[{"type":"file_search_call","queries":["q"],"results":[{"file_id":"file-1","filename":"a.txt","score":0.9,"text":"hit"}]}]}"#,
        )
        .unwrap();
        let result = responses_payload(&body).unwrap();
        assert_eq!(result.queries, ["q"]);
        assert_eq!(result.hits[0].file_id, "file-1");
        assert_eq!(result.hits[0].filename.as_deref(), Some("a.txt"));
        assert_eq!(result.hits[0].score, Some(0.9));
        assert_eq!(result.hits[0].text.as_deref(), Some("hit"));
        assert!(result
            .metadata
            .as_json_source()
            .contains("file_search_call"));
    }

    #[test]
    fn streaming_completion_deduplicates_output_item_and_final_response() {
        let call =
            json!({"type":"file_search_call","queries":["q"],"results":[{"file_id":"file-1"}]});
        let body = JsonPayload::parse(&json!({"output":[call]}).to_string()).unwrap();
        let mut stream = FileSearchStream::default();
        let mut events = Vec::new();
        stream.emit_payload(&body, &mut events);
        stream.emit_payload(&body, &mut events);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::FileSearch { .. }));
    }
}
