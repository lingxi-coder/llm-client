//! Qwen Responses knowledge-base search decoding.

use lingxi_agent_api::protocol::{FileSearchHit, FileSearchResult, StreamEvent};
use serde_json::{json, Value};
use std::collections::BTreeSet;

pub(crate) fn responses(body: &Value) -> Option<FileSearchResult> {
    let calls: Vec<_> = body
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("file_search_call"))
        .cloned()
        .collect();
    result_from_calls(calls)
}

fn result_from_calls(calls: Vec<Value>) -> Option<FileSearchResult> {
    if calls.is_empty() {
        return None;
    }
    let mut queries = Vec::new();
    let mut hits = Vec::new();
    for call in &calls {
        queries.extend(
            call.get("queries")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned),
        );
        hits.extend(
            call.get("results")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(decode_hit),
        );
    }
    Some(FileSearchResult {
        queries,
        hits,
        metadata: json!({"file_search_calls": calls}),
    })
}

fn decode_hit(value: &Value) -> Option<FileSearchHit> {
    let file_id = value
        .get("file_id")
        .or_else(|| value.get("id"))
        .and_then(Value::as_str)?
        .to_owned();
    Some(FileSearchHit {
        file_id,
        filename: value
            .get("filename")
            .and_then(Value::as_str)
            .map(str::to_owned),
        score: value.get("score").and_then(Value::as_f64),
        text: value.get("text").and_then(Value::as_str).map(str::to_owned),
    })
}

#[derive(Debug, Default)]
pub(crate) struct FileSearchStream {
    emitted_calls: BTreeSet<String>,
}

impl FileSearchStream {
    pub(crate) fn emit(&mut self, body: &Value, out: &mut Vec<StreamEvent>) {
        let Some(result) = responses(body) else {
            return;
        };
        let Some(calls) = result
            .metadata
            .get("file_search_calls")
            .and_then(Value::as_array)
        else {
            return;
        };
        for call in calls {
            let fingerprint = call.to_string();
            if !self.emitted_calls.insert(fingerprint) {
                continue;
            }
            let Some(event) = result_from_calls(vec![call.clone()]) else {
                continue;
            };
            out.push(StreamEvent::FileSearch { result: event });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_results_keep_queries_and_provider_metadata() {
        let result = responses(&json!({"output":[{"type":"file_search_call","queries":["q"],"results":[{"file_id":"file-1","filename":"a.txt","score":0.9,"text":"hit"}]}]})).unwrap();
        assert_eq!(result.queries, ["q"]);
        assert_eq!(result.hits[0].file_id, "file-1");
        assert_eq!(result.hits[0].filename.as_deref(), Some("a.txt"));
        assert_eq!(result.hits[0].score, Some(0.9));
        assert_eq!(result.hits[0].text.as_deref(), Some("hit"));
    }

    #[test]
    fn streaming_completion_deduplicates_output_item_and_final_response() {
        let call =
            json!({"type":"file_search_call","queries":["q"],"results":[{"file_id":"file-1"}]});
        let mut stream = FileSearchStream::default();
        let mut events = Vec::new();
        stream.emit(&json!({"output":[call.clone()]}), &mut events);
        stream.emit(&json!({"output":[call]}), &mut events);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::FileSearch { .. }));
    }
}
