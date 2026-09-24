//! Native attribution metadata is retained alongside a convenience source list.
use crate::protocol::{StreamEvent, WebCitation, WebSearchResult};
use serde_json::{json, Value};

fn citations(value: &Value, out: &mut Vec<WebCitation>) {
    match value {
        Value::Array(values) => values.iter().for_each(|v| citations(v, out)),
        Value::Object(object) => {
            let source = match value.get("type").and_then(Value::as_str) {
                Some("url_citation") => Some(value.get("url_citation").unwrap_or(value)),
                Some("web_search_result_location" | "web_search_result" | "url") => Some(value),
                _ if value.get("link").is_some() => Some(value),
                _ => value.get("web"),
            };
            if let Some(source) = source {
                if let Some(url) = source
                    .get("url")
                    .or_else(|| source.get("uri"))
                    .or_else(|| source.get("link"))
                    .and_then(Value::as_str)
                {
                    let title = source
                        .get("title")
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                    if !out.iter().any(|c| c.url == url && c.title == title) {
                        out.push(WebCitation {
                            url: url.to_owned(),
                            title,
                        });
                    }
                }
            }
            for nested in object.values() {
                citations(nested, out);
            }
        }
        _ => {}
    }
}

pub(crate) fn result(metadata: Value) -> Option<WebSearchResult> {
    if metadata.as_object().is_none_or(|o| o.is_empty()) {
        return None;
    }
    let mut sources = Vec::new();
    citations(&metadata, &mut sources);
    Some(WebSearchResult {
        citations: sources,
        metadata,
    })
}

pub(crate) fn chat(message: &Value) -> Option<WebSearchResult> {
    let annotations: Vec<_> = message
        .get("annotations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|a| a.get("type").and_then(Value::as_str) == Some("url_citation"))
        .cloned()
        .collect();
    if annotations.is_empty() {
        None
    } else {
        result(json!({"annotations": annotations}))
    }
}

/// GLM search results live at the response/frame root, including frames with
/// no choices. Keep native `refer` markers, snippets and dates for attribution.
pub(crate) fn chat_with_search(message: &Value, body: &Value) -> Option<WebSearchResult> {
    let mut metadata = chat(message)
        .map(|s| s.metadata)
        .unwrap_or_else(|| json!({}));
    if let Some(search) = body.get("web_search").filter(|v| !v.is_null()) {
        metadata["web_search"] = search.clone();
    }
    result(metadata)
}

pub(crate) fn responses(body: &Value) -> Option<WebSearchResult> {
    let mut metadata = serde_json::Map::new();
    if let Some(counts) = body.get("server_side_tool_usage").filter(|v| !v.is_null()) {
        metadata.insert("server_side_tool_usage".into(), counts.clone());
    }
    let mut annotations = Vec::new();
    let mut calls = Vec::new();
    for (output_index, item) in body
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        if item.get("type").and_then(Value::as_str) == Some("web_search_call") {
            calls.push(item.clone());
        }
        for (content_index, part) in item
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            for annotation in part
                .get("annotations")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if annotation.get("type").and_then(Value::as_str) == Some("url_citation") {
                    annotations.push(json!({"output_index":output_index,"content_index":content_index,"annotation":annotation}));
                }
            }
        }
    }
    if !annotations.is_empty() {
        metadata.insert("annotations".into(), annotations.into());
    }
    if !calls.is_empty() {
        metadata.insert("web_search_calls".into(), calls.into());
    }
    result(metadata.into())
}

pub(crate) fn anthropic(body: &Value) -> Option<WebSearchResult> {
    let blocks: Vec<_> = body
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|b| {
            is_anthropic_search_block(b)
                || b.get("citations")
                    .and_then(Value::as_array)
                    .is_some_and(|cs| {
                        cs.iter().any(|c| {
                            c.get("type").and_then(Value::as_str)
                                == Some("web_search_result_location")
                        })
                    })
        })
        .cloned()
        .collect();
    if blocks.is_empty() {
        None
    } else {
        result(json!({"content":blocks}))
    }
}

pub(crate) fn is_anthropic_search_block(block: &Value) -> bool {
    block.get("type").and_then(Value::as_str) == Some("web_search_tool_result")
        || (block.get("type").and_then(Value::as_str) == Some("server_tool_use")
            && block.get("name").and_then(Value::as_str) == Some("web_search"))
}

pub(crate) fn gemini(candidate: &Value) -> Option<WebSearchResult> {
    candidate
        .get("groundingMetadata")
        .filter(|m| !m.is_null())
        .and_then(|m| result(json!({"groundingMetadata":m})))
}

/// Array records are per-frame additions; repeated terminal records are suppressed.
/// Non-array metadata (Gemini groundingMetadata) is a native snapshot; changed
/// snapshots are emitted intact so chunk indices and grounding supports stay aligned.
#[derive(Debug, Default)]
pub(crate) struct SearchStream {
    seen: Vec<(String, Value)>,
    snapshots: serde_json::Map<String, Value>,
}
impl SearchStream {
    fn record(&mut self, key: &str, value: &Value) -> bool {
        if self.seen.iter().any(|(k, v)| k == key && v == value) {
            return false;
        }
        self.seen.push((key.to_owned(), value.clone()));
        true
    }

    pub(crate) fn emit(&mut self, search: Option<WebSearchResult>, out: &mut Vec<StreamEvent>) {
        let Some(search) = search else {
            return;
        };
        let mut metadata = serde_json::Map::new();
        for (key, value) in search.metadata.as_object().into_iter().flatten() {
            if let Some(records) = value.as_array() {
                let fresh: Vec<_> = records
                    .iter()
                    .filter(|v| self.record(key, v))
                    .cloned()
                    .collect();
                if !fresh.is_empty() {
                    metadata.insert(key.clone(), fresh.into());
                }
            } else if self.snapshots.get(key) != Some(value) {
                self.snapshots.insert(key.clone(), value.clone());
                metadata.insert(key.clone(), value.clone());
            }
        }
        if let Some(result) = result(metadata.into()) {
            out.push(StreamEvent::WebSearch { result });
        }
    }
}

/// Preserve provider server-search counters without mixing them into token usage.
pub(crate) fn with_usage(
    search: Option<WebSearchResult>,
    usage: Option<&Value>,
) -> Option<WebSearchResult> {
    let mut metadata = search.map(|s| s.metadata).unwrap_or_else(|| json!({}));
    let mut counters = serde_json::Map::new();
    if let Some(usage) = usage {
        if let Some(value) = usage.pointer("/server_tool_use").filter(|value| {
            value
                .get("web_search_requests")
                .is_some_and(|count| count.as_u64().is_some())
        }) {
            // Preserve all provider counters (for example web_fetch_requests)
            // while only exposing this metadata when an actual web search ran.
            counters.insert("server_tool_use".into(), value.clone());
        }
        if let Some(value) = usage.get("server_side_tool_usage").filter(|value| {
            value.get("web_search").is_some_and(|v| !v.is_null())
                || value
                    .get("web_search_requests")
                    .is_some_and(|v| v.as_u64().is_some())
                || value
                    .get("web_search_calls")
                    .is_some_and(|v| v.as_u64().is_some())
        }) {
            counters.insert("server_side_tool_usage".into(), value.clone());
        }
        if let Some(value) = usage
            .pointer("/x_tools/web_search")
            .filter(|value| !value.is_null())
        {
            counters.insert("x_tools".into(), json!({"web_search":value}));
        }
    }
    if !counters.is_empty() {
        metadata["usage"] = counters.into();
    }
    result(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_search_usage_does_not_claim_that_web_search_occurred() {
        assert!(with_usage(
            None,
            Some(&json!({
                "x_tools":{"file_search":{"count":1}}
            }))
        )
        .is_none());
    }
}
