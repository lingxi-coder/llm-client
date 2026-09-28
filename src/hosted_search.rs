//! Provider search result projection for clients rendering hosted search.
use serde::Serialize;
use serde_json::{json, Value};
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum SearchResultEntry {
    /// Free-form text accumulated from one or more consecutive `text` blocks
    /// (also used for the `Web search error: <code>` string emitted on a
    /// `web_search_tool_result` error payload).
    Text(String),
    /// Structured search result from a `web_search_tool_result` block, shaped
    /// `{ "tool_use_id": <id>, "content": [{ "title", "url" }, ...] }`.
    Hit(Value),
}
pub fn parse_response_content(content: &[Value]) -> Vec<SearchResultEntry> {
    let mut out: Vec<SearchResultEntry> = Vec::new();
    let mut text_acc = String::new();
    let mut in_text = true;

    for original in content {
        let block = original
            .get("value")
            .filter(|_| original.get("type").and_then(Value::as_str) == Some("provider_content"))
            .unwrap_or(original);
        match block.get("type").and_then(Value::as_str).unwrap_or("") {
            "server_tool_use" => {
                if in_text {
                    in_text = false;
                    let trimmed = text_acc.trim();
                    if !trimmed.is_empty() {
                        out.push(SearchResultEntry::Text(trimmed.to_string()));
                    }
                    text_acc.clear();
                }
            }
            "web_search_tool_result" => match block.get("content") {
                // Success case — `content` is an array of search hits.
                Some(Value::Array(items)) => {
                    let hits: Vec<Value> = items
                        .iter()
                        .map(|r| {
                            json!({
                                "title": r.get("title").cloned().unwrap_or(Value::Null),
                                "url": r.get("url").cloned().unwrap_or(Value::Null),
                            })
                        })
                        .collect();
                    out.push(SearchResultEntry::Hit(json!({
                        "tool_use_id": block.get("tool_use_id").cloned().unwrap_or(Value::Null),
                        "content": hits,
                    })));
                }
                // Error case — `content` is a `WebSearchToolResultError`.
                other => {
                    let code = other.and_then(|c| c.get("error_code")).map_or_else(
                        || "undefined".to_string(),
                        |v| match v {
                            Value::String(s) => s.clone(),
                            _ => v.to_string(),
                        },
                    );
                    out.push(SearchResultEntry::Text(format!("Web search error: {code}")));
                }
            },
            "text" => {
                let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                if in_text {
                    text_acc.push_str(text);
                } else {
                    in_text = true;
                    text_acc = text.to_string();
                }
            }
            _ => {}
        }
    }

    // Flush any trailing buffered text (`if (textAcc.length)` in upstream).
    if !text_acc.is_empty() {
        out.push(SearchResultEntry::Text(text_acc.trim().to_string()));
    }
    out
}
/// Whether a decoded provider block identifies a hosted search operation.
/// Responses supplies the native call when it completes; this is still a
/// progress update before the final model response. Citations are not calls.
pub fn search_started(block: &Value) -> bool {
    let block = block
        .get("value")
        .filter(|_| block.get("type").and_then(Value::as_str) == Some("provider_content"))
        .unwrap_or(block);
    match block.get("type").and_then(Value::as_str) {
        Some("server_tool_use") => block.get("name").and_then(Value::as_str) == Some("web_search"),
        Some("web_search_call") => true,
        _ => false,
    }
}
pub fn supports(profile: &crate::protocol::ProviderProfile) -> bool {
    supports_with_config(profile, &Default::default())
}
pub fn supports_with_config(
    profile: &crate::protocol::ProviderProfile,
    config: &crate::protocol::WebSearchConfig,
) -> bool {
    let mut request = serde_json::from_value::<crate::protocol::ChatRequest>(
        json!({"model":"capability-probe", "messages":[]}),
    )
    .expect("static search request");
    request.set_hosted_web_search(Some(config.clone()));
    crate::codecs::web_search::apply(&request, profile, &mut Default::default()).is_ok()
}
/// Add normalized citations for providers without Anthropic result blocks.
pub fn append_citations(results: &mut Vec<SearchResultEntry>, observations: Option<&Value>) {
    if results
        .iter()
        .any(|entry| matches!(entry, SearchResultEntry::Hit(_)))
    {
        return;
    }
    let Some(observations) = observations.and_then(Value::as_array) else {
        return;
    };
    let mut seen = std::collections::BTreeSet::new();
    let mut hits = Vec::new();
    for observation in observations {
        if let Ok(result) =
            serde_json::from_value::<crate::protocol::WebSearchResult>(observation.clone())
        {
            for citation in result.citations {
                if seen.insert(citation.url.clone()) {
                    hits.push(json!({"title":citation.title,"url":citation.url}));
                }
            }
        }
    }
    if !hits.is_empty() {
        results.push(SearchResultEntry::Hit(
            json!({"tool_use_id":null,"content":hits}),
        ));
    }
}

#[cfg(test)]
mod progress_tests {
    use super::*;
    #[test]
    fn native_search_calls_are_distinct_from_citations_and_client_tools() {
        for block in [
            json!({"type":"server_tool_use","name":"web_search"}),
            json!({"type":"web_search_call","id":"ws_1","status":"completed"}),
        ] {
            assert!(search_started(&block));
            assert!(search_started(
                &json!({"type":"provider_content","value":block})
            ));
        }
        for block in [
            json!({"type":"url_citation","url":"https://example.com"}),
            json!({"type":"function_call","name":"web_search"}),
            json!({"type":"server_tool_use","name":"web_fetch"}),
        ] {
            assert!(!search_started(&block));
        }
    }
}

/// Tracks query occurrences in cumulative grounding snapshots. Citations alone
/// do not establish a search count, and repeated snapshots add no progress.
#[derive(Default)]
pub struct SearchProgress {
    queries: std::collections::BTreeMap<String, u64>,
}
impl SearchProgress {
    /// Number of newly observed searches (not the number of result citations).
    pub fn observe(&mut self, result: &crate::protocol::WebSearchResult) -> u64 {
        let Some(queries) = result
            .metadata
            .pointer("/groundingMetadata/webSearchQueries")
            .and_then(Value::as_array)
        else {
            return 0;
        };
        let mut snapshot = std::collections::BTreeMap::new();
        for query in queries.iter().filter_map(Value::as_str) {
            *snapshot.entry(query.to_owned()).or_insert(0_u64) += 1;
        }
        let mut added = 0;
        for (query, count) in snapshot {
            let previous = self.queries.entry(query).or_default();
            added += count.saturating_sub(*previous);
            *previous = (*previous).max(count);
        }
        added
    }
}

#[cfg(test)]
mod query_progress_tests {
    use super::*;
    #[test]
    fn grounding_snapshots_count_queries_not_citations_or_repeated_snapshots() {
        let mut progress = SearchProgress::default();
        let result = |queries| crate::protocol::WebSearchResult {
            metadata: json!({"groundingMetadata":{"webSearchQueries":queries}}),
            ..Default::default()
        };
        assert_eq!(progress.observe(&result(json!(["rust"]))), 1);
        assert_eq!(progress.observe(&result(json!(["rust"]))), 0);
        assert_eq!(progress.observe(&result(json!(["rust", "rust docs"]))), 1);
        assert_eq!(progress.observe(&result(json!(["rust"]))), 0);
        assert_eq!(progress.observe(&Default::default()), 0);
    }
}
