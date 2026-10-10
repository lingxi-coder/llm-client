//! Provider search result projection for clients rendering hosted search.
use crate::protocol::JsonPayload;
use serde::Serialize;
use serde_json::json;
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum SearchResultEntry {
    /// Free-form text accumulated from one or more consecutive `text` blocks
    /// (also used for the `Web search error: <code>` string emitted on a
    /// `web_search_tool_result` error payload).
    Text(JsonPayload),
    /// Structured search result from a `web_search_tool_result` block, shaped
    /// `{ "tool_use_id": <id>, "content": [{ "title", "url" }, ...] }`.
    Hit(JsonPayload),
}

fn null_payload() -> JsonPayload {
    JsonPayload::parse("null").expect("null is valid JSON")
}

fn scalar_payload(parent: &JsonPayload, pointer: &str) -> JsonPayload {
    let Some(payload) = parent.payload_at(pointer) else {
        return null_payload();
    };
    match payload.kind_at("") {
        Some(
            crate::response_json::FlatJsonKind::String
            | crate::response_json::FlatJsonKind::Boolean
            | crate::response_json::FlatJsonKind::Null
            | crate::response_json::FlatJsonKind::Number,
        ) => payload,
        _ => null_payload(),
    }
}

fn object_payload(fields: &[(&str, JsonPayload)]) -> JsonPayload {
    fields.iter().fold(
        JsonPayload::parse("{}").expect("empty object is valid JSON"),
        |object, (name, value)| {
            object
                .with_object_field(name, value)
                .expect("field insertion into generated object succeeds")
        },
    )
}

fn raw_array(items: &[JsonPayload]) -> JsonPayload {
    let mut source = String::from("[");
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            source.push(',');
        }
        source.push_str(item.as_json_source());
    }
    source.push(']');
    JsonPayload::parse(&source).expect("array assembled from JSON payloads is valid")
}

fn string_units_or_source(payload: &JsonPayload) -> Vec<u16> {
    payload
        .root_string_code_units()
        .unwrap_or_else(|| payload.as_json_source().encode_utf16().collect())
}

pub fn parse_response_content(content: &[JsonPayload]) -> Vec<SearchResultEntry> {
    let mut out: Vec<SearchResultEntry> = Vec::new();
    let mut text_acc = Vec::new();
    let mut in_text = true;

    fn trim_js_whitespace(units: &[u16]) -> &[u16] {
        fn is_js_whitespace(unit: u16) -> bool {
            matches!(
                unit,
                0x0009..=0x000D
                    | 0x0020
                    | 0x00A0
                    | 0x1680
                    | 0x2000..=0x200A
                    | 0x2028..=0x2029
                    | 0x202F
                    | 0x205F
                    | 0x3000
                    | 0xFEFF
            )
        }
        let start = units
            .iter()
            .position(|unit| !is_js_whitespace(*unit))
            .unwrap_or(units.len());
        let end = units
            .iter()
            .rposition(|unit| !is_js_whitespace(*unit))
            .map_or(start, |index| index + 1);
        &units[start..end]
    }

    fn flush_text(out: &mut Vec<SearchResultEntry>, text_acc: &mut Vec<u16>) {
        let trimmed = trim_js_whitespace(text_acc);
        if !trimmed.is_empty() {
            out.push(SearchResultEntry::Text(
                JsonPayload::string_from_utf16_code_units(trimmed),
            ));
        }
        text_acc.clear();
    }

    fn append_text(units: &[u16], text_acc: &mut Vec<u16>, in_text: &mut bool) {
        if *in_text {
            text_acc.extend_from_slice(units);
        } else {
            *in_text = true;
            text_acc.clear();
            text_acc.extend_from_slice(units);
        }
    }

    fn append_provider_content(
        block: &JsonPayload,
        out: &mut Vec<SearchResultEntry>,
        text_acc: &mut Vec<u16>,
        in_text: &mut bool,
    ) {
        match block.string_at("/type").as_deref() {
            Some("server_tool_use") => {
                if *in_text {
                    flush_text(out, text_acc);
                    *in_text = false;
                }
            }
            Some("web_search_tool_result") => {
                // Text blocks are buffered so adjacent blocks can be joined,
                // but an immediate hit/error result must stay after the text
                // that preceded it in the provider's content array.
                flush_text(out, text_acc);
                *in_text = false;
                let content = block.payload_at("/content");
                if let Some(items) = content.as_ref().filter(|items| items.is_root_array()) {
                    let hits = (0..items.array_len("").unwrap_or_default())
                        .map(|index| {
                            let item = items.array_item(index);
                            item.map_or_else(
                                || {
                                    object_payload(&[
                                        ("title", null_payload()),
                                        ("url", null_payload()),
                                    ])
                                },
                                |item| {
                                    object_payload(&[
                                        ("title", scalar_payload(&item, "/title")),
                                        ("url", scalar_payload(&item, "/url")),
                                    ])
                                },
                            )
                        })
                        .collect::<Vec<_>>();
                    let content = raw_array(&hits);
                    out.push(SearchResultEntry::Hit(object_payload(&[
                        ("tool_use_id", scalar_payload(block, "/tool_use_id")),
                        ("content", content),
                    ])));
                } else {
                    let code = content
                        .and_then(|content| content.payload_at("/error_code"))
                        .map_or_else(
                            || "undefined".encode_utf16().collect(),
                            |code| string_units_or_source(&code),
                        );
                    let mut message = "Web search error: ".encode_utf16().collect::<Vec<_>>();
                    message.extend(code);
                    out.push(SearchResultEntry::Text(
                        JsonPayload::string_from_utf16_code_units(&message),
                    ));
                }
            }
            Some("text") => {
                let text = block.string_units_at("/text").unwrap_or_default();
                append_text(&text, text_acc, in_text);
            }
            _ => {}
        }
    }

    for block in content {
        append_provider_content(block, &mut out, &mut text_acc, &mut in_text);
    }

    // Flush any trailing buffered text (`if (textAcc.length)` in upstream).
    if !text_acc.is_empty() {
        let trimmed = trim_js_whitespace(&text_acc);
        out.push(SearchResultEntry::Text(
            JsonPayload::string_from_utf16_code_units(trimmed),
        ));
    }
    out
}
/// Whether a decoded provider block identifies a hosted search operation.
/// Responses supplies the native call when it completes; this is still a
/// progress update before the final model response. Citations are not calls.
pub fn search_started(block: &JsonPayload) -> bool {
    match block.string_at("/type").as_deref() {
        Some("server_tool_use") => block.string_at("/name").as_deref() == Some("web_search"),
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
pub fn append_citations(results: &mut Vec<SearchResultEntry>, observations: Option<&JsonPayload>) {
    if results
        .iter()
        .any(|entry| matches!(entry, SearchResultEntry::Hit(_)))
    {
        return;
    }
    let Some(observations) = observations.filter(|value| value.is_root_array()) else {
        return;
    };
    let mut seen = std::collections::BTreeSet::<Vec<u16>>::new();
    let mut hits = Vec::new();
    for index in 0..observations.array_len("").unwrap_or_default() {
        let Some(observation) = observations.array_item(index) else {
            continue;
        };
        let Some(citations) = observation.payload_at("/citations") else {
            continue;
        };
        for index in 0..citations.array_len("").unwrap_or_default() {
            let Some(citation) = citations.array_item(index) else {
                continue;
            };
            let title = scalar_payload(&citation, "/title");
            let url = scalar_payload(&citation, "/url");
            let Some(url_units) = url.root_string_code_units() else {
                continue;
            };
            if seen.insert(url_units) {
                hits.push(object_payload(&[("title", title), ("url", url)]));
            }
        }
    }
    if !hits.is_empty() {
        let content = raw_array(&hits);
        results.push(SearchResultEntry::Hit(object_payload(&[
            ("tool_use_id", null_payload()),
            ("content", content),
        ])));
    }
}

#[cfg(test)]
mod progress_tests {
    use super::*;
    use crate::protocol::JsonPayload;
    use serde_json::Value;

    fn provider_block(value: Value) -> JsonPayload {
        JsonPayload::from_generated_value(&value).unwrap()
    }

    #[test]
    fn native_search_calls_are_distinct_from_citations_and_client_tools() {
        for block in [
            provider_block(json!({"type":"server_tool_use","name":"web_search"})),
            provider_block(json!({"type":"web_search_call","id":"ws_1","status":"completed"})),
        ] {
            assert!(search_started(&block));
        }
        for block in [
            provider_block(json!({"type":"url_citation","url":"https://example.com"})),
            provider_block(json!({"type":"function_call","name":"web_search"})),
            provider_block(json!({"type":"server_tool_use","name":"web_fetch"})),
        ] {
            assert!(!search_started(&block));
        }
    }
}

#[cfg(test)]
mod source_projection_tests {
    use super::*;

    #[test]
    fn native_text_and_hit_scalars_keep_source_code_units_and_numbers() {
        let blocks = [
            JsonPayload::parse(r#"{"type":"text","text":"lead\ud800tail"}"#).unwrap(),
            JsonPayload::parse(
                r#"{"type":"web_search_tool_result","tool_use_id":"call\ud800","content":[{"title":"hit\ud800","url":"https://example.test/a?x=1e400","score":1e400}]}"#,
            )
            .unwrap(),
        ];

        let entries = parse_response_content(&blocks);
        let SearchResultEntry::Text(text) = &entries[0] else {
            panic!("expected source-backed text result")
        };
        assert_eq!(
            text.root_string_code_units().as_deref(),
            Some(
                "lead"
                    .encode_utf16()
                    .chain([0xD800])
                    .chain("tail".encode_utf16())
                    .collect::<Vec<_>>()
                    .as_slice()
            )
        );

        let SearchResultEntry::Hit(hit) = &entries[1] else {
            panic!("expected source-backed search result")
        };
        assert_eq!(
            hit.payload_at("/tool_use_id")
                .and_then(|value| value.root_string_code_units())
                .as_deref(),
            Some(
                "call"
                    .encode_utf16()
                    .chain([0xD800])
                    .collect::<Vec<_>>()
                    .as_slice()
            )
        );
        assert_eq!(
            hit.payload_at("/content/0/title")
                .and_then(|value| value.root_string_code_units())
                .as_deref(),
            Some(
                "hit"
                    .encode_utf16()
                    .chain([0xD800])
                    .collect::<Vec<_>>()
                    .as_slice()
            )
        );
        assert_eq!(
            hit.raw_source_at("/content/0/url").as_deref(),
            Some(r#""https://example.test/a?x=1e400""#)
        );
        assert!(hit.as_json_source().contains("1e400"));
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
        let Some(count) = result
            .metadata
            .array_len("/groundingMetadata/webSearchQueries")
        else {
            return 0;
        };
        let mut snapshot = std::collections::BTreeMap::new();
        for index in 0..count {
            let Some(units) = result
                .metadata
                .string_units_at(&format!("/groundingMetadata/webSearchQueries/{index}"))
            else {
                continue;
            };
            let Ok(query) = String::from_utf16(&units) else {
                continue;
            };
            *snapshot.entry(query).or_insert(0_u64) += 1;
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
            metadata: crate::protocol::JsonPayload::parse(
                &json!({"groundingMetadata":{"webSearchQueries":queries}}).to_string(),
            )
            .unwrap(),
            ..Default::default()
        };
        assert_eq!(progress.observe(&result(json!(["rust"]))), 1);
        assert_eq!(progress.observe(&result(json!(["rust"]))), 0);
        assert_eq!(progress.observe(&result(json!(["rust", "rust docs"]))), 1);
        assert_eq!(progress.observe(&result(json!(["rust"]))), 0);
        assert_eq!(progress.observe(&Default::default()), 0);
    }
}
