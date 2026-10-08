//! Native attribution metadata is retained alongside a convenience source list.
use crate::protocol::{JsonPayload, StreamEvent, WebCitation, WebSearchResult};
use std::collections::{HashMap, HashSet};

fn node_string(node: &crate::response_json::JsonNodeRef<'_>) -> Option<String> {
    String::from_utf16(&node.string_units()?).ok()
}

fn field<'a>(
    fields: &'a [crate::response_json::JsonObjectEntryRef<'a>],
    key: &str,
) -> Option<&'a crate::response_json::JsonNodeRef<'a>> {
    let expected: Vec<u16> = key.encode_utf16().collect();
    fields
        .iter()
        .find(|entry| entry.key_units == expected)
        .map(|entry| &entry.value)
}

fn field_string(
    fields: &[crate::response_json::JsonObjectEntryRef<'_>],
    key: &str,
) -> Option<String> {
    node_string(field(fields, key)?)
}

fn citations(metadata: &JsonPayload, out: &mut Vec<WebCitation>) {
    let Ok(scanner) = metadata.scanner() else {
        return;
    };
    let mut pending = vec![scanner.root_node()];
    while let Some(node) = pending.pop() {
        match node.kind() {
            crate::response_json::FlatJsonKind::Array => {
                for index in (0..node.array_len().unwrap_or(0)).rev() {
                    if let Some(child) = node.array_item(index) {
                        pending.push(child);
                    }
                }
            }
            crate::response_json::FlatJsonKind::Object => {
                let Some(entries) = node.object_entries() else {
                    continue;
                };
                let tag = field_string(&entries, "type");
                let source = match tag.as_deref() {
                    Some("url_citation") => field(&entries, "url_citation").unwrap_or(&node),
                    Some("web_search_result_location" | "web_search_result" | "url") => &node,
                    _ if field(&entries, "link").is_some() => &node,
                    _ if field(&entries, "maps").is_some() => field(&entries, "maps").unwrap(),
                    _ if field(&entries, "retrievedUrl").is_some()
                        && field_string(&entries, "urlRetrievalStatus").as_deref()
                            == Some("URL_RETRIEVAL_STATUS_SUCCESS") =>
                    {
                        &node
                    }
                    _ => match field(&entries, "web") {
                        Some(value) => value,
                        None => {
                            for entry in entries.into_iter().rev() {
                                pending.push(entry.value);
                            }
                            continue;
                        }
                    },
                };
                if source.kind() == crate::response_json::FlatJsonKind::Object {
                    if let Some(source_fields) = source.object_entries() {
                        let url = ["url", "uri", "googleMapsUri", "retrievedUrl", "link"]
                            .into_iter()
                            .find_map(|key| field_string(&source_fields, key));
                        if let Some(url) = url {
                            let title = field_string(&source_fields, "title");
                            if !out
                                .iter()
                                .any(|citation| citation.url == url && citation.title == title)
                            {
                                out.push(WebCitation { url, title });
                            }
                        }
                    }
                }
                for entry in entries.into_iter().rev() {
                    pending.push(entry.value);
                }
            }
            _ => {}
        }
    }
}

pub(crate) fn result_payload(metadata: JsonPayload) -> Option<WebSearchResult> {
    if metadata.kind_at("") != Some(crate::response_json::FlatJsonKind::Object)
        || metadata.object_entries_at("")?.is_empty()
    {
        return None;
    }
    let mut sources = Vec::new();
    citations(&metadata, &mut sources);
    Some(WebSearchResult {
        citations: sources,
        metadata,
    })
}

/// Extract source-backed search attribution before the response body becomes a recursive Value.
pub(crate) fn responses_payload(body: &JsonPayload) -> Option<WebSearchResult> {
    let mut metadata = Vec::new();
    if body
        .kind_at("/server_side_tool_usage")
        .is_some_and(|kind| kind != crate::response_json::FlatJsonKind::Null)
    {
        metadata.push((
            "server_side_tool_usage".encode_utf16().collect(),
            body.raw_source_at("/server_side_tool_usage")?,
        ));
    }
    let mut annotations = Vec::new();
    let mut calls = Vec::new();
    for output_index in 0..body.array_len("/output").unwrap_or(0) {
        let item = format!("/output/{output_index}");
        if body.string_at(&format!("{item}/type")).as_deref() == Some("web_search_call") {
            calls.push(body.raw_source_at(&item)?);
        }
        for content_index in 0..body.array_len(&format!("{item}/content")).unwrap_or(0) {
            let part = format!("{item}/content/{content_index}");
            for annotation_index in 0..body.array_len(&format!("{part}/annotations")).unwrap_or(0) {
                let annotation = format!("{part}/annotations/{annotation_index}");
                if body.string_at(&format!("{annotation}/type")).as_deref() != Some("url_citation")
                {
                    continue;
                }
                let value = body.raw_source_at(&annotation)?;
                let fields = vec![
                    (
                        "output_index".encode_utf16().collect(),
                        output_index.to_string(),
                    ),
                    (
                        "content_index".encode_utf16().collect(),
                        content_index.to_string(),
                    ),
                    ("annotation".encode_utf16().collect(), value),
                ];
                annotations.push(
                    JsonPayload::object_from_fields(&fields)
                        .ok()?
                        .as_json_source()
                        .to_owned(),
                );
            }
        }
    }
    if !annotations.is_empty() {
        metadata.push((
            "annotations".encode_utf16().collect(),
            array_source(&annotations),
        ));
    }
    if !calls.is_empty() {
        metadata.push((
            "web_search_calls".encode_utf16().collect(),
            array_source(&calls),
        ));
    }
    result_payload(JsonPayload::object_from_fields(&metadata).ok()?)
}

pub(crate) fn anthropic_payload_blocks(blocks: &[JsonPayload]) -> Option<WebSearchResult> {
    let mut selected = Vec::new();
    for block in blocks {
        let search_block = block.string_at("/type").as_deref() == Some("web_search_tool_result")
            || (block.string_at("/type").as_deref() == Some("server_tool_use")
                && block.string_at("/name").as_deref() == Some("web_search"));
        let mut has_location = false;
        for index in 0..block.array_len("/citations").unwrap_or(0) {
            if block
                .string_at(&format!("/citations/{index}/type"))
                .as_deref()
                == Some("web_search_result_location")
            {
                has_location = true;
                break;
            }
        }
        if search_block || has_location {
            selected.push(block.as_json_source().to_owned());
        }
    }
    if selected.is_empty() {
        return None;
    }
    let content = array_source(&selected);
    result_payload(
        JsonPayload::object_from_fields(&[("content".encode_utf16().collect(), content)]).ok()?,
    )
}

pub(crate) fn anthropic_payload_is_search_block(block: &JsonPayload) -> bool {
    block.string_at("/type").as_deref() == Some("web_search_tool_result")
        || (block.string_at("/type").as_deref() == Some("server_tool_use")
            && block.string_at("/name").as_deref() == Some("web_search"))
        || (0..block.array_len("/citations").unwrap_or(0)).any(|index| {
            block
                .string_at(&format!("/citations/{index}/type"))
                .as_deref()
                == Some("web_search_result_location")
        })
}

pub(crate) fn anthropic_event_frame(event: &JsonPayload) -> Option<WebSearchResult> {
    let metadata = JsonPayload::object_from_fields(&[(
        "events".encode_utf16().collect(),
        array_source(&[event.as_json_source().to_owned()]),
    )])
    .ok()?;
    result_payload(metadata)
}

pub(crate) fn gemini_payload(candidate: &JsonPayload) -> Option<WebSearchResult> {
    let mut metadata = Vec::new();
    for key in ["groundingMetadata", "urlContextMetadata"] {
        let pointer = format!("/{key}");
        if candidate
            .kind_at(&pointer)
            .is_some_and(|kind| kind != crate::response_json::FlatJsonKind::Null)
        {
            metadata.push((
                key.encode_utf16().collect(),
                candidate.raw_source_at(&pointer)?,
            ));
        }
    }
    result_payload(JsonPayload::object_from_fields(&metadata).ok()?)
}

pub(crate) fn responses_item(item: &JsonPayload) -> Option<WebSearchResult> {
    if item.string_at("/type").as_deref() != Some("web_search_call") {
        return None;
    }
    let metadata = JsonPayload::object_from_fields(&[(
        "web_search_calls".encode_utf16().collect(),
        array_source(&[item.as_json_source().to_owned()]),
    )])
    .ok()?;
    result_payload(metadata)
}

pub(crate) fn responses_annotation(
    output_index: usize,
    content_index: usize,
    annotation: &JsonPayload,
) -> Option<WebSearchResult> {
    if annotation.string_at("/type").as_deref() != Some("url_citation") {
        return None;
    }
    let value = JsonPayload::object_from_fields(&[
        (
            "output_index".encode_utf16().collect(),
            output_index.to_string(),
        ),
        (
            "content_index".encode_utf16().collect(),
            content_index.to_string(),
        ),
        (
            "annotation".encode_utf16().collect(),
            annotation.as_json_source().to_owned(),
        ),
    ])
    .ok()?;
    let metadata = JsonPayload::object_from_fields(&[(
        "annotations".encode_utf16().collect(),
        array_source(&[value.as_json_source().to_owned()]),
    )])
    .ok()?;
    result_payload(metadata)
}

pub(crate) fn chat_payload(
    message: &JsonPayload,
    web_search: Option<&JsonPayload>,
) -> Option<WebSearchResult> {
    let mut metadata = Vec::new();
    let mut annotations = Vec::new();
    for index in 0..message.array_len("/annotations").unwrap_or(0) {
        let pointer = format!("/annotations/{index}");
        if message.string_at(&format!("{pointer}/type")).as_deref() == Some("url_citation") {
            annotations.push(message.raw_source_at(&pointer)?);
        }
    }
    if !annotations.is_empty() {
        metadata.push((
            "annotations".encode_utf16().collect(),
            array_source(&annotations),
        ));
    }
    if let Some(web_search) = web_search {
        if web_search
            .kind_at("")
            .is_some_and(|kind| kind != crate::response_json::FlatJsonKind::Null)
        {
            let source = web_search
                .canonical_json(crate::exact_json::JsonEncoding::JavaScript)
                .ok()?;
            metadata.push(("web_search".encode_utf16().collect(), source));
        }
    }
    result_payload(JsonPayload::object_from_fields(&metadata).ok()?)
}

pub(crate) fn chat_response_payload(
    root: &JsonPayload,
    message_pointer: &str,
) -> Option<WebSearchResult> {
    let message = root
        .raw_source_at(message_pointer)
        .and_then(|source| JsonPayload::parse(&source).ok())
        .unwrap_or_else(|| JsonPayload::parse("{}").expect("empty object is JSON"));
    let web_search = root
        .raw_source_at("/web_search")
        .and_then(|source| JsonPayload::parse(&source).ok());
    chat_payload(&message, web_search.as_ref())
}

fn array_source(items: &[String]) -> String {
    format!("[{}]", items.join(","))
}

/// Array records are per-frame additions; repeated terminal records are suppressed.
/// Non-array metadata (including Gemini grounding and URL Context data) is a
/// native snapshot; changed snapshots are emitted intact so source references
/// and grounding supports stay aligned.
#[derive(Debug, Default)]
pub(crate) struct SearchStream {
    seen: HashSet<(Vec<u16>, String)>,
    snapshots: HashMap<Vec<u16>, String>,
}
impl SearchStream {
    pub(crate) fn emit(&mut self, search: Option<WebSearchResult>, out: &mut Vec<StreamEvent>) {
        let Some(search) = search else {
            return;
        };
        let Ok(scanner) = search.metadata.scanner() else {
            return;
        };
        let Some(entries) = scanner.root_node().object_entries() else {
            return;
        };
        let mut delta = Vec::new();
        for entry in entries {
            if entry.value.kind() == crate::response_json::FlatJsonKind::Array {
                let mut fresh = Vec::new();
                for index in 0..entry.value.array_len().unwrap_or(0) {
                    let Some(record) = entry.value.array_item(index) else {
                        continue;
                    };
                    let Ok(source) = crate::exact_json::serialize_json_node(
                        record,
                        crate::exact_json::JsonEncoding::JavaScript,
                    ) else {
                        continue;
                    };
                    if self.seen.insert((entry.key_units.to_vec(), source.clone())) {
                        fresh.push(source);
                    }
                }
                if !fresh.is_empty() {
                    delta.push((entry.key_units.to_vec(), format!("[{}]", fresh.join(","))));
                }
            } else if let Ok(source) = crate::exact_json::serialize_json_node(
                entry.value,
                crate::exact_json::JsonEncoding::JavaScript,
            ) {
                if self.snapshots.get(entry.key_units) != Some(&source) {
                    self.snapshots
                        .insert(entry.key_units.to_vec(), source.clone());
                    delta.push((entry.key_units.to_vec(), source));
                }
            }
        }
        if let Ok(metadata) = JsonPayload::object_from_fields(&delta) {
            if let Some(result) = result_payload(metadata) {
                out.push(StreamEvent::WebSearch { result });
            }
        }
    }
}

/// Preserve provider server-search counters without mixing them into token usage.
pub(crate) fn with_usage_payload(
    search: Option<WebSearchResult>,
    usage: Option<&JsonPayload>,
) -> Option<WebSearchResult> {
    let metadata = search
        .map(|result| result.metadata)
        .unwrap_or_else(|| JsonPayload::parse("{}").expect("empty object is JSON"));
    let mut counters = Vec::new();
    if let Some(usage) = usage {
        let valid_count = |pointer: &str| {
            usage.kind_at(pointer) == Some(crate::response_json::FlatJsonKind::Number)
                && usage
                    .raw_source_at(pointer)
                    .is_some_and(|source| source.parse::<u64>().is_ok())
        };
        if valid_count("/server_tool_use/web_search_requests") {
            if let Some(source) = usage.raw_source_at("/server_tool_use") {
                counters.push(("server_tool_use".encode_utf16().collect(), source));
            }
        }
        let web_search = usage.kind_at("/server_side_tool_usage/web_search");
        if web_search.is_some_and(|kind| kind != crate::response_json::FlatJsonKind::Null)
            || valid_count("/server_side_tool_usage/web_search_requests")
            || valid_count("/server_side_tool_usage/web_search_calls")
        {
            if let Some(source) = usage.raw_source_at("/server_side_tool_usage") {
                counters.push(("server_side_tool_usage".encode_utf16().collect(), source));
            }
        }
        if usage
            .kind_at("/x_tools/web_search")
            .is_some_and(|kind| kind != crate::response_json::FlatJsonKind::Null)
        {
            if let Some(source) = usage.raw_source_at("/x_tools/web_search") {
                if let Ok(metadata) = JsonPayload::object_from_fields(&[(
                    "web_search".encode_utf16().collect(),
                    source,
                )]) {
                    counters.push((
                        "x_tools".encode_utf16().collect(),
                        metadata.as_json_source().into(),
                    ));
                }
            }
        }
    }
    if counters.is_empty() {
        return result_payload(metadata);
    }
    let counters = JsonPayload::object_from_fields(&counters).ok()?;
    let metadata = metadata
        .with_canonical_fields(
            crate::exact_json::JsonEncoding::JavaScript,
            &[("usage", counters.as_json_source())],
        )
        .ok()?;
    result_payload(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_search_usage_does_not_claim_that_web_search_occurred() {
        let usage = JsonPayload::parse(r#"{"x_tools":{"file_search":{"count":1}}}"#).unwrap();
        assert!(with_usage_payload(None, Some(&usage)).is_none());
    }
}
