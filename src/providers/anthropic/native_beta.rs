//! Source-preserving late request policy for host-directed Native beta refusal retries.
use crate::protocol::{JsonPayload, LlmError, ProtocolFamily};
use crate::transport::HttpRequest;

pub(crate) const CONTEXT_1M_BETA_HEADER: &str = "context-1m-2025-08-07";

/// Remove Native's refused context-1m token from the final Anthropic wire.
/// Headers apply to all supported Native Anthropic routes; only Bedrock has
/// the mirrored `anthropic_beta` body member.
pub(crate) fn suppress_context_1m_beta(
    request: &mut HttpRequest,
    protocol: ProtocolFamily,
) -> Result<(), LlmError> {
    if !matches!(
        protocol,
        ProtocolFamily::AnthropicMessages
            | ProtocolFamily::FoundryClaude
            | ProtocolFamily::BedrockClaude
            | ProtocolFamily::VertexClaude
    ) {
        return Ok(());
    }

    suppress_header_token(&mut request.headers);
    if protocol == ProtocolFamily::BedrockClaude {
        suppress_bedrock_body_token(request)?;
    }
    Ok(())
}

fn suppress_header_token(headers: &mut Vec<(String, String)>) {
    let mut filtered = Vec::with_capacity(headers.len());
    for (name, value) in std::mem::take(headers) {
        if !name.eq_ignore_ascii_case("anthropic-beta") {
            filtered.push((name, value));
            continue;
        }
        let kept = value
            .split(',')
            .map(str::trim)
            .filter(|token| *token != CONTEXT_1M_BETA_HEADER)
            .collect::<Vec<_>>();
        if !kept.is_empty() {
            filtered.push((name, kept.join(",")));
        }
    }
    *headers = filtered;
}

fn suppress_bedrock_body_token(request: &mut HttpRequest) -> Result<(), LlmError> {
    let source = std::str::from_utf8(&request.body).map_err(|error| LlmError::InvalidRequest {
        message: format!("Bedrock refusal retry body is not UTF-8 JSON: {error}"),
    })?;
    let body = JsonPayload::parse(source).map_err(|error| LlmError::InvalidRequest {
        message: format!("Bedrock refusal retry body is not JSON: {error}"),
    })?;
    if !body.is_root_object() {
        return Ok(());
    }
    let scanner = body.scanner()?;
    const BETA_POINTER: &str = "/anthropic_beta";
    if scanner.kind_at(BETA_POINTER) != Some(crate::response_json::FlatJsonKind::Array) {
        return Ok(());
    }

    let beta_units = CONTEXT_1M_BETA_HEADER.encode_utf16().collect::<Vec<_>>();
    let source = body.as_json_source();
    let item_count = scanner.array_len(BETA_POINTER).unwrap_or_default();
    let mut item_ranges = Vec::with_capacity(item_count);
    let mut remove_items = Vec::with_capacity(item_count);
    for index in 0..item_count {
        let item_pointer = format!("{BETA_POINTER}/{index}");
        item_ranges.push(
            scanner
                .source_range_at(&item_pointer)
                .expect("validated array item has a source range"),
        );
        remove_items
            .push(scanner.string_units_at(&item_pointer).as_deref() == Some(beta_units.as_slice()));
    }

    // Select commas by maximal target runs. Keeping the comma before a middle
    // run joins its surrounding survivors; a suffix run must also remove the
    // comma before the run or it would leave a trailing comma.
    let mut removals = Vec::new();
    let mut index = 0;
    while index < item_count {
        if !remove_items[index] {
            index += 1;
            continue;
        }
        let run_start = index;
        while index < item_count && remove_items[index] {
            index += 1;
        }
        let run_end = index - 1;
        removals.extend(item_ranges[run_start..=run_end].iter().cloned());

        if item_count >= 2 {
            let suffix_run = run_end + 1 == item_count;
            let first_comma = if suffix_run {
                run_start.saturating_sub(1)
            } else {
                run_start
            };
            let last_comma = if suffix_run { item_count - 2 } else { run_end };
            for comma_index in first_comma..=last_comma {
                let delimiter_range =
                    item_ranges[comma_index].end..item_ranges[comma_index + 1].start;
                let comma = source[delimiter_range.clone()]
                    .find(',')
                    .expect("valid array items have one comma between them");
                let comma_start = delimiter_range.start + comma;
                removals.push(comma_start..comma_start + 1);
            }
        }
    }
    if removals.is_empty() {
        return Ok(());
    }

    // Splice only each refused item and comma. This retains all other request
    // bytes, including field-key spelling, whitespace, nested values, numeric
    // lexemes, and lone-surrogate escapes.
    removals.sort_by_key(|range| range.start);
    let mut merged_removals: Vec<std::ops::Range<usize>> = Vec::with_capacity(removals.len());
    for range in removals {
        if let Some(previous) = merged_removals.last_mut() {
            if range.start <= previous.end {
                previous.end = previous.end.max(range.end);
                continue;
            }
        }
        merged_removals.push(range);
    }
    let removed_bytes = merged_removals
        .iter()
        .map(|range| range.end - range.start)
        .sum::<usize>();
    let mut updated = String::with_capacity(source.len() - removed_bytes);
    let mut cursor = 0;
    for range in merged_removals {
        updated.push_str(&source[cursor..range.start]);
        cursor = range.end;
    }
    updated.push_str(&source[cursor..]);
    request.body = bytes::Bytes::copy_from_slice(updated.as_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: &str, betas: &str) -> HttpRequest {
        HttpRequest {
            method: "POST".into(),
            url: "https://example.test/v1/messages".into(),
            headers: vec![
                ("x-preserve".into(), "same".into()),
                ("Anthropic-Beta".into(), betas.into()),
            ],
            http1_header_layout: None,
            body: bytes::Bytes::copy_from_slice(body.as_bytes()),
            timeout: None,
        }
    }

    #[test]
    fn bedrock_removes_all_exact_members_and_preserves_raw_array_and_sibling_sources() {
        let source = r#"{"messages":[],"anthropic_beta":["before",null,"context-1m-2025-08-07",1.00,{"nested":"\ud800","exp":1.0e+04},"context-1m-2025-08-07","after"],"opaque":{"surrogate":"\ud800","number":1.000}}"#;
        let mut request = request(
            source,
            "before, context-1m-2025-08-07,after,context-1m-2025-08-07",
        );

        suppress_context_1m_beta(&mut request, ProtocolFamily::BedrockClaude).unwrap();

        assert_eq!(
            request.headers,
            vec![
                ("x-preserve".into(), "same".into()),
                ("Anthropic-Beta".into(), "before,after".into()),
            ]
        );
        assert_eq!(
            std::str::from_utf8(&request.body).unwrap(),
            r#"{"messages":[],"anthropic_beta":["before",null,1.00,{"nested":"\ud800","exp":1.0e+04},"after"],"opaque":{"surrogate":"\ud800","number":1.000}}"#
        );
    }

    #[test]
    fn missing_null_nonarray_and_unrelated_protocol_body_values_are_untouched() {
        for source in [
            r#"{"messages":[]}"#,
            r#"{"anthropic_beta":null,"opaque":"\ud800"}"#,
            r#"{"anthropic_beta":"context-1m-2025-08-07","opaque":1.00}"#,
            r#"{"anthropic_beta":["other-beta",null],"opaque":1.00}"#,
        ] {
            let mut request = request(source, "context-1m-2025-08-07,other-beta");
            suppress_context_1m_beta(&mut request, ProtocolFamily::BedrockClaude).unwrap();
            let body = std::str::from_utf8(&request.body).unwrap();
            assert_eq!(body, source);
            assert_eq!(request.headers[1].1, "other-beta");
        }

        let body = r#"{"anthropic_beta":["context-1m-2025-08-07"],"opaque":"\ud800"}"#;
        let mut vertex = request(body, "context-1m-2025-08-07,keep");
        suppress_context_1m_beta(&mut vertex, ProtocolFamily::VertexClaude).unwrap();
        assert_eq!(std::str::from_utf8(&vertex.body).unwrap(), body);
        assert_eq!(vertex.headers[1].1, "keep");

        let mut other = request(body, "context-1m-2025-08-07,keep");
        let original = other.clone();
        suppress_context_1m_beta(&mut other, ProtocolFamily::OpenAiResponses).unwrap();
        assert_eq!(other, original);
    }

    #[test]
    fn header_matching_removes_only_complete_case_sensitive_beta_tokens() {
        let mut request = request(
            "{}",
            "prefix-context-1m-2025-08-07,context-1m-2025-08-07-extra,Context-1m-2025-08-07,context-1m-2025-08-07",
        );

        suppress_context_1m_beta(&mut request, ProtocolFamily::AnthropicMessages).unwrap();

        assert_eq!(
            request.headers[1].1,
            "prefix-context-1m-2025-08-07,context-1m-2025-08-07-extra,Context-1m-2025-08-07"
        );
    }

    #[test]
    fn bedrock_empty_result_preserves_unrelated_source() {
        let source =
            r#"{  "\u0061nthropic_beta" : [ "context-1m-2025-08-07" ], "opaque" : "\ud800" }"#;
        let mut request = request(source, "context-1m-2025-08-07");

        suppress_context_1m_beta(&mut request, ProtocolFamily::BedrockClaude).unwrap();

        assert_eq!(
            std::str::from_utf8(&request.body).unwrap(),
            r#"{  "\u0061nthropic_beta" : [  ], "opaque" : "\ud800" }"#
        );
    }

    #[test]
    fn bedrock_last_member_removal_preserves_remaining_wire_source() {
        let source = r#"{  "\u0061nthropic_beta" : [ "keep" , "context-1m-2025-08-07" ], "opaque" : "\ud800" }"#;
        let mut request = request(source, "context-1m-2025-08-07");

        suppress_context_1m_beta(&mut request, ProtocolFamily::BedrockClaude).unwrap();

        assert_eq!(
            std::str::from_utf8(&request.body).unwrap(),
            r#"{  "\u0061nthropic_beta" : [ "keep"   ], "opaque" : "\ud800" }"#
        );
    }

    #[test]
    fn consecutive_target_runs_preserve_valid_array_and_raw_survivor_source() {
        for (source, expected) in [
            (
                r#"{"anthropic_beta":["context-1m-2025-08-07","context-1m-2025-08-07","keep"]}"#,
                r#"{"anthropic_beta":["keep"]}"#,
            ),
            (
                r#"{"anthropic_beta":["left","context-1m-2025-08-07","context-1m-2025-08-07","right"]}"#,
                r#"{"anthropic_beta":["left","right"]}"#,
            ),
            (
                r#"{"anthropic_beta":["keep","context-1m-2025-08-07","context-1m-2025-08-07"]}"#,
                r#"{"anthropic_beta":["keep"]}"#,
            ),
            (
                r#"{"anthropic_beta":["context-1m-2025-08-07","context-1m-2025-08-07"]}"#,
                r#"{"anthropic_beta":[]}"#,
            ),
            (
                r#"{"anthropic_beta":["context-1m-2025-08-07","one","context-1m-2025-08-07","two","context-1m-2025-08-07"]}"#,
                r#"{"anthropic_beta":["one","two"]}"#,
            ),
            (
                r#"{ "anthropic_beta" : [ "keep" , "context-1m-2025-08-07" , "context-1m-2025-08-07" ], "opaque" : "\ud800", "number" : 1.00 }"#,
                r#"{ "anthropic_beta" : [ "keep"     ], "opaque" : "\ud800", "number" : 1.00 }"#,
            ),
        ] {
            let mut request = request(source, "context-1m-2025-08-07");
            suppress_context_1m_beta(&mut request, ProtocolFamily::BedrockClaude).unwrap();
            let actual = std::str::from_utf8(&request.body).unwrap();
            assert_eq!(actual, expected, "source={source}");
            assert!(JsonPayload::parse(actual).is_ok());
        }
    }
}
