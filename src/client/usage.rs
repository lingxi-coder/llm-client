//! Merging and validating a provider's usage report across a stream.
//!
//! Ported from the previous project's `model_attempt.rs`
//! (`merge_attempt_usage`, `complete_usage_for`) and `stream_accumulator.rs`
//! (`merge_usage`). Two ideas carry over; both are easy to get wrong and hard
//! to notice.
//!
//! **Merge on presence, not on value.** A stream reports usage twice: a seed
//! when the message opens and a correction when it closes. Taking the later
//! value only when it is non-zero — which is the obvious way to write it —
//! cannot tell "this counter is genuinely zero now" from "this frame did not
//! mention the counter". A model that writes cache on the first turn and reads
//! it on the second legitimately reports `cache_creation_input_tokens: 0` at the
//! end, and a value-based merge keeps the stale seed instead.
//!
//! **A report is only worth billing from if it is self-consistent.** The
//! buckets we publish are derived by subtraction, so a provider that sends a
//! cached count larger than its own prompt count would leave us reporting a
//! saturated zero as though it were a measurement. Better to say the report is
//! incomplete.
//!
//! Where the previous project carried the raw provider JSON on the usage struct
//! and re-read it later, these functions are called by the stream decoders,
//! which already hold the frame they are parsing. Nothing has to be carried.

use serde_json::Value;

/// A counter's value, if the report actually states it as a number.
///
/// `None` means the field was absent, null, or not a number — all of which are
/// "not stated", as distinct from "stated as zero".
#[must_use]
pub(crate) fn counter(raw: &Value, key: &str) -> Option<u64> {
    raw.get(key).and_then(Value::as_u64)
}

/// A counter at a nested path, with the same rule.
#[must_use]
pub(crate) fn counter_at(raw: &Value, path: &str) -> Option<u64> {
    raw.pointer(path).and_then(Value::as_u64)
}

/// Fold a later usage report over an earlier one, key by key.
///
/// A key the later frame states wins, including when it states zero. A key it
/// omits keeps whatever the seed said. One level of nesting is merged the same
/// way, so a wire that splits a counter into a sub-object (cache writes by TTL)
/// does not lose the seed's split to a delta that only restates the total.
pub(crate) fn fold(seed: &mut Value, delta: &Value) {
    let (Some(seed_obj), Some(delta_obj)) = (seed.as_object_mut(), delta.as_object()) else {
        *seed = delta.clone();
        return;
    };
    for (key, value) in delta_obj {
        match (seed_obj.get_mut(key), value) {
            (Some(existing), Value::Object(_)) if existing.is_object() => fold(existing, value),
            _ => {
                seed_obj.insert(key.clone(), value.clone());
            }
        }
    }
}

/// Which counters a wire names, so one validator can serve every wire.
///
/// `total` is checked against `input + output` when the provider sends one.
/// `thoughts_are_extra` distinguishes the two conventions: on one wire the
/// thinking count sits outside the output count and has to be added before the
/// total will reconcile; on the others it is already inside.
pub(crate) struct ReportShape {
    pub input: &'static str,
    pub output: &'static str,
    pub total: &'static [&'static str],
    /// `(path, whether it is bounded by input rather than output)`
    pub subsets: &'static [(&'static str, bool)],
    pub thoughts: Option<&'static str>,
    pub thoughts_are_extra: bool,
    /// The nested object splitting cache writes by TTL, if the wire has one:
    /// `(object key, the total it must account for, its parts)`.
    ///
    /// The first part is the derivable one — the total minus every other part —
    /// so a report may omit it and still be complete.
    pub cache_creation: Option<(&'static str, &'static str, &'static [&'static str])>,
}

/// Whether a usage report is complete and internally consistent.
///
/// Both required counters must be explicitly numeric: a default zero from a
/// frame that never mentioned them is not a measurement. Every subset must fit
/// inside the counter it is a subset of, because we publish the difference.
/// Any total the provider states must reconcile.
#[must_use]
pub(crate) fn is_complete(raw: &Value, shape: &ReportShape) -> bool {
    let (Some(input), Some(output)) = (counter(raw, shape.input), counter(raw, shape.output))
    else {
        return false;
    };

    // A field that is present but not a number is a malformed report, not an
    // absent counter.
    for (path, _) in shape.subsets {
        if raw.pointer(path).is_some_and(|v| v.as_u64().is_none()) {
            return false;
        }
    }
    for (path, bounded_by_input) in shape.subsets {
        let ceiling = if *bounded_by_input { input } else { output };
        if counter_at(raw, path).is_some_and(|n| n > ceiling) {
            return false;
        }
    }
    // Cached reads and writes partition the prompt, so checking each one
    // against input separately is insufficient when both are present.
    let cached_input = shape
        .subsets
        .iter()
        .filter(|(_, input)| *input)
        .try_fold(0_u64, |sum, (path, _)| {
            sum.checked_add(counter_at(raw, path).unwrap_or(0))
        });
    if cached_input.is_none_or(|sum| sum > input) {
        return false;
    }

    let thoughts = match shape.thoughts {
        Some(key) => {
            if raw.get(key).is_some_and(|v| v.as_u64().is_none()) {
                return false;
            }
            counter(raw, key).unwrap_or(0)
        }
        None => 0,
    };

    let Some(mut expected) = input.checked_add(output) else {
        return false;
    };
    if shape.thoughts_are_extra {
        let Some(sum) = expected.checked_add(thoughts) else {
            return false;
        };
        expected = sum;
    }
    for key in shape.total {
        let Some(stated) = raw.get(key) else { continue };
        let Some(stated) = stated.as_u64() else {
            return false;
        };
        if stated != expected {
            return false;
        }
    }

    if let Some((group, total_key, parts)) = shape.cache_creation {
        if let Some(creation) = raw.get(group) {
            if !creation.is_object() {
                return false;
            }
            let Some(total) = counter(raw, total_key) else {
                return false;
            };
            let mut split = 0_u64;
            for key in parts {
                let Some(value) = creation.get(*key) else {
                    continue;
                };
                let Some(n) = value.as_u64() else {
                    return false;
                };
                let Some(sum) = split.checked_add(n) else {
                    return false;
                };
                split = sum;
            }
            // A split that overshoots its own total is malformed either way.
            //
            // Otherwise the parts must account for the total exactly — with one
            // exception, and it is not symmetric. The first part is the
            // derivable one: given the total and every other part, it is their
            // difference. So a report that omits only that part is still
            // complete. A report that states it and stops is not, because what
            // is left over cannot be attributed to any particular one of the
            // remaining tariffs, and they are priced differently.
            let derivable_missing = creation.get(parts[0]).is_none();
            let others_stated = parts[1..].iter().any(|k| creation.get(*k).is_some());
            if split > total || (!(derivable_missing && others_stated) && split != total) {
                return false;
            }
        }
    }

    true
}

pub(crate) const ANTHROPIC: ReportShape = ReportShape {
    input: "input_tokens",
    output: "output_tokens",
    total: &[],
    subsets: &[],
    thoughts: None,
    thoughts_are_extra: false,
    cache_creation: Some((
        "cache_creation",
        "cache_creation_input_tokens",
        &["ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"],
    )),
};

pub(crate) const OPENAI_CHAT: ReportShape = ReportShape {
    input: "prompt_tokens",
    output: "completion_tokens",
    total: &["total_tokens"],
    subsets: &[
        ("/prompt_tokens_details/cached_tokens", true),
        ("/prompt_tokens_details/cache_write_tokens", true),
        ("/completion_tokens_details/reasoning_tokens", false),
    ],
    thoughts: None,
    thoughts_are_extra: false,
    cache_creation: None,
};

pub(crate) const OPENAI_RESPONSES: ReportShape = ReportShape {
    input: "input_tokens",
    output: "output_tokens",
    total: &["total_tokens"],
    subsets: &[
        ("/input_tokens_details/cached_tokens", true),
        ("/output_tokens_details/reasoning_tokens", false),
    ],
    thoughts: None,
    thoughts_are_extra: false,
    cache_creation: None,
};

pub(crate) const GEMINI: ReportShape = ReportShape {
    input: "promptTokenCount",
    output: "candidatesTokenCount",
    total: &["totalTokenCount"],
    subsets: &[("/cachedContentTokenCount", true)],
    thoughts: Some("thoughtsTokenCount"),
    // This wire counts thinking outside the candidate tokens, so a stated total
    // only reconciles once they are added back.
    thoughts_are_extra: true,
    cache_creation: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_report_missing_either_required_counter_is_not_complete() {
        // A counter the provider never stated decodes to zero like any other,
        // which is exactly why absence has to be checked before the value.
        assert!(!is_complete(&json!({"prompt_tokens": 10}), &OPENAI_CHAT));
        assert!(!is_complete(
            &json!({"completion_tokens": 10}),
            &OPENAI_CHAT
        ));
        assert!(is_complete(
            &json!({"prompt_tokens": 10, "completion_tokens": 2}),
            &OPENAI_CHAT
        ));
    }

    #[test]
    fn disjoint_cache_buckets_must_fit_the_prompt_together() {
        assert!(!is_complete(
            &json!({
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 6, "cache_write_tokens": 6}
            }),
            &OPENAI_CHAT
        ));
    }

    #[test]
    fn normalized_totals_must_not_overflow_even_without_a_reported_total() {
        assert!(!is_complete(
            &json!({"promptTokenCount": 1, "candidatesTokenCount": u64::MAX, "thoughtsTokenCount": 1}),
            &GEMINI
        ));
        assert!(!is_complete(
            &json!({"prompt_tokens": u64::MAX, "completion_tokens": 1}),
            &OPENAI_CHAT
        ));
    }

    #[test]
    fn a_subset_larger_than_what_it_is_a_subset_of_is_not_complete() {
        // We publish the difference, so this would saturate to zero and read as
        // a measurement rather than as the contradiction it is.
        assert!(!is_complete(
            &json!({
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 11},
            }),
            &OPENAI_CHAT
        ));
        assert!(!is_complete(
            &json!({
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "completion_tokens_details": {"reasoning_tokens": 3},
            }),
            &OPENAI_CHAT
        ));
        assert!(is_complete(
            &json!({
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 10},
                "completion_tokens_details": {"reasoning_tokens": 2},
            }),
            &OPENAI_CHAT
        ));
    }

    #[test]
    fn a_stated_total_has_to_reconcile() {
        assert!(is_complete(
            &json!({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}),
            &OPENAI_CHAT
        ));
        assert!(!is_complete(
            &json!({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 13}),
            &OPENAI_CHAT
        ));
    }

    #[test]
    fn the_two_thinking_conventions_reconcile_differently() {
        // Here the thinking count sits outside the candidate count, so the
        // provider's own total only adds up once it is included. Applying the
        // other wire's convention would reject a perfectly good report.
        let report = json!({
            "promptTokenCount": 1000,
            "candidatesTokenCount": 50,
            "thoughtsTokenCount": 7,
            "totalTokenCount": 1057,
        });
        assert!(is_complete(&report, &GEMINI));

        let as_if_folded_in = json!({
            "promptTokenCount": 1000,
            "candidatesTokenCount": 50,
            "thoughtsTokenCount": 7,
            "totalTokenCount": 1050,
        });
        assert!(!is_complete(&as_if_folded_in, &GEMINI));
    }

    #[test]
    fn a_counter_that_is_present_but_not_a_number_is_malformed() {
        for bad in [json!("12"), json!(null), json!(-1), json!(1.5)] {
            assert!(
                !is_complete(
                    &json!({
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": bad},
                    }),
                    &OPENAI_CHAT
                ),
                "{bad} is not a token count"
            );
        }
    }

    #[test]
    fn folding_takes_a_stated_zero_and_keeps_an_unstated_counter() {
        let mut seed = json!({"input_tokens": 10, "cache_creation_input_tokens": 500});
        fold(
            &mut seed,
            &json!({"output_tokens": 4, "cache_creation_input_tokens": 0}),
        );
        assert_eq!(
            seed,
            json!({
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_creation_input_tokens": 0,
            }),
            "the zero replaces; the untouched counter survives"
        );
    }

    #[test]
    fn folding_reaches_into_a_nested_split() {
        let mut seed = json!({"cache_creation": {"ephemeral_5m_input_tokens": 30}});
        fold(
            &mut seed,
            &json!({"cache_creation": {"ephemeral_1h_input_tokens": 70}}),
        );
        assert_eq!(
            seed,
            json!({"cache_creation": {
                "ephemeral_5m_input_tokens": 30,
                "ephemeral_1h_input_tokens": 70,
            }}),
            "a delta restating one tariff must not drop the other"
        );
    }
}
