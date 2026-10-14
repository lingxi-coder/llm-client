//! Typed Native 2.1.295 model-validation suggestion facts.
//!
//! The SDK owns the baked fallback catalog and source ordering. The host owns
//! the selected route and supplies the current effective `hb()[catalogKey]`
//! values after Native's `Op(modelOverrides)` and provider-specific runtime
//! discovery have been applied.

use super::model_complete::NativeModelObservation;
use super::model_complete_catalog::{
    native_model_suggestion_target, native_static_provider_defaults, NativeModelProvider,
};
use super::JsonPayload;
use std::collections::BTreeMap;

/// Host facts read by Native `ikn(model)` after a trigger row is matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeModelSuggestionFacts {
    /// Exact current Native `hb()[catalogConfigKey]` values by catalog ID,
    /// after the provider's `go(Pe())` map, `Op(modelOverrides)`, and any live
    /// provider discovery (such as Bedrock inference profiles) have resolved.
    /// Values remain root JSON strings so isolated UTF-16 units survive.
    pub effective_model_by_catalog_id: NativeModelObservation<BTreeMap<String, JsonPayload>>,
    /// `ANTHROPIC_DEFAULT_OPUS_MODEL` as observed by the host. Native uses a
    /// nullish check for the fable-to-Opus suggestion, preserving empty text.
    pub anthropic_default_opus_model: NativeModelObservation<Option<JsonPayload>>,
}

impl NativeModelSuggestionFacts {
    /// Explicitly represent a caller that has not read any suggestion inputs.
    #[must_use]
    pub fn unobserved() -> Self {
        Self {
            effective_model_by_catalog_id: NativeModelObservation::Unobserved,
            anthropic_default_opus_model: NativeModelObservation::Unobserved,
        }
    }
}

/// Static `go(provider)` seed values for Native's 20 model-config rows.
///
/// Missing route IDs use the first non-null provider row in Native `Yl`/`Ho`
/// order. If no row exists for a provider, that catalog entry uses its
/// first-party ID. This is only a seed: the host must apply current
/// `modelOverrides` and provider discovery before using values as Native
/// `hb()[catalogConfigKey]` observations.
#[must_use]
pub fn native_static_model_suggestion_defaults(
    provider: NativeModelProvider,
) -> BTreeMap<String, JsonPayload> {
    native_static_provider_defaults(provider)
        .into_iter()
        .map(|(catalog_id, model)| (catalog_id.to_owned(), JsonPayload::string(model)))
        .collect()
}

/// Resolve Native `ikn(model)` after model-access/provider selection. An
/// `Unobserved` result means the host omitted a fact Native reads; an observed
/// `None` means the source route gate or catalog scan has no suggestion.
#[must_use]
pub fn native_model_suggestion(
    input_model_utf16: &[u16],
    provider: NativeModelObservation<NativeModelProvider>,
    facts: &NativeModelSuggestionFacts,
) -> NativeModelObservation<Option<JsonPayload>> {
    let provider = match provider {
        NativeModelObservation::Unobserved => return NativeModelObservation::Unobserved,
        NativeModelObservation::Observed(provider) => provider,
    };
    if matches!(
        provider,
        NativeModelProvider::FirstParty
            | NativeModelProvider::AnthropicAws
            | NativeModelProvider::AnthropicGoogleCloud
            | NativeModelProvider::Gateway
    ) {
        return NativeModelObservation::Observed(None);
    }

    let Some(target) = native_model_suggestion_target(input_model_utf16) else {
        return NativeModelObservation::Observed(None);
    };

    // Native's fable-5 candidate checks the Opus env value with `??`, so a
    // defined empty string is an actual suggestion. If absent, fall through to
    // the host's already-resolved hb()[catalogConfigKey] value.
    if target.trigger_family == super::model_complete_catalog::NativeModelFamily::Fable
        && target.fallback_family == super::model_complete_catalog::NativeModelFamily::Opus
    {
        match &facts.anthropic_default_opus_model {
            NativeModelObservation::Unobserved => return NativeModelObservation::Unobserved,
            NativeModelObservation::Observed(Some(value)) => {
                return if value.is_root_string() {
                    NativeModelObservation::Observed(Some(value.clone()))
                } else {
                    NativeModelObservation::Unobserved
                };
            }
            NativeModelObservation::Observed(None) => {}
        }
    }

    match &facts.effective_model_by_catalog_id {
        NativeModelObservation::Unobserved => NativeModelObservation::Unobserved,
        NativeModelObservation::Observed(defaults) => {
            let Some(value) = defaults.get(target.fallback_catalog_id) else {
                return NativeModelObservation::Unobserved;
            };
            if !value.is_root_string() {
                return NativeModelObservation::Unobserved;
            }
            NativeModelObservation::Observed(Some(value.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        native_model_suggestion, native_static_model_suggestion_defaults,
        NativeModelSuggestionFacts,
    };
    use crate::protocol::{
        JsonPayload, NativeModelObservation as Observation, NativeModelProvider,
    };

    fn facts(provider: NativeModelProvider) -> NativeModelSuggestionFacts {
        NativeModelSuggestionFacts {
            effective_model_by_catalog_id: Observation::Observed(
                native_static_model_suggestion_defaults(provider),
            ),
            anthropic_default_opus_model: Observation::Observed(None),
        }
    }

    fn suggest(model: &str, provider: NativeModelProvider) -> Option<JsonPayload> {
        let facts = facts(provider);
        match native_model_suggestion(
            &model.encode_utf16().collect::<Vec<_>>(),
            Observation::Observed(provider),
            &facts,
        ) {
            Observation::Observed(value) => value,
            Observation::Unobserved => panic!("test facts are complete"),
        }
    }

    #[test]
    fn fallback_3p_triggers_follow_native_sorted_substring_scan() {
        let rows = [
            ("request fable_5_1", "claude-fable-5"),
            ("fable-5", "claude-opus-5-5"),
            ("opus-5-5", "claude-opus-5"),
            ("opus_5", "claude-opus-4-8"),
            ("opus-4-8", "claude-opus-4-7"),
            ("opus-4-7", "claude-opus-4-6"),
            ("opus-4-6", "claude-opus-4-5"),
            ("opus-4-5", "claude-opus-4-1"),
            ("sonnet-5-5", "claude-sonnet-5"),
            ("sonnet-5", "claude-sonnet-4-6"),
            ("sonnet-4-6", "claude-sonnet-4-5"),
            ("sonnet-4-5", "claude-sonnet-4-0"),
            ("haiku-5-5", "claude-haiku-4-5"),
        ];
        let effective_model_by_catalog_id =
            native_static_model_suggestion_defaults(NativeModelProvider::Bedrock)
                .into_keys()
                .map(|catalog_id| {
                    let value = catalog_id.clone();
                    (catalog_id, JsonPayload::string(&value))
                })
                .collect();
        let identity_facts = NativeModelSuggestionFacts {
            effective_model_by_catalog_id: Observation::Observed(effective_model_by_catalog_id),
            anthropic_default_opus_model: Observation::Observed(None),
        };

        for (input, expected_catalog_id) in rows {
            let result = native_model_suggestion(
                &input.encode_utf16().collect::<Vec<_>>(),
                Observation::Observed(NativeModelProvider::Bedrock),
                &identity_facts,
            );
            assert_eq!(
                match result {
                    Observation::Observed(Some(payload)) => payload.string_at(""),
                    other => panic!("unexpected result for {input}: {other:?}"),
                },
                Some(expected_catalog_id.to_owned()),
                "{input}"
            );
        }
    }

    #[test]
    fn static_provider_seed_contains_only_native_model_config_rows_and_exact_ids() {
        for provider in [NativeModelProvider::Bedrock, NativeModelProvider::Foundry] {
            let defaults = native_static_model_suggestion_defaults(provider);
            assert_eq!(defaults.len(), 20);
            assert!(!defaults.contains_key("claude-mythos-5"));
            assert!(!defaults.contains_key("claude-mythos-5-1"));
        }
        let bedrock = native_static_model_suggestion_defaults(NativeModelProvider::Bedrock);
        assert_eq!(
            bedrock
                .get("claude-opus-4-1")
                .and_then(|payload| payload.string_at("")),
            Some("us.anthropic.claude-opus-4-1-20250805-v1:0".to_owned())
        );
        let vertex = native_static_model_suggestion_defaults(NativeModelProvider::Vertex);
        assert_eq!(
            vertex
                .get("claude-opus-4-1")
                .and_then(|payload| payload.string_at("")),
            Some("claude-opus-4-1@20250805".to_owned())
        );
        let foundry = native_static_model_suggestion_defaults(NativeModelProvider::Foundry);
        assert_eq!(
            foundry
                .get("claude-opus-4-1")
                .and_then(|payload| payload.string_at("")),
            Some("claude-opus-4-1".to_owned())
        );
    }

    #[test]
    fn first_party_family_and_gateway_routes_suppress_suggestions() {
        for provider in [
            NativeModelProvider::FirstParty,
            NativeModelProvider::AnthropicAws,
            NativeModelProvider::AnthropicGoogleCloud,
            NativeModelProvider::Gateway,
        ] {
            assert_eq!(suggest("unknown opus-5 model", provider), None);
        }
        assert!(suggest("opus-5", NativeModelProvider::Foundry).is_some());
    }

    #[test]
    fn current_host_effective_values_are_used_per_catalog_id() {
        let provider = NativeModelProvider::Vertex;
        let mut effective = native_static_model_suggestion_defaults(provider);
        effective.insert(
            "claude-opus-4-1".to_owned(),
            JsonPayload::string("host-resolved-opus-4-1-after-Op"),
        );
        let facts = NativeModelSuggestionFacts {
            effective_model_by_catalog_id: Observation::Observed(effective),
            anthropic_default_opus_model: Observation::Observed(None),
        };
        let result = native_model_suggestion(
            &"opus-4-5".encode_utf16().collect::<Vec<_>>(),
            Observation::Observed(provider),
            &facts,
        );
        assert_eq!(
            match result {
                Observation::Observed(Some(payload)) => payload.string_at(""),
                other => panic!("unexpected suggestion result: {other:?}"),
            },
            Some("host-resolved-opus-4-1-after-Op".to_owned())
        );
    }

    #[test]
    fn effective_values_preserve_utf16_and_host_applies_truthy_overrides() {
        let provider = NativeModelProvider::Bedrock;
        let facts = NativeModelSuggestionFacts {
            effective_model_by_catalog_id: Observation::Observed({
                let mut effective = native_static_model_suggestion_defaults(provider);
                effective.insert(
                    "claude-opus-4-1".to_owned(),
                    JsonPayload::string_from_utf16_code_units(&[0xd800]),
                );
                effective
            }),
            anthropic_default_opus_model: Observation::Observed(None),
        };
        let result = native_model_suggestion(
            &"opus-4-5".encode_utf16().collect::<Vec<_>>(),
            Observation::Observed(provider),
            &facts,
        );
        let Observation::Observed(Some(suggestion)) = result else {
            panic!("expected suggestion");
        };
        assert_eq!(suggestion.root_string_code_units(), Some(vec![0xd800]));

        // The host already applied Native's truthy-only `Op()` overlay, so an
        // empty configured override leaves this current effective value intact.
        let facts = NativeModelSuggestionFacts {
            effective_model_by_catalog_id: Observation::Observed(
                native_static_model_suggestion_defaults(provider),
            ),
            anthropic_default_opus_model: Observation::Observed(None),
        };
        let result = native_model_suggestion(
            &"opus-4-5".encode_utf16().collect::<Vec<_>>(),
            Observation::Observed(provider),
            &facts,
        );
        assert_eq!(
            match result {
                Observation::Observed(Some(payload)) => payload.string_at(""),
                other => panic!("unexpected suggestion result: {other:?}"),
            },
            Some("us.anthropic.claude-opus-4-1-20250805-v1:0".to_owned())
        );
    }

    #[test]
    fn fable_opus_environment_override_is_nullish_not_truthy() {
        let provider = NativeModelProvider::Foundry;
        let mut facts = facts(provider);
        facts.anthropic_default_opus_model = Observation::Observed(Some(JsonPayload::string("")));
        assert_eq!(
            match native_model_suggestion(
                &"fable-5".encode_utf16().collect::<Vec<_>>(),
                Observation::Observed(provider),
                &facts,
            ) {
                Observation::Observed(Some(payload)) => payload.string_at(""),
                other => panic!("unexpected suggestion result: {other:?}"),
            },
            Some(String::new())
        );
        facts.anthropic_default_opus_model = Observation::Observed(None);
        assert_eq!(
            suggest("fable-5", provider).and_then(|payload| payload.string_at("")),
            Some("claude-opus-5-5".to_owned())
        );
    }

    #[test]
    fn unknown_host_facts_remain_unobserved_and_nonmatches_are_absent() {
        let facts = NativeModelSuggestionFacts {
            effective_model_by_catalog_id: Observation::Unobserved,
            anthropic_default_opus_model: Observation::Unobserved,
        };
        assert_eq!(
            native_model_suggestion(
                &"opus-5".encode_utf16().collect::<Vec<_>>(),
                Observation::Observed(NativeModelProvider::Bedrock),
                &facts,
            ),
            Observation::Unobserved
        );
        assert_eq!(
            native_model_suggestion(
                &"no model token".encode_utf16().collect::<Vec<_>>(),
                Observation::Observed(NativeModelProvider::Bedrock),
                &facts,
            ),
            Observation::Observed(None)
        );
        assert_eq!(
            native_model_suggestion(
                &"opus-5".encode_utf16().collect::<Vec<_>>(),
                Observation::Unobserved,
                &facts,
            ),
            Observation::Unobserved
        );
    }
}
