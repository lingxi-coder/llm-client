//! The shipped provider presets: routing joined to the vendored catalog.
//!
//! These name providers on purpose — they assert what the data files say. Gate
//! 30 scans `src/` only, because a test reading a shipped data file is not a
//! code path that adds a provider.

use lingxi_agent_api::protocol::{BillingMode, ProtocolFamily, ProviderProfile, Submission, Usage};
use lingxi_llm_client::presets::{builtin, merge};
use lingxi_llm_client::LlmClientBuilder;
use std::sync::Arc;

mod support;

#[test]
fn metered_glm_never_inherits_subscription_zero_prices() {
    let profile = builtin()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "glm")
        .unwrap();
    assert_eq!(profile.pricing.billing_mode, BillingMode::PerToken);
    let client = LlmClientBuilder::with_transport(Arc::new(support::NoHttp), &[profile])
        .build()
        .unwrap();
    let route = client.resolve("glm/glm-4.7").unwrap();
    let usage = Usage {
        input_tokens: 1_000,
        output_tokens: 100,
        ..Usage::default()
    };
    assert!(client
        .estimate_cost(&route, &usage, Submission::Interactive)
        .unwrap()
        .is_none());
}

#[test]
fn every_preset_resolves_to_a_vendored_catalog_slice() {
    let presets = builtin().expect("the routing table and every slice must parse");
    assert!(presets.len() >= 7, "{}", presets.len());
    for p in &presets {
        assert!(
            !p.models.is_empty(),
            "preset {:?} has no models, so nothing can resolve to it",
            p.profile_name
        );
        assert!(p.base_url.starts_with("https://"), "{:?}", p.profile_name);
    }
}

#[test]
fn the_catalog_supplies_context_windows_rather_than_this_repo_guessing_them() {
    let presets = builtin().unwrap();
    let with_window = presets
        .iter()
        .flat_map(|p| &p.models)
        .filter(|m| m.metadata.context_window_tokens.is_some())
        .count();
    let total: usize = presets.iter().map(|p| p.models.len()).sum();
    assert!(
        with_window * 2 > total,
        "most models should carry a published context window ({with_window} of \
         {total}); a hand-written list is what this replaced"
    );
}

#[test]
fn presets_and_their_models_are_ordered_so_two_dumps_agree() {
    let presets = builtin().unwrap();
    let names: Vec<&str> = presets.iter().map(|p| p.profile_name.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(
        names, sorted,
        "the preset order comes from a directory listing, which is not ordered \
         on its own (gate 42)"
    );

    for p in &presets {
        let ids: Vec<&String> = p.models.iter().map(|m| &m.request_model).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(
            ids, sorted,
            "{:?}'s models must be sorted too",
            p.profile_name
        );
    }
}

#[test]
fn every_wire_this_crate_speaks_has_at_least_one_preset() {
    use lingxi_agent_api::protocol::ProtocolFamily;
    let served: std::collections::BTreeSet<_> =
        builtin().unwrap().iter().map(|p| p.protocol).collect();
    for family in [
        ProtocolFamily::OpenAiChat,
        ProtocolFamily::OpenAiResponses,
        ProtocolFamily::AnthropicMessages,
        ProtocolFamily::GeminiGenerateContent,
    ] {
        assert!(
            served.contains(&family),
            "{family:?} has a codec but no preset, so nobody reaches it without \
             writing settings by hand"
        );
    }
}

#[test]
fn a_vendor_publishing_two_wires_is_one_group_with_two_connections() {
    let presets = builtin().unwrap();
    for group in ["zhipu", "xai"] {
        let wires: std::collections::BTreeSet<_> = presets
            .iter()
            .filter(|p| p.group() == group)
            .map(|p| p.protocol)
            .collect();
        assert_eq!(
            wires.len(),
            2,
            "{group} ships an OpenAI-compatible endpoint and an \
             Anthropic-compatible one; neither is 'the compatible one'"
        );
        assert!(wires.contains(&ProtocolFamily::AnthropicMessages));
        assert!(wires.contains(&ProtocolFamily::OpenAiChat));
    }
}

#[test]
fn a_vendor_reached_over_two_wires_keeps_one_credential() {
    let presets = builtin().unwrap();
    let creds: std::collections::BTreeSet<_> = presets
        .iter()
        .filter(|p| p.group() == "xai")
        .map(|p| format!("{:?}", p.credential))
        .collect();
    assert_eq!(
        creds.len(),
        1,
        "one account reached two ways is still one key; two env vars would make \
         a user set the same secret twice and wonder which is live"
    );
}

#[test]
fn connections_billed_differently_are_not_interchangeable() {
    let presets = builtin().unwrap();
    for group in ["zhipu", "kimi"] {
        let modes: std::collections::BTreeSet<_> = presets
            .iter()
            .filter(|p| p.group() == group)
            .map(|p| p.pricing.billing_mode)
            .collect();
        assert!(
            modes.contains(&BillingMode::Subscription) && modes.contains(&BillingMode::PerToken),
            "{group} pairs a plan with a metered endpoint, which is exactly the \
             case the failover chain must refuse to cross"
        );
    }
}

#[test]
fn every_connection_of_a_group_has_a_distinct_order() {
    let presets = builtin().unwrap();
    let mut groups: std::collections::BTreeMap<&str, Vec<u32>> = Default::default();
    for p in &presets {
        groups
            .entry(p.group())
            .or_default()
            .push(p.connection.order);
    }
    for (group, mut orders) in groups {
        let before = orders.len();
        orders.sort_unstable();
        orders.dedup();
        assert_eq!(
            orders.len(),
            before,
            "{group} has two connections at the same order; which is tried first \
             would then depend on the profile name alone"
        );
    }
}

#[test]
fn the_quirk_flags_are_set_where_the_wire_cannot_infer_them() {
    let presets = builtin().unwrap();
    let flag = |name: &str, key: &str| {
        presets
            .iter()
            .find(|p| p.profile_name == name)
            .and_then(|p| p.extra.get(key))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    };
    assert!(
        flag("deepseek", "preserve_reasoning_content"),
        "thinking is on by default here and the endpoint rejects a later turn \
         that omits that turn's reasoning"
    );
    assert!(
        flag("deepseek", "thinking_rejects_forced_tool_choice"),
        "with thinking on this endpoint answers 400 to `required` and to a \
         named function, though `none`, `auto` and the tools are all fine"
    );
    assert!(
        flag("kimi", "stream_usage_opt_in"),
        "without it the transcript and every token counter downstream see zero"
    );
    assert!(
        flag("openai", "supports_previous_response_id"),
        "only a provider that persists Responses state may receive a continuation id"
    );
}

#[test]
fn a_user_entry_replaces_the_preset_of_the_same_name() {
    let mine: ProviderProfile = serde_json::from_value(serde_json::json!({
        "provider_id": "mine",
        "profile_name": "deepseek",
        "base_url": "https://internal.example",
        "protocol": "open_ai_chat",
        "auth": "none",
        "models": [{"display_model": "m", "request_model": "m", "billing_model": "m"}],
    }))
    .unwrap();

    let merged = merge(vec![mine]).unwrap();
    let hits: Vec<_> = merged
        .iter()
        .filter(|p| p.profile_name == "deepseek")
        .collect();
    assert_eq!(hits.len(), 1, "the preset does not survive beside it");
    assert_eq!(hits[0].base_url, "https://internal.example");
    assert!(merged.len() > 1, "the rest of the presets are still there");
}

#[test]
fn a_model_resolves_through_the_client_built_from_the_presets() {
    use lingxi_llm_client::LlmClientBuilder;
    let http = std::sync::Arc::new(support::NoHttp);
    let presets = builtin().unwrap();

    // OAuth remains host-specific and must still fail build validation when
    // no matching authenticator is registered.
    let mut oauth_profile = presets
        .first()
        .expect("there is at least one preset")
        .clone();
    oauth_profile.auth = lingxi_agent_api::protocol::AuthStrategy::OAuthBearer;
    let missing =
        match LlmClientBuilder::with_transport(http.clone(), &[oauth_profile.clone()]).build() {
            Err(e) => e,
            Ok(_) => panic!("OAuth requires a host authenticator"),
        };
    assert!(
        format!("{missing}").contains(&oauth_profile.profile_name),
        "the error names the profile a user has to fix: {missing}"
    );

    // Replace authentication for this offline catalog-only test.
    let mut b = LlmClientBuilder::with_transport(http, &presets);
    for strategy in lingxi_agent_api::protocol::AuthStrategy::ALL {
        b.register_authenticator(strategy, std::sync::Arc::new(support::NoAuth));
    }
    let client = b.build().expect("every preset's wire has a codec");

    let listed = client.models();
    assert!(
        listed.len() > 20,
        "the catalog is the model list now, not a hand-written one: {}",
        listed.len()
    );
    let one = listed.first().expect("at least one model").id.clone();
    assert!(
        client.resolve(&one).is_ok(),
        "a listed model must resolve: {one}"
    );
}

// --- pricing ---------------------------------------------------------------

/// Unix seconds for a UTC instant, without a calendar dependency.
/// 2026-01-05 is a Monday; 2026-01-10 is the Saturday of that week.
fn utc(days_after_2026_01_05: u64, hour: u64, minute: u64) -> u64 {
    // 2026-01-05T00:00:00Z
    const MONDAY: u64 = 1_767_571_200;
    MONDAY + days_after_2026_01_05 * 86_400 + hour * 3_600 + minute * 60
}

#[test]
fn every_priced_model_keeps_its_buckets_apart() {
    let presets = builtin().unwrap();
    let priced: Vec<_> = presets
        .iter()
        .flat_map(|p| &p.models)
        .filter_map(|m| m.pricing.as_ref())
        .collect();
    assert!(
        priced.len() > 100,
        "the catalog publishes prices and they must survive the conversion: {}",
        priced.len()
    );
    assert!(
        priced.iter().any(|p| p.cache_read_per_million.is_some()),
        "a cache read is typically a tenth of the input rate; folding it into \
         input misprices every cached turn"
    );
    assert!(
        priced.iter().any(|p| p.reasoning_per_million.is_some()),
        "a provider that bills thinking tokens separately is not describable \
         without this bucket"
    );
    assert!(
        priced.iter().all(|p| p.source.is_some()),
        "every price records where it came from, so a stale estimate is traceable"
    );
}

#[test]
fn the_deepseek_style_schedule_halves_the_bill_outside_peak() {
    let presets = builtin().unwrap();
    let p = presets
        .iter()
        .find(|p| p.profile_name == "deepseek")
        .expect("preset");
    let schedule = p
        .pricing
        .peak
        .as_ref()
        .expect("this vendor publishes peak and off-peak rates");
    let model = p
        .models
        .iter()
        .find(|m| m.request_model == "deepseek-flash")
        .expect("model");
    let listed = model.pricing.as_ref().expect("priced");

    // Inside a published peak window on a weekday: the listed rates stand.
    let peak = listed.at(Submission::Interactive, Some(schedule), utc(0, 2, 0));
    assert_eq!(peak.input_per_million, listed.input_per_million);
    assert_eq!(peak.output_per_million, listed.output_per_million);

    // Between the two windows on the same weekday is off-peak.
    let between = listed.at(Submission::Interactive, Some(schedule), utc(0, 5, 0));
    assert_eq!(
        between.output_per_million,
        listed.output_per_million.map(|v| v / 2.0)
    );

    // And the whole weekend is off-peak, peak hours or not.
    let saturday = listed.at(Submission::Interactive, Some(schedule), utc(5, 2, 0));
    assert_eq!(
        saturday.input_per_million,
        listed.input_per_million.map(|v| v / 2.0),
        "billing the weekend at peak would overstate the cost of most of a week"
    );
    assert_eq!(
        saturday.cache_read_per_million,
        listed.cache_read_per_million.map(|v| v / 2.0),
        "the discount applies to every bucket, not only to input"
    );
}

#[test]
fn a_provider_with_no_schedule_is_billed_at_its_listed_rates() {
    let presets = builtin().unwrap();
    let flat = presets
        .iter()
        .find(|p| p.pricing.peak.is_none() && p.models.iter().any(|m| m.pricing.is_some()))
        .expect("most providers charge one rate");
    let model = flat.models.iter().find(|m| m.pricing.is_some()).unwrap();
    let listed = model.pricing.as_ref().unwrap();
    let resolved = listed.at(Submission::Interactive, None, utc(5, 2, 0));
    assert_eq!(resolved.input_per_million, listed.input_per_million);
    assert_eq!(resolved.output_per_million, listed.output_per_million);
    assert_eq!(
        resolved.cache_read_per_million,
        listed.cache_read_per_million
    );
    assert_eq!(
        resolved.cache_write_per_million,
        listed.cache_write_per_million
    );
    assert_eq!(resolved.reasoning_per_million, listed.reasoning_per_million);
    assert_eq!(resolved.source, listed.source, "provenance survives");
    assert_eq!(
        resolved.batch, None,
        "a resolved rate is what you pay, not a table to resolve again"
    );
}

#[test]
fn nothing_a_route_declares_is_lost_to_a_table_header() {
    // TOML binds every bare key after a `[table]` header to that table. A route
    // that puts `[pricing.peak]` above `extra = {...}` therefore parses cleanly
    // and drops the quirk flags into the schedule instead. That happened while
    // writing this, so the check is on the outcome: what the file declares has
    // to survive the parse.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/providers");
    let presets = builtin().unwrap();
    let mut checked = 0;

    for entry in std::fs::read_dir(&dir).expect("the preset directory exists") {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "toml") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(&path).unwrap();
        let route = text.split("\n[[model]]").next().unwrap();
        let parsed = presets
            .iter()
            .find(|p| p.profile_name == name)
            .unwrap_or_else(|| panic!("{name} parsed"));
        checked += 1;

        if route.contains("\nextra = {") {
            assert!(
                parsed.extra.is_object(),
                "{name} declares `extra` but it did not reach the profile"
            );
        }
        if route.contains("\nconnection = {") {
            assert!(
                parsed.connection != Default::default(),
                "{name} declares `connection` but it did not reach the profile"
            );
        }
        if route.contains("[pricing.peak]") {
            assert!(
                parsed.pricing.peak.is_some(),
                "{name} declares a peak schedule but it did not reach the profile"
            );
        }
    }
    assert!(checked >= 12, "only {checked} presets were checked");
}

/// The upstream catalog repeats a model under `<id>:batch` at its batch-API
/// rate. That is a pricing row, not a model: the discount comes from the
/// endpoint you post to, not from the model string, so such an id would be
/// resolvable here and then rejected on the wire. `vendor-catalog.py` folds the
/// rates onto the base model instead; this fails if a refresh ever lets one
/// through as an id of its own.
#[test]
fn a_catalog_pricing_variant_is_not_a_routable_model() {
    let presets = builtin().unwrap();
    let leaked: Vec<String> = presets
        .iter()
        .flat_map(|p| p.models.iter().map(move |m| (p, m)))
        .filter(|(_, m)| {
            [&m.request_model, &m.display_model, &m.billing_model]
                .iter()
                .any(|s| s.ends_with(":batch"))
        })
        .map(|(p, m)| format!("{}/{}", p.profile_name, m.request_model))
        .collect();
    assert!(
        leaked.is_empty(),
        "these ids are batch pricing rows, not wire models: {leaked:?}"
    );
}

/// Folding is the half that can regress quietly: dropping a `:batch` row is
/// easy, keeping its price is the part worth asserting.
#[test]
fn the_batch_rates_ride_on_the_model_they_belong_to() {
    let presets = builtin().unwrap();
    let p = presets
        .iter()
        .find(|p| p.profile_name == "anthropic")
        .expect("preset");
    let m = p
        .models
        .iter()
        .find(|m| m.request_model == "claude-opus-5")
        .expect("model");
    let listed = m.pricing.as_ref().expect("priced");
    let batch = listed
        .batch
        .as_ref()
        .expect("this vendor publishes a batch rate for this model");
    assert_eq!(listed.input_per_million, Some(5.0));
    assert_eq!(batch.input_per_million, Some(2.5));
    assert_eq!(batch.output_per_million, Some(12.5));
    assert_eq!(batch.cache_write_per_million, Some(3.125));
}

/// On a first-party preset the two rows are the same seller, so the batch rate
/// is a real discount: every bucket at exactly half. Asserted only here — see
/// `a_batch_rate_is_carried_as_published_not_assumed_to_be_a_discount` for why
/// the aggregator gets no such claim.
#[test]
fn a_first_party_batch_rate_is_exactly_half_the_listed_one() {
    let presets = builtin().unwrap();
    let p = presets
        .iter()
        .find(|p| p.profile_name == "anthropic")
        .expect("preset");
    let mut checked = 0;
    for m in &p.models {
        let Some(listed) = m.pricing.as_ref() else {
            continue;
        };
        let Some(batch) = listed.batch.as_ref() else {
            continue;
        };
        for (bucket, b, l) in [
            ("input", batch.input_per_million, listed.input_per_million),
            (
                "output",
                batch.output_per_million,
                listed.output_per_million,
            ),
            (
                "cache_read",
                batch.cache_read_per_million,
                listed.cache_read_per_million,
            ),
            (
                "cache_write",
                batch.cache_write_per_million,
                listed.cache_write_per_million,
            ),
        ] {
            let (Some(b), Some(l)) = (b, l) else { continue };
            assert!(
                (b - l / 2.0).abs() < 1e-9,
                "{}: batch {bucket} {b} is not half the listed {l}",
                m.request_model
            );
            checked += 1;
        }
    }
    assert!(checked >= 40, "only {checked} buckets compared");
}

/// The discount is per bucket, not one multiplier over the model: this vendor
/// halves input, output and reasoning and leaves the cache read alone. A scalar
/// could not record that, which is why `BatchPricing` repeats the buckets.
#[test]
fn a_batch_discount_can_skip_a_bucket() {
    let presets = builtin().unwrap();
    let p = presets
        .iter()
        .find(|p| p.profile_name == "openrouter")
        .expect("preset");
    let m = p
        .models
        .iter()
        .find(|m| m.request_model == "google/gemini-2.5-pro")
        .expect("model");
    let listed = m.pricing.as_ref().expect("priced");
    let batch = listed.batch.as_ref().expect("batch rates");
    assert_eq!(listed.input_per_million, Some(1.25));
    assert_eq!(batch.input_per_million, Some(0.625), "input is halved");
    assert_eq!(batch.output_per_million, Some(5.0), "so is output");
    assert_eq!(
        batch.cache_read_per_million, listed.cache_read_per_million,
        "but the cache read is not discounted at all"
    );
}

/// An aggregator quotes the cheapest host it can route to, and its batch row
/// may be a different host than its listed row — some are dearer than the rate
/// they supposedly discount. So the fold carries what is published and asserts
/// no direction: only that the number is a usable price and that the rows did
/// not silently stop being folded at all.
#[test]
fn a_batch_rate_is_carried_as_published_not_assumed_to_be_a_discount() {
    let presets = builtin().unwrap();
    let mut carried = 0;
    for p in &presets {
        for m in &p.models {
            let Some(batch) = m.pricing.as_ref().and_then(|x| x.batch.as_ref()) else {
                continue;
            };
            carried += 1;
            for (bucket, v) in [
                ("input", batch.input_per_million),
                ("output", batch.output_per_million),
                ("cache_read", batch.cache_read_per_million),
                ("cache_write", batch.cache_write_per_million),
                ("reasoning", batch.reasoning_per_million),
            ] {
                let Some(v) = v else { continue };
                assert!(
                    v.is_finite() && v >= 0.0,
                    "{}/{}: batch {bucket} is {v}",
                    p.profile_name,
                    m.request_model
                );
            }
        }
    }
    assert_eq!(
        carried, 78,
        "the catalog ships 78 batch rate sets; a refresh that changes this \
         should change this number deliberately"
    );
}

/// An aggregator serves a free tier over the same endpoint and the same key as
/// its metered models, so how a model is billed cannot be a property of the
/// connection. These carry it themselves.
#[test]
fn an_aggregators_free_tier_is_billed_differently_from_its_metered_models() {
    let presets = builtin().unwrap();
    let p = presets
        .iter()
        .find(|p| p.profile_name == "openrouter")
        .expect("preset");
    assert_eq!(
        p.pricing.billing_mode,
        BillingMode::PerToken,
        "the connection's default"
    );

    let free: Vec<&str> = p
        .models
        .iter()
        .filter(|m| m.billing_mode_on(&p.pricing) == BillingMode::Free)
        .map(|m| m.request_model.as_str())
        .collect();
    assert_eq!(free.len(), 18, "the vendor marks these itself");
    assert!(free.iter().all(|id| id.ends_with(":free")));

    let metered = p
        .models
        .iter()
        .find(|m| m.request_model == "anthropic/claude-opus-5")
        .expect("model");
    assert_eq!(
        metered.billing_mode_on(&p.pricing),
        BillingMode::PerToken,
        "everything unmarked falls back to the connection"
    );
}

/// The rule that must not be applied: several models are priced at zero because
/// a subscription already covers them. Covered is not free, and reading a zero
/// price as free would let a request move off a plan and start charging.
#[test]
fn a_model_priced_at_zero_is_not_thereby_free() {
    let presets = builtin().unwrap();
    let mut checked = 0;
    for p in &presets {
        if p.pricing.billing_mode != BillingMode::Subscription {
            continue;
        }
        for m in &p.models {
            let all_zero = m.pricing.as_ref().is_some_and(|pr| {
                [
                    pr.input_per_million,
                    pr.output_per_million,
                    pr.cache_read_per_million,
                    pr.cache_write_per_million,
                ]
                .iter()
                .flatten()
                .all(|v| *v == 0.0)
            });
            if !all_zero {
                continue;
            }
            checked += 1;
            assert_eq!(
                m.billing_mode_on(&p.pricing),
                BillingMode::Subscription,
                "{}/{} is covered by a plan, not free",
                p.profile_name,
                m.request_model
            );
        }
    }
    assert!(
        checked >= 10,
        "only {checked} zero-priced subscription models; this guards against a \
         refresh deciding they are free"
    );
}

/// The one field an app needs to turn "you have no key for this provider" into
/// a link the user can follow. It is hand-authored per preset because the page
/// where a key is created is not derivable from a base URL, and the vendor's
/// name for itself is not the profile name — which is a file stem like
/// `grok-anthropic`.
#[test]
fn every_shipped_provider_says_what_it_is_called_and_where_to_get_a_key() {
    let presets = builtin().unwrap();
    assert_eq!(presets.len(), 15);
    for p in &presets {
        let info = &p.info;
        let name = info
            .display_name
            .as_deref()
            .unwrap_or_else(|| panic!("{} has no display name", p.profile_name));
        assert!(!name.trim().is_empty(), "{}", p.profile_name);
        assert_ne!(
            name, p.profile_name,
            "{} is showing its file stem, not a name a user would recognise",
            p.profile_name
        );
        let key_url = info
            .api_key_url
            .as_deref()
            .unwrap_or_else(|| panic!("{} has no api key page", p.profile_name));
        assert!(
            key_url.starts_with("https://"),
            "{}: {key_url}",
            p.profile_name
        );
    }
}

/// Every URL we hand a user has to be one, and none of the optional fields may
/// be present-but-empty — an empty string renders as a dead link, which is
/// worse than an absent one the app can skip.
#[test]
fn no_provider_link_is_present_but_useless() {
    for p in builtin().unwrap() {
        let i = &p.info;
        for (field, value) in [
            ("console_url", &i.console_url),
            ("api_key_url", &i.api_key_url),
            ("docs_url", &i.docs_url),
        ] {
            if let Some(v) = value {
                assert!(
                    v.starts_with("https://") && v.len() > "https://".len(),
                    "{} {field}: {v:?}",
                    p.profile_name
                );
            }
        }
        for (field, value) in [
            ("display_name", &i.display_name),
            ("description", &i.description),
            ("credential_hint", &i.credential_hint),
        ] {
            if let Some(v) = value {
                assert!(!v.trim().is_empty(), "{} {field} is blank", p.profile_name);
            }
        }
    }
}

/// `scripts/vendor-catalog.py` regenerates a preset by replacing everything from
/// the first `[[model]]` onward. Metadata lives above that line, so a refresh
/// keeps it — but only while it stays above. A bare key that slips below the
/// line binds into that model's table instead and is silently lost from the
/// route.
#[test]
fn provider_metadata_survives_a_catalog_regeneration() {
    // Read the files the script rewrites, not the compiled-in copies: this is
    // a claim about the data on disk.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/providers");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).expect("the preset directory exists") {
        let path = entry.expect("readable").path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        seen += 1;
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(&path).expect("readable");
        let head = text.split("\n[[model]]").next().expect("a route header");
        for key in ["display_name", "api_key_url"] {
            assert!(
                head.contains(&format!("{key} = ")),
                "{name}: {key} is below the first [[model]] and a refresh would drop it"
            );
        }
    }
    assert_eq!(seen, 15, "every preset was checked");
}
