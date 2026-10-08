use lingxi_llm_client::protocol::JsonPayload;
use lingxi_llm_client::providers::anthropic::fallback_request::{
    classify_server_fallback_rejection, server_fallback_repair_facts, LaneMode,
    ServerFallbackRepairCause as Cause,
};
use serde_json::Value;

fn native_vm_fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/server_fallback_repair_2_1_288.json")).unwrap()
}

#[test]
fn classifier_and_retry_facts_match_extracted_2_1_288_vm() {
    let fixture = native_vm_fixture();
    assert_eq!(fixture["version"], "2.1.288");
    assert_eq!(
        fixture["native_sha256"],
        "bbe93063f7a0879a1021b2891e5c9354e5b3b98433e32efe6750f7710afed750"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 64);

    for case in cases {
        let native = &case["actual"];
        let expected_cause = match native["cause"].as_str() {
            Some("category_beta_header") => Some(Cause::CategoryBetaHeader),
            Some("default_unconfigured") => Some(Cause::DefaultUnconfigured),
            // bMe also classifies a generic server-side fallback beta, but
            // that cause retires the June explicit beta through another path.
            _ => None,
        };
        let cause = classify_server_fallback_rejection(
            case["status"].as_u64().unwrap() as u16,
            &JsonPayload::from_generated_value(&case["error_body"])
                .expect("fixture error body is valid JSON"),
        );
        assert_eq!(cause, expected_cause, "native case: {}", case["name"]);

        let facts = cause.and_then(|cause| {
            server_fallback_repair_facts(
                cause,
                case["fallback_parameter_added"].as_bool().unwrap(),
                case["repair_already_attempted"].as_bool().unwrap(),
            )
        });
        let native_category_retry =
            native["retry"].as_str() == Some("retry:server-fallback-category-strip");
        assert_eq!(
            facts.is_some(),
            native_category_retry,
            "native retry case: {} / a1={} / attempted={}",
            case["name"],
            case["fallback_parameter_added"],
            case["repair_already_attempted"]
        );
        if let Some(facts) = facts {
            assert_eq!(facts.cause, cause.unwrap());
            assert_eq!(facts.rejected_mode, LaneMode::Default);
        }
    }
}

#[test]
fn native_a1_fixture_covers_default_explicit_and_transport_gates() {
    let fixture = native_vm_fixture();
    let cases = fixture["a1Cases"].as_array().unwrap();
    assert_eq!(cases.len(), 24);
    assert!(cases.iter().any(|case| {
        case["Tu"] == true
            && case["fallback"] == "default"
            && case["sent"]
                .as_array()
                .unwrap()
                .iter()
                .any(|beta| beta == "server-side-fallback-2026-07-01")
            && case["actual"]["present"] == true
            && case["actual"]["mode"] == "default"
    }));
    assert!(cases.iter().any(|case| {
        case["Tu"] == true
            && case["fallback"].is_array()
            && case["sent"]
                .as_array()
                .unwrap()
                .iter()
                .any(|beta| beta == "server-side-fallback-2026-06-01")
            && case["actual"]["present"] == true
            && case["actual"]["mode"] == "explicit"
    }));
    assert!(cases.iter().any(|case| {
        case["Tu"] == false && case["fallback"] == "default" && case["actual"]["present"] == false
    }));
}
