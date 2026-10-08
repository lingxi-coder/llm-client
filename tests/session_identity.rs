use lingxi_llm_client::providers::anthropic::session_identity::{
    session_identity, SessionEnvironment,
};
#[test]
fn current_native_session_identity_cases() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/session_identity_2_1_287.json")).unwrap();
    for row in fixture["cases"].as_array().unwrap() {
        let input = &row["input"];
        let actual = session_identity(SessionEnvironment {
            access_token: input["access_token"].as_str(),
            cached_access_token: input["cached_access_token"].as_str(),
            remote: input["remote"].as_str(),
            environment_kind: input["environment_kind"].as_str(),
            entrypoint: input["entrypoint"].as_str(),
        });
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            row["expected"],
            "{row}"
        );
    }
}
