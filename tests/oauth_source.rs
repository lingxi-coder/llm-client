use lingxi_llm_client::providers::anthropic::oauth_source::{
    select_oauth_source, OAuthSource, OAuthSourceInputs,
};
#[test]
fn current_native_oauth_source_cases() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/oauth_source_2_1_288.json")).unwrap();
    let saved = vec!["stored:scope".to_owned()];
    for row in fixture["cases"].as_array().unwrap() {
        let i = &row["input"];
        let descriptor_scopes: Option<Vec<String>> =
            serde_json::from_value(i["descriptor_scopes"].clone()).unwrap();
        let selected = select_oauth_source(OAuthSourceInputs {
            environment_token: i["environment_token"].as_str(),
            environment_scopes: i["environment_scopes"].as_str(),
            descriptor_token: i["descriptor_token"].as_str(),
            descriptor_scopes: descriptor_scopes.as_deref(),
            from_background_snapshot: i["from_background_snapshot"].as_bool().unwrap(),
            host_managed: i["host_managed"].as_bool().unwrap(),
            stored_token: i["stored_token"].as_str(),
            stored_scopes: Some(&saved),
        });
        let actual=selected.map(|s|serde_json::json!({"access_token":s.access_token,"scopes":s.scopes,"source":match s.source {OAuthSource::Environment=>"environment",OAuthSource::Descriptor=>"descriptor",OAuthSource::Store=>"store"}}));
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            row["expected"],
            "{row}"
        );
    }
}

#[test]
fn native_293_raw_token_truthiness_and_empty_fallback() {
    for token in [" token ", " ", "\t"] {
        let selected = select_oauth_source(OAuthSourceInputs {
            environment_token: Some(token),
            descriptor_token: Some("fd"),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(selected.access_token, token);
        assert_eq!(selected.source, OAuthSource::Environment);
    }

    let selected = select_oauth_source(OAuthSourceInputs {
        environment_token: Some(""),
        descriptor_token: Some(" fd "),
        stored_token: Some("stored"),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(selected.access_token, " fd ");
    assert_eq!(selected.source, OAuthSource::Descriptor);
}
