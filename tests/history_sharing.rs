use lingxi_llm_client::protocol::JsonPayload;

#[test]
fn clones_share_source_and_replacements_preserve_other_snapshots() {
    let source = r#"{"text":"\ud800","integer":9007199254740993,"object":{"x":1}}"#;
    let original = JsonPayload::parse(source).unwrap();
    let snapshot = original.clone();
    assert_eq!(
        original.as_json_source().as_ptr(),
        snapshot.as_json_source().as_ptr()
    );
    let edited = original
        .with_object_field("text", &JsonPayload::string("new"))
        .unwrap();
    assert_eq!(snapshot.as_json_source(), source);
    assert_ne!(edited, snapshot);
    let roundtrip: JsonPayload =
        serde_json::from_str(&serde_json::to_string(&snapshot).unwrap()).unwrap();
    assert_eq!(snapshot, roundtrip);
    assert_eq!(
        roundtrip.as_json_source().as_ptr(),
        roundtrip.clone().as_json_source().as_ptr()
    );
}
