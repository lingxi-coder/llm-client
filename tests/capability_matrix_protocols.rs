//! Offline evidence integrity for alternate provider routes.
use chrono::NaiveDate;
use serde_json::Value;

fn evidence_date(value: &Value) -> NaiveDate {
    NaiveDate::parse_from_str(value.as_str().unwrap(), "%Y-%m-%d").unwrap()
}

#[test]
fn alternate_catalog_evidence_is_complete_and_keeps_unknowns() {
    let matrix: Value =
        serde_json::from_str(include_str!("../data/capability-matrix-protocols.json")).unwrap();
    assert_eq!(matrix["live_validation"], "not_run");
    // The initial review was on this date; later additions keep their own dates.
    let initial_review_date = NaiveDate::from_ymd_opt(2026, 10, 12).unwrap();
    let snapshot_date = evidence_date(&matrix["evidence_date"]);
    for p in matrix["profiles"].as_array().unwrap() {
        let raw: toml::Value = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(p["catalog_path"].as_str().unwrap()),
        )
        .unwrap()
        .parse()
        .unwrap();
        assert_eq!(p["protocol"].as_str(), raw["protocol"].as_str());
        assert_eq!(p["base_url"].as_str(), raw["base_url"].as_str());
        assert_eq!(
            p["models"].as_array().unwrap().len(),
            raw["model"].as_array().unwrap().len()
        );
        for m in p["models"].as_array().unwrap() {
            assert_eq!(m["live_validation"], "not_run");
            let model_date = evidence_date(&m["evidence_date"]);
            assert!(initial_review_date <= model_date && model_date <= snapshot_date);
            assert_eq!(
                m["cells"].as_object().unwrap().len(),
                p["operation_ids"].as_array().unwrap().len()
            );
            for cell in m["cells"].as_object().unwrap().values() {
                for id in cell["source_ids"].as_array().unwrap() {
                    let source = &matrix["sources"][id.as_str().unwrap()];
                    assert!(source["url"].as_str().unwrap().starts_with("https://"));
                    let source_date = evidence_date(&source["evidence_date"]);
                    assert!(initial_review_date <= source_date && source_date <= model_date);
                }
            }
        }
    }
}
