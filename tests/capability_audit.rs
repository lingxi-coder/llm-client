//! Keep the evidence index complete as built-in provider profiles change.
use serde_json::Value;
use std::{collections::BTreeSet, fs, path::Path};

#[test]
fn every_bundled_profile_has_one_evidence_entry() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let audit: Value = serde_json::from_str(include_str!("../data/capability-audit.json")).unwrap();
    assert_eq!(audit["schema_version"], 1);
    assert_eq!(audit["status"], "source_index_only_not_a_support_matrix");
    for matrix in audit["evidence_matrices"].as_array().unwrap() {
        let file = matrix.as_str().unwrap();
        assert!(file.starts_with("capability-matrix-") && file.ends_with(".json"));
        assert!(root.join("data").join(file).is_file());
    }

    let mut indexed = BTreeSet::new();
    let providers = audit["providers"].as_array().unwrap();
    for provider in providers {
        let id = provider["provider_id"].as_str().unwrap();
        assert!(!id.is_empty());
        assert!(provider["live_validation"].is_string());
        assert!(provider["sources"]
            .as_array()
            .unwrap()
            .iter()
            .all(|source| {
                source
                    .as_str()
                    .is_some_and(|url| url.starts_with("https://"))
            }));
        for profile in provider["profiles"].as_array().unwrap() {
            let profile = profile.as_str().unwrap();
            assert!(
                indexed.insert(profile.to_owned()),
                "duplicate evidence entry: {profile}"
            );
            let definition: toml::Value =
                fs::read_to_string(root.join("data/providers").join(format!("{profile}.toml")))
                    .unwrap()
                    .parse()
                    .unwrap();
            assert_eq!(
                definition["provider_id"].as_str(),
                Some(id),
                "provider identity differs for {profile}"
            );
        }
    }

    let bundled = fs::read_dir(root.join("data/providers"))
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.unwrap();
            let path = entry.path();
            (path.extension().and_then(|extension| extension.to_str()) == Some("toml"))
                .then(|| path.file_stem().unwrap().to_str().unwrap().to_owned())
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(indexed, bundled, "provider evidence index is stale");
}
