//! Keep the independently researched evidence matrices aligned with the live
//! bundled catalog. Historical retired rows are not part of this coverage.
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[test]
fn evidence_matrices_cover_every_current_builtin_model_once() {
    let mut indexed: BTreeMap<String, (String, BTreeSet<String>)> = BTreeMap::new();
    for source in [
        include_str!("../data/capability-matrix-openai-gemini.json"),
        include_str!("../data/capability-matrix-china.json"),
        include_str!("../data/capability-matrix-west.json"),
    ] {
        let matrix: Value = serde_json::from_str(source).unwrap();
        assert_eq!(matrix["schema_version"], 1);
        assert_eq!(matrix["live_validation"], "not_run");
        for profile in matrix["profiles"].as_array().unwrap() {
            let name = profile["profile"].as_str().unwrap();
            let provider = profile["provider_id"].as_str().unwrap();
            let models = profile["models"].as_array().unwrap();
            let model_ids = models
                .iter()
                .map(|model| model["model_id"].as_str().unwrap().to_owned())
                .collect::<BTreeSet<_>>();
            assert_eq!(model_ids.len(), models.len(), "duplicate model in {name}");
            assert!(
                indexed
                    .insert(name.to_owned(), (provider.to_owned(), model_ids))
                    .is_none(),
                "duplicate profile {name}"
            );
        }
    }

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/providers");
    let mut bundled = BTreeSet::new();
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|part| part.to_str()) != Some("toml") {
            continue;
        }
        let name = path.file_stem().unwrap().to_str().unwrap();
        bundled.insert(name.to_owned());
        let toml: toml::Value = fs::read_to_string(&path).unwrap().parse().unwrap();
        let provider = toml["provider_id"].as_str().unwrap();
        let models = toml
            .get("model")
            .and_then(toml::Value::as_array)
            .map(|rows| {
                rows.iter()
                    .map(|row| row["id"].as_str().unwrap().to_owned())
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        let (indexed_provider, indexed_models) = indexed
            .get(name)
            .unwrap_or_else(|| panic!("missing profile {name}"));
        assert_eq!(indexed_provider, provider, "provider differs for {name}");
        assert_eq!(indexed_models, &models, "model coverage differs for {name}");
    }
    assert_eq!(indexed.keys().cloned().collect::<BTreeSet<_>>(), bundled);
}
