//! Evidence coverage is separate from runtime routing and live API validation.
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

const MATRIX: &str = include_str!("../data/capability-matrix-openai-gemini.json");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Matrix {
    schema_version: u32,
    status: String,
    evidence_date: String,
    live_validation: String,
    scope: serde_json::Value,
    basis_definitions: BTreeMap<String, String>,
    operations: BTreeMap<String, Operation>,
    sources: BTreeMap<String, Source>,
    profiles: Vec<Profile>,
    services: Vec<Service>,
    retired_models: Vec<RetiredModel>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Operation {
    category: String,
    scope: String,
    endpoint: String,
    description: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    url: String,
    evidence_date: String,
    section: String,
    note: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    profile: String,
    provider_id: String,
    catalog_path: String,
    operation_ids: Vec<String>,
    models: Vec<Model>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {
    model_id: String,
    evidence_date: String,
    live_validation: String,
    account_region_validation: String,
    availability: Availability,
    cells: BTreeMap<String, Cell>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RetiredModel {
    profile: String,
    provider_id: String,
    model_id: String,
    evidence_date: String,
    live_validation: String,
    account_region_validation: String,
    availability: Availability,
    cells: BTreeMap<String, Cell>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Availability {
    state: String,
    source_ids: Vec<String>,
    note: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cell {
    status: Status,
    source_ids: Vec<String>,
    basis: String,
    note: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Supported,
    Unsupported,
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Service {
    profile: String,
    operation_id: String,
    evidence_date: String,
    live_validation: String,
    account_region_validation: String,
    cell: Cell,
}

fn matrix() -> Matrix {
    serde_json::from_str(MATRIX).expect("the evidence matrix must match its schema")
}

fn model<'a>(matrix: &'a Matrix, profile: &str, id: &str) -> &'a Model {
    matrix
        .profiles
        .iter()
        .find(|row| row.profile == profile)
        .unwrap()
        .models
        .iter()
        .find(|row| row.model_id == id)
        .unwrap()
}

fn status(matrix: &Matrix, profile: &str, id: &str, operation: &str) -> Status {
    model(matrix, profile, id).cells[operation].status
}

#[test]
fn every_raw_catalog_model_has_exactly_one_complete_operation_row() {
    let matrix = matrix();
    let catalogs = [
        (
            "openai",
            "openai",
            include_str!("../data/providers/openai.toml"),
        ),
        (
            "gemini",
            "google",
            include_str!("../data/providers/gemini.toml"),
        ),
    ];
    assert_eq!(matrix.profiles.len(), catalogs.len());
    assert_eq!(
        matrix.scope["catalog_profiles"],
        serde_json::json!(["openai", "gemini"])
    );
    let mut profiles = BTreeSet::new();
    let required_categories: BTreeSet<_> = [
        "structured_output",
        "prompt_cache",
        "hosted_tools",
        "retrieval",
        "embeddings",
        "batch_background",
        "audio_realtime",
    ]
    .into_iter()
    .collect();
    for (profile, provider_id, catalog) in catalogs {
        let rows = matrix
            .profiles
            .iter()
            .find(|row| row.profile == profile)
            .expect("both catalog profiles must be audited");
        assert!(profiles.insert(&rows.profile));
        assert_eq!(rows.provider_id, provider_id);
        assert_eq!(rows.catalog_path, format!("data/providers/{profile}.toml"));
        let catalog: toml::Value = toml::from_str(catalog).unwrap();
        // Raw catalog rows include embedding, realtime and agent models filtered
        // out of completion presets. Auditing only builtin().models loses them.
        let expected: BTreeSet<_> = catalog["model"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].as_str().unwrap())
            .collect();
        let actual: BTreeSet<_> = rows
            .models
            .iter()
            .map(|row| row.model_id.as_str())
            .collect();
        assert_eq!(actual.len(), rows.models.len(), "duplicate {profile} model");
        assert_eq!(actual, expected, "raw catalog coverage drift for {profile}");

        let operations: BTreeSet<_> = rows.operation_ids.iter().map(String::as_str).collect();
        assert_eq!(operations.len(), rows.operation_ids.len());
        let categories: BTreeSet<_> = operations
            .iter()
            .map(|id| {
                let operation = &matrix.operations[*id];
                assert_eq!(operation.scope, "model");
                operation.category.as_str()
            })
            .collect();
        assert!(required_categories.is_subset(&categories));
        for row in &rows.models {
            assert_eq!(
                row.cells
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
                operations,
                "{} must have an explicit cell, including Unknown, for every operation",
                row.model_id
            );
        }
    }
}

fn check_evidence_date(matrix: &Matrix, date: &str) {
    let date = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap();
    let latest = chrono::NaiveDate::parse_from_str(&matrix.evidence_date, "%Y-%m-%d").unwrap();
    assert!(
        date <= latest,
        "evidence is newer than the matrix review date"
    );
}

fn check_sources(matrix: &Matrix, profile: &str, ids: &[String], used: &mut BTreeSet<String>) {
    assert!(
        !ids.is_empty(),
        "even Unknown needs a reviewed source context"
    );
    let unique: BTreeSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), ids.len());
    for id in ids {
        let source = matrix.sources.get(id).expect("dangling source reference");
        let url = url::Url::parse(&source.url).expect("source must be a URL");
        assert_eq!(url.scheme(), "https");
        let allowed = match profile {
            "openai" => matches!(
                url.host_str(),
                Some("developers.openai.com" | "platform.openai.com")
            ),
            "gemini" => matches!(url.host_str(), Some("ai.google.dev")),
            _ => panic!("unexpected profile {profile}"),
        };
        assert!(
            allowed,
            "source must be first-party for {profile}: {}",
            source.url
        );
        check_evidence_date(matrix, &source.evidence_date);
        assert!(!source.section.trim().is_empty());
        if let Some(note) = &source.note {
            assert!(!note.trim().is_empty());
        }
        used.insert(id.clone());
    }
}

fn check_cell(matrix: &Matrix, profile: &str, cell: &Cell, used: &mut BTreeSet<String>) {
    check_sources(matrix, profile, &cell.source_ids, used);
    assert!(matrix.basis_definitions.contains_key(&cell.basis));
    match cell.status {
        Status::Unknown => assert_eq!(cell.basis, "not_established"),
        Status::Unsupported => assert_eq!(cell.basis, "explicit_limitation"),
        Status::Supported => assert!(matches!(
            cell.basis.as_str(),
            "explicit_support" | "documented_family" | "documented_alias"
        )),
    }
    if let Some(note) = &cell.note {
        assert!(!note.trim().is_empty());
    }
}

#[test]
fn every_claim_resolves_to_dated_first_party_evidence_without_live_claims() {
    let matrix = matrix();
    assert_eq!(matrix.schema_version, 1);
    assert_eq!(
        matrix.status,
        "documentation_evidence_matrix_not_runtime_configuration"
    );
    assert_eq!(matrix.live_validation, "not_run");
    assert_eq!(
        matrix.scope["implementation_validation"],
        "not_assessed_by_this_matrix"
    );
    let mut used_sources = BTreeSet::new();
    let mut used_operations = BTreeSet::new();
    for profile in &matrix.profiles {
        for row in &profile.models {
            check_evidence_date(&matrix, &row.evidence_date);
            assert_eq!(row.live_validation, "not_run");
            assert_eq!(row.account_region_validation, "unknown");
            assert!(matches!(
                row.availability.state.as_str(),
                "unknown" | "documented" | "restricted" | "deprecated" | "shut_down"
            ));
            assert!(!row.availability.note.trim().is_empty());
            check_sources(
                &matrix,
                &profile.profile,
                &row.availability.source_ids,
                &mut used_sources,
            );
            for (operation, cell) in &row.cells {
                used_operations.insert(operation.clone());
                check_cell(&matrix, &profile.profile, cell, &mut used_sources);
            }
        }
    }
    let mut retired_ids = BTreeSet::new();
    for row in &matrix.retired_models {
        assert!(retired_ids.insert((&row.profile, &row.model_id)));
        let profile = matrix
            .profiles
            .iter()
            .find(|p| p.profile == row.profile)
            .unwrap();
        assert_eq!(row.provider_id, profile.provider_id);
        assert!(!profile
            .models
            .iter()
            .any(|active| active.model_id == row.model_id));
        check_evidence_date(&matrix, &row.evidence_date);
        assert_eq!(row.live_validation, "not_run");
        assert_eq!(row.account_region_validation, "unknown");
        assert_eq!(row.availability.state, "shut_down");
        assert!(!row.availability.note.trim().is_empty());
        check_sources(
            &matrix,
            &row.profile,
            &row.availability.source_ids,
            &mut used_sources,
        );
        assert_eq!(
            row.cells.keys().collect::<BTreeSet<_>>(),
            profile.operation_ids.iter().collect()
        );
        for (operation, cell) in &row.cells {
            assert_eq!(matrix.operations[operation].scope, "model");
            used_operations.insert(operation.clone());
            check_cell(&matrix, &row.profile, cell, &mut used_sources);
        }
    }
    let mut services = BTreeSet::new();
    for service in &matrix.services {
        assert!(services.insert((&service.profile, &service.operation_id)));
        assert!(service
            .operation_id
            .starts_with(&format!("{}.", service.profile)));
        assert_eq!(matrix.operations[&service.operation_id].scope, "service");
        check_evidence_date(&matrix, &service.evidence_date);
        assert_eq!(service.live_validation, "not_run");
        assert_eq!(service.account_region_validation, "unknown");
        check_cell(&matrix, &service.profile, &service.cell, &mut used_sources);
        used_operations.insert(service.operation_id.clone());
    }
    assert_eq!(used_sources, matrix.sources.keys().cloned().collect());
    assert_eq!(used_operations, matrix.operations.keys().cloned().collect());
    for operation in matrix.operations.values() {
        assert!(!operation.endpoint.is_empty());
        assert!(!operation.description.is_empty());
    }
}

#[test]
fn openai_snapshots_and_cache_generations_do_not_inherit_broad_family_claims() {
    let matrix = matrix();
    assert_eq!(
        status(
            &matrix,
            "openai",
            "gpt-4o-2024-05-13",
            "responses.json_schema"
        ),
        Status::Unsupported
    );
    assert_eq!(
        status(
            &matrix,
            "openai",
            "gpt-4o-2024-08-06",
            "responses.json_schema"
        ),
        Status::Supported
    );
    assert_eq!(
        status(&matrix, "openai", "gpt-5.2-pro", "responses.json_schema"),
        Status::Unsupported
    );
    assert_eq!(
        status(
            &matrix,
            "openai",
            "gpt-5.4-pro",
            "responses.code_interpreter"
        ),
        Status::Unsupported
    );
    assert_eq!(
        status(&matrix, "openai", "gpt-5.4", "responses.cache_breakpoints"),
        Status::Unsupported
    );
    assert_eq!(
        status(
            &matrix,
            "openai",
            "gpt-5.4",
            "responses.cache_retention_24h"
        ),
        Status::Supported
    );
    assert_eq!(
        status(
            &matrix,
            "openai",
            "gpt-6-sol",
            "responses.cache_breakpoints"
        ),
        Status::Supported
    );
    assert_eq!(
        status(&matrix, "openai", "gpt-6-sol", "responses.cache_ttl_30m"),
        Status::Supported
    );
    assert_eq!(
        status(
            &matrix,
            "openai",
            "gpt-6-sol",
            "responses.cache_retention_24h"
        ),
        Status::Unknown
    );
    // The Spark page could not establish HTTP API operations. Name similarity
    // to Codex models must not turn its catalog claims into API evidence.
    assert!(model(&matrix, "openai", "gpt-5.3-codex-spark")
        .cells
        .values()
        .all(|cell| cell.status == Status::Unknown));
}

#[test]
fn gemini_endpoint_restrictions_do_not_override_agent_or_resource_contracts() {
    let matrix = matrix();
    assert_eq!(
        status(
            &matrix,
            "gemini",
            "gemini-3.1-pro-preview",
            "interactions.remote_mcp"
        ),
        Status::Unsupported
    );
    assert_eq!(
        status(
            &matrix,
            "gemini",
            "deep-research-max-preview-04-2026",
            "interactions.remote_mcp"
        ),
        Status::Supported
    );
    assert_eq!(
        status(
            &matrix,
            "gemini",
            "deep-research-max-preview-04-2026",
            "interactions.background"
        ),
        Status::Supported
    );
    assert_eq!(
        status(
            &matrix,
            "gemini",
            "deep-research-max-preview-04-2026",
            "generate_content.json_schema"
        ),
        Status::Unsupported
    );
    assert_eq!(
        status(
            &matrix,
            "gemini",
            "gemini-2.5-flash",
            "gemini.batch.generate_content.submit"
        ),
        Status::Supported
    );
    assert_eq!(
        status(&matrix, "gemini", "gemini-2.5-flash", "interactions.batch"),
        Status::Unsupported
    );
    assert_eq!(
        status(
            &matrix,
            "gemini",
            "gemini-2.5-flash",
            "interactions.explicit_cache"
        ),
        Status::Unsupported
    );
    let resource = matrix
        .services
        .iter()
        .find(|row| row.operation_id == "gemini.cache.resources.create")
        .unwrap();
    assert_eq!(resource.cell.status, Status::Supported);
    // A supported resource endpoint does not prove every model accepts it.
    assert_eq!(
        status(
            &matrix,
            "gemini",
            "gemini-2.5-flash",
            "generate_content.cached_content"
        ),
        Status::Unknown
    );
}

#[test]
fn gemini_batch_generation_and_embedding_have_distinct_model_evidence() {
    let matrix = matrix();
    let gemini = matrix
        .profiles
        .iter()
        .find(|profile| profile.profile == "gemini")
        .unwrap();
    let openai = matrix
        .profiles
        .iter()
        .find(|profile| profile.profile == "openai")
        .unwrap();

    assert!(!gemini.operation_ids.iter().any(|id| id == "batch.submit"));
    assert!(openai.operation_ids.iter().any(|id| id == "batch.submit"));
    let generate = &matrix.operations["gemini.batch.generate_content.submit"];
    let embed = &matrix.operations["gemini.batch.embed_content.submit"];
    assert_eq!(generate.scope, "model");
    assert_eq!(
        generate.endpoint,
        "POST /v1beta/models/{model}:batchGenerateContent"
    );
    assert_eq!(embed.scope, "model");
    assert_eq!(
        embed.endpoint,
        "POST /v1beta/models/{model}:asyncBatchEmbedContent"
    );

    assert_eq!(
        status(
            &matrix,
            "gemini",
            "gemini-embedding-001",
            "gemini.batch.embed_content.submit"
        ),
        Status::Supported
    );
    assert_eq!(
        status(
            &matrix,
            "gemini",
            "gemini-embedding-2",
            "gemini.batch.embed_content.submit"
        ),
        Status::Supported
    );
    assert_eq!(
        status(
            &matrix,
            "gemini",
            "gemini-embedding-001",
            "gemini.batch.generate_content.submit"
        ),
        Status::Unknown
    );
    assert_eq!(
        status(
            &matrix,
            "gemini",
            "gemini-2.5-flash",
            "gemini.batch.embed_content.submit"
        ),
        Status::Unknown
    );

    for operation_id in [
        "gemini.batch.get",
        "gemini.batch.list",
        "gemini.batch.cancel",
        "gemini.batch.delete",
        "gemini.batch.results",
    ] {
        let service = matrix
            .services
            .iter()
            .find(|service| service.operation_id == operation_id)
            .unwrap();
        assert_eq!(matrix.operations[operation_id].scope, "service");
        assert_eq!(service.cell.status, Status::Supported);
    }
}

#[test]
fn gemini_model_directory_is_service_scoped_and_does_not_imply_embedding_support() {
    let matrix = matrix();
    let gemini = matrix
        .profiles
        .iter()
        .find(|profile| profile.profile == "gemini")
        .unwrap();
    assert_eq!(gemini.operation_ids.len(), 21);
    assert!(!gemini
        .operation_ids
        .iter()
        .any(|operation| operation.starts_with("gemini.models.")));

    let list = &matrix.operations["gemini.models.list"];
    let get = &matrix.operations["gemini.models.get"];
    assert_eq!(list.category, "model_catalog");
    assert_eq!(list.scope, "service");
    assert_eq!(list.endpoint, "GET /v1beta/models");
    assert!(list.description.contains("supportedGenerationMethods"));
    assert!(list.description.contains("embedContent"));
    assert_eq!(get.category, "model_catalog");
    assert_eq!(get.scope, "service");
    assert_eq!(get.endpoint, "GET /v1beta/{name=models/*}");

    let source = &matrix.sources["google.models-api"];
    assert_eq!(source.url, "https://ai.google.dev/api/models");
    assert!(source.section.contains("models.list"));
    assert!(source.section.contains("models.get"));

    for operation_id in ["gemini.models.list", "gemini.models.get"] {
        let service = matrix
            .services
            .iter()
            .find(|service| service.operation_id == operation_id)
            .unwrap();
        assert_eq!(service.profile, "gemini");
        assert_eq!(service.cell.status, Status::Supported);
        assert_eq!(service.cell.basis, "explicit_support");
        assert_eq!(
            service.cell.source_ids,
            vec!["google.models-api".to_owned()]
        );
        assert_eq!(service.live_validation, "not_run");
        assert_eq!(service.account_region_validation, "unknown");
        assert!(service
            .cell
            .note
            .as_deref()
            .is_some_and(|note| { note.contains("account") && note.contains("regional") }));
    }
    assert!(matrix.services.iter().any(|service| {
        service.operation_id == "gemini.models.list"
            && service.cell.note.as_deref().is_some_and(|note| {
                note.contains("client-side projection")
                    && note.contains("not a guarantee that every listed model supports embeddings")
            })
    }));
    assert_eq!(
        status(&matrix, "gemini", "gemini-2.5-flash", "embeddings.create"),
        Status::Unknown,
        "catalog filtering must not turn every listed model into embedding support"
    );
}

#[test]
fn unknown_aliases_and_availability_remain_separate_from_feature_support() {
    let matrix = matrix();
    for id in ["gemini-flash-latest", "gemini-flash-lite-latest"] {
        assert_eq!(
            status(&matrix, "gemini", id, "generate_content.json_schema"),
            Status::Unknown
        );
        assert_eq!(
            status(&matrix, "gemini", id, "interactions.create"),
            Status::Unknown
        );
    }
    let retired_ids: BTreeSet<_> = matrix
        .retired_models
        .iter()
        .map(|r| r.model_id.as_str())
        .collect();
    assert_eq!(
        retired_ids,
        [
            "gemini-3-pro-image-preview",
            "gemini-3.1-flash-image-preview",
            "gemini-3.1-flash-lite-preview"
        ]
        .into_iter()
        .collect()
    );
    for row in &matrix.retired_models {
        assert_eq!(row.availability.state, "shut_down");
        assert_eq!(row.availability.source_ids, ["google.deprecations"]);
        assert!(row.availability.note.contains(
            if row.model_id == "gemini-3.1-flash-lite-preview" {
                "2026-05-25"
            } else {
                "2026-06-25"
            }
        ));
    }
    for id in [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ] {
        assert_eq!(
            model(&matrix, "gemini", id).availability.state,
            "restricted"
        );
    }
    assert_eq!(
        status(&matrix, "gemini", "lyria-3-clip-preview", "audio.output"),
        Status::Supported
    );
    assert_eq!(
        status(&matrix, "gemini", "lyria-3-pro-preview", "audio.output"),
        Status::Unknown
    );
    assert_eq!(
        status(
            &matrix,
            "gemini",
            "gemini-3.1-flash-tts-preview",
            "audio.speech"
        ),
        Status::Supported
    );
    assert_eq!(
        status(
            &matrix,
            "gemini",
            "gemini-3.1-flash-tts-preview",
            "realtime.connect"
        ),
        Status::Unsupported
    );
}
