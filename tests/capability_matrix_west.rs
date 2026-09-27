//! Evidence coverage is separate from runtime routing and live API validation.
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

const MATRIX: &str = include_str!("../data/capability-matrix-west.json");

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

fn check_evidence_date(matrix: &Matrix, evidence_date: &str) {
    let snapshot = chrono::NaiveDate::parse_from_str(&matrix.evidence_date, "%Y-%m-%d")
        .expect("matrix evidence_date must be a date");
    let evidence = chrono::NaiveDate::parse_from_str(evidence_date, "%Y-%m-%d")
        .expect("record evidence_date must be a date");
    assert!(
        evidence <= snapshot,
        "record date {evidence_date} is newer than matrix snapshot {}",
        matrix.evidence_date
    );
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
fn every_remaining_raw_catalog_row_has_complete_operation_evidence() {
    let matrix = matrix();
    let already_audited = [
        "openai", "google", "deepseek", "kimi", "minimax", "qwen", "zhipu",
    ];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut expected_profiles = BTreeMap::new();
    for entry in std::fs::read_dir(root.join("data/providers")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("toml") {
            continue;
        }
        let catalog: toml::Value =
            toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let provider = catalog["provider_id"].as_str().unwrap();
        if !already_audited.contains(&provider) {
            expected_profiles.insert(
                path.file_stem().unwrap().to_str().unwrap().to_owned(),
                catalog,
            );
        }
    }
    let actual_profiles: BTreeSet<_> = matrix.profiles.iter().map(|p| p.profile.clone()).collect();
    assert_eq!(
        actual_profiles.len(),
        matrix.profiles.len(),
        "duplicate profile"
    );
    assert_eq!(actual_profiles, expected_profiles.keys().cloned().collect());
    assert_eq!(
        matrix.scope["catalog_profiles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap().to_owned())
            .collect::<BTreeSet<_>>(),
        actual_profiles
    );
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
    for profile in &matrix.profiles {
        let catalog = &expected_profiles[&profile.profile];
        assert_eq!(
            profile.provider_id,
            catalog["provider_id"].as_str().unwrap()
        );
        assert_eq!(
            profile.catalog_path,
            format!("data/providers/{}.toml", profile.profile)
        );
        let expected: BTreeSet<_> = catalog["model"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["id"].as_str().unwrap())
            .collect();
        let actual: BTreeSet<_> = profile.models.iter().map(|r| r.model_id.as_str()).collect();
        assert_eq!(actual.len(), profile.models.len(), "duplicate model");
        assert_eq!(
            actual, expected,
            "raw model coverage for {}",
            profile.profile
        );
        let operations: BTreeSet<_> = profile.operation_ids.iter().map(String::as_str).collect();
        assert_eq!(operations.len(), profile.operation_ids.len());
        let categories: BTreeSet<_> = operations
            .iter()
            .map(|id| {
                let operation = &matrix.operations[*id];
                assert_eq!(operation.scope, "model");
                operation.category.as_str()
            })
            .collect();
        assert!(required_categories.is_subset(&categories));
        for row in &profile.models {
            assert_eq!(
                row.cells
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
                operations,
                "explicit cells, including Unknown, for {}",
                row.model_id
            );
        }
    }
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
            "anthropic" => matches!(url.host_str(), Some("platform.claude.com")),
            "grok" | "grok-responses" | "grok-anthropic" => {
                matches!(url.host_str(), Some("docs.x.ai"))
            }
            "openrouter" => matches!(url.host_str(), Some("openrouter.ai")),
            "github-copilot" => matches!(url.host_str(), Some("docs.github.com")),
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
    chrono::NaiveDate::parse_from_str(&matrix.evidence_date, "%Y-%m-%d")
        .expect("matrix evidence_date must be a date");
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
fn claude_versioned_tools_and_embeddings_keep_explicit_limits() {
    let m = matrix();
    for operation in ["messages.web_search", "messages.web_fetch"] {
        assert_eq!(
            status(&m, "anthropic", "claude-sonnet-4-6", operation),
            Status::Supported
        );
        assert_eq!(
            status(&m, "anthropic", "claude-sonnet-4-5-20250929", operation),
            Status::Unknown
        );
    }
    let haiku = "claude-haiku-4-5-20251001";
    assert_eq!(
        status(&m, "anthropic", haiku, "messages.code_execution"),
        Status::Supported
    );
    assert_eq!(
        status(&m, "anthropic", haiku, "messages.programmatic_tools"),
        Status::Unsupported
    );
    assert_eq!(
        status(&m, "anthropic", "claude-sonnet-5", "messages.tool_search"),
        Status::Unknown
    );
    let profile = m
        .profiles
        .iter()
        .find(|p| p.profile == "anthropic")
        .unwrap();
    for row in &profile.models {
        assert_eq!(row.cells["embeddings.create"].status, Status::Unsupported);
        assert_eq!(
            row.cells["messages.json_schema_with_citations"].status,
            Status::Unsupported
        );
        assert_eq!(row.cells["batch.submit"].status, Status::Supported);
        assert_eq!(row.availability.state, "documented");
    }
}

#[test]
fn anthropic_stable_client_toolsets_have_exact_model_and_platform_boundaries() {
    let m = matrix();
    let profile = m
        .profiles
        .iter()
        .find(|p| p.profile == "anthropic")
        .unwrap();
    // The official compatibility table names eight model versions. Six are
    // present as raw rows in this catalog; the two limited-availability
    // Mythos IDs are intentionally not synthesized as catalog rows.
    let expected_supported: BTreeSet<_> = [
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-sonnet-5",
    ]
    .into_iter()
    .collect();
    let documented_ids = [
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-mythos-5",
        "claude-mythos-5-1",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-sonnet-5",
    ];
    for source in ["anthropic.browser_toolset", "anthropic.computer_toolset"] {
        let note = m.sources[source].note.as_deref().unwrap();
        for id in documented_ids {
            assert!(note.contains(id), "{source} does not record {id}");
        }
    }
    assert!(!profile.models.iter().any(|row| matches!(
        row.model_id.as_str(),
        "claude-mythos-5" | "claude-mythos-5-1"
    )));

    for operation in ["messages.browser_toolset", "messages.computer_toolset"] {
        assert_eq!(m.operations[operation].category, "client_toolsets");
        assert_eq!(m.operations[operation].scope, "model");
        let actual_supported: BTreeSet<_> = profile
            .models
            .iter()
            .filter(|row| row.cells[operation].status == Status::Supported)
            .map(|row| row.model_id.as_str())
            .collect();
        assert_eq!(actual_supported, expected_supported, "{operation}");
        for row in &profile.models {
            if !expected_supported.contains(&row.model_id.as_str()) {
                assert_eq!(
                    row.cells[operation].status,
                    Status::Unknown,
                    "{}",
                    row.model_id
                );
            }
        }
    }

    for (operation, expected) in [
        ("anthropic.vertex.browser_toolset", Status::Supported),
        ("anthropic.vertex.computer_toolset", Status::Supported),
        ("anthropic.foundry.browser_toolset", Status::Unsupported),
        ("anthropic.foundry.computer_toolset", Status::Unsupported),
    ] {
        assert_eq!(m.operations[operation].scope, "service");
        let service = m
            .services
            .iter()
            .find(|service| service.operation_id == operation)
            .unwrap();
        assert_eq!(service.cell.status, expected, "{operation}");
        assert_eq!(service.live_validation, "not_run");
        assert_eq!(service.account_region_validation, "unknown");
    }
}

#[test]
fn foundry_hosting_and_tool_versions_have_service_evidence_without_model_inheritance() {
    let m = matrix();
    let service = |id: &str| m.services.iter().find(|s| s.operation_id == id).unwrap();
    let foundry_operations = [
        "anthropic.foundry.tool_search",
        "anthropic.foundry.azure.web_fetch",
        "anthropic.foundry.anthropic.web_fetch",
        "anthropic.foundry.remote_mcp",
    ];
    for id in foundry_operations {
        let operation = &m.operations[id];
        assert_eq!(operation.scope, "service");
        assert!(operation
            .endpoint
            .contains("services.ai.azure.com/anthropic/v1/messages"));
        let record = service(id);
        assert_eq!(record.cell.status, Status::Supported);
        assert_eq!(record.evidence_date, "2026-09-27");
        assert_eq!(record.live_validation, "not_run");
        assert_eq!(record.account_region_validation, "unknown");
        assert!(record
            .cell
            .source_ids
            .iter()
            .any(|s| s == "anthropic.foundry"));
        assert!(m
            .profiles
            .iter()
            .all(|p| !p.operation_ids.iter().any(|op| op == id)
                && p.models.iter().all(|model| !model.cells.contains_key(id))));
    }
    for id in [
        "anthropic.foundry.tool_search",
        "anthropic.foundry.remote_mcp",
    ] {
        let record = service(id);
        let note = record.cell.note.as_deref().unwrap();
        assert!(note.contains("Hosted on Azure") && note.contains("Hosted on Anthropic"));
        assert!(record
            .cell
            .source_ids
            .iter()
            .any(|s| s == "anthropic.feature_overview"));
    }
    let azure = service("anthropic.foundry.azure.web_fetch")
        .cell
        .note
        .as_deref()
        .unwrap();
    assert!(azure.contains("only web_fetch_20250910"));
    assert!(azure.contains("use_cache") && azure.contains("response_inclusion"));
    assert!(azure.contains("HTTP 400") && azure.contains("No live rejection was tested"));
    let anthropic = service("anthropic.foundry.anthropic.web_fetch")
        .cell
        .note
        .as_deref()
        .unwrap();
    for version in ["20250910", "20260209", "20260309", "20260318"] {
        assert!(anthropic.contains(&format!("web_fetch_{version}")));
    }
    let mcp = service("anthropic.foundry.remote_mcp")
        .cell
        .note
        .as_deref()
        .unwrap();
    assert!(mcp.contains("mcp-client-2025-11-20 beta"));
    assert!(mcp.contains("not a stable toolset"));
    let pinned = service("anthropic.api.mcp_tool_list_pinning");
    assert_eq!(pinned.cell.status, Status::Supported);
    assert_eq!(
        m.operations[&pinned.operation_id].endpoint,
        "POST https://api.anthropic.com/v1/messages"
    );
    assert!(pinned
        .cell
        .note
        .as_deref()
        .unwrap()
        .contains("Foundry support remains unestablished"));
    // Independent platform evidence must not fabricate a Foundry catalog or
    // relax previously documented stable client-toolset exclusions.
    assert!(m.profiles.iter().all(|p| !p.profile.contains("foundry")));
    for id in [
        "anthropic.foundry.browser_toolset",
        "anthropic.foundry.computer_toolset",
    ] {
        assert_eq!(service(id).cell.status, Status::Unsupported);
    }
}

#[test]
fn xai_protocol_and_batch_model_boundaries_remain_independent() {
    let m = matrix();
    for profile in ["grok", "grok-responses", "grok-anthropic"] {
        for id in ["grok-4.20", "grok-4.20-multi-agent", "grok-4.3"] {
            assert_eq!(status(&m, profile, id, "batch.submit"), Status::Supported);
        }
        for id in ["grok-4.5", "grok-4.6", "grok-4.7", "grok-build-0.1"] {
            assert_eq!(status(&m, profile, id, "batch.submit"), Status::Unsupported);
        }
    }
    for row in &m
        .profiles
        .iter()
        .find(|p| p.profile == "grok-anthropic")
        .unwrap()
        .models
    {
        assert!(row
            .cells
            .iter()
            .filter(|(op, _)| op.starts_with("messages."))
            .all(|(_, cell)| cell.status == Status::Unknown));
    }
    assert_eq!(
        status(&m, "grok-responses", "grok-4.7", "responses.background"),
        Status::Unsupported
    );
    assert_eq!(
        status(&m, "grok", "grok-4.7", "chat.deferred"),
        Status::Supported
    );
    assert_eq!(
        status(&m, "grok", "grok-4.7", "chat.remote_mcp"),
        Status::Unknown
    );
    assert_eq!(
        status(&m, "grok-responses", "grok-4.7", "responses.remote_mcp"),
        Status::Supported
    );
    assert!(model(&m, "grok", "grok-4.7").cells["chat.deferred"]
        .note
        .as_ref()
        .unwrap()
        .contains("once"));
    assert!(m.operations["grok.collections.create"]
        .endpoint
        .contains("management-api.x.ai"));
    assert!(m.operations["grok.collections.search"]
        .endpoint
        .contains("https://api.x.ai"));
}

#[test]
fn xai_streaming_tts_and_custom_voice_endpoints_are_service_only() {
    let m = matrix();
    let service = |id: &str| m.services.iter().find(|s| s.operation_id == id).unwrap();
    let expected = [
        "grok.audio.synthesize_stream",
        "grok.custom_voices.create",
        "grok.custom_voices.list",
        "grok.custom_voices.get",
        "grok.custom_voices.update",
        "grok.custom_voices.delete",
        "grok.custom_voices.audio",
    ];
    for id in expected {
        assert_eq!(m.operations[id].scope, "service");
        let entry = service(id);
        assert_eq!(entry.cell.status, Status::Supported);
        assert_eq!(entry.evidence_date, "2026-09-26");
        assert_eq!(entry.live_validation, "not_run");
        assert_eq!(entry.account_region_validation, "unknown");
    }

    assert_eq!(
        m.operations["grok.audio.synthesize_stream"].endpoint,
        "wss://api.x.ai/v1/tts"
    );
    assert_eq!(
        m.operations["grok.custom_voices.audio"].endpoint,
        "GET https://api.x.ai/v1/custom-voices/{voice_id}/audio"
    );
    assert!(m.operations["grok.custom_voices.create"]
        .description
        .contains("Enterprise-gated"));
    let custom_source = &m.sources["xai.custom_voices"];
    assert_eq!(custom_source.evidence_date, "2026-09-26");
    assert!(custom_source.url.ends_with("/audio/custom-voices"));
    assert!(custom_source.note.as_deref().unwrap().contains("Illinois"));
    let tts_source = &m.sources["xai.tts"];
    assert_eq!(tts_source.evidence_date, "2026-09-26");
    assert!(tts_source.note.as_deref().unwrap().contains("2026-09-19"));

    for profile in ["grok", "grok-responses", "grok-anthropic"] {
        let profile = m.profiles.iter().find(|p| p.profile == profile).unwrap();
        for id in expected {
            assert!(!profile
                .operation_ids
                .iter()
                .any(|model_op| model_op.as_str() == id));
        }
    }
}

#[test]
fn openrouter_evidence_uses_schema_flags_modalities_and_batch_variants() {
    let m = matrix();
    for id in ["google/lyria-3-clip-preview", "google/lyria-3-pro-preview"] {
        assert_eq!(
            status(&m, "openrouter", id, "chat.json_schema"),
            Status::Unsupported
        );
    }
    assert_eq!(
        status(&m, "openrouter", "aion-labs/aion-2.0", "chat.json_schema"),
        Status::Unknown
    );
    assert_eq!(
        status(&m, "openrouter", "openai/gpt-4o", "chat.json_schema"),
        Status::Supported
    );
    assert_eq!(
        status(&m, "openrouter", "openai/gpt-4o", "batch.submit"),
        Status::Supported
    );
    assert_eq!(
        status(&m, "openrouter", "openai/gpt-5.6-sol", "chat.cache_ttl_30m"),
        Status::Supported
    );
    assert_eq!(
        status(&m, "openrouter", "openai/gpt-4o", "chat.cache_ttl_30m"),
        Status::Unknown
    );
    assert_eq!(
        status(&m, "openrouter", "openai/gpt-4o", "chat.response_cache"),
        Status::Supported
    );
    assert_eq!(
        status(&m, "openrouter", "openai/gpt-audio", "chat.audio_input"),
        Status::Supported
    );
    assert_eq!(
        status(&m, "openrouter", "openai/gpt-audio", "chat.audio_output"),
        Status::Supported
    );
    assert_eq!(
        status(&m, "openrouter", "openai/gpt-audio", "audio.synthesize"),
        Status::Unknown
    );
    let profile = m
        .profiles
        .iter()
        .find(|p| p.profile == "openrouter")
        .unwrap();
    let missing = m.scope["openrouter_directory"]["missing_catalog_models"]
        .as_array()
        .unwrap();
    let documented = profile
        .models
        .iter()
        .filter(|r| r.availability.state == "documented")
        .count();
    assert_eq!(
        documented,
        m.scope["openrouter_directory"]["matched_catalog_models"]
            .as_u64()
            .unwrap() as usize
    );
    assert_eq!(documented + missing.len(), profile.models.len());
    for id in missing {
        let row = model(&m, "openrouter", id.as_str().unwrap());
        assert_eq!(row.availability.state, "unknown");
        assert!(row
            .cells
            .values()
            .all(|cell| cell.status == Status::Unknown));
    }
    assert_eq!(
        profile
            .models
            .iter()
            .filter(|r| r.cells["chat.audio_input"].status == Status::Supported)
            .count(),
        27
    );
    assert_eq!(
        profile
            .models
            .iter()
            .filter(|r| r.cells["chat.audio_output"].status == Status::Supported)
            .count(),
        4
    );
    assert_eq!(
        profile
            .models
            .iter()
            .filter(|r| r.cells["batch.submit"].status == Status::Supported)
            .count(),
        63
    );
}

#[test]
fn copilot_and_independent_services_do_not_inherit_native_chat_capabilities() {
    let m = matrix();
    let copilot = m
        .profiles
        .iter()
        .find(|p| p.profile == "github-copilot")
        .unwrap();
    assert!(copilot
        .models
        .iter()
        .all(|r| r.availability.state == "unknown"
            && r.cells.values().all(|c| c.status == Status::Unknown)));
    let service = |id: &str| m.services.iter().find(|s| s.operation_id == id).unwrap();
    for id in [
        "openrouter.embeddings.create",
        "openrouter.rerank.create",
        "openrouter.audio.transcribe",
        "openrouter.audio.synthesize",
        "openrouter.responses.tool_search",
        "openrouter.messages.shell",
        "grok.realtime.connect",
    ] {
        assert_eq!(service(id).cell.status, Status::Supported);
    }
    assert_eq!(
        service("openrouter.batch.cancel").cell.status,
        Status::Unknown
    );
    assert_eq!(
        service("openrouter.batch.delete").cell.status,
        Status::Supported
    );
    assert_eq!(
        status(&m, "openrouter", "openai/gpt-4o", "chat.tool_search"),
        Status::Unsupported
    );
    assert_eq!(
        status(&m, "openrouter", "openai/gpt-4o", "embeddings.create"),
        Status::Unknown
    );
    assert_eq!(
        status(&m, "grok", "grok-4.7", "realtime.connect"),
        Status::Unknown
    );
}

#[test]
fn foundry_execution_evidence_distinguishes_hosting_without_claiming_live_access() {
    let matrix = matrix();
    for feature in ["code_execution", "programmatic_tools"] {
        for (hosting, status) in [
            ("anthropic", Status::Supported),
            ("azure", Status::Unsupported),
        ] {
            let id = format!("anthropic.foundry.{hosting}.{feature}");
            let service = matrix
                .services
                .iter()
                .find(|row| row.operation_id == id)
                .unwrap();
            assert_eq!(service.cell.status, status);
            assert_eq!(service.live_validation, "not_run");
            assert_eq!(service.account_region_validation, "unknown");
            assert!(service
                .cell
                .source_ids
                .iter()
                .any(|id| id == "anthropic.foundry"));
        }
    }
}

#[test]
fn foundry_skills_evidence_separates_content_download_and_hosting() {
    let m = matrix();
    for hosting in ["anthropic", "azure"] {
        for operation in [
            "create",
            "list",
            "get",
            "delete",
            "versions.create",
            "versions.list",
            "versions.get",
            "versions.delete",
            "versions.download",
        ] {
            let id = format!("anthropic.foundry.{hosting}.skills.{operation}");
            let row = m
                .services
                .iter()
                .find(|row| row.operation_id == id)
                .unwrap();
            let supported = hosting == "anthropic" && operation != "versions.download";
            assert_eq!(
                row.cell.status,
                if supported {
                    Status::Supported
                } else {
                    Status::Unsupported
                }
            );
            assert_eq!(row.live_validation, "not_run");
            assert_eq!(row.account_region_validation, "unknown");
        }
    }
}
