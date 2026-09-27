//! Offline coverage and evidence integrity, not live provider acceptance.
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

const PROVIDERS: &[&str] = &["qwen", "deepseek", "kimi", "minimax", "zhipu"];

fn matrix() -> Value {
    serde_json::from_str(include_str!("../data/capability-matrix-china.json")).unwrap()
}

fn strings(value: &Value) -> BTreeSet<&str> {
    let values = value.as_array().expect("expected string array");
    let result = values.iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(values.len(), BTreeSet::len(&result), "duplicate string");
    result
}

fn catalog(profile: &str) -> toml::Value {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data/providers")
            .join(format!("{profile}.toml")),
    )
    .unwrap()
    .parse()
    .unwrap()
}

fn first_party(url: &str) -> bool {
    let url = url::Url::parse(url).unwrap();
    url.scheme() == "https"
        && url.username().is_empty()
        && url.password().is_none()
        && match url.host_str().unwrap() {
            "help.aliyun.com"
            | "www.alibabacloud.com"
            | "api-docs.deepseek.com"
            | "platform.kimi.com"
            | "platform.kimi.ai"
            | "platform.moonshot.cn"
            | "platform.minimax.io"
            | "platform.minimax.cn"
            | "platform.minimaxi.com"
            | "docs.bigmodel.cn"
            | "docs.z.ai" => true,
            "github.com" => ["/MetaGLM/", "/zai-org/", "/MiniMax-AI/"]
                .iter()
                .any(|p| url.path().starts_with(p)),
            _ => false,
        }
}

fn check_sources(value: &Value, matrix: &Value, required: bool) {
    let ids = strings(value);
    assert!(!required || !ids.is_empty(), "claim has no evidence");
    for id in ids {
        let source = matrix["sources"].get(id).expect("unknown source ID");
        assert!(first_party(source["url"].as_str().unwrap()));
        let date = source["evidence_date"].as_str().unwrap();
        assert!(chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").is_ok());
        assert!(date <= matrix["evidence_date"].as_str().unwrap());
        assert!(!source["section"].as_str().unwrap().is_empty());
    }
}

fn check_cell(cell: &Value, matrix: &Value) {
    let status = cell["status"].as_str().unwrap();
    let basis = cell["basis"].as_str().unwrap();
    match status {
        "supported" => assert!(matches!(basis, "explicit_support" | "documented_family")),
        "unsupported" => assert_eq!(basis, "explicit_limitation"),
        "unknown" => assert_eq!(basis, "not_established"),
        _ => panic!("invalid capability status {status}"),
    }
    assert!(matrix["basis_definitions"].get(basis).is_some());
    check_sources(&cell["source_ids"], matrix, status != "unknown");
    for key in cell.as_object().unwrap().keys() {
        assert!(matches!(
            key.as_str(),
            "status" | "source_ids" | "basis" | "note"
        ));
    }
}

fn check_record(record: &Value, matrix: &Value) {
    let record_date =
        chrono::NaiveDate::parse_from_str(record["evidence_date"].as_str().unwrap(), "%Y-%m-%d")
            .unwrap();
    let snapshot_date =
        chrono::NaiveDate::parse_from_str(matrix["evidence_date"].as_str().unwrap(), "%Y-%m-%d")
            .unwrap();
    assert!(
        record_date <= snapshot_date,
        "record is newer than snapshot"
    );
    assert_eq!(record["live_validation"], "not_run");
    assert_eq!(record["account_region_validation"], "unknown");
    let availability = &record["availability"];
    let state = availability["state"].as_str().unwrap();
    assert!(matches!(state, "documented" | "unknown" | "shut_down"));
    assert!(!availability["note"].as_str().unwrap().is_empty());
    check_sources(&availability["source_ids"], matrix, state != "unknown");
}

#[test]
fn every_china_provider_profile_and_raw_model_row_is_covered_once() {
    let matrix = matrix();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/providers");
    let expected_profiles = fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().and_then(|x| x.to_str()) == Some("toml"))
        .filter_map(|path| {
            let definition: toml::Value = fs::read_to_string(&path).unwrap().parse().unwrap();
            PROVIDERS
                .contains(&definition["provider_id"].as_str().unwrap())
                .then(|| path.file_stem().unwrap().to_str().unwrap().to_owned())
        })
        .collect::<BTreeSet<_>>();
    let mut actual_profiles = BTreeSet::new();
    for profile in matrix["profiles"].as_array().unwrap() {
        let name = profile["profile"].as_str().unwrap();
        assert!(actual_profiles.insert(name.to_owned()), "duplicate {name}");
        let definition = catalog(name);
        for key in ["provider_id", "protocol", "base_url"] {
            assert_eq!(
                profile[key].as_str(),
                definition[key].as_str(),
                "{name}: {key}"
            );
        }
        assert_eq!(
            profile["catalog_path"],
            format!("data/providers/{name}.toml")
        );
        let expected_regions = definition["regions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<BTreeSet<_>>();
        assert_eq!(strings(&profile["catalog_regions"]), expected_regions);
        assert!(!profile["region"].as_str().unwrap().is_empty());
        assert!(matches!(
            profile["account_product"].as_str().unwrap(),
            "metered_api" | "coding_subscription"
        ));
        let expected_models = definition
            .get("model")
            .and_then(toml::Value::as_array)
            .into_iter()
            .flatten()
            .map(|model| model["id"].as_str().unwrap())
            .collect::<BTreeSet<_>>();
        let mut actual_models = BTreeSet::new();
        let expected_operations = strings(&profile["operation_ids"]);
        let prefix = match profile["protocol"].as_str().unwrap() {
            "open_ai_chat" => "chat.",
            "open_ai_responses" => "responses.",
            "anthropic_messages" => "messages.",
            protocol => panic!("new protocol needs matrix review: {protocol}"),
        };
        let registered_operations = matrix["operations"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(id, op)| {
                op["scope"] == "model"
                    && (id.starts_with(prefix)
                        || !["chat.", "responses.", "messages."]
                            .iter()
                            .any(|prefix| id.starts_with(prefix)))
            })
            .map(|(id, _)| id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            expected_operations, registered_operations,
            "{name}: omitted registry operation"
        );
        for model in profile["models"].as_array().unwrap() {
            let id = model["model_id"].as_str().unwrap();
            assert!(actual_models.insert(id), "duplicate {name}/{id}");
            let operations = model["cells"]
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            assert_eq!(
                expected_operations, operations,
                "missing operation {name}/{id}"
            );
        }
        assert_eq!(
            expected_models, actual_models,
            "stale catalog coverage: {name}"
        );
    }
    assert_eq!(expected_profiles, actual_profiles);
    assert_eq!(
        strings(&matrix["scope"]["catalog_profiles"]),
        actual_profiles.iter().map(String::as_str).collect()
    );
}

#[test]
fn evidence_schema_is_valid_and_every_claim_resolves_to_a_dated_first_party_source() {
    let matrix = matrix();
    assert_eq!(matrix["schema_version"], 1);
    assert_eq!(
        matrix["status"],
        "documentation_evidence_matrix_not_runtime_configuration"
    );
    assert_eq!(matrix["live_validation"], "not_run");
    assert_eq!(
        matrix["scope"]["implementation_validation"],
        "not_assessed_by_this_matrix"
    );
    let operations = matrix["operations"].as_object().unwrap();
    for operation in operations.values() {
        assert!(matches!(
            operation["scope"].as_str().unwrap(),
            "model" | "service"
        ));
        for key in ["category", "endpoint", "description"] {
            assert!(!operation[key].as_str().unwrap().is_empty());
        }
        assert!(matches!(
            operation["category"].as_str().unwrap(),
            "completion"
                | "structured_output"
                | "prompt_cache"
                | "hosted_tools"
                | "retrieval"
                | "embeddings"
                | "batch_background"
                | "audio_realtime"
                | "images"
        ));
        if let Some(profile_ids) = operation.get("profile_ids") {
            assert_eq!(operation["scope"], "service");
            let applicable_profiles = strings(profile_ids);
            let catalog_profiles = matrix["profiles"]
                .as_array()
                .unwrap()
                .iter()
                .map(|profile| profile["profile"].as_str().unwrap())
                .collect::<BTreeSet<_>>();
            assert!(!applicable_profiles.is_empty());
            assert!(applicable_profiles.is_subset(&catalog_profiles));
        }
    }
    for profile in matrix["profiles"].as_array().unwrap() {
        for model in profile["models"].as_array().unwrap() {
            check_record(model, &matrix);
            for (id, cell) in model["cells"].as_object().unwrap() {
                assert_eq!(operations[id]["scope"], "model");
                check_cell(cell, &matrix);
            }
        }
    }
    for retired in matrix["retired_models"].as_array().unwrap() {
        check_record(retired, &matrix);
        assert_eq!(retired["availability"]["state"], "shut_down");
        for (id, cell) in retired["cells"].as_object().unwrap() {
            assert_eq!(operations[id]["scope"], "model");
            check_cell(cell, &matrix);
        }
    }
    for service in matrix["services"].as_array().unwrap() {
        check_record(service, &matrix);
        assert_eq!(
            operations[service["operation_id"].as_str().unwrap()]["scope"],
            "service"
        );
        check_cell(&service["cell"], &matrix);
        if let Some(models) = service.get("documented_models") {
            assert!(!strings(models).is_empty());
            assert_eq!(service["cell"]["status"], "supported");
        }
    }
    let urls = matrix["sources"]
        .as_object()
        .unwrap()
        .values()
        .map(|s| s["url"].as_str().unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        urls.len(),
        matrix["sources"].as_object().unwrap().len(),
        "duplicate source URLs"
    );
    let mut referenced = BTreeSet::new();
    for profile in matrix["profiles"].as_array().unwrap() {
        for model in profile["models"].as_array().unwrap() {
            referenced.extend(strings(&model["availability"]["source_ids"]));
            for cell in model["cells"].as_object().unwrap().values() {
                referenced.extend(strings(&cell["source_ids"]));
            }
        }
    }
    for service in matrix["services"].as_array().unwrap() {
        referenced.extend(strings(&service["availability"]["source_ids"]));
        referenced.extend(strings(&service["cell"]["source_ids"]));
    }
    for retired in matrix["retired_models"].as_array().unwrap() {
        referenced.extend(strings(&retired["availability"]["source_ids"]));
        for cell in retired["cells"].as_object().unwrap().values() {
            referenced.extend(strings(&cell["source_ids"]));
        }
    }
    assert_eq!(
        referenced,
        matrix["sources"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect(),
        "unused source records conceal stale evidence"
    );
}

#[test]
fn independent_services_and_image_catalog_rows_have_complete_unique_keys() {
    let matrix = matrix();
    let mut expected = BTreeSet::new();
    for profile in matrix["profiles"].as_array().unwrap() {
        let name = profile["profile"].as_str().unwrap();
        let definition = catalog(name);
        for (id, op) in matrix["operations"].as_object().unwrap() {
            let profile_applies = op
                .get("profile_ids")
                .and_then(Value::as_array)
                .is_none_or(|profile_ids| profile_ids.iter().any(|id| id.as_str() == Some(name)));
            if op["scope"] == "service" && !id.starts_with("service.images.") && profile_applies {
                expected.insert((
                    name.to_owned(),
                    id.to_owned(),
                    String::new(),
                    String::new(),
                    String::new(),
                ));
            }
        }
        for model in definition
            .get("images")
            .and_then(|v| v.get("models"))
            .and_then(toml::Value::as_array)
            .into_iter()
            .flatten()
        {
            for op in ["service.images.generate", "service.images.edit"] {
                assert!(expected.insert((
                    name.to_owned(),
                    op.to_owned(),
                    model["display_model"].as_str().unwrap().to_owned(),
                    model["request_model"].as_str().unwrap().to_owned(),
                    model["route"].as_str().unwrap().to_owned()
                )));
            }
        }
    }
    let mut actual = BTreeSet::new();
    for service in matrix["services"].as_array().unwrap() {
        let name = service["profile"].as_str().unwrap();
        let profile = matrix["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["profile"] == name)
            .expect("orphan service");
        assert_eq!(service["provider_id"], profile["provider_id"]);
        assert_eq!(service["region"], profile["region"]);
        if let Some(profile_ids) =
            matrix["operations"][service["operation_id"].as_str().unwrap()].get("profile_ids")
        {
            assert!(
                strings(profile_ids).contains(name),
                "service record is outside its operation profile scope"
            );
        }
        let key = (
            name.to_owned(),
            service["operation_id"].as_str().unwrap().to_owned(),
            service["catalog_model_id"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            service["request_model"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            service["catalog_route"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        );
        assert!(actual.insert(key), "duplicate service record");
    }
    assert_eq!(
        expected, actual,
        "independent service or image row coverage differs"
    );
}

#[test]
fn minimax_voice_operations_are_scoped_and_documented_for_both_regions() {
    let matrix = matrix();
    assert_eq!(matrix["evidence_date"], "2026-09-27");
    assert_eq!(matrix["live_validation"], "not_run");
    assert_eq!(
        matrix["sources"]["minimax.messages.intl"]["evidence_date"],
        "2026-09-25"
    );

    let voice_sources = [
        (
            "service.voice.clone",
            "minimax.voice_clone.intl",
            "minimax.voice_clone.cn",
            "POST /v1/voice_clone",
            "voice-cloning-clone",
        ),
        (
            "service.voice.design",
            "minimax.voice_design.intl",
            "minimax.voice_design.cn",
            "POST /v1/voice_design",
            "voice-design-design",
        ),
        (
            "service.voice.get",
            "minimax.voice_get.intl",
            "minimax.voice_get.cn",
            "POST /v1/get_voice",
            "voice-management-get",
        ),
        (
            "service.voice.delete",
            "minimax.voice_delete.intl",
            "minimax.voice_delete.cn",
            "POST /v1/delete_voice",
            "voice-management-delete",
        ),
    ];
    for (operation_id, intl_source, cn_source, endpoint, page_slug) in voice_sources {
        let operation = &matrix["operations"][operation_id];
        assert_eq!(operation["scope"], "service");
        assert_eq!(operation["endpoint"], endpoint);
        assert_eq!(
            strings(&operation["profile_ids"]),
            BTreeSet::from(["minimax-intl", "minimax"])
        );

        for (profile, region, source_id, host) in [
            (
                "minimax-intl",
                "international",
                intl_source,
                "platform.minimax.io",
            ),
            (
                "minimax",
                "china_mainland",
                cn_source,
                "platform.minimax.cn",
            ),
        ] {
            let service = matrix["services"]
                .as_array()
                .unwrap()
                .iter()
                .find(|service| {
                    service["profile"] == profile && service["operation_id"] == operation_id
                })
                .expect("both regional MiniMax profiles have a voice-service record");
            assert_eq!(service["region"], region);
            assert_eq!(service["evidence_date"], "2026-09-26");
            assert_eq!(service["live_validation"], "not_run");
            assert_eq!(service["account_region_validation"], "unknown");
            assert_eq!(service["availability"]["state"], "documented");
            assert_eq!(
                service["availability"]["source_ids"],
                serde_json::json!([source_id])
            );
            assert_eq!(service["cell"]["status"], "supported");
            assert_eq!(service["cell"]["basis"], "explicit_support");
            assert_eq!(
                service["cell"]["source_ids"],
                serde_json::json!([source_id])
            );
            assert_eq!(matrix["sources"][source_id]["evidence_date"], "2026-09-26");
            assert_eq!(
                matrix["sources"][source_id]["url"],
                format!("https://{host}/docs/api-reference/{page_slug}")
            );
        }
    }

    for profile in matrix["profiles"].as_array().unwrap() {
        if matches!(
            profile["profile"].as_str().unwrap(),
            "minimax-intl" | "minimax"
        ) {
            for model in profile["models"].as_array().unwrap() {
                for (operation_id, _, _, _, _) in voice_sources {
                    assert!(model["cells"].get(operation_id).is_none());
                }
            }
        } else {
            for service in matrix["services"].as_array().unwrap() {
                assert!(
                    service["profile"] != profile["profile"]
                        || !voice_sources
                            .iter()
                            .any(|(id, _, _, _, _)| service["operation_id"] == *id)
                );
            }
        }
    }
}

#[test]
fn minimax_streaming_and_bidi_tts_routes_are_separate_and_region_scoped() {
    let matrix = matrix();
    let operation_id = "service.audio.synthesize_bidi";
    let operation = &matrix["operations"][operation_id];
    assert_eq!(operation["scope"], "service");
    assert_eq!(
        operation["endpoint"],
        "provider-specific WSS endpoint /ws/v1/t2a_v2_bidi"
    );
    assert_eq!(
        strings(&operation["profile_ids"]),
        BTreeSet::from(["minimax-intl", "minimax"])
    );
    assert!(operation["description"]
        .as_str()
        .unwrap()
        .contains("separate from ordinary streaming-text TTS"));

    for (profile, region, source_id, host) in [
        (
            "minimax-intl",
            "international",
            "minimax.speech_bidi.intl",
            "api.minimax.io",
        ),
        (
            "minimax",
            "china_mainland",
            "minimax.speech_bidi.cn",
            "api.minimax.cn",
        ),
    ] {
        let service = matrix["services"]
            .as_array()
            .unwrap()
            .iter()
            .find(|service| {
                service["profile"] == profile && service["operation_id"] == operation_id
            })
            .expect("both MiniMax regions have a Bidi TTS service record");
        assert_eq!(service["region"], region);
        assert_eq!(service["evidence_date"], "2026-09-26");
        assert_eq!(service["live_validation"], "not_run");
        assert_eq!(service["account_region_validation"], "unknown");
        assert_eq!(service["availability"]["state"], "documented");
        assert_eq!(service["cell"]["status"], "supported");
        assert_eq!(service["cell"]["basis"], "explicit_support");
        assert_eq!(
            service["availability"]["source_ids"],
            serde_json::json!([source_id])
        );
        assert_eq!(
            service["cell"]["source_ids"],
            serde_json::json!([source_id])
        );

        let source = &matrix["sources"][source_id];
        assert_eq!(source["evidence_date"], "2026-09-26");
        assert_eq!(
            source["url"],
            format!(
                "https://platform.minimax.{}/docs/api-reference/speech-t2a-websocket-bidi",
                if profile == "minimax-intl" {
                    "io"
                } else {
                    "cn"
                }
            )
        );
        let section = source["section"].as_str().unwrap();
        assert!(section.contains("page-embedded AsyncAPI"));
        assert!(section.contains(&format!("wss://{host}/ws/v1/t2a_v2_bidi")));
    }

    let bidi_rows = matrix["services"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|service| service["operation_id"] == operation_id)
        .collect::<Vec<_>>();
    assert_eq!(bidi_rows.len(), 2);
    assert_eq!(
        bidi_rows
            .iter()
            .map(|service| service["profile"].as_str().unwrap())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["minimax-intl", "minimax"])
    );

    let mainland_stream = matrix["services"]
        .as_array()
        .unwrap()
        .iter()
        .find(|service| {
            service["profile"] == "minimax"
                && service["operation_id"] == "service.audio.speech_streaming_text"
        })
        .unwrap();
    assert_eq!(mainland_stream["cell"]["status"], "supported");
    assert_eq!(mainland_stream["evidence_date"], "2026-09-26");
    assert_eq!(
        mainland_stream["cell"]["source_ids"],
        serde_json::json!(["minimax.speech_streaming.cn"])
    );
    let normal_source = &matrix["sources"]["minimax.speech_streaming.cn"];
    assert_eq!(
        normal_source["url"],
        "https://platform.minimax.cn/docs/api-reference/speech-t2a-websocket"
    );
    let normal_section = normal_source["section"].as_str().unwrap();
    assert!(normal_section.contains("page-embedded AsyncAPI"));
    assert!(normal_section.contains("wss://api.minimax.cn/ws/v1/t2a_v2"));

    for profile in matrix["profiles"].as_array().unwrap() {
        let name = profile["profile"].as_str().unwrap();
        if matches!(name, "minimax-intl" | "minimax") {
            for model in profile["models"].as_array().unwrap() {
                assert!(model["cells"].get(operation_id).is_none());
            }
        } else {
            assert!(!bidi_rows.iter().any(|service| service["profile"] == name));
        }
    }
}

#[test]
fn qwen_beijing_rag_service_operations_have_endpoint_evidence_only() {
    let matrix = matrix();
    assert_eq!(matrix["evidence_date"], "2026-09-27");
    assert_eq!(matrix["services"].as_array().unwrap().len(), 606);
    assert_eq!(matrix["sources"]["qwen.rag"]["evidence_date"], "2026-09-25");

    let endpoint_rows = [
        (
            "service.knowledge.chunks.add",
            "qwen.rag.chunks.add",
            "rag-api-add-chunk",
            "POST /api/v1/indices/rag/index/chunk/create",
        ),
        (
            "service.knowledge.chunks.list",
            "qwen.rag.chunks.list",
            "rag-api-list-chunks",
            "POST /api/v1/indices/rag/index/chunklist",
        ),
        (
            "service.knowledge.chunks.update",
            "qwen.rag.chunks.update",
            "rag-api-update-chunk",
            "POST /api/v1/indices/rag/index/chunk/update",
        ),
        (
            "service.knowledge.chunks.delete",
            "qwen.rag.chunks.delete",
            "rag-api-delete-chunk",
            "POST /api/v1/indices/rag/index/chunk/delete",
        ),
        (
            "service.knowledge.data_files.list",
            "qwen.rag.data_files.list",
            "rag-api-list-file",
            "POST /api/v1/connector/dash/listFile",
        ),
        (
            "service.knowledge.data_files.delete",
            "qwen.rag.data_files.delete",
            "rag-api-delete-file",
            "POST /api/v1/connector/dash/deleteFile",
        ),
        (
            "service.knowledge.data_files.tags.update",
            "qwen.rag.data_files.tags",
            "rag-api-batch-update-tag",
            "POST /api/v1/connector/dash/batchUpdateFileTag",
        ),
        (
            "service.knowledge.monitoring",
            "qwen.rag.monitoring",
            "rag-api-get-index-monitor",
            "POST /api/v1/indices/rag/index/monitor",
        ),
    ];

    for (operation_id, source_id, page, endpoint) in endpoint_rows {
        let operation = &matrix["operations"][operation_id];
        assert_eq!(operation["scope"], "service");
        assert_eq!(operation["endpoint"], endpoint);
        assert_eq!(strings(&operation["profile_ids"]), BTreeSet::from(["qwen"]));

        let source = &matrix["sources"][source_id];
        assert_eq!(source["evidence_date"], "2026-09-26");
        assert_eq!(
            source["url"],
            format!("https://help.aliyun.com/en/model-studio/{page}")
        );
        assert!(source["section"].as_str().unwrap().contains(endpoint));

        let matching = matrix["services"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|service| service["operation_id"] == operation_id)
            .collect::<Vec<_>>();
        assert_eq!(matching.len(), 1);
        let service = matching[0];
        assert_eq!(service["profile"], "qwen");
        assert_eq!(service["provider_id"], "qwen");
        assert_eq!(service["region"], "china_beijing");
        assert_eq!(service["evidence_date"], "2026-09-26");
        assert_eq!(service["live_validation"], "not_run");
        assert_eq!(service["account_region_validation"], "unknown");
        assert_eq!(service["availability"]["state"], "documented");
        assert_eq!(service["cell"]["status"], "supported");
        assert_eq!(service["cell"]["basis"], "explicit_support");
        assert_eq!(
            service["availability"]["source_ids"],
            serde_json::json!([source_id])
        );
        assert_eq!(
            service["cell"]["source_ids"],
            serde_json::json!([source_id])
        );

        for profile in matrix["profiles"].as_array().unwrap() {
            for model in profile["models"].as_array().unwrap() {
                assert!(model["cells"].get(operation_id).is_none());
            }
        }
        for service in matrix["services"].as_array().unwrap() {
            assert!(service["profile"] == "qwen" || service["operation_id"] != operation_id);
        }
    }
}

#[test]
fn qwen_http_tts_sse_is_separate_from_websocket_tts_and_asr_evidence() {
    let matrix = matrix();
    let source = &matrix["sources"]["qwen.tts.api"];
    assert_eq!(source["evidence_date"], "2026-09-26");
    assert_eq!(
        source["url"],
        "https://help.aliyun.com/en/model-studio/qwen-tts-api"
    );
    let section = source["section"].as_str().unwrap();
    for documented in [
        "X-DashScope-SSE: enable",
        "intermediate audio.data contains Base64 segments",
        "final chunk has empty audio.data plus audio.url",
        "finish_reason is null while generation is in progress and stop on normal completion",
        "labels its curl example Singapore",
    ] {
        assert!(
            section.contains(documented),
            "missing TTS evidence: {documented}"
        );
    }

    let speech_rows = matrix["services"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|service| {
            service["provider_id"] == "qwen"
                && service["operation_id"] == "service.audio.speech"
                && service["cell"]["status"] == "supported"
        })
        .collect::<Vec<_>>();
    assert_eq!(speech_rows.len(), 4);
    for service in speech_rows {
        assert_eq!(service["evidence_date"], "2026-09-26");
        assert!(strings(&service["cell"]["source_ids"]).contains("qwen.tts.api"));
        assert!(strings(&service["availability"]["source_ids"]).contains("qwen.tts.api"));
    }

    for profile in ["qwen", "qwen-intl"] {
        assert_eq!(
            service_cell(&matrix, profile, "service.audio.speech_streaming_text")["status"],
            "supported"
        );
        assert!(!strings(
            &service_cell(&matrix, profile, "service.audio.speech_streaming_text")["source_ids"]
        )
        .contains("qwen.tts.api"));
        assert!(!strings(
            &service_cell(&matrix, profile, "service.audio.transcriptions")["source_ids"]
        )
        .contains("qwen.tts.api"));
        assert!(!strings(
            &service_cell(&matrix, profile, "service.realtime.connect")["source_ids"]
        )
        .contains("qwen.tts.api"));
    }
}

#[test]
fn qwen_realtime_tts_is_region_scoped_and_separate_from_omni_realtime() {
    let matrix = matrix();
    let operation_id = "service.audio.speech_streaming_text";
    assert_eq!(matrix["operations"][operation_id]["scope"], "service");
    assert!(matrix["operations"][operation_id]["description"]
        .as_str()
        .unwrap()
        .contains("not conversation"));

    let evidence = [
        (
            "qwen.tts.realtime.endpoint",
            "interactive-process-of-qwen-tts-realtime-synthesis",
        ),
        ("qwen.tts.realtime.models", "model-pricing"),
        (
            "qwen.tts.realtime.client",
            "qwen-tts-realtime-client-events",
        ),
        (
            "qwen.tts.realtime.server",
            "qwen-tts-realtime-server-events",
        ),
    ];
    for (source_id, page) in evidence {
        let source = &matrix["sources"][source_id];
        assert_eq!(source["evidence_date"], "2026-09-26");
        assert_eq!(
            source["url"],
            format!("https://help.aliyun.com/en/model-studio/{page}")
        );
    }
    let endpoint_section = matrix["sources"]["qwen.tts.realtime.endpoint"]["section"]
        .as_str()
        .unwrap();
    for documented in [
        "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-tts-flash-realtime",
        "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model=qwen3-tts-flash-realtime",
        "regional API key is required",
    ] {
        assert!(endpoint_section.contains(documented));
    }
    let models_section = matrix["sources"]["qwen.tts.realtime.models"]["section"]
        .as_str()
        .unwrap();
    assert!(models_section.contains("China (Beijing) includes `qwen-tts-realtime`"));
    assert!(models_section.contains("Singapore lists `qwen3-tts-flash-realtime`"));

    let documented_profiles = ["qwen", "qwen-intl", "qwen-search", "qwen-search-intl"];
    let source_ids = [
        "qwen.tts.realtime.endpoint",
        "qwen.tts.realtime.models",
        "qwen.tts.realtime.client",
        "qwen.tts.realtime.server",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    for profile in matrix["profiles"].as_array().unwrap() {
        let name = profile["profile"].as_str().unwrap();
        if !name.starts_with("qwen") {
            continue;
        }
        let row = matrix["services"]
            .as_array()
            .unwrap()
            .iter()
            .find(|service| service["profile"] == name && service["operation_id"] == operation_id)
            .unwrap();
        if documented_profiles.contains(&name) {
            assert_eq!(row["cell"]["status"], "supported");
            assert_eq!(row["availability"]["state"], "documented");
            assert_eq!(row["evidence_date"], "2026-09-26");
            assert_eq!(
                strings(&row["cell"]["source_ids"])
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<BTreeSet<_>>(),
                source_ids
            );
            assert_eq!(
                strings(&row["availability"]["source_ids"])
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<BTreeSet<_>>(),
                source_ids
            );
            assert_eq!(row["live_validation"], "not_run");
            assert_eq!(row["account_region_validation"], "unknown");
        } else {
            assert_eq!(row["cell"]["status"], "unknown");
            assert!(strings(&row["cell"]["source_ids"])
                .iter()
                .all(|source_id| !source_ids.contains(*source_id)));
        }

        for model in profile["models"].as_array().unwrap() {
            assert!(model["cells"].get(operation_id).is_none());
        }
        let omni = matrix["services"]
            .as_array()
            .unwrap()
            .iter()
            .find(|service| {
                service["profile"] == name && service["operation_id"] == "service.realtime.connect"
            })
            .unwrap();
        assert!(strings(&omni["cell"]["source_ids"])
            .iter()
            .all(|source_id| !source_ids.contains(*source_id)));
    }
}

#[test]
fn qwen_category_connector_and_oss_operations_are_beijing_service_rows() {
    let matrix = matrix();
    let endpoint_rows = [
        (
            "service.knowledge.categories.list",
            "qwen.rag.categories.list",
            "rag-api-list-category",
            "POST /api/v1/connector/dash/listCategory",
        ),
        (
            "service.knowledge.categories.create",
            "qwen.rag.categories.create",
            "rag-api-add-category",
            "POST /api/v1/connector/dash/addCategory",
        ),
        (
            "service.knowledge.categories.delete",
            "qwen.rag.categories.delete",
            "rag-api-delete-category",
            "POST /api/v1/connector/dash/deleteCategory",
        ),
        (
            "service.knowledge.connectors.create",
            "qwen.rag.connectors.create",
            "rag-api-add-connector",
            "POST /api/v1/connector/dash/addConnector",
        ),
        (
            "service.knowledge.connectors.get",
            "qwen.rag.connectors.get",
            "rag-api-get-connector",
            "POST /api/v1/connector/dash/getConnector",
        ),
        (
            "service.knowledge.data_files.import_oss",
            "qwen.rag.data_files.import_oss",
            "rag-api-oss-import",
            "POST /api/v1/connector/dash/addFilesFromAuthorizedOss",
        ),
    ];
    assert_eq!(matrix["services"].as_array().unwrap().len(), 606);

    for (operation_id, source_id, page, endpoint) in endpoint_rows {
        let operation = &matrix["operations"][operation_id];
        assert_eq!(operation["scope"], "service");
        assert_eq!(operation["endpoint"], endpoint);
        assert_eq!(strings(&operation["profile_ids"]), BTreeSet::from(["qwen"]));

        let source = &matrix["sources"][source_id];
        assert_eq!(source["evidence_date"], "2026-09-26");
        assert_eq!(
            source["url"],
            format!("https://help.aliyun.com/en/model-studio/{page}")
        );
        assert!(source["section"].as_str().unwrap().contains(endpoint));

        let rows = matrix["services"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|service| service["operation_id"] == operation_id)
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 1);
        let row = rows[0];
        assert_eq!(row["profile"], "qwen");
        assert_eq!(row["region"], "china_beijing");
        assert_eq!(row["evidence_date"], "2026-09-26");
        assert_eq!(row["live_validation"], "not_run");
        assert_eq!(row["account_region_validation"], "unknown");
        assert_eq!(row["availability"]["state"], "documented");
        assert_eq!(row["cell"]["status"], "supported");
        assert_eq!(row["cell"]["basis"], "explicit_support");
        assert_eq!(row["cell"]["source_ids"], serde_json::json!([source_id]));
        assert_eq!(
            row["availability"]["source_ids"],
            serde_json::json!([source_id])
        );

        for profile in matrix["profiles"].as_array().unwrap() {
            for model in profile["models"].as_array().unwrap() {
                assert!(model["cells"].get(operation_id).is_none());
            }
        }
        for service in matrix["services"].as_array().unwrap() {
            assert!(service["profile"] == "qwen" || service["operation_id"] != operation_id);
        }
    }
}

fn model_cell<'a>(matrix: &'a Value, profile: &str, model: &str, op: &str) -> &'a Value {
    &matrix["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["profile"] == profile)
        .unwrap()["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["model_id"] == model)
        .unwrap()["cells"][op]
}

fn service_cell<'a>(matrix: &'a Value, profile: &str, op: &str) -> &'a Value {
    &matrix["services"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["profile"] == profile && s["operation_id"] == op)
        .unwrap()["cell"]
}

#[test]
fn documented_limits_do_not_leak_across_protocols_regions_or_account_products() {
    let m = matrix();
    assert_eq!(
        model_cell(&m, "qwen-search", "qwen3.8-max", "responses.background")["status"],
        "unsupported"
    );
    assert_eq!(
        model_cell(&m, "qwen", "qwen3.8-max", "batch.submit")["status"],
        "supported"
    );
    assert_eq!(
        model_cell(&m, "qwen-intl", "qwen3.8-max", "batch.submit")["status"],
        "unknown"
    );
    assert_eq!(
        service_cell(&m, "qwen", "service.knowledge.create")["status"],
        "supported"
    );
    assert_eq!(
        service_cell(&m, "qwen-intl", "service.knowledge.create")["status"],
        "unknown"
    );
    for p in ["kimi-search", "kimi-search-intl"] {
        assert_eq!(
            model_cell(&m, p, "kimi-k3", "responses.cache_ttl")["status"],
            "supported"
        );
        assert_eq!(
            model_cell(&m, p, "kimi-k3", "responses.cache_breakpoints")["status"],
            "unsupported"
        );
        assert_eq!(
            model_cell(&m, p, "kimi-k3", "batch.submit")["status"],
            "unsupported"
        );
    }
    for p in ["minimax", "minimax-intl"] {
        assert_eq!(
            model_cell(&m, p, "MiniMax-M3", "messages.remote_mcp")["status"],
            "unsupported"
        );
    }
    assert_eq!(
        service_cell(&m, "minimax-intl", "service.audio.speech_streaming_text")["status"],
        "supported"
    );
    assert_eq!(
        service_cell(&m, "minimax-intl", "service.realtime.connect")["status"],
        "unknown"
    );
    assert_eq!(
        service_cell(&m, "glm", "service.batch.create")["status"],
        "supported"
    );
    for p in ["glm-coding", "zai", "zai-coding"] {
        assert_eq!(
            service_cell(&m, p, "service.batch.create")["status"],
            "unknown"
        );
    }
    assert_eq!(
        model_cell(&m, "zai", "glm-4.7", "chat.json_schema")["status"],
        "unknown"
    );
    assert_eq!(
        model_cell(&m, "kimi-code", "k3", "chat.create")["status"],
        "unknown"
    );
}

#[test]
fn retired_models_remain_visible_and_service_support_is_not_chat_model_support() {
    let m = matrix();
    let international_k25 = m["retired_models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["profile"] == "kimi-intl" && model["model_id"] == "kimi-k2.5")
        .expect("international retirement needs its own evidence record");
    assert_eq!(international_k25["region"], "international");
    assert_eq!(
        international_k25["availability"]["source_ids"],
        serde_json::json!(["kimi.models.intl"])
    );
    assert_eq!(
        m["sources"]["kimi.models.intl"]["url"],
        "https://platform.kimi.ai/docs/models"
    );
    let mut historical = BTreeSet::new();
    for model in m["retired_models"].as_array().unwrap() {
        let profile = model["profile"].as_str().unwrap();
        let id = model["model_id"].as_str().unwrap();
        assert!(historical.insert((profile, id)), "duplicate retired model");
        assert_eq!(model["cells"]["chat.create"]["status"], "unsupported");
        assert_eq!(model["availability"]["state"], "shut_down");
        let definition = catalog(profile);
        assert_eq!(
            definition["provider_id"].as_str(),
            model["provider_id"].as_str()
        );
        assert!(
            !definition["model"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["id"].as_str() == Some(id)),
            "retired row is still active"
        );
    }
    for (profile, model) in [("glm", "glm-4.7"), ("qwen", "qwen3.8-max")] {
        assert_eq!(
            service_cell(&m, profile, "service.embeddings.create")["status"],
            "supported"
        );
        assert_eq!(
            model_cell(&m, profile, model, "embeddings.create")["status"],
            "unknown"
        );
    }
}

#[test]
fn qwen_standalone_asr_realtime_is_region_scoped_and_distinct() {
    let matrix = matrix();
    let operation_id = "service.audio.transcriptions_realtime";
    let documented_profiles = BTreeMap::from([
        ("qwen", "china_beijing"),
        ("qwen-search", "china_beijing"),
        ("qwen-intl", "singapore"),
        ("qwen-search-intl", "singapore"),
    ]);
    let source_ids = BTreeSet::from([
        "qwen.asr.realtime.endpoint",
        "qwen.asr.realtime.client",
        "qwen.asr.realtime.server",
        "qwen.asr.realtime.model",
        "qwen.asr.realtime.guide",
    ]);
    let operation = &matrix["operations"][operation_id];
    assert_eq!(operation["scope"], "service");
    assert_eq!(
        operation["endpoint"],
        "WSS /api-ws/v1/realtime?model=<model_name>"
    );
    assert_eq!(
        strings(&operation["profile_ids"]),
        documented_profiles.keys().copied().collect()
    );
    for other in [
        "service.audio.transcriptions",
        "service.realtime.connect",
        "service.audio.speech_streaming_text",
    ] {
        assert_ne!(operation_id, other);
    }

    for (source_id, page) in [
        (
            "qwen.asr.realtime.endpoint",
            "qwen-asr-realtime-interaction-process",
        ),
        (
            "qwen.asr.realtime.client",
            "qwen-asr-realtime-client-events",
        ),
        (
            "qwen.asr.realtime.server",
            "qwen-asr-realtime-server-events",
        ),
        ("qwen.asr.realtime.model", "qwen3-asr-flash-realtime"),
        (
            "qwen.asr.realtime.guide",
            "real-time-speech-recognition-user-guide",
        ),
    ] {
        let source = &matrix["sources"][source_id];
        assert_eq!(source["evidence_date"], "2026-09-26");
        assert_eq!(
            source["url"],
            format!("https://help.aliyun.com/en/model-studio/{page}")
        );
    }
    for endpoint in [
        "wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=<model_name>",
        "wss://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model=<model_name>",
        "Authorization: Bearer <API key>",
    ] {
        assert!(matrix["sources"]["qwen.asr.realtime.endpoint"]["section"]
            .as_str()
            .unwrap()
            .contains(endpoint));
    }
    let client = matrix["sources"]["qwen.asr.realtime.client"]["section"]
        .as_str()
        .unwrap();
    for event in [
        "`session.update`",
        "`input_audio_buffer.append`",
        "`input_audio_buffer.commit`",
        "`session.finish`",
        "`session.finished`",
        "VAD and Manual modes",
    ] {
        assert!(
            client.contains(event),
            "missing ASR client evidence: {event}"
        );
    }
    let server = matrix["sources"]["qwen.asr.realtime.server"]["section"]
        .as_str()
        .unwrap();
    for event in [
        "`session.created`",
        "`conversation.item.input_audio_transcription.text`",
        "`conversation.item.input_audio_transcription.completed`",
        "`session.finished`",
        "`error` event",
    ] {
        assert!(
            server.contains(event),
            "missing ASR server evidence: {event}"
        );
    }
    let documented_models = BTreeSet::from([
        "qwen3-asr-flash-realtime",
        "qwen3-asr-flash-realtime-2026-02-10",
        "qwen3-asr-flash-realtime-2025-10-27",
    ]);
    for source_id in ["qwen.asr.realtime.model", "qwen.asr.realtime.guide"] {
        let section = matrix["sources"][source_id]["section"].as_str().unwrap();
        for model in &documented_models {
            assert!(section.contains(model), "missing model evidence: {model}");
        }
    }

    let rows = matrix["services"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|service| service["operation_id"] == operation_id)
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), documented_profiles.len());
    for row in &rows {
        let profile = row["profile"].as_str().unwrap();
        assert_eq!(row["provider_id"], "qwen");
        assert_eq!(row["region"], documented_profiles[profile]);
        assert_eq!(row["evidence_date"], "2026-09-26");
        assert_eq!(row["live_validation"], "not_run");
        assert_eq!(row["account_region_validation"], "unknown");
        assert_eq!(row["availability"]["state"], "documented");
        assert_eq!(row["cell"]["status"], "supported");
        assert_eq!(row["cell"]["basis"], "explicit_support");
        assert_eq!(strings(&row["cell"]["source_ids"]), source_ids);
        assert_eq!(strings(&row["availability"]["source_ids"]), source_ids);
        assert_eq!(strings(&row["documented_models"]), documented_models);
    }
    for profile in matrix["profiles"].as_array().unwrap() {
        let name = profile["profile"].as_str().unwrap();
        if name.starts_with("qwen") {
            assert_eq!(
                rows.iter().any(|row| row["profile"] == name),
                documented_profiles.contains_key(name),
                "unexpected Qwen ASR Realtime scope for {name}"
            );
        }
        for model in profile["models"].as_array().unwrap() {
            assert!(model["cells"].get(operation_id).is_none());
        }
    }

    for profile in documented_profiles.keys() {
        for existing_operation in [
            "service.audio.transcriptions",
            "service.realtime.connect",
            "service.audio.speech_streaming_text",
        ] {
            let existing_sources =
                strings(&service_cell(&matrix, profile, existing_operation)["source_ids"]);
            assert!(
                existing_sources.is_disjoint(&source_ids),
                "ASR Realtime evidence leaked into {existing_operation} for {profile}"
            );
        }
    }
}

#[test]
fn qwen_published_knowledge_search_is_distinct_and_beijing_only() {
    let matrix = matrix();
    let operation_id = "service.knowledge.search_published";
    let source_id = "qwen.rag.knowledge_search";
    let operation = &matrix["operations"][operation_id];
    assert_eq!(operation["scope"], "service");
    assert_eq!(
        operation["endpoint"],
        "POST /api/v1/indices/knowledge/search"
    );
    assert_eq!(strings(&operation["profile_ids"]), BTreeSet::from(["qwen"]));
    assert!(operation["description"]
        .as_str()
        .unwrap()
        .contains("published Qwen Knowledge Retrieval service"));

    let source = &matrix["sources"][source_id];
    assert_eq!(source["evidence_date"], "2026-09-26");
    assert_eq!(
        source["url"],
        "https://help.aliyun.com/en/model-studio/knowledgesearch"
    );
    let section = source["section"].as_str().unwrap();
    for documented in [
        "POST /api/v1/indices/knowledge/search",
        "{workspaceId}.cn-beijing.maas.aliyuncs.com",
        "already be created and published",
        "`agent_id`",
        "Agent-not-published",
        "does not create or publish the service",
    ] {
        assert!(
            section.contains(documented),
            "missing Knowledge Search evidence: {documented}"
        );
    }

    let rows = matrix["services"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|service| service["operation_id"] == operation_id)
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 1);
    let row = rows[0];
    assert_eq!(row["profile"], "qwen");
    assert_eq!(row["provider_id"], "qwen");
    assert_eq!(row["region"], "china_beijing");
    assert_eq!(row["evidence_date"], "2026-09-26");
    assert_eq!(row["live_validation"], "not_run");
    assert_eq!(row["account_region_validation"], "unknown");
    assert_eq!(row["availability"]["state"], "documented");
    assert_eq!(row["cell"]["status"], "supported");
    assert_eq!(row["cell"]["basis"], "explicit_support");
    assert_eq!(
        strings(&row["cell"]["source_ids"]),
        BTreeSet::from([source_id])
    );
    assert_eq!(
        strings(&row["availability"]["source_ids"]),
        BTreeSet::from([source_id])
    );

    for profile in matrix["profiles"].as_array().unwrap() {
        for model in profile["models"].as_array().unwrap() {
            assert!(model["cells"].get(operation_id).is_none());
        }
    }
    for service in matrix["services"].as_array().unwrap() {
        if service["operation_id"] == "service.knowledge.search" {
            assert!(!strings(&service["cell"]["source_ids"]).contains(source_id));
            assert!(!strings(&service["availability"]["source_ids"]).contains(source_id));
        }
    }
}

#[test]
fn qwen_published_knowledge_chat_is_beijing_only_and_requires_streaming() {
    let matrix = matrix();
    let operation_id = "service.knowledge.chat_published";
    let source_id = "qwen.rag.knowledge_chat";
    let operation = &matrix["operations"][operation_id];
    assert_eq!(operation["scope"], "service");
    assert_eq!(
        operation["endpoint"],
        "POST /api/v2/apps/knowledge/chat (SSE)"
    );
    assert_eq!(strings(&operation["profile_ids"]), BTreeSet::from(["qwen"]));
    assert!(operation["description"]
        .as_str()
        .unwrap()
        .contains("previously created and published service"));

    let source = &matrix["sources"][source_id];
    assert_eq!(source["evidence_date"], "2026-09-26");
    assert_eq!(
        source["url"],
        "https://help.aliyun.com/en/model-studio/knowledgechat"
    );
    let section = source["section"].as_str().unwrap();
    for documented in [
        "POST /api/v2/apps/knowledge/chat",
        "Beijing workspace",
        "SSE response",
        "created and published",
        "`agent_id`",
        "Agent-not-published",
        "full `messages` history",
        "`stream: true`",
        "no model-selection field",
        "not evidence for generic model cache capabilities",
    ] {
        assert!(
            section.contains(documented),
            "missing Knowledge Chat evidence: {documented}"
        );
    }

    let rows = matrix["services"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|service| service["operation_id"] == operation_id)
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 1);
    let row = rows[0];
    assert_eq!(row["profile"], "qwen");
    assert_eq!(row["provider_id"], "qwen");
    assert_eq!(row["region"], "china_beijing");
    assert_eq!(row["evidence_date"], "2026-09-26");
    assert_eq!(row["live_validation"], "not_run");
    assert_eq!(row["account_region_validation"], "unknown");
    assert_eq!(row["availability"]["state"], "documented");
    assert_eq!(row["cell"]["status"], "supported");
    assert_eq!(row["cell"]["basis"], "explicit_support");
    assert_eq!(
        strings(&row["cell"]["source_ids"]),
        BTreeSet::from([source_id])
    );
    assert_eq!(
        strings(&row["availability"]["source_ids"]),
        BTreeSet::from([source_id])
    );

    for profile in matrix["profiles"].as_array().unwrap() {
        for model in profile["models"].as_array().unwrap() {
            assert!(model["cells"].get(operation_id).is_none());
        }
    }
    for service in matrix["services"].as_array().unwrap() {
        if service["operation_id"] == "service.knowledge.search_published" {
            assert!(!strings(&service["cell"]["source_ids"]).contains(source_id));
            assert!(!strings(&service["availability"]["source_ids"]).contains(source_id));
        }
    }
}
