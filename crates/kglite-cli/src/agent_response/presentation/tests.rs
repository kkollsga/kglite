use std::fs;

use serde_json::json;

use super::*;

fn large_envelope() -> Value {
    json!({
        "schema_version": 1,
        "kind": "cypher_result",
        "columns": ["definition"],
        "rows": (0..80).map(|index| json!([format!("row {index}: {}", "界'".repeat(600))])).collect::<Vec<_>>(),
        "diagnostics": null,
        "coverage": {"executed_rows": 80, "database_population": "unknown"},
        "identity": {"graph": "/gone/demo.kgl", "steering": []},
        "operation": {"mutation": false, "output_format": "rows"},
        "representation": {"text_complete": false, "text_kind": "agent_json"}
        ,"navigation": {
            "targets": [
                {"purpose":"continue rows","json_pointer":"/rows","offset":16,"response":{"mode":"bounded","max_bytes":4096}},
                {"purpose":"columns","json_pointer":"/columns","offset":0,"response":{"mode":"bounded","max_bytes":4096}},
                {"purpose":"diagnostics","json_pointer":"/diagnostics","offset":0,"response":{"mode":"bounded","max_bytes":4096}},
                {"purpose":"coverage","json_pointer":"/coverage","offset":0,"response":{"mode":"bounded","max_bytes":4096}}
            ],
            "observed_value_targets": [{"json_pointer":"/rows/0/0","offset":0,"response":{"mode":"bounded","max_bytes":4096}}]
        }
    })
}

fn preview_body(value: &Value) -> Value {
    serde_json::from_str(value.pointer("/content/0/text").unwrap().as_str().unwrap()).unwrap()
}

#[test]
fn durable_translation_fits_final_serialized_budget_and_emits_executable_commands() {
    let temp = tempfile::tempdir().unwrap();
    let value = present(
        temp.path().join("cache"),
        "/workspace/with space",
        large_envelope(),
        false,
        AgentOptions {
            max_bytes: Some(MIN_RESPONSE_BYTES),
            full: false,
        },
    );
    assert!(serialized_size(&value) <= MIN_RESPONSE_BYTES);
    let body = preview_body(&value);
    let budget = &body["response_budget"];
    let handle = budget["result_id"].as_str().unwrap();
    assert!(handle.starts_with("kgr_"));
    let selected = budget["next"]["selected_value"].as_str().unwrap();
    assert!(selected.starts_with(&format!("kglite response expand {handle}")));
    assert!(!selected.contains("--response-full"));
    assert!(budget["next"]["full_result"]
        .as_str()
        .unwrap()
        .ends_with("--response-full"));
    assert_eq!(budget["domain_commands"][0]["json_pointer"], "/rows");
    assert!(budget["domain_commands"][0]["command"]
        .as_str()
        .unwrap()
        .contains("--response-max-bytes 4096"));
    assert_eq!(budget["domain_commands"][4]["json_pointer"], "/rows/37");
    assert_eq!(budget["domain_commands"][5]["json_pointer"], "/rows/0/0");
}

#[test]
fn unavoidable_translation_overage_reports_exact_final_size() {
    let envelope = large_envelope();
    let result = fit_translated_preview(
        envelope.clone(),
        &envelope,
        "workspace",
        AgentOptions {
            max_bytes: Some(MIN_RESPONSE_BYTES),
            full: false,
        },
        &"h".repeat(10_000),
    );
    let actual = serialized_size(&result);
    assert!(actual > MIN_RESPONSE_BYTES);
    assert_eq!(
        result["retention"]["actual_bytes"].as_u64(),
        Some(actual as u64)
    );
}

#[test]
fn later_process_loads_by_root_and_handle_then_recovers_exact_original() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("cache");
    let original = large_envelope();
    let preview = present(
        root.clone(),
        "/workspace/original",
        original.clone(),
        false,
        AgentOptions::default(),
    );
    let body = preview_body(&preview);
    let handle = body["response_budget"]["result_id"]
        .as_str()
        .unwrap()
        .to_owned();

    let recovered = expand(
        root,
        &handle,
        String::new(),
        0,
        AgentOptions {
            full: true,
            max_bytes: None,
        },
    )
    .unwrap();
    assert_eq!(recovered["structuredContent"], original);
}

#[test]
fn expansion_selects_escaped_pointer_and_unicode_offset_without_graph_context() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("cache");
    let mut original = large_envelope();
    original["diagnostics"] = json!({"a/b~c": "αβγ界".repeat(2_000)});
    let preview = present(
        root.clone(),
        "/graph/removed",
        original.clone(),
        false,
        AgentOptions::default(),
    );
    let body = preview_body(&preview);
    let handle = body["response_budget"]["result_id"].as_str().unwrap();
    let selected = expand(
        root,
        handle,
        "/diagnostics/a~1b~0c".into(),
        3,
        AgentOptions::default(),
    )
    .unwrap();
    let body = preview_body(&selected);
    assert_eq!(body["response_budget"]["preview"]["offset"], 3);
    assert!(body["response_budget"]["preview"]["excerpt"]
        .as_str()
        .unwrap()
        .starts_with("界αβγ"));
    assert!(serialized_size(&selected) <= DEFAULT_RESPONSE_BYTES);
}

#[test]
fn invalid_controls_are_rejected_without_touching_cache_root() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("cache");
    assert!(AgentOptions {
        full: false,
        max_bytes: Some(MIN_RESPONSE_BYTES - 1),
    }
    .validate()
    .is_err());
    assert!(AgentOptions {
        full: true,
        max_bytes: Some(DEFAULT_RESPONSE_BYTES),
    }
    .validate()
    .is_err());
    assert!(!root.exists());
}

#[cfg(unix)]
#[test]
fn retention_failure_returns_complete_success_with_warning_and_no_handle() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("public-cache");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).unwrap();
    let mut envelope = large_envelope();
    envelope["operation"]["mutation"] = json!(true);
    let result = present(
        root,
        "mutation-workspace",
        envelope,
        false,
        AgentOptions::default(),
    );
    assert_eq!(result["isError"], false);
    assert_eq!(result["retention"]["retained"], false);
    assert_eq!(result["structuredContent"]["operation"]["mutation"], true);
    assert!(result["retention"]["warning"]
        .as_str()
        .unwrap()
        .contains("was not rerun"));
    assert!(temporary_id(&result).is_none());
}
