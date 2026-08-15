use std::sync::Arc;

use kglite::api::cypher::CypherResult;
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::{KgError, Value as KgliteValue};
use rmcp::model::CallToolResult;
use serde_json::{json, Map, Value};

use super::errors::RecipeErrorEnvelope;
use super::result::{list_recipe_queries, run_recipe_query, serialize_success};
use super::wire::{ListRecipeQueriesArgs, RunRecipeQueryArgs};
use super::{RecipeCatalog, RECIPE_RESULT_ROW_LIMIT};
use crate::tools::GraphState;
use crate::{WorkspaceGraphHooks, WorkspaceGraphMode, WorkspaceGraphResult};

fn catalog(cypher: &str) -> RecipeCatalog {
    RecipeCatalog::from_manifest_value(Some(&json!({
        "review": {
            "description": "Review operations.",
            "queries": {
                "lookup": {
                    "description": "Look up values.",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": false
                    },
                    "cypher": cypher
                }
            }
        }
    })))
    .unwrap()
}

fn args(include_cypher: bool) -> RunRecipeQueryArgs {
    RunRecipeQueryArgs {
        recipe: "review".into(),
        query: "lookup".into(),
        variables: Map::from_iter([("name".into(), json!("Ada"))]),
        include_cypher,
    }
}

fn canonical_code_review_catalog() -> RecipeCatalog {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/local_code_review_mcp.yaml");
    let manifest = mcp_methods::server::load_manifest(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    RecipeCatalog::from_manifest_value(manifest.extensions.get("cypher_recipes"))
        .expect("canonical recipe catalog")
}

fn canonical_args(query: &str) -> RunRecipeQueryArgs {
    RunRecipeQueryArgs {
        recipe: "code_review".into(),
        query: query.into(),
        variables: Map::from_iter([("qualified_name".into(), json!("pkg.target"))]),
        include_cypher: false,
    }
}

fn assert_text_matches_structured(result: &CallToolResult) -> Value {
    let structured = result
        .structured_content
        .clone()
        .expect("structured content");
    let text = result.content[0].as_text().expect("text fallback");
    assert_eq!(
        serde_json::from_str::<Value>(&text.text).unwrap(),
        structured
    );
    structured
}

#[test]
fn variables_are_required_and_unknown_root_fields_are_rejected() {
    let missing = serde_json::from_value::<RunRecipeQueryArgs>(json!({
        "recipe": "review", "query": "lookup"
    }))
    .unwrap_err();
    assert!(missing.to_string().contains("variables"));

    let malformed = serde_json::from_value::<RunRecipeQueryArgs>(json!({
        "recipe": "review", "query": "lookup", "variables": {}, "extra": true
    }))
    .unwrap_err();
    assert!(malformed.to_string().contains("unknown field"));
}

#[test]
fn compact_and_focused_listing_have_stable_disclosure() {
    let catalog = catalog("RETURN $name AS name");
    let compact = serde_json::to_value(list_recipe_queries(
        &catalog,
        ListRecipeQueriesArgs::default(),
    ))
    .unwrap();
    assert_eq!(compact["recipes"][0]["query_count"], 1);
    assert!(compact["recipes"][0].get("queries").is_none());

    let focused = serde_json::to_value(list_recipe_queries(
        &catalog,
        ListRecipeQueriesArgs {
            recipe: Some("review".into()),
        },
    ))
    .unwrap();
    assert_eq!(focused["recipes"][0]["queries"][0]["name"], "lookup");
    assert_eq!(
        focused["recipes"][0]["queries"][0]["parameters"]["type"],
        "object"
    );
    assert!(!focused.to_string().contains("RETURN $name"));
}

#[test]
fn positional_serialization_preserves_empty_rows_and_natural_values() {
    let catalog = catalog("RETURN $name AS name");
    let query = catalog.get("review").unwrap().get("lookup").unwrap();
    let empty = cypher_result(vec!["name"], vec![]);
    let output = serde_json::to_value(serialize_success(&args(false), query, &empty)).unwrap();
    assert_eq!(output["result"]["columns"], json!(["name"]));
    assert_eq!(output["result"]["rows"], json!([]));
    assert_eq!(output["result"]["row_count"], 0);

    let values = cypher_result(
        vec!["n", "items"],
        vec![vec![
            KgliteValue::Int64(7),
            KgliteValue::List(vec![KgliteValue::String("x".into())]),
        ]],
    );
    let output = serde_json::to_value(serialize_success(&args(false), query, &values)).unwrap();
    assert_eq!(output["result"]["rows"], json!([[7, ["x"]]]));
}

/// A recipe query that returns a node hands the agent a JSON object, not
/// the engine's Rust `Debug` string. Until 0.16.1 `serialize_success` sent
/// `"Node(NodeValue { id: 7, .. })"` — unparseable, and the same
/// `kglite_value_to_json` defect the C ABI and CLI `--mode json` shared.
#[test]
fn a_returned_node_serializes_as_an_object_not_a_debug_string() {
    let catalog = catalog("RETURN $name AS name");
    let query = catalog.get("review").unwrap().get("lookup").unwrap();
    let mut properties = std::collections::BTreeMap::new();
    properties.insert("name".to_string(), KgliteValue::String("Ada".into()));
    let node = KgliteValue::Node(Box::new(kglite::api::NodeValue {
        id: 7,
        labels: vec!["Person".to_string()],
        properties,
    }));
    let result = cypher_result(vec!["n"], vec![vec![node]]);
    let output = serde_json::to_value(serialize_success(&args(false), query, &result)).unwrap();
    assert!(
        !output.to_string().contains("NodeValue"),
        "the Debug rendering leaked into the recipe result: {output}"
    );
    assert_eq!(
        output["result"]["rows"],
        json!([[{"id": 7, "labels": ["Person"], "properties": {"name": "Ada"}}]])
    );
}

#[test]
fn exact_cap_returns_all_or_an_error_with_observed_count() {
    let catalog = catalog("RETURN $name AS name");
    let query = catalog.get("review").unwrap().get("lookup").unwrap();
    let result = |count| {
        cypher_result(
            vec!["n"],
            (0..count)
                .map(|value| vec![KgliteValue::Int64(value as i64)])
                .collect(),
        )
    };
    let exact = serde_json::to_value(serialize_success(
        &args(false),
        query,
        &result(RECIPE_RESULT_ROW_LIMIT),
    ))
    .unwrap();
    assert_eq!(exact["result"]["row_count"], RECIPE_RESULT_ROW_LIMIT);

    let overflow = serde_json::to_value(serialize_success(
        &args(false),
        query,
        &result(RECIPE_RESULT_ROW_LIMIT + 1),
    ))
    .unwrap();
    assert_eq!(overflow["code"], "result_limit_exceeded");
    assert_eq!(overflow["details"]["limit"], RECIPE_RESULT_ROW_LIMIT);
    assert_eq!(
        overflow["details"]["observed_count"],
        RECIPE_RESULT_ROW_LIMIT + 1
    );
    assert!(overflow.get("result").is_none());
}

#[test]
fn audit_fields_are_paired_and_text_matches_structured_json() {
    let catalog = catalog("RETURN $name AS name");
    let query = catalog.get("review").unwrap().get("lookup").unwrap();
    let result = cypher_result(vec!["name"], vec![vec![KgliteValue::String("Ada".into())]]);
    let hidden = serialize_success(&args(false), query, &result).into_call_tool_result();
    let hidden = assert_text_matches_structured(&hidden);
    assert!(hidden.get("cypher").is_none());
    assert!(hidden.get("parameters").is_none());

    let shown = serialize_success(&args(true), query, &result).into_call_tool_result();
    let shown = assert_text_matches_structured(&shown);
    assert_eq!(shown["cypher"], "RETURN $name AS name");
    assert_eq!(shown["parameters"], json!({"name": "Ada"}));
}

#[test]
fn invalid_variables_expose_categories_and_obey_audit_pairing() {
    let catalog = catalog("RETURN $name AS name");
    let query = catalog.get("review").unwrap().get("lookup").unwrap();
    let bad = RunRecipeQueryArgs {
        variables: Map::from_iter([("other".into(), json!(3))]),
        ..args(true)
    };
    let error = query.validate_variables(&bad.variables).unwrap_err();
    let value =
        serde_json::to_value(RecipeErrorEnvelope::invalid_variables(&bad, query, error)).unwrap();
    assert_eq!(value["code"], "invalid_variables");
    assert_eq!(value["details"]["missing"], json!(["name"]));
    assert_eq!(value["details"]["unknown"], json!(["other"]));
    assert_eq!(value["cypher"], "RETURN $name AS name");
    assert_eq!(value["parameters"], json!({"other": 3}));
}

#[test]
fn query_failure_categories_use_raw_cause_and_preserve_position() {
    let catalog = catalog("RETURN $name AS name");
    let query = catalog.get("review").unwrap().get("lookup").unwrap();
    let cases = [
        (
            "CALL rev_diff: this graph has no `revs` property — it is not a multi-rev graph. Build one.",
            "multi_revision_graph_required",
        ),
        (
            "CALL rev_diff: revision \"missing\" is not present in this graph. Available revs: [main].",
            "unknown_revision",
        ),
        ("ordinary execution failure", "cypher_execution"),
    ];
    for (message, category) in cases {
        let error = KgError::CypherExecution {
            message: message.into(),
            position: Some((3, 7)),
        };
        let value = serde_json::to_value(RecipeErrorEnvelope::query_failed(
            &args(false),
            query,
            error,
        ))
        .unwrap();
        assert_eq!(value["details"]["cause"]["category"], category);
        assert_eq!(value["details"]["cause"]["kglite_code"], "CypherExecution");
        assert_eq!(
            value["details"]["cause"]["position"],
            json!({"line": 3, "column": 7})
        );
    }
}

#[test]
fn no_graph_is_structured_and_error_text_matches_json() {
    let output = run_recipe_query(
        &GraphState::default(),
        &catalog("RETURN $name AS name"),
        args(false),
    );
    let result = output.into_call_tool_result();
    assert_eq!(result.is_error, Some(true));
    let value = assert_text_matches_structured(&result);
    assert_eq!(value["code"], "no_active_graph");
}

#[test]
fn known_stale_workspace_returns_no_recipe_data() {
    let workspace = tempfile::tempdir().unwrap();
    let hooks = WorkspaceGraphHooks {
        build: Box::new(|request| {
            if !request.root().exists() {
                return Err("source tree disappeared".to_string());
            }
            new_dir_graph_in_mode(StorageMode::Memory, None)
                .map(Arc::new)
                .map(WorkspaceGraphResult::new)
                .map_err(|error| error.to_string())
        }),
        is_relevant: Box::new(|_| true),
    };
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(Arc::new(hooks)));
    state
        .build_workspace_graph(workspace.path(), None)
        .expect("install initial workspace graph");
    let missing_root = workspace.path().to_path_buf();
    drop(workspace);
    assert!(!missing_root.exists());
    state.tag_workspace_graph_dirty(&[missing_root.join("deleted.rs")]);

    let result = run_recipe_query(&state, &catalog("RETURN $name AS name"), args(true))
        .into_call_tool_result();
    assert_eq!(result.is_error, Some(true));
    let value = assert_text_matches_structured(&result);
    assert_eq!(value["code"], "stale_graph");
    assert_eq!(value["details"]["reason"], "rebuild_failed");
    assert!(value["details"]["failure_message"]
        .as_str()
        .is_some_and(|message| message.contains("source tree disappeared")));
    assert_eq!(value["cypher"], "RETURN $name AS name");
    assert!(value.get("result").is_none());
}

#[test]
fn canonical_code_review_queries_execute_with_resolve_first_semantics() {
    let hooks = WorkspaceGraphHooks {
        build: Box::new(|_| {
            let mut graph = new_dir_graph_in_mode(StorageMode::Memory, None)
                .map_err(|error| error.to_string())?;
            let params = std::collections::HashMap::new();
            let options = kglite::api::session::ExecuteOptions::eager(&params);
            kglite::api::session::execute_mut(
                &mut graph,
                "CREATE \
                 (target:Function {id:'pkg.target', qualified_name:'pkg.target', file_path:'src/target.rs', line_number:10, is_test:false}), \
                 (caller:Function {id:'pkg.caller', qualified_name:'pkg.caller', file_path:'src/caller.rs', line_number:20, is_test:false}), \
                 (middle:Function {id:'pkg.middle', qualified_name:'pkg.middle', file_path:'src/middle.rs', line_number:30, is_test:false}), \
                 (test:Function {id:'pkg.test_target', qualified_name:'pkg.test_target', file_path:'tests/test_target.rs', line_number:40, is_test:true}), \
                 (orphan:Function {id:'pkg.orphan', qualified_name:'pkg.orphan', file_path:'src/orphan.rs', line_number:50, is_test:false})",
                &options,
            )
            .map_err(|error| error.to_string())?;
            kglite::api::session::execute_mut(
                &mut graph,
                "MATCH (target:Function {id:'pkg.target'}), \
                       (caller:Function {id:'pkg.caller'}), \
                       (middle:Function {id:'pkg.middle'}), \
                       (test:Function {id:'pkg.test_target'}) \
                 CREATE (caller)-[:CALLS]->(target), \
                        (middle)-[:CALLS]->(target), \
                        (test)-[:CALLS]->(middle)",
                &options,
            )
            .map_err(|error| error.to_string())?;
            Ok(WorkspaceGraphResult::new(Arc::new(graph)))
        }),
        is_relevant: Box::new(|_| true),
    };
    let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
        .with_workspace_graph(Some(Arc::new(hooks)));
    state
        .build_workspace_graph(std::path::Path::new("/example"), None)
        .expect("install canonical example fixture");
    let catalog = canonical_code_review_catalog();

    let resolve = serde_json::to_value(run_recipe_query(
        &state,
        &catalog,
        canonical_args("resolve_function"),
    ))
    .unwrap();
    assert_eq!(resolve["result"]["row_count"], 1);
    assert_eq!(resolve["result"]["rows"][0][0], "pkg.target");

    let callers = serde_json::to_value(run_recipe_query(
        &state,
        &catalog,
        canonical_args("direct_callers"),
    ))
    .unwrap();
    assert_eq!(callers["result"]["row_count"], 2);
    assert_eq!(callers["result"]["rows"][0][0], "pkg.caller");
    assert_eq!(callers["result"]["rows"][1][0], "pkg.middle");

    let tests = serde_json::to_value(run_recipe_query(
        &state,
        &catalog,
        canonical_args("affected_tests"),
    ))
    .unwrap();
    assert_eq!(tests["result"]["row_count"], 1);
    assert_eq!(tests["result"]["rows"][0][0], "pkg.test_target");

    let missing = RunRecipeQueryArgs {
        variables: Map::from_iter([("qualified_name".into(), json!("pkg.missing"))]),
        ..canonical_args("resolve_function")
    };
    let missing = serde_json::to_value(run_recipe_query(&state, &catalog, missing)).unwrap();
    assert_eq!(missing["result"]["rows"], json!([]));
    assert_eq!(missing["result"]["row_count"], 0);

    let orphan_variables = Map::from_iter([("qualified_name".into(), json!("pkg.orphan"))]);
    let orphan_resolve = RunRecipeQueryArgs {
        variables: orphan_variables.clone(),
        ..canonical_args("resolve_function")
    };
    let orphan_resolve =
        serde_json::to_value(run_recipe_query(&state, &catalog, orphan_resolve)).unwrap();
    assert_eq!(orphan_resolve["result"]["row_count"], 1);

    let orphan_callers = RunRecipeQueryArgs {
        variables: orphan_variables,
        ..canonical_args("direct_callers")
    };
    let orphan_callers =
        serde_json::to_value(run_recipe_query(&state, &catalog, orphan_callers)).unwrap();
    assert_eq!(orphan_callers["result"]["rows"], json!([]));
    assert_eq!(orphan_callers["result"]["row_count"], 0);
}

fn cypher_result(columns: Vec<&str>, rows: Vec<Vec<KgliteValue>>) -> CypherResult {
    CypherResult {
        columns: columns.into_iter().map(str::to_string).collect(),
        rows,
        stats: None,
        profile: None,
        diagnostics: None,
        lazy: None,
    }
}
