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
    state.tag_workspace_graph_dirty();

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
