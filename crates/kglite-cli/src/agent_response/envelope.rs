use std::path::Path;

use kglite::api::cypher;
use kglite::api::session::ExecuteOutcome;
use serde_json::{json, Value};

pub(crate) fn result_envelope(outcome: &ExecuteOutcome, query: &str, graph: &Path) -> Value {
    let result = &outcome.result;
    let rows: Vec<Vec<Value>> = result
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(kglite::api::param::kglite_value_to_json)
                .collect()
        })
        .collect();
    let diagnostics = result
        .diagnostics
        .as_ref()
        .map(|value| json!(value))
        .unwrap_or(Value::Null);
    let (engine_row_limit, rows_before_engine_limit) = result
        .diagnostics
        .as_ref()
        .map(|value| (value.row_limit, value.total_rows))
        .unwrap_or((None, None));
    let features = cypher::query_features(query).ok();
    let query_literal_limits = features
        .as_ref()
        .map(|value| value.literal_limits.clone())
        .unwrap_or_default();
    // A literal LIMIT may be nested, so equality is evidence about the
    // executed result only; it is not a database-population claim.
    let literal_limit_status = if query_literal_limits.is_empty() {
        "no_literal_limit"
    } else if query_literal_limits
        .iter()
        .any(|limit| *limit >= 0 && rows.len() == *limit as usize)
    {
        "executed_rows_match_a_literal_limit"
    } else {
        "executed_rows_do_not_match_a_literal_limit"
    };
    let navigation = navigation(&result.columns, &rows, 0);
    json!({
        "schema_version": 1,
        "kind": "cypher_result",
        "columns": result.columns,
        "rows": rows,
        "diagnostics": diagnostics,
        "coverage": {
            "executed_rows": rows.len(),
            "engine_row_limit": engine_row_limit,
            "rows_before_engine_limit": rows_before_engine_limit,
            "query_literal_limits": query_literal_limits,
            "literal_limit_status": literal_limit_status,
            "database_population": "unknown"
        },
        "identity": {
            "footer": format!("Graph: {}", graph.display()),
            "rebuild_warning": null,
            "steering": ["Expansion reads retained evidence and never reruns this query."]
        },
        "operation": {
            "mutation": outcome.is_mutation,
            "output_format": if outcome.output_format == cypher::OutputFormat::Csv { "csv" } else { "rows" }
        },
        "representation": {
            "text_complete": false,
            "text_kind": "agent_json",
            "text_rows": 0,
            "text_row_limit": 0
        },
        "navigation": navigation
    })
}

pub(crate) fn error_envelope(error: &anyhow::Error, query: &str, graph: &Path) -> Value {
    json!({
        "schema_version": 1,
        "kind": "cypher_result",
        "columns": [],
        "rows": [],
        "diagnostics": {"errors": [{"message": format!("{error:#}")}]},
        "coverage": {
            "executed_rows": 0,
            "engine_row_limit": null,
            "rows_before_engine_limit": null,
            "query_literal_limits": cypher::query_features(query).map(|v| v.literal_limits).unwrap_or_default(),
            "literal_limit_status": "operation_failed",
            "database_population": "unknown"
        },
        "identity": {"footer": format!("Graph: {}", graph.display()), "rebuild_warning": null, "steering": []},
        "operation": {"mutation": null, "output_format": "rows"},
        "representation": {"text_complete": true, "text_kind": "error", "text_rows": 0, "text_row_limit": 0},
        "navigation": navigation(&[], &[], 0)
    })
}

fn navigation(columns: &[String], rows: &[Vec<Value>], text_rows: usize) -> Value {
    const COLUMN_LIMIT: usize = 24;
    const LABEL_LIMIT: usize = 128;
    json!({
        "schema_version": 1,
        "available": {
            "row_count": rows.len(),
            "column_count": columns.len(),
            "columns": columns.iter().take(COLUMN_LIMIT)
                .map(|name| name.chars().take(LABEL_LIMIT).collect::<String>()).collect::<Vec<_>>(),
            "sections": [
                {"json_pointer":"/columns","description":"ordered column names"},
                {"json_pointer":"/rows","description":"positional rows in original order"},
                {"json_pointer":"/diagnostics","description":"engine diagnostics and warnings"},
                {"json_pointer":"/coverage","description":"query and executor limit facts"},
                {"json_pointer":"/identity","description":"active graph and response steering"},
                {"json_pointer":"/operation","description":"read/write and output format"},
                {"json_pointer":"/representation","description":"text presentation completeness"}
            ]
        },
        "omissions": {"column_names": columns.len().saturating_sub(COLUMN_LIMIT), "complete_column_names_at": "/columns"},
        "targets": [
            target("continue rows after the text presentation", "/rows", text_rows),
            target("inspect ordered column names", "/columns", 0),
            target("inspect warnings and execution facts", "/diagnostics", 0),
            target("inspect query and executor limits", "/coverage", 0)
        ],
        "observed_value_targets": observed_value_targets(rows),
        "composition": "Use a target's json_pointer, offset and response in the generated response expand command."
    })
}

fn target(purpose: &str, pointer: &str, offset: usize) -> Value {
    json!({"purpose":purpose,"json_pointer":pointer,"offset":offset,"response":{"mode":"bounded","max_bytes":4096}})
}

fn observed_value_targets(rows: &[Vec<Value>]) -> Vec<Value> {
    fn visit(value: &Value, pointer: String, depth: usize, targets: &mut Vec<Value>) {
        if targets.len() >= 16 {
            return;
        }
        match value {
            Value::Array(values) if depth < 6 => {
                for (index, value) in values.iter().enumerate() {
                    visit(value, format!("{pointer}/{index}"), depth + 1, targets);
                }
            }
            Value::Object(values) if depth < 6 => {
                for (name, value) in values {
                    let escaped = name.replace('~', "~0").replace('/', "~1");
                    visit(value, format!("{pointer}/{escaped}"), depth + 1, targets);
                }
            }
            _ => targets.push(json!({"json_pointer":pointer,"offset":0,"response":{"mode":"bounded","max_bytes":4096}})),
        }
    }
    let mut targets = Vec::new();
    for (row, values) in rows.iter().take(2).enumerate() {
        for (column, value) in values.iter().enumerate() {
            visit(value, format!("/rows/{row}/{column}"), 0, &mut targets);
        }
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_canonical_envelope_matches_v1_field_contract() {
        let outcome = ExecuteOutcome {
            result: cypher::CypherResult {
                columns: vec!["duplicate".into(), "duplicate".into()],
                rows: vec![vec![kglite::api::Value::Null, kglite::api::Value::Int64(7)]],
                stats: None,
                profile: None,
                diagnostics: None,
                lazy: None,
            },
            is_mutation: false,
            output_format: cypher::OutputFormat::Default,
            explain: false,
        };
        let value = result_envelope(&outcome, "RETURN null, 7 LIMIT 1", Path::new("g.kgl"));
        assert_eq!(
            value
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            [
                "columns",
                "coverage",
                "diagnostics",
                "identity",
                "kind",
                "navigation",
                "operation",
                "representation",
                "rows",
                "schema_version",
            ]
        );
        assert_eq!(value["columns"], json!(["duplicate", "duplicate"]));
        assert_eq!(value["rows"], json!([[null, 7]]));
        assert_eq!(
            value["coverage"],
            json!({
                "executed_rows":1,"engine_row_limit":null,"rows_before_engine_limit":null,
                "query_literal_limits":[1],"literal_limit_status":"executed_rows_match_a_literal_limit",
                "database_population":"unknown"
            })
        );
        assert_eq!(
            value["representation"],
            json!({"text_complete":false,"text_kind":"agent_json","text_rows":0,"text_row_limit":0})
        );
        assert_eq!(value["navigation"]["available"]["row_count"], 1);
        assert_eq!(
            value["navigation"]["observed_value_targets"][1]["json_pointer"],
            "/rows/0/1"
        );
    }
}
