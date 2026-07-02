//! Shared Cypher execution helpers for interactive and one-shot CLI modes.

use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Result;
use kglite::api::param::kglite_value_to_json;
use kglite::api::session::{execute_mut, execute_read, ExecuteOptions, ExecuteOutcome};
use kglite::api::{make_dir_graph_mut, DirGraph, Value};

use crate::format::{render, Mode};

/// Per-query knobs shared by the REPL and one-shot commands.
#[derive(Debug, Default)]
pub struct QueryOptions {
    pub cancel: Option<&'static AtomicBool>,
    pub write_scope: Option<HashSet<String>>,
    pub git_sha: Option<String>,
    pub modified_by: Option<String>,
}

/// Execute one Cypher statement through the mutable session path.
///
/// `execute_mut` internally keeps read queries read-only, so this single seam
/// supports both the single-user REPL and write-enabled one-shot commands.
pub fn execute(
    graph: &mut Arc<DirGraph>,
    query: &str,
    params: &HashMap<String, Value>,
    options: &QueryOptions,
) -> Result<ExecuteOutcome> {
    let mut opts = ExecuteOptions::new(params);
    opts.cancel = options.cancel;
    opts.write_scope = options.write_scope.as_ref();
    opts.git_sha = options.git_sha.as_deref();
    opts.modified_by = options.modified_by.as_deref();

    let g = make_dir_graph_mut(graph);
    Ok(execute_mut(g, query, &opts)?)
}

/// Execute one read-only Cypher statement.
pub fn execute_readonly(
    graph: &Arc<DirGraph>,
    query: &str,
    params: &HashMap<String, Value>,
) -> Result<ExecuteOutcome> {
    let opts = ExecuteOptions::new(params);
    Ok(execute_read(graph, query, &opts)?)
}

/// Render a Cypher outcome in the requested CLI mode.
pub fn render_outcome(mode: Mode, outcome: &ExecuteOutcome) -> String {
    let r = &outcome.result;
    render(mode, &r.columns, &r.rows)
}

/// Convert a Cypher outcome to typed JSON rows for agent protocols.
pub fn outcome_rows_json(outcome: &ExecuteOutcome) -> serde_json::Value {
    let r = &outcome.result;
    let arr: Vec<serde_json::Value> = r
        .rows
        .iter()
        .map(|row| {
            let mut obj = serde_json::Map::new();
            for (i, col) in r.columns.iter().enumerate() {
                let value = row
                    .get(i)
                    .map(kglite_value_to_json)
                    .unwrap_or(serde_json::Value::Null);
                obj.insert(col.clone(), value);
            }
            serde_json::Value::Object(obj)
        })
        .collect();
    serde_json::Value::Array(arr)
}

/// Write CLI output, treating a closed downstream pipe as successful exit.
pub fn write_stdout(text: &str) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    match stdout.write_all(text.as_bytes()) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => return Ok(()),
        Err(e) => return Err(e),
    }
    match stdout.write_all(b"\n") {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(e) => Err(e),
    }
}

pub fn parse_write_scope(raw: Option<&str>) -> Option<HashSet<String>> {
    raw.map(|s| {
        s.split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_string)
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_write_scope_splits_commas_and_ignores_blanks() {
        let scope = parse_write_scope(Some("Plan, Task,,Artifact ")).unwrap();
        assert!(scope.contains("Plan"));
        assert!(scope.contains("Task"));
        assert!(scope.contains("Artifact"));
        assert_eq!(scope.len(), 3);
    }

    #[test]
    fn parse_write_scope_none_is_unrestricted() {
        assert!(parse_write_scope(None).is_none());
    }
}
