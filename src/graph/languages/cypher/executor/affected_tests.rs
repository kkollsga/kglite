//! `CALL affected_tests({files: [...], max_depth: N?}) YIELD test_file, depth`
//!
//! Given a seed set of changed file paths, BFS over inbound `IMPORTS` edges
//! (the "who imports me" direction) and yield the subset of reached File
//! nodes whose `is_test` property is true.
//!
//! Builds on the File → File IMPORTS edges added in 0.9.34: the seed files
//! are looked up in the File-type index by their `path` property, and the
//! traversal exclusively follows IMPORTS into the File-Class layer.
//!
//! Seeds themselves are skipped from the output even if they are tests —
//! the procedure answers "what other tests are affected", not "are the
//! changed files tests". Pre-existing edges between non-File nodes are
//! ignored: only IMPORTS edges count toward depth.
//!
//! `max_depth` defaults to 10 (a generous cap, since typical impact webs
//! are < 5 hops on real codebases). Pass `max_depth: 0` to get only the
//! direct importers of the seed files.
//!
//! @procedure: affected_tests

use std::collections::{HashMap, VecDeque};

use petgraph::graph::NodeIndex;
use petgraph::Direction;

use super::super::ast::YieldItem;
use super::super::result::ResultRow;
use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead;

const PROC: &str = "affected_tests";

pub(super) fn execute_affected_tests(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let files = require_files_param(params)?;
    if files.is_empty() {
        return Ok(Vec::new());
    }
    let max_depth = params
        .get("max_depth")
        .and_then(|v| match v {
            Value::Int64(n) => Some((*n).max(0) as usize),
            Value::Float64(n) => Some((*n).max(0.0) as usize),
            _ => None,
        })
        .unwrap_or(10);

    // YIELD columns are both optional individually — the user may yield
    // just `test_file` (the common "give me a path list" case) or both.
    // Validation of unknown YIELD names happens upstream in execute_call.
    let test_file_var = yield_alias(yield_items, "test_file");
    let depth_var = yield_alias(yield_items, "depth");
    if test_file_var.is_none() && depth_var.is_none() {
        return Err(format!(
            "CALL {PROC}: must YIELD at least one of 'test_file', 'depth'."
        ));
    }

    // Build path → NodeIndex lookup for File nodes. File's id_alias is
    // `path`, so the id field carries the file path.
    let file_idx = match graph.type_indices.get("File") {
        Some(idx) => idx,
        None => return Ok(Vec::new()), // not a code graph
    };
    let mut path_to_idx: HashMap<String, NodeIndex> = HashMap::new();
    for nidx in file_idx.iter() {
        if let Some(node) = graph.graph.node_weight(nidx) {
            if let Value::String(s) = node.id().as_ref() {
                path_to_idx.insert(s.clone(), nidx);
            }
        }
    }

    // BFS over inbound IMPORTS edges starting from each seed file.
    // Seeds that don't resolve to a known File node are silently skipped —
    // operators pass changed-file lists (e.g. from `git diff --name-only`)
    // and untracked/deleted files shouldn't error the impact analysis.
    let imports_key = InternedKey::from_str("IMPORTS");
    let mut depth_of: HashMap<NodeIndex, usize> = HashMap::new();
    let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();

    for seed in &files {
        if let Some(&nidx) = path_to_idx.get(seed) {
            if depth_of.insert(nidx, 0).is_none() {
                queue.push_back((nidx, 0));
            }
        }
    }

    while let Some((nidx, d)) = queue.pop_front() {
        if d >= max_depth {
            continue;
        }
        for er in graph.graph.edges_directed(nidx, Direction::Incoming) {
            if er.weight().connection_type != imports_key {
                continue;
            }
            let importer = er.source();
            if depth_of.contains_key(&importer) {
                continue;
            }
            depth_of.insert(importer, d + 1);
            queue.push_back((importer, d + 1));
        }
    }

    // Emit one row per visited *test* file, skipping the seeds themselves.
    let mut rows: Vec<(String, ResultRow)> = Vec::new();
    for (nidx, depth) in &depth_of {
        if *depth == 0 {
            continue;
        }
        let node = match graph.graph.node_weight(*nidx) {
            Some(n) => n,
            None => continue,
        };
        let is_test = matches!(
            node.get_property("is_test").as_deref(),
            Some(Value::Boolean(true))
        );
        if !is_test {
            continue;
        }
        let path = match node.id().as_ref() {
            Value::String(s) => s.clone(),
            _ => continue,
        };
        let mut row = ResultRow::new();
        if let Some(name) = &test_file_var {
            row.projected
                .insert(name.clone(), Value::String(path.clone()));
        }
        if let Some(name) = &depth_var {
            row.projected
                .insert(name.clone(), Value::Int64(*depth as i64));
        }
        rows.push((path, row));
    }

    // Sort for deterministic output (test_file ASC). The path is captured
    // alongside the row so sort works even when the caller didn't yield it.
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(rows.into_iter().map(|(_, row)| row).collect())
}

/// Resolve the `files` parameter — accepts either a single path string or
/// a list of paths. Empty lists are allowed (return Ok(empty)) so callers
/// can pass dynamically built path lists without special-casing the empty
/// case. Only a missing or wrong-type parameter is an error.
fn require_files_param(params: &HashMap<String, Value>) -> Result<Vec<String>, String> {
    match params.get("files") {
        Some(Value::String(s)) => {
            if s.starts_with('[') {
                // List literal serialized as JSON-ish string.
                let items = super::helpers::parse_list_value(&Value::String(s.clone()));
                Ok(items
                    .into_iter()
                    .filter_map(|v| match v {
                        Value::String(s) => Some(s),
                        _ => None,
                    })
                    .collect())
            } else {
                Ok(vec![s.clone()])
            }
        }
        Some(other) => Err(format!(
            "CALL {PROC}: parameter 'files' must be a list of file paths (got {other:?})."
        )),
        None => Err(format!(
            "CALL {PROC}: missing required parameter 'files'. \
             Use map syntax — e.g. CALL {PROC}({{files: ['src/foo.py'], max_depth: 5}})."
        )),
    }
}

/// Return the alias the caller gave a particular YIELD column, or `None`
/// if they didn't ask for it. Unknown YIELD names are rejected upstream
/// by the executor's `valid_yields` check, so we don't need to validate
/// names here.
fn yield_alias(yield_items: &[YieldItem], expected: &str) -> Option<String> {
    yield_items
        .iter()
        .find(|y| y.name == expected)
        .map(|item| item.alias.clone().unwrap_or_else(|| expected.to_string()))
}
