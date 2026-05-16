//! `KnowledgeGraph::explore(query)` — one-call codebase exploration.
//!
//! For code-tree graphs, lexically rank Function/Class/Interface nodes
//! against a free-text query, take the top `max_entities`, 2-hop
//! traverse along CALLS/USES_TYPE/HAS_METHOD/DEFINES, and return a
//! markdown report with entry points, a relationship map, and grouped
//! source slices.
//!
//! The lexical ranker is deliberately simple — exact-name match wins
//! big, name-contains is next, signature/docstring matches trail. We
//! don't currently use the vector embedder even when configured;
//! semantic re-ranking is a follow-up once we know what fixture-shape
//! makes a useful evaluation harness.
//!
//! Source slices come from `KnowledgeGraph::source_roots` if present,
//! otherwise we resolve `file_path` against the cwd. Reading happens
//! synchronously; for code-tree graphs with thousands of files the
//! total work is bounded by `max_entities` (typically 10).
//!
//! ## Output shape
//!
//! ```text
//! ## Query
//! <query>
//!
//! ## Entry points (3)
//! 1. **Function `parse_query`** — `src/parser.rs:42-78`
//!    signature: `pub fn parse_query(s: &str) -> Query`
//! 2. ...
//!
//! ## Related (12)
//! - Function `tokenize` — `src/lexer.rs:14-30`
//! - ...
//!
//! ## Source
//!
//! ### src/parser.rs
//! ```rust
//! 42  pub fn parse_query(s: &str) -> Query {
//! 43      ...
//! ```

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use petgraph::graph::NodeIndex;
use petgraph::Direction;

use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead;
use crate::graph::KnowledgeGraph;

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Int64(n) => n.to_string(),
        Value::Float64(f) => f.to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::Null => String::new(),
        other => format!("{:?}", other),
    }
}

/// Tunable knobs. Defaults match the pymethod's defaults.
#[derive(Debug, Clone)]
pub struct ExploreOptions {
    pub max_entities: usize,
    pub max_depth: usize,
    pub include_source: bool,
    /// Maximum total source-section bytes returned in the markdown body.
    /// Prevents a runaway query from emitting a hundred-KB blob.
    pub source_budget_bytes: usize,
}

impl Default for ExploreOptions {
    fn default() -> Self {
        Self {
            max_entities: 10,
            max_depth: 2,
            include_source: true,
            source_budget_bytes: 32 * 1024,
        }
    }
}

#[derive(Debug)]
struct Hit {
    nidx: NodeIndex,
    score: i64,
    qname: String,
    name: String,
    kind: String,
    file_path: Option<String>,
    line_number: Option<i64>,
    end_line: Option<i64>,
    signature: Option<String>,
}

const ENTRY_NODE_TYPES: &[&str] = &[
    "Function",
    "Class",
    "Struct",
    "Interface",
    "Trait",
    "Protocol",
    "Enum",
];

/// Edge types to traverse when collecting the neighborhood around each
/// entry point. Hops are undirected; we expand both inbound and
/// outbound matching edges so callers and callees both surface.
const TRAVERSAL_EDGES: &[&str] = &[
    "CALLS",
    "USES_TYPE",
    "HAS_METHOD",
    "DEFINES",
    "REFERENCES_FN",
];

/// Top-level entry point. Runs the search + traversal + rendering and
/// returns a markdown string.
pub fn explore_markdown(
    kg: &KnowledgeGraph,
    query: &str,
    opts: &ExploreOptions,
    source_roots: &[PathBuf],
) -> String {
    let query_trim = query.trim();
    if query_trim.is_empty() {
        return "## Query\n\n_empty query_\n".to_string();
    }
    let entry_hits = lexical_search(kg, query_trim, opts.max_entities);

    if entry_hits.is_empty() {
        return format!(
            "## Query\n\n`{query_trim}`\n\n_No matching Function/Class/Interface nodes in this graph._\n"
        );
    }

    let related = traverse(kg, &entry_hits, opts.max_depth);

    let mut out = String::new();
    out.push_str("## Query\n\n");
    out.push_str(&format!("`{query_trim}`\n\n"));

    out.push_str(&format!("## Entry points ({})\n\n", entry_hits.len()));
    for (i, hit) in entry_hits.iter().enumerate() {
        out.push_str(&format!(
            "{}. **{} `{}`** — `{}:{}-{}`\n",
            i + 1,
            hit.kind,
            hit.qname,
            hit.file_path.as_deref().unwrap_or("(unknown)"),
            hit.line_number.unwrap_or(0),
            hit.end_line.unwrap_or(0),
        ));
        if let Some(sig) = &hit.signature {
            if !sig.is_empty() {
                out.push_str(&format!("   signature: `{}`\n", truncate(sig, 200)));
            }
        }
    }
    out.push('\n');

    if !related.is_empty() {
        out.push_str(&format!("## Related ({})\n\n", related.len()));
        for hit in &related {
            out.push_str(&format!(
                "- {} `{}` — `{}:{}-{}`\n",
                hit.kind,
                hit.qname,
                hit.file_path.as_deref().unwrap_or("(unknown)"),
                hit.line_number.unwrap_or(0),
                hit.end_line.unwrap_or(0),
            ));
        }
        out.push('\n');
    }

    if opts.include_source {
        out.push_str("## Source\n\n");
        render_source_sections(
            &mut out,
            &entry_hits,
            &related,
            source_roots,
            opts.source_budget_bytes,
        );
    }

    out
}

fn lexical_search(kg: &KnowledgeGraph, query: &str, max_entities: usize) -> Vec<Hit> {
    let q_lower = query.to_lowercase();
    let mut hits: Vec<Hit> = Vec::new();

    for node_type in ENTRY_NODE_TYPES {
        let Some(idx_ref) = kg.inner.type_indices.get(node_type) else {
            continue;
        };
        for nidx in idx_ref.iter() {
            let Some(node) = kg.inner.graph.node_weight(nidx) else {
                continue;
            };
            let title = value_to_string(&node.title());
            let title_lower = title.to_lowercase();
            let mut score: i64 = 0;
            if title_lower == q_lower {
                score += 100;
            } else if title_lower.contains(&q_lower) {
                // Shorter names with a substring match get a boost so
                // `parse` ranks `parse_query` above `query_parser_helper`.
                score += 30 + (40 / (title_lower.len() as i64).max(1));
            }
            let signature = node
                .get_property("signature")
                .as_deref()
                .map(value_to_string);
            if let Some(sig) = signature.as_deref().map(str::to_lowercase) {
                if sig.contains(&q_lower) {
                    score += 10;
                }
            }
            let docstring = node
                .get_property("docstring")
                .as_deref()
                .map(value_to_string);
            if let Some(doc) = docstring.as_deref().map(str::to_lowercase) {
                if doc.contains(&q_lower) {
                    score += 5;
                }
            }
            if score == 0 {
                continue;
            }
            let file_path = node
                .get_property("file_path")
                .as_deref()
                .map(value_to_string);
            let line_number = node
                .get_property("line_number")
                .as_deref()
                .and_then(|v| match v {
                    Value::Int64(n) => Some(*n),
                    _ => None,
                });
            let end_line = node
                .get_property("end_line")
                .as_deref()
                .and_then(|v| match v {
                    Value::Int64(n) => Some(*n),
                    _ => None,
                });
            let qname = value_to_string(&node.id());
            hits.push(Hit {
                nidx,
                score,
                qname,
                name: title,
                kind: node_type.to_string(),
                file_path,
                line_number,
                end_line,
                signature,
            });
        }
    }

    hits.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.qname.cmp(&b.qname)));
    hits.truncate(max_entities);
    hits
}

fn traverse(kg: &KnowledgeGraph, seeds: &[Hit], max_depth: usize) -> Vec<Hit> {
    if max_depth == 0 {
        return Vec::new();
    }
    let edge_keys: Vec<InternedKey> = TRAVERSAL_EDGES
        .iter()
        .map(|n| InternedKey::from_str(n))
        .collect();
    let seed_set: HashSet<NodeIndex> = seeds.iter().map(|h| h.nidx).collect();
    let mut visited: HashSet<NodeIndex> = seed_set.clone();
    let mut frontier: Vec<NodeIndex> = seeds.iter().map(|h| h.nidx).collect();

    for _ in 0..max_depth {
        let mut next: Vec<NodeIndex> = Vec::new();
        for nidx in &frontier {
            for dir in [Direction::Outgoing, Direction::Incoming] {
                for er in kg.inner.graph.edges_directed(*nidx, dir) {
                    if !edge_keys.contains(&er.weight().connection_type) {
                        continue;
                    }
                    let other = if dir == Direction::Outgoing {
                        er.target()
                    } else {
                        er.source()
                    };
                    if visited.insert(other) {
                        next.push(other);
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }

    let mut out: Vec<Hit> = Vec::new();
    for nidx in visited {
        if seed_set.contains(&nidx) {
            continue;
        }
        let Some(node) = kg.inner.graph.node_weight(nidx) else {
            continue;
        };
        let kind = node.get_node_type_ref(&kg.inner.interner).to_string();
        // Only surface code-entity neighbors. Edges may also lead to
        // File / Module / Route nodes which are useful but visually
        // noisy in the "Related" list.
        if !ENTRY_NODE_TYPES.contains(&kind.as_str()) {
            continue;
        }
        let qname = value_to_string(&node.id());
        let name = value_to_string(&node.title());
        let file_path = node
            .get_property("file_path")
            .as_deref()
            .map(value_to_string);
        let line_number = node
            .get_property("line_number")
            .as_deref()
            .and_then(|v| match v {
                Value::Int64(n) => Some(*n),
                _ => None,
            });
        let end_line = node
            .get_property("end_line")
            .as_deref()
            .and_then(|v| match v {
                Value::Int64(n) => Some(*n),
                _ => None,
            });
        out.push(Hit {
            nidx,
            score: 0,
            qname,
            name,
            kind,
            file_path,
            line_number,
            end_line,
            signature: None,
        });
    }
    out.sort_by(|a, b| a.qname.cmp(&b.qname));
    out
}

/// Source rendering: dedupe (file, range) ranges across entries +
/// related, merge overlapping/adjacent ranges per file, then read
/// each merged span from disk (best-effort against `source_roots`).
fn render_source_sections(
    out: &mut String,
    entries: &[Hit],
    related: &[Hit],
    source_roots: &[PathBuf],
    budget: usize,
) {
    // Only entry points get source — neighborhood is referenced by
    // file:line in the Related list. Including 50 neighbor bodies
    // blows the budget on any non-trivial codebase.
    let mut by_file: HashMap<String, Vec<(i64, i64, String)>> = HashMap::new();
    for hit in entries {
        let (Some(path), Some(start), Some(end)) = (&hit.file_path, hit.line_number, hit.end_line)
        else {
            continue;
        };
        if start <= 0 || end < start {
            continue;
        }
        by_file
            .entry(path.clone())
            .or_default()
            .push((start, end, hit.name.clone()));
    }

    let _ = related; // see comment above
    let mut total = 0usize;
    let mut files: Vec<&String> = by_file.keys().collect();
    files.sort();
    for file in files {
        let mut ranges = by_file.get(file).cloned().unwrap_or_default();
        ranges.sort_by_key(|(s, _, _)| *s);
        let merged = merge_ranges(&ranges);

        let Some(content) = read_file(file, source_roots) else {
            out.push_str(&format!("### `{file}`\n_could not read source_\n\n"));
            continue;
        };
        let lines: Vec<&str> = content.lines().collect();

        out.push_str(&format!("### `{file}`\n\n"));
        for (s, e, names) in merged {
            if total >= budget {
                out.push_str("_… truncated (source budget reached) …_\n\n");
                return;
            }
            let s_idx = (s as usize).saturating_sub(1);
            let e_idx = (e as usize).min(lines.len());
            if s_idx >= lines.len() {
                continue;
            }
            out.push_str(&format!("<!-- {} -->\n```\n", names.join(", "),));
            for (line_no, line) in lines[s_idx..e_idx].iter().enumerate() {
                let actual_lineno = (s as usize) + line_no;
                let next = format!("{actual_lineno:>5}  {line}\n");
                total += next.len();
                out.push_str(&next);
                if total >= budget {
                    out.push_str("```\n_… truncated …_\n\n");
                    return;
                }
            }
            out.push_str("```\n\n");
        }
    }
}

/// Merge `(start, end, name)` triples that overlap or are within a
/// small gap. The name list accumulates so the rendered comment can
/// attribute the merged span back to all entry points that produced it.
fn merge_ranges(ranges: &[(i64, i64, String)]) -> Vec<(i64, i64, Vec<String>)> {
    if ranges.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<(i64, i64, Vec<String>)> = Vec::new();
    for (s, e, name) in ranges {
        match out.last_mut() {
            // Adjacent or overlapping: extend the last span.
            Some(last) if *s <= last.1 + 1 => {
                last.1 = last.1.max(*e);
                last.2.push(name.clone());
            }
            _ => out.push((*s, *e, vec![name.clone()])),
        }
    }
    out
}

fn read_file(path: &str, source_roots: &[PathBuf]) -> Option<String> {
    if let Ok(content) = std::fs::read_to_string(path) {
        return Some(content);
    }
    for root in source_roots {
        let candidate = root.join(path);
        if let Ok(content) = std::fs::read_to_string(&candidate) {
            return Some(content);
        }
    }
    None
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut end = n;
    while !s.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    format!("{}…", &s[..end])
}
