//! Cross-language HTTP boundary edges (Phase C.1).
//!
//! A client HTTP call and a server route handler can't be linked by any
//! single-file parse — they're in different files, often different languages.
//! This post-graph pass detects client calls in source and links the calling
//! `Function` to the server `Route` node sharing the same normalized path:
//!
//! ```text
//! Function -[CALLS_SERVICE]-> Route -[HANDLES]-> Function (handler)
//! ```
//!
//! so reverse-impact from a handler reaches every client through the route,
//! across the language boundary (e.g. a TS `fetch('/api/users')` reaching the
//! Python FastAPI handler for `/api/users`).
//!
//! Detection is best-effort source matching (the URL literal isn't on
//! `FunctionInfo.calls`), so every edge is tagged `confidence = "inferred"` —
//! never a parsed fact. Server routes come from the existing `routes/`
//! detectors; this only adds the client side, so it activates whenever the
//! graph already has `Route` nodes.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;

use crate::datatypes::values::{DataFrame, Value};
use crate::graph::dir_graph::DirGraph;
use crate::graph::mutation::maintain;
use crate::graph::storage::GraphRead;

/// Regexes capturing the URL/path literal (group 1) of a client HTTP call
/// across languages. Literal first-argument only — the common, unambiguous
/// shape; dynamic URLs are intentionally not guessed.
fn client_patterns() -> &'static Vec<Regex> {
    static PATS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATS.get_or_init(|| {
        [
            // JS / TS
            r#"fetch\(\s*['"`]([^'"`]+)"#,
            r#"axios\s*\.\s*(?:get|post|put|delete|patch|head)\(\s*['"`]([^'"`]+)"#,
            // Python
            r#"requests\s*\.\s*(?:get|post|put|delete|patch|head)\(\s*['"]([^'"]+)"#,
            r#"httpx\s*\.\s*(?:get|post|put|delete|patch|head)\(\s*['"]([^'"]+)"#,
            // Rust (reqwest)
            r#"reqwest::(?:blocking::)?get\(\s*"([^"]+)"#,
            // Go (net/http)
            r#"http\.(?:Get|Post|Head|PostForm)\(\s*"([^"]+)"#,
        ]
        .iter()
        .map(|p| Regex::new(p).expect("valid client-call regex"))
        .collect()
    })
}

/// Reduce a URL or path to a comparable route path: drop scheme+host, query,
/// and fragment. Returns `None` for values that aren't an absolute path
/// (relative imports, template variables, bare hostnames).
fn normalize_path(raw: &str) -> Option<String> {
    let mut s = raw.trim();
    if let Some(pos) = s.find("://") {
        let after = &s[pos + 3..];
        s = match after.find('/') {
            Some(slash) => &after[slash..],
            None => "/",
        };
    }
    s = s.split(['?', '#']).next().unwrap_or(s);
    if !s.starts_with('/') {
        return None;
    }
    Some(s.to_string())
}

/// Split a path into segments, mapping framework parameter syntax — Flask
/// `<int:id>`, FastAPI/Express `{id}` / `:id` — to a `{}` wildcard so a
/// concrete client path can match a server template.
fn route_segments(path: &str) -> Vec<String> {
    let norm = normalize_path(path).unwrap_or_else(|| path.to_string());
    norm.trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|seg| {
            if seg.starts_with('{') || seg.starts_with('<') || seg.starts_with(':') {
                "{}".to_string()
            } else {
                seg.to_string()
            }
        })
        .collect()
}

fn segments(path: &str) -> Vec<String> {
    path.trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// A concrete client path matches a route template when the segment counts
/// agree and each route segment is a wildcard or an exact literal match.
fn path_matches(client: &[String], route: &[String]) -> bool {
    client.len() == route.len() && client.iter().zip(route).all(|(c, r)| r == "{}" || r == c)
}

/// Link client HTTP calls to server `Route` nodes by normalized path. Returns
/// the number of `CALLS_SERVICE` edges added. No-op when the graph has no
/// routes (nothing to link to).
pub fn ingest_http_cross_edges(
    graph: &mut DirGraph,
    root: &Path,
    verbose: bool,
) -> Result<usize, String> {
    // 1. Routes: (route_id, route_segments).
    let mut routes: Vec<(String, Vec<String>)> = Vec::new();
    for idx in graph.graph.node_indices() {
        let Some(node) = graph.get_node(idx) else {
            continue;
        };
        if node.node_type_str(&graph.interner) != "Route" {
            continue;
        }
        let Value::String(route_id) = node.id().as_ref().clone() else {
            continue;
        };
        if let Some(Value::String(path)) = node.get_property("path").as_deref() {
            routes.push((route_id, route_segments(path)));
        }
    }
    if routes.is_empty() {
        return Ok(0);
    }

    // 2. Functions grouped by file: file_path -> [(qname, start_line, end_line)].
    let mut by_file: HashMap<String, Vec<(String, usize, usize)>> = HashMap::new();
    for idx in graph.graph.node_indices() {
        let Some(node) = graph.get_node(idx) else {
            continue;
        };
        if node.node_type_str(&graph.interner) != "Function" {
            continue;
        }
        let Value::String(qname) = node.id().as_ref().clone() else {
            continue;
        };
        let Some(Value::String(file_path)) = node.get_property("file_path").as_deref().cloned()
        else {
            continue;
        };
        let start = match node.get_property("line_number").as_deref() {
            Some(Value::Int64(n)) => *n as usize,
            _ => continue,
        };
        let end = match node.get_property("end_line").as_deref() {
            Some(Value::Int64(n)) => *n as usize,
            _ => start,
        };
        by_file
            .entry(file_path)
            .or_default()
            .push((qname, start, end));
    }

    // 3. Scan each function's source slice for client calls; match path → route.
    let pats = client_patterns();
    let mut edges: Vec<(String, String)> = Vec::new();
    for (file_path, fns) in &by_file {
        let Ok(src) = std::fs::read_to_string(root.join(file_path)) else {
            continue;
        };
        let lines: Vec<&str> = src.lines().collect();
        for (qname, start, end) in fns {
            let lo = start.saturating_sub(1);
            let hi = (*end).min(lines.len());
            for line in lines.get(lo..hi).unwrap_or(&[]) {
                // Skip whole-line comments (keeps `http://` in real calls intact,
                // unlike a naive `//` split).
                let t = line.trim_start();
                if t.starts_with("//") || t.starts_with('#') || t.starts_with('*') {
                    continue;
                }
                for re in pats {
                    for cap in re.captures_iter(line) {
                        let Some(url) = cap.get(1) else { continue };
                        let Some(path) = normalize_path(url.as_str()) else {
                            continue;
                        };
                        let segs = segments(&path);
                        for (route_id, rsegs) in &routes {
                            if path_matches(&segs, rsegs) {
                                edges.push((qname.clone(), route_id.clone()));
                            }
                        }
                    }
                }
            }
        }
    }

    edges.sort();
    edges.dedup();
    if edges.is_empty() {
        return Ok(0);
    }
    let n = edges.len();
    let rows: Vec<Vec<Value>> = edges
        .into_iter()
        .map(|(s, t)| {
            vec![
                Value::String(s),
                Value::String(t),
                Value::String("inferred".to_string()),
            ]
        })
        .collect();
    let df = DataFrame::from_cypher_rows(
        vec![
            "source_id".to_string(),
            "target_id".to_string(),
            "confidence".to_string(),
        ],
        rows,
    )?;
    maintain::add_connections(
        graph,
        df,
        "CALLS_SERVICE".to_string(),
        "Function".to_string(),
        "source_id".to_string(),
        "Route".to_string(),
        "target_id".to_string(),
        None,
        None,
        Some("update".to_string()),
    )?;
    if verbose {
        eprintln!("[cross-lang] {n} CALLS_SERVICE edge(s)");
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_host_query_fragment() {
        assert_eq!(
            normalize_path("https://api.x/users?q=1#f").as_deref(),
            Some("/users")
        );
        assert_eq!(normalize_path("/api/users").as_deref(), Some("/api/users"));
        assert_eq!(normalize_path("relative/x"), None);
    }

    #[test]
    fn parameterized_route_matches_concrete_path() {
        assert!(path_matches(
            &segments("/users/7"),
            &route_segments("/users/{id}")
        ));
        assert!(path_matches(
            &segments("/users/7"),
            &route_segments("/users/<int:id>")
        ));
        assert!(!path_matches(
            &segments("/users/7/x"),
            &route_segments("/users/{id}")
        ));
        assert!(!path_matches(
            &segments("/orders/7"),
            &route_segments("/users/{id}")
        ));
    }
}
