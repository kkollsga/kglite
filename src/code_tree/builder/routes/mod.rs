//! Web-framework route extraction.
//!
//! Recognizes URL-routing patterns across web frameworks and synthesizes
//! `Route` nodes linked to handler `Function`s via `HANDLES` edges. The
//! per-framework files under this directory each implement a single
//! `detect(parse) -> (Vec<RouteNode>, Vec<RouteEdge>)` entry point.
//!
//! Direction: `Route -[HANDLES]-> Function`. Reads naturally as "this
//! route handles via that function", so the typical query
//!
//! ```cypher
//! MATCH (r:Route)-[:HANDLES]->(f:Function) WHERE r.path STARTS WITH '/api'
//! RETURN r.method, r.path, f.qualified_name
//! ```
//!
//! returns one row per endpoint.
//!
//! Frameworks shipped in 0.9.34:
//!   - **Flask** — `@app.route(...)`, `@blueprint.route(...)`, method-shortcuts
//!     `@app.get(...)`, `@app.post(...)`, etc.
//!   - **FastAPI** — `@app.get(...)`, `@router.post(...)`, all HTTP verbs.
//!   - **Django** — `urlpatterns = [path('users/', view)]` in `urls.py`.
//!
//! Express, Axum, Rails, Spring, Laravel and the rest of CodeGraph's
//! framework list need parser-side capture of call-arguments (e.g.
//! `app.get('/x', handler)` in TS) which `FunctionInfo.calls` doesn't
//! preserve today. They land as follow-up PRs once the parser model
//! gains a `function_calls_with_args` channel; adding each new framework
//! after that is one new file in this directory plus a line below.

use crate::code_tree::models::{ConstantInfo, FunctionInfo};

mod django;
mod fastapi;
mod flask;

/// A discovered URL endpoint. The graph stores one Route node per
/// `(framework, method, path)` triple; conflicts (two handlers for the
/// same triple) are kept as parallel edges, since they often indicate
/// legitimate stacked decorators (`@app.get @app.post` on the same fn).
#[derive(Debug)]
pub struct RouteNode {
    /// Stable id — `"{FRAMEWORK}::{METHOD}::{PATH}"` (e.g. `"flask::GET::/users/{id}"`).
    pub id: String,
    /// Display name — the URL path (e.g. `"/users/{id}"`).
    pub name: String,
    /// URL path or pattern, framework-native syntax preserved.
    pub path: String,
    /// HTTP method or `"ANY"` for `@app.route(...)` without `methods=`.
    pub method: String,
    /// `"flask"` | `"fastapi"` | `"django"` (and more later).
    pub framework: String,
    /// File where the route is declared.
    pub file_path: String,
    /// Source line of the declaration.
    pub line_number: u32,
}

/// `Route -[HANDLES]-> Function` — links a route to the function that
/// handles its requests. Multiple routes can point at the same handler
/// (`@app.get @app.post on same fn`) and one handler can be hit by
/// multiple routes (no uniqueness constraint).
#[derive(Debug)]
pub struct RouteEdge {
    pub route_id: String,
    pub function_qname: String,
}

/// Run every registered framework detector over the parse result and
/// concatenate their outputs. The order in this match arm fixes the
/// node-id stable-prefix order — newer frameworks land at the end.
pub fn build_routes(
    functions: &[FunctionInfo],
    constants: &[ConstantInfo],
) -> (Vec<RouteNode>, Vec<RouteEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for (det_nodes, det_edges) in [
        flask::detect(functions),
        fastapi::detect(functions),
        django::detect(constants, functions),
    ] {
        nodes.extend(det_nodes);
        edges.extend(det_edges);
    }
    (nodes, edges)
}

// ── Per-framework shared helpers ────────────────────────────────────

/// Decorator parser: split `"app.route('/x', methods=['GET'])"` into
/// `("app.route", "'/x', methods=['GET']")`. Returns `None` if there's
/// no call-syntax (`@property` etc.).
pub(super) fn split_decorator(raw: &str) -> Option<(&str, &str)> {
    let open = raw.find('(')?;
    let close = raw.rfind(')')?;
    if close < open {
        return None;
    }
    Some((raw[..open].trim(), &raw[open + 1..close]))
}

/// Extract the first positional string-literal argument from a decorator
/// arg-list. Walks `"'/users', methods=['GET']"` and returns `"/users"`.
/// Handles single and double quotes; ignores f-string prefixes since path
/// patterns are almost always plain literals.
pub(super) fn first_string_literal(args: &str) -> Option<String> {
    let bytes = args.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b' ' || b == b'\t' {
            i += 1;
            continue;
        }
        if b == b'\'' || b == b'"' {
            let quote = b;
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != quote {
                // Skip simple `\<x>` escapes — path literals rarely have them
                // but this keeps the scan correct for `'\'`.
                if bytes[j] == b'\\' && j + 1 < bytes.len() {
                    j += 2;
                    continue;
                }
                j += 1;
            }
            if j < bytes.len() {
                return Some(std::str::from_utf8(&bytes[start..j]).ok()?.to_string());
            }
            return None;
        }
        // Anything that isn't whitespace or a quote means the first arg is
        // not a string literal (e.g. a variable, an f-string, a list).
        return None;
    }
    None
}

/// Find the value of a keyword argument like `methods=['GET', 'POST']`
/// in a decorator arg-list. Returns the raw string between the brackets
/// or after `=`, unstripped. Used to extract Flask's `methods=` and
/// `methods=` arguments.
pub(super) fn keyword_arg<'a>(args: &'a str, key: &str) -> Option<&'a str> {
    // Naive but sufficient for the shapes we care about: split at the
    // key, ensure preceded by start or `,`, look for `=`, then scan
    // until top-level comma respecting `[]`/`()`/quotes.
    let pat = format!("{key}=");
    let mut start = 0;
    while let Some(rel) = args[start..].find(&pat) {
        let abs = start + rel;
        let preceding_ok = abs == 0 || {
            let b = args.as_bytes()[abs - 1];
            b == b' ' || b == b','
        };
        if !preceding_ok {
            start = abs + 1;
            continue;
        }
        let after = abs + pat.len();
        // Scan until top-level comma.
        let bytes = args.as_bytes();
        let mut depth = 0i32;
        let mut in_quote: Option<u8> = None;
        let mut j = after;
        while j < bytes.len() {
            let c = bytes[j];
            if let Some(q) = in_quote {
                if c == b'\\' && j + 1 < bytes.len() {
                    j += 2;
                    continue;
                }
                if c == q {
                    in_quote = None;
                }
            } else {
                match c {
                    b'\'' | b'"' => in_quote = Some(c),
                    b'[' | b'(' | b'{' => depth += 1,
                    b']' | b')' | b'}' => depth -= 1,
                    b',' if depth == 0 => break,
                    _ => {}
                }
            }
            j += 1;
        }
        return Some(args[after..j].trim());
    }
    None
}

/// Parse a Python list-of-strings literal like `"['GET', 'POST']"` into
/// owned uppercased strings. Single-element forms work either with or
/// without brackets.
pub(super) fn parse_methods_list(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(trimmed);
    let mut out = Vec::new();
    for piece in inner.split(',') {
        let p = piece.trim().trim_matches(['\'', '"']);
        if !p.is_empty() {
            out.push(p.to_ascii_uppercase());
        }
    }
    out
}

/// Stable Route id used as both the node id and the source side of the
/// HANDLES edge. Keeps a parsable shape if anyone wants to split it.
pub(super) fn make_route_id(framework: &str, method: &str, path: &str) -> String {
    format!("{framework}::{method}::{path}")
}
