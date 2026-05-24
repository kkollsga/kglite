//! FastAPI route extraction.
//!
//! Looks for the FastAPI-typical decorator shape:
//!   - `@router.get('/x')` / `@router.post(...)` — APIRouter pattern.
//!   - `@app.get('/x')` / `@app.post(...)` with a FastAPI app.
//!
//! The Flask detector also accepts `@app.get(...)` — both emit a Route
//! with their respective framework label. We can't statically know
//! whether `app` is a Flask Flask or a FastAPI FastAPI without type
//! info, so deduplication happens at the graph level (the schema keys
//! Route by `(framework, method, path)`, so the same id from two
//! detectors merges into one node by upsert semantics).

use super::{first_string_literal, make_route_id, split_decorator, RouteEdge, RouteNode};
use crate::code_tree::models::FunctionInfo;

const FRAMEWORK: &str = "fastapi";

const METHODS: &[&str] = &[
    "get", "post", "put", "delete", "patch", "options", "head", "trace",
];

pub(super) fn detect(functions: &[FunctionInfo]) -> (Vec<RouteNode>, Vec<RouteEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for fn_info in functions {
        for raw in &fn_info.decorators {
            let Some((head, args)) = split_decorator(raw) else {
                continue;
            };
            let suffix = head.rsplit('.').next().unwrap_or(head).to_ascii_lowercase();
            if !METHODS.contains(&suffix.as_str()) {
                continue;
            }
            // Recognise FastAPI-typical holders. `app` is also Flask;
            // we still emit a fastapi-tagged Route — see module docs.
            let holder = head.rsplit('.').nth(1).unwrap_or("").to_ascii_lowercase();
            if holder != "router" && holder != "api_router" && holder != "app" {
                continue;
            }
            let Some(path) = first_string_literal(args) else {
                continue;
            };
            let method = suffix.to_ascii_uppercase();
            let id = make_route_id(FRAMEWORK, &method, &path);
            nodes.push(RouteNode {
                id: id.clone(),
                name: path.clone(),
                path: path.clone(),
                method,
                framework: FRAMEWORK.to_string(),
                file_path: fn_info.file_path.clone(),
                line_number: fn_info.line_number,
            });
            edges.push(RouteEdge {
                route_id: id,
                function_qname: fn_info.qualified_name.clone(),
            });
        }
    }
    (nodes, edges)
}
