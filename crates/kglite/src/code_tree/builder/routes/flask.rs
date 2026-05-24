//! Flask route extraction.
//!
//! Recognises three decorator shapes on Python functions:
//!
//! 1. `@app.route('/x')` / `@app.route('/x', methods=['POST'])`
//! 2. `@app.get('/x')`, `@app.post('/x')`, ... (method shortcuts)
//! 3. `@blueprint.route(...)` / `@bp.route(...)` — any name ending in
//!    `.route` follows the same arg shape, so we accept the suffix
//!    pattern rather than enumerating blueprint variable names.
//!
//! FastAPI shares decorator shapes 1 and 2 but emits the `fastapi`
//! framework label — see `fastapi.rs`. The two detectors deliberately
//! cooperate by tagging the decorator-callee name into their own
//! framework label, so a `@router.post(...)` registered by a FastAPI
//! detector doesn't get a duplicate Flask emission.

use super::{
    first_string_literal, keyword_arg, make_route_id, parse_methods_list, split_decorator,
    RouteEdge, RouteNode,
};
use crate::code_tree::models::FunctionInfo;

const FRAMEWORK: &str = "flask";

/// Method shortcuts that map directly to HTTP verbs. Any decorator
/// suffixed with `.METHOD` where METHOD is in this list registers a
/// route for that verb. We accept the suffix to cover both `app.get`
/// (the Flask app instance) and `blueprint.get`.
const METHOD_SHORTCUTS: &[&str] = &["get", "post", "put", "delete", "patch", "options", "head"];

pub(super) fn detect(functions: &[FunctionInfo]) -> (Vec<RouteNode>, Vec<RouteEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for fn_info in functions {
        for raw in &fn_info.decorators {
            let Some((head, args)) = split_decorator(raw) else {
                continue;
            };
            let suffix = head.rsplit('.').next().unwrap_or(head).to_ascii_lowercase();

            // Reject decorators that are FastAPI's exclusive markers
            // (e.g. `router.get` on an APIRouter). We use the variable
            // name preceding `.` as a hint: `router` and `api_router`
            // are FastAPI conventions. False positives are rare and
            // both detectors emit a route with their own framework
            // label, so the worst case is a duplicated route — fine.
            if is_fastapi_holder(head) {
                continue;
            }

            // `@app.route('/x')` — path is positional, method via kwarg.
            if suffix == "route" {
                if let Some(path) = first_string_literal(args) {
                    let methods = keyword_arg(args, "methods")
                        .map(parse_methods_list)
                        .unwrap_or_else(|| vec!["ANY".to_string()]);
                    for method in methods {
                        emit(&mut nodes, &mut edges, fn_info, &path, &method);
                    }
                }
                continue;
            }

            // `@app.get('/x')` / `.post` / ... — method baked into the suffix.
            if METHOD_SHORTCUTS.contains(&suffix.as_str()) {
                if let Some(path) = first_string_literal(args) {
                    emit(
                        &mut nodes,
                        &mut edges,
                        fn_info,
                        &path,
                        &suffix.to_ascii_uppercase(),
                    );
                }
            }
        }
    }
    (nodes, edges)
}

fn emit(
    nodes: &mut Vec<RouteNode>,
    edges: &mut Vec<RouteEdge>,
    fn_info: &FunctionInfo,
    path: &str,
    method: &str,
) {
    let id = make_route_id(FRAMEWORK, method, path);
    nodes.push(RouteNode {
        id: id.clone(),
        name: path.to_string(),
        path: path.to_string(),
        method: method.to_string(),
        framework: FRAMEWORK.to_string(),
        file_path: fn_info.file_path.clone(),
        line_number: fn_info.line_number,
    });
    edges.push(RouteEdge {
        route_id: id,
        function_qname: fn_info.qualified_name.clone(),
    });
}

fn is_fastapi_holder(head: &str) -> bool {
    // The variable preceding `.method` is the routing-app instance.
    let holder = head.rsplit('.').nth(1).unwrap_or("").to_ascii_lowercase();
    holder == "router" || holder == "api_router"
}
