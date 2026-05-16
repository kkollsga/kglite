//! Django route extraction.
//!
//! Django expresses routes as Python data — typically a top-level
//! `urlpatterns = [path('users/', UserView.as_view(), name='users'), ...]`
//! in a file named `urls.py`. The code_tree parser captures these as
//! `ConstantInfo` rows with `name = "urlpatterns"` and the source-text
//! preview of the list in `value_preview`.
//!
//! The detector scans every `urlpatterns` constant, extracts each
//! `path(...)` / `re_path(...)` / `url(...)` call's path and view
//! reference, and emits a `Route` node + a `HANDLES` edge to the view
//! function if it resolves against the project's Function set.
//!
//! Limitations (acceptable for v1):
//!  - `value_preview` is the first ~100 chars of the urlpatterns
//!    literal. Long urls.py files get truncated. The CHANGELOG notes
//!    this; the fix is a parser-side full-text capture, deferred.
//!  - Class-based views written as `UserView.as_view()` resolve on the
//!    bare `as_view` name, which collides across all CBVs in a project.
//!    Ambiguity-skipping (same policy as DECORATES) prevents false
//!    edges but loses the connection. Class-view linkage warrants a
//!    follow-up pass that resolves on the class name instead.

use std::collections::HashMap;

use super::{make_route_id, RouteEdge, RouteNode};
use crate::code_tree::models::{ConstantInfo, FunctionInfo};

const FRAMEWORK: &str = "django";

pub(super) fn detect(
    constants: &[ConstantInfo],
    functions: &[FunctionInfo],
) -> (Vec<RouteNode>, Vec<RouteEdge>) {
    // Build a bare-name → unique qualified_name index for view-handler
    // resolution. Multiple matches → skip (we don't guess across
    // namespaces — same policy as DECORATES).
    let mut by_name: HashMap<&str, Vec<&str>> = HashMap::new();
    for f in functions {
        by_name
            .entry(f.name.as_str())
            .or_default()
            .push(f.qualified_name.as_str());
    }

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for c in constants {
        if c.name != "urlpatterns" {
            continue;
        }
        let Some(preview) = c.value_preview.as_deref() else {
            continue;
        };
        // urls.py routes are file-scoped (not always literally named
        // `urls.py` — Django also accepts `routing.py` etc.) but the
        // parsed file path is preserved on the constant.
        for entry in iter_path_calls(preview) {
            let id = make_route_id(FRAMEWORK, &entry.method, &entry.path);
            nodes.push(RouteNode {
                id: id.clone(),
                name: entry.path.clone(),
                path: entry.path,
                method: entry.method,
                framework: FRAMEWORK.to_string(),
                file_path: c.file_path.clone(),
                line_number: c.line_number,
            });
            // Resolve view to a Function node if unambiguous.
            if let Some(view) = entry.view_name.as_deref() {
                let bare = strip_to_bare_name(view);
                if let Some(candidates) = by_name.get(bare) {
                    if candidates.len() == 1 {
                        edges.push(RouteEdge {
                            route_id: id,
                            function_qname: candidates[0].to_string(),
                        });
                    }
                }
            }
        }
    }
    (nodes, edges)
}

struct PathCall {
    /// HTTP verb is not encoded at the URL-router layer in Django, so we
    /// stamp every urlpattern entry as `"ANY"`. View-level method
    /// restriction lives in `@require_http_methods` decorators which
    /// the DECORATES pass already records on the view function.
    method: String,
    path: String,
    view_name: Option<String>,
}

/// Walk a `urlpatterns` text preview and yield one `PathCall` per
/// `path(...)` / `re_path(...)` / `url(...)` call. Tolerant to partial
/// inputs — truncated previews simply yield fewer entries.
fn iter_path_calls(preview: &str) -> Vec<PathCall> {
    let mut out = Vec::new();
    let bytes = preview.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Match call openers: `path(`, `re_path(`, `url(`.
        let kind_len = if preview[i..].starts_with("path(") {
            "path(".len()
        } else if preview[i..].starts_with("re_path(") {
            "re_path(".len()
        } else if preview[i..].starts_with("url(") {
            "url(".len()
        } else {
            i += 1;
            continue;
        };
        let body_start = i + kind_len;
        let body_end = match scan_balanced(bytes, body_start, b'(', b')') {
            Some(j) => j,
            None => break, // truncated; stop
        };
        let args = &preview[body_start..body_end];
        let Some(path) = first_call_arg_string(args) else {
            i = body_end + 1;
            continue;
        };
        let view_name = second_call_arg_ident(args);
        out.push(PathCall {
            method: "ANY".to_string(),
            path,
            view_name,
        });
        i = body_end + 1;
    }
    out
}

fn scan_balanced(bytes: &[u8], from: usize, open: u8, close: u8) -> Option<usize> {
    let mut depth = 1i32;
    let mut in_quote: Option<u8> = None;
    let mut i = from;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = in_quote {
            if c == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if c == q {
                in_quote = None;
            }
        } else {
            match c {
                b'\'' | b'"' => in_quote = Some(c),
                x if x == open => depth += 1,
                x if x == close => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

fn first_call_arg_string(args: &str) -> Option<String> {
    super::first_string_literal(args)
}

/// Best-effort extraction of the second positional arg as an
/// identifier expression. Handles `UserView.as_view()`, plain `view`,
/// and dotted refs. Returns `None` if the second arg is a call to
/// `include()` (those are nested urlpatterns, not handlers).
fn second_call_arg_ident(args: &str) -> Option<String> {
    let bytes = args.as_bytes();
    // Skip first arg (string literal).
    let first_end = skip_first_arg(bytes)?;
    let mut i = first_end + 1;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b',') {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    // Read up to next top-level comma.
    let mut depth = 0i32;
    let mut in_quote: Option<u8> = None;
    let mut j = i;
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
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth -= 1,
                b',' if depth == 0 => break,
                _ => {}
            }
        }
        j += 1;
    }
    let raw = args[i..j].trim();
    if raw.starts_with("include(") {
        return None;
    }
    Some(raw.to_string())
}

fn skip_first_arg(bytes: &[u8]) -> Option<usize> {
    // First arg is a quoted string (per first_call_arg_string).
    let mut i = 0;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    let q = bytes[i];
    if q != b'\'' && q != b'"' {
        return None;
    }
    let mut j = i + 1;
    while j < bytes.len() && bytes[j] != q {
        if bytes[j] == b'\\' && j + 1 < bytes.len() {
            j += 2;
            continue;
        }
        j += 1;
    }
    if j < bytes.len() {
        Some(j)
    } else {
        None
    }
}

fn strip_to_bare_name(view: &str) -> &str {
    // `UserView.as_view()` → `as_view`; `mod.helper` → `helper`; `foo` → `foo`.
    let head = view.split('(').next().unwrap_or(view);
    head.rsplit('.').next().unwrap_or(head)
}
