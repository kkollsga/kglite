//! DataFrame builders for code-tree relationships.

use super::entity_frames::{bool_col, build_df, int_col, str_col};
use crate::code_tree::models::ParseResult;
use crate::datatypes::values::{ColumnType, DataFrame};
use std::collections::{BTreeMap, HashSet};

// ── Edge DataFrame builders ─────────────────────────────────────────

pub(super) fn contains_edges_df(edges: &[super::super::other_edges::ContainsEdge]) -> DataFrame {
    let parent: Vec<Option<String>> = edges.iter().map(|e| Some(e.parent.clone())).collect();
    let child: Vec<Option<String>> = edges.iter().map(|e| Some(e.child.clone())).collect();
    build_df(vec![
        ("parent", ColumnType::String, str_col(parent)),
        ("child", ColumnType::String, str_col(child)),
    ])
}

pub(super) fn import_edges_df(edges: &[super::super::other_edges::ImportEdge]) -> DataFrame {
    let src: Vec<Option<String>> = edges.iter().map(|e| Some(e.file_path.clone())).collect();
    let tgt: Vec<Option<String>> = edges.iter().map(|e| Some(e.module.clone())).collect();
    build_df(vec![
        ("file_path", ColumnType::String, str_col(src)),
        ("module", ColumnType::String, str_col(tgt)),
    ])
}

pub(super) fn file_import_edges_df(
    edges: &[super::super::other_edges::FileImportEdge],
) -> DataFrame {
    let src: Vec<Option<String>> = edges.iter().map(|e| Some(e.source.clone())).collect();
    let tgt: Vec<Option<String>> = edges.iter().map(|e| Some(e.target.clone())).collect();
    let count: Vec<Option<i64>> = edges.iter().map(|e| Some(e.import_count)).collect();
    build_df(vec![
        ("source", ColumnType::String, str_col(src)),
        ("target", ColumnType::String, str_col(tgt)),
        ("import_count", ColumnType::Int64, int_col(count)),
    ])
}

pub(super) fn route_nodes_df(nodes: &[super::super::routes::RouteNode]) -> DataFrame {
    let id: Vec<Option<String>> = nodes.iter().map(|n| Some(n.id.clone())).collect();
    let name: Vec<Option<String>> = nodes.iter().map(|n| Some(n.name.clone())).collect();
    let path: Vec<Option<String>> = nodes.iter().map(|n| Some(n.path.clone())).collect();
    let method: Vec<Option<String>> = nodes.iter().map(|n| Some(n.method.clone())).collect();
    let framework: Vec<Option<String>> = nodes.iter().map(|n| Some(n.framework.clone())).collect();
    let file_path: Vec<Option<String>> = nodes.iter().map(|n| Some(n.file_path.clone())).collect();
    let line_number: Vec<Option<i64>> = nodes.iter().map(|n| Some(n.line_number as i64)).collect();
    build_df(vec![
        ("id", ColumnType::String, str_col(id)),
        ("name", ColumnType::String, str_col(name)),
        ("path", ColumnType::String, str_col(path)),
        ("method", ColumnType::String, str_col(method)),
        ("framework", ColumnType::String, str_col(framework)),
        ("file_path", ColumnType::String, str_col(file_path)),
        ("line_number", ColumnType::Int64, int_col(line_number)),
    ])
}

pub(super) fn route_edges_df(edges: &[super::super::routes::RouteEdge]) -> DataFrame {
    let route_id: Vec<Option<String>> = edges.iter().map(|e| Some(e.route_id.clone())).collect();
    let func: Vec<Option<String>> = edges
        .iter()
        .map(|e| Some(e.function_qname.clone()))
        .collect();
    build_df(vec![
        ("route_id", ColumnType::String, str_col(route_id)),
        ("function_qname", ColumnType::String, str_col(func)),
    ])
}

pub(super) fn decorates_edges_df(edges: &[super::super::other_edges::DecoratesEdge]) -> DataFrame {
    let dec: Vec<Option<String>> = edges.iter().map(|e| Some(e.decorator.clone())).collect();
    let fun: Vec<Option<String>> = edges.iter().map(|e| Some(e.function.clone())).collect();
    let name: Vec<Option<String>> = edges
        .iter()
        .map(|e| Some(e.decorator_name.clone()))
        .collect();
    build_df(vec![
        ("decorator", ColumnType::String, str_col(dec)),
        ("function", ColumnType::String, str_col(fun)),
        ("decorator_name", ColumnType::String, str_col(name)),
    ])
}

pub(super) fn call_edges_df(edges: &[super::super::call_edges::CallEdge]) -> DataFrame {
    let caller: Vec<Option<String>> = edges.iter().map(|e| Some(e.caller.clone())).collect();
    let callee: Vec<Option<String>> = edges.iter().map(|e| Some(e.callee.clone())).collect();
    let lines: Vec<Option<String>> = edges.iter().map(|e| Some(e.call_lines.clone())).collect();
    let count: Vec<Option<i64>> = edges.iter().map(|e| Some(e.call_count)).collect();
    build_df(vec![
        ("caller", ColumnType::String, str_col(caller)),
        ("callee", ColumnType::String, str_col(callee)),
        ("call_lines", ColumnType::String, str_col(lines)),
        ("call_count", ColumnType::Int64, int_col(count)),
    ])
}

pub(super) fn implements_edges_df(edges: &[super::super::type_edges::ImplementsEdge]) -> DataFrame {
    let type_name: Vec<Option<String>> = edges.iter().map(|e| Some(e.type_name.clone())).collect();
    let iface: Vec<Option<String>> = edges
        .iter()
        .map(|e| Some(e.interface_name.clone()))
        .collect();
    build_df(vec![
        ("type_name", ColumnType::String, str_col(type_name)),
        ("interface_name", ColumnType::String, str_col(iface)),
    ])
}

pub(super) fn extends_edges_df(edges: &[super::super::type_edges::ExtendsEdge]) -> DataFrame {
    let child: Vec<Option<String>> = edges.iter().map(|e| Some(e.child_name.clone())).collect();
    let parent: Vec<Option<String>> = edges.iter().map(|e| Some(e.parent_name.clone())).collect();
    build_df(vec![
        ("child_name", ColumnType::String, str_col(child)),
        ("parent_name", ColumnType::String, str_col(parent)),
    ])
}

pub(super) fn has_method_edges_df(edges: &[super::super::type_edges::HasMethodEdge]) -> DataFrame {
    let owner: Vec<Option<String>> = edges.iter().map(|e| Some(e.owner.clone())).collect();
    let method: Vec<Option<String>> = edges.iter().map(|e| Some(e.method.clone())).collect();
    build_df(vec![
        ("owner", ColumnType::String, str_col(owner)),
        ("method", ColumnType::String, str_col(method)),
    ])
}

pub(super) fn uses_type_edges_df(edges: &[super::super::other_edges::UsesTypeEdge]) -> DataFrame {
    let fns: Vec<Option<String>> = edges.iter().map(|e| Some(e.function.clone())).collect();
    let types: Vec<Option<String>> = edges.iter().map(|e| Some(e.type_name.clone())).collect();
    let positions: Vec<Option<String>> =
        edges.iter().map(|e| Some(e.position.to_string())).collect();
    build_df(vec![
        ("function", ColumnType::String, str_col(fns)),
        ("type_name", ColumnType::String, str_col(types)),
        ("position", ColumnType::String, str_col(positions)),
    ])
}

pub(super) fn references_edges_df(
    edges: &[super::super::other_edges::ReferencesEdge],
) -> DataFrame {
    let fns: Vec<Option<String>> = edges.iter().map(|e| Some(e.function.clone())).collect();
    let consts: Vec<Option<String>> = edges.iter().map(|e| Some(e.constant.clone())).collect();
    let lines: Vec<Option<i64>> = edges.iter().map(|e| Some(e.line as i64)).collect();
    build_df(vec![
        ("function", ColumnType::String, str_col(fns)),
        ("constant", ColumnType::String, str_col(consts)),
        ("line", ColumnType::Int64, int_col(lines)),
    ])
}

pub(super) fn references_fn_edges_df(
    edges: &[super::super::other_edges::ReferencesFnEdge],
) -> DataFrame {
    let callers: Vec<Option<String>> = edges.iter().map(|e| Some(e.caller.clone())).collect();
    let callees: Vec<Option<String>> = edges.iter().map(|e| Some(e.callee.clone())).collect();
    let lines: Vec<Option<i64>> = edges.iter().map(|e| Some(e.line as i64)).collect();
    build_df(vec![
        ("caller", ColumnType::String, str_col(callers)),
        ("callee", ColumnType::String, str_col(callees)),
        ("line", ColumnType::Int64, int_col(lines)),
    ])
}

pub(super) fn module_contains_file_df(
    edges: &[super::super::other_edges::ModuleContainsFileEdge],
) -> DataFrame {
    let m: Vec<Option<String>> = edges.iter().map(|e| Some(e.module.clone())).collect();
    let p: Vec<Option<String>> = edges.iter().map(|e| Some(e.file_path.clone())).collect();
    build_df(vec![
        ("module", ColumnType::String, str_col(m)),
        ("file_path", ColumnType::String, str_col(p)),
    ])
}

pub(super) fn pyo3_binds_df(edges: &[super::super::other_edges::PyO3BindsEdge]) -> DataFrame {
    let py: Vec<Option<String>> = edges.iter().map(|e| Some(e.py_function.clone())).collect();
    let rs: Vec<Option<String>> = edges
        .iter()
        .map(|e| Some(e.rust_function.clone()))
        .collect();
    build_df(vec![
        ("py_function", ColumnType::String, str_col(py)),
        ("rust_function", ColumnType::String, str_col(rs)),
    ])
}

pub(super) fn ffi_exposes_df(edges: &[super::super::other_edges::FfiExposesEdge]) -> DataFrame {
    let m: Vec<Option<String>> = edges.iter().map(|e| Some(e.module_fn.clone())).collect();
    let t: Vec<Option<String>> = edges.iter().map(|e| Some(e.target_qname.clone())).collect();
    let py: Vec<Option<String>> = edges.iter().map(|e| Some(e.py_name.clone())).collect();
    build_df(vec![
        ("module_fn", ColumnType::String, str_col(m)),
        ("target_qname", ColumnType::String, str_col(t)),
        ("py_name", ColumnType::String, str_col(py)),
    ])
}

pub(super) fn external_nodes_df(nodes: &[super::super::type_edges::ExternalNode]) -> DataFrame {
    let qn: Vec<Option<String>> = nodes
        .iter()
        .map(|n| Some(n.qualified_name.clone()))
        .collect();
    let name: Vec<Option<String>> = nodes.iter().map(|n| Some(n.name.clone())).collect();
    let ext: Vec<Option<bool>> = nodes.iter().map(|_| Some(true)).collect();
    build_df(vec![
        ("qualified_name", ColumnType::String, str_col(qn)),
        ("name", ColumnType::String, str_col(name)),
        ("is_external", ColumnType::Boolean, bool_col(ext)),
    ])
}

pub struct DefinesEdge {
    pub source_type: String,
    pub source_id: String,
    pub target_type: String,
    pub target_id: String,
}

pub(super) fn defines_edges(result: &ParseResult) -> Vec<DefinesEdge> {
    let mut out = Vec::new();
    // File DEFINES Function — every function the file textually defines,
    // including class methods. The previous `if !is_method` filter dropped
    // every C# method (the language has no top-level functions), so 305k+
    // method nodes carried no DEFINES edge from a File and looked like
    // "external stubs" to any query that joined through File. The
    // logical Class -[HAS_METHOD]-> Function edge is still emitted
    // separately by the type-edge builder.
    for f in &result.functions {
        out.push(DefinesEdge {
            source_type: "File".into(),
            source_id: f.file_path.clone(),
            target_type: "Function".into(),
            target_id: f.qualified_name.clone(),
        });
    }
    // File DEFINES Class / Struct / Mixin / Enum / Interface / Protocol / Trait / Constant
    for c in &result.classes {
        let target_type = super::super::class_node_type(&c.kind);
        out.push(DefinesEdge {
            source_type: "File".into(),
            source_id: c.file_path.clone(),
            target_type: target_type.into(),
            target_id: c.qualified_name.clone(),
        });
    }
    for e in &result.enums {
        out.push(DefinesEdge {
            source_type: "File".into(),
            source_id: e.file_path.clone(),
            target_type: "Enum".into(),
            target_id: e.qualified_name.clone(),
        });
    }
    for i in &result.interfaces {
        let tt = match i.kind.as_str() {
            "trait" => "Trait",
            "protocol" => "Protocol",
            _ => "Interface",
        };
        out.push(DefinesEdge {
            source_type: "File".into(),
            source_id: i.file_path.clone(),
            target_type: tt.into(),
            target_id: i.qualified_name.clone(),
        });
    }
    for c in &result.constants {
        out.push(DefinesEdge {
            source_type: "File".into(),
            source_id: c.file_path.clone(),
            target_type: "Constant".into(),
            target_id: c.qualified_name.clone(),
        });
    }
    for e in &result.elements {
        out.push(DefinesEdge {
            source_type: "File".into(),
            source_id: e.file_path.clone(),
            target_type: "Element".into(),
            target_id: e.qualified_name.clone(),
        });
    }
    for s in &result.selectors {
        out.push(DefinesEdge {
            source_type: "File".into(),
            source_id: s.file_path.clone(),
            target_type: "Selector".into(),
            target_id: s.qualified_name.clone(),
        });
    }
    out
}

/// BTreeMap, not HashMap: the caller feeds these frames to `add_connections`,
/// which treats the first-ever batch of a connection type as initial load and
/// skips edge-existence checks — so whichever pair iterates first gets
/// different dedup semantics. Hash iteration order is randomized per process,
/// which made total edge counts flap run-to-run (three stable values on
/// distillPDF). Ordered iteration + within-batch consolidation below make the
/// result independent of which pair goes first.
pub(super) fn defines_edges_df(edges: &[DefinesEdge]) -> BTreeMap<(String, String), DataFrame> {
    let mut by_pair: BTreeMap<(String, String), Vec<&DefinesEdge>> = BTreeMap::new();
    for e in edges {
        by_pair
            .entry((e.source_type.clone(), e.target_type.clone()))
            .or_default()
            .push(e);
    }
    by_pair
        .into_iter()
        .map(|(pair, list)| {
            // Consolidate duplicate (source, target) rows: a file that defines
            // the same selector/element name twice would otherwise become a
            // parallel duplicate edge on the skip-existence-check initial-load
            // path — batch.rs makes within-chunk consolidation the caller's
            // responsibility in that mode.
            let mut seen: HashSet<(&str, &str)> = HashSet::with_capacity(list.len());
            let mut src: Vec<Option<String>> = Vec::with_capacity(list.len());
            let mut tgt: Vec<Option<String>> = Vec::with_capacity(list.len());
            for e in &list {
                if seen.insert((e.source_id.as_str(), e.target_id.as_str())) {
                    src.push(Some(e.source_id.clone()));
                    tgt.push(Some(e.target_id.clone()));
                }
            }
            let df = build_df(vec![
                ("source", ColumnType::String, str_col(src)),
                ("target", ColumnType::String, str_col(tgt)),
            ]);
            (pair, df)
        })
        .collect()
}
