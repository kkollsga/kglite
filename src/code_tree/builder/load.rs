//! Phase 3: load parsed entities into a KnowledgeGraph.
//!
//! Builds `crate::datatypes::DataFrame` objects directly from Rust record
//! vectors and hands them to `crate::graph::mutation::maintain::add_nodes` /
//! `add_connections` — no pandas, no PyO3 round-trip.

use crate::code_tree::models::{
    AttributeInfo, ClassInfo, ConstantInfo, EnumInfo, FieldEntry, FileInfo, FunctionInfo,
    InterfaceInfo, ParseResult, ProjectInfo,
};
use crate::datatypes::values::{ColumnData, ColumnType, DataFrame};
use crate::graph::mutation::maintain;
use crate::graph::{get_graph_mut, KnowledgeGraph};
use std::collections::{BTreeMap, HashMap};

pub struct ModuleRecord {
    pub qualified_name: String,
    pub name: String,
    pub language: String,
    pub is_test: bool,
}

/// Synthesize Module nodes from parsed files.
///
/// Each file's `module_path` defines a leaf module; every prefix of that path
/// becomes an ancestor module (same shape as builder.py::_build_modules).
///
/// Skips path segments whose name is purely ASCII digits — they appear when
/// a parser falls back to file-path-derived module names and the path
/// contains numeric directories (dotnet/runtime's
/// `tests/JIT/Regression/Runtime_<bug-id>/...` test layout in particular).
/// A bare integer is never a meaningful module name, and emitting them
/// pollutes the schema with thousands of `Module {title="125042"}` nodes.
pub fn build_modules(files: &[FileInfo]) -> Vec<ModuleRecord> {
    let mut seen: BTreeMap<String, ModuleRecord> = BTreeMap::new();
    for f in files {
        if f.module_path.is_empty() {
            continue;
        }
        let sep = pick_sep(&f.language);
        let parts: Vec<&str> = f.module_path.split(sep).collect();
        for end in 1..=parts.len() {
            let leaf = parts[end - 1];
            if is_numeric_segment(leaf) {
                continue;
            }
            let qname = parts[..end].join(sep);
            let name = leaf.to_string();
            seen.entry(qname.clone()).or_insert(ModuleRecord {
                qualified_name: qname,
                name,
                language: f.language.clone(),
                is_test: f.is_test && end == parts.len(),
            });
        }
    }
    seen.into_values().collect()
}

/// True when `s` is non-empty and made up entirely of ASCII digits.
fn is_numeric_segment(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_tree::models::FileInfo;

    fn file_with_module(language: &str, module_path: &str) -> FileInfo {
        FileInfo {
            path: format!("{}/dummy", module_path),
            filename: "dummy".into(),
            loc: 0,
            module_path: module_path.into(),
            language: language.into(),
            submodule_declarations: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            annotations: None,
            is_test: false,
            skip_reason: None,
        }
    }

    #[test]
    fn build_modules_skips_numeric_leaf() {
        // dotnet-style path with a numeric bug-id directory.
        let files = vec![file_with_module("csharp", "tests.JIT.Regression.125042")];
        let modules = build_modules(&files);
        let names: Vec<&str> = modules.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["tests", "JIT", "Regression"]);
        // The qualified ancestor "tests.JIT.Regression.125042" should NOT exist.
        assert!(!modules
            .iter()
            .any(|m| m.qualified_name == "tests.JIT.Regression.125042"));
    }

    #[test]
    fn build_modules_skips_numeric_intermediate() {
        // A numeric mid-segment must drop only itself and any descendants
        // whose deepest segment is also numeric. Non-numeric ancestors live.
        let files = vec![file_with_module("csharp", "a.123.c")];
        let modules = build_modules(&files);
        let qnames: Vec<&str> = modules.iter().map(|m| m.qualified_name.as_str()).collect();
        assert!(qnames.contains(&"a"));
        assert!(qnames.contains(&"a.123.c")); // c is alphanumeric — kept
        assert!(!qnames.contains(&"a.123")); // numeric leaf — dropped
    }

    #[test]
    fn build_modules_keeps_alphanumeric() {
        let files = vec![file_with_module("csharp", "Foo.Bar.V2")];
        let modules = build_modules(&files);
        let qnames: Vec<&str> = modules.iter().map(|m| m.qualified_name.as_str()).collect();
        assert!(qnames.contains(&"Foo"));
        assert!(qnames.contains(&"Foo.Bar"));
        assert!(qnames.contains(&"Foo.Bar.V2"));
    }

    #[test]
    fn is_numeric_segment_detection() {
        assert!(is_numeric_segment("0"));
        assert!(is_numeric_segment("125042"));
        assert!(!is_numeric_segment(""));
        assert!(!is_numeric_segment("v2"));
        assert!(!is_numeric_segment("Runtime_125042"));
        assert!(!is_numeric_segment("12.5")); // dot is not a digit
    }
}

fn pick_sep(language: &str) -> &'static str {
    match language {
        "rust" | "cpp" | "c" => "::",
        "python" | "java" | "csharp" => ".",
        "typescript" | "javascript" | "go" => "/",
        _ => ".",
    }
}

/// Build a DataFrame with the given columns, all of type `String` (most code
/// entity properties are strings). Numeric/boolean columns must be added
/// separately via `add_column_typed`.
fn df_with_cols(columns: &[&str]) -> DataFrame {
    DataFrame::new(
        columns
            .iter()
            .map(|c| (c.to_string(), ColumnType::String))
            .collect(),
    )
}

/// Add a pre-built `ColumnData::*` to an existing DataFrame.
fn add_typed_col(df: &mut DataFrame, name: &str, ct: ColumnType, data: ColumnData) {
    df.add_column(name.to_string(), ct, data)
        .unwrap_or_else(|e| panic!("add_column({name}) failed: {e}"));
}

/// Convenience: produce a fresh DataFrame with the given (name, type) columns
/// whose data is the corresponding `ColumnData::*` vec.
fn build_df(cols: Vec<(&str, ColumnType, ColumnData)>) -> DataFrame {
    let mut out = DataFrame::new(Vec::new());
    for (name, ct, data) in cols {
        add_typed_col(&mut out, name, ct, data);
    }
    out
}

fn str_col(values: Vec<Option<String>>) -> ColumnData {
    ColumnData::String(values)
}
fn int_col(values: Vec<Option<i64>>) -> ColumnData {
    ColumnData::Int64(values)
}
fn bool_col(values: Vec<Option<bool>>) -> ColumnData {
    ColumnData::Boolean(values)
}

fn py_err<S: Into<String>>(msg: S) -> String {
    msg.into()
}

/// Read a boolean flag from `FunctionInfo.metadata` with a `false` default.
/// Used to promote per-language metadata flags into typed DataFrame columns.
fn meta_bool(f: &FunctionInfo, key: &str) -> bool {
    f.metadata
        .get(key)
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Same, but for ClassInfo.
fn class_meta_bool(c: &ClassInfo, key: &str) -> bool {
    c.metadata
        .get(key)
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

// ── Entity → DataFrame builders ─────────────────────────────────────

fn files_df(files: &[FileInfo]) -> DataFrame {
    let path = files.iter().map(|f| Some(f.path.clone())).collect();
    let filename = files.iter().map(|f| Some(f.filename.clone())).collect();
    let loc = files.iter().map(|f| Some(f.loc as i64)).collect();
    let module_path = files.iter().map(|f| Some(f.module_path.clone())).collect();
    let language = files.iter().map(|f| Some(f.language.clone())).collect();
    let is_test = files.iter().map(|f| Some(f.is_test)).collect();
    let annotations = files
        .iter()
        .map(|f| {
            f.annotations
                .as_ref()
                .and_then(|a| serde_json::to_string(a).ok())
        })
        .collect();
    let skip_reason = files.iter().map(|f| f.skip_reason.clone()).collect();
    build_df(vec![
        ("path", ColumnType::String, str_col(path)),
        ("filename", ColumnType::String, str_col(filename)),
        ("loc", ColumnType::Int64, int_col(loc)),
        ("module", ColumnType::String, str_col(module_path)),
        ("language", ColumnType::String, str_col(language)),
        ("is_test", ColumnType::Boolean, bool_col(is_test)),
        ("annotations", ColumnType::String, str_col(annotations)),
        ("skip_reason", ColumnType::String, str_col(skip_reason)),
    ])
}

fn modules_df(modules: &[ModuleRecord]) -> DataFrame {
    build_df(vec![
        (
            "qualified_name",
            ColumnType::String,
            str_col(
                modules
                    .iter()
                    .map(|m| Some(m.qualified_name.clone()))
                    .collect(),
            ),
        ),
        // 0.9.30: `module` is an alias for qualified_name on Module
        // nodes so the same property name works across File/Module/
        // Function/Class/Constant/Enum/Interface — agents can write
        // `MATCH (n) WHERE n.module STARTS WITH 'xarray.core' RETURN n`
        // without branching on label.
        (
            "module",
            ColumnType::String,
            str_col(
                modules
                    .iter()
                    .map(|m| Some(m.qualified_name.clone()))
                    .collect(),
            ),
        ),
        (
            "name",
            ColumnType::String,
            str_col(modules.iter().map(|m| Some(m.name.clone())).collect()),
        ),
        (
            "language",
            ColumnType::String,
            str_col(modules.iter().map(|m| Some(m.language.clone())).collect()),
        ),
        (
            "is_test",
            ColumnType::Boolean,
            bool_col(modules.iter().map(|m| Some(m.is_test)).collect()),
        ),
    ])
}

fn functions_df(
    fns: &[FunctionInfo],
    file_is_test: &HashMap<&str, bool>,
    file_to_module: &HashMap<&str, &str>,
) -> DataFrame {
    build_df(vec![
        (
            "qualified_name",
            ColumnType::String,
            str_col(fns.iter().map(|f| Some(f.qualified_name.clone())).collect()),
        ),
        // 0.9.30: `module` is the dotted module path of the file this
        // function lives in — the same key Files carry. Lets agents
        // write `WHERE f.module STARTS WITH 'xarray.core'` against
        // Function nodes the same way they would against File nodes.
        // Pre-0.9.30 the property only existed on File + Module nodes,
        // which silently returned zero rows for cross-type module
        // filters.
        (
            "module",
            ColumnType::String,
            str_col(
                fns.iter()
                    .map(|f| {
                        file_to_module
                            .get(f.file_path.as_str())
                            .map(|m| (*m).to_string())
                    })
                    .collect(),
            ),
        ),
        (
            "name",
            ColumnType::String,
            str_col(fns.iter().map(|f| Some(f.name.clone())).collect()),
        ),
        (
            "visibility",
            ColumnType::String,
            str_col(fns.iter().map(|f| Some(f.visibility.clone())).collect()),
        ),
        (
            "is_async",
            ColumnType::Boolean,
            bool_col(fns.iter().map(|f| Some(f.is_async)).collect()),
        ),
        (
            "is_method",
            ColumnType::Boolean,
            bool_col(fns.iter().map(|f| Some(f.is_method)).collect()),
        ),
        (
            "signature",
            ColumnType::String,
            str_col(fns.iter().map(|f| Some(f.signature.clone())).collect()),
        ),
        (
            "file_path",
            ColumnType::String,
            str_col(fns.iter().map(|f| Some(f.file_path.clone())).collect()),
        ),
        (
            "line_number",
            ColumnType::Int64,
            int_col(fns.iter().map(|f| Some(f.line_number as i64)).collect()),
        ),
        (
            "end_line",
            ColumnType::Int64,
            int_col(fns.iter().map(|f| f.end_line.map(|e| e as i64)).collect()),
        ),
        (
            "docstring",
            ColumnType::String,
            str_col(fns.iter().map(|f| f.docstring.clone()).collect()),
        ),
        (
            "return_type",
            ColumnType::String,
            str_col(fns.iter().map(|f| f.return_type.clone()).collect()),
        ),
        (
            "type_parameters",
            ColumnType::String,
            str_col(fns.iter().map(|f| f.type_parameters.clone()).collect()),
        ),
        (
            "decorators",
            ColumnType::String,
            str_col(
                fns.iter()
                    .map(|f| {
                        if f.decorators.is_empty() {
                            None
                        } else {
                            Some(f.decorators.join(","))
                        }
                    })
                    .collect(),
            ),
        ),
        (
            "is_test",
            ColumnType::Boolean,
            bool_col(
                fns.iter()
                    .map(|f| {
                        let in_test_file = file_is_test
                            .get(f.file_path.as_str())
                            .copied()
                            .unwrap_or(false);
                        Some(meta_bool(f, "is_test") || in_test_file)
                    })
                    .collect(),
            ),
        ),
        (
            "is_pymethod",
            ColumnType::Boolean,
            bool_col(
                fns.iter()
                    .map(|f| Some(meta_bool(f, "is_pymethod")))
                    .collect(),
            ),
        ),
        (
            "is_pymodule",
            ColumnType::Boolean,
            bool_col(
                fns.iter()
                    .map(|f| Some(meta_bool(f, "is_pymodule")))
                    .collect(),
            ),
        ),
        (
            "is_ffi",
            ColumnType::Boolean,
            bool_col(fns.iter().map(|f| Some(meta_bool(f, "is_ffi"))).collect()),
        ),
        (
            "ffi_kind",
            ColumnType::String,
            str_col(
                fns.iter()
                    .map(|f| {
                        f.metadata
                            .get("ffi_kind")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                    })
                    .collect(),
            ),
        ),
        (
            "is_static",
            ColumnType::Boolean,
            bool_col(
                fns.iter()
                    .map(|f| Some(meta_bool(f, "is_static")))
                    .collect(),
            ),
        ),
        (
            "is_abstract",
            ColumnType::Boolean,
            bool_col(
                fns.iter()
                    .map(|f| Some(meta_bool(f, "is_abstract")))
                    .collect(),
            ),
        ),
        (
            "is_property",
            ColumnType::Boolean,
            bool_col(
                fns.iter()
                    .map(|f| Some(meta_bool(f, "is_property")))
                    .collect(),
            ),
        ),
        (
            "is_classmethod",
            ColumnType::Boolean,
            bool_col(
                fns.iter()
                    .map(|f| Some(meta_bool(f, "is_classmethod")))
                    .collect(),
            ),
        ),
        (
            "branch_count",
            ColumnType::Int64,
            int_col(
                fns.iter()
                    .map(|f| f.branch_count.map(|v| v as i64))
                    .collect(),
            ),
        ),
        (
            "param_count",
            ColumnType::Int64,
            int_col(
                fns.iter()
                    .map(|f| f.param_count.map(|v| v as i64))
                    .collect(),
            ),
        ),
        (
            "max_nesting",
            ColumnType::Int64,
            int_col(
                fns.iter()
                    .map(|f| f.max_nesting.map(|v| v as i64))
                    .collect(),
            ),
        ),
        (
            "is_recursive",
            ColumnType::Boolean,
            bool_col(fns.iter().map(|f| f.is_recursive).collect()),
        ),
        (
            "parameters",
            ColumnType::String,
            str_col(
                fns.iter()
                    .map(|f| {
                        if f.parameters.is_empty() {
                            None
                        } else {
                            serde_json::to_string(&f.parameters).ok()
                        }
                    })
                    .collect(),
            ),
        ),
    ])
}

fn classes_df(
    classes: &[ClassInfo],
    attrs_by_owner: &HashMap<String, Vec<&AttributeInfo>>,
    file_to_module: &HashMap<&str, &str>,
) -> DataFrame {
    build_df(vec![
        (
            "qualified_name",
            ColumnType::String,
            str_col(
                classes
                    .iter()
                    .map(|c| Some(c.qualified_name.clone()))
                    .collect(),
            ),
        ),
        // 0.9.30: see functions_df for rationale on the `module` property.
        (
            "module",
            ColumnType::String,
            str_col(
                classes
                    .iter()
                    .map(|c| {
                        file_to_module
                            .get(c.file_path.as_str())
                            .map(|m| (*m).to_string())
                    })
                    .collect(),
            ),
        ),
        (
            "name",
            ColumnType::String,
            str_col(classes.iter().map(|c| Some(c.name.clone())).collect()),
        ),
        (
            "kind",
            ColumnType::String,
            str_col(classes.iter().map(|c| Some(c.kind.clone())).collect()),
        ),
        (
            "visibility",
            ColumnType::String,
            str_col(classes.iter().map(|c| Some(c.visibility.clone())).collect()),
        ),
        (
            "file_path",
            ColumnType::String,
            str_col(classes.iter().map(|c| Some(c.file_path.clone())).collect()),
        ),
        (
            "line_number",
            ColumnType::Int64,
            int_col(classes.iter().map(|c| Some(c.line_number as i64)).collect()),
        ),
        (
            "end_line",
            ColumnType::Int64,
            int_col(
                classes
                    .iter()
                    .map(|c| c.end_line.map(|e| e as i64))
                    .collect(),
            ),
        ),
        (
            "docstring",
            ColumnType::String,
            str_col(classes.iter().map(|c| c.docstring.clone()).collect()),
        ),
        (
            "bases",
            ColumnType::String,
            str_col(
                classes
                    .iter()
                    .map(|c| {
                        if c.bases.is_empty() {
                            None
                        } else {
                            Some(c.bases.join(", "))
                        }
                    })
                    .collect(),
            ),
        ),
        (
            "type_parameters",
            ColumnType::String,
            str_col(classes.iter().map(|c| c.type_parameters.clone()).collect()),
        ),
        (
            "fields",
            ColumnType::String,
            str_col(
                classes
                    .iter()
                    .map(|c| {
                        let entries: Vec<FieldEntry> = attrs_by_owner
                            .get(&c.qualified_name)
                            .map(|v| {
                                v.iter()
                                    .map(|a| FieldEntry {
                                        name: a.name.clone(),
                                        r#type: a.type_annotation.clone(),
                                        visibility: a.visibility.clone(),
                                        default: a.default_value.clone(),
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        if entries.is_empty() {
                            None
                        } else {
                            serde_json::to_string(&entries).ok()
                        }
                    })
                    .collect(),
            ),
        ),
        (
            "is_pyclass",
            ColumnType::Boolean,
            bool_col(
                classes
                    .iter()
                    .map(|c| Some(class_meta_bool(c, "is_pyclass")))
                    .collect(),
            ),
        ),
    ])
}

fn enums_df(enums: &[EnumInfo], file_to_module: &HashMap<&str, &str>) -> DataFrame {
    build_df(vec![
        (
            "qualified_name",
            ColumnType::String,
            str_col(
                enums
                    .iter()
                    .map(|e| Some(e.qualified_name.clone()))
                    .collect(),
            ),
        ),
        // 0.9.30: see functions_df for rationale on the `module` property.
        (
            "module",
            ColumnType::String,
            str_col(
                enums
                    .iter()
                    .map(|e| {
                        file_to_module
                            .get(e.file_path.as_str())
                            .map(|m| (*m).to_string())
                    })
                    .collect(),
            ),
        ),
        (
            "name",
            ColumnType::String,
            str_col(enums.iter().map(|e| Some(e.name.clone())).collect()),
        ),
        (
            "visibility",
            ColumnType::String,
            str_col(enums.iter().map(|e| Some(e.visibility.clone())).collect()),
        ),
        (
            "file_path",
            ColumnType::String,
            str_col(enums.iter().map(|e| Some(e.file_path.clone())).collect()),
        ),
        (
            "line_number",
            ColumnType::Int64,
            int_col(enums.iter().map(|e| Some(e.line_number as i64)).collect()),
        ),
        (
            "end_line",
            ColumnType::Int64,
            int_col(enums.iter().map(|e| e.end_line.map(|x| x as i64)).collect()),
        ),
        (
            "docstring",
            ColumnType::String,
            str_col(enums.iter().map(|e| e.docstring.clone()).collect()),
        ),
        (
            "variants",
            ColumnType::String,
            str_col(
                enums
                    .iter()
                    .map(|e| {
                        if e.variants.is_empty() {
                            None
                        } else {
                            Some(e.variants.join(", "))
                        }
                    })
                    .collect(),
            ),
        ),
    ])
}

fn interfaces_df(ifs: &[InterfaceInfo], file_to_module: &HashMap<&str, &str>) -> DataFrame {
    build_df(vec![
        (
            "qualified_name",
            ColumnType::String,
            str_col(ifs.iter().map(|i| Some(i.qualified_name.clone())).collect()),
        ),
        // 0.9.30: see functions_df for rationale on the `module` property.
        (
            "module",
            ColumnType::String,
            str_col(
                ifs.iter()
                    .map(|i| {
                        file_to_module
                            .get(i.file_path.as_str())
                            .map(|m| (*m).to_string())
                    })
                    .collect(),
            ),
        ),
        (
            "name",
            ColumnType::String,
            str_col(ifs.iter().map(|i| Some(i.name.clone())).collect()),
        ),
        (
            "kind",
            ColumnType::String,
            str_col(ifs.iter().map(|i| Some(i.kind.clone())).collect()),
        ),
        (
            "visibility",
            ColumnType::String,
            str_col(ifs.iter().map(|i| Some(i.visibility.clone())).collect()),
        ),
        (
            "file_path",
            ColumnType::String,
            str_col(ifs.iter().map(|i| Some(i.file_path.clone())).collect()),
        ),
        (
            "line_number",
            ColumnType::Int64,
            int_col(ifs.iter().map(|i| Some(i.line_number as i64)).collect()),
        ),
        (
            "end_line",
            ColumnType::Int64,
            int_col(ifs.iter().map(|i| i.end_line.map(|x| x as i64)).collect()),
        ),
        (
            "docstring",
            ColumnType::String,
            str_col(ifs.iter().map(|i| i.docstring.clone()).collect()),
        ),
        (
            "type_parameters",
            ColumnType::String,
            str_col(ifs.iter().map(|i| i.type_parameters.clone()).collect()),
        ),
    ])
}

fn constants_df(consts: &[ConstantInfo], file_to_module: &HashMap<&str, &str>) -> DataFrame {
    build_df(vec![
        (
            "qualified_name",
            ColumnType::String,
            str_col(
                consts
                    .iter()
                    .map(|c| Some(c.qualified_name.clone()))
                    .collect(),
            ),
        ),
        // 0.9.30: see functions_df for rationale on the `module` property.
        (
            "module",
            ColumnType::String,
            str_col(
                consts
                    .iter()
                    .map(|c| {
                        file_to_module
                            .get(c.file_path.as_str())
                            .map(|m| (*m).to_string())
                    })
                    .collect(),
            ),
        ),
        (
            "name",
            ColumnType::String,
            str_col(consts.iter().map(|c| Some(c.name.clone())).collect()),
        ),
        (
            "kind",
            ColumnType::String,
            str_col(consts.iter().map(|c| Some(c.kind.clone())).collect()),
        ),
        (
            "type_annotation",
            ColumnType::String,
            str_col(consts.iter().map(|c| c.type_annotation.clone()).collect()),
        ),
        (
            "value_preview",
            ColumnType::String,
            str_col(consts.iter().map(|c| c.value_preview.clone()).collect()),
        ),
        (
            "visibility",
            ColumnType::String,
            str_col(consts.iter().map(|c| Some(c.visibility.clone())).collect()),
        ),
        (
            "file_path",
            ColumnType::String,
            str_col(consts.iter().map(|c| Some(c.file_path.clone())).collect()),
        ),
        (
            "line_number",
            ColumnType::Int64,
            int_col(consts.iter().map(|c| Some(c.line_number as i64)).collect()),
        ),
    ])
}

// ── Edge DataFrame builders ─────────────────────────────────────────

fn has_submodule_df(modules: &[ModuleRecord]) -> DataFrame {
    let mut parent: Vec<Option<String>> = Vec::new();
    let mut child: Vec<Option<String>> = Vec::new();
    for m in modules {
        for sep in ["::", ".", "/"] {
            if let Some(idx) = m.qualified_name.rfind(sep) {
                let p = &m.qualified_name[..idx];
                if modules.iter().any(|o| o.qualified_name == p) {
                    parent.push(Some(p.to_string()));
                    child.push(Some(m.qualified_name.clone()));
                    break;
                }
            }
        }
    }
    build_df(vec![
        ("parent", ColumnType::String, str_col(parent)),
        ("child", ColumnType::String, str_col(child)),
    ])
}

fn contains_edges_df(edges: &[super::other_edges::ContainsEdge]) -> DataFrame {
    let parent: Vec<Option<String>> = edges.iter().map(|e| Some(e.parent.clone())).collect();
    let child: Vec<Option<String>> = edges.iter().map(|e| Some(e.child.clone())).collect();
    build_df(vec![
        ("parent", ColumnType::String, str_col(parent)),
        ("child", ColumnType::String, str_col(child)),
    ])
}

fn import_edges_df(edges: &[super::other_edges::ImportEdge]) -> DataFrame {
    let src: Vec<Option<String>> = edges.iter().map(|e| Some(e.file_path.clone())).collect();
    let tgt: Vec<Option<String>> = edges.iter().map(|e| Some(e.module.clone())).collect();
    build_df(vec![
        ("file_path", ColumnType::String, str_col(src)),
        ("module", ColumnType::String, str_col(tgt)),
    ])
}

fn file_import_edges_df(edges: &[super::other_edges::FileImportEdge]) -> DataFrame {
    let src: Vec<Option<String>> = edges.iter().map(|e| Some(e.source.clone())).collect();
    let tgt: Vec<Option<String>> = edges.iter().map(|e| Some(e.target.clone())).collect();
    let count: Vec<Option<i64>> = edges.iter().map(|e| Some(e.import_count)).collect();
    build_df(vec![
        ("source", ColumnType::String, str_col(src)),
        ("target", ColumnType::String, str_col(tgt)),
        ("import_count", ColumnType::Int64, int_col(count)),
    ])
}

fn route_nodes_df(nodes: &[super::routes::RouteNode]) -> DataFrame {
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

fn route_edges_df(edges: &[super::routes::RouteEdge]) -> DataFrame {
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

fn decorates_edges_df(edges: &[super::other_edges::DecoratesEdge]) -> DataFrame {
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

fn call_edges_df(edges: &[super::call_edges::CallEdge]) -> DataFrame {
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

fn implements_edges_df(edges: &[super::type_edges::ImplementsEdge]) -> DataFrame {
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

fn extends_edges_df(edges: &[super::type_edges::ExtendsEdge]) -> DataFrame {
    let child: Vec<Option<String>> = edges.iter().map(|e| Some(e.child_name.clone())).collect();
    let parent: Vec<Option<String>> = edges.iter().map(|e| Some(e.parent_name.clone())).collect();
    build_df(vec![
        ("child_name", ColumnType::String, str_col(child)),
        ("parent_name", ColumnType::String, str_col(parent)),
    ])
}

fn has_method_edges_df(edges: &[super::type_edges::HasMethodEdge]) -> DataFrame {
    let owner: Vec<Option<String>> = edges.iter().map(|e| Some(e.owner.clone())).collect();
    let method: Vec<Option<String>> = edges.iter().map(|e| Some(e.method.clone())).collect();
    build_df(vec![
        ("owner", ColumnType::String, str_col(owner)),
        ("method", ColumnType::String, str_col(method)),
    ])
}

fn uses_type_edges_df(edges: &[super::other_edges::UsesTypeEdge]) -> DataFrame {
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

fn references_edges_df(edges: &[super::other_edges::ReferencesEdge]) -> DataFrame {
    let fns: Vec<Option<String>> = edges.iter().map(|e| Some(e.function.clone())).collect();
    let consts: Vec<Option<String>> = edges.iter().map(|e| Some(e.constant.clone())).collect();
    let lines: Vec<Option<i64>> = edges.iter().map(|e| Some(e.line as i64)).collect();
    build_df(vec![
        ("function", ColumnType::String, str_col(fns)),
        ("constant", ColumnType::String, str_col(consts)),
        ("line", ColumnType::Int64, int_col(lines)),
    ])
}

fn references_fn_edges_df(edges: &[super::other_edges::ReferencesFnEdge]) -> DataFrame {
    let callers: Vec<Option<String>> = edges.iter().map(|e| Some(e.caller.clone())).collect();
    let callees: Vec<Option<String>> = edges.iter().map(|e| Some(e.callee.clone())).collect();
    let lines: Vec<Option<i64>> = edges.iter().map(|e| Some(e.line as i64)).collect();
    build_df(vec![
        ("caller", ColumnType::String, str_col(callers)),
        ("callee", ColumnType::String, str_col(callees)),
        ("line", ColumnType::Int64, int_col(lines)),
    ])
}

fn module_contains_file_df(edges: &[super::other_edges::ModuleContainsFileEdge]) -> DataFrame {
    let m: Vec<Option<String>> = edges.iter().map(|e| Some(e.module.clone())).collect();
    let p: Vec<Option<String>> = edges.iter().map(|e| Some(e.file_path.clone())).collect();
    build_df(vec![
        ("module", ColumnType::String, str_col(m)),
        ("file_path", ColumnType::String, str_col(p)),
    ])
}

fn pyo3_binds_df(edges: &[super::other_edges::PyO3BindsEdge]) -> DataFrame {
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

fn ffi_exposes_df(edges: &[super::other_edges::FfiExposesEdge]) -> DataFrame {
    let m: Vec<Option<String>> = edges.iter().map(|e| Some(e.module_fn.clone())).collect();
    let t: Vec<Option<String>> = edges.iter().map(|e| Some(e.target_qname.clone())).collect();
    let py: Vec<Option<String>> = edges.iter().map(|e| Some(e.py_name.clone())).collect();
    build_df(vec![
        ("module_fn", ColumnType::String, str_col(m)),
        ("target_qname", ColumnType::String, str_col(t)),
        ("py_name", ColumnType::String, str_col(py)),
    ])
}

fn external_nodes_df(nodes: &[super::type_edges::ExternalNode]) -> DataFrame {
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

fn defines_edges(result: &ParseResult) -> Vec<DefinesEdge> {
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
    // File DEFINES Class / Struct / Enum / Interface / Protocol / Trait / Constant
    for c in &result.classes {
        let target_type = match c.kind.as_str() {
            "struct" => "Struct",
            _ => "Class",
        };
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
    out
}

fn defines_edges_df(edges: &[DefinesEdge]) -> HashMap<(String, String), DataFrame> {
    let mut by_pair: HashMap<(String, String), Vec<&DefinesEdge>> = HashMap::new();
    for e in edges {
        by_pair
            .entry((e.source_type.clone(), e.target_type.clone()))
            .or_default()
            .push(e);
    }
    by_pair
        .into_iter()
        .map(|(pair, list)| {
            let src: Vec<Option<String>> = list.iter().map(|e| Some(e.source_id.clone())).collect();
            let tgt: Vec<Option<String>> = list.iter().map(|e| Some(e.target_id.clone())).collect();
            let df = build_df(vec![
                ("source", ColumnType::String, str_col(src)),
                ("target", ColumnType::String, str_col(tgt)),
            ]);
            (pair, df)
        })
        .collect()
}

// ── Entry point ────────────────────────────────────────────────────

pub fn load_into_graph(
    result: &ParseResult,
    project_info: Option<&ProjectInfo>,
) -> Result<KnowledgeGraph, String> {
    let verbose = std::env::var_os("KGLITE_CODE_TREE_VERBOSE").is_some();
    let mark = |t: std::time::Instant, label: &str| {
        if verbose {
            eprintln!("[timing]   {}: {:.3}s", label, t.elapsed().as_secs_f64());
        }
    };
    let mut kg = KnowledgeGraph::new_empty();
    let graph = get_graph_mut(&mut kg.inner);
    let t_start = std::time::Instant::now();

    // ── Project / Dependency / HAS_SOURCE (from manifest) ──────────────
    if let Some(info) = project_info {
        let df = build_df(vec![
            (
                "name",
                ColumnType::String,
                str_col(vec![Some(info.name.clone())]),
            ),
            (
                "version",
                ColumnType::String,
                str_col(vec![info.version.clone()]),
            ),
            (
                "description",
                ColumnType::String,
                str_col(vec![info.description.clone()]),
            ),
            (
                "languages",
                ColumnType::String,
                str_col(vec![if info.languages.is_empty() {
                    None
                } else {
                    Some(info.languages.join(", "))
                }]),
            ),
            (
                "authors",
                ColumnType::String,
                str_col(vec![if info.authors.is_empty() {
                    None
                } else {
                    Some(info.authors.join(", "))
                }]),
            ),
            (
                "license",
                ColumnType::String,
                str_col(vec![info.license.clone()]),
            ),
            (
                "repository",
                ColumnType::String,
                str_col(vec![info.repository_url.clone()]),
            ),
            (
                "build_system",
                ColumnType::String,
                str_col(vec![info.build_system.clone()]),
            ),
            (
                "crate_type",
                ColumnType::String,
                str_col(vec![info.metadata.get("crate_type").and_then(|v| {
                    v.as_array().map(|arr| {
                        arr.iter()
                            .filter_map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                })]),
            ),
            (
                "manifest",
                ColumnType::String,
                str_col(vec![Some(info.manifest_path.clone())]),
            ),
        ]);
        maintain::add_nodes(
            graph,
            df,
            "Project".into(),
            "name".into(),
            Some("name".into()),
            None,
        )
        .map_err(py_err)?;

        if !info.dependencies.is_empty() {
            let dep_ids: Vec<Option<String>> = info
                .dependencies
                .iter()
                .map(|d| {
                    Some(match &d.group {
                        Some(g) => format!("{}::{}", d.name, g),
                        None => d.name.clone(),
                    })
                })
                .collect();
            let names: Vec<Option<String>> = info
                .dependencies
                .iter()
                .map(|d| Some(d.name.clone()))
                .collect();
            let specs: Vec<Option<String>> = info
                .dependencies
                .iter()
                .map(|d| d.version_spec.clone())
                .collect();
            let is_dev: Vec<Option<bool>> = info
                .dependencies
                .iter()
                .map(|d| if d.is_dev { Some(true) } else { None })
                .collect();
            let is_optional: Vec<Option<bool>> = info
                .dependencies
                .iter()
                .map(|d| if d.is_optional { Some(true) } else { None })
                .collect();
            let groups: Vec<Option<String>> =
                info.dependencies.iter().map(|d| d.group.clone()).collect();
            let df = build_df(vec![
                ("dep_id", ColumnType::String, str_col(dep_ids.clone())),
                ("name", ColumnType::String, str_col(names)),
                ("version_spec", ColumnType::String, str_col(specs)),
                ("is_dev", ColumnType::Boolean, bool_col(is_dev)),
                ("is_optional", ColumnType::Boolean, bool_col(is_optional)),
                ("group", ColumnType::String, str_col(groups)),
            ]);
            maintain::add_nodes(
                graph,
                df,
                "Dependency".into(),
                "dep_id".into(),
                Some("name".into()),
                None,
            )
            .map_err(py_err)?;
        }
    }

    let modules = build_modules(&result.files);
    let known_modules: std::collections::HashSet<String> =
        modules.iter().map(|m| m.qualified_name.clone()).collect();

    // Attribute lookup for JSON embedding.
    let mut attrs_by_owner: HashMap<String, Vec<&AttributeInfo>> = HashMap::new();
    for a in &result.attributes {
        attrs_by_owner
            .entry(a.owner_qualified_name.clone())
            .or_default()
            .push(a);
    }

    mark(t_start, "setup+project/deps");
    let t_nodes = std::time::Instant::now();
    // 0.9.30: file_path → module_path lookup shared by every entity
    // df builder so Function/Class/Constant/Enum/Interface/Trait/
    // Protocol/Struct all carry a `module` property derived from the
    // file they live in. Closes the operator-reported friction where
    // `WHERE f.module STARTS WITH '...'` silently returned 0 rows on
    // non-File node types pre-0.9.30.
    let file_to_module: HashMap<&str, &str> = result
        .files
        .iter()
        .map(|f| (f.path.as_str(), f.module_path.as_str()))
        .collect();
    // ── Node insertions ─────────────────────────────────────────
    if !result.files.is_empty() {
        maintain::add_nodes(
            graph,
            files_df(&result.files),
            "File".into(),
            "path".into(),
            Some("filename".into()),
            None,
        )
        .map_err(py_err)?;
    }
    if !modules.is_empty() {
        maintain::add_nodes(
            graph,
            modules_df(&modules),
            "Module".into(),
            "qualified_name".into(),
            Some("name".into()),
            None,
        )
        .map_err(py_err)?;
    }
    if !result.functions.is_empty() {
        let file_is_test: HashMap<&str, bool> = result
            .files
            .iter()
            .map(|f| (f.path.as_str(), f.is_test))
            .collect();
        maintain::add_nodes(
            graph,
            functions_df(&result.functions, &file_is_test, &file_to_module),
            "Function".into(),
            "qualified_name".into(),
            Some("name".into()),
            None,
        )
        .map_err(py_err)?;
    }
    // Separate struct vs class
    let (structs, classes): (Vec<_>, Vec<_>) =
        result.classes.iter().partition(|c| c.kind == "struct");
    if !structs.is_empty() {
        let structs_owned: Vec<ClassInfo> = structs.into_iter().cloned().collect();
        maintain::add_nodes(
            graph,
            classes_df(&structs_owned, &attrs_by_owner, &file_to_module),
            "Struct".into(),
            "qualified_name".into(),
            Some("name".into()),
            None,
        )
        .map_err(py_err)?;
    }
    if !classes.is_empty() {
        let classes_owned: Vec<ClassInfo> = classes.into_iter().cloned().collect();
        maintain::add_nodes(
            graph,
            classes_df(&classes_owned, &attrs_by_owner, &file_to_module),
            "Class".into(),
            "qualified_name".into(),
            Some("name".into()),
            None,
        )
        .map_err(py_err)?;
    }
    if !result.enums.is_empty() {
        maintain::add_nodes(
            graph,
            enums_df(&result.enums, &file_to_module),
            "Enum".into(),
            "qualified_name".into(),
            Some("name".into()),
            None,
        )
        .map_err(py_err)?;
    }
    // Split interfaces by kind
    let (traits, others): (Vec<_>, Vec<_>) =
        result.interfaces.iter().partition(|i| i.kind == "trait");
    let (protocols, ifaces): (Vec<_>, Vec<_>) =
        others.into_iter().partition(|i| i.kind == "protocol");
    if !traits.is_empty() {
        let v: Vec<InterfaceInfo> = traits.into_iter().cloned().collect();
        maintain::add_nodes(
            graph,
            interfaces_df(&v, &file_to_module),
            "Trait".into(),
            "qualified_name".into(),
            Some("name".into()),
            None,
        )
        .map_err(py_err)?;
    }
    if !protocols.is_empty() {
        let v: Vec<InterfaceInfo> = protocols.into_iter().cloned().collect();
        maintain::add_nodes(
            graph,
            interfaces_df(&v, &file_to_module),
            "Protocol".into(),
            "qualified_name".into(),
            Some("name".into()),
            None,
        )
        .map_err(py_err)?;
    }
    if !ifaces.is_empty() {
        let v: Vec<InterfaceInfo> = ifaces.into_iter().cloned().collect();
        maintain::add_nodes(
            graph,
            interfaces_df(&v, &file_to_module),
            "Interface".into(),
            "qualified_name".into(),
            Some("name".into()),
            None,
        )
        .map_err(py_err)?;
    }
    if !result.constants.is_empty() {
        maintain::add_nodes(
            graph,
            constants_df(&result.constants, &file_to_module),
            "Constant".into(),
            "qualified_name".into(),
            Some("name".into()),
            None,
        )
        .map_err(py_err)?;
    }

    mark(t_nodes, "nodes");
    let t_typeedges = std::time::Instant::now();
    // ── Type-relationship-derived external stubs (before HAS_METHOD/IMPLEMENTS use them) ─
    // `name_to_qname` is the legacy first-match lookup, retained for the
    // HAS_METHOD owner-resolution helpers; `build_type_edges` now also
    // consults file-level imports and per-type kind, so interface targets
    // route to Interface nodes even when an unrelated class shares the name.
    let mut name_to_qname: HashMap<String, String> = HashMap::new();
    for c in &result.classes {
        name_to_qname.insert(c.name.clone(), c.qualified_name.clone());
    }
    for i in &result.interfaces {
        name_to_qname.insert(i.name.clone(), i.qualified_name.clone());
    }
    for e in &result.enums {
        name_to_qname.insert(e.name.clone(), e.qualified_name.clone());
    }

    let type_out = super::type_edges::build_type_edges(
        &result.type_relationships,
        &result.files,
        &result.classes,
        &result.interfaces,
        &mut name_to_qname,
    );

    // External trait stubs — only if Trait type was already registered by a
    // parsed interface above. Otherwise silently drop them (same behaviour as
    // Python when the schema doesn't have the target node type).
    if !type_out.external_traits.is_empty() && graph.has_node_type("Trait") {
        maintain::add_nodes(
            graph,
            external_nodes_df(&type_out.external_traits),
            "Trait".into(),
            "qualified_name".into(),
            Some("name".into()),
            Some("skip".into()),
        )
        .map_err(py_err)?;
    }
    if !type_out.external_classes.is_empty() {
        let target = if graph.has_node_type("Class") {
            Some("Class")
        } else if graph.has_node_type("Struct") {
            Some("Struct")
        } else {
            None
        };
        if let Some(target) = target {
            maintain::add_nodes(
                graph,
                external_nodes_df(&type_out.external_classes),
                target.into(),
                "qualified_name".into(),
                Some("name".into()),
                Some("skip".into()),
            )
            .map_err(py_err)?;
        }
    }

    mark(t_typeedges, "type_edges build+external stubs");
    let t_routes = std::time::Instant::now();
    // Route nodes + HANDLES edges — web-framework URL endpoints
    // synthesized from decorators and urls.py constants. Per-framework
    // detectors live under `builder/routes/`. Adding a new framework is
    // one new file in that directory plus a line in routes/mod.rs.
    let (route_nodes, route_edges) =
        super::routes::build_routes(&result.functions, &result.constants);
    if !route_nodes.is_empty() {
        maintain::add_nodes(
            graph,
            route_nodes_df(&route_nodes),
            "Route".into(),
            "id".into(),
            Some("name".into()),
            None,
        )
        .map_err(py_err)?;
    }
    if !route_edges.is_empty() {
        maintain::add_connections(
            graph,
            route_edges_df(&route_edges),
            "HANDLES".into(),
            "Route".into(),
            "route_id".into(),
            "Function".into(),
            "function_qname".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }

    mark(t_routes, "routes");
    let t_edges = std::time::Instant::now();
    // ── Edge insertions ─────────────────────────────────────────
    // Module HAS_SUBMODULE Module — built from submodule declarations.
    // (Python uses the same "contains" source as HAS_SUBMODULE; no separate
    // CONTAINS edge type is emitted.)
    let contains = super::other_edges::build_contains_edges(&result.files);
    if !contains.is_empty() {
        maintain::add_connections(
            graph,
            contains_edges_df(&contains),
            "HAS_SUBMODULE".into(),
            "Module".into(),
            "parent".into(),
            "Module".into(),
            "child".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }

    // Module HAS_FILE File — closes the natural top-down walk from a Module
    // to the source files (and their Functions/Classes). (Edge name avoids
    // `CONTAINS`, which is a reserved Cypher keyword for substring matching.)
    let mod_contains_file = super::other_edges::build_module_contains_file_edges(&result.files);
    if !mod_contains_file.is_empty() {
        maintain::add_connections(
            graph,
            module_contains_file_df(&mod_contains_file),
            "HAS_FILE".into(),
            "Module".into(),
            "module".into(),
            "File".into(),
            "file_path".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }

    // File DEFINES *
    let defines = defines_edges(result);
    for ((src_type, tgt_type), df) in defines_edges_df(&defines) {
        if df.row_count() == 0 {
            continue;
        }
        maintain::add_connections(
            graph,
            df,
            "DEFINES".into(),
            src_type,
            "source".into(),
            tgt_type,
            "target".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }

    // File IMPORTS Module (only edges to known modules).
    let imports = super::other_edges::build_import_edges(&result.files, &known_modules);
    if !imports.is_empty() {
        maintain::add_connections(
            graph,
            import_edges_df(&imports),
            "IMPORTS".into(),
            "File".into(),
            "file_path".into(),
            "Module".into(),
            "module".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }

    // File IMPORTS File — direct file-level dependency edges, resolved via
    // the project's `module_path → file_path` reverse index. Sibling to the
    // File → Module IMPORTS edge above; both ship per build.
    let module_to_file: HashMap<String, String> = result
        .files
        .iter()
        .filter(|f| !f.module_path.is_empty())
        .map(|f| (f.module_path.clone(), f.path.clone()))
        .collect();
    let file_imports = super::other_edges::build_file_import_edges(&result.files, &module_to_file);
    if !file_imports.is_empty() {
        maintain::add_connections(
            graph,
            file_import_edges_df(&file_imports),
            "IMPORTS".into(),
            "File".into(),
            "source".into(),
            "File".into(),
            "target".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }

    // Project DEPENDS_ON Dependency + Project HAS_SOURCE File (manifest).
    if let Some(info) = project_info {
        if !info.dependencies.is_empty() {
            let proj: Vec<Option<String>> = info
                .dependencies
                .iter()
                .map(|_| Some(info.name.clone()))
                .collect();
            let dep_ids: Vec<Option<String>> = info
                .dependencies
                .iter()
                .map(|d| {
                    Some(match &d.group {
                        Some(g) => format!("{}::{}", d.name, g),
                        None => d.name.clone(),
                    })
                })
                .collect();
            let df = build_df(vec![
                ("project", ColumnType::String, str_col(proj)),
                ("dep_id", ColumnType::String, str_col(dep_ids)),
            ]);
            maintain::add_connections(
                graph,
                df,
                "DEPENDS_ON".into(),
                "Project".into(),
                "project".into(),
                "Dependency".into(),
                "dep_id".into(),
                None,
                None,
                None,
            )
            .map_err(py_err)?;
        }
        if !result.files.is_empty() {
            let proj: Vec<Option<String>> = result
                .files
                .iter()
                .map(|_| Some(info.name.clone()))
                .collect();
            let files: Vec<Option<String>> =
                result.files.iter().map(|f| Some(f.path.clone())).collect();
            let df = build_df(vec![
                ("project", ColumnType::String, str_col(proj)),
                ("file", ColumnType::String, str_col(files)),
            ]);
            maintain::add_connections(
                graph,
                df,
                "HAS_SOURCE".into(),
                "Project".into(),
                "project".into(),
                "File".into(),
                "file".into(),
                None,
                None,
                None,
            )
            .map_err(py_err)?;
        }
    }

    mark(
        t_edges,
        "edges: submodule+contains+defines+imports+depends+hasrc",
    );
    let t_calls = std::time::Instant::now();
    // Function CALLS Function (5-tier resolution).
    // Union noise names from every parser that contributed (language detection
    // by qualified_name separator would be stricter, but the Python impl
    // merges them all into one frozen set too).
    let mut noise: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for name in super::super::parsers::python::PYTHON_NOISE_NAMES {
        noise.insert(*name);
    }
    for name in super::super::parsers::rust_lang::RUST_NOISE_NAMES {
        noise.insert(*name);
    }
    for name in super::super::parsers::typescript::JSTS_NOISE_NAMES {
        noise.insert(*name);
    }
    for name in super::super::parsers::go::GO_NOISE_NAMES {
        noise.insert(*name);
    }
    for name in super::super::parsers::java::JAVA_NOISE_NAMES {
        noise.insert(*name);
    }
    for name in super::super::parsers::csharp::CSHARP_NOISE_NAMES {
        noise.insert(*name);
    }
    for name in super::super::parsers::cpp::CPP_NOISE_NAMES {
        noise.insert(*name);
    }
    let call_edges =
        super::call_edges::build_call_edges(&result.functions, &result.files, &noise, 5);
    if !call_edges.is_empty() {
        maintain::add_connections(
            graph,
            call_edges_df(&call_edges),
            "CALLS".into(),
            "Function".into(),
            "caller".into(),
            "Function".into(),
            "callee".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }

    mark(t_calls, "calls");
    let t_iface = std::time::Instant::now();
    // IMPLEMENTS / EXTENDS / HAS_METHOD — source/target types are picked from
    // whatever is registered in the graph schema. Python uses the same default
    // chain (Class → Struct → Trait → Interface → Protocol).
    // Snapshot the relevant schema checks up front so the later mutable
    // borrows of `graph` (during add_connections) don't conflict.
    let has_class = graph.has_node_type("Class");
    let has_struct = graph.has_node_type("Struct");
    let has_trait = graph.has_node_type("Trait");
    let has_protocol = graph.has_node_type("Protocol");
    let has_interface = graph.has_node_type("Interface");

    let pick = |defaults: &[(&'static str, bool)]| -> Option<&'static str> {
        defaults.iter().find(|(_, exists)| *exists).map(|(n, _)| *n)
    };

    if !type_out.implements.is_empty() {
        // Route IMPLEMENTS per-row based on the resolved source/target types.
        // Python's _add_typed_connections does the equivalent via name_to_qname.
        let mut qname_to_type: HashMap<String, &'static str> = HashMap::new();
        for c in &result.classes {
            let nt = if c.kind == "struct" {
                "Struct"
            } else {
                "Class"
            };
            qname_to_type.insert(c.qualified_name.clone(), nt);
            qname_to_type.insert(c.name.clone(), nt);
        }
        for e in &result.enums {
            qname_to_type.insert(e.qualified_name.clone(), "Enum");
            qname_to_type.insert(e.name.clone(), "Enum");
        }
        for i in &result.interfaces {
            let nt = match i.kind.as_str() {
                "trait" => "Trait",
                "protocol" => "Protocol",
                _ => "Interface",
            };
            qname_to_type.insert(i.qualified_name.clone(), nt);
            qname_to_type.insert(i.name.clone(), nt);
        }
        // External stubs we just inserted: traits → Trait, classes → Class/Struct.
        let ext_trait_type = if graph.has_node_type("Trait") {
            Some("Trait")
        } else if graph.has_node_type("Protocol") {
            Some("Protocol")
        } else if graph.has_node_type("Interface") {
            Some("Interface")
        } else {
            None
        };
        if let Some(nt) = ext_trait_type {
            for ext in &type_out.external_traits {
                qname_to_type.insert(ext.qualified_name.clone(), nt);
                qname_to_type.insert(ext.name.clone(), nt);
            }
        }
        let ext_class_type = if graph.has_node_type("Class") {
            Some("Class")
        } else if graph.has_node_type("Struct") {
            Some("Struct")
        } else {
            None
        };
        if let Some(nt) = ext_class_type {
            for ext in &type_out.external_classes {
                qname_to_type.insert(ext.qualified_name.clone(), nt);
                qname_to_type.insert(ext.name.clone(), nt);
            }
        }

        let default_src = pick(&[("Class", has_class), ("Struct", has_struct)]).unwrap_or("Class");
        let default_tgt = pick(&[
            ("Protocol", has_protocol),
            ("Trait", has_trait),
            ("Interface", has_interface),
        ])
        .unwrap_or("Protocol");

        let mut by_pair: BTreeMap<
            (&'static str, &'static str),
            Vec<&super::type_edges::ImplementsEdge>,
        > = BTreeMap::new();
        for edge in &type_out.implements {
            let src = qname_to_type
                .get(&edge.type_name)
                .copied()
                .unwrap_or(default_src);
            let tgt = qname_to_type
                .get(&edge.interface_name)
                .copied()
                .unwrap_or(default_tgt);
            by_pair.entry((src, tgt)).or_default().push(edge);
        }

        for ((src, tgt), edges) in by_pair {
            if !graph.has_node_type(src) || !graph.has_node_type(tgt) {
                continue;
            }
            let owned: Vec<super::type_edges::ImplementsEdge> = edges
                .iter()
                .map(|e| super::type_edges::ImplementsEdge {
                    type_name: e.type_name.clone(),
                    interface_name: e.interface_name.clone(),
                })
                .collect();
            maintain::add_connections(
                graph,
                implements_edges_df(&owned),
                "IMPLEMENTS".into(),
                src.into(),
                "type_name".into(),
                tgt.into(),
                "interface_name".into(),
                None,
                None,
                None,
            )
            .map_err(py_err)?;
        }
    }
    if !type_out.extends.is_empty() {
        let src = pick(&[("Class", has_class), ("Struct", has_struct)]);
        if let Some(src) = src {
            maintain::add_connections(
                graph,
                extends_edges_df(&type_out.extends),
                "EXTENDS".into(),
                src.into(),
                "child_name".into(),
                src.into(),
                "parent_name".into(),
                None,
                None,
                None,
            )
            .map_err(py_err)?;
        }
    }
    if !type_out.has_method.is_empty() {
        // Build qualified_name → node_type for every parsed owner type.
        let mut qname_to_type: HashMap<String, &'static str> = HashMap::new();
        for c in &result.classes {
            let nt = if c.kind == "struct" {
                "Struct"
            } else {
                "Class"
            };
            qname_to_type.insert(c.qualified_name.clone(), nt);
        }
        for i in &result.interfaces {
            let nt = match i.kind.as_str() {
                "trait" => "Trait",
                "protocol" => "Protocol",
                _ => "Interface",
            };
            qname_to_type.insert(i.qualified_name.clone(), nt);
        }
        for e in &result.enums {
            qname_to_type.insert(e.qualified_name.clone(), "Enum");
        }

        let default_src = pick(&[
            ("Class", has_class),
            ("Struct", has_struct),
            ("Trait", has_trait),
            ("Interface", has_interface),
            ("Protocol", has_protocol),
        ]);

        // Group edges by inferred source type (owner's node type).
        let mut by_src: BTreeMap<&'static str, Vec<&super::type_edges::HasMethodEdge>> =
            BTreeMap::new();
        for edge in &type_out.has_method {
            let src = qname_to_type
                .get(&edge.owner)
                .copied()
                .unwrap_or(default_src.unwrap_or("Class"));
            by_src.entry(src).or_default().push(edge);
        }

        for (src, edges) in by_src {
            if !graph.has_node_type(src) {
                continue;
            }
            let owned: Vec<super::type_edges::HasMethodEdge> = edges
                .iter()
                .map(|e| super::type_edges::HasMethodEdge {
                    owner: e.owner.clone(),
                    method: e.method.clone(),
                })
                .collect();
            maintain::add_connections(
                graph,
                has_method_edges_df(&owned),
                "HAS_METHOD".into(),
                src.into(),
                "owner".into(),
                "Function".into(),
                "method".into(),
                None,
                None,
                None,
            )
            .map_err(py_err)?;
        }
    }

    mark(t_iface, "implements+extends+has_method");
    let t_uses = std::time::Instant::now();
    // USES_TYPE (one edge batch per target node type).
    let uses_type = super::other_edges::build_uses_type_edges(
        &result.functions,
        &result.classes,
        &result.enums,
        &result.interfaces,
    );
    for (target_type, edges) in uses_type {
        if edges.is_empty() {
            continue;
        }
        maintain::add_connections(
            graph,
            uses_type_edges_df(&edges),
            "USES_TYPE".into(),
            "Function".into(),
            "function".into(),
            target_type.into(),
            "type_name".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }

    mark(t_uses, "uses_type");
    let t_refs = std::time::Instant::now();
    // REFERENCES (Function → Constant) — name-keyed identifier resolution.
    let refs = super::other_edges::build_references_edges(&result.functions, &result.constants);
    if !refs.is_empty() {
        maintain::add_connections(
            graph,
            references_edges_df(&refs),
            "REFERENCES".into(),
            "Function".into(),
            "function".into(),
            "Constant".into(),
            "constant".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }

    mark(t_refs, "references");
    let t_refs_fn = std::time::Instant::now();
    // REFERENCES_FN (Function → Function) — bare-identifier function
    // pointers passed to higher-order calls.
    let refs_fn = super::other_edges::build_references_fn_edges(&result.functions);
    if !refs_fn.is_empty() {
        maintain::add_connections(
            graph,
            references_fn_edges_df(&refs_fn),
            "REFERENCES_FN".into(),
            "Function".into(),
            "caller".into(),
            "Function".into(),
            "callee".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }

    mark(t_refs_fn, "references_fn");
    let t_decorates = std::time::Instant::now();
    // Function DECORATES Function — resolve `FunctionInfo.decorators` strings
    // against the project's function set. Skips unresolved (third-party)
    // decorators silently; the `decorator_name` property keeps the raw
    // source literal.
    let decorates = super::other_edges::build_decorates_edges(&result.functions);
    if !decorates.is_empty() {
        maintain::add_connections(
            graph,
            decorates_edges_df(&decorates),
            "DECORATES".into(),
            "Function".into(),
            "decorator".into(),
            "Function".into(),
            "function".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }

    mark(t_decorates, "decorates");
    let t_ffi = std::time::Instant::now();
    // FFI EXPOSES.
    let ffi = super::other_edges::build_ffi_exposes_edges(&result.functions, &result.classes);
    if !ffi.is_empty() {
        // Batch by target_type.
        let (structs, fns): (Vec<_>, Vec<_>) = ffi.iter().partition(|e| e.target_type == "Struct");
        if !structs.is_empty() {
            let v: Vec<_> = structs.into_iter().cloned().collect();
            maintain::add_connections(
                graph,
                ffi_exposes_df(&v),
                "EXPOSES".into(),
                "Function".into(),
                "module_fn".into(),
                "Struct".into(),
                "target_qname".into(),
                None,
                None,
                None,
            )
            .map_err(py_err)?;
        }
        if !fns.is_empty() {
            let v: Vec<_> = fns.into_iter().cloned().collect();
            maintain::add_connections(
                graph,
                ffi_exposes_df(&v),
                "EXPOSES".into(),
                "Function".into(),
                "module_fn".into(),
                "Function".into(),
                "target_qname".into(),
                None,
                None,
                None,
            )
            .map_err(py_err)?;
        }
    }

    mark(t_ffi, "ffi_exposes");
    let t_binds = std::time::Instant::now();
    // Function -[BINDS]-> Function — Python wrapper → underlying Rust pymethod.
    let binds = super::other_edges::build_pyo3_binds_edges(&result.functions);
    if !binds.is_empty() {
        maintain::add_connections(
            graph,
            pyo3_binds_df(&binds),
            "BINDS".into(),
            "Function".into(),
            "py_function".into(),
            "Function".into(),
            "rust_function".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }
    mark(t_binds, "pyo3_binds");
    let t_proc = std::time::Instant::now();
    // Procedure nodes — synthesized from `@procedure: NAME` doc-comment
    // annotations on Function nodes. Each annotated function emits one
    // Procedure node and a `Procedure -[IMPLEMENTED_BY]-> Function` edge.
    let proc_pairs: Vec<(String, String)> = result
        .functions
        .iter()
        .flat_map(|f| {
            f.procedure_names
                .iter()
                .map(move |n| (n.clone(), f.qualified_name.clone()))
        })
        .collect();
    if !proc_pairs.is_empty() {
        // Dedup procedure names — multiple functions implementing the same
        // procedure name keep separate edges to that single Procedure node.
        let mut proc_names: Vec<String> = proc_pairs.iter().map(|(n, _)| n.clone()).collect();
        proc_names.sort();
        proc_names.dedup();
        let proc_df = build_df(vec![
            (
                "name",
                ColumnType::String,
                str_col(proc_names.iter().map(|n| Some(n.clone())).collect()),
            ),
            (
                "qualified_name",
                ColumnType::String,
                str_col(proc_names.iter().map(|n| Some(n.clone())).collect()),
            ),
        ]);
        maintain::add_nodes(
            graph,
            proc_df,
            "Procedure".into(),
            "qualified_name".into(),
            Some("name".into()),
            None,
        )
        .map_err(py_err)?;

        let edge_df = build_df(vec![
            (
                "procedure",
                ColumnType::String,
                str_col(proc_pairs.iter().map(|(n, _)| Some(n.clone())).collect()),
            ),
            (
                "function",
                ColumnType::String,
                str_col(proc_pairs.iter().map(|(_, q)| Some(q.clone())).collect()),
            ),
        ]);
        maintain::add_connections(
            graph,
            edge_df,
            "IMPLEMENTED_BY".into(),
            "Procedure".into(),
            "procedure".into(),
            "Function".into(),
            "function".into(),
            None,
            None,
            None,
        )
        .map_err(py_err)?;
    }
    mark(t_proc, "procedures");
    Ok(kg)
}

impl Clone for super::other_edges::FfiExposesEdge {
    fn clone(&self) -> Self {
        Self {
            module_fn: self.module_fn.clone(),
            target_qname: self.target_qname.clone(),
            target_type: self.target_type,
            py_name: self.py_name.clone(),
        }
    }
}
