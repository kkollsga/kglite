//! DataFrame builders for code-tree entity nodes.

use super::ModuleRecord;
use crate::code_tree::models::{
    AttributeInfo, ClassInfo, ConstantInfo, EnumInfo, FieldEntry, FileInfo, FunctionInfo,
    InterfaceInfo,
};
use crate::datatypes::values::{ColumnData, ColumnType, DataFrame};
use std::collections::HashMap;

/// Add a pre-built `ColumnData::*` to an existing DataFrame.
pub(super) fn add_typed_col(df: &mut DataFrame, name: &str, ct: ColumnType, data: ColumnData) {
    df.add_column(name.to_string(), ct, data)
        .unwrap_or_else(|e| panic!("add_column({name}) failed: {e}"));
}

/// Convenience: produce a fresh DataFrame with the given (name, type) columns
/// whose data is the corresponding `ColumnData::*` vec.
pub(super) fn build_df(cols: Vec<(&str, ColumnType, ColumnData)>) -> DataFrame {
    let mut out = DataFrame::new(Vec::new());
    for (name, ct, data) in cols {
        add_typed_col(&mut out, name, ct, data);
    }
    out
}

pub(super) fn str_col(values: Vec<Option<String>>) -> ColumnData {
    ColumnData::String(values)
}
pub(super) fn int_col(values: Vec<Option<i64>>) -> ColumnData {
    ColumnData::Int64(values)
}
pub(super) fn bool_col(values: Vec<Option<bool>>) -> ColumnData {
    ColumnData::Boolean(values)
}

pub(super) fn py_err<S: Into<String>>(msg: S) -> String {
    msg.into()
}

/// Read a boolean flag from `FunctionInfo.metadata` with a `false` default.
/// Used to promote per-language metadata flags into typed DataFrame columns.
pub(super) fn meta_bool(f: &FunctionInfo, key: &str) -> bool {
    f.metadata
        .get(key)
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Same, but for ClassInfo.
pub(super) fn class_meta_bool(c: &ClassInfo, key: &str) -> bool {
    c.metadata
        .get(key)
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Path-based benchmark-suite provenance — the benchmark analogue of the
/// `is_test` heuristic. Directory-based (not filename-based) because benchmark
/// runners reuse `test_`-style function names: ASV's `asv_bench/` and the
/// conventional `benchmarks/` / `bench/` directories are the reliable signal.
/// Computed from the file path so every node type (File / Module / Function /
/// Class) tags consistently without threading a flag through every parser.
pub(super) fn path_is_benchmark(path: &str) -> bool {
    let p = path.replace('\\', "/");
    let seg = |needle: &str| p.starts_with(needle) || p.contains(&format!("/{needle}"));
    seg("asv_bench/") || seg("benchmarks/") || seg("bench/")
}

/// A file whose contents were skipped as machine-produced (generated or
/// minified). `too_large` is a size cap, not a provenance signal, so it does
/// not count as generated.
pub(super) fn skip_reason_is_generated(skip_reason: Option<&str>) -> bool {
    matches!(skip_reason, Some("generated") | Some("minified"))
}

// ── Entity → DataFrame builders ─────────────────────────────────────

pub(super) fn files_df(files: &[FileInfo]) -> DataFrame {
    let path = files.iter().map(|f| Some(f.path.clone())).collect();
    let filename = files.iter().map(|f| Some(f.filename.clone())).collect();
    let loc = files.iter().map(|f| Some(f.loc as i64)).collect();
    let module_path = files.iter().map(|f| Some(f.module_path.clone())).collect();
    let language = files.iter().map(|f| Some(f.language.clone())).collect();
    let is_test = files.iter().map(|f| Some(f.is_test)).collect();
    let is_benchmark = files
        .iter()
        .map(|f| Some(path_is_benchmark(&f.path)))
        .collect();
    let is_generated = files
        .iter()
        .map(|f| Some(skip_reason_is_generated(f.skip_reason.as_deref())))
        .collect();
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
        ("is_benchmark", ColumnType::Boolean, bool_col(is_benchmark)),
        ("is_generated", ColumnType::Boolean, bool_col(is_generated)),
        ("annotations", ColumnType::String, str_col(annotations)),
        ("skip_reason", ColumnType::String, str_col(skip_reason)),
    ])
}

pub(super) fn modules_df(modules: &[ModuleRecord]) -> DataFrame {
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
        (
            "is_benchmark",
            ColumnType::Boolean,
            bool_col(modules.iter().map(|m| Some(m.is_benchmark)).collect()),
        ),
    ])
}

pub(super) fn functions_df(
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
            "is_benchmark",
            ColumnType::Boolean,
            bool_col(
                fns.iter()
                    .map(|f| Some(path_is_benchmark(&f.file_path)))
                    .collect(),
            ),
        ),
        // Every Function is in-repo (parsed from a source file) — there is no
        // external Function stub the way Class/Struct get `is_external = true`
        // bases merged in. Emit `false` explicitly so the documented
        // `WHERE n.is_external = false` library-only filter works uniformly on
        // Function as it does on Class/File (A1a, operator feedback 2026-06-17).
        (
            "is_external",
            ColumnType::Boolean,
            bool_col(fns.iter().map(|_| Some(false)).collect()),
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
        // Dart callable metadata (set by the Dart parser; false/absent
        // for other languages).
        (
            "is_constructor",
            ColumnType::Boolean,
            bool_col(
                fns.iter()
                    .map(|f| Some(meta_bool(f, "is_constructor")))
                    .collect(),
            ),
        ),
        (
            "is_factory",
            ColumnType::Boolean,
            bool_col(
                fns.iter()
                    .map(|f| Some(meta_bool(f, "is_factory")))
                    .collect(),
            ),
        ),
        (
            "flutter_build",
            ColumnType::Boolean,
            bool_col(
                fns.iter()
                    .map(|f| Some(meta_bool(f, "flutter_build")))
                    .collect(),
            ),
        ),
        (
            "accessor",
            ColumnType::String,
            str_col(
                fns.iter()
                    .map(|f| {
                        f.metadata
                            .get("accessor")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                    })
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

pub(super) fn classes_df(
    classes: &[ClassInfo],
    attrs_by_owner: &HashMap<String, Vec<&AttributeInfo>>,
    file_to_module: &HashMap<&str, &str>,
    file_is_test: &HashMap<&str, bool>,
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
        // Provenance — same path-based signals as Function/File, so a class
        // defined in a test or benchmark file (e.g. xarray's PlotTestCase)
        // can be excluded from fan-out / centrality queries.
        (
            "is_test",
            ColumnType::Boolean,
            bool_col(
                classes
                    .iter()
                    .map(|c| {
                        Some(
                            file_is_test
                                .get(c.file_path.as_str())
                                .copied()
                                .unwrap_or(false),
                        )
                    })
                    .collect(),
            ),
        ),
        (
            "is_benchmark",
            ColumnType::Boolean,
            bool_col(
                classes
                    .iter()
                    .map(|c| Some(path_is_benchmark(&c.file_path)))
                    .collect(),
            ),
        ),
        // Internal definitions are explicitly `false`, not absent: external
        // stubs (bases resolved from imports/stdlib) are added later by
        // `external_nodes_df` with `is_external = true` onto this same Class /
        // Struct node type via skip-on-conflict. Emitting `false` here means
        // `WHERE n.is_external = false` selects in-repo nodes — without it the
        // property is null on internal nodes and the intuitive filter matches
        // nothing.
        (
            "is_external",
            ColumnType::Boolean,
            bool_col(classes.iter().map(|_| Some(false)).collect()),
        ),
        // Dart Flutter pass: "stateless" / "stateful" / "state" for a
        // widget subclass, absent otherwise.
        (
            "flutter_widget",
            ColumnType::String,
            str_col(
                classes
                    .iter()
                    .map(|c| {
                        c.metadata
                            .get("flutter_widget")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                    })
                    .collect(),
            ),
        ),
    ])
}

pub(super) fn enums_df(enums: &[EnumInfo], file_to_module: &HashMap<&str, &str>) -> DataFrame {
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

pub(super) fn interfaces_df(
    ifs: &[InterfaceInfo],
    file_to_module: &HashMap<&str, &str>,
) -> DataFrame {
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
        // See classes_df: explicit `false` so external Trait stubs (added with
        // `is_external = true`) and internal interfaces share one boolean
        // column rather than leaving internal nodes null.
        (
            "is_external",
            ColumnType::Boolean,
            bool_col(ifs.iter().map(|_| Some(false)).collect()),
        ),
    ])
}

pub(super) fn constants_df(
    consts: &[ConstantInfo],
    file_to_module: &HashMap<&str, &str>,
) -> DataFrame {
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

pub(super) fn elements_df(elements: &[crate::code_tree::models::ElementInfo]) -> DataFrame {
    build_df(vec![
        (
            "qualified_name",
            ColumnType::String,
            str_col(
                elements
                    .iter()
                    .map(|e| Some(e.qualified_name.clone()))
                    .collect(),
            ),
        ),
        (
            "name",
            ColumnType::String,
            str_col(elements.iter().map(|e| Some(e.name.clone())).collect()),
        ),
        (
            "tag",
            ColumnType::String,
            str_col(elements.iter().map(|e| Some(e.tag.clone())).collect()),
        ),
        (
            "kind",
            ColumnType::String,
            str_col(elements.iter().map(|e| Some(e.kind.clone())).collect()),
        ),
        (
            "html_id",
            ColumnType::String,
            str_col(elements.iter().map(|e| e.id.clone()).collect()),
        ),
        (
            "action",
            ColumnType::String,
            str_col(elements.iter().map(|e| e.action.clone()).collect()),
        ),
        (
            "method",
            ColumnType::String,
            str_col(elements.iter().map(|e| e.method.clone()).collect()),
        ),
        (
            "file_path",
            ColumnType::String,
            str_col(elements.iter().map(|e| Some(e.file_path.clone())).collect()),
        ),
        (
            "line_number",
            ColumnType::Int64,
            int_col(
                elements
                    .iter()
                    .map(|e| Some(e.line_number as i64))
                    .collect(),
            ),
        ),
        (
            "end_line",
            ColumnType::Int64,
            int_col(
                elements
                    .iter()
                    .map(|e| e.end_line.map(|v| v as i64))
                    .collect(),
            ),
        ),
    ])
}

pub(super) fn selectors_df(selectors: &[crate::code_tree::models::SelectorInfo]) -> DataFrame {
    build_df(vec![
        (
            "qualified_name",
            ColumnType::String,
            str_col(
                selectors
                    .iter()
                    .map(|s| Some(s.qualified_name.clone()))
                    .collect(),
            ),
        ),
        (
            "name",
            ColumnType::String,
            str_col(selectors.iter().map(|s| Some(s.name.clone())).collect()),
        ),
        (
            "kind",
            ColumnType::String,
            str_col(selectors.iter().map(|s| Some(s.kind.clone())).collect()),
        ),
        (
            "file_path",
            ColumnType::String,
            str_col(
                selectors
                    .iter()
                    .map(|s| Some(s.file_path.clone()))
                    .collect(),
            ),
        ),
        (
            "line_number",
            ColumnType::Int64,
            int_col(
                selectors
                    .iter()
                    .map(|s| Some(s.line_number as i64))
                    .collect(),
            ),
        ),
        (
            "end_line",
            ColumnType::Int64,
            int_col(
                selectors
                    .iter()
                    .map(|s| s.end_line.map(|v| v as i64))
                    .collect(),
            ),
        ),
    ])
}

pub(super) fn element_contains_edges_df(
    elements: &[crate::code_tree::models::ElementInfo],
) -> DataFrame {
    let mut parents: Vec<Option<String>> = Vec::new();
    let mut children: Vec<Option<String>> = Vec::new();
    for e in elements {
        if let Some(p) = &e.parent_qname {
            parents.push(Some(p.clone()));
            children.push(Some(e.qualified_name.clone()));
        }
    }
    build_df(vec![
        ("parent", ColumnType::String, str_col(parents)),
        ("child", ColumnType::String, str_col(children)),
    ])
}
