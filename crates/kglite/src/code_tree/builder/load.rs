//! Phase 3: load parsed entities into a KnowledgeGraph.
//!
//! Builds `crate::datatypes::DataFrame` objects directly from Rust record
//! vectors and hands them to `crate::graph::mutation::maintain::add_nodes` /
//! `add_connections` — no pandas, no PyO3 round-trip.

use crate::code_tree::models::{
    AttributeInfo, ClassInfo, FileInfo, InterfaceInfo, ParseResult, ProjectInfo,
};
use crate::datatypes::values::ColumnType;
use crate::graph::mutation::maintain;
// Build a `DirGraph` directly instead of the binding-flavored
// `KnowledgeGraph` wrapper — keeps code_tree engine-only. The
// pyapi callsite (`code_tree.build` pyfunction) wraps via
// `KnowledgeGraph::from_arc` at the Python boundary.
use crate::graph::dir_graph::DirGraph;
use std::collections::{BTreeMap, HashMap};

mod edge_frames;
mod entity_frames;

pub use edge_frames::DefinesEdge;
use edge_frames::*;
use entity_frames::*;

pub struct ModuleRecord {
    pub qualified_name: String,
    pub name: String,
    pub language: String,
    pub is_test: bool,
    pub is_benchmark: bool,
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
                is_benchmark: path_is_benchmark(&f.path) && end == parts.len(),
            });
        }
    }
    seen.into_values().collect()
}

/// True when `s` is non-empty and made up entirely of ASCII digits.
fn is_numeric_segment(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

fn pick_sep(language: &str) -> &'static str {
    match language {
        "rust" | "cpp" | "c" => "::",
        "python" | "java" | "csharp" => ".",
        "typescript" | "javascript" | "go" => "/",
        _ => ".",
    }
}

// ── Entry point ────────────────────────────────────────────────────

pub fn load_into_graph(
    result: &ParseResult,
    project_info: Option<&ProjectInfo>,
) -> Result<
    (
        std::sync::Arc<DirGraph>,
        super::call_edges::CallResolutionStats,
    ),
    String,
> {
    let verbose = std::env::var_os("KGLITE_CODE_TREE_VERBOSE").is_some();
    let mark = |started: std::time::Instant, label: &str| {
        if verbose {
            eprintln!(
                "[timing]   {}: {:.3}s",
                label,
                started.elapsed().as_secs_f64()
            );
        }
    };
    let mut pipeline = LoadPipeline::new(result, project_info);

    macro_rules! run_stage {
        ($method:ident, $label:literal) => {{
            let started = std::time::Instant::now();
            pipeline.$method()?;
            mark(started, $label);
        }};
    }
    run_stage!(project_and_dependencies, "setup+project/deps");
    run_stage!(entity_nodes, "nodes");
    run_stage!(external_type_stubs, "type_edges build+external stubs");
    run_stage!(routes, "routes");
    run_stage!(
        structural_edges,
        "edges: submodule+contains+defines+imports+depends+hasrc"
    );
    run_stage!(call_edges, "calls");
    run_stage!(type_relationship_edges, "implements+extends+has_method");
    run_stage!(uses_type_edges, "uses_type");
    run_stage!(reference_edges, "references");
    run_stage!(function_reference_edges, "references_fn");
    run_stage!(decorator_edges, "decorates");
    run_stage!(ffi_edges, "ffi_exposes");
    run_stage!(pyo3_binding_edges, "pyo3_binds");
    run_stage!(procedures, "procedures");

    Ok((std::sync::Arc::new(pipeline.graph), pipeline.call_stats))
}

struct LoadPipeline<'a> {
    result: &'a ParseResult,
    project_info: Option<&'a ProjectInfo>,
    graph: DirGraph,
    modules: Vec<ModuleRecord>,
    known_modules: std::collections::HashSet<String>,
    attrs_by_owner: HashMap<String, Vec<&'a AttributeInfo>>,
    type_edges: Option<super::type_edges::TypeEdgeOutput>,
    call_stats: super::call_edges::CallResolutionStats,
}

impl<'a> LoadPipeline<'a> {
    fn new(result: &'a ParseResult, project_info: Option<&'a ProjectInfo>) -> Self {
        Self {
            result,
            project_info,
            graph: DirGraph::new(),
            modules: Vec::new(),
            known_modules: std::collections::HashSet::new(),
            attrs_by_owner: HashMap::new(),
            type_edges: None,
            call_stats: super::call_edges::CallResolutionStats::default(),
        }
    }

    fn project_and_dependencies(&mut self) -> Result<(), String> {
        let result = self.result;
        let project_info = self.project_info;
        let graph = &mut self.graph;

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

        self.modules = build_modules(&result.files);
        self.known_modules = self
            .modules
            .iter()
            .map(|module| module.qualified_name.clone())
            .collect();
        for attribute in &result.attributes {
            self.attrs_by_owner
                .entry(attribute.owner_qualified_name.clone())
                .or_default()
                .push(attribute);
        }
        Ok(())
    }

    fn entity_nodes(&mut self) -> Result<(), String> {
        let result = self.result;
        let graph = &mut self.graph;
        let modules = &self.modules;
        let attrs_by_owner = &self.attrs_by_owner;
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
                modules_df(modules),
                "Module".into(),
                "qualified_name".into(),
                Some("name".into()),
                None,
            )
            .map_err(py_err)?;
        }
        // file_path → is_test, shared by Function and Class node construction so a
        // class defined in a test file inherits the file's test provenance.
        let file_is_test: HashMap<&str, bool> = result
            .files
            .iter()
            .map(|f| (f.path.as_str(), f.is_test))
            .collect();
        if !result.functions.is_empty() {
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
        // Separate struct / mixin / class — each is a distinct graph node label.
        let (structs, non_structs): (Vec<_>, Vec<_>) =
            result.classes.iter().partition(|c| c.kind == "struct");
        let (mixins, classes): (Vec<_>, Vec<_>) =
            non_structs.into_iter().partition(|c| c.kind == "mixin");
        if !structs.is_empty() {
            let structs_owned: Vec<ClassInfo> = structs.into_iter().cloned().collect();
            maintain::add_nodes(
                graph,
                classes_df(
                    &structs_owned,
                    attrs_by_owner,
                    &file_to_module,
                    &file_is_test,
                ),
                "Struct".into(),
                "qualified_name".into(),
                Some("name".into()),
                None,
            )
            .map_err(py_err)?;
        }
        if !mixins.is_empty() {
            let mixins_owned: Vec<ClassInfo> = mixins.into_iter().cloned().collect();
            maintain::add_nodes(
                graph,
                classes_df(
                    &mixins_owned,
                    attrs_by_owner,
                    &file_to_module,
                    &file_is_test,
                ),
                "Mixin".into(),
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
                classes_df(
                    &classes_owned,
                    attrs_by_owner,
                    &file_to_module,
                    &file_is_test,
                ),
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
        // Element nodes — HTML structural elements (headings, sections,
        // forms) emitted by the 0.9.36 HTML parser. The HTML parser
        // imposes its own emission filter (only nodes with semantic
        // interest), so we just shovel the prepared list into the graph.
        if !result.elements.is_empty() {
            maintain::add_nodes(
                graph,
                elements_df(&result.elements),
                "Element".into(),
                "qualified_name".into(),
                Some("name".into()),
                None,
            )
            .map_err(py_err)?;
        }
        // Selector nodes — CSS rule_sets emitted by the 0.9.36 CSS parser
        // (one per rule_set, regardless of selector-list count).
        if !result.selectors.is_empty() {
            maintain::add_nodes(
                graph,
                selectors_df(&result.selectors),
                "Selector".into(),
                "qualified_name".into(),
                Some("name".into()),
                None,
            )
            .map_err(py_err)?;
        }

        Ok(())
    }

    fn external_type_stubs(&mut self) -> Result<(), String> {
        let result = self.result;
        let graph = &mut self.graph;
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

        self.type_edges = Some(type_out);
        Ok(())
    }

    fn routes(&mut self) -> Result<(), String> {
        let result = self.result;
        let graph = &mut self.graph;
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

        Ok(())
    }

    fn structural_edges(&mut self) -> Result<(), String> {
        let result = self.result;
        let project_info = self.project_info;
        let graph = &mut self.graph;
        let known_modules = &self.known_modules;
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

        // Element HAS_CHILD Element — outline edges from the HTML parser
        // (0.9.36). Edge name avoids `CONTAINS`, which is a reserved
        // Cypher keyword for substring matching (same rule that drove
        // `HAS_FILE` instead of `CONTAINS` on Module→File).
        if !result.elements.is_empty() {
            let contains_df = element_contains_edges_df(&result.elements);
            if contains_df.row_count() > 0 {
                maintain::add_connections(
                    graph,
                    contains_df,
                    "HAS_CHILD".into(),
                    "Element".into(),
                    "parent".into(),
                    "Element".into(),
                    "child".into(),
                    None,
                    None,
                    None,
                )
                .map_err(py_err)?;
            }
        }

        // File IMPORTS Module (only edges to known modules).
        let imports = super::other_edges::build_import_edges(&result.files, known_modules);
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
        let file_imports =
            super::other_edges::build_file_import_edges(&result.files, &module_to_file);
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

        Ok(())
    }

    fn call_edges(&mut self) -> Result<(), String> {
        let result = self.result;
        let graph = &mut self.graph;
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
        for name in super::super::parsers::swift::SWIFT_NOISE_NAMES {
            noise.insert(*name);
        }
        for name in super::super::parsers::php::PHP_NOISE_NAMES {
            noise.insert(*name);
        }
        let (call_edges, call_stats) = super::call_edges::build_call_edges(
            &result.functions,
            &result.files,
            &noise,
            5,
            &result.type_relationships,
        );
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

        self.call_stats = call_stats;
        Ok(())
    }

    fn type_relationship_edges(&mut self) -> Result<(), String> {
        let result = self.result;
        let graph = &mut self.graph;
        let type_out = self
            .type_edges
            .as_ref()
            .expect("type-edge stage must run first");
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
                let nt = super::class_node_type(&c.kind);
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

            let default_src =
                pick(&[("Class", has_class), ("Struct", has_struct)]).unwrap_or("Class");
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
                qname_to_type.insert(c.qualified_name.clone(), super::class_node_type(&c.kind));
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

        Ok(())
    }

    fn uses_type_edges(&mut self) -> Result<(), String> {
        let result = self.result;
        let graph = &mut self.graph;
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

        Ok(())
    }

    fn reference_edges(&mut self) -> Result<(), String> {
        let result = self.result;
        let graph = &mut self.graph;
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

        Ok(())
    }

    fn function_reference_edges(&mut self) -> Result<(), String> {
        let result = self.result;
        let graph = &mut self.graph;
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

        Ok(())
    }

    fn decorator_edges(&mut self) -> Result<(), String> {
        let result = self.result;
        let graph = &mut self.graph;
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

        Ok(())
    }

    fn ffi_edges(&mut self) -> Result<(), String> {
        let result = self.result;
        let graph = &mut self.graph;
        // FFI EXPOSES.
        let ffi = super::other_edges::build_ffi_exposes_edges(&result.functions, &result.classes);
        if !ffi.is_empty() {
            // Batch by target_type.
            let (structs, fns): (Vec<_>, Vec<_>) =
                ffi.iter().partition(|e| e.target_type == "Struct");
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

        Ok(())
    }

    fn pyo3_binding_edges(&mut self) -> Result<(), String> {
        let result = self.result;
        let graph = &mut self.graph;
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
        Ok(())
    }

    fn procedures(&mut self) -> Result<(), String> {
        let result = self.result;
        let graph = &mut self.graph;
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
        Ok(())
    }
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
