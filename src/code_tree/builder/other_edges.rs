//! USES_TYPE, CONTAINS, IMPORTS, FFI_EXPOSES edges.

use crate::code_tree::models::{
    ClassInfo, ConstantInfo, EnumInfo, FileInfo, FunctionInfo, InterfaceInfo,
};
use aho_corasick::{AhoCorasick, MatchKind};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};

fn get_separator(language: &str) -> &'static str {
    match language {
        "rust" | "cpp" => "::",
        "python" | "java" | "csharp" => ".",
        _ => "/",
    }
}

pub struct ContainsEdge {
    pub parent: String,
    pub child: String,
}

pub struct ImportEdge {
    pub file_path: String,
    pub module: String,
}

/// `File -[IMPORTS]-> File` — direct file-to-file dependency.
///
/// Sibling to `ImportEdge` (File → Module). Each source file's import strings
/// are resolved against the project's `module_path → file_path` reverse index
/// using the same longest-prefix walk as `build_import_edges`. Multiple
/// imports from the same source file resolving to the same target file are
/// aggregated into a single edge with `import_count` ≥ 1.
///
/// Powers transitive file-level impact analysis: "given changed files, which
/// other files are affected?" is one Cypher hop (e.g.
/// `MATCH (f:File)-[:IMPORTS]->(t:File {is_test: true}) ...`).
pub struct FileImportEdge {
    pub source: String,
    pub target: String,
    pub import_count: i64,
}

pub struct UsesTypeEdge {
    pub function: String,
    pub type_name: String,
    /// Target node type: "Struct" | "Class" | "Enum" | "Trait" | "Protocol" | "Interface"
    pub target_node_type: &'static str,
    /// Where in the function signature this type appears. Aggregates across
    /// all sites in the same function — a type used as both a parameter and
    /// a return value yields `"both"`. Values: `"parameter"` | `"return"` |
    /// `"both"` | `"signature"`. `"signature"` is the fallback when the
    /// parser couldn't extract structured parameters (typically the AC
    /// scanner found the type embedded in the signature string).
    ///
    /// Cypher: `WHERE r.position IN ['parameter','both']` for "consumes T",
    /// `WHERE r.position IN ['return','both']` for "produces T".
    pub position: &'static str,
}

pub struct FfiExposesEdge {
    pub module_fn: String,
    pub target_qname: String,
    pub target_type: &'static str,
    pub py_name: String,
}

/// `Module -[CONTAINS]-> File` edges — one per file pointing to its leaf module.
///
/// `build_modules` synthesizes a Module node for every prefix of a file's
/// `module_path`; the leaf module's qualified_name equals `file.module_path`
/// exactly. Without this edge, "what's in module X" requires a string
/// `STARTS WITH` filter; with it, the natural top-down walk works:
///
/// ```cypher
/// MATCH (m:Module {qualified_name: 'crate::graph::cypher'})
///       -[:HAS_SUBMODULE*0..]->(:Module)-[:CONTAINS]->(f:File)-[:DEFINES]->(fn:Function)
/// RETURN fn.qualified_name
/// ```
pub struct ModuleContainsFileEdge {
    pub module: String,
    pub file_path: String,
}

pub fn build_module_contains_file_edges(files: &[FileInfo]) -> Vec<ModuleContainsFileEdge> {
    files
        .iter()
        .filter(|f| !f.module_path.is_empty())
        .map(|f| ModuleContainsFileEdge {
            module: f.module_path.clone(),
            file_path: f.path.clone(),
        })
        .collect()
}

/// Module CONTAINS Module edges from file submodule declarations.
pub fn build_contains_edges(files: &[FileInfo]) -> Vec<ContainsEdge> {
    let mut out = Vec::new();
    for f in files {
        let sep = get_separator(&f.language);
        for sub in &f.submodule_declarations {
            out.push(ContainsEdge {
                parent: f.module_path.clone(),
                child: format!("{}{}{}", f.module_path, sep, sub),
            });
        }
    }
    out
}

/// File IMPORTS Module edges — resolve each import string against known modules.
pub fn build_import_edges(files: &[FileInfo], known_modules: &HashSet<String>) -> Vec<ImportEdge> {
    let mut out = Vec::new();
    for f in files {
        let sep = get_separator(&f.language);
        for use_path in &f.imports {
            let parts: Vec<&str> = use_path.split(sep).collect();
            for end in (1..=parts.len()).rev() {
                let candidate = parts[..end].join(sep);
                if known_modules.contains(&candidate) {
                    out.push(ImportEdge {
                        file_path: f.path.clone(),
                        module: candidate,
                    });
                    break;
                }
            }
        }
    }
    out
}

/// `File -[IMPORTS]-> File` edges — resolve each import string to a project
/// file via the `module_path → file_path` reverse index.
///
/// Walks the import path from longest to shortest prefix (mirroring
/// `build_import_edges`'s module resolution) and lands on the first file
/// whose `module_path` matches a prefix candidate. Self-imports are skipped.
/// Multiple imports from the same source resolving to the same target are
/// aggregated into a single edge whose `import_count` records the multiplicity.
pub fn build_file_import_edges(
    files: &[FileInfo],
    module_to_file: &HashMap<String, String>,
) -> Vec<FileImportEdge> {
    let mut counts: HashMap<(String, String), i64> = HashMap::new();
    for f in files {
        let sep = get_separator(&f.language);
        for use_path in &f.imports {
            let parts: Vec<&str> = use_path.split(sep).collect();
            for end in (1..=parts.len()).rev() {
                let candidate = parts[..end].join(sep);
                if let Some(target_file) = module_to_file.get(&candidate) {
                    if target_file != &f.path {
                        *counts
                            .entry((f.path.clone(), target_file.clone()))
                            .or_insert(0) += 1;
                    }
                    break;
                }
            }
        }
    }
    counts
        .into_iter()
        .map(|((source, target), import_count)| FileImportEdge {
            source,
            target,
            import_count,
        })
        .collect()
}

/// USES_TYPE edges: scan function signature/return_type for known type names.
///
/// Returns a map from target node type → list of edges, because add_connections
/// must be called separately for each distinct target type.
pub fn build_uses_type_edges(
    functions: &[FunctionInfo],
    classes: &[ClassInfo],
    enums: &[EnumInfo],
    interfaces: &[InterfaceInfo],
) -> BTreeMap<&'static str, Vec<UsesTypeEdge>> {
    // Collect known type names → (qualified_name, node_type).
    let mut type_lookup: HashMap<String, (String, &'static str)> = HashMap::new();
    for c in classes {
        if c.name.chars().count() > 1 {
            let target = if c.kind == "struct" {
                "Struct"
            } else {
                "Class"
            };
            type_lookup.insert(c.name.clone(), (c.qualified_name.clone(), target));
        }
    }
    for e in enums {
        if e.name.chars().count() > 1 {
            type_lookup.insert(e.name.clone(), (e.qualified_name.clone(), "Enum"));
        }
    }
    for i in interfaces {
        if i.name.chars().count() > 1 {
            let target = match i.kind.as_str() {
                "trait" => "Trait",
                "protocol" => "Protocol",
                _ => "Interface",
            };
            type_lookup.insert(i.name.clone(), (i.qualified_name.clone(), target));
        }
    }

    if type_lookup.is_empty() {
        return BTreeMap::new();
    }

    // Flatten type names into a stable-ordered Vec so pattern IDs from
    // Aho-Corasick map back to the right (qname, node_type) tuple.
    // Longest-match-first so "MyCollection" wins over "Collection".
    let mut names: Vec<String> = type_lookup.keys().cloned().collect();
    names.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    let pattern_meta: Vec<(String, &'static str)> = names
        .iter()
        .map(|n| {
            let (q, t) = type_lookup.get(n).unwrap();
            (q.clone(), *t)
        })
        .collect();

    let ac = match AhoCorasick::builder()
        .match_kind(MatchKind::LeftmostLongest)
        .build(&names)
    {
        Ok(ac) => ac,
        Err(_) => return BTreeMap::new(),
    };

    // Per-function scan in parallel. Scans signature/return_type/each parameter
    // separately, tracks the *set* of positions per pattern, then collapses to
    // a single position per (function, type) so we emit at most one USES_TYPE
    // edge per node pair. (The graph engine keys edges by (src, type, tgt) —
    // multiple edges with same nodes would overwrite.)
    //
    // Position bitset: bit 0 = parameter, bit 1 = return, bit 2 = signature,
    // bit 3 = receiver. Receiver is treated as an input position distinct
    // from `parameter` because `func (c *Call) lock()`-style methods consume
    // their receiver type implicitly — users querying "who consumes T" want
    // both parameter and receiver matches, but they're semantically different.
    const POS_PARAM: u8 = 1 << 0;
    const POS_RETURN: u8 = 1 << 1;
    const POS_SIGNATURE: u8 = 1 << 2;
    const POS_RECEIVER: u8 = 1 << 3;

    let per_fn: Vec<Vec<(u32, &'static str, String, &'static str)>> = functions
        .par_iter()
        .map(|fn_info| {
            // pat_id → bitset of positions seen in this function.
            let mut seen: HashMap<u32, u8> = HashMap::new();

            let scan = |text: &str, pos_bit: u8, seen: &mut HashMap<u32, u8>| {
                if text.is_empty() {
                    return;
                }
                let bytes = text.as_bytes();
                for m in ac.find_iter(text) {
                    let start = m.start();
                    let end = m.end();
                    let before_ok = start == 0
                        || !bytes[start - 1].is_ascii_alphanumeric() && bytes[start - 1] != b'_';
                    let after_ok = end == text.len()
                        || !bytes[end].is_ascii_alphanumeric() && bytes[end] != b'_';
                    if !before_ok || !after_ok {
                        continue;
                    }
                    let pat_id = m.pattern().as_usize() as u32;
                    *seen.entry(pat_id).or_insert(0) |= pos_bit;
                }
            };

            // 1. Each structured parameter type — clean per-position attribution.
            //    Receivers (Go `(c *Call)`, Rust `&self`) get POS_RECEIVER instead
            //    of POS_PARAM so the resulting edge is labeled `position="receiver"`.
            for p in &fn_info.parameters {
                if let Some(t) = &p.type_annotation {
                    let pos_bit = if p.kind == crate::code_tree::models::ParameterKind::Receiver {
                        POS_RECEIVER
                    } else {
                        POS_PARAM
                    };
                    scan(t, pos_bit, &mut seen);
                }
            }
            // 2. Return type.
            if let Some(rt) = &fn_info.return_type {
                scan(rt, POS_RETURN, &mut seen);
            }
            // 3. Signature fallback — only when structured parameters are
            // empty (parser couldn't extract them). Without this, legacy
            // parses lose USES_TYPE coverage entirely. Tagged "signature"
            // so callers know it's a less-precise attribution.
            let has_param_types = fn_info
                .parameters
                .iter()
                .any(|p| p.type_annotation.is_some());
            if !has_param_types && !fn_info.signature.is_empty() {
                scan(&fn_info.signature, POS_SIGNATURE, &mut seen);
            }

            // Collapse bitset to a single label. {param, return, receiver} are
            // semantic positions; signature is the fallback. If two or more
            // semantic positions fire (e.g. receiver + return on a chaining
            // method), collapse to "both". Pure receiver-only stays "receiver".
            seen.into_iter()
                .map(|(pat_id, bits)| {
                    let semantic_count = (bits & POS_PARAM != 0) as u8
                        + (bits & POS_RETURN != 0) as u8
                        + (bits & POS_RECEIVER != 0) as u8;
                    let position = if semantic_count >= 2 {
                        "both"
                    } else if bits & POS_RECEIVER != 0 {
                        "receiver"
                    } else if bits & POS_PARAM != 0 {
                        "parameter"
                    } else if bits & POS_RETURN != 0 {
                        "return"
                    } else if bits & POS_SIGNATURE != 0 {
                        "signature"
                    } else {
                        unreachable!("at least one position bit must be set");
                    };
                    let (qname, target) = &pattern_meta[pat_id as usize];
                    (pat_id, *target, qname.clone(), position)
                })
                .collect()
        })
        .collect();

    let mut by_target_type: BTreeMap<&'static str, Vec<UsesTypeEdge>> = BTreeMap::new();
    for (fn_info, matches) in functions.iter().zip(per_fn.into_iter()) {
        for (_pat_id, target, qname, position) in matches {
            by_target_type
                .entry(target)
                .or_default()
                .push(UsesTypeEdge {
                    function: fn_info.qualified_name.clone(),
                    type_name: qname,
                    target_node_type: target,
                    position,
                });
        }
    }

    by_target_type
}

pub struct ReferencesEdge {
    pub function: String,
    pub constant: String,
    /// Line number in the function body where the reference appears.
    pub line: u32,
}

pub struct ReferencesFnEdge {
    pub caller: String,
    pub callee: String,
    pub line: u32,
}

/// `Function -[DECORATES]-> Function` — resolved decorator-to-decoratee edges.
///
/// Per-language parsers already populate `FunctionInfo.decorators` with the
/// raw decorator strings (`"property"`, `"functools.wraps"`, `"app.route('/x')"`).
/// This pass strips any call-args, extracts the terminal segment as a bare
/// name, and resolves it against the project's Function set the same way
/// `build_call_edges` does for CALLS.
///
/// Direction: `decorator -[DECORATES]-> function` reads naturally as
/// "this decorator decorates that function". Third-party decorators
/// (`@pytest.fixture`, `@app.route` from a Flask app) that don't have a
/// matching Function node are silently dropped — the absence of an edge
/// is correct (we can't resolve into code we don't parse) and mirrors
/// `build_call_edges`'s same-file/global-fallback handling.
pub struct DecoratesEdge {
    pub decorator: String,
    pub function: String,
    /// Raw decorator string from source (e.g. `"functools.wraps"` or
    /// `"app.route('/users/{id}')"`). Preserved on the edge so callers
    /// who want the original literal don't have to reparse the
    /// Function.decorators property.
    pub decorator_name: String,
}

/// REFERENCES edges from `Function` to `Constant` — emit one row per
/// `(function, constant)` pair where the constant's terminal name
/// appears in the function body's identifier stream.
///
/// Per-language parsers populate `FunctionInfo.references` with
/// constant-style identifier candidates (the Rust parser uses
/// `SCREAMING_SNAKE_CASE` as the heuristic). This pass resolves each
/// candidate against the project's constant set and dedupes per
/// `(function, constant)` pair so a constant referenced N times in
/// one function still produces a single edge.
pub fn build_references_edges(
    functions: &[FunctionInfo],
    constants: &[ConstantInfo],
) -> Vec<ReferencesEdge> {
    if constants.is_empty() {
        return Vec::new();
    }

    // Name-keyed lookup: constant short-name → qualified_name. When two
    // constants share the same name (cross-module), we keep both —
    // emit edges to all matches. This mirrors how the type-name resolver
    // handles ambiguity (it doesn't disambiguate by import scope yet).
    let mut by_name: HashMap<&str, Vec<&str>> = HashMap::new();
    for c in constants {
        by_name
            .entry(c.name.as_str())
            .or_default()
            .push(c.qualified_name.as_str());
    }

    let mut out: Vec<ReferencesEdge> = Vec::new();
    for f in functions {
        if f.references.is_empty() {
            continue;
        }
        // Dedup per (function, constant_qname) — a function that uses
        // the same constant on three lines emits one edge.
        let mut seen: HashSet<&str> = HashSet::new();
        for (ident, line) in &f.references {
            let Some(matches) = by_name.get(ident.as_str()) else {
                continue;
            };
            for &qname in matches {
                if seen.insert(qname) {
                    out.push(ReferencesEdge {
                        function: f.qualified_name.clone(),
                        constant: qname.to_string(),
                        line: *line,
                    });
                }
            }
        }
    }
    out
}

/// `Function -[REFERENCES_FN]-> Function` — bare/scoped identifiers
/// passed as arguments to higher-order calls (`iter.and_then(some_fn)`).
/// The referenced function isn't invoked at the reference site, so this
/// is intentionally a different edge type from `CALLS`. Dead-code
/// analysis can union the two: a function with `CALLS ∪ REFERENCES_FN`
/// = 0 inbound is genuinely uncalled.
///
/// Resolution mirrors `build_call_edges`'s name-keyed lookup: only
/// emit an edge when the identifier matches exactly one function in
/// the project (skip ambiguous matches to avoid noise from
/// argument-name collisions with unrelated functions).
pub fn build_references_fn_edges(functions: &[FunctionInfo]) -> Vec<ReferencesFnEdge> {
    if functions.is_empty() {
        return Vec::new();
    }
    let mut by_name: HashMap<&str, Vec<&str>> = HashMap::new();
    for f in functions {
        by_name
            .entry(f.name.as_str())
            .or_default()
            .push(f.qualified_name.as_str());
    }

    let mut out: Vec<ReferencesFnEdge> = Vec::new();
    for f in functions {
        if f.function_refs.is_empty() {
            continue;
        }
        let caller = f.qualified_name.as_str();
        let mut seen: HashSet<&str> = HashSet::new();
        for (ident, line) in &f.function_refs {
            let Some(matches) = by_name.get(ident.as_str()) else {
                continue;
            };
            // Only emit on unambiguous matches — if the bare name maps
            // to multiple functions, skip rather than guess. Function
            // pointers passed as arguments don't carry receiver-type
            // info that the call-edge resolver could use to narrow.
            if matches.len() != 1 {
                continue;
            }
            let target = matches[0];
            if target == caller {
                continue;
            }
            if seen.insert(target) {
                out.push(ReferencesFnEdge {
                    caller: caller.to_string(),
                    callee: target.to_string(),
                    line: *line,
                });
            }
        }
    }
    out
}

/// `Function -[BINDS]-> Function` — Python wrapper to its underlying Rust impl.
///
/// PyO3 exposes a `#[pyclass]` Rust struct (e.g. `KnowledgeGraph`) and its
/// `#[pymethods]` block to Python. The Python class shows up in a `.pyi`
/// stub like `kglite.KnowledgeGraph.add_nodes`, while the Rust side has
/// `crate::graph::pyapi::*::KnowledgeGraph::add_nodes` (with
/// `metadata.is_pymethod == true`). Method names are 1:1 by PyO3 contract.
///
/// Closes the cross-language graph: `MATCH (py)-[:BINDS]->(rs)-[:CALLS*]->(impl)`
/// traces a request from the Python entry point down to deep Rust impl.
///
/// Resolution rules:
/// - Python side: Function whose `qualified_name` matches `<pkg>.<Class>.<method>`
///   *and* whose `is_method == true`.
/// - Rust side: Function whose name == `<method>`, owner == `<Class>`, and
///   `metadata["is_pymethod"] == true`.
/// - Skip ambiguous Rust matches (multiple pymethods with same `(Class, method)`)
///   to avoid guessing — wouldn't compile under PyO3 anyway, but be defensive.
pub struct PyO3BindsEdge {
    pub py_function: String,
    pub rust_function: String,
}

pub fn build_pyo3_binds_edges(functions: &[FunctionInfo]) -> Vec<PyO3BindsEdge> {
    // Index Rust pymethods by (parent_struct_short_name, method_short_name).
    let mut rust_idx: HashMap<(String, String), Vec<&str>> = HashMap::new();
    for f in functions {
        if !f
            .metadata
            .get("is_pymethod")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            continue;
        }
        // Derive parent struct short name from the qualified name.
        // Rust pymethod qnames look like `crate::a::b::KnowledgeGraph::add_nodes`.
        let parts: Vec<&str> = f.qualified_name.split("::").collect();
        if parts.len() < 2 {
            continue;
        }
        let parent = parts[parts.len() - 2].to_string();
        let method = parts[parts.len() - 1].to_string();
        rust_idx
            .entry((parent, method))
            .or_default()
            .push(f.qualified_name.as_str());
    }

    let mut out = Vec::new();
    for f in functions {
        // Python class methods come from `.pyi` stubs and look like
        // `kglite.KnowledgeGraph.add_nodes`. The split separator is `.`.
        if !f.qualified_name.contains('.') || !f.is_method {
            continue;
        }
        let parts: Vec<&str> = f.qualified_name.split('.').collect();
        if parts.len() < 3 {
            continue;
        }
        let py_class = parts[parts.len() - 2].to_string();
        let py_method = parts[parts.len() - 1].to_string();
        let Some(matches) = rust_idx.get(&(py_class, py_method)) else {
            continue;
        };
        if matches.len() != 1 {
            continue; // ambiguous — skip
        }
        out.push(PyO3BindsEdge {
            py_function: f.qualified_name.clone(),
            rust_function: matches[0].to_string(),
        });
    }
    out
}

/// `Function -[DECORATES]-> Function` — resolve each parsed decorator
/// string to its target function. Strips call-args (`@app.route('/x')` →
/// `app.route`) and the namespace prefix (`functools.wraps` → `wraps`)
/// before consulting a bare-name index built from every project Function.
///
/// Unambiguous matches (exactly one qualified-name candidate) emit an
/// edge. Ambiguous bare names are skipped — duplicating the call-edge
/// resolver's stance: without import-scope info we'd guess, and a wrong
/// edge is worse than a missing one for downstream queries that count
/// `WHERE (dec)-[:DECORATES]->(fn) RETURN dec.name`. Self-decoration is
/// suppressed (would only happen on malformed input).
pub fn build_decorates_edges(functions: &[FunctionInfo]) -> Vec<DecoratesEdge> {
    if functions.is_empty() {
        return Vec::new();
    }
    // bare name → list of qualified_names that share that short name.
    let mut by_name: HashMap<&str, Vec<&str>> = HashMap::new();
    for f in functions {
        by_name
            .entry(f.name.as_str())
            .or_default()
            .push(f.qualified_name.as_str());
    }

    let mut out: Vec<DecoratesEdge> = Vec::new();
    for f in functions {
        if f.decorators.is_empty() {
            continue;
        }
        let function_qname = f.qualified_name.as_str();
        // Dedup per (decorator_qname → function) — a function with two
        // decorators that happen to resolve to the same target only
        // emits one edge. Carries the *first* raw decorator_name we
        // saw so the property remains stable.
        let mut seen: HashSet<&str> = HashSet::new();
        for raw in &f.decorators {
            // Strip call args: `app.route('/x', methods=['GET'])` → `app.route`.
            let head = raw.split('(').next().unwrap_or(raw).trim();
            if head.is_empty() {
                continue;
            }
            // Take the terminal segment after the last `.` or `::` — that's
            // the bare function name we look up. `functools.wraps` → `wraps`.
            let bare = head
                .rsplit_once("::")
                .map(|(_, t)| t)
                .or_else(|| head.rsplit_once('.').map(|(_, t)| t))
                .unwrap_or(head);
            let Some(candidates) = by_name.get(bare) else {
                continue;
            };
            if candidates.len() != 1 {
                continue; // ambiguous bare name — skip rather than guess
            }
            let target = candidates[0];
            if target == function_qname {
                continue; // self-decoration — defensive
            }
            if seen.insert(target) {
                out.push(DecoratesEdge {
                    decorator: target.to_string(),
                    function: function_qname.to_string(),
                    decorator_name: raw.clone(),
                });
            }
        }
    }
    out
}

/// FFI EXPOSES edges — #[pymodule] fn → each #[pyclass]/#[pyfunction] item.
pub fn build_ffi_exposes_edges(
    functions: &[FunctionInfo],
    classes: &[ClassInfo],
) -> Vec<FfiExposesEdge> {
    let pymodule_fns: Vec<&FunctionInfo> = functions
        .iter()
        .filter(|f| {
            f.metadata
                .get("is_pymodule")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        })
        .collect();
    if pymodule_fns.is_empty() {
        return Vec::new();
    }

    let pyclass_items: Vec<(&ClassInfo, String)> = classes
        .iter()
        .filter(|c| {
            c.metadata
                .get("is_pyclass")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        })
        .map(|c| {
            let py_name = c
                .metadata
                .get("py_name")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| c.name.clone());
            (c, py_name)
        })
        .collect();

    let pyfunc_items: Vec<(&FunctionInfo, String)> = functions
        .iter()
        .filter(|f| {
            !f.is_method
                && !f
                    .metadata
                    .get("is_pymodule")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                && f.metadata.get("ffi_kind").and_then(|v| v.as_str()) == Some("pyo3")
        })
        .map(|f| {
            let py_name = f
                .metadata
                .get("py_name")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| f.name.clone());
            (f, py_name)
        })
        .collect();

    let mut out = Vec::new();
    for mod_fn in &pymodule_fns {
        for (c, py_name) in &pyclass_items {
            out.push(FfiExposesEdge {
                module_fn: mod_fn.qualified_name.clone(),
                target_qname: c.qualified_name.clone(),
                target_type: "Struct",
                py_name: py_name.clone(),
            });
        }
        for (f, py_name) in &pyfunc_items {
            out.push(FfiExposesEdge {
                module_fn: mod_fn.qualified_name.clone(),
                target_qname: f.qualified_name.clone(),
                target_type: "Function",
                py_name: py_name.clone(),
            });
        }
    }
    out
}
