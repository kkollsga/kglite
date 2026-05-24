//! CALLS edge resolution — scope-aware name matching with import context.
//!
//! Ported from builder.py::_build_call_edges, then extended with a
//! namespace-import tier so that languages with explicit `using`/`import`
//! directives (C#, Java, TS, Python, Go) disambiguate same-named symbols
//! across namespaces. On dotnet/runtime this lifts CALLS resolution from
//! ~9% to a meaningfully higher rate by pinning calls like
//! `Assert.True` to the Assert class actually imported by the caller.

use crate::code_tree::models::{FileInfo, FunctionInfo};
use rayon::prelude::*;
use std::collections::HashMap;

/// One resolved caller → callee edge, with call-site line numbers.
pub struct CallEdge {
    pub caller: String,
    pub callee: String,
    /// Comma-separated sorted unique line numbers.
    pub call_lines: String,
    pub call_count: i64,
}

/// True if `qname` lives under any of the namespace prefixes in `scopes`.
/// A "lives under" match requires `qname` to start with `scope` followed by
/// a `.` or `::` separator — `System` matches `System.IO.Stream` but not
/// `Systemic`.
fn qname_starts_with_any(qname: &str, scopes: &[String]) -> bool {
    for scope in scopes {
        if scope.is_empty() {
            continue;
        }
        if qname.len() > scope.len()
            && qname.starts_with(scope.as_str())
            && (qname.as_bytes()[scope.len()] == b'.'
                || (qname.len() > scope.len() + 1 && &qname[scope.len()..scope.len() + 2] == "::"))
        {
            return true;
        }
    }
    false
}

fn infer_lang_group(qname: &str) -> &'static str {
    if qname.contains("::") {
        "rust_cpp"
    } else if qname.contains('/') {
        "go_ts_js"
    } else {
        "python_java"
    }
}

/// Run the 5-tier resolution over every parsed function's call sites.
///
/// Tiers (first non-empty wins):
///   0. Receiver hint: `Receiver.method` → narrow by owner short-name
///   1. Same owner: caller and target share qualified prefix
///   2. Same file
///   3. Same language group (separator convention)
///   4. Global fallback (all targets with matching bare name)
///
/// Calls whose bare name appears in `excluded_names` are skipped (stdlib noise).
/// Calls with more than `max_targets` resolvable targets are dropped as too
/// ambiguous.
pub fn build_call_edges(
    functions: &[FunctionInfo],
    files: &[FileInfo],
    excluded_names: &std::collections::HashSet<&str>,
    max_targets: usize,
) -> Vec<CallEdge> {
    let verbose = std::env::var_os("KGLITE_CODE_TREE_VERBOSE").is_some();
    let t0 = std::time::Instant::now();
    // Bare name → every qualified_name that matches.
    let mut name_lookup: HashMap<&str, Vec<&str>> = HashMap::new();
    for fn_info in functions {
        name_lookup
            .entry(fn_info.name.as_str())
            .or_default()
            .push(fn_info.qualified_name.as_str());
    }

    // qualified_name → owner short name (last segment of owner prefix).
    // qualified_name → owner prefix (everything before the final separator).
    let mut qname_to_owner: HashMap<&str, &str> = HashMap::new();
    let mut qname_to_prefix: HashMap<&str, &str> = HashMap::new();
    for fn_info in functions {
        let qn = fn_info.qualified_name.as_str();
        for sep in ["::", ".", "/"] {
            if let Some(idx) = qn.rfind(sep) {
                let owner_path = &qn[..idx];
                qname_to_prefix.insert(qn, owner_path);
                // Find the last separator inside owner_path (any of ::, ., /).
                let mut short = owner_path;
                for sep2 in ["::", ".", "/"] {
                    if let Some(i2) = owner_path.rfind(sep2) {
                        short = &owner_path[i2 + sep2.len()..];
                        break;
                    }
                }
                qname_to_owner.insert(qn, short);
                break;
            }
        }
    }

    // qualified_name → file_path (for tier 2).
    let qname_to_file: HashMap<&str, &str> = functions
        .iter()
        .map(|f| (f.qualified_name.as_str(), f.file_path.as_str()))
        .collect();

    // file_path → imported namespace prefixes. Empty for files whose
    // language doesn't track imports as namespace names.
    let file_imports: HashMap<&str, &Vec<String>> = files
        .iter()
        .filter(|f| !f.imports.is_empty())
        .map(|f| (f.path.as_str(), &f.imports))
        .collect();

    if verbose {
        eprintln!(
            "[calls]     lookup build: {:.3}s",
            t0.elapsed().as_secs_f64()
        );
    }
    let t_match = std::time::Instant::now();

    // Parallelise the per-function match loop: each caller's edges are
    // independent, so we collect per-function edge vectors and merge.
    // Keys stay as &str (borrowed from `functions`) to avoid alloc per edge.
    let per_fn: Vec<Vec<(&str, &str, u32)>> = functions
        .par_iter()
        .map(|fn_info| {
            let caller_qn = fn_info.qualified_name.as_str();
            let caller_lang = infer_lang_group(caller_qn);
            let caller_prefix = qname_to_prefix.get(caller_qn).copied();
            let caller_owner = qname_to_owner.get(caller_qn).copied();
            let caller_file = fn_info.file_path.as_str();

            let mut out: Vec<(&str, &str, u32)> = Vec::new();

            for (called_name, line) in &fn_info.calls {
                let (explicit_hint, method_name) = match called_name.rfind('.') {
                    Some(idx) => (Some(&called_name[..idx]), &called_name[idx + 1..]),
                    None => (None, called_name.as_str()),
                };

                if excluded_names.contains(method_name) {
                    continue;
                }
                let Some(candidates) = name_lookup.get(method_name) else {
                    continue;
                };

                if candidates.len() == 1 {
                    let target = candidates[0];
                    if target != caller_qn {
                        out.push((caller_qn, target, *line));
                    }
                    continue;
                }

                let mut targets: &[&str] = candidates.as_slice();
                let mut filtered: Vec<&str>;

                // Tier 0: receiver-type filter. Two sources of hints —
                // `(explicit_hint, owner_short_match)`:
                //
                //   - Explicit hint from `obj.method()` — the receiver
                //     identifier's text. Already extracted at parse time
                //     (e.g. `cfg.read` becomes `("cfg", "read")`).
                //   - Implicit hint from `self.method()` / bare-name
                //     calls inside a method body — use the caller's own
                //     owner short name as the receiver type. Resolves
                //     `Foo::caller -> Foo::method` correctly when the
                //     same method name exists on multiple structs.
                let implicit_hint = if explicit_hint.is_none() {
                    caller_owner
                } else {
                    None
                };
                if let Some(hint) = explicit_hint.or(implicit_hint) {
                    filtered = targets
                        .iter()
                        .copied()
                        .filter(|t| qname_to_owner.get(t).copied() == Some(hint))
                        .collect();
                    if !filtered.is_empty() {
                        targets = &filtered[..];
                    }
                }

                if targets.len() > 1 {
                    if let Some(prefix) = caller_prefix {
                        let narrowed: Vec<&str> = targets
                            .iter()
                            .copied()
                            .filter(|t| qname_to_prefix.get(t).copied() == Some(prefix))
                            .collect();
                        if !narrowed.is_empty() {
                            filtered = narrowed;
                            targets = &filtered[..];
                        }
                    }
                }

                // Tier 2.5: namespace-import scope. Prefer candidates whose
                // qname lives under a namespace the caller's file imports
                // (or under the caller's own namespace). Critical for
                // disambiguating `Assert.True` across xunit / fluentassertions
                // / project-local assertion helpers — the caller's `using`
                // list pins the correct one.
                if targets.len() > 1 {
                    if let Some(imports) = file_imports.get(caller_file) {
                        let narrowed: Vec<&str> = targets
                            .iter()
                            .copied()
                            .filter(|t| qname_starts_with_any(t, imports))
                            .collect();
                        if !narrowed.is_empty() {
                            filtered = narrowed;
                            targets = &filtered[..];
                        }
                    }
                }

                if targets.len() > 1 {
                    let narrowed: Vec<&str> = targets
                        .iter()
                        .copied()
                        .filter(|t| qname_to_file.get(t).copied() == Some(caller_file))
                        .collect();
                    if !narrowed.is_empty() {
                        filtered = narrowed;
                        targets = &filtered[..];
                    }
                }

                if targets.len() > 1 {
                    let narrowed: Vec<&str> = targets
                        .iter()
                        .copied()
                        .filter(|t| infer_lang_group(t) == caller_lang)
                        .collect();
                    if !narrowed.is_empty() {
                        filtered = narrowed;
                        targets = &filtered[..];
                    }
                }

                if targets.len() > max_targets {
                    continue;
                }

                for &target in targets {
                    if target != caller_qn {
                        out.push((caller_qn, target, *line));
                    }
                }
            }
            out
        })
        .collect();

    // Merge into the final dedupe map sequentially — 200K inserts is ~5ms.
    let total: usize = per_fn.iter().map(|v| v.len()).sum();
    let mut seen: HashMap<(&str, &str), Vec<u32>> = HashMap::with_capacity(total);
    for edges in per_fn {
        for (caller, callee, line) in edges {
            seen.entry((caller, callee)).or_default().push(line);
        }
    }

    if verbose {
        eprintln!(
            "[calls]     match loop:   {:.3}s ({} entries)",
            t_match.elapsed().as_secs_f64(),
            seen.len()
        );
    }
    let t_out = std::time::Instant::now();

    // Sort keys for deterministic output (match Python's ordered dict).
    let mut keys: Vec<(&str, &str)> = seen.keys().copied().collect();
    keys.sort_unstable();

    let result: Vec<CallEdge> = keys
        .into_iter()
        .map(|(caller, callee)| {
            let mut lines = seen.remove(&(caller, callee)).unwrap_or_default();
            lines.sort_unstable();
            lines.dedup();
            let count = lines.len() as i64;
            let mut call_lines = String::with_capacity(lines.len() * 4);
            for (i, l) in lines.iter().enumerate() {
                if i > 0 {
                    call_lines.push(',');
                }
                use std::fmt::Write;
                let _ = write!(call_lines, "{}", l);
            }
            CallEdge {
                caller: caller.to_string(),
                callee: callee.to_string(),
                call_lines,
                call_count: count,
            }
        })
        .collect();
    if verbose {
        eprintln!(
            "[calls]     output build: {:.3}s",
            t_out.elapsed().as_secs_f64()
        );
    }
    result
}
