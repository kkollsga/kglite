//! CALLS edge resolution — scope-aware name matching with import context.
//!
//! Ported from builder.py::_build_call_edges, then extended with a
//! namespace-import tier so that languages with explicit `using`/`import`
//! directives (C#, Java, TS, Python, Go) disambiguate same-named symbols
//! across namespaces. On dotnet/runtime this lifts CALLS resolution from
//! ~9% to a meaningfully higher rate by pinning calls like
//! `Assert.True` to the Assert class actually imported by the caller.

use crate::code_tree::models::{FileInfo, FunctionInfo, TypeRelationship};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// One resolved caller → callee edge, with call-site line numbers.
pub struct CallEdge {
    pub caller: String,
    pub callee: String,
    /// Comma-separated sorted unique line numbers.
    pub call_lines: String,
    pub call_count: i64,
}

/// Aggregate counters describing how the resolver classified every call
/// site in one `build_call_edges` pass — the measurement substrate the
/// re-resolution phases (and the `code_tree_stats` dev bin) track.
///
/// The denominator for resolver *quality* is `total_calls - excluded_noise`.
/// Of those: `no_candidate` reference a bare name absent from the project
/// (external / stdlib — nothing we could resolve to); `ambiguous_dropped`
/// still had more than `max_targets` candidates after every tier;
/// `resolved_call_sites` matched at least one in-project symbol.
/// `resolved_edges` is the de-duplicated caller→callee pair count actually
/// emitted (one call site can fan out to several when tiers can't separate
/// overloads, and repeated calls on different lines collapse to one edge).
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct CallResolutionStats {
    pub total_calls: u64,
    pub excluded_noise: u64,
    pub no_candidate: u64,
    pub ambiguous_dropped: u64,
    pub resolved_call_sites: u64,
    pub resolved_edges: u64,
    /// Subset of `resolved_call_sites` pinned via the inheritance tier — a
    /// `self.method()` whose method is defined on an ancestor (EXTENDS /
    /// IMPLEMENTS), not the caller's own type. The headline win of the
    /// inheritance-aware resolution.
    pub resolved_via_inheritance: u64,
}

/// Per-function scratch counters, summed into [`CallResolutionStats`] after
/// the parallel match loop. Kept `Copy` so the rayon reduce stays alloc-free.
#[derive(Debug, Clone, Copy, Default)]
struct Counts {
    total: u64,
    excluded: u64,
    no_candidate: u64,
    ambiguous: u64,
    resolved: u64,
    inherited: u64,
}

/// Terminal segment of a `::` / `.` / `/`-separated type name — the form
/// stored in `qname_to_owner`, so ancestor lookups match call candidates.
fn short_type_name(name: &str) -> &str {
    let mut cut = 0usize;
    for sep in ["::", ".", "/"] {
        if let Some(i) = name.rfind(sep) {
            let after = i + sep.len();
            if after > cut {
                cut = after;
            }
        }
    }
    &name[cut..]
}

/// type short-name → transitive ancestor short-names, derived from the
/// EXTENDS / IMPLEMENTS relationships in the parse. Borrowed from
/// `rels`, so the map lives as long as the caller's `type_relationships`.
fn build_ancestor_map(rels: &[TypeRelationship]) -> HashMap<&str, HashSet<&str>> {
    let mut parents: HashMap<&str, Vec<&str>> = HashMap::new();
    for tr in rels {
        if tr.relationship == "extends" || tr.relationship == "implements" {
            if let Some(tgt) = tr.target_type.as_deref() {
                parents
                    .entry(short_type_name(&tr.source_type))
                    .or_default()
                    .push(short_type_name(tgt));
            }
        }
    }
    let mut out: HashMap<&str, HashSet<&str>> = HashMap::with_capacity(parents.len());
    for &child in parents.keys() {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = parents.get(child).cloned().unwrap_or_default();
        while let Some(p) = stack.pop() {
            // Guard against inheritance cycles (malformed input) via `seen`.
            if p != child && seen.insert(p) {
                if let Some(gp) = parents.get(p) {
                    stack.extend(gp.iter().copied());
                }
            }
        }
        out.insert(child, seen);
    }
    out
}

/// Per-function output of the parallel match loop: borrowed
/// `(caller, callee, line)` tuples plus that function's [`Counts`].
type FnMatchResult<'a> = (Vec<(&'a str, &'a str, u32)>, Counts);

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
///   0b. Inheritance: a self-call to a method defined on an ancestor (EXTENDS/IMPLEMENTS, not the caller's own type) resolves to the unique inherited definition
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
    type_relationships: &[TypeRelationship],
) -> (Vec<CallEdge>, CallResolutionStats) {
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

    // type short-name → transitive ancestors, for the inheritance tier.
    let ancestors = build_ancestor_map(type_relationships);

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
    let per_fn: Vec<FnMatchResult> = functions
        .par_iter()
        .map(|fn_info| {
            let caller_qn = fn_info.qualified_name.as_str();
            let caller_lang = infer_lang_group(caller_qn);
            let caller_prefix = qname_to_prefix.get(caller_qn).copied();
            let caller_owner = qname_to_owner.get(caller_qn).copied();
            let caller_file = fn_info.file_path.as_str();

            let mut out: Vec<(&str, &str, u32)> = Vec::new();
            let mut counts = Counts::default();

            for (called_name, line) in &fn_info.calls {
                counts.total += 1;
                let (explicit_hint, method_name) = match called_name.rfind('.') {
                    Some(idx) => (Some(&called_name[..idx]), &called_name[idx + 1..]),
                    None => (None, called_name.as_str()),
                };

                if excluded_names.contains(method_name) {
                    counts.excluded += 1;
                    continue;
                }
                let Some(candidates) = name_lookup.get(method_name) else {
                    counts.no_candidate += 1;
                    continue;
                };

                if candidates.len() == 1 {
                    counts.resolved += 1;
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
                let mut owner_hint_hit = false;
                if let Some(hint) = explicit_hint.or(implicit_hint) {
                    filtered = targets
                        .iter()
                        .copied()
                        .filter(|t| qname_to_owner.get(t).copied() == Some(hint))
                        .collect();
                    if !filtered.is_empty() {
                        targets = &filtered[..];
                        owner_hint_hit = true;
                    }
                }

                // Inheritance tier: a `self.method()` whose method isn't
                // defined on the caller's own type resolves to the method
                // *inherited* from an ancestor (EXTENDS / IMPLEMENTS). Only
                // fires for implicit (self) calls whose direct-owner filter
                // above found nothing — `obj.method()` is left alone (we
                // can't infer obj's type). Conservative: a unique inherited
                // definition resolves immediately; a diamond narrows the set
                // and defers to the later tiers.
                if !owner_hint_hit && targets.len() > 1 {
                    if let Some(owner) = implicit_hint {
                        if let Some(anc) = ancestors.get(owner) {
                            let inh: Vec<&str> = candidates
                                .iter()
                                .copied()
                                .filter(|t| {
                                    qname_to_owner
                                        .get(t)
                                        .copied()
                                        .is_some_and(|o| anc.contains(o))
                                })
                                .collect();
                            if inh.len() == 1 {
                                counts.resolved += 1;
                                counts.inherited += 1;
                                if inh[0] != caller_qn {
                                    out.push((caller_qn, inh[0], *line));
                                }
                                continue;
                            } else if !inh.is_empty() {
                                filtered = inh;
                                targets = &filtered[..];
                            }
                        }
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
                    counts.ambiguous += 1;
                    continue;
                }

                counts.resolved += 1;
                for &target in targets {
                    if target != caller_qn {
                        out.push((caller_qn, target, *line));
                    }
                }
            }
            (out, counts)
        })
        .collect();

    // Aggregate per-function counters into the pass-level stats.
    let mut stats = CallResolutionStats::default();
    for (_, c) in &per_fn {
        stats.total_calls += c.total;
        stats.excluded_noise += c.excluded;
        stats.no_candidate += c.no_candidate;
        stats.ambiguous_dropped += c.ambiguous;
        stats.resolved_call_sites += c.resolved;
        stats.resolved_via_inheritance += c.inherited;
    }

    // Merge into the final dedupe map sequentially — 200K inserts is ~5ms.
    let total: usize = per_fn.iter().map(|(v, _)| v.len()).sum();
    let mut seen: HashMap<(&str, &str), Vec<u32>> = HashMap::with_capacity(total);
    for (edges, _) in per_fn {
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
    stats.resolved_edges = result.len() as u64;
    if verbose {
        eprintln!(
            "[calls]     output build: {:.3}s ({} edges, {}/{} call sites resolved)",
            t_out.elapsed().as_secs_f64(),
            stats.resolved_edges,
            stats.resolved_call_sites,
            stats.total_calls.saturating_sub(stats.excluded_noise),
        );
    }
    (result, stats)
}

#[cfg(test)]
mod stats_tests {
    use super::*;
    use crate::code_tree::models::{FileInfo, FunctionInfo, TypeRelationship};

    fn func(qn: &str, file: &str, calls: &[(&str, u32)]) -> FunctionInfo {
        FunctionInfo {
            name: qn.rsplit(['.', ':']).next().unwrap_or(qn).to_string(),
            qualified_name: qn.to_string(),
            file_path: file.to_string(),
            calls: calls.iter().map(|(n, l)| (n.to_string(), *l)).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn stats_classify_every_call_site() {
        // `a.foo` calls `bar` twice (resolvable, single candidate → 1 edge),
        // an external name (no candidate), and a noise name (excluded).
        let functions = vec![
            func(
                "a.foo",
                "a.py",
                &[("bar", 1), ("bar", 2), ("external_thing", 3), ("noisy", 4)],
            ),
            func("a.bar", "a.py", &[]),
        ];
        let files = vec![FileInfo {
            path: "a.py".into(),
            ..Default::default()
        }];
        let mut noise = std::collections::HashSet::new();
        noise.insert("noisy");

        let (edges, stats) = build_call_edges(&functions, &files, &noise, 5, &[]);

        assert_eq!(stats.total_calls, 4);
        assert_eq!(stats.excluded_noise, 1);
        assert_eq!(stats.no_candidate, 1);
        assert_eq!(stats.ambiguous_dropped, 0);
        assert_eq!(stats.resolved_call_sites, 2); // two `bar` sites
        assert_eq!(stats.resolved_edges, 1); // collapsed to one a.foo→a.bar edge
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn inheritance_tier_resolves_self_call_to_ancestor_method() {
        // `m` exists on Base and Other. Derived extends Base. A self-call to
        // `m()` from Derived must resolve to the inherited Base.m — not Other.m
        // (which the same-file / global fallbacks could otherwise pick).
        let files = vec![FileInfo {
            path: "a.py".into(),
            ..Default::default()
        }];
        let functions = vec![
            func("mod.Base.m", "a.py", &[]),
            func("mod.Other.m", "a.py", &[]),
            func("mod.Derived.caller", "a.py", &[("m", 1)]),
        ];
        let rels = vec![TypeRelationship {
            source_type: "mod.Derived".into(),
            target_type: Some("mod.Base".into()),
            relationship: "extends".into(),
            methods: vec![],
        }];
        let noise = std::collections::HashSet::new();

        let (edges, stats) = build_call_edges(&functions, &files, &noise, 5, &rels);

        assert_eq!(stats.resolved_via_inheritance, 1);
        let pairs: Vec<(&str, &str)> = edges
            .iter()
            .map(|e| (e.caller.as_str(), e.callee.as_str()))
            .collect();
        assert!(pairs.contains(&("mod.Derived.caller", "mod.Base.m")));
        assert!(!pairs.iter().any(|(_, callee)| *callee == "mod.Other.m"));
    }

    #[test]
    fn no_type_relationships_means_no_inheritance_resolution() {
        // Same shape, but without the EXTENDS relationship the self-call to a
        // multi-owner `m` is left to the ordinary tiers (no inheritance pin).
        let files = vec![FileInfo {
            path: "a.py".into(),
            ..Default::default()
        }];
        let functions = vec![
            func("mod.Base.m", "a.py", &[]),
            func("mod.Other.m", "a.py", &[]),
            func("mod.Derived.caller", "a.py", &[("m", 1)]),
        ];
        let noise = std::collections::HashSet::new();

        let (_edges, stats) = build_call_edges(&functions, &files, &noise, 5, &[]);
        assert_eq!(stats.resolved_via_inheritance, 0);
    }
}
