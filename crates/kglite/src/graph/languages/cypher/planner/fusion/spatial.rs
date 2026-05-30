//! Spatial-join fusion — `wkt_*` distance/contains predicates fused into a
//! single bounded scan.
//!
//! Split out of the former monolithic `fusion.rs` (0.10.10).

use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::PatternElement;
use crate::graph::languages::cypher::ast::*;
use crate::graph::schema::DirGraph;

// ============================================================================
// Spatial-join fusion: MATCH (s:A), (w:B) WHERE contains(s, w) [AND rest]
// ============================================================================

/// Try to strip `contains(var, var)` from a predicate, returning
/// (container_var, probe_var, remainder_predicate).
/// Returns `None` if the predicate doesn't match the required shape.
///
/// Matches these AST shapes (see where_clause.rs `try_extract_contains_filter`):
///   - `contains(a, b) <> false` (parser truthy wrapper) → primary case
///   - `contains(a, b) <> false AND rest` or `rest AND contains(...)` → with remainder
///
/// Does NOT match: `NOT contains(...)`, `contains(a, point(…))` (constant point),
/// disjunctions, or any non-variable first/second arg.
fn extract_spatial_join_contains(
    pred: &Predicate,
) -> Option<(
    String,
    String,
    crate::graph::languages::cypher::ast::SpatialProbeKind,
    Option<Predicate>,
)> {
    match pred {
        Predicate::Comparison {
            left,
            operator: ComparisonOp::NotEquals,
            right: Expression::Literal(Value::Boolean(false)),
        } => {
            let (c, p, k) = extract_contains_call_vars(left)?;
            Some((c, p, k, None))
        }
        Predicate::And(l, r) => {
            if let Some((c, p, k, None)) = extract_spatial_join_contains(l) {
                return Some((c, p, k, Some((**r).clone())));
            }
            if let Some((c, p, k, None)) = extract_spatial_join_contains(r) {
                return Some((c, p, k, Some((**l).clone())));
            }
            None
        }
        _ => None,
    }
}

/// Match a `contains(Variable, ProbeExpr)` function call where ProbeExpr is
/// either a bare `Variable` (Location-style probe) or `centroid(Variable)`
/// (Centroid-style probe). Returns `(container_var, probe_var, probe_kind)`.
fn extract_contains_call_vars(
    expr: &Expression,
) -> Option<(
    String,
    String,
    crate::graph::languages::cypher::ast::SpatialProbeKind,
)> {
    use crate::graph::languages::cypher::ast::SpatialProbeKind;
    if let Expression::FunctionCall { name, args, .. } = expr {
        if name != "contains" || args.len() != 2 {
            return None;
        }
        let c = match &args[0] {
            Expression::Variable(n) => n.clone(),
            _ => return None,
        };
        let (p, kind) = match &args[1] {
            Expression::Variable(n) => (n.clone(), SpatialProbeKind::Location),
            // `centroid(probe_var)` — probe by computing the probe geometry's
            // centroid. Lets the fast path fire on the common
            // point-in-polygon-via-centroid pipeline.
            Expression::FunctionCall {
                name: inner_name,
                args: inner_args,
                ..
            } if inner_name == "centroid" && inner_args.len() == 1 => match &inner_args[0] {
                Expression::Variable(n) => (n.clone(), SpatialProbeKind::Centroid),
                _ => return None,
            },
            _ => return None,
        };
        if c == p {
            return None;
        }
        Some((c, p, kind))
    } else {
        None
    }
}

/// Rewrite spatial-join shapes into `Clause::SpatialJoin`.
///
/// Two shapes are recognized:
///
/// 1. **Single-MATCH** (the original `MATCH (a:T1), (b:T2) WHERE
///    contains(a, b) [AND rest]` form). Probe via location config.
///
/// 2. **Multi-MATCH** (`MATCH (p:T1) [WHERE pre1] MATCH (s:T2) WHERE
///    contains(s, centroid(p)) [AND rest]`). Probe via centroid of
///    the probe-side geometry. Common in point-in-polygon enrichment
///    pipelines (Sodir prospect → structural-element classification,
///    well → license area, …) which previously fell off the fast
///    path because the planner's old gate required a single MATCH.
///
/// Preconditions for the SpatialJoin rewrite (any miss → no rewrite):
/// - Two single-node patterns, each with `variable` and `node_type`.
/// - WHERE matches `contains(c, p) <> false` or `contains(c, centroid(p))
///   <> false` (parser's truthy wrapper), possibly ANDed with a residual.
/// - Container type has `SpatialConfig::geometry`. Probe type has
///   `SpatialConfig::location` (Location probe) or
///   `SpatialConfig::geometry` (Centroid probe).
/// - The two contains() variables bind to the two patterns (either order).
/// - For the multi-MATCH form, an optional WHERE between the two MATCH
///   clauses is folded into the SpatialJoin's residual `remainder` so
///   per-probe filters (e.g. `p.wkt_geometry IS NOT NULL`) survive.
pub(crate) fn fuse_spatial_join(query: &mut CypherQuery, graph: &DirGraph) {
    // The spatial-join executor builds its R-tree from the primary
    // `type_indices` of the container/probe types and silently drops any
    // `extra_labels` on those patterns. On a multi-label graph that misses
    // secondary-labelled nodes, so bail to the general `MATCH (a:T1),(b:T2)
    // WHERE contains(...)` path (two matcher scans + predicate), which is
    // multi-label correct. Single-label graphs are unaffected.
    if graph.has_secondary_labels {
        return;
    }
    let mut i = 0;
    while i < query.clauses.len() {
        if try_fuse_spatial_single_match(query, graph, i)
            || try_fuse_spatial_multi_match(query, graph, i)
        {
            // Cursor stays put — the rewritten SpatialJoin sits at i.
        }
        i += 1;
    }
}

/// Single-MATCH form: `MATCH (a:T1), (b:T2) WHERE contains(a, b) [AND rest]`.
/// Returns true iff a rewrite was committed at `i`.
fn try_fuse_spatial_single_match(query: &mut CypherQuery, graph: &DirGraph, i: usize) -> bool {
    if i + 1 >= query.clauses.len() {
        return false;
    }
    let eligible = matches!(
        (&query.clauses[i], &query.clauses[i + 1]),
        (Clause::Match(_), Clause::Where(_))
    );
    if !eligible {
        return false;
    }

    let (p0_var, p0_type, p1_var, p1_type) = {
        let mc = match &query.clauses[i] {
            Clause::Match(m) => m,
            _ => return false,
        };
        if mc.patterns.len() != 2
            || !mc.path_assignments.is_empty()
            || mc.limit_hint.is_some()
            || mc.distinct_node_hint.is_some()
        {
            return false;
        }
        let (v0, t0) = match extract_single_typed_node(&mc.patterns[0]) {
            Some(x) => x,
            None => return false,
        };
        let (v1, t1) = match extract_single_typed_node(&mc.patterns[1]) {
            Some(x) => x,
            None => return false,
        };
        (v0, t0, v1, t1)
    };

    let (container_var, probe_var, probe_kind, remainder) = {
        let w = match &query.clauses[i + 1] {
            Clause::Where(w) => w,
            _ => return false,
        };
        match extract_spatial_join_contains(&w.predicate) {
            Some(x) => x,
            None => return false,
        }
    };

    let (container_type, probe_type) = if container_var == p0_var && probe_var == p1_var {
        (p0_type.clone(), p1_type.clone())
    } else if container_var == p1_var && probe_var == p0_var {
        (p1_type.clone(), p0_type.clone())
    } else {
        return false;
    };

    if !spatial_schema_ok(graph, &container_type, &probe_type, probe_kind) {
        return false;
    }

    query.clauses.remove(i + 1);
    query.clauses[i] = Clause::SpatialJoin {
        container_var,
        probe_var,
        container_type,
        probe_type,
        probe_kind,
        remainder,
    };
    true
}

/// Multi-MATCH form: `MATCH (p:T1) [WHERE pre1] MATCH (s:T2) WHERE
/// contains(s, centroid(p)) [AND rest]`. Returns true iff a rewrite was
/// committed at `i`.
fn try_fuse_spatial_multi_match(query: &mut CypherQuery, graph: &DirGraph, i: usize) -> bool {
    use crate::graph::languages::cypher::ast::SpatialProbeKind;

    // Window: Match[i] [Where[i+1]] Match[i+ofs] Where[i+ofs+1].
    if i + 2 >= query.clauses.len() {
        return false;
    }
    let probe_pre_where_idx: Option<usize> = match (
        matches!(query.clauses.get(i + 1), Some(Clause::Where(_))),
        matches!(query.clauses.get(i + 2), Some(Clause::Match(_))),
    ) {
        (true, true) => Some(i + 1),
        (false, _) => None,
        _ => return false,
    };
    let m1_idx = probe_pre_where_idx.map_or(i + 1, |_| i + 2);
    let w_idx = m1_idx + 1;
    if !matches!(query.clauses.get(m1_idx), Some(Clause::Match(_))) {
        return false;
    }
    if !matches!(query.clauses.get(w_idx), Some(Clause::Where(_))) {
        return false;
    }

    // Both MATCH clauses must be single-pattern, single-node, no edges,
    // no path assignments, no hints.
    let extract_single = |c: &Clause| -> Option<(String, String)> {
        let mc = match c {
            Clause::Match(m) => m,
            _ => return None,
        };
        if mc.patterns.len() != 1
            || !mc.path_assignments.is_empty()
            || mc.limit_hint.is_some()
            || mc.distinct_node_hint.is_some()
        {
            return None;
        }
        extract_single_typed_node(&mc.patterns[0])
    };
    let (m0_var, m0_type) = match extract_single(&query.clauses[i]) {
        Some(x) => x,
        None => return false,
    };
    let (m1_var, m1_type) = match extract_single(&query.clauses[m1_idx]) {
        Some(x) => x,
        None => return false,
    };
    if m0_var == m1_var {
        return false;
    }

    // The trailing WHERE must hold a contains(c, centroid(p)) call (or
    // equivalent). Centroid probe is the only mode that makes sense
    // here — Location probe uses a single MATCH cartesian and is
    // already handled by `try_fuse_spatial_single_match`.
    let (container_var, probe_var, probe_kind, remainder) = {
        let w = match &query.clauses[w_idx] {
            Clause::Where(w) => w,
            _ => return false,
        };
        match extract_spatial_join_contains(&w.predicate) {
            Some(x) => x,
            None => return false,
        }
    };
    if probe_kind != SpatialProbeKind::Centroid {
        return false;
    }

    // Either MATCH may carry the container or the probe — the contains
    // call decides, not pattern position. Fail fast if the call vars
    // don't map cleanly onto the two MATCHes.
    let (cont_pat_type, probe_pat_type, probe_pat_is_first) =
        if container_var == m0_var && probe_var == m1_var {
            (m0_type.clone(), m1_type.clone(), false)
        } else if container_var == m1_var && probe_var == m0_var {
            (m1_type.clone(), m0_type.clone(), true)
        } else {
            return false;
        };
    // The pre-WHERE (if any) sits between the two MATCHes and references
    // the probe in the canonical Sodir shape. If the probe is the first
    // MATCH, the pre-WHERE references the container instead — still
    // valid; we fold it into the residual either way.
    let _ = probe_pat_is_first;

    if !spatial_schema_ok(graph, &cont_pat_type, &probe_pat_type, probe_kind) {
        return false;
    }

    // Fold the optional pre-WHERE between the two MATCHes into the
    // SpatialJoin's residual predicate so per-pattern filters
    // (e.g. `p.wkt_geometry IS NOT NULL`) still apply. The R-tree probe
    // naturally drops probes without geometry, but a user predicate may
    // filter more.
    let merged_remainder = match (probe_pre_where_idx, remainder) {
        (None, r) => r,
        (Some(idx), r) => {
            let pre = match &query.clauses[idx] {
                Clause::Where(w) => w.predicate.clone(),
                _ => return false,
            };
            Some(match r {
                Some(rest) => Predicate::And(Box::new(pre), Box::new(rest)),
                None => pre,
            })
        }
    };

    // Commit: remove the trailing WHERE, the second MATCH, and the
    // optional pre-WHERE; replace the first MATCH with the SpatialJoin.
    query.clauses.remove(w_idx);
    query.clauses.remove(m1_idx);
    if let Some(pre_idx) = probe_pre_where_idx {
        query.clauses.remove(pre_idx);
    }
    query.clauses[i] = Clause::SpatialJoin {
        container_var,
        probe_var,
        container_type: cont_pat_type,
        probe_type: probe_pat_type,
        probe_kind,
        remainder: merged_remainder,
    };
    true
}

/// Extract `(variable, node_type)` from a 1-element Node pattern.
fn extract_single_typed_node(
    pat: &crate::graph::core::pattern_matching::Pattern,
) -> Option<(String, String)> {
    if pat.elements.len() != 1 {
        return None;
    }
    match &pat.elements[0] {
        PatternElement::Node(np) => {
            let v = np.variable.as_ref()?.clone();
            let t = np.node_type.as_ref()?.clone();
            Some((v, t))
        }
        _ => None,
    }
}

/// Schema gate per probe-kind:
/// - Container always needs `SpatialConfig::geometry`.
/// - Location probe needs `SpatialConfig::location`.
/// - Centroid probe needs `SpatialConfig::geometry` (so we can compute centroid).
fn spatial_schema_ok(
    graph: &DirGraph,
    container_type: &str,
    probe_type: &str,
    probe_kind: crate::graph::languages::cypher::ast::SpatialProbeKind,
) -> bool {
    use crate::graph::languages::cypher::ast::SpatialProbeKind;
    let container_ok = graph
        .get_spatial_config(container_type)
        .is_some_and(|c| c.geometry.is_some());
    let probe_ok = match probe_kind {
        SpatialProbeKind::Location => graph
            .get_spatial_config(probe_type)
            .is_some_and(|c| c.location.is_some()),
        SpatialProbeKind::Centroid => graph
            .get_spatial_config(probe_type)
            .is_some_and(|c| c.geometry.is_some()),
    };
    container_ok && probe_ok
}

#[cfg(test)]
#[path = "../fusion_spatial_tests.rs"]
mod spatial_join_tests;
