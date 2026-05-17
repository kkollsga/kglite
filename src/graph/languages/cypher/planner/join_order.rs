//! Join-order optimisation — pick pattern-start nodes, reorder MATCH
//! patterns by estimated selectivity.

use super::super::ast::*;
use crate::graph::core::pattern_matching::{PatternElement, PropertyMatcher};
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;
use std::collections::{HashMap, HashSet};

pub(super) fn optimize_pattern_start_node(query: &mut CypherQuery, graph: &DirGraph) {
    use crate::graph::core::pattern_matching::EdgeDirection;

    // Track variables bound by earlier clauses so an unconstrained pattern
    // node like `(p)` in M2 — which will be pre-bound at runtime — is treated
    // as effectively-anchored (selectivity 1). Without this, the planner sees
    // `(p)-[:T]->(target:Type)` and reverses it because `(p)` is statically
    // unconstrained → looks worst-case, even though it'll resolve to a single
    // pre-bound NodeIndex when the executor reaches this clause.
    let mut bound_vars: HashSet<String> = HashSet::new();

    for clause in &mut query.clauses {
        let (patterns, path_assignments) = match clause {
            Clause::Match(m) => (&mut m.patterns, &m.path_assignments),
            Clause::OptionalMatch(m) => (&mut m.patterns, &m.path_assignments),
            // Other clauses don't introduce node bindings the optimizer cares
            // about; advance without modifying patterns.
            _ => continue,
        };
        for (pi, pattern) in patterns.iter_mut().enumerate() {
            if pattern.elements.len() < 3 {
                continue;
            }
            // Don't reverse patterns with path assignments — breaks path semantics
            if path_assignments.iter().any(|pa| pa.pattern_index == pi) {
                continue;
            }

            let first_node = match &pattern.elements[0] {
                PatternElement::Node(np) => np,
                _ => continue,
            };
            let last_node = match pattern.elements.last() {
                Some(PatternElement::Node(np)) => np,
                _ => continue,
            };

            // Reversing is safe for undirected and variable-length edges:
            // - `Both` flips to `Both` (identity).
            // - Var-length without path assignment is symmetric — `(a)-[*1..3]-(b)`
            //   is `(b)-[*1..3]-(a)` and `(a)-[*1..3]->(b)` reversed yields
            //   `(b)<-[*1..3]-(a)` (same edges traversed in reverse). Path-bound
            //   patterns are protected by the `path_assignments` check above.
            // No early-exit needed for direction/var-length anymore.

            let first_sel = estimate_node_selectivity_in_context(first_node, graph, &bound_vars);
            let last_sel = estimate_node_selectivity_in_context(last_node, graph, &bound_vars);

            // Only reverse if last node is significantly more selective (5× threshold).
            // A 5x advantage already saves 80% of expansion work. `saturating_mul`
            // because unconstrained nodes report `usize::MAX` and would otherwise
            // overflow.
            if last_sel.saturating_mul(5) >= first_sel {
                continue;
            }

            // Reverse: flip element order and flip each edge direction
            pattern.elements.reverse();
            for elem in &mut pattern.elements {
                if let PatternElement::Edge(ep) = elem {
                    ep.direction = match ep.direction {
                        EdgeDirection::Outgoing => EdgeDirection::Incoming,
                        EdgeDirection::Incoming => EdgeDirection::Outgoing,
                        EdgeDirection::Both => EdgeDirection::Both,
                    };
                }
            }
        }

        // Accumulate node variables introduced by this clause's patterns so
        // subsequent clauses see them as bound.
        for pattern in patterns.iter() {
            for elem in &pattern.elements {
                if let PatternElement::Node(np) = elem {
                    if let Some(ref v) = np.variable {
                        bound_vars.insert(v.clone());
                    }
                }
            }
        }
    }
}

/// Selectivity estimate that knows about variables bound by earlier clauses.
/// A pre-bound node resolves to a single NodeIndex at runtime, so its
/// effective candidate count is 1 — the most selective possible.
fn estimate_node_selectivity_in_context(
    np: &crate::graph::core::pattern_matching::NodePattern,
    graph: &DirGraph,
    bound_vars: &HashSet<String>,
) -> usize {
    if let Some(ref v) = np.variable {
        if bound_vars.contains(v) {
            return 1;
        }
    }
    estimate_node_selectivity(np, graph)
}

/// Estimate the number of candidate nodes for a node pattern.
/// Lower = more selective = better as start node.
pub(super) fn estimate_node_selectivity(
    np: &crate::graph::core::pattern_matching::NodePattern,
    graph: &DirGraph,
) -> usize {
    let type_count = np
        .node_type
        .as_ref()
        .and_then(|t| graph.type_indices.get(t))
        .map(|idx| idx.len())
        .unwrap_or(GraphRead::node_count(&graph.graph));

    // Unconstrained nodes (no type, no properties) match every node in the
    // graph — they represent the *worst* possible start node. Returning
    // `usize::MAX` ensures the optimizer never picks an unconstrained node
    // over a constrained one, regardless of how the graph is populated.
    // (On a freshly-created graph, `type_count = 0`, which would otherwise
    // make unconstrained nodes look maximally selective.)
    let unconstrained = np.node_type.is_none();
    // Floor typed-no-property and empty-property branches at 1 so they never
    // beat a legitimately-anchored node (`{id: X}` returns 1, pre-bound vars
    // also map to 1 in the in-context estimator). Without the floor, a typed
    // node on an empty graph reports 0 and the optimizer reverses toward it
    // even when the other end is a bound anchor.
    match &np.properties {
        None if unconstrained => usize::MAX,
        None => type_count.max(1),
        Some(props) if props.is_empty() && unconstrained => usize::MAX,
        Some(props) if props.is_empty() => type_count.max(1),
        Some(props) => {
            // {id: X} is always selectivity 1 regardless of type
            for (prop, matcher) in props {
                if prop == "id" {
                    match matcher {
                        PropertyMatcher::Equals(_) | PropertyMatcher::EqualsParam(_) => return 1,
                        PropertyMatcher::In(vals) => return vals.len(),
                        _ => {}
                    }
                }
            }
            // Check if any property has equality on an indexed field
            if let Some(ref nt) = np.node_type {
                for (prop, matcher) in props {
                    match matcher {
                        PropertyMatcher::Equals(val) => {
                            let key = (nt.clone(), prop.clone());
                            if graph.property_indices.contains_key(&key) {
                                if let Some(results) = graph.lookup_by_index(nt, prop, val) {
                                    return results.len().max(1);
                                }
                                return 1;
                            }
                        }
                        PropertyMatcher::In(vals) => return vals.len(),
                        _ => {}
                    }
                }
            }
            // Heuristic: equality filters on string properties typically have many
            // distinct values (~100x reduction each). Range/comparison filters are
            // gentler (~10x). Multiple filters multiply the reduction.
            let eq_count = props
                .values()
                .filter(|m| {
                    matches!(
                        m,
                        PropertyMatcher::Equals(_) | PropertyMatcher::EqualsParam(_)
                    )
                })
                .count();
            let other_count = props.len() - eq_count;
            let mut est = type_count;
            for _ in 0..eq_count {
                est /= 100;
            }
            for _ in 0..other_count {
                est /= 10;
            }
            est.max(1)
        }
    }
}

/// Reorder consecutive MATCH clauses by edge-type total-count cost.
///
/// For a span of `MATCH … MATCH …` clauses where every clause has an
/// `{id: X}` anchor on one endpoint and a known connection type, drive
/// the cheaper one first. The cost proxy is the connection type's total
/// edge count from `edge_type_counts_cache` — O(1) per clause when the
/// cache is populated (always for graphs loaded from disk; warmed during
/// build for in-memory graphs that exercise it).
///
/// **The motivating case (Wikidata, 124M nodes / 861M edges):**
/// ```cypher
/// MATCH (p)-[:P31]->({id:5})       -- ~80M P31 edges total
/// MATCH (p)-[:P27]->({id:183})     -- ~3M P27 edges total
/// RETURN p.title LIMIT 20
/// ```
/// Without this pass the executor enumerates 13.4M humans then filters
/// each by P27 — observed at ~500s. Driving from M2 first (3M Germans →
/// per-row P31 check) is ~25× cheaper.
///
/// Safety conditions (any miss → no reorder):
/// - At least 2 consecutive `Match` clauses (not OPTIONAL, no path
///   assignments).
/// - `edge_type_counts_cache` is populated (avoids triggering a fresh
///   O(E) scan at plan time).
/// - Every clause in the span has an `{id: X}` anchor on at least one
///   endpoint of every pattern, AND a connection type whose total-edge
///   count is known. This restricts us to queries where the proxy is a
///   real upper bound on expansion size.
/// - All clauses in the span share at least one variable (otherwise no
///   join, no benefit).
/// - The cost ordering would actually change.
///
/// Runs *before* `optimize_pattern_start_node` so subsequent reversal
/// sees the new clause order and accumulates `bound_vars` correctly.
pub(super) fn reorder_match_clauses(query: &mut CypherQuery, graph: &DirGraph) {
    if !graph.has_edge_type_counts_cache() {
        return;
    }
    let edge_counts = graph.get_edge_type_counts();

    // 0.9.35 (AgensGraph-inspired): when the label-pair connectivity
    // cache is also populated, use the per-triple `(src_type, edge_type,
    // tgt_type)` counts instead of the broader edge-type total. Drops
    // the cost-estimate error on label-asymmetric patterns from "all
    // edges of type R" to "only edges of type R between the matched
    // labels" — typically 10–100× tighter on Wikidata-shaped graphs.
    // Gating on `has_type_connectivity_cache()` mirrors the existing
    // `has_edge_type_counts_cache()` gate so plan-time stays O(1).
    let triple_counts: Option<HashMap<(String, String, String), usize>> =
        if graph.has_type_connectivity_cache() {
            graph.get_type_connectivity().map(|triples| {
                triples
                    .into_iter()
                    .map(|t| ((t.src, t.conn, t.tgt), t.count))
                    .collect()
            })
        } else {
            None
        };

    let mut i = 0;
    while i < query.clauses.len() {
        // Find a span of consecutive non-OPTIONAL MATCH clauses with no
        // path assignments. Stops at any other clause kind (WITH, WHERE,
        // RETURN, etc.) to preserve their semantic boundaries.
        let mut j = i;
        while j < query.clauses.len() {
            match &query.clauses[j] {
                Clause::Match(m) if m.path_assignments.is_empty() => j += 1,
                _ => break,
            }
        }
        if j - i < 2 {
            i = j.max(i + 1);
            continue;
        }

        // Estimate cost for each MATCH in the span. Bail on the whole span
        // if any clause is unscoreable — partial knowledge could mislead
        // a sort.
        let mut costs: Vec<usize> = Vec::with_capacity(j - i);
        let mut all_scored = true;
        for k in i..j {
            let m = match &query.clauses[k] {
                Clause::Match(m) => m,
                _ => unreachable!(),
            };
            match estimate_match_edge_cost(m, &edge_counts, triple_counts.as_ref()) {
                Some(c) => costs.push(c),
                None => {
                    all_scored = false;
                    break;
                }
            }
        }
        if !all_scored {
            i = j;
            continue;
        }

        if !shares_variable_across(&query.clauses[i..j]) {
            i = j;
            continue;
        }

        // Stable sort by (cost, original_index) so equal costs preserve
        // textual order (no churn).
        let mut order: Vec<(usize, usize)> = costs.iter().copied().enumerate().collect();
        order.sort_by_key(|&(orig, c)| (c, orig));

        let already_sorted = order
            .iter()
            .enumerate()
            .all(|(pos, &(orig, _))| pos == orig);
        if !already_sorted {
            let extracted: Vec<Clause> = query.clauses.drain(i..j).collect();
            for (offset, &(orig, _)) in order.iter().enumerate() {
                query.clauses.insert(i + offset, extracted[orig].clone());
            }
        }

        i = j;
    }
}

/// Cost proxy for a MATCH clause: sum of total edge counts (over all
/// connection types in its patterns), provided every pattern is
/// id-anchored. Returns `None` if the clause is unscoreable under the
/// safety rules in [`reorder_match_clauses`].
fn estimate_match_edge_cost(
    m: &MatchClause,
    edge_counts: &HashMap<String, usize>,
    triple_counts: Option<&HashMap<(String, String, String), usize>>,
) -> Option<usize> {
    let mut total: usize = 0;
    for pattern in &m.patterns {
        if pattern.elements.len() < 3 {
            // A node-only pattern carries no edge cost; ordering it
            // relative to edge-bearing patterns is meaningless under
            // this proxy. Bail.
            return None;
        }
        // Need at least one id-anchored endpoint on every pattern in
        // the clause. Mid-pattern nodes are not checked — typical case
        // is `(node)-[:T]->(node)`.
        let first = &pattern.elements[0];
        let last = pattern.elements.last().unwrap();
        if !is_id_anchored(first) && !is_id_anchored(last) {
            return None;
        }
        // Sum the edge count for every typed edge in the pattern.
        // Prefer the label-pair triple count (src_type, edge, tgt_type)
        // when both endpoints are labelled AND triple_counts is
        // populated — this drops the cost estimate from "all R edges"
        // to "R edges only between (T1, T2)", which on label-skewed
        // graphs (humans-in-Germany style queries) is the difference
        // between picking the right driving side and not.
        let elems = &pattern.elements;
        for idx in 0..elems.len() {
            let ep = match &elems[idx] {
                PatternElement::Edge(ep) => ep,
                _ => continue,
            };
            let ct = ep.connection_type.as_ref()?;
            let mut count: Option<usize> = None;
            if let Some(triples) = triple_counts {
                // Lookup neighbouring node-type labels. Pattern shape
                // is always (node, edge, node, edge, ...) so the
                // surrounding nodes are at idx-1 and idx+1. Fall back
                // to per-edge total when either side is untyped or the
                // (src, edge, tgt) triple isn't in the cache.
                let src_label = idx
                    .checked_sub(1)
                    .and_then(|i| elems.get(i))
                    .and_then(node_label);
                let tgt_label = elems.get(idx + 1).and_then(node_label);
                if let (Some(sl), Some(tl)) = (src_label, tgt_label) {
                    let key_fwd = (sl.clone(), ct.clone(), tl.clone());
                    let key_rev = (tl, ct.clone(), sl);
                    // Direction-agnostic for `()-[]-()` patterns:
                    // honour both directions, take the sum.
                    let fwd = triples.get(&key_fwd).copied().unwrap_or(0);
                    let rev = triples.get(&key_rev).copied().unwrap_or(0);
                    if fwd > 0 || rev > 0 {
                        count = Some(fwd + rev);
                    }
                }
            }
            let resolved = match count {
                Some(c) => c,
                None => *edge_counts.get(ct)?,
            };
            total = total.saturating_add(resolved);
        }
    }
    Some(total)
}

/// Extract the node-type label (e.g. `Person`) from a NodePattern
/// element. Returns `None` for edges, anonymous nodes, or nodes
/// without a label.
fn node_label(elem: &PatternElement) -> Option<String> {
    let np = match elem {
        PatternElement::Node(np) => np,
        _ => return None,
    };
    np.node_type.clone()
}

fn is_id_anchored(elem: &PatternElement) -> bool {
    let np = match elem {
        PatternElement::Node(np) => np,
        _ => return false,
    };
    let props = match &np.properties {
        Some(p) => p,
        None => return false,
    };
    props.iter().any(|(prop, matcher)| {
        prop == "id"
            && matches!(
                matcher,
                PropertyMatcher::Equals(_) | PropertyMatcher::EqualsParam(_)
            )
    })
}

fn shares_variable_across(clauses: &[Clause]) -> bool {
    let mut common: Option<HashSet<String>> = None;
    for clause in clauses {
        let m = match clause {
            Clause::Match(m) => m,
            _ => return false,
        };
        let vars: HashSet<String> = m
            .patterns
            .iter()
            .flat_map(|p| p.elements.iter())
            .filter_map(|e| match e {
                PatternElement::Node(np) => np.variable.clone(),
                _ => None,
            })
            .collect();
        common = Some(match common {
            None => vars,
            Some(prev) => prev.intersection(&vars).cloned().collect(),
        });
    }
    common.is_some_and(|s| !s.is_empty())
}

/// Reorder patterns within a MATCH clause so the most selective pattern runs first.
///
/// For `MATCH (n)-[:P31]->({id:6256}), (n)-[:P30]->({id:46})`, the pattern with
/// the more selective start node should execute first to minimize the number of
/// rows passed to subsequent patterns via shared-variable join.
///
/// Estimates selectivity by looking at the first node of each pattern (after
/// start-node optimization has already picked the best direction).
pub(super) fn reorder_match_patterns(query: &mut CypherQuery, graph: &DirGraph) {
    let mut bound_vars: HashSet<String> = HashSet::new();

    for clause in &mut query.clauses {
        let mc = match clause {
            Clause::Match(mc) => mc,
            Clause::OptionalMatch(mc) => {
                // OPTIONAL MATCH still binds vars for downstream clauses;
                // accumulate but don't reorder OPTIONAL MATCH patterns.
                for pat in mc.patterns.iter() {
                    for elem in &pat.elements {
                        if let PatternElement::Node(np) = elem {
                            if let Some(ref v) = np.variable {
                                bound_vars.insert(v.clone());
                            }
                        }
                    }
                }
                continue;
            }
            _ => continue,
        };
        if mc.patterns.len() < 2 || !mc.path_assignments.is_empty() {
            // Still accumulate vars even when not reordering.
            for pat in mc.patterns.iter() {
                for elem in &pat.elements {
                    if let PatternElement::Node(np) = elem {
                        if let Some(ref v) = np.variable {
                            bound_vars.insert(v.clone());
                        }
                    }
                }
            }
            continue;
        }
        // Estimate selectivity for each pattern based on its start node,
        // accounting for variables already bound by prior clauses.
        let mut pattern_scores: Vec<(usize, usize)> = mc
            .patterns
            .iter()
            .enumerate()
            .map(|(i, pat)| {
                let sel = if let Some(PatternElement::Node(np)) = pat.elements.first() {
                    estimate_node_selectivity_in_context(np, graph, &bound_vars)
                } else {
                    usize::MAX
                };
                (i, sel)
            })
            .collect();

        // Sort by selectivity (lower = more selective = should go first)
        pattern_scores.sort_by_key(|&(_, sel)| sel);

        // Only reorder if the order actually changes
        let already_ordered = pattern_scores
            .iter()
            .enumerate()
            .all(|(pos, &(idx, _))| pos == idx);
        if !already_ordered {
            let old_patterns = std::mem::take(&mut mc.patterns);
            mc.patterns = pattern_scores
                .iter()
                .map(|&(idx, _)| old_patterns[idx].clone())
                .collect();
        }

        // Accumulate vars from this clause's (possibly reordered) patterns.
        for pat in mc.patterns.iter() {
            for elem in &pat.elements {
                if let PatternElement::Node(np) = elem {
                    if let Some(ref v) = np.variable {
                        bound_vars.insert(v.clone());
                    }
                }
            }
        }
    }
}
