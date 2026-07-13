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
    let (type_count, secondary_count) = np.node_type.as_ref().map_or_else(
        || (GraphRead::node_count(&graph.graph), 0),
        |node_type| {
            let primary = graph
                .type_indices
                .get(node_type)
                .map_or(0, |indices| indices.len());
            let secondary = graph
                .secondary_label_index
                .get(&crate::graph::schema::InternedKey::from_str(node_type))
                .map_or(0, Vec::len);
            (primary.saturating_add(secondary), secondary)
        },
    );

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
                        PropertyMatcher::Equals(_) | PropertyMatcher::EqualsParam(_) => {
                            return 1usize.saturating_add(secondary_count);
                        }
                        PropertyMatcher::In(vals) => {
                            if vals.is_empty() {
                                return 0;
                            }
                            let unique: HashSet<_> = vals.iter().collect();
                            if let Some(node_type) = np.node_type.as_deref() {
                                let hits: HashSet<_> = unique
                                    .into_iter()
                                    .filter_map(|value| {
                                        graph.lookup_by_id_readonly(node_type, value)
                                    })
                                    .collect();
                                return hits.len().saturating_add(secondary_count);
                            }
                            return unique.len();
                        }
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
                                    return results.len().saturating_add(secondary_count).max(1);
                                }
                                return 1;
                            }
                        }
                        PropertyMatcher::In(vals) => {
                            if vals.is_empty() {
                                return 0;
                            }
                            let key = (nt.clone(), prop.clone());
                            if graph.property_indices.contains_key(&key) {
                                let mut hits = HashSet::new();
                                for value in vals {
                                    if let Some(indices) = graph.lookup_by_index(nt, prop, value) {
                                        hits.extend(indices);
                                    }
                                }
                                return hits.len().saturating_add(secondary_count);
                            }
                        }
                        _ => {}
                    }
                }
            }
            // Per-property reduction. These props are all NON-indexed (an
            // indexed equality returned exact selectivity above). Equality on
            // a typed node uses the real distinct-value count (NDV) when
            // available — `type_count / ndv` — so a low-cardinality field
            // (bool ≈ /2, enum ≈ /k) isn't mis-rated as highly selective and a
            // high-cardinality one isn't under-rated. Falls back to the flat
            // ~100x heuristic when NDV is unavailable (no type, or type too
            // large to scan). Range/other filters use the gentler ~10x guess.
            let mut est = type_count;
            for (prop, matcher) in props {
                match matcher {
                    PropertyMatcher::Equals(_) | PropertyMatcher::EqualsParam(_) => {
                        match np
                            .node_type
                            .as_ref()
                            .and_then(|nt| graph.property_ndv(nt, prop))
                        {
                            Some(ndv) => est /= ndv.max(1),
                            None => est /= 100,
                        }
                    }
                    PropertyMatcher::In(values) => {
                        if values.is_empty() {
                            return 0;
                        }
                        let unique_count = values.iter().collect::<HashSet<_>>().len();
                        let reduced = match np
                            .node_type
                            .as_ref()
                            .and_then(|nt| graph.property_ndv(nt, prop))
                        {
                            Some(ndv) => est
                                .saturating_mul(unique_count.min(ndv))
                                .div_ceil(ndv.max(1)),
                            None => est / 10,
                        };
                        // Match cardinality is not the whole start-node cost:
                        // without an index, finding those matches still scans
                        // the type. Retain a conservative fraction of that
                        // scan cost so a unique-looking IN value cannot tie an
                        // O(1) id/index anchor and suppress pattern reversal.
                        let scan_cost_floor = type_count.div_ceil(10);
                        est = reduced.max(scan_cost_floor).max(1);
                    }
                    _ => est /= 10,
                }
            }
            est.max(1)
        }
    }
}

/// Reorder consecutive MATCH clauses by clear anchors, then edge cost.
///
/// For a span of `MATCH … MATCH …` clauses sharing a variable, stably promote
/// clauses with an `{id: X}` endpoint before unanchored clauses. If every
/// clause is anchored and cached connection cardinalities are available,
/// retain the existing finer edge-count ordering within that anchored span.
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
/// - All clauses in the span share at least one variable (otherwise no
///   join, no benefit).
/// - Anchor promotion is stable within anchored/unanchored classes. Edge-cost
///   sorting additionally requires a populated cache and every clause to be
///   anchorable/scorable; otherwise only the stable promotion is used.
/// - The cost ordering would actually change.
///
/// Runs *before* `optimize_pattern_start_node` so subsequent reversal
/// sees the new clause order and accumulates `bound_vars` correctly.
pub(super) fn reorder_match_clauses(query: &mut CypherQuery, graph: &DirGraph) {
    let edge_counts = graph
        .has_edge_type_counts_cache()
        .then(|| graph.get_edge_type_counts());

    // 0.9.35 (AgensGraph-inspired): when the label-pair connectivity
    // cache is also populated, use the per-triple `(src_type, edge_type,
    // tgt_type)` counts instead of the broader edge-type total. Drops
    // the cost-estimate error on label-asymmetric patterns from "all
    // edges of type R" to "only edges of type R between the matched
    // labels" — typically 10–100× tighter on Wikidata-shaped graphs.
    // Gating on `has_type_connectivity_cache()` mirrors the existing
    // `has_edge_type_counts_cache()` gate so plan-time stays O(1).
    let triple_counts: Option<HashMap<(String, String, String), usize>> =
        if edge_counts.is_some() && graph.has_type_connectivity_cache() {
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

        if !shares_variable_across(&query.clauses[i..j]) {
            i = j;
            continue;
        }

        // Prefer the existing, more precise edge-count ordering when every
        // clause is scoreable. Mixed anchored/unanchored spans deliberately
        // fall back to a two-class stable partition: it is safe, predictable,
        // and avoids pretending an unanchored scan has a trustworthy cost.
        let costs = edge_counts.as_ref().and_then(|counts| {
            (i..j)
                .map(|k| match &query.clauses[k] {
                    Clause::Match(m) => estimate_match_edge_cost(m, counts, triple_counts.as_ref()),
                    _ => unreachable!(),
                })
                .collect::<Option<Vec<_>>>()
        });
        let mut order: Vec<usize> = (i..j).collect();
        if let Some(costs) = costs {
            order.sort_by_key(|&absolute| (costs[absolute - i], absolute));
        } else {
            order.sort_by_key(|&absolute| {
                let Clause::Match(m) = &query.clauses[absolute] else {
                    unreachable!()
                };
                (!match_is_id_anchored(m), absolute)
            });
        }

        let already_sorted = order
            .iter()
            .enumerate()
            .all(|(offset, &absolute)| i + offset == absolute);
        if !already_sorted {
            let extracted: Vec<Clause> = query.clauses.drain(i..j).collect();
            for (offset, &absolute) in order.iter().enumerate() {
                query
                    .clauses
                    .insert(i + offset, extracted[absolute - i].clone());
            }
        }

        i = j;
    }
}

/// True when every pattern in a MATCH has a literal/parameter ID anchor on an
/// endpoint. This is intentionally narrower than general selectivity: it is
/// the only mixed-clause promotion shape this pass can prove independently of
/// graph statistics.
fn match_is_id_anchored(m: &MatchClause) -> bool {
    !m.patterns.is_empty()
        && m.patterns.iter().all(|pattern| {
            pattern.elements.first().is_some_and(is_id_anchored)
                || pattern.elements.last().is_some_and(is_id_anchored)
        })
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
        let (first, last) = match (pattern.elements.first(), pattern.elements.last()) {
            (Some(first), Some(last)) => (first, last),
            // Empty patterns can't be produced by the parser; bail rather
            // than panic in release if a pass ever emits one.
            _ => return None,
        };
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

/// A simple cyclic pattern (ring) extracted from a MATCH pattern's elements —
/// `k` distinct nodes joined by `k` clean single-typed edges, where the start
/// variable repeats exactly once (closing the ring). Cloned out of the AST so
/// the pass can re-root freely, then write a fresh `elements` vec back.
struct SimpleCycle {
    nodes: Vec<crate::graph::core::pattern_matching::NodePattern>,
    edges: Vec<crate::graph::core::pattern_matching::EdgePattern>,
    k: usize,
}

impl SimpleCycle {
    /// Recognise `[N, E, N, E, …, N]` (len `2k+1`, `k ≥ 2`) where the first and
    /// last node share a variable, every ring node has a *distinct* `Some`
    /// variable, and every edge is "clean" — single connection type, exactly
    /// one hop, no edge variable / properties / inline filter / path-info.
    /// Anything else returns `None` (⇒ the caller leaves the pattern untouched),
    /// which keeps the rewrite confined to the shape it can prove equivalent.
    fn detect(elements: &[PatternElement]) -> Option<SimpleCycle> {
        if elements.len() < 5 || elements.len().is_multiple_of(2) {
            return None;
        }
        let k = elements.len() / 2; // ring nodes == ring edges
        if k < 2 {
            return None;
        }

        // First and last must be nodes sharing a variable (the closing cycle).
        let (first_var, last_var) = match (&elements[0], elements.last()?) {
            (PatternElement::Node(a), PatternElement::Node(b)) => {
                (a.variable.as_ref()?, b.variable.as_ref()?)
            }
            _ => return None,
        };
        if first_var != last_var {
            return None;
        }

        let mut nodes = Vec::with_capacity(k);
        let mut edges = Vec::with_capacity(k);
        for (i, el) in elements.iter().enumerate() {
            match (i % 2, el) {
                // ring nodes are the even indices except the repeated last one
                (0, PatternElement::Node(np)) if i + 1 < elements.len() => nodes.push(np.clone()),
                (0, PatternElement::Node(_)) => {} // trailing repeat — skip
                (1, PatternElement::Edge(ep)) => edges.push(ep.clone()),
                _ => return None,
            }
        }
        if nodes.len() != k || edges.len() != k {
            return None;
        }

        // Every ring node distinct & named (a non-adjacent variable repeat would
        // be a figure-eight, not a simple ring — out of scope, bail).
        let mut seen = HashSet::with_capacity(k);
        for np in &nodes {
            let v = np.variable.as_ref()?;
            if !seen.insert(v.clone()) {
                return None;
            }
        }

        // Clean edges only — keeps re-rooting (which may flip directions)
        // provably result-preserving and avoids edge-binding/path concerns.
        // NB: `needs_path_info` is the parser's default (`true`) until a later
        // pass clears it — NOT a cleanliness signal — so it is deliberately not
        // checked. `var_length: None` already guarantees a single fixed edge.
        for ep in &edges {
            if ep.variable.is_some()
                || ep.connection_type.is_none()
                || ep.connection_types.is_some()
                || ep.var_length.is_some()
                || ep.properties.is_some()
                || ep.edge_filter.is_some()
            {
                return None;
            }
        }

        Some(SimpleCycle { nodes, edges, k })
    }

    /// Edge `j` connects `nodes[j]` and `nodes[(j+1) % k]` (per its written
    /// direction). Re-emit the ring as a linear `elements` vec rooted at
    /// `root`, walking forward or reflected so the *cheaper* incident edge of
    /// the root drives first. Reflected edges have their direction flipped so
    /// the traversal semantics are identical to the original ring.
    fn linearize(
        &self,
        root: usize,
        edge_counts: Option<&HashMap<String, usize>>,
    ) -> Vec<PatternElement> {
        use crate::graph::core::pattern_matching::EdgeDirection;
        let k = self.k;
        let cost = |ep: &crate::graph::core::pattern_matching::EdgePattern| -> usize {
            match (edge_counts, ep.connection_type.as_ref()) {
                (Some(m), Some(ct)) => m.get(ct).copied().unwrap_or(usize::MAX),
                _ => 0, // unknown ⇒ treat equal; forward wins the tie below
            }
        };
        // Forward drives edges[root]; reflected drives edges[root-1].
        let forward = cost(&self.edges[root]) <= cost(&self.edges[(root + k - 1) % k]);

        let flip = |mut ep: crate::graph::core::pattern_matching::EdgePattern| {
            ep.direction = match ep.direction {
                EdgeDirection::Outgoing => EdgeDirection::Incoming,
                EdgeDirection::Incoming => EdgeDirection::Outgoing,
                EdgeDirection::Both => EdgeDirection::Both,
            };
            ep
        };

        let mut out = Vec::with_capacity(2 * k + 1);
        out.push(PatternElement::Node(self.nodes[root].clone()));
        for m in 0..k {
            let (edge_idx, next_node) = if forward {
                ((root + m) % k, (root + m + 1) % k)
            } else {
                let ej = (root + k - 1 - m) % k;
                (ej, ej) // reflected: next node is the edge's lower-index endpoint
            };
            let ep = self.edges[edge_idx].clone();
            out.push(PatternElement::Edge(if forward { ep } else { flip(ep) }));
            out.push(PatternElement::Node(self.nodes[next_node].clone()));
        }
        out
    }
}

/// Re-root a *simple cyclic* pattern at its most-selective node.
///
/// A cycle such as
/// `(p:Person)-[:WORKS_AT]->(c:Company)-[:OWNS]->(pr:Project)<-[:CONTRIBUTES_TO]-(p)`
/// is evaluated left-to-right from `elements[0]`. Starting at the
/// largest-cardinality node (`p` — every Person) materialises a huge
/// intermediate set before the cycle closes. `optimize_pattern_start_node`
/// can't help: it only *reverses*, and a cycle's two ends are the same node, so
/// first/last selectivity are equal.
///
/// This pass rotates the ring so the **most-selective** node starts the walk,
/// and (when the edge-type-count cache is warm) orients it so the cheaper
/// incident edge drives first. The cycle-closing segment then lands on an
/// already-bound node, which the matcher confirms with an O(1) adjacency check
/// (`bound_target` → `expand_from_node`'s `target_hint`) rather than a full
/// expansion.
///
/// **Shape-gated for zero acyclic regression.** Fires ONLY on a simple ring of
/// clean single-typed edges whose start variable repeats exactly once, and only
/// re-roots when the new root is ≥`ROOT_GAIN`× more selective than the written
/// one (and only when that root isn't already `elements[0]`). Every other
/// pattern is left byte-identical, so acyclic queries are provably unaffected.
pub(super) fn reorder_cyclic_pattern_edges(query: &mut CypherQuery, graph: &DirGraph) {
    /// Re-root only on a clear selectivity win, to avoid churn on marginal
    /// cases where the cost proxy could mislead.
    const ROOT_GAIN: usize = 4;

    // Edge orientation is a refinement that needs the count cache; rooting only
    // needs node selectivity (always available). Skip the cache scan if cold.
    let edge_counts = if graph.has_edge_type_counts_cache() {
        Some(graph.get_edge_type_counts())
    } else {
        None
    };

    for clause in &mut query.clauses {
        // OPTIONAL MATCH excluded: its null-extension semantics make re-rooting
        // riskier and cyclic OPTIONAL patterns are vanishingly rare.
        let (patterns, path_assignments) = match clause {
            Clause::Match(m) => (&mut m.patterns, &m.path_assignments),
            _ => continue,
        };
        for (pi, pattern) in patterns.iter_mut().enumerate() {
            if path_assignments.iter().any(|pa| pa.pattern_index == pi) {
                continue; // path semantics depend on written order
            }
            let Some(ring) = SimpleCycle::detect(&pattern.elements) else {
                continue;
            };
            let sels: Vec<usize> = ring
                .nodes
                .iter()
                .map(|np| estimate_node_selectivity(np, graph))
                .collect();
            let root = (0..ring.k).min_by_key(|&j| sels[j]).unwrap_or(0);
            // Clear-win gate; also a no-op when the written root is already best
            // (root == 0 ⇒ sels[root] == sels[0] ⇒ condition false).
            if sels[root].saturating_mul(ROOT_GAIN) >= sels[0] {
                continue;
            }
            pattern.elements = ring.linearize(root, edge_counts.as_ref());
        }
    }
}
