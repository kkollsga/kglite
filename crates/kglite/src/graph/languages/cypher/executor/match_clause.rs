//! Cypher executor — match_clause methods.

use super::super::ast::*;
use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{
    EdgeDirection, MatchBinding, NodePattern, Pattern, PatternElement, PatternExecutor,
    PatternMatch, PropertyMatcher,
};
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};

/// Collect the edge indices a [`PatternMatch`] consumed, deduplicated into
/// `out`. Fixed-length hops live on the compact `exact_path` trail (named
/// *and* anonymous — the matcher tracks it whenever `needs_path_info` is
/// set, the parser default for fixed edges); named fixed edges additionally
/// appear as `Edge` bindings, and var-length matches carry their hop list
/// in `VariableLengthPath` bindings. Used to enforce openCypher
/// relationship uniqueness (the trail rule) across the comma patterns of a
/// single MATCH clause.
pub(super) fn match_edge_indices(m: &PatternMatch, out: &mut Vec<EdgeIndex>) {
    let push = |edge: EdgeIndex, out: &mut Vec<EdgeIndex>| {
        if !out.contains(&edge) {
            out.push(edge);
        }
    };
    if let Some((_, path)) = m.exact_path.as_deref() {
        for hop in path {
            push(hop.edge, out);
        }
    }
    for (_, binding) in &m.bindings {
        match binding {
            MatchBinding::Edge { edge_index, .. } => push(*edge_index, out),
            MatchBinding::VariableLengthPath { path, .. } => {
                for hop in path {
                    push(hop.edge, out);
                }
            }
            _ => {}
        }
    }
}

/// Edge indices a freshly-created row consumed, read back off its bindings:
/// named edges plus every fixed-trail (`__fixed_path`) and var-length path
/// hop. Only valid as a *clause-local* edge set when the row was produced
/// by a single pattern of the current clause (no inherited bindings) — the
/// first-MATCH first-pattern case.
pub(super) fn row_edge_indices(row: &ResultRow) -> Vec<EdgeIndex> {
    let mut out = Vec::new();
    for (_, eb) in &row.edge_bindings {
        if !out.contains(&eb.edge_index) {
            out.push(eb.edge_index);
        }
    }
    for (_, pb) in &row.path_bindings {
        for hop in &pb.path {
            if !out.contains(&hop.edge) {
                out.push(hop.edge);
            }
        }
    }
    out
}

/// NULL-pad pattern-introduced variables on `row`: every node/edge variable
/// declared by the clause's patterns that the row doesn't already bind gets
/// an explicit `projected: NULL` entry. Recording the null (rather than
/// leaving the name absent) is what lets downstream consumers distinguish a
/// bound-but-null variable — e.g. a later MATCH re-using a relationship
/// variable must match NOTHING when it's null, and `bindings_compatible`
/// can only see that through the projected entry.
pub(super) fn null_pad_pattern_vars(row: &mut ResultRow, clause: &MatchClause) {
    for pattern in &clause.patterns {
        for elem in &pattern.elements {
            match elem {
                PatternElement::Node(np) => {
                    if let Some(ref var) = np.variable {
                        if !row.node_bindings.contains_key(var) && !row.projected.contains_key(var)
                        {
                            row.projected.insert(var.clone(), Value::Null);
                        }
                    }
                }
                PatternElement::Edge(ep) => {
                    if let Some(ref var) = ep.variable {
                        if !row.edge_bindings.contains_key(var) && !row.projected.contains_key(var)
                        {
                            row.projected.insert(var.clone(), Value::Null);
                        }
                    }
                }
            }
        }
    }
}

/// True when relationship uniqueness must be enforced across this MATCH
/// clause's comma patterns: at least two patterns contain an edge element.
/// (Within one pattern the matcher's `reuses_bound_relationship` already
/// enforces the trail rule; a single-edge-pattern clause has nothing to
/// cross-check, so the hot single-pattern path pays nothing.)
pub(super) fn clause_needs_rel_uniqueness(clause: &MatchClause) -> bool {
    patterns_need_rel_uniqueness(&clause.patterns)
}

/// Pattern-list form of [`clause_needs_rel_uniqueness`] — shared with the
/// `EXISTS { p1, p2, … }` subquery evaluation, whose comma patterns form
/// one clause scope under the same openCypher trail rule.
pub(super) fn patterns_need_rel_uniqueness(patterns: &[Pattern]) -> bool {
    patterns.len() > 1 && patterns.iter().filter(|p| p.elements.len() > 1).count() >= 2
}

/// Group-aware form for pattern subqueries: relationship uniqueness must
/// be enforced only when some single clause group (`groups[i]` — comma-
/// joined patterns share a group, each `MATCH` separator starts a new one)
/// contains two or more edge-bearing patterns. Groups are emitted in
/// non-decreasing order by the parser, so a run-length scan suffices.
pub(super) fn grouped_patterns_need_rel_uniqueness(patterns: &[Pattern], groups: &[usize]) -> bool {
    let mut run_group: Option<usize> = None;
    let mut run_edge_count = 0usize;
    for (pi, pattern) in patterns.iter().enumerate() {
        if pattern.elements.len() <= 1 {
            continue;
        }
        let group = groups.get(pi).copied().unwrap_or(0);
        if run_group == Some(group) {
            run_edge_count += 1;
            if run_edge_count >= 2 {
                return true;
            }
        } else {
            run_group = Some(group);
            run_edge_count = 1;
        }
    }
    false
}

/// Resolve a relationship variable already bound on the incoming row —
/// either as a live `EdgeBinding` (MATCH/WITH carry-through) or as a
/// projected `Value::Relationship` (e.g. `UNWIND collect(r) AS r`).
/// Returns `(source, target, edge_index)` of the stored edge.
pub(super) fn row_bound_edge(
    row: &ResultRow,
    var: &str,
) -> Option<(NodeIndex, NodeIndex, EdgeIndex)> {
    if let Some(eb) = row.edge_bindings.get(var) {
        return Some((eb.source, eb.target, eb.edge_index));
    }
    if let Some(Value::Relationship(rel)) = row.projected.get(var) {
        return Some((
            NodeIndex::new(rel.start_id as usize),
            NodeIndex::new(rel.end_id as usize),
            EdgeIndex::new(rel.id as usize),
        ));
    }
    None
}

/// Resolve a node variable carried on the incoming row only as a projected
/// VALUE — `Value::Node` (e.g. `UNWIND collect(n) AS n`, `WITH n` after the
/// fold pass rewrote bindings) or the transient `Value::NodeRef`. Both carry
/// the petgraph NodeIndex (see `materialize_node_value`). Returns `None` for
/// live `node_bindings` entries (the caller reads those directly) and for
/// any other value shape.
pub(super) fn row_projected_node(row: &ResultRow, var: &str) -> Option<NodeIndex> {
    match row.projected.get(var) {
        Some(Value::Node(nv)) => Some(NodeIndex::new(nv.id as usize)),
        Some(Value::NodeRef(i)) => Some(NodeIndex::new(*i as usize)),
        _ => None,
    }
}

/// Seed pattern node variables that are pre-bound on the row only as
/// projected `Value::Node` / `Value::NodeRef` values, so the
/// `PatternExecutor` anchors at that node instead of enumerating every
/// candidate. Purely a search-space constraint — `bindings_compatible`
/// enforces the node identity itself. Returns `None` (no allocation) when
/// nothing needs seeding — the common path.
fn seed_projected_node_vars(pattern: &Pattern, row: &ResultRow) -> Option<Bindings<NodeIndex>> {
    let mut extended: Option<Bindings<NodeIndex>> = None;
    for elem in &pattern.elements {
        let PatternElement::Node(np) = elem else {
            continue;
        };
        let Some(var) = np.variable.as_ref() else {
            continue;
        };
        if row.node_bindings.contains_key(var) {
            continue; // an existing binding always wins
        }
        let Some(idx) = row_projected_node(row, var) else {
            continue;
        };
        let ext = extended.get_or_insert_with(|| row.node_bindings.clone());
        if !ext.contains_key(var) {
            ext.insert(var.clone(), idx);
        }
    }
    extended
}

/// When a pattern re-uses a variable that is already bound on the incoming
/// row only as a projected value (openCypher: the pattern must then match
/// exactly that entity), seed the pattern so the `PatternExecutor` expands
/// around the bound entity instead of enumerating every candidate:
/// - a relationship variable bound as `EdgeBinding` / `Value::Relationship`
///   seeds its adjacent node variables from the stored edge's endpoints;
/// - a node variable bound as `Value::Node` / `Value::NodeRef` seeds
///   itself (via `seed_projected_node_vars`).
///
/// Purely a search-space constraint — `bindings_compatible` enforces the
/// identities themselves, including the undirected (`Both`) edge case this
/// helper skips because the orientation is ambiguous. Returns `None` (no
/// allocation) when no variable of the pattern is pre-bound — the common
/// path.
pub(super) fn seed_prebound_pattern_vars(
    pattern: &Pattern,
    row: &ResultRow,
) -> Option<Bindings<NodeIndex>> {
    let mut extended = seed_projected_node_vars(pattern, row);
    for (i, elem) in pattern.elements.iter().enumerate() {
        let PatternElement::Edge(ep) = elem else {
            continue;
        };
        if ep.var_length.is_some() {
            continue;
        }
        let Some(var) = ep.variable.as_deref() else {
            continue;
        };
        let Some((src, tgt, _)) = row_bound_edge(row, var) else {
            continue;
        };
        // Pattern-order endpoints: `Outgoing` means left-to-right.
        let (left, right) = match ep.direction {
            EdgeDirection::Outgoing => (src, tgt),
            EdgeDirection::Incoming => (tgt, src),
            EdgeDirection::Both => continue,
        };
        let mut seed = |node_elem: Option<&PatternElement>, idx: NodeIndex| {
            let Some(PatternElement::Node(np)) = node_elem else {
                return;
            };
            let Some(node_var) = np.variable.as_ref() else {
                return;
            };
            if row.node_bindings.contains_key(node_var) {
                return; // an existing binding always wins
            }
            let ext = extended.get_or_insert_with(|| row.node_bindings.clone());
            if !ext.contains_key(node_var) {
                ext.insert(node_var.clone(), idx);
            }
        };
        seed(i.checked_sub(1).and_then(|j| pattern.elements.get(j)), left);
        seed(pattern.elements.get(i + 1), right);
    }
    extended
}

/// True when a `count(...)` call is `count(*)` (or the empty-arg form,
/// treated as star elsewhere in the fused paths too). `count(var)`
/// returns false. The two forms diverge on unmatched upstream rows
/// under OPTIONAL MATCH: the null-padded row counts for `count(*)`
/// but not for `count(var)`.
fn count_call_is_star(args: &[Expression]) -> bool {
    args.is_empty() || matches!(args[0], Expression::Star)
}

/// Replace every `count(...)` function-call sub-tree in `expr` with a
/// literal: `count(*)` becomes `star_value`, `count(var)` becomes
/// `var_value`. Used by the fused OPTIONAL MATCH path to evaluate
/// derived expressions like `total - count(rp)` against a
/// per-upstream-row count without re-running the OPTIONAL expansion.
/// Only count() is substituted because that's the only aggregate the
/// fused path computes inline; other aggregates are rejected upstream
/// by `is_fusable_*_clause`.
fn substitute_count_with_value(expr: &Expression, star_value: i64, var_value: i64) -> Expression {
    let subst = |e: &Expression| Box::new(substitute_count_with_value(e, star_value, var_value));
    match expr {
        Expression::FunctionCall { name, args, .. } if name == "count" => {
            let value = if count_call_is_star(args) {
                star_value
            } else {
                var_value
            };
            Expression::Literal(Value::Int64(value))
        }
        Expression::Add(l, r) => Expression::Add(subst(l), subst(r)),
        Expression::Subtract(l, r) => Expression::Subtract(subst(l), subst(r)),
        Expression::Multiply(l, r) => Expression::Multiply(subst(l), subst(r)),
        Expression::Divide(l, r) => Expression::Divide(subst(l), subst(r)),
        Expression::Modulo(l, r) => Expression::Modulo(subst(l), subst(r)),
        Expression::Negate(inner) => Expression::Negate(subst(inner)),
        // Concat / Case / etc. — leave alone; the gating function only
        // accepts shapes this helper covers, so the fall-through path
        // never sees a containing aggregate.
        other => other.clone(),
    }
}

impl<'a> CypherExecutor<'a> {
    pub(super) fn pattern_match_to_row(&self, m: PatternMatch) -> ResultRow {
        let PatternMatch {
            bindings,
            exact_path,
        } = m;
        let binding_count = bindings.len();
        let mut row = ResultRow::with_capacity(binding_count, binding_count / 2, 0);

        if let Some(exact_path) = exact_path {
            let (source, path) = *exact_path;
            row.path_bindings.insert(
                "__fixed_path".to_string(),
                PathBinding {
                    source,
                    hops: path.len(),
                    path,
                },
            );
        }

        for (var, binding) in bindings {
            match binding {
                MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => {
                    row.node_bindings.insert(var, index);
                }
                MatchBinding::Edge {
                    source,
                    target,
                    edge_index,
                    ..
                } => {
                    row.edge_bindings.insert(
                        var,
                        EdgeBinding {
                            source,
                            target,
                            edge_index,
                        },
                    );
                }
                MatchBinding::VariableLengthPath {
                    source, hops, path, ..
                } => {
                    row.path_bindings
                        .insert(var, PathBinding { source, hops, path });
                }
            }
        }

        row
    }

    /// Expand a single upstream row through the OPTIONAL MATCH
    /// patterns, returning the rows produced. Same semantics as the
    /// per-row body of [`Self::execute_optional_match`] minus the
    /// outer `existing.rows.is_empty()` first-clause case (the
    /// streaming path only enters when the upstream is non-empty).
    /// Used by [`super::stream::optional_match`].
    pub(super) fn stream_expand_optional(
        &self,
        clause: &MatchClause,
        row: &ResultRow,
    ) -> Result<Vec<ResultRow>, String> {
        self.expand_optional_match_row(clause, row, 0)
    }

    /// Expand one upstream row through ALL comma patterns of an OPTIONAL
    /// MATCH as one joined unit: each pattern extends the working row set
    /// produced by the previous one (cross-join with shared-variable and
    /// relationship-uniqueness constraints), exactly like a regular
    /// multi-pattern MATCH. An empty return means the *joined* pattern set
    /// produced no match — the caller emits the single null-padded
    /// fallback row (openCypher: OPTIONAL MATCH succeeds or fails as a
    /// unit; per-pattern partial rows are not a thing). Earlier code
    /// expanded each pattern independently against the original row and
    /// unioned the results, yielding quasi-independent half-rows.
    ///
    /// `budget_rows_base` is the caller's already-materialized row count,
    /// so the budget reservation reflects the true total.
    fn expand_optional_match_row(
        &self,
        clause: &MatchClause,
        row: &ResultRow,
        budget_rows_base: usize,
    ) -> Result<Vec<ResultRow>, String> {
        let enforce_rel_uniqueness = clause_needs_rel_uniqueness(clause);
        let mut row_set: Vec<ResultRow> = vec![row.clone()];
        // Relationship-uniqueness bookkeeping, parallel to `row_set`: the
        // edges each working row consumed within THIS clause. Only
        // maintained when two or more comma patterns carry edges.
        let mut edge_sets: Vec<Vec<EdgeIndex>> = if enforce_rel_uniqueness {
            vec![Vec::new()]
        } else {
            Vec::new()
        };
        for pattern in &clause.patterns {
            if row_set.is_empty() {
                break;
            }
            let mut expanded: Vec<ResultRow> = Vec::new();
            let mut expanded_sets: Vec<Vec<EdgeIndex>> = Vec::new();
            for (ci, cur) in row_set.iter().enumerate() {
                // Resolve EqualsVar references against the working row so a
                // later pattern can reference variables bound by an earlier
                // one within the same OPTIONAL MATCH.
                let resolved;
                let pat = if Self::pattern_has_vars(pattern) {
                    resolved = self.resolve_pattern_vars(pattern, cur);
                    &resolved
                } else {
                    pattern
                };
                let seeded = seed_prebound_pattern_vars(pat, cur);
                let pre_bindings = seeded.as_ref().unwrap_or(&cur.node_bindings);
                let executor = PatternExecutor::with_bindings_and_params(
                    self.graph,
                    self.budget_probe_limit(None),
                    pre_bindings,
                    self.params,
                )
                .set_deadline(self.deadline)
                .set_cancel(self.cancel);
                let matches = executor.execute(pat)?;
                self.budget
                    .check_work(matches.len(), "OPTIONAL MATCH expansion")?;
                for m in &matches {
                    if !self.bindings_compatible(cur, m) {
                        continue;
                    }
                    if enforce_rel_uniqueness {
                        let mut m_edges = Vec::new();
                        match_edge_indices(m, &mut m_edges);
                        if m_edges.iter().any(|e| edge_sets[ci].contains(e)) {
                            continue;
                        }
                        let mut next_set = edge_sets[ci].clone();
                        next_set.extend(m_edges);
                        expanded_sets.push(next_set);
                    }
                    self.budget.reserve_rows(
                        budget_rows_base + expanded.len(),
                        1,
                        "OPTIONAL MATCH",
                    )?;
                    let mut new_row = cur.clone();
                    self.merge_match_into_row(&mut new_row, m);
                    expanded.push(new_row);
                }
            }
            row_set = expanded;
            if enforce_rel_uniqueness {
                edge_sets = expanded_sets;
            }
        }
        Ok(row_set)
    }

    /// Merge a PatternMatch's bindings into an existing ResultRow
    pub(super) fn merge_match_into_row(&self, row: &mut ResultRow, m: &PatternMatch) {
        if let Some((source, path)) = m.exact_path.as_deref() {
            row.path_bindings.insert(
                "__fixed_path".to_string(),
                PathBinding {
                    source: *source,
                    hops: path.len(),
                    path: path.clone(),
                },
            );
        }
        for (var, binding) in &m.bindings {
            match binding {
                MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => {
                    row.node_bindings.insert(var.clone(), *index);
                }
                MatchBinding::Edge {
                    source,
                    target,
                    edge_index,
                    ..
                } => {
                    row.edge_bindings.insert(
                        var.clone(),
                        EdgeBinding {
                            source: *source,
                            target: *target,
                            edge_index: *edge_index,
                        },
                    );
                }
                MatchBinding::VariableLengthPath {
                    source, hops, path, ..
                } => {
                    row.path_bindings.insert(
                        var.clone(),
                        PathBinding {
                            source: *source,
                            hops: *hops,
                            path: path.clone(),
                        },
                    );
                }
            }
        }
    }

    /// Synthesize a PathBinding from a multi-hop pattern.
    /// Iterates ALL pattern elements to capture every hop, not just the first.
    pub(super) fn synthesize_path_from_pattern(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        row: &ResultRow,
    ) -> Option<PathBinding> {
        let mut node_vars: Vec<&str> = Vec::new();
        let mut edge_vars: Vec<Option<&str>> = Vec::new();
        let mut edge_element_indices: Vec<usize> = Vec::new();
        for (element_index, elem) in pattern.elements.iter().enumerate() {
            match elem {
                PatternElement::Node(np) => {
                    if let Some(ref v) = np.variable {
                        node_vars.push(v);
                    }
                }
                PatternElement::Edge(ep) => {
                    edge_vars.push(ep.variable.as_deref());
                    edge_element_indices.push(element_index);
                }
            }
        }
        if node_vars.len() < 2 || edge_vars.is_empty() {
            return None;
        }
        let source_idx = row.node_bindings.get(node_vars[0])?;

        // Build full path: for each edge, record the target node and edge type
        let mut path = Vec::with_capacity(edge_vars.len());
        for (i, edge_var) in edge_vars.iter().enumerate() {
            let node_idx = row.node_bindings.get(node_vars[i + 1])?;
            let binding_name = edge_var
                .map(str::to_string)
                // PatternExecutor names internal fixed-edge bindings with
                // the following node element's index.
                .unwrap_or_else(|| format!("__anon_edge_{}", edge_element_indices[i] + 1));
            let edge = row.edge_bindings.get(&binding_name)?;
            path.push(crate::graph::core::pattern_matching::PathHop {
                node: *node_idx,
                edge: edge.edge_index,
                connection_type: self
                    .graph
                    .graph
                    .edge_weight(edge.edge_index)?
                    .connection_type,
            });
        }

        Some(PathBinding {
            source: *source_idx,
            hops: edge_vars.len(),
            path,
        })
    }

    // ========================================================================
    // OPTIONAL MATCH
    // ========================================================================

    pub(super) fn execute_optional_match(
        &self,
        clause: &MatchClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        if existing.rows.is_empty() {
            // OPTIONAL MATCH as first clause: try regular match, but if
            // nothing matches, return one row with all variables set to NULL
            let columns = existing.columns.clone();
            let result = self.execute_match(clause, existing, None)?;
            if !result.rows.is_empty() {
                return Ok(result);
            }
            let mut null_row = ResultRow::new();
            for pattern in &clause.patterns {
                for elem in &pattern.elements {
                    match elem {
                        PatternElement::Node(np) => {
                            if let Some(ref var) = np.variable {
                                null_row.projected.insert(var.clone(), Value::Null);
                            }
                        }
                        PatternElement::Edge(ep) => {
                            if let Some(ref var) = ep.variable {
                                null_row.projected.insert(var.clone(), Value::Null);
                            }
                        }
                    }
                }
            }
            return Ok(ResultSet {
                rows: vec![null_row],
                columns,
                lazy_return_items: None,
            });
        }

        let mut new_rows = Vec::with_capacity(existing.rows.len());

        for row in &existing.rows {
            let expanded = self.expand_optional_match_row(clause, row, new_rows.len())?;
            if expanded.is_empty() {
                // The joined pattern set produced no match: keep the row,
                // recording an explicit NULL for every pattern variable so
                // downstream clauses see them as bound-but-null.
                self.budget
                    .reserve_rows(new_rows.len(), 1, "OPTIONAL MATCH")?;
                let mut keep = row.clone();
                null_pad_pattern_vars(&mut keep, clause);
                new_rows.push(keep);
            } else {
                new_rows.extend(expanded);
            }
        }

        Ok(ResultSet {
            rows: new_rows,
            columns: existing.columns,
            lazy_return_items: None,
        })
    }

    /// Fast-path count for simple node-edge-node patterns when one end is pre-bound.
    /// Returns Some(count) if the fast-path applies, None to fall back to PatternExecutor.
    ///
    /// For pattern `(a:Type)-[:REL]->(b)` where `b` is already bound in the row:
    /// Instead of scanning all Type nodes and checking edges (O(|Type|)),
    /// traverse edges directly from the bound node (O(degree)).
    /// Fast path for EXISTS / NOT EXISTS: when the subquery is a single
    /// 3-element pattern (node-edge-node) with exactly one node already bound
    /// from the outer row, we can check edge existence directly via
    /// `edges_directed()` instead of creating a full PatternExecutor.
    /// Returns `Some(true/false)` if the fast path applies, `None` otherwise.
    pub(super) fn try_fast_exists_check(
        &self,
        patterns: &[Pattern],
        where_clause: &Option<Box<Predicate>>,
        row: &ResultRow,
    ) -> Option<Result<bool, String>> {
        if patterns.len() != 1 {
            return None;
        }
        let pattern = &patterns[0];
        if pattern.elements.len() != 3 {
            return None;
        }

        let node_a = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return None,
        };
        let edge = match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return None,
        };
        let node_b = match &pattern.elements[2] {
            PatternElement::Node(np) => np,
            _ => return None,
        };

        // Skip variable-length edges and edge property filters
        if edge.var_length.is_some() || edge.properties.is_some() {
            return None;
        }

        // A pre-bound relationship variable pins the pattern to exactly
        // that edge; the direct edges_directed sweep below never checks
        // edge identity, so fall back to the full executor (whose
        // bindings_compatible enforces it).
        if let Some(var) = edge.variable.as_deref() {
            if row_bound_edge(row, var).is_some() {
                return None;
            }
        }

        // Same for node variables carried only as projected VALUES
        // (`UNWIND collect(n) AS n` → Value::Node, or an OPTIONAL MATCH
        // miss → projected Null): they constrain the pattern to exactly
        // that node (or to nothing, for Null). The sweep below only
        // consults `node_bindings`, so fall back to the full executor.
        for np in [node_a, node_b] {
            if let Some(var) = np.variable.as_deref() {
                if !row.node_bindings.contains_key(var) && row.projected.contains_key(var) {
                    return None;
                }
            }
        }

        // Determine which node is bound from the outer row
        let a_bound = node_a
            .variable
            .as_ref()
            .and_then(|v| row.node_bindings.get(v).copied());
        let b_bound = node_b
            .variable
            .as_ref()
            .and_then(|v| row.node_bindings.get(v).copied());

        let (bound_idx, other_node, other_var, direction) = match (a_bound, b_bound) {
            (Some(idx), None) => {
                let dir = match edge.direction {
                    EdgeDirection::Outgoing => Direction::Outgoing,
                    EdgeDirection::Incoming => Direction::Incoming,
                    EdgeDirection::Both => return None,
                };
                (idx, node_b, &node_b.variable, dir)
            }
            (None, Some(idx)) => {
                let dir = match edge.direction {
                    EdgeDirection::Outgoing => Direction::Incoming,
                    EdgeDirection::Incoming => Direction::Outgoing,
                    EdgeDirection::Both => return None,
                };
                (idx, node_a, &node_a.variable, dir)
            }
            _ => return None, // both bound or neither — fall back
        };

        let interned_conn = edge.connection_type.as_deref().map(InternedKey::from_str);

        // Pre-allocate a mutable row for WHERE evaluation (avoids clone per edge)
        let (has_where, mut eval_row) = if where_clause.is_some() {
            let mut r = row.clone(); // single clone
            if let Some(ref var) = other_var {
                r.node_bindings.insert(var.clone(), NodeIndex::new(0)); // placeholder
            }
            (true, r)
        } else {
            (false, ResultRow::new()) // unused placeholder
        };

        for edge_ref in
            self.graph
                .graph
                .edges_directed_filtered(bound_idx, direction, interned_conn)
        {
            if let Some(ik) = interned_conn {
                if edge_ref.weight().connection_type != ik {
                    continue;
                }
            }

            let other_idx = if direction == Direction::Outgoing {
                edge_ref.target()
            } else {
                edge_ref.source()
            };

            // Check target node type (O(1) mmap read, no materialization)
            if let Some(ref req_type) = other_node.node_type {
                if let Some(nt) = self.graph.graph.node_type_of(other_idx) {
                    if self.graph.interner.resolve(nt) != req_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Check target node inline properties — bail to slow path
            // for non-trivial matchers (EqualsParam, EqualsVar, etc.)
            if let Some(ref props) = other_node.properties {
                if let Some(nd) = self.graph.graph.node_weight(other_idx) {
                    let mut all_match = true;
                    // Resolve aliases against the target node's type so
                    // `{id: 20}` / `{nid: 'Q76'}` / `{label: 'X'}` /
                    // `{title: 'X'}` all reach the right column.
                    // Without this, get_property("id") misses because
                    // id lives in the id_column, not the regular
                    // property map — which silently dropped EXISTS
                    // inline-property predicates before.
                    let tgt_type_str = nd.node_type_str(&self.graph.interner);
                    for (key, matcher) in props {
                        let resolved = self.graph.resolve_alias(tgt_type_str, key);
                        let val: Option<std::borrow::Cow<'_, Value>> = if resolved == "id" {
                            Some(nd.id())
                        } else if resolved == "title" {
                            Some(nd.title())
                        } else if let Some(v) = nd.get_property(resolved) {
                            // Stored property wins (KG-1).
                            Some(v)
                        } else {
                            // Structural convenience fallback for soft aliases.
                            match crate::graph::schema::soft_alias_fallback(resolved) {
                                Some(crate::graph::schema::SoftAliasFallback::Title) => {
                                    Some(nd.title())
                                }
                                Some(crate::graph::schema::SoftAliasFallback::TypeString) => {
                                    Some(std::borrow::Cow::Owned(Value::String(
                                        tgt_type_str.to_string(),
                                    )))
                                }
                                None => None,
                            }
                        };
                        let ok = match matcher {
                            PropertyMatcher::Equals(expected) => val.as_deref().is_some_and(|v| {
                                crate::graph::core::filtering::values_equal(v, expected)
                            }),
                            PropertyMatcher::In(values) => val.as_deref().is_some_and(|v| {
                                values
                                    .iter()
                                    .any(|exp| crate::graph::core::filtering::values_equal(v, exp))
                            }),
                            // Complex matchers — fall back to slow path
                            _ => return None,
                        };
                        if !ok {
                            all_match = false;
                            break;
                        }
                    }
                    if !all_match {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Check WHERE clause — reuse pre-allocated row, just update binding
            if has_where {
                if let Some(ref var) = other_var {
                    eval_row.node_bindings.insert(var.clone(), other_idx);
                }
                match self.evaluate_predicate(
                    where_clause
                        .as_ref()
                        .expect("invariant: has_where guards Some(where_clause)"),
                    &eval_row,
                ) {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(e) => return Some(Err(e)),
                }
            }

            return Some(Ok(true)); // Found a match
        }
        Some(Ok(false)) // No match found
    }

    /// Count matches of a simple 3-element `Node-Edge-Node` pattern from a
    /// bound endpoint, without materializing rows. Multi-edges between the
    /// same pair each count. Returns `Ok(None)` when the pattern shape isn't
    /// supported (caller falls back to the full PatternExecutor).
    pub(super) fn try_count_simple_pattern(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        bindings: &Bindings<NodeIndex>,
    ) -> Result<Option<i64>, String> {
        self.count_simple_pattern_from_bound(pattern, bindings, false)
    }

    /// Distinct-peer variant of `try_count_simple_pattern`: returns the number
    /// of distinct peer NodeIndices reachable along the edge from the bound
    /// node, instead of the raw edge count. Used for `count(DISTINCT v)`
    /// where `v` is the unbound node variable. Multi-edges between the same
    /// pair collapse to one. Undirected `[r]-` patterns dedup peers across
    /// both directions (a node reachable both ways counts once).
    pub(super) fn try_count_distinct_peers(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        bindings: &Bindings<NodeIndex>,
    ) -> Result<Option<i64>, String> {
        self.count_simple_pattern_from_bound(pattern, bindings, true)
    }

    /// Shared implementation behind [`Self::try_count_simple_pattern`]
    /// (`distinct_peers = false`, raw edge count) and
    /// [`Self::try_count_distinct_peers`] (`distinct_peers = true`, distinct
    /// peer count). Exactly one copy of the direction mapping and filter
    /// plumbing lives here — the two pre-unification copies had drifted:
    /// one post-filtered `connection_type`, the other trusted
    /// `edges_directed_filtered` to do it and silently over-counted edges
    /// of other connection types on memory/mapped storage.
    fn count_simple_pattern_from_bound(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        bindings: &Bindings<NodeIndex>,
        distinct_peers: bool,
    ) -> Result<Option<i64>, String> {
        // Only handle simple 3-element patterns: Node-Edge-Node
        if pattern.elements.len() != 3 {
            return Ok(None);
        }

        let node_a = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return Ok(None),
        };
        let edge = match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return Ok(None),
        };
        let node_b = match &pattern.elements[2] {
            PatternElement::Node(np) => np,
            _ => return Ok(None),
        };

        // Don't use fast-path for variable-length edges or edge property filters
        if edge.var_length.is_some() || edge.properties.is_some() {
            return Ok(None);
        }

        // Determine which end is bound
        let a_bound = node_a
            .variable
            .as_ref()
            .and_then(|v| bindings.get(v).copied());
        let b_bound = node_b
            .variable
            .as_ref()
            .and_then(|v| bindings.get(v).copied());

        // We need exactly one end bound for the fast-path to help.
        // Undirected `[r]-` patterns count both incoming and outgoing —
        // both directions are swept and summed (dedup'd for distinct peers).
        //
        // Contract: the caller guarantees that the bound NodeIndex satisfies
        // any property filter on the bound side of the pattern (the upstream
        // PatternExecutor that produced the binding already applied those
        // filters). We therefore ignore the bound node's `properties` here
        // and only consult `other_props` for the unbound peer. Earlier code
        // bailed out (`return Ok(None)`) when the bound side had properties,
        // which silently produced 0 in the `.unwrap_or(0)` fused callers.
        type CountFastPath<'p> = (
            NodeIndex,
            &'p Option<String>,
            &'p Option<HashMap<String, PropertyMatcher>>,
            &'p [Direction],
        );
        let (bound_idx, other_type, other_props, traverse_dirs): CountFastPath =
            match (a_bound, b_bound) {
                (None, Some(b_idx)) => {
                    let dirs: &[Direction] = match edge.direction {
                        EdgeDirection::Outgoing => &[Direction::Incoming], // (a)->b: b has incoming
                        EdgeDirection::Incoming => &[Direction::Outgoing], // (a)<-b: b has outgoing
                        EdgeDirection::Both => &[Direction::Outgoing, Direction::Incoming],
                    };
                    (b_idx, &node_a.node_type, &node_a.properties, dirs)
                }
                (Some(a_idx), None) => {
                    let dirs: &[Direction] = match edge.direction {
                        EdgeDirection::Outgoing => &[Direction::Outgoing],
                        EdgeDirection::Incoming => &[Direction::Incoming],
                        EdgeDirection::Both => &[Direction::Outgoing, Direction::Incoming],
                    };
                    (a_idx, &node_b.node_type, &node_b.properties, dirs)
                }
                _ => return Ok(None), // both bound or neither bound — fall back
            };

        let conn_type = edge.connection_type.as_deref();
        let interned_conn = conn_type.map(InternedKey::from_str);
        let interned_other_type = other_type.as_ref().map(|t| InternedKey::from_str(t));

        // Fast path (raw counts only): when no property filters on the
        // unbound node *and* no inline edge filter, use count_edges_filtered
        // which avoids EdgeData materialization entirely. On disk with
        // sorted CSR: binary search + sequential count (zero allocations).
        // Distinct-peer counting can't use it (needs peer identity, not an
        // edge count); when a filter is set we fall through — the slow loop
        // below checks each edge without ever building a row.
        if !distinct_peers && other_props.is_none() && edge.edge_filter.is_none() {
            let mut count: usize = 0;
            for &dir in traverse_dirs {
                count = count.saturating_add(self.graph.graph.count_edges_filtered(
                    bound_idx,
                    dir,
                    interned_conn,
                    interned_other_type,
                    self.deadline,
                )?);
            }
            return Ok(Some(count as i64));
        }

        // Slow path: iterate incident edges. This loop can cover millions of
        // edges for hub nodes (Q5 has ~40 M incoming P31 edges), so check the
        // deadline every 1 M iterations. For undirected `[r]-` patterns we
        // sweep both directions.
        let pe = PatternExecutor::new_lightweight_with_params(self.graph, None, self.params);
        let mut count: i64 = 0;
        let mut peers: HashSet<NodeIndex> = HashSet::new();
        let mut iter: usize = 0;
        // The inline filter (if any) was pushed by the planner from a
        // downstream WHERE; apply it per edge so the count reflects the
        // post-filter row set.
        let edge_filter = edge.edge_filter.as_ref();

        for &dir in traverse_dirs {
            // Translate the matcher's `direction` into the
            // peer_is_start boolean RelEdgePredicate works with.
            // bound_idx is always the anchor; `dir` describes its
            // outward direction. For `Source` anchor:
            //   Outgoing → peer = edge.target → peer_is_start = false
            //   Incoming → peer = edge.source → peer_is_start = true
            // For `Target` anchor (right-bound) it's reversed.
            let peer_is_start = if let Some(f) = edge_filter {
                use crate::graph::core::pattern_matching::pattern::AnchorSide;
                match (f.anchor, dir) {
                    (AnchorSide::Source, Direction::Outgoing) => false,
                    (AnchorSide::Source, Direction::Incoming) => true,
                    (AnchorSide::Target, Direction::Outgoing) => true,
                    (AnchorSide::Target, Direction::Incoming) => false,
                }
            } else {
                false
            };

            for edge_ref in self
                .graph
                .graph
                .edges_directed_filtered(bound_idx, dir, interned_conn)
            {
                iter += 1;
                if iter.is_multiple_of(1 << 20) {
                    self.check_deadline()?;
                }
                // `edges_directed_filtered` is a hint — the disk backend
                // pre-filters by connection type, but the memory/mapped
                // backends return every edge in the given direction (see
                // the trait contract in storage/mod.rs), so we must
                // post-filter on `connection_type` here.
                if let Some(required_conn) = interned_conn {
                    if edge_ref.weight().connection_type != required_conn {
                        continue;
                    }
                }
                let other_idx = if dir == Direction::Outgoing {
                    edge_ref.target()
                } else {
                    edge_ref.source()
                };
                // Distinct peers: a peer that already passed all filters via
                // another edge is settled — skip the filter work.
                if distinct_peers && peers.contains(&other_idx) {
                    continue;
                }

                if let Some(required_type) = interned_other_type {
                    if let Some(nt) = self.graph.graph.node_type_of(other_idx) {
                        if nt != required_type {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                if let Some(filter) = edge_filter {
                    let edge_data = edge_ref.weight();
                    let conn_ty = edge_data.connection_type;
                    let edge_source = edge_ref.source();
                    let edge_target = edge_ref.target();
                    if !filter.predicate.eval(
                        conn_ty,
                        peer_is_start,
                        edge_source,
                        edge_target,
                        &|prop: &str| edge_data.get_property(prop).cloned(),
                    ) {
                        continue;
                    }
                }

                if let Some(ref props) = other_props {
                    if !pe.node_matches_properties_pub(other_idx, props) {
                        continue;
                    }
                }

                if distinct_peers {
                    peers.insert(other_idx);
                } else {
                    count += 1;
                }
            }
        }

        Ok(Some(if distinct_peers {
            peers.len() as i64
        } else {
            count
        }))
    }

    /// Count matches for a 5-element pattern (a)-[e1]->(b)<-[e2]-(c)
    /// from a bound first node, without materializing intermediate rows.
    /// Traverses: first_node --e1--> middle_nodes --e2--> count last nodes.
    pub(super) fn count_two_hop_pattern(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        first_idx: NodeIndex,
    ) -> Result<i64, String> {
        self.count_two_hop_from_anchor(pattern, first_idx, false)
    }

    /// Count matches for a 5-element pattern traversed in reverse:
    /// (a)-[e1]->(b)-[e2]->(c) counted from c (position 4) backward.
    /// Reads elements [3],[2],[1],[0] with flipped edge directions.
    pub(super) fn count_two_hop_pattern_reverse(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        last_idx: NodeIndex,
    ) -> Result<i64, String> {
        self.count_two_hop_from_anchor(pattern, last_idx, true)
    }

    /// Shared implementation behind [`Self::count_two_hop_pattern`]
    /// (`reverse = false`, anchored at element 0) and
    /// [`Self::count_two_hop_pattern_reverse`] (`reverse = true`, anchored
    /// at element 4 with both edge directions flipped). One copy of the
    /// direction mapping and connection-type post-filtering.
    fn count_two_hop_from_anchor(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        anchor_idx: NodeIndex,
        reverse: bool,
    ) -> Result<i64, String> {
        use petgraph::Direction;

        // Extract pattern elements. Hop 1 leaves the anchor; hop 2 reaches
        // the far endpoint whose type is checked per counted match.
        let (hop1_elem, hop2_elem, end_elem) = if reverse {
            (
                &pattern.elements[3],
                &pattern.elements[1],
                &pattern.elements[0],
            )
        } else {
            (
                &pattern.elements[1],
                &pattern.elements[3],
                &pattern.elements[4],
            )
        };
        let hop1_edge = match hop1_elem {
            PatternElement::Edge(ep) => ep,
            _ => return Ok(0),
        };
        let mid_node = match &pattern.elements[2] {
            PatternElement::Node(np) => np,
            _ => return Ok(0),
        };
        let hop2_edge = match hop2_elem {
            PatternElement::Edge(ep) => ep,
            _ => return Ok(0),
        };
        let end_node = match end_elem {
            PatternElement::Node(np) => np,
            _ => return Ok(0),
        };

        // Traversal direction of each hop; a reversed traversal flips the
        // pattern's edge directions. `Both` is unsupported in this fused path.
        let map_dir = |d: EdgeDirection| -> Option<Direction> {
            match d {
                EdgeDirection::Outgoing if reverse => Some(Direction::Incoming),
                EdgeDirection::Outgoing => Some(Direction::Outgoing),
                EdgeDirection::Incoming if reverse => Some(Direction::Outgoing),
                EdgeDirection::Incoming => Some(Direction::Incoming),
                EdgeDirection::Both => None,
            }
        };
        let Some(dir1) = map_dir(hop1_edge.direction) else {
            return Ok(0);
        };
        let Some(dir2) = map_dir(hop2_edge.direction) else {
            return Ok(0);
        };
        let interned_conn1 = hop1_edge
            .connection_type
            .as_deref()
            .map(InternedKey::from_str);
        let interned_conn2 = hop2_edge
            .connection_type
            .as_deref()
            .map(InternedKey::from_str);

        // Node-type check without materialization (O(1) mmap read on disk).
        let node_type_matches = |idx: NodeIndex, want: &Option<String>| -> bool {
            match want {
                None => true,
                Some(want_ty) => self
                    .graph
                    .graph
                    .node_type_of(idx)
                    .is_some_and(|nt| self.graph.interner.resolve(nt) == *want_ty),
            }
        };

        let mut total: i64 = 0;
        let mut work = 0usize;

        // First hop: anchor --hop1--> middle nodes
        for e1_ref in self
            .graph
            .graph
            .edges_directed_filtered(anchor_idx, dir1, interned_conn1)
        {
            self.check_interrupt_periodic(work)?;
            work = work.saturating_add(1);
            // `edges_directed_filtered` is a hint (see storage/mod.rs) —
            // post-filter connection type for memory/mapped backends.
            if let Some(ik) = interned_conn1 {
                if e1_ref.weight().connection_type != ik {
                    continue;
                }
            }
            let mid_idx = if dir1 == Direction::Outgoing {
                e1_ref.target()
            } else {
                e1_ref.source()
            };
            if !node_type_matches(mid_idx, &mid_node.node_type) {
                continue;
            }

            // Second hop: mid_idx --hop2--> end nodes (just count)
            for e2_ref in self
                .graph
                .graph
                .edges_directed_filtered(mid_idx, dir2, interned_conn2)
            {
                self.check_interrupt_periodic(work)?;
                work = work.saturating_add(1);
                if let Some(ik) = interned_conn2 {
                    if e2_ref.weight().connection_type != ik {
                        continue;
                    }
                }
                if e2_ref.id() == e1_ref.id() {
                    continue;
                }
                let end_idx = if dir2 == Direction::Outgoing {
                    e2_ref.target()
                } else {
                    e2_ref.source()
                };
                if !node_type_matches(end_idx, &end_node.node_type) {
                    continue;
                }
                total += 1;
            }
        }

        Ok(total)
    }
}

include!("match_clause/fused_match.rs");
