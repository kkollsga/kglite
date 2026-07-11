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
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};

/// Replace every `count(...)` function-call sub-tree in `expr` with a
/// literal `Value::Int64(value)`. Used by the fused OPTIONAL MATCH
/// path to evaluate derived expressions like `total - count(rp)`
/// against a per-upstream-row count without re-running the OPTIONAL
/// expansion. Only count() is substituted because that's the only
/// aggregate the fused path computes inline; other aggregates are
/// rejected upstream by `is_fusable_*_clause`.
fn substitute_count_with_value(expr: &Expression, value: i64) -> Expression {
    match expr {
        Expression::FunctionCall { name, .. } if name == "count" => {
            Expression::Literal(Value::Int64(value))
        }
        Expression::Add(l, r) => Expression::Add(
            Box::new(substitute_count_with_value(l, value)),
            Box::new(substitute_count_with_value(r, value)),
        ),
        Expression::Subtract(l, r) => Expression::Subtract(
            Box::new(substitute_count_with_value(l, value)),
            Box::new(substitute_count_with_value(r, value)),
        ),
        Expression::Multiply(l, r) => Expression::Multiply(
            Box::new(substitute_count_with_value(l, value)),
            Box::new(substitute_count_with_value(r, value)),
        ),
        Expression::Divide(l, r) => Expression::Divide(
            Box::new(substitute_count_with_value(l, value)),
            Box::new(substitute_count_with_value(r, value)),
        ),
        Expression::Modulo(l, r) => Expression::Modulo(
            Box::new(substitute_count_with_value(l, value)),
            Box::new(substitute_count_with_value(r, value)),
        ),
        Expression::Negate(inner) => {
            Expression::Negate(Box::new(substitute_count_with_value(inner, value)))
        }
        // Concat / Case / etc. — leave alone; the gating function only
        // accepts shapes this helper covers, so the fall-through path
        // never sees a containing aggregate.
        other => other.clone(),
    }
}

impl<'a> CypherExecutor<'a> {
    pub(super) fn pattern_match_to_row(&self, m: PatternMatch) -> ResultRow {
        let binding_count = m.bindings.len();
        let mut row = ResultRow::with_capacity(binding_count, binding_count / 2, 0);

        for (var, binding) in m.bindings {
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
                    let string_path: Vec<(petgraph::graph::NodeIndex, String)> = path
                        .iter()
                        .map(|(idx, ik)| (*idx, self.graph.interner.resolve(*ik).to_string()))
                        .collect();
                    row.path_bindings.insert(
                        var,
                        PathBinding {
                            source,
                            hops,
                            path: string_path,
                        },
                    );
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
        let mut expanded = Vec::new();
        for pattern in &clause.patterns {
            let resolved;
            let pat = if Self::pattern_has_vars(pattern) {
                resolved = self.resolve_pattern_vars(pattern, row);
                &resolved
            } else {
                pattern
            };
            let pe = PatternExecutor::with_bindings_and_params(
                self.graph,
                None,
                &row.node_bindings,
                self.params,
            )
            .set_deadline(self.deadline)
            .set_cancel(self.cancel);
            let matches = pe.execute(pat)?;
            for m in &matches {
                if !self.bindings_compatible(row, m) {
                    continue;
                }
                let mut new_row = row.clone();
                self.merge_match_into_row(&mut new_row, m);
                expanded.push(new_row);
            }
        }
        Ok(expanded)
    }

    /// Merge a PatternMatch's bindings into an existing ResultRow
    pub(super) fn merge_match_into_row(&self, row: &mut ResultRow, m: &PatternMatch) {
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
                    let string_path: Vec<(petgraph::graph::NodeIndex, String)> = path
                        .iter()
                        .map(|(idx, ik)| (*idx, self.graph.interner.resolve(*ik).to_string()))
                        .collect();
                    row.path_bindings.insert(
                        var.clone(),
                        PathBinding {
                            source: *source,
                            hops: *hops,
                            path: string_path,
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
        let mut edge_types: Vec<&str> = Vec::new();
        for elem in &pattern.elements {
            match elem {
                PatternElement::Node(np) => {
                    if let Some(ref v) = np.variable {
                        node_vars.push(v);
                    }
                }
                PatternElement::Edge(ep) => {
                    edge_types.push(ep.connection_type.as_deref().unwrap_or(""));
                }
            }
        }
        if node_vars.len() < 2 || edge_types.is_empty() {
            return None;
        }
        let source_idx = row.node_bindings.get(node_vars[0])?;

        // Build full path: for each edge, record the target node and edge type
        let mut path = Vec::with_capacity(edge_types.len());
        for (i, edge_type) in edge_types.iter().enumerate() {
            let node_idx = row.node_bindings.get(node_vars[i + 1])?;
            path.push((*node_idx, edge_type.to_string()));
        }

        Some(PathBinding {
            source: *source_idx,
            hops: edge_types.len(),
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
            let mut found_any = false;

            for pattern in &clause.patterns {
                // Resolve EqualsVar references against current row
                let resolved;
                let pat = if Self::pattern_has_vars(pattern) {
                    resolved = self.resolve_pattern_vars(pattern, row);
                    &resolved
                } else {
                    pattern
                };
                let executor = PatternExecutor::with_bindings_and_params(
                    self.graph,
                    self.budget_probe_limit(None),
                    &row.node_bindings,
                    self.params,
                )
                .set_deadline(self.deadline)
                .set_cancel(self.cancel);
                let matches = executor.execute(pat)?;
                self.budget
                    .check_work(matches.len(), "OPTIONAL MATCH expansion")?;

                for m in &matches {
                    if !self.bindings_compatible(row, m) {
                        continue;
                    }
                    self.budget
                        .reserve_rows(new_rows.len(), 1, "OPTIONAL MATCH")?;
                    let mut new_row = row.clone();
                    self.merge_match_into_row(&mut new_row, m);
                    new_rows.push(new_row);
                    found_any = true;
                }
            }

            if !found_any {
                // Keep the row - OPTIONAL MATCH produces NULLs for unmatched variables
                self.budget
                    .reserve_rows(new_rows.len(), 1, "OPTIONAL MATCH")?;
                new_rows.push(row.clone());
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

    pub(super) fn try_count_simple_pattern(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        bindings: &Bindings<NodeIndex>,
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

        // Don't use fast-path if the bound (group-key) node has property filters
        // — the caller already filtered it. The unbound node's properties are
        // checked inline during counting (supports WHERE push-down on target).
        // Both nodes having properties is rare and we fall back for it.

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
        // Undirected `[r]-` patterns count both incoming and outgoing — the
        // fast path issues two `count_edges_filtered` calls and sums.
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

        // Fast path: when no property filters on the unbound node *and*
        // no inline edge filter, use count_edges_filtered which avoids
        // EdgeData materialization entirely. On disk with sorted CSR:
        // binary search + sequential count (zero allocations). When an
        // edge_filter is set we fall through — the slow loop below
        // checks each edge's predicate without ever building a row.
        if other_props.is_none() && edge.edge_filter.is_none() {
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

        // Slow path: property filters require per-node property access.
        // This loop can iterate millions of edges for hub nodes (Q5 has ~40 M
        // incoming P31 edges), so check the deadline every 1 M iterations.
        // For undirected `[r]-` patterns we sweep both directions and sum.
        let pe = PatternExecutor::new_lightweight_with_params(self.graph, None, self.params);
        let mut count: i64 = 0;
        let mut iter: usize = 0;
        // The inline filter (if any) was pushed by the planner from a
        // downstream WHERE; apply it per edge so the count reflects the
        // post-filter row set. AnchorSide::Source assumption matches
        // `try_count_simple_pattern`'s `(Some(a_idx), None)` arm — the
        // bound side is the pattern's left endpoint.
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
                // Connection type already filtered by edges_directed_filtered
                let other_idx = if dir == Direction::Outgoing {
                    edge_ref.target()
                } else {
                    edge_ref.source()
                };

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

                count += 1;
            }
        }

        Ok(Some(count))
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

        if edge.var_length.is_some() || edge.properties.is_some() {
            return Ok(None);
        }

        let a_bound = node_a
            .variable
            .as_ref()
            .and_then(|v| bindings.get(v).copied());
        let b_bound = node_b
            .variable
            .as_ref()
            .and_then(|v| bindings.get(v).copied());

        // Same shape as `try_count_simple_pattern`'s `CountFastPath` alias —
        // and the same caller-guarantees contract: the bound NodeIndex
        // already satisfies any bound-side property filter, so we ignore
        // `node_a.properties` / `node_b.properties` for the bound side and
        // only consult `other_props` for the unbound peer.
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
                        EdgeDirection::Outgoing => &[Direction::Incoming],
                        EdgeDirection::Incoming => &[Direction::Outgoing],
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
                _ => return Ok(None),
            };

        let conn_type = edge.connection_type.as_deref();
        let interned_conn = conn_type.map(InternedKey::from_str);
        let interned_other_type = other_type.as_ref().map(|t| InternedKey::from_str(t));

        // Always iterate edges and collect peers into a HashSet — there's no
        // pre-built "distinct peers" index. The edge-count fast path
        // (`count_edges_filtered`) doesn't help here because we need
        // peer-uniqueness, not edge count. Polling the deadline every 1 M
        // iterations matches the surrounding helpers.
        let pe = PatternExecutor::new_lightweight_with_params(self.graph, None, self.params);
        let mut peers: HashSet<NodeIndex> = HashSet::new();
        let mut iter: usize = 0;
        let edge_filter = edge.edge_filter.as_ref();

        for &dir in traverse_dirs {
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
                // `edges_directed_filtered` is a hint — the in-memory backend
                // returns every edge in the given direction regardless of
                // `interned_conn`, so we must post-filter by connection type.
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
                if peers.contains(&other_idx) {
                    continue;
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
                if let Some(required_type) = interned_other_type {
                    if let Some(nt) = self.graph.graph.node_type_of(other_idx) {
                        if nt != required_type {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                if let Some(ref props) = other_props {
                    if !pe.node_matches_properties_pub(other_idx, props) {
                        continue;
                    }
                }
                peers.insert(other_idx);
            }
        }

        Ok(Some(peers.len() as i64))
    }

    /// Count matches for a 5-element pattern (a)-[e1]->(b)<-[e2]-(c)
    /// from a bound first node, without materializing intermediate rows.
    /// Traverses: first_node --e1--> middle_nodes --e2--> count last nodes.
    pub(super) fn count_two_hop_pattern(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        first_idx: NodeIndex,
    ) -> Result<i64, String> {
        use petgraph::Direction;

        // Extract pattern elements
        let edge1 = match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return Ok(0),
        };
        let mid_node = match &pattern.elements[2] {
            PatternElement::Node(np) => np,
            _ => return Ok(0),
        };
        let edge2 = match &pattern.elements[3] {
            PatternElement::Edge(ep) => ep,
            _ => return Ok(0),
        };
        let last_node = match &pattern.elements[4] {
            PatternElement::Node(np) => np,
            _ => return Ok(0),
        };

        let dir1 = match edge1.direction {
            EdgeDirection::Outgoing => Direction::Outgoing,
            EdgeDirection::Incoming => Direction::Incoming,
            EdgeDirection::Both => return Ok(0), // unsupported in fused path
        };
        let interned_conn1 = edge1.connection_type.as_deref().map(InternedKey::from_str);

        let dir2 = match edge2.direction {
            EdgeDirection::Outgoing => Direction::Outgoing,
            EdgeDirection::Incoming => Direction::Incoming,
            EdgeDirection::Both => return Ok(0),
        };
        let interned_conn2 = edge2.connection_type.as_deref().map(InternedKey::from_str);

        let mut total: i64 = 0;
        let mut work = 0usize;

        // First hop: first_idx --e1--> middle nodes
        for e1_ref in self
            .graph
            .graph
            .edges_directed_filtered(first_idx, dir1, interned_conn1)
        {
            self.check_interrupt_periodic(work)?;
            work = work.saturating_add(1);
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
            // Check middle node type (O(1) mmap read, no materialization)
            if let Some(ref mid_type) = mid_node.node_type {
                if let Some(nt) = self.graph.graph.node_type_of(mid_idx) {
                    if self.graph.interner.resolve(nt) != mid_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Second hop: mid_idx --e2--> last nodes (just count)
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
                let last_idx = if dir2 == Direction::Outgoing {
                    e2_ref.target()
                } else {
                    e2_ref.source()
                };
                // Check last node type (O(1) mmap read, no materialization)
                if let Some(ref last_type) = last_node.node_type {
                    if let Some(nt) = self.graph.graph.node_type_of(last_idx) {
                        if self.graph.interner.resolve(nt) != last_type {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                total += 1;
            }
        }

        Ok(total)
    }

    /// Count matches for a 5-element pattern traversed in reverse:
    /// (a)-[e1]->(b)-[e2]->(c) counted from c (position 4) backward.
    /// Reads elements [3],[2],[1],[0] with flipped edge directions.
    pub(super) fn count_two_hop_pattern_reverse(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        last_idx: NodeIndex,
    ) -> Result<i64, String> {
        use petgraph::Direction;

        // Read pattern elements in reverse
        let edge2 = match &pattern.elements[3] {
            PatternElement::Edge(ep) => ep,
            _ => return Ok(0),
        };
        let mid_node = match &pattern.elements[2] {
            PatternElement::Node(np) => np,
            _ => return Ok(0),
        };
        let edge1 = match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return Ok(0),
        };
        let first_node = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return Ok(0),
        };

        // Flip edge2 direction (we're traversing from c back toward b)
        let dir2 = match edge2.direction {
            EdgeDirection::Outgoing => Direction::Incoming,
            EdgeDirection::Incoming => Direction::Outgoing,
            EdgeDirection::Both => return Ok(0),
        };
        let interned_conn2 = edge2.connection_type.as_deref().map(InternedKey::from_str);

        // Flip edge1 direction (from b back toward a)
        let dir1 = match edge1.direction {
            EdgeDirection::Outgoing => Direction::Incoming,
            EdgeDirection::Incoming => Direction::Outgoing,
            EdgeDirection::Both => return Ok(0),
        };
        let interned_conn1 = edge1.connection_type.as_deref().map(InternedKey::from_str);

        let mut total: i64 = 0;
        let mut work = 0usize;

        // First hop: last_idx --reverse(e2)--> middle nodes
        for e2_ref in self
            .graph
            .graph
            .edges_directed_filtered(last_idx, dir2, interned_conn2)
        {
            self.check_interrupt_periodic(work)?;
            work = work.saturating_add(1);
            if let Some(ik) = interned_conn2 {
                if e2_ref.weight().connection_type != ik {
                    continue;
                }
            }
            let mid_idx = if dir2 == Direction::Outgoing {
                e2_ref.target()
            } else {
                e2_ref.source()
            };
            // Check middle node type (O(1) mmap read, no materialization)
            if let Some(ref mid_type) = mid_node.node_type {
                if let Some(nt) = self.graph.graph.node_type_of(mid_idx) {
                    if self.graph.interner.resolve(nt) != mid_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Second hop: mid_idx --reverse(e1)--> first nodes (just count)
            for e1_ref in self
                .graph
                .graph
                .edges_directed_filtered(mid_idx, dir1, interned_conn1)
            {
                self.check_interrupt_periodic(work)?;
                work = work.saturating_add(1);
                if let Some(ik) = interned_conn1 {
                    if e1_ref.weight().connection_type != ik {
                        continue;
                    }
                }
                let first_idx = if dir1 == Direction::Outgoing {
                    e1_ref.target()
                } else {
                    e1_ref.source()
                };
                // Check first node type (O(1) mmap read, no materialization)
                if let Some(ref first_type) = first_node.node_type {
                    if let Some(nt) = self.graph.graph.node_type_of(first_idx) {
                        if self.graph.interner.resolve(nt) != first_type {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                total += 1;
            }
        }

        Ok(total)
    }
}

include!("match_clause/fused_match.rs");
