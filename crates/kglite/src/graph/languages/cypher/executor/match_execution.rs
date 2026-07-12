//! Cypher executor — MATCH clause execution: pattern-variable resolution,
//! the first-MATCH pattern loop and the subsequent-MATCH shared-variable
//! join, including cross-pattern relationship uniqueness (the openCypher
//! trail rule) and pre-bound relationship-variable constraints.
//!
//! Split out of `executor/mod.rs` (0.12.x) — the mod file keeps the
//! CypherExecutor struct + orchestration; the MATCH machinery lives here.

use super::*;

impl<'a> CypherExecutor<'a> {
    // ========================================================================
    // Variable resolution for pattern properties
    // ========================================================================

    /// Resolve `EqualsVar(name)` and `EqualsNodeProp { var, prop }` references
    /// in pattern properties against the current row. Converts them to
    /// `Equals(value)` so the PatternExecutor can match them (and pick an
    /// indexed lookup if one is available). Enables:
    ///   `WITH "Oslo" AS city MATCH (n:Person {city: city}) RETURN n`  (EqualsVar)
    ///   `MATCH (a) MATCH (b) WHERE b.x = a.y` after planner pushdown  (EqualsNodeProp)
    ///
    /// When a reference cannot be resolved (unknown var, missing property, or
    /// null), the matcher is replaced with `In(vec![])` so the pattern yields
    /// no candidates — Cypher equality treats null as never-equal.
    pub(super) fn resolve_pattern_vars(&self, pattern: &Pattern, row: &ResultRow) -> Pattern {
        let mut resolved = pattern.clone();
        for element in &mut resolved.elements {
            let props = match element {
                PatternElement::Node(np) => &mut np.properties,
                PatternElement::Edge(ep) => &mut ep.properties,
            };
            if let Some(props) = props {
                for matcher in props.values_mut() {
                    match matcher {
                        PropertyMatcher::EqualsVar(name) => {
                            // Check projected scalars (WITH/UNWIND ... AS varName)
                            if let Some(val) = row.projected.get(name) {
                                if matches!(val, Value::Null) {
                                    *matcher = PropertyMatcher::In(Vec::new());
                                } else {
                                    *matcher = PropertyMatcher::Equals(val.clone());
                                }
                            } else {
                                *matcher = PropertyMatcher::In(Vec::new());
                            }
                        }
                        PropertyMatcher::EqualsNodeProp { var, prop } => {
                            // Resolve by reading the referenced node's property:
                            // first a bound node, then a projected node VALUE
                            // (NodeRef/Node) — e.g. `WITH collect(x)[0] AS first
                            // MATCH (b {id: first.id})`.
                            let val = row
                                .node_bindings
                                .get(var)
                                .and_then(|idx| self.graph.graph.node_weight(*idx))
                                .map(|node| helpers::resolve_node_property(node, prop, self.graph))
                                .or_else(|| match row.projected.get(var) {
                                    Some(Value::NodeRef(i)) => self
                                        .graph
                                        .graph
                                        .node_weight(petgraph::graph::NodeIndex::new(*i as usize))
                                        .map(|n| {
                                            helpers::resolve_node_property(n, prop, self.graph)
                                        }),
                                    Some(Value::Node(nv)) => nv.properties.get(prop).cloned(),
                                    // A projected MAP value — e.g. a row from
                                    // `UNWIND $rows AS x MATCH (n {id: x.id})`.
                                    // Read the member directly; previously this
                                    // fell through to `In([])` and silently
                                    // matched nothing.
                                    Some(Value::Map(m)) => m.get(prop).cloned(),
                                    _ => None,
                                });
                            match val {
                                Some(v) if !matches!(v, Value::Null) => {
                                    *matcher = PropertyMatcher::Equals(v);
                                }
                                _ => {
                                    *matcher = PropertyMatcher::In(Vec::new());
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        resolved
    }

    /// Check if a pattern contains any deferred-resolution matchers.
    pub(super) fn pattern_has_vars(pattern: &Pattern) -> bool {
        for element in &pattern.elements {
            let props = match element {
                PatternElement::Node(np) => &np.properties,
                PatternElement::Edge(ep) => &ep.properties,
            };
            if let Some(props) = props {
                for matcher in props.values() {
                    if matches!(
                        matcher,
                        PropertyMatcher::EqualsVar(_) | PropertyMatcher::EqualsNodeProp { .. }
                    ) {
                        return true;
                    }
                }
            }
        }
        false
    }

    // ========================================================================
    // MATCH
    // ========================================================================

    pub(super) fn execute_match(
        &self,
        clause: &MatchClause,
        existing: ResultSet,
        inline_where: Option<&Predicate>,
    ) -> Result<ResultSet, String> {
        // Check for shortestPath assignments
        if let Some(pa) = clause.path_assignments.first() {
            if pa.is_shortest_path {
                return self.execute_shortest_path_match(clause, pa, existing);
            }
        }

        let limit_hint = clause.limit_hint;
        // When an inline WHERE is present, the pattern executor must NOT
        // pre-cap candidates at limit_hint — WHERE may filter some out
        // and we'd return fewer than `limit` rows. Apply the limit after
        // WHERE filtering instead (see the post-filter break below).
        let pattern_limit = if inline_where.is_some() {
            None
        } else {
            limit_hint
        };
        let pattern_limit = self.budget_probe_limit(pattern_limit);

        // Relationship uniqueness (the openCypher trail rule) applies across
        // the comma patterns of ONE MATCH clause: two different pattern
        // edges may not bind the same relationship. Only enforced when at
        // least two patterns carry edges — single-pattern clauses (the hot
        // path) pay nothing. Edges may repeat across separate MATCH clauses.
        let enforce_rel_uniqueness = match_clause::clause_needs_rel_uniqueness(clause);

        let mut result_rows = if existing.rows.is_empty() {
            // First MATCH: execute patterns to produce initial bindings
            let mut all_rows = Vec::new();
            // Parallel to `all_rows` when `enforce_rel_uniqueness`: the edge
            // indices each row consumed within this clause.
            let mut clause_edge_sets: Vec<Vec<petgraph::graph::EdgeIndex>> = Vec::new();

            for (pi, pattern) in clause.patterns.iter().enumerate() {
                if pi == 0 {
                    // First pattern - create initial rows
                    // limit_hint is safe for edge patterns: PatternExecutor
                    // only enforces max_matches at the last hop.
                    // When a residual (non-pushable) WHERE is fused into this
                    // MATCH, the matcher's per-target dedup must NOT run: it
                    // keeps one arbitrary representative per distinct target,
                    // which may fail the predicate while a suppressed match
                    // with the same target would have passed. Filter first,
                    // dedup after (in the loop below) instead.
                    let matcher_distinct_target = if inline_where.is_none() {
                        clause.distinct_node_hint.clone()
                    } else {
                        None
                    };
                    let executor = PatternExecutor::new_lightweight_with_params(
                        self.graph,
                        pattern_limit,
                        self.params,
                    )
                    .set_deadline(self.deadline)
                    .set_cancel(self.cancel)
                    .set_distinct_target(matcher_distinct_target);
                    let matches = executor.execute(pattern)?;
                    self.budget.check_work(matches.len(), "MATCH expansion")?;

                    // When distinct_node_hint is set, pre-dedup by NodeIndex to avoid
                    // creating ResultRows for matches that would be DISTINCT-removed later.
                    if let Some(ref dedup_var) = clause.distinct_node_hint {
                        use crate::graph::core::pattern_matching::MatchBinding;
                        let mut seen: rustc_hash::FxHashSet<_> =
                            rustc_hash::FxHashSet::with_capacity_and_hasher(
                                matches.len().min(10000),
                                Default::default(),
                            );
                        for m in matches {
                            // Resolve the dedup variable's node index first so an
                            // already-kept target is skipped before row conversion.
                            let dedup_idx = m
                                .bindings
                                .iter()
                                .find(|(name, _)| name == dedup_var)
                                .and_then(|(_, b)| match b {
                                    MatchBinding::Node { index, .. } => Some(*index),
                                    MatchBinding::NodeRef(index) => Some(*index),
                                    _ => None,
                                });
                            if let Some(idx) = dedup_idx {
                                if seen.contains(&idx) {
                                    continue;
                                }
                            }
                            let row = self.pattern_match_to_row(m);
                            // Residual WHERE fused into this MATCH: filter BEFORE
                            // the dedup insert so the kept representative is a row
                            // that passed the predicate (filter-then-dedup).
                            if let Some(pred) = inline_where {
                                match self.evaluate_predicate(pred, &row) {
                                    Ok(true) => {}
                                    Ok(false) => continue,
                                    Err(e) => return Err(e),
                                }
                            }
                            if let Some(idx) = dedup_idx {
                                seen.insert(idx);
                            }
                            self.budget.reserve_rows(all_rows.len(), 1, "MATCH")?;
                            all_rows.push(row);
                            // Stop after limit matching rows (not candidates)
                            if let Some(limit) = limit_hint {
                                if all_rows.len() >= limit {
                                    break;
                                }
                            }
                        }
                    } else {
                        for m in matches {
                            let row = self.pattern_match_to_row(m);
                            // Inline WHERE: evaluate predicate before collecting
                            if let Some(pred) = inline_where {
                                match self.evaluate_predicate(pred, &row) {
                                    Ok(true) => {}           // Keep row
                                    Ok(false) => continue,   // Skip non-matching row
                                    Err(e) => return Err(e), // Propagate errors (e.g., missing param)
                                }
                            }
                            self.budget.reserve_rows(all_rows.len(), 1, "MATCH")?;
                            all_rows.push(row);
                            // Stop after limit matching rows (not candidates)
                            if let Some(limit) = limit_hint {
                                if all_rows.len() >= limit {
                                    break;
                                }
                            }
                        }
                    }
                    // Post-match truncation: for edge patterns without inline WHERE,
                    // limit_hint wasn't passed to the PatternExecutor, so truncate here.
                    if inline_where.is_none() {
                        if let Some(limit) = limit_hint {
                            all_rows.truncate(limit);
                        }
                    }
                    // Rows from the first pattern hold exactly that pattern's
                    // bindings, so its consumed edges can be read back off
                    // the rows (named edges + fixed/var-length path hops).
                    if enforce_rel_uniqueness {
                        clause_edge_sets = all_rows
                            .iter()
                            .map(match_clause::row_edge_indices)
                            .collect();
                    }
                } else {
                    if all_rows.is_empty() {
                        // An earlier pattern produced no rows: the comma
                        // patterns of one MATCH join, so the clause result is
                        // empty. Without this break the next pattern would
                        // re-enter the "first pattern" branch and fabricate
                        // rows that ignore the empty pattern entirely.
                        break;
                    }
                    // Subsequent patterns: use shared-variable join
                    // Pass existing node bindings as pre-bindings to constrain the pattern
                    let has_vars = Self::pattern_has_vars(pattern);
                    // Move rows out so we can iterate by value (enables move-on-last)
                    let old_rows = std::mem::take(&mut all_rows);
                    let old_sets = std::mem::take(&mut clause_edge_sets);
                    let mut new_rows = Vec::with_capacity(old_rows.len());
                    let mut new_sets: Vec<Vec<petgraph::graph::EdgeIndex>> = Vec::new();
                    for (ri, mut existing_row) in old_rows.into_iter().enumerate() {
                        // Calculate remaining budget for this expansion
                        let remaining = limit_hint.map(|l| l.saturating_sub(new_rows.len()));
                        if remaining == Some(0) {
                            break;
                        }
                        // Resolve EqualsVar references against current row
                        let resolved;
                        let pat = if has_vars {
                            resolved = self.resolve_pattern_vars(pattern, &existing_row);
                            &resolved
                        } else {
                            pattern
                        };
                        // A relationship variable re-used from a prior clause
                        // pins the pattern to that edge — seed its endpoints
                        // so the executor doesn't enumerate every edge.
                        let seeded = match_clause::seed_bound_edge_endpoints(pat, &existing_row);
                        let pre_bindings = seeded.as_ref().unwrap_or(&existing_row.node_bindings);
                        let executor = PatternExecutor::with_bindings_and_params(
                            self.graph,
                            self.budget_probe_limit(remaining),
                            pre_bindings,
                            self.params,
                        )
                        .set_deadline(self.deadline)
                        .set_cancel(self.cancel);
                        let matches = executor.execute(pat)?;
                        self.budget.check_work(matches.len(), "MATCH join")?;
                        // Collect compatible matches (with their clause-local
                        // edge sets when uniqueness is enforced) for the
                        // move-on-last optimization.
                        let row_edges = old_sets.get(ri);
                        let compatible: Vec<(
                            &crate::graph::core::pattern_matching::PatternMatch,
                            Vec<petgraph::graph::EdgeIndex>,
                        )> = matches
                            .iter()
                            .filter(|m| self.bindings_compatible(&existing_row, m))
                            .filter_map(|m| {
                                if !enforce_rel_uniqueness {
                                    return Some((m, Vec::new()));
                                }
                                let mut m_edges = Vec::new();
                                match_clause::match_edge_indices(m, &mut m_edges);
                                let prior = row_edges.map(Vec::as_slice).unwrap_or(&[]);
                                if m_edges.iter().any(|e| prior.contains(e)) {
                                    return None; // trail rule: edge re-use across patterns
                                }
                                let mut next = prior.to_vec();
                                next.extend(m_edges);
                                Some((m, next))
                            })
                            .collect();
                        let total = compatible.len();
                        for (i, (m, edges)) in compatible.into_iter().enumerate() {
                            if i + 1 == total {
                                // Last compatible match: move row instead of cloning
                                self.merge_match_into_row(&mut existing_row, m);
                                self.budget.reserve_rows(new_rows.len(), 1, "MATCH join")?;
                                new_rows.push(existing_row);
                                if enforce_rel_uniqueness {
                                    new_sets.push(edges);
                                }
                                break;
                            }
                            let mut new_row = existing_row.clone();
                            self.merge_match_into_row(&mut new_row, m);
                            self.budget.reserve_rows(new_rows.len(), 1, "MATCH join")?;
                            new_rows.push(new_row);
                            if enforce_rel_uniqueness {
                                new_sets.push(edges);
                            }
                            if limit_hint.is_some_and(|l| new_rows.len() >= l) {
                                break;
                            }
                        }
                        if limit_hint.is_some_and(|l| new_rows.len() >= l) {
                            break;
                        }
                    }
                    all_rows = new_rows;
                    clause_edge_sets = new_sets;
                }
            }
            all_rows
        } else {
            // Subsequent MATCH: expand each existing row with new patterns
            let mut new_rows = Vec::with_capacity(existing.rows.len());

            // Build a query-local equality index per pattern when the
            // shape qualifies (single typed-node + one EqualsVar/
            // EqualsNodeProp matcher) and the outer-row count justifies
            // the build cost. Avoids the per-row full-type scan that
            // `PatternExecutor::execute` would otherwise do.
            let transient_indexes: Vec<Option<transient_index::TransientEqIndex>> = clause
                .patterns
                .iter()
                .map(|p| {
                    transient_index::TransientEqIndex::try_build(self.graph, p, existing.rows.len())
                })
                .collect();

            // Comma-separated patterns CROSS-JOIN: each pattern expands the
            // working set produced by the previous one (seeded with the incoming
            // row), not independent rows. Earlier this branch pushed a separate
            // row per pattern, so `WITH/UNWIND … MATCH (a),(b)` produced
            // half-rows ({a, null}, {null, b}) instead of the joined {a, b} —
            // which in turn made `… CREATE (a)-[:R]->(b)` mis-bind and create
            // spurious nodes. The single-pattern case (the hot path) reduces to
            // one chain step and keeps the executor's `remaining` limit cap.
            let single_pattern = clause.patterns.len() == 1;
            for row in &existing.rows {
                if limit_hint.is_some_and(|l| new_rows.len() >= l) {
                    break;
                }
                let mut row_set: Vec<ResultRow> = vec![row.clone()];
                // Relationship-uniqueness bookkeeping, parallel to `row_set`:
                // the edges each working row consumed within THIS clause.
                let mut edge_sets: Vec<Vec<petgraph::graph::EdgeIndex>> = if enforce_rel_uniqueness
                {
                    vec![Vec::new()]
                } else {
                    Vec::new()
                };
                for (pi, pattern) in clause.patterns.iter().enumerate() {
                    if row_set.is_empty() {
                        break;
                    }
                    // For a single pattern we can still cap the executor at the
                    // outer LIMIT; for a cross-join the per-pattern count isn't
                    // the final count, so don't pre-cap (apply at push instead).
                    let exec_limit = if single_pattern {
                        limit_hint.map(|l| l.saturating_sub(new_rows.len()))
                    } else {
                        None
                    };
                    let exec_limit = self.budget_probe_limit(exec_limit);
                    let mut expanded: Vec<ResultRow> = Vec::with_capacity(row_set.len());
                    let mut expanded_sets: Vec<Vec<petgraph::graph::EdgeIndex>> = Vec::new();
                    for (ci, cur) in row_set.iter().enumerate() {
                        // Fast path: probe the transient index when one was built
                        // and the bind-var isn't already constrained by a prior
                        // binding. (Transient indexes only cover single-node
                        // patterns, so the clause-local edge set is unchanged.)
                        if let Some(idx) = &transient_indexes[pi] {
                            if !cur.node_bindings.contains_key(idx.bind_var.as_str()) {
                                if let Some(probe) = idx.probe_value(cur, self.graph) {
                                    for &node_idx in idx.lookup(&probe) {
                                        self.budget.reserve_rows(
                                            expanded.len(),
                                            1,
                                            "MATCH indexed join",
                                        )?;
                                        let mut nr = cur.clone();
                                        nr.node_bindings.insert(idx.bind_var.clone(), node_idx);
                                        expanded.push(nr);
                                        if enforce_rel_uniqueness {
                                            expanded_sets.push(edge_sets[ci].clone());
                                        }
                                    }
                                }
                                continue;
                            }
                        }

                        // Resolve EqualsVar / EqualsNodeProp references against
                        // the current (partially-bound) row.
                        let resolved;
                        let pat = if Self::pattern_has_vars(pattern) {
                            resolved = self.resolve_pattern_vars(pattern, cur);
                            &resolved
                        } else {
                            pattern
                        };
                        // A relationship variable re-used from a prior clause
                        // pins the pattern to that edge — seed its endpoints
                        // so the executor doesn't enumerate every edge.
                        let seeded = match_clause::seed_bound_edge_endpoints(pat, cur);
                        let pre_bindings = seeded.as_ref().unwrap_or(&cur.node_bindings);
                        let executor = PatternExecutor::with_bindings_and_params(
                            self.graph,
                            exec_limit,
                            pre_bindings,
                            self.params,
                        )
                        .set_deadline(self.deadline)
                        .set_cancel(self.cancel);
                        let matches = executor.execute(pat)?;
                        self.budget.check_work(matches.len(), "MATCH join")?;
                        for m in &matches {
                            if !self.bindings_compatible(cur, m) {
                                continue;
                            }
                            if enforce_rel_uniqueness {
                                let mut m_edges = Vec::new();
                                match_clause::match_edge_indices(m, &mut m_edges);
                                if m_edges.iter().any(|e| edge_sets[ci].contains(e)) {
                                    continue; // trail rule: edge re-use across patterns
                                }
                                let mut next = edge_sets[ci].clone();
                                next.extend(m_edges);
                                expanded_sets.push(next);
                            }
                            let mut nr = cur.clone();
                            self.merge_match_into_row(&mut nr, m);
                            self.budget.reserve_rows(expanded.len(), 1, "MATCH join")?;
                            expanded.push(nr);
                        }
                    }
                    row_set = expanded;
                    if enforce_rel_uniqueness {
                        edge_sets = expanded_sets;
                    }
                }
                for r in row_set {
                    self.budget.reserve_rows(new_rows.len(), 1, "MATCH join")?;
                    new_rows.push(r);
                    if limit_hint.is_some_and(|l| new_rows.len() >= l) {
                        break;
                    }
                }
            }
            new_rows
        };

        // Propagate path bindings for non-shortestPath path assignments.
        // For `MATCH p = (a)-[r:REL*1..3]->(b)`, alias the edge's
        // VariableLengthPath binding under the path variable `p`.
        // For single-hop `MATCH p = (a)-[:REL]->(b)`, synthesize a PathBinding
        // from the edge binding.
        for pa in &clause.path_assignments {
            if pa.is_shortest_path {
                continue;
            }
            // Identify the VLP edge variable from this pattern so we look up
            // the correct path binding (not just the first one in the map).
            let vlp_edge_var: Option<String> =
                clause.patterns.get(pa.pattern_index).and_then(|pat| {
                    pat.elements.iter().find_map(|elem| {
                        if let PatternElement::Edge(ep) = elem {
                            if ep.var_length.is_some() {
                                return ep.variable.clone();
                            }
                        }
                        None
                    })
                });

            for row in &mut result_rows {
                // First try: find the VLP binding matching this pattern's edge variable
                let path_binding = if let Some(ref vlp_var) = vlp_edge_var {
                    row.path_bindings.get(vlp_var).cloned()
                } else {
                    // Fallback: pick first path binding (single-path case)
                    row.path_bindings.iter().next().map(|(_, pb)| pb.clone())
                };
                if let Some(pb) = path_binding {
                    row.path_bindings.insert(pa.variable.clone(), pb);
                } else {
                    // No variable-length path found: synthesize the exact
                    // fixed-length trail from its named/internal edge bindings.
                    if let Some(pattern) = clause.patterns.get(pa.pattern_index) {
                        if let Some(pb) = self.synthesize_path_from_pattern(pattern, row) {
                            row.path_bindings.insert(pa.variable.clone(), pb);
                        }
                    }
                }
            }
        }

        // Enforce max_rows limit if configured
        self.budget.check_rows(result_rows.len(), "MATCH")?;

        Ok(ResultSet {
            rows: result_rows,
            columns: existing.columns,
            lazy_return_items: None,
        })
    }
}
