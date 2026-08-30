//! Cypher executor — MATCH clause execution: pattern-variable resolution,
//! the first-MATCH pattern loop and the subsequent-MATCH shared-variable
//! join, including cross-pattern relationship uniqueness (the openCypher
//! trail rule) and pre-bound relationship-variable constraints.

use super::*;
use crate::graph::core::membership::MembershipSet;

// The fused MATCH+WHERE row loop below stays **sequential, on measurement.**
//
// It was implemented partitioned (`into_par_iter` over the match vector,
// order-preserving, gated to `limit_hint.is_none()` + no `distinct_node_hint`
// + an unbounded `max_work_units` + non-disk + non-spatial) and then removed,
// because it is a *regression*: release, 1M-node graphgen fixture, 792k
// surviving rows, min of 7, four runs —
//
//   MATCH (p:Person) WHERE p.score > 0.99 RETURN p.name
//     sequential 188-203 ms   partitioned 241-246 ms   = 0.78-0.84x
//   ... RETURN p.name, p.age
//     sequential 203 ms       partitioned 236 ms       = 0.86x
//
// `pattern_match_to_row` allocates a `ResultRow` (three `Bindings` vectors)
// per match, so the loop is allocator-bound: ten threads contend for the
// allocator instead of sharing work. It is the same shape, and the same
// verdict, as the projection fan-out measured at 1.96x *slower* below
// its crossover — see `parallel::PROJECTION_MIN_ROWS`. The scan feeding this
// loop *is* partitioned and wins 4-6x; the win is simply not here.
//
// Three correctness gates were worked out for that attempt and are recorded
// because they constrain any future one: `distinct_node_hint` cannot be
// partitioned as written (its dedup tests `seen` *before* evaluating the
// predicate, so a duplicate the sequential path never evaluates would be
// evaluated by a partition that has not seen the original — turning a
// predicate error into a divergence); `max_work_units` cannot reproduce the
// sequential error message (which names the count at which the cap was
// crossed, an offset a partition cannot know); and `limit_hint` has no
// partitioned stopping point.

/// The per-clause invariants of one subsequent-MATCH execution: everything
/// [`CypherExecutor::expand_driving_row`] needs that is the same for every
/// driving row, computed once by [`CypherExecutor::subsequent_match_rows`].
struct DrivingRowPlan<'p> {
    clause: &'p MatchClause,
    transient_indexes: Vec<Option<transient_index::TransientEqIndex>>,
    limit_hint: Option<usize>,
    enforce_rel_uniqueness: bool,
    /// The variable one seen-set spans across driving rows, or `None` — see
    /// [`CypherExecutor::cross_row_dedup_var`].
    dedup_var: Option<&'p str>,
}

impl<'a> CypherExecutor<'a> {
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
                                    *matcher = PropertyMatcher::In(MembershipSet::default());
                                } else {
                                    *matcher = PropertyMatcher::Equals(val.clone());
                                }
                            } else {
                                *matcher = PropertyMatcher::In(MembershipSet::default());
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
                                .and_then(|idx| self.graph.graph.node_view(*idx))
                                .map(|node| helpers::resolve_node_property(node, prop, self.graph))
                                .or_else(|| match row.projected.get(var) {
                                    Some(Value::NodeRef(i)) => self
                                        .graph
                                        .graph
                                        .node_view(petgraph::graph::NodeIndex::new(*i as usize))
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
                                    *matcher = PropertyMatcher::In(MembershipSet::default());
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

    /// The variable the *pattern matcher* may deduplicate by while it expands,
    /// or `None` to leave the dedup to the row loop below.
    ///
    /// Deduplicating inside the matcher is what keeps a multi-source expansion's
    /// match vector proportional to the distinct target set rather than to
    /// `sources x targets` — the matcher's `distinct_seen` is shared across
    /// every source row, so a target already emitted is skipped before a
    /// `PatternMatch` is built for it. (It skips *emission*, never traversal:
    /// each source still runs its own BFS through nodes an earlier source
    /// already reached.)
    ///
    /// The hazard it trades against is that the matcher keeps one arbitrary
    /// representative per target, and a fused WHERE may reject exactly that one
    /// while a suppressed match on the same target would have passed. Two
    /// things answer it. Here, the *heuristic*: a predicate that can read the
    /// dedup variable itself is left to the row loop, because such a predicate
    /// genuinely filters targets and would send every call down the retry path.
    /// In [`Self::first_pattern_rows`], the *proof*: any row that fails the
    /// predicate under matcher-level dedup invalidates the pass, which is then
    /// redone without it. So this function only has to be a good guess —
    /// [`match_clause::predicate_may_read_var`] answering `true` too often costs
    /// the optimization, never an answer.
    fn matcher_distinct_target<'c>(
        clause: &'c MatchClause,
        inline_where: Option<&Predicate>,
    ) -> Option<&'c str> {
        let hint = clause.distinct_node_hint.as_ref()?.var.as_str();
        match inline_where {
            None => Some(hint),
            Some(pred) if !match_clause::predicate_may_read_var(pred, hint) => Some(hint),
            Some(_) => None,
        }
    }

    /// Execute the clause's first pattern and turn its matches into rows,
    /// applying the fused WHERE and the `distinct_node_hint` dedup.
    ///
    /// `matcher_distinct_target` is the variable [`Self::matcher_distinct_target`]
    /// licensed the matcher to deduplicate by. Returns `Ok(None)` when that
    /// license turned out to be wrong — a kept representative failed the fused
    /// WHERE, so a match this pass never saw might have passed on the same
    /// target. The caller redoes the pattern with `None`, which cannot fail the
    /// same way because then every match reaches the predicate.
    fn first_pattern_rows(
        &self,
        clause: &MatchClause,
        pattern: &Pattern,
        pattern_limit: Option<usize>,
        limit_hint: Option<usize>,
        inline_where: Option<&Predicate>,
        matcher_distinct_target: Option<String>,
    ) -> Result<Option<Vec<ResultRow>>, String> {
        let matcher_deduped = matcher_distinct_target.is_some();
        // A slot anchor (`WHERE elementId(v) = …`) seeds the variable as a
        // pre-binding, turning the leading scan into a point lookup.
        // Search-space only — the predicate stays.
        let unbound: Bindings<petgraph::graph::NodeIndex> = Bindings::new();
        let anchors = match_clause::seed_clause_node_anchors(clause, &unbound);
        let executor = match anchors.as_ref() {
            Some(pre_bindings) => PatternExecutor::with_bindings_and_params(
                self.graph,
                pattern_limit,
                pre_bindings,
                self.params,
            ),
            None => {
                PatternExecutor::new_lightweight_with_params(self.graph, pattern_limit, self.params)
            }
        }
        .set_deadline(self.deadline)
        .set_cancel(self.cancel)
        .set_parallel(self.parallel)
        .set_match_ceiling(self.budget.match_ceiling("MATCH expansion"))
        .set_distinct_target(matcher_distinct_target);
        let matches = executor.execute(pattern)?;
        self.budget.check_work(matches.len(), "MATCH expansion")?;

        // Every match becomes a row when nothing can drop one: with no fused
        // predicate none is filtered, and under matcher-level dedup a filtered
        // row invalidates the whole pass (below), so the count is exact. Sizing
        // the vector up front then costs nothing and removes the geometric
        // growth's last doubling, which holds the old and new buffers at once —
        // on a 19k-row k-hop that realloc, not the rows, was the peak.
        let exact_rows = (inline_where.is_none() || matcher_deduped)
            .then(|| limit_hint.map_or(matches.len(), |l| l.min(matches.len())));
        let mut rows: Vec<ResultRow> = Vec::with_capacity(exact_rows.unwrap_or(0));
        // When distinct_node_hint is set, pre-dedup by NodeIndex to avoid
        // creating ResultRows for matches that would be DISTINCT-removed later.
        let mut seen: rustc_hash::FxHashSet<petgraph::graph::NodeIndex> =
            rustc_hash::FxHashSet::with_capacity_and_hasher(
                if clause.distinct_node_hint.is_some() {
                    matches.len().min(10000)
                } else {
                    0
                },
                Default::default(),
            );
        // Every unit here is a row this loop builds, filters and retains, and
        // the enclosing matcher has already finished: nothing further down
        // polls until the whole vector is converted. A 1.9M-match MATCH spent
        // that entire conversion past its deadline before this poll existed
        // (the kglite-visual OOM), so the loop that charges `reserve_rows`
        // charges the interrupt at the same stride.
        let mut work = 0usize;
        for m in matches {
            self.check_interrupt_periodic(work)?;
            work = work.saturating_add(1);
            let dedup_idx = clause
                .distinct_node_hint
                .as_ref()
                .and_then(|hint| match_clause::match_node_index(&m, &hint.var));
            if let Some(idx) = dedup_idx {
                if seen.contains(&idx) {
                    continue;
                }
            }
            let row = self.pattern_match_to_row(m);
            // Residual WHERE fused into this MATCH: filter BEFORE the dedup
            // insert so the kept representative is a row that passed the
            // predicate (filter-then-dedup).
            if let Some(pred) = inline_where {
                match self.evaluate_predicate(pred, &row) {
                    Ok(true) => {}
                    Ok(false) => {
                        if matcher_deduped {
                            // The matcher already discarded this target's other
                            // matches; one of them may have passed. This pass
                            // cannot answer the clause.
                            return Ok(None);
                        }
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }
            if let Some(idx) = dedup_idx {
                seen.insert(idx);
            }
            self.budget.reserve_rows(rows.len(), 1, "MATCH")?;
            rows.push(row);
            // Stop after limit matching rows (not candidates)
            if let Some(limit) = limit_hint {
                if rows.len() >= limit {
                    break;
                }
            }
        }
        // Redundant with the in-loop break above, which caps `rows` at
        // `limit_hint` — except for `limit_hint == 0`, where that break only
        // fires after the first push.
        if inline_where.is_none() {
            if let Some(limit) = limit_hint {
                rows.truncate(limit);
            }
        }
        Ok(Some(rows))
    }

    pub(super) fn execute_match(
        &self,
        clause: &MatchClause,
        existing: ResultSet,
        inline_where: Option<&Predicate>,
    ) -> Result<ResultSet, String> {
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

            // One monotonic counter for the whole comma-pattern join, so the
            // poll fires once per `INTERRUPT_POLL_INTERVAL` units however the
            // work is split between driving rows and the matches they expand
            // to — a single driving row with a huge compatible set is bounded
            // by the inner poll, a million cheap rows by the outer one.
            let mut join_work = 0usize;
            for (pi, pattern) in clause.patterns.iter().enumerate() {
                if pi == 0 {
                    // First pattern — create the initial rows. The matcher may
                    // be licensed to deduplicate by `distinct_node_hint` during
                    // expansion (bounding the match vector by the *distinct
                    // target* count instead of sources x targets); when it is,
                    // `first_pattern_rows` returns `None` if a kept
                    // representative turned out to fail the fused WHERE, and the
                    // uncapped-retry below redoes the pattern without it.
                    let licensed =
                        Self::matcher_distinct_target(clause, inline_where).map(str::to_string);
                    all_rows = match self.first_pattern_rows(
                        clause,
                        pattern,
                        pattern_limit,
                        limit_hint,
                        inline_where,
                        licensed,
                    )? {
                        Some(rows) => rows,
                        None => self
                            .first_pattern_rows(
                                clause,
                                pattern,
                                pattern_limit,
                                limit_hint,
                                inline_where,
                                None,
                            )?
                            .expect("no matcher dedup leaves nothing to invalidate"),
                    };
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
                    let has_vars = Self::pattern_has_vars(pattern);
                    // Move rows out so we can iterate by value (enables move-on-last)
                    let old_rows = std::mem::take(&mut all_rows);
                    let old_sets = std::mem::take(&mut clause_edge_sets);
                    let mut new_rows = Vec::with_capacity(old_rows.len());
                    let mut new_sets: Vec<Vec<petgraph::graph::EdgeIndex>> = Vec::new();
                    for (ri, mut existing_row) in old_rows.into_iter().enumerate() {
                        self.check_interrupt_periodic(join_work)?;
                        join_work = join_work.saturating_add(1);
                        let remaining = limit_hint.map(|l| l.saturating_sub(new_rows.len()));
                        if remaining == Some(0) {
                            break;
                        }
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
                        let seeded = match_clause::seed_prebound_pattern_vars(pat, &existing_row);
                        // Block-scoped: the PatternExecutor holds the disk
                        // arena guard (drop glue), so its borrow of
                        // `existing_row` via `pre_bindings` must end before
                        // the move/merge below.
                        let matches = {
                            let base = seeded.as_ref().unwrap_or(&existing_row.node_bindings);
                            let anchored = match_clause::seed_clause_node_anchors(clause, base);
                            let pre_bindings = anchored.as_ref().unwrap_or(base);
                            self.materializing_executor(
                                self.budget_probe_limit(remaining),
                                pre_bindings,
                                "MATCH join",
                            )
                            .execute(pat)?
                        };
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
                            self.check_interrupt_periodic(join_work)?;
                            join_work = join_work.saturating_add(1);
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
            self.subsequent_match_rows(
                clause,
                &existing.rows,
                limit_hint,
                inline_where,
                enforce_rel_uniqueness,
            )?
        };

        // Propagate path bindings for non-shortestPath path assignments.
        // For `MATCH p = (a)-[r:REL*1..3]->(b)`, alias the edge's
        // VariableLengthPath binding under the path variable `p`.
        // For single-hop `MATCH p = (a)-[:REL]->(b)`, synthesize a PathBinding
        // from the edge binding.
        // Runs over the finished row set, once per path assignment, cloning a
        // path binding per row — the last unbounded per-row pass of the clause.
        let mut path_work = 0usize;
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
                self.check_interrupt_periodic(path_work)?;
                path_work = path_work.saturating_add(1);
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

        self.budget.check_rows(result_rows.len(), "MATCH")?;

        Ok(ResultSet {
            rows: result_rows,
            columns: existing.columns,
            lazy_return_items: None,
        })
    }

    /// The variable ONE seen-set may span **every driving row** of a
    /// subsequent MATCH, or `None` to leave each driving row independent.
    ///
    /// [`Self::matcher_distinct_target`] shares a seen-set across the source
    /// rows of a *single* expansion. This shares one across the separate
    /// expansions the subsequent-MATCH branch runs — one `PatternExecutor` per
    /// driving row — which is the same optimization for the UNWIND spelling of
    /// a reachability query (`UNWIND $ids AS i MATCH (p {id: i})-[*1..3]->(f)
    /// RETURN count(DISTINCT f)`) as the WHERE-IN spelling already gets. Like
    /// that one it skips *emission* only: each driving row still traverses
    /// every node an earlier one reached.
    ///
    /// The extra thing that has to hold here is that a driving row may
    /// contribute **no rows at all** — every target it reaches having been
    /// reached already — without the answer noticing. Only the aggregate route
    /// of the hint guarantees it (see [`DistinctNodeHint::aggregate_only`]:
    /// every projection item is a multiplicity-invariant aggregate over the
    /// dedup variable, so no other variable of the dropped row is readable).
    /// The remaining conditions are the licence's mechanical preconditions.
    fn cross_row_dedup_var<'c>(
        clause: &'c MatchClause,
        inline_where: Option<&Predicate>,
        limit_hint: Option<usize>,
        enforce_rel_uniqueness: bool,
    ) -> Option<&'c str> {
        let hint = clause.distinct_node_hint.as_ref()?;
        if !hint.aggregate_only {
            return None;
        }
        // A fused WHERE cannot reach this branch — fusion is gated on an empty
        // incoming result set — so the filter-after-dedup hazard
        // [`Self::first_pattern_rows`] answers with a retry has no analogue
        // here. Refuse the licence rather than assume the gate.
        if inline_where.is_some() {
            return None;
        }
        // A LIMIT decides *which* rows survive, so suppressing a duplicate
        // target changes which driving rows reach the cap.
        if limit_hint.is_some() {
            return None;
        }
        // The trail rule filters matches after the matcher has already
        // discarded a target's other representatives. (Unreachable: it needs
        // two edge-carrying patterns, and the hint needs exactly one pattern.)
        if enforce_rel_uniqueness {
            return None;
        }
        // Exactly one pattern, carrying at least one edge. The single pattern
        // is what the planner's hint already requires; the edge is what makes
        // this worth doing — a node-only pattern binds one target per driving
        // row and is the shape `transient_index` serves without an executor at
        // all, so its rows would never reach the dedup.
        let [pattern] = clause.patterns.as_slice() else {
            return None;
        };
        if pattern.elements.len() < 2 {
            return None;
        }
        Some(hint.var.as_str())
    }

    /// Execute the clause against a non-empty incoming result set: every
    /// existing row drives its own expansion and is replaced by the rows it
    /// produces.
    fn subsequent_match_rows(
        &self,
        clause: &MatchClause,
        existing_rows: &[ResultRow],
        limit_hint: Option<usize>,
        inline_where: Option<&Predicate>,
        enforce_rel_uniqueness: bool,
    ) -> Result<Vec<ResultRow>, String> {
        let mut new_rows = Vec::with_capacity(existing_rows.len());

        let plan = DrivingRowPlan {
            clause,
            // Build a query-local equality index per pattern when the
            // shape qualifies (single typed-node + one EqualsVar/
            // EqualsNodeProp matcher) and the outer-row count justifies
            // the build cost. Avoids the per-row full-type scan that
            // `PatternExecutor::execute` would otherwise do.
            transient_indexes: clause
                .patterns
                .iter()
                .map(|p| {
                    transient_index::TransientEqIndex::try_build(self.graph, p, existing_rows.len())
                })
                .collect(),
            limit_hint,
            enforce_rel_uniqueness,
            dedup_var: Self::cross_row_dedup_var(
                clause,
                inline_where,
                limit_hint,
                enforce_rel_uniqueness,
            ),
        };
        // Targets already emitted by an earlier driving row, when the clause
        // licenses one shared seen-set — see [`Self::cross_row_dedup_var`].
        // A target lands here only once a match carrying it has actually
        // become a row, so nothing a later filter or a matcher retry discarded
        // can mark a target as answered.
        let mut seen: std::collections::HashSet<petgraph::graph::NodeIndex> =
            std::collections::HashSet::new();

        // Same monotonic-counter shape as the comma-pattern join above: a
        // driving row that produces nothing still advances the poll, so a
        // clause that filters everything out is bounded too.
        let mut work = 0usize;
        for row in existing_rows {
            self.check_interrupt_periodic(work)?;
            work = work.saturating_add(1);
            if limit_hint.is_some_and(|l| new_rows.len() >= l) {
                break;
            }
            let produced = self.expand_driving_row(&plan, row, new_rows.len(), &mut seen)?;
            for r in produced {
                self.check_interrupt_periodic(work)?;
                work = work.saturating_add(1);
                self.budget.reserve_rows(new_rows.len(), 1, "MATCH join")?;
                new_rows.push(r);
                if limit_hint.is_some_and(|l| new_rows.len() >= l) {
                    break;
                }
            }
        }
        Ok(new_rows)
    }

    /// Expand one driving row through the clause's patterns, cross-joining
    /// them, and return the rows it produced.
    ///
    /// `seen` carries the cross-row dedup: the targets earlier driving rows
    /// already emitted, under [`DrivingRowPlan::dedup_var`]. The matcher only
    /// reads it; it is extended here, by exactly the matches that became rows.
    fn expand_driving_row(
        &self,
        plan: &DrivingRowPlan<'_>,
        row: &ResultRow,
        produced_so_far: usize,
        seen: &mut std::collections::HashSet<petgraph::graph::NodeIndex>,
    ) -> Result<Vec<ResultRow>, String> {
        let DrivingRowPlan {
            clause,
            transient_indexes,
            limit_hint,
            enforce_rel_uniqueness,
            dedup_var,
        } = plan;
        let (limit_hint, enforce_rel_uniqueness, dedup_var) =
            (*limit_hint, *enforce_rel_uniqueness, *dedup_var);
        // Comma-separated patterns CROSS-JOIN: each pattern expands the
        // working set produced by the previous one (seeded with the incoming
        // row), not independent rows. Earlier this branch pushed a separate
        // row per pattern, so `WITH/UNWIND … MATCH (a),(b)` produced
        // half-rows ({a, null}, {null, b}) instead of the joined {a, b} —
        // which in turn made `… CREATE (a)-[:R]->(b)` mis-bind and create
        // spurious nodes. The single-pattern case (the hot path) reduces to
        // one chain step and keeps the executor's `remaining` limit cap.
        let single_pattern = clause.patterns.len() == 1;
        // One driving row can fan out to millions across the cross-join, so the
        // two emit loops below poll on their own counter; the caller polls the
        // driving rows themselves.
        let mut work = 0usize;
        let mut row_set: Vec<ResultRow> = vec![row.clone()];
        // Relationship-uniqueness bookkeeping, parallel to `row_set`:
        // the edges each working row consumed within THIS clause.
        let mut edge_sets: Vec<Vec<petgraph::graph::EdgeIndex>> = if enforce_rel_uniqueness {
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
                limit_hint.map(|l| l.saturating_sub(produced_so_far))
            } else {
                None
            };
            let exec_limit = self.budget_probe_limit(exec_limit);
            let mut expanded: Vec<ResultRow> = Vec::with_capacity(row_set.len());
            let mut expanded_sets: Vec<Vec<petgraph::graph::EdgeIndex>> = Vec::new();
            for (ci, cur) in row_set.iter().enumerate() {
                // Fast path: probe the transient index when one was built
                // and the bind-var isn't already constrained by a prior
                // binding — live (`node_bindings`) or projected value
                // (`UNWIND collect(n) AS n` → Value::Node; OPTIONAL
                // MATCH miss → Null). Projected constraints are
                // enforced by `bindings_compatible` on the general
                // path, which the probe would bypass. (Transient
                // indexes only cover single-node patterns, so the
                // clause-local edge set is unchanged.)
                if let Some(idx) = &transient_indexes[pi] {
                    if !cur.node_bindings.contains_key(idx.bind_var.as_str())
                        && !cur.projected.contains_key(idx.bind_var.as_str())
                    {
                        if let Some(probe) = idx.probe_value(cur, self.graph) {
                            for &node_idx in idx.lookup(&probe) {
                                self.check_interrupt_periodic(work)?;
                                work = work.saturating_add(1);
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
                // A working row that already constrains the dedup variable
                // pins the pattern to one target, so there is nothing here to
                // deduplicate; sharing the set would only let one working row
                // silence another's mandatory binding.
                let cur_dedup_var = dedup_var.filter(|var| {
                    !cur.node_bindings.contains_key(var) && !cur.projected.contains_key(var)
                });
                let matches =
                    self.driving_row_matches(clause, pat, cur, exec_limit, cur_dedup_var, seen)?;
                self.budget.check_work(matches.len(), "MATCH join")?;
                for m in &matches {
                    self.check_interrupt_periodic(work)?;
                    work = work.saturating_add(1);
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
                    // Record the target only now: the row exists, so no later
                    // driving row is entitled to emit it again.
                    if let Some(var) = cur_dedup_var {
                        if let Some(idx) = match_clause::match_node_index(m, var) {
                            seen.insert(idx);
                        }
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
        Ok(row_set)
    }

    /// One working row's pattern matches, with the cross-row dedup applied if
    /// this row holds the licence.
    ///
    /// Matcher-level dedup keeps one arbitrary match per target, and
    /// [`Self::bindings_compatible`] can reject exactly that one while a
    /// suppressed match on the same target would have passed — which would
    /// lose the target for every *later* driving row too, since the loser
    /// never reaches the shared set. That is the same hazard
    /// [`Self::first_pattern_rows`] answers with an uncapped retry, and the
    /// same answer: the moment a deduplicated pass produces an incompatible
    /// match, redo it without the dedup, where every match reaches the check.
    fn driving_row_matches(
        &self,
        clause: &MatchClause,
        pat: &Pattern,
        cur: &ResultRow,
        exec_limit: Option<usize>,
        dedup_var: Option<&str>,
        seen: &std::collections::HashSet<petgraph::graph::NodeIndex>,
    ) -> Result<Vec<crate::graph::core::pattern_matching::PatternMatch>, String> {
        let seeded = match_clause::seed_prebound_pattern_vars(pat, cur);
        let base = seeded.as_ref().unwrap_or(&cur.node_bindings);
        let anchored = match_clause::seed_clause_node_anchors(clause, base);
        let pre_bindings = anchored.as_ref().unwrap_or(base);
        // The PatternExecutor holds the disk arena guard (drop glue) and
        // borrows `seen`; both end within each `run` call, before the caller
        // extends the set.
        let run = |distinct: Option<&str>| -> Result<Vec<_>, String> {
            self.materializing_executor(exec_limit, pre_bindings, "MATCH join")
                .set_distinct_target(distinct.map(str::to_string))
                .set_distinct_prior(distinct.map(|_| seen))
                .execute(pat)
        };
        let matches = run(dedup_var)?;
        if dedup_var.is_some() && matches.iter().any(|m| !self.bindings_compatible(cur, m)) {
            return run(None);
        }
        Ok(matches)
    }
}
