//! Cypher executor — MATCH clause execution: pattern-variable resolution,
//! the first-MATCH pattern loop and the subsequent-MATCH shared-variable
//! join, including cross-pattern relationship uniqueness (the openCypher
//! trail rule) and pre-bound relationship-variable constraints.

use super::*;
use crate::graph::core::membership::MembershipSet;
use crate::graph::core::pattern_matching::PatternMatch;

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

/// The widest buffer of first-pattern matches a `MATCH` held at once — the
/// whole vector, or the widest chunk — so a test can pin what a filtered
/// `LIMIT` materialises without measuring the process.
#[cfg(test)]
pub(crate) mod first_rows_probe {
    thread_local! {
        static WIDEST: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    pub(crate) fn note(held: usize) {
        WIDEST.with(|w| w.set(w.get().max(held)));
    }

    /// The widest buffer since the last call.
    pub(crate) fn take() -> usize {
        WIDEST.with(|w| w.replace(0))
    }
}

#[cfg(test)]
#[path = "filtered_limit_tests.rs"]
mod filtered_limit_tests;

/// Turns a first pattern's matches into rows: the fused WHERE, the
/// `distinct_node_hint` dedup and the limit, in the order the materialized
/// route always applied them. Fed one vector or one chunk at a time, so the
/// chunked route keeps exactly the rows the whole-vector route would.
struct MatchRowCollector<'c> {
    clause: &'c MatchClause,
    inline_where: Option<&'c Predicate>,
    limit_hint: Option<usize>,
    matcher_deduped: bool,
    bind_paths: Option<bool>,
    seen: rustc_hash::FxHashSet<petgraph::graph::NodeIndex>,
    rows: Vec<ResultRow>,
    work: usize,
}

impl<'c> MatchRowCollector<'c> {
    fn new(
        clause: &'c MatchClause,
        inline_where: Option<&'c Predicate>,
        limit_hint: Option<usize>,
        capacity: usize,
        matcher_deduped: bool,
    ) -> Self {
        Self {
            clause,
            inline_where,
            limit_hint,
            matcher_deduped,
            bind_paths: None,
            seen: Default::default(),
            rows: Vec::with_capacity(capacity),
            work: 0,
        }
    }

    /// Fold `matches` in. `Ok(None)`: a matcher-deduplicated representative
    /// failed the predicate, so this pass cannot answer the clause.
    /// `Ok(Some(true))`: the limit is filled.
    //
    // Every unit here is a row this loop builds, filters and retains, and
    // the enclosing matcher has already finished: nothing further down
    // polls until the whole vector is converted. A 1.9M-match MATCH spent
    // that entire conversion past its deadline before this poll existed
    // (the kglite-visual OOM), so the loop that charges `reserve_rows`
    // charges the interrupt at the same stride.
    fn take_all(
        &mut self,
        executor: &CypherExecutor<'_>,
        matches: Vec<PatternMatch>,
    ) -> Result<Option<bool>, String> {
        let bind_paths = *self.bind_paths.get_or_insert_with(|| {
            CypherExecutor::binds_paths_before_where(self.clause, self.inline_where)
        });
        for m in matches {
            executor.check_interrupt_periodic(self.work)?;
            self.work = self.work.saturating_add(1);
            let dedup_idx = self
                .clause
                .distinct_node_hint
                .as_ref()
                .and_then(|hint| match_clause::match_node_index(&m, &hint.var));
            if let Some(idx) = dedup_idx {
                if self.seen.contains(&idx) {
                    continue;
                }
            }
            let mut row = executor.pattern_match_to_row(m);
            if bind_paths {
                executor.bind_row_paths(self.clause, &mut row);
            }
            // Residual WHERE fused into this MATCH: filter BEFORE the dedup
            // insert so the kept representative is a row that passed the
            // predicate (filter-then-dedup).
            if let Some(pred) = self.inline_where {
                if !executor.evaluate_predicate(pred, &row)? {
                    if self.matcher_deduped {
                        // The matcher already discarded this target's other
                        // matches; one of them may have passed. This pass
                        // cannot answer the clause.
                        return Ok(None);
                    }
                    continue;
                }
            }
            if let Some(idx) = dedup_idx {
                self.seen.insert(idx);
            }
            executor.budget.reserve_rows(self.rows.len(), 1, "MATCH")?;
            self.rows.push(row);
            // Stop after limit matching rows (not candidates)
            if self
                .limit_hint
                .is_some_and(|limit| self.rows.len() >= limit)
            {
                return Ok(Some(true));
            }
        }
        Ok(Some(false))
    }
}

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
    /// no candidates — Cypher equality treats null as never-equal. An inline
    /// map expression (`EqualsExpr`) that fails to evaluate is an error, as
    /// the same expression in a `CREATE` map or a `WHERE` is.
    pub(super) fn resolve_pattern_vars(
        &self,
        pattern: &Pattern,
        row: &ResultRow,
    ) -> Result<Pattern, String> {
        let mut resolved = pattern.clone();
        for element in &mut resolved.elements {
            let props = match element {
                PatternElement::Node(np) => &mut np.properties,
                PatternElement::Edge(ep) => &mut ep.properties,
            };
            if let Some(props) = props {
                for matcher in props.values_mut() {
                    match matcher {
                        PropertyMatcher::EqualsExpr(expr) => {
                            *matcher = match self.evaluate_expression(expr, row)? {
                                Value::Null => PropertyMatcher::In(MembershipSet::default()),
                                value => PropertyMatcher::Equals(value),
                            };
                        }
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
                        PropertyMatcher::EqualsNodeProp { var, prop }
                            if transient_index::reference_needs_evaluation(row, var) =>
                        {
                            let access = Expression::PropertyAccess {
                                variable: var.clone(),
                                property: prop.clone(),
                            };
                            *matcher = match self.evaluate_expression(&access, row)? {
                                Value::Null => PropertyMatcher::In(MembershipSet::default()),
                                value => PropertyMatcher::Equals(value),
                            };
                        }
                        PropertyMatcher::EqualsNodeProp { var, prop } => {
                            let val =
                                transient_index::node_prop_reference(self.graph, row, var, prop);
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
        Ok(resolved)
    }

    /// `pattern` resolved against `row` when it has deferred matchers
    /// (`has_vars`, from [`Self::pattern_has_vars`]), else borrowed as is.
    fn pattern_for_row<'p>(
        &self,
        pattern: &'p Pattern,
        has_vars: bool,
        row: &ResultRow,
    ) -> Result<std::borrow::Cow<'p, Pattern>, String> {
        Ok(if has_vars {
            std::borrow::Cow::Owned(self.resolve_pattern_vars(pattern, row)?)
        } else {
            std::borrow::Cow::Borrowed(pattern)
        })
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
                        PropertyMatcher::EqualsVar(_)
                            | PropertyMatcher::EqualsNodeProp { .. }
                            | PropertyMatcher::EqualsExpr(_)
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
        // The opening MATCH has no row, so an inline-map expression can only
        // read constants and parameters — resolve it against the empty row.
        let resolved;
        let pattern = if Self::pattern_has_vars(pattern) {
            resolved = self.resolve_pattern_vars(pattern, &ResultRow::new())?;
            &resolved
        } else {
            pattern
        };
        let matcher_deduped = matcher_distinct_target.is_some();
        // A slot anchor (`WHERE elementId(v) = …`) seeds the variable as a
        // pre-binding, turning the leading scan into a point lookup.
        // Search-space only — the predicate stays.
        let unbound: Bindings<petgraph::graph::NodeIndex> = Bindings::new();
        let anchors = match_clause::seed_clause_node_anchors(clause, &unbound);
        let executor = self
            .pattern_executor(pattern_limit, anchors.as_ref())
            .set_match_ceiling(self.budget.match_ceiling("MATCH expansion"))
            .set_distinct_target(matcher_distinct_target);
        // A residual WHERE under a LIMIT: the matcher cannot be capped (it
        // does not know how many matches the predicate rejects), so drain it a
        // slice of start nodes at a time and stop once the limit is filled.
        // Materialising every match first held ~670 MB to return 12 000 rows
        // of a 1M-node type. Matcher-level dedup is whole-pass (its retry
        // cannot take back rows), so a licensed dedup keeps the whole route.
        if let (Some(_), Some(limit), false) = (inline_where, limit_hint, matcher_deduped) {
            if let Some(mut chunker) = executor.begin_chunks(pattern)? {
                let mut rows = MatchRowCollector::new(clause, inline_where, limit_hint, 0, false);
                while let Some(chunk) = executor.next_chunk(pattern, &mut chunker)? {
                    #[cfg(test)]
                    first_rows_probe::note(chunk.len());
                    self.budget.check_work(chunk.len(), "MATCH expansion")?;
                    if rows.take_all(self, chunk)?.is_some_and(|full| full) {
                        break;
                    }
                }
                let mut rows = rows.rows;
                rows.truncate(limit);
                return Ok(Some(rows));
            }
        }
        let matches = executor.execute(pattern)?;
        #[cfg(test)]
        first_rows_probe::note(matches.len());
        self.budget.check_work(matches.len(), "MATCH expansion")?;

        // Every match becomes a row when nothing can drop one: with no fused
        // predicate none is filtered, and under matcher-level dedup a filtered
        // row invalidates the whole pass (below), so the count is exact. Sizing
        // the vector up front then costs nothing and removes the geometric
        // growth's last doubling, which holds the old and new buffers at once —
        // on a 19k-row k-hop that realloc, not the rows, was the peak.
        let exact_rows = (inline_where.is_none() || matcher_deduped)
            .then(|| limit_hint.map_or(matches.len(), |l| l.min(matches.len())));
        let mut rows = MatchRowCollector::new(
            clause,
            inline_where,
            limit_hint,
            exact_rows.unwrap_or(0),
            matcher_deduped,
        );
        // When distinct_node_hint is set, pre-dedup by NodeIndex to avoid
        // creating ResultRows for matches that would be DISTINCT-removed later.
        rows.seen.reserve(if clause.distinct_node_hint.is_some() {
            matches.len().min(10000)
        } else {
            0
        });
        if rows.take_all(self, matches)?.is_none() {
            return Ok(None);
        }
        let mut rows = rows.rows;
        // Redundant with the in-loop break, which caps `rows` at `limit_hint` —
        // except for `limit_hint == 0`, where that break only fires after the
        // first push.
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
                return self.execute_shortest_path_match(clause, pa, existing, inline_where);
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

        // `first_pattern_rows` binds paths itself when a fused WHERE may read
        // one; only the leading MATCH's single pattern carries a fused WHERE.
        let paths_bound_early =
            existing.rows.is_empty() && Self::binds_paths_before_where(clause, inline_where);
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
                        let pat = &*self.pattern_for_row(pattern, has_vars, &existing_row)?;
                        // A relationship variable re-used from a prior clause
                        // pins the pattern to that edge — seed its endpoints
                        // so the executor doesn't enumerate every edge.
                        let seeded = match_clause::seed_prebound_pattern_vars(pat, &existing_row);
                        // Block-scoped: the PatternExecutor's borrow of
                        // `existing_row` via `pre_bindings` must end before
                        // the move/merge below.
                        let matches = {
                            let base = seeded.as_ref().unwrap_or(&existing_row.node_bindings);
                            let anchored = self.seed_row_anchors(clause, base, &existing_row);
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
                                self.merge_match_into_row(&mut existing_row, m, pi);
                                self.budget.reserve_rows(new_rows.len(), 1, "MATCH join")?;
                                new_rows.push(existing_row);
                                if enforce_rel_uniqueness {
                                    new_sets.push(edges);
                                }
                                break;
                            }
                            let mut new_row = existing_row.clone();
                            self.merge_match_into_row(&mut new_row, m, pi);
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

        // Bind path variables over the finished row set — the last unbounded
        // per-row pass of the clause.
        if !paths_bound_early
            && clause
                .path_assignments
                .iter()
                .any(|pa| !pa.is_shortest_path)
        {
            let mut path_work = 0usize;
            for row in &mut result_rows {
                self.check_interrupt_periodic(path_work)?;
                path_work = path_work.saturating_add(1);
                self.bind_row_paths(clause, row);
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
            // `PatternExecutor::execute` would otherwise do. A probe binds
            // nodes without the matcher, so none is built under a graph
            // filter, whose guard lives in the matcher.
            transient_indexes: clause
                .patterns
                .iter()
                .map(|p| {
                    if self.graph_filter().is_some() {
                        return None;
                    }
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
                if let Some(predicate) = inline_where {
                    if !self.evaluate_predicate(predicate, &r)? {
                        continue;
                    }
                }
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
                    debug_assert!(
                        self.graph_filter().is_none(),
                        "transient equality index reached under a graph filter"
                    );
                    // A probe the index cannot key (a cross-kind text rule)
                    // falls through to the per-row matcher below.
                    let hits = idx.hits_for_row(cur, self.graph);
                    if let Some(hits) = hits {
                        for &node_idx in hits {
                            self.check_interrupt_periodic(work)?;
                            work = work.saturating_add(1);
                            self.budget
                                .reserve_rows(expanded.len(), 1, "MATCH indexed join")?;
                            let mut nr = cur.clone();
                            nr.node_bindings.insert(idx.bind_var.clone(), node_idx);
                            expanded.push(nr);
                            if enforce_rel_uniqueness {
                                expanded_sets.push(edge_sets[ci].clone());
                            }
                        }
                        continue;
                    }
                }

                // Resolve EqualsVar / EqualsNodeProp references against
                // the current (partially-bound) row.
                let resolved;
                let pat = if Self::pattern_has_vars(pattern) {
                    resolved = self.resolve_pattern_vars(pattern, cur)?;
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
                    self.merge_match_into_row(&mut nr, m, pi);
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
        let anchored = self.seed_row_anchors(clause, base, cur);
        let pre_bindings = anchored.as_ref().unwrap_or(base);
        // The PatternExecutor borrows `seen`; the borrow ends within each `run`
        // call, before the caller extends the set.
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

/// Fold every row-independent inline-map expression (`{id: toUpper('a')}`,
/// `{id: $list[0]}`) to the literal matcher it evaluates to, before the
/// planner runs, so index selection reads literal equality. A scope under a
/// valid-time guard is not folded (the folding executor has no filter); the
/// fused node scans decline an expression value, so it resolves per row. Row-dependent expressions stay deferred and resolve per row;
/// one that fails to evaluate here is also left in place, so the row-time
/// resolution reports the error. Builds no executor when the query has no
/// such expression.
pub(crate) fn fold_constant_inline_maps(
    query: &mut CypherQuery,
    graph: &DirGraph,
    params: &HashMap<String, Value>,
) {
    let mut executor = None;
    fold_clauses(&mut query.clauses, &mut |expr| {
        let mut refs = std::collections::HashSet::new();
        crate::graph::languages::cypher::planner::collect_expression_refs(expr, &mut refs);
        if !refs.is_empty() {
            return None;
        }
        let executor =
            executor.get_or_insert_with(|| CypherExecutor::with_params(graph, params, None));
        match executor.evaluate_expression(expr, &ResultRow::new()) {
            Ok(Value::Null) => Some(PropertyMatcher::In(MembershipSet::default())),
            Ok(value) => Some(PropertyMatcher::Equals(value)),
            Err(_) => None,
        }
    });
}

fn fold_clauses(
    clauses: &mut [Clause],
    fold: &mut impl FnMut(&Expression) -> Option<PropertyMatcher>,
) {
    for clause in clauses {
        match clause {
            Clause::Match(m) | Clause::OptionalMatch(m) => {
                for element in m.patterns.iter_mut().flat_map(|p| p.elements.iter_mut()) {
                    let props = match element {
                        PatternElement::Node(np) => np.properties.as_mut(),
                        PatternElement::Edge(ep) => ep.properties.as_mut(),
                    };
                    for matcher in props.into_iter().flat_map(|map| map.values_mut()) {
                        if let PropertyMatcher::EqualsExpr(expr) = matcher {
                            if let Some(folded) = fold(expr) {
                                *matcher = folded;
                            }
                        }
                    }
                }
            }
            Clause::CallSubquery { body, .. } => fold_clauses(&mut body.clauses, fold),
            _ => {}
        }
    }
}
