//! Hop expansion for [`PatternExecutor`] — start-node seeding, the per-hop
//! parallel and sequential expansion loops, and the small predicates they
//! share.
//!
//! Split out of `matcher.rs` for the source-quality line ceiling; these are
//! inherent methods on `PatternExecutor`, so the split is a file boundary only.

use super::*;
use crate::graph::parallel::{self, ParallelInterrupt};

/// How many matches one parallel job may hold unreported before it publishes
/// them to the shared ceiling counter.
///
/// Set from measurement, not taste. Publishing on every source row cost
/// **+78% to +84%** on `two_edge_relationship_text_filter` across two agreeing
/// release runs — that cell's rows are short (~11 ns each) and an
/// eight-thread-contended `fetch_add` is comparable to the row itself. At 256
/// the same cell measures +1.5-1.7%, which is inside the capture's own noise.
///
/// The number that matters at the other end is how much a job can hold back:
/// rayon splits this region finely (measured ~7 rows per `map_init` call on a
/// 9 000-row hop), so a job only publishes when its rows are *wide*. That is
/// the shape a memory ceiling is for; a narrow hop is bounded at the hop
/// boundary instead — see [`PatternExecutor::expand_hop_parallel`].
const CEILING_PUBLISH_STRIDE: usize = 256;

/// Whether one node variable is written more than once in `pattern`
/// (`(a)-[]->(b)-[]->(a)`). Such a pattern constrains a later hop against an
/// earlier binding, so partial matches on the same node are not
/// interchangeable and must not be deduplicated by node index.
fn repeats_a_node_variable(pattern: &Pattern) -> bool {
    let mut seen: Vec<&str> = Vec::new();
    for element in &pattern.elements {
        let PatternElement::Node(node) = element else {
            continue;
        };
        let Some(var) = node.variable.as_deref() else {
            continue;
        };
        if seen.contains(&var) {
            return true;
        }
        seen.push(var);
    }
    false
}

/// Collapse the partial matches of one intermediate hop to one per node, when
/// `collapsible` says the caller proved they are interchangeable. Returns them
/// unchanged otherwise.
///
/// The optimization exists because `distinct_target_var` (the planner's
/// `distinct_node_hint`) means only the last hop's target reaches the answer,
/// so carrying N partials through an anonymous intermediate is wasted work.
/// It is only legal while nothing downstream can tell two partials apart, and
/// two things can:
///
/// - **The relationships they already consumed.** Cypher paths are trails, so
///   a later hop consults them (`reuses_bound_relationship`) and two partials
///   on the same node continue *differently*. Keeping one silently deletes
///   the other's continuations: on a ring with stride-7 chords,
///   `(a {id: 0})-[:R]-()-[:R]-()-[:R]-(b) RETURN DISTINCT b.id` lost node 7,
///   whose only three-hop trail runs through an intermediate the dedup had
///   already claimed for a route that had consumed the last relationship.
///   `mark_disjoint_fixed_trails` is what usually clears this: pairwise-
///   disjoint hop types record no trail at all.
/// - **A node variable the pattern binds twice** (`(a)-[]->()-[]->(a)`).
///   `target_satisfies_bindings` compares a later hop's target against the
///   earlier binding, which differs per partial — `(a:N)-[:A]->()-[:B]->(a)
///   RETURN DISTINCT a.id` returned 1 of 3 rows.
fn dedup_interchangeable_partials(
    matches: Vec<PatternMatch>,
    indices: Vec<NodeIndex>,
    collapsible: bool,
) -> (Vec<PatternMatch>, Vec<NodeIndex>) {
    if !collapsible {
        return (matches, indices);
    }
    let mut seen = HashSet::with_capacity(indices.len());
    let mut kept_matches = Vec::with_capacity(indices.len());
    let mut kept_indices = Vec::with_capacity(indices.len());
    for (partial, index) in matches.into_iter().zip(indices) {
        if seen.insert(index) {
            kept_matches.push(partial);
            kept_indices.push(index);
        }
    }
    (kept_matches, kept_indices)
}

/// Everything about one expansion hop that is invariant across the matches
/// being expanded. Built once per hop by [`PatternExecutor::execute`] — the
/// internal binding name in particular must not be formatted afresh for every
/// matched relationship on the expansion hot path.
struct HopPlan<'p> {
    edge: &'p EdgePattern,
    node: &'p NodePattern,
    /// Internal binding name for an anonymous variable-length path, when the
    /// hop needs path info and the edge has no user variable.
    anonymous_path_var: Option<String>,
    /// Whether an exact fixed-length trail must be extended at this hop.
    track_fixed_trail: bool,
    /// Last hop of the pattern: where `max_matches` is exact and where
    /// distinct-target dedup applies.
    is_last_hop: bool,
    /// How many matches this hop may keep. Exact `max_matches` at the last
    /// hop; at an intermediate hop an *advisory* overcommit (or `None` on the
    /// uncapped retry pass) — see [`CapPass`].
    limit: Option<usize>,
    /// Whether this hop may hand its remaining budget down into a
    /// variable-length expansion so the BFS stops at the first rows it needs.
    ///
    /// The expansion is upstream of three filters this loop applies —
    /// [`reuses_bound_relationship`], [`PatternExecutor::target_satisfies_bindings`]
    /// and the distinct-target dedup — so truncating it is only sound when
    /// none of them can reject a row. If one could, a cap of *n* would return
    /// *n* rows that all get dropped and the hop would answer "no more rows"
    /// when more existed. False for every hop where that is possible, which
    /// restores the pre-cap behaviour (the expansion runs to completion and
    /// this loop truncates).
    ///
    /// Two of the three are decided per hop and live here; the third
    /// (`bound_target`) is per match and is checked at the call site.
    var_length_cap_safe: bool,
}

/// Whether an [`PatternExecutor::execute_pass`] applies the advisory
/// candidate caps.
///
/// The caps (100× on start nodes, 50× on intermediate hops) are a
/// *selectivity heuristic*: they assume roughly one candidate in a hundred
/// survives the pattern's filters. When that assumption is wrong — an
/// unlabeled start whose relationship-typed sources enumerate late, a sparse
/// label, a sparse intermediate hop — enforcing the heuristic as a bound
/// silently returns zero or partial rows. So they stay for speed, but a pass
/// that hit one and came back short is re-run [`CapPass::Uncapped`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum CapPass {
    /// First pass: advisory caps on, cap bites recorded.
    Capped,
    /// Retry pass: no pre-caps on start nodes or intermediate hops. The exact
    /// `max_matches` early-exit at the last hop stays — correctness never
    /// needed the pre-caps, only the final limit.
    Uncapped,
}

impl<'a> PatternExecutor<'a> {
    /// Choose and materialise the start-node set for `pattern`, capped at
    /// `source_cap`. Runs once per [`Self::execute`] call, before the
    /// expansion loop — never per row.
    fn seed_start_nodes(
        &self,
        pattern: &Pattern,
        first_node: &NodePattern,
        has_edges: bool,
        source_cap: Option<usize>,
    ) -> Result<Vec<NodeIndex>, String> {
        // Pre-bound first nodes (e.g. `MATCH (f {id: X}) MATCH (f)-[:R]->(c)`)
        // must skip the inverted-index fast path — that path returns every source
        // for the edge type, ignoring the binding. find_matching_nodes resolves
        // the variable to a single node directly.
        let first_is_prebound = first_node
            .variable
            .as_ref()
            .map(|v| self.pre_bindings.get(v).is_some())
            .unwrap_or(false);

        // Untyped source with typed edges: the connection-type inverted index
        // names the nodes with such outgoing edges, instead of scanning all 124M.
        let mut initial_nodes = if !first_is_prebound
            && has_edges
            && first_node.node_type.is_none()
            && first_node.properties.is_none()
        {
            // `[:A|B]` needs the sources of EVERY branch: taking only the
            // singular `connection_type` dropped every start node whose sole
            // matching edge was on a later branch.
            let edge_conn_types: Option<Vec<InternedKey>> =
                if let Some(PatternElement::Edge(ep)) = pattern.elements.get(1) {
                    if ep.var_length.is_none() {
                        match ep.conn_filter() {
                            ConnTypeFilter::Any => None,
                            ConnTypeFilter::One(key) => Some(vec![key]),
                            ConnTypeFilter::AnyOf(keys) => Some(keys),
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
            // The inverted index only covers outgoing sources.
            let is_outgoing = if let Some(PatternElement::Edge(ep)) = pattern.elements.get(1) {
                ep.direction == EdgeDirection::Outgoing
            } else {
                false
            };
            if let (Some(conn_types), true) = (edge_conn_types, is_outgoing) {
                // Pass `source_cap` through so we don't eagerly copy the
                // whole 400 MB source list from the inverted index for
                // a query that only needs 1 000 of them. An alternation
                // unions the per-branch source lists; if the index is
                // unavailable for ANY branch the union would be short, so
                // the whole lookup falls back to the full node scan.
                let mut union: Vec<NodeIndex> = Vec::new();
                let mut complete = true;
                for ct in conn_types {
                    match self
                        .graph
                        .graph
                        .sources_for_conn_type_bounded(ct, source_cap)
                    {
                        Some(sources) => {
                            // A read that came back exactly full to the cap may
                            // have left later sources unseen; the index cannot
                            // say which. Treat it as a bite — a false positive
                            // costs at most one retry, and only on a pass that
                            // already came back short.
                            if source_cap.is_some_and(|cap| sources.len() >= cap) {
                                self.note_cap_truncated();
                            }
                            union.extend(sources.into_iter().map(|s| NodeIndex::new(s as usize)));
                        }
                        None => {
                            complete = false;
                            break;
                        }
                    }
                }
                if complete {
                    // A node can source more than one branch; the caller
                    // treats each start node once.
                    union.sort_unstable();
                    union.dedup();
                    union
                } else {
                    self.find_matching_nodes(first_node)?
                }
            } else {
                self.find_matching_nodes(first_node)?
            }
        } else {
            self.find_matching_nodes(first_node)?
        };
        if let Some(cap) = source_cap {
            if initial_nodes.len() > cap {
                // Only advisory when the pattern has edges — see `execute`,
                // where a single-node pattern's cap is the exact limit.
                if has_edges {
                    self.note_cap_truncated();
                }
                initial_nodes.truncate(cap);
            }
        }
        Ok(initial_nodes)
    }

    /// Execute the pattern and return all matches.
    ///
    /// Runs [`Self::execute_pass`] with the advisory caps on. If that pass hit
    /// a cap *and* came back short of `max_matches`, the short result is not
    /// evidence that the pattern has no more rows — the caps are a selectivity
    /// heuristic, not a bound — so the whole pattern is re-run once with the
    /// pre-caps off. Re-entering `execute_pass` (rather than resuming the
    /// capped one) is what keeps the distinct-target dedup, the lazy seeding
    /// and the per-hop bookkeeping consistent: the retry is an ordinary
    /// execution that happens to have no pre-caps.
    ///
    /// A dense pattern — one where the cap is met by real rows — never reads
    /// the bit for anything: it returns `max_matches` rows and returns here.
    pub fn execute(&self, pattern: &Pattern) -> Result<Vec<PatternMatch>, String> {
        self.take_cap_truncated();
        let matches = self.execute_pass(pattern, CapPass::Capped)?;
        if self.max_matches.is_some_and(|max| matches.len() < max) && self.take_cap_truncated() {
            return self.execute_pass(pattern, CapPass::Uncapped);
        }
        Ok(matches)
    }

    /// Build one hop's [`HopPlan`].
    fn plan_hop<'p>(
        &self,
        edge_pattern: &'p EdgePattern,
        node_pattern: &'p NodePattern,
        element_index: usize,
        is_last_hop: bool,
        earlier_relationship_state: bool,
        pass: CapPass,
    ) -> HopPlan<'p> {
        HopPlan {
            edge: edge_pattern,
            node: node_pattern,
            anonymous_path_var: (edge_pattern.variable.is_none()
                && edge_pattern.needs_path_info
                && edge_pattern.var_length.is_some())
            .then(|| format!("__anon_vlpath_{element_index}")),
            track_fixed_trail: edge_pattern.var_length.is_none() && edge_pattern.needs_path_info,
            is_last_hop,
            // At the last hop `max_matches` is exact. At an intermediate hop it
            // is a generous overcommit (50×) that avoids expanding far more
            // intermediates than needed — advisory, because a sparse
            // intermediate can push the rows that survive to the last hop past
            // it. The uncapped retry drops it entirely.
            limit: if is_last_hop {
                self.max_matches
            } else {
                match pass {
                    CapPass::Capped => self.max_matches.map(|m| m.saturating_mul(50).max(1000)),
                    CapPass::Uncapped => None,
                }
            },
            var_length_cap_safe: edge_pattern.var_length.is_some()
                && !earlier_relationship_state
                && !self
                    .distinct_target_var
                    .as_deref()
                    .is_some_and(|dtv| node_pattern.variable.as_deref() == Some(dtv)),
        }
    }

    /// One execution of `pattern` under the given cap regime.
    fn execute_pass(&self, pattern: &Pattern, pass: CapPass) -> Result<Vec<PatternMatch>, String> {
        if pattern.elements.is_empty() {
            return Ok(Vec::new());
        }

        let first_node = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => {
                return Err(
                    "Pattern must start with a node in parentheses. Example: (n:Person) or ()"
                        .to_string(),
                )
            }
        };

        let has_edges = pattern.elements.len() > 1;
        let source_cap = if has_edges {
            // Multi-hop with LIMIT: cap sources to avoid O(N) allocation +
            // PatternMatch construction for millions of nodes; the expansion
            // loop enforces exact max_matches via early-exit. The 100x headroom
            // assumes ~1% of sources produce a match, but the start-node set is
            // relationship-type-blind, so it is a selectivity guess, not a
            // bound — a short result under it is retried uncapped by `execute`.
            match pass {
                CapPass::Capped => self.max_matches.map(|m| m.saturating_mul(100).max(1000)),
                CapPass::Uncapped => None,
            }
        } else {
            // Single-node pattern: exact truncation — `find_matching_nodes` has
            // already applied every filter the pattern has, so any
            // `max_matches` of these rows is a correct answer.
            self.max_matches
        };
        let initial_nodes = self.seed_start_nodes(pattern, first_node, has_edges, source_cap)?;

        // Under a cap the first hop stops as soon as `max_matches` rows exist,
        // so the start nodes past that point are never read — and `source_cap`
        // is deliberately 100x the cap, to survive a sparse pattern. Seeding
        // all of them up front therefore built a `PatternMatch` (a `Vec`
        // allocation plus the variable name) for 10 000 nodes to serve 100
        // rows on the tracked `return_node_rel_node_100` cell, where the build
        // and its matching drop were 60% of `execute`. Under a cap the seeds
        // are built as the expansion reaches them instead (`seeds_pending`).
        // The uncapped path stays eager: it consumes every seed anyway, and
        // its parallel branch needs the vector to zip against.
        let mut seeds_pending = has_edges && self.max_matches.is_some();
        let mut matches: Vec<PatternMatch> = if seeds_pending {
            Vec::new()
        } else {
            initial_nodes
                .iter()
                .map(|&idx| self.seed_match(first_node, idx))
                .collect()
        };

        let mut current_indices: Vec<NodeIndex> = initial_nodes;

        // One reusable visited buffer for the whole pass: a variable-length hop
        // marks it per source row instead of allocating and zeroing a
        // graph-sized `Vec<bool>` for each one.
        let mut visited = VisitedStamps::default();

        let mut distinct_seen: HashSet<NodeIndex> = if self.distinct_target_var.is_some() {
            HashSet::with_capacity(current_indices.len())
        } else {
            HashSet::new()
        };

        // The two ways a later hop can tell two partial matches apart — see
        // `dedup_interchangeable_partials`, which refuses to collapse them.
        let repeats_a_node_variable = repeats_a_node_variable(pattern);
        let mut relationship_state_recorded = false;

        let mut i = 1;
        while i < pattern.elements.len() {
            // max_matches is enforced DURING expansion (inner-loop checks below),
            // not between hops, to avoid breaking before edges are expanded.
            let is_last_hop = i + 2 >= pattern.elements.len();
            if let Some(msg) = self.interrupt_reason() {
                return Err(msg);
            }

            let edge_pattern = match &pattern.elements[i] {
                PatternElement::Edge(ep) => ep,
                _ => return Err("Expected edge pattern after node. Use -[:TYPE]-> for outgoing, <-[:TYPE]- for incoming.".to_string()),
            };

            i += 1;
            if i >= pattern.elements.len() {
                return Err("Edge pattern must be followed by a node pattern. Example: ()-[:KNOWS]->(n:Person)".to_string());
            }

            let node_pattern = match &pattern.elements[i] {
                PatternElement::Node(np) => np,
                _ => return Err("Expected node pattern after edge. Complete the pattern with a node: ()-[:EDGE]->(node)".to_string()),
            };

            // Whether any EARLIER hop recorded relationship identity: only
            // then can `reuses_bound_relationship` reject a row of this hop.
            // Read before this hop folds itself in, below.
            let earlier_relationship_state = relationship_state_recorded;

            let hop = self.plan_hop(
                edge_pattern,
                node_pattern,
                i,
                is_last_hop,
                earlier_relationship_state,
                pass,
            );

            relationship_state_recorded |= hop.track_fixed_trail
                || hop.edge.variable.is_some()
                || hop.anonymous_path_var.is_some();

            // `!seeds_pending` is implied by `max_matches.is_none()` — it is
            // named because the parallel branch zips `matches` against
            // `current_indices`, and a pending seed leaves `matches` empty.
            let (mut new_matches, mut new_indices) = if !seeds_pending
                && matches.len() >= EXPANSION_RAYON_THRESHOLD
                && self.max_matches.is_none()
            {
                self.expand_hop_parallel(&matches, &current_indices, &hop)?
            } else {
                self.expand_hop_sequential(
                    &matches,
                    &current_indices,
                    &hop,
                    seeds_pending.then_some(first_node),
                    &mut distinct_seen,
                    &mut visited,
                )?
            };

            if let Some(msg) = self.interrupt_reason() {
                return Err(msg);
            }

            // The parallel path cannot early-exit, so truncate its overflow here.
            if let Some(max) = hop.limit {
                if new_matches.len() > max {
                    if !is_last_hop {
                        self.note_cap_truncated();
                    }
                    new_matches.truncate(max);
                    new_indices.truncate(max);
                }
            }

            let collapsible = self.distinct_target_var.is_some()
                && !relationship_state_recorded
                && !repeats_a_node_variable
                && i + 1 < pattern.elements.len()
                && node_pattern.variable.is_none();
            (matches, current_indices) =
                dedup_interchangeable_partials(new_matches, new_indices, collapsible);
            // Every later hop reads real matches, not start-node seeds.
            seeds_pending = false;
            i += 1;
        }

        Ok(matches)
    }

    /// Expand one hop across every current match in parallel — each match's
    /// `expand_from_node` is independent. Only reachable with no
    /// `max_matches`: there is no early exit to honour, so every match is
    /// expanded and the caller truncates.
    ///
    /// Errors (deadline, cancellation, expansion failure) are captured by the
    /// shared [`ParallelInterrupt`] latch and the first message is propagated
    /// after the parallel section, so a 100M-source expansion cannot run past
    /// its deadline in a worker thread. The region runs on the dedicated query
    /// pool so its workers have `QUERY_THREAD_STACK_SIZE` stacks.
    fn expand_hop_parallel(
        &self,
        matches: &[PatternMatch],
        current_indices: &[NodeIndex],
        hop: &HopPlan<'_>,
    ) -> Result<(Vec<PatternMatch>, Vec<NodeIndex>), String> {
        let interrupt = ParallelInterrupt::new(|| self.interrupt_reason());
        // Matches emitted by every job so far. The sequential path reads
        // `new_matches.len()` directly; here the buffer is spread across
        // workers until `collect`, so the ceiling needs a shared count —
        // published in blocks of `CEILING_PUBLISH_STRIDE`, never per row, for
        // the cost recorded on that constant.
        //
        // A job whose rows are wide enough to fill a block stops the region
        // *while it is filling*, which is the runaway this ceiling exists for.
        // A job that ends below a block keeps its remainder, so a hop of many
        // narrow rows is bounded at the hop boundary instead, by the
        // `results.len()` check after the region. Both are error paths, never
        // truncation, so neither can turn a too-large answer into a wrong one.
        let produced = std::sync::atomic::AtomicUsize::new(0);
        let results: Vec<(PatternMatch, NodeIndex)> = parallel::install(|| {
            matches
                .par_iter()
                .zip(current_indices.par_iter())
                // `map_init` gives each worker its own reusable visited buffer,
                // so a variable-length hop's marks cost a stamp bump per row
                // here too — the sequential path's buffer cannot be shared
                // across threads. It carries the unpublished match count for
                // the same reason: both are per-worker state.
                .map_init(
                    || (VisitedStamps::default(), 0usize),
                    |(visited, unpublished), (current_match, &source_idx)| {
                        // Short-circuit once any thread has detected a timeout/error,
                        // and independently check the deadline from each thread.
                        // One expansion dwarfs the probe, so poll on every match.
                        if interrupt.check_each().is_err() {
                            return Vec::new();
                        }
                        let Some(expansions) = interrupt.capture(self.expand_from_node(
                            source_idx,
                            hop.edge,
                            hop.node,
                            None,
                            self.bound_target(hop.node, current_match),
                            visited,
                        )) else {
                            return Vec::new();
                        };
                        let kept: Vec<_> = expansions
                            .into_iter()
                            .filter_map(|(target_idx, edge_binding)| {
                                if reuses_bound_relationship(current_match, &edge_binding) {
                                    return None;
                                }
                                if !self.target_satisfies_bindings(
                                    hop.node,
                                    current_match,
                                    target_idx,
                                ) {
                                    return None;
                                }
                                Some((
                                    self.extend_match(current_match, hop, edge_binding, target_idx),
                                    target_idx,
                                ))
                            })
                            .collect();
                        *unpublished += kept.len();
                        if *unpublished >= CEILING_PUBLISH_STRIDE {
                            let held = produced
                                .fetch_add(*unpublished, std::sync::atomic::Ordering::Relaxed)
                                + *unpublished;
                            *unpublished = 0;
                            if interrupt.capture(self.check_match_ceiling(held)).is_none() {
                                return Vec::new();
                            }
                        }
                        kept
                    },
                )
                .flatten()
                .collect()
        });
        interrupt.finish()?;
        // The workers' unpublished remainders never reached the counter, so
        // the authoritative total is the collected buffer itself.
        self.check_match_ceiling(results.len())?;
        // The sequential path dedups distinct targets inline; the parallel path
        // cannot, without synchronization.
        let needs_dedup = hop.is_last_hop
            && self
                .distinct_target_var
                .as_ref()
                .is_some_and(|dtv| hop.node.variable.as_deref() == Some(dtv.as_str()));
        if needs_dedup {
            let mut seen_targets = HashSet::new();
            Ok(results
                .into_iter()
                .filter(|(_, target_idx)| seen_targets.insert(*target_idx))
                .unzip())
        } else {
            Ok(results.into_iter().unzip())
        }
    }

    /// Expand one hop sequentially, honouring `max_matches` by early exit.
    ///
    /// `lazy_seed` is `Some(first_node)` while the start-node seeds have not
    /// been materialised: this hop builds each one as it reaches it, so the
    /// early exit above is what bounds how many ever get built. `None` means
    /// `matches` already holds one entry per entry of `current_indices`.
    fn expand_hop_sequential(
        &self,
        matches: &[PatternMatch],
        current_indices: &[NodeIndex],
        hop: &HopPlan<'_>,
        lazy_seed: Option<&NodePattern>,
        distinct_seen: &mut HashSet<NodeIndex>,
        visited: &mut VisitedStamps,
    ) -> Result<(Vec<PatternMatch>, Vec<NodeIndex>), String> {
        let mut new_matches = Vec::new();
        let mut new_indices = Vec::new();
        let mut expand_count: usize = 0;
        let hop_limit = hop.limit;
        for (position, &source_idx) in current_indices.iter().enumerate() {
            if hop_limit.is_some_and(|max| new_matches.len() >= max) {
                break;
            }
            let pending_seed;
            let current_match = if let Some(first_node) = lazy_seed {
                pending_seed = self.seed_match(first_node, source_idx);
                &pending_seed
            } else if let Some(m) = matches.get(position) {
                m
            } else {
                // Unreachable: the two vectors are pushed in lockstep.
                break;
            };
            let mut remaining = hop_limit.map(|max| max.saturating_sub(new_matches.len()));
            let hint = self.bound_target(hop.node, current_match);
            // A variable-length expansion is capped only where every row it
            // returns is one this loop keeps — see `HopPlan::var_length_cap_safe`
            // for the two hop-level conditions; `hint` is the third: a bound
            // target variable means `target_satisfies_bindings` rejects every
            // peer but one, so a truncated expansion could return only peers.
            if hop.edge.var_length.is_some() && !(hop.var_length_cap_safe && hint.is_none()) {
                remaining = None;
            }
            let expansions =
                self.expand_from_node(source_idx, hop.edge, hop.node, remaining, hint, visited)?;
            for (target_idx, edge_binding) in expansions {
                if reuses_bound_relationship(current_match, &edge_binding) {
                    continue;
                }
                expand_count += 1;
                if expand_count.is_multiple_of(1024) {
                    if let Some(msg) = self.interrupt_reason() {
                        return Err(msg);
                    }
                    // Same stride as the interrupt poll, and for the same
                    // reason: `new_matches` is this hop's held buffer, and
                    // `expand_count` only ever runs ahead of it, so the
                    // ceiling is tested at least once per 1024 pushes.
                    self.check_match_ceiling(new_matches.len())?;
                }
                if hop_limit.is_some_and(|max| new_matches.len() >= max) {
                    break;
                }
                if !self.target_satisfies_bindings(hop.node, current_match, target_idx) {
                    continue;
                }
                // Distinct-target dedup: at the last hop, skip targets already
                // seen — by this pass, or by an earlier execution the caller is
                // sharing a dedup across (`distinct_prior`, read-only).
                if hop.is_last_hop {
                    if let Some(ref dtv) = self.distinct_target_var {
                        if hop.node.variable.as_deref() == Some(dtv.as_str())
                            && (self
                                .distinct_prior
                                .is_some_and(|prior| prior.contains(&target_idx))
                                || !distinct_seen.insert(target_idx))
                        {
                            continue;
                        }
                    }
                }
                new_matches.push(self.extend_match(current_match, hop, edge_binding, target_idx));
                new_indices.push(target_idx);
            }
        }
        // An intermediate hop that filled its advisory limit was cut short:
        // the loop may have stopped before the last source, and
        // `expand_from_node` itself stops at `remaining` — so a hop that
        // produced exactly the limit cannot tell "that is all there was" from
        // "the rest was dropped". Either way the matches that would have
        // reached the final hop may be in the part never expanded, so the pass
        // is not authoritative; a false positive costs one retry. The last hop
        // is exempt: there `hop.limit` IS `max_matches`, and filling it is the
        // answer.
        if !hop.is_last_hop && hop_limit.is_some_and(|max| new_matches.len() >= max) {
            self.note_cap_truncated();
        }
        self.check_match_ceiling(new_matches.len())?;
        Ok((new_matches, new_indices))
    }

    /// Whether `target_idx` is allowed for this hop's node pattern: it must
    /// agree with any pre-binding of the variable, and with any earlier
    /// binding of the same variable inside this pattern.
    #[inline]
    fn target_satisfies_bindings(
        &self,
        node_pattern: &NodePattern,
        current_match: &PatternMatch,
        target_idx: NodeIndex,
    ) -> bool {
        let Some(ref var) = node_pattern.variable else {
            return true;
        };
        if let Some(&bound_idx) = self.pre_bindings.get(var) {
            if target_idx != bound_idx {
                return false;
            }
        }
        let already_bound = current_match.bindings.iter().find_map(|(name, binding)| {
            if name == var {
                match binding {
                    MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => Some(*index),
                    _ => None,
                }
            } else {
                None
            }
        });
        already_bound.is_none_or(|bound_idx| target_idx == bound_idx)
    }

    #[inline]
    fn extend_match(
        &self,
        current_match: &PatternMatch,
        hop: &HopPlan<'_>,
        edge_binding: MatchBinding,
        target_idx: NodeIndex,
    ) -> PatternMatch {
        let mut new_match = current_match.clone();
        if hop.track_fixed_trail {
            extend_fixed_trail(&mut new_match, &edge_binding);
        }
        if let Some(ref var) = hop.edge.variable {
            new_match.bindings.push((var.clone(), edge_binding));
        } else if let Some(ref internal_var) = hop.anonymous_path_var {
            new_match
                .bindings
                .push((internal_var.clone(), edge_binding));
        }
        if let Some(ref var) = hop.node.variable {
            new_match
                .bindings
                .push((var.clone(), self.node_to_binding(target_idx)));
        }
        new_match
    }

    /// The start-node binding for one match — the first node pattern's
    /// variable bound to `idx`, or an empty match when it is anonymous.
    #[inline]
    fn seed_match(&self, first_node: &NodePattern, idx: NodeIndex) -> PatternMatch {
        let mut pm = PatternMatch {
            bindings: Vec::new(),
            exact_path: None,
        };
        if let Some(ref var) = first_node.variable {
            pm.bindings.push((var.clone(), self.node_to_binding(idx)));
        }
        pm
    }
}

#[cfg(test)]
mod tests {
    use super::repeats_a_node_variable;
    use crate::graph::core::pattern_matching::parse_pattern;

    #[test]
    fn a_node_variable_written_twice_is_reported() {
        for text in [
            "(a)-[:A]->(b)-[:B]->(a)",
            "(a)-[:A]->()-[:B]->(a)",
            "(a:N)-[:A]->(a)",
        ] {
            let pattern = parse_pattern(text).unwrap_or_else(|e| panic!("{text}: {e}"));
            assert!(repeats_a_node_variable(&pattern), "{text}");
        }
    }

    #[test]
    fn distinct_and_anonymous_variables_are_not_repeats() {
        for text in [
            "(a)-[:A]->(b)-[:B]->(c)",
            "(a)-[:A]->()-[:B]->(b)",
            "()-[:A]->()-[:B]->()",
        ] {
            let pattern = parse_pattern(text).unwrap_or_else(|e| panic!("{text}: {e}"));
            assert!(!repeats_a_node_variable(&pattern), "{text}");
        }
    }
}
