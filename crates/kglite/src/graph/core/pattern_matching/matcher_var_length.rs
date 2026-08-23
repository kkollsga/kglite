//! Variable-length segment (`-[:T*min..max]-`) expansion for
//! [`PatternExecutor`].
//!
//! Two implementations of the same relation, chosen per segment:
//!
//! - [`PatternExecutor::expand_var_length`] — the exact one. A BFS over
//!   *trails*: it carries each candidate's relationship sequence, so Cypher's
//!   rule that one relationship occurs at most once per path is enforced and
//!   `p = ...` / a named edge variable can be answered.
//! - [`PatternExecutor::expand_var_length_fast`] — a BFS over *distances*
//!   with one global visited set. That is a different relation, and it only
//!   coincides with the trail one under the conditions documented on it.
//!
//! Split out of `matcher.rs` (at its file-size ceiling); these are inherent
//! methods on `PatternExecutor`, so the split is a file boundary only.

use super::*;
use crate::graph::core::iterators::GraphEdgeRef;
use petgraph::graph::EdgeIndex;
use std::collections::VecDeque;

/// Edge examinations [`PatternExecutor::source_closed_trail`] may spend before
/// giving up. The probe enumerates trails, not nodes, so it has no polynomial
/// bound of its own; the budget is what makes it safe to run per source node.
/// Exhausting it is not a wrong answer — the caller falls back to the exact
/// per-path expansion.
const CLOSED_TRAIL_PROBE_BUDGET: usize = 20_000;

/// One variable-length segment as both expansions read it: the pattern pair
/// that defines it plus its resolved hop range. Bundled because the two
/// entry points take it whole and nothing varies these four independently.
pub(super) struct VarLengthSegment<'p> {
    pub edge: &'p EdgePattern,
    pub node: &'p NodePattern,
    pub min_hops: usize,
    pub max_hops: usize,
}

/// How many nodes a *capped* row may mark before its marks move from a hash set
/// to the graph-sized dense array.
///
/// Only capped rows start sparse at all (see [`VisitedStamps::begin`]), so this
/// is the backstop for a cap generous enough that the BFS runs on anyway. Small
/// on purpose: past it the set is pure overhead on top of the array the row
/// ends up allocating regardless.
const VISITED_DENSE_PROMOTION: usize = 128;

/// Reusable "already reached" marks for [`PatternExecutor::expand_var_length_fast`].
///
/// It replaces a `vec![false; node_bound]` **per source row** — an allocation
/// plus a zeroing pass whose cost scaled with the *graph* rather than with the
/// work done. Measured on a 50-row EXISTS witness: 31.5 µs at 10k nodes,
/// 36.7 µs at 40k, 88.9 µs at 160k, against a flat ~16 µs fixed-hop control —
/// ~7.4 ns per 1 000 nodes per row, for a BFS that stops at the first witness
/// and touches a handful of nodes.
///
/// Two things fix that, and both are needed. **Lazy sizing:** a row whose caller
/// capped the result count may stop after a handful of nodes, so it marks into a
/// hash set and only allocates the dense array if it turns out to keep going.
/// That is what the EXISTS shape needs, because its pattern predicate builds one
/// `PatternExecutor` per candidate row, so a per-executor buffer would be
/// re-allocated just as often as the per-row one was. An *uncapped* row sweeps
/// its whole reachable set by definition and starts dense. **Generation
/// stamps:** once the array exists, the following rows of the same expansion
/// re-use it by bumping a stamp instead of re-zeroing it. The stamp is a `u8`
/// deliberately — the footprint then stays exactly the one byte per node the
/// `Vec<bool>` had, and the only price is re-zeroing once every 255 rows when
/// the generation wraps.
#[derive(Default)]
pub(super) struct VisitedStamps {
    /// Live while `dense` is empty: the nodes this row has marked.
    sparse: rustc_hash::FxHashSet<usize>,
    /// Graph-sized marks, allocated on promotion and re-used by later rows.
    dense: Vec<u8>,
    /// Never zero: zero is the "no row has marked this node" value `dense` is
    /// (re)filled with, so a stale mark can never equal a live generation.
    generation: u8,
    node_bound: usize,
}

impl VisitedStamps {
    /// Start a new row's marks over `node_bound` nodes. O(marks written by the
    /// previous row) while sparse, O(1) once dense — except on the wrap.
    ///
    /// `may_stop_early` is the caller's result cap: without one the row visits
    /// its entire reachable set, so the dense array is the right shape from the
    /// first mark and the sparse phase would be pure overhead.
    fn begin(&mut self, node_bound: usize, may_stop_early: bool) {
        self.node_bound = node_bound;
        self.sparse.clear();
        if self.dense.is_empty() {
            if !may_stop_early {
                self.promote();
            }
            return;
        }
        if self.dense.len() < node_bound {
            self.dense.resize(node_bound, 0);
        }
        self.generation = match self.generation.checked_add(1) {
            Some(next) => next,
            None => {
                self.dense.fill(0);
                1
            }
        };
    }

    #[inline]
    fn is_visited(&self, index: usize) -> bool {
        if self.dense.is_empty() {
            self.sparse.contains(&index)
        } else {
            self.dense.get(index).is_some_and(|m| *m == self.generation)
        }
    }

    #[inline]
    fn visit(&mut self, index: usize) {
        if !self.dense.is_empty() {
            if let Some(mark) = self.dense.get_mut(index) {
                *mark = self.generation;
            }
            return;
        }
        self.sparse.insert(index);
        if self.sparse.len() >= VISITED_DENSE_PROMOTION {
            self.promote();
        }
    }

    /// Move this row's marks into the dense array, which every later row of the
    /// same expansion then re-uses.
    #[cold]
    fn promote(&mut self) {
        self.dense = vec![0u8; self.node_bound.max(self.sparse.len())];
        self.generation = 1;
        for &index in &self.sparse {
            if let Some(mark) = self.dense.get_mut(index) {
                *mark = 1;
            }
        }
        self.sparse.clear();
    }
}

/// What the source node contributes to its *own* variable-length segment.
///
/// Split out of the answer itself so the expensive arm can be deferred: under
/// a `max_results` cap a witness found by the BFS answers the segment, and the
/// probe below never has to run.
enum SourceRole {
    /// The source is not a legal target of its own segment.
    Absent,
    /// `min_hops == 0`: the zero-length row, and nothing else.
    ZeroHop(MatchBinding),
    /// Directed `min_hops == 1`: leaving the source unvisited lets the
    /// ordinary BFS rediscover it at its shortest closed walk, which in a
    /// directed graph is a simple cycle and therefore a trail.
    Rediscover,
    /// Undirected `min_hops == 1`: only [`PatternExecutor::source_closed_trail`]
    /// can answer, and it costs up to [`CLOSED_TRAIL_PROBE_BUDGET`] edge
    /// examinations — so it runs last, and only when the rows already found
    /// have not satisfied the caller's cap.
    Probe,
}

enum ClosedTrail {
    /// A closed trail of this many hops leaves and returns to the source.
    Found(usize),
    /// No closed trail within `max_hops` exists — proven exhaustively.
    Absent,
    /// The probe ran out of budget. Nothing is proven either way, so the
    /// caller must answer the whole segment with the exact per-path
    /// expansion.
    Undecided,
}

/// A segment's relationship-type filter, interned once per expansion so the
/// inner loop compares `u64`s instead of strings.
struct ConnFilter {
    /// `[:A|B]` — the relationship's type must be one of these.
    any_of: Option<Vec<InternedKey>>,
    /// `[:A]` — a single type. Also handed to the backend, where a disk CSR
    /// can pre-filter on it without materialising `EdgeData`.
    one: Option<InternedKey>,
}

impl ConnFilter {
    fn new(edge_pattern: &EdgePattern) -> Self {
        let any_of: Option<Vec<InternedKey>> = edge_pattern
            .connection_types
            .as_ref()
            .map(|types| types.iter().map(|t| InternedKey::from_str(t)).collect());
        let one = if any_of.is_none() {
            edge_pattern
                .connection_type
                .as_ref()
                .map(|ct| InternedKey::from_str(ct))
        } else {
            None
        };
        ConnFilter { any_of, one }
    }

    #[inline]
    fn backend_hint(&self) -> Option<InternedKey> {
        self.one
    }

    #[inline]
    fn accepts(&self, conn_type: InternedKey) -> bool {
        match (&self.any_of, self.one) {
            (Some(keys), _) => keys.contains(&conn_type),
            (None, Some(key)) => conn_type == key,
            (None, None) => true,
        }
    }
}

#[inline]
fn cap_reached(found: usize, max_results: Option<usize>) -> bool {
    max_results.is_some_and(|max| found >= max)
}

fn segment_directions(edge_pattern: &EdgePattern) -> &'static [Direction] {
    match edge_pattern.direction {
        EdgeDirection::Outgoing => &[Direction::Outgoing],
        EdgeDirection::Incoming => &[Direction::Incoming],
        EdgeDirection::Both => &[Direction::Outgoing, Direction::Incoming],
    }
}

impl<'a> PatternExecutor<'a> {
    /// `Some(connection_type)` when this relationship passes the segment's
    /// type and property filters, `None` when it is rejected.
    ///
    /// The type check is a `u64` compare and never materialises anything; the
    /// edge's properties are read only when the pattern actually asks for
    /// them, which on a disk backend is the difference between reading the
    /// endpoint table and reading the property blob.
    #[inline]
    fn var_length_edge_accepts(
        &self,
        edge: &GraphEdgeRef<'_>,
        edge_pattern: &EdgePattern,
        conn: &ConnFilter,
    ) -> Option<InternedKey> {
        let conn_type = edge.connection_type();
        if !conn.accepts(conn_type) {
            return None;
        }
        if let Some(props) = edge_pattern.properties.as_ref() {
            let edge_data = edge.weight();
            let matches = props.iter().all(|(key, matcher)| {
                edge_data
                    .get_property(key)
                    .map(|v| self.value_matches(v, matcher))
                    .unwrap_or(false)
            });
            if !matches {
                return None;
            }
        }
        Some(conn_type)
    }

    /// Whether `idx` is an acceptable target of this variable-length segment.
    ///
    /// Mirrors the emission test inside the expansion loops exactly, including
    /// the planner's `skip_target_type_check` guarantee — a node arriving over
    /// a relationship of the marked type has the pattern's label by
    /// construction.
    #[inline]
    fn matches_var_length_target(
        &self,
        idx: NodeIndex,
        edge_pattern: &EdgePattern,
        node_pattern: &NodePattern,
    ) -> bool {
        if !edge_pattern.skip_target_type_check
            && !self.node_matches_pattern_labels(idx, node_pattern)
        {
            return false;
        }
        match node_pattern.properties.as_ref() {
            Some(props) => self.node_matches_properties(idx, props),
            None => true,
        }
    }

    /// Does a closed trail of length `1..=max_hops` leave and return to
    /// `source` under this edge pattern?
    ///
    /// A breadth-first enumeration of trails from `source`, stopping at the
    /// first relationship that lands back on it — so it returns the *shortest*
    /// closed trail, usually at depth 1 or 2, and it explores nothing at all
    /// when the source has no matching relationships. There is no visited set:
    /// a trail may revisit nodes, only relationships are unique, and dropping
    /// a revisit would lose exactly the answers this exists to find.
    /// [`CLOSED_TRAIL_PROBE_BUDGET`] bounds the enumeration.
    fn source_closed_trail(
        &self,
        source: NodeIndex,
        edge_pattern: &EdgePattern,
        max_hops: usize,
        conn: &ConnFilter,
        directions: &[Direction],
    ) -> Result<ClosedTrail, String> {
        let mut queue: VecDeque<(NodeIndex, usize, Vec<EdgeIndex>)> = VecDeque::new();
        queue.push_back((source, 0, Vec::new()));
        let mut budget = CLOSED_TRAIL_PROBE_BUDGET;
        let mut popped: usize = 0;

        while let Some((current, depth, trail)) = queue.pop_front() {
            popped += 1;
            if popped.is_multiple_of(256) {
                if let Some(msg) = self.interrupt_reason() {
                    return Err(msg);
                }
            }
            for &direction in directions {
                for edge in self.graph.graph.edges_directed_filtered(
                    current,
                    direction,
                    conn.backend_hint(),
                ) {
                    if budget == 0 {
                        return Ok(ClosedTrail::Undecided);
                    }
                    budget -= 1;

                    if self
                        .var_length_edge_accepts(&edge, edge_pattern, conn)
                        .is_none()
                    {
                        continue;
                    }
                    let edge_index = edge.id();
                    if trail.contains(&edge_index) {
                        continue;
                    }
                    let target = match direction {
                        Direction::Outgoing => edge.target(),
                        Direction::Incoming => edge.source(),
                    };
                    if target == source {
                        return Ok(ClosedTrail::Found(depth + 1));
                    }
                    if depth + 1 < max_hops {
                        let mut next = trail.clone();
                        next.push(edge_index);
                        queue.push_back((target, depth + 1, next));
                    }
                }
            }
        }

        Ok(ClosedTrail::Absent)
    }

    /// What the source node contributes to its own segment.
    ///
    /// Cypher's trail semantics make the source a legal target of its own
    /// segment whenever a closed trail of length in `[max(min_hops, 1),
    /// max_hops]` returns to it — a fact the distance BFS structurally cannot
    /// see, because it pre-marks the source visited. `min_hops == 0` already
    /// answers the source at zero hops, so only `min_hops == 1` asks.
    ///
    /// A **directed** segment gets the answer for free: the shortest closed
    /// *walk* from a node in a directed graph never repeats a vertex — a
    /// repeat could be cut out for a shorter walk — so it is a simple cycle
    /// and therefore a trail. Leaving the source unvisited lets the ordinary
    /// BFS discover it at exactly that length.
    ///
    /// An **undirected** segment's shortest closed walk is the degenerate
    /// there-and-back over one relationship, which the trail rule forbids, so
    /// the BFS would answer 2 for every source with a neighbour. That case
    /// needs the explicit trail probe, which the caller runs only if it still
    /// needs the row.
    fn var_length_source_role(
        &self,
        source: NodeIndex,
        edge_pattern: &EdgePattern,
        node_pattern: &NodePattern,
        min_hops: usize,
        max_hops: usize,
    ) -> SourceRole {
        if min_hops == 0 {
            return if self.matches_var_length_target_strictly(source, node_pattern) {
                SourceRole::ZeroHop(self.zero_length_binding(source, 0))
            } else {
                SourceRole::Absent
            };
        }
        if max_hops == 0 || !self.matches_var_length_target(source, edge_pattern, node_pattern) {
            return SourceRole::Absent;
        }
        if !matches!(edge_pattern.direction, EdgeDirection::Both) {
            SourceRole::Rediscover
        } else {
            SourceRole::Probe
        }
    }

    /// The zero-hop arm's target test: the source did not arrive over a
    /// relationship, so the planner's `skip_target_type_check` guarantee does
    /// not cover it and the labels are always checked.
    #[inline]
    fn matches_var_length_target_strictly(
        &self,
        idx: NodeIndex,
        node_pattern: &NodePattern,
    ) -> bool {
        self.node_matches_pattern_labels(idx, node_pattern)
            && match node_pattern.properties.as_ref() {
                Some(props) => self.node_matches_properties(idx, props),
                None => true,
            }
    }

    #[inline]
    fn zero_length_binding(&self, node: NodeIndex, hops: usize) -> MatchBinding {
        MatchBinding::VariableLengthPath {
            source: node,
            target: node,
            hops,
            path: Vec::new(),
        }
    }

    /// Answers **distance** reachability: each node is visited at most once,
    /// so hub nodes are never re-explored at deeper depths. Cypher asks for
    /// **trail** reachability, and the two relations only coincide when
    /// `min_hops <= 1`:
    ///
    /// - For `min_hops <= 1` every node at distance `1..=max_hops` is
    ///   trail-reachable (a shortest path is a trail) and every
    ///   trail-reachable node is within that distance — *except the source
    ///   itself*, handled by [`Self::var_length_source_results`].
    /// - For `min_hops >= 2` there is no set-based computation to make:
    ///   `(a)-[:R*2..2]-(b)` on a triangle is trail-reachable from `a` to both
    ///   peers and distance-reachable to neither. Those stay on the per-path
    ///   expansion; [`Self::expand_var_length`] enforces it.
    ///
    /// `max_results` is an **exact** cap the caller has authorised: it returns
    /// as soon as that many rows exist, and the deferred source probe below is
    /// then skipped entirely — a witness already answers the segment. Only
    /// callers whose post-filters accept every row this returns may pass one
    /// (see `HopPlan::var_length_cap_safe`).
    ///
    /// Returns `Ok(None)` when source inclusion could not be decided within
    /// budget — the caller must then answer the whole segment with the exact
    /// per-path expansion.
    fn expand_var_length_fast(
        &self,
        source: NodeIndex,
        segment: &VarLengthSegment<'_>,
        max_results: Option<usize>,
        visited: &mut VisitedStamps,
    ) -> Result<Option<Vec<(NodeIndex, MatchBinding)>>, String> {
        let VarLengthSegment {
            edge: edge_pattern,
            node: node_pattern,
            min_hops,
            max_hops,
        } = *segment;
        debug_assert!(
            min_hops <= 1,
            "the distance BFS is not trail-equivalent for min_hops >= 2"
        );

        let directions = segment_directions(edge_pattern);
        let conn = ConnFilter::new(edge_pattern);

        let role =
            self.var_length_source_role(source, edge_pattern, node_pattern, min_hops, max_hops);
        let mut results: Vec<(NodeIndex, MatchBinding)> = Vec::new();
        let mut leave_source_unvisited = false;
        let mut probe_pending = false;
        match role {
            SourceRole::ZeroHop(binding) => results.push((source, binding)),
            SourceRole::Rediscover => leave_source_unvisited = true,
            SourceRole::Probe => probe_pending = true,
            SourceRole::Absent => {}
        }
        if cap_reached(results.len(), max_results) {
            return Ok(Some(results));
        }

        // Global visited set — each node is explored at most once. Reusing the
        // caller's `VisitedStamps` is what keeps the cost proportional to the
        // marks this row writes rather than to the graph's node count.
        visited.begin(self.graph.graph.node_bound(), max_results.is_some());
        if !leave_source_unvisited {
            visited.visit(source.index());
        }

        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        queue.push_back((source, 0));

        let mut iter_count: usize = 0;

        while let Some((current, depth)) = queue.pop_front() {
            iter_count += 1;
            if iter_count & 511 == 0 {
                if let Some(dl) = self.deadline {
                    if Instant::now() > dl {
                        return Err("Query timed out".to_string());
                    }
                }
            }
            if depth >= max_hops {
                continue;
            }

            for &direction in directions {
                let edges = self.graph.graph.edges_directed_filtered(
                    current,
                    direction,
                    conn.backend_hint(),
                );

                let mut inner_iter: usize = 0;
                for edge in edges {
                    inner_iter += 1;
                    // Inner-loop deadline check. A 1-2 hop fan-out from a hub
                    // like Q42 can push hundreds of millions of inner
                    // iterations between the outer `iter_count & 511` check
                    // — without this the 20 s deadline overshoots to 30+ s.
                    if inner_iter.is_multiple_of(1 << 20) {
                        if let Some(dl) = self.deadline {
                            if Instant::now() > dl {
                                return Err("Query timed out".to_string());
                            }
                        }
                    }
                    if self
                        .var_length_edge_accepts(&edge, edge_pattern, &conn)
                        .is_none()
                    {
                        continue;
                    }

                    let target = match direction {
                        Direction::Outgoing => edge.target(),
                        Direction::Incoming => edge.source(),
                    };

                    let target_idx = target.index();
                    if visited.is_visited(target_idx) {
                        continue;
                    }
                    visited.visit(target_idx);

                    let new_depth = depth + 1;

                    if new_depth >= min_hops
                        && self.matches_var_length_target(target, edge_pattern, node_pattern)
                    {
                        results.push((
                            target,
                            MatchBinding::VariableLengthPath {
                                source,
                                target,
                                hops: new_depth,
                                path: Vec::new(),
                            },
                        ));
                        if cap_reached(results.len(), max_results) {
                            // The caller asked for this many rows and no more,
                            // so the deferred source probe is moot: whatever it
                            // would have added, these rows already answer.
                            return Ok(Some(results));
                        }
                    }

                    // The source is never re-expanded: reaching it here means it
                    // was left unvisited for the closed-trail answer, and every
                    // relationship it has was already walked at depth 0.
                    if new_depth < max_hops && target != source {
                        queue.push_back((target, new_depth));
                    }
                }
            }
        }

        // Deferred: the undirected closed-trail probe. Reached only when the
        // BFS did not already satisfy the caller's cap, so an existence check
        // with a witness never pays for it. `insert(0, …)` keeps the source
        // row where the eager version put it — first.
        if probe_pending {
            match self.source_closed_trail(source, edge_pattern, max_hops, &conn, directions)? {
                ClosedTrail::Found(hops) => {
                    results.insert(0, (source, self.zero_length_binding(source, hops)));
                }
                ClosedTrail::Absent => {}
                ClosedTrail::Undecided => return Ok(None),
            }
        }

        Ok(Some(results))
    }

    /// The exact expansion: a BFS over trails within the hop range, carrying
    /// each candidate's relationship sequence.
    ///
    /// `max_results` is an exact cap: expansion stops as soon as that many
    /// rows exist. A `min_hops >= 2` segment cannot use the set-based path at
    /// all, but it can still stop at the first complete trail — which is all
    /// an existence check needs.
    pub(super) fn expand_var_length(
        &self,
        source: NodeIndex,
        segment: &VarLengthSegment<'_>,
        max_results: Option<usize>,
        visited: &mut VisitedStamps,
    ) -> Result<Vec<(NodeIndex, MatchBinding)>, String> {
        let VarLengthSegment {
            edge: edge_pattern,
            node: node_pattern,
            min_hops,
            max_hops,
        } = *segment;
        // Fast path: when path info isn't needed, use global-dedup BFS.
        // `min_hops <= 1` is a *correctness* condition, not a heuristic — see
        // `expand_var_length_fast`. The planner's `mark_fast_var_length_paths`
        // gate applies the same rule; this is the executor-side backstop for
        // every other producer of an `EdgePattern`. `Ok(None)` means the fast
        // path could not decide source inclusion within budget.
        if !edge_pattern.needs_path_info && min_hops <= 1 {
            if let Some(fast) =
                self.expand_var_length_fast(source, segment, max_results, visited)?
            {
                return Ok(fast);
            }
        }

        let mut results = Vec::new();
        let directions = segment_directions(edge_pattern);
        let conn = ConnFilter::new(edge_pattern);

        // BFS state: (current_node, depth, exact trail).  Edge identity is
        // required both for parallel-edge cardinality and for the Cypher rule
        // that a relationship occurs at most once in a path.
        type PathInfo = Vec<PathHop>;
        let mut queue: VecDeque<(NodeIndex, usize, PathInfo)> = VecDeque::new();

        queue.push_back((source, 0, Vec::new()));

        if min_hops == 0 && self.matches_var_length_target_strictly(source, node_pattern) {
            results.push((source, self.zero_length_binding(source, 0)));
            if cap_reached(results.len(), max_results) {
                return Ok(results);
            }
        }

        let mut vlp_count: usize = 0;
        while let Some((current, depth, path)) = queue.pop_front() {
            vlp_count += 1;
            if vlp_count.is_multiple_of(512) {
                if let Some(dl) = self.deadline {
                    if Instant::now() > dl {
                        return Err("Query timed out".to_string());
                    }
                }
                // The frontier is the other buffer this loop holds, and it
                // grows even where the node pattern rejects every target (so
                // `results` stays small). Each entry owns its trail, so it is
                // the more expensive of the two per element.
                self.check_match_ceiling(queue.len())?;
            }
            if depth >= max_hops {
                continue;
            }

            // Collected before any is walked, so an edge yielded twice by the
            // two directional iterators is admitted once (see below).
            let mut valid_targets: Vec<PathHop> = Vec::new();

            for &direction in directions {
                let edges = self.graph.graph.edges_directed_filtered(
                    current,
                    direction,
                    conn.backend_hint(),
                );

                for edge in edges {
                    let Some(conn_type) = self.var_length_edge_accepts(&edge, edge_pattern, &conn)
                    else {
                        continue;
                    };

                    let target = match direction {
                        Direction::Outgoing => edge.target(),
                        Direction::Incoming => edge.source(),
                    };

                    let edge_index = edge.id();
                    // Paths are relationship-unique trails. Repeated nodes are
                    // valid, but traversing the same edge again (including in
                    // reverse for an undirected pattern) is not.
                    if path.iter().any(|hop| hop.edge == edge_index) {
                        continue;
                    }
                    // A self-loop appears in both directional iterators for an
                    // undirected edge. It is still one candidate relationship.
                    if valid_targets.iter().any(|hop| hop.edge == edge_index) {
                        continue;
                    }

                    valid_targets.push(PathHop {
                        node: target,
                        edge: edge_index,
                        connection_type: conn_type,
                    });
                }
            }

            let new_depth = depth + 1;

            for hop in valid_targets {
                let target = hop.node;
                let needs_queue = new_depth < max_hops;

                let mut new_path = path.clone();
                new_path.push(hop);

                if new_depth >= min_hops
                    && self.matches_var_length_target(target, edge_pattern, node_pattern)
                {
                    let path_for_binding = if needs_queue {
                        new_path.clone()
                    } else {
                        std::mem::take(&mut new_path)
                    };
                    results.push((
                        target,
                        MatchBinding::VariableLengthPath {
                            source,
                            target,
                            hops: new_depth,
                            path: path_for_binding,
                        },
                    ));
                    if cap_reached(results.len(), max_results) {
                        return Ok(results);
                    }
                    // Trail expansion is combinatorial in the hop count, and
                    // `max_results` is `None` on every path whose post-filters
                    // could reject a row — so without this the only bound on
                    // `results` is the number of trails the pattern admits.
                    // Tested per push: it is one comparison against a
                    // register-resident `Option<usize>`, and the alternative
                    // (a stride) lets a single high-degree pop overshoot by
                    // its whole fan-out.
                    self.check_match_ceiling(results.len())?;
                }

                if needs_queue {
                    queue.push_back((target, new_depth, new_path));
                }
            }
        }

        Ok(results)
    }
}
