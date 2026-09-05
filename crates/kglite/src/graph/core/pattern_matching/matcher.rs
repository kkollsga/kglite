use crate::datatypes::values::Value;
use crate::graph::core::filtering::{compare_values, str_values_equal, values_equal};
use crate::graph::dir_graph::indexes::predicate_queries::string_index_hits;
use crate::graph::languages::cypher::executor::budget::MatchCeiling;
use crate::graph::languages::cypher::result::Bindings;
use crate::graph::schema::{DirGraph, InternedKey, NodeData};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::{GraphRead, NodeView};
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use crate::graph::parallel::{self, ParallelInterrupt};

use super::closure_probe;
use super::column_filter::{self, ColumnFilter};
use super::pattern::{
    AnchorSide, ConnTypeFilter, EdgeDirection, EdgePattern, MatchBinding, NodePattern, PathHop,
    Pattern, PatternElement, PatternMatch, PropertyMatcher,
};

/// Minimum match count to use parallel expansion via rayon.
/// Set high: each expand_from_node does light work (a few edge iterations),
/// so rayon overhead only pays off for very large match sets. Also avoids
/// contention when multiple queries run concurrently (shared thread pool).
const EXPANSION_RAYON_THRESHOLD: usize = 8192;

/// Candidate-scan partitions per worker. More than one so a partition holding
/// the expensive rows cannot stall the scan, few enough that concatenating them
/// stays a rounding error. Matches the fused scan aggregate's factor.
const CANDIDATE_PARTITIONS_PER_WORKER: usize = 4;

/// One property matcher with its field name already alias-resolved, and that
/// name's interned key precomputed.
struct ResolvedMatcher<'a> {
    field: &'a str,
    key: InternedKey,
    matcher: &'a PropertyMatcher,
}

/// Everything a candidate scan can resolve once per node **type** instead of
/// once per candidate — all of it a function of the node's type alone.
///
/// Resolving them per node cost a full-type text-filter scan roughly 40% of its
/// runtime: two `String`-keyed hash probes in `DirGraph::resolve_alias`, one
/// interner probe for the type name, one FNV hash per property, and one store
/// probe inside `GraphRead::node_view`, all recomputing the same answer 10 000
/// times.
struct TypeScanMemo<'a> {
    /// The type this memo is valid for. A mixed candidate stream (primary
    /// `type_indices` ∪ secondary-label hits) rebuilds when this changes, so
    /// the memo never answers for the wrong type.
    type_key: InternedKey,
    type_str: &'a str,
    store: Option<&'a std::sync::Arc<ColumnStore>>,
    props: Vec<ResolvedMatcher<'a>>,
    /// The same matchers compiled to the columns they read, when every one of
    /// them resolves through exactly one column of this type's store. `None`
    /// means the row route answers — see [`ColumnFilter`] for the decline list.
    filter: Option<ColumnFilter<'a>>,
}

/// Whether adding `candidate` would reuse a relationship already consumed by
/// this pattern match. Cypher paths are trails: nodes may repeat, edges may not.
fn reuses_bound_relationship(current: &PatternMatch, candidate: &MatchBinding) -> bool {
    let fixed_path_uses = |edge| {
        current
            .exact_path
            .as_deref()
            .is_some_and(|(_, path)| path.iter().any(|hop| hop.edge == edge))
    };
    let candidate_edges = match candidate {
        MatchBinding::Edge { edge_index, .. } => std::slice::from_ref(edge_index),
        MatchBinding::VariableLengthPath { path, .. } => {
            return path.iter().any(|hop| {
                fixed_path_uses(hop.edge)
                    || current.bindings.iter().any(|(_, binding)| match binding {
                        MatchBinding::Edge { edge_index, .. } => *edge_index == hop.edge,
                        MatchBinding::VariableLengthPath { path, .. } => {
                            path.iter().any(|bound| bound.edge == hop.edge)
                        }
                        _ => false,
                    })
            });
        }
        _ => return false,
    };

    candidate_edges.iter().any(|candidate_edge| {
        fixed_path_uses(*candidate_edge)
            || current.bindings.iter().any(|(_, binding)| match binding {
                MatchBinding::Edge { edge_index, .. } => *edge_index == *candidate_edge,
                MatchBinding::VariableLengthPath { path, .. } => {
                    path.iter().any(|hop| hop.edge == *candidate_edge)
                }
                _ => false,
            })
    })
}

fn extend_fixed_trail(current: &mut PatternMatch, candidate: &MatchBinding) {
    let MatchBinding::Edge {
        source,
        target,
        edge_index,
        connection_type,
        ..
    } = candidate
    else {
        return;
    };
    let hop = PathHop {
        node: *target,
        edge: *edge_index,
        connection_type: *connection_type,
    };

    if let Some(exact_path) = &mut current.exact_path {
        exact_path.1.push(hop);
        return;
    }

    current.exact_path = Some(Box::new((*source, vec![hop])));
}

/// Return the ordered list of index-name candidates to try when the
/// cross-type fast path sees a query for `prop`. The first entry is
/// always `prop` itself.
///
/// Two sources of aliases: the hardcoded `title ↔ label ↔ name` family, and the
/// per-type `title_field_aliases` / `id_field_aliases` registered on `DirGraph`
/// — so a type that registered `'original_name'` as its title alias also serves
/// `{title: 'X'}` from the `original_name` index, with no new config API.
fn global_alias_candidates(prop: &str, graph: &DirGraph) -> Vec<String> {
    let mut out: Vec<String> = vec![prop.to_string()];
    let (family, per_type_map): (&[&str], &FxHashMap<String, String>) = match prop {
        "title" | "label" | "name" => (&["title", "label", "name"], &graph.title_field_aliases),
        // `nid`/`qid` are NO LONGER id-aliases (0.11.0 cross-mode parity): the
        // node id is the compact integer in every mode, and the string form
        // (`"Q42"`) is the plain `nid` property — so `{nid: X}` resolves as an
        // ordinary (indexed) string property, identically across modes, rather
        // than coercing into the integer id-index. Only `id` (+ per-type
        // user aliases) routes to the id-index.
        "id" => (&["id"], &graph.id_field_aliases),
        _ => return out,
    };
    for &sibling in family {
        let s = sibling.to_string();
        if !out.contains(&s) {
            out.push(s);
        }
    }
    for alias in per_type_map.values() {
        if !out.contains(alias) {
            out.push(alias.clone());
        }
    }
    out
}

/// `str::ends_with` with the mismatch decided on one byte.
///
/// `str::ends_with` on a runtime-length pattern lowers to a `memcmp` call
/// through the dynamic-linker stub, and a filtered scan calls it once per
/// candidate row while almost every row fails. Comparing the last byte first
/// takes the call out of the failing path, which is the path a scan is.
#[inline]
fn str_ends_with(s: &str, suffix: &str) -> bool {
    let (haystack, needle) = (s.as_bytes(), suffix.as_bytes());
    match (needle.last(), haystack.last()) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(n), Some(h)) => {
            n == h && haystack.len() >= needle.len() && {
                let at = haystack.len() - needle.len();
                &haystack[at..] == needle
            }
        }
    }
}

/// Drop repeat `NodeIndex` entries from an index-built candidate list,
/// keeping each node's **first** occurrence.
///
/// One pass, so the no-duplicate case (every list-driven anchor's normal
/// shape) pays one hash insert per candidate and no reallocation.
fn dedup_candidates(candidates: &mut Vec<NodeIndex>) {
    if candidates.len() < 2 {
        return;
    }
    let mut seen: rustc_hash::FxHashSet<NodeIndex> =
        rustc_hash::FxHashSet::with_capacity_and_hasher(candidates.len(), Default::default());
    candidates.retain(|&idx| seen.insert(idx));
}

/// `str::starts_with`, first byte first — see [`str_ends_with`].
#[inline]
fn str_starts_with(s: &str, prefix: &str) -> bool {
    let (haystack, needle) = (s.as_bytes(), prefix.as_bytes());
    match (needle.first(), haystack.first()) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(n), Some(h)) => {
            n == h && haystack.len() >= needle.len() && &haystack[..needle.len()] == needle
        }
    }
}

/// The string test a matcher reduces to, or `None` when its answer needs more
/// than the field's string form.
///
/// These four matchers are exactly the ones
/// [`PatternExecutor::value_matches`] decides by looking at a `Value::String`
/// and nothing else — every other value shape answers `false` there, which is
/// what [`crate::graph::storage::StrField::is`] returns for
/// `NotString`/`Absent`. Keeping this function beside `value_matches` is the
/// whole safety argument: they must agree row for row.
///
/// `Equals` is [`str_values_equal`] — `values_equal`'s string arm,
/// JSON-single-element unwrapping included — because on the identity fields
/// this route replaces `value_matches`, which used `values_equal`. Stored
/// user properties are answered before this by the byte fast path in
/// [`PatternExecutor::prop_matches`], which calls the same function through
/// `str_prop_eq`, so the two routes cannot disagree.
pub(super) fn str_field_test(matcher: &PropertyMatcher) -> Option<impl Fn(&str) -> bool + '_> {
    if !matches!(
        matcher,
        PropertyMatcher::Equals(Value::String(_))
            | PropertyMatcher::StartsWith(_)
            | PropertyMatcher::EndsWith(_)
            | PropertyMatcher::Contains(_)
    ) {
        return None;
    }
    Some(move |s: &str| match matcher {
        PropertyMatcher::Equals(Value::String(target)) => str_values_equal(s, target),
        PropertyMatcher::StartsWith(prefix) => str_starts_with(s, prefix),
        PropertyMatcher::EndsWith(suffix) => str_ends_with(s, suffix),
        PropertyMatcher::Contains(needle) => s.contains(needle.as_str()),
        _ => unreachable!("guarded by the matches! above"),
    })
}

/// Whether `value` satisfies `matcher`, given the query's parameters.
///
/// Cross-type numeric comparison throughout (Int64 <-> UniqueId <-> Float64).
/// Free rather than a `PatternExecutor` method because the column-major scan
/// filter needs it and holds no executor — a scan carrying its own copy of
/// these comparisons is a scan that can disagree with the row route about what
/// a query means.
pub(super) fn value_matches(
    params: &HashMap<String, Value>,
    value: &Value,
    matcher: &PropertyMatcher,
) -> bool {
    // Cypher three-valued logic, as `WHERE` applies it in
    // `executor::helpers::evaluate_comparison`: a comparison with a NULL
    // operand is NULL, and a NULL row is filtered out. `values_equal` already
    // encodes that for equality, but the ordering matchers reach
    // `compare_values`, which sorts NULL *below* every value (its ORDER BY
    // duty) and so answered `x < 5` with `true` for a NULL `x`; `In` likewise
    // matched a NULL element in the set. The planner now drops a `WHERE` these
    // matchers provably enforce (`where_subsumed_by_pattern`), so the two
    // evaluators disagreeing here would be a wrong answer rather than a
    // redundant filter.
    if matches!(value, Value::Null) {
        return false;
    }
    match matcher {
        PropertyMatcher::Equals(expected) => values_equal(value, expected),
        PropertyMatcher::EqualsParam(name) => params
            .get(name.as_str())
            .is_some_and(|expected| values_equal(value, expected)),
        // EqualsVar / EqualsNodeProp should be resolved to Equals before
        // pattern matching. If they reach here unresolved, no match is possible.
        PropertyMatcher::EqualsVar(_) | PropertyMatcher::EqualsNodeProp { .. } => false,
        // One coercion-normalized probe against the set the planner built
        // with the pattern — not a scan of the list per candidate node.
        PropertyMatcher::In(values) => values.matches(value),
        PropertyMatcher::GreaterThan(threshold) => {
            compare_values(value, threshold) == Some(std::cmp::Ordering::Greater)
        }
        PropertyMatcher::GreaterOrEqual(threshold) => {
            matches!(
                compare_values(value, threshold),
                Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            )
        }
        PropertyMatcher::LessThan(threshold) => {
            compare_values(value, threshold) == Some(std::cmp::Ordering::Less)
        }
        PropertyMatcher::LessOrEqual(threshold) => {
            matches!(
                compare_values(value, threshold),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            )
        }
        PropertyMatcher::Range {
            lower,
            lower_inclusive,
            upper,
            upper_inclusive,
        } => {
            let above_lower = if *lower_inclusive {
                matches!(
                    compare_values(value, lower),
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                )
            } else {
                compare_values(value, lower) == Some(std::cmp::Ordering::Greater)
            };
            let below_upper = if *upper_inclusive {
                matches!(
                    compare_values(value, upper),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )
            } else {
                compare_values(value, upper) == Some(std::cmp::Ordering::Less)
            };
            above_lower && below_upper
        }
        PropertyMatcher::StartsWith(prefix) => match value {
            Value::String(s) => str_starts_with(s, prefix),
            _ => false,
        },
        PropertyMatcher::Contains(needle) => match value {
            Value::String(s) => s.contains(needle.as_str()),
            _ => false,
        },
        PropertyMatcher::EndsWith(suffix) => match value {
            Value::String(s) => str_ends_with(s, suffix),
            _ => false,
        },
    }
}

/// Executes graph pattern matching against a `DirGraph`.
///
/// Takes a parsed `Pattern` and finds all subgraph matches using
/// BFS expansion from type-indexed starting nodes. Supports variable
/// binding, property filters, edge direction, variable-length paths,
/// and optional pre-bound variables for Cypher integration.
pub struct PatternExecutor<'a> {
    graph: &'a DirGraph,
    max_matches: Option<usize>,
    pre_bindings: &'a Bindings<NodeIndex>,
    /// When true, node_to_binding() and edge bindings skip cloning
    /// properties/title/id (the Cypher executor only uses `index`).
    lightweight: bool,
    /// Query parameters for resolving $param references in inline properties
    params: &'a HashMap<String, Value>,
    deadline: Option<Instant>,
    /// Optional cooperative-cancellation flag, polled at the same
    /// checkpoints as `deadline` (one relaxed atomic load). Set by a
    /// binding's signal model (the Python wheel's SIGINT handler) so a
    /// long scan/expansion can be interrupted. `None` = never cancelled.
    cancel: Option<&'static AtomicBool>,
    /// When set, deduplicate results by NodeIndex of the named variable.
    /// At the last hop expansion, paths leading to already-seen target nodes
    /// are skipped, avoiding PatternMatch cloning and allocation overhead.
    distinct_target_var: Option<String>,
    /// Targets an *earlier* execution already emitted, when the caller is
    /// deduplicating one variable across a series of executions (the Cypher
    /// executor's subsequent-MATCH branch builds one executor per driving
    /// row). Consulted alongside this execution's own seen-set, and
    /// **never written to**: the caller inserts a target only once a match
    /// carrying it has actually become a row, so neither the capped/uncapped
    /// retry inside [`PatternExecutor::execute`] nor a match this executor's
    /// caller later discards can leave a target marked as emitted.
    distinct_prior: Option<&'a HashSet<NodeIndex>>,
    /// Opt-in parallel runtime for this execution (`ExecuteOptions::parallel`,
    /// threaded down like `cancel`). Default `false`. A permission, not an
    /// instruction: the candidate scan still applies its own runtime row ×
    /// cost-class gate before it fans out.
    parallel: bool,
    /// Set by [`PatternExecutor::note_cap_truncated`] whenever one of the
    /// *advisory* candidate caps under `max_matches` actually discarded
    /// candidates — see `matcher_expansion.rs`. Those caps are a selectivity
    /// heuristic, so a short result with this bit set is not evidence that the
    /// pattern has no more rows: `execute` re-runs the pattern once with the
    /// pre-caps off. `AtomicBool` rather than `Cell` because the parallel
    /// expansion path captures `&self` across rayon workers; a `Relaxed` store
    /// on the (rare) truncation path is not on the hot path.
    cap_truncated: AtomicBool,
    /// The absolute ceiling this execution's in-flight match buffers are held
    /// to, set by a caller that **retains** the matches — see
    /// [`MatchCeiling`] for the per-call-site classification. `None` (the
    /// default) leaves the expansion unbounded, which is correct for a caller
    /// that only counts or scans.
    ///
    /// Distinct from `max_matches`, and deliberately so: `max_matches` is a
    /// *limit* the matcher may satisfy by stopping early, and setting one
    /// changes the plan (lazy seeding, no parallel hop expansion). This is a
    /// *ceiling* the matcher may only satisfy by erroring, so it changes
    /// nothing about how the pattern is executed.
    match_ceiling: Option<MatchCeiling>,
    /// Holds the disk materialization arenas alive for this executor's
    /// lifetime (arena protocol in `storage/disk/graph.rs`, enforced by a
    /// debug assert). Acquired in every constructor so pattern matching is
    /// guard-covered no matter which surface spawned it (Cypher executor,
    /// fluent API, MERGE matching). `None` on memory/mapped backends —
    /// one enum match at construction on the in-memory hot path.
    _arena_guard: Option<crate::graph::storage::disk::graph::DiskQueryGuard>,
}

static EMPTY_PARAMS: std::sync::LazyLock<HashMap<String, Value>> =
    std::sync::LazyLock::new(HashMap::new);

static EMPTY_BINDINGS: std::sync::LazyLock<Bindings<NodeIndex>> =
    std::sync::LazyLock::new(Bindings::new);

impl<'a> PatternExecutor<'a> {
    pub fn new(graph: &'a DirGraph, max_matches: Option<usize>) -> Self {
        PatternExecutor {
            graph,
            max_matches,
            pre_bindings: &EMPTY_BINDINGS,
            lightweight: false,
            params: &EMPTY_PARAMS,
            deadline: None,
            cancel: None,
            distinct_target_var: None,
            distinct_prior: None,
            parallel: false,
            cap_truncated: AtomicBool::new(false),
            match_ceiling: None,
            _arena_guard: graph.graph.begin_query(),
        }
    }

    pub fn new_lightweight_with_params(
        graph: &'a DirGraph,
        max_matches: Option<usize>,
        params: &'a HashMap<String, Value>,
    ) -> Self {
        PatternExecutor {
            graph,
            max_matches,
            pre_bindings: &EMPTY_BINDINGS,
            lightweight: true,
            params,
            deadline: None,
            cancel: None,
            distinct_target_var: None,
            distinct_prior: None,
            parallel: false,
            cap_truncated: AtomicBool::new(false),
            match_ceiling: None,
            _arena_guard: graph.graph.begin_query(),
        }
    }

    pub fn with_bindings_and_params(
        graph: &'a DirGraph,
        max_matches: Option<usize>,
        pre_bindings: &'a Bindings<NodeIndex>,
        params: &'a HashMap<String, Value>,
    ) -> Self {
        PatternExecutor {
            graph,
            max_matches,
            pre_bindings,
            lightweight: true,
            params,
            deadline: None,
            cancel: None,
            distinct_target_var: None,
            distinct_prior: None,
            parallel: false,
            cap_truncated: AtomicBool::new(false),
            match_ceiling: None,
            _arena_guard: graph.graph.begin_query(),
        }
    }

    pub fn set_deadline(mut self, deadline: Option<Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    pub fn set_cancel(mut self, cancel: Option<&'static AtomicBool>) -> Self {
        self.cancel = cancel;
        self
    }

    /// Opt this execution in to the parallel runtime — a per-query property the
    /// Cypher executor threads down from `ExecuteOptions`, like `cancel`.
    pub fn set_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Combined deadline + cancellation poll; `Some(message)` aborts the run.
    /// The String is allocated only on the (rare) abort path.
    #[inline]
    fn interrupt_reason(&self) -> Option<String> {
        if let Some(dl) = self.deadline {
            if Instant::now() > dl {
                return Some("Query timed out".to_string());
            }
        }
        if let Some(c) = &self.cancel {
            if c.load(std::sync::atomic::Ordering::Relaxed) {
                return Some("Query cancelled".to_string());
            }
        }
        None
    }

    /// Record that an advisory candidate cap discarded candidates — see
    /// [`PatternExecutor::cap_truncated`].
    #[inline]
    fn note_cap_truncated(&self) {
        self.cap_truncated
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    #[inline]
    fn take_cap_truncated(&self) -> bool {
        self.cap_truncated
            .swap(false, std::sync::atomic::Ordering::Relaxed)
    }

    /// Deduplicate results by the named variable — see
    /// [`PatternExecutor::distinct_target_var`].
    pub fn set_distinct_target(mut self, var: Option<String>) -> Self {
        self.distinct_target_var = var;
        self
    }

    /// Seed the distinct-target dedup with targets an earlier execution
    /// already emitted — see [`PatternExecutor::distinct_prior`]. Only has an
    /// effect together with [`Self::set_distinct_target`].
    pub fn set_distinct_prior(mut self, prior: Option<&'a HashSet<NodeIndex>>) -> Self {
        self.distinct_prior = prior;
        self
    }

    /// Hold this execution's in-flight match buffers to an absolute ceiling.
    /// Set by callers that retain the matches; see [`MatchCeiling`].
    pub fn set_match_ceiling(mut self, ceiling: Option<MatchCeiling>) -> Self {
        self.match_ceiling = ceiling;
        self
    }

    /// Fail if `held` matches in one buffer would breach the ceiling.
    ///
    /// Called from the expansion loops, so the common case is one `Option`
    /// test and one comparison; the message is built behind `#[cold]`.
    #[inline]
    fn check_match_ceiling(&self, held: usize) -> Result<(), String> {
        match self.match_ceiling {
            Some(ceiling) => ceiling.check(held),
            None => Ok(()),
        }
    }

    /// Public wrapper for [`Self::find_matching_nodes`] — shortestPath
    /// endpoints and the fused node-scan operators.
    pub fn find_matching_nodes_pub(&self, pattern: &NodePattern) -> Result<Vec<NodeIndex>, String> {
        self.find_matching_nodes(pattern)
    }

    fn find_matching_nodes(&self, pattern: &NodePattern) -> Result<Vec<NodeIndex>, String> {
        let extra_keys: Vec<InternedKey> = pattern
            .extra_labels
            .iter()
            .map(|label| InternedKey::from_str(label))
            .collect();

        if let Some(ref var) = pattern.variable {
            if let Some(&idx) = self.pre_bindings.get(var) {
                if let Some(node) = self.graph.graph.node_view(idx) {
                    if pattern.node_type.is_some() {
                        let labels = self.graph.node_labels(idx);
                        // Alternation: ANY branch may carry the binding;
                        // plain/AND-chain: the type plus every extra must.
                        let alts = pattern.label_alternatives();
                        if !alts
                            .iter()
                            .any(|l| labels.contains(&InternedKey::from_str(l)))
                        {
                            return Ok(vec![]);
                        }
                        for extra in &pattern.extra_labels {
                            let key = InternedKey::from_str(extra);
                            if !labels.contains(&key) {
                                return Ok(vec![]);
                            }
                        }
                        // The view is only an existence check — the label tests
                        // read `labels`, so nothing here consumes `node`.
                        let _ = node;
                    }
                    if let Some(ref props) = pattern.properties {
                        if !self.node_matches_properties(idx, props) {
                            return Ok(vec![]);
                        }
                    }
                    return Ok(vec![idx]);
                }
                return Ok(vec![]);
            }
        }

        if pattern.properties.as_ref().is_some_and(|properties| {
            properties
                .values()
                .any(|matcher| matches!(matcher, PropertyMatcher::In(values) if values.is_empty()))
        }) {
            return Ok(Vec::new());
        }

        // `:A|B|C` alternation. Index-served first where every branch can
        // answer the pattern's equality predicates; otherwise the union of
        // every branch's carriers, scanned.
        if let Some(alts) = &pattern.alt_labels {
            if let Some(props) = pattern.properties.as_ref() {
                if let Some(hit) = self.try_alternation_probe(alts, props, &extra_keys) {
                    return hit;
                }
            }
            return self.scan_label_union(alts, pattern.properties.as_ref(), &extra_keys);
        }

        if let Some(ref node_type) = pattern.node_type {
            let secondary = if self.graph.has_secondary_labels {
                self.graph
                    .secondary_label_index
                    .get(&InternedKey::from_str(node_type))
                    .filter(|bucket| !bucket.is_empty())
            } else {
                None
            };

            // Only this label's secondary carriers affect primary-index completeness.
            // Union their filtered scan with indexed hits instead of scanning every primary.
            if let Some(ref props) = pattern.properties {
                if let Some(hit) = self.try_closure_probe(node_type, props, &extra_keys) {
                    return hit;
                }
                if let Some(indexed) = self
                    .try_index_lookup(node_type, props)
                    .or_else(|| self.try_global_index_lookup_typed(node_type, props))
                {
                    let mut out = self.filter_node_candidates(&indexed, None, &extra_keys)?;
                    if let Some(secondary) = secondary {
                        out.extend(self.filter_node_candidates(
                            secondary.as_slice(),
                            Some(props),
                            &extra_keys,
                        )?);
                    }
                    return Ok(out);
                }
            }

            // The choke-point API forbids primary==secondary on the same node,
            // so this type_indices ∪ secondary_label_index has no duplicates.
            let mut candidates = self
                .graph
                .type_indices
                .get(node_type)
                .map(|indices| indices.to_vec())
                .unwrap_or_default();
            if let Some(secondary) = secondary {
                candidates.extend(secondary.iter().copied());
            }
            if candidates.is_empty() {
                return Ok(Vec::new());
            }
            if pattern.properties.is_none() && extra_keys.is_empty() {
                return Ok(candidates);
            }
            self.filter_node_candidates(&candidates, pattern.properties.as_ref(), &extra_keys)
        } else if let Some(ref props) = pattern.properties {
            // Fast path: untyped node with {id: X} — cross-type id lookup, one
            // O(1) id-index probe per type, so O(types): fast even at 132K types.
            //
            // Only `{id: N}` routes to the id-index here; `{nid: 'Q76'}` is a
            // plain string property (0.11.0) served by the cross-type
            // global-property-index path below in O(log N).
            // Params resolve here exactly as the typed path does
            // (try_index_lookup's EqualsParam arm) — pre-fix `{id: $x}` fell
            // past this anchor into the full scan, so the literal and the
            // parameter spelling answered DIFFERENT rows on graphs with
            // duplicate ids (measured 2026-08-15: 1 vs 68).
            let id_val_opt = ["id"].iter().find_map(|k| match props.get(*k) {
                Some(PropertyMatcher::Equals(v)) => Some(v),
                Some(PropertyMatcher::EqualsParam(name)) => self.params.get(name.as_str()),
                _ => None,
            });
            if let Some(id_val) = id_val_opt {
                // Union over every type's id index — one node per (type, id),
                // the semantics the duplicate-id warning documents. Pre-fix
                // this returned on the FIRST type with a hit, collapsing
                // cross-type id collisions to one arbitrary node (HashMap key
                // order — nondeterministic across processes).
                let mut hits: Vec<petgraph::graph::NodeIndex> = Vec::new();
                for node_type in self.graph.type_indices.keys() {
                    if let Some(idx) = self.graph.lookup_by_id_readonly(node_type, id_val) {
                        if props.len() == 1 || self.node_matches_properties(idx, props) {
                            hits.push(idx);
                        }
                    }
                }
                // Deterministic row order regardless of type-map iteration.
                hits.sort_unstable();
                return Ok(hits);
            }
            // Cross-type fast paths: for any Equals(String) or
            // StartsWith(String), consult the persistent global index
            // if one exists for that property. Turns `MATCH (n {label:
            // 'Norway'})` into O(log N) without requiring a type label.
            // Alias-aware via `global_alias_candidates`, so an index built as
            // `create_global_index('label')` still serves `{title: 'X'}`.
            for (prop, matcher) in props {
                let alias_candidates = global_alias_candidates(prop, self.graph);
                match matcher {
                    PropertyMatcher::Equals(Value::String(s)) => {
                        for idx_name in &alias_candidates {
                            if let Some(candidates) = string_index_hits(s, |key| {
                                self.graph
                                    .graph
                                    .lookup_by_property_eq_any_type(idx_name, key)
                            }) {
                                if props.len() == 1 {
                                    return Ok(candidates);
                                }
                                let filtered = candidates
                                    .into_iter()
                                    .filter(|&idx| self.node_matches_properties(idx, props))
                                    .collect();
                                return Ok(filtered);
                            }
                        }
                    }
                    PropertyMatcher::StartsWith(prefix) => {
                        for idx_name in &alias_candidates {
                            if let Some(candidates) = self
                                .graph
                                .graph
                                .lookup_by_property_prefix_any_type(idx_name, prefix, usize::MAX)
                            {
                                if props.len() == 1 {
                                    return Ok(candidates);
                                }
                                let filtered = candidates
                                    .into_iter()
                                    .filter(|&idx| self.node_matches_properties(idx, props))
                                    .collect();
                                return Ok(filtered);
                            }
                        }
                    }
                    _ => {}
                }
            }
            // No id property, no global index — scan all nodes with property filter.
            let g = &self.graph.graph;
            let mut out = Vec::new();
            for (i, idx) in g.node_indices().enumerate() {
                if i & 0xFFF == 0 {
                    self.check_scan_deadline()?;
                }
                if self.node_matches_properties(idx, props) {
                    out.push(idx);
                }
            }
            Ok(out)
        } else {
            let g = &self.graph.graph;
            let mut out = Vec::with_capacity(g.node_count());
            for (i, idx) in g.node_indices().enumerate() {
                if i & 0xFFF == 0 {
                    self.check_scan_deadline()?;
                }
                out.push(idx);
            }
            Ok(out)
        }
    }

    /// Apply label intersections and any properties not already covered by an
    /// index to a candidate stream. The shared loop keeps deadline/cancellation
    /// checks identical for primary scans and secondary-label fallbacks.
    ///
    /// This is the one place a *stream* of candidates is property-filtered, so
    /// it is where per-type resolution is hoisted out of the per-node work —
    /// see [`TypeScanMemo`].
    fn filter_node_candidates<'p>(
        &'p self,
        candidates: &[NodeIndex],
        props: Option<&'p HashMap<String, PropertyMatcher>>,
        extra_keys: &[InternedKey],
    ) -> Result<Vec<NodeIndex>, String> {
        if self.may_fan_out_candidate_scan(candidates, props) {
            return self.filter_candidates_parallel(candidates, props, extra_keys);
        }
        let interrupt = ParallelInterrupt::new(|| self.check_scan_deadline().err());
        self.filter_candidate_partition(candidates, props, extra_keys, &interrupt)
    }

    /// Whether the candidate scan may fan out.
    ///
    /// **Write-freedom is provable here rather than argued.** Unlike the
    /// Cypher scan operators, this loop evaluates `PropertyMatcher`s, never
    /// Cypher expressions, so it cannot re-enter the interpreter: every call it
    /// makes — `node_has_label`, `interner::try_resolve`, `column_store`,
    /// `resolve_alias`, `ColumnFilter::compile`, `node_view`, `value_matches` —
    /// is a plain read on the memory and mapped backends. There is nothing to
    /// pre-warm and no spatial exclusion to make: the per-node spatial cache
    /// belongs to the Cypher executor and is unreachable from here.
    ///
    /// **Disk stays excluded** even though the arena hazard the rest of the
    /// engine has on disk is genuinely bypassed here (`owned_node_data`
    /// materialises into the caller's frame rather than parking a record in the
    /// shared query arena): disk mode is deferred wholesale, and an exclusion
    /// uniform with the Cypher scan operators is one rule to state to users —
    /// disk ignores `parallel`.
    fn may_fan_out_candidate_scan(
        &self,
        candidates: &[NodeIndex],
        props: Option<&HashMap<String, PropertyMatcher>>,
    ) -> bool {
        if !self.parallel || self.graph.graph.is_disk() {
            return false;
        }
        // The column-filter test overrides are thread-local *controls*: a
        // worker would not see them, so a forced row route would silently stop
        // being forced and the differential sweep would compare the compiled
        // filter with itself. Refuse to fan out while one is set — that closes
        // the hole by construction rather than by convention.
        if column_filter::scan_overrides_active() {
            return false;
        }
        parallel::should_fan_out(
            candidates.len(),
            self.candidate_scan_cost(candidates, props),
        )
    }

    /// Which side of the runtime gate this scan's per-candidate work sits on.
    ///
    /// The memo's own column-vs-row split *is* the cost class: when
    /// `ColumnFilter::compile` accepts every matcher the test is a typed column
    /// read and a compare (tens of ns), and when it declines the candidate goes
    /// through `node_matches_resolved` — a `NodeView`, a property fetch and a
    /// `Value` comparison per matcher, which is where the ~100× spread lives.
    /// Probing the first candidate's type is enough: a mixed stream rebuilds
    /// the memo per type, but the overwhelming majority of a scan is one type,
    /// and mis-classifying a mixed stream only moves a threshold.
    fn candidate_scan_cost(
        &self,
        candidates: &[NodeIndex],
        props: Option<&HashMap<String, PropertyMatcher>>,
    ) -> parallel::CostClass {
        let Some(props) = props else {
            // No property matchers at all — the loop is a label check per
            // candidate, the cheapest shape there is.
            return parallel::CostClass::Compiled;
        };
        let compiled = candidates
            .first()
            .and_then(|&idx| self.graph.graph.node_weight(idx))
            .and_then(|data| self.build_type_scan_memo(data.node_type, props))
            .is_some_and(|memo| memo.filter.is_some());
        if compiled {
            parallel::CostClass::Compiled
        } else {
            parallel::CostClass::Interpreted
        }
    }

    /// Fan the candidate scan across the query pool.
    ///
    /// Order-preserving by construction: `par_chunks` partitions the
    /// candidate vector by index range, `collect` on an indexed parallel
    /// iterator restores partition order, and the partitions are concatenated
    /// in that order — so the surviving candidates come back in exactly the
    /// order the sequential scan would have produced. That matters more here
    /// than almost anywhere else in the engine: bucket order of an
    /// un-`ORDER BY`'d MATCH is a documented, test-gated invariant.
    fn filter_candidates_parallel<'p>(
        &'p self,
        candidates: &[NodeIndex],
        props: Option<&'p HashMap<String, PropertyMatcher>>,
        extra_keys: &[InternedKey],
    ) -> Result<Vec<NodeIndex>, String> {
        #[cfg(test)]
        parallel::PARALLEL_CANDIDATE_SCANS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let interrupt = ParallelInterrupt::new(|| self.check_scan_deadline().err());
        let partitions = (rayon::current_num_threads() * CANDIDATE_PARTITIONS_PER_WORKER).max(1);
        let chunk_len = candidates.len().div_ceil(partitions).max(1);
        let parts: Vec<(Vec<NodeIndex>, usize)> = parallel::install(|| {
            candidates
                .par_chunks(chunk_len)
                .map(|chunk| {
                    // Difference the compiled-filter meter around this
                    // partition so the rows a worker answered for are folded
                    // back into the measuring thread below. Compiles to
                    // nothing outside `cfg(test)`.
                    let before = column_filter::local_rows_filtered();
                    let kept =
                        self.filter_candidate_partition(chunk, props, extra_keys, &interrupt)?;
                    Ok((kept, column_filter::local_rows_filtered() - before))
                })
                .collect::<Result<Vec<_>, String>>()
        })?;
        let mut out = Vec::with_capacity(parts.iter().map(|(part, _)| part.len()).sum());
        let mut filtered = 0usize;
        for (part, rows) in parts {
            filtered += rows;
            out.extend(part);
        }
        column_filter::add_rows_filtered(filtered);
        Ok(out)
    }

    /// Filter one contiguous range of candidates. Owns its [`TypeScanMemo`] —
    /// the memo is per-node-type mutable state, so a partition cannot share
    /// one.
    fn filter_candidate_partition<'p, F>(
        &'p self,
        candidates: &[NodeIndex],
        props: Option<&'p HashMap<String, PropertyMatcher>>,
        extra_keys: &[InternedKey],
        interrupt: &ParallelInterrupt<F>,
    ) -> Result<Vec<NodeIndex>, String>
    where
        F: Fn() -> Option<String> + Sync,
    {
        let mut out = Vec::new();
        let mut memo: Option<TypeScanMemo<'p>> = None;
        // Resolved once: only the disk backend materialises into an arena.
        let scoped_materialization = self.graph.graph.is_disk();
        for (i, &idx) in candidates.iter().enumerate() {
            interrupt.check(i)?;
            if !extra_keys.is_empty()
                && !extra_keys
                    .iter()
                    .all(|&key| self.graph.node_has_label(idx, key))
            {
                continue;
            }
            let Some(properties) = props else {
                out.push(idx);
                continue;
            };
            // On disk, materialize into this frame: `node_weight` would park a
            // record in the query arena for every node the scan walks, and the
            // scan drops each one immediately (storage/disk/query_arena.rs).
            // Heap backends borrow straight out of the graph. Both branches
            // then run the same body below, over a plain `&NodeData` — no
            // closure, which would force `memo` out of registers for the whole
            // loop and cost the heap path ~8% on a 50k-node filtered scan.
            let owned;
            let data = if scoped_materialization {
                owned = self.graph.graph.owned_node_data(idx);
                owned.as_ref()
            } else {
                self.graph.graph.node_weight(idx)
            };
            let Some(data) = data else {
                continue;
            };
            if memo
                .as_ref()
                .is_none_or(|memo| memo.type_key != data.node_type)
            {
                memo = self.build_type_scan_memo(data.node_type, properties);
            }
            let Some(memo) = memo.as_ref() else {
                continue;
            };
            // Column-major where the type's store can answer every matcher on
            // its own (`ColumnFilter`), row-major otherwise — and row-major for
            // the individual node a compiled filter hands back, which is how a
            // node carrying an inline identity value stays correct in a graph
            // whose other nodes are columnar.
            let matched = memo
                .filter
                .as_ref()
                .and_then(|filter| {
                    let row = data.properties.columnar_row_id()?;
                    filter.matches(data, row, self.params)
                })
                .unwrap_or_else(|| {
                    self.node_matches_resolved(self.node_view_of(data, memo.store), memo)
                });
            if matched {
                out.push(idx);
            }
        }
        Ok(out)
    }

    /// Deadline check used by all full-type / unanchored scans in this
    /// file. Poll every 4096 nodes — amortised overhead is negligible
    /// (≤ 1 `Instant::now()` per ~4K pattern comparisons) while keeping
    /// the worst-case response time under a few milliseconds past the
    /// deadline.
    #[inline]
    fn check_scan_deadline(&self) -> Result<(), String> {
        if let Some(dl) = self.deadline {
            if Instant::now() > dl {
                return Err("Query timed out during node scan. Hint: add an index on a \
                     predicate property (create_index), anchor with \
                     MATCH (n {id: ...}), or raise timeout_ms."
                    .to_string());
            }
        }
        if let Some(c) = &self.cancel {
            if c.load(std::sync::atomic::Ordering::Relaxed) {
                return Err("Query cancelled".to_string());
            }
        }
        Ok(())
    }

    /// Cross-type global-index fast path for **typed** patterns.
    ///
    /// The untyped branch above already consults the global index. On
    /// Wikidata-scale disk graphs the common shape is typed — `MATCH
    /// (n:Human {title: 'Barack Obama'})` — and there's no per-type
    /// index built for that 13M-row type. Without this fast path the
    /// executor falls through to a full-type scan (10–14s, usually a
    /// timeout).
    ///
    /// So: consult the cross-type global index (built once at save-time,
    /// covering every node type), then filter its hits by `node_type_of(idx)` —
    /// O(hits), microseconds, for a query hitting a handful of rows. Alias-aware
    /// via `global_alias_candidates`, so an index built as `global_index_label_*`
    /// still serves `{title: 'X'}`.
    ///
    /// `None` = no global index covered any pushable predicate in `props`; the
    /// caller falls through to the type-scan path.
    /// Every branch's carriers, unioned and property-filtered — the
    /// alternation path that is always correct and never indexed.
    ///
    /// A node can sit in two branches at once (primary `:A`, secondary `:B`),
    /// so the dedup is mandatory; sorting also keeps the candidate order
    /// stable across runs.
    fn scan_label_union(
        &self,
        alts: &[String],
        props: Option<&HashMap<String, PropertyMatcher>>,
        extra_keys: &[InternedKey],
    ) -> Result<Vec<NodeIndex>, String> {
        let mut candidates: Vec<NodeIndex> = Vec::new();
        for label in alts {
            candidates.extend(self.graph.nodes_with_label(label));
        }
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        if props.is_none() && extra_keys.is_empty() {
            return Ok(candidates);
        }
        self.filter_node_candidates(&candidates, props, extra_keys)
    }

    /// Per-branch index probes for `MATCH (n:A|B {p: v})` — the alternation
    /// mirror of the single-label indexed path in `find_matching_nodes`.
    ///
    /// **Precondition, all-or-nothing:** every branch label must answer a
    /// point lookup of every *bound* equality property, from an index or from
    /// being that label's id field
    /// ([`closure_probe::type_covers_property`], the same rule the closure
    /// probe applies to its members). One uncovered branch declines the whole
    /// probe: a partial union would drop that branch's rows silently, and a
    /// scan is only ever allowed to be traded for a *complete* answer.
    ///
    /// **Rewrite:** per branch, the index answer over its primary bucket plus
    /// a filtered scan of that label's secondary carriers. Carriers hold the
    /// label without being of that primary type, so no per-type index sees
    /// them and they must still be walked. `try_index_lookup`'s unified miss
    /// contract is what makes the union possible at all — a covered
    /// value-miss contributes nothing instead of declining the probe.
    ///
    /// **The dedup is soundness, not tidiness.** Sibling label overlap is
    /// legal (primary `:A` + secondary `:B`), so branch A's index hit and
    /// branch B's carrier scan can name the same node, and a `MATCH` binds
    /// each node once.
    ///
    /// **Why-bail:** `None` — caller keeps the scan — when the pattern binds
    /// no equality predicate (nothing to probe with) or coverage is partial.
    fn try_alternation_probe(
        &self,
        alts: &[String],
        props: &HashMap<String, PropertyMatcher>,
        extra_keys: &[InternedKey],
    ) -> Option<Result<Vec<NodeIndex>, String>> {
        let equality_props = self.bound_equality_prop_names(props);
        if equality_props.is_empty() {
            return None;
        }
        let covered = alts.iter().all(|branch| {
            equality_props
                .iter()
                .all(|prop| closure_probe::type_covers_property(self.graph, branch, prop))
        });
        covered.then(|| self.alternation_probe_candidates(alts, props, extra_keys))
    }

    /// The probe body for [`Self::try_alternation_probe`]; see its doc for the
    /// precondition every branch has already satisfied.
    fn alternation_probe_candidates(
        &self,
        alts: &[String],
        props: &HashMap<String, PropertyMatcher>,
        extra_keys: &[InternedKey],
    ) -> Result<Vec<NodeIndex>, String> {
        let mut out: Vec<NodeIndex> = Vec::new();
        for branch in alts {
            let Some(hits) = self.try_index_lookup(branch, props) else {
                // Declaration coverage does not prove a query value's domain.
                // Decline the whole union if any branch cannot answer it.
                return self.scan_label_union(alts, Some(props), extra_keys);
            };
            out.extend(hits);
            if self.graph.has_secondary_labels {
                if let Some(carriers) = self
                    .graph
                    .secondary_label_index
                    .get(&InternedKey::from_str(branch))
                {
                    out.extend(self.filter_node_candidates(
                        carriers.as_slice(),
                        Some(props),
                        &[],
                    )?);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        if extra_keys.is_empty() {
            return Ok(out);
        }
        self.filter_node_candidates(&out, None, extra_keys)
    }

    /// The pattern's equality-predicate property names whose value is known
    /// here: a literal, or a parameter this execution actually bound. Shared
    /// by the two index probes so they agree on what "the pattern probes
    /// with" means — and so neither treats an unbound `$p` as probeable.
    fn bound_equality_prop_names<'p>(
        &self,
        props: &'p HashMap<String, PropertyMatcher>,
    ) -> Vec<&'p str> {
        props
            .iter()
            .filter(|(_, matcher)| match matcher {
                PropertyMatcher::Equals(_) => true,
                PropertyMatcher::EqualsParam(name) => self.params.contains_key(name.as_str()),
                _ => false,
            })
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// The closure-aware probe: for `MATCH (n:Ancestor {p: v})` where
    /// `Ancestor` is a **Closed** managed label, run the property-index
    /// lookup per declared member type, union, and finish with the
    /// extra-label filter — the complete answer, so the caller returns it
    /// without the secondary-bucket union (the probes subsume the bucket:
    /// Closed means the engine is its only writer). `None` — caller falls
    /// back to the correct scan — whenever
    /// [`closure_probe::closure_probe_members`] finds the shape ineligible:
    /// an Open label may hold carriers no descendant probe covers, and a
    /// member that cannot answer the lookup from an index would contribute a
    /// silent row loss rather than a slow scan.
    fn try_closure_probe(
        &self,
        node_type: &str,
        props: &HashMap<String, PropertyMatcher>,
        extra_keys: &[InternedKey],
    ) -> Option<Result<Vec<NodeIndex>, String>> {
        let unioned = self.try_closure_index_lookup(node_type, props)?;
        Some(self.filter_node_candidates(&unioned, None, extra_keys))
    }

    /// Eligibility comes from the shared predicate, which the EXPLAIN
    /// renderer reads too. Runtime additionally proves the query value's
    /// key domain and may decline an unsupported value. The property names handed
    /// to it are the pattern's equality keys, matching what
    /// `closure_probe_ops` reads off the AST, plus the parameterised
    /// equalities whose value is bound here (EXPLAIN cannot see those, so it
    /// stays conservatively unmarked for them).
    ///
    /// Each eligible member then contributes its own hits, **a value-miss
    /// contributing none**. That is the whole fix: a unique value lives in at
    /// most one member's index, so before `lookup_by_index` learned to say
    /// "proven empty" the `?` below declined on the first member that did not
    /// hold it — structurally, for every closure with two live members.
    fn try_closure_index_lookup(
        &self,
        node_type: &str,
        props: &HashMap<String, PropertyMatcher>,
    ) -> Option<Vec<NodeIndex>> {
        let equality_props = self.bound_equality_prop_names(props);
        let members = closure_probe::closure_probe_members(self.graph, node_type, &equality_props)?;
        let mut out = Vec::new();
        for member in members {
            // Per-primary-type buckets are disjoint. Coverage is structural;
            // unsupported query values still decline the entire union here.
            out.extend(self.try_index_lookup(&member, props)?);
        }
        Some(out)
    }

    fn try_global_index_lookup_typed(
        &self,
        node_type: &str,
        props: &HashMap<String, PropertyMatcher>,
    ) -> Option<Vec<NodeIndex>> {
        let expected = InternedKey::from_str(node_type);
        for (prop, matcher) in props {
            let aliases = global_alias_candidates(prop, self.graph);
            match matcher {
                PropertyMatcher::Equals(Value::String(s)) => {
                    for alias in &aliases {
                        if let Some(candidates) = string_index_hits(s, |key| {
                            self.graph.graph.lookup_by_property_eq_any_type(alias, key)
                        }) {
                            let filtered: Vec<NodeIndex> = candidates
                                .into_iter()
                                .filter(|&idx| self.graph.graph.node_type_of(idx) == Some(expected))
                                .filter(|&idx| {
                                    props.len() == 1 || self.node_matches_properties(idx, props)
                                })
                                .collect();
                            return Some(filtered);
                        }
                    }
                }
                PropertyMatcher::StartsWith(prefix) => {
                    for alias in &aliases {
                        if let Some(candidates) = self
                            .graph
                            .graph
                            .lookup_by_property_prefix_any_type(alias, prefix, usize::MAX)
                        {
                            let filtered: Vec<NodeIndex> = candidates
                                .into_iter()
                                .filter(|&idx| self.graph.graph.node_type_of(idx) == Some(expected))
                                .filter(|&idx| {
                                    props.len() == 1 || self.node_matches_properties(idx, props)
                                })
                                .collect();
                            return Some(filtered);
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Index-served `IN`-list anchors: `{id: IN [...]}` and `{p: IN [...]}`
    /// where `p` carries a per-type property index.
    ///
    /// Returns `Some(candidates)` when an index answered (the same
    /// proven-empty contract as [`Self::try_index_lookup`], whose IN arms
    /// these are), `None` when no index covers any `IN` in `props` and the
    /// caller should keep trying its other anchors.
    ///
    /// **The candidate set is deduplicated, keeping first occurrence.** These
    /// anchors are driven by the *list*, one index probe per element, so a
    /// list that names the same node twice — literal duplicates
    /// (`WHERE n.id IN [1, 1, 2]`), coercion-equal spellings (`[1, 1.0]`), or
    /// two values of an indexed property held by one node — used to emit that
    /// node once per element. A `MATCH` binds each node once: `count(n)` over
    /// `[1, 1, 2]` answered 3 where the scan path, every other anchor, and
    /// Neo4j answer 2. Dedup here rather than in the list because equal
    /// *values* are not the only way two elements land on one node, and
    /// because this is upstream of every `max_matches` cap and of both
    /// `CapPass` passes — a duplicate must not consume a row of the cap.
    ///
    /// First-occurrence order preserves the list order these anchors have
    /// always returned (`IN [3, 1, 2]` → nodes 3, 1, 2).
    fn try_in_list_lookup(
        &self,
        node_type: &str,
        props: &HashMap<String, PropertyMatcher>,
    ) -> Option<Vec<NodeIndex>> {
        // Fast path: IN on id field — O(k) lookups via id index
        if let Some(PropertyMatcher::In(values)) = props.get("id") {
            let mut result = Vec::with_capacity(values.len());
            for val in values {
                if let Some(idx) = self.graph.lookup_by_id_readonly(node_type, val) {
                    result.push(idx);
                }
            }
            dedup_candidates(&mut result);
            if props.len() > 1 {
                result.retain(|&idx| self.node_matches_properties(idx, props));
            }
            return Some(result);
        }

        // Fast path: IN on any indexed property — O(k) lookups via property index
        for (prop_name, matcher) in props {
            if let PropertyMatcher::In(values) = matcher {
                if prop_name == "id" {
                    continue; // handled above
                }
                if !self.graph.index_answers_point_lookup(node_type, prop_name) {
                    continue;
                }
                let mut result = Vec::with_capacity(values.len());
                for val in values {
                    result.extend(self.graph.lookup_by_index(node_type, prop_name, val)?);
                }
                dedup_candidates(&mut result);
                if props.len() > 1 {
                    result.retain(|&idx| self.node_matches_properties(idx, props));
                }
                return Some(result);
            }
        }

        None
    }

    /// Try to use property indexes for faster node lookup.
    ///
    /// `Some(v)` and `None` are not interchangeable: `None` sends
    /// [`Self::find_matching_nodes`] into a scan of every node of the type,
    /// `Some(v)` is taken verbatim (unioned with a filtered scan of the
    /// queried label's secondary-label carriers, which no index covers). So
    /// `Some(vec![])` is how an anchor says *proven empty, do not scan*.
    ///
    /// **A key miss on the id anchors is proven empty, not an unbuilt index.**
    /// Both id anchors — `{id: IN [...]}` and the `{id: v}` / `{<alias>: v}`
    /// equality below — read the per-type id index through
    /// `DirGraph::lookup_by_id_readonly`, which self-heals: it builds and
    /// caches the type's index on a miss (`IdIndexStore::lookup_or_build`,
    /// issue #20). By the time it answers `None` the index therefore exists
    /// and is authoritative over exactly the candidate set a scan would walk
    /// (`type_indices`), so falling through could only ever re-derive the same
    /// empty answer — at O(V) per absent key: 0.39 ms at 50k nodes and 1.56 ms
    /// at 200k against ~2.5 µs for a hit, and 6.7 s for an UNWIND over 16k
    /// absent ids. Empty also holds when the pattern carries further
    /// predicates: a conjunction with one false conjunct is empty.
    ///
    /// This trust is what obliges `TypeIdIndex::get` to coerce over the same
    /// numeric family as `values_equal` — a coercion the index declines but a
    /// scan would have accepted is now a lost row, not a slow one.
    ///
    /// **A value-miss on a covered property index is proven empty too**, for
    /// the same reason and under the same obligation: `lookup_by_index` answers
    /// only for `(node_type, property)` pairs whose index holds the value-space
    /// a scan reads, and index completeness is a maintained invariant. Before
    /// that, `MATCH (n:Student {email: 'absent'})` re-derived its empty answer
    /// by walking the whole type.
    fn try_index_lookup(
        &self,
        node_type: &str,
        props: &HashMap<String, PropertyMatcher>,
    ) -> Option<Vec<NodeIndex>> {
        // A known-empty IN list is an immediate empty candidate set even when
        // the property has no index. Avoid falling through to a full type scan.
        if props
            .values()
            .any(|matcher| matches!(matcher, PropertyMatcher::In(values) if values.is_empty()))
        {
            return Some(Vec::new());
        }

        if let Some(result) = self.try_in_list_lookup(node_type, props) {
            return Some(result);
        }

        let mut equality_props: Vec<(&String, &Value)> = props
            .iter()
            .filter_map(|(k, v)| match v {
                PropertyMatcher::Equals(val) => Some((k, val)),
                PropertyMatcher::EqualsParam(name) => {
                    self.params.get(name.as_str()).map(|val| (k, val))
                }
                // EqualsVar / In / comparisons are handled separately
                _ => None,
            })
            .collect();

        let has_comparison = props.values().any(|m| {
            matches!(
                m,
                PropertyMatcher::GreaterThan(_)
                    | PropertyMatcher::GreaterOrEqual(_)
                    | PropertyMatcher::LessThan(_)
                    | PropertyMatcher::LessOrEqual(_)
                    | PropertyMatcher::Range { .. }
            )
        });

        let has_prefix = props
            .values()
            .any(|matcher| matches!(matcher, PropertyMatcher::StartsWith(_)));
        if equality_props.is_empty() && !has_comparison && !has_prefix {
            return None;
        }

        // Try ID index for {id: value} patterns — O(1) lookup. Only the
        // canonical `id` and the user-declared per-type ID alias route here
        // (`add_nodes(df, "Star", "starId", "title")` makes `starId` the id
        // alias for :Star); `nid`/`qid` are plain string properties (0.11.0).
        if equality_props.len() == 1 {
            let (prop_name, value) = equality_props[0];
            let is_id_alias = prop_name.as_str() == "id"
                || self
                    .graph
                    .id_field_aliases
                    .get(node_type)
                    .map(|alias| alias == prop_name.as_str())
                    .unwrap_or(false);
            if is_id_alias {
                if let Some(idx) = self.graph.lookup_by_id_readonly(node_type, value) {
                    return Some(vec![idx]);
                }
                return Some(Vec::new()); // key miss, not a missing index
            }
        }

        if equality_props.len() >= 2 {
            // Composite keys are stored under their property names sorted
            // (`create_composite_index`), so probing in the pattern's own
            // order would miss. Sort in-place — equality_props is a local vec
            // of references, cheap to reorder.
            equality_props.sort_by(|a, b| a.0.cmp(b.0));
            let names: Vec<String> = equality_props.iter().map(|(k, _)| (*k).clone()).collect();
            let values: Vec<Value> = equality_props.iter().map(|(_, v)| (*v).clone()).collect();
            if let Some(results) = self
                .graph
                .lookup_by_composite_predicate(node_type, &names, &values)
            {
                if equality_props.len() == props.len() {
                    return Some(results);
                }
                let filtered = results
                    .into_iter()
                    .filter(|&idx| self.node_matches_properties(idx, props))
                    .collect();
                return Some(filtered);
            }
        }

        for (prop, value) in &equality_props {
            if let Some(results) = self.graph.lookup_by_index(node_type, prop, value) {
                if equality_props.len() == 1 && props.len() == 1 {
                    return Some(results);
                } else {
                    let filtered = results
                        .into_iter()
                        .filter(|&idx| self.node_matches_properties(idx, props))
                        .collect();
                    return Some(filtered);
                }
            }
        }

        // Persistent disk-backed property index (string equality).
        // `lookup_by_property_eq` returns `Some(Vec)` only when a
        // persistent index for `(node_type, prop)` exists; otherwise
        // `None` so we fall through to scan. Only Value::String values
        // are indexable today.
        for (prop, value) in &equality_props {
            if let Value::String(s) = value {
                if let Some(results) = string_index_hits(s, |key| {
                    self.graph.graph.lookup_by_property_eq(node_type, prop, key)
                }) {
                    if equality_props.len() == 1 && props.len() == 1 {
                        return Some(results);
                    }
                    let filtered = results
                        .into_iter()
                        .filter(|&idx| self.node_matches_properties(idx, props))
                        .collect();
                    return Some(filtered);
                }
            }
        }

        // Persistent disk-backed prefix index (STARTS WITH). Same `None` /
        // `Some` semantics as the equality path. Uses `usize::MAX` as the cap;
        // outer LIMIT pushdown is not wired into matcher state yet.
        for (prop, matcher) in props {
            if let PropertyMatcher::StartsWith(prefix) = matcher {
                if let Some(results) =
                    self.graph
                        .graph
                        .lookup_by_property_prefix(node_type, prop, prefix, usize::MAX)
                {
                    if props.len() == 1 {
                        return Some(results);
                    }
                    let filtered = results
                        .into_iter()
                        .filter(|&idx| self.node_matches_properties(idx, props))
                        .collect();
                    return Some(filtered);
                }
            }
        }

        for (prop, matcher) in props {
            use std::ops::Bound;
            let bounds: Option<(Bound<&Value>, Bound<&Value>)> = match matcher {
                PropertyMatcher::GreaterThan(v) => Some((Bound::Excluded(v), Bound::Unbounded)),
                PropertyMatcher::GreaterOrEqual(v) => Some((Bound::Included(v), Bound::Unbounded)),
                PropertyMatcher::LessThan(v) => Some((Bound::Unbounded, Bound::Excluded(v))),
                PropertyMatcher::LessOrEqual(v) => Some((Bound::Unbounded, Bound::Included(v))),
                PropertyMatcher::Range {
                    lower,
                    lower_inclusive,
                    upper,
                    upper_inclusive,
                } => {
                    let lo = if *lower_inclusive {
                        Bound::Included(lower)
                    } else {
                        Bound::Excluded(lower)
                    };
                    let hi = if *upper_inclusive {
                        Bound::Included(upper)
                    } else {
                        Bound::Excluded(upper)
                    };
                    Some((lo, hi))
                }
                _ => None,
            };
            if let Some((lo, hi)) = bounds {
                if let Some(results) = self.graph.lookup_range(node_type, prop, lo, hi) {
                    // Range candidates preserve the shared ordering policy;
                    // MATCH must still reject NULL and apply its own predicates.
                    let filtered = results
                        .into_iter()
                        .filter(|&idx| self.node_matches_properties(idx, props))
                        .collect();
                    return Some(filtered);
                }
            }
        }

        None
    }

    /// Public wrapper for node property matching, used by the fused node-scan
    /// operators and by peer filtering in edge expansion.
    pub fn node_matches_properties_pub(
        &self,
        idx: NodeIndex,
        props: &HashMap<String, PropertyMatcher>,
    ) -> bool {
        self.node_matches_properties(idx, props)
    }

    /// The single-node entry point: everything in [`TypeScanMemo`] is resolved
    /// inline here because a lone node cannot amortise it. Scans over a
    /// candidate stream must go through [`Self::filter_node_candidates`], which
    /// resolves per *type* instead of per node.
    ///
    /// One implementation for every backend: `NodeView` resolves the node's
    /// column store once per node, not once per property read.
    fn node_matches_properties(
        &self,
        idx: NodeIndex,
        props: &HashMap<String, PropertyMatcher>,
    ) -> bool {
        // Disk materializes into this frame (see `filter_node_candidates`):
        // the record is consumed here, so it must not enter the query arena.
        // Heap backends keep the direct borrow.
        let owned;
        let data = if self.graph.graph.is_disk() {
            owned = self.graph.graph.owned_node_data(idx);
            owned.as_ref()
        } else {
            self.graph.graph.node_weight(idx)
        };
        let Some(data) = data else {
            return false;
        };
        self.node_data_matches_properties(data, props)
    }

    /// [`Self::node_matches_properties`] against an already-borrowed record.
    #[inline]
    fn node_data_matches_properties(
        &self,
        data: &NodeData,
        props: &HashMap<String, PropertyMatcher>,
    ) -> bool {
        let Some(type_str) = self.graph.interner.try_resolve(data.node_type) else {
            return false;
        };
        let node = self.node_view_of(data, self.graph.graph.column_store(data.node_type));
        props.iter().all(|(key, matcher)| {
            let field = self.graph.resolve_alias(type_str, key);
            self.prop_matches(node, type_str, field, InternedKey::from_str(field), matcher)
        })
    }

    /// Pair a node's weight with the column store its *type* lives in, without
    /// re-probing the backend's store map when the caller already resolved it.
    ///
    /// Equivalent to [`GraphRead::node_view`], which resolves the store from
    /// the node's own type on every call — the cost a scan hoists out of its
    /// loop.
    #[inline]
    fn node_view_of<'d>(
        &self,
        data: &'d NodeData,
        store: Option<&'d std::sync::Arc<ColumnStore>>,
    ) -> NodeView<'d> {
        let resolved = data
            .properties
            .columnar_row_id()
            .and_then(|row_id| store.map(|store| (&**store, row_id)));
        NodeView::new(data, resolved)
    }

    /// Match one node against matchers whose field names this node's type has
    /// already resolved.
    fn node_matches_resolved(&self, node: NodeView<'_>, memo: &TypeScanMemo<'_>) -> bool {
        memo.props.iter().all(|resolved| {
            self.prop_matches(
                node,
                memo.type_str,
                resolved.field,
                resolved.key,
                resolved.matcher,
            )
        })
    }

    /// One alias-resolved property matcher against one node.
    ///
    /// `field` is the alias-resolved field name and `key` its interned form.
    /// Both are pure functions of `(node type, user key)` — never of the node —
    /// which is what lets a typed scan resolve them once per type.
    #[inline]
    fn prop_matches(
        &self,
        node: NodeView<'_>,
        type_str: &str,
        field: &str,
        key: InternedKey,
        matcher: &PropertyMatcher,
    ) -> bool {
        // Byte equality against a stored user property answers from the column
        // without building a `StrField` at all — worth its own arm because
        // `StrField` is wider than a register pair, so the general route below
        // returns it through memory once per candidate row.
        if !matches!(
            field,
            "name" | "title" | "id" | "type" | "node_type" | "label"
        ) {
            if let PropertyMatcher::Equals(Value::String(target)) = matcher {
                return node.str_prop_eq(key, target) == Some(true);
            }
        }

        // Zero-alloc route for every other matcher whose answer is a function
        // of the string form alone — the identity fields' equality included.
        // Under columnar storage the owned `Value::String` the general path
        // materialises is one heap allocation *per candidate row*, which is the
        // whole cost of a text-filter scan.
        if let Some(test) = str_field_test(matcher) {
            return node.resolved_field_str(type_str, field, key).is(test);
        }

        // Identity fields, then a stored property (a user `label`/`type`/
        // `name`… wins — KG-1), then the structural soft-alias fallback. Shared
        // with the planner's NDV statistic so the two cannot disagree about
        // what a filter on `field` sees.
        match node.resolved_field(type_str, field, key) {
            Some(v) => self.value_matches(&v, matcher),
            None => false,
        }
    }

    /// Resolve, once, everything a candidate scan would otherwise redo for
    /// every node of the same type. `None` when the type key is unknown to the
    /// interner, which reads as "no node of this type matches".
    fn build_type_scan_memo<'m>(
        &'m self,
        type_key: InternedKey,
        props: &'m HashMap<String, PropertyMatcher>,
    ) -> Option<TypeScanMemo<'m>> {
        let type_str = self.graph.interner.try_resolve(type_key)?;
        let store = self.graph.graph.column_store(type_key);
        let resolved: Vec<ResolvedMatcher<'m>> = props
            .iter()
            .map(|(key, matcher)| {
                let field = self.graph.resolve_alias(type_str, key);
                ResolvedMatcher {
                    field,
                    key: InternedKey::from_str(field),
                    matcher,
                }
            })
            .collect();
        let filter = column_filter::column_filter_enabled()
            .then(|| {
                ColumnFilter::compile(store, resolved.iter().map(|r| (r.field, r.key, r.matcher)))
            })
            .flatten();
        Some(TypeScanMemo {
            type_key,
            type_str,
            store,
            props: resolved,
            filter,
        })
    }

    /// The free [`value_matches`], with `self.params` supplied — the same body
    /// the column-major scan filter ([`super::column_filter`]) evaluates.
    #[inline]
    fn value_matches(&self, value: &Value, matcher: &PropertyMatcher) -> bool {
        value_matches(self.params, value, matcher)
    }

    /// Whether `idx` satisfies a node pattern's label constraints — its
    /// `node_type` (matched as primary OR secondary label) and every
    /// `extra_label` — multi-label aware via `DirGraph::node_has_label`.
    /// Properties are matched separately. Used by edge-expansion target
    /// filtering so a typed endpoint like `(b:VIP)` matches nodes carrying
    /// `VIP` as a secondary label, not only as their primary type.
    fn node_matches_pattern_labels(&self, idx: NodeIndex, node_pattern: &NodePattern) -> bool {
        // Alternation: any branch admits; plain patterns are the one-branch
        // case of the same rule. Extras (empty under alternation) must all
        // hold.
        let alts = node_pattern.label_alternatives();
        if !alts.is_empty()
            && !alts
                .iter()
                .any(|l| self.graph.node_has_label(idx, InternedKey::from_str(l)))
        {
            return false;
        }
        node_pattern
            .extra_labels
            .iter()
            .all(|l| self.graph.node_has_label(idx, InternedKey::from_str(l)))
    }

    /// If this node-pattern's variable is *already* bound — externally (an
    /// UNWIND pre-binding) or earlier in the same pattern (a cycle that
    /// re-uses a variable, e.g. `(p)-[]->(c)-[]->(pr)<-[]-(p)`) — return the
    /// bound node index. The matching segment then only needs to confirm the
    /// edge to that one node (passed as `expand_from_node`'s `target_hint`)
    /// rather than expanding every neighbour and discarding all but one.
    /// `None` ⇒ the variable is new (or anonymous) ⇒ a normal full expansion.
    fn bound_target(
        &self,
        node_pattern: &NodePattern,
        current_match: &PatternMatch,
    ) -> Option<NodeIndex> {
        let var = node_pattern.variable.as_ref()?;
        if let Some(&idx) = self.pre_bindings.get(var) {
            return Some(idx);
        }
        current_match.bindings.iter().find_map(|(name, binding)| {
            if name == var {
                match binding {
                    MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => Some(*index),
                    _ => None,
                }
            } else {
                None
            }
        })
    }

    /// Whether [`Self::expand_disk_peers`] can answer this hop.
    ///
    /// The sweep skips `EdgeData` materialization entirely, which on a disk
    /// graph is the difference between reading `edge_endpoints.bin` (13 GB on
    /// Wikidata) and not. It can only do that when nothing downstream needs
    /// the relationship: no named variable, no property filter, no trail, and
    /// a single connection type the CSR can pre-filter on. The `is_disk()`
    /// gate keeps memory/mapped on the ordinary path, where materialization is
    /// already free via petgraph.
    fn disk_peer_sweep_applies(&self, edge_pattern: &EdgePattern) -> bool {
        edge_pattern.variable.is_none()
            && edge_pattern.properties.is_none()
            && !edge_pattern.needs_path_info
            && edge_pattern.connection_types.is_none()
            && self.graph.graph.is_disk()
    }

    /// One hop over the disk CSR's peer list, without materialising an edge.
    fn expand_disk_peers(
        &self,
        source: NodeIndex,
        edge_pattern: &EdgePattern,
        node_pattern: &NodePattern,
        max_results: Option<usize>,
        target_hint: Option<NodeIndex>,
    ) -> Vec<(NodeIndex, MatchBinding)> {
        let conn_u64 = edge_pattern
            .connection_type
            .as_ref()
            .map(|ct| InternedKey::from_str(ct).as_u64());
        let directions: &[Direction] = match edge_pattern.direction {
            EdgeDirection::Outgoing => &[Direction::Outgoing],
            EdgeDirection::Incoming => &[Direction::Incoming],
            EdgeDirection::Both => &[Direction::Outgoing, Direction::Incoming],
        };
        let mut results = Vec::new();
        for &dir in directions {
            for (peer_idx, _edge_idx) in self.graph.graph.iter_peers_filtered(source, dir, conn_u64)
            {
                if max_results.is_some_and(|max| results.len() >= max) {
                    break;
                }
                if target_hint.is_some_and(|hint| peer_idx != hint) {
                    continue;
                }
                if !edge_pattern.skip_target_type_check
                    && !self.node_matches_pattern_labels(peer_idx, node_pattern)
                {
                    continue;
                }
                if let Some(ref props) = node_pattern.properties {
                    if !self.node_matches_properties(peer_idx, props) {
                        continue;
                    }
                }
                // Placeholder binding — the caller won't use it (no variable).
                results.push((peer_idx, MatchBinding::NodeRef(peer_idx)));
            }
        }
        results
    }

    fn expand_from_node(
        &self,
        source: NodeIndex,
        edge_pattern: &EdgePattern,
        node_pattern: &NodePattern,
        max_results: Option<usize>,
        // When the segment's target variable is already bound (an UNWIND
        // pre-binding or a cycle that re-binds an earlier variable), only the
        // edge(s) to that one node can match. Rejecting every other peer here —
        // before binding construction and the caller's per-result scan — turns
        // an expand-all-then-filter (O(degree)) into a targeted check. Skipped
        // for variable-length segments (those return via `expand_var_length`).
        target_hint: Option<NodeIndex>,
        // Reusable visited marks for the fast variable-length BFS, owned by the
        // hop loop so a per-row buffer allocation+zeroing that scaled with the
        // graph becomes a stamp bump. Unused by every other expansion shape.
        visited: &mut VisitedStamps,
    ) -> Result<Vec<(NodeIndex, MatchBinding)>, String> {
        // Early exit: if the specified connection type doesn't exist in the graph, skip all iteration
        if let Some(ref types) = edge_pattern.connection_types {
            if !types.iter().any(|t| self.graph.has_connection_type(t)) {
                return Ok(Vec::new());
            }
        } else if let Some(ref conn_type) = edge_pattern.connection_type {
            if !self.graph.has_connection_type(conn_type) {
                return Ok(Vec::new());
            }
        }

        // `max_results` reaches a variable-length expansion only when every row
        // it returns survives the post-expansion filters
        // (`HopPlan::var_length_cap_safe`), so the BFS may stop once filled.
        if let Some((min_hops, max_hops)) = edge_pattern.var_length {
            return self.expand_var_length(
                source,
                &VarLengthSegment {
                    edge: edge_pattern,
                    node: node_pattern,
                    min_hops,
                    max_hops,
                },
                max_results,
                visited,
            );
        }

        if self.disk_peer_sweep_applies(edge_pattern) {
            return Ok(self.expand_disk_peers(
                source,
                edge_pattern,
                node_pattern,
                max_results,
                target_hint,
            ));
        }

        let mut results = Vec::new();

        // Static slice, no heap alloc.
        let directions: &[Direction] = match edge_pattern.direction {
            EdgeDirection::Outgoing => &[Direction::Outgoing],
            EdgeDirection::Incoming => &[Direction::Incoming],
            EdgeDirection::Both => &[Direction::Outgoing, Direction::Incoming],
        };

        // Pre-intern connection type(s) for fast u64 == u64 comparison in inner loop
        let conn_keys: Option<Vec<InternedKey>> = edge_pattern
            .connection_types
            .as_ref()
            .map(|types| types.iter().map(|t| InternedKey::from_str(t)).collect());
        let conn_key = if conn_keys.is_none() {
            edge_pattern
                .connection_type
                .as_ref()
                .map(|ct| InternedKey::from_str(ct))
        } else {
            None
        };

        for &direction in directions {
            // Pre-filter by single connection type in DiskGraph (skips materialization)
            let edges = self
                .graph
                .graph
                .edges_directed_filtered(source, direction, conn_key);

            for edge in edges {
                // Connection-type check uses the cheap accessor — on disk this
                // avoids materialising the edge (heap alloc + property clone)
                // for every edge just to read its type. A single conn_key is
                // already pre-filtered by DiskGraph (this is then a no-op);
                // multi-type conn_keys still need the post-filter.
                let conn_type = edge.connection_type();
                if let Some(ref keys) = conn_keys {
                    if !keys.contains(&conn_type) {
                        continue;
                    }
                } else if let Some(key) = conn_key {
                    if conn_type != key {
                        continue;
                    }
                }

                // Inline edge filter pushed from a downstream WHERE: eliminates
                // rows the post-expansion WHERE would have discarded anyway, so
                // the dominant cost (binding allocation + node-property reads
                // below) never happens. Reads edge properties, so it
                // materialises the edge (lazy on disk) only when a filter exists.
                if let Some(ref filter) = edge_pattern.edge_filter {
                    let edge_data = edge.weight();
                    let edge_source = edge.source();
                    let edge_target = edge.target();
                    // Map the matcher's `direction` onto "is the peer
                    // node on the edge's start side?" — the form
                    // RelEdgePredicate works with.
                    let peer_is_start = match (filter.anchor, direction) {
                        (AnchorSide::Source, Direction::Outgoing) => false,
                        (AnchorSide::Source, Direction::Incoming) => true,
                        (AnchorSide::Target, Direction::Outgoing) => true,
                        (AnchorSide::Target, Direction::Incoming) => false,
                    };
                    let keep = filter.predicate.eval(
                        conn_type,
                        peer_is_start,
                        edge_source,
                        edge_target,
                        &|prop: &str| edge_data.get_property(prop).cloned(),
                    );
                    if !keep {
                        continue;
                    }
                }

                // Check edge properties if specified — materialise lazily.
                if let Some(ref props) = edge_pattern.properties {
                    let edge_data = edge.weight();
                    let matches = props.iter().all(|(key, matcher)| {
                        edge_data
                            .get_property(key)
                            .map(|v| self.value_matches(v, matcher))
                            .unwrap_or(false)
                    });
                    if !matches {
                        continue;
                    }
                }

                let target = match direction {
                    Direction::Outgoing => edge.target(),
                    Direction::Incoming => edge.source(),
                };

                // Bound-target fast reject: the edge doesn't reach the one
                // already-bound node, so skip label/property checks + binding.
                if target_hint.is_some_and(|h| target != h) {
                    continue;
                }

                // Primary + secondary labels; skipped when the edge type
                // guarantees the target's type.
                if !edge_pattern.skip_target_type_check
                    && !self.node_matches_pattern_labels(target, node_pattern)
                {
                    continue;
                }

                if let Some(ref props) = node_pattern.properties {
                    if !self.node_matches_properties(target, props) {
                        continue;
                    }
                }

                // Index-only binding: `conn_type` was already read via the
                // cheap accessor above, and consumers resolve edge properties
                // from the graph on demand, so no edge materialisation or
                // property-map clone happens here even when the edge variable
                // is named.
                let edge_binding = MatchBinding::Edge {
                    source,
                    target,
                    edge_index: edge.id(),
                    connection_type: conn_type,
                };

                results.push((target, edge_binding));
                if max_results.is_some_and(|max| results.len() >= max) {
                    return Ok(results);
                }
            }
        }

        Ok(results)
    }

    /// In lightweight mode (Cypher executor path), only `index` is populated
    /// since the executor resolves node data on demand via graph lookups.
    fn node_to_binding(&self, idx: NodeIndex) -> MatchBinding {
        if self.lightweight {
            return MatchBinding::NodeRef(idx);
        }
        if let Some(node) = self.graph.graph.node_view(idx) {
            let node_title = node.title();
            let title_str = match &*node_title {
                Value::String(s) => s.clone(),
                Value::Int64(i) => i.to_string(),
                Value::Float64(f) => f.to_string(),
                Value::UniqueId(u) => u.to_string(),
                _ => format!("{:?}", *node_title),
            };
            MatchBinding::Node {
                index: idx,
                node_type: node.node_type_str(&self.graph.interner).to_string(),
                title: title_str,
                id: node.id().into_owned(),
                properties: node.properties_cloned(&self.graph.interner),
            }
        } else {
            MatchBinding::Node {
                index: idx,
                node_type: "Unknown".to_string(),
                title: "Unknown".to_string(),
                id: Value::Null,
                properties: HashMap::new(),
            }
        }
    }
}

#[path = "matcher_expansion.rs"]
mod expansion;

#[path = "matcher_var_length.rs"]
mod var_length;

use var_length::{VarLengthSegment, VisitedStamps};

#[cfg(test)]
#[path = "matcher_id_lookup_tests.rs"]
mod id_lookup_tests;

#[cfg(test)]
#[path = "matcher_composite_index_tests.rs"]
mod composite_index_tests;

#[cfg(test)]
#[path = "matcher_limit_seed_tests.rs"]
mod limit_seed_tests;

#[cfg(test)]
#[path = "matcher_ceiling_tests.rs"]
mod ceiling_tests;

#[cfg(test)]
#[path = "matcher_property_index_tests.rs"]
mod property_index_tests;
