// Matcher — executes parsed Pattern against a DirGraph.
//
// PatternExecutor implements a BFS expansion state machine with
// variable bindings, property filters, edge direction, variable-length
// paths, and Rayon-parallelised expansion for large match sets.

use crate::datatypes::values::Value;
use crate::graph::core::filtering::{compare_values, values_equal};
use crate::graph::languages::cypher::result::Bindings;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use super::pattern::{
    AnchorSide, EdgeDirection, EdgePattern, MatchBinding, NodePattern, Pattern, PatternElement,
    PatternMatch, PropertyMatcher,
};

/// Minimum match count to use parallel expansion via rayon.
/// Set high: each expand_from_node does light work (a few edge iterations),
/// so rayon overhead only pays off for very large match sets. Also avoids
/// contention when multiple queries run concurrently (shared thread pool).
const EXPANSION_RAYON_THRESHOLD: usize = 8192;

/// Return the ordered list of index-name candidates to try when the
/// cross-type fast path sees a query for `prop`. The first entry is
/// always `prop` itself.
///
/// Two sources of aliases:
///   1. Hardcoded families — `title ↔ label ↔ name` and `id ↔ nid ↔
///      qid`. Covers the common KGLite conventions without any
///      per-graph config.
///   2. Per-type `title_field_aliases` / `id_field_aliases` on
///      `DirGraph`. If any node type registered `'original_name'` as
///      its title alias, a query for `{title: 'X'}` falls back to the
///      `original_name` index too. Derived automatically from the
///      graph's existing schema — no new config API.
fn global_alias_candidates(prop: &str, graph: &DirGraph) -> Vec<String> {
    let mut out: Vec<String> = vec![prop.to_string()];
    let (family, per_type_map): (&[&str], &HashMap<String, String>) = match prop {
        "title" | "label" | "name" => (&["title", "label", "name"], &graph.title_field_aliases),
        "id" | "nid" | "qid" => (&["id", "nid", "qid"], &graph.id_field_aliases),
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

// ============================================================================
// Executor
// ============================================================================

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
    /// Optional deadline for aborting long-running pattern execution.
    deadline: Option<Instant>,
    /// When set, deduplicate results by NodeIndex of the named variable.
    /// At the last hop expansion, paths leading to already-seen target nodes
    /// are skipped, avoiding PatternMatch cloning and allocation overhead.
    distinct_target_var: Option<String>,
}

/// Static empty params for constructors that don't take parameters.
static EMPTY_PARAMS: std::sync::LazyLock<HashMap<String, Value>> =
    std::sync::LazyLock::new(HashMap::new);

/// Static empty bindings for constructors that don't take pre-bindings.
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
            distinct_target_var: None,
        }
    }

    /// Lightweight executor with query parameters for resolving $param in inline properties
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
            distinct_target_var: None,
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
            distinct_target_var: None,
        }
    }

    /// Set a deadline for pattern execution. Returns self for chaining.
    pub fn set_deadline(mut self, deadline: Option<Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    /// Set a distinct target variable for deduplication during pattern matching.
    /// At the last hop, paths leading to already-seen target NodeIndex values
    /// are skipped, avoiding PatternMatch cloning overhead.
    pub fn set_distinct_target(mut self, var: Option<String>) -> Self {
        self.distinct_target_var = var;
        self
    }

    /// Execute the pattern and return all matches
    pub fn execute(&self, pattern: &Pattern) -> Result<Vec<PatternMatch>, String> {
        if pattern.elements.is_empty() {
            return Ok(Vec::new());
        }

        // Start with the first node pattern
        let first_node = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => {
                return Err(
                    "Pattern must start with a node in parentheses. Example: (n:Person) or ()"
                        .to_string(),
                )
            }
        };

        // Find all nodes matching the first pattern.
        // For multi-hop patterns with max_matches, cap the source candidates to avoid
        // O(N) allocation when only a small number of results are needed (e.g. LIMIT 10
        // on an 11M-node type). The expansion loop enforces the exact max_matches.
        let has_edges = pattern.elements.len() > 1;
        let source_cap = if has_edges {
            // Multi-hop with LIMIT: cap sources to avoid O(N) allocation + PatternMatch
            // construction for millions of nodes. The expansion loop enforces exact
            // max_matches via early-exit. 100x headroom handles sparse match patterns
            // (each source needs only a 1% chance of producing a match to hit the limit).
            self.max_matches.map(|m| m.saturating_mul(100).max(1000))
        } else {
            // Single-node pattern: exact truncation
            self.max_matches
        };
        // Pre-bound first nodes (e.g. `MATCH (f {id: X}) MATCH (f)-[:R]->(c)`)
        // must skip the inverted-index fast path — that path returns every source
        // for the edge type, ignoring the binding. find_matching_nodes resolves
        // the variable to a single node directly.
        let first_is_prebound = first_node
            .variable
            .as_ref()
            .map(|v| self.pre_bindings.get(v).is_some())
            .unwrap_or(false);

        // Try connection-type inverted index for untyped source nodes with typed edges.
        // Instead of iterating all 124M nodes hoping to find P31 sources, the inverted
        // index gives us exactly which nodes have P31 outgoing edges.
        let mut initial_nodes = if !first_is_prebound
            && has_edges
            && first_node.node_type.is_none()
            && first_node.properties.is_none()
        {
            // Check if the first edge has a connection type we can look up
            let edge_conn_type = if let Some(PatternElement::Edge(ep)) = pattern.elements.get(1) {
                if ep.var_length.is_none() {
                    ep.connection_type
                        .as_ref()
                        .map(|ct| InternedKey::from_str(ct))
                } else {
                    None
                }
            } else {
                None
            };
            // Check edge direction — inverted index only covers outgoing sources
            let is_outgoing = if let Some(PatternElement::Edge(ep)) = pattern.elements.get(1) {
                ep.direction == EdgeDirection::Outgoing
            } else {
                false
            };
            if let (Some(ct), true) = (edge_conn_type, is_outgoing) {
                // Pass `source_cap` through so we don't eagerly copy the
                // whole 400 MB source list from the inverted index for
                // a query that only needs 1 000 of them.
                if let Some(sources) = self
                    .graph
                    .graph
                    .sources_for_conn_type_bounded(ct, source_cap)
                {
                    // Convert u32 source IDs to NodeIndex
                    sources
                        .into_iter()
                        .map(|s| NodeIndex::new(s as usize))
                        .collect()
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
                initial_nodes.truncate(cap);
            }
        }

        // Initialize matches with first node bindings
        let mut matches: Vec<PatternMatch> = initial_nodes
            .iter()
            .map(|&idx| {
                let mut pm = PatternMatch {
                    bindings: Vec::new(),
                };
                if let Some(ref var) = first_node.variable {
                    pm.bindings.push((var.clone(), self.node_to_binding(idx)));
                }
                pm
            })
            .collect();

        // Track current node indices for each match
        let mut current_indices: Vec<NodeIndex> = initial_nodes;

        // Pre-allocate dedup set for distinct_target_var optimization
        let mut distinct_seen: HashSet<NodeIndex> = if self.distinct_target_var.is_some() {
            HashSet::with_capacity(matches.len())
        } else {
            HashSet::new()
        };

        // Process edge-node pairs
        let mut i = 1;
        while i < pattern.elements.len() {
            // max_matches is enforced DURING expansion (inner-loop checks below),
            // not between hops, to avoid breaking before edges are expanded.
            let is_last_hop = i + 2 >= pattern.elements.len();
            if let Some(dl) = self.deadline {
                if Instant::now() > dl {
                    return Err("Query timed out".to_string());
                }
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

            // Expand each current match
            let (mut new_matches, mut new_indices) = if matches.len() >= EXPANSION_RAYON_THRESHOLD
                && self.max_matches.is_none()
            {
                // Parallel expansion — each match's expand_from_node is independent.
                // Errors (e.g. deadline exceeded) are captured via AtomicBool and
                // the first error message is saved for propagation after the parallel section.
                let had_error = std::sync::atomic::AtomicBool::new(false);
                let first_error: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
                let results: Vec<(PatternMatch, NodeIndex)> = matches
                    .par_iter()
                    .zip(current_indices.par_iter())
                    .flat_map(|(current_match, &source_idx)| {
                        // Short-circuit once any thread has detected a timeout/error,
                        // and independently check the deadline from each thread so a
                        // parallel expansion over 100M+ sources cannot run unbounded.
                        if had_error.load(std::sync::atomic::Ordering::Relaxed) {
                            return Vec::new();
                        }
                        if let Some(dl) = self.deadline {
                            if Instant::now() > dl {
                                if !had_error.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                    *first_error.lock().unwrap() =
                                        Some("Query timed out".to_string());
                                }
                                return Vec::new();
                            }
                        }
                        let expansions = match self.expand_from_node(
                            source_idx,
                            edge_pattern,
                            node_pattern,
                            None,
                        ) {
                            Ok(exp) => exp,
                            Err(e) => {
                                if !had_error.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                    *first_error.lock().unwrap() = Some(e);
                                }
                                return Vec::new();
                            }
                        };
                        expansions
                            .into_iter()
                            .filter_map(|(target_idx, edge_binding)| {
                                if let Some(ref var) = node_pattern.variable {
                                    if let Some(&bound_idx) = self.pre_bindings.get(var) {
                                        if target_idx != bound_idx {
                                            return None;
                                        }
                                    }
                                    // Enforce intra-pattern variable constraint
                                    let already_bound = current_match.bindings.iter().find_map(
                                        |(name, binding)| {
                                            if name == var {
                                                match binding {
                                                    MatchBinding::Node { index, .. }
                                                    | MatchBinding::NodeRef(index) => Some(*index),
                                                    _ => None,
                                                }
                                            } else {
                                                None
                                            }
                                        },
                                    );
                                    if let Some(bound_idx) = already_bound {
                                        if target_idx != bound_idx {
                                            return None;
                                        }
                                    }
                                }
                                let mut new_match = current_match.clone();
                                if let Some(ref var) = edge_pattern.variable {
                                    new_match.bindings.push((var.clone(), edge_binding));
                                } else if edge_pattern.needs_path_info
                                    && matches!(
                                        edge_binding,
                                        MatchBinding::VariableLengthPath { .. }
                                    )
                                {
                                    new_match
                                        .bindings
                                        .push((format!("__anon_vlpath_{}", i), edge_binding));
                                }
                                if let Some(ref var) = node_pattern.variable {
                                    new_match
                                        .bindings
                                        .push((var.clone(), self.node_to_binding(target_idx)));
                                }
                                Some((new_match, target_idx))
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                // Propagate any error that occurred during parallel expansion
                if had_error.load(std::sync::atomic::Ordering::Relaxed) {
                    let err = first_error
                        .into_inner()
                        .unwrap()
                        .unwrap_or_else(|| "parallel expansion failed".to_string());
                    return Err(err);
                }
                // Apply distinct-target dedup for parallel results (sequential path
                // does this inline, but parallel path can't without synchronization).
                let needs_dedup = i + 2 >= pattern.elements.len()
                    && self
                        .distinct_target_var
                        .as_ref()
                        .is_some_and(|dtv| node_pattern.variable.as_deref() == Some(dtv.as_str()));
                if needs_dedup {
                    let mut seen_targets = HashSet::new();
                    let filtered: Vec<_> = results
                        .into_iter()
                        .filter(|(_, target_idx)| seen_targets.insert(*target_idx))
                        .collect();
                    filtered.into_iter().unzip()
                } else {
                    results.into_iter().unzip()
                }
            } else {
                // Sequential expansion with max_matches early-exit
                let mut new_matches_seq = Vec::new();
                let mut new_indices_seq = Vec::new();
                let mut expand_count: usize = 0;
                // At the last hop, enforce exact max_matches.
                // At intermediate hops, use a generous overcommit (50x) to avoid
                // expanding far more intermediates than needed while ensuring
                // enough survive to produce max_matches final results.
                let hop_limit = if is_last_hop {
                    self.max_matches
                } else {
                    self.max_matches.map(|m| m.saturating_mul(50).max(1000))
                };
                for (current_match, &source_idx) in matches.iter().zip(current_indices.iter()) {
                    if hop_limit.is_some_and(|max| new_matches_seq.len() >= max) {
                        break;
                    }
                    let remaining = hop_limit.map(|max| max.saturating_sub(new_matches_seq.len()));
                    let expansions =
                        self.expand_from_node(source_idx, edge_pattern, node_pattern, remaining)?;
                    for (target_idx, edge_binding) in expansions {
                        expand_count += 1;
                        if expand_count.is_multiple_of(1024) {
                            if let Some(dl) = self.deadline {
                                if Instant::now() > dl {
                                    return Err("Query timed out".to_string());
                                }
                            }
                        }
                        if hop_limit.is_some_and(|max| new_matches_seq.len() >= max) {
                            break;
                        }
                        if let Some(ref var) = node_pattern.variable {
                            if let Some(&bound_idx) = self.pre_bindings.get(var) {
                                if target_idx != bound_idx {
                                    continue;
                                }
                            }
                            // Enforce intra-pattern variable constraint:
                            // if this variable was already bound earlier in the
                            // same pattern, the target must match that binding.
                            let already_bound =
                                current_match.bindings.iter().find_map(|(name, binding)| {
                                    if name == var {
                                        match binding {
                                            MatchBinding::Node { index, .. }
                                            | MatchBinding::NodeRef(index) => Some(*index),
                                            _ => None,
                                        }
                                    } else {
                                        None
                                    }
                                });
                            if let Some(bound_idx) = already_bound {
                                if target_idx != bound_idx {
                                    continue;
                                }
                            }
                        }
                        // Distinct-target dedup: at the last hop, skip targets already seen
                        if i + 1 >= pattern.elements.len() {
                            if let Some(ref dtv) = self.distinct_target_var {
                                if node_pattern.variable.as_deref() == Some(dtv.as_str())
                                    && !distinct_seen.insert(target_idx)
                                {
                                    continue;
                                }
                            }
                        }
                        let mut new_match = current_match.clone();
                        if let Some(ref var) = edge_pattern.variable {
                            new_match.bindings.push((var.clone(), edge_binding));
                        } else if edge_pattern.needs_path_info
                            && matches!(edge_binding, MatchBinding::VariableLengthPath { .. })
                        {
                            new_match
                                .bindings
                                .push((format!("__anon_vlpath_{}", i), edge_binding));
                        }
                        if let Some(ref var) = node_pattern.variable {
                            new_match
                                .bindings
                                .push((var.clone(), self.node_to_binding(target_idx)));
                        }
                        new_matches_seq.push(new_match);
                        new_indices_seq.push(target_idx);
                    }
                }
                (new_matches_seq, new_indices_seq)
            };

            // Check deadline after expansion (covers both parallel and sequential paths)
            if let Some(dl) = self.deadline {
                if Instant::now() > dl {
                    return Err("Query timed out".to_string());
                }
            }

            // Apply hop limit truncation (for parallel path which can't early-exit)
            let truncate_limit = if is_last_hop {
                self.max_matches
            } else {
                self.max_matches.map(|m| m.saturating_mul(50).max(1000))
            };
            if let Some(max) = truncate_limit {
                new_matches.truncate(max);
                new_indices.truncate(max);
            }

            // Intermediate dedup: when distinct_target_var is set and this is
            // NOT the final hop and the current node is anonymous (no variable),
            // deduplicate by NodeIndex to reduce work at subsequent hops.
            if self.distinct_target_var.is_some()
                && i + 1 < pattern.elements.len()
                && node_pattern.variable.is_none()
            {
                let mut seen_idx = HashSet::with_capacity(new_indices.len());
                let mut deduped_matches = Vec::with_capacity(new_indices.len());
                let mut deduped_indices = Vec::with_capacity(new_indices.len());
                for (m, idx) in new_matches.into_iter().zip(new_indices) {
                    if seen_idx.insert(idx) {
                        deduped_matches.push(m);
                        deduped_indices.push(idx);
                    }
                }
                matches = deduped_matches;
                current_indices = deduped_indices;
            } else {
                matches = new_matches;
                current_indices = new_indices;
            }
            i += 1;
        }

        Ok(matches)
    }

    /// Public wrapper for find_matching_nodes (used by Cypher executor for shortestPath)
    pub fn find_matching_nodes_pub(&self, pattern: &NodePattern) -> Result<Vec<NodeIndex>, String> {
        self.find_matching_nodes(pattern)
    }

    /// Find all nodes matching a node pattern
    fn find_matching_nodes(&self, pattern: &NodePattern) -> Result<Vec<NodeIndex>, String> {
        // Multi-label aware: when the pattern carries secondary labels
        // OR the graph has secondary labels in play, take a slower
        // union path that consults both `type_indices` (primary) and
        // `secondary_label_index`. Single-label graphs hit the
        // unchanged hot path.
        let needs_secondary_path = !pattern.extra_labels.is_empty()
            || (pattern.node_type.is_some() && self.graph.has_secondary_labels);

        // If variable is pre-bound, return only that node (if it matches filters)
        if let Some(ref var) = pattern.variable {
            if let Some(&idx) = self.pre_bindings.get(var) {
                if let Some(node) = self.graph.graph.node_weight(idx) {
                    if let Some(ref node_type) = pattern.node_type {
                        let primary_key = InternedKey::from_str(node_type);
                        let labels = self.graph.node_labels(idx);
                        if !labels.contains(&primary_key) {
                            return Ok(vec![]);
                        }
                        for extra in &pattern.extra_labels {
                            let key = InternedKey::from_str(extra);
                            if !labels.contains(&key) {
                                return Ok(vec![]);
                            }
                        }
                        // Suppress unused-binding warning when no extras.
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

        if let Some(ref node_type) = pattern.node_type {
            // Try property index acceleration when we have both type and properties.
            // Skip the shortcuts when we need the secondary-label path —
            // property_indices is keyed by primary type, so it misses
            // multi-labelled nodes whose secondary is the queried label.
            if !needs_secondary_path {
                if let Some(ref props) = pattern.properties {
                    if let Some(indexed) = self.try_index_lookup(node_type, props) {
                        return Ok(indexed);
                    }
                    if let Some(indexed) = self.try_global_index_lookup_typed(node_type, props) {
                        return Ok(indexed);
                    }
                }
            }

            // Gather candidates: primary type_indices ∪ secondary_label_index.
            // The choke-point API forbids primary==secondary on the same
            // node, so the union has no duplicates.
            let candidates: Vec<NodeIndex> = if needs_secondary_path {
                let primary = self
                    .graph
                    .type_indices
                    .get(node_type)
                    .map(|v| v.to_vec())
                    .unwrap_or_default();
                let key = InternedKey::from_str(node_type);
                let secondary = self
                    .graph
                    .secondary_label_index
                    .get(&key)
                    .cloned()
                    .unwrap_or_default();
                if primary.is_empty() && secondary.is_empty() {
                    return Ok(vec![]);
                }
                let mut out = primary;
                out.extend(secondary);
                out
            } else {
                match self.graph.type_indices.get(node_type) {
                    Some(indices) => indices.to_vec(),
                    None => return Ok(vec![]),
                }
            };

            // AND-intersect extra labels (Cypher `MATCH (n:A:B:C)` semantics).
            let filter_extras = !pattern.extra_labels.is_empty();
            let extra_keys: Vec<InternedKey> = if filter_extras {
                pattern
                    .extra_labels
                    .iter()
                    .map(|s| InternedKey::from_str(s))
                    .collect()
            } else {
                Vec::new()
            };

            if let Some(ref props) = pattern.properties {
                let mut out = Vec::new();
                for (i, idx) in candidates.iter().copied().enumerate() {
                    if i & 0xFFF == 0 {
                        self.check_scan_deadline()?;
                    }
                    if filter_extras {
                        let labels = self.graph.node_labels(idx);
                        if !extra_keys.iter().all(|k| labels.contains(k)) {
                            continue;
                        }
                    }
                    if self.node_matches_properties(idx, props) {
                        out.push(idx);
                    }
                }
                Ok(out)
            } else if filter_extras {
                let out = candidates
                    .into_iter()
                    .filter(|&idx| {
                        let labels = self.graph.node_labels(idx);
                        extra_keys.iter().all(|k| labels.contains(k))
                    })
                    .collect();
                Ok(out)
            } else {
                Ok(candidates)
            }
        } else if let Some(ref props) = pattern.properties {
            // Fast path: untyped node with {id: X} — cross-type id lookup.
            // Tries lookup_by_id_readonly on each type. When id_indices are built,
            // each lookup is O(1). Total: O(types) which is fast even for 132K types.
            //
            // Alias-aware: `{nid: 'Q76'}` / `{qid: 'Q76'}` also take this
            // path. Without the alias check the query falls through to a
            // 124M-row scan on Wikidata (times out).
            let id_val_opt = ["id", "nid", "qid"].iter().find_map(|k| {
                if let Some(PropertyMatcher::Equals(v)) = props.get(*k) {
                    Some(v)
                } else {
                    None
                }
            });
            if let Some(id_val) = id_val_opt {
                for node_type in self.graph.type_indices.keys() {
                    if let Some(idx) = self.graph.lookup_by_id_readonly(node_type, id_val) {
                        if props.len() == 1 || self.node_matches_properties(idx, props) {
                            return Ok(vec![idx]);
                        }
                    }
                }
                return Ok(vec![]);
            }
            // Cross-type fast paths: for any Equals(String) or
            // StartsWith(String), consult the persistent global index
            // if one exists for that property. Turns `MATCH (n {label:
            // 'Norway'})` into O(log N) without requiring a type label.
            //
            // Alias-aware: if the literal property name misses, also
            // try common title/id aliases (title↔label↔name,
            // id↔nid↔qid). That way an agent who built the index as
            // `create_global_index('label')` but queries with
            // `{title: 'X'}` still hits the fast path.
            for (prop, matcher) in props {
                let alias_candidates = global_alias_candidates(prop, self.graph);
                match matcher {
                    PropertyMatcher::Equals(Value::String(s)) => {
                        for idx_name in &alias_candidates {
                            if let Some(candidates) =
                                self.graph.graph.lookup_by_property_eq_any_type(idx_name, s)
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
            // No type, no properties — all nodes
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
    /// Strategy: consult the cross-type global index (built once at
    /// save-time, covering every node type), then filter by
    /// `node_type_of(idx)`. For a query that hits a handful of rows
    /// across the whole graph, the filter is O(hits) — microseconds —
    /// and avoids the 13M-row scan entirely.
    ///
    /// Alias-aware via `global_alias_candidates` so an index built as
    /// `global_index_label_*` still serves `{title: 'X'}` queries.
    ///
    /// Returns `None` if no global index matches any alias for any
    /// pushable predicate in `props`, leaving the caller to fall
    /// through to the existing type-scan path.
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
                        if let Some(candidates) =
                            self.graph.graph.lookup_by_property_eq_any_type(alias, s)
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

    /// Try to use property indexes for faster node lookup.
    /// Returns None if no indexes cover the requested properties.
    fn try_index_lookup(
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
            // Apply remaining property filters if any (e.g. {id: IN [...], status: "active"})
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
                let key = (node_type.to_string(), prop_name.clone());
                if !self.graph.property_indices.contains_key(&key) {
                    continue;
                }
                let mut result = Vec::with_capacity(values.len());
                for val in values {
                    if let Some(indices) = self.graph.lookup_by_index(node_type, prop_name, val) {
                        result.extend(indices);
                    }
                }
                if props.len() > 1 {
                    result.retain(|&idx| self.node_matches_properties(idx, props));
                }
                return Some(result);
            }
        }

        // Extract equality values from PropertyMatcher (resolve params)
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

        // Check if any comparison/range matchers exist (for range index path below)
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

        if equality_props.is_empty() && !has_comparison {
            return None;
        }

        // Try ID index for {id: value} patterns — O(1) lookup.
        // Alias-aware: `nid` / `qid` anchor via the same per-type
        // id_index that `id` does — same index, same semantics.
        //
        // Phase A.3 / 0.9.53 fix: also routes through `lookup_by_id_readonly`
        // when the queried property is the user-declared ID alias for
        // this type (e.g. `add_nodes(df, "Star", "starId", "title")`
        // makes `starId` an alias for the canonical id). Pre-fix, those
        // queries fell through to a full type scan.
        if equality_props.len() == 1 {
            let (prop_name, value) = equality_props[0];
            let is_id_alias = matches!(prop_name.as_str(), "id" | "nid" | "qid")
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
                // Fall through: id_index not built yet, use scan below
            }
        }

        // Try composite index for multi-property patterns
        if equality_props.len() >= 2 {
            // Sort in-place — equality_props is a local vec of references, cheap to reorder
            equality_props.sort_by(|a, b| a.0.cmp(b.0));
            let names: Vec<String> = equality_props.iter().map(|(k, _)| (*k).clone()).collect();
            let values: Vec<Value> = equality_props.iter().map(|(_, v)| (*v).clone()).collect();
            if let Some(results) = self
                .graph
                .lookup_by_composite_index(node_type, &names, &values)
            {
                if equality_props.len() == props.len() {
                    // Composite index covers all properties
                    return Some(results);
                }
                // Filter remaining non-indexed properties
                let filtered = results
                    .into_iter()
                    .filter(|&idx| self.node_matches_properties(idx, props))
                    .collect();
                return Some(filtered);
            }
        }

        // Try single property index
        for (prop, value) in &equality_props {
            if let Some(results) = self.graph.lookup_by_index(node_type, prop, value) {
                if equality_props.len() == 1 && props.len() == 1 {
                    // Index covers all properties — return directly
                    return Some(results);
                } else {
                    // Index covers one property — filter remaining manually
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
                if let Some(results) = self.graph.graph.lookup_by_property_eq(node_type, prop, s) {
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

        // Persistent disk-backed prefix index (STARTS WITH). Same
        // `None` / `Some` semantics as the equality path — `None` means
        // no index and the caller falls through to scan. Uses
        // `usize::MAX` as the cap; outer LIMIT pushdown is not wired
        // into matcher state yet.
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

        // Try range index for comparison/range matchers
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
                    if props.len() == 1 {
                        return Some(results);
                    }
                    // Filter remaining non-indexed properties
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

    /// Public wrapper for node property matching, used by FusedNodeScanAggregate.
    pub fn node_matches_properties_pub(
        &self,
        idx: NodeIndex,
        props: &HashMap<String, PropertyMatcher>,
    ) -> bool {
        self.node_matches_properties(idx, props)
    }

    /// Check if a node matches property filters
    /// Optimized: Uses references instead of cloning values
    fn node_matches_properties(
        &self,
        idx: NodeIndex,
        props: &HashMap<String, PropertyMatcher>,
    ) -> bool {
        // Disk-graph fast path: read properties directly from columnar storage
        // without materializing full NodeData. Avoids arena allocation and
        // unnecessary id/title reads. Gated by backend type so in-memory graphs
        // take the unchanged, faster path below.
        if self.graph.graph.is_disk() {
            return self.node_matches_properties_columnar(idx, props);
        }

        // In-memory path: node_weight() is cheap (pointer chase, no allocation)
        if let Some(node) = self.graph.graph.node_weight(idx) {
            for (key, matcher) in props {
                let resolved = self
                    .graph
                    .resolve_alias(node.node_type_str(&self.graph.interner), key);

                // Zero-alloc fast path for `Equals(String)` on a user property.
                // For columnar storage this bypasses cloning bytes out of the
                // mmap into an owned String; for Map/Compact it avoids an
                // unnecessary Value comparison.
                if resolved != "name"
                    && resolved != "title"
                    && resolved != "id"
                    && resolved != "type"
                    && resolved != "node_type"
                    && resolved != "label"
                {
                    if let PropertyMatcher::Equals(Value::String(target)) = matcher {
                        match node
                            .properties
                            .str_prop_eq(InternedKey::from_str(resolved), target)
                        {
                            Some(true) => continue,
                            Some(false) => return false,
                            None => return false,
                        }
                    }
                }

                let value: Option<Cow<'_, Value>> = if resolved == "name" || resolved == "title" {
                    Some(node.title())
                } else if resolved == "id" {
                    Some(node.id())
                } else if resolved == "type" || resolved == "node_type" || resolved == "label" {
                    Some(Cow::Owned(Value::String(
                        node.node_type_str(&self.graph.interner).to_string(),
                    )))
                } else {
                    node.get_property(resolved)
                };

                match value {
                    Some(v) => {
                        if !self.value_matches(&v, matcher) {
                            return false;
                        }
                    }
                    None => return false,
                }
            }
            true
        } else {
            false
        }
    }

    /// Disk-graph columnar fast path for property matching.
    /// Reads individual column values without full NodeData materialization.
    fn node_matches_properties_columnar(
        &self,
        idx: NodeIndex,
        props: &HashMap<String, PropertyMatcher>,
    ) -> bool {
        let node_type_key = match self.graph.graph.node_type_of(idx) {
            Some(k) => k,
            None => return false,
        };
        let type_str = match self.graph.interner.try_resolve(node_type_key) {
            Some(s) => s,
            None => return false,
        };

        for (key, matcher) in props {
            let resolved = self.graph.resolve_alias(type_str, key);

            // Zero-alloc fast path: literal-string equality against a user
            // property skips materialising `Value::String(owned)` per node.
            // Critical on mapped-mode graphs where `get_node_property` clones
            // bytes out of the mmap for every string read.
            if resolved != "name"
                && resolved != "title"
                && resolved != "id"
                && resolved != "type"
                && resolved != "node_type"
                && resolved != "label"
            {
                if let PropertyMatcher::Equals(Value::String(target)) = matcher {
                    let k = InternedKey::from_str(resolved);
                    match self.graph.graph.str_prop_eq(idx, k, target) {
                        Some(true) => continue,
                        Some(false) => return false,
                        None => return false,
                    }
                }
            }

            let value: Option<Cow<'_, Value>> = if resolved == "name" || resolved == "title" {
                self.graph.graph.get_node_title(idx).map(Cow::Owned)
            } else if resolved == "id" {
                self.graph.graph.get_node_id(idx).map(Cow::Owned)
            } else if resolved == "type" || resolved == "node_type" || resolved == "label" {
                Some(Cow::Owned(Value::String(type_str.to_string())))
            } else {
                self.graph
                    .graph
                    .get_node_property(idx, InternedKey::from_str(resolved))
                    .map(Cow::Owned)
            };

            match value {
                Some(v) => {
                    if !self.value_matches(&v, matcher) {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }

    /// Check if a value matches a property matcher.
    /// Uses cross-type numeric comparison (Int64 <-> UniqueId <-> Float64).
    fn value_matches(&self, value: &Value, matcher: &PropertyMatcher) -> bool {
        match matcher {
            PropertyMatcher::Equals(expected) => values_equal(value, expected),
            PropertyMatcher::EqualsParam(name) => self
                .params
                .get(name.as_str())
                .is_some_and(|expected| values_equal(value, expected)),
            // EqualsVar / EqualsNodeProp should be resolved to Equals before
            // pattern matching. If they reach here unresolved, no match is possible.
            PropertyMatcher::EqualsVar(_) | PropertyMatcher::EqualsNodeProp { .. } => false,
            PropertyMatcher::In(values) => values.iter().any(|v| values_equal(value, v)),
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
                Value::String(s) => s.starts_with(prefix.as_str()),
                _ => false,
            },
        }
    }

    /// Expand from a source node via an edge pattern to nodes matching node pattern
    fn expand_from_node(
        &self,
        source: NodeIndex,
        edge_pattern: &EdgePattern,
        node_pattern: &NodePattern,
        max_results: Option<usize>,
    ) -> Result<Vec<(NodeIndex, MatchBinding)>, String> {
        // Early exit: if the specified connection type doesn't exist in the graph, skip all iteration
        if let Some(ref types) = edge_pattern.connection_types {
            // Multi-type: at least one must exist
            if !types.iter().any(|t| self.graph.has_connection_type(t)) {
                return Ok(Vec::new());
            }
        } else if let Some(ref conn_type) = edge_pattern.connection_type {
            if !self.graph.has_connection_type(conn_type) {
                return Ok(Vec::new());
            }
        }

        // Check for variable-length path
        if let Some((min_hops, max_hops)) = edge_pattern.var_length {
            return self.expand_var_length(source, edge_pattern, node_pattern, min_hops, max_hops);
        }

        // Lightweight fast path: when the edge has no named variable and no property
        // filters, skip EdgeData materialization entirely. For disk graphs this avoids
        // reading edge_endpoints.bin (13 GB on Wikidata), cutting I/O in half.
        // Only for single connection type (not multi-type) and directed edges.
        // The `is_disk()` gate avoids the redundant work on memory/mapped, where
        // EdgeData materialization is already free via petgraph. The trait's
        // `iter_peers_filtered` dispatches to the disk CSR fast-path.
        if edge_pattern.variable.is_none()
            && edge_pattern.properties.is_none()
            && !edge_pattern.needs_path_info
            && edge_pattern.connection_types.is_none()
            && self.graph.graph.is_disk()
        {
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
                for (peer_idx, _edge_idx) in
                    self.graph.graph.iter_peers_filtered(source, dir, conn_u64)
                {
                    if max_results.is_some_and(|max| results.len() >= max) {
                        break;
                    }
                    // Check target node type
                    if !edge_pattern.skip_target_type_check {
                        if let Some(ref node_type) = node_pattern.node_type {
                            if let Some(nt) = self.graph.graph.node_type_of(peer_idx) {
                                if nt != InternedKey::from_str(node_type) {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }
                    }
                    // Check target node properties
                    if let Some(ref props) = node_pattern.properties {
                        if !self.node_matches_properties(peer_idx, props) {
                            continue;
                        }
                    }
                    // Placeholder binding — caller won't use it (variable is None)
                    let binding = MatchBinding::NodeRef(peer_idx);
                    results.push((peer_idx, binding));
                }
            }
            return Ok(results);
        }

        let mut results = Vec::new();

        // Determine which directions to check (static slice, no heap alloc)
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
                let edge_data = edge.weight();

                // Check connection type if specified (u64 == u64)
                // For single conn_key, DiskGraph already pre-filtered; this is a no-op.
                // For multi-type conn_keys, post-filter is still needed.
                if let Some(ref keys) = conn_keys {
                    if !keys.contains(&edge_data.connection_type) {
                        continue;
                    }
                } else if let Some(key) = conn_key {
                    if edge_data.connection_type != key {
                        continue;
                    }
                }

                // Inline edge filter pushed from a downstream WHERE.
                // Skip if the predicate rejects this edge — eliminates
                // rows the post-expansion WHERE would have discarded
                // anyway, so the dominant cost (binding allocation +
                // node-property reads below) never happens. The
                // `if let Some` guards the no-filter hot path with a
                // single branch-predicted check.
                if let Some(ref filter) = edge_pattern.edge_filter {
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
                    let conn_ty = edge_data.connection_type;
                    let keep = filter.predicate.eval(
                        conn_ty,
                        peer_is_start,
                        edge_source,
                        edge_target,
                        &|prop: &str| edge_data.get_property(prop).cloned(),
                    );
                    if !keep {
                        continue;
                    }
                }

                // Check edge properties if specified
                if let Some(ref props) = edge_pattern.properties {
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

                // Get target node
                let target = match direction {
                    Direction::Outgoing => edge.target(),
                    Direction::Incoming => edge.source(),
                };

                // Check if target matches node pattern (skip when edge type guarantees it)
                // Uses O(1) node_type_of() to avoid full materialization
                if !edge_pattern.skip_target_type_check {
                    if let Some(ref node_type) = node_pattern.node_type {
                        if let Some(nt) = self.graph.graph.node_type_of(target) {
                            if nt != InternedKey::from_str(node_type) {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }
                }

                // Check node properties if specified
                if let Some(ref props) = node_pattern.properties {
                    if !self.node_matches_properties(target, props) {
                        continue;
                    }
                }

                // Create edge binding — skip expensive clones when the edge has
                // no named variable (the caller will drop the binding unused).
                let edge_binding = if edge_pattern.variable.is_some() {
                    let edge_data = edge.weight();
                    MatchBinding::Edge {
                        source,
                        target,
                        edge_index: edge.id(),
                        connection_type: edge_data.connection_type,
                        properties: edge_data.properties_cloned(&self.graph.interner),
                    }
                } else {
                    MatchBinding::Edge {
                        source,
                        target,
                        edge_index: edge.id(),
                        connection_type: InternedKey::default(),
                        properties: HashMap::new(),
                    }
                };

                results.push((target, edge_binding));
                if max_results.is_some_and(|max| results.len() >= max) {
                    return Ok(results);
                }
            }
        }

        Ok(results)
    }

    /// Fast variable-length path expansion using global BFS dedup.
    /// Used when path info is not needed (no `p = ...`, no named edge variable).
    /// Each node is visited at most once, eliminating redundant re-exploration
    /// from hub nodes at deeper depths.
    fn expand_var_length_fast(
        &self,
        source: NodeIndex,
        edge_pattern: &EdgePattern,
        node_pattern: &NodePattern,
        min_hops: usize,
        max_hops: usize,
    ) -> Result<Vec<(NodeIndex, MatchBinding)>, String> {
        use std::collections::VecDeque;

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

        // Global visited set — each node is explored at most once.
        // Vec<bool> is faster than HashSet for dense NodeIndex (no hashing, cache-friendly).
        let mut visited = vec![false; self.graph.graph.node_bound()];
        visited[source.index()] = true;

        // Queue: (node, depth) — no path vector needed
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        queue.push_back((source, 0));

        let mut results = Vec::new();

        // Zero-hop case: if min_hops == 0, the source node itself is a valid result
        if min_hops == 0 {
            let node_matches = if let Some(ref node_type) = node_pattern.node_type {
                self.graph
                    .graph
                    .node_type_of(source)
                    .map(|nt| nt == InternedKey::from_str(node_type))
                    .unwrap_or(false)
            } else {
                true
            };
            let props_match = if let Some(ref props) = node_pattern.properties {
                self.node_matches_properties(source, props)
            } else {
                true
            };
            if node_matches && props_match {
                results.push((
                    source,
                    MatchBinding::VariableLengthPath {
                        source,
                        target: source,
                        hops: 0,
                        path: Vec::new(),
                    },
                ));
            }
        }

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
                let edges = self
                    .graph
                    .graph
                    .edges_directed_filtered(current, direction, conn_key);

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
                    let edge_data = edge.weight();

                    // Check connection type(s) (u64 == u64)
                    if let Some(ref keys) = conn_keys {
                        if !keys.contains(&edge_data.connection_type) {
                            continue;
                        }
                    } else if let Some(key) = conn_key {
                        if edge_data.connection_type != key {
                            continue;
                        }
                    }

                    // Check edge properties
                    if let Some(ref props) = edge_pattern.properties {
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

                    // Global dedup — skip if already visited at any depth
                    let target_idx = target.index();
                    if visited[target_idx] {
                        continue;
                    }
                    visited[target_idx] = true;

                    let new_depth = depth + 1;

                    // Check if target is a valid result (within hop range + matches node pattern)
                    if new_depth >= min_hops {
                        let node_matches = if edge_pattern.skip_target_type_check {
                            true
                        } else if let Some(ref node_type) = node_pattern.node_type {
                            self.graph
                                .graph
                                .node_type_of(target)
                                .map(|nt| nt == InternedKey::from_str(node_type))
                                .unwrap_or(false)
                        } else {
                            true
                        };

                        let props_match = if let Some(ref props) = node_pattern.properties {
                            self.node_matches_properties(target, props)
                        } else {
                            true
                        };

                        if node_matches && props_match {
                            let edge_binding = MatchBinding::VariableLengthPath {
                                source,
                                target,
                                hops: new_depth,
                                path: Vec::new(),
                            };
                            results.push((target, edge_binding));
                        }
                    }

                    // Continue exploring if we haven't reached max depth
                    if new_depth < max_hops {
                        queue.push_back((target, new_depth));
                    }
                }
            }
        }

        Ok(results)
    }

    /// Expand via variable-length path (BFS within hop range)
    /// Optimized: Only clones paths when branching (multiple valid targets from same node)
    fn expand_var_length(
        &self,
        source: NodeIndex,
        edge_pattern: &EdgePattern,
        node_pattern: &NodePattern,
        min_hops: usize,
        max_hops: usize,
    ) -> Result<Vec<(NodeIndex, MatchBinding)>, String> {
        // Fast path: when path info isn't needed, use global-dedup BFS
        if !edge_pattern.needs_path_info {
            return self.expand_var_length_fast(
                source,
                edge_pattern,
                node_pattern,
                min_hops,
                max_hops,
            );
        }

        use std::collections::VecDeque;

        let mut results = Vec::new();

        // Determine which directions to check (avoid allocation with static slice)
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

        // BFS state: (current_node, depth, path_info)
        // path_info stores the path taken for creating variable-length edge binding
        type PathInfo = Vec<(NodeIndex, InternedKey)>;
        let mut queue: VecDeque<(NodeIndex, usize, PathInfo)> = VecDeque::new();
        let mut visited_at_depth: HashMap<(NodeIndex, usize), bool> = HashMap::new();

        queue.push_back((source, 0, Vec::new()));

        // Zero-hop case: if min_hops == 0, the source node itself is a valid result
        // (matching "zero hops" means the source IS the target).
        if min_hops == 0 {
            let node_matches = if let Some(ref node_type) = node_pattern.node_type {
                self.graph
                    .graph
                    .node_type_of(source)
                    .map(|nt| nt == InternedKey::from_str(node_type))
                    .unwrap_or(false)
            } else {
                true
            };
            let props_match = if let Some(ref props) = node_pattern.properties {
                self.node_matches_properties(source, props)
            } else {
                true
            };
            if node_matches && props_match {
                results.push((
                    source,
                    MatchBinding::VariableLengthPath {
                        source,
                        target: source,
                        hops: 0,
                        path: Vec::new(),
                    },
                ));
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
            }
            if depth >= max_hops {
                continue;
            }

            // First pass: collect all valid targets to know how many branches we'll have
            // This avoids cloning paths unnecessarily when only one target exists
            let mut valid_targets: Vec<(NodeIndex, InternedKey)> = Vec::new();

            for &direction in directions {
                let edges = self
                    .graph
                    .graph
                    .edges_directed_filtered(current, direction, conn_key);

                for edge in edges {
                    let edge_data = edge.weight();

                    // Check connection type(s) if specified (u64 == u64)
                    if let Some(ref keys) = conn_keys {
                        if !keys.contains(&edge_data.connection_type) {
                            continue;
                        }
                    } else if let Some(key) = conn_key {
                        if edge_data.connection_type != key {
                            continue;
                        }
                    }

                    // Check edge properties if specified
                    if let Some(ref props) = edge_pattern.properties {
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

                    // Get target node
                    let target = match direction {
                        Direction::Outgoing => edge.target(),
                        Direction::Incoming => edge.source(),
                    };

                    // Skip if we've visited this node at this depth (prevent cycles at same depth)
                    let visit_key = (target, depth + 1);
                    if visited_at_depth.contains_key(&visit_key) {
                        continue;
                    }
                    visited_at_depth.insert(visit_key, true);

                    valid_targets.push((target, edge_data.connection_type));
                }
            }

            // Second pass: process valid targets with smart path management
            let new_depth = depth + 1;

            for (target, conn_type) in valid_targets {
                let needs_queue = new_depth < max_hops;

                let mut new_path = path.clone();
                new_path.push((target, conn_type));

                // If we're within the valid hop range and target matches node pattern, add to results
                if new_depth >= min_hops {
                    let node_matches = if edge_pattern.skip_target_type_check {
                        true
                    } else if let Some(ref node_type) = node_pattern.node_type {
                        self.graph
                            .graph
                            .node_type_of(target)
                            .map(|nt| nt == InternedKey::from_str(node_type))
                            .unwrap_or(false)
                    } else {
                        true
                    };

                    let props_match = if let Some(ref props) = node_pattern.properties {
                        self.node_matches_properties(target, props)
                    } else {
                        true
                    };

                    if node_matches && props_match {
                        // Create binding - clone path only if we also need it for queue
                        let path_for_binding = if needs_queue {
                            new_path.clone()
                        } else {
                            std::mem::take(&mut new_path)
                        };
                        let edge_binding = MatchBinding::VariableLengthPath {
                            source,
                            target,
                            hops: new_depth,
                            path: path_for_binding,
                        };
                        results.push((target, edge_binding));
                    }
                }

                // Continue exploring if we haven't reached max depth
                if needs_queue {
                    queue.push_back((target, new_depth, new_path));
                }
            }
        }

        Ok(results)
    }

    /// Convert a node to a binding.
    /// In lightweight mode (Cypher executor path), only `index` is populated
    /// since the executor resolves node data on demand via graph lookups.
    fn node_to_binding(&self, idx: NodeIndex) -> MatchBinding {
        if self.lightweight {
            return MatchBinding::NodeRef(idx);
        }
        if let Some(node) = self.graph.graph.node_weight(idx) {
            let node_title = node.title();
            let title_str = match &*node_title {
                Value::String(s) => s.clone(),
                Value::Int64(i) => i.to_string(),
                Value::Float64(f) => f.to_string(),
                Value::UniqueId(u) => u.to_string(),
                _ => format!("{:?}", &*node_title),
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

// ============================================================================
