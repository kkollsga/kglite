//! Hop expansion for [`PatternExecutor`] — start-node seeding, the per-hop
//! parallel and sequential expansion loops, and the small predicates they
//! share.
//!
//! Split out of `matcher.rs` to keep that file under the source-quality line
//! ceiling, matching `matcher_id_lookup_tests.rs`. These are inherent methods
//! on `PatternExecutor`, so the split is purely a file boundary.

use super::*;
use crate::graph::parallel::{self, ParallelInterrupt};

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

        // Try connection-type inverted index for untyped source nodes with typed edges.
        // Instead of iterating all 124M nodes hoping to find P31 sources, the inverted
        // index gives us exactly which nodes have P31 outgoing edges.
        let mut initial_nodes = if !first_is_prebound
            && has_edges
            && first_node.node_type.is_none()
            && first_node.properties.is_none()
        {
            // Check if the first edge has connection type(s) we can look up.
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
            // Check edge direction — inverted index only covers outgoing sources
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
                initial_nodes.truncate(cap);
            }
        }
        Ok(initial_nodes)
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
        let initial_nodes = self.seed_start_nodes(pattern, first_node, has_edges, source_cap)?;

        // Initialize matches with first node bindings.
        //
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

        // Track current node indices for each match
        let mut current_indices: Vec<NodeIndex> = initial_nodes;

        // Pre-allocate dedup set for distinct_target_var optimization
        let mut distinct_seen: HashSet<NodeIndex> = if self.distinct_target_var.is_some() {
            HashSet::with_capacity(current_indices.len())
        } else {
            HashSet::new()
        };

        // Process edge-node pairs
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

            // Internal binding names are invariant for the whole hop. Build
            // them once rather than formatting a fresh String for every
            // matched relationship on the expansion hot path.
            let hop = HopPlan {
                edge: edge_pattern,
                node: node_pattern,
                anonymous_path_var: (edge_pattern.variable.is_none()
                    && edge_pattern.needs_path_info
                    && edge_pattern.var_length.is_some())
                .then(|| format!("__anon_vlpath_{i}")),
                track_fixed_trail: edge_pattern.var_length.is_none()
                    && edge_pattern.needs_path_info,
                is_last_hop,
            };

            // Expand each current match.
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
                )?
            };

            // Check deadline / cancellation after expansion (covers both
            // parallel and sequential paths)
            if let Some(msg) = self.interrupt_reason() {
                return Err(msg);
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
        let results: Vec<(PatternMatch, NodeIndex)> = parallel::install(|| {
            matches
                .par_iter()
                .zip(current_indices.par_iter())
                .flat_map(|(current_match, &source_idx)| {
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
                    )) else {
                        return Vec::new();
                    };
                    expansions
                        .into_iter()
                        .filter_map(|(target_idx, edge_binding)| {
                            if reuses_bound_relationship(current_match, &edge_binding) {
                                return None;
                            }
                            if !self.target_satisfies_bindings(hop.node, current_match, target_idx)
                            {
                                return None;
                            }
                            Some((
                                self.extend_match(current_match, hop, edge_binding, target_idx),
                                target_idx,
                            ))
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        });
        // Propagate any error that occurred during parallel expansion
        interrupt.finish()?;
        // Apply distinct-target dedup for parallel results (the sequential
        // path does this inline, but the parallel path can't without
        // synchronization).
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
    ) -> Result<(Vec<PatternMatch>, Vec<NodeIndex>), String> {
        let mut new_matches = Vec::new();
        let mut new_indices = Vec::new();
        let mut expand_count: usize = 0;
        // At the last hop, enforce exact max_matches.
        // At intermediate hops, use a generous overcommit (50x) to avoid
        // expanding far more intermediates than needed while ensuring
        // enough survive to produce max_matches final results.
        let hop_limit = if hop.is_last_hop {
            self.max_matches
        } else {
            self.max_matches.map(|m| m.saturating_mul(50).max(1000))
        };
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
                // Stopping here is what the `zip` this replaced did.
                break;
            };
            let remaining = hop_limit.map(|max| max.saturating_sub(new_matches.len()));
            let hint = self.bound_target(hop.node, current_match);
            let expansions =
                self.expand_from_node(source_idx, hop.edge, hop.node, remaining, hint)?;
            for (target_idx, edge_binding) in expansions {
                if reuses_bound_relationship(current_match, &edge_binding) {
                    continue;
                }
                expand_count += 1;
                if expand_count.is_multiple_of(1024) {
                    if let Some(msg) = self.interrupt_reason() {
                        return Err(msg);
                    }
                }
                if hop_limit.is_some_and(|max| new_matches.len() >= max) {
                    break;
                }
                if !self.target_satisfies_bindings(hop.node, current_match, target_idx) {
                    continue;
                }
                // Distinct-target dedup: at the last hop, skip targets already seen
                if hop.is_last_hop {
                    if let Some(ref dtv) = self.distinct_target_var {
                        if hop.node.variable.as_deref() == Some(dtv.as_str())
                            && !distinct_seen.insert(target_idx)
                        {
                            continue;
                        }
                    }
                }
                new_matches.push(self.extend_match(current_match, hop, edge_binding, target_idx));
                new_indices.push(target_idx);
            }
        }
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

    /// `current_match` plus this hop's edge and target bindings.
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
