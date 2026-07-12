impl<'a> CypherExecutor<'a> {
    /// Fused OPTIONAL MATCH + WITH count() execution.
    /// Instead of expanding each input row into N matched rows then aggregating,
    /// count compatible matches directly per input row — O(N×degree) with zero
    /// intermediate row allocation.
    pub(super) fn execute_fused_optional_match_aggregate(
        &self,
        match_clause: &MatchClause,
        with_clause: &WithClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        if existing.rows.is_empty() {
            return Ok(existing);
        }

        // Items split into three buckets:
        // - group keys (Variable / PropertyAccess on pre-OPTIONAL var)
        // - pure count aggregates (`count(rp)` directly)
        // - derived expressions whose only aggregates are count() — e.g.
        //   `total - count(rp) AS cultural`. The fused operator computes
        //   count once per upstream row, substitutes it into each
        //   derived expression, and evaluates the result. Same row cost
        //   as the pure-count path; avoids the OPTIONAL MATCH expansion
        //   that the materialized executor would otherwise run.
        let mut group_key_indices = Vec::new();
        let mut count_items: Vec<(usize, &ReturnItem)> = Vec::new();
        let mut derived_items: Vec<(usize, &ReturnItem)> = Vec::new();

        for (i, item) in with_clause.items.iter().enumerate() {
            if is_aggregate_expression(&item.expression) {
                if matches!(
                    &item.expression,
                    Expression::FunctionCall { name, .. } if name == "count"
                ) {
                    count_items.push((i, item));
                } else {
                    derived_items.push((i, item));
                }
            } else {
                group_key_indices.push(i);
            }
        }

        let mut result_rows = Vec::with_capacity(existing.rows.len());

        for (scan_count, row) in existing.rows.iter().enumerate() {
            if scan_count.is_multiple_of(2048) {
                self.check_deadline()?;
            }
            // Count compatible matches for each pattern without materializing rows
            let mut match_count: i64 = 0;

            for pattern in &match_clause.patterns {
                // Fast-path: direct edge traversal when one end is pre-bound
                if let Some(fast_count) =
                    self.try_count_simple_pattern(pattern, &row.node_bindings)?
                {
                    match_count += fast_count;
                } else {
                    // Fall back to full PatternExecutor
                    let executor = PatternExecutor::with_bindings_and_params(
                        self.graph,
                        None,
                        &row.node_bindings,
                        self.params,
                    )
                    .set_deadline(self.deadline)
                    .set_cancel(self.cancel);
                    let matches = executor.execute(pattern)?;

                    for m in &matches {
                        if self.bindings_compatible(row, m) {
                            match_count += 1;
                        }
                    }
                }
            }

            // OPTIONAL MATCH semantics: an upstream row with zero pattern
            // matches still emits one null-padded row, so `count(*)` is
            // max(match_count, 1), while `count(var)` — over a variable
            // bound only by this OPTIONAL MATCH — counts non-null bindings,
            // which is exactly match_count.
            let star_count = match_count.max(1);

            // Build projected values for this row
            let mut projected = Bindings::with_capacity(
                group_key_indices.len() + count_items.len() + derived_items.len(),
            );

            // Group key pass-throughs
            for &idx in &group_key_indices {
                let item = &with_clause.items[idx];
                let key = return_item_column_name(item);
                let val = self.evaluate_expression(&item.expression, row)?;
                projected.insert(key, val);
            }

            // Derived expressions with embedded count() — substitute the
            // computed count into every count(...) sub-tree, then run
            // through the standard expression evaluator. The row's
            // projected bindings (e.g. `total` from a prior WITH) are
            // already in scope.
            for &(_, item) in &derived_items {
                let key = return_item_column_name(item);
                let substituted =
                    substitute_count_with_value(&item.expression, star_count, match_count);
                let val = self.evaluate_expression(&substituted, row)?;
                projected.insert(key, val);
            }

            // Count aggregates: count(*) vs count(var) per item (see above)
            for &(_, item) in &count_items {
                let key = return_item_column_name(item);
                let value = match &item.expression {
                    Expression::FunctionCall { args, .. } if count_call_is_star(args) => star_count,
                    _ => match_count,
                };
                projected.insert(key, Value::Int64(value));
            }

            // Create result row preserving bindings for group-key variables
            let mut new_row = ResultRow::from_projected(projected);
            for &idx in &group_key_indices {
                if let Expression::Variable(var) = &with_clause.items[idx].expression {
                    if let Some(&node_idx) = row.node_bindings.get(var) {
                        new_row.node_bindings.insert(var.clone(), node_idx);
                    }
                    if let Some(edge) = row.edge_bindings.get(var) {
                        new_row.edge_bindings.insert(var.clone(), *edge);
                    }
                    if let Some(path) = row.path_bindings.get(var) {
                        new_row.path_bindings.insert(var.clone(), path.clone());
                    }
                }
            }

            result_rows.push(new_row);
        }

        // Output columns come from this fused operator's own
        // WITH/RETURN items, not the upstream's. Earlier code
        // re-used `existing.columns`, which silently inherited the
        // pre-OPTIONAL columns and dropped the post-OPTIONAL ones —
        // visible as `KeyError` in Python clients reading by name.
        let columns: Vec<String> = with_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();
        let mut result = ResultSet {
            rows: result_rows,
            columns,
            lazy_return_items: None,
        };

        // Apply optional WHERE on the aggregated rows (e.g. WHERE cnt > 3)
        if let Some(ref where_clause) = with_clause.where_clause {
            result = self.execute_where(where_clause, result)?;
        }

        Ok(result)
    }

    /// Count all matches for a simple one- or two-hop pattern without
    /// materializing a `ResultRow` per path. The planner admits only a lone,
    /// non-distinct `count(*)` and patterns supported by the existing exact
    /// per-endpoint counters.
    fn execute_fused_global_pattern_count(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        return_clause: &ReturnClause,
    ) -> Result<ResultSet, String> {
        let last_elem_idx = pattern.elements.len() - 1;
        let first_var = match &pattern.elements[0] {
            PatternElement::Node(np) => np.variable.as_ref(),
            _ => None,
        };
        let last_var = match &pattern.elements[last_elem_idx] {
            PatternElement::Node(np) => np.variable.as_ref(),
            _ => None,
        };
        let (group_elem_idx, group_var) = if let Some(var) = first_var {
            (0, var)
        } else if let Some(var) = last_var {
            (last_elem_idx, var)
        } else {
            return Err("FusedMatchReturnAggregate: count pattern has no endpoint variable".into());
        };

        let group_only_pattern = crate::graph::core::pattern_matching::Pattern {
            elements: vec![pattern.elements[group_elem_idx].clone()],
        };
        let executor = PatternExecutor::new_lightweight_with_params(self.graph, None, self.params)
            .set_deadline(self.deadline)
            .set_cancel(self.cancel);
        let group_matches = executor.execute(&group_only_pattern)?;
        let mut total = 0i64;

        for (scan_count, matched) in group_matches.iter().enumerate() {
            if scan_count.is_multiple_of(2048) {
                self.check_deadline()?;
            }
            let Some(node_idx) = matched.bindings.iter().find_map(|(name, binding)| {
                if name != group_var {
                    return None;
                }
                match binding {
                    MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => Some(*index),
                    _ => None,
                }
            }) else {
                continue;
            };

            let count = if pattern.elements.len() == 5 {
                if group_elem_idx == 0 {
                    self.count_two_hop_pattern(pattern, node_idx)?
                } else {
                    self.count_two_hop_pattern_reverse(pattern, node_idx)?
                }
            } else {
                let mut bindings = Bindings::with_capacity(1);
                bindings.insert(group_var.clone(), node_idx);
                self.try_count_simple_pattern(pattern, &bindings)?.ok_or(
                    "FusedMatchReturnAggregate: unsupported count-only pattern",
                )?
            };
            total = total
                .checked_add(count)
                .ok_or("count(*) overflow while executing fused pattern count")?;
        }

        self.budget
            .check_work(usize::try_from(total).unwrap_or(usize::MAX), "fused pattern count")?;
        let item = &return_clause.items[0];
        let column = return_item_column_name(item);
        let mut projected = Bindings::with_capacity(1);
        projected.insert(column.clone(), Value::Int64(total));
        Ok(ResultSet {
            rows: vec![ResultRow::from_projected(projected)],
            columns: vec![column],
            lazy_return_items: None,
        })
    }

    /// Fused MATCH + RETURN with count() aggregation.
    /// Instead of materializing all (node, edge, node) rows and then grouping,
    /// match only the first-pattern nodes (group keys) and count edges directly.
    pub(super) fn execute_fused_match_return_aggregate(
        &self,
        match_clause: &MatchClause,
        return_clause: &ReturnClause,
        top_k: &Option<(usize, bool, usize)>,
        candidate_emit: &Option<(usize, bool, usize)>,
        distinct_count: bool,
        _existing: ResultSet,
    ) -> Result<ResultSet, String> {
        // The MATCH must have exactly 1 pattern with 3 or 5 elements (validated by planner)
        let pattern = &match_clause.patterns[0];

        if return_clause
            .items
            .iter()
            .all(|item| is_aggregate_expression(&item.expression))
        {
            return self.execute_fused_global_pattern_count(pattern, return_clause);
        }

        // Extract node variables from pattern
        let first_var = match &pattern.elements[0] {
            PatternElement::Node(np) => np.variable.as_ref(),
            _ => return Err("FusedMatchReturnAggregate: expected node pattern".into()),
        };
        let last_elem_idx = pattern.elements.len() - 1;
        let second_var = match &pattern.elements[last_elem_idx] {
            PatternElement::Node(np) => np.variable.as_ref(),
            _ => return Err("FusedMatchReturnAggregate: expected node pattern".into()),
        };

        // Determine which variable is the group key by checking RETURN items.
        // The planner guarantees all non-aggregate items reference the same variable.
        let group_var: &str = {
            let mut gv = None;
            for item in &return_clause.items {
                if !is_aggregate_expression(&item.expression) {
                    gv = match &item.expression {
                        Expression::PropertyAccess { variable, .. } => Some(variable.as_str()),
                        Expression::Variable(v) => Some(v.as_str()),
                        _ => None,
                    };
                    break;
                }
            }
            gv.ok_or("FusedMatchReturnAggregate: no group-by variable found")?
        };

        // Determine which pattern element index is the group key
        let group_elem_idx = if first_var.is_some_and(|v| v == group_var) {
            0
        } else if second_var.is_some_and(|v| v == group_var) {
            last_elem_idx
        } else {
            return Err("FusedMatchReturnAggregate: group variable not in pattern".into());
        };

        // Identify which RETURN items are group keys vs aggregates
        let mut group_key_indices = Vec::new();
        let mut count_indices = Vec::new();
        for (i, item) in return_clause.items.iter().enumerate() {
            if is_aggregate_expression(&item.expression) {
                count_indices.push(i);
            } else {
                group_key_indices.push(i);
            }
        }

        // Helper: extract node index from a match binding
        let extract_node_idx = |m: &crate::graph::core::pattern_matching::PatternMatch| -> Option<petgraph::graph::NodeIndex> {
            m.bindings.iter().find_map(|(name, binding)| {
                if name == group_var {
                    match binding {
                        MatchBinding::Node { index, .. } => Some(*index),
                        MatchBinding::NodeRef(index) => Some(*index),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        };

        // Helper: count edges (or distinct peers, when `distinct_count` is set)
        // for a node. Returns Result so the deadline surfaced by the inner
        // counters can propagate through the surrounding heap/loop and
        // terminate the query cleanly.
        let count_for_node = |node_idx: petgraph::graph::NodeIndex| -> Result<i64, String> {
            if pattern.elements.len() == 5 {
                // 5-element patterns aren't supported with DISTINCT yet; the
                // planner restricts `distinct_count` to 3-element patterns,
                // so this branch is non-distinct only.
                if group_elem_idx == 0 {
                    self.count_two_hop_pattern(pattern, node_idx)
                } else {
                    self.count_two_hop_pattern_reverse(pattern, node_idx)
                }
            } else {
                let mut bindings_for_count = Bindings::with_capacity(1);
                bindings_for_count.insert(group_var.to_string(), node_idx);
                if distinct_count {
                    Ok(self
                        .try_count_distinct_peers(pattern, &bindings_for_count)?
                        .unwrap_or(0))
                } else {
                    Ok(self
                        .try_count_simple_pattern(pattern, &bindings_for_count)?
                        .unwrap_or(0))
                }
            }
        };

        // Helper: build a result row for a (node_idx, count) pair
        let build_row =
            |node_idx: petgraph::graph::NodeIndex, match_count: i64| -> Result<ResultRow, String> {
                let mut tmp_row = ResultRow::new();
                tmp_row
                    .node_bindings
                    .insert(group_var.to_string(), node_idx);

                let mut projected = Bindings::with_capacity(return_clause.items.len());
                for &idx in &group_key_indices {
                    let item = &return_clause.items[idx];
                    let key = return_item_column_name(item);
                    let val = self.evaluate_expression(&item.expression, &tmp_row)?;
                    projected.insert(key, val);
                }
                for &idx in &count_indices {
                    let item = &return_clause.items[idx];
                    let key = return_item_column_name(item);
                    projected.insert(key, Value::Int64(match_count));
                }
                let mut new_row = ResultRow::from_projected(projected);
                new_row
                    .node_bindings
                    .insert(group_var.to_string(), node_idx);
                Ok(new_row)
            };

        let result_rows = if let Some(&(_, descending, limit)) = top_k.as_ref() {
            use std::cmp::Reverse;
            use std::collections::BinaryHeap;

            // Edge-centric aggregation: for 3-element patterns with a typed connection,
            // scan ALL edges of that type once and accumulate counts by peer. O(E_type)
            // sequential I/O instead of O(all_nodes × per_node_lookup).
            // This is critical for untyped group nodes (e.g., RETURN b.title, count(a))
            // where the node-centric path would iterate 124M nodes.
            //
            // Skipped when `distinct_count` is set: this path counts edges,
            // not distinct peers, and would overcount for any pattern with
            // multi-edges between the same pair.
            let edge_conn_type = match &pattern.elements[1] {
                PatternElement::Edge(ep) => ep.connection_type.as_ref(),
                _ => None,
            };
            let edge_direction = match &pattern.elements[1] {
                PatternElement::Edge(ep) => Some(ep.direction),
                _ => None,
            };
            let group_node_props = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(np) => &np.properties,
                _ => &None,
            };
            let group_node_type = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(np) => np.node_type.as_deref(),
                _ => None,
            };
            if let (false, 3, Some(ct_str), None) = (
                distinct_count,
                pattern.elements.len(),
                edge_conn_type,
                group_node_props.as_ref(),
            ) {
                let conn_key = InternedKey::from_str(ct_str);
                // Determine whether `group` is the SEMANTIC TARGET of the edge.
                // The persistent peer-count histogram is keyed by edge target,
                // so the fast path applies whenever group=target — regardless
                // of whether the planner reversed the pattern.
                //
                //   user wrote                  →  after `optimize_pattern_start_node`
                //   (a)-[:E]->(b:T)             →  (b:T)<-[:E]-(a)
                //   group_elem_idx = 2          →  group_elem_idx = 0
                //   edge.direction = Outgoing   →  edge.direction = Incoming
                //
                // In both shapes `b` is the semantic target. lookup_peer_counts
                // serves both. group=source (e.g. RETURN a, count(b)) needs a
                // different histogram; that case still falls back to slow path.
                let group_is_target = matches!(
                    (group_elem_idx, edge_direction),
                    (2, Some(EdgeDirection::Outgoing)) | (0, Some(EdgeDirection::Incoming))
                );

                if group_is_target {
                    self.check_deadline()?;
                    // Fast path: persistent per-(conn_type, peer) histogram
                    // answers in O(distinct-peers). Falls back to edge_endpoints
                    // scan for in-memory graphs and older disk graphs that
                    // lack the histogram.
                    let counts = if let Some(cached) = self.graph.graph.lookup_peer_counts(conn_key)
                    {
                        cached
                    } else {
                        self.graph.graph.count_edges_grouped_by_peer(
                            conn_key,
                            Direction::Outgoing,
                            self.deadline,
                        )?
                    };
                    // Optional per-peer type filter. When the group node carries
                    // a `:Type` label, restrict peers to that type via O(log n)
                    // binary search on `type_indices[T]` (sorted by construction
                    // — see `TypeNodesRef::binary_search_idx`). Pure CPU work;
                    // avoids the random mmap reads of `node_type_of` on disk-
                    // backed graphs that dominated the pre-fix wall time.
                    let type_index_view =
                        group_node_type.and_then(|nt| self.graph.type_indices.get(nt));
                    let peer_passes_type = |peer: u32| -> bool {
                        match &type_index_view {
                            None => true,
                            Some(view) => view
                                .binary_search_idx(petgraph::graph::NodeIndex::new(peer as usize)),
                        }
                    };
                    // Top-K from the counts HashMap
                    let heap: BinaryHeap<Reverse<(i64, u32)>> = if descending {
                        let mut h = BinaryHeap::with_capacity(limit + 1);
                        for (&peer, &count) in &counts {
                            if !peer_passes_type(peer) {
                                continue;
                            }
                            h.push(Reverse((count, peer)));
                            if h.len() > limit {
                                h.pop();
                            }
                        }
                        h
                    } else {
                        // For ASC we need a max-heap — use negative trick
                        let mut h = BinaryHeap::with_capacity(limit + 1);
                        for (&peer, &count) in &counts {
                            if !peer_passes_type(peer) {
                                continue;
                            }
                            h.push(Reverse((-count, peer)));
                            if h.len() > limit {
                                h.pop();
                            }
                        }
                        h
                    };

                    let top: Vec<_> = heap.into_sorted_vec();
                    let mut rows = Vec::with_capacity(top.len());
                    for Reverse((score, peer)) in &top {
                        let count = if descending { *score } else { -*score };
                        let node_idx = petgraph::graph::NodeIndex::new(*peer as usize);
                        rows.push(build_row(node_idx, count)?);
                    }
                    return Ok(ResultSet {
                        rows,
                        columns: return_clause
                            .items
                            .iter()
                            .map(return_item_column_name)
                            .collect(),
                        lazy_return_items: None,
                    });
                }

                // Group at SOURCE — semantic dual of the target case. The
                // persistent histogram is keyed by edge target so we can't
                // just look up; instead, do one sequential pass over
                // `for_each_edge_of_conn_type` (O(matching edges) on disk
                // via conn_type_index_*, NOT a full edge_endpoints scan) and
                // accumulate counts keyed by source. For Wikidata's typical
                // edge types (P166, P527, P57, ...) that's 200k–10M entries
                // — a couple hundred ms vs the 30s timeout the prior slow
                // node-centric path was hitting on `MATCH (h:human)-[:P166]
                // ->(award) ...`.
                let group_is_source = matches!(
                    (group_elem_idx, edge_direction),
                    (0, Some(EdgeDirection::Outgoing)) | (2, Some(EdgeDirection::Incoming))
                );
                if group_is_source {
                    self.check_deadline()?;
                    // No persistent source-keyed histogram exists, so we
                    // accept a sequential scan of edge_endpoints to build
                    // the equivalent on the fly. `count_edges_grouped_by_peer`
                    // with Direction::Incoming is the source-keyed dual of
                    // the target-keyed call, and it's already MADV_SEQUENTIAL
                    // tuned (~14s for Wikidata's 13.8 GB edge_endpoints,
                    // bounded by the deadline).
                    //
                    // The earlier-considered `for_each_edge_of_conn_type` path
                    // (using `conn_type_index_sources` + per-source CSR walks)
                    // is asymptotically O(distinct sources × log fan-out)
                    // but its random reads on cold mmap pages thrash the page
                    // cache — measured at >100s on the same query that the
                    // sequential variant runs in 14s. Sequential I/O wins
                    // even when total bytes are higher (see
                    // `feedback_disk_io_patterns.md`).
                    let counts = self.graph.graph.count_edges_grouped_by_peer(
                        conn_key,
                        Direction::Incoming,
                        self.deadline,
                    )?;
                    // Same per-source type filter as the target branch — sorted
                    // `type_indices[T]` + binary search.
                    let type_index_view =
                        group_node_type.and_then(|nt| self.graph.type_indices.get(nt));
                    let source_passes_type = |src: u32| -> bool {
                        match &type_index_view {
                            None => true,
                            Some(view) => view
                                .binary_search_idx(petgraph::graph::NodeIndex::new(src as usize)),
                        }
                    };
                    let heap: BinaryHeap<Reverse<(i64, u32)>> = if descending {
                        let mut h = BinaryHeap::with_capacity(limit + 1);
                        for (&src, &count) in &counts {
                            if !source_passes_type(src) {
                                continue;
                            }
                            h.push(Reverse((count, src)));
                            if h.len() > limit {
                                h.pop();
                            }
                        }
                        h
                    } else {
                        let mut h = BinaryHeap::with_capacity(limit + 1);
                        for (&src, &count) in &counts {
                            if !source_passes_type(src) {
                                continue;
                            }
                            h.push(Reverse((-count, src)));
                            if h.len() > limit {
                                h.pop();
                            }
                        }
                        h
                    };
                    let top: Vec<_> = heap.into_sorted_vec();
                    let mut rows = Vec::with_capacity(top.len());
                    for Reverse((score, src)) in &top {
                        let count = if descending { *score } else { -*score };
                        let node_idx = petgraph::graph::NodeIndex::new(*src as usize);
                        rows.push(build_row(node_idx, count)?);
                    }
                    return Ok(ResultSet {
                        rows,
                        columns: return_clause
                            .items
                            .iter()
                            .map(return_item_column_name)
                            .collect(),
                        lazy_return_items: None,
                    });
                }
            }

            // Node-centric top-K path (for typed group nodes or group=source patterns)
            // Get group node candidates directly from type_indices (streaming, no alloc)
            let group_node_type = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(np) => np.node_type.as_deref(),
                _ => None,
            };
            let group_node_props = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(np) => &np.properties,
                _ => &None,
            };
            let group_indices: Vec<petgraph::graph::NodeIndex> = if let Some(nt) = group_node_type {
                self.graph
                    .type_indices
                    .get(nt)
                    .map(|v| v.to_vec())
                    .unwrap_or_default()
            } else {
                {
                    let g = &self.graph.graph;
                    g.node_indices().collect()
                }
            };

            // Property filter executor (if group node has inline properties)
            let prop_executor = group_node_props.as_ref().map(|_| {
                PatternExecutor::new_lightweight_with_params(self.graph, None, self.params)
            });

            if descending {
                let mut heap: BinaryHeap<Reverse<(i64, petgraph::graph::NodeIndex)>> =
                    BinaryHeap::with_capacity(limit + 1);
                for (scan_count, &node_idx) in group_indices.iter().enumerate() {
                    if scan_count.is_multiple_of(10000) {
                        self.check_deadline()?;
                    }
                    // Property filter on group node
                    if let Some(ref props) = group_node_props {
                        if !prop_executor
                            .as_ref()
                            .expect(
                                "invariant: prop_executor is Some when group_node_props is Some",
                            )
                            .node_matches_properties_pub(node_idx, props)
                        {
                            continue;
                        }
                    }
                    let count = count_for_node(node_idx)?;
                    if count == 0 {
                        continue;
                    }
                    heap.push(Reverse((count, node_idx)));
                    if heap.len() > limit {
                        heap.pop();
                    }
                }
                let top: Vec<_> = heap
                    .into_sorted_vec()
                    .into_iter()
                    .map(|Reverse(x)| x)
                    .collect();
                let mut rows = Vec::with_capacity(top.len());
                for (count, node_idx) in top {
                    rows.push(build_row(node_idx, count)?);
                }
                rows
            } else {
                let mut heap: BinaryHeap<(i64, petgraph::graph::NodeIndex)> =
                    BinaryHeap::with_capacity(limit + 1);
                for (scan_count, &node_idx) in group_indices.iter().enumerate() {
                    if scan_count.is_multiple_of(10000) {
                        self.check_deadline()?;
                    }
                    if let Some(ref props) = group_node_props {
                        if !prop_executor
                            .as_ref()
                            .expect(
                                "invariant: prop_executor is Some when group_node_props is Some",
                            )
                            .node_matches_properties_pub(node_idx, props)
                        {
                            continue;
                        }
                    }
                    let count = count_for_node(node_idx)?;
                    if count == 0 {
                        continue;
                    }
                    heap.push((count, node_idx));
                    if heap.len() > limit {
                        heap.pop();
                    }
                }
                let top: Vec<_> = heap.into_sorted_vec();
                let mut rows = Vec::with_capacity(top.len());
                for (count, node_idx) in top {
                    rows.push(build_row(node_idx, count)?);
                }
                rows
            }
        } else {
            // Non-top-k: use edge-centric aggregation when the pattern is a
            // 3-element typed edge and the group key is the target node. This
            // replaces an O(|target-nodes| * avg-degree) per-node scan with a
            // single O(|edges-of-type|) sequential pass — essential when the
            // group variable has no type filter (124 M target candidates on
            // Wikidata would OOM or time out).
            let edge_conn_type = match &pattern.elements[1] {
                PatternElement::Edge(ep) => ep.connection_type.as_ref(),
                _ => None,
            };
            let edge_direction_nontopk = match &pattern.elements[1] {
                PatternElement::Edge(ep) => Some(ep.direction),
                _ => None,
            };
            let group_node_props_nontopk = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(np) => &np.properties,
                _ => &None,
            };
            let group_node_type_nontopk = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(np) => np.node_type.as_deref(),
                _ => None,
            };
            // Same direction-aware "group is target" predicate as the top-K
            // branch (see comment there for the post-reversal case). Pre-fix
            // this read `, 2` against group_elem_idx, which silently bailed
            // typed-target queries to the slow node-centric scan. The fast
            // path's `lookup_peer_counts` is keyed by edge target, so it
            // serves both AST shapes.
            let group_is_target_nontopk = matches!(
                (group_elem_idx, edge_direction_nontopk),
                (2, Some(EdgeDirection::Outgoing)) | (0, Some(EdgeDirection::Incoming))
            );
            let edge_centric_rows = if let (false, 3, Some(ct_str), None, true) = (
                distinct_count,
                pattern.elements.len(),
                edge_conn_type,
                group_node_props_nontopk.as_ref(),
                group_is_target_nontopk,
            ) {
                let conn_key = InternedKey::from_str(ct_str);
                self.check_deadline()?;
                // Fast path: persistent histogram. See matching comment at the
                // top-k branch.
                let counts = if let Some(cached) = self.graph.graph.lookup_peer_counts(conn_key) {
                    cached
                } else {
                    self.graph.graph.count_edges_grouped_by_peer(
                        conn_key,
                        Direction::Outgoing,
                        self.deadline,
                    )?
                };
                // Optional per-peer type filter — same shape as the top-K
                // branch. Drops peers whose type doesn't match the group
                // node's `:Type` label before any row is materialised; for
                // 295k peers + a downstream LIMIT this avoids 13s of
                // build_row work that would be thrown away.
                let type_index_view_nontopk =
                    group_node_type_nontopk.and_then(|nt| self.graph.type_indices.get(nt));
                let peer_passes_type_nontopk = |peer: u32| -> bool {
                    match &type_index_view_nontopk {
                        None => true,
                        Some(view) => {
                            view.binary_search_idx(petgraph::graph::NodeIndex::new(peer as usize))
                        }
                    }
                };

                // 0.8.12 phase-4: multi-key ORDER BY LIMIT was kept in the
                // pipeline (fusion set `candidate_emit` instead of
                // `top_k`). Trim via a heap on the primary key, grab the
                // threshold, then build rows only for entries whose
                // primary count is ≥ threshold. Downstream OrderBy +
                // Limit re-sort with the full multi-key spec and trim
                // to K. For P31-class-counts-shaped data this drops
                // `build_row` calls (each of which resolves `c.title`)
                // from O(distinct peers) to O(~K).
                let emit_rows: Vec<ResultRow> =
                    if let Some(&(_, descending, k)) = candidate_emit.as_ref() {
                        use std::cmp::Reverse;
                        use std::collections::BinaryHeap;
                        let threshold: i64 = if descending {
                            let mut h: BinaryHeap<Reverse<i64>> = BinaryHeap::with_capacity(k + 1);
                            for (&peer, &c) in &counts {
                                if !peer_passes_type_nontopk(peer) {
                                    continue;
                                }
                                h.push(Reverse(c));
                                if h.len() > k {
                                    h.pop();
                                }
                            }
                            h.peek().map(|Reverse(c)| *c).unwrap_or(i64::MIN)
                        } else {
                            let mut h: BinaryHeap<i64> = BinaryHeap::with_capacity(k + 1);
                            for (&peer, &c) in &counts {
                                if !peer_passes_type_nontopk(peer) {
                                    continue;
                                }
                                h.push(c);
                                if h.len() > k {
                                    h.pop();
                                }
                            }
                            h.peek().copied().unwrap_or(i64::MAX)
                        };
                        let mut rows = Vec::new();
                        for (&peer, &count) in &counts {
                            if !peer_passes_type_nontopk(peer) {
                                continue;
                            }
                            let keep = if descending {
                                count >= threshold
                            } else {
                                count <= threshold
                            };
                            if !keep {
                                continue;
                            }
                            self.check_deadline()?;
                            let node_idx = petgraph::graph::NodeIndex::new(peer as usize);
                            rows.push(build_row(node_idx, count)?);
                        }
                        rows
                    } else {
                        let mut rows = Vec::with_capacity(counts.len());
                        for (peer, count) in counts {
                            if !peer_passes_type_nontopk(peer) {
                                continue;
                            }
                            self.check_deadline()?;
                            let node_idx = petgraph::graph::NodeIndex::new(peer as usize);
                            rows.push(build_row(node_idx, count)?);
                        }
                        rows
                    };
                Some(emit_rows)
            } else {
                None
            };

            if let Some(rows) = edge_centric_rows {
                rows
            } else {
                // Node-centric fallback: the only path that actually needs
                // `group_matches`. Computing it earlier turned an untyped
                // group target (e.g. `c` in
                // `MATCH ()-[:P31]->(c) RETURN c.title, count(*) …`) into
                // a full-graph node scan ahead of the histogram fast path
                // — 14.7 M nodes on wiki1000m, ~3.5 s of work the fast
                // path never reads. Build it here, where it's used.
                let group_only_pattern = crate::graph::core::pattern_matching::Pattern {
                    elements: vec![pattern.elements[group_elem_idx].clone()],
                };
                let executor =
                    PatternExecutor::new_lightweight_with_params(self.graph, None, self.params)
                        .set_deadline(self.deadline)
                        .set_cancel(self.cancel);
                let group_matches = executor.execute(&group_only_pattern)?;
                let mut rows = Vec::with_capacity(group_matches.len());
                for (scan_count, m) in group_matches.iter().enumerate() {
                    if scan_count.is_multiple_of(2048) {
                        self.check_deadline()?;
                    }
                    let Some(node_idx) = extract_node_idx(m) else {
                        continue;
                    };
                    let match_count = count_for_node(node_idx)?;
                    // MATCH semantics: skip nodes with zero matching edges
                    if match_count == 0 {
                        continue;
                    }
                    rows.push(build_row(node_idx, match_count)?);
                }
                rows
            }
        };

        // Apply HAVING post-aggregation. Cheap: the row set is at most the
        // number of distinct group keys, which is bounded by the type/peer
        // cardinality (thousands to tens of thousands), not the edge count.
        let mut result_rows = result_rows;
        if let Some(ref having) = return_clause.having {
            augment_rows_with_aggregate_keys(&mut result_rows, &return_clause.items);
            result_rows.retain(|row| self.evaluate_predicate(having, row).unwrap_or(false));
        }

        let columns: Vec<String> = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();

        Ok(ResultSet {
            rows: result_rows,
            columns,
            lazy_return_items: None,
        })
    }

    /// Candidate node indices for a fused single-node scan, multi-label
    /// aware. Mirrors `PatternExecutor::find_matching_nodes`'s
    /// `needs_secondary_path`: unions the primary type bucket with the
    /// secondary-label index for the node type, then AND-intersects any
    /// extra labels (`MATCH (n:A:B)`). A `None` node type yields a full
    /// node scan. The choke-point API (`DirGraph::add_node_label`) forbids
    /// a node holding the same key as both primary and secondary, so the
    /// union is duplicate-free.
    ///
    /// Without this, the fused scan paths would miss secondary-only-labelled
    /// nodes (`MATCH (n:SecLabel)`) and over-include when extra labels are
    /// meant to narrow the match (`MATCH (n:Type:SecLabel)`).
    fn fused_scan_candidates(&self, node_pattern: &NodePattern) -> Vec<NodeIndex> {
        let Some(nt) = node_pattern.node_type.as_deref() else {
            return self.graph.graph.node_indices().collect();
        };
        let candidates = self.graph.nodes_with_label(nt);
        if node_pattern.extra_labels.is_empty() {
            return candidates;
        }
        let extra_keys: Vec<InternedKey> = node_pattern
            .extra_labels
            .iter()
            .map(|s| InternedKey::from_str(s))
            .collect();
        candidates
            .into_iter()
            .filter(|&idx| {
                extra_keys
                    .iter()
                    .all(|&k| self.graph.node_has_label(idx, k))
            })
            .collect()
    }

    /// Fused MATCH (n:Type) [WHERE ...] RETURN group_keys, agg_funcs(...)
    /// Single-pass node scan: iterates nodes directly, evaluates group keys
    /// and aggregates without creating intermediate ResultRows.
    pub(super) fn execute_fused_node_scan_aggregate(
        &self,
        match_clause: &MatchClause,
        where_predicate: Option<&Predicate>,
        return_clause: &ReturnClause,
    ) -> Result<ResultSet, String> {
        use crate::graph::core::pattern_matching::PatternElement;

        // Extract node variable and type from the single-element pattern
        let pattern = &match_clause.patterns[0];
        let node_pattern = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return Err("FusedNodeScanAggregate: expected node pattern".into()),
        };
        let node_var = node_pattern.variable.as_deref().unwrap_or("_n");

        // Get candidate node indices (multi-label aware).
        let node_indices = self.fused_scan_candidates(node_pattern);

        // Classify RETURN items into group keys and aggregates
        let mut group_key_indices = Vec::new();
        let mut agg_indices = Vec::new();
        for (i, item) in return_clause.items.iter().enumerate() {
            if is_aggregate_expression(&item.expression) {
                agg_indices.push(i);
            } else {
                group_key_indices.push(i);
            }
        }

        // Pre-fold group key and aggregate expressions
        let folded_group_exprs: Vec<Expression> = group_key_indices
            .iter()
            .map(|&i| self.fold_constants_expr(&return_clause.items[i].expression))
            .collect();

        // Which aggregates are count(DISTINCT …) — tracked per group via a value
        // set rather than a running count.
        let agg_is_distinct: Vec<bool> = agg_indices
            .iter()
            .map(|&i| {
                matches!(&return_clause.items[i].expression,
                    Expression::FunctionCall { name, distinct: true, .. }
                        if name.eq_ignore_ascii_case("count"))
            })
            .collect();

        // Pre-fold WHERE predicate once (converts In → InLiteralSet with HashSet, etc.)
        let folded_where = where_predicate.map(|p| self.fold_constants_pred(p));
        let folded_where_ref = folded_where.as_ref();

        // Single-pass: iterate nodes, evaluate group keys, update accumulators
        // Use a single reusable ResultRow to avoid per-node allocation
        let mut eval_row = ResultRow::new();
        eval_row
            .node_bindings
            .insert(node_var.to_string(), petgraph::graph::NodeIndex::new(0));

        // Create PatternExecutor once for property matching (if needed)
        let pattern_executor = if node_pattern.properties.is_some() {
            Some(PatternExecutor::new_lightweight_with_params(
                self.graph,
                None,
                self.params,
            ))
        } else {
            None
        };

        // Inline accumulators for aggregation during scan
        struct InlineAccumulators {
            counts: Vec<i64>,
            sums: Vec<f64>,
            mins: Vec<Option<Value>>,
            maxs: Vec<Option<Value>>,
            // Per-agg value set for count(DISTINCT …); None for non-distinct aggs.
            distinct_sets: Vec<Option<FxHashSet<Value>>>,
        }

        // Groups: (group_key_values, first_node_idx_for_binding)
        let mut groups: Vec<(Vec<Value>, petgraph::graph::NodeIndex)> = Vec::new();
        let mut group_accumulators: Vec<InlineAccumulators> = Vec::new();
        let mut group_index_map: FxHashMap<Vec<Value>, usize> = FxHashMap::default();

        // Perf: reusable per-row scratch buffers — avoids a heap allocation per
        // passing row for the group key and aggregate-input vectors (the inner
        // loop's dominant cost on scan-heavy filters/aggregates).
        let mut key_values: Vec<Value> = Vec::with_capacity(folded_group_exprs.len());
        let mut agg_vals: Vec<Value> = Vec::with_capacity(agg_indices.len());

        for (scan_count, &node_idx) in node_indices.iter().enumerate() {
            // Timeout check every 10,000 iterations (matches fused_match_return pattern)
            if scan_count % 10_000 == 0 && scan_count > 0 {
                self.check_deadline()?;
            }
            // Check pattern properties using PatternExecutor's matching logic
            if let Some(ref props) = node_pattern.properties {
                if !pattern_executor
                    .as_ref()
                    .expect("invariant: pattern_executor is Some when node has property filters")
                    .node_matches_properties_pub(node_idx, props)
                {
                    continue;
                }
            }

            // Set the node binding for expression evaluation
            *eval_row
                .node_bindings
                .get_mut(node_var)
                .expect("invariant: node_var binding inserted upstream by pattern match") =
                node_idx;

            // Check WHERE predicate. Use the compiled fast path when available
            // (pre-resolved accessors, no per-row interpreter overhead);
            // otherwise fall back to the generic evaluator.
            if let Some(pred) = folded_where_ref {
                if !self.evaluate_predicate(pred, &eval_row).unwrap_or(false) {
                    continue;
                }
            }

            // Evaluate group key (reuse the scratch buffer — no per-row alloc)
            key_values.clear();
            for expr in &folded_group_exprs {
                key_values.push(
                    self.evaluate_expression(expr, &eval_row)
                        .unwrap_or(Value::Null),
                );
            }

            // Evaluate all aggregate expressions for this node (reuse buffer)
            agg_vals.clear();
            for &ai in &agg_indices {
                let item = &return_clause.items[ai];
                let v = match &item.expression {
                    Expression::FunctionCall {
                        name,
                        args,
                        distinct,
                    } => {
                        if args.is_empty() || matches!(args[0], Expression::Star) {
                            Value::Boolean(true) // count(*) marker — always counted
                        } else if !*distinct
                            && name.eq_ignore_ascii_case("count")
                            && matches!(&args[0], Expression::Variable(v)
                                if eval_row.node_bindings.get(v).is_some()
                                    || eval_row.edge_bindings.get(v).is_some())
                        {
                            // count(n) over a bound node/edge variable: the binding
                            // is always present, so this is equivalent to count(*).
                            // Avoid materializing the full node Value (every property
                            // cloned into a BTreeMap) per row just to test non-null.
                            Value::Boolean(true)
                        } else {
                            self.evaluate_expression(&args[0], &eval_row)
                                .unwrap_or(Value::Null)
                        }
                    }
                    _ => self
                        .evaluate_expression(&item.expression, &eval_row)
                        .unwrap_or(Value::Null),
                };
                agg_vals.push(v);
            }

            if let Some(&group_idx) = group_index_map.get(&key_values) {
                // Update accumulators
                let acc = &mut group_accumulators[group_idx];
                for (ai, _) in agg_indices.iter().enumerate() {
                    let val = &agg_vals[ai];
                    // count(DISTINCT …): dedup non-null values in the per-agg set.
                    if agg_is_distinct[ai] {
                        if !matches!(val, Value::Null) {
                            acc.distinct_sets[ai]
                                .get_or_insert_with(FxHashSet::default)
                                .insert(val.clone());
                        }
                        continue;
                    }
                    // Only count non-null values (count(*) uses Boolean marker)
                    if !matches!(val, Value::Null) {
                        acc.counts[ai] += 1;
                    }
                    if let Some(f) = value_to_f64(val) {
                        acc.sums[ai] += f;
                    }
                    if !matches!(val, Value::Null) {
                        // Phase A.2 / C4 — short-circuit on is_none()
                        // guarantees the unwrap can't fire, but the
                        // .expect() makes the invariant explicit if a
                        // future refactor reorders the conditions.
                        if acc.mins[ai].is_none()
                            || crate::graph::core::filtering::compare_values(
                                val,
                                acc.mins[ai]
                                    .as_ref()
                                    .expect("invariant: is_none() short-circuited above"),
                            ) == Some(std::cmp::Ordering::Less)
                        {
                            acc.mins[ai] = Some(val.clone());
                        }
                        if acc.maxs[ai].is_none()
                            || crate::graph::core::filtering::compare_values(
                                val,
                                acc.maxs[ai]
                                    .as_ref()
                                    .expect("invariant: is_none() short-circuited above"),
                            ) == Some(std::cmp::Ordering::Greater)
                        {
                            acc.maxs[ai] = Some(val.clone());
                        }
                    }
                }
            } else {
                let group_idx = groups.len();
                group_index_map.insert(key_values.clone(), group_idx);
                groups.push((key_values.clone(), node_idx));

                // Initialize accumulators
                let na = agg_indices.len();
                let mut acc = InlineAccumulators {
                    counts: vec![0i64; na],
                    sums: vec![0.0f64; na],
                    mins: vec![None; na],
                    maxs: vec![None; na],
                    distinct_sets: agg_is_distinct
                        .iter()
                        .map(|&d| if d { Some(FxHashSet::default()) } else { None })
                        .collect(),
                };
                for (ai, _) in agg_indices.iter().enumerate() {
                    let val = &agg_vals[ai];
                    if agg_is_distinct[ai] {
                        if !matches!(val, Value::Null) {
                            acc.distinct_sets[ai]
                                .get_or_insert_with(FxHashSet::default)
                                .insert(val.clone());
                        }
                        continue;
                    }
                    if !matches!(val, Value::Null) {
                        acc.counts[ai] = 1;
                        if let Some(f) = value_to_f64(val) {
                            acc.sums[ai] = f;
                        }
                        acc.mins[ai] = Some(val.clone());
                        acc.maxs[ai] = Some(val.clone());
                    }
                }
                group_accumulators.push(acc);
            }
        }

        // Build result rows from groups
        let columns: Vec<String> = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();

        // Handle empty-set aggregation: pure aggregation with no group keys
        // and no matching nodes should return one row with defaults (count=0, sum=0, etc.)
        if groups.is_empty() && group_key_indices.is_empty() {
            let empty_rows: Vec<&ResultRow> = Vec::new();
            let mut projected = Bindings::with_capacity(return_clause.items.len());
            for &item_idx in &agg_indices {
                let item = &return_clause.items[item_idx];
                let key = return_item_column_name(item);
                let val = self.evaluate_aggregate_with_rows(&item.expression, &empty_rows)?;
                projected.insert(key, val);
            }
            return Ok(ResultSet {
                rows: vec![ResultRow::from_projected(projected)],
                columns,
                lazy_return_items: None,
            });
        }

        let mut result_rows = Vec::with_capacity(groups.len());

        for (gi, (group_key_values, first_node_idx)) in groups.iter().enumerate() {
            let mut projected = Bindings::with_capacity(return_clause.items.len());

            // Add group key values
            for (ki, &item_idx) in group_key_indices.iter().enumerate() {
                let key = return_item_column_name(&return_clause.items[item_idx]);
                projected.insert(key, group_key_values[ki].clone());
            }

            // Emit aggregate values from accumulators
            let acc = &group_accumulators[gi];
            for (ai, &item_idx) in agg_indices.iter().enumerate() {
                let item = &return_clause.items[item_idx];
                let key = return_item_column_name(item);
                let val = match &item.expression {
                    Expression::FunctionCall {
                        name,
                        args,
                        distinct,
                    } => {
                        if *distinct {
                            // count(DISTINCT …): number of distinct non-null values.
                            Value::Int64(
                                acc.distinct_sets[ai]
                                    .as_ref()
                                    .map(|s| s.len() as i64)
                                    .unwrap_or(0),
                            )
                        } else {
                            match name.as_str() {
                                "count" => Value::Int64(acc.counts[ai]),
                                "sum" => {
                                    if acc.counts[ai] == 0 {
                                        Value::Int64(0)
                                    } else {
                                        // Check if input is integer-typed
                                        let is_int = acc.mins[ai].as_ref().is_some_and(|v| {
                                            matches!(v, Value::Int64(_) | Value::UniqueId(_))
                                        });
                                        if is_int {
                                            Value::Int64(acc.sums[ai] as i64)
                                        } else {
                                            Value::Float64(acc.sums[ai])
                                        }
                                    }
                                }
                                "avg" | "mean" | "average" => {
                                    if acc.counts[ai] == 0 {
                                        Value::Null
                                    } else {
                                        Value::Float64(acc.sums[ai] / acc.counts[ai] as f64)
                                    }
                                }
                                "min" => acc.mins[ai].clone().unwrap_or(Value::Null),
                                "max" => acc.maxs[ai].clone().unwrap_or(Value::Null),
                                _ => {
                                    // Unsupported aggregate — fall back to evaluate
                                    let mut tmp_row = ResultRow::new();
                                    tmp_row
                                        .node_bindings
                                        .insert(node_var.to_string(), *first_node_idx);
                                    self.evaluate_expression(&args[0], &tmp_row)?
                                }
                            }
                        }
                    }
                    _ => Value::Null,
                };
                projected.insert(key, val);
            }

            let mut row = ResultRow::from_projected(projected);
            row.node_bindings
                .insert(node_var.to_string(), *first_node_idx);
            result_rows.push(row);
        }

        // Handle HAVING
        if let Some(ref having) = return_clause.having {
            augment_rows_with_aggregate_keys(&mut result_rows, &return_clause.items);
            result_rows.retain(|row| self.evaluate_predicate(having, row).unwrap_or(false));
        }

        // Handle DISTINCT
        if return_clause.distinct {
            let mut seen = HashSet::new();
            result_rows.retain(|row| {
                let key: Vec<Value> = columns
                    .iter()
                    .map(|c| row.projected.get(c).cloned().unwrap_or(Value::Null))
                    .collect();
                seen.insert(key)
            });
        }

        Ok(ResultSet {
            rows: result_rows,
            columns,
            lazy_return_items: None,
        })
    }

    /// Fused MATCH (n:Type) [WHERE] RETURN expressions ORDER BY expr LIMIT k.
    /// Single-pass scan: iterates nodes, evaluates sort key per node, maintains
    /// K-element top-K via sorted Vec (insertion sort). RETURN expressions are
    /// only evaluated for the K winners. Avoids materializing all rows.
    pub(super) fn execute_fused_node_scan_top_k(
        &self,
        match_clause: &MatchClause,
        where_predicate: Option<&Predicate>,
        return_clause: &ReturnClause,
        sort_expression: &Expression,
        descending: bool,
        limit: usize,
    ) -> Result<ResultSet, String> {
        use crate::graph::core::pattern_matching::PatternElement;

        let pattern = &match_clause.patterns[0];
        let node_pattern = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return Err("FusedNodeScanTopK: expected node pattern".into()),
        };
        let node_var = node_pattern.variable.as_deref().unwrap_or("_n");

        // Get candidate node indices (multi-label aware).
        let node_indices = self.fused_scan_candidates(node_pattern);

        // Pattern property filter
        let pattern_executor = if node_pattern.properties.is_some() {
            Some(PatternExecutor::new_lightweight_with_params(
                self.graph,
                None,
                self.params,
            ))
        } else {
            None
        };

        // Pre-fold expressions
        let folded_sort = self.fold_constants_expr(sort_expression);
        let folded_where = where_predicate.map(|p| self.fold_constants_pred(p));
        let folded_where_ref = folded_where.as_ref();

        // Single reusable eval row
        let mut eval_row = ResultRow::new();
        eval_row
            .node_bindings
            .insert(node_var.to_string(), petgraph::graph::NodeIndex::new(0));

        // Top-K: sorted Vec of (sort_value, node_idx). Insertion sort for small K.
        let mut top_k: Vec<(Value, petgraph::graph::NodeIndex)> = Vec::with_capacity(limit + 1);

        for (scan_count, &node_idx) in node_indices.iter().enumerate() {
            // Periodic deadline check
            if scan_count.is_multiple_of(10000) {
                self.check_deadline()?;
            }

            // Pattern property filter
            if let Some(ref props) = node_pattern.properties {
                if !pattern_executor
                    .as_ref()
                    .expect("invariant: pattern_executor is Some when node has property filters")
                    .node_matches_properties_pub(node_idx, props)
                {
                    continue;
                }
            }

            // Set node binding for expression evaluation
            *eval_row
                .node_bindings
                .get_mut(node_var)
                .expect("invariant: node_var binding inserted upstream by pattern match") =
                node_idx;

            // WHERE filter
            if let Some(pred) = folded_where_ref {
                if !self.evaluate_predicate(pred, &eval_row).unwrap_or(false) {
                    continue;
                }
            }

            // Evaluate sort key
            let sort_val = self.evaluate_expression(&folded_sort, &eval_row)?;
            if matches!(sort_val, Value::Null) {
                continue;
            }

            // Insert into top-K sorted Vec
            let pos = if descending {
                top_k.partition_point(|(existing, _)| {
                    crate::graph::core::filtering::compare_values(existing, &sort_val)
                        .is_some_and(|o| o != std::cmp::Ordering::Less)
                })
            } else {
                top_k.partition_point(|(existing, _)| {
                    crate::graph::core::filtering::compare_values(existing, &sort_val)
                        .is_some_and(|o| o != std::cmp::Ordering::Greater)
                })
            };
            if pos < limit {
                top_k.insert(pos, (sort_val, node_idx));
                if top_k.len() > limit {
                    top_k.pop();
                }
            }
        }

        // Build RETURN expressions only for the K winners
        let folded_return_exprs: Vec<Expression> = return_clause
            .items
            .iter()
            .map(|item| self.fold_constants_expr(&item.expression))
            .collect();
        let columns: Vec<String> = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();

        let mut result_rows = Vec::with_capacity(top_k.len());
        for (_, winner_idx) in &top_k {
            *eval_row
                .node_bindings
                .get_mut(node_var)
                .expect("invariant: node_var binding inserted upstream") = *winner_idx;
            let mut projected = Bindings::with_capacity(columns.len());
            for (j, expr) in folded_return_exprs.iter().enumerate() {
                let val = self.evaluate_expression(expr, &eval_row)?;
                projected.insert(columns[j].clone(), val);
            }
            result_rows.push(ResultRow::from_projected(projected));
        }

        Ok(ResultSet {
            rows: result_rows,
            columns,
            lazy_return_items: None,
        })
    }

    /// Fused MATCH + WITH count() — same as `execute_fused_match_return_aggregate`
    /// but produces ResultSet for pipeline continuation (WITH semantics).
    ///
    /// When `secondary_match` is `Some`, the planner has folded a second
    /// adjacent MATCH whose edge variable is consumed only by the WITH's
    /// count(). The primary `match_clause` enumerates group keys (via the
    /// fully-executed pattern, so its filters apply); the secondary clause's
    /// pattern provides the count shape (edge type/direction/target filter).
    /// Per group key the executor calls `try_count_simple_pattern` against
    /// the secondary pattern, which uses the existing degree-fast-path
    /// (count_edges_filtered) without materializing edge rows.
    pub(super) fn execute_fused_match_with_aggregate(
        &self,
        match_clause: &MatchClause,
        with_clause: &WithClause,
        secondary_match: Option<&MatchClause>,
        top_k: Option<&AggregateTopK>,
        distinct_count: bool,
        _existing: ResultSet,
    ) -> Result<ResultSet, String> {
        let pattern = &match_clause.patterns[0];

        let first_var = match &pattern.elements[0] {
            PatternElement::Node(np) => np.variable.as_ref(),
            _ => return Err("FusedMatchWithAggregate: expected node pattern".into()),
        };
        let second_var = match &pattern.elements[2] {
            PatternElement::Node(np) => np.variable.as_ref(),
            _ => return Err("FusedMatchWithAggregate: expected node pattern".into()),
        };

        // Determine which variable is the group key. The non-aggregate
        // items in the WITH project either the group variable directly
        // (`w`) or one of its properties (`w.name`); both shapes resolve
        // to the same group key for our purposes.
        let group_var: &str = {
            let mut gv = None;
            for item in &with_clause.items {
                if !is_aggregate_expression(&item.expression) {
                    match &item.expression {
                        Expression::Variable(v) => {
                            gv = Some(v.as_str());
                            break;
                        }
                        Expression::PropertyAccess { variable, .. } => {
                            gv = Some(variable.as_str());
                            break;
                        }
                        _ => {}
                    }
                }
            }
            gv.ok_or("FusedMatchWithAggregate: no group-by variable found")?
        };

        let group_elem_idx = if first_var.is_some_and(|v| v == group_var) {
            0
        } else if second_var.is_some_and(|v| v == group_var) {
            2
        } else {
            return Err("FusedMatchWithAggregate: group variable not in pattern".into());
        };

        // Identify group key and count items
        let mut group_key_indices = Vec::new();
        let mut count_indices = Vec::new();
        for (i, item) in with_clause.items.iter().enumerate() {
            if is_aggregate_expression(&item.expression) {
                count_indices.push(i);
            } else {
                group_key_indices.push(i);
            }
        }

        let columns: Vec<String> = with_clause
            .items
            .iter()
            .map(|item| {
                item.alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}", item.expression))
            })
            .collect();

        // 0.8.12 phase-3: edge-centric aggregation via peer_count_histogram.
        // Pattern must be 3 elements, group on the target (element 2),
        // target has no property constraints, edge has a typed connection.
        // Source may have a node-type constraint if a cheap uniformity
        // check proves every source of the edge type already has that
        // type. For wiki-style queries like
        //   MATCH (h:Q5)-[:P27]->(c) WITH c, count(h) AS k
        // this drops wall time from O(|tgt nodes| × avg in-degree) to
        // O(|distinct peers|) by consulting the pre-built histogram.
        //
        // The fast path is tried BEFORE computing `group_matches`
        // because `group_matches = executor.execute(&MATCH (c))` for
        // an untyped group target scans every node in the graph — on
        // wiki1000m that's a 14.7 M-node full scan (~3 s) that the
        // histogram path never looks at. Running it only when the
        // slow path actually fires cuts `WITH P27 count` from 5.4 s
        // to under 500 ms at 1 B triples.
        // Histogram fast path only applies to the single-MATCH shape — the
        // two-MATCH variant has a separate pattern driving the count, so the
        // histogram (keyed on M1's edge type) doesn't answer the right
        // question. Skip it when secondary_match is set, and skip it for
        // distinct counts (the histogram counts edges, not distinct peers).
        if secondary_match.is_none() && !distinct_count {
            if let Some(rows) = self.try_fast_with_aggregate_via_histogram(
                pattern,
                with_clause,
                &columns,
                group_var,
                group_elem_idx,
                &group_key_indices,
                &count_indices,
            )? {
                return Ok(ResultSet {
                    rows,
                    columns,
                    lazy_return_items: None,
                });
            }
        }

        // Fast path didn't apply (non-disk backend, unsupported pattern
        // shape, two-MATCH fusion, etc.). Now enumerate group keys for the
        // fall-back aggregation.
        //
        // Single-MATCH case: the group node is one end of M1's edge —
        // execute just that node pattern. Counts via M1's full pattern
        // filter out non-edge-target nodes downstream (count == 0 → skip).
        //
        // Two-MATCH case: M1 carries the constraint that defines the group
        // key set (e.g. `(w)-[:P106]->({nid:'Q36180'})` only matches w's
        // that are writers). Execute M1 fully so its filters apply. The
        // count then runs against M2's pattern, which is anchored on the
        // shared variable per group key.
        let executor = PatternExecutor::new_lightweight_with_params(self.graph, None, self.params)
            .set_deadline(self.deadline)
            .set_cancel(self.cancel);
        let count_pattern: &crate::graph::core::pattern_matching::Pattern =
            if let Some(m2) = secondary_match {
                &m2.patterns[0]
            } else {
                pattern
            };
        let group_matches = if secondary_match.is_some() {
            executor.execute(pattern)?
        } else {
            let group_only_pattern = crate::graph::core::pattern_matching::Pattern {
                elements: vec![pattern.elements[group_elem_idx].clone()],
            };
            executor.execute(&group_only_pattern)?
        };

        // Phase 1 — sequential: extract distinct group-key NodeIndices from
        // the match set. Dedup applies to the two-MATCH path because M1's
        // full execution can yield duplicate `w` bindings (one per edge
        // satisfying M1's constraint). The single-MATCH path's
        // group_only_pattern already produces unique nodes.
        let mut group_keys: Vec<NodeIndex> = Vec::with_capacity(group_matches.len());
        let mut seen_group_keys: HashSet<NodeIndex> = HashSet::new();
        for m in &group_matches {
            let node_idx = m.bindings.iter().find_map(|(name, binding)| {
                if name == group_var {
                    match binding {
                        MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => {
                            Some(*index)
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            });
            let Some(node_idx) = node_idx else {
                continue;
            };
            if secondary_match.is_some() && !seen_group_keys.insert(node_idx) {
                continue;
            }
            group_keys.push(node_idx);
        }

        // Phase 2 — parallel: degree count per group key. Each
        // try_count_simple_pattern call is read-only against the graph and
        // independent of every other call, so rayon over many keys
        // overlaps the per-call mmap reads instead of serialising them.
        // Sequential fallback for small group sets so rayon overhead doesn't
        // dominate.
        const PARALLEL_COUNT_THRESHOLD: usize = 4_096;
        let group_var_owned = group_var.to_string();
        let count_one = |idx: NodeIndex| -> Result<(NodeIndex, i64), String> {
            let mut bindings = Bindings::with_capacity(1);
            bindings.insert(group_var_owned.clone(), idx);
            let c = if distinct_count {
                self.try_count_distinct_peers(count_pattern, &bindings)?
                    .unwrap_or(0)
            } else {
                self.try_count_simple_pattern(count_pattern, &bindings)?
                    .unwrap_or(0)
            };
            Ok((idx, c))
        };
        let counts: Vec<(NodeIndex, i64)> = if group_keys.len() >= PARALLEL_COUNT_THRESHOLD {
            group_keys
                .par_iter()
                .map(|&idx| count_one(idx))
                .collect::<Result<_, _>>()?
        } else {
            let mut sequential = Vec::with_capacity(group_keys.len());
            for &idx in &group_keys {
                sequential.push(count_one(idx)?);
            }
            sequential
        };

        // Phase 2.5 — when the planner absorbed a downstream `ORDER BY
        // <count_alias> {DESC|ASC} LIMIT k`, trim the count vec to the K
        // winners *before* row construction. Property-evaluation per row
        // is the tail cost (each `evaluate_expression` does a few mmap
        // reads); skipping it for non-winners is the whole point of
        // pushing the top-K hint into the fused stage.
        let counts: Vec<(NodeIndex, i64)> = if let Some(tk) = top_k {
            let mut filtered: Vec<(NodeIndex, i64)> =
                counts.into_iter().filter(|&(_, c)| c > 0).collect();
            if tk.descending {
                filtered.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));
            } else {
                filtered.sort_unstable_by_key(|a| a.1);
            }
            filtered.truncate(tk.limit);
            filtered
        } else {
            counts
        };

        // Phase 3 — sequential: project group keys + counts into result rows.
        // Row construction uses the executor's expression evaluator which
        // isn't trivially parallelisable; the per-row work is tiny next to
        // the count phase, so leaving it sequential is fine.
        let mut result_rows = Vec::with_capacity(counts.len());

        for (node_idx, match_count) in counts {
            // Skip nodes with 0 matches (MATCH semantics — no outer join)
            if match_count == 0 {
                continue;
            }

            // Build a temporary row for evaluating group-key expressions
            let mut tmp_row = ResultRow::new();
            tmp_row
                .node_bindings
                .insert(group_var.to_string(), node_idx);

            let mut projected = Bindings::with_capacity(with_clause.items.len());

            for &idx in &group_key_indices {
                let item = &with_clause.items[idx];
                let key = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}", item.expression));
                let val = self.evaluate_expression(&item.expression, &tmp_row)?;
                projected.insert(key, val);
            }

            for &idx in &count_indices {
                let item = &with_clause.items[idx];
                let key = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}", item.expression));
                projected.insert(key, Value::Int64(match_count));
            }

            let mut new_row = ResultRow::from_projected(projected);
            new_row
                .node_bindings
                .insert(group_var.to_string(), node_idx);
            result_rows.push(new_row);
        }

        // Apply WITH WHERE filter if present
        if let Some(ref where_clause) = with_clause.where_clause {
            let folded = self.fold_constants_pred(&where_clause.predicate);
            result_rows.retain(|row| self.evaluate_predicate(&folded, row).unwrap_or(false));
        }

        Ok(ResultSet {
            rows: result_rows,
            columns,
            lazy_return_items: None,
        })
    }

    /// 0.8.12 phase-3 fast path for
    ///   `MATCH (src [:Type])-[:T]->(tgt) WITH tgt, count(src) [AS k] ...`
    /// — answers in O(|distinct peers|) via the `peer_count_histogram`
    /// instead of the per-source iteration that the generic path takes.
    /// Returns `Ok(None)` when the pattern shape, the target
    /// constraints, or the histogram availability make this path unsafe
    /// — caller then uses the per-source iteration.
    ///
    /// Preconditions for the fast path:
    ///   1. Pattern is exactly 3 elements: node, edge, node.
    ///   2. Group variable is the *target* (element index 2).
    ///   3. Edge has a connection type (`[:T]`) — required to look up
    ///      the histogram at all.
    ///   4. Target element has no property constraints (`{…}`) — the
    ///      histogram counts every peer, so an added property filter
    ///      would require post-filter which defeats the point.
    ///   5. Source's type constraint (if any) is a no-op on this edge
    ///      type: every node in `sources_for_conn_type_bounded(T)` has
    ///      the constrained type. Otherwise using the unfiltered
    ///      histogram would overcount.
    ///
    /// Histogram fallback isn't implemented here — when
    /// `lookup_peer_counts` returns `None` (memory / mapped backends,
    /// or older disk graphs) we return `Ok(None)` so the caller takes
    /// the per-source path.
    #[allow(clippy::too_many_arguments)]
    fn try_fast_with_aggregate_via_histogram(
        &self,
        pattern: &Pattern,
        with_clause: &WithClause,
        columns: &[String],
        group_var: &str,
        group_elem_idx: usize,
        group_key_indices: &[usize],
        count_indices: &[usize],
    ) -> Result<Option<Vec<ResultRow>>, String> {
        if pattern.elements.len() != 3 {
            return Ok(None);
        }
        // Histogram fast path counts every edge of the given type — it
        // can't apply an arbitrary `edge_filter` pushed from a WHERE
        // clause. Bail when one is present so the caller falls back to
        // per-source iteration via `try_count_simple_pattern`, which
        // does honor the filter inline.
        let edge_pat = match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return Ok(None),
        };
        if edge_pat.edge_filter.is_some() {
            return Ok(None);
        }
        // Same direction-aware "group is target" predicate as the RETURN-
        // aggregate fast paths above. Pre-fix this only matched the user-
        // written shape (group_elem_idx == 2 with Outgoing edge), so the
        // post-`optimize_pattern_start_node` form (group_elem_idx == 0
        // with Incoming) silently bailed even though `lookup_peer_counts`
        // (target-keyed) serves both shapes.
        let group_is_target = matches!(
            (group_elem_idx, edge_pat.direction),
            (2, EdgeDirection::Outgoing) | (0, EdgeDirection::Incoming)
        );
        if !group_is_target {
            return Ok(None);
        }
        let edge_conn_type = edge_pat.connection_type.as_deref();
        let Some(ct_str) = edge_conn_type else {
            return Ok(None);
        };
        // The element index of the SOURCE side (non-group) of the pattern,
        // which is also the side whose props/type the type-anchor logic
        // below cares about. Mirrors the planner-reversal duality.
        let source_elem_idx = if group_elem_idx == 2 { 0 } else { 2 };
        // Target must have no property constraint; it's the group key.
        let (tgt_props, src_type, src_props) = match (
            &pattern.elements[source_elem_idx],
            &pattern.elements[group_elem_idx],
        ) {
            (PatternElement::Node(src), PatternElement::Node(tgt)) => {
                (&tgt.properties, src.node_type.as_deref(), &src.properties)
            }
            _ => return Ok(None),
        };
        if tgt_props.is_some() || src_props.is_some() {
            return Ok(None);
        }

        let conn_key = InternedKey::from_str(ct_str);
        let want_type_key = src_type.map(InternedKey::from_str);

        // Two fast paths. (A) no source constraint → precomputed
        // `peer_count_histogram`, O(distinct peers). (B) source has a
        // type constraint → single-pass sweep of the edge-type's
        // matching edges via `for_each_edge_of_conn_type`, filtering
        // sources by `node_type_of` and accumulating per-peer counts.
        //
        // Path (B) previously iterated per source and called
        // `edges_directed_filtered` for each; every matching edge went
        // through `DiskEdges::next → make_edge_ref → materialize_edge`,
        // which heap-allocated a `Box<EdgeData>` and took the
        // `edge_arena` Mutex for every edge. On wiki1000m (~11 M P27
        // edges) the per-query arena growth hit an allocator-growth
        // cliff (426 ms at 500 M → 5387 ms at 1 B). The callback form
        // reads only the (src, tgt) pair we need — no allocation, no
        // arena growth — and restores the expected ~2× scaling.
        let counts: std::collections::HashMap<u32, i64> = if let Some(want_key) = want_type_key {
            if !self.graph.has_connection_type(ct_str) {
                return Ok(Some(Vec::new()));
            }
            // Disk-only: at small scale use the source-centric
            // `for_each_edge_of_conn_type` (cheaper when matching
            // sources are a small fraction of the graph and the
            // `edge_endpoints` array fits in L3 cache). At large scale
            // switch to a linear sweep of `edge_endpoints` — the
            // source-centric path binary-searches each source's CSR
            // slice, reading `edge_endpoints[edge_idx]` randomly; on
            // wiki1000m (247 MB endpoints, far above the ~32 MB SLC)
            // those reads miss cache on every comparison, blowing
            // aggregation out to ~4.5 s. Sequential access is bound by
            // memory bandwidth (~5 ms for 250 MB) and restores the
            // expected ~2× scaling from 500 M → 1 B.
            use crate::graph::storage::backend::GraphBackend;
            let disk = match &self.graph.graph {
                GraphBackend::Disk(dg) => dg.as_ref(),
                _ => return Ok(None),
            };
            let conn_u64 = conn_key.as_u64();
            let mut counts: std::collections::HashMap<u32, i64> = std::collections::HashMap::new();
            let mut deadline_iter: usize = 0;
            let mut deadline_err: Option<String> = None;
            // Threshold chosen so `edge_endpoints` (~16 B/edge) sits
            // comfortably above L3/SLC (~32 MB on Apple Silicon, ~32–
            // 64 MB on server CPUs) — past that the source-centric
            // binary search's per-comparison random reads become the
            // dominant cost. Below this, both paths are sub-200 ms on
            // Wikidata-style data, so the choice doesn't matter.
            const LINEAR_SCAN_EDGE_COUNT_THRESHOLD: usize = 4_000_000;
            if disk.edge_count() >= LINEAR_SCAN_EDGE_COUNT_THRESHOLD {
                disk.scan_edges_of_conn_type_linear(conn_u64, |src, tgt, _edge_idx| {
                    deadline_iter = deadline_iter.wrapping_add(1);
                    if deadline_iter & ((1 << 17) - 1) == 0 {
                        if let Err(e) = self.check_deadline() {
                            deadline_err = Some(e);
                            return false;
                        }
                    }
                    if disk.node_type_of(src) != Some(want_key) {
                        return true;
                    }
                    *counts.entry(tgt.index() as u32).or_insert(0) += 1;
                    true
                });
            } else {
                self.graph.graph.for_each_edge_of_conn_type(
                    conn_key,
                    |src, tgt, _edge_idx, _props| {
                        deadline_iter = deadline_iter.wrapping_add(1);
                        if deadline_iter & ((1 << 14) - 1) == 0 {
                            if let Err(e) = self.check_deadline() {
                                deadline_err = Some(e);
                                return false;
                            }
                        }
                        if self.graph.graph.node_type_of(src) != Some(want_key) {
                            return true;
                        }
                        *counts.entry(tgt.index() as u32).or_insert(0) += 1;
                        true
                    },
                );
            }
            if let Some(e) = deadline_err {
                return Err(e);
            }
            counts
        } else {
            let Some(h) = self.graph.graph.lookup_peer_counts(conn_key) else {
                return Ok(None);
            };
            h
        };

        let _ = columns; // column names are the caller's ResultSet wrap
        let mut rows: Vec<ResultRow> = Vec::with_capacity(counts.len());
        for (&peer, &count) in &counts {
            let node_idx = NodeIndex::new(peer as usize);

            // Build temporary row so group-key expressions (e.g.
            // `c.title`) can resolve via the evaluator.
            let mut tmp_row = ResultRow::new();
            tmp_row
                .node_bindings
                .insert(group_var.to_string(), node_idx);

            let mut projected = Bindings::with_capacity(with_clause.items.len());
            for &idx in group_key_indices {
                let item = &with_clause.items[idx];
                let key = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}", item.expression));
                let val = self.evaluate_expression(&item.expression, &tmp_row)?;
                projected.insert(key, val);
            }
            for &idx in count_indices {
                let item = &with_clause.items[idx];
                let key = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}", item.expression));
                projected.insert(key, Value::Int64(count));
            }

            let mut new_row = ResultRow::from_projected(projected);
            new_row
                .node_bindings
                .insert(group_var.to_string(), node_idx);
            rows.push(new_row);
        }

        // Apply WITH WHERE filter (mirrors the slow path's behavior so
        // `count(h) > 5` etc. still work).
        if let Some(ref where_clause) = with_clause.where_clause {
            let folded = self.fold_constants_pred(&where_clause.predicate);
            rows.retain(|row| self.evaluate_predicate(&folded, row).unwrap_or(false));
        }

        Ok(Some(rows))
    }
}
