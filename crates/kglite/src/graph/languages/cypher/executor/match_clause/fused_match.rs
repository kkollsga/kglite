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

        let carried_vars = grouping_variables(&with_clause.items);
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
                    .set_cancel(self.cancel)
            .set_parallel(self.parallel)
            .set_parallel(self.parallel);
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

            // Create the result row, preserving bindings for every variable the
            // grouping keys read (see `helpers::carry_group_bindings`). This
            // operator is what an `OPTIONAL MATCH` + `count()` query fuses
            // into, so narrowing the carry-over to bare-variable keys here is
            // what made `ORDER BY t.priority` differ between the OPTIONAL and
            // non-OPTIONAL spellings of the same query.
            let mut new_row = ResultRow::from_projected(projected);
            carry_group_bindings(&carried_vars, row, &mut new_row);

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
            .set_cancel(self.cancel)
            .set_parallel(self.parallel)
            .set_parallel(self.parallel);
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

        let FusedAggregateShape {
            group_var,
            group_elem_idx,
            group_key_indices,
            count_indices,
        } = fused_aggregate_shape(pattern, return_clause)?;
        let last_elem_idx = pattern.elements.len() - 1;

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

        let property_grouping = group_key_indices.iter().all(|&idx| {
            matches!(
                return_clause.items[idx].expression,
                Expression::PropertyAccess { .. }
            )
        });

        // Property grouping cannot accumulate by NodeIndex: two distinct
        // endpoints with the same property value (including missing values,
        // which resolve to NULL) form one Cypher group. Count each matching
        // endpoint without materialising edge rows, resolve its key tuple
        // once, and merge the additive counts by Value before top-K.
        let result_rows = if property_grouping {
            let group_candidate_count = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(node) => node
                    .node_type
                    .as_deref()
                    .and_then(|node_type| self.graph.type_indices.get(node_type))
                    .map_or_else(|| self.graph.graph.node_count(), |nodes| nodes.len()),
                _ => 0,
            };
            // Peer-count aggregation removes one adjacency lookup per
            // candidate. Use it only once candidate
            // cardinality is material relative to graph size; small SODIR
            // types are much faster through direct degree probes.
            let peer_counts_worthwhile = group_candidate_count.saturating_mul(100)
                >= self.graph.graph.edge_count();
            // Heap backends cache these counts by relationship type after the
            // first scan. Exactness requires either an unconstrained opposite
            // endpoint or schema metadata proving its primary label for every
            // edge of this type. More complex endpoint/relationship predicates
            // bail to the node-centric counter below.
            let peer_counts = if !distinct_count
                && pattern.elements.len() == 3
                && peer_counts_worthwhile
                && (self.graph.graph.is_memory() || self.graph.graph.is_mapped())
            {
                let edge = match &pattern.elements[1] {
                    PatternElement::Edge(edge)
                        if edge.var_length.is_none()
                            && edge.connection_types.is_none()
                            && edge.properties.as_ref().is_none_or(|props| props.is_empty())
                            && edge.edge_filter.is_none() =>
                    {
                        Some(edge)
                    }
                    _ => None,
                };
                let other_elem_idx = if group_elem_idx == 0 { 2 } else { 0 };
                let other = match &pattern.elements[other_elem_idx] {
                    PatternElement::Node(node)
                        if node.extra_labels.is_empty()
                            && node.properties.as_ref().is_none_or(|props| props.is_empty()) =>
                    {
                        Some(node)
                    }
                    _ => None,
                };

                if let (Some(edge), Some(other), Some(conn_type)) =
                    (edge, other, edge.and_then(|edge| edge.connection_type.as_ref()))
                {
                    let group_is_target = matches!(
                        (group_elem_idx, edge.direction),
                        (2, EdgeDirection::Outgoing) | (0, EdgeDirection::Incoming)
                    );
                    let group_is_source = matches!(
                        (group_elem_idx, edge.direction),
                        (0, EdgeDirection::Outgoing) | (2, EdgeDirection::Incoming)
                    );
                    let other_type_guaranteed = match other.node_type.as_deref() {
                        None => true,
                        Some(expected) => self
                            .graph
                            .connection_type_metadata
                            .get(conn_type)
                            .is_some_and(|info| {
                                let endpoint_types = if group_is_target {
                                    &info.source_types
                                } else {
                                    &info.target_types
                                };
                                endpoint_types.len() == 1 && endpoint_types.contains(expected)
                            }),
                    };
                    if (group_is_target || group_is_source) && other_type_guaranteed {
                        let conn_key = InternedKey::from_str(conn_type);
                        let direction = if group_is_target {
                            Direction::Outgoing
                        } else {
                            Direction::Incoming
                        };
                        Some(match self
                            .graph
                            .graph
                            .cached_edge_counts_grouped_by_peer(
                                conn_key,
                                direction,
                                self.deadline,
                            )?
                        {
                            Some(counts) => counts,
                            None => Arc::new(self.graph.graph.count_edges_grouped_by_peer(
                                conn_key,
                                direction,
                                self.deadline,
                            )?),
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            // A plain typed endpoint already has an exact node list. Reading
            // that list directly avoids allocating one PatternMatch plus a
            // String binding for every candidate before aggregation.
            let direct_group_nodes = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(node)
                    if node.extra_labels.is_empty()
                        && node.properties.as_ref().is_none_or(|props| props.is_empty()) =>
                {
                    node.node_type
                        .as_deref()
                        .and_then(|node_type| self.graph.type_indices.get(node_type))
                        .map(|nodes| nodes.to_vec())
                }
                _ => None,
            };
            let group_matches = if direct_group_nodes.is_none() {
                let group_only_pattern = crate::graph::core::pattern_matching::Pattern {
                    elements: vec![pattern.elements[group_elem_idx].clone()],
                };
                let executor =
                    PatternExecutor::new_lightweight_with_params(self.graph, None, self.params)
                        .set_deadline(self.deadline)
                        .set_cancel(self.cancel)
            .set_parallel(self.parallel)
            .set_parallel(self.parallel);
                Some(executor.execute(&group_only_pattern)?)
            } else {
                None
            };
            let group_node_indices = direct_group_nodes.unwrap_or_else(|| {
                group_matches
                    .as_ref()
                    .expect("fallback group matches built")
                    .iter()
                    .filter_map(&extract_node_idx)
                    .collect()
            });

            let group_node_type = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(node) => node.node_type.as_deref(),
                _ => None,
            };
            // A typed group has one alias table for every candidate. Resolve
            // aliases once here instead of repeating two String-keyed lookups
            // for every peer (notably `name` on Keyword/Judge/Function). The
            // interned key is hoisted with them: every group property read
            // re-hashed its own name once per row otherwise.
            let direct_group_properties: Vec<(String, InternedKey, bool)> = group_key_indices
                .iter()
                .map(|&idx| match &return_clause.items[idx].expression {
                    Expression::PropertyAccess { property, .. } => {
                        let (resolved, pre_resolved) = match group_node_type {
                            Some(node_type) => {
                                (self.graph.resolve_alias(node_type, property).to_string(), true)
                            }
                            None => (property.clone(), false),
                        };
                        let key = InternedKey::from_str(&resolved);
                        (resolved, key, pre_resolved)
                    }
                    _ => unreachable!("property_grouping checked every group expression"),
                })
                .collect();

            // (resolved group values, representative node, merged count)
            let mut groups: Vec<(Vec<Value>, petgraph::graph::NodeIndex, i64)> = Vec::new();
            let mut group_index: FxHashMap<Vec<Value>, usize> = FxHashMap::default();
            // Scanned rows outnumber groups by orders of magnitude in the shape
            // this path exists for, so the key is built into a reused buffer and
            // only cloned when it opens a new group.
            let mut key_scratch: Vec<Value> = Vec::with_capacity(group_key_indices.len());
            // One store handle per node type instead of one probe per row:
            // `node_view` re-resolves the backend's per-type map every call, and
            // a grouped scan calls it once per candidate.
            let mut store_memo: Option<(InternedKey, Option<&Arc<ColumnStore>>)> = None;
            let mut eval_row = ResultRow::new();
            eval_row
                .node_bindings
                .insert(group_var.to_string(), petgraph::graph::NodeIndex::new(0));

            for (scan_count, node_idx) in group_node_indices.into_iter().enumerate() {
                if scan_count.is_multiple_of(2048) {
                    self.check_deadline()?;
                }
                let match_count = if let Some(counts) = &peer_counts {
                    counts.get(&(node_idx.index() as u32)).copied().unwrap_or(0)
                } else {
                    count_for_node(node_idx)?
                };
                if match_count == 0 {
                    continue;
                }
                key_scratch.clear();
                if !self.graph.graph.is_disk() {
                    let node = self.graph.graph.node_weight(node_idx).map(|data| {
                        let store = match store_memo {
                            Some((type_key, store)) if type_key == data.node_type => store,
                            _ => {
                                let store = self.graph.graph.column_store(data.node_type);
                                store_memo = Some((data.node_type, store));
                                store
                            }
                        };
                        let resolved = data
                            .properties
                            .columnar_row_id()
                            .and_then(|row_id| store.map(|store| (&**store, row_id)));
                        NodeView::new(data, resolved)
                    });
                    key_scratch.extend(direct_group_properties.iter().map(
                        |(property, key, pre_resolved)| match node {
                            Some(node) if *pre_resolved => {
                                resolve_node_property_keyed(node, property, *key, self.graph)
                            }
                            Some(node) => resolve_node_property(node, property, self.graph),
                            None => Value::Null,
                        },
                    ));
                } else {
                    *eval_row
                        .node_bindings
                        .get_mut(group_var)
                        .expect("group binding inserted before property aggregation") = node_idx;
                    for &idx in &group_key_indices {
                        key_scratch.push(self.evaluate_expression(
                            &return_clause.items[idx].expression,
                            &eval_row,
                        )?);
                    }
                }
                if let Some(&group_idx) = group_index.get(&key_scratch) {
                    groups[group_idx].2 = groups[group_idx]
                        .2
                        .checked_add(match_count)
                        .ok_or("count overflow while merging property groups")?;
                } else {
                    let group_idx = groups.len();
                    group_index.insert(key_scratch.clone(), group_idx);
                    groups.push((key_scratch.clone(), node_idx, match_count));
                }
            }

            if let Some(&(_, descending, limit)) = top_k.as_ref() {
                if descending {
                    groups.sort_unstable_by(|a, b| b.2.cmp(&a.2).then_with(|| a.1.cmp(&b.1)));
                } else {
                    groups.sort_unstable_by(|a, b| a.2.cmp(&b.2).then_with(|| a.1.cmp(&b.1)));
                }
                groups.truncate(limit);
            } else if let Some(&(_, descending, limit)) = candidate_emit.as_ref() {
                let mut counts: Vec<i64> = groups.iter().map(|group| group.2).collect();
                if descending {
                    counts.sort_unstable_by(|a, b| b.cmp(a));
                } else {
                    counts.sort_unstable();
                }
                if let Some(&threshold) = counts.get(limit.saturating_sub(1)) {
                    groups.retain(|group| {
                        if descending {
                            group.2 >= threshold
                        } else {
                            group.2 <= threshold
                        }
                    });
                }
            }

            let mut rows = Vec::with_capacity(groups.len());
            for (values, representative, count) in groups {
                let mut projected = Bindings::with_capacity(return_clause.items.len());
                for (&idx, value) in group_key_indices.iter().zip(values) {
                    projected.insert(
                        return_item_column_name(&return_clause.items[idx]),
                        value,
                    );
                }
                for &idx in &count_indices {
                    projected.insert(
                        return_item_column_name(&return_clause.items[idx]),
                        Value::Int64(count),
                    );
                }
                let mut row = ResultRow::from_projected(projected);
                row.node_bindings
                    .insert(group_var.to_string(), representative);
                rows.push(row);
            }
            rows
        } else if let Some(&(_, descending, limit)) = top_k.as_ref() {
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
            let other_elem_idx = if group_elem_idx == 0 {
                last_elem_idx
            } else {
                0
            };
            let other_node_unconstrained = matches!(
                &pattern.elements[other_elem_idx],
                PatternElement::Node(np)
                    if np.node_type.is_none()
                        && np.extra_labels.is_empty()
                        && np.properties.as_ref().is_none_or(|props| props.is_empty())
            );
            let edge_histogram_safe = matches!(
                &pattern.elements[1],
                PatternElement::Edge(ep)
                    if ep.connection_types.is_none() && ep.edge_filter.is_none()
            );
            if let (false, 3, Some(ct_str), None, true, true) = (
                distinct_count,
                pattern.elements.len(),
                edge_conn_type,
                group_node_props.as_ref(),
                other_node_unconstrained,
                edge_histogram_safe,
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
            let other_elem_idx_nontopk = if group_elem_idx == 0 {
                last_elem_idx
            } else {
                0
            };
            let other_node_unconstrained_nontopk = matches!(
                &pattern.elements[other_elem_idx_nontopk],
                PatternElement::Node(np)
                    if np.node_type.is_none()
                        && np.extra_labels.is_empty()
                        && np.properties.as_ref().is_none_or(|props| props.is_empty())
            );
            let edge_histogram_safe_nontopk = matches!(
                &pattern.elements[1],
                PatternElement::Edge(ep)
                    if ep.connection_types.is_none() && ep.edge_filter.is_none()
            );
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
            let edge_centric_rows = if let (false, 3, Some(ct_str), None, true, true, true) = (
                distinct_count,
                pattern.elements.len(),
                edge_conn_type,
                group_node_props_nontopk.as_ref(),
                group_is_target_nontopk,
                other_node_unconstrained_nontopk,
                edge_histogram_safe_nontopk,
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
                        .set_cancel(self.cancel)
            .set_parallel(self.parallel)
            .set_parallel(self.parallel);
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

    /// Discover fused-scan candidates through the shared pattern matcher.
    /// This preserves multi-label semantics while reusing id, equality, IN,
    /// range, prefix, global, and backend-specific indexes. The fused operator
    /// still owns expression evaluation and aggregation/top-K maintenance.
    fn fused_scan_candidates(&self, node_pattern: &NodePattern) -> Result<Vec<NodeIndex>, String> {
        PatternExecutor::new_lightweight_with_params(self.graph, None, self.params)
            .set_deadline(self.deadline)
            .set_cancel(self.cancel)
            .set_parallel(self.parallel)
            .find_matching_nodes_pub(node_pattern)
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
        let node_indices = self.fused_scan_candidates(node_pattern)?;

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

        // A probe row carrying exactly the scan's bindings. The real per-row
        // row is built per partition; this one only answers "is this variable
        // bound by the scan?" for the `count(<bound var>)` test below.
        let mut eval_row = ResultRow::new();
        eval_row
            .node_bindings
            .insert(node_var.to_string(), petgraph::graph::NodeIndex::new(0));

        // `None` marks an aggregate whose input is the constant "row present"
        // marker: `count(*)`, and `count(<bound var>)`, which cannot be null and
        // so must not materialise the node value per row. The test reads the
        // scan's bindings, which are fixed for the whole scan.
        let folded_agg_args: Vec<Option<Expression>> = agg_indices
            .iter()
            .map(|&ai| match &return_clause.items[ai].expression {
                Expression::FunctionCall {
                    name,
                    args,
                    distinct,
                } => {
                    // `count(*)`, and `count(<bound var>)` — whose binding is
                    // always present, so materialising the node value only to
                    // test it for null is pure waste.
                    let is_row_marker = args.is_empty()
                        || matches!(args[0], Expression::Star)
                        || (!*distinct
                            && name.eq_ignore_ascii_case("count")
                            && matches!(&args[0], Expression::Variable(v)
                                if eval_row.node_bindings.get(v).is_some()
                                    || eval_row.edge_bindings.get(v).is_some()));
                    // Fold the argument, as the materialized aggregation path
                    // does — this operator evaluated it unfolded.
                    (!is_row_marker).then(|| self.fold_constants_expr(&args[0]))
                }
                other => Some(self.fold_constants_expr(other)),
            })
            .collect();

        // Compile the per-row expressions against the scan's single node
        // variable: property routes and borrowed string comparisons resolved
        // once per node type, not once per row. Anything not modelled stays on
        // the interpreter (see `scan_eval`).
        let mut compiler = ScanCompiler::new(self, node_var);
        let compiled_group: Vec<ScanExpr<'_>> = folded_group_exprs
            .iter()
            .map(|expr| compiler.expr(expr))
            .collect();
        let compiled_agg: Vec<Option<ScanExpr<'_>>> = folded_agg_args
            .iter()
            .map(|arg| arg.as_ref().map(|expr| compiler.expr(expr)))
            .collect();
        let compiled_where = folded_where_ref.map(|pred| compiler.pred(pred));
        let mut runtime = compiler.finish();
        let needs_node = !runtime.is_empty();

        // Everything the per-node loop needs that is invariant across
        // partitions. The compiled trees are immutable and shared; only the
        // `ScanRuntime` (route table + store memo) is per-partition mutable
        // state, which is what makes this loop partitionable at all.
        let ctx = ScanPartitionCtx {
            node_var,
            compiled_where: compiled_where.as_ref(),
            compiled_group: &compiled_group,
            compiled_agg: &compiled_agg,
            agg_is_distinct: &agg_is_distinct,
            needs_node,
        };

        let ScanPartial {
            groups,
            accumulators: group_accumulators,
            ..
        } = if self.may_fan_out_scan(node_indices.len(), &ctx) {
            self.scan_partitions_parallel(&node_indices, &ctx, &runtime)?
        } else {
            let interrupt = ParallelInterrupt::new(|| self.check_deadline().err());
            self.scan_partition(&node_indices, &ctx, &mut runtime, &interrupt)?
        };

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
                                    // Zero *numeric* values sums to Int64(0) —
                                    // the same answer the materialized path
                                    // gives when `collect_numeric_values` comes
                                    // back empty.
                                    if acc.numeric_counts[ai] == 0 {
                                        Value::Int64(0)
                                    } else {
                                        // Integer-typed iff every numeric
                                        // input was an `Int64` and the total is
                                        // whole — the streaming path's rule.
                                        let is_int =
                                            acc.sum_was_int[ai] && acc.sums[ai].fract() == 0.0;
                                        if is_int {
                                            Value::Int64(acc.sums[ai] as i64)
                                        } else {
                                            Value::Float64(acc.sums[ai])
                                        }
                                    }
                                }
                                "avg" | "mean" | "average" => {
                                    // Divide by the numeric count, not the
                                    // non-null count: a single string cell in
                                    // an otherwise numeric property must not
                                    // drag the average down (it contributes
                                    // nothing to `sums`).
                                    if acc.numeric_counts[ai] == 0 {
                                        Value::Null
                                    } else {
                                        Value::Float64(
                                            acc.sums[ai] / acc.numeric_counts[ai] as f64,
                                        )
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

    /// Fused MATCH (n:Type) [WHERE] RETURN expressions ORDER BY keys LIMIT k.
    /// Single-pass scan: iterates nodes, evaluates the sort-key tuple per node,
    /// maintains a K-element heap. RETURN expressions are only evaluated for the
    /// K winners. Avoids materializing all rows.
    ///
    /// Ranking is [`super::ordering::compare_sort_keys`] over the whole key tuple —
    /// the same comparison `execute_order_by` uses — so the fused scan and the
    /// unfused `ORDER BY` + `LIMIT` pipeline select and order identical rows,
    /// including ties and NULL keys.
    pub(super) fn execute_fused_node_scan_top_k(
        &self,
        match_clause: &MatchClause,
        where_predicate: Option<&Predicate>,
        return_clause: &ReturnClause,
        sort_keys: &[FusedSortKey],
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
        let node_indices = self.fused_scan_candidates(node_pattern)?;

        // Pre-fold expressions
        let folded_keys: Vec<Expression> = sort_keys
            .iter()
            .map(|key| self.fold_constants_expr(&key.expression))
            .collect();
        let specs: Vec<SortSpec> = sort_keys
            .iter()
            .map(|key| SortSpec {
                ascending: key.ascending,
                nulls: key.nulls,
            })
            .collect();
        let folded_where = where_predicate.map(|p| self.fold_constants_pred(p));
        let folded_where_ref = folded_where.as_ref();

        // Single reusable eval row
        let mut eval_row = ResultRow::new();
        eval_row
            .node_bindings
            .insert(node_var.to_string(), petgraph::graph::NodeIndex::new(0));

        let mut collector: TopKCollector<petgraph::graph::NodeIndex> =
            TopKCollector::new(specs, limit);
        let mut key_buf: Vec<Value> = Vec::with_capacity(folded_keys.len());

        // Both the sort keys and the filter run on every candidate, so both are
        // compiled against the scan's node variable (see `scan_eval`); the
        // RETURN expressions run only for the K winners and stay interpreted.
        let mut compiler = ScanCompiler::new(self, node_var);
        let compiled_keys: Vec<ScanExpr<'_>> =
            folded_keys.iter().map(|expr| compiler.expr(expr)).collect();
        let compiled_where = folded_where_ref.map(|pred| compiler.pred(pred));
        let mut runtime = compiler.finish();
        let needs_node = !runtime.is_empty();

        for (scan_count, &node_idx) in node_indices.iter().enumerate() {
            // Periodic deadline check
            if scan_count.is_multiple_of(10000) {
                self.check_deadline()?;
            }

            // Set node binding for expression evaluation
            *eval_row
                .node_bindings
                .get_mut(node_var)
                .expect("invariant: node_var binding inserted upstream by pattern match") =
                node_idx;

            // One view per node — the store handle and the property routes are
            // memoised by node type.
            let node = if needs_node {
                runtime.bind(self.graph, node_idx)
            } else {
                None
            };

            // WHERE filter. Errors are swallowed (a row whose predicate cannot
            // be evaluated does not match), as `evaluate_predicate(..)
            // .unwrap_or(false)` did.
            if let Some(pred) = &compiled_where {
                if !matches!(pred.eval(self, &runtime, node, &eval_row), Ok(Some(true))) {
                    continue;
                }
            }

            // Evaluate the sort-key tuple; only a candidate that would enter
            // the top-K pays for an owned key tuple.
            key_buf.clear();
            for expr in &compiled_keys {
                key_buf.push(expr.eval(self, &runtime, node, &eval_row)?);
            }
            if collector.accepts(&key_buf, scan_count) {
                collector.push(&key_buf, scan_count, node_idx);
            }
        }
        let winners = collector.into_sorted();

        // Build RETURN expressions only for the K winners. A column that *is* a
        // sort key reuses the computed key instead of a second evaluation.
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
        let mut key_of_item: Vec<Option<usize>> = vec![None; return_clause.items.len()];
        for (key_idx, key) in sort_keys.iter().enumerate() {
            if let Some(item_idx) = key.return_item {
                if key_of_item[item_idx].is_none() {
                    key_of_item[item_idx] = Some(key_idx);
                }
            }
        }

        let mut result_rows = Vec::with_capacity(winners.len());
        for (keys, winner_idx) in &winners {
            *eval_row
                .node_bindings
                .get_mut(node_var)
                .expect("invariant: node_var binding inserted upstream") = *winner_idx;
            let mut projected = Bindings::with_capacity(columns.len());
            for (j, column) in columns.iter().enumerate() {
                let val = match key_of_item[j] {
                    Some(key_idx) => keys[key_idx].clone(),
                    None => self.evaluate_expression(&folded_return_exprs[j], &eval_row)?,
                };
                projected.insert(column.clone(), val);
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
            .set_cancel(self.cancel)
            .set_parallel(self.parallel)
            .set_parallel(self.parallel);
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
            // Dedicated pool + interrupt poll. Per-key work here is a whole
            // pattern count, so poll on every key rather than per chunk.
            let interrupt =
                crate::graph::parallel::ParallelInterrupt::new(|| self.check_deadline().err());
            let keys = &group_keys;
            crate::graph::parallel::install(|| {
                keys.par_iter()
                    .map(|&idx| {
                        interrupt.check_each()?;
                        count_one(idx)
                    })
                    .collect::<Result<_, _>>()
            })?
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
        // The histogram is keyed by a single connection type, so an
        // alternation would need its per-type histograms merged. Reading the
        // singular `connection_type` instead counted only the first branch —
        // a wrong answer. Bail to the caller's per-source counter, which
        // honours every branch: merging k histograms is a plan-shape choice
        // this fast path does not need to make, and the fallback is correct
        // at the cost of the O(distinct-peers) shortcut for alternations only.
        if edge_pat.connection_types.is_some() {
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
        Ok(Some(self.histogram_counts_to_rows(
            &counts,
            with_clause,
            group_var,
            group_key_indices,
            count_indices,
        )?))
    }

    /// Result assembly for [`Self::try_fast_with_aggregate_via_histogram`]:
    /// turn per-peer counts into projected rows and apply the `WITH … WHERE`
    /// filter. Runs once, after the counting sweep.
    fn histogram_counts_to_rows(
        &self,
        counts: &std::collections::HashMap<u32, i64>,
        with_clause: &WithClause,
        group_var: &str,
        group_key_indices: &[usize],
        count_indices: &[usize],
    ) -> Result<Vec<ResultRow>, String> {
        let mut rows: Vec<ResultRow> = Vec::with_capacity(counts.len());
        for (&peer, &count) in counts {
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

        Ok(rows)
    }
}

/// Running aggregate state for one group of a fused node scan.
///
/// Every supported aggregate is folded into the same handful of running values
/// plus, for `count(DISTINCT …)`, a per-aggregate value set — so one pass over
/// the scan answers `count` / `sum` / `avg` / `min` / `max` together, without
/// materialising the group's rows.
///
/// Two counts, not one: `count()` counts non-null values, while `sum()` and
/// `avg()` see only the numeric ones. They diverge on a mixed-type property,
/// which is what made `avg` over `[10, 20, 'hello']` answer 30/3.
struct InlineAccumulators {
    counts: Vec<i64>,
    /// Per-aggregate count of the values that were actually *numeric* — the
    /// divisor `avg()` needs, and the emptiness test `sum()` needs. It differs
    /// from `counts` (all non-null values) exactly when a property holds mixed
    /// types: `[10, 20, 'hello']` is 3 non-null values but 2 numeric ones, and
    /// dividing the numeric sum by 3 is the wrong average.
    numeric_counts: Vec<i64>,
    /// Whether every numeric value this aggregate saw was an `Int64` — the
    /// integer-ness probe `sum()` emits with, matching the streaming
    /// aggregate's `sum_was_int` (`stream::aggregate::AggState`), the path
    /// these queries take when they do not fuse.
    ///
    /// Deriving it from `mins` instead (as this operator used to) reads the
    /// *smallest* value under the cross-type order — where a string outranks
    /// every number — so one string cell turned `sum()` over an integer
    /// property into a float on the fused path only.
    sum_was_int: Vec<bool>,
    sums: Vec<f64>,
    mins: Vec<Option<Value>>,
    maxs: Vec<Option<Value>>,
    /// Per-aggregate value set for `count(DISTINCT …)`; `None` for the rest.
    distinct_sets: Vec<Option<FxHashSet<Value>>>,
}

impl InlineAccumulators {
    fn new(agg_is_distinct: &[bool]) -> Self {
        let width = agg_is_distinct.len();
        InlineAccumulators {
            counts: vec![0i64; width],
            numeric_counts: vec![0i64; width],
            sum_was_int: vec![true; width],
            sums: vec![0.0f64; width],
            mins: vec![None; width],
            maxs: vec![None; width],
            distinct_sets: agg_is_distinct
                .iter()
                .map(|&distinct| distinct.then(FxHashSet::default))
                .collect(),
        }
    }

    /// Fold a later partition's accumulator for the same group into this one.
    ///
    /// The associative combine every aggregate this operator serves admits:
    /// `count`/`sum` add, `min`/`max` take the extreme under the same
    /// `total_order` the row path uses, and `count(DISTINCT …)` unions the
    /// value sets. `avg` is derived from `sums`/`counts` at emission, so it
    /// merges for free. Nothing here reads row order, which is why the
    /// partitioned scan returns byte-identical results.
    fn merge(&mut self, other: InlineAccumulators) {
        for (ai, count) in other.counts.iter().enumerate() {
            self.counts[ai] += count;
            self.numeric_counts[ai] += other.numeric_counts[ai];
            self.sum_was_int[ai] &= other.sum_was_int[ai];
            self.sums[ai] += other.sums[ai];
        }
        for (ai, min) in other.mins.into_iter().enumerate() {
            if let Some(val) = min {
                let replace = match &self.mins[ai] {
                    None => true,
                    Some(cur) => {
                        crate::graph::core::filtering::total_order(&val, cur)
                            == std::cmp::Ordering::Less
                    }
                };
                if replace {
                    self.mins[ai] = Some(val);
                }
            }
        }
        for (ai, max) in other.maxs.into_iter().enumerate() {
            if let Some(val) = max {
                let replace = match &self.maxs[ai] {
                    None => true,
                    Some(cur) => {
                        crate::graph::core::filtering::total_order(&val, cur)
                            == std::cmp::Ordering::Greater
                    }
                };
                if replace {
                    self.maxs[ai] = Some(val);
                }
            }
        }
        for (ai, set) in other.distinct_sets.into_iter().enumerate() {
            if let Some(values) = set {
                self.distinct_sets[ai]
                    .get_or_insert_with(FxHashSet::default)
                    .extend(values);
            }
        }
    }

    /// Fold one row's value for aggregate `ai` into the running state.
    ///
    /// `Null` contributes to nothing: it is not counted, not summed, and never
    /// becomes a `min`/`max` — the same rule the materialized aggregation path
    /// applies. A non-null value that is not numeric (a string in an otherwise
    /// numeric property) counts and can win `min`/`max`, but contributes to
    /// neither `sums` nor `numeric_counts` — `sum`/`avg` see only numbers.
    ///
    /// `count(*)` and `count(<bound var>)` arrive as a non-null `Boolean`
    /// marker, so they count without materialising a node value.
    fn absorb(&mut self, ai: usize, val: &Value, distinct: bool) {
        if distinct {
            // count(DISTINCT …): dedup non-null values in the per-agg set.
            if !matches!(val, Value::Null) {
                self.distinct_sets[ai]
                    .get_or_insert_with(FxHashSet::default)
                    .insert(val.clone());
            }
            return;
        }
        if matches!(val, Value::Null) {
            return;
        }
        self.counts[ai] += 1;
        if let Some(f) = value_to_f64(val) {
            self.numeric_counts[ai] += 1;
            self.sums[ai] += f;
            // UniqueId and Float64 both force a Float64 sum, as they do on the
            // streaming path.
            if !matches!(val, Value::Int64(_)) {
                self.sum_was_int[ai] = false;
            }
        }
        // Phase A.2 / C4 — short-circuit on is_none() guarantees the unwrap
        // can't fire, but the .expect() makes the invariant explicit if a
        // future refactor reorders the conditions.
        if self.mins[ai].is_none()
            || crate::graph::core::filtering::total_order(
                val,
                self.mins[ai]
                    .as_ref()
                    .expect("invariant: is_none() short-circuited above"),
            ) == std::cmp::Ordering::Less
        {
            self.mins[ai] = Some(val.clone());
        }
        if self.maxs[ai].is_none()
            || crate::graph::core::filtering::total_order(
                val,
                self.maxs[ai]
                    .as_ref()
                    .expect("invariant: is_none() short-circuited above"),
            ) == std::cmp::Ordering::Greater
        {
            self.maxs[ai] = Some(val.clone());
        }
    }
}

/// The query-shape half of `execute_fused_match_return_aggregate`, resolved
/// once per query before the row loop: which variable groups, which pattern
/// end it binds to, and which RETURN items are keys vs aggregates.
struct FusedAggregateShape<'q> {
    group_var: &'q str,
    group_elem_idx: usize,
    group_key_indices: Vec<usize>,
    count_indices: Vec<usize>,
}

fn fused_aggregate_shape<'q>(
    pattern: &'q Pattern,
    return_clause: &'q ReturnClause,
) -> Result<FusedAggregateShape<'q>, String> {
    let first_var = match &pattern.elements[0] {
        PatternElement::Node(np) => np.variable.as_ref(),
        _ => return Err("FusedMatchReturnAggregate: expected node pattern".into()),
    };
    let last_elem_idx = pattern.elements.len() - 1;
    let second_var = match &pattern.elements[last_elem_idx] {
        PatternElement::Node(np) => np.variable.as_ref(),
        _ => return Err("FusedMatchReturnAggregate: expected node pattern".into()),
    };

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

    let group_elem_idx = if first_var.is_some_and(|v| v == group_var) {
        0
    } else if second_var.is_some_and(|v| v == group_var) {
        last_elem_idx
    } else {
        return Err("FusedMatchReturnAggregate: group variable not in pattern".into());
    };

    let mut group_key_indices = Vec::new();
    let mut count_indices = Vec::new();
    for (i, item) in return_clause.items.iter().enumerate() {
        if is_aggregate_expression(&item.expression) {
            count_indices.push(i);
        } else {
            group_key_indices.push(i);
        }
    }
    Ok(FusedAggregateShape {
        group_var,
        group_elem_idx,
        group_key_indices,
        count_indices,
    })
}
