//! Indexed vector retrieval over materialized rows or a proven whole type.

use super::helpers::*;
use super::*;
use crate::graph::schema::EmbeddingStore;
use rustc_hash::FxHashMap;

struct HnswScoreArgs {
    variable: String,
    property: String,
    query: Vec<f32>,
    options: vector_options::VectorOptions,
}

enum HnswOutcome {
    Indexed(ResultSet, RetrievalDiagnostics),
    Exact(RetrievalDiagnostics),
}

struct HnswRowCoverage {
    node_to_row: FxHashMap<usize, usize>,
    ordered_whole_store: bool,
}

enum HnswPopulation<'r> {
    Rows(&'r ResultSet),
    WholeType {
        nodes: crate::graph::storage::disk::type_index::TypeNodesRef<'r>,
        variable: &'r str,
    },
}

impl HnswPopulation<'_> {
    fn row(&self, index: usize) -> std::borrow::Cow<'_, ResultRow> {
        match self {
            Self::Rows(rows) => std::borrow::Cow::Borrowed(&rows.rows[index]),
            Self::WholeType { nodes, variable } => {
                let mut row = ResultRow::new();
                row.node_bindings.insert(
                    (*variable).to_owned(),
                    nodes.get(index).expect("validated retrieval position"),
                );
                std::borrow::Cow::Owned(row)
            }
        }
    }
}

impl<'a> CypherExecutor<'a> {
    /// Admit only an initial, unfiltered typed scan whose embedding slots are
    /// exactly its ordered membership. Correlated seeds, labels, constraints,
    /// stale indexes and explicit exact policy keep the materialized route.
    pub(super) fn try_hnsw_entry(&self, clauses: &[Clause]) -> Result<Option<ResultSet>, String> {
        let [Clause::Match(matched), Clause::FusedVectorScoreTopK {
            return_clause,
            score_item_index,
            descending: true,
            limit,
        }, ..] = clauses
        else {
            return Ok(None);
        };
        let [pattern] = matched.patterns.as_slice() else {
            return Ok(None);
        };
        let [PatternElement::Node(node)] = pattern.elements.as_slice() else {
            return Ok(None);
        };
        let (Some(variable), Some(node_type)) = (&node.variable, &node.node_type) else {
            return Ok(None);
        };
        if *limit == 0
            || node.properties.is_some()
            || node.multi_label_constrained()
            || !node.label_params.is_empty()
            || !matched.path_assignments.is_empty()
            || !matched.node_anchors.is_empty()
            || matched.where_clause.is_some()
            || matched.limit_hint.is_some()
            || matched.distinct_node_hint.is_some()
            || self
                .graph
                .secondary_label_index
                .get(&InternedKey::from_str(node_type))
                .is_some_and(|nodes| !nodes.is_empty())
        {
            return Ok(None);
        }
        let Some(nodes) = self.graph.type_indices.get(node_type) else {
            return Ok(None);
        };
        if nodes.is_empty() {
            return Ok(None);
        }
        self.budget.check_work(nodes.len(), "MATCH")?;
        let score_expr =
            self.fold_constants_expr(&return_clause.items[*score_item_index].expression);
        if let Expression::FunctionCall { args, .. } = &score_expr {
            if (3..=5).contains(&args.len()) && self.requested_retrieval_policy(args)? == "exact" {
                return Ok(None);
            }
        }
        let mut seed = ResultRow::new();
        seed.node_bindings.insert(
            variable.clone(),
            nodes.get(0).expect("nonempty type bucket"),
        );
        let Some(args) = self.hnsw_score_args(&score_expr, &seed)? else {
            return Ok(None);
        };
        if args.variable != *variable || args.options.exact {
            return Ok(None);
        }
        let Some(store) = self.graph.embedding_store(node_type, &args.property) else {
            return Ok(None);
        };
        // A pending refresh belongs to the established path, including its
        // warnings and fallback. Never pay or report a refresh twice on a bail.
        if !store.has_index() || store.index_is_stale() {
            return Ok(None);
        }
        match self.try_hnsw_fused_top_k(
            &score_expr,
            true,
            *limit,
            &HnswPopulation::WholeType { nodes, variable },
            return_clause,
            *score_item_index,
        )? {
            HnswOutcome::Indexed(result, info) => {
                self.record_retrieval(info);
                Ok(Some(result))
            }
            HnswOutcome::Exact(_) => Ok(None),
        }
    }

    /// Parse the constant arguments required by the HNSW fused path.
    /// Returning `None` delegates unsupported expression shapes to the exact
    /// scorer; evaluation errors keep their established error channel.
    fn hnsw_score_args(
        &self,
        score_expr: &Expression,
        first_row: &ResultRow,
    ) -> Result<Option<HnswScoreArgs>, String> {
        let args = match score_expr {
            Expression::FunctionCall { name, args, .. }
                if name == "vector_score" && (3..=5).contains(&args.len()) =>
            {
                args
            }
            _ => return Ok(None),
        };
        // ANN cannot use the first row's selectors for every other row.
        // This also recognizes constant options maps without broadly changing
        // expression folding or accepting non-deterministic calls.
        if VectorScoreCache::key_for(args).is_none() {
            return Ok(None);
        }
        let variable = match &args[0] {
            Expression::Variable(variable) => variable.clone(),
            _ => return Ok(None),
        };
        let property = match self.evaluate_expression(&args[1], first_row)? {
            Value::String(property) => property,
            _ => return Ok(None),
        };
        let query = self.extract_float_list(&args[2], first_row)?;
        let tail = args[3..]
            .iter()
            .map(|expr| self.evaluate_expression(expr, first_row))
            .collect::<Result<Vec<_>, _>>()?;
        let options = vector_options::parse(&tail)?;
        Ok(Some(HnswScoreArgs {
            variable,
            property,
            query,
            options,
        }))
    }

    /// Build the node-to-row lookup while validating the single-type,
    /// duplicate-free contract required by the HNSW path. The common ordered
    /// whole-store case is proved during this same walk, avoiding the rejected
    /// second store-sized membership pass.
    fn hnsw_row_coverage(
        &self,
        variable: &str,
        node_type: &str,
        store: &EmbeddingStore,
        first_idx: petgraph::graph::NodeIndex,
        result_set: &ResultSet,
    ) -> Option<HnswRowCoverage> {
        let mut node_to_row =
            FxHashMap::with_capacity_and_hasher(result_set.rows.len(), Default::default());
        node_to_row.insert(first_idx.index(), 0);
        let mut ordered_whole_store = result_set.rows.len() == store.len()
            && store.slot_to_node.first() == Some(&first_idx.index());
        if !ordered_whole_store && !store.node_to_slot.contains_key(&first_idx.index()) {
            return None;
        }

        for (row_index, row) in result_set.rows.iter().enumerate().skip(1) {
            let idx = *row.node_bindings.get(variable)?;
            if ordered_whole_store && store.slot_to_node.get(row_index) == Some(&idx.index()) {
                if node_to_row.insert(idx.index(), row_index).is_some() {
                    return None;
                }
                continue;
            }
            ordered_whole_store = false;
            let current_type = self
                .graph
                .graph
                .node_view(idx)?
                .node_type_str(&self.graph.interner);
            // Unembedded rows score NULL and precede numeric scores in DESC.
            // ANN cannot omit them, even if it found enough numeric candidates.
            if current_type != node_type
                || !store.node_to_slot.contains_key(&idx.index())
                || node_to_row.insert(idx.index(), row_index).is_some()
            {
                return None;
            }
        }
        Some(HnswRowCoverage {
            node_to_row,
            ordered_whole_store,
        })
    }

    /// Project RETURN expressions for HNSW winners using their pre-computed
    /// exact-scale scores. This mirrors Phase 3 of the exact fused path.
    pub(super) fn project_hnsw_winners(
        &self,
        scored: Vec<(usize, f64)>,
        score_expr: &Expression,
        result_set: &ResultSet,
        return_clause: &ReturnClause,
        score_item_index: usize,
    ) -> Result<ResultSet, String> {
        self.project_retrieval_winners(
            scored,
            score_expr,
            &HnswPopulation::Rows(result_set),
            return_clause,
            score_item_index,
        )
    }

    fn project_retrieval_winners(
        &self,
        scored: Vec<(usize, f64)>,
        score_expr: &Expression,
        population: &HnswPopulation<'_>,
        return_clause: &ReturnClause,
        score_item_index: usize,
    ) -> Result<ResultSet, String> {
        let columns = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();
        let folded_exprs: Vec<Expression> = return_clause
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                if index == score_item_index {
                    score_expr.clone()
                } else {
                    self.fold_constants_expr(&item.expression)
                }
            })
            .collect();

        let mut rows = Vec::with_capacity(scored.len());
        for (row_index, score) in scored {
            let row = population.row(row_index);
            let mut projected = Bindings::with_capacity(return_clause.items.len());
            for (index, item) in return_clause.items.iter().enumerate() {
                let value = if index == score_item_index {
                    Value::Float64(score)
                } else {
                    self.evaluate_expression(&folded_exprs[index], &row)?
                };
                projected.insert(return_item_column_name(item), value);
            }
            rows.push(ResultRow {
                node_bindings: row.node_bindings.clone(),
                edge_bindings: row.edge_bindings.clone(),
                path_bindings: row.path_bindings.clone(),
                projected,
            });
        }
        Ok(ResultSet {
            rows,
            columns,
            lazy_return_items: None,
        })
    }

    /// Use HNSW candidates only for constant selectors over one fully embedded,
    /// duplicate-free row population and a compatible index metric. Re-score
    /// survivors on the scalar score scale. Explicit exact policy bypasses
    /// both index refresh and selection. Unsupported shape, stale index,
    /// incompatible metric or filtered underfill delegates to the exact scan
    /// with the actual decline reason; no hypothetical route is reported.
    fn try_hnsw_fused_top_k(
        &self,
        score_expr: &Expression,
        descending: bool,
        limit: usize,
        population: &HnswPopulation<'_>,
        return_clause: &ReturnClause,
        score_item_index: usize,
    ) -> Result<HnswOutcome, String> {
        use crate::graph::algorithms::vector as vs;
        let mut info = RetrievalDiagnostics::exact("unsupported_shape");
        if let Expression::FunctionCall { args, .. } = score_expr {
            if (3..=5).contains(&args.len()) {
                info.requested_policy = self.requested_retrieval_policy(args)?;
            }
        }

        // ANN models "top-k most similar" — descending score, non-empty limit.
        if !descending || limit == 0 {
            return Ok(HnswOutcome::Exact(info.fallback("unsupported_shape")));
        }

        let first_row = population.row(0);
        let args = match self.hnsw_score_args(score_expr, &first_row)? {
            Some(args) => args,
            None => return Ok(HnswOutcome::Exact(info.fallback("row_dependent_selectors"))),
        };
        info.requested_policy = if args.options.exact { "exact" } else { "auto" }.into();

        if args.options.exact {
            return Ok(HnswOutcome::Exact(info.fallback("forced_exact")));
        }

        // Resolve the first row's store before building membership. This lets
        // the existing row walk prove the common ordered whole-store shape by
        // comparing each binding with the parallel slot_to_node entry, without
        // a second store-sized HashMap lookup pass.
        let first_idx = match first_row.node_bindings.get(&args.variable) {
            Some(&idx) => idx,
            None => return Ok(HnswOutcome::Exact(info.fallback("unsupported_shape"))),
        };
        let node_type = match self.graph.graph.node_view(first_idx) {
            Some(node) => node.node_type_str(&self.graph.interner).to_string(),
            None => return Ok(HnswOutcome::Exact(info.fallback("unsupported_shape"))),
        };
        let store = match self.graph.embedding_store(&node_type, &args.property) {
            Some(store) => store,
            None => return Ok(HnswOutcome::Exact(info.fallback("unsupported_shape"))),
        };
        let coverage = match population {
            HnswPopulation::Rows(result_set) => {
                match self.hnsw_row_coverage(
                    &args.variable,
                    &node_type,
                    store,
                    first_idx,
                    result_set,
                ) {
                    Some(coverage) => Some(coverage),
                    None => return Ok(HnswOutcome::Exact(info.fallback("row_coverage"))),
                }
            }
            HnswPopulation::WholeType { nodes, .. } => {
                if nodes.len() != store.len() {
                    return Ok(HnswOutcome::Exact(info.fallback("row_coverage")));
                }
                for (position, (node, &stored)) in nodes.iter().zip(&store.slot_to_node).enumerate()
                {
                    if position % INTERRUPT_POLL_INTERVAL == 0 {
                        self.check_deadline()?;
                    }
                    if node.index() != stored {
                        return Ok(HnswOutcome::Exact(info.fallback("row_coverage")));
                    }
                }
                None
            }
        };
        info.store = Some(format!("{node_type}.{}", args.property));

        let index = match store.index_for_query(self.graph.read_only) {
            Some(i) => i,
            None => {
                // No index, a read-only graph, or a delta too large to fold in
                // inline — all three fall through to the exact scan, which is
                // the *oracle* the approximate path is measured against. A
                // stale vector index therefore costs latency and nothing else,
                // which is why this arm serves rows rather than nulls. The
                // warning is for the one case the caller can act on: an index
                // exists and has fallen behind what a query will catch up.
                if store.has_index() && store.index_is_stale() {
                    self.warn(format!(
                        "vector index '{}.{}' is behind its store by {} vectors, over its \
                         auto_refresh_limit of {} — this query was served by exact scan. \
                         Rebuild with build_vector_index() to restore the index path.",
                        node_type,
                        args.property,
                        store.delta_size(),
                        store.auto_refresh_limit(),
                    ));
                }
                let reason = if store.has_index() {
                    "stale_index"
                } else {
                    "no_index"
                };
                return Ok(HnswOutcome::Exact(info.fallback(reason)));
            }
        };
        if args.query.len() != store.dimension {
            return Ok(HnswOutcome::Exact(info.fallback("unsupported_shape"))); // let the exact path raise the dimension error
        }
        // Resolve metric: explicit > stored > cosine; Poincaré → exact.
        let metric =
            match args.options.metric.or_else(|| {
                vs::DistanceMetric::from_name(store.metric.as_deref().unwrap_or("cosine"))
            }) {
                Some(metric) => metric,
                None => return Ok(HnswOutcome::Exact(info.fallback("unsupported_shape"))),
            };
        if crate::graph::algorithms::hnsw::HnswMetric::from_distance(metric) != Some(index.metric())
        {
            return Ok(HnswOutcome::Exact(info.fallback("metric_mismatch")));
        }

        // HNSW search → membership filter → re-score for exact score scale.
        let scorer = vs::Scorer::new(metric, &args.query);
        let query_norm = vs::dot_product(&args.query, &args.query).sqrt();
        let whole_store = coverage.as_ref().is_none_or(|coverage| {
            coverage.ordered_whole_store
                || (coverage.node_to_row.len() >= store.len()
                    && vs::store_is_fully_selected(store, |node| {
                        coverage.node_to_row.contains_key(&node)
                    }))
        });
        let k_fetch = limit.saturating_mul(4).max(limit).min(store.len());
        let ef = k_fetch.max(index.params().ef_search);
        let raw = index.search(
            &args.query,
            query_norm,
            k_fetch,
            Some(ef),
            &store.data,
            &store.norms,
        );

        let mut scored: Vec<(usize, f64)> = Vec::with_capacity(limit.min(raw.len()));
        for (slot, _dist) in raw {
            let node_raw = store.slot_to_node[slot as usize];
            let row_index = match &coverage {
                Some(coverage) => coverage.node_to_row.get(&node_raw).copied(),
                None => Some(slot as usize),
            };
            if let Some(ri) = row_index {
                let start = slot as usize * store.dimension;
                let emb = &store.data[start..start + store.dimension];
                let norm = store.norms[slot as usize];
                scored.push((ri, scorer.score(&args.query, emb, norm) as f64));
            }
        }
        // Stable sort: ties keep row order (matches the exact path's behaviour
        // closely enough; ANN is approximate by contract anyway).
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(limit);

        // Filtered + underfilled (a tight WHERE ate the over-fetch) → exact scan.
        if !whole_store && scored.len() < limit {
            return Ok(HnswOutcome::Exact(info.fallback("filtered_underfill")));
        }

        let result = self.project_retrieval_winners(
            scored,
            score_expr,
            population,
            return_clause,
            score_item_index,
        )?;
        info.actual_mode = "hnsw".into();
        info.fallback_reason = None;
        Ok(HnswOutcome::Indexed(result, info))
    }

    pub(super) fn execute_fused_vector_score_top_k(
        &self,
        return_clause: &ReturnClause,
        score_item_index: usize,
        descending: bool,
        limit: usize,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        if result_set.rows.is_empty() || limit == 0 {
            let columns: Vec<String> = return_clause
                .items
                .iter()
                .map(return_item_column_name)
                .collect();
            return Ok(ResultSet {
                rows: Vec::new(),
                columns,
                lazy_return_items: None,
            });
        }

        let score_expr =
            self.fold_constants_expr(&return_clause.items[score_item_index].expression);

        // HNSW fast path: when the score is `vector_score` over a single type
        // whose store carries a built index, search the index instead of scoring
        // every row (the same opt-in approximate path the fluent API auto-uses).
        // Declines to the exact scan below whenever it isn't applicable (no index, unsupported metric, filtered+underfilled,
        // mixed types, duplicate node bindings, ASC order).
        match self.try_hnsw_fused_top_k(
            &score_expr,
            descending,
            limit,
            &HnswPopulation::Rows(&result_set),
            return_clause,
            score_item_index,
        )? {
            HnswOutcome::Indexed(rs, info) => {
                self.record_retrieval(info);
                return Ok(rs);
            }
            HnswOutcome::Exact(info) => self.record_retrieval(info),
        }

        // The generic collector preserves NULLs and stable ties. Reusing it
        // also keeps exact fallback aligned with ordinary ORDER BY semantics.
        let sort_keys = [FusedSortKey {
            expression: score_expr,
            ascending: !descending,
            nulls: if descending {
                NullsPlacement::First
            } else {
                NullsPlacement::Last
            },
            return_item: Some(score_item_index),
        }];
        self.execute_fused_order_by_top_k(return_clause, &sort_keys, limit, result_set)
    }
}
