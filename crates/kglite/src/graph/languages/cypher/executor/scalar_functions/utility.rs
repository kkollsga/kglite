//! Cypher scalar functions — utility category. Split out of the monolithic
//! `evaluate_scalar_function` dispatcher. Routed from
//! `super::evaluate_scalar_function`; returns `Ok(None)` when `name` is not
//! one of this category's functions so the dispatcher tries the next.
use super::super::helpers::*;
use super::super::*;
use super::shared::*;
use crate::datatypes::values::Value;
use crate::graph::algorithms::vector as vs;
use crate::graph::languages::cypher::planner::simplification::TEXT_SCORE_STORES_PARAM;
use crate::graph::storage::GraphRead;
use crate::graph::text_indexes;

impl<'a> CypherExecutor<'a> {
    pub(super) fn eval_utility_fn(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<Value>, String> {
        let result: Result<Value, String> = match name {
            "vector_score" => self.eval_vector_score(args, row),
            // text_bm25(n, 'property', 'query text') — BM25 relevance of one
            // row's document against a query, or null when that row has no
            // document. Lives in `utility` rather than `string` because it is
            // the same shape as its neighbours here — a *node*-and-store
            // scalar (vector_score, embedding_norm, text_score) — while
            // `string`'s text_* family are pure string→scalar functions that
            // never touch the graph.
            "text_bm25" => self.eval_text_bm25(args, row),
            // score_fuse(s1, s2, … [, [w1, w2, …]]) — one number out of
            // several ranked lanes. Registered in `utility` beside the lanes
            // it fuses (text_bm25, vector_score, text_score) rather than in
            // `numeric`, because what it is *for* is hybrid retrieval: the
            // three names a caller needs for one query are then one
            // `SHOW FUNCTIONS` category apart, not two. It touches no graph
            // state, so it is also the one function here that would work
            // unchanged in any other module.
            "score_fuse" => self.eval_score_fuse(args, row),
            // randomUUID() — RFC 4122 version-4 UUID string. Non-
            // deterministic; classified alongside rand() in
            // `is_row_independent` (where_clause.rs) so constant folding
            // never collapses it to a single value across rows. No `uuid`
            // crate dependency — 128 random bits from the same
            // thread-local xorshift64 PRNG rand() uses (two u64 draws),
            // version (4) and variant (10xx) bits stamped per the v4
            // layout. Registered under the lowercased key `randomuuid`;
            // the canonical Cypher spelling is randomUUID().
            "randomuuid" => {
                if !args.is_empty() {
                    return Err("randomUUID() takes no arguments".into());
                }
                let (hi, lo) = next_random_u128_halves();
                // Stamp version 4 into the high u64 (bits 12-15 of the
                // time_hi_and_version field) and variant 10xx into the
                // low u64 (top two bits of clock_seq_hi).
                let hi = (hi & 0xFFFF_FFFF_FFFF_0FFF) | 0x0000_0000_0000_4000;
                let lo = (lo & 0x3FFF_FFFF_FFFF_FFFF) | 0x8000_0000_0000_0000;
                let uuid = format!(
                    "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                    hi >> 32,
                    (hi >> 16) & 0xFFFF,
                    hi & 0xFFFF,
                    lo >> 48,
                    lo & 0xFFFF_FFFF_FFFF,
                );
                Ok(Value::String(uuid))
            }
            // localdatetime() / localtime() / time() dispatch in
            // temporal.rs via eval_local_temporal (documented on the
            // helper in mod.rs).
            "rand" | "random" => {
                // Top 53 bits → f64 mantissa to avoid precision loss.
                let x = next_random_u64();
                let val = ((x >> 11) as f64) / ((1u64 << 53) as f64);
                Ok(Value::Float64(val))
            }

            // ── Temporal filtering functions ──────────────────────────────
            "valid_at" => self.eval_valid_at(args, row),
            "valid_during" => self.eval_valid_during(args, row),
            // Aggregate functions should not be evaluated per-row
            "count" | "sum" | "avg" | "min" | "max" | "collect" | "mean" | "std" | "stdev" => {
                Err(format!(
                    "Aggregate function '{}' cannot be used outside of RETURN/WITH",
                    name
                ))
            }
            // embedding_norm(entity, property) → Float64
            // Returns the L2 norm of the entity's embedding vector.
            // Useful for inferring hierarchy depth in Poincaré embeddings
            // (norm close to 0 = root/general, norm close to 1 = leaf/specific).
            "embedding_norm" => self.eval_embedding_norm(args, row),
            "text_score" => Err(
                "text_score() requires set_embedder(). Call g.set_embedder(model) first."
                    .to_string(),
            ),
            // parse_json(s) — recursively parse a JSON string into structured
            // Value (Map / List / scalars) so Cypher can predicate over data
            // that is stored as a JSON string. The code graph keeps
            // Function.parameters / Class.fields as JSON arrays-of-objects
            // (the columnar store is scalar-only), so this unlocks queries like
            //   MATCH (f:Function)
            //   WHERE any(p IN parse_json(f.parameters) WHERE p.type = 'Dataset')
            // Returns Null on a non-string arg or on invalid JSON (Neo4j-style
            // lenient: bad input is null, not an error).
            "parse_json" | "from_json" => {
                if args.len() != 1 {
                    return Err("parse_json() requires exactly 1 argument".to_string());
                }
                match self.evaluate_expression(&args[0], row)? {
                    Value::String(s) => Ok(serde_json::from_str::<serde_json::Value>(&s)
                        .map(|j| crate::param::json_value_to_kglite_value(&j))
                        .unwrap_or(Value::Null)),
                    Value::Null => Ok(Value::Null),
                    _ => Ok(Value::Null),
                }
            }
            _ => return Ok(None),
        };
        result.map(Some)
    }
}

impl CypherExecutor<'_> {
    /// `text_bm25(node, 'property', 'query text')` → BM25 relevance of that one
    /// row's document, or null when the row has no document.
    fn eval_text_bm25(&self, args: &[Expression], row: &ResultRow) -> Result<Value, String> {
        if args.len() != 3 {
            return Err("text_bm25() requires 3 arguments: (node, property, query_text)".into());
        }
        let node_idx = match &args[0] {
            Expression::Variable(var) => match row.node_bindings.get(var) {
                Some(&idx) => idx,
                None => return self.eval_text_bm25_non_node(args, row),
            },
            _ => return self.eval_text_bm25_non_node(args, row),
        };
        let node_type = match self.graph.graph.node_view(node_idx) {
            Some(n) => n.node_type_str(&self.graph.interner),
            None => return Ok(Value::Null),
        };
        if let Some(cache) = self.tb_cache.get() {
            if !cache.relationship
                && cache.node_type == node_type
                && cache.keys.as_ref().is_some_and(|(property, query)| {
                    property.matches(&args[1]) && query.matches(&args[2])
                })
            {
                return self.score_text_bm25_row(cache, node_idx);
            }
        }
        let cache = self.prepare_text_bm25(args, row, node_type)?;
        let scored = self.score_text_bm25_row(&cache, node_idx);
        // One slot: the first (type, property, query) triple to arrive keeps
        // it, and anything else re-prepares per row rather than reading an
        // answer that is not its own. A row-dependent argument is never cached
        // at all — its value is allowed to change between rows.
        if cache.keys.is_some() {
            let _ = self.tb_cache.set(cache);
        }
        scored
    }

    fn eval_vector_score(&self, args: &[Expression], row: &ResultRow) -> Result<Value, String> {
        if args.len() < 3 || args.len() > 5 {
            return Err(
                "vector_score() requires 3-5 arguments: (entity, property, query_vector [, metric] [, options])"
                    .into(),
            );
        }

        // A binding is addressed by name; anything else is evaluated and has
        // to arrive as a node or relationship *value* (`collect(n)[0]`,
        // `nodes(p)[0]`, a `CALL {}` column). Both reach the same scoring
        // below.
        let variable = match &args[0] {
            Expression::Variable(var) => Some(var),
            _ => None,
        };

        if let Some(variable) = variable {
            if let Some(edge) = row.edge_bindings.get(variable) {
                return self.eval_edge_vector_score(args, row, *edge);
            }
            if row.path_bindings.contains_key(variable) {
                return Err(
                    "vector_score(): first argument must be a node or relationship variable".into(),
                );
            }
        }
        // Per-row: look up node type → embedding store → compute similarity
        let (node_idx, node_type) =
            match variable.and_then(|var| row.node_bindings.get(var)) {
                Some(&node_idx) => match self.graph.graph.node_view(node_idx) {
                    Some(n) => (node_idx, n.node_type_str(&self.graph.interner)),
                    None => return Ok(Value::Null),
                },
                None => match self.evaluate_expression(&args[0], row)? {
                    Value::Relationship(relationship) => {
                        let edge = self.projected_relationship_binding(&relationship)?;
                        return self.eval_edge_vector_score(args, row, edge);
                    }
                    Value::Node(node) => match self.projected_node_binding(&node) {
                        Some(resolved) => resolved,
                        None => return Ok(Value::Null),
                    },
                    Value::Null => return Ok(Value::Null),
                    _ => return Err(
                        "vector_score(): first argument must be a node or relationship variable"
                            .into(),
                    ),
                },
            };

        // The constant arguments, parsed once per call site — or per
        // row when this call's arguments are row-dependent, or when
        // every cache slot already belongs to another call site.
        let uncached;
        let c = match self.vs_cache.get(args, node_type) {
            Some(cached) => cached,
            None => match self
                .vs_cache
                .park(self.prepare_vector_score(args, row, node_type)?)
            {
                Ok(parked) => parked,
                Err(entry) => {
                    uncached = entry;
                    &uncached
                }
            },
        };

        let store = match self.graph.embedding_store(node_type, &c.prop_name) {
            Some(s) => s,
            None => return Err(self.missing_embedding_error(node_type, &c.prop_name)),
        };

        Self::check_vector_score_dimension(c.query_vec.len(), store.dimension)?;

        match store.get_embedding_with_norm(node_idx.index()) {
            Some((embedding, norm)) => {
                let score = c.scorer.score(&c.query_vec, embedding, norm);
                Ok(Value::Float64(score as f64))
            }
            None => Ok(Value::Null),
        }
    }

    /// `embedding_norm(entity, 'property')` → the L2 norm of that entity's
    /// stored vector, or `Null` when it has none.
    ///
    /// Accepts the same first argument `vector_score` does: a node or
    /// relationship binding, or a materialised value of either. A value whose
    /// slot is dead, stale or retyped scores `Null`; a *store* the entity's
    /// type does not have is an error, because that is a query about a column
    /// that does not exist rather than a row without a vector.
    fn eval_embedding_norm(&self, args: &[Expression], row: &ResultRow) -> Result<Value, String> {
        if args.len() != 2 {
            return Err("embedding_norm() requires 2 arguments: (entity, property)".into());
        }
        let variable = match &args[0] {
            Expression::Variable(var) => Some(var),
            _ => None,
        };
        let prop_name = match self.evaluate_expression(&args[1], row)? {
            Value::String(s) => s,
            _ => {
                return Err(
                    "embedding_norm(): second argument must be a string property name".into(),
                )
            }
        };
        if let Some(variable) = variable {
            if let Some(edge) = row.edge_bindings.get(variable) {
                return Ok(norm_of(self.edge_embedding(
                    *edge,
                    &prop_name,
                    "embedding_norm",
                )?));
            }
            if row.path_bindings.contains_key(variable) {
                return Err(
                    "embedding_norm(): first argument must be a node or relationship variable"
                        .into(),
                );
            }
        }
        let embedding =
            match variable.and_then(|variable| row.node_bindings.get(variable)) {
                Some(&node_idx) => {
                    let Some(node_type) = self
                        .graph
                        .graph
                        .node_view(node_idx)
                        .map(|node| node.node_type_str(&self.graph.interner))
                    else {
                        return Ok(Value::Null);
                    };
                    self.node_embedding(node_idx, node_type, &prop_name, "embedding_norm")?
                }
                None => match self.evaluate_expression(&args[0], row)? {
                    Value::Relationship(relationship) => {
                        let edge = self.projected_relationship_binding(&relationship)?;
                        self.edge_embedding(edge, &prop_name, "embedding_norm")?
                    }
                    Value::Node(node) => match self.projected_node_binding(&node) {
                        Some((node_idx, node_type)) => {
                            self.node_embedding(node_idx, node_type, &prop_name, "embedding_norm")?
                        }
                        None => return Ok(Value::Null),
                    },
                    Value::Null => return Ok(Value::Null),
                    _ => return Err(
                        "embedding_norm(): first argument must be a node or relationship variable"
                            .into(),
                    ),
                },
            };
        Ok(norm_of(embedding))
    }

    /// One node's stored vector. `Ok(None)` is "this node has no vector";
    /// a missing store is the error.
    fn node_embedding(
        &self,
        node_idx: petgraph::graph::NodeIndex,
        node_type: &str,
        prop_name: &str,
        caller: &str,
    ) -> Result<Option<&[f32]>, String> {
        let store = self
            .graph
            .embedding_store(node_type, prop_name)
            .ok_or_else(|| {
                format!("{caller}(): no embedding '{prop_name}' found for node type '{node_type}'")
            })?;
        Ok(store.get_embedding(node_idx.index()))
    }

    /// One relationship's stored vector. A binding this statement has
    /// invalidated, or a dead slot, has no vector rather than an error.
    fn edge_embedding(
        &self,
        edge: EdgeBinding,
        prop_name: &str,
        caller: &str,
    ) -> Result<Option<&[f32]>, String> {
        if !self.relationship_binding_is_current(&edge) {
            return Ok(None);
        }
        let Some(weight) = self.graph.graph.edge_weight(edge.edge_index) else {
            return Ok(None);
        };
        let relationship_type = weight.connection_type_str(&self.graph.interner);
        let store = self
            .graph
            .edge_embeddings
            .get(&(relationship_type.to_string(), prop_name.to_string()))
            .ok_or_else(|| {
                format!(
                    "{caller}(): no embedding '{prop_name}' found for relationship type \
                     '{relationship_type}'"
                )
            })?;
        Ok(store.get(edge.edge_index))
    }

    /// Resolve a materialised node **value** — `collect(n)[0]`, an `UNWIND`
    /// element, a `CALL {}` column, `nodes(p)`, `head(...)` — back to the slot
    /// it names, with that slot's current primary type.
    ///
    /// `None` when the slot is dead or now carries a different type than the
    /// value's primary label, and the scalars turn that into `Null` rather
    /// than an error: a node value is a snapshot, and a snapshot of something
    /// that has since been deleted or replaced has no score, it is not a
    /// malformed argument. `NodeValue::labels` is primary-first
    /// (`DirGraph::node_labels`), and a value for a node deleted earlier in
    /// the same statement carries no labels at all, so it never matches.
    fn projected_node_binding(
        &self,
        node: &crate::datatypes::values::NodeValue,
    ) -> Option<(petgraph::graph::NodeIndex, &str)> {
        let node_idx = petgraph::graph::NodeIndex::new(node.id as usize);
        let node_type = self
            .graph
            .graph
            .node_view(node_idx)?
            .node_type_str(&self.graph.interner);
        node.labels
            .first()
            .is_some_and(|label| label == node_type)
            .then_some((node_idx, node_type))
    }

    pub(super) fn projected_relationship_binding(
        &self,
        relationship: &crate::datatypes::values::RelValue,
    ) -> Result<EdgeBinding, String> {
        let edge = EdgeBinding {
            source: petgraph::graph::NodeIndex::new(relationship.start_id as usize),
            target: petgraph::graph::NodeIndex::new(relationship.end_id as usize),
            edge_index: petgraph::graph::EdgeIndex::new(relationship.id as usize),
            incarnation: relationship.incarnation,
        };
        if self.relationship_identities.is_some() && edge.incarnation.is_none() {
            return Err("relationship value was not bound by this statement".into());
        }
        Ok(edge)
    }

    fn eval_edge_vector_score(
        &self,
        args: &[Expression],
        row: &ResultRow,
        edge: EdgeBinding,
    ) -> Result<Value, String> {
        if !self.relationship_binding_is_current(&edge) {
            return Ok(Value::Null);
        }
        let Some(weight) = self.graph.graph.edge_weight(edge.edge_index) else {
            return Ok(Value::Null);
        };
        let relationship_type = weight.connection_type_str(&self.graph.interner);
        let prop_name = match self.evaluate_expression(&args[1], row)? {
            Value::String(s) => s,
            _ => {
                return Err("vector_score(): second argument must be a string property name".into())
            }
        };
        let query_vec = self.extract_float_list(&args[2], row)?;
        crate::graph::embedding_validation::validate_finite_vector(&query_vec)
            .map_err(|error| format!("vector_score(): invalid query vector: {error}"))?;
        let store = self
            .graph
            .edge_embeddings
            .get(&(relationship_type.to_string(), prop_name.clone()))
            .ok_or_else(|| self.missing_edge_embedding_error(relationship_type, &prop_name))?;
        Self::check_vector_score_dimension(query_vec.len(), store.dimension())?;
        let tail = args[3..]
            .iter()
            .map(|expr| self.evaluate_expression(expr, row))
            .collect::<Result<Vec<_>, _>>()?;
        let options = super::super::vector_options::parse(&tail)?;
        let metric = match options.metric {
            Some(metric) => metric,
            None => vs::DistanceMetric::from_name(store.metric().unwrap_or("cosine")).ok_or_else(
                || {
                    format!(
                        "vector_score(): unknown stored metric '{}'",
                        store.metric().unwrap_or("cosine")
                    )
                },
            )?,
        };
        let scorer = vs::Scorer::new(metric, &query_vec);
        match store.get_with_norm(edge.edge_index) {
            Some((embedding, norm)) => Ok(Value::Float64(
                scorer.score(&query_vec, embedding, norm) as f64,
            )),
            None => Ok(Value::Null),
        }
    }

    pub(in crate::graph::languages::cypher::executor) fn check_vector_score_dimension(
        query_dimension: usize,
        embedding_dimension: usize,
    ) -> Result<(), String> {
        if query_dimension != embedding_dimension {
            return Err(format!(
                "vector_score(): query vector dimension {query_dimension} does not match embedding dimension {embedding_dimension}",
            ));
        }
        Ok(())
    }

    /// Parse `vector_score()`'s constant arguments — property name, query
    /// vector, and the metric (explicit argument, else the store's own, else
    /// cosine).
    ///
    /// The returned entry carries the key it was prepared under, so the caller
    /// can park it for the rest of the scan; a row-dependent argument yields a
    /// keyless entry that scores this row only.
    pub(in crate::graph::languages::cypher::executor) fn prepare_vector_score(
        &self,
        args: &[Expression],
        row: &ResultRow,
        node_type: &str,
    ) -> Result<VectorScoreCache, String> {
        #[cfg(test)]
        VECTOR_SCORE_PREPARES.with(|count| count.set(count.get() + 1));

        let prop_name = match self.evaluate_expression(&args[1], row)? {
            Value::String(s) => s,
            _ => {
                return Err("vector_score(): second argument must be a string property name".into())
            }
        };
        let query_vec = self.extract_float_list(&args[2], row)?;
        crate::graph::embedding_validation::validate_finite_vector(&query_vec)
            .map_err(|error| format!("vector_score(): invalid query vector: {error}"))?;
        let tail = args[3..]
            .iter()
            .map(|expr| self.evaluate_expression(expr, row))
            .collect::<Result<Vec<_>, _>>()?;
        let options = super::super::vector_options::parse(&tail)?;
        let store = self
            .graph
            .embedding_store(node_type, &prop_name)
            .ok_or_else(|| self.missing_embedding_error(node_type, &prop_name))?;
        let metric = match options.metric {
            Some(metric) => metric,
            None => {
                let name = store.metric.as_deref().unwrap_or("cosine");
                vs::DistanceMetric::from_name(name)
                    .ok_or_else(|| format!("vector_score(): unknown stored metric '{name}'"))?
            }
        };
        let scorer = vs::Scorer::new(metric, &query_vec);
        Ok(VectorScoreCache {
            keys: VectorScoreCache::key_for(args),
            node_type: node_type.to_string(),
            prop_name,
            query_vec,
            scorer,
        })
    }

    /// Score one row against an already-prepared query.
    ///
    /// The generation is read *after* the read guard is taken: a refresh bumps
    /// it while holding the write lock, so a guard held across the comparison
    /// is what makes the answer still true when the score below is computed.
    /// A mismatch means a concurrent refresh renumbered the dictionary, and the
    /// cached term ids may now name other terms — re-prepare rather than score
    /// against them.
    ///
    /// `None` from the index is an **unindexed** row and surfaces as null; an
    /// indexed row sharing no term with the query is `Some(0.0)` and surfaces
    /// as `0.0`. Collapsing the two would make "not searchable yet" look like
    /// "searched, no match".
    fn score_text_bm25_row(
        &self,
        cache: &TextBm25Cache,
        node: petgraph::graph::NodeIndex,
    ) -> Result<Value, String> {
        let Some(query_text) = cache.query_text.as_deref() else {
            return Ok(Value::Null);
        };
        let Some(store) =
            text_indexes::text_index_store(self.graph, &cache.node_type, &cache.prop_name)
        else {
            return Err(missing_text_index_error(
                self.graph,
                &cache.node_type,
                &cache.prop_name,
            ));
        };
        let view = store.read();
        let score = if store.generation() == cache.generation {
            view.score(node, &cache.prepared)
        } else {
            view.score(node, &view.prepare_query(query_text))
        };
        Ok(score.map_or(Value::Null, Value::Float64))
    }

    /// Resolve `text_bm25()`'s constant arguments, run the query-entry
    /// freshness policy once, and tokenize the query.
    ///
    /// **The freshness policy (release-train-0-16-10, decision 11a).** The
    /// check runs only here — in a query that actually calls `text_bm25` on
    /// this index — so every other query pays nothing for it. A delta within
    /// the index's `auto_refresh_limit` is folded in inline before the query is
    /// prepared, which is what lets a node created after the build score
    /// without an explicit rebuild. A delta over the limit, or a read-only
    /// graph (where a refresh would be the one write a read-only handle
    /// performed), serves the index as it stands: the rows it has no document
    /// for score null, and the query carries a warning naming the delta and the
    /// call that fixes it. A query never silently absorbs a post-bulk-ingest
    /// catch-up: it absorbs at most the ceiling the index's author set, and
    /// never more than one rebuild's worth of work, because a fold past the
    /// measured crossover rebuilds instead of splicing
    /// (`text_indexes::rebuild_beats_folding`).
    pub(in crate::graph::languages::cypher::executor) fn prepare_text_bm25(
        &self,
        args: &[Expression],
        row: &ResultRow,
        node_type: &str,
    ) -> Result<TextBm25Cache, String> {
        let prop_name = match self.evaluate_expression(&args[1], row)? {
            Value::String(s) => s,
            _ => return Err("text_bm25(): second argument must be a string property name".into()),
        };
        let query_text = match self.evaluate_expression(&args[2], row)? {
            Value::String(s) => Some(s),
            Value::Null => None,
            _ => return Err("text_bm25(): third argument must be a query string".into()),
        };
        let Some(store) = text_indexes::text_index_store(self.graph, node_type, &prop_name) else {
            return Err(missing_text_index_error(self.graph, node_type, &prop_name));
        };

        if store.is_stale(self.graph) {
            if !self.graph.read_only && store.can_auto_refresh(self.graph) {
                text_indexes::refresh_text_index(self.graph, node_type, &prop_name);
            } else {
                // "up to": the delta over-counts — it is a watermark gap plus a
                // dirty set, and a slot in either may turn out to hold no
                // document at all.
                let reason = if self.graph.read_only {
                    "and this graph is read-only, so a query cannot catch it up".to_string()
                } else {
                    format!(
                        "over its auto_refresh_limit of {}",
                        store.auto_refresh_limit()
                    )
                };
                self.warn(format!(
                    "text index '{}.{}' is stale: up to {} documents are unindexed, {} — a new \
                     node scores null and a changed one scores its previously indexed text. \
                     Rebuild with build_text_index('{}', '{}').",
                    node_type,
                    prop_name,
                    store.delta_size(self.graph),
                    reason,
                    node_type,
                    prop_name,
                ));
            }
        }

        let view = store.read();
        // Read under the guard, for the reason `score_text_bm25_row` documents.
        let generation = store.generation();
        let prepared = match query_text.as_deref() {
            Some(text) => view.prepare_query(text),
            None => Default::default(),
        };
        drop(view);
        Ok(TextBm25Cache {
            node_type: node_type.to_string(),
            relationship: false,
            keys: ArgKey::of(&args[1]).zip(ArgKey::of(&args[2])),
            query_text,
            prepared,
            prop_name,
            generation,
        })
    }

    /// `score_fuse(s1, s2, … [, [w1, w2, …]])` — the weighted mean of the
    /// signals that are **present**, so one Cypher query can rank by several
    /// retrieval lanes at once.
    ///
    /// **Absent is `null`, `NaN` and `±inf`**, and an absent signal leaves the
    /// average — its weight leaves the denominator with it. That is a
    /// deliberate departure from the null-in/null-out rule the other scalars
    /// follow (see `vector.rs`): a lane reports `null` for a row it *could not
    /// see* (no document in the text index, no stored embedding), and both
    /// alternatives are wrong for a ranking. Nulling the whole row deletes a
    /// document the other lane found; folding the absence in as `0.0` ranks it
    /// below a document both lanes actively disliked. Averaging the lanes that
    /// did run keeps the row comparable on the evidence that exists. Only
    /// "every signal absent" makes the call `null` — there is then nothing to
    /// rank on.
    ///
    /// **The trailing argument decides the shape**: a list there is the weight
    /// vector, anything else is one more score — the same list-or-variadic rule
    /// `text_contains_any` uses, and unambiguous because a score is a number.
    /// The one case it cannot see is a *`null`* in the last position, which is
    /// read as an absent score rather than a missing weights list; the
    /// documented spelling is a list literal or a non-null parameter.
    ///
    /// No cache: every argument is an ordinary expression the executor has
    /// already evaluated for this row, and there is nothing to prepare once per
    /// call site the way `text_bm25` and `vector_score` prepare a query.
    fn eval_score_fuse(&self, args: &[Expression], row: &ResultRow) -> Result<Value, String> {
        const USAGE: &str = "score_fuse() takes 2 or more scores and an optional trailing weights \
                             list: score_fuse(s1, s2, … [, [w1, w2, …]])";
        if args.len() < 2 {
            return Err(USAGE.into());
        }
        // Evaluated once and kept: re-reading it as a score below would
        // evaluate a non-deterministic argument (rand(), randomUUID()) twice.
        let last = self.evaluate_expression(&args[args.len() - 1], row)?;
        let (scores, weights) = match &last {
            Value::List(items) => (&args[..args.len() - 1], Some(items.as_slice())),
            _ => (args, None),
        };
        if scores.len() < 2 {
            return Err(USAGE.into());
        }
        if let Some(weights) = weights {
            if weights.len() != scores.len() {
                return Err(format!(
                    "score_fuse(): {} weights for {} scores — the weights list needs one entry per \
                     score, in the same order",
                    weights.len(),
                    scores.len()
                ));
            }
        }

        let mut weighted_sum = 0.0f64;
        let mut weight_total = 0.0f64;
        for (position, arg) in scores.iter().enumerate() {
            // Every weight is validated, present signal or not: a malformed
            // weights list is a query bug, and which signals happen to be
            // absent on this row must not decide whether it is reported.
            let weight = match weights {
                Some(weights) => score_fuse_weight(weights, position)?,
                None => 1.0,
            };
            let value = if weights.is_none() && position + 1 == scores.len() {
                last.clone()
            } else {
                self.evaluate_expression(arg, row)?
            };
            if matches!(value, Value::Null) {
                continue;
            }
            let Some(score) = value_to_f64(&value) else {
                return Err(format!(
                    "score_fuse(): argument {} must be a number or null, got {}",
                    position + 1,
                    value.type_name()
                ));
            };
            // NaN and ±inf carry no rank position, so they mean the same thing
            // `null` does: this lane produced nothing for this row.
            if !score.is_finite() {
                continue;
            }
            weighted_sum += weight * score;
            weight_total += weight;
        }
        if weight_total == 0.0 {
            // Every signal absent, or every present signal weighted zero: the
            // mean is 0/0, and `null` is Cypher's word for undefined.
            return Ok(Value::Null);
        }
        Ok(Value::Float64(weighted_sum / weight_total))
    }
}

/// One `score_fuse` weight, validated. Rejects a non-number, a non-finite
/// value, and a negative weight: each of those is a query bug that a silent
/// substitution would turn into a plausible-looking ranking. A negative weight
/// in particular would rank *against* the lane it names while the totals still
/// look like an average.
fn score_fuse_weight(weights: &[Value], position: usize) -> Result<f64, String> {
    let value = &weights[position];
    let Some(weight) = value_to_f64(value) else {
        return Err(format!(
            "score_fuse(): weight {} must be a number, got {}",
            position + 1,
            value.type_name()
        ));
    };
    if !weight.is_finite() || weight < 0.0 {
        return Err(format!(
            "score_fuse(): weight {} must be a finite number ≥ 0, got {weight}",
            position + 1
        ));
    }
    Ok(weight)
}

/// Prefix of every [`missing_text_index_error`] message.
const NO_TEXT_INDEX_PREFIX: &str = "text_bm25(): no text index on '";

/// Prefix of every `vector_score` missing-store message
/// ([`CypherExecutor::missing_embedding_error`] and its relationship twin).
const NO_EMBEDDING_PREFIX: &str = "vector_score(): no embedding '";

/// Prefix of the same messages for a `text_score` call.
const TEXT_SCORE_NO_EMBEDDING_PREFIX: &str = "text_score(): no embedding for property '";

/// The error for `text_bm25(n, '<property>', …)` when the node's type carries no
/// text index over that property.
///
/// Ranking is opt-in, so "no index" is a hard error rather than a null column:
/// a query that silently returned null for every row would look like a corpus
/// with no matches. Names the other properties indexed on the type when there
/// are any — a misspelled property is the likely reason to be here.
fn missing_text_index_error(graph: &DirGraph, node_type: &str, prop_name: &str) -> String {
    let base = format!(
        "{NO_TEXT_INDEX_PREFIX}{node_type}.{prop_name}'. BM25 ranking is opt-in — build \
         one with build_text_index('{node_type}', '{prop_name}'); every binding reaches it \
         (Python, Rust, and the C ABI's kglite_session_build_text_index)."
    );
    let indexed: Vec<&str> = text_indexes::list_text_indexes(graph)
        .into_iter()
        .filter(|(indexed_type, _, _)| *indexed_type == node_type)
        .map(|(_, property, _)| property)
        .collect();
    if indexed.is_empty() {
        base
    } else {
        format!(
            "{base} Indexed on '{node_type}' today: {}.",
            indexed.join(", ")
        )
    }
}

impl CypherExecutor<'_> {
    /// The error for `vector_score(n, '<name>', …)` / `text_score(n,
    /// '<property>', …)` when the node's type has no such store.
    ///
    /// `text_score` reaches the scorer rewritten to `vector_score` over the
    /// store `<property>_emb`; the rewrite lists those stores under
    /// [`TEXT_SCORE_STORES_PARAM`], so the message names the function and the
    /// property the user wrote. `vector_score` is named in *store* terms while
    /// every other surface is named in *source column* terms, so a
    /// `vector_score` caller who used the column name is shown the store
    /// spelling that does exist.
    pub(in crate::graph::languages::cypher::executor) fn missing_embedding_error(
        &self,
        node_type: &str,
        prop_name: &str,
    ) -> String {
        if let Some(property) = self.text_score_property(prop_name) {
            let base =
                format!("{TEXT_SCORE_NO_EMBEDDING_PREFIX}{property}' on node type '{node_type}'");
            if let Some(hint) = store_name_hint(property, 'n', |store| {
                self.graph.embedding_store(node_type, store).is_some()
            }) {
                return format!("{base}. {hint}");
            }
            return format!(
                "{base}. Embed it first with embed_texts('{node_type}', '{property}')."
            );
        }
        let base = format!("{NO_EMBEDDING_PREFIX}{prop_name}' found for node type '{node_type}'");
        let suffixed = crate::graph::embeddings::store_name(prop_name);
        match self.graph.embedding_store(node_type, &suffixed) {
            Some(_) => format!(
                "{base}. Did you mean '{suffixed}'? vector_score() takes the embedding \
                 store name; text_score(n, '{prop_name}', <query text>) takes the text column."
            ),
            None => base,
        }
    }

    /// The relationship twin of [`Self::missing_embedding_error`].
    pub(in crate::graph::languages::cypher::executor) fn missing_edge_embedding_error(
        &self,
        relationship_type: &str,
        prop_name: &str,
    ) -> String {
        let store_exists = |store: &str| {
            self.graph
                .edge_embeddings
                .contains_key(&(relationship_type.to_string(), store.to_string()))
        };
        let Some(property) = self.text_score_property(prop_name) else {
            let base = format!(
                "{NO_EMBEDDING_PREFIX}{prop_name}' found for relationship type \
                 '{relationship_type}'"
            );
            let suffixed = crate::graph::embeddings::store_name(prop_name);
            return if store_exists(&suffixed) {
                format!(
                    "{base}. Did you mean '{suffixed}'? vector_score() takes the embedding \
                     store name; text_score(r, '{prop_name}', <query text>) takes the text column."
                )
            } else {
                base
            };
        };
        let base = format!(
            "{TEXT_SCORE_NO_EMBEDDING_PREFIX}{property}' on relationship type '{relationship_type}'"
        );
        if let Some(hint) = store_name_hint(property, 'r', store_exists) {
            return format!("{base}. {hint}");
        }
        format!(
            "{base}. Embed it first with MATCH ()-[r:{relationship_type}]->() \
             WITH collect(r) AS rs CALL db.relationship_embeddings.embed({{type: \
             '{relationship_type}', text_column: '{property}', relationships: rs}})."
        )
    }

    /// The source property a `text_score` call wrote, when `store` is one the
    /// text_score rewrite produced in this statement.
    fn text_score_property<'s>(&self, store: &'s str) -> Option<&'s str> {
        let Some(Value::List(stores)) = self.params.get(TEXT_SCORE_STORES_PARAM) else {
            return None;
        };
        stores
            .iter()
            .any(|listed| matches!(listed, Value::String(s) if s == store))
            .then(|| store.strip_suffix("_emb").unwrap_or(store))
    }
}

/// The remedy for `text_score(x, '<store name>', …)`: the caller wrote an
/// existing store's name where its text column belongs. Offering the usual
/// "embed it first" remedy there would embed a property named after the store
/// and create an empty `<store>_emb`, after which the query returns null.
/// `None` when `property` is not the name of an existing store.
fn store_name_hint(
    property: &str,
    variable: char,
    store_exists: impl Fn(&str) -> bool,
) -> Option<String> {
    let column = crate::graph::embeddings::text_column_of(property)?;
    store_exists(property).then(|| {
        format!(
            "'{property}' is the embedding store of '{column}'. Did you mean '{column}'? \
             text_score({variable}, '{column}', <query text>) takes the text column."
        )
    })
}

/// `embedding_norm`'s tail: the L2 norm of a vector, or `Null` for an entity
/// that has none.
fn norm_of(embedding: Option<&[f32]>) -> Value {
    match embedding {
        Some(vector) => Value::Float64(vector.iter().map(|x| x * x).sum::<f32>().sqrt() as f64),
        None => Value::Null,
    }
}
