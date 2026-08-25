//! Cypher scalar functions — utility category. Split out of the monolithic
//! `evaluate_scalar_function` dispatcher; arms are verbatim. Routed from
//! `super::evaluate_scalar_function`; returns `Ok(None)` when `name` is not
//! one of this category's functions so the dispatcher tries the next.
use super::super::helpers::*;
use super::super::*;
use super::shared::*;
use crate::datatypes::values::Value;
use crate::graph::algorithms::vector as vs;
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
            "vector_score" => {
                if args.len() < 3 || args.len() > 4 {
                    return Err(
                        "vector_score() requires 3-4 arguments: (node, property, query_vector [, metric])"
                            .into(),
                    );
                }

                // Arg 0: node variable → resolve to NodeIndex (changes per row)
                let node_idx = match &args[0] {
                    Expression::Variable(var) => match row.node_bindings.get(var) {
                        Some(&idx) => idx,
                        None => return Ok(Some(Value::Null)),
                    },
                    _ => {
                        return Err("vector_score(): first argument must be a node variable".into())
                    }
                };

                // The constant arguments, parsed once per call site — or per
                // row when this call's arguments are row-dependent, or when
                // every cache slot already belongs to another call site.
                let uncached;
                let c = match self.vs_cache.get(args) {
                    Some(cached) => cached,
                    None => match self.vs_cache.park(self.prepare_vector_score(args, row)?) {
                        Ok(parked) => parked,
                        Err(entry) => {
                            uncached = entry;
                            &uncached
                        }
                    },
                };

                // Per-row: look up node type → embedding store → compute similarity
                let node_type = match self.graph.graph.node_view(node_idx) {
                    Some(n) => n.node_type_str(&self.graph.interner),
                    None => return Ok(Some(Value::Null)),
                };

                let store = match self.graph.embedding_store(node_type, &c.prop_name) {
                    Some(s) => s,
                    None => {
                        return Err(missing_embedding_error(self.graph, node_type, &c.prop_name))
                    }
                };

                if c.query_vec.len() != store.dimension {
                    return Err(format!(
                        "vector_score(): query vector dimension {} does not match embedding dimension {}",
                        c.query_vec.len(),
                        store.dimension
                    ));
                }

                match store.get_embedding_with_norm(node_idx.index()) {
                    Some((embedding, norm)) => {
                        let score = c.scorer.score(&c.query_vec, embedding, norm);
                        Ok(Value::Float64(score as f64))
                    }
                    None => Ok(Value::Null),
                }
            }
            // text_bm25(n, 'property', 'query text') — BM25 relevance of one
            // row's document against a query, or null when that row has no
            // document. Lives in `utility` rather than `string` because it is
            // the same shape as its neighbours here — a *node*-and-store
            // scalar (vector_score, embedding_norm, text_score) — while
            // `string`'s text_* family are pure string→scalar functions that
            // never touch the graph.
            "text_bm25" => self.eval_text_bm25(args, row),
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
            // embedding_norm(node, property) → Float64
            // Returns the L2 norm of the node's embedding vector.
            // Useful for inferring hierarchy depth in Poincaré embeddings
            // (norm close to 0 = root/general, norm close to 1 = leaf/specific).
            "embedding_norm" => {
                if args.len() != 2 {
                    return Err("embedding_norm() requires 2 arguments: (node, property)".into());
                }
                let node_idx = match &args[0] {
                    Expression::Variable(var) => match row.node_bindings.get(var) {
                        Some(&idx) => idx,
                        None => return Ok(Some(Value::Null)),
                    },
                    _ => {
                        return Err(
                            "embedding_norm(): first argument must be a node variable".into()
                        )
                    }
                };
                let prop_name = match self.evaluate_expression(&args[1], row)? {
                    Value::String(s) => s,
                    _ => {
                        return Err(
                            "embedding_norm(): second argument must be a string property name"
                                .into(),
                        )
                    }
                };
                let node_type = match self.graph.graph.node_view(node_idx) {
                    Some(n) => n.node_type_str(&self.graph.interner),
                    None => return Ok(Some(Value::Null)),
                };
                let store = match self.graph.embedding_store(node_type, &prop_name) {
                    Some(s) => s,
                    None => {
                        return Err(format!(
                            "embedding_norm(): no embedding '{}' found for node type '{}'",
                            prop_name, node_type
                        ))
                    }
                };
                match store.get_embedding(node_idx.index()) {
                    Some(emb) => {
                        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                        Ok(Value::Float64(norm as f64))
                    }
                    None => Ok(Value::Null),
                }
            }
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
                        .map(|j| json_to_value(&j))
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
    /// `valid_at(entity, date, 'from_field', 'to_field')` → Boolean.
    /// True when `from_field <= date <= to_field`; a null field is an
    /// open-ended boundary and always passes.
    fn eval_valid_at(&self, args: &[Expression], row: &ResultRow) -> Result<Value, String> {
        if args.len() != 4 {
            return Err(
                "valid_at() requires 4 arguments: (entity, date, from_field, to_field)".into(),
            );
        }
        let var_name = match &args[0] {
            Expression::Variable(v) => v,
            _ => {
                return Err(
                    "valid_at(): first argument must be a node or relationship variable".into(),
                )
            }
        };
        let date_val = self.evaluate_expression(&args[1], row)?;
        let from_field = match self.evaluate_expression(&args[2], row)? {
            Value::String(s) => s,
            _ => return Err("valid_at(): from_field (3rd arg) must be a string".into()),
        };
        let to_field = match self.evaluate_expression(&args[3], row)? {
            Value::String(s) => s,
            _ => return Err("valid_at(): to_field (4th arg) must be a string".into()),
        };
        let from_val = self.resolve_property(var_name, &from_field, row)?;
        let to_val = self.resolve_property(var_name, &to_field, row)?;
        let from_ok = match &from_val {
            Value::Null => true,
            _ => evaluate_comparison(&from_val, &ComparisonOp::LessThanEq, &date_val)?,
        };
        let to_ok = match &to_val {
            Value::Null => true,
            _ => evaluate_comparison(&to_val, &ComparisonOp::GreaterThanEq, &date_val)?,
        };
        Ok(Value::Boolean(from_ok && to_ok))
    }

    /// `valid_during(entity, start, end, 'from_field', 'to_field')` → Boolean.
    /// True when the entity's interval overlaps `[start, end]`; a null field is
    /// an open-ended boundary and always passes.
    fn eval_valid_during(&self, args: &[Expression], row: &ResultRow) -> Result<Value, String> {
        if args.len() != 5 {
            return Err(
                "valid_during() requires 5 arguments: (entity, start, end, from_field, to_field)"
                    .into(),
            );
        }
        let var_name = match &args[0] {
            Expression::Variable(v) => v,
            _ => {
                return Err(
                    "valid_during(): first argument must be a node or relationship variable".into(),
                )
            }
        };
        let start_val = self.evaluate_expression(&args[1], row)?;
        let end_val = self.evaluate_expression(&args[2], row)?;
        let from_field = match self.evaluate_expression(&args[3], row)? {
            Value::String(s) => s,
            _ => return Err("valid_during(): from_field (4th arg) must be a string".into()),
        };
        let to_field = match self.evaluate_expression(&args[4], row)? {
            Value::String(s) => s,
            _ => return Err("valid_during(): to_field (5th arg) must be a string".into()),
        };
        let from_val = self.resolve_property(var_name, &from_field, row)?;
        let to_val = self.resolve_property(var_name, &to_field, row)?;
        let from_ok = match &from_val {
            Value::Null => true,
            _ => evaluate_comparison(&from_val, &ComparisonOp::LessThanEq, &end_val)?,
        };
        let to_ok = match &to_val {
            Value::Null => true,
            _ => evaluate_comparison(&to_val, &ComparisonOp::GreaterThanEq, &start_val)?,
        };
        Ok(Value::Boolean(from_ok && to_ok))
    }

    /// `text_bm25(node, 'property', 'query text')` → BM25 relevance of that one
    /// row's document, or null when the row has no document.
    fn eval_text_bm25(&self, args: &[Expression], row: &ResultRow) -> Result<Value, String> {
        if args.len() != 3 {
            return Err("text_bm25() requires 3 arguments: (node, property, query_text)".into());
        }
        let node_idx = match &args[0] {
            Expression::Variable(var) => match row.node_bindings.get(var) {
                Some(&idx) => idx,
                None => return Ok(Value::Null),
            },
            _ => return Err("text_bm25(): first argument must be a node variable".into()),
        };
        let node_type = match self.graph.graph.node_view(node_idx) {
            Some(n) => n.node_type_str(&self.graph.interner),
            None => return Ok(Value::Null),
        };
        if let Some(cache) = self.tb_cache.get() {
            if cache.node_type == node_type
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

    /// Parse `vector_score()`'s constant arguments — property name, query
    /// vector, and the metric (explicit argument, else the store's own, else
    /// cosine).
    ///
    /// The returned entry carries the key it was prepared under, so the caller
    /// can park it for the rest of the scan; a row-dependent argument yields a
    /// keyless entry that scores this row only.
    fn prepare_vector_score(
        &self,
        args: &[Expression],
        row: &ResultRow,
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
        let metric_name = if args.len() > 3 {
            match self.evaluate_expression(&args[3], row)? {
                Value::String(s) => s,
                _ => "cosine".to_string(),
            }
        } else {
            self.graph
                .embeddings
                .iter()
                .find(|((_, pn), _)| pn == &prop_name)
                .and_then(|(_, store)| store.metric.clone())
                .unwrap_or_else(|| "cosine".to_string())
        };
        let metric = vs::DistanceMetric::from_name(&metric_name).ok_or_else(|| {
            format!(
                "vector_score(): unknown metric '{}'. Use 'cosine', 'dot_product', 'euclidean', or 'poincare'.",
                metric_name
            )
        })?;
        let scorer = vs::Scorer::new(metric, &query_vec);
        Ok(VectorScoreCache {
            keys: VectorScoreCache::key_for(args),
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
    /// rebuild.
    fn prepare_text_bm25(
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
                    "text index '{}.{}' is stale: up to {} documents are unindexed, {} — those \
                     rows score null. Rebuild with build_text_index('{}', '{}').",
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
            keys: ArgKey::of(&args[1]).zip(ArgKey::of(&args[2])),
            query_text,
            prepared,
            prop_name,
            generation,
        })
    }
}

/// The error for `text_bm25(n, '<property>', …)` when the node's type carries no
/// text index over that property.
///
/// Ranking is opt-in, so "no index" is a hard error rather than a null column:
/// a query that silently returned null for every row would look like a corpus
/// with no matches. Names the other properties indexed on the type when there
/// are any — a misspelled property is the likely reason to be here.
fn missing_text_index_error(graph: &DirGraph, node_type: &str, prop_name: &str) -> String {
    let base = format!(
        "text_bm25(): no text index on '{node_type}.{prop_name}'. BM25 ranking is opt-in — build \
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

/// The error for `vector_score(n, '<name>', …)` when no store of that name
/// exists on the node's type.
///
/// `vector_score` is named in *store* terms (`'summary_emb'`) while every other
/// surface — `set_embeddings`, `text_score`, the Python API — is named in
/// *source column* terms (`'summary'`). A caller who reaches for the column
/// name here gets an error naming a store that does exist under the spelling
/// they didn't use, so the message hands them both ways out. When the name is
/// genuinely unknown there is nothing to suggest and the plain message stands.
fn missing_embedding_error(graph: &DirGraph, node_type: &str, prop_name: &str) -> String {
    let base =
        format!("vector_score(): no embedding '{prop_name}' found for node type '{node_type}'");
    let suffixed = crate::graph::embeddings::store_name(prop_name);
    match graph.embedding_store(node_type, &suffixed) {
        Some(_) => format!(
            "{base}. Did you mean '{suffixed}'? vector_score() takes the embedding \
             store name; text_score(n, '{prop_name}', <query text>) takes the text column."
        ),
        None => base,
    }
}
