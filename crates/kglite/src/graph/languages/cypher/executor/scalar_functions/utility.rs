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

                // Get or initialize cache — constant args parsed once, reused for all rows
                let c = match self.vs_cache.get() {
                    Some(c) => c,
                    None => {
                        let prop_name = match self.evaluate_expression(&args[1], row)? {
                            Value::String(s) => s,
                            _ => return Err(
                                "vector_score(): second argument must be a string property name"
                                    .into(),
                            ),
                        };
                        let query_vec = self.extract_float_list(&args[2], row)?;
                        // Resolve metric: explicit arg > stored metric > cosine default
                        let metric_name = if args.len() > 3 {
                            match self.evaluate_expression(&args[3], row)? {
                                Value::String(s) => s,
                                _ => "cosine".to_string(),
                            }
                        } else {
                            // Look up stored metric from the embedding store
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
                        let _ = self.vs_cache.set(VectorScoreCache {
                            prop_name,
                            query_vec,
                            scorer,
                        });
                        self.vs_cache.get().unwrap()
                    }
                };

                // Per-row: look up node type → embedding store → compute similarity
                let node_type = match self.graph.graph.node_weight(node_idx) {
                    Some(n) => n.node_type_str(&self.graph.interner),
                    None => return Ok(Some(Value::Null)),
                };

                let store = match self.graph.embedding_store(node_type, &c.prop_name) {
                    Some(s) => s,
                    None => {
                        return Err(format!(
                            "vector_score(): no embedding '{}' found for node type '{}'",
                            c.prop_name, node_type
                        ))
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
            // ── Timeseries functions ──────────────────────────────────────
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
            // localdatetime() / localtime() / time() — wall-clock
            // temporal "now" values. KGLite's Value has no time-of-day
            // variant (Value::DateTime wraps a date-only NaiveDate; see
            // datatypes/values.rs), so these return ISO-8601 *strings*
            // rather than lying about sub-day precision via a date-only
            // Value::DateTime. The no-arg form returns local now; the
            // single-string form validates/normalises and returns Null on
            // unparseable input (mirrors datetime()'s Null-on-bad-input
            // contract). Classified like datetime() — NOT added to the
            // is_row_independent non-deterministic list, matching its
            // sibling (the 0-arg "now" forms are evaluated per the same
            // folding rules datetime() already follows).
            "rand" | "random" => {
                // Top 53 bits → f64 mantissa to avoid precision loss.
                let x = next_random_u64();
                let val = ((x >> 11) as f64) / ((1u64 << 53) as f64);
                Ok(Value::Float64(val))
            }

            // ── Temporal filtering functions ──────────────────────────────
            "valid_at" => {
                // valid_at(entity, date, 'from_field', 'to_field') → Boolean
                // True when entity.from_field <= date AND entity.to_field >= date.
                // NULL fields = open-ended (always pass).
                if args.len() != 4 {
                    return Err(
                        "valid_at() requires 4 arguments: (entity, date, from_field, to_field)"
                            .into(),
                    );
                }
                let var_name =
                    match &args[0] {
                        Expression::Variable(v) => v,
                        _ => return Err(
                            "valid_at(): first argument must be a node or relationship variable"
                                .into(),
                        ),
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
                // NULL = open-ended boundary
                let from_ok = match &from_val {
                    Value::Null => true,
                    _ => {
                        evaluate_comparison(&from_val, &ComparisonOp::LessThanEq, &date_val, None)?
                    }
                };
                let to_ok = match &to_val {
                    Value::Null => true,
                    _ => {
                        evaluate_comparison(&to_val, &ComparisonOp::GreaterThanEq, &date_val, None)?
                    }
                };
                Ok(Value::Boolean(from_ok && to_ok))
            }
            "valid_during" => {
                // valid_during(entity, start, end, 'from_field', 'to_field') → Boolean
                // Overlap: entity.from_field <= end AND entity.to_field >= start.
                // NULL fields = open-ended (always pass).
                if args.len() != 5 {
                    return Err(
                        "valid_during() requires 5 arguments: (entity, start, end, from_field, to_field)"
                            .into(),
                    );
                }
                let var_name = match &args[0] {
                    Expression::Variable(v) => v,
                    _ => return Err(
                        "valid_during(): first argument must be a node or relationship variable"
                            .into(),
                    ),
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
                // Overlap: entity.from <= query_end AND entity.to >= query_start
                let from_ok = match &from_val {
                    Value::Null => true,
                    _ => evaluate_comparison(&from_val, &ComparisonOp::LessThanEq, &end_val, None)?,
                };
                let to_ok = match &to_val {
                    Value::Null => true,
                    _ => evaluate_comparison(
                        &to_val,
                        &ComparisonOp::GreaterThanEq,
                        &start_val,
                        None,
                    )?,
                };
                Ok(Value::Boolean(from_ok && to_ok))
            }

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
                let node_type = match self.graph.graph.node_weight(node_idx) {
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
