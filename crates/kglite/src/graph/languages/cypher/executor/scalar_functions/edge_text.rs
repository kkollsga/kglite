//! `text_bm25(r, 'property', 'query')` on a relationship — the scalar's
//! relationship arm, kept beside the node arm in `utility.rs` rather than
//! inside it so the node path is untouched.
//!
//! A relationship arrives as a MATCH binding or as a value (`collect(r)[0]`,
//! `UNWIND`, the `relationship` column of `db.relationship_embeddings.query`, a
//! `CALL { }` column). A value resolves through
//! `projected_relationship_binding`, and either shape goes through
//! `relationship_binding_is_current`, so a binding whose slot was retired and
//! reused earlier in the statement scores null instead of scoring the slot's
//! new owner — the contract `vector_score(r, …)` already keeps.
use super::super::*;
use crate::datatypes::values::Value;
use crate::graph::storage::GraphRead;
use crate::graph::text_indexes::edge_text::{edge_text_index_store, refresh_edge_text_index};

impl<'a> CypherExecutor<'a> {
    /// `text_bm25` whose first argument is not a bound node: a relationship
    /// binding or value is scored against the relationship text index, a
    /// null or any other unbound name is null, and anything else is refused.
    pub(super) fn eval_text_bm25_non_node(
        &self,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        if let Expression::Variable(var) = &args[0] {
            if let Some(edge) = row.edge_bindings.get(var) {
                return self.eval_edge_text_bm25(args, row, *edge);
            }
        }
        match self.evaluate_expression(&args[0], row)? {
            Value::Relationship(relationship) => {
                let edge = self.projected_relationship_binding(&relationship)?;
                self.eval_edge_text_bm25(args, row, edge)
            }
            // An unbound OPTIONAL MATCH name, or a carried node value: the
            // node lane scores only bound nodes, and has always answered null
            // here.
            Value::Null | Value::Node(_) | Value::NodeRef(_) => Ok(Value::Null),
            other => Err(format!(
                "text_bm25(): first argument must be a node or a relationship, got {}",
                other.type_name()
            )),
        }
    }

    fn eval_edge_text_bm25(
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
        let rel_type = weight.connection_type_str(&self.graph.interner);
        if let Some(cache) = self.tb_cache.get() {
            if cache.relationship
                && cache.node_type == rel_type
                && cache.keys.as_ref().is_some_and(|(property, query)| {
                    property.matches(&args[1]) && query.matches(&args[2])
                })
            {
                return self.score_edge_text_bm25_row(cache, edge.edge_index);
            }
        }
        let cache = self.prepare_edge_text_bm25(args, row, rel_type)?;
        let scored = self.score_edge_text_bm25_row(&cache, edge.edge_index);
        if cache.keys.is_some() {
            let _ = self.tb_cache.set(cache);
        }
        scored
    }

    /// The relationship twin of `score_text_bm25_row`: the same generation
    /// check, the same null (no document) / `0.0` (no shared term) split.
    fn score_edge_text_bm25_row(
        &self,
        cache: &TextBm25Cache,
        edge: petgraph::graph::EdgeIndex,
    ) -> Result<Value, String> {
        let Some(query_text) = cache.query_text.as_deref() else {
            return Ok(Value::Null);
        };
        let Some(store) = edge_text_index_store(self.graph, &cache.node_type, &cache.prop_name)
        else {
            return Err(missing_edge_text_index_error(
                &cache.node_type,
                &cache.prop_name,
            ));
        };
        let view = store.read();
        let score = if store.generation() == cache.generation {
            view.score_relationship(edge, &cache.prepared)
        } else {
            view.score_relationship(edge, &view.prepare_query(query_text))
        };
        Ok(score.map_or(Value::Null, Value::Float64))
    }

    /// The relationship twin of `prepare_text_bm25`, with the same query-entry
    /// freshness policy: a delta within `auto_refresh_limit` is folded in, a
    /// larger one (or a read-only graph) is served as it stands with a warning.
    fn prepare_edge_text_bm25(
        &self,
        args: &[Expression],
        row: &ResultRow,
        rel_type: &str,
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
        let Some(store) = edge_text_index_store(self.graph, rel_type, &prop_name) else {
            return Err(missing_edge_text_index_error(rel_type, &prop_name));
        };
        if store.relationship_is_stale(self.graph) {
            if !self.graph.read_only && store.relationship_can_auto_refresh(self.graph) {
                refresh_edge_text_index(self.graph, rel_type, &prop_name);
            } else {
                let reason = if self.graph.read_only {
                    "and this graph is read-only, so a query cannot catch it up".to_string()
                } else {
                    format!(
                        "over its auto_refresh_limit of {}",
                        store.auto_refresh_limit()
                    )
                };
                self.warn(format!(
                    "relationship text index '{rel_type}.{prop_name}' is stale: up to {} \
                     documents are unindexed, {reason} — a new relationship scores null and a \
                     changed one scores its previously indexed text. Refresh with \
                     CALL db.relationship_text_index.refresh({{type: '{rel_type}', text_column: \
                     '{prop_name}'}}).",
                    store.relationship_delta_size(self.graph),
                ));
            }
        }
        let view = store.read();
        let generation = store.generation();
        let prepared = match query_text.as_deref() {
            Some(text) => view.prepare_query(text),
            None => Default::default(),
        };
        drop(view);
        Ok(TextBm25Cache {
            node_type: rel_type.to_string(),
            relationship: true,
            keys: ArgKey::of(&args[1]).zip(ArgKey::of(&args[2])),
            query_text,
            prepared,
            prop_name,
            generation,
        })
    }
}

/// The error for `text_bm25(r, '<property>', …)` when the relationship type
/// carries no text index over that property. Starts with the node lane's
/// prefix, so both lanes read alike.
fn missing_edge_text_index_error(rel_type: &str, prop_name: &str) -> String {
    format!(
        "text_bm25(): no text index on '{rel_type}.{prop_name}' (relationship). BM25 ranking \
         is opt-in — build one with CALL db.relationship_text_index.build({{type: '{rel_type}', \
         property: '{prop_name}'}})."
    )
}
