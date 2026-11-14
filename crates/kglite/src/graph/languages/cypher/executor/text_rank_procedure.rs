//! `db.text.rank`: BM25 over a graph-selected scope, with no index.
//!
//! `CALL db.text.rank({type, property, query, within?, limit?}) YIELD node,
//! score` ranks the nodes of `within` (every node of the type without it) by
//! how well their `property` matches `query`, treating that scope as the whole
//! corpus: `N`, the mean length and each term's document frequency are the
//! scope's, computed from its text at call time. Nothing is declared, stored
//! or cached, so the cost is proportional to the scope's text.
//!
//! The answer is what a text index built over the scope's nodes alone would
//! return for the same query: same tokenizer, same scoring function (shared
//! with the heap index), same order (score descending, node slot ascending),
//! one row per document sharing a term with the query.
//!
//! Documents follow the heap index's rule: a string is itself, a list of
//! strings (nulls skipped) is its members joined by one space, and a null,
//! absent or other value is not a document (it gets no row and is not counted
//! in `N`). The corpus is the nodes whose primary type is `type`, as the heap
//! index sees them. The column driver supplies the bytes when it can prove the
//! column; otherwise the values come through `NodeView`, with the same result.

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use super::procedure_params::reject_unknown_keys;
use super::*;
use crate::graph::algorithms::text_index::scoped::ScopedBm25;
use crate::graph::core::pattern_matching::text_column_scan::{
    for_each_document, note_text_decline, note_text_served,
};
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead;
use crate::graph::text_indexes::document_text;

const PROC: &str = "db.text.rank";
const KEYS: &[&str] = &["type", "property", "query", "within", "limit"];

fn string_param(params: &HashMap<String, Value>, key: &str) -> Result<String, String> {
    match params.get(key) {
        Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
        Some(other) => Err(format!(
            "CALL {PROC}: '{key}' must be a non-empty string, got {}",
            other.type_name()
        )),
        None => Err(format!("CALL {PROC}: missing parameter '{key}'")),
    }
}

fn limit_param(params: &HashMap<String, Value>) -> Result<Option<usize>, String> {
    match params.get("limit") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Int64(n)) if *n >= 0 => Ok(Some(*n as usize)),
        Some(other) => Err(format!(
            "CALL {PROC}: 'limit' must be a non-negative integer, got {}",
            other.type_name()
        )),
    }
}

fn within(params: &HashMap<String, Value>) -> Result<Option<Vec<NodeIndex>>, String> {
    let items = match params.get("within") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::List(items)) => items,
        Some(other) => {
            return Err(format!(
                "CALL {PROC}: 'within' must be a list of nodes, got {}",
                other.type_name()
            ))
        }
    };
    items
        .iter()
        .map(|item| match item {
            Value::NodeRef(i) => Ok(NodeIndex::new(*i as usize)),
            Value::Node(node) => Ok(NodeIndex::new(node.id as usize)),
            other => Err(format!(
                "CALL {PROC}: 'within' must hold only nodes, found {}",
                other.type_name()
            )),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

impl CypherExecutor<'_> {
    pub(super) fn text_rank(
        &self,
        params: &HashMap<String, Value>,
        yields: &[YieldItem],
    ) -> Result<Vec<ResultRow>, String> {
        reject_unknown_keys(
            &format!("CALL {PROC}"),
            params.keys().map(String::as_str),
            KEYS,
        )?;
        let node_type = string_param(params, "type")?;
        let property = string_param(params, "property")?;
        let query = string_param(params, "query")?;
        let scope = within(params)?;
        let limit = limit_param(params)?;

        let mut corpus = ScopedBm25::new(&query);
        if !corpus.no_terms() {
            self.collect_documents(&node_type, &property, scope.as_deref(), &mut corpus)?;
        }
        let ranked = corpus.top(limit);

        let mut rows = Vec::with_capacity(ranked.len());
        for doc in ranked {
            let node = NodeIndex::new(doc.slot as usize);
            let mut row = ResultRow::new();
            for item in yields {
                let alias = item.alias.clone().unwrap_or_else(|| item.name.clone());
                match item.name.as_str() {
                    "node" => {
                        row.node_bindings.insert(alias, node);
                    }
                    "score" => {
                        row.projected.insert(alias, Value::Float64(doc.score));
                    }
                    _ => {}
                }
            }
            rows.push(row);
        }
        Ok(rows)
    }

    /// Feeds every document of the scope to `corpus`: from the column when the
    /// driver proves it, else through `NodeView`.
    fn collect_documents(
        &self,
        node_type: &str,
        property: &str,
        scope: Option<&[NodeIndex]>,
        corpus: &mut ScopedBm25,
    ) -> Result<(), String> {
        let interrupt = self.interrupt();
        let served = for_each_document(
            self.graph,
            node_type,
            property,
            scope,
            interrupt,
            &mut |node, bytes| {
                corpus.add_document(node.index() as u32, &String::from_utf8_lossy(bytes));
            },
        )
        .map_err(|e| e.message(&interrupt))?;
        match served {
            Ok(()) => {
                note_text_served();
                Ok(())
            }
            Err(decline) => {
                note_text_decline(decline);
                self.collect_documents_per_row(node_type, property, scope, corpus)
            }
        }
    }

    fn collect_documents_per_row(
        &self,
        node_type: &str,
        property: &str,
        scope: Option<&[NodeIndex]>,
        corpus: &mut ScopedBm25,
    ) -> Result<(), String> {
        let type_key = InternedKey::from_str(node_type);
        let candidates: Vec<NodeIndex> = match scope {
            None => self
                .graph
                .type_indices
                .get(node_type)
                .map(|n| n.to_vec())
                .unwrap_or_default(),
            Some(scope) => {
                let mut nodes: Vec<NodeIndex> = scope
                    .iter()
                    .copied()
                    .filter(|&n| {
                        self.graph
                            .graph
                            .node_weight(n)
                            .is_some_and(|data| data.node_type == type_key)
                    })
                    .collect();
                nodes.sort_unstable();
                nodes.dedup();
                nodes
            }
        };
        let field = self.graph.resolve_alias(node_type, property);
        let key = InternedKey::from_str(field);
        for (i, node) in candidates.into_iter().enumerate() {
            self.check_interrupt_periodic(i)?;
            let Some(view) = self.graph.graph.node_view(node) else {
                continue;
            };
            if let Some(text) = document_text(&view, node_type, field, key) {
                corpus.add_document(node.index() as u32, &text);
            }
        }
        Ok(())
    }
}
