//! `db.text.search`: exact text matches with their positions.
//!
//! `CALL db.text.search({type, property, query, mode?, ignore_case?, within?,
//! limit?, snippet_chars?}) YIELD node, property, offset, length, char_offset,
//! snippet` — one row per occurrence, in node order and then offset order.
//! `offset` and `length` are **byte** positions in the property's current
//! value; `char_offset` counts characters. Only string values have
//! occurrences; a list or non-string value yields none.
//!
//! The column driver answers when it can prove the column. When it declines,
//! the same rows come from running the matcher over the label's nodes through
//! `NodeView`, so the answer never depends on the route. "The label's nodes"
//! are the nodes the primary type holds plus its secondary-label carriers,
//! as `MATCH (n:Label)` visits them; ontology closures are not followed.

use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;

use super::procedure_params::reject_unknown_keys;
use super::*;
use crate::datatypes::prop_map::PropMap;
use crate::graph::core::pattern_matching::text_column_scan::{
    note_text_decline, note_text_served, text_column_scan, TextAnswer, TextMode, TextScanRequest,
};
use crate::graph::schema::InternedKey;
use crate::graph::storage::{GraphRead, StrField};
use crate::graph::text_scan::spec::{snippet, TextQuery};

const PROC: &str = "db.text.search";
const KEYS: &[&str] = &[
    "type",
    "property",
    "query",
    "mode",
    "ignore_case",
    "within",
    "limit",
    "snippet_chars",
];
const DEFAULT_SNIPPET_CHARS: usize = 40;

struct Hit {
    node: NodeIndex,
    offset: usize,
    len: usize,
}

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

fn count_param(params: &HashMap<String, Value>, key: &str) -> Result<Option<usize>, String> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Int64(n)) if *n >= 0 => Ok(Some(*n as usize)),
        Some(other) => Err(format!(
            "CALL {PROC}: '{key}' must be a non-negative integer, got {}",
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
    pub(super) fn text_search(
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
        let text = string_param(params, "query")?;
        let scope = within(params)?;
        let limit = count_param(params, "limit")?;
        let context = count_param(params, "snippet_chars")?.unwrap_or(DEFAULT_SNIPPET_CHARS);
        let mut options = Vec::new();
        for key in ["mode", "ignore_case"] {
            if let Some(value) = params.get(key) {
                options.push((key.to_string(), value.clone()));
            }
        }
        let options = Value::Map(PropMap::from_pairs(options));
        let who = format!("CALL {PROC}");
        let query = TextQuery::new(&text, Some(&options), &who)?;
        let matcher = query.compile(&who)?;

        let interrupt = self.interrupt();
        let outcome = text_column_scan(
            self.graph,
            &TextScanRequest {
                node_type: &node_type,
                property: &property,
                matcher: &matcher,
                scope: scope.as_deref(),
                mode: TextMode::Hits { max: limit },
                interrupt,
                serial: false,
                candidate_cap: None,
            },
        )
        .map_err(|e| e.message(&interrupt))?;
        let hits: Vec<Hit> = match outcome {
            Ok(TextAnswer::Hits { hits, .. }) => {
                note_text_served();
                hits.into_iter()
                    .map(|h| Hit {
                        node: h.node,
                        offset: h.offset as usize,
                        len: h.len as usize,
                    })
                    .collect()
            }
            Ok(_) => unreachable!("hits mode answers with hits"),
            Err(decline) => {
                note_text_decline(decline);
                self.text_search_per_row(&node_type, &property, &matcher, scope.as_deref(), limit)?
            }
        };

        let field = self.graph.resolve_alias(&node_type, &property);
        let key = InternedKey::from_str(field);
        let mut rows = Vec::with_capacity(hits.len());
        for (i, hit) in hits.iter().enumerate() {
            self.check_interrupt_periodic(i)?;
            let value = self
                .graph
                .graph
                .node_view(hit.node)
                .map(
                    |view| match view.resolved_field_str(&node_type, field, key) {
                        StrField::Str(text) => text.into_owned(),
                        _ => String::new(),
                    },
                )
                .unwrap_or_default();
            let mut row = ResultRow::new();
            for item in yields {
                let alias = item.alias.clone().unwrap_or_else(|| item.name.clone());
                match item.name.as_str() {
                    "node" => {
                        row.node_bindings.insert(alias, hit.node);
                    }
                    "property" => {
                        row.projected.insert(alias, Value::String(property.clone()));
                    }
                    "offset" => {
                        row.projected.insert(alias, Value::Int64(hit.offset as i64));
                    }
                    "length" => {
                        row.projected.insert(alias, Value::Int64(hit.len as i64));
                    }
                    "char_offset" => {
                        let chars = value.get(..hit.offset).map_or(0, |s| s.chars().count());
                        row.projected.insert(alias, Value::Int64(chars as i64));
                    }
                    "snippet" => {
                        row.projected.insert(
                            alias,
                            Value::String(snippet(&value, hit.offset, hit.len, context)),
                        );
                    }
                    _ => {}
                }
            }
            rows.push(row);
        }
        Ok(rows)
    }

    /// The matcher over the label's nodes (or the scope's), through
    /// `NodeView`: the answer the column driver reproduces.
    fn text_search_per_row(
        &self,
        node_type: &str,
        property: &str,
        matcher: &crate::graph::text_scan::matcher::Matcher,
        scope: Option<&[NodeIndex]>,
        limit: Option<usize>,
    ) -> Result<Vec<Hit>, String> {
        let type_key = InternedKey::from_str(node_type);
        let label_nodes = |graph: &DirGraph| -> Vec<NodeIndex> {
            let mut nodes = graph
                .type_indices
                .get(node_type)
                .map(|n| n.to_vec())
                .unwrap_or_default();
            if graph.has_secondary_labels {
                if let Some(bucket) = graph.secondary_label_index.get(&type_key) {
                    nodes.extend(bucket.iter().copied());
                }
            }
            nodes
        };
        let candidates: Vec<NodeIndex> = match scope {
            None => label_nodes(self.graph),
            Some(scope) => {
                let members: HashSet<NodeIndex> = label_nodes(self.graph).into_iter().collect();
                let mut seen = HashSet::new();
                scope
                    .iter()
                    .copied()
                    .filter(|n| members.contains(n) && seen.insert(*n))
                    .collect()
            }
        };
        let field = self.graph.resolve_alias(node_type, property);
        let key = InternedKey::from_str(field);
        let mut hits = Vec::new();
        for (i, node) in candidates.into_iter().enumerate() {
            self.check_interrupt_periodic(i)?;
            let Some(view) = self.graph.graph.node_view(node) else {
                continue;
            };
            let StrField::Str(text) = view.resolved_field_str(node_type, field, key) else {
                continue;
            };
            matcher.find_in_record(text.as_bytes(), &mut |start, end| {
                hits.push(Hit {
                    node,
                    offset: start,
                    len: end - start,
                });
                limit.is_none_or(|l| hits.len() < l)
            });
            if limit.is_some_and(|l| hits.len() >= l) {
                break;
            }
        }
        Ok(hits)
    }
}
