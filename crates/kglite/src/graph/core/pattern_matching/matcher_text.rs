//! The text column driver as a candidate source for a node pattern.
//!
//! A [`PropertyMatcher::TextScan`] in a pattern's properties says "these
//! patterns find a superset of the nodes the query's text predicate accepts".
//! When no index serves the pattern, the driver supplies the candidates; the
//! caller then runs every property matcher, the text one included, over them,
//! and the predicate the matcher came from stays in the plan. So the driver
//! may return more nodes than the predicate accepts, never fewer; and a
//! decline returns `None`, which sends the pattern down the unchanged scan.

use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;

use super::PatternExecutor;
use crate::graph::algorithms::Interrupt;
use crate::graph::core::pattern_matching::pattern::PropertyMatcher;
use crate::graph::core::pattern_matching::text_column_scan::{
    note_text_decline, note_text_served, text_column_scan, TextAnswer, TextMode, TextScanRequest,
};

/// Candidates above this share of the type (a tenth, never fewer than
/// [`CAP_FLOOR`]) make the scan decline: the caller re-tests every candidate,
/// so a scan that matches most of the type saves nothing over the per-row scan.
/// Measured on a 58k-node type: a term in 23% of nodes ran 12% slower through
/// the scan, one in 1% ran 2x faster.
fn candidate_cap(graph: &crate::graph::schema::DirGraph, node_type: &str) -> usize {
    let bucket = graph.type_indices.get(node_type).map_or(0, |n| n.len());
    (bucket / 10).max(CAP_FLOOR)
}

const CAP_FLOOR: usize = 4096;

impl PatternExecutor<'_> {
    /// The pattern's nodes found through a text-scan matcher, already put
    /// through every property matcher; `None` when it does not apply.
    ///
    /// Called once no index served the pattern. A text predicate's column scan
    /// is the next cheapest source: it declines (`None`) on secondary-label
    /// carriers and every other state it cannot prove, and the candidates it
    /// does return still pass through every property matcher.
    pub(super) fn text_scan_nodes(
        &self,
        pattern: &super::NodePattern,
        extra_keys: &[crate::graph::schema::InternedKey],
        has_secondary: bool,
    ) -> Result<Option<Vec<NodeIndex>>, String> {
        let (Some(props), Some(node_type)) =
            (pattern.properties.as_ref(), pattern.node_type.as_deref())
        else {
            return Ok(None);
        };
        if !extra_keys.is_empty() || has_secondary {
            return Ok(None);
        }
        match self.try_text_scan(node_type, props)? {
            Some(found) => self
                .filter_node_candidates(&found, Some(props), extra_keys)
                .map(Some),
            None => Ok(None),
        }
    }

    /// Candidates for `node_type` from the first (by property name) text-scan
    /// matcher in `props`; `None` when there is none or the driver declined.
    pub(super) fn try_text_scan(
        &self,
        node_type: &str,
        props: &HashMap<String, PropertyMatcher>,
    ) -> Result<Option<Vec<NodeIndex>>, String> {
        let mut specs: Vec<_> = props
            .iter()
            .filter_map(|(key, matcher)| match matcher {
                PropertyMatcher::TextScan(spec) => Some((key.as_str(), spec)),
                _ => None,
            })
            .collect();
        specs.sort_by_key(|(key, _)| *key);
        let Some((property, spec)) = specs.first() else {
            return Ok(None);
        };
        let Some(matchers) = spec.matchers() else {
            return Ok(None);
        };
        let interrupt = Interrupt {
            deadline: self.deadline,
            cancel: self.cancel,
        };
        let mut sets: Vec<Vec<NodeIndex>> = Vec::with_capacity(matchers.len());
        for matcher in matchers {
            let outcome = text_column_scan(
                self.graph,
                &TextScanRequest {
                    node_type,
                    property,
                    matcher,
                    scope: None,
                    mode: TextMode::Exists,
                    interrupt,
                    serial: false,
                    candidate_cap: Some(candidate_cap(self.graph, node_type)),
                },
            );
            match outcome {
                Err(_) => {
                    return Err(self
                        .interrupt_reason()
                        .unwrap_or_else(|| "Query interrupted".to_string()))
                }
                Ok(Err(decline)) => {
                    note_text_decline(decline);
                    return Ok(None);
                }
                Ok(Ok(TextAnswer::Nodes(nodes))) => sets.push(nodes),
                Ok(Ok(_)) => return Ok(None),
            }
        }
        note_text_served();
        if sets.len() == 1 {
            return Ok(sets.pop());
        }
        // The union, in the order the type's bucket lists its nodes.
        let wanted: HashSet<NodeIndex> = sets.into_iter().flatten().collect();
        let Some(bucket) = self.graph.type_indices.get(node_type) else {
            return Ok(Some(Vec::new()));
        };
        Ok(Some(bucket.iter().filter(|n| wanted.contains(n)).collect()))
    }
}
