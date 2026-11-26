//! Pair uniqueness (`distinct`) for the bulk relationship loaders.
//!
//! The frame-side companion of `ontology::distinct_gate`, run in the same
//! pre-pass as the other relationship rules so a refusal leaves the graph
//! untouched. Only a frame that folds rows **independently** can create a
//! duplicate: a merging load folds every row onto the one edge its endpoint
//! pair holds, so it can never leave two (see the table in
//! [`super::rel_constraint_gate`]). An independent frame is judged as the
//! relationships it adds: its own rows grouped by `(source, target, listed
//! property values)`, plus any stored edge of the same group.

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use super::rel_constraint_gate::{ConnectionBatchGate, FramePlan, RowFolding};
use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::ontology::distinct_gate::{
    flag_group, group_properties, group_value, interned_keys, weight_key, GroupKey,
};
use crate::graph::ontology::predicates::source_exempt;
use crate::graph::storage::GraphRead;

impl ConnectionBatchGate<'_> {
    /// The group key of frame row `row`, reading the listed properties from
    /// the frame's columns (a missing column and a null cell are both absent).
    fn distinct_key(&self, row: usize, properties: &[String]) -> GroupKey {
        properties
            .iter()
            .map(|name| {
                self.property_columns
                    .iter()
                    .find(|(column, _, _)| column == name)
                    .and_then(|(_, _, index)| self.rows.cell(row, *index))
                    .and_then(|v| group_value(Some(&v)))
            })
            .collect()
    }

    /// Judge the duplicates an independent frame would add.
    pub(super) fn judge_distinct(&self, graph: &DirGraph, plan: &mut FramePlan) {
        let Some(rules) = plan.rules.as_ref() else {
            return;
        };
        let Some(rule) = rules.distinct() else {
            return;
        };
        if self.folding != RowFolding::Independent && self.start_key.is_none() {
            return;
        }
        let properties = group_properties(graph, self.connection_type, rule.properties());
        let conn_key = crate::graph::schema::InternedKey::from_str(self.connection_type);
        let keys = interned_keys(&properties);

        let mut groups: HashMap<(NodeIndex, NodeIndex, GroupKey), usize> = HashMap::new();
        for (row, source, target) in self.matched {
            *groups
                .entry((*source, *target, self.distinct_key(*row, &properties)))
                .or_default() += 1;
        }
        let mut stored_by_source: HashMap<NodeIndex, HashMap<(NodeIndex, GroupKey), usize>> =
            HashMap::new();
        for ((source, target, key), added) in groups {
            let stored = stored_by_source.entry(source).or_insert_with(|| {
                let mut held: HashMap<(NodeIndex, GroupKey), usize> = HashMap::new();
                for edge in graph
                    .graph
                    .edges_directed(source, petgraph::Direction::Outgoing)
                    .filter(|e| e.weight().connection_type == conn_key)
                {
                    let group = weight_key(&edge.weight(), &keys);
                    *held.entry((edge.target(), group)).or_default() += 1;
                }
                held
            });
            let total = added + stored.get(&(target, key)).copied().unwrap_or(0);
            if total > 1 {
                flag_group(
                    graph,
                    rules,
                    rule,
                    self.connection_type,
                    source,
                    added.min(total - 1),
                    &mut plan.tally,
                );
                if plan.tally.is_refused() {
                    return;
                }
            }
        }

        // Rows held back for stub vivification have no stored twin.
        let Some((source_type, _)) = self.endpoint_types else {
            return;
        };
        let mut stubs: HashMap<(&Value, &Value, GroupKey), usize> = HashMap::new();
        for (row, source, target) in self.deferred {
            *stubs
                .entry((source, target, self.distinct_key(*row, &properties)))
                .or_default() += 1;
        }
        let exempt = source_exempt(
            &graph.ontology,
            rules.decl(),
            "distinct",
            source_type,
            |_| false,
        );
        for count in stubs.into_values().filter(|c| *c > 1) {
            if !exempt {
                plan.tally.flag_many(
                    rule.severity(),
                    rule.violation(self.connection_type, source_type, count - 1),
                    count - 1,
                );
            }
            if plan.tally.is_refused() {
                return;
            }
        }
    }
}
