//! BM25 postings retrieval with one guarded population-membership proof.

use super::retrieval::RetrievalPopulation;
use super::*;
use crate::graph::storage::disk::type_index::TypeNodesRef;
use crate::graph::text_indexes::{text_index_store, TextIndexRead};
use petgraph::graph::NodeIndex;

enum TextRowLookup<'r> {
    Rows(Vec<u32>),
    WholeType(&'r TypeNodesRef<'r>),
}

impl TextRowLookup<'_> {
    fn position(&self, node: NodeIndex) -> Option<usize> {
        match self {
            Self::Rows(slots) => slots.binary_search(&(node.index() as u32)).ok(),
            Self::WholeType(nodes) => {
                let (mut low, mut high) = (0, nodes.len());
                while low < high {
                    let mid = low + (high - low) / 2;
                    match nodes.get(mid)?.index().cmp(&node.index()) {
                        std::cmp::Ordering::Less => low = mid + 1,
                        std::cmp::Ordering::Greater => high = mid,
                        std::cmp::Ordering::Equal => return Some(mid),
                    }
                }
                None
            }
        }
    }
}

impl CypherExecutor<'_> {
    pub(super) fn try_text_retrieval_entry(
        &self,
        matched: &MatchClause,
        return_clause: &ReturnClause,
        score_item_index: usize,
        sort_keys: &[FusedSortKey],
        limit: usize,
    ) -> Result<Option<ResultSet>, String> {
        let [key] = sort_keys else {
            return Ok(None);
        };
        if limit == 0
            || key.ascending
            || key.nulls != NullsPlacement::First
            || key.return_item != Some(score_item_index)
        {
            return Ok(None);
        }
        let Some(population) = self.plain_retrieval_population(matched)? else {
            return Ok(None);
        };
        let score_expr =
            self.fold_constants_expr(&return_clause.items[score_item_index].expression);
        self.try_text_index_fused_top_k(
            &score_expr,
            true,
            limit,
            &population,
            return_clause,
            score_item_index,
        )
    }

    /// Top-k uses the scalar's score kernel and summation order. Its slot tie
    /// order equals row tie order only for a complete, strictly ascending
    /// population. Missing documents score NULL and outrank positives under
    /// DESC, so count equality alone cannot authorize this shortcut.
    ///
    /// Row-dependent arguments, ASC and underfilled postings retain scalar
    /// ranking, including zero/NULL fill. Whole-type trials additionally leave
    /// stale-index refresh and warnings to the established materialized route.
    pub(super) fn try_text_index_fused_top_k(
        &self,
        score_expr: &Expression,
        descending: bool,
        limit: usize,
        population: &RetrievalPopulation<'_>,
        return_clause: &ReturnClause,
        score_item_index: usize,
    ) -> Result<Option<ResultSet>, String> {
        if !descending || limit == 0 || population.len() == 0 {
            return Ok(None);
        }
        let Expression::FunctionCall { name, args, .. } = score_expr else {
            return Ok(None);
        };
        if name != "text_bm25" || args.len() != 3 {
            return Ok(None);
        }
        let Expression::Variable(variable) = &args[0] else {
            return Ok(None);
        };
        if ArgKey::of(&args[1]).is_none() || ArgKey::of(&args[2]).is_none() {
            return Ok(None);
        }
        let first_row = population.row(0);
        let Some(&first_idx) = first_row.node_bindings.get(variable) else {
            return Ok(None);
        };
        let Some(node) = self.graph.graph.node_view(first_idx) else {
            return Ok(None);
        };
        let node_type = node.node_type_str(&self.graph.interner);
        if matches!(population, RetrievalPopulation::WholeType { .. })
            && !self.clean_text_entry_store(args, &first_row, node_type, population.len(), limit)?
        {
            return Ok(None);
        }
        let cache = self.prepare_text_bm25(args, &first_row, node_type)?;
        if cache.query_text.is_none() {
            return Ok(None);
        }
        let Some(store) = text_index_store(self.graph, node_type, &cache.prop_name) else {
            return Ok(None);
        };
        let view = store.read();
        if store.generation() != cache.generation
            || view.documents() != population.len()
            || limit > view.documents()
        {
            return Ok(None);
        }
        let hits = view.top_k(&cache.prepared, limit);
        // Actual underfill is known only after query preparation/search. Bail
        // before the population walk; scalar ranking supplies rows scoring zero.
        if hits.len() < limit {
            return Ok(None);
        }
        let Some(lookup) = self.text_population_coverage(variable, node_type, population, &view)?
        else {
            return Ok(None);
        };
        let scored: Vec<_> = hits
            .into_iter()
            .filter_map(|(node, score)| {
                lookup
                    .position(node)
                    .map(|position| (position, Value::Float64(score)))
            })
            .collect();
        drop(view);
        if scored.len() < limit {
            return Ok(None);
        }
        self.project_retrieval_winners(
            scored.into_iter(),
            score_expr,
            population,
            return_clause,
            score_item_index,
        )
        .map(Some)
    }

    fn clean_text_entry_store(
        &self,
        args: &[Expression],
        row: &ResultRow,
        node_type: &str,
        candidates: usize,
        limit: usize,
    ) -> Result<bool, String> {
        let Value::String(property) = self.evaluate_expression(&args[1], row)? else {
            return Ok(false);
        };
        let Some(store) = text_index_store(self.graph, node_type, &property) else {
            return Ok(false);
        };
        if store.is_stale(self.graph) {
            return Ok(false);
        }
        let view = store.read();
        Ok(view.documents() == candidates && limit <= view.documents())
    }

    fn text_population_coverage<'r>(
        &self,
        variable: &str,
        node_type: &str,
        population: &'r RetrievalPopulation<'_>,
        view: &TextIndexRead<'_>,
    ) -> Result<Option<TextRowLookup<'r>>, String> {
        match population {
            RetrievalPopulation::Rows(rows) => {
                let mut slots = Vec::with_capacity(rows.rows.len());
                let nodes = rows.rows.iter().map(|row| {
                    let node = row.node_bindings.get(variable).copied();
                    if let Some(node) = node {
                        slots.push(node.index() as u32);
                    }
                    node
                });
                Ok(self
                    .ordered_text_membership(nodes, node_type, view)?
                    .then_some(TextRowLookup::Rows(slots)))
            }
            RetrievalPopulation::WholeType { nodes, .. } => Ok(self
                .ordered_text_membership(nodes.iter().map(Some), node_type, view)?
                .then_some(TextRowLookup::WholeType(nodes))),
        }
    }

    /// Count equality is checked under the same guard before this walk.
    /// Strict ordering proves uniqueness and preserves scalar tie order;
    /// membership of every node then proves equality of the two populations.
    fn ordered_text_membership(
        &self,
        nodes: impl Iterator<Item = Option<NodeIndex>>,
        node_type: &str,
        view: &TextIndexRead<'_>,
    ) -> Result<bool, String> {
        let type_key = InternedKey::from_str(node_type);
        let mut previous = None;
        for (position, node) in nodes.enumerate() {
            if position % INTERRUPT_POLL_INTERVAL == 0 {
                self.check_deadline()?;
            }
            let Some(node) = node else {
                return Ok(false);
            };
            let slot = node.index();
            if previous.is_some_and(|last| last >= slot)
                || self.graph.graph.node_type_of(node) != Some(type_key)
                || !view.contains_node(node)
            {
                return Ok(false);
            }
            previous = Some(slot);
        }
        Ok(true)
    }
}
