// src/graph/cypher/mod.rs
// Cypher query language implementation for kglite
//
// Architecture:
//   Query String -> Tokenizer -> Parser -> AST -> Planner -> Executor -> Result
//
// The MATCH clause delegates pattern parsing to crate::graph::core::pattern_matching::parse_pattern()
// WHERE/RETURN/ORDER BY etc. are handled by the Cypher-level parser and executor.

pub mod ast;
pub mod executor;
pub mod parse_cache;
pub mod parser;
pub mod planner;
pub mod py_convert;
pub mod result;
pub mod tokenizer;
mod window;

// Re-exports for convenience.
//
// Phase A.3 / 0.9.53 (Issue #2): `parse_cypher` is the cached wrapper.
// Direct callers (cypher() / Transaction.cypher() / mcp-server) all go
// through the cache. The raw uncached parser lives at
// `parser::parse_cypher`; only the cache implementation itself and a
// handful of planner-internal unit tests bypass the cache.
pub use ast::OutputFormat;
pub use executor::{execute_mutable, is_mutation_query, CypherExecutor};
pub use parse_cache::parse_cypher_cached as parse_cypher;
pub use planner::mark_lazy_eligibility;
pub use planner::optimize;
pub use planner::schema_check::validate_schema;
pub use planner::simplification::rewrite_text_score;
pub use result::CypherResult;

use crate::datatypes::values::Value;
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;

use ast::*;

/// Estimate the number of rows a MATCH clause will produce based on type_indices.
fn estimate_match_rows(m: &MatchClause, graph: &DirGraph) -> Option<usize> {
    let types = collect_node_types(m);
    if types.is_empty() {
        // Untyped scan — total node count
        Some(graph.graph.node_count())
    } else {
        // Use the smallest type's count as the estimate (join selectivity heuristic)
        types
            .iter()
            .map(|t| graph.type_indices.get(t.as_str()).map_or(0, |v| v.len()))
            .min()
    }
}

/// Collect node types from a MatchClause's patterns.
fn collect_node_types(m: &MatchClause) -> Vec<String> {
    use crate::graph::core::pattern_matching::PatternElement;
    let mut types = Vec::new();
    for pattern in &m.patterns {
        for element in &pattern.elements {
            if let PatternElement::Node(np) = element {
                if let Some(ref t) = np.node_type {
                    types.push(t.clone());
                }
            }
        }
    }
    types
}

/// Generate a structured query plan as a CypherResult with columns
/// [step, operation, estimated_rows].
pub fn generate_explain_result(query: &CypherQuery, graph: &DirGraph) -> result::CypherResult {
    let mut rows = Vec::new();

    for (i, clause) in query.clauses.iter().enumerate() {
        let step = (i + 1) as i64;
        let operation = executor::clause_display_name(clause);
        let est = match clause {
            Clause::Match(m) | Clause::OptionalMatch(m) => estimate_match_rows(m, graph)
                .map(|e| Value::Int64(e as i64))
                .unwrap_or(Value::Null),
            Clause::FusedCountAll { .. }
            | Clause::FusedMatchReturnAggregate { .. }
            | Clause::FusedOptionalMatchAggregate { .. }
            | Clause::FusedCountTypedEdge { .. }
            | Clause::FusedCountAnchoredEdges { .. } => Value::Int64(1),
            Clause::FusedCountTypedNode { node_type, .. } => {
                let n = graph
                    .type_indices
                    .get(node_type.as_str())
                    .map_or(0, |v| v.len());
                Value::Int64(n.min(1) as i64)
            }
            Clause::FusedCountByType { .. } => Value::Int64(graph.type_indices.len() as i64),
            Clause::FusedVectorScoreTopK { limit, .. }
            | Clause::FusedOrderByTopK { limit, .. }
            | Clause::FusedNodeScanTopK { limit, .. } => Value::Int64(*limit as i64),
            _ => Value::Null,
        };

        rows.push(vec![Value::Int64(step), Value::String(operation), est]);
    }

    // Add optimizations as a final metadata row
    let mut optimizations = Vec::new();
    for clause in &query.clauses {
        match clause {
            Clause::FusedOptionalMatchAggregate { .. } => {
                optimizations.push("optional_match_fusion");
            }
            Clause::FusedVectorScoreTopK { .. } => {
                optimizations.push("vector_score_topk_fusion");
            }
            Clause::FusedOrderByTopK { .. } => {
                optimizations.push("order_by_topk_fusion");
            }
            Clause::FusedNodeScanTopK { .. } => {
                optimizations.push("node_scan_topk_fusion");
            }
            Clause::FusedCountAll { .. }
            | Clause::FusedCountByType { .. }
            | Clause::FusedCountEdgesByType { .. }
            | Clause::FusedCountTypedNode { .. }
            | Clause::FusedCountTypedEdge { .. }
            | Clause::FusedCountAnchoredEdges { .. }
            | Clause::FusedMatchReturnAggregate { .. } => {
                optimizations.push("count_fusion");
            }
            _ => {}
        }
    }

    result::CypherResult {
        columns: vec!["step".into(), "operation".into(), "estimated_rows".into()],
        rows,
        stats: None,
        profile: None,
        diagnostics: None,
        lazy: None,
    }
}
