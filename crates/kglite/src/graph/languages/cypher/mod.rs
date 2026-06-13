// crates/kglite/src/graph/languages/cypher/mod.rs
// Cypher query language implementation for the kglite engine.
//
// Architecture:
//   Query String -> Tokenizer -> Parser -> AST -> Planner -> Executor -> Result
//
// The MATCH clause delegates pattern parsing to
// crate::graph::core::pattern_matching::parse_pattern() — WHERE /
// RETURN / ORDER BY etc. are handled by the Cypher-level parser
// and executor.
//
// The Python-facing conversion helpers (py_convert.rs) live in
// the kglite-py wrapper crate — they're not part of the engine.

pub mod ast;
pub mod executor;
pub mod parse_cache;
pub mod parser;
pub mod planner;
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
pub use planner::schema_check::{validate_schema, warn_unknown_pattern_refs};
pub use planner::simplification::rewrite_text_score;
pub use result::CypherResult;

use crate::datatypes::values::Value;
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;

use ast::*;

/// Estimate the number of rows a MATCH clause will produce based
/// on type_indices.
fn estimate_match_rows(m: &MatchClause, graph: &DirGraph) -> Option<usize> {
    let types = collect_node_types(m);
    if types.is_empty() {
        // Untyped scan — total node count
        Some(graph.graph.node_count())
    } else {
        // Use the smallest type's count as the estimate (join
        // selectivity heuristic)
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

/// Parse a query and classify whether it mutates the graph. Returns
/// `(parsed, is_mutation)`. Convenience for the "every binding
/// pre-parses to check mutation status before applying its
/// per-binding policy" pattern.
///
/// Each binding still owns its policy:
/// - MCP server rejects all mutations on the `cypher_query` tool
/// - Bolt server rejects auto-commit mutations + reject any mutation
///   when `--readonly` is set
/// - Python wheel allows mutations unless the graph is `read_only`
///
/// What's shared is the SEQUENCE: parse, classify, then decide. This
/// helper bundles those two steps so call-sites become one line plus
/// the policy check.
///
/// Lifted from `kglite-bolt-server::backend.rs` +
/// `kglite-mcp-server::tools.rs` in 2026-05-25 — both wrote the same
/// `parse_cypher() + is_mutation_query()` pair identically.
pub fn parse_with_mutation_check(
    query: &str,
) -> Result<(ast::CypherQuery, bool), crate::error::KgError> {
    let parsed = parse_cypher(query)?;
    let is_mutation = is_mutation_query(&parsed);
    Ok((parsed, is_mutation))
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

    // (Optimisation tags collected here in the previous tree; kept
    // as a metadata-row placeholder for future EXPLAIN extensions.
    // Behavior unchanged from pre-G state — these tags were
    // collected but not yet surfaced into the result rows.)

    result::CypherResult {
        columns: vec!["step".into(), "operation".into(), "estimated_rows".into()],
        rows,
        stats: None,
        profile: None,
        diagnostics: None,
        lazy: None,
    }
}
