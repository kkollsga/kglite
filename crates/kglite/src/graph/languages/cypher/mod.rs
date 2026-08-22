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
pub mod dynamic_labels;
pub mod executor;
pub mod parse_cache;
pub mod parser;
pub mod plan_cache;
pub mod planner;
pub mod result;
#[cfg(test)]
mod stack_probe;
pub mod tokenizer;
pub mod value_codec;
mod window;

// Re-exports for convenience.
//
// Phase A.3 / 0.9.53 (Issue #2): `parse_cypher` is the cached wrapper.
// Direct callers (cypher() / Transaction.cypher() / mcp-server) all go
// through the cache. The raw uncached parser lives at
// `parser::parse_cypher`; only the cache implementation itself and a
// handful of planner-internal unit tests bypass the cache.
// `execute_mutable` is re-exported by `api::cypher` straight from
// `executor::write`; the session layer reaches for
// `execute_mutable_with_csv`, which carries the LOAD CSV capability.
pub use executor::{is_mutation_query, CypherExecutor};
pub use parse_cache::parse_cypher_cached as parse_cypher;
pub use planner::mark_lazy_eligibility;
pub use planner::optimize;
pub use planner::schema_check::validate_schema;
pub(crate) use planner::schema_check::{
    collect_query_warnings, emit_query_warnings, strict_read_error,
};
pub use planner::simplification::rewrite_text_score;

use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{
    EdgeDirection, EdgePattern, NodePattern, Pattern, PatternElement,
};
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

/// Return the distinct parameter names referenced by Cypher source.
///
/// Names are returned in first-appearance order. Parameter-looking text inside
/// quoted string literals or `//` comments is ignored because collection goes
/// through the canonical Cypher tokenizer. This performs lexical validation;
/// callers that also need the query AST should parse the query separately.
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
pub fn parameter_names(query: &str) -> Result<Vec<String>, crate::error::KgError> {
    let positioned = tokenizer::tokenize_cypher_with_positions(query).map_err(|message| {
        crate::error::KgError::CypherSyntax {
            message,
            line: None,
            col: None,
        }
    })?;

    let mut names = Vec::new();
    for (token, _) in positioned.tokens {
        if let tokenizer::CypherToken::Parameter(name) = token {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    Ok(names)
}

/// Security- and transport-relevant facts derived from one parsed query.
///
/// Bindings use this narrow summary to enforce policies without exposing or
/// duplicating the internal clause AST. Literal limits are collected from the
/// full query tree (including set-operation branches and `CALL {}` bodies) in
/// source traversal order; parameterized or computed limits are not literals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryFeatures {
    pub is_mutation: bool,
    pub explain: bool,
    pub profile: bool,
    pub format_csv: bool,
    pub has_load_csv: bool,
    pub literal_limits: Vec<i64>,
}

/// Parse Cypher once and return the policy-relevant query features.
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
pub fn query_features(query: &str) -> Result<QueryFeatures, crate::error::KgError> {
    let parsed = parse_cypher(query)?;
    let mut features = QueryFeatures {
        is_mutation: is_mutation_query(&parsed),
        explain: false,
        profile: false,
        format_csv: false,
        has_load_csv: false,
        literal_limits: Vec::new(),
    };
    collect_query_features(&parsed, &mut features);
    Ok(features)
}

fn collect_query_features(query: &CypherQuery, features: &mut QueryFeatures) {
    features.explain |= query.explain;
    features.profile |= query.profile;
    features.format_csv |= query.output_format == OutputFormat::Csv;
    for clause in &query.clauses {
        match clause {
            Clause::LoadCsv(_) => features.has_load_csv = true,
            Clause::Limit(limit) => {
                if let Expression::Literal(Value::Int64(value)) = &limit.count {
                    features.literal_limits.push(*value);
                }
            }
            Clause::CallSubquery { body, .. } => collect_query_features(body, features),
            Clause::Union(union) => collect_query_features(&union.query, features),
            _ => {}
        }
    }
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
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
pub fn parse_with_mutation_check(
    query: &str,
) -> Result<(ast::CypherQuery, bool), crate::error::KgError> {
    let parsed = parse_cypher(query)?;
    let is_mutation = is_mutation_query(&parsed);
    Ok((parsed, is_mutation))
}

/// Render the node side of an expansion as `(:Type)` / `(:A:B)` / `()`.
fn explain_node_display(node: Option<&NodePattern>) -> String {
    let Some(node) = node else {
        return "()".to_string();
    };
    let Some(primary) = node.node_type.as_deref() else {
        return "()".to_string();
    };
    let mut out = format!("(:{primary}");
    for label in &node.extra_labels {
        out.push(':');
        out.push_str(label);
    }
    out.push(')');
    out
}

/// Render an edge pattern's type slot as `:A`, `:A|B`, or the empty string.
fn explain_edge_types(edge: &EdgePattern) -> String {
    match edge.connection_types.as_deref() {
        Some(types) if !types.is_empty() => format!(":{}", types.join("|")),
        _ => edge
            .connection_type
            .as_deref()
            .map(|t| format!(":{t}"))
            .unwrap_or_default(),
    }
}

/// One `Expand` operator row per variable-length edge in `patterns`, in
/// pattern order.
///
/// A variable-length edge is a [`PatternElement`] *inside* a MATCH clause, so
/// the clause-granular plan above can never show it: `MATCH (a)-[:R*2..3]->(b)`
/// and `MATCH (a)-[:R]->(b)` produce the identical `Match` row even though the
/// former is where the whole query's cost lives. These rows make the expansion
/// visible. `estimated_rows` stays `Null` — no cardinality model covers
/// variable-length expansion (the cost model is predicate-only and join
/// ordering excludes var-length), and a fabricated number would be worse than
/// none.
fn var_length_expand_ops(patterns: &[Pattern]) -> Vec<String> {
    let node_at = |pattern: &Pattern, idx: Option<usize>| -> String {
        let node = idx
            .and_then(|i| pattern.elements.get(i))
            .and_then(|e| match e {
                PatternElement::Node(n) => Some(n),
                PatternElement::Edge(_) => None,
            });
        explain_node_display(node)
    };

    let mut ops = Vec::new();
    for pattern in patterns {
        for (i, element) in pattern.elements.iter().enumerate() {
            let PatternElement::Edge(edge) = element else {
                continue;
            };
            let Some((min, max)) = edge.var_length else {
                continue;
            };
            let left = node_at(pattern, i.checked_sub(1));
            let right = node_at(pattern, Some(i + 1));
            let body = format!("[{}*{min}..{max}]", explain_edge_types(edge));
            let rendered = match edge.direction {
                EdgeDirection::Outgoing => format!("{left}-{body}->{right}"),
                EdgeDirection::Incoming => format!("{left}<-{body}-{right}"),
                EdgeDirection::Both => format!("{left}-{body}-{right}"),
            };
            ops.push(format!("Expand {rendered}"));
        }
    }
    ops
}

/// Generate a structured query plan as a CypherResult with columns
/// [step, operation, estimated_rows].
pub fn generate_explain_result(query: &CypherQuery, graph: &DirGraph) -> result::CypherResult {
    let mut rows = Vec::new();

    for clause in query.clauses.iter() {
        // `step` numbers rows, not clauses: a MATCH carrying variable-length
        // edges contributes an Expand row per edge after its own row, and the
        // column stays contiguous.
        let step = (rows.len() + 1) as i64;
        let operation = executor::clause_display_name(clause);
        let est = match clause {
            Clause::Match(m) | Clause::OptionalMatch(m) => estimate_match_rows(m, graph)
                .map(|e| Value::Int64(e as i64))
                .unwrap_or(Value::Null),
            Clause::FusedCountAll { .. }
            | Clause::FusedCountAllEdges { .. }
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

        if let Clause::Match(m) | Clause::OptionalMatch(m) = clause {
            for op in var_length_expand_ops(&m.patterns) {
                rows.push(vec![
                    Value::Int64((rows.len() + 1) as i64),
                    Value::String(op),
                    Value::Null,
                ]);
            }
        }
    }

    for pass in &query.optimizer_tags {
        rows.push(vec![
            Value::Int64((rows.len() + 1) as i64),
            Value::String(format!("OptimizerPass {pass}")),
            Value::Null,
        ]);
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

#[cfg(test)]
mod parameter_name_tests {
    use crate::api::cypher::parameter_names;
    use crate::error::{KgError, KgErrorCode};

    #[test]
    fn ignores_comments_and_string_literals() {
        let query = r#"
            MATCH (n)
            WHERE n.name = $name
              AND n.note = 'literal $ignored'
              AND n.other = "$also_ignored"
            // $commented_out
            RETURN n
        "#;

        assert_eq!(parameter_names(query).unwrap(), ["name"]);
    }

    #[test]
    fn deduplicates_in_first_appearance_order() {
        let query = "RETURN $second, $first, $second, $third, $first";

        assert_eq!(
            parameter_names(query).unwrap(),
            ["second", "first", "third"]
        );
    }

    #[test]
    fn finds_parameters_in_nested_expressions() {
        let query = "RETURN coalesce($fallback, {items: [$first, {value: $second}]})";

        assert_eq!(
            parameter_names(query).unwrap(),
            ["fallback", "first", "second"]
        );
    }

    #[test]
    fn invalid_parameter_syntax_is_a_typed_cypher_error() {
        let error = parameter_names("RETURN $").unwrap_err();

        assert!(matches!(error, KgError::CypherSyntax { .. }));
        assert_eq!(error.code(), KgErrorCode::CypherSyntax);
    }
}

#[cfg(test)]
mod query_feature_tests {
    use crate::api::cypher::query_features;

    #[test]
    fn reports_top_level_modes_and_mutation() {
        let explain = query_features("EXPLAIN RETURN 1 FORMAT CSV").unwrap();
        assert!(explain.explain);
        assert!(explain.format_csv);
        assert!(!explain.profile);
        assert!(!explain.is_mutation);

        let mutation = query_features("CREATE (:Thing)").unwrap();
        assert!(mutation.is_mutation);
    }

    #[test]
    fn reports_load_csv_without_matching_string_or_comment_text() {
        let load = query_features("LOAD CSV FROM 'rows.csv' AS row RETURN row").unwrap();
        assert!(load.has_load_csv);

        let ordinary = query_features("// LOAD CSV\nRETURN 'LOAD CSV' AS text").unwrap();
        assert!(!ordinary.has_load_csv);
    }

    #[test]
    fn collects_only_literal_limits_across_the_query_tree() {
        let features = query_features(
            "CALL { RETURN 1 AS n LIMIT 10 } RETURN n LIMIT $outer \
             UNION ALL RETURN 2 AS n LIMIT 200",
        )
        .unwrap();
        assert_eq!(features.literal_limits, [10, 200]);

        let lookalikes = query_features("RETURN 'LIMIT 200' AS text").unwrap();
        assert!(lookalikes.literal_limits.is_empty());
    }
}

#[cfg(test)]
mod var_length_expand_tests {
    use super::{var_length_expand_ops, Clause};

    fn ops(query: &str) -> Vec<String> {
        let parsed = super::parser::parse_cypher(query).expect("parse");
        parsed
            .clauses
            .iter()
            .filter_map(|c| match c {
                Clause::Match(m) | Clause::OptionalMatch(m) => Some(m),
                _ => None,
            })
            .flat_map(|m| var_length_expand_ops(&m.patterns))
            .collect()
    }

    #[test]
    fn renders_one_row_per_var_length_edge() {
        assert_eq!(
            ops("MATCH (a:Person)-[:KNOWS*2..3]->(b:Person) RETURN a"),
            ["Expand (:Person)-[:KNOWS*2..3]->(:Person)"]
        );
    }

    #[test]
    fn fixed_length_edges_produce_no_row() {
        assert!(ops("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a").is_empty());
    }

    #[test]
    fn every_var_length_edge_appears_in_pattern_order() {
        assert_eq!(
            ops("MATCH (a:A)-[:R*1..2]->(b:B)-[:S*3..4]->(c) RETURN a"),
            ["Expand (:A)-[:R*1..2]->(:B)", "Expand (:B)-[:S*3..4]->()",]
        );
    }

    #[test]
    fn renders_direction_types_and_the_star_default_bounds() {
        assert_eq!(
            ops("MATCH (a:Person)<-[:KNOWS|LIKES*2..3]-(b) RETURN a"),
            ["Expand (:Person)<-[:KNOWS|LIKES*2..3]-()"]
        );
        assert_eq!(ops("MATCH (a)-[*]-(b) RETURN a"), ["Expand ()-[*1..10]-()"]);
    }
}
