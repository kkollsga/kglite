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
mod explain;
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
// `parse_cypher` is the cached wrapper.
// Direct callers (cypher() / Transaction.cypher() / mcp-server) all go
// through the cache. The raw uncached parser lives at
// `parser::parse_cypher`; only the cache implementation itself and a
// handful of planner-internal unit tests bypass the cache.
// `execute_mutable` is re-exported by `api::cypher` straight from
// `executor::write`; the session layer reaches for
// `execute_mutable_with_csv`, which carries the LOAD CSV capability.
pub use executor::{is_mutation_query, CypherExecutor};
pub use explain::generate_explain_result;
pub use parse_cache::parse_cypher_cached as parse_cypher;
pub use planner::mark_lazy_eligibility;
pub use planner::optimize;
pub use planner::schema_check::validate_schema;
pub(crate) use planner::schema_check::{
    collect_query_warnings, emit_query_warnings, strict_read_error, strict_type_error,
};
pub use planner::simplification::rewrite_text_score;

use crate::datatypes::values::Value;

use ast::*;

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
