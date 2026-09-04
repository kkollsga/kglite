//! EXPLAIN plan rendering: the operator rows and the row estimate a
//! `EXPLAIN <query>` answers with.
//!
//! Split out of `cypher/mod.rs`, which is a module *root* — its job is the
//! submodule tree and the crate-facing re-exports, and the plan renderer had
//! grown past the file's line cap sitting inside it.
//!
//! The renderer is fully static: it reads the parsed query and the graph's
//! statistics, never the executor. Eligibility for the `ClosureProbe` row is
//! not decided here — it comes from
//! [`crate::graph::core::pattern_matching::closure_probe`], the same predicate
//! the matcher gates on, so a plan row cannot claim a probe the runtime
//! declines.

use super::ast::*;
use super::{executor, result};
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{
    closure_probe, EdgeDirection, EdgePattern, NodePattern, Pattern, PatternElement,
    PropertyMatcher,
};
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;

/// Estimate the number of rows a MATCH clause will produce, from the label
/// cardinalities its patterns constrain.
///
/// Counts through [`DirGraph::label_cardinality`], the same primary +
/// secondary-bucket answer the join-order model uses: a label that only ever
/// appears as a secondary one (every materialized ontology supertype) has an
/// empty `type_indices` bucket, and counting that alone reported 0 rows for a
/// pattern matching every member.
fn estimate_match_rows(m: &MatchClause, graph: &DirGraph) -> Option<usize> {
    let types = collect_node_types(m);
    if types.is_empty() {
        // Untyped scan — total node count
        Some(graph.graph.node_count())
    } else {
        // Use the smallest type's count as the estimate (join
        // selectivity heuristic)
        types.iter().map(|t| graph.label_cardinality(t)).min()
    }
}

/// Collect node types from a MatchClause's patterns.
fn collect_node_types(m: &MatchClause) -> Vec<String> {
    use crate::graph::core::pattern_matching::PatternElement;
    let mut types = Vec::new();
    for pattern in &m.patterns {
        for element in &pattern.elements {
            if let PatternElement::Node(np) = element {
                for t in np.label_alternatives() {
                    types.push(t.clone());
                }
            }
        }
    }
    types
}

/// Render the node side of an expansion as `(:Type)` / `(:A:B)` / `()`.
fn explain_node_display(node: Option<&NodePattern>) -> String {
    let Some(node) = node else {
        return "()".to_string();
    };
    let Some(primary) = node.node_type.as_deref() else {
        return "()".to_string();
    };
    if let Some(alts) = &node.alt_labels {
        return format!("(:{})", alts.join("|"));
    }
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

/// One `ClosureProbe` operator row per node pattern whose ontology closure the
/// matcher can answer from member indexes, in pattern order.
///
/// Rendered as `ClosureProbe :Person (Student, Teacher)` — the supertype the
/// query names, then the live member types the probe would visit. Eligibility
/// comes from [`closure_probe::closure_probe_members`], the same predicate the
/// matcher answers to, so a plan row can never claim a probe the runtime
/// declines.
///
/// Only *inline* equality properties count. Pushdown has already run by the
/// time EXPLAIN renders (the planner moves eligible `WHERE` equalities into the
/// pattern), but a value written as a parameter is still unresolved here, so
/// `{p: $v}` stays conservatively unmarked rather than promising a plan the
/// bound value might not get. `estimated_rows` stays `Null`: the probe's row
/// count is the value's, and no static model knows it.
fn closure_probe_ops(patterns: &[Pattern], graph: &DirGraph) -> Vec<String> {
    let mut ops = Vec::new();
    for pattern in patterns {
        for element in &pattern.elements {
            let PatternElement::Node(np) = element else {
                continue;
            };
            if np.alt_labels.is_some() {
                continue;
            }
            let (Some(node_type), Some(props)) = (np.node_type.as_deref(), np.properties.as_ref())
            else {
                continue;
            };
            let equality_props: Vec<&str> = props
                .iter()
                .filter(|(_, matcher)| matches!(matcher, PropertyMatcher::Equals(_)))
                .map(|(name, _)| name.as_str())
                .collect();
            if let Some(members) =
                closure_probe::closure_probe_members(graph, node_type, &equality_props)
            {
                ops.push(format!(
                    "ClosureProbe :{node_type} ({})",
                    members.join(", ")
                ));
            }
        }
    }
    ops
}

/// Generate a structured query plan as a CypherResult with columns
/// [step, operation, estimated_rows].
pub fn generate_explain_result(query: &CypherQuery, graph: &DirGraph) -> result::CypherResult {
    let mut rows = Vec::new();

    for clause in query.clauses.iter() {
        // `step` numbers rows, not clauses: a MATCH contributes a ClosureProbe
        // row per eligible closure and an Expand row per variable-length edge
        // after its own row, and the column stays contiguous.
        let step = (rows.len() + 1) as i64;
        let mut operation = executor::clause_display_name(clause);
        if let Clause::FusedVectorScoreTopK {
            return_clause,
            score_item_index,
            ..
        } = clause
        {
            operation.push_str(&format!(
                " [requested={}]",
                requested_vector_policy(&return_clause.items[*score_item_index].expression)
            ));
        }
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
            // One row out, exactly like every other single-count fusion. The
            // sibling above reports 0 for an empty type; an alternation is
            // minted only over branch labels the pass accepted, so the row
            // count does not depend on which of them happen to be populated.
            Clause::FusedCountLabelUnion { .. } => Value::Int64(1),
            Clause::FusedCountByType { .. } => Value::Int64(graph.type_indices.len() as i64),
            Clause::FusedVectorScoreTopK { limit, .. }
            | Clause::FusedTextBm25TopK { limit, .. }
            | Clause::FusedOrderByTopK { limit, .. }
            | Clause::FusedNodeScanTopK { limit, .. } => Value::Int64(*limit as i64),
            _ => Value::Null,
        };

        rows.push(vec![Value::Int64(step), Value::String(operation), est]);

        if let Clause::Match(m) | Clause::OptionalMatch(m) = clause {
            for op in closure_probe_ops(&m.patterns, graph)
                .into_iter()
                .chain(var_length_expand_ops(&m.patterns))
            {
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
mod var_length_expand_tests {
    use super::{var_length_expand_ops, Clause};

    fn ops(query: &str) -> Vec<String> {
        let parsed = super::super::parser::parse_cypher(query).expect("parse");
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

/// EXPLAIN does not evaluate parameters or row expressions, and never claims an actual route.
fn requested_vector_policy(expression: &Expression) -> &'static str {
    let Expression::FunctionCall { args, .. } = expression else {
        return "dynamic";
    };
    match args.last() {
        Some(Expression::MapLiteral(items)) => {
            match items
                .iter()
                .find(|(key, _)| key == "exact")
                .map(|(_, value)| value)
            {
                Some(Expression::Literal(Value::Boolean(true))) => "exact",
                None | Some(Expression::Literal(Value::Boolean(false))) => "auto",
                _ => "dynamic",
            }
        }
        Some(Expression::Literal(Value::Map(items))) => {
            match items
                .iter()
                .find(|(key, _)| *key == "exact")
                .map(|(_, value)| value)
            {
                Some(Value::Boolean(true)) => "exact",
                None | Some(Value::Boolean(false)) => "auto",
                _ => "dynamic",
            }
        }
        _ if args.len() <= 3 => "auto",
        Some(Expression::Literal(_)) if args.len() == 4 => "auto",
        _ => "dynamic",
    }
}
