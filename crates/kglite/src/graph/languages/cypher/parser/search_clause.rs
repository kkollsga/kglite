//! Cypher parser: the Cypher 25 `SEARCH` subclause, desugared to the vector
//! query procedures.
//!
//! ```text
//! MATCH (e[:L]) SEARCH e IN (VECTOR INDEX name FOR vec LIMIT k) [SCORE AS s]
//!   => CALL db.index.vector.queryNodes(name, k, vec) YIELD node AS e, score AS s
//! MATCH ()-[e]->() SEARCH e IN (VECTOR INDEX …)
//!   => CALL db.index.vector.queryRelationships(…) YIELD relationship AS e, score AS s
//! ```
//!
//! No executor operator exists for `SEARCH`: the rewrite happens here, so the
//! procedure's index resolution, errors and scores are the clause's.

use super::super::ast::*;
use super::super::tokenizer::CypherToken;
use super::CypherParser;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::PatternElement;

pub(super) const SEARCH_PLACEMENT: &str =
    "SEARCH must directly follow the MATCH that introduces its variable, with no WHERE in \
     between: MATCH (e) SEARCH e IN (VECTOR INDEX name FOR vector LIMIT k)";

/// What the `MATCH` preceding a `SEARCH` binds.
struct SearchBinding {
    variable: String,
    relationship: bool,
    label: Option<String>,
}

impl CypherParser {
    /// True at `SEARCH <ident> IN`.
    pub(super) fn at_search_clause(&self) -> bool {
        self.peek_soft_word("SEARCH")
            && matches!(self.peek_at(1), Some(CypherToken::Identifier(_)))
            && matches!(self.peek_at(2), Some(CypherToken::In))
    }

    /// A `MATCH`, plus the `SEARCH` that directly follows it, if any.
    pub(super) fn parse_match_with_search(
        &mut self,
        clauses: &mut Vec<Clause>,
    ) -> Result<(), String> {
        clauses.push(self.parse_match_clause(false)?);
        if self.at_search_clause() {
            self.parse_search_clause(clauses)?;
        }
        Ok(())
    }

    /// Parse the clause and replace the `MATCH` that introduces its variable
    /// (the last element of `clauses`) with the equivalent `CALL`.
    fn parse_search_clause(&mut self, clauses: &mut Vec<Clause>) -> Result<(), String> {
        self.expect_soft_word("SEARCH", "SEARCH clause")?;
        let variable = self.consume_identifier()?;
        let binding = search_binding(clauses.last(), &variable)?;
        self.expect(&CypherToken::In)?;
        self.expect(&CypherToken::LParen)?;
        self.expect_soft_word("VECTOR", "SEARCH (only `VECTOR INDEX` is supported)")?;
        self.expect_soft_word("INDEX", "SEARCH")?;
        let index_name = match self.advance().cloned() {
            Some(CypherToken::Identifier(name)) => name,
            other => {
                return Err(format!(
                    "SEARCH expects an index name after VECTOR INDEX, got {}",
                    super::describe_token_opt(other.as_ref())
                ))
            }
        };
        self.expect_soft_word("FOR", "SEARCH")?;
        let query = self.parse_expression()?;
        if self.check(&CypherToken::Where) {
            return Err(
                "SEARCH ... WHERE (a filter inside the index search) is not supported: the \
                 filter would have to apply before the top-k cut, and this engine cannot do \
                 that. Filter the rows after the SEARCH with a WHERE, which keeps the top-k \
                 hits that pass, or use CALL db.node_embeddings.query with a filter"
                    .to_string(),
            );
        }
        if !self.check(&CypherToken::Limit) {
            return Err(
                "SEARCH requires a LIMIT inside the parentheses: SEARCH e IN (VECTOR INDEX \
                 name FOR vector LIMIT k)"
                    .to_string(),
            );
        }
        self.advance();
        let limit = self.parse_expression()?;
        self.expect(&CypherToken::RParen)?;
        let score = if self.peek_soft_word("SCORE") {
            self.advance();
            self.expect(&CypherToken::As)?;
            Some(self.try_consume_alias_name()?)
        } else {
            None
        };

        let mut procedure_name = if binding.relationship {
            "db.index.vector.queryRelationships"
        } else {
            "db.index.vector.queryNodes"
        }
        .to_string();
        let mut parameters = vec![
            (
                "indexName".to_string(),
                Expression::Literal(Value::String(index_name)),
            ),
            ("numberOfNearestNeighbours".to_string(), limit),
            ("query".to_string(), query),
        ];
        crate::graph::languages::cypher::executor::procedure_router::route(
            &mut procedure_name,
            &mut parameters,
        )?;
        let mut yield_items = vec![YieldItem {
            name: if binding.relationship {
                "relationship"
            } else {
                "node"
            }
            .to_string(),
            alias: Some(binding.variable.clone()),
        }];
        if let Some(alias) = score {
            yield_items.push(YieldItem {
                name: "score".to_string(),
                alias: Some(alias),
            });
        }
        clauses.pop();
        clauses.push(Clause::Call(CallClause {
            procedure_name,
            parameters,
            yield_items,
            release: Vec::new(),
        }));
        // The index's own label already bounds the hits; a pattern label
        // that names another label leaves nothing after this filter.
        if let Some(label) = binding.label {
            clauses.push(Clause::Where(WhereClause {
                predicate: Predicate::LabelCheck {
                    variable: binding.variable,
                    label,
                    label_param: None,
                },
            }));
        }
        Ok(())
    }
}

/// The variable and label a `MATCH` hands to `SEARCH`, or why it cannot.
fn search_binding(previous: Option<&Clause>, variable: &str) -> Result<SearchBinding, String> {
    let directly = SEARCH_PLACEMENT;
    let Some(Clause::Match(matched)) = previous else {
        return Err(directly.to_string());
    };
    let single = matched.patterns.len() == 1 && matched.path_assignments.is_empty();
    if !single || matched.where_clause.is_some() {
        return Err(format!(
            "SEARCH needs a MATCH of exactly one node or relationship pattern. {directly}"
        ));
    }
    let elements = &matched.patterns[0].elements;
    let mismatch = |found: Option<&String>| {
        format!(
            "SEARCH variable '{variable}' is not the variable of the preceding MATCH ({})",
            found.map_or("it binds none".to_string(), |v| format!("'{v}'"))
        )
    };
    match elements.as_slice() {
        [PatternElement::Node(node)] => {
            if node.variable.as_deref() != Some(variable) {
                return Err(mismatch(node.variable.as_ref()));
            }
            if node.properties.is_some()
                || node.multi_label_constrained()
                || !node.label_params.is_empty()
            {
                return Err(
                    "SEARCH needs a plain node pattern: one optional label, no inline \
                            properties; the index decides which nodes are searched"
                        .to_string(),
                );
            }
            Ok(SearchBinding {
                variable: variable.to_string(),
                relationship: false,
                label: node.node_type.clone(),
            })
        }
        [PatternElement::Node(left), PatternElement::Edge(edge), PatternElement::Node(right)] => {
            if edge.variable.as_deref() != Some(variable) {
                return Err(mismatch(edge.variable.as_ref()));
            }
            let bare_node = |n: &crate::graph::core::pattern_matching::NodePattern| {
                n.variable.is_none()
                    && n.node_type.is_none()
                    && n.properties.is_none()
                    && !n.multi_label_constrained()
            };
            if !bare_node(left)
                || !bare_node(right)
                || edge.connection_type.is_some()
                || edge.properties.is_some()
                || edge.var_length.is_some()
            {
                return Err(
                    "SEARCH over relationships takes the bare form ()-[e]->(): no \
                            endpoint variables or labels, no relationship type, no properties"
                        .to_string(),
                );
            }
            Ok(SearchBinding {
                variable: variable.to_string(),
                relationship: true,
                label: None,
            })
        }
        _ => Err(format!(
            "SEARCH needs a MATCH of exactly one node or relationship pattern. {directly}"
        )),
    }
}
