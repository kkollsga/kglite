//! Cypher parser: WHERE predicate tree (OR / XOR / AND / NOT / comparisons).

use super::super::ast::*;
use super::super::tokenizer::CypherToken;
use super::CypherParser;
use crate::datatypes::values::Value;

impl CypherParser {
    pub(super) fn parse_where_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Where)?;
        let predicate = self.parse_predicate()?;
        Ok(Clause::Where(WhereClause { predicate }))
    }

    /// Parse predicate with OR as lowest precedence
    pub(super) fn parse_predicate(&mut self) -> Result<Predicate, String> {
        self.parse_or_predicate()
    }

    /// Parse OR expressions (lowest precedence)
    pub(super) fn parse_or_predicate(&mut self) -> Result<Predicate, String> {
        let mut left = self.parse_xor_predicate()?;

        while self.check(&CypherToken::Or) {
            self.advance();
            let right = self.parse_xor_predicate()?;
            left = Predicate::Or(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse XOR expressions (precedence between OR and AND)
    pub(super) fn parse_xor_predicate(&mut self) -> Result<Predicate, String> {
        let mut left = self.parse_and_predicate()?;

        while self.check(&CypherToken::Xor) {
            self.advance();
            let right = self.parse_and_predicate()?;
            left = Predicate::Xor(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse AND expressions
    pub(super) fn parse_and_predicate(&mut self) -> Result<Predicate, String> {
        let mut left = self.parse_not_predicate()?;

        while self.check(&CypherToken::And) {
            self.advance();
            let right = self.parse_not_predicate()?;
            left = Predicate::And(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse NOT expressions
    pub(super) fn parse_not_predicate(&mut self) -> Result<Predicate, String> {
        if self.check(&CypherToken::Not) {
            self.advance();
            let inner = self.parse_not_predicate()?;
            Ok(Predicate::Not(Box::new(inner)))
        } else {
            self.parse_comparison_predicate()
        }
    }

    /// Parse comparison expressions and IS NULL/IS NOT NULL/IN
    pub(super) fn parse_comparison_predicate(&mut self) -> Result<Predicate, String> {
        // Check for EXISTS { pattern }
        if self.check(&CypherToken::Exists) {
            self.advance(); // consume EXISTS
            if self.check(&CypherToken::LBrace) {
                self.advance(); // consume {
                let patterns = self.parse_exists_patterns()?;
                // Check for optional WHERE clause inside EXISTS { MATCH ... WHERE ... }
                let where_clause = if self.check(&CypherToken::Where) {
                    self.advance(); // consume WHERE
                    Some(Box::new(self.parse_predicate()?))
                } else {
                    None
                };
                self.expect(&CypherToken::RBrace)?;
                return Ok(Predicate::Exists {
                    patterns,
                    where_clause,
                });
            } else if self.check(&CypherToken::LParen) {
                self.advance(); // consume outer (
                                // Support EXISTS((...)) — inner parens are the pattern
                if self.check(&CypherToken::LParen) {
                    let pattern_str = self.extract_pattern_string()?;
                    let pattern =
                        crate::graph::core::pattern_matching::parse_pattern(&pattern_str)?;
                    self.expect(&CypherToken::RParen)?; // consume outer )
                    return Ok(Predicate::Exists {
                        patterns: vec![pattern],
                        where_clause: None,
                    });
                } else if self.looks_like_property_access() {
                    // Neo4j-style `exists(n.prop)` for property-existence
                    // checks. KGLite doesn't support this form — point the
                    // user at the modern equivalent rather than the pattern
                    // error, which sends them down the wrong rabbit hole.
                    return Err(
                        "exists(n.prop) is Neo4j legacy syntax for property-existence. \
                         Use `WHERE n.prop IS NOT NULL` instead. \
                         (For pattern-existence, EXISTS takes a pattern: \
                         `EXISTS { (n)-[:REL]->() }` or `EXISTS((n)-[:REL]->())`.)"
                            .to_string(),
                    );
                } else {
                    return Err("EXISTS(...) requires a pattern in parentheses, e.g. \
                         EXISTS((n)-[:REL]->()). For property-existence checks, \
                         use `WHERE n.prop IS NOT NULL` (Neo4j-style \
                         `exists(n.prop)` is not supported)."
                        .to_string());
                }
            } else {
                return Err("Expected '{' or '(' after EXISTS".to_string());
            }
        }

        // Check for parenthesized predicate
        if self.check(&CypherToken::LParen) {
            // Could be a parenthesized predicate or the start of a pattern
            // Peek ahead to determine
            if self.looks_like_pattern_start() {
                // Inline pattern predicate — desugar to EXISTS { pattern }
                let pattern_str = self.extract_pattern_string()?;
                let pattern = crate::graph::core::pattern_matching::parse_pattern(&pattern_str)?;
                return Ok(Predicate::Exists {
                    patterns: vec![pattern],
                    where_clause: None,
                });
            }
            self.advance(); // consume (
            let pred = self.parse_predicate()?;
            self.expect(&CypherToken::RParen)?;
            return Ok(pred);
        }

        let left = self.parse_expression()?;

        // Label-check predicate: `WHERE n:Label` (and chained `n:A:B` = AND).
        // Only applies when the LHS is a bare Variable — `count(n):Foo` isn't valid.
        if self.check(&CypherToken::Colon) {
            if let Expression::Variable(var) = &left {
                let var = var.clone();
                self.advance(); // consume :
                let first_label = match self.advance().cloned() {
                    Some(CypherToken::Identifier(name)) => name,
                    other => {
                        return Err(format!("Expected label name after ':', got {:?}", other));
                    }
                };
                let mut pred = Predicate::LabelCheck {
                    variable: var.clone(),
                    label: first_label,
                };
                while self.check(&CypherToken::Colon) {
                    self.advance();
                    let next_label = match self.advance().cloned() {
                        Some(CypherToken::Identifier(name)) => name,
                        other => {
                            return Err(format!("Expected label name after ':', got {:?}", other));
                        }
                    };
                    pred = Predicate::And(
                        Box::new(pred),
                        Box::new(Predicate::LabelCheck {
                            variable: var.clone(),
                            label: next_label,
                        }),
                    );
                }
                return Ok(pred);
            }
        }

        // parse_expression() may have already consumed IS NULL / IS NOT NULL
        // and returned Expression::IsNull/IsNotNull — convert to Predicate form
        if let Expression::IsNull(inner) = left {
            return Ok(Predicate::IsNull(*inner));
        }
        if let Expression::IsNotNull(inner) = left {
            return Ok(Predicate::IsNotNull(*inner));
        }

        // Check for IS NULL / IS NOT NULL (fallback for non-expression contexts)
        if self.check(&CypherToken::Is) {
            self.advance(); // consume IS
            if self.check(&CypherToken::Not) {
                self.advance(); // consume NOT
                self.expect(&CypherToken::Null)?;
                return Ok(Predicate::IsNotNull(left));
            } else {
                self.expect(&CypherToken::Null)?;
                return Ok(Predicate::IsNull(left));
            }
        }

        // Check for IN
        if self.check(&CypherToken::In) {
            self.advance();
            if self.check(&CypherToken::LBracket) {
                let list = self.parse_list_expression()?;
                return Ok(Predicate::In { expr: left, list });
            } else {
                // IN with a variable, parameter, or function call expression
                let list_expr = self.parse_expression()?;
                return Ok(Predicate::InExpression {
                    expr: left,
                    list_expr,
                });
            }
        }

        // Check for STARTS WITH
        if self.check(&CypherToken::StartsWith) {
            self.advance(); // consume STARTS
                            // Expect WITH keyword (we use With token)
            self.expect(&CypherToken::With)?;
            let pattern = self.parse_expression()?;
            return Ok(Predicate::StartsWith {
                expr: left,
                pattern,
            });
        }

        // Check for ENDS WITH
        if self.check(&CypherToken::EndsWith) {
            self.advance(); // consume ENDS
            self.expect(&CypherToken::With)?;
            let pattern = self.parse_expression()?;
            return Ok(Predicate::EndsWith {
                expr: left,
                pattern,
            });
        }

        // Check for CONTAINS
        if self.check(&CypherToken::Contains) {
            self.advance();
            let pattern = self.parse_expression()?;
            return Ok(Predicate::Contains {
                expr: left,
                pattern,
            });
        }

        // Check for comparison operator
        let operator = match self.peek() {
            Some(CypherToken::Equals) => ComparisonOp::Equals,
            Some(CypherToken::NotEquals) => ComparisonOp::NotEquals,
            Some(CypherToken::LessThan) => ComparisonOp::LessThan,
            Some(CypherToken::LessThanEquals) => ComparisonOp::LessThanEq,
            Some(CypherToken::GreaterThan) => ComparisonOp::GreaterThan,
            Some(CypherToken::GreaterThanEquals) => ComparisonOp::GreaterThanEq,
            Some(CypherToken::RegexMatch) => ComparisonOp::RegexMatch,
            _ => {
                // No operator - the expression itself is a boolean predicate
                // Convert expression to a comparison: expr <> false (truthy check)
                return Ok(Predicate::Comparison {
                    left: left.clone(),
                    operator: ComparisonOp::NotEquals,
                    right: Expression::Literal(Value::Boolean(false)),
                });
            }
        };

        self.advance(); // consume operator
        let right = self.parse_expression()?;

        Ok(Predicate::Comparison {
            left,
            operator,
            right,
        })
    }

    /// Parse a [value, value, ...] list for IN clause
    pub(super) fn parse_list_expression(&mut self) -> Result<Vec<Expression>, String> {
        self.expect(&CypherToken::LBracket)?;
        let mut items = Vec::new();

        if !self.check(&CypherToken::RBracket) {
            items.push(self.parse_expression()?);
            while self.check(&CypherToken::Comma) {
                self.advance();
                items.push(self.parse_expression()?);
            }
        }

        self.expect(&CypherToken::RBracket)?;
        Ok(items)
    }

    /// True when the next three tokens are `<ident> . <ident>` — i.e. the
    /// shape of a property access (`n.prop`). Used by `parse_comparison_predicate`
    /// to recognise the Neo4j-legacy `exists(n.prop)` form and steer the
    /// user to `IS NOT NULL` instead of the generic pattern-required error.
    pub(super) fn looks_like_property_access(&self) -> bool {
        matches!(self.peek(), Some(CypherToken::Identifier(_)))
            && matches!(self.peek_at(1), Some(CypherToken::Dot))
            && matches!(self.peek_at(2), Some(CypherToken::Identifier(_)))
    }

    /// Quick lookahead to check if ( starts a pattern (node pattern) vs a parenthesized predicate
    pub(super) fn looks_like_pattern_start(&self) -> bool {
        // Pattern: (var:Type), (:Type), (), (var)-[...]->()
        // Predicate: (expr op expr), (NOT ...)
        match self.peek_at(1) {
            Some(CypherToken::RParen) => {
                // () or (var) closed immediately — pattern if followed by - or <
                matches!(
                    self.peek_at(2),
                    Some(CypherToken::Dash) | Some(CypherToken::LessThan)
                )
            }
            Some(CypherToken::Colon) => true, // (:Type)
            Some(CypherToken::Identifier(_)) => {
                match self.peek_at(2) {
                    Some(CypherToken::Colon) => true, // (var:Type
                    Some(CypherToken::RParen) => {
                        // (var) — pattern if followed by - or <  e.g. (p)-[:REL]->()
                        matches!(
                            self.peek_at(3),
                            Some(CypherToken::Dash) | Some(CypherToken::LessThan)
                        )
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
}
