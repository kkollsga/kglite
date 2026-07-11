//! Cypher parser: expressions (arithmetic, function calls, CASE, list operations).

use super::super::ast::*;
use super::super::tokenizer::CypherToken;
use super::CypherParser;
use crate::datatypes::values::Value;

impl CypherParser {
    pub(super) fn parse_expression_with_predicates(&mut self) -> Result<Expression, String> {
        let expr = self.parse_expression()?;
        // Check for trailing comparison/predicate operators
        match self.peek() {
            Some(CypherToken::Equals)
            | Some(CypherToken::NotEquals)
            | Some(CypherToken::LessThan)
            | Some(CypherToken::GreaterThan)
            | Some(CypherToken::LessThanEquals)
            | Some(CypherToken::GreaterThanEquals)
            | Some(CypherToken::RegexMatch) => {
                let operator = match self.peek() {
                    Some(CypherToken::Equals) => ComparisonOp::Equals,
                    Some(CypherToken::NotEquals) => ComparisonOp::NotEquals,
                    Some(CypherToken::LessThan) => ComparisonOp::LessThan,
                    Some(CypherToken::GreaterThan) => ComparisonOp::GreaterThan,
                    Some(CypherToken::LessThanEquals) => ComparisonOp::LessThanEq,
                    Some(CypherToken::GreaterThanEquals) => ComparisonOp::GreaterThanEq,
                    Some(CypherToken::RegexMatch) => ComparisonOp::RegexMatch,
                    _ => unreachable!(),
                };
                self.advance(); // consume operator
                let right = self.parse_expression()?;
                Ok(Expression::PredicateExpr(Box::new(Predicate::Comparison {
                    left: expr,
                    operator,
                    right,
                })))
            }
            Some(CypherToken::StartsWith) => {
                self.advance(); // consume STARTS
                self.expect(&CypherToken::With)?; // consume WITH
                let pattern = self.parse_expression()?;
                Ok(Expression::PredicateExpr(Box::new(Predicate::StartsWith {
                    expr,
                    pattern,
                })))
            }
            Some(CypherToken::EndsWith) => {
                self.advance(); // consume ENDS
                self.expect(&CypherToken::With)?; // consume WITH
                let pattern = self.parse_expression()?;
                Ok(Expression::PredicateExpr(Box::new(Predicate::EndsWith {
                    expr,
                    pattern,
                })))
            }
            Some(CypherToken::Contains) => {
                self.advance(); // consume CONTAINS
                let pattern = self.parse_expression()?;
                Ok(Expression::PredicateExpr(Box::new(Predicate::Contains {
                    expr,
                    pattern,
                })))
            }
            Some(CypherToken::In) => {
                self.advance(); // consume IN
                if self.check(&CypherToken::LBracket) {
                    let list = self.parse_list_expression()?;
                    Ok(Expression::PredicateExpr(Box::new(Predicate::In {
                        expr,
                        list,
                    })))
                } else {
                    let list_expr = self.parse_expression()?;
                    Ok(Expression::PredicateExpr(Box::new(
                        Predicate::InExpression { expr, list_expr },
                    )))
                }
            }
            _ => Ok(expr),
        }
    }

    pub(super) fn parse_expression(&mut self) -> Result<Expression, String> {
        let expr = self.parse_additive_expression()?;
        // Check for IS NULL / IS NOT NULL postfix
        if self.peek() == Some(&CypherToken::Is) {
            self.advance(); // consume IS
            if self.peek() == Some(&CypherToken::Not) {
                self.advance(); // consume NOT
                self.expect(&CypherToken::Null)?;
                return Ok(Expression::IsNotNull(Box::new(expr)));
            } else {
                self.expect(&CypherToken::Null)?;
                return Ok(Expression::IsNull(Box::new(expr)));
            }
        }
        Ok(expr)
    }

    pub(super) fn parse_additive_expression(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_multiplicative_expression()?;

        loop {
            match self.peek() {
                Some(CypherToken::Plus) => {
                    self.advance();
                    let right = self.parse_multiplicative_expression()?;
                    left = Expression::Add(Box::new(left), Box::new(right));
                }
                Some(CypherToken::Dash) => {
                    // Dash could be subtraction or edge syntax - only treat as subtraction
                    // if we're in an expression context (not at clause boundary)
                    // Heuristic: if next token after dash is a number, identifier, or '(',
                    // it's subtraction. Otherwise, stop.
                    if self.peek_at(1).is_some_and(|t| {
                        matches!(
                            t,
                            CypherToken::IntLit(_)
                                | CypherToken::FloatLit(_)
                                | CypherToken::Identifier(_)
                                | CypherToken::LParen
                        )
                    }) {
                        // But check it's not an edge pattern (dash followed by bracket)
                        if self.peek_at(1) == Some(&CypherToken::LBracket) {
                            break;
                        }
                        self.advance();
                        let right = self.parse_multiplicative_expression()?;
                        left = Expression::Subtract(Box::new(left), Box::new(right));
                    } else {
                        break;
                    }
                }
                Some(CypherToken::DoublePipe) => {
                    self.advance();
                    let right = self.parse_multiplicative_expression()?;
                    left = Expression::Concat(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    pub(super) fn parse_multiplicative_expression(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_unary_expression()?;

        loop {
            match self.peek() {
                Some(CypherToken::Star) => {
                    self.advance();
                    let right = self.parse_unary_expression()?;
                    left = Expression::Multiply(Box::new(left), Box::new(right));
                }
                Some(CypherToken::Slash) => {
                    self.advance();
                    let right = self.parse_unary_expression()?;
                    left = Expression::Divide(Box::new(left), Box::new(right));
                }
                Some(CypherToken::Percent) => {
                    self.advance();
                    let right = self.parse_unary_expression()?;
                    left = Expression::Modulo(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    pub(super) fn parse_unary_expression(&mut self) -> Result<Expression, String> {
        let expr = if self.check(&CypherToken::Dash) {
            self.advance();
            let inner = self.parse_primary_expression()?;
            Expression::Negate(Box::new(inner))
        } else {
            self.parse_primary_expression()?
        };
        self.parse_postfix(expr)
    }

    /// Parse postfix operators: expr[index] or expr[start..end]
    pub(super) fn parse_postfix(&mut self, mut expr: Expression) -> Result<Expression, String> {
        while self.check(&CypherToken::LBracket) {
            self.advance(); // consume [

            if self.check(&CypherToken::DotDot) {
                // [..end] — slice with no start. Also accept [..] (both
                // ends omitted) — openCypher allows it as "the whole list",
                // and ListSlice with start=None/end=None already evaluates
                // that way. Mirror the [start..] / [start..end] guard
                // pattern just below.
                self.advance(); // consume ..
                let end_expr = if self.check(&CypherToken::RBracket) {
                    None
                } else {
                    Some(Box::new(self.parse_expression()?))
                };
                self.expect(&CypherToken::RBracket)?;
                expr = Expression::ListSlice {
                    expr: Box::new(expr),
                    start: None,
                    end: end_expr,
                };
            } else {
                let first = self.parse_expression()?;
                if self.check(&CypherToken::DotDot) {
                    // [start..] or [start..end]
                    self.advance(); // consume ..
                    let end_expr = if self.check(&CypherToken::RBracket) {
                        None
                    } else {
                        Some(Box::new(self.parse_expression()?))
                    };
                    self.expect(&CypherToken::RBracket)?;
                    expr = Expression::ListSlice {
                        expr: Box::new(expr),
                        start: Some(Box::new(first)),
                        end: end_expr,
                    };
                } else {
                    // [index] — plain index access
                    self.expect(&CypherToken::RBracket)?;
                    expr = Expression::IndexAccess {
                        expr: Box::new(expr),
                        index: Box::new(first),
                    };
                }
            }
        }
        Ok(expr)
    }

    pub(super) fn parse_primary_expression(&mut self) -> Result<Expression, String> {
        match self.peek().cloned() {
            // Numeric literals
            Some(CypherToken::IntLit(n)) => {
                self.advance();
                Ok(Expression::Literal(Value::Int64(n)))
            }
            Some(CypherToken::FloatLit(f)) => {
                self.advance();
                Ok(Expression::Literal(Value::Float64(f)))
            }

            // String literal
            Some(CypherToken::StringLit(s)) => {
                self.advance();
                Ok(Expression::Literal(Value::String(s)))
            }

            // Boolean literals
            Some(CypherToken::True) => {
                self.advance();
                Ok(Expression::Literal(Value::Boolean(true)))
            }
            Some(CypherToken::False) => {
                self.advance();
                Ok(Expression::Literal(Value::Boolean(false)))
            }

            // NULL literal
            Some(CypherToken::Null) => {
                self.advance();
                Ok(Expression::Literal(Value::Null))
            }

            // Star (for count(*))
            Some(CypherToken::Star) => {
                self.advance();
                Ok(Expression::Star)
            }

            // Parenthesized expression
            Some(CypherToken::LParen) => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(&CypherToken::RParen)?;
                Ok(expr)
            }

            // List literal [...] or list comprehension [x IN list WHERE ... | expr]
            Some(CypherToken::LBracket) => {
                self.advance(); // consume [

                // Check for list comprehension: [x IN list ...]
                // Look for: Identifier IN
                if matches!(self.peek(), Some(CypherToken::Identifier(_)))
                    && self.peek_at(1) == Some(&CypherToken::In)
                {
                    return self.parse_list_comprehension();
                }

                // Otherwise: list literal [expr, expr, ...]
                let mut items = Vec::new();
                if !self.check(&CypherToken::RBracket) {
                    items.push(self.parse_expression()?);
                    while self.check(&CypherToken::Comma) {
                        self.advance();
                        items.push(self.parse_expression()?);
                    }
                }
                self.expect(&CypherToken::RBracket)?;
                Ok(Expression::ListLiteral(items))
            }

            // CASE expression
            Some(CypherToken::Case) => {
                self.advance();
                self.parse_case_expression()
            }

            // Parameter: $name
            Some(CypherToken::Parameter(name)) => {
                self.advance();
                Ok(Expression::Parameter(name))
            }

            // Identifier: could be variable, property access, function call, or list quantifier
            Some(CypherToken::Identifier(name)) => {
                self.advance();

                // Check for list quantifier: any/none/single(var IN list WHERE pred)
                if self.check(&CypherToken::LParen) {
                    let quantifier = match name.to_lowercase().as_str() {
                        "any" => Some(ListQuantifier::Any),
                        "none" => Some(ListQuantifier::None),
                        "single" => Some(ListQuantifier::Single),
                        _ => None,
                    };
                    if let Some(q) = quantifier {
                        if matches!(self.peek_at(1), Some(CypherToken::Identifier(_)))
                            && self.peek_at(2) == Some(&CypherToken::In)
                        {
                            return self.parse_list_quantifier_expr(q);
                        }
                    }
                    // Check for reduce(acc = init, var IN list | body)
                    if name.eq_ignore_ascii_case("reduce")
                        && matches!(self.peek_at(1), Some(CypherToken::Identifier(_)))
                        && self.peek_at(2) == Some(&CypherToken::Equals)
                    {
                        return self.parse_reduce_expr();
                    }
                    let func_expr = self.parse_function_call(name)?;
                    // Check for property access on function result: func().property
                    if self.check(&CypherToken::Dot) {
                        self.advance(); // consume dot
                        let prop = self.expect_name("property name after '.'")?;
                        return Ok(Expression::ExprPropertyAccess {
                            expr: Box::new(func_expr),
                            property: prop,
                        });
                    }
                    return Ok(func_expr);
                }

                // Check for namespaced function call: `duration.between(...)`,
                // `point.distance(...)`, etc. Pattern is
                // `<ident>.<ident>(`. 0.9.0 §3 introduces the first
                // namespaced function `duration.between`.
                if self.check(&CypherToken::Dot)
                    && matches!(self.peek_at(1), Some(CypherToken::Identifier(_)))
                    && self.peek_at(2) == Some(&CypherToken::LParen)
                {
                    self.advance(); // dot
                    let sub = match self.advance().cloned() {
                        Some(CypherToken::Identifier(s)) => s,
                        _ => unreachable!("guarded by peek_at(1)"),
                    };
                    let qualified_name = format!("{}.{}", name, sub);
                    let func_expr = self.parse_function_call(qualified_name)?;
                    // Allow trailing `.field` chains on the result, e.g.
                    // `duration.between(d1, d2).days`.
                    let mut expr = func_expr;
                    while self.check(&CypherToken::Dot) {
                        self.advance();
                        let field = self.expect_name("property name after '.'")?;
                        expr = Expression::ExprPropertyAccess {
                            expr: Box::new(expr),
                            property: field,
                        };
                    }
                    return Ok(expr);
                }

                // Check for property access: identifier.property,
                // and chained accessors like `n.joined.year` (datetime
                // field accessors per 0.9.0 §3) — repeatedly wrap in
                // ExprPropertyAccess.
                if self.check(&CypherToken::Dot) {
                    self.advance(); // consume dot
                    let first_prop = self.expect_name("property name after '.'")?;
                    let mut expr = Expression::PropertyAccess {
                        variable: name,
                        property: first_prop,
                    };
                    while self.check(&CypherToken::Dot) {
                        self.advance(); // consume the chained dot
                        let next_prop = self.expect_name("property name after '.'")?;
                        expr = Expression::ExprPropertyAccess {
                            expr: Box::new(expr),
                            property: next_prop,
                        };
                    }
                    Ok(expr)
                }
                // Identifier followed by `{` is ambiguous between two shapes:
                //   1. Map projection: `n { .prop1, .prop2, alias: expr }`
                //   2. Subquery expression: `count { (a)-[:REL]->() }`
                // Disambiguate on the identifier name — openCypher reserves
                // `count { ... }` for the subquery form. Any other identifier
                // stays on the map-projection path.
                else if self.check(&CypherToken::LBrace) {
                    if name.eq_ignore_ascii_case("count") {
                        self.parse_count_subquery()
                    } else {
                        self.parse_map_projection(name)
                    }
                } else {
                    Ok(Expression::Variable(name))
                }
            }

            // Map literal: {key: expr, key2: expr, ...}
            Some(CypherToken::LBrace) => {
                self.advance(); // consume {
                self.parse_map_literal()
            }

            // ALL(var IN list WHERE pred) — ALL is a keyword token
            Some(CypherToken::All)
                if self.peek_at(1) == Some(&CypherToken::LParen)
                    && matches!(self.peek_at(2), Some(CypherToken::Identifier(_)))
                    && self.peek_at(3) == Some(&CypherToken::In) =>
            {
                self.advance(); // consume ALL
                self.parse_list_quantifier_expr(ListQuantifier::All)
            }

            // Keywords that can also be function names when followed by (
            Some(CypherToken::Contains) if self.peek_at(1) == Some(&CypherToken::LParen) => {
                self.advance();
                self.parse_function_call("contains".to_string())
            }

            Some(t) => Err(format!("Unexpected token in expression: {:?}", t)),
            None => Err("Unexpected end of query in expression".to_string()),
        }
    }

    /// Parse function call: name(args...)
    pub(super) fn parse_function_call(&mut self, name: String) -> Result<Expression, String> {
        // Normalize function name to lowercase once at parse time so downstream
        // dispatch (planner, executor, aggregate detection) doesn't pay a
        // `.to_lowercase()` per row. Cypher function names are case-insensitive.
        let name = if name
            .chars()
            .all(|c| c.is_ascii_lowercase() || !c.is_alphabetic())
        {
            name
        } else {
            name.to_ascii_lowercase()
        };
        // KGLite extensions have a canonical namespace so they cannot be
        // mistaken for language-standard functions. Keep accepting historical
        // flat spellings by normalising both forms to the same AST; planner
        // rewrites and execution therefore remain identical.
        let name = name
            .strip_prefix("kglite.")
            .unwrap_or(name.as_str())
            .to_string();
        self.expect(&CypherToken::LParen)?;

        // 0.9.0 §6 — `size(<pattern-expression>)` is the openCypher
        // form returning the count of matches of an inline pattern.
        // After consuming `size(`, if the next token is `(` AND the
        // following tokens look like a pattern start, dispatch to the
        // count-subquery code path (semantically equivalent to
        // `count { <pattern> }` from 0.8.16). Comma-separated patterns
        // are unsupported here — Cypher only allows a single pattern
        // expression as size()'s arg.
        if name == "size" && self.check(&CypherToken::LParen) && self.looks_like_pattern_start() {
            let patterns = self.parse_pattern_subquery_patterns(&CypherToken::RParen)?;
            self.expect(&CypherToken::RParen)?; // close size(
            return Ok(Expression::CountSubquery {
                patterns,
                where_clause: None,
            });
        }

        // Check for DISTINCT
        let distinct = if self.check(&CypherToken::Distinct) {
            self.advance();
            true
        } else {
            false
        };

        let mut args = Vec::new();

        if !self.check(&CypherToken::RParen) {
            args.push(self.parse_expression()?);
            while self.check(&CypherToken::Comma) {
                self.advance();
                args.push(self.parse_expression()?);
            }
        }

        self.expect(&CypherToken::RParen)?;

        // Check for window function: func() OVER (PARTITION BY ... ORDER BY ...)
        if self.check(&CypherToken::Over) {
            let lower = name.to_lowercase();
            if !matches!(lower.as_str(), "row_number" | "rank" | "dense_rank") {
                return Err(format!(
                    "OVER clause is only supported for window functions (row_number, rank, dense_rank), not '{}'",
                    name
                ));
            }
            self.advance(); // consume OVER
            self.expect(&CypherToken::LParen)?;

            // Optional PARTITION BY
            let partition_by = if self.check(&CypherToken::Partition) {
                self.advance(); // consume PARTITION
                self.expect(&CypherToken::By)?;
                let mut exprs = vec![self.parse_expression()?];
                while self.check(&CypherToken::Comma) {
                    self.advance();
                    exprs.push(self.parse_expression()?);
                }
                exprs
            } else {
                vec![]
            };

            // ORDER BY (required for window functions)
            if !self.check(&CypherToken::Order) {
                return Err("Window function requires ORDER BY in OVER clause".into());
            }
            self.advance(); // consume ORDER
            self.expect(&CypherToken::By)?;
            let mut order_by = vec![self.parse_order_item()?];
            while self.check(&CypherToken::Comma) {
                self.advance();
                order_by.push(self.parse_order_item()?);
            }

            self.expect(&CypherToken::RParen)?;

            return Ok(Expression::WindowFunction {
                name: lower,
                partition_by,
                order_by,
            });
        }

        Ok(Expression::FunctionCall {
            name,
            args,
            distinct,
        })
    }

    // ========================================================================
    // Count subquery
    // ========================================================================

    /// Parse `count { <pattern(s)> [WHERE <pred>] }` after the `count`
    /// identifier has been consumed; LBrace is next. Mirrors the
    /// `EXISTS { ... }` parse body in
    /// [`super::predicate::parse_comparison_predicate`], but wrapped
    /// as an [`Expression::CountSubquery`] for use in WITH / RETURN /
    /// ORDER BY etc.
    pub(super) fn parse_count_subquery(&mut self) -> Result<Expression, String> {
        self.expect(&CypherToken::LBrace)?;
        let patterns = self.parse_exists_patterns()?;
        let where_clause = if self.check(&CypherToken::Where) {
            self.advance(); // consume WHERE
            Some(Box::new(self.parse_predicate()?))
        } else {
            None
        };
        self.expect(&CypherToken::RBrace)?;
        Ok(Expression::CountSubquery {
            patterns,
            where_clause,
        })
    }

    // ========================================================================
    // Map Projection
    // ========================================================================

    /// Parse map projection: variable { .prop1, .prop2, alias: expr }
    /// The variable name has already been consumed; LBrace is next.
    pub(super) fn parse_map_projection(&mut self, variable: String) -> Result<Expression, String> {
        self.expect(&CypherToken::LBrace)?;

        let mut items = Vec::new();

        while !self.check(&CypherToken::RBrace) {
            if !items.is_empty() {
                self.expect(&CypherToken::Comma)?;
            }

            // Check for .property shorthand or .* all-properties
            if self.check(&CypherToken::Dot) {
                self.advance(); // consume dot
                match self.advance().cloned() {
                    Some(CypherToken::Identifier(prop)) => {
                        items.push(MapProjectionItem::Property(prop));
                    }
                    Some(CypherToken::Star) => {
                        items.push(MapProjectionItem::AllProperties);
                    }
                    _ => {
                        return Err(
                            "Expected property name or '*' after '.' in map projection".into()
                        )
                    }
                }
            } else {
                // alias: expression
                let key = match self.advance().cloned() {
                    Some(CypherToken::Identifier(name)) => name,
                    other => {
                        return Err(format!(
                            "Expected property name or .property in map projection, got {:?}",
                            other
                        ))
                    }
                };
                self.expect(&CypherToken::Colon)?;
                let expr = self.parse_expression()?;
                items.push(MapProjectionItem::Alias { key, expr });
            }
        }

        self.expect(&CypherToken::RBrace)?;

        Ok(Expression::MapProjection { variable, items })
    }

    /// Parse map literal: {key: expr, key2: expr, ...}
    /// The opening LBrace has already been consumed.
    pub(super) fn parse_map_literal(&mut self) -> Result<Expression, String> {
        let mut entries = Vec::new();

        if !self.check(&CypherToken::RBrace) {
            loop {
                let key = match self.advance().cloned() {
                    Some(CypherToken::Identifier(name)) => name,
                    other => {
                        return Err(format!("Expected key name in map literal, got {:?}", other))
                    }
                };
                self.expect(&CypherToken::Colon)?;
                let expr = self.parse_expression()?;
                entries.push((key, expr));

                if self.check(&CypherToken::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        self.expect(&CypherToken::RBrace)?;
        Ok(Expression::MapLiteral(entries))
    }

    // ========================================================================
    // CASE Expression
    // ========================================================================

    /// Parse CASE expression (CASE token already consumed)
    /// Generic form: CASE WHEN predicate THEN result [WHEN ...] [ELSE default] END
    /// Simple form:  CASE operand WHEN value THEN result [WHEN ...] [ELSE default] END
    pub(super) fn parse_case_expression(&mut self) -> Result<Expression, String> {
        // Determine form: if next token is WHEN, it's generic; otherwise parse operand
        let operand = if self.check(&CypherToken::When) {
            None
        } else {
            Some(Box::new(self.parse_expression()?))
        };

        let mut when_clauses = Vec::new();

        // Parse WHEN ... THEN ... pairs
        while self.check(&CypherToken::When) {
            self.advance(); // consume WHEN

            let condition = if operand.is_some() {
                // Simple form: WHEN value — compare against operand
                CaseCondition::Expression(self.parse_expression()?)
            } else {
                // Generic form: WHEN predicate — evaluated as boolean
                CaseCondition::Predicate(self.parse_predicate()?)
            };

            self.expect(&CypherToken::Then)?;
            let result = self.parse_expression()?;
            when_clauses.push((condition, result));
        }

        if when_clauses.is_empty() {
            return Err("CASE expression requires at least one WHEN clause".to_string());
        }

        // Optional ELSE
        let else_expr = if self.check(&CypherToken::Else) {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        self.expect(&CypherToken::End)?;

        Ok(Expression::Case {
            operand,
            when_clauses,
            else_expr,
        })
    }

    /// Parse list comprehension: x IN list_expr WHERE predicate | map_expr ]
    /// Opening [ already consumed.
    pub(super) fn parse_list_comprehension(&mut self) -> Result<Expression, String> {
        // Variable name
        let variable = match self.advance() {
            Some(CypherToken::Identifier(name)) => name.clone(),
            _ => return Err("Expected variable name in list comprehension".to_string()),
        };

        self.expect(&CypherToken::In)?;
        let list_expr = self.parse_expression()?;

        // Optional WHERE filter
        let filter = if self.check(&CypherToken::Where) {
            self.advance();
            Some(Box::new(self.parse_predicate()?))
        } else {
            None
        };

        // Optional | map_expr
        let map_expr = if self.check(&CypherToken::Pipe) {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        self.expect(&CypherToken::RBracket)?;

        Ok(Expression::ListComprehension {
            variable,
            list_expr: Box::new(list_expr),
            filter,
            map_expr,
        })
    }

    /// Parse list quantifier expression: (variable IN list_expr WHERE predicate)
    /// The quantifier keyword has been consumed; LParen is next.
    pub(super) fn parse_list_quantifier_expr(
        &mut self,
        quantifier: ListQuantifier,
    ) -> Result<Expression, String> {
        self.expect(&CypherToken::LParen)?;

        // Variable name
        let variable = match self.advance().cloned() {
            Some(CypherToken::Identifier(name)) => name,
            other => {
                return Err(format!(
                    "Expected variable name in list predicate, got {:?}",
                    other
                ))
            }
        };

        self.expect(&CypherToken::In)?;
        let list_expr = self.parse_expression()?;

        // WHERE predicate
        self.expect(&CypherToken::Where)?;
        let predicate = self.parse_predicate()?;

        self.expect(&CypherToken::RParen)?;

        Ok(Expression::QuantifiedList {
            quantifier,
            variable,
            list_expr: Box::new(list_expr),
            filter: Box::new(predicate),
        })
    }

    /// Parse `reduce(acc = init, var IN list | body)`.
    /// The `reduce` identifier has been consumed; LParen is next.
    pub(super) fn parse_reduce_expr(&mut self) -> Result<Expression, String> {
        self.expect(&CypherToken::LParen)?;

        // accumulator name
        let accumulator = match self.advance().cloned() {
            Some(CypherToken::Identifier(name)) => name,
            other => {
                return Err(format!(
                    "Expected accumulator name after reduce(, got {:?}",
                    other
                ))
            }
        };

        self.expect(&CypherToken::Equals)?;
        let init = self.parse_expression()?;
        self.expect(&CypherToken::Comma)?;

        // iteration variable
        let variable = match self.advance().cloned() {
            Some(CypherToken::Identifier(name)) => name,
            other => {
                return Err(format!(
                    "Expected iteration variable in reduce(), got {:?}",
                    other
                ))
            }
        };

        self.expect(&CypherToken::In)?;
        let list_expr = self.parse_expression()?;
        self.expect(&CypherToken::Pipe)?;
        let body = self.parse_expression()?;
        self.expect(&CypherToken::RParen)?;

        Ok(Expression::Reduce {
            accumulator,
            init: Box::new(init),
            variable,
            list_expr: Box::new(list_expr),
            body: Box::new(body),
        })
    }

    // ========================================================================
    // RETURN Clause
    // ========================================================================
}
