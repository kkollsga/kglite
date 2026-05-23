//! Cypher parser: top-level clauses other than MATCH / WHERE.
//!
//! RETURN, WITH, ORDER BY, LIMIT, SKIP, UNWIND, UNION,
//! CREATE, SET, DELETE, REMOVE, MERGE, CALL.

use super::super::ast::*;
use super::super::tokenizer::CypherToken;
use super::CypherParser;

impl CypherParser {
    pub(super) fn parse_return_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Return)?;

        let distinct = if self.check(&CypherToken::Distinct) {
            self.advance();
            true
        } else {
            false
        };

        let items = self.parse_return_items()?;

        // Optional HAVING clause for post-aggregation filtering
        let having = if self.check(&CypherToken::Having) {
            self.advance();
            Some(self.parse_predicate()?)
        } else {
            None
        };

        Ok(Clause::Return(ReturnClause {
            items,
            distinct,
            having,
            lazy_eligible: false,
            group_limit_hint: None,
        }))
    }

    /// Parse comma-separated return items: expr AS alias, expr AS alias, ...
    pub(super) fn parse_return_items(&mut self) -> Result<Vec<ReturnItem>, String> {
        let mut items = Vec::new();
        items.push(self.parse_return_item()?);

        while self.check(&CypherToken::Comma) {
            self.advance();
            items.push(self.parse_return_item()?);
        }

        Ok(items)
    }

    pub(super) fn parse_return_item(&mut self) -> Result<ReturnItem, String> {
        let expression = self.parse_expression_with_predicates()?;

        let alias = if self.check(&CypherToken::As) {
            self.advance();
            Some(self.try_consume_alias_name()?)
        } else {
            None
        };

        Ok(ReturnItem { expression, alias })
    }

    // ========================================================================
    // WITH Clause
    // ========================================================================

    pub(super) fn parse_with_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::With)?;

        let distinct = if self.check(&CypherToken::Distinct) {
            self.advance();
            true
        } else {
            false
        };

        let items = self.parse_return_items()?;

        // Check for optional HAVING or WHERE in WITH
        let where_clause = if self.check(&CypherToken::Having) || self.check(&CypherToken::Where) {
            self.advance();
            Some(WhereClause {
                predicate: self.parse_predicate()?,
            })
        } else {
            None
        };

        Ok(Clause::With(WithClause {
            items,
            distinct,
            where_clause,
            group_limit_hint: None,
        }))
    }

    // ========================================================================
    // ORDER BY Clause
    // ========================================================================

    pub(super) fn parse_order_by_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Order)?;
        self.expect(&CypherToken::By)?;

        let mut items = Vec::new();
        items.push(self.parse_order_item()?);

        while self.check(&CypherToken::Comma) {
            self.advance();
            items.push(self.parse_order_item()?);
        }

        Ok(Clause::OrderBy(OrderByClause { items }))
    }

    pub(super) fn parse_order_item(&mut self) -> Result<OrderItem, String> {
        let expression = self.parse_expression()?;

        let ascending = match self.peek() {
            Some(CypherToken::Asc) => {
                self.advance();
                true
            }
            Some(CypherToken::Desc) => {
                self.advance();
                false
            }
            _ => true, // default ascending
        };

        // 0.9.0 §2 — optional NULLS FIRST / NULLS LAST modifier.
        // Default placement (when omitted) is NULLS LAST for ASC and
        // NULLS FIRST for DESC, computed at sort time via
        // OrderItem::effective_nulls().
        let nulls = if matches!(self.peek(), Some(CypherToken::Nulls)) {
            self.advance();
            match self.peek() {
                Some(CypherToken::Identifier(ident)) if ident.eq_ignore_ascii_case("first") => {
                    self.advance();
                    Some(crate::graph::languages::cypher::ast::NullsPlacement::First)
                }
                Some(CypherToken::Identifier(ident)) if ident.eq_ignore_ascii_case("last") => {
                    self.advance();
                    Some(crate::graph::languages::cypher::ast::NullsPlacement::Last)
                }
                other => {
                    return Err(format!(
                        "Expected FIRST or LAST after NULLS in ORDER BY, found {:?}",
                        other
                    ));
                }
            }
        } else {
            None
        };

        Ok(OrderItem {
            expression,
            ascending,
            nulls,
        })
    }

    // ========================================================================
    // LIMIT / SKIP
    // ========================================================================

    pub(super) fn parse_limit_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Limit)?;
        let count = self.parse_expression()?;
        Ok(Clause::Limit(LimitClause { count }))
    }

    pub(super) fn parse_skip_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Skip)?;
        let count = self.parse_expression()?;
        Ok(Clause::Skip(SkipClause { count }))
    }

    // ========================================================================
    // UNWIND / UNION (Phase 3 stubs)
    // ========================================================================

    pub(super) fn parse_unwind_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Unwind)?;
        let expression = self.parse_expression()?;
        self.expect(&CypherToken::As)?;
        let alias = self.try_consume_alias_name()?;
        Ok(Clause::Unwind(UnwindClause { expression, alias }))
    }

    pub(super) fn parse_union_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Union)?;
        let all = if self.check(&CypherToken::All) {
            self.advance();
            true
        } else {
            false
        };

        // Parse the rest as a new query
        let query = self.parse_query()?;

        Ok(Clause::Union(UnionClause {
            all,
            query: Box::new(query),
            kind: SetOpKind::Union,
        }))
    }

    pub(super) fn parse_intersect_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Intersect)?;
        let query = self.parse_query()?;
        Ok(Clause::Union(UnionClause {
            all: false,
            query: Box::new(query),
            kind: SetOpKind::Intersect,
        }))
    }

    pub(super) fn parse_except_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Except)?;
        let query = self.parse_query()?;
        Ok(Clause::Union(UnionClause {
            all: false,
            query: Box::new(query),
            kind: SetOpKind::Except,
        }))
    }

    // ========================================================================
    // CREATE Clause
    // ========================================================================

    pub(super) fn parse_create_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Create)?;
        let mut patterns = Vec::new();

        loop {
            patterns.push(self.parse_create_pattern()?);
            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(Clause::Create(CreateClause { patterns }))
    }

    /// Parse a single CREATE path pattern: (node)-[edge]->(node)...
    pub(super) fn parse_create_pattern(&mut self) -> Result<CreatePattern, String> {
        let mut elements = Vec::new();
        elements.push(CreateElement::Node(self.parse_create_node()?));

        // Parse optional edge-node chains
        while matches!(
            self.peek(),
            Some(CypherToken::Dash) | Some(CypherToken::LessThan)
        ) {
            elements.push(CreateElement::Edge(self.parse_create_edge()?));
            elements.push(CreateElement::Node(self.parse_create_node()?));
        }

        Ok(CreatePattern { elements })
    }

    /// Parse a node in a CREATE pattern: (var:Label {key: expr, ...})
    pub(super) fn parse_create_node(&mut self) -> Result<CreateNodePattern, String> {
        self.expect(&CypherToken::LParen)?;
        let mut variable = None;
        let mut label = None;
        let mut properties = Vec::new();

        // Parse optional variable name
        if let Some(CypherToken::Identifier(_)) = self.peek() {
            // It's a variable if followed by : or { or )
            // (not a property access or function call)
            if let Some(CypherToken::Identifier(name)) = self.peek().cloned() {
                self.advance();
                variable = Some(name);
            }
        }

        // Parse optional :Label
        if self.check(&CypherToken::Colon) {
            self.advance();
            if let Some(CypherToken::Identifier(name)) = self.peek().cloned() {
                self.advance();
                label = Some(name);
            } else {
                return Err("Expected label name after ':'".to_string());
            }
        }

        // Parse optional {key: expr, ...}
        if self.check(&CypherToken::LBrace) {
            properties = self.parse_create_properties()?;
        }

        self.expect(&CypherToken::RParen)?;
        Ok(CreateNodePattern {
            variable,
            label,
            properties,
        })
    }

    /// Parse CREATE properties: {key: expr, key: expr, ...}
    pub(super) fn parse_create_properties(&mut self) -> Result<Vec<(String, Expression)>, String> {
        self.expect(&CypherToken::LBrace)?;
        let mut props = Vec::new();

        if !self.check(&CypherToken::RBrace) {
            loop {
                let key = match self.peek().cloned() {
                    Some(CypherToken::Identifier(k)) => {
                        self.advance();
                        k
                    }
                    other => {
                        return Err(format!("Expected property key, got {:?}", other));
                    }
                };
                self.expect(&CypherToken::Colon)?;
                let value_expr = self.parse_expression()?;
                props.push((key, value_expr));

                if self.check(&CypherToken::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        self.expect(&CypherToken::RBrace)?;
        Ok(props)
    }

    /// Parse an edge in a CREATE pattern: -[var:TYPE {props}]-> or <-[var:TYPE {props}]-
    pub(super) fn parse_create_edge(&mut self) -> Result<CreateEdgePattern, String> {
        // Handle direction prefix: <- means incoming
        let incoming = if self.check(&CypherToken::LessThan) {
            self.advance();
            true
        } else {
            false
        };

        self.expect(&CypherToken::Dash)?;
        self.expect(&CypherToken::LBracket)?;

        let mut variable = None;
        let mut connection_type = None;
        let mut properties = Vec::new();

        // Parse optional variable name
        if let Some(CypherToken::Identifier(_)) = self.peek() {
            // Check if followed by : (variable:TYPE) or ] (just variable)
            if matches!(
                self.peek_at(1),
                Some(CypherToken::Colon) | Some(CypherToken::RBracket)
            ) {
                if let Some(CypherToken::Identifier(name)) = self.peek().cloned() {
                    self.advance();
                    variable = Some(name);
                }
            }
        }

        // Parse :TYPE (required for CREATE)
        if self.check(&CypherToken::Colon) {
            self.advance();
            if let Some(CypherToken::Identifier(name)) = self.peek().cloned() {
                self.advance();
                connection_type = Some(name);
            } else {
                return Err("Expected relationship type after ':'".to_string());
            }
        }

        let conn_type = connection_type
            .ok_or_else(|| "CREATE requires a relationship type (e.g. [:KNOWS])".to_string())?;

        // Parse optional properties
        if self.check(&CypherToken::LBrace) {
            properties = self.parse_create_properties()?;
        }

        self.expect(&CypherToken::RBracket)?;
        self.expect(&CypherToken::Dash)?;

        // Handle direction suffix
        let direction = if self.check(&CypherToken::GreaterThan) {
            self.advance();
            if incoming {
                return Err("Cannot have both < and > in CREATE edge pattern".to_string());
            }
            CreateEdgeDirection::Outgoing
        } else if incoming {
            CreateEdgeDirection::Incoming
        } else {
            return Err("CREATE edges must have a direction (-> or <-)".to_string());
        };

        Ok(CreateEdgePattern {
            variable,
            connection_type: conn_type,
            direction,
            properties,
        })
    }

    // ========================================================================
    // SET Clause
    // ========================================================================

    pub(super) fn parse_set_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Set)?;
        let items = self.parse_set_items()?;
        Ok(Clause::Set(SetClause { items }))
    }

    /// Parse comma-separated SET items (shared by SET and MERGE ON CREATE/ON MATCH)
    pub(super) fn parse_set_items(&mut self) -> Result<Vec<SetItem>, String> {
        let mut items = Vec::new();

        loop {
            let var_name = match self.peek().cloned() {
                Some(CypherToken::Identifier(name)) => {
                    self.advance();
                    name
                }
                other => {
                    return Err(format!("Expected variable name in SET, got {:?}", other));
                }
            };

            if self.check(&CypherToken::Dot) {
                // Property assignment: var.prop = expr
                self.advance(); // consume .
                let prop_name = match self.peek().cloned() {
                    Some(CypherToken::Identifier(name)) => {
                        self.advance();
                        name
                    }
                    other => {
                        return Err(format!("Expected property name after '.', got {:?}", other));
                    }
                };
                self.expect(&CypherToken::Equals)?;
                let expression = self.parse_expression()?;
                items.push(SetItem::Property {
                    variable: var_name,
                    property: prop_name,
                    expression,
                });
            } else if self.check(&CypherToken::Colon) {
                // Label assignment: var:Label
                self.advance();
                let label = match self.peek().cloned() {
                    Some(CypherToken::Identifier(name)) => {
                        self.advance();
                        name
                    }
                    other => {
                        return Err(format!("Expected label name after ':', got {:?}", other));
                    }
                };
                items.push(SetItem::Label {
                    variable: var_name,
                    label,
                });
            } else {
                return Err("Expected '.' or ':' after variable name in SET".to_string());
            }

            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(items)
    }

    // ========================================================================
    // DELETE Clause
    // ========================================================================

    pub(super) fn parse_delete_clause(&mut self) -> Result<Clause, String> {
        let detach = if self.check(&CypherToken::Detach) {
            self.advance(); // consume DETACH
            true
        } else {
            false
        };
        self.expect(&CypherToken::Delete)?;

        let mut expressions = Vec::new();
        loop {
            let expr = match self.peek().cloned() {
                Some(CypherToken::Identifier(name)) => {
                    self.advance();
                    Expression::Variable(name)
                }
                other => {
                    return Err(format!("Expected variable name in DELETE, got {:?}", other));
                }
            };
            expressions.push(expr);

            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(Clause::Delete(DeleteClause {
            detach,
            expressions,
        }))
    }

    // ========================================================================
    // REMOVE Clause
    // ========================================================================

    pub(super) fn parse_remove_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Remove)?;
        let mut items = Vec::new();

        loop {
            let var_name = match self.peek().cloned() {
                Some(CypherToken::Identifier(name)) => {
                    self.advance();
                    name
                }
                other => {
                    return Err(format!("Expected variable name in REMOVE, got {:?}", other));
                }
            };

            if self.check(&CypherToken::Dot) {
                // Property removal: var.prop
                self.advance(); // consume .
                let prop_name = match self.peek().cloned() {
                    Some(CypherToken::Identifier(name)) => {
                        self.advance();
                        name
                    }
                    other => {
                        return Err(format!(
                            "Expected property name after '.' in REMOVE, got {:?}",
                            other
                        ));
                    }
                };
                items.push(RemoveItem::Property {
                    variable: var_name,
                    property: prop_name,
                });
            } else if self.check(&CypherToken::Colon) {
                // Label removal: var:Label
                self.advance(); // consume :
                let label = match self.peek().cloned() {
                    Some(CypherToken::Identifier(name)) => {
                        self.advance();
                        name
                    }
                    other => {
                        return Err(format!(
                            "Expected label name after ':' in REMOVE, got {:?}",
                            other
                        ));
                    }
                };
                items.push(RemoveItem::Label {
                    variable: var_name,
                    label,
                });
            } else {
                return Err("Expected '.' or ':' after variable name in REMOVE".to_string());
            }

            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(Clause::Remove(RemoveClause { items }))
    }

    // ========================================================================
    // MERGE Clause
    // ========================================================================

    pub(super) fn parse_merge_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Merge)?;
        let pattern = self.parse_create_pattern()?;

        let mut on_create = None;
        let mut on_match = None;

        // Parse optional ON CREATE SET / ON MATCH SET (can appear in either order)
        while self.check(&CypherToken::On) {
            self.advance(); // consume ON
            match self.peek() {
                Some(CypherToken::Create) => {
                    self.advance(); // consume CREATE
                    self.expect(&CypherToken::Set)?;
                    on_create = Some(self.parse_set_items()?);
                }
                Some(CypherToken::Match) => {
                    self.advance(); // consume MATCH
                    self.expect(&CypherToken::Set)?;
                    on_match = Some(self.parse_set_items()?);
                }
                other => {
                    return Err(format!(
                        "Expected CREATE or MATCH after ON in MERGE, got {:?}",
                        other
                    ));
                }
            }
        }

        Ok(Clause::Merge(MergeClause {
            pattern,
            on_create,
            on_match,
        }))
    }

    // ========================================================================
    // CALL Clause
    // ========================================================================

    pub(super) fn parse_call_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Call)?;

        // Parse procedure name (may be namespaced: `db.labels`, `apoc.coll.sum`).
        // The tokenizer splits these into Identifier/Dot/Identifier sequences;
        // we re-join them into a single flat `String` so the executor dispatch
        // can match on the qualified name. Phase A.3 (Bolt-compat: `db.*`).
        let mut procedure_name = match self.peek().cloned() {
            Some(CypherToken::Identifier(name)) => {
                self.advance();
                name
            }
            other => {
                return Err(format!(
                    "Expected procedure name after CALL (e.g. `pagerank`, `db.labels`), \
                     got {:?}",
                    other
                ));
            }
        };
        while self.check(&CypherToken::Dot) {
            self.advance(); // consume `.`
            match self.peek().cloned() {
                Some(CypherToken::Identifier(part)) => {
                    self.advance();
                    procedure_name.push('.');
                    procedure_name.push_str(&part);
                }
                other => {
                    return Err(format!(
                        "Expected identifier after `.` in procedure name, got {:?}",
                        other
                    ));
                }
            }
        }

        // Parse argument list: ( [{key: val, ...}] )
        self.expect(&CypherToken::LParen)?;
        let parameters = if self.check(&CypherToken::LBrace) {
            self.parse_create_properties()?
        } else if !self.check(&CypherToken::RParen) {
            return Err(format!(
                "CALL parameters must use map syntax: CALL {}({{key: value, ...}}). \
                 Example: CALL {}({{damping_factor: 0.85}})",
                procedure_name, procedure_name
            ));
        } else {
            Vec::new()
        };
        self.expect(&CypherToken::RParen)?;

        // Parse YIELD clause (required)
        if !self.check(&CypherToken::Yield) {
            return Err(
                "CALL requires a YIELD clause, e.g. CALL pagerank() YIELD node, score".to_string(),
            );
        }
        self.advance(); // consume YIELD

        let yield_items = self.parse_yield_items()?;
        if yield_items.is_empty() {
            return Err("YIELD requires at least one column name".to_string());
        }

        Ok(Clause::Call(CallClause {
            procedure_name,
            parameters,
            yield_items,
        }))
    }

    /// Parse comma-separated YIELD items: name [AS alias], ...
    pub(super) fn parse_yield_items(&mut self) -> Result<Vec<YieldItem>, String> {
        let mut items = Vec::new();

        loop {
            let name = match self.peek().cloned() {
                Some(CypherToken::Identifier(n)) => {
                    self.advance();
                    n
                }
                other => {
                    return Err(format!("Expected column name in YIELD, got {:?}", other));
                }
            };

            let alias = if self.check(&CypherToken::As) {
                self.advance();
                Some(self.try_consume_alias_name()?)
            } else {
                None
            };

            items.push(YieldItem { name, alias });

            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(items)
    }
}
