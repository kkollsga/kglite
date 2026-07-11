//! Parser regression tests extracted from mod.rs.

use super::*;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_match_return() {
        let query = parse_cypher("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(query.clauses.len(), 2);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Return(_)));
    }

    #[test]
    fn test_match_where_return() {
        let query =
            parse_cypher("MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name").unwrap();
        assert_eq!(query.clauses.len(), 3);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Where(_)));
        assert!(matches!(&query.clauses[2], Clause::Return(_)));

        // Check WHERE predicate
        if let Clause::Where(w) = &query.clauses[1] {
            if let Predicate::Comparison {
                left,
                operator,
                right,
            } = &w.predicate
            {
                assert!(
                    matches!(left, Expression::PropertyAccess { variable, property }
                    if variable == "n" && property == "age")
                );
                assert_eq!(*operator, ComparisonOp::GreaterThan);
                assert!(matches!(right, Expression::Literal(Value::Int64(30))));
            } else {
                panic!("Expected comparison predicate");
            }
        } else {
            panic!("Expected WHERE clause");
        }

        // Check RETURN alias
        if let Clause::Return(r) = &query.clauses[2] {
            assert_eq!(r.items.len(), 1);
            assert_eq!(r.items[0].alias, Some("name".to_string()));
        }
    }

    #[test]
    fn test_where_and_or() {
        let query = parse_cypher(
            "MATCH (n:Person) WHERE n.age > 18 AND n.city = 'Oslo' OR n.vip = true RETURN n",
        )
        .unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            // Should be: (age > 18 AND city = 'Oslo') OR vip = true
            assert!(matches!(&w.predicate, Predicate::Or(_, _)));
        }
    }

    #[test]
    fn test_where_not() {
        let query = parse_cypher("MATCH (n:Person) WHERE NOT n.active = false RETURN n").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            assert!(matches!(&w.predicate, Predicate::Not(_)));
        }
    }

    #[test]
    fn test_where_is_null() {
        let query = parse_cypher("MATCH (n:Person) WHERE n.email IS NULL RETURN n").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            assert!(matches!(&w.predicate, Predicate::IsNull(_)));
        }
    }

    #[test]
    fn test_where_is_not_null() {
        let query = parse_cypher("MATCH (n:Person) WHERE n.email IS NOT NULL RETURN n").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            assert!(matches!(&w.predicate, Predicate::IsNotNull(_)));
        }
    }

    #[test]
    fn test_where_in_list() {
        let query = parse_cypher(
            "MATCH (n:Person) WHERE n.city IN ['Oslo', 'Bergen', 'Trondheim'] RETURN n",
        )
        .unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            if let Predicate::In { expr: _, list } = &w.predicate {
                assert_eq!(list.len(), 3);
            } else {
                panic!("Expected IN predicate");
            }
        }
    }

    #[test]
    fn test_return_multiple_items() {
        let query =
            parse_cypher("MATCH (n:Person) RETURN n.name AS name, n.age AS age, n.city").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert_eq!(r.items.len(), 3);
            assert_eq!(r.items[0].alias, Some("name".to_string()));
            assert_eq!(r.items[1].alias, Some("age".to_string()));
            assert_eq!(r.items[2].alias, None);
        }
    }

    #[test]
    fn test_return_distinct() {
        let query = parse_cypher("MATCH (n:Person) RETURN DISTINCT n.city").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert!(r.distinct);
        }
    }

    #[test]
    fn test_return_function_call() {
        let query = parse_cypher("MATCH (n:Person) RETURN count(n) AS total").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::FunctionCall {
                name,
                args,
                distinct,
            } = &r.items[0].expression
            {
                assert_eq!(name, "count");
                assert_eq!(args.len(), 1);
                assert!(!distinct);
            } else {
                panic!("Expected function call");
            }
        }
    }

    #[test]
    fn test_return_count_star() {
        let query = parse_cypher("MATCH (n:Person) RETURN count(*) AS total").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::FunctionCall { args, .. } = &r.items[0].expression {
                assert!(matches!(&args[0], Expression::Star));
            }
        }
    }

    #[test]
    fn test_return_count_distinct() {
        let query =
            parse_cypher("MATCH (n:Person) RETURN count(DISTINCT n.city) AS cities").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::FunctionCall { distinct, .. } = &r.items[0].expression {
                assert!(distinct);
            }
        }
    }

    #[test]
    fn test_order_by_limit_skip() {
        let query =
            parse_cypher("MATCH (n:Person) RETURN n.name ORDER BY n.age DESC SKIP 5 LIMIT 10")
                .unwrap();

        assert!(matches!(&query.clauses[2], Clause::OrderBy(_)));
        assert!(matches!(&query.clauses[3], Clause::Skip(_)));
        assert!(matches!(&query.clauses[4], Clause::Limit(_)));

        if let Clause::OrderBy(o) = &query.clauses[2] {
            assert_eq!(o.items.len(), 1);
            assert!(!o.items[0].ascending);
        }
    }

    #[test]
    fn test_with_clause() {
        let query = parse_cypher(
            "MATCH (n:Person) WITH n.city AS city, count(n) AS cnt WHERE cnt > 5 RETURN city, cnt",
        )
        .unwrap();

        assert!(matches!(&query.clauses[1], Clause::With(_)));
        if let Clause::With(w) = &query.clauses[1] {
            assert_eq!(w.items.len(), 2);
            assert!(w.where_clause.is_some());
        }
    }

    #[test]
    fn test_optional_match() {
        let query =
            parse_cypher("MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(f:Person) RETURN n, f")
                .unwrap();

        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::OptionalMatch(_)));
        assert!(matches!(&query.clauses[2], Clause::Return(_)));
    }

    #[test]
    fn test_match_with_edge_pattern() {
        let query =
            parse_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name").unwrap();

        if let Clause::Match(m) = &query.clauses[0] {
            assert_eq!(m.patterns.len(), 1);
            assert_eq!(m.patterns[0].elements.len(), 3); // node, edge, node
        }
    }

    #[test]
    fn test_match_with_var_length() {
        let query = parse_cypher("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) RETURN a, b").unwrap();

        assert!(matches!(&query.clauses[0], Clause::Match(_)));
    }

    #[test]
    fn test_multiple_match_patterns() {
        let query = parse_cypher("MATCH (a:Person), (b:Company) RETURN a, b").unwrap();

        if let Clause::Match(m) = &query.clauses[0] {
            assert_eq!(m.patterns.len(), 2);
        }
    }

    #[test]
    fn test_case_insensitive() {
        let query = parse_cypher("match (n:Person) where n.age > 30 return n").unwrap();
        assert_eq!(query.clauses.len(), 3);
    }

    #[test]
    fn test_arithmetic_in_return() {
        let query =
            parse_cypher("MATCH (n:Product) RETURN n.price * 1.1 AS price_with_tax").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert!(matches!(&r.items[0].expression, Expression::Multiply(_, _)));
        }
    }

    #[test]
    fn test_where_contains() {
        let query = parse_cypher("MATCH (n:Person) WHERE n.name CONTAINS 'son' RETURN n").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            assert!(matches!(&w.predicate, Predicate::Contains { .. }));
        }
    }

    #[test]
    fn test_unwind() {
        let query = parse_cypher("UNWIND [1, 2, 3] AS x RETURN x").unwrap();

        assert!(matches!(&query.clauses[0], Clause::Unwind(_)));
        if let Clause::Unwind(u) = &query.clauses[0] {
            assert_eq!(u.alias, "x");
        }
    }

    #[test]
    fn test_case_generic_form() {
        let query = parse_cypher(
            "MATCH (n:Person) RETURN CASE WHEN n.age > 18 THEN 'adult' ELSE 'minor' END AS category",
        )
        .unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert!(
                matches!(&r.items[0].expression, Expression::Case { operand, .. } if operand.is_none())
            );
            assert_eq!(r.items[0].alias, Some("category".to_string()));
        } else {
            panic!("Expected RETURN clause");
        }
    }

    #[test]
    fn test_case_simple_form() {
        let query = parse_cypher(
            "MATCH (n:Person) RETURN CASE n.city WHEN 'Oslo' THEN 'capital' WHEN 'Bergen' THEN 'west' ELSE 'other' END",
        )
        .unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::Case {
                operand,
                when_clauses,
                else_expr,
            } = &r.items[0].expression
            {
                assert!(operand.is_some());
                assert_eq!(when_clauses.len(), 2);
                assert!(else_expr.is_some());
            } else {
                panic!("Expected CASE expression");
            }
        }
    }

    #[test]
    fn test_case_no_else() {
        let query =
            parse_cypher("MATCH (n:Person) RETURN CASE WHEN n.age > 18 THEN 'adult' END").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::Case { else_expr, .. } = &r.items[0].expression {
                assert!(else_expr.is_none());
            } else {
                panic!("Expected CASE expression");
            }
        }
    }

    #[test]
    fn test_parameter_in_expression() {
        let query = parse_cypher("MATCH (n:Person) WHERE n.age > $min_age RETURN n.name").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            if let Predicate::Comparison { right, .. } = &w.predicate {
                assert!(matches!(right, Expression::Parameter(name) if name == "min_age"));
            } else {
                panic!("Expected comparison predicate");
            }
        }
    }

    #[test]
    fn test_parameter_in_return() {
        let query = parse_cypher("MATCH (n:Person) RETURN n.name, $label AS label").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert!(
                matches!(&r.items[1].expression, Expression::Parameter(name) if name == "label")
            );
        }
    }

    // ========================================================================
    // CREATE Clause
    // ========================================================================

    #[test]
    fn test_parse_create_node() {
        let query = parse_cypher("CREATE (n:Person {name: 'Alice', age: 30})").unwrap();
        assert_eq!(query.clauses.len(), 1);

        if let Clause::Create(c) = &query.clauses[0] {
            assert_eq!(c.patterns.len(), 1);
            assert_eq!(c.patterns[0].elements.len(), 1);
            if let CreateElement::Node(np) = &c.patterns[0].elements[0] {
                assert_eq!(np.variable, Some("n".to_string()));
                assert_eq!(np.label, Some("Person".to_string()));
                assert_eq!(np.properties.len(), 2);
                assert_eq!(np.properties[0].0, "name");
                assert_eq!(np.properties[1].0, "age");
            } else {
                panic!("Expected node element");
            }
        } else {
            panic!("Expected CREATE clause");
        }
    }

    #[test]
    fn test_parse_create_edge() {
        let query = parse_cypher("MATCH (a:Person), (b:Person) CREATE (a)-[:KNOWS]->(b)").unwrap();
        assert_eq!(query.clauses.len(), 2);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Create(_)));

        if let Clause::Create(c) = &query.clauses[1] {
            assert_eq!(c.patterns[0].elements.len(), 3); // node, edge, node
            if let CreateElement::Edge(ep) = &c.patterns[0].elements[1] {
                assert_eq!(ep.connection_type, "KNOWS");
                assert_eq!(ep.direction, CreateEdgeDirection::Outgoing);
            } else {
                panic!("Expected edge element");
            }
        }
    }

    #[test]
    fn test_parse_create_path() {
        let query =
            parse_cypher("CREATE (a:Person {name: 'A'})-[:KNOWS]->(b:Person {name: 'B'})").unwrap();

        if let Clause::Create(c) = &query.clauses[0] {
            assert_eq!(c.patterns[0].elements.len(), 3);
            assert!(matches!(&c.patterns[0].elements[0], CreateElement::Node(_)));
            assert!(matches!(&c.patterns[0].elements[1], CreateElement::Edge(_)));
            assert!(matches!(&c.patterns[0].elements[2], CreateElement::Node(_)));
        }
    }

    #[test]
    fn test_parse_create_with_params() {
        let query = parse_cypher("CREATE (n:Person {name: $name, age: $age})").unwrap();

        if let Clause::Create(c) = &query.clauses[0] {
            if let CreateElement::Node(np) = &c.patterns[0].elements[0] {
                assert!(matches!(&np.properties[0].1, Expression::Parameter(n) if n == "name"));
                assert!(matches!(&np.properties[1].1, Expression::Parameter(n) if n == "age"));
            }
        }
    }

    #[test]
    fn test_parse_create_incoming_edge() {
        let query =
            parse_cypher("MATCH (a:Person), (b:Person) CREATE (a)<-[:FOLLOWS]-(b)").unwrap();

        if let Clause::Create(c) = &query.clauses[1] {
            if let CreateElement::Edge(ep) = &c.patterns[0].elements[1] {
                assert_eq!(ep.connection_type, "FOLLOWS");
                assert_eq!(ep.direction, CreateEdgeDirection::Incoming);
            }
        }
    }

    // ========================================================================
    // SET Clause
    // ========================================================================

    #[test]
    fn test_parse_set_property() {
        let query = parse_cypher("MATCH (n:Person) SET n.age = 31").unwrap();
        assert_eq!(query.clauses.len(), 2);
        assert!(matches!(&query.clauses[1], Clause::Set(_)));

        if let Clause::Set(s) = &query.clauses[1] {
            assert_eq!(s.items.len(), 1);
            if let SetItem::Property {
                variable,
                property,
                expression,
            } = &s.items[0]
            {
                assert_eq!(variable, "n");
                assert_eq!(property, "age");
                assert!(matches!(expression, Expression::Literal(Value::Int64(31))));
            }
        }
    }

    #[test]
    fn test_parse_set_multiple() {
        let query = parse_cypher("MATCH (n:Person) SET n.age = 31, n.city = 'Bergen'").unwrap();

        if let Clause::Set(s) = &query.clauses[1] {
            assert_eq!(s.items.len(), 2);
            if let SetItem::Property { property, .. } = &s.items[0] {
                assert_eq!(property, "age");
            }
            if let SetItem::Property { property, .. } = &s.items[1] {
                assert_eq!(property, "city");
            }
        }
    }

    #[test]
    fn test_parse_set_expression() {
        let query = parse_cypher("MATCH (n:Person) SET n.salary = n.salary * 1.1").unwrap();

        if let Clause::Set(s) = &query.clauses[1] {
            if let SetItem::Property { expression, .. } = &s.items[0] {
                assert!(matches!(expression, Expression::Multiply(_, _)));
            }
        }
    }

    #[test]
    fn test_parse_match_create_set_return() {
        let query = parse_cypher(
            "MATCH (a:Person) CREATE (a)-[:RATED]->(r:Review {text: 'Great'}) SET a.reviews = a.reviews + 1 RETURN a, r",
        ).unwrap();

        assert_eq!(query.clauses.len(), 4);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Create(_)));
        assert!(matches!(&query.clauses[2], Clause::Set(_)));
        assert!(matches!(&query.clauses[3], Clause::Return(_)));
    }

    // ========================================================================
    // DELETE Clause
    // ========================================================================

    #[test]
    fn test_parse_delete() {
        let query = parse_cypher("MATCH (n:Person) DELETE n").unwrap();
        assert_eq!(query.clauses.len(), 2);
        if let Clause::Delete(d) = &query.clauses[1] {
            assert!(!d.detach);
            assert_eq!(d.expressions.len(), 1);
            assert!(matches!(&d.expressions[0], Expression::Variable(v) if v == "n"));
        } else {
            panic!("Expected DELETE clause");
        }
    }

    #[test]
    fn test_parse_detach_delete() {
        let query = parse_cypher("MATCH (n:Person) DETACH DELETE n").unwrap();
        if let Clause::Delete(d) = &query.clauses[1] {
            assert!(d.detach);
            assert_eq!(d.expressions.len(), 1);
        } else {
            panic!("Expected DELETE clause");
        }
    }

    #[test]
    fn test_parse_delete_multiple() {
        let query = parse_cypher("MATCH (a)-[r]->(b) DELETE a, r, b").unwrap();
        if let Clause::Delete(d) = &query.clauses[1] {
            assert_eq!(d.expressions.len(), 3);
        }
    }

    // ========================================================================
    // REMOVE Clause
    // ========================================================================

    #[test]
    fn test_parse_remove_property() {
        let query = parse_cypher("MATCH (n:Person) REMOVE n.age").unwrap();
        assert!(matches!(&query.clauses[1], Clause::Remove(_)));
        if let Clause::Remove(r) = &query.clauses[1] {
            assert_eq!(r.items.len(), 1);
            if let RemoveItem::Property { variable, property } = &r.items[0] {
                assert_eq!(variable, "n");
                assert_eq!(property, "age");
            } else {
                panic!("Expected property removal");
            }
        }
    }

    #[test]
    fn test_parse_remove_multiple() {
        let query = parse_cypher("MATCH (n:Person) REMOVE n.age, n.city").unwrap();
        if let Clause::Remove(r) = &query.clauses[1] {
            assert_eq!(r.items.len(), 2);
        }
    }

    #[test]
    fn test_parse_remove_label() {
        let query = parse_cypher("MATCH (n:Person) REMOVE n:Temporary").unwrap();
        if let Clause::Remove(r) = &query.clauses[1] {
            assert!(
                matches!(&r.items[0], RemoveItem::Label { variable, label } if variable == "n" && label == "Temporary")
            );
        }
    }

    // ========================================================================
    // MERGE Clause
    // ========================================================================

    #[test]
    fn test_parse_merge_node() {
        let query = parse_cypher("MERGE (n:Person {name: 'Alice'})").unwrap();
        assert_eq!(query.clauses.len(), 1);
        assert!(matches!(&query.clauses[0], Clause::Merge(_)));
        if let Clause::Merge(m) = &query.clauses[0] {
            assert_eq!(m.pattern.elements.len(), 1);
            assert!(m.on_create.is_none());
            assert!(m.on_match.is_none());
        }
    }

    #[test]
    fn test_parse_merge_on_create() {
        let query =
            parse_cypher("MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.age = 30").unwrap();
        if let Clause::Merge(m) = &query.clauses[0] {
            assert!(m.on_create.is_some());
            assert!(m.on_match.is_none());
            assert_eq!(m.on_create.as_ref().unwrap().len(), 1);
        }
    }

    #[test]
    fn test_parse_merge_on_match() {
        let query =
            parse_cypher("MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.visits = 1").unwrap();
        if let Clause::Merge(m) = &query.clauses[0] {
            assert!(m.on_create.is_none());
            assert!(m.on_match.is_some());
        }
    }

    #[test]
    fn test_parse_merge_both() {
        let query = parse_cypher(
            "MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.age = 30 ON MATCH SET n.visits = 1",
        )
        .unwrap();
        if let Clause::Merge(m) = &query.clauses[0] {
            assert!(m.on_create.is_some());
            assert!(m.on_match.is_some());
        }
    }

    #[test]
    fn test_parse_merge_relationship() {
        let query = parse_cypher("MATCH (a:Person), (b:Person) MERGE (a)-[r:KNOWS]->(b)").unwrap();
        assert_eq!(query.clauses.len(), 2);
        if let Clause::Merge(m) = &query.clauses[1] {
            assert_eq!(m.pattern.elements.len(), 3);
        }
    }

    #[test]
    fn test_reserved_word_as_alias() {
        // Keywords should be valid alias names after AS
        for keyword in &[
            "optional", "match", "where", "return", "order", "limit", "type", "set", "all",
            "distinct", "contains", "exists", "null", "true", "false", "in", "is", "not",
        ] {
            let query_str = format!("MATCH (n) RETURN n AS {}", keyword);
            let query = parse_cypher(&query_str)
                .unwrap_or_else(|e| panic!("Failed to parse 'RETURN n AS {}': {}", keyword, e));
            if let Clause::Return(ret) = &query.clauses[1] {
                assert_eq!(
                    ret.items[0].alias.as_deref(),
                    Some(*keyword),
                    "Alias should be '{}' for keyword",
                    keyword
                );
            } else {
                panic!("Expected RETURN clause");
            }
        }
    }

    #[test]
    fn test_reserved_word_as_unwind_alias() {
        let query = parse_cypher("UNWIND [1,2] AS optional").unwrap();
        if let Clause::Unwind(u) = &query.clauses[0] {
            assert_eq!(u.alias, "optional");
        } else {
            panic!("Expected UNWIND clause");
        }
    }

    #[test]
    fn test_reserved_word_as_yield_alias() {
        let query = parse_cypher("CALL pagerank() YIELD node AS optional, score AS limit").unwrap();
        if let Clause::Call(c) = &query.clauses[0] {
            assert_eq!(c.yield_items[0].alias.as_deref(), Some("optional"));
            assert_eq!(c.yield_items[1].alias.as_deref(), Some("limit"));
        } else {
            panic!("Expected CALL clause");
        }
    }

    // ========================================================================
    // CALL { } subqueries (Phase 1: parser + AST node)
    // ========================================================================

    #[test]
    fn test_call_subquery_uncorrelated() {
        let query =
            parse_cypher("CALL { MATCH (n:Person) RETURN count(n) AS c } RETURN c").unwrap();
        assert_eq!(query.clauses.len(), 2);
        if let Clause::CallSubquery { import, body } = &query.clauses[0] {
            assert!(import.is_empty(), "uncorrelated subquery has no imports");
            assert_eq!(body.clauses.len(), 2);
            assert!(matches!(&body.clauses[0], Clause::Match(_)));
            assert!(matches!(&body.clauses[1], Clause::Return(_)));
        } else {
            panic!("Expected CallSubquery, got {:?}", query.clauses[0]);
        }
        assert!(matches!(&query.clauses[1], Clause::Return(_)));
    }

    #[test]
    fn test_call_subquery_correlated_importing_with() {
        let query = parse_cypher(
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } \
             RETURN p.name, c",
        )
        .unwrap();
        assert_eq!(query.clauses.len(), 3);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        if let Clause::CallSubquery { import, body } = &query.clauses[1] {
            assert_eq!(import, &vec!["p".to_string()]);
            // The importing WITH is stripped from the body.
            assert_eq!(body.clauses.len(), 2);
            assert!(matches!(&body.clauses[0], Clause::Match(_)));
            assert!(matches!(&body.clauses[1], Clause::Return(_)));
        } else {
            panic!("Expected CallSubquery, got {:?}", query.clauses[1]);
        }
    }

    #[test]
    fn test_call_subquery_multi_import() {
        let query = parse_cypher(
            "MATCH (p:Person), (q:Person) CALL { WITH p, q MATCH (p)-[:KNOWS]->(q) \
             RETURN count(*) AS c } RETURN c",
        )
        .unwrap();
        if let Clause::CallSubquery { import, .. } = &query.clauses[1] {
            assert_eq!(import, &vec!["p".to_string(), "q".to_string()]);
        } else {
            panic!("Expected CallSubquery");
        }
    }

    #[test]
    fn test_call_subquery_nested_braces() {
        // Nested CALL {} plus a map literal containing a '}' must not close early.
        let query = parse_cypher(
            "CALL { CALL { MATCH (n) RETURN n LIMIT 1 } MATCH (m {tag: 'x}y'}) \
             RETURN m AS r, {a: 1, b: 2} AS meta } RETURN r, meta",
        )
        .unwrap();
        assert_eq!(query.clauses.len(), 2);
        if let Clause::CallSubquery { import, body } = &query.clauses[0] {
            assert!(import.is_empty());
            // body = [nested CallSubquery, MATCH, RETURN]
            assert_eq!(body.clauses.len(), 3);
            assert!(matches!(&body.clauses[0], Clause::CallSubquery { .. }));
            assert!(matches!(&body.clauses[1], Clause::Match(_)));
            assert!(matches!(&body.clauses[2], Clause::Return(_)));
        } else {
            panic!("Expected outer CallSubquery, got {:?}", query.clauses[0]);
        }
    }

    #[test]
    fn test_call_subquery_map_literal_not_closing() {
        // A RETURN'd map literal at the end of the body must not be mistaken
        // for the subquery's closing brace.
        let query = parse_cypher("CALL { MATCH (n) RETURN {a: n.x} AS m } RETURN m").unwrap();
        assert_eq!(query.clauses.len(), 2);
        assert!(matches!(&query.clauses[0], Clause::CallSubquery { .. }));
        assert!(matches!(&query.clauses[1], Clause::Return(_)));
    }

    #[test]
    fn test_call_subquery_missing_closing_brace() {
        let err = parse_cypher("CALL { MATCH (n:Person) RETURN n").unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains('}') || msg.to_lowercase().contains("close"),
            "expected a missing-brace error, got: {msg}"
        );
    }

    #[test]
    fn test_call_subquery_importing_with_alias_rejected() {
        // `WITH p AS x` in the importing position is illegal.
        let err = parse_cypher(
            "MATCH (p:Person) CALL { WITH p AS x MATCH (x) RETURN count(x) AS c } RETURN c",
        )
        .unwrap_err();
        let msg = format!("{}", err).to_lowercase();
        assert!(
            msg.contains("importing with"),
            "expected importing-WITH violation, got: {msg}"
        );
    }

    #[test]
    fn test_call_subquery_importing_with_projection_rejected() {
        // `WITH p.name` (a projection) in the importing position is illegal.
        let err = parse_cypher("MATCH (p:Person) CALL { WITH p.name MATCH (n) RETURN n } RETURN n")
            .unwrap_err();
        let msg = format!("{}", err).to_lowercase();
        assert!(
            msg.contains("importing with"),
            "expected importing-WITH violation, got: {msg}"
        );
    }

    #[test]
    fn test_call_procedure_still_parses() {
        // The existing CALL procedure form must be unaffected by the
        // subquery branch.
        let query = parse_cypher("CALL pagerank() YIELD node, score RETURN node, score").unwrap();
        assert!(matches!(&query.clauses[0], Clause::Call(_)));
        if let Clause::Call(c) = &query.clauses[0] {
            assert_eq!(c.procedure_name, "pagerank");
            assert_eq!(c.yield_items.len(), 2);
        }
    }
}
