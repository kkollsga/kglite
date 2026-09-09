use super::super::super::ast::*;
use super::{clause_needs_implicit_row, is_mutation_query};

fn query(clauses: Vec<Clause>) -> CypherQuery {
    CypherQuery {
        clauses,
        explain: false,
        profile: false,
        output_format: OutputFormat::Default,
        optimizer_tags: Vec::new(),
    }
}

fn create_clause() -> Clause {
    Clause::Create(CreateClause {
        patterns: Vec::new(),
    })
}

fn return_clause() -> Clause {
    Clause::Return(ReturnClause {
        items: Vec::new(),
        distinct: false,
        having: None,
        lazy_eligible: false,
        group_limit_hint: None,
    })
}

fn call_subquery(body: CypherQuery) -> Clause {
    Clause::CallSubquery {
        import: CallSubqueryImport::Legacy(Vec::new()),
        body: Box::new(body),
    }
}

#[test]
fn plain_read_is_not_a_mutation() {
    assert!(!is_mutation_query(&query(vec![return_clause()])));
}

#[test]
fn top_level_write_is_a_mutation() {
    assert!(is_mutation_query(&query(vec![create_clause()])));
}

#[test]
fn leading_call_subquery_gets_the_implicit_start_row() {
    let call = call_subquery(query(vec![return_clause()]));
    assert!(clause_needs_implicit_row(&call));
}

#[test]
fn write_inside_call_subquery_body_is_a_mutation() {
    let call = call_subquery(query(vec![create_clause(), return_clause()]));
    assert!(is_mutation_query(&query(vec![call, return_clause()])));
}

#[test]
fn nested_write_inside_call_subquery_body_is_a_mutation() {
    let inner_call = call_subquery(query(vec![create_clause(), return_clause()]));
    let outer_call = call_subquery(query(vec![inner_call, return_clause()]));
    assert!(is_mutation_query(&query(vec![outer_call])));
}

#[test]
fn read_only_call_subquery_body_is_not_a_mutation() {
    let call = call_subquery(query(vec![return_clause()]));
    assert!(!is_mutation_query(&query(vec![call, return_clause()])));
}

#[test]
fn write_inside_union_arm_is_a_mutation() {
    let arm = Box::new(query(vec![create_clause(), return_clause()]));
    let union = Clause::Union(UnionClause {
        all: false,
        query: arm,
        kind: SetOpKind::Union,
    });
    assert!(is_mutation_query(&query(vec![return_clause(), union])));
}
