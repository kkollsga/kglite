use super::*;
use std::collections::HashMap;

#[test]
fn compiled_membership_preserves_nested_unknown_under_not() {
    let graph = DirGraph::new();
    let params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &params, None);
    let row = ResultRow::new();
    for count in [8, 9] {
        for (probe, expected) in [
            (Value::List(vec![Value::Null]), None),
            (Value::List(vec![Value::Int64(1)]), Some(true)),
            (Value::List(vec![]), Some(false)),
        ] {
            let predicate = Predicate::InLiteralSet {
                expr: Expression::Literal(probe),
                values: MembershipSet::new(
                    (1..=count)
                        .map(|i| Value::List(vec![Value::Int64(i)]))
                        .collect(),
                ),
            };
            let mut compiler = ScanCompiler::new(&executor, "n");
            let compiled = compiler.pred(&predicate);
            // A Generic fallback would exercise the interpreter and miss this arm.
            assert!(matches!(&compiled, ScanPred::InLiteralSet { .. }));
            let runtime = compiler.finish();
            assert_eq!(
                compiled.eval(&executor, &runtime, None, &row).unwrap(),
                expected
            );
            let negated = ScanPred::Not(Box::new(compiled));
            assert_eq!(
                negated.eval(&executor, &runtime, None, &row).unwrap(),
                expected.map(|v| !v)
            );
        }
    }
}
