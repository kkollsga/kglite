//! Post-pass structural invariants, checked after every optimizer pass in
//! debug builds. Extracted from `planner/mod.rs` (which is the PASSES table
//! and the pass wrappers) — a failed invariant names the pass that broke it,
//! which is what turns a wrong-answer bug into a one-line diagnosis.

#[cfg(debug_assertions)]
use super::*;

/// Sanity checks on the post-pass IR. Debug-only — release builds pay
/// nothing. Catches the class of bug where pass X corrupts the IR and a
/// downstream pass or the executor crashes 200 lines later with a
/// confusing error. Each check is permissive (only catches definitely-
/// invalid shapes); we'd rather miss a subtle bug than panic on a valid
/// query the writer of an invariant didn't anticipate.
#[cfg(debug_assertions)]
pub(super) fn debug_check_invariants(query: &CypherQuery, after_pass_name: &str) {
    if let Err(msg) = check_match_patterns_non_empty(query) {
        panic!("Pass `{after_pass_name}` produced invalid IR: {msg}");
    }
    if let Err(msg) = check_return_with_items_non_empty(query) {
        panic!("Pass `{after_pass_name}` produced invalid IR: {msg}");
    }
    if let Err(msg) = check_limit_skip_nonnegative(query) {
        panic!("Pass `{after_pass_name}` produced invalid IR: {msg}");
    }
}

/// Every Match / OptionalMatch must have at least one pattern, and each
/// pattern at least one element. Catches passes that delete the last
/// pattern but leave the clause shell.
#[cfg(debug_assertions)]
fn check_match_patterns_non_empty(query: &CypherQuery) -> Result<(), String> {
    for (idx, clause) in query.clauses.iter().enumerate() {
        let mc = match clause {
            Clause::Match(m) | Clause::OptionalMatch(m) => m,
            _ => continue,
        };
        if mc.patterns.is_empty() {
            return Err(format!("Match clause at index {idx} has no patterns"));
        }
        for (pi, p) in mc.patterns.iter().enumerate() {
            if p.elements.is_empty() {
                return Err(format!(
                    "Match clause at index {idx}, pattern {pi} has no elements"
                ));
            }
        }
    }
    Ok(())
}

/// Return / With must project at least one item. Catches passes that
/// leave a stub Return after consuming its only item into a fused clause.
#[cfg(debug_assertions)]
fn check_return_with_items_non_empty(query: &CypherQuery) -> Result<(), String> {
    for (idx, clause) in query.clauses.iter().enumerate() {
        match clause {
            Clause::Return(r) if r.items.is_empty() => {
                return Err(format!("Return clause at index {idx} has no items"));
            }
            Clause::With(w) if w.items.is_empty() => {
                return Err(format!("With clause at index {idx} has no items"));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Literal LIMIT / SKIP values must be non-negative. Catches passes
/// that synthesize a literal hint (e.g. fusion top-K) and forget to
/// clamp at zero. Non-literal values (parameters, expressions) are left
/// alone — the executor handles those at runtime.
#[cfg(debug_assertions)]
fn check_limit_skip_nonnegative(query: &CypherQuery) -> Result<(), String> {
    for (idx, clause) in query.clauses.iter().enumerate() {
        match clause {
            Clause::Limit(l) => {
                if let Expression::Literal(Value::Int64(n)) = &l.count {
                    if *n < 0 {
                        return Err(format!(
                            "Limit clause at index {idx} has negative literal {n}"
                        ));
                    }
                }
            }
            Clause::Skip(s) => {
                if let Expression::Literal(Value::Int64(n)) = &s.count {
                    if *n < 0 {
                        return Err(format!(
                            "Skip clause at index {idx} has negative literal {n}"
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

// Note: a `check_terminal_return_position` invariant was prototyped here
// and removed — the parser legitimately produces `RETURN ... WHERE ...`
// for queries where the WHERE syntactically trails the RETURN (test:
// test_edge_properties.py). Without a clear oracle for "what's a valid
// post-RETURN clause", a position check creates false positives. The
// non-empty-patterns and non-empty-items checks above stay because they
// have unambiguous oracles.
