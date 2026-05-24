//! Query-interface languages.
//!
//! Each language under `languages/` produces results by calling
//! into the shared primitives in `crate::graph::core`. Languages
//! are peers — a new language (SPARQL, GraphQL, custom DSL) slots
//! in alongside `cypher` and `fluent` without disturbing either.

pub mod cypher;
pub mod fluent;
