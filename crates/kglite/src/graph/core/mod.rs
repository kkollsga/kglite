//! Shared query-execution engine — used by Cypher AND the fluent API.
//!
//! Pattern matching, filtering, traversal, property retrieval,
//! calculations, and statistics live here. The cypher/ module calls
//! into query/ for its MATCH implementation; the fluent API uses it
//! via filtering/traversal helpers.

pub mod calculations;
pub mod data_retrieval;
pub mod filtering;
pub mod iterators;
pub mod pattern_matching;
pub mod statistics;
pub mod traversal;
pub mod value_operations;
