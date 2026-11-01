//! Pattern matching — parse and match Cypher-style patterns against a DirGraph.
//!
//! Split into submodules:
//! - [`pattern`] — AST types (Pattern, NodePattern, EdgePattern, PropertyMatcher, etc.)
//! - [`parser`] — tokenizer + Parser that turns pattern strings into the AST
//! - [`matcher`] — PatternExecutor state machine that runs matches against a graph
//! - [`closure_probe`] — when an ontology closure can be answered by member
//!   index lookups; shared with the Cypher EXPLAIN renderer
//!
//! The PropertyMatcher enum is defined in `pattern` with its cases, its
//! construction in the Parser, and its evaluation in `matcher::PatternExecutor`.
//! The three files deliberately mirror the lifecycle of a pattern: parsed,
//! typed, then matched.

pub(crate) mod closure_probe;
pub(crate) mod column_filter;
// Unwired spike: only its tests and the release probe call it so far.
#[allow(dead_code)]
mod column_kernel;
pub mod matcher;
mod numeric_cell_test;
pub mod parser;
pub mod pattern;

pub use matcher::PatternExecutor;
pub use parser::parse_pattern;
pub use pattern::{
    anon_vlpath_name, EdgeDirection, EdgePattern, MatchBinding, NodePattern, ParamLabel, PathHop,
    Pattern, PatternElement, PatternMatch, PropertyMatcher, ANON_VLPATH_PREFIX,
};
