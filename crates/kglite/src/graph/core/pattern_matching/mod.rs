//! Pattern matching — parse and match Cypher-style patterns against a DirGraph.
//!
//! Split into three submodules:
//! - [`pattern`] — AST types (Pattern, NodePattern, EdgePattern, PropertyMatcher, etc.)
//! - [`parser`] — tokenizer + Parser that turns pattern strings into the AST
//! - [`matcher`] — PatternExecutor state machine that runs matches against a graph
//!
//! The PropertyMatcher enum is defined in `pattern` with its cases, its
//! construction in the Parser, and its evaluation in `matcher::PatternExecutor`.
//! The three files deliberately mirror the lifecycle of a pattern: parsed,
//! typed, then matched.

pub mod matcher;
pub mod parser;
pub mod pattern;

pub use matcher::PatternExecutor;
pub use parser::parse_pattern;
pub use pattern::{
    EdgeDirection, MatchBinding, NodePattern, Pattern, PatternElement, PatternMatch,
    PropertyMatcher,
};
