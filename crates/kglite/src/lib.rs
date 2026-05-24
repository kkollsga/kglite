//! kglite-core — pure-Rust core of the kglite knowledge graph engine.
//!
//! ## Phase G crate split
//!
//! Currently named `kglite-core` to avoid a workspace name conflict
//! with the existing root crate `kglite`. G.4 will rename this to
//! `kglite` once the root crate is relocated to `crates/kglite-py/`.
//! End-state matches the polars convention: `crates/kglite/` is the
//! pure-Rust core publishable on crates.io; `crates/kglite-py/` is
//! the PyO3 wrapper that maturin builds into the wheel.
//!
//! ## Public API
//!
//! Downstream Rust consumers (kglite-py for the Python wheel,
//! kglite-bolt-server, kglite-mcp-server, future Go/TypeScript/JVM
//! bindings) should depend on items in the curated [`api`] module —
//! those are the items that get semver guarantees. Anything else is
//! an implementation detail.
//!
//! See `docs/explanation/embedding-kglite.md` (G.5) for the embedder
//! guide.

// Phase A.2 / C2 — crate-wide allow for clippy::result_large_err.
// `KgError` is intentionally rich (16 variants spanning Cypher /
// schema / IO / argument validation) so its size pushes past clippy's
// default 128-byte threshold. Boxing the error variant in every
// `Result<T, KgError>` would add an allocation per error path for no
// real benefit — error paths aren't hot. Standard pattern for crates
// with a unified typed error.
#![allow(clippy::result_large_err)]
// Phase G.3a — when the engine moved from a cdylib (root crate) into
// an rlib (this crate), clippy's lint set widened because more items
// became reachable across crate boundaries. The noisy lints below
// were tolerated in the root crate's pub(crate) surface; they're
// follow-up cleanup tasks but not blockers. Two of them
// (hidden_glob_reexports, private_interfaces) directly reflect the
// rushed wide-public visibility bumps in G.3a — proper accessor
// methods would resolve them.
#![allow(clippy::new_without_default)]
#![allow(clippy::len_without_is_empty)]
#![allow(private_interfaces)]
#![allow(hidden_glob_reexports)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::result_unit_err)]

pub mod code_tree;
pub mod datasets;
pub mod datatypes;
pub mod error;
pub mod graph;

/// Curated stable Rust API. Downstream consumers should depend on
/// items here, not on the underlying module structure (which may
/// move between minor releases).
pub mod api {
    pub use crate::code_tree::builder::run_with_options as build_code_tree;
    pub use crate::datatypes::values::{NodeValue, PathValue, RelValue};
    pub use crate::datatypes::Value;
    pub use crate::error::{KgError, KgErrorCode};
    pub use crate::graph::dir_graph::DirGraph;
    #[cfg(feature = "fastembed")]
    pub use crate::graph::embedder::fastembed::FastEmbedAdapter;
    pub use crate::graph::embedder::Embedder;
    pub use crate::graph::explore::{explore_markdown, ExploreOptions};
    pub use crate::graph::introspection::describe::compute_description;
    pub use crate::graph::introspection::schema_overview::compute_schema;
    pub use crate::graph::introspection::SchemaOverview;
    pub use crate::graph::introspection::{ConnectionDetail, CypherDetail, FluentDetail};
    pub use crate::graph::io::file::{load_file, save_graph};
    pub use crate::graph::{SourceLocation, SourceLookup};

    /// Cypher parser + planner + executor primitives. Downstream
    /// consumers can build their own custom Cypher pipelines using
    /// these items; for the canonical pipeline see [`session`].
    pub mod cypher {
        pub use crate::graph::languages::cypher::ast::CypherQuery;
        pub use crate::graph::languages::cypher::ast::OutputFormat;
        pub use crate::graph::languages::cypher::executor::write::execute_mutable;
        pub use crate::graph::languages::cypher::executor::CypherExecutor;
        pub use crate::graph::languages::cypher::generate_explain_result;
        pub use crate::graph::languages::cypher::is_mutation_query;
        pub use crate::graph::languages::cypher::parser::parse_cypher;
        pub use crate::graph::languages::cypher::planner;
        pub use crate::graph::languages::cypher::planner::mark_lazy_eligibility;
        pub use crate::graph::languages::cypher::planner::schema_check::validate_schema;
        pub use crate::graph::languages::cypher::planner::simplification::rewrite_text_score;
        pub use crate::graph::languages::cypher::result::CypherResult;
    }

    /// Canonical query + transaction surface — single source of
    /// truth for the Cypher pipeline + snapshot/working CoW
    /// transaction model. See `docs/explanation/session.md`.
    pub mod session {
        pub use crate::graph::session::{
            execute_mut, execute_read, CommitOutcome, ExecuteOptions, ExecuteOutcome, Session,
            Transaction,
        };
    }
}
