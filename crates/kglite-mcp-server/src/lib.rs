//! `kglite-mcp-server` — single MCP server for KGLite knowledge graphs.
//!
//! Layers KGLite graph/query/source tools on top of the generic `mcp-server`
//! framework. The graph tools close over a [`GraphState`] holding the active
//! pure-Rust graph.
//!
//! ## Two frontends, one library
//!
//! The server body lives in [`run`] (sync, builds its own tokio
//! runtime) so it can be driven from two places without duplication:
//!
//! - the thin `src/main.rs` binary (`cargo install kglite-mcp-server`),
//!   which is libpython-free; and
//! - the `kglite` Python wheel, whose PyO3 wrapper calls [`run`]
//!   inside `py.detach(...)` (GIL released) so `pip install kglite` ships
//!   the exact same server with no separate wheel and no duplicated engine.
//!
//! The library never links libpython — it depends only on the pure-Rust
//! `kglite` core. The wheel's `.so` and the standalone binary share this
//! one engine build.
//!
//! Modes:
//! - `--graph X.kgl` — load a pre-built graph file at boot.
//! - `--workspace DIR` — multi-repo source workspace. Graph construction
//!   requires injected [`WorkspaceGraphHooks`] (for example from `codingest-mcp`).
//! - `--watch DIR` — file-watcher mode. With an injected producer, changes
//!   rebuild and atomically swap the active graph.
//! - `--source-root DIR` — generic file-tree mode (no graph).
//! - bare — framework + manifest tools only.

use anyhow::{Context, Result};
use clap::Parser;

mod activation;
mod boot;
mod bundled_overrides;
mod cli;
mod code_source;
mod csv_http;
mod cypher_tools;
mod embedder;
mod explore;
mod extensions;
mod modes;
mod recipe_queries;
mod selftest;
mod server_run;
mod skills;
mod tools;
mod value_codecs;
mod watcher;

pub(crate) use activation::*;
pub(crate) use boot::*;
pub(crate) use bundled_overrides::*;
pub(crate) use cli::*;
pub(crate) use embedder::*;
pub(crate) use extensions::*;
pub(crate) use modes::*;
pub(crate) use server_run::*;
pub(crate) use skills::*;
pub(crate) use watcher::*;

pub use crate::embedder::PyEmbedderFactory;
pub use crate::extensions::{
    DomainGraphContext, DomainGraphState, DomainToolRegistrar, DomainToolRegistry, ServerExtensions,
};
pub use crate::tools::{
    WorkspaceGraphBuildFn, WorkspaceGraphChanges, WorkspaceGraphHooks, WorkspaceGraphMode,
    WorkspaceGraphRelevance, WorkspaceGraphRelevanceFn, WorkspaceGraphRequest,
    WorkspaceGraphResult,
};

/// Run the MCP server to completion over stdio.
///
/// `args` is a full argv vector (program name in `args[0]`, as clap
/// expects). The binary passes `std::env::args()`; the PyO3 wrapper
/// passes a synthesised `["kglite-mcp-server", ...sys.argv[1:]]`.
///
/// Synchronous by design (see CLAUDE.md "core is sync, bindings own
/// async"): this builds its own multi-thread tokio runtime and blocks
/// on the stdio serve loop. The PyO3 wrapper calls it inside
/// `py.detach(...)`, so the GIL is released for the server's entire
/// lifetime and the Python process simply *becomes* the server.
pub fn run<I, T>(args: I) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    run_with_options(args, None, ServerExtensions::default())
}

/// Like [`run`], but with an optional Python-embedder factory for the
/// `extensions.embedder.library` Python path (`sentence-transformers` /
/// `fastembed`). The standalone binary calls [`run`] (no factory); the
/// kglite wheel passes a factory that builds the named Python embedder.
pub fn run_with_embedder_factory<I, T>(args: I, factory: Option<PyEmbedderFactory>) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    run_with_options(args, factory, ServerExtensions::default())
}

/// Run the server with downstream boot-time extension points.
///
/// Domain tools are registered after KGLite's built-ins and manifest Cypher
/// tools, but before skill finalisation. This lets `tool_registered:` skill
/// predicates see downstream tools while keeping the generic graph/Cypher
/// routes owned and protected by KGLite.
pub fn run_with_extensions<I, T>(args: I, extensions: ServerExtensions) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    run_with_options(args, None, extensions)
}

fn run_with_options<I, T>(
    args: I,
    factory: Option<PyEmbedderFactory>,
    extensions: ServerExtensions,
) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let argv: Vec<std::ffi::OsString> = args.into_iter().map(Into::into).collect();
    let cli = Cli::parse_from(argv.iter().cloned());
    // `--selftest` is a diagnostic mode, not a serve mode: it re-spawns this
    // binary with the operator's other flags and drives a real handshake. It
    // runs before the tokio runtime is built (the child owns the async serve
    // loop; the parent's RPC client is plain blocking I/O).
    if cli.selftest {
        return selftest::run_selftest(&cli, &argv);
    }
    // Tool handlers are synchronous closures that run the Cypher pipeline
    // inline on a worker thread (see `tools.rs`), and that pipeline recurses
    // per level of expression nesting. Tokio's default 2 MiB worker stack is
    // too small for the deepest query the parser accepts, and a Rust stack
    // overflow aborts the process rather than failing the one request — so
    // give workers the same headroom the CLI's main thread has.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(kglite::api::session::QUERY_THREAD_STACK_SIZE)
        .build()
        .context("failed to build tokio runtime")?;
    runtime.block_on(run_async(cli, factory, extensions))
}
