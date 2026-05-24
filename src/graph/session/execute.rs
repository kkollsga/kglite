//! Cypher pipeline orchestration — single source of truth.
//!
//! Mirrors the canonical pipeline that previously lived inline at
//! `src/graph/pyapi/kg_core.rs::cypher`:
//!
//! ```text
//! parse_cypher → validate_schema → rewrite_text_score (+embed if needed)
//!   → optimize_with_disabled → [mark_lazy_eligibility] → is_mutation_query
//!   → generate_explain_result | execute | execute_mutable
//! ```
//!
//! [`execute_read`] takes `&DirGraph` (auto-commit reads + in-tx reads
//! against working/snapshot). [`execute_mut`] takes `&mut DirGraph`
//! (in-tx writes against `Transaction::working_mut()`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use crate::datatypes::Value;
use crate::error::KgError;
use crate::graph::dir_graph::DirGraph;
use crate::graph::embedder::Embedder;
use crate::graph::languages::cypher;
use crate::graph::languages::cypher::ast::{CypherQuery, OutputFormat};
use crate::graph::languages::cypher::result::CypherResult;

/// Per-query knobs. Borrowed for the duration of one execute call.
/// Default values match the kg_core.rs Python boundary's defaults
/// (lazy_eligible=true, no deadline, no max_rows, no disabled passes,
/// no embedder).
pub struct ExecuteOptions<'a> {
    /// Parameter bindings (`$x` references). Empty map = no params.
    pub params: &'a HashMap<String, Value>,
    /// Optional execution deadline. Past this, the executor returns
    /// `CypherTimeout`. None = no deadline.
    pub deadline: Option<Instant>,
    /// Optional row cap. None = no cap.
    pub max_rows: Option<usize>,
    /// Lazy-projection mode.
    ///
    /// - `true` (Python default): call `mark_lazy_eligibility` after
    ///   optimize + pass `streaming=true` to the executor. The
    ///   `CypherResult.lazy` field may be `Some(LazyResultDescriptor)`;
    ///   callers that want eager rows must materialize via the lazy
    ///   helper in `src/graph/pyapi/result_view.rs`.
    /// - `false` (bolt-server, mcp-server): skip
    ///   `mark_lazy_eligibility` + pass `streaming=false`. The
    ///   executor materializes every row into `CypherResult.rows`.
    ///
    /// **Important:** setting `lazy_eligible=true` without having a
    /// lazy-materializer to consume `result.lazy` results in
    /// silently empty row sets — exactly the bolt-server bug fixed
    /// during the robustness pass. Default to `false` for safety;
    /// the Python boundary flips it to `true` to benefit from the
    /// lazy path in interactive use.
    pub lazy_eligible: bool,
    /// Optional set of planner passes to disable. None means "use
    /// the static empty set" (no allocation; the common case).
    pub disabled_passes: Option<&'a HashSet<String>>,
    /// Optional embedder for `text_score()` queries. If a query
    /// uses `text_score()` and this is `None`, execute returns
    /// `KgError::Argument("text_score requires embedder ...")`.
    pub embedder: Option<Arc<dyn Embedder>>,
}

impl<'a> ExecuteOptions<'a> {
    /// Conservative defaults: `lazy_eligible: false` (safe for
    /// every consumer that doesn't have a lazy materializer), no
    /// deadline, no max_rows, no disabled passes, no embedder.
    /// Caller is expected to override at least `params`.
    pub fn new(params: &'a HashMap<String, Value>) -> Self {
        Self {
            params,
            deadline: None,
            max_rows: None,
            lazy_eligible: false,
            disabled_passes: None,
            embedder: None,
        }
    }
}

/// Result of a successful execute. Wraps `CypherResult` with the
/// metadata callers need for output serialization (CSV, DataFrame,
/// PackStream record emission).
pub struct ExecuteOutcome {
    pub result: CypherResult,
    /// `true` when the query was a CREATE/SET/DELETE/REMOVE/MERGE.
    /// Read-only callers can pre-reject by checking this on a
    /// dry-run; in practice `execute_read` rejects mutations
    /// upfront via `KgError::Argument`.
    pub is_mutation: bool,
    /// Set when the user passes `RETURN ... FORMAT CSV` (kglite
    /// extension); pyapi + mcp-server format the result accordingly.
    pub output_format: OutputFormat,
    /// Set when the user prefixed the query with `EXPLAIN`. The
    /// `result` contains the rendered plan rows rather than real
    /// data; callers may want to format / display differently.
    pub explain: bool,
}

/// Read-only execution. Errors if the query mutates.
///
/// Caller responsibilities:
/// - Provide a `&DirGraph` (snapshot for auto-commit, or
///   `tx.current()` for in-tx reads).
/// - Decode params (`Bolt`/`Py` → `Value`) before calling.
/// - Map the returned `KgError` to the binding's error type
///   (PyErr subclass via `From`, `BoltError` via the
///   `kg_to_bolt`/`string_to_bolt` helpers in bolt-server).
pub fn execute_read(
    graph: &DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
) -> Result<ExecuteOutcome, KgError> {
    let parsed = prepare(graph, query, opts)?;
    let is_mutation = cypher::is_mutation_query(&parsed);

    // EXPLAIN: render plan rows, skip execution.
    if parsed.explain {
        let result = cypher::generate_explain_result(&parsed, graph);
        return Ok(ExecuteOutcome {
            result,
            is_mutation,
            output_format: parsed.output_format,
            explain: true,
        });
    }

    if is_mutation {
        return Err(KgError::Argument(
            "execute_read called with a mutation query (CREATE/SET/DELETE/REMOVE/MERGE) \
             — use execute_mut against a mutable graph view"
                .to_string(),
        ));
    }

    let result = cypher::CypherExecutor::with_params(graph, opts.params, opts.deadline)
        .with_max_rows(opts.max_rows)
        .with_streaming(opts.lazy_eligible)
        .execute(&parsed)
        .map_err(|message| KgError::CypherExecution {
            message,
            position: None,
        })?;

    Ok(ExecuteOutcome {
        result,
        is_mutation: false,
        output_format: parsed.output_format,
        explain: false,
    })
}

/// Mutating execution. Caller passes `&mut DirGraph` (typically
/// from `Transaction::working_mut()`). For pure reads, use
/// [`execute_read`] instead.
///
/// Note: a read query passed to `execute_mut` runs against the
/// mutable graph view as a read. The function returns
/// `is_mutation: false` in that case so the caller knows nothing
/// was changed.
pub fn execute_mut(
    graph: &mut DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
) -> Result<ExecuteOutcome, KgError> {
    let parsed = prepare(graph, query, opts)?;
    let is_mutation = cypher::is_mutation_query(&parsed);

    if parsed.explain {
        let result = cypher::generate_explain_result(&parsed, graph);
        return Ok(ExecuteOutcome {
            result,
            is_mutation,
            output_format: parsed.output_format,
            explain: true,
        });
    }

    let result = if is_mutation {
        cypher::execute_mutable(graph, &parsed, opts.params.clone(), opts.deadline).map_err(
            |message| KgError::CypherExecution {
                message,
                position: None,
            },
        )?
    } else {
        cypher::CypherExecutor::with_params(graph, opts.params, opts.deadline)
            .with_max_rows(opts.max_rows)
            .with_streaming(opts.lazy_eligible)
            .execute(&parsed)
            .map_err(|message| KgError::CypherExecution {
                message,
                position: None,
            })?
    };

    Ok(ExecuteOutcome {
        result,
        is_mutation,
        output_format: parsed.output_format,
        explain: false,
    })
}

/// Shared preparation: parse → validate → rewrite_text_score →
/// optimize → optional mark_lazy. Returns the parsed+optimized AST.
fn prepare(
    graph: &DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
) -> Result<CypherQuery, KgError> {
    let mut parsed = cypher::parse_cypher(query)?;

    // Schema validation — property typos in pattern literals
    // (`{ttle: 'Alice'}`) get caught with a "did you mean?" hint.
    cypher::validate_schema(&parsed, graph).map_err(KgError::from)?;

    // text_score() rewrite + embed. If the query uses text_score()
    // we need an embedder; reject otherwise.
    let rewrite = cypher::rewrite_text_score(&mut parsed, opts.params).map_err(|message| {
        KgError::CypherExecution {
            message,
            position: None,
        }
    })?;
    if !rewrite.texts_to_embed.is_empty() && !parsed.explain {
        // text_score() requires the binding to embed the texts +
        // inject the resulting vectors into the params map BEFORE
        // calling session::execute_*. Phase E1 doesn't propagate
        // a mutable params map; pyapi will keep its own embed loop
        // wrapping the call (Python needs py.detach around it
        // anyway). bolt-server + mcp-server don't wire text_score
        // today; they get a clean error pointing at the limitation.
        //
        // TODO(future): take params by `&mut HashMap`, accept an
        // optional embedder + inline the embed loop here so all
        // bindings share it.
        if opts.embedder.is_none() {
            return Err(KgError::Argument(
                "text_score() requires a registered embedder; \
                 pass one via ExecuteOptions::embedder or call the \
                 binding's higher-level cypher() entry point that \
                 inlines the embed loop"
                    .to_string(),
            ));
        }
        return Err(KgError::CypherExecution {
            message: "text_score() embed loop not yet wired through session::execute — \
                      the embed step still lives in the binding (pyapi/kg_core.rs); \
                      call the binding's cypher() entry point until E2 lifts it"
                .to_string(),
            position: None,
        });
    }

    // Optimize. Empty disabled-set is the common case; avoid the
    // HashSet allocation when no passes are disabled.
    let disabled_default = cypher::planner::empty_disabled_set();
    let disabled_ref = opts.disabled_passes.unwrap_or(disabled_default);
    cypher::planner::optimize_with_disabled(&mut parsed, graph, opts.params, disabled_ref);

    // Lazy marking — only when the caller asked for it. Without
    // this call, the executor materializes rows eagerly. With it,
    // `result.lazy` may be Some and `result.rows` empty; the
    // caller must handle materialization (Python's ResultView
    // does; bolt-server doesn't, so it passes `lazy_eligible: false`).
    if opts.lazy_eligible {
        cypher::mark_lazy_eligibility(&mut parsed);
    }

    Ok(parsed)
}
