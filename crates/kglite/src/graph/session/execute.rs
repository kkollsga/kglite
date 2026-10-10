//! Cypher pipeline orchestration — single source of truth.
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

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use super::CancelToken;
use crate::datatypes::Value;
use crate::error::KgError;
use crate::graph::dir_graph::rollback::StatementCheckpoint;
use crate::graph::dir_graph::DirGraph;
use crate::graph::edge_embedding_generation::EmbeddingExecutionService;
use crate::graph::embedder::Embedder;
use crate::graph::languages::cypher;
use crate::graph::languages::cypher::ast::{
    Clause, CreateElement, CreatePattern, CypherQuery, OutputFormat, RemoveItem, SetItem,
};
use crate::graph::languages::cypher::executor::load_csv::CsvImportPolicy;
use crate::graph::languages::cypher::result::{
    CypherResult, QueryDiagnostics, TemporalDiagnostics,
};
use crate::graph::languages::cypher::value_codec::ValueCodec;

/// Per-query knobs. Borrowed for the duration of one execute call.
pub struct ExecuteOptions<'a> {
    /// Parameter bindings (`$x` references).
    pub params: &'a HashMap<String, Value>,
    /// Past this, the executor returns `CypherTimeout`.
    pub deadline: Option<Instant>,
    /// The instant the caller's timeout was resolved to [`Self::deadline`].
    /// A `CypherTimeout` measures both its `elapsed_ms` and its `limit_ms`
    /// from here, so it reports the configured limit even when the binding
    /// spent part of the budget before execution began (converting a large
    /// parameter, forking a transaction's working copy). `None` measures from
    /// the start of execution. [`Self::set_timeout_ms`] sets both fields.
    pub deadline_origin: Option<Instant>,
    /// Work budget for one query, **not** a result-row cap.
    ///
    /// It is charged against intermediate rows, retained collection items, and
    /// scan work units — every quantity the executor holds or walks on the way
    /// to an answer, which is why the count can far exceed the rows the caller
    /// eventually receives. Exceeding it *fails the query*:
    ///
    /// ```text
    /// Query produced 3 rows while executing MATCH, exceeding the
    /// max_work_units budget of 2. Add a LIMIT clause or raise max_work_units.
    /// ```
    ///
    /// A query is never truncated to this number — a caller who wants N rows
    /// back writes `LIMIT N`. `None` (the default on every surface) leaves the
    /// engine's 10,000,000-unit `MAX_UNBOUNDED_ROWS` backstop in charge of
    /// materialized quantities; `Some(n)` replaces that backstop with `n` and
    /// additionally charges scan work against it.
    pub max_work_units: Option<usize>,
    /// Cap on the result rows this execution **retains** — the deliberate
    /// opposite number to [`Self::max_work_units`], not a rename of it:
    ///
    /// | | bounds | on overrun |
    /// |---|---|---|
    /// | `max_work_units` | work: intermediate rows, retained items, scans | **errors** |
    /// | `row_limit` | rows handed back to the caller | **truncates, with a signal** |
    ///
    /// The query still runs to completion and still computes every row —
    /// ORDER BY sorts the whole set, aggregation folds the whole set — and
    /// only the *retention* of the finished rows stops at the cap. So the
    /// rows kept are the first `n` of the answer the caller would have got
    /// uncapped, and with an ORDER BY they are its genuine top-`n`. An
    /// explicit `LIMIT m` in the query is applied first, so the effective cap
    /// is `min(m, n)`.
    ///
    /// Truncation is **never silent**. `QueryDiagnostics` carries the cap in
    /// `row_limit`, the exact pre-truncation total in `total_rows` (populated
    /// only when rows were actually dropped, and exact on every execution
    /// path), and a warning in `warnings` that reaches the ordinary
    /// query-warning channel — so `"showing 5,000 of 412,003"` is answerable
    /// from the result alone.
    ///
    /// Applies to a mutation's trailing `RETURN` exactly as it does to a read:
    /// the writes still all happen and `MutationStats` still counts them all,
    /// because this caps what is *reported*, never what is *changed*.
    /// `EXPLAIN` is exempt — a rendered plan is not result data.
    ///
    /// `Some(0)` is legal and means "retain nothing, still tell me the total".
    /// `None` (the default on every surface) retains everything.
    ///
    /// One wart worth knowing: a query with **no** `RETURN` clause infers its
    /// columns from the rows it kept, so capping such a query to 0 yields no
    /// columns either.
    pub row_limit: Option<usize>,
    /// Lazy-projection mode.
    ///
    /// - `true` (the Python live graph): `mark_lazy_eligibility` runs after
    ///   optimize, so `CypherResult.lazy` may be `Some(LazyResultDescriptor)`
    ///   and `rows` empty; the caller must materialize via the lazy helper in
    ///   `pyapi/result_view.rs`, which pins the graph the rows are read from.
    /// - `false` (every other caller): the executor materializes every row
    ///   into `CypherResult.rows`.
    ///
    /// **Important:** setting `true` without a lazy-materializer to consume
    /// `result.lazy` yields silently empty row sets — the bolt-server bug
    /// fixed during the robustness pass. Default `false` for safety.
    pub lazy_eligible: bool,
    /// Run a trailing aggregate (`WITH`/`RETURN` with an aggregate, optionally
    /// `ORDER BY … LIMIT`) through the streaming pipeline, which folds rows
    /// into aggregate state as they are matched instead of materializing
    /// them first. Its rows are ordinary materialized rows, so it needs no
    /// lazy materializer and is independent of [`Self::lazy_eligible`].
    /// `false` in [`Self::eager`].
    pub streaming: bool,
    /// Planner passes to disable. `None` uses the static empty set — no
    /// allocation, and the common case.
    pub disabled_passes: Option<&'a HashSet<String>>,
    /// Embedder for `text_score()` queries. A `text_score()` query with `None`
    /// here fails with `KgError::CypherExecution`.
    pub embedder: Option<Arc<dyn Embedder>>,
    /// Optional operator-declared value codecs. When set, query-side
    /// literals bound to a codec'd property are decoded before
    /// validation/optimization (`'Q42'` → `42`), and result columns
    /// that are direct projections of a codec'd property are encoded
    /// back (`42` → `'Q42'`). `None`/empty = no transform (the common
    /// case; zero hot-path cost). See `cypher::value_codec`.
    pub value_codecs: Option<&'a [ValueCodec]>,
    /// Cooperative-cancellation handle. The executor and pattern matcher poll
    /// it at the same checkpoints they poll `deadline` (one relaxed atomic
    /// load per ~4K comparisons); once [`CancelToken::cancel`] is called from
    /// any thread, the run aborts with [`KgError::Cancelled`]. `None` = never
    /// cancelled, zero hot-path cost. A token already cancelled before the
    /// call fails it before any work. Keep a clone alive until the call
    /// returns (see [`CancelToken`]).
    pub cancel: Option<CancelToken>,
    /// Optional role-scoped write whitelist (integrity, not secrecy — e.g. a
    /// coding role may write `Plan`/`Task` but not `Algorithm`). `None` =
    /// unrestricted (the default; zero hot-path cost); an empty set denies
    /// every mutation. Only meaningful on the mutation path (`execute_mut`).
    ///
    /// When `Some`, a **node** write — `CREATE`, `MERGE`'s create arm, `SET`
    /// (property, map or label), `REMOVE` (property or label), `DELETE`,
    /// `DETACH DELETE`, node-type index/constraint DDL — is judged by the
    /// node's *stored* type, so a pattern label cannot widen the scope. A
    /// **relationship** write (edge `CREATE`, `DELETE r`, `SET r.p`,
    /// `REMOVE r.p`) is allowed iff at least one endpoint's stored type is in
    /// the set; `DETACH DELETE`'s incident-edge collateral is authorized by
    /// the node delete and not re-checked per far endpoint. Relationship
    /// *constraint* DDL and `db.cdc.enable`/`db.cdc.disable` are outside the
    /// perimeter, as are the bulk loaders (this is a per-execution concept).
    /// The enforcement sites are the `enforce_*_write_scope` family in
    /// `languages::cypher::executor::write_scope`.
    pub write_scope: Option<&'a HashSet<String>>,
    /// Caller-supplied freshness provenance, stamped alongside `updated_at` on
    /// writes to `auto_timestamp` types: the git SHA the writer is working
    /// against and an actor id. `None` = not supplied. Mutation path only.
    pub git_sha: Option<&'a str>,
    pub modified_by: Option<&'a str>,
    /// Whether this execution may read local files through `LOAD CSV`, and
    /// from where. Defaults to [`CsvImportPolicy::Denied`], deliberately: `file://`
    /// means the server's filesystem, so a binding that never considered
    /// `LOAD CSV` must not hand its callers a file-read primitive by omission.
    /// In-process bindings (the Python wheel, the CLI) grant
    /// [`CsvImportPolicy::LocalFilesystem`] because their caller already has
    /// the host process's access; the Bolt server grants a
    /// [`CsvImportPolicy::Directory`] only when started with
    /// `--allow-csv-import <DIR>`; the MCP server grants nothing.
    pub csv_import: CsvImportPolicy,
    /// Opt in to the parallel runtime for this query. Default `false`
    /// everywhere, mirroring Neo4j's `CYPHER runtime=parallel` posture: one
    /// heavy analytical query may use the whole machine, but a server's cores
    /// belong to its concurrent clients, so nothing turns this on by
    /// omission. Only operators that can partition deterministically honour
    /// it, and each still applies its own runtime row × cost-class gate
    /// ([`crate::graph::parallel::should_fan_out`]) — `true` is a permission,
    /// not an instruction.
    ///
    /// The Python wheel exposes it as `kg.cypher(parallel=True)` and the CLI
    /// as `--parallel`, both per query. The MCP server exposes it as an
    /// operator decision instead — `--parallel` / `extensions.parallel`,
    /// applied to its read seam only — because the agent on the other end
    /// cannot know how many cores the deployment is willing to spend. The
    /// Bolt server sets it nowhere: it multiplexes concurrent sessions over
    /// one process, so the cores are already committed.
    pub parallel: bool,
}

impl<'a> ExecuteOptions<'a> {
    /// Synonym for [`Self::eager`], kept for Rust-convention API discovery;
    /// `eager` is the intent-named factory call-sites prefer.
    pub fn new(params: &'a HashMap<String, Value>) -> Self {
        Self::eager(params)
    }

    /// Eager-execution defaults — the constructor for any binding without a
    /// lazy result materializer (`lazy_eligible: false`). Override individual
    /// fields after construction.
    pub fn eager(params: &'a HashMap<String, Value>) -> Self {
        Self {
            params,
            deadline: None,
            deadline_origin: None,
            max_work_units: None,
            row_limit: None,
            lazy_eligible: false,
            streaming: false,
            disabled_passes: None,
            embedder: None,
            value_codecs: None,
            cancel: None,
            write_scope: None,
            git_sha: None,
            modified_by: None,
            csv_import: CsvImportPolicy::Denied,
            parallel: false,
        }
    }

    /// Grant `LOAD CSV` filesystem access for this execution.
    ///
    /// Builder form so a call-site reads as an explicit grant rather than a
    /// field assignment buried among defaults.
    pub fn with_csv_import(mut self, policy: CsvImportPolicy) -> Self {
        self.csv_import = policy;
        self
    }

    /// Give this execution a deadline `timeout_ms` from now — `None` or `0`
    /// means none — measured from now (see [`Self::deadline_origin`]).
    pub fn set_timeout_ms(&mut self, timeout_ms: Option<u64>) {
        let span = super::query_defaults::deadline_span(timeout_ms);
        self.deadline_origin = span.map(|(origin, _)| origin);
        self.deadline = span.map(|(_, deadline)| deadline);
    }

    /// Opt this execution in to the parallel runtime. Builder form for the
    /// same reason [`Self::with_csv_import`] has one.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

#[inline]
fn is_cancelled(opts: &ExecuteOptions<'_>) -> bool {
    opts.cancel.as_ref().is_some_and(CancelToken::is_cancelled)
}

/// Whether [`ExecuteOptions::deadline`] has passed.
///
/// Read only on an error path, so the clock read is free to the happy path.
/// It is the *state* the executor aborted on, re-asked — deliberately not a
/// match on the abort message, which is prose and drifts.
#[inline]
fn deadline_expired(opts: &ExecuteOptions<'_>) -> bool {
    opts.deadline.is_some_and(|dl| Instant::now() > dl)
}

/// `(elapsed_ms, limit_ms)`, both measured from the deadline's origin — the
/// instant the caller resolved it, else `started`.
fn deadline_figures(opts: &ExecuteOptions<'_>, started: Instant) -> (u64, Option<u64>) {
    let origin = opts.deadline_origin.unwrap_or(started);
    (
        origin.elapsed().as_millis() as u64,
        opts.deadline
            .map(|dl| dl.saturating_duration_since(origin).as_millis() as u64),
    )
}

/// Build the typed timeout. Unlike [`attach_diagnostics`], its `elapsed_ms`
/// runs from the deadline's origin, so it and `limit_ms` share one clock.
fn timeout_err(opts: &ExecuteOptions<'_>, started: Instant, message: String) -> KgError {
    let (elapsed_ms, limit_ms) = deadline_figures(opts, started);
    KgError::CypherTimeout {
        elapsed_ms,
        limit_ms: limit_ms.unwrap_or(0),
        message,
    }
}

/// Map an executor error string to a typed [`KgError`], reading *why* the run
/// stopped off the interrupt state rather than off the message:
///
/// 1. the cancel flag is raised → [`KgError::Cancelled`] (the binding maps
///    that to its interrupt type — `KeyboardInterrupt` in the Python wheel);
/// 2. the deadline has passed → [`KgError::CypherTimeout`], which is what
///    every binding's timeout surface (`kglite.CypherTimeoutError`, Bolt's
///    `TransactionTimedOut`, the C `CypherTimeout` status, HTTP 408) has
///    always claimed to carry and, before this, never received;
/// 3. otherwise → `CypherExecution`.
///
/// The deadline arm can only misattribute a failure that happens *after* the
/// deadline passed but before the next poll would have aborted the run
/// anyway — a query already over budget, reported as over budget.
#[inline]
pub(super) fn exec_err(opts: &ExecuteOptions<'_>, started: Instant, message: String) -> KgError {
    if is_cancelled(opts) {
        KgError::Cancelled
    } else if deadline_expired(opts) {
        timeout_err(opts, started, message)
    } else {
        KgError::CypherExecution {
            message,
            position: None,
        }
    }
}

/// [`exec_err`] for the mutation path, which can additionally recover the
/// structured constraint violation parked by
/// [`DirGraph::record_constraint_violation`].
///
/// Cancellation keeps precedence: an interrupt is a user action and stays
/// `KgError::Cancelled` regardless of what the aborted statement had parked.
/// A parked violation outranks the deadline the other way round — the
/// statement produced a real, reportable answer about the data, and a deadline
/// that happened to be past by the time it did must not hide it.
/// The park is drained on every path so nothing survives into a later run.
fn mutation_err(
    graph: &mut DirGraph,
    opts: &ExecuteOptions<'_>,
    started: Instant,
    message: String,
) -> KgError {
    if is_cancelled(opts) {
        graph.clear_pending_constraint_violation();
        return KgError::Cancelled;
    }
    if let Some(violation) = graph.take_constraint_error(&message) {
        return violation;
    }
    if deadline_expired(opts) {
        return timeout_err(opts, started, message);
    }
    KgError::CypherExecution {
        message,
        position: None,
    }
}

/// Result of a successful execute. Wraps `CypherResult` with the
/// metadata callers need for output serialization (CSV, DataFrame,
/// PackStream record emission).
pub struct ExecuteOutcome {
    pub result: CypherResult,
    /// `true` when the query was a CREATE/SET/DELETE/REMOVE/MERGE.
    /// `execute_read` rejects those upfront via `KgError::Argument`.
    pub is_mutation: bool,
    /// Set when the user passes `RETURN ... FORMAT CSV` (kglite
    /// extension); pyapi + mcp-server format the result accordingly.
    pub output_format: OutputFormat,
    /// Set when the user prefixed the query with `EXPLAIN`, in which case
    /// `result` holds rendered plan rows rather than data.
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
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
pub fn execute_read(
    graph: &DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
) -> Result<ExecuteOutcome, KgError> {
    let (outcome, id_warnings) =
        crate::graph::dir_graph::collect_id_warnings(|| read_statement(graph, query, opts));
    outcome.map(|outcome| with_id_warnings(outcome, id_warnings))
}

// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
pub(super) fn read_statement(
    graph: &DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
) -> Result<ExecuteOutcome, KgError> {
    let started = Instant::now();
    let (
        PreparedQuery {
            plan: parsed,
            params,
            encode_plan,
            warnings,
        },
        echo,
    ) = timeless_route(graph, query, prepare(graph, query, opts, false)?, opts)?;
    let is_mutation = cypher::is_mutation_query(&parsed);
    // Attribute the plan-cache events `prepare` just caused, now that the
    // statement kind is known. Test-only; see `plan_cache::instrumentation`.
    #[cfg(test)]
    cypher::plan_cache::instrumentation::classify_pending(is_mutation);

    if parsed.explain {
        let mut result = cypher::generate_explain_result(&parsed, graph);
        attach_diagnostics(&mut result, &warnings, started, opts, echo);
        return Ok(ExecuteOutcome {
            result,
            is_mutation,
            output_format: parsed.output_format,
            explain: true,
        });
    }

    if is_mutation {
        return Err(KgError::Argument(
            "execute_read called with a mutation query (CREATE/SET/DELETE/REMOVE/MERGE, \
             CREATE INDEX/DROP INDEX) — use execute_mut against a mutable graph view"
                .to_string(),
        ));
    }

    let mut result = cypher::CypherExecutor::with_params(graph, &params, opts.deadline)
        .with_max_work_units(opts.max_work_units)
        .with_row_limit(opts.row_limit)
        .with_streaming(opts.streaming)
        .with_parallel(opts.parallel)
        .with_cancel(opts.cancel.as_ref().map(CancelToken::flag))
        .with_csv_import(opts.csv_import.clone())
        .execute(&parsed)
        .map_err(|message| exec_err(opts, started, message))?;
    // Encode codec'd-property result columns back to the typed form
    // (`42` → `'Q42'`). Eager rows only; lazy results (Python's streaming
    // path) materialize later and aren't covered — the configured consumer
    // (mcp-server) runs eager.
    cypher::value_codec::apply_encode(&mut result, &encode_plan);
    super::resolve_noderefs(&graph.graph, &mut result.rows);
    attach_diagnostics(&mut result, &warnings, started, opts, echo);

    Ok(ExecuteOutcome {
        result,
        is_mutation: false,
        output_format: parsed.output_format,
        explain: false,
    })
}

/// Whether every write a statement can make to a disk graph's column stores
/// goes through the two channels `StatementCheckpoint::DiskCells` journals:
/// the staged-node flush (every `SET`/`REMOVE` of a property, including a
/// `MERGE`'s `ON CREATE`/`ON MATCH` items) and the node append of a `CREATE` or
/// a `MERGE` that creates.
///
/// A whitelist: a statement with no write at all is not one, and everything it
/// does not name falls out — `FOREACH`, procedure calls, `LOAD CSV`, label
/// changes, `SET var += map` and nested `SET` paths — because each of those
/// either reaches a store another way or has not been shown not to.
/// `DELETE` qualifies: it frees the node's slot and edges (restored from the
/// snapshot) and writes no column.
fn writes_only_journaled_disk_cells(query: &CypherQuery) -> bool {
    use cypher::ast::{RemoveItem, SetItem};
    let plain_set =
        |item: &SetItem| matches!(item, SetItem::Property { path, .. } if path.is_empty());
    let mut any_write = false;
    let journaled = query.clauses.iter().all(|clause| match clause {
        Clause::Set(set) => {
            any_write = true;
            set.items.iter().all(plain_set)
        }
        Clause::Remove(remove) => {
            any_write = true;
            remove
                .items
                .iter()
                .all(|item| matches!(item, RemoveItem::Property { .. }))
        }
        Clause::Merge(merge) => {
            any_write = true;
            merge.on_create.iter().flatten().all(plain_set)
                && merge.on_match.iter().flatten().all(plain_set)
        }
        Clause::Create(_) | Clause::Delete(_) => {
            any_write = true;
            true
        }
        Clause::Match(_)
        | Clause::OptionalMatch(_)
        | Clause::Where(_)
        | Clause::Filter(_)
        | Clause::Return(_)
        | Clause::Finish
        | Clause::With(_)
        | Clause::OrderBy(_)
        | Clause::Skip(_)
        | Clause::Limit(_)
        | Clause::Unwind(_) => true,
        _ => false,
    });
    journaled && any_write
}

/// Whether this statement has no fallible operation after its first write.
///
/// This is intentionally a proof whitelist, not a general optimiser:
///
/// - one standalone node `CREATE` evaluates properties, validates scope/schema,
///   and checks primary-key uniqueness before inserting its only node;
/// - a terminal variable-only `DELETE` collects bindings and validates plain
///   delete edge constraints before removing anything.
///
/// Both shapes are safe only on the default in-memory backend (a `DELETE` also
/// under the durable capture wrapper) and without an execution budget, which
/// is checked after a write. A `CREATE` additionally needs no enforced node
/// ontology: that rule is judged on the stored node after the insert, and a
/// refusal then has to roll the insert back. Deadline/cancellation is safe: CREATE polls immediately before insertion, and DELETE immediately
/// before its non-interruptible removal phase, so a deadline error from either
/// has applied nothing — which is also why the late-statement check in
/// `mut_statement` runs only when a checkpoint is open. Every other mutation
/// retains the full rollback checkpoint.
fn can_skip_rollback_checkpoint(
    graph: &DirGraph,
    query: &CypherQuery,
    opts: &ExecuteOptions<'_>,
) -> bool {
    if query.profile || opts.max_work_units.is_some() {
        return false;
    }

    match query.clauses.as_slice() {
        // An enforced node rule is judged after the insert, and a refusal then
        // needs the checkpoint to undo it.
        [Clause::Create(create)] => {
            graph.graph.supports_checkpoint_free_mutation()
                && !graph.ontology_node_gate
                && !graph.ontology_tx_judges_here()
                && matches!(
                    create.patterns.as_slice(),
                    [pattern] if matches!(pattern.elements.as_slice(), [CreateElement::Node(_)])
                )
        }
        clauses => {
            let Some((Clause::Delete(delete), prefix)) = clauses.split_last() else {
                return false;
            };
            graph.graph.supports_checkpoint_free_delete()
                && !graph.ontology_tx_judges_here()
                && delete
                    .expressions
                    .iter()
                    .all(|expr| matches!(expr, cypher::ast::Expression::Variable(_)))
                && prefix
                    .iter()
                    .all(|clause| !cypher::executor::write::clause_is_mutation(clause))
        }
    }
}

/// Mutating execution. Caller passes `&mut DirGraph` (typically from
/// `Transaction::working_mut()`). For pure reads, use [`execute_read`].
///
/// A read query passed here runs against the mutable graph view as a read and
/// returns `is_mutation: false`, so the caller knows nothing was changed.
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
pub fn execute_mut(
    graph: &mut DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
) -> Result<ExecuteOutcome, KgError> {
    let (outcome, id_warnings) = crate::graph::dir_graph::collect_id_warnings(|| {
        mut_statement(
            graph,
            query,
            opts,
            &mut StatementCheckpoint::None,
            false,
            false,
        )
    });
    outcome.map(|outcome| with_id_warnings(outcome, id_warnings))
}

/// [`execute_mut`] for a caller that must still be able to undo the statement
/// after it succeeded (a write-ahead-log append that fails, a panic unwinding
/// past the statement): the statement's rollback checkpoint is parked in
/// `held` instead of being closed.
///
/// The caller owns closing it: `held.take()` then `commit` once the write is
/// final, or `rollback` to restore the pre-statement graph. A checkpoint
/// parked in `held` is also what a caught panic rolls back, so it is stored
/// there *before* the statement's first write. `keep_undo` forces a checkpoint
/// even for the shapes `can_skip_rollback_checkpoint` proves cannot fail, since
/// the caller's own later step can.
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
pub(super) fn execute_mut_held(
    graph: &mut DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
    held: &mut StatementCheckpoint,
    keep_undo: bool,
) -> Result<ExecuteOutcome, KgError> {
    let (outcome, id_warnings) = crate::graph::dir_graph::collect_id_warnings(|| {
        mut_statement(graph, query, opts, held, true, keep_undo)
    });
    outcome.map(|outcome| with_id_warnings(outcome, id_warnings))
}

/// Add the duplicate-id warnings a statement raised to its diagnostics, echoed
/// like the other query warnings. A failed statement is rolled back, so the
/// caller drops its warnings with it.
fn with_id_warnings(mut outcome: ExecuteOutcome, id_warnings: Vec<String>) -> ExecuteOutcome {
    if !id_warnings.is_empty() {
        cypher::emit_query_warnings(&id_warnings);
        let diagnostics = outcome
            .result
            .diagnostics
            .get_or_insert_with(Default::default);
        for warning in id_warnings {
            if !diagnostics.warnings.contains(&warning) {
                diagnostics.warnings.push(warning);
            }
        }
    }
    outcome
}

// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
fn mut_statement(
    graph: &mut DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
    slot: &mut StatementCheckpoint,
    parked: bool,
    keep_undo: bool,
) -> Result<ExecuteOutcome, KgError> {
    let started = Instant::now();
    let (
        PreparedQuery {
            plan: parsed,
            params,
            encode_plan,
            warnings,
        },
        echo,
    ) = timeless_route(graph, query, prepare(graph, query, opts, false)?, opts)?;
    let is_mutation = cypher::is_mutation_query(&parsed);
    // See the identical call in `execute_read`. Test-only.
    #[cfg(test)]
    cypher::plan_cache::instrumentation::classify_pending(is_mutation);

    // EXPLAIN never executes the mutation. Return before collision preflight,
    // disk promotion, or an atomic rollback checkpoint so inspecting a write
    // plan remains a read-only, O(plan) operation.
    if parsed.explain {
        let mut result = cypher::generate_explain_result(&parsed, graph);
        attach_diagnostics(&mut result, &warnings, started, opts, echo);
        return Ok(ExecuteOutcome {
            result,
            is_mutation,
            output_format: parsed.output_format,
            explain: true,
        });
    }

    if is_mutation {
        // A `&mut DirGraph` reached here without passing `make_dir_graph_mut`
        // (a transaction working copy, an embedder's owned graph). Build a
        // deferred load's indexes before the statement writes into them —
        // `DirGraph::indexes_deferred` for why it must be before.
        graph.materialize_indexes();
        let mut names = Vec::new();
        collect_mutation_names(&parsed, &mut names);
        graph
            .interner
            .validate_names(names)
            .map_err(KgError::from)?;
    }

    // A statement is atomic even when the caller supplied an already-
    // materialized transaction working copy. Open a checkpoint whenever the
    // executor can still fail after its first write; the narrow in-memory
    // exemptions are proved in `can_skip_rollback_checkpoint`.
    //
    // The checkpoint is an undo journal (O(changes)) wherever the journal can
    // reverse everything the statement may touch, and a whole-graph clone
    // (O(V+E)) otherwise; `dir_graph::rollback` owns that decision. Either way
    // it MUST be closed on every exit path — `commit` is what uninstalls the
    // capture journal.
    *slot = if is_mutation && (keep_undo || !can_skip_rollback_checkpoint(graph, &parsed, opts)) {
        StatementCheckpoint::open_for_statement(
            graph,
            cypher::executor::write::mutates_cdc_configuration(&parsed),
            // Only a disk graph has a route for it; skip the clause walk elsewhere.
            graph.graph.as_disk().is_some() && writes_only_journaled_disk_cells(&parsed),
        )
    } else {
        StatementCheckpoint::None
    };

    if is_mutation {
        if let Err(error) = graph.prepare_mutation() {
            std::mem::replace(slot, StatementCheckpoint::None).rollback(graph);
            return Err(KgError::FileIo(error));
        }
    }

    let mut result = if is_mutation {
        let interrupt = crate::graph::algorithms::Interrupt {
            deadline: opts.deadline,
            cancel: opts.cancel.as_ref().map(CancelToken::flag),
        };
        // Foreign model callbacks are available only to mutable CALL
        // dispatch. The read executor and its parallel paths never receive
        // this borrowed service.
        let embedding_service = opts
            .embedder
            .as_deref()
            .map(|model| EmbeddingExecutionService { model, interrupt });
        // Install the execution-scoped write whitelist for the duration of this
        // mutation, then clear it unconditionally (even on error) so it never
        // leaks into a later execution on the same working copy.
        graph.active_write_scope = opts.write_scope.cloned();
        // Same lifecycle as the write scope: no violation parked by an earlier
        // execution on this working copy may be read by this one.
        graph.clear_pending_constraint_violation();
        let r = graph.with_write_provenance(opts.git_sha, opts.modified_by, |graph| {
            cypher::executor::write::execute_mutable_with_csv(
                graph,
                &parsed,
                params,
                interrupt,
                cypher::executor::write::MutationLimits {
                    max_work_units: opts.max_work_units,
                    row_limit: opts.row_limit,
                },
                &opts.csv_import,
                embedding_service.as_ref(),
            )
        });
        graph.active_write_scope = None;
        // Write loops poll before each row and clause, so a statement whose
        // last write (or trailing RETURN) finished past the deadline would
        // otherwise commit and report success. Only with a checkpoint: a
        // checkpoint-free statement polled for the last time before its first
        // write, and a deadline error must never follow writes it cannot undo.
        let r = r.and_then(|result| {
            if matches!(slot, StatementCheckpoint::None) {
                return Ok(result);
            }
            cypher::executor::check_statement_interrupt(&interrupt).map(|()| result)
        });
        let r = match r {
            Ok(result) => result,
            Err(message) => {
                // Recover the structured violation *before* rolling back, so
                // the typed error never depends on what a checkpoint restore
                // does to the graph's transient fields.
                let error = mutation_err(graph, opts, started, message);
                std::mem::replace(slot, StatementCheckpoint::None).rollback(graph);
                return Err(error);
            }
        };
        if !parked {
            std::mem::replace(slot, StatementCheckpoint::None).commit(graph);
        }
        // A Cypher write occurred — advance the graph version so any
        // version-keyed caches (the plan cache) and OCC see the change.
        // Bumps the working copy directly so a read-after-write *within* the
        // same transaction re-plans against the mutated state; the eventual
        // commit recomputes the live version independently (see Session::commit).
        graph.bump_version();
        // Re-enforce `set_memory_limit` over what the statement just wrote.
        // A write that *creates* a column — a SET for a property the type has
        // never carried — puts O(rows) of fresh heap behind a limit last
        // checked at consolidation time; without this, writing one new
        // property permanently escapes the only bound a caller can place on
        // the columnar heap. A no-op (one `Option` test) with no limit set, an
        // O(columns) heap sum with one; materialisation runs only when over.
        // A parked checkpoint can still be rolled back, and a spill moves the
        // columns the journal replays into: the caller spills once final.
        if !parked {
            graph.maybe_spill_columns();
        }
        r
    } else {
        cypher::CypherExecutor::with_params(graph, &params, opts.deadline)
            .with_max_work_units(opts.max_work_units)
            .with_row_limit(opts.row_limit)
            .with_streaming(opts.streaming)
            .with_parallel(opts.parallel)
            .with_cancel(opts.cancel.as_ref().map(CancelToken::flag))
            .execute(&parsed)
            .map_err(|message| exec_err(opts, started, message))?
    };
    // Encode codec'd-property result columns (e.g. `CREATE (...) RETURN n.id`
    // reads back `'Q42'`). Eager path only; see execute_read.
    cypher::value_codec::apply_encode(&mut result, &encode_plan);
    super::resolve_noderefs(&graph.graph, &mut result.rows);
    attach_diagnostics(&mut result, &warnings, started, opts, echo);

    Ok(ExecuteOutcome {
        result,
        is_mutation,
        output_format: parsed.output_format,
        explain: false,
    })
}

fn collect_pattern_names<'a>(pattern: &'a CreatePattern, out: &mut Vec<&'a str>) {
    for element in &pattern.elements {
        match element {
            CreateElement::Node(node) => {
                out.extend(node.label.as_deref());
                out.extend(node.extra_labels.iter().map(String::as_str));
                out.extend(node.properties.iter().map(|(name, _)| name.as_str()));
            }
            CreateElement::Edge(edge) => {
                out.push(edge.connection_type.as_str());
                out.extend(edge.properties.iter().map(|(name, _)| name.as_str()));
            }
        }
    }
}

fn collect_set_names<'a>(items: &'a [SetItem], out: &mut Vec<&'a str>) {
    for item in items {
        match item {
            SetItem::Property { property, .. } => out.push(property),
            SetItem::Label { label, .. } => out.push(label),
            SetItem::Map { .. } => {}
        }
    }
}

fn collect_mutation_names<'a>(query: &'a CypherQuery, out: &mut Vec<&'a str>) {
    collect_clause_names(&query.clauses, out);
}

fn collect_clause_names<'a>(clauses: &'a [Clause], out: &mut Vec<&'a str>) {
    for clause in clauses {
        match clause {
            Clause::Create(create) => {
                for pattern in &create.patterns {
                    collect_pattern_names(pattern, out);
                }
            }
            Clause::Set(set) => collect_set_names(&set.items, out),
            Clause::Remove(remove) => {
                for item in &remove.items {
                    match item {
                        RemoveItem::Property { property, .. } => out.push(property),
                        RemoveItem::Label { label, .. } => out.push(label),
                    }
                }
            }
            Clause::Merge(merge) => {
                collect_pattern_names(&merge.pattern, out);
                if let Some(items) = &merge.on_create {
                    collect_set_names(items, out);
                }
                if let Some(items) = &merge.on_match {
                    collect_set_names(items, out);
                }
            }
            Clause::Foreach { body, .. } => collect_clause_names(body, out),
            Clause::CallSubquery { body, .. } => collect_mutation_names(body, out),
            Clause::Union(union) => collect_mutation_names(&union.query, out),
            _ => {}
        }
    }
}

/// Output of [`prepare`]: the parsed+optimized query, the (possibly
/// embedding-augmented) param map, the column-indexed value-codec encode plan
/// (empty when no codecs apply), and the non-fatal schema warnings this
/// statement earned.
pub(super) struct PreparedQuery {
    pub(super) plan: Arc<CypherQuery>,
    pub(super) params: HashMap<String, Value>,
    pub(super) encode_plan: Vec<Option<ValueCodec>>,
    /// Unknown-label / unknown-relationship-type / absent-property warnings.
    /// Behind an `Arc` because the plan cache stores them alongside the plan.
    pub(super) warnings: Arc<[String]>,
}

fn plan_scope(
    graph: &DirGraph,
    opts: &ExecuteOptions<'_>,
    suppress_default: bool,
) -> cypher::plan_cache::PlanScope {
    cypher::plan_cache::PlanScope {
        graph_id: graph.graph_id(),
        version: graph.version(),
        schema_locked: graph.schema_locked,
        lazy: opts.lazy_eligible,
        suppress_default,
        valid_time_default: graph.valid_time_default.cache_code(),
    }
}

/// The plan-cache hit path: a hit skips everything below the lookup in
/// [`prepare`] — parse, validate, the schema pass, optimize. Entries are
/// stored post lazy-marking for this `lazy_eligible`, so a hit is a pure `Arc`
/// clone. The caller gates on the `cacheable` predicate.
///
/// A hit also skips parameter-presence validation. That is sound because an
/// entry only exists after the full parsed AST was checked against the same
/// empty parameter map. A statement that references any parameter errors above
/// the insert, so a repeated unbound statement misses and raises again. Pinned
/// by `session::param_presence_tests`.
///
/// The warnings come out of the entry rather than being recomputed: recomputing
/// needs the parsed AST, and not skipping the parse is exactly what this early
/// return exists to avoid. Their validity is the cache's own soundness argument
/// — they are a pure function of `(query, graph schema)` and the key pins the
/// graph state. Stderr repeats them per call, as it did when every call parsed.
fn cached_plan(
    graph: &DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
    suppress_default: bool,
) -> Option<PreparedQuery> {
    let cached = cypher::plan_cache::get(plan_scope(graph, opts, suppress_default), query)?;
    cypher::emit_query_warnings(&cached.warnings);
    Some(PreparedQuery {
        plan: cached.plan,
        params: HashMap::new(),
        encode_plan: Vec::new(),
        warnings: cached.warnings,
    })
}

/// Shared preparation for both execution paths (the pipeline is in the module
/// header).
///
/// The returned param map is borrowed from `opts.params` in the common case;
/// a `text_score()` query clones-on-write to inject the embedding vectors.
///
/// **GIL note for binding implementers.** If `opts.embedder` is a
/// Python-backed embedder (PyEmbedderAdapter), the binding MUST release the
/// GIL before calling `execute_read`/`execute_mut` (Python's `py.detach`).
/// The embed call below re-acquires it briefly to invoke Python; failing to
/// release first deadlocks.
///
/// `suppress_default` keeps lowering from adding the default valid-time
/// context — the session's plain-plan re-prepare of a text whose default
/// context would filter nothing. It is part of the plan cache key.
// KgError carries query context; boxing it would only burden an error path.
#[allow(clippy::result_large_err)]
pub(super) fn prepare(
    graph: &DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
    suppress_default: bool,
) -> Result<PreparedQuery, KgError> {
    // Open a plan-cache attribution window for this statement. Test-only; the
    // caller closes it with `classify_pending` once it knows `is_mutation`.
    #[cfg(test)]
    cypher::plan_cache::instrumentation::begin_prepare();
    // Plan cache: a param-less, codec-free, no-disabled-passes query against an
    // unchanged graph reuses its fully-optimized plan, skipping parse + validate
    // + optimize. Keyed on (graph_id, version, lazy_eligible) so any mutation
    // invalidates it and it never leaks across graphs (see
    // `cypher::plan_cache`). [`cached_plan`] carries the hit-path soundness
    // argument.
    let cacheable = opts.params.is_empty()
        && opts.disabled_passes.is_none_or(|s| s.is_empty())
        && opts.value_codecs.is_none_or(|c| c.is_empty());
    if cacheable {
        if let Some(prepared) = cached_plan(graph, query, opts, suppress_default) {
            return Ok(prepared);
        }
    }

    with_query_stack(query, || {
        prepare_uncached(graph, query, opts, suppress_default, cacheable)
    })
}

/// Stack one nesting level of the parsed statement can cost the post-parse
/// walkers in `prepare` (dynamic-label binding, schema validation, warning
/// collection, `text_score` rewrite, planner). Measured per level at the
/// parser's ceiling: ~18.7 KiB for the dynamic-label walk in a debug build,
/// which is the largest; release walkers are well under a quarter of that.
const PREPARE_STACK_PER_LEVEL: usize = if cfg!(debug_assertions) {
    24 * 1024
} else {
    8 * 1024
};

/// Run `f` on a stack that can hold the deepest AST `query` could parse into.
///
/// Only the parser grows its own stack; the walkers `prepare` runs afterwards
/// recurse on the caller's. At the parser's nesting ceiling those walkers need
/// more than [`QUERY_THREAD_STACK_SIZE`] in a debug build, and a stack
/// overflow aborts the process. A query of N bytes cannot nest deeper than N
/// levels, so the bytes bound the need without a second walk of the text:
/// short statements — nearly all of them — see no extra stack, and the segment
/// is allocated only when the thread's remaining stack falls under the bound.
fn with_query_stack<R>(query: &str, f: impl FnOnce() -> R) -> R {
    let levels = query.len().min(cypher::parser::MAX_EXPRESSION_DEPTH);
    let need = levels * PREPARE_STACK_PER_LEVEL;
    let headroom = 512 * 1024;
    stacker::maybe_grow(need + headroom, need + 4 * headroom, f)
}

/// The cache-miss half of [`prepare`]: parse, validate, plan, and cache.
// KgError carries query context; boxing it would only burden an error path.
#[allow(clippy::result_large_err)]
fn prepare_uncached(
    graph: &DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
    suppress_default: bool,
    cacheable: bool,
) -> Result<PreparedQuery, KgError> {
    let mut parsed = cypher::parse_cypher(query)?;
    parsed.suppress_default = suppress_default;

    // Dynamic labels / relationship types (`MATCH (n:$label)`): bind them from
    // the caller's parameters FIRST, so validation, optimization and execution
    // all see an ordinary literal label. The parser cannot do this — parsed
    // ASTs are cached by query text and re-run with different parameters — so
    // it leaves a marker for this pass. The same pass rejects an inline
    // property map whose value names a parameter the caller did not bind
    // (`MATCH (v {flag: $flag})`), which the matcher's `bool`-returning filter
    // could only answer as "no match". See `cypher::dynamic_labels`.
    cypher::dynamic_labels::resolve(&mut parsed, opts.params)?;

    if let Some(name) = cypher::parameter_presence::first_missing_parameter(&parsed, opts.params) {
        return Err(KgError::CypherExecution {
            message: format!("Missing parameter: ${name}"),
            position: None,
        });
    }

    // value_codecs: decode operator-declared literals bound to a codec'd
    // property (`{id:'Q42'}` / `WHERE n.id = 'Q42'` → `42`) BEFORE anything
    // else, so validation, optimization, and execution all treat the decoded
    // form as canonical. No-op (one is_empty check) when none are configured.
    let codecs = opts.value_codecs.unwrap_or(&[]);
    cypher::value_codec::apply_decode(&mut parsed, codecs);
    // Build the result-side encode plan now, while the RETURN clause is a clean
    // pre-optimize projection (fusion later rewrites *how* columns are computed,
    // not the output schema). Column-indexed; empty when no codecs / no RETURN.
    let encode_plan = cypher::value_codec::build_encode_plan(&parsed, codecs);

    // Property typos in pattern literals (`{ttle: 'Alice'}`) are rejected
    // here, with a "did you mean?" hint.
    cypher::validate_schema(&parsed, graph).map_err(KgError::from)?;

    // Non-fatal: a MATCH that references an unknown node label or relationship
    // type — the most common "why is my query empty?" typo. Computed once and
    // consumed twice: stderr now (interactive users), and
    // `QueryDiagnostics.warnings` at the end of execute (every programmatic
    // surface, including the MCP server, which is where an agent reads them).
    let collected = cypher::collect_query_warnings(&parsed, graph, opts.params);
    // ...with one exception, and it is a *disposition* change, not a second
    // walk: under `lock_schema()` the absent-property subset becomes fatal.
    // `MATCH (p:Person) WHERE p.agee = 1` returning `[]` and `RETURN p.agee`
    // returning a null column are the read-side twins of the pattern-literal
    // typo `validate_schema` already rejects above, and a lock exists to catch
    // exactly that. Reversed arrows and unknown labels/rel-types stay warnings:
    // the first is heuristic, and the second is legal zero-row Cypher whose
    // locked-schema *label* case is already fatal via `validate_label`, so
    // promoting here would double-report it.
    //
    // The declared-type family (`WHERE p.age > 'forty'` on an `IS :: INTEGER`
    // property) promotes the same way, but only in part: a mismatch against a
    // write-enforced `IS :: T` constraint is a guarantee about every row the
    // query can see, while one against a `define_schema()` field type rests on
    // a declaration nothing checks at write time — so that half stays a
    // warning in both schema states, and stays cacheable with it.
    let strict_absent = !collected.absent_property.is_empty();
    let strict_type = collected
        .type_mismatch
        .iter()
        .any(|finding| finding.promotable());
    if graph.schema_locked {
        if let Some(error) = cypher::strict_read_error(&collected.absent_property, graph) {
            return Err(KgError::from(error));
        }
        if let Some(error) = cypher::strict_type_error(&collected.type_mismatch) {
            return Err(KgError::from(error));
        }
    }
    let warnings: Arc<[String]> = collected.into_messages().into();
    cypher::emit_query_warnings(&warnings);

    // Rewrites `text_score(...)` calls to `vector_score(...)` and
    // the embedding `query` procedures' `text` option to `vector`, collecting the
    // texts to embed alongside.
    let rewrite = cypher::rewrite_text_score(&mut parsed, opts.params).map_err(|message| {
        KgError::CypherExecution {
            message,
            position: None,
        }
    })?;

    // EXPLAIN renders plan rows without executing, so it needs no embedding.
    let mut params: Cow<'_, HashMap<String, Value>> =
        if !rewrite.texts_to_embed.is_empty() && !parsed.explain {
            Cow::Owned(embed_into_params(opts, &rewrite)?)
        } else {
            Cow::Borrowed(opts.params)
        };
    record_text_score_stores(&mut params, &rewrite);

    let disabled_default = cypher::planner::empty_disabled_set();
    let disabled_ref = opts.disabled_passes.unwrap_or(disabled_default);
    cypher::planner::optimize_with_disabled(&mut parsed, graph, &params, disabled_ref);
    cypher::valid_time::check_executable(&parsed)
        .map_err(|message| exec_err(opts, Instant::now(), message))?;

    // Lazy marking — only when the caller asked for it. Done BEFORE caching so
    // the cached plan is ready-to-execute for this `lazy_eligible` (the cache
    // key includes it), making hits a pure Arc clone. Without this the executor
    // materializes rows eagerly; with it, `result.lazy` may be Some and
    // `result.rows` empty and the caller must materialize (Python's ResultView
    // does; bolt-server doesn't, so it passes `lazy_eligible: false`).
    if opts.lazy_eligible {
        cypher::mark_lazy_eligibility(&mut parsed);
    }

    let plan = Arc::new(parsed);
    // Cache the ready-to-execute plan. Only when `params` stayed empty — a
    // `text_score()` rewrite injects embedding params, making the plan
    // call-specific, so those are never cached (and thus never hit above).
    //
    // And only for **reads**. A mutation's key carries the graph version, and
    // a successful mutation bumps that version immediately after this insert
    // (`bump_version`), so the entry is stale the instant it lands: measured
    // at 600 identical serial writes → 600 insertions, **0 hits**, 88
    // evictions, and a shared 512-entry cache left entirely full of entries
    // only the writer could ever have reached. Skipping the insert costs a
    // writer nothing and stops a write loop evicting every *other* graph's
    // live read plans out of a process-global cache.
    //
    // Two same-version replays lose their reuse, deliberately: transactions
    // forked from one base version (same `graph_id` + `version`), and a retry
    // of a mutation that errored before `bump_version`. A narrow window traded
    // for a per-write cost every serial writer pays; pinned in both directions
    // by `session::plan_cache_cost_tests`.
    //
    // The **lookup** above deliberately stays. `prepare` runs before anything
    // has parsed the query, so this classification does not exist yet there,
    // and buying it early means a `parse_cypher_cached` AST clone (71 ns for
    // `RETURN 1`, measured 2026-09-29, growing with the AST) on the read-hit path that the plan cache exists to
    // keep at ~1.9 us. With no mutation ever inserted, a mutation's lookup is
    // a guaranteed miss: one shared read lock and one hash, and nothing more.
    //
    // And never for a statement carrying findings a lock would promote, which
    // is what makes the promotions above independent of the cache: a hit
    // returns before the schema pass runs and cannot re-decide anything, so
    // refusing to store such a plan makes a hit *prove* there was nothing to
    // promote. "Prime unlocked → lock_schema() → rerun the same text" then
    // raises whether or not locking bumped the graph version (through the
    // Python/`api` surface it does — `make_dir_graph_mut`; a core caller
    // flipping the flag on an owned `DirGraph` does not).
    //
    // The exclusion tracks the promotable subset exactly: every
    // absent-property statement (that family promotes wholesale), but only the
    // *declared*-type mismatches — a `define_schema()`-sourced one can never
    // become an error, so its plan stays cached and its warning rides the
    // entry. What stays excluded reads an all-null column or asks a question
    // no row can answer, which no hot loop should contain.
    if cacheable
        && params.is_empty()
        && !strict_absent
        && !strict_type
        && !cypher::is_mutation_query(&plan)
    {
        cypher::plan_cache::insert(
            plan_scope(graph, opts, suppress_default),
            query,
            plan.clone(),
            Arc::clone(&warnings),
        );
    }

    Ok(PreparedQuery {
        plan,
        params: params.into_owned(),
        encode_plan,
        warnings,
    })
}

/// A statement under `FOR VALID_TIME AS OF` (written, or the graph's default
/// of today) whose filter would remove nothing — every declared target of the
/// graph timeless at this execution's instant — runs the plan of its text
/// without the prefix and without the default: the normal planner, plan cache
/// and fused routes, and the same rows. Re-decided on every
/// execution (see `valid_time::timeless_plain_text`); EXPLAIN keeps the
/// guarded plan. Also returns the statement's valid-time echo, taken from the
/// plan with the context, and `None` without one.
#[allow(clippy::result_large_err)] // KgError carries query context, as `prepare`'s does.
pub(super) fn timeless_route(
    graph: &DirGraph,
    query: &str,
    prepared: PreparedQuery,
    opts: &ExecuteOptions<'_>,
) -> Result<(PreparedQuery, Option<TemporalDiagnostics>), KgError> {
    if prepared.plan.context.is_none() {
        return Ok((prepared, None));
    }
    let plan = &prepared.plan;
    match cypher::valid_time::timeless_plain_text(query, plan, graph, &prepared.params) {
        Some(plain) => {
            let echo = cypher::valid_time::temporal_echo(plan, graph, &prepared.params, "plain");
            Ok((prepare(graph, &plain, opts, true)?, echo))
        }
        None => {
            let echo = cypher::valid_time::temporal_echo(plan, graph, &prepared.params, "guarded");
            Ok((prepared, echo))
        }
    }
}

/// Attach `QueryDiagnostics` to a finished result — the single place every
/// execution path (read, mutation, EXPLAIN) leaves them.
///
/// `prepare_warnings` are the schema warnings from [`prepare`]; the executor
/// may already have parked runtime ones (procedure scoping) on the result's
/// diagnostics, and those keep their place after the schema ones, which
/// explain an empty result before any runtime advisory does.
///
/// `timeout_ms` is the deadline that was actually in force, measured from its
/// origin (see [`ExecuteOptions::deadline_origin`]). The Python read path
/// overwrites it with the caller's resolved `timeout_ms`.
fn attach_diagnostics(
    result: &mut CypherResult,
    prepare_warnings: &[String],
    started: Instant,
    opts: &ExecuteOptions<'_>,
    echo: Option<TemporalDiagnostics>,
) {
    let mut diagnostics = result.diagnostics.take().unwrap_or_default();
    diagnostics.temporal = echo.map(|echo| Box::new(finish_echo(echo, &diagnostics)));
    if !prepare_warnings.is_empty() {
        let mut merged = prepare_warnings.to_vec();
        merged.append(&mut diagnostics.warnings);
        diagnostics.warnings = merged;
    }
    // `elapsed_ms` is this statement's duration, not time since the deadline
    // was resolved (a transaction's deadline is resolved at `begin()`);
    // `timeout_ms` is the configured limit.
    diagnostics.elapsed_ms = started.elapsed().as_millis() as u64;
    diagnostics.timeout_ms = deadline_figures(opts, started).1;
    result.diagnostics = Some(diagnostics);
}

/// `echo` with what the executor recorded: a vector retrieval's masked route,
/// and whether an algorithm ran on the valid slice.
fn finish_echo(
    mut echo: TemporalDiagnostics,
    diagnostics: &QueryDiagnostics,
) -> TemporalDiagnostics {
    echo.slice = diagnostics.temporal.as_ref().is_some_and(|t| t.slice);
    echo.retrieval = diagnostics.retrieval.iter().find_map(|record| {
        if record.actual_mode == "hnsw_mask" {
            Some("hnsw_mask".to_string())
        } else {
            let reason = record.fallback_reason.as_deref().unwrap_or_default();
            reason
                .starts_with("exact_mask")
                .then(|| "exact_mask".into())
        }
    });
    echo
}

/// Lets a missing-store error name `text_score` and the source property the
/// user wrote instead of the `vector_score` / `<property>_emb` it became.
fn record_text_score_stores(
    params: &mut Cow<'_, HashMap<String, Value>>,
    rewrite: &cypher::planner::simplification::TextScoreRewrite,
) {
    if rewrite.text_score_stores.is_empty() {
        return;
    }
    params.to_mut().insert(
        cypher::planner::simplification::TEXT_SCORE_STORES_PARAM.to_string(),
        Value::List(
            rewrite
                .text_score_stores
                .iter()
                .map(|store| Value::String(store.clone()))
                .collect(),
        ),
    );
}

/// Run the embedder on collected texts; inject the vectors into a clone of
/// the param map. Caller-supplied params are not mutated.
// KgError carries query context; boxing it would only burden an error path.
#[allow(clippy::result_large_err)]
fn embed_into_params(
    opts: &ExecuteOptions<'_>,
    rewrite: &cypher::planner::simplification::TextScoreRewrite,
) -> Result<HashMap<String, Value>, KgError> {
    let model = opts
        .embedder
        .as_ref()
        .ok_or_else(|| KgError::CypherExecution {
            message: "Embedding query text for text_score() or \
                      CALL db.node_embeddings.query / db.relationship_embeddings.query({text: ...}) \
                      requires a registered \
                      embedding model. \
                      Call g.set_embedder(model) first (Python) or pass an embedder \
                      via ExecuteOptions::embedder (downstream Rust consumers), \
                      or pass the query as a vector."
                .to_string(),
            position: None,
        })?;
    model.load().map_err(|message| KgError::CypherExecution {
        message,
        position: None,
    })?;
    let texts: Vec<String> = rewrite
        .texts_to_embed
        .iter()
        .map(|(_, t)| t.clone())
        .collect();
    let embed_result = model.embed(&texts);
    let dimension = model.dimension();
    model.unload();
    let embeddings: Vec<Vec<f32>> = embed_result.map_err(|message| KgError::CypherExecution {
        message,
        position: None,
    })?;
    if embeddings.len() != texts.len() {
        return Err(KgError::CypherExecution {
            message: format!(
                "text_score: model.embed() returned {} vectors for {} texts",
                embeddings.len(),
                texts.len()
            ),
            position: None,
        });
    }
    for vector in &embeddings {
        if vector.len() != dimension {
            return Err(KgError::CypherExecution {
                message: format!(
                    "text_score: model returned a vector of dimension {} (expected {})",
                    vector.len(),
                    dimension
                ),
                position: None,
            });
        }
        crate::graph::embedding_validation::validate_finite_vector(vector).map_err(|error| {
            KgError::CypherExecution {
                message: format!("text_score: model returned an invalid vector: {error}"),
                position: None,
            }
        })?;
    }
    let mut params = opts.params.clone();
    for (i, (param_name, _)) in rewrite.texts_to_embed.iter().enumerate() {
        // Native `Value::List`, not a JSON string: the same shape a caller
        // who supplies their own query vector passes in, so both routes into
        // `vector_score` converge and neither re-parses per row.
        let vector = Value::List(
            embeddings[i]
                .iter()
                .map(|f| Value::Float64(*f as f64))
                .collect(),
        );
        params.insert(param_name.clone(), vector);
    }
    Ok(params)
}

#[cfg(test)]
mod version_soundness_tests {
    use super::*;
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::storage::GraphRead;

    /// A Cypher write through `execute_mut` must advance the graph version so
    /// version-keyed caches (the plan cache) and a read-after-write within the
    /// same transaction observe the change.
    #[test]
    fn execute_mut_write_bumps_version() {
        let mut g = DirGraph::new();
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        let before = g.version();
        execute_mut(&mut g, "CREATE (:Item {id: 1})", &opts).expect("create");
        assert!(
            g.version() > before,
            "a Cypher write must bump version (was {before}, now {})",
            g.version()
        );
    }

    /// A read must NOT bump the version — otherwise repeated reads would
    /// perpetually invalidate the plan cache.
    #[test]
    fn execute_read_does_not_bump_version() {
        let mut g = DirGraph::new();
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        execute_mut(&mut g, "CREATE (:Item {id: 1})", &opts).expect("create");
        let after_write = g.version();
        let _ = execute_read(&g, "MATCH (n:Item) RETURN n.id", &opts).expect("read");
        assert_eq!(g.version(), after_write, "a read must not bump version");
    }

    #[test]
    fn cypher_collision_is_typed_and_atomic() {
        let mut g = DirGraph::new();
        let incoming = "CollisionType";
        g.interner
            .try_register(
                crate::graph::schema::InternedKey::from_str(incoming),
                "conflicting-existing",
            )
            .unwrap();
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        let error = match execute_mut(&mut g, "CREATE (:CollisionType {id: 1})", &opts) {
            Err(error) => error,
            Ok(_) => panic!("colliding Cypher name must be rejected"),
        };
        assert!(matches!(error, KgError::InternerCollision(_)));
        assert_eq!(g.graph.node_count(), 0);
        assert_eq!(g.version(), 0);
    }

    #[test]
    fn checkpointed_multi_create_rolls_back_late_expression_error() {
        let mut g = DirGraph::new();
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        let error = match execute_mut(
            &mut g,
            "CREATE (:Item {id: 1}), (:Item {id: 2, broken: duration({months: 2147483648})})",
            &opts,
        ) {
            Err(error) => error,
            Ok(_) => panic!("the second CREATE expression must fail"),
        };
        assert!(
            error.to_string().contains("duration()"),
            "unexpected error: {error}"
        );
        assert_eq!(g.graph.node_count(), 0, "the first CREATE must roll back");
        assert_eq!(g.version(), 0);
    }

    #[test]
    fn checkpoint_free_mutations_cancel_before_their_first_write() {
        use std::sync::atomic::AtomicBool;
        static CANCEL: AtomicBool = AtomicBool::new(false);

        let mut g = DirGraph::new();
        let params = HashMap::new();
        let base_opts = ExecuteOptions::eager(&params);
        execute_mut(&mut g, "CREATE (:Item {id: 1})", &base_opts).unwrap();

        CANCEL.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut cancelled_opts = ExecuteOptions::eager(&params);
        cancelled_opts.cancel = Some(CancelToken::from_static(&CANCEL));
        assert!(matches!(
            execute_mut(&mut g, "MATCH (n:Item) DELETE n", &cancelled_opts),
            Err(KgError::Cancelled)
        ));
        assert_eq!(g.graph.node_count(), 1);
        CANCEL.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// A terminal `DELETE` skips the undo journal under the durable capture
    /// wrapper — the journal holds every removed node's full record until the
    /// statement commits — while a `CREATE`, a `DELETE` behind a write, and a
    /// profiled run keep their checkpoints.
    #[test]
    fn terminal_delete_skips_the_checkpoint_under_the_capture_wrapper() {
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        let skips = |g: &DirGraph, q: &str| {
            let parsed = cypher::parse_cypher(q).unwrap();
            can_skip_rollback_checkpoint(g, &parsed, &opts)
        };
        let delete = "MATCH (n:Item) DETACH DELETE n";
        let plain = DirGraph::new();
        assert!(skips(&plain, delete), "non-vacuity: a plain graph skips");

        let mut durable = DirGraph::new();
        crate::graph::storage::recording::wrap_for_durability(&mut durable).unwrap();
        assert!(durable.graph.is_recording());
        assert!(skips(&durable, delete));
        assert!(skips(&durable, "MATCH (n:Item) DELETE n"));
        assert!(!skips(&durable, "CREATE (:Item {id: 1})"));
        assert!(!skips(
            &durable,
            "MATCH (n:Item) SET n.v = 1 WITH n DETACH DELETE n"
        ));
        assert!(!skips(&durable, "PROFILE MATCH (n:Item) DETACH DELETE n"));
    }
}
