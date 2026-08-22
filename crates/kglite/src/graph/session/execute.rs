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

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use crate::datatypes::Value;
use crate::error::KgError;
use crate::graph::dir_graph::rollback::StatementCheckpoint;
use crate::graph::dir_graph::DirGraph;
use crate::graph::embedder::Embedder;
use crate::graph::languages::cypher;
use crate::graph::languages::cypher::ast::{
    Clause, CreateElement, CreatePattern, CypherQuery, OutputFormat, RemoveItem, SetItem,
};
use crate::graph::languages::cypher::executor::load_csv::CsvImportPolicy;
use crate::graph::languages::cypher::result::CypherResult;
use crate::graph::languages::cypher::value_codec::ValueCodec;

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
    /// Optional operator-declared value codecs. When set, query-side
    /// literals bound to a codec'd property are decoded before
    /// validation/optimization (`'Q42'` → `42`), and result columns
    /// that are direct projections of a codec'd property are encoded
    /// back (`42` → `'Q42'`). `None`/empty = no transform (the common
    /// case; zero hot-path cost). See `cypher::value_codec`.
    pub value_codecs: Option<&'a [ValueCodec]>,
    /// Optional cooperative-cancellation flag. The executor and pattern
    /// matcher poll it at the same checkpoints they poll `deadline`
    /// (one relaxed atomic load per ~4K comparisons); once set, the
    /// run aborts with [`KgError::Cancelled`]. `None` = never cancelled
    /// (zero hot-path cost). This is the engine-agnostic primitive each
    /// binding flips from its own signal model — the Python wheel wires
    /// it to a scoped SIGINT handler so Ctrl-C interrupts long queries;
    /// servers leave it `None` and use their own deadline/teardown.
    ///
    /// A `&'static` flag (not an owned `Arc`) because the only setter is
    /// a process-global signal handler, which can't capture state — the
    /// Python wheel points this at a `static AtomicBool` its SIGINT
    /// handler flips. Bindings that need this provide a `'static` flag;
    /// the rest pass `None`.
    pub cancel: Option<&'static AtomicBool>,
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
    /// from where.
    ///
    /// Defaults to [`CsvImportPolicy::Denied`], and deliberately so: `file://`
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
    /// The Bolt and MCP servers deliberately never set it (v1); the Python
    /// wheel exposes it as `kg.cypher(parallel=True)` and the CLI as
    /// `--parallel`.
    pub parallel: bool,
}

impl<'a> ExecuteOptions<'a> {
    /// Conservative defaults: `lazy_eligible: false` (safe for
    /// every consumer that doesn't have a lazy materializer), no
    /// deadline, no max_rows, no disabled passes, no embedder.
    /// Caller is expected to override at least `params`.
    ///
    /// Same as [`Self::eager`] — the two are synonyms. `new` is
    /// kept for Rust-convention API discovery; `eager` is the
    /// intent-named factory call-sites prefer.
    pub fn new(params: &'a HashMap<String, Value>) -> Self {
        Self::eager(params)
    }

    /// Eager-execution defaults — the safe default for any binding
    /// that doesn't have a lazy result materializer.
    ///
    /// This is the constructor non-Python bindings should reach for:
    /// `lazy_eligible: false`, no deadline, no max_rows, no disabled
    /// passes, no embedder. Override individual fields after
    /// construction if needed (deadline for timeouts, embedder when
    /// `text_score()` queries are expected).
    ///
    /// Lifted in 2026-05-25 to give the call-site the intent-named
    /// shape — previously mcp-server / bolt-server constructed the
    /// struct manually with identical defaults; now they call
    /// `ExecuteOptions::eager(params)` for self-documenting code.
    pub fn eager(params: &'a HashMap<String, Value>) -> Self {
        Self {
            params,
            deadline: None,
            max_rows: None,
            lazy_eligible: false,
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

    /// Opt this execution in to the parallel runtime.
    ///
    /// Builder form for the same reason [`Self::with_csv_import`] has one: a
    /// call-site reads as an explicit grant rather than a field assignment
    /// buried among defaults.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

/// Map an executor error string to a typed [`KgError`]. When the
/// caller's cooperative-cancellation flag is set, an aborted run is
/// reported as [`KgError::Cancelled`] (the binding maps that to its
/// interrupt type — `KeyboardInterrupt` in the Python wheel) rather
/// than a misleading `CypherExecution`. Otherwise it's a plain
/// execution error. `cancel == None` (every server binding) always
/// takes the `CypherExecution` branch, so behaviour is unchanged there.
#[inline]
fn is_cancelled(opts: &ExecuteOptions<'_>) -> bool {
    opts.cancel
        .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
}

#[inline]
fn exec_err(opts: &ExecuteOptions<'_>, message: String) -> KgError {
    if is_cancelled(opts) {
        KgError::Cancelled
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
/// The park is drained on every path so nothing survives into a later run.
fn mutation_err(graph: &mut DirGraph, opts: &ExecuteOptions<'_>, message: String) -> KgError {
    if is_cancelled(opts) {
        graph.clear_pending_constraint_violation();
        return KgError::Cancelled;
    }
    graph
        .take_constraint_error(&message)
        .unwrap_or(KgError::CypherExecution {
            message,
            position: None,
        })
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
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
pub fn execute_read(
    graph: &DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
) -> Result<ExecuteOutcome, KgError> {
    let started = Instant::now();
    let PreparedQuery {
        plan: parsed,
        params,
        encode_plan,
        warnings,
    } = prepare(graph, query, opts)?;
    let is_mutation = cypher::is_mutation_query(&parsed);
    // Attribute the plan-cache events `prepare` just caused, now that the
    // statement kind is known. Test-only; see `plan_cache::instrumentation`.
    #[cfg(test)]
    cypher::plan_cache::instrumentation::classify_pending(is_mutation);

    // EXPLAIN: render plan rows, skip execution.
    if parsed.explain {
        let mut result = cypher::generate_explain_result(&parsed, graph);
        attach_diagnostics(&mut result, &warnings, started, opts);
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
        .with_max_rows(opts.max_rows)
        .with_streaming(opts.lazy_eligible)
        .with_parallel(opts.parallel)
        .with_cancel(opts.cancel)
        .with_csv_import(opts.csv_import.clone())
        .execute(&parsed)
        .map_err(|message| exec_err(opts, message))?;
    // value_codecs: encode codec'd-property result columns back to the typed
    // form (`42` → `'Q42'`). Applies to eager rows; lazy results (Python's
    // streaming path) materialize later and aren't covered — the configured
    // consumer (mcp-server) runs eager.
    cypher::value_codec::apply_encode(&mut result, &encode_plan);
    attach_diagnostics(&mut result, &warnings, started, opts);

    Ok(ExecuteOutcome {
        result,
        is_mutation: false,
        output_format: parsed.output_format,
        explain: false,
    })
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
/// Both shapes are safe only on the default in-memory backend and without an
/// execution budget, which is checked after a write. Deadline/cancellation is
/// safe: CREATE polls before insertion, while DELETE completes every poll and
/// validation in its preflight phase before entering a non-interruptible commit
/// phase. Every other mutation retains the full rollback checkpoint.
fn can_skip_rollback_checkpoint(
    graph: &DirGraph,
    query: &CypherQuery,
    opts: &ExecuteOptions<'_>,
) -> bool {
    if !graph.graph.supports_checkpoint_free_mutation() || query.profile || opts.max_rows.is_some()
    {
        return false;
    }

    match query.clauses.as_slice() {
        [Clause::Create(create)] => matches!(
            create.patterns.as_slice(),
            [pattern] if matches!(pattern.elements.as_slice(), [CreateElement::Node(_)])
        ),
        clauses => {
            let Some((Clause::Delete(delete), prefix)) = clauses.split_last() else {
                return false;
            };
            delete
                .expressions
                .iter()
                .all(|expr| matches!(expr, cypher::ast::Expression::Variable(_)))
                && prefix
                    .iter()
                    .all(|clause| !cypher::executor::write::clause_is_mutation(clause))
        }
    }
}

/// Mutating execution. Caller passes `&mut DirGraph` (typically
/// from `Transaction::working_mut()`). For pure reads, use
/// [`execute_read`] instead.
///
/// Note: a read query passed to `execute_mut` runs against the
/// mutable graph view as a read. The function returns
/// `is_mutation: false` in that case so the caller knows nothing
/// was changed.
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
pub fn execute_mut(
    graph: &mut DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
) -> Result<ExecuteOutcome, KgError> {
    let started = Instant::now();
    let PreparedQuery {
        plan: parsed,
        params,
        encode_plan,
        warnings,
    } = prepare(graph, query, opts)?;
    let is_mutation = cypher::is_mutation_query(&parsed);
    // See the identical call in `execute_read`. Test-only.
    #[cfg(test)]
    cypher::plan_cache::instrumentation::classify_pending(is_mutation);

    // EXPLAIN never executes the mutation. Return before collision preflight,
    // disk promotion, or an atomic rollback checkpoint so inspecting a write
    // plan remains a read-only, O(plan) operation.
    if parsed.explain {
        let mut result = cypher::generate_explain_result(&parsed, graph);
        attach_diagnostics(&mut result, &warnings, started, opts);
        return Ok(ExecuteOutcome {
            result,
            is_mutation,
            output_format: parsed.output_format,
            explain: true,
        });
    }

    if is_mutation {
        let mut names = Vec::new();
        collect_mutation_names(&parsed, &mut names);
        graph
            .interner
            .validate_names(names)
            .map_err(KgError::from)?;
    }

    // A statement is atomic even when the caller supplied an already-
    // materialized transaction working copy. Open a checkpoint whenever the
    // executor can still fail after its first write. A deliberately narrow
    // in-memory fast path covers two shapes whose executors finish all
    // fallible work before mutating; see `can_skip_rollback_checkpoint`.
    //
    // The checkpoint is an undo journal (O(changes)) wherever the journal can
    // reverse everything the statement may touch, and a whole-graph clone
    // (O(V+E)) otherwise; `dir_graph::rollback` owns that decision and
    // documents it. Either way it MUST be closed on every exit path —
    // `commit` is what uninstalls the capture journal.
    let checkpoint = if is_mutation && !can_skip_rollback_checkpoint(graph, &parsed, opts) {
        StatementCheckpoint::open(graph)
    } else {
        StatementCheckpoint::None
    };

    if is_mutation {
        if let Err(error) = graph.prepare_disk_mutation() {
            checkpoint.rollback(graph);
            return Err(KgError::FileIo(error));
        }
    }

    let mut result = if is_mutation {
        let interrupt = crate::graph::algorithms::Interrupt {
            deadline: opts.deadline,
            cancel: opts.cancel,
        };
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
                opts.max_rows,
                &opts.csv_import,
            )
        });
        graph.active_write_scope = None;
        let r = match r {
            Ok(result) => result,
            Err(message) => {
                // Recover the structured violation *before* rolling back, so
                // the typed error never depends on what a checkpoint restore
                // does to the graph's transient fields.
                let error = mutation_err(graph, opts, message);
                checkpoint.rollback(graph);
                return Err(error);
            }
        };
        checkpoint.commit(graph);
        // A Cypher write occurred — advance the graph version so any
        // version-keyed caches (the plan cache) and OCC see the change.
        // Bumps the working copy directly so a read-after-write *within* the
        // same transaction re-plans against the mutated state; the eventual
        // commit recomputes the live version independently (see Session::commit).
        graph.bump_version();
        // Re-enforce `set_memory_limit` over what the statement just wrote.
        // A write that *creates* a column — a SET for a property the type has
        // never carried — puts O(rows) of fresh heap behind a limit that was
        // last checked at consolidation time, and before this nothing looked
        // again: the only bound a caller can place on the columnar heap could
        // be escaped by writing one new property, permanently. A no-op (one
        // `Option` test) when no limit is set, and an O(columns) heap sum when
        // one is; the materialisation only runs when the sum is over.
        graph.maybe_spill_columns();
        r
    } else {
        cypher::CypherExecutor::with_params(graph, &params, opts.deadline)
            .with_max_rows(opts.max_rows)
            .with_streaming(opts.lazy_eligible)
            .with_parallel(opts.parallel)
            .with_cancel(opts.cancel)
            .execute(&parsed)
            .map_err(|message| exec_err(opts, message))?
    };
    // Encode codec'd-property result columns (e.g. `CREATE (...) RETURN n.id`
    // reads back `'Q42'`). Eager path only; see execute_read.
    cypher::value_codec::apply_encode(&mut result, &encode_plan);
    attach_diagnostics(&mut result, &warnings, started, opts);

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

/// Shared preparation: parse → validate → rewrite_text_score → embed
/// (if needed) → optimize → optional mark_lazy. Returns the
/// parsed+optimized AST + the (possibly-augmented-with-embeddings)
/// param map.
///
/// The params map is borrowed from `opts.params` in the common case
/// (no text_score). When text_score() is present, we clone-on-write
/// to inject the embedding result vectors into the map — the
/// returned `HashMap<String, Value>` is owned in that case.
///
/// **GIL note for binding implementers.** If `opts.embedder` is a
/// Python-backed embedder (PyEmbedderAdapter), the binding MUST
/// release the GIL before calling `execute_read`/`execute_mut`
/// (Python's `py.detach`). The embed call inside this fn will then
/// re-acquire the GIL briefly to invoke Python; if you forget to
/// release first, it deadlocks.
/// Output of [`prepare`]: the parsed+optimized query, the (possibly
/// embedding-augmented) param map, the column-indexed value-codec encode plan
/// (empty when no codecs apply), and the non-fatal schema warnings this
/// statement earned.
struct PreparedQuery {
    plan: Arc<CypherQuery>,
    params: HashMap<String, Value>,
    encode_plan: Vec<Option<ValueCodec>>,
    /// Unknown-label / unknown-relationship-type / absent-property warnings,
    /// computed **once** here and then used twice: emitted to stderr for
    /// interactive users, and attached to `QueryDiagnostics.warnings` for every
    /// programmatic surface. Behind an `Arc` because the plan cache stores them
    /// alongside the plan — see the note at the cache lookup below.
    warnings: Arc<[String]>,
}

// KgError carries query context; boxing it would only burden an error path.
#[allow(clippy::result_large_err)]
fn prepare(
    graph: &DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
) -> Result<PreparedQuery, KgError> {
    // Open a plan-cache attribution window for this statement. Test-only; the
    // caller closes it with `classify_pending` once it knows `is_mutation`.
    #[cfg(test)]
    cypher::plan_cache::instrumentation::begin_prepare();
    // Plan cache: a param-less, codec-free, no-disabled-passes query against an
    // unchanged graph reuses its fully-optimized plan, skipping parse + validate
    // + optimize. Keyed on (graph_id, version) so any mutation invalidates it
    // and it never leaks across graphs (see `cypher::plan_cache`). Lazy-marking
    // is applied fresh per call since it depends on `opts.lazy_eligible`.
    let cacheable = opts.params.is_empty()
        && opts.disabled_passes.is_none_or(|s| s.is_empty())
        && opts.value_codecs.is_none_or(|c| c.is_empty());
    if cacheable {
        if let Some(cached) =
            cypher::plan_cache::get(graph.graph_id(), graph.version(), opts.lazy_eligible, query)
        {
            // Stored post lazy-marking for this `lazy_eligible` — a hit is a
            // pure Arc clone, no parse / validate / optimize / mutation.
            //
            // A hit also skips `dynamic_labels::resolve`, which is where an
            // unbound `$parameter` — a label position or an inline
            // property-map value — is rejected. That is sound rather than a
            // hole: an entry only exists because some earlier call reached
            // the end of this function with `params` empty, and that call ran
            // the pass against the same empty map, which binds nothing. So a
            // cached plan provably contains no parameter reference at all, and
            // a hit is only consulted when `params` is empty. A statement that
            // *does* reference one errors above the insert and leaves nothing
            // behind, so its second call misses and raises again. Pinned by
            // `session::param_presence_tests`.
            //
            // The warnings come out of the entry rather than being recomputed:
            // recomputing needs the parsed AST, and not skipping the parse is
            // exactly what this early return exists to avoid. Their validity
            // is the cache's own soundness argument — they are a pure function
            // of `(query, graph schema)` and the key pins the graph state.
            // Stderr repeats them per call, as it did when every call parsed.
            cypher::emit_query_warnings(&cached.warnings);
            return Ok(PreparedQuery {
                plan: cached.plan,
                params: HashMap::new(),
                encode_plan: Vec::new(),
                warnings: cached.warnings,
            });
        }
    }

    let mut parsed = cypher::parse_cypher(query)?;

    // Dynamic labels / relationship types (`MATCH (n:$label)`): bind them from
    // the caller's parameters FIRST, so validation, optimization and execution
    // all see an ordinary literal label. The parser cannot do this — parsed
    // ASTs are cached by query text and re-run with different parameters — so
    // it leaves a marker for this pass. The same pass rejects an inline
    // property map whose value names a parameter the caller did not bind
    // (`MATCH (v {flag: $flag})`), which the matcher's `bool`-returning filter
    // could only answer as "no match". See `cypher::dynamic_labels`.
    cypher::dynamic_labels::resolve(&mut parsed, opts.params)?;

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

    // Schema validation — property typos in pattern literals
    // (`{ttle: 'Alice'}`) get caught with a "did you mean?" hint.
    cypher::validate_schema(&parsed, graph).map_err(KgError::from)?;

    // Non-fatal: a MATCH that references an unknown node label or relationship
    // type — the most common "why is my query empty?" typo. Computed once and
    // consumed twice: stderr now (interactive users), and
    // `QueryDiagnostics.warnings` at the end of execute (every programmatic
    // surface, including the MCP server, which is where an agent reads them).
    let collected = cypher::collect_query_warnings(&parsed, graph);
    // ...with one exception, and it is a *disposition* change, not a second
    // walk: under `lock_schema()` the absent-property subset becomes fatal.
    // `MATCH (p:Person) WHERE p.agee = 1` returning `[]` and `RETURN p.agee`
    // returning a null column are the read-side twins of the pattern-literal
    // typo `validate_schema` already rejects above, and a lock exists to catch
    // exactly that. Reversed arrows and unknown labels/rel-types stay warnings:
    // the first is heuristic, and the second is legal zero-row Cypher whose
    // locked-schema *label* case is already fatal via `validate_label`, so
    // promoting here would double-report it.
    let strict_absent = !collected.absent_property.is_empty();
    if graph.schema_locked {
        if let Some(error) = cypher::strict_read_error(&collected.absent_property, graph) {
            return Err(KgError::from(error));
        }
    }
    let warnings: Arc<[String]> = collected.into_messages().into();
    cypher::emit_query_warnings(&warnings);

    // text_score() rewrite. Scans for `text_score(...)` calls in the
    // AST and rewrites them to `vector_score(...)`, collecting the
    // texts to embed alongside.
    let rewrite = cypher::rewrite_text_score(&mut parsed, opts.params).map_err(|message| {
        KgError::CypherExecution {
            message,
            position: None,
        }
    })?;

    // If text_score(...) was used (and we're NOT in EXPLAIN mode —
    // EXPLAIN renders plan rows without executing, so no embedding
    // needed), run the embedder and inject the result vectors into
    // the param map. Otherwise pass the caller's params through.
    let params: Cow<'_, HashMap<String, Value>> =
        if !rewrite.texts_to_embed.is_empty() && !parsed.explain {
            Cow::Owned(embed_into_params(opts, &rewrite)?)
        } else {
            Cow::Borrowed(opts.params)
        };

    // Optimize. Empty disabled-set is the common case; avoid the
    // HashSet allocation when no passes are disabled.
    let disabled_default = cypher::planner::empty_disabled_set();
    let disabled_ref = opts.disabled_passes.unwrap_or(disabled_default);
    cypher::planner::optimize_with_disabled(&mut parsed, graph, &params, disabled_ref);

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
    // Two shapes did reuse a cached mutation plan and deliberately no longer
    // do: transactions forked from one base version (same `graph_id` +
    // `version`), and a retry of a mutation that errored before
    // `bump_version`. Both are same-version replays — a narrow window traded
    // for a per-write cost every serial writer pays. See
    // `session::plan_cache_cost_tests`, which pins both directions.
    //
    // The **lookup** above deliberately stays. `prepare` runs before anything
    // has parsed the query, so this classification does not exist yet there,
    // and buying it early means a `parse_cypher_cached` AST clone (~700 ns per
    // its own module docs) on the read-hit path that the plan cache exists to
    // keep at ~1.9 us. With no mutation ever inserted, a mutation's lookup is
    // a guaranteed miss: one shared read lock and one hash, and nothing more.
    // ...and never for a statement carrying absent-property findings, which is
    // what makes the strict-read promotion above independent of the cache. A
    // hit returns before the schema pass runs, so it cannot re-decide anything;
    // by refusing to store such a plan, a hit *proves* there was nothing to
    // promote, and "prime unlocked → lock_schema() → rerun the same text"
    // raises whether or not locking bumped the graph version (through the
    // Python/`api` surface it does — `make_dir_graph_mut`; a core caller
    // flipping the flag on an owned `DirGraph` does not). The cost is confined
    // to statements the engine has already diagnosed as reading an all-null
    // column, which no hot loop should contain.
    if cacheable && params.is_empty() && !strict_absent && !cypher::is_mutation_query(&plan) {
        cypher::plan_cache::insert(
            graph.graph_id(),
            graph.version(),
            opts.lazy_eligible,
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

/// Attach `QueryDiagnostics` to a finished result — the single place every
/// execution path (read, mutation, EXPLAIN) leaves them.
///
/// `prepare_warnings` are the schema warnings from [`prepare`]; the executor
/// may already have parked runtime ones (procedure scoping) on the result's
/// diagnostics, and those keep their place after the schema ones, which
/// explain an empty result before any runtime advisory does.
///
/// `timeout_ms` is derived from the deadline that was actually in force. A
/// binding that knows the configured figure (the wheel reports the caller's
/// `timeout_ms`, including its `Some(0)` = "disabled" escape hatch) overwrites
/// it; core can only see the instant.
fn attach_diagnostics(
    result: &mut CypherResult,
    prepare_warnings: &[String],
    started: Instant,
    opts: &ExecuteOptions<'_>,
) {
    let mut diagnostics = result.diagnostics.take().unwrap_or_default();
    if !prepare_warnings.is_empty() {
        let mut merged = prepare_warnings.to_vec();
        merged.append(&mut diagnostics.warnings);
        diagnostics.warnings = merged;
    }
    diagnostics.elapsed_ms = started.elapsed().as_millis() as u64;
    diagnostics.timeout_ms = opts
        .deadline
        .map(|deadline| deadline.saturating_duration_since(started).as_millis() as u64);
    result.diagnostics = Some(diagnostics);
}

/// Run the embedder on collected texts; inject the JSON-encoded
/// vectors into a clone of the param map. Caller-supplied params
/// are not mutated. Returns the augmented map.
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
            message: "text_score() requires a registered embedding model. \
                      Call g.set_embedder(model) first (Python) or pass an embedder \
                      via ExecuteOptions::embedder (downstream Rust consumers)."
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
        static CANCEL: AtomicBool = AtomicBool::new(false);

        let mut g = DirGraph::new();
        let params = HashMap::new();
        let base_opts = ExecuteOptions::eager(&params);
        execute_mut(&mut g, "CREATE (:Item {id: 1})", &base_opts).unwrap();

        CANCEL.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut cancelled_opts = ExecuteOptions::eager(&params);
        cancelled_opts.cancel = Some(&CANCEL);
        assert!(matches!(
            execute_mut(&mut g, "MATCH (n:Item) DELETE n", &cancelled_opts),
            Err(KgError::Cancelled)
        ));
        assert_eq!(g.graph.node_count(), 1);
        CANCEL.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}
