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
use std::sync::Arc;
use std::time::Instant;

use crate::datatypes::Value;
use crate::error::KgError;
use crate::graph::dir_graph::DirGraph;
use crate::graph::embedder::Embedder;
use crate::graph::languages::cypher;
use crate::graph::languages::cypher::ast::{CypherQuery, OutputFormat};
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
    let (parsed, params, encode_plan) = prepare(graph, query, opts)?;
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

    let mut result = cypher::CypherExecutor::with_params(graph, &params, opts.deadline)
        .with_max_rows(opts.max_rows)
        .with_streaming(opts.lazy_eligible)
        .execute(&parsed)
        .map_err(|message| KgError::CypherExecution {
            message,
            position: None,
        })?;
    // value_codecs: encode codec'd-property result columns back to the typed
    // form (`42` → `'Q42'`). Applies to eager rows; lazy results (Python's
    // streaming path) materialize later and aren't covered — the configured
    // consumer (mcp-server) runs eager.
    cypher::value_codec::apply_encode(&mut result, &encode_plan);

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
    let (parsed, params, encode_plan) = prepare(graph, query, opts)?;
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

    let mut result = if is_mutation {
        let r =
            cypher::execute_mutable(graph, &parsed, params, opts.deadline).map_err(|message| {
                KgError::CypherExecution {
                    message,
                    position: None,
                }
            })?;
        // A Cypher write occurred — advance the graph version so any
        // version-keyed caches (the plan cache) and OCC see the change.
        // Bumps the working copy directly so a read-after-write *within* the
        // same transaction re-plans against the mutated state; the eventual
        // commit recomputes the live version independently (see Session::commit).
        graph.bump_version();
        r
    } else {
        cypher::CypherExecutor::with_params(graph, &params, opts.deadline)
            .with_max_rows(opts.max_rows)
            .with_streaming(opts.lazy_eligible)
            .execute(&parsed)
            .map_err(|message| KgError::CypherExecution {
                message,
                position: None,
            })?
    };
    // Encode codec'd-property result columns (e.g. `CREATE (...) RETURN n.id`
    // reads back `'Q42'`). Eager path only; see execute_read.
    cypher::value_codec::apply_encode(&mut result, &encode_plan);

    Ok(ExecuteOutcome {
        result,
        is_mutation,
        output_format: parsed.output_format,
        explain: false,
    })
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
/// embedding-augmented) param map, and the column-indexed value-codec encode
/// plan (empty when no codecs apply).
type PreparedQuery = (
    Arc<CypherQuery>,
    HashMap<String, Value>,
    Vec<Option<ValueCodec>>,
);

fn prepare(
    graph: &DirGraph,
    query: &str,
    opts: &ExecuteOptions<'_>,
) -> Result<PreparedQuery, KgError> {
    // Plan cache: a param-less, codec-free, no-disabled-passes query against an
    // unchanged graph reuses its fully-optimized plan, skipping parse + validate
    // + optimize. Keyed on (graph_id, version) so any mutation invalidates it
    // and it never leaks across graphs (see `cypher::plan_cache`). Lazy-marking
    // is applied fresh per call since it depends on `opts.lazy_eligible`.
    let cacheable = opts.params.is_empty()
        && opts.disabled_passes.is_none_or(|s| s.is_empty())
        && opts.value_codecs.is_none_or(|c| c.is_empty());
    if cacheable {
        if let Some(plan) =
            cypher::plan_cache::get(graph.graph_id(), graph.version(), opts.lazy_eligible, query)
        {
            // Stored post lazy-marking for this `lazy_eligible` — a hit is a
            // pure Arc clone, no parse / validate / optimize / mutation.
            return Ok((plan, HashMap::new(), Vec::new()));
        }
    }

    let mut parsed = cypher::parse_cypher(query)?;

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

    // Non-fatal: warn (stderr) when a MATCH references an unknown node label
    // or relationship type — the most common "why is my query empty?" typo.
    cypher::warn_unknown_pattern_refs(&parsed, graph);

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
    if cacheable && params.is_empty() {
        cypher::plan_cache::insert(
            graph.graph_id(),
            graph.version(),
            opts.lazy_eligible,
            query,
            plan.clone(),
        );
    }

    Ok((plan, params.into_owned(), encode_plan))
}

/// Run the embedder on collected texts; inject the JSON-encoded
/// vectors into a clone of the param map. Caller-supplied params
/// are not mutated. Returns the augmented map.
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
        let json = format!(
            "[{}]",
            embeddings[i]
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        params.insert(param_name.clone(), Value::String(json));
    }
    Ok(params)
}

#[cfg(test)]
mod version_soundness_tests {
    use super::*;
    use crate::graph::dir_graph::DirGraph;

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
}
