//! The server's canonical Cypher seam: [`GraphState`]'s read/template entry
//! points and the shared execution + rendering helpers behind them.

use std::collections::HashMap;

use anyhow::Result;
use kglite::api::cypher;
use kglite::api::cypher::ValueCodec;
use kglite::api::session::ExecuteOutcome;
use kglite::api::{KnowledgeGraph, Value};

use crate::tools::*;

impl GraphState {
    /// Run a parameterised Cypher template against the active graph.
    /// Used by the YAML-declared `tools[].cypher` registration path
    /// (see [`crate::cypher_tools::register_cypher_tools`]).
    ///
    /// `Err` carries the agent-facing failure text — the same bytes this
    /// returned inside `Ok` before the fallible seam landed, now separated so
    /// the route can put it in an MCP error envelope (`isError: true`) instead
    /// of an answer a programmatic client cannot tell from one.
    pub fn run_cypher_template(
        &self,
        template: &str,
        args: &serde_json::Map<String, serde_json::Value>,
        csv_http: Option<&crate::csv_http::CsvHttpConfig>,
    ) -> Result<String, String> {
        match self.execute_cypher_read(template, params_from_json(Some(args))) {
            Ok(outcome) => render_cypher_output(
                &outcome.result,
                outcome.output_format == cypher::OutputFormat::Csv,
                csv_http,
            )
            .map_err(|error| cypher_tool_error(&error)),
            Err(error) => Err(legacy_cypher_error(&error)),
        }
    }

    /// Execute read-only Cypher and preserve its structured outcome.
    ///
    /// The lazy rebuild completes before the active-graph guard is acquired;
    /// that guard then remains held for the complete eager execution. This is
    /// the shared entry point for automation-safe structured MCP routes. It
    /// intentionally performs no text rendering or result postprocessing.
    pub(crate) fn execute_cypher_read(
        &self,
        query: &str,
        params: HashMap<String, Value>,
    ) -> std::result::Result<ExecuteOutcome, CypherRunError> {
        self.ensure_graph_fresh();
        let guard = read_lock(&self.inner);
        let active = guard.as_ref().ok_or(CypherRunError::NoActiveGraph)?;
        execute_cypher_inner(&active.kg, query, params, self.value_codecs())
    }

    /// Execute read-only Cypher only when the current workspace graph is known
    /// fresh.
    ///
    /// Freshness is ensured exactly once. A remaining typed rebuild failure is
    /// returned before the active graph is borrowed or the query is parsed, so
    /// stale data can never escape through a structured evidence route. The
    /// query then executes directly against the installed generation while its
    /// read guard is held; calling [`Self::execute_cypher_read`] here would
    /// incorrectly run freshness handling a second time.
    pub(crate) fn execute_cypher_read_strict(
        &self,
        query: &str,
        params: HashMap<String, Value>,
    ) -> std::result::Result<ExecuteOutcome, StrictCypherReadError> {
        self.ensure_graph_fresh();
        if let Some(failure) = self.workspace_rebuild_failure() {
            return Err(StrictCypherReadError::StaleGraph(failure));
        }
        let guard = read_lock(&self.inner);
        let active = guard
            .as_ref()
            .ok_or(StrictCypherReadError::Cypher(CypherRunError::NoActiveGraph))?;
        execute_cypher_inner(&active.kg, query, params, self.value_codecs())
            .map_err(StrictCypherReadError::Cypher)
    }
}

/// Convert a `serde_json::Value` into a Cypher param `Value`. Mirrors
/// the Python boundary's `py_value_to_value` for the JSON subset.
///
/// As of the 2026-05-25 binding-framework lift this is a 1-line
/// delegate to `kglite::api::param::json_value_to_kglite_value`,
/// which any REST/gRPC binding can call directly.
pub(crate) fn json_to_value(v: &serde_json::Value) -> Value {
    kglite::api::param::json_value_to_kglite_value(v)
}

/// Build the engine's parameter map from a tool call's JSON object.
///
/// One conversion for both parameter sources: the manifest template route,
/// which has bound `$name` placeholders since it shipped, and the
/// `cypher_query` tools' `params` argument, which until now had none — so
/// `$param` was unbindable over MCP by construction while `describe()`'s own
/// examples taught the inline-map form that needs it. `None` is the
/// no-parameters call and yields an empty map.
pub(crate) fn params_from_json(
    args: Option<&serde_json::Map<String, serde_json::Value>>,
) -> HashMap<String, Value> {
    args.map(|args| {
        args.iter()
            .map(|(k, v)| (k.clone(), json_to_value(v)))
            .collect()
    })
    .unwrap_or_default()
}

/// Execute read-only Cypher without choosing a presentation format.
///
/// This is the canonical MCP execution seam: policy, eager materialization,
/// embedder wiring, and value codecs are applied once, while callers retain
/// [`ExecuteOutcome`] for structured serialization or legacy rendering.
pub(crate) fn execute_cypher_inner(
    kg: &KnowledgeGraph,
    query: &str,
    params: HashMap<String, Value>,
    value_codecs: Option<&[ValueCodec]>,
) -> std::result::Result<ExecuteOutcome, CypherRunError> {
    // MCP rejects mutations regardless of read-only graph mode. Pre-parse so
    // the policy failure remains distinct from an engine execution failure.
    let (_, is_mutation) =
        kglite::api::cypher::parse_with_mutation_check(query).map_err(CypherRunError::engine)?;
    if is_mutation {
        return Err(CypherRunError::MutationNotAllowed);
    }

    // Eager rows are required by both the legacy formatters and structured
    // routes. The embedder and codecs match the pre-extraction execution path.
    let mut opts = kglite::api::session::ExecuteOptions::eager(&params);
    opts.embedder = kg.embedder().cloned();
    opts.value_codecs = value_codecs;
    kglite::api::session::execute_read(kg.dir(), query, &opts).map_err(CypherRunError::engine)
}

/// Run a Cypher query against the given KnowledgeGraph snapshot. Picks
/// between read and write paths based on `is_mutation_query`; on success
/// returns the rendered tool body (capped CSV when `FORMAT CSV` is in the
/// query, inline 15-row preview otherwise).
pub(crate) fn run_cypher_inner(
    kg: &KnowledgeGraph,
    query: &str,
    params: HashMap<String, Value>,
    value_codecs: Option<&[ValueCodec]>,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> std::result::Result<String, String> {
    let outcome =
        execute_cypher_inner(kg, query, params, value_codecs).map_err(|error| error.to_string())?;
    render_cypher_output(
        &outcome.result,
        outcome.output_format == cypher::OutputFormat::Csv,
        csv_http,
    )
}

/// The engine's non-fatal query warnings, rendered as a trailing block, or
/// the empty string for a clean query.
///
/// This is the whole point of D1p: the engine has always known that
/// `MATCH (v:vessel)` on a graph of `Vessel`s returns nothing because of a
/// typo, and said so — on **stderr**, which no MCP client ever sees. An agent
/// got a confident "No results." and, measured against a 15-query
/// self-correction harness, scored zero on every silently-wrong query. The
/// block is appended by [`render_cypher_output`], the one seam every Cypher
/// tool response (direct tool, manifest template, write ack) passes through.
pub(crate) fn cypher_warning_block(result: &cypher::CypherResult) -> String {
    let warnings = match result.diagnostics.as_ref() {
        Some(diagnostics) if !diagnostics.warnings.is_empty() => &diagnostics.warnings,
        _ => return String::new(),
    };
    let mut out = String::from("\nwarnings:\n");
    for warning in warnings {
        out.push_str("  - ");
        out.push_str(warning);
        out.push('\n');
    }
    out
}

/// Render a `CypherResult` for the MCP text surface: CSV (via the csv_http
/// server, or inline capped at [`INLINE_CSV_ROW_LIMIT`] rows) or a 15-row
/// inline preview, followed by the engine's warning block when the query
/// earned one. Shared by the read path and the write path so both format
/// results identically.
pub(crate) fn render_cypher_output(
    result: &cypher::CypherResult,
    output_csv: bool,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> Result<String, String> {
    render_cypher_body(result, output_csv, csv_http)
        .map(|body| format!("{body}{}", cypher_warning_block(result)))
}

fn render_cypher_body(
    result: &cypher::CypherResult,
    output_csv: bool,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> Result<String, String> {
    if output_csv {
        let csv = result.to_csv();
        if let Some(cfg) = csv_http {
            match crate::csv_http::write_csv(cfg, &csv) {
                Ok(name) => {
                    let url = cfg.url_for(&name);
                    // 0.9.19 fix: count rows from the CSV body, not from
                    // `result.rows.len()`. The planner's lazy_eligible
                    // pass leaves `rows` empty for simple
                    // MATCH-RETURN-LIMIT queries and materialises through
                    // the lazy descriptor (or streaming pipeline) — the
                    // CSV is correct but `rows.len()` reads 0 and the
                    // operator-facing status says "0 row(s) written".
                    // Counting newlines in the CSV agrees with what the
                    // file actually contains.
                    let row_count = count_csv_rows(&csv);
                    Ok(format!(
                        "FORMAT CSV: {row_count} row(s) written to {url}\n\
                         Fetch with: curl {url}"
                    ))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "csv_http write_csv failed; falling back to inline");
                    Ok(cap_inline_csv(&csv))
                }
            }
        } else {
            Ok(cap_inline_csv(&csv))
        }
    } else {
        Ok(format_cypher_inline(result))
    }
}

/// Maximum data rows an inline `FORMAT CSV` body may carry over MCP.
///
/// Deliberately the same number as the structured recipe route's cap, and
/// written as an alias of it so the two can never drift: an agent that learns
/// "200 rows is what one MCP call returns" from one route must not be taught
/// a different number by the other.
pub(crate) const INLINE_CSV_ROW_LIMIT: usize = crate::recipe_queries::RECIPE_RESULT_ROW_LIMIT;

/// Trim an inline CSV body to [`INLINE_CSV_ROW_LIMIT`] data rows, appending a
/// notice that names the true row count, the full byte size, and the escape
/// hatch that returns the complete file.
///
/// The uncapped path was the single largest response this server could
/// produce: an external eval measured 283,686 characters (~71k tokens) from
/// one `FORMAT CSV` call on a 5,420-node graph, on a tool whose own
/// description recommended `FORMAT CSV` for large results. The inline
/// 15-row preview has always been capped; the CSV branch never was, so the
/// budget-safe formatting an agent thought it was choosing did the opposite.
///
/// The notice, not a silent trim, is the point: an agent that cannot see the
/// total re-runs the same query hoping for more, and one that cannot see the
/// escape hatch has no way to obtain the rest. `csv_http_server` stays
/// opt-in — it binds a port and writes files, which no query should be able
/// to turn on — so the notice names it as an operator action.
pub(crate) fn cap_inline_csv(csv: &str) -> String {
    let total_rows = count_csv_rows(csv);
    if total_rows <= INLINE_CSV_ROW_LIMIT {
        return csv.to_string();
    }
    // Header plus the first N data rows, re-joined with the newline
    // terminator `lines()` strips.
    let mut out = String::with_capacity(csv.len().min(64 * 1024));
    for line in csv.lines().take(INLINE_CSV_ROW_LIMIT + 1) {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&format!(
        "\nFORMAT CSV truncated: showing the first {INLINE_CSV_ROW_LIMIT} of {total_rows} row(s) \
         ({} bytes in full). Narrow the query (WHERE / LIMIT / SKIP / aggregate) to fit, or ask \
         the operator to enable extensions.csv_http_server in the server manifest, which returns \
         the complete CSV as a fetch URL instead of inline text.\n",
        csv.len()
    ));
    out
}

/// Render a CypherResult as an inline 15-row preview (header + repr per
/// row). Matches the format the pre-0.9.18 Python shim produced via
/// `format_cypher_result`.
pub(crate) fn format_cypher_inline(result: &cypher::CypherResult) -> String {
    let len = result.rows.len();
    if len == 0 {
        return "No results.".to_string();
    }
    let header = if len > 15 {
        format!("{len} row(s) (showing first 15):\n")
    } else {
        format!("{len} row(s):\n")
    };
    let mut out = header;
    out.push_str(&result.columns.join("\t"));
    out.push('\n');
    for row in result.rows.iter().take(15) {
        for (i, val) in row.iter().enumerate() {
            if i > 0 {
                out.push('\t');
            }
            push_value_repr(&mut out, val);
        }
        out.push('\n');
    }
    out
}

/// Count data rows in a CSV string, defined as (newline-terminated lines) - 1
/// for the header. Trailing newlines after the last row don't add to the
/// count. Handles the edge cases: empty string → 0, header-only → 0,
/// header + N rows → N. Quoted newlines inside cells aren't recognised
/// here — kglite's `csv_value` doesn't emit Value variants that contain
/// embedded newlines, so a plain `lines()` count agrees with row count.
pub(crate) fn count_csv_rows(csv: &str) -> usize {
    let line_count = csv.lines().count();
    line_count.saturating_sub(1)
}

pub(crate) fn push_value_repr(out: &mut String, val: &Value) {
    use std::fmt::Write;
    match val {
        Value::Null => out.push_str("null"),
        Value::String(s) => {
            let _ = write!(out, "{s:?}");
        }
        Value::Int64(n) => {
            let _ = write!(out, "{n}");
        }
        Value::Float64(f) => {
            let _ = write!(out, "{f}");
        }
        Value::Boolean(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::UniqueId(u) => {
            let _ = write!(out, "{u}");
        }
        Value::DateTime(d) => out.push_str(&d.format("%Y-%m-%d").to_string()),
        Value::Timestamp(dt) => out.push_str(&dt.format("%Y-%m-%dT%H:%M:%S").to_string()),
        Value::Point { lat, lon } => {
            let _ = write!(out, "POINT({lon} {lat})");
        }
        Value::Duration {
            months,
            days,
            seconds,
        } => {
            let _ = write!(out, "duration(M={months}, D={days}, S={seconds})");
        }
        Value::NodeRef(idx) => {
            let _ = write!(out, "node[{idx}]");
        }
        // Phase A.1 / C5 — collection / graph-entity variants. Render
        // as compact JSON for the MCP text surface; the structured
        // form is already what agents consume via `to_dicts()` /
        // `to_list()`. Falls back to `?` on serialisation failure
        // (shouldn't happen — these all derive Serialize).
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => {
            let _ = write!(
                out,
                "{}",
                serde_json::to_string(val).unwrap_or_else(|_| "?".to_string())
            );
        }
    }
}
