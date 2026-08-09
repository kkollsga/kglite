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
    pub fn run_cypher_template(
        &self,
        template: &str,
        args: &serde_json::Map<String, serde_json::Value>,
        csv_http: Option<&crate::csv_http::CsvHttpConfig>,
    ) -> String {
        let mut params = HashMap::new();
        for (k, v) in args {
            params.insert(k.clone(), json_to_value(v));
        }
        match self.execute_cypher_read(template, params) {
            Ok(outcome) => render_cypher_output(
                &outcome.result,
                outcome.output_format == cypher::OutputFormat::Csv,
                csv_http,
            )
            .unwrap_or_else(|error| cypher_tool_error(&error)),
            Err(error) => legacy_cypher_error(&error),
        }
    }

    /// Execute read-only Cypher and preserve its structured outcome.
    ///
    /// The lazy rebuild completes before the active-graph guard is acquired;
    /// that guard then remains held for the complete eager execution. This is
    /// the shared entry point for automation-safe structured MCP routes. It
    /// intentionally performs no text rendering or result postprocessing.
    #[allow(clippy::result_large_err)]
    pub(crate) fn execute_cypher_read(
        &self,
        query: &str,
        params: HashMap<String, Value>,
    ) -> std::result::Result<ExecuteOutcome, CypherRunError> {
        self.ensure_workspace_graph_fresh();
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
    #[allow(clippy::result_large_err)]
    pub(crate) fn execute_cypher_read_strict(
        &self,
        query: &str,
        params: HashMap<String, Value>,
    ) -> std::result::Result<ExecuteOutcome, StrictCypherReadError> {
        self.ensure_workspace_graph_fresh();
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

/// Execute read-only Cypher without choosing a presentation format.
///
/// This is the canonical MCP execution seam: policy, eager materialization,
/// embedder wiring, and value codecs are applied once, while callers retain
/// [`ExecuteOutcome`] for structured serialization or legacy rendering.
#[allow(clippy::result_large_err)]
pub(crate) fn execute_cypher_inner(
    kg: &KnowledgeGraph,
    query: &str,
    params: HashMap<String, Value>,
    value_codecs: Option<&[ValueCodec]>,
) -> std::result::Result<ExecuteOutcome, CypherRunError> {
    // MCP rejects mutations regardless of read-only graph mode. Pre-parse so
    // the policy failure remains distinct from an engine execution failure.
    let (_, is_mutation) =
        kglite::api::cypher::parse_with_mutation_check(query).map_err(CypherRunError::Engine)?;
    if is_mutation {
        return Err(CypherRunError::MutationNotAllowed);
    }

    // Eager rows are required by both the legacy formatters and structured
    // routes. The embedder and codecs match the pre-extraction execution path.
    let mut opts = kglite::api::session::ExecuteOptions::eager(&params);
    opts.embedder = kg.embedder().cloned();
    opts.value_codecs = value_codecs;
    kglite::api::session::execute_read(kg.dir(), query, &opts).map_err(CypherRunError::Engine)
}

/// Run a Cypher query against the given KnowledgeGraph snapshot. Picks
/// between read and write paths based on `is_mutation_query`; on success
/// returns the rendered tool body (CSV when `FORMAT CSV` is in the
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

/// Render a `CypherResult` for the MCP text surface: CSV (inline or via the
/// csv_http server) or a 15-row inline preview. Shared by the read path and
/// the write path so both format results identically.
pub(crate) fn render_cypher_output(
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
                    Ok(csv)
                }
            }
        } else {
            Ok(csv)
        }
    } else {
        Ok(format_cypher_inline(result))
    }
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
