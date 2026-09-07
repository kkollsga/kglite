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
        csv_http: &crate::csv_http::CsvHttpState,
    ) -> Result<String, String> {
        match self.execute_cypher_read(template, params_from_json(Some(args))?) {
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
        execute_cypher_inner(&active.kg, query, params, self.exec_policy())
    }

    /// Execute read-only Cypher only when the current workspace graph is known
    /// fresh.
    ///
    /// Freshness is ensured exactly once. A remaining typed rebuild failure is
    /// returned before the active graph is borrowed or the query is parsed, so
    /// stale data can never escape through a structured evidence route. The
    /// query then executes directly against the installed graph while its
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
        execute_cypher_inner(&active.kg, query, params, self.exec_policy())
            .map_err(StrictCypherReadError::Cypher)
    }
}

/// Build the engine's parameter map from a tool call's JSON object.
///
/// Query admission rejects integer tokens outside `i64` and explicit numeric
/// tokens outside finite `f64`, preserving the nested JSON path in the error.
pub(crate) fn params_from_json(
    args: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<HashMap<String, Value>, String> {
    args.map(kglite::api::param::json_object_to_query_value_map)
        .transpose()
        .map(|params| params.unwrap_or_default())
        .map_err(|error| error.to_string())
}

/// The boot-decided engine settings a Cypher route applies to every
/// execution, travelling as one value.
///
/// One struct rather than a parameter per setting: these are all read off the
/// same [`GraphState`] at the same moment, and a route that threads them
/// individually can silently drop one — which is indistinguishable, from the
/// outside, from the knob having no effect. That is also why the per-call
/// deadline lives here rather than as a seventh positional argument.
#[derive(Clone, Copy, Default)]
pub(crate) struct ExecPolicy<'a> {
    /// Manifest-declared literal codecs (`extensions.value_codecs`).
    pub(crate) value_codecs: Option<&'a [ValueCodec]>,
    /// The operator's parallel-runtime opt-in. Honoured on reads only — see
    /// [`execute_cypher_inner`] and `run_cypher_write`.
    pub(crate) parallel: bool,
    /// This call's deadline, in milliseconds. `None` takes the shared
    /// [`kglite::api::session::DEFAULT_TIMEOUT_MS`]; `Some(0)` disables the
    /// deadline. Boot supplies `None`; the `cypher_query` tool's `timeout_ms`
    /// argument overrides it per call.
    ///
    /// This server *adopts* the default (the Python surface's, one constant
    /// shared through the core) rather than running unbounded, because an
    /// agent has no cancel channel and a runaway read holds the active graph's
    /// read lock — which stalls `ensure_reloaded_graph_fresh`'s single-flight
    /// rebuild gate, and with it every later tool call. It is a liveness
    /// property, not a preference.
    pub(crate) timeout_ms: Option<u64>,
}

impl<'a> ExecPolicy<'a> {
    /// This policy with one call's `timeout_ms` laid over it.
    pub(crate) fn with_timeout_ms(self, timeout_ms: Option<u64>) -> Self {
        Self { timeout_ms, ..self }
    }

    /// The instant this execution must stop at.
    pub(crate) fn deadline(&self) -> Option<std::time::Instant> {
        kglite::api::session::QueryDefaults::default()
            .resolve(self.timeout_ms, None, None)
            .deadline
    }
}

/// Execute read-only Cypher without choosing a presentation format.
///
/// This is the canonical MCP execution seam: policy, eager materialization,
/// embedder wiring, and value codecs are applied once, while callers retain
/// [`ExecuteOutcome`] for structured serialization or legacy rendering.
///
/// Every read reaching the engine passes through here — the built-in
/// `cypher_query`, manifest `tools[].cypher` templates, recipe routes, and a
/// domain tool's `run_cypher` — so `policy.parallel` is applied once, here,
/// rather than at each of those four registration sites.
pub(crate) fn execute_cypher_inner(
    kg: &KnowledgeGraph,
    query: &str,
    params: HashMap<String, Value>,
    policy: ExecPolicy<'_>,
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
    opts.value_codecs = policy.value_codecs;
    // A permission, not an instruction: the engine still applies its own
    // per-operator row × cost-class gate, so a small query is unaffected.
    opts.parallel = policy.parallel;
    opts.deadline = policy.deadline();
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
    policy: ExecPolicy<'_>,
    csv_http: &crate::csv_http::CsvHttpState,
) -> std::result::Result<String, String> {
    let outcome =
        execute_cypher_inner(kg, query, params, policy).map_err(|error| error.to_string())?;
    render_cypher_output(
        &outcome.result,
        outcome.output_format == cypher::OutputFormat::Csv,
        csv_http,
    )
}

/// The engine's non-fatal query warnings, rendered as a trailing block, or
/// the empty string for a clean query.
///
/// Warnings must reach the tool response: MCP clients do not see engine stderr.
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

/// Shared tail for row previews, CSV links and mutation acknowledgments.
pub(crate) fn cypher_diagnostics_block(result: &cypher::CypherResult) -> String {
    let mut out = cypher_warning_block(result);
    if let Some(d) = &result.diagnostics {
        if !d.retrieval.is_empty() {
            out.push_str(&format!(
                "\nretrieval: {}\n",
                serde_json::json!(d.retrieval)
            ));
        }
    }
    out
}

/// Render a `CypherResult` for the MCP text surface: CSV (via the csv_http
/// server, or inline capped at [`INLINE_CSV_ROW_LIMIT`] rows) or a 15-row
/// inline preview, followed by execution diagnostics when present. Shared by the read path and the write path so both format
/// results identically.
pub(crate) fn render_cypher_output(
    result: &cypher::CypherResult,
    output_csv: bool,
    csv_http: &crate::csv_http::CsvHttpState,
) -> Result<String, String> {
    render_cypher_body(result, output_csv, csv_http)
        .map(|body| format!("{body}{}", cypher_diagnostics_block(result)))
}

fn render_cypher_body(
    result: &cypher::CypherResult,
    output_csv: bool,
    csv_http: &crate::csv_http::CsvHttpState,
) -> Result<String, String> {
    if output_csv {
        let csv = result.to_csv();
        if let Some(cfg) = csv_http.config() {
            match crate::csv_http::write_csv(cfg, &csv) {
                Ok(name) => {
                    let url = cfg.url_for(&name);
                    // Count rows from the CSV body, not from
                    // `result.rows.len()`. The planner's lazy_eligible
                    // pass leaves `rows` empty for simple
                    // MATCH-RETURN-LIMIT queries and materialises through
                    // the lazy descriptor (or streaming pipeline) — the
                    // CSV is correct but `rows.len()` reads 0 and the
                    // operator-facing status says "0 row(s) written".
                    // Counting logical RFC records in the CSV agrees with
                    // what the file actually contains, including multiline
                    // quoted fields.
                    let row_count = count_csv_rows(&csv);
                    Ok(format!(
                        "FORMAT CSV: {row_count} row(s) written to {url}\n\
                         Fetch with: curl {url}"
                    ))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "csv_http write_csv failed; falling back to inline");
                    Ok(cap_inline_csv(&csv, csv_http))
                }
            }
        } else {
            Ok(cap_inline_csv(&csv, csv_http))
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
pub(crate) fn cap_inline_csv(csv: &str, csv_http: &crate::csv_http::CsvHttpState) -> String {
    let records = csv_record_summary(csv);
    let total_rows = records.data_rows;
    if total_rows <= INLINE_CSV_ROW_LIMIT {
        return match inline_csv_reason(csv_http) {
            // A body that fits carries no truncation notice, so this is the
            // only place a degraded server can tell the agent that the fetch
            // URL its manifest promised is not coming.
            Some(reason) => format!("{csv}\nFORMAT CSV: {reason}\n"),
            None => csv.to_string(),
        };
    }
    // Header plus the first N complete RFC CSV records, byte-for-byte.
    let prefix_end = records.capped_prefix_end;
    let mut out = String::with_capacity(csv.len().min(64 * 1024));
    out.push_str(&csv[..prefix_end]);
    if !out.ends_with('\n') && !out.ends_with('\r') {
        out.push('\n');
    }
    let escape_hatch = match inline_csv_reason(csv_http) {
        Some(reason) => format!(". {reason}"),
        None => ", or ask the operator to enable extensions.csv_http_server in the server \
                 manifest, which returns the complete CSV as a fetch URL instead of inline text."
            .to_string(),
    };
    out.push_str(&format!(
        "\nFORMAT CSV truncated: showing the first {INLINE_CSV_ROW_LIMIT} of {total_rows} row(s) \
         ({} bytes in full). Narrow the query (WHERE / LIMIT / SKIP / aggregate) to fit{escape_hatch}\n",
        csv.len()
    ));
    out
}

/// Why this `FORMAT CSV` answer is inline when the server was configured to
/// answer with a fetch URL. `None` is the plain case — no `csv_http_server` in
/// the manifest, nothing to explain beyond the row cap.
///
/// The `Up` arm is reachable only from the write-failure fallback in
/// [`render_cypher_body`]: a live listener that renders inline did so because
/// the file could not be written.
fn inline_csv_reason(csv_http: &crate::csv_http::CsvHttpState) -> Option<String> {
    use crate::csv_http::CsvHttpState;
    match csv_http {
        CsvHttpState::Off => None,
        CsvHttpState::Failed { reason, .. } => Some(format!(
            "extensions.csv_http_server is configured on this server but its listener did not \
             start, so the fetch URL it would have returned is unavailable ({reason}). Ask the \
             operator to free that port, or to drop the pinned `port:` from the manifest so the \
             OS assigns a free one."
        )),
        CsvHttpState::Up(_) => Some(
            "extensions.csv_http_server is running on this server but the CSV file could not be \
             written, so this result is inline."
                .to_string(),
        ),
    }
}

/// Render a CypherResult as an inline 15-row preview (header + repr per
/// row).
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

/// Count logical RFC CSV data records, excluding the header.
pub(crate) fn count_csv_rows(csv: &str) -> usize {
    csv_record_summary(csv).data_rows
}

struct CsvRecordSummary {
    data_rows: usize,
    capped_prefix_end: usize,
}

/// Count logical records and retain the byte boundary after the header and
/// first [`INLINE_CSV_ROW_LIMIT`] data records without allocating per row.
fn csv_record_summary(csv: &str) -> CsvRecordSummary {
    let bytes = csv.as_bytes();
    let mut in_quotes = false;
    let mut index = 0;
    let mut record_start = 0;
    let mut records = 0usize;
    let mut capped_prefix_end = None;
    while index < bytes.len() {
        match bytes[index] {
            b'"' if in_quotes && bytes.get(index + 1) == Some(&b'"') => index += 2,
            b'"' => {
                in_quotes = !in_quotes;
                index += 1;
            }
            b'\r' | b'\n' if !in_quotes => {
                index += 1;
                if bytes[index - 1] == b'\r' && bytes.get(index) == Some(&b'\n') {
                    index += 1;
                }
                records += 1;
                if records == INLINE_CSV_ROW_LIMIT + 1 {
                    capped_prefix_end = Some(index);
                }
                record_start = index;
            }
            _ => index += 1,
        }
    }
    if record_start < bytes.len() {
        records += 1;
        if records == INLINE_CSV_ROW_LIMIT + 1 {
            capped_prefix_end = Some(bytes.len());
        }
    }
    CsvRecordSummary {
        data_rows: records.saturating_sub(1),
        capped_prefix_end: capped_prefix_end.unwrap_or(bytes.len()),
    }
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
        Value::Timestamp(dt) => out.push_str(&dt.format("%Y-%m-%dT%H:%M:%S%.f").to_string()),
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
        // Collection / graph-entity variants go through the shared
        // converter, so this text surface publishes the same object shape as
        // the C ABI, the CLI's `--format json` and MCP recipe results — the
        // one `docs/python/value-projection.md` documents. Serialising the
        // `Value` enum directly would emit serde's externally-tagged
        // persistence encoding (`{"Relationship":{...,"w":{"Float64":1.5}}}`,
        // nested null as the string `"Null"`), which an agent reading this
        // preview has no accessor to undo.
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => {
            let _ = write!(out, "{}", kglite::api::param::kglite_value_to_json(val));
        }
    }
}

#[cfg(test)]
mod fractional_timestamp_contract_tests {
    use super::*;
    fn stamp(text: &str) -> Value {
        Value::Timestamp(
            chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f").unwrap(),
        )
    }
    #[test]
    fn fractional_timestamp_text_keeps_exact_value_and_whole_second_spelling() {
        for text in [
            "2025-01-02T03:04:05",
            "2025-01-02T03:04:05.123456789",
            "1969-12-31T23:59:59.500",
        ] {
            let mut actual = String::new();
            push_value_repr(&mut actual, &stamp(text));
            assert_eq!(actual, text);
        }
    }
}

#[cfg(test)]
mod natural_json_contract_tests {
    use super::*;
    use kglite::api::{NodeValue, PathValue, PropMap, RelValue};

    fn repr(val: &Value) -> String {
        let mut out = String::new();
        push_value_repr(&mut out, val);
        out
    }

    fn props(pairs: Vec<(&str, Value)>) -> PropMap {
        PropMap::from_pairs(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    fn node(id: u32, label: &str, name: &str) -> NodeValue {
        NodeValue {
            id,
            labels: vec![label.to_string()],
            properties: props(vec![("name", Value::String(name.to_string()))]),
        }
    }

    fn rel() -> RelValue {
        RelValue {
            id: 0,
            start_id: 1,
            end_id: 2,
            rel_type: "LINK".to_string(),
            properties: props(vec![("w", Value::Float64(1.5))]),
        }
    }

    /// The inline preview publishes the same object shape as the C ABI, the
    /// CLI's `--format json` and the MCP recipe results — the shape
    /// `docs/python/value-projection.md` documents. A serde-derived
    /// externally-tagged rendering (`{"Relationship":{…}}`, `{"Float64":1.5}`,
    /// `"Null"`) is a different wire, and an agent has no accessor to undo it.
    #[test]
    fn list_of_scalars_renders_as_a_bare_json_array() {
        let val = Value::List(vec![Value::Int64(1), Value::String("a".to_string())]);
        assert_eq!(repr(&val), r#"[1,"a"]"#);
    }

    #[test]
    fn a_nested_null_renders_as_json_null_not_the_string_null() {
        let val = Value::List(vec![Value::List(vec![Value::Int64(1), Value::Null])]);
        let text = repr(&val);
        assert_eq!(text, "[[1,null]]");
        assert!(!text.contains(r#""Null""#), "tagged null leaked: {text}");
        assert!(!text.contains("Int64"), "tagged scalar leaked: {text}");
    }

    #[test]
    fn a_map_renders_as_a_json_object_with_a_null_member() {
        let val = Value::Map(props(vec![("a", Value::Int64(1)), ("b", Value::Null)]));
        assert_eq!(repr(&val), r#"{"a":1,"b":null}"#);
    }

    #[test]
    fn a_node_renders_with_the_published_field_names() {
        let val = Value::Node(Box::new(node(7, "P", "x")));
        let text = repr(&val);
        assert_eq!(text, r#"{"id":7,"labels":["P"],"properties":{"name":"x"}}"#);
        assert!(!text.contains(r#""Node""#), "tagged node leaked: {text}");
    }

    #[test]
    fn a_relationship_renders_with_the_published_field_names_and_a_bare_float() {
        let text = repr(&Value::Relationship(Box::new(rel())));
        assert_eq!(
            text,
            r#"{"end":2,"id":0,"properties":{"w":1.5},"start":1,"type":"LINK"}"#
        );
        assert!(!text.contains("rel_type"), "internal field leaked: {text}");
        assert!(!text.contains("Float64"), "tagged scalar leaked: {text}");
    }

    #[test]
    fn a_path_renders_as_nodes_and_relationships() {
        let val = Value::Path(Box::new(PathValue {
            nodes: vec![node(1, "P", "a"), node(2, "P", "b")],
            rels: vec![rel()],
        }));
        let text = repr(&val);
        assert_eq!(
            text,
            r#"{"nodes":[{"id":1,"labels":["P"],"properties":{"name":"a"}},{"id":2,"labels":["P"],"properties":{"name":"b"}}],"relationships":[{"end":2,"id":0,"properties":{"w":1.5},"start":1,"type":"LINK"}]}"#
        );
        assert!(!text.contains(r#""Path""#), "tagged path leaked: {text}");
    }
}
