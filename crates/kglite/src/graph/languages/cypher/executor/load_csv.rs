//! `LOAD CSV` execution — a streaming external row source.
//!
//! # Why this is a driver, not a clause
//!
//! Every other clause is a `ResultSet -> ResultSet` function, and the drivers
//! in [`super::CypherExecutor::execute`] / [`super::write::execute_mutable`]
//! materialize a `Vec<ResultRow>` at each clause boundary. If `LOAD CSV` were
//! an ordinary clause it would have to return the *whole file* as one
//! `ResultSet` — and a multi-gigabyte CSV is exactly the input a bulk import
//! arrives with. Peak memory would scale with file size, which the
//! bounded-memory rule forbids.
//!
//! So `LOAD CSV` instead **drives** the clauses that follow it: read a bounded
//! batch of rows, run the rest of the pipeline over that batch, concatenate
//! the output, repeat. Memory is bounded by (batch size × row width) plus
//! whatever the query's own output legitimately needs — never by the file.
//!
//! # When batching is equivalent, and when it is not
//!
//! Running the suffix once per batch and concatenating is only equal to
//! running it once over every row when every remaining clause is *row-local*.
//! An aggregate, `ORDER BY`, `SKIP`/`LIMIT`, `DISTINCT`, `UNION`, or a
//! procedure call reasons over the whole row set, so per-batch execution would
//! produce one answer per batch instead of one answer overall — a silent wrong
//! result, the one outcome worse than an error.
//!
//! [`batching_barrier`] detects those shapes. When one is present the file is
//! read into a single batch instead, capped at [`MAX_MATERIALIZED_ROWS`] so a
//! query that cannot stream fails with an explanation rather than an OOM.
//! `LOAD CSV … CREATE/MERGE/SET` — the ingest shape the feature exists for —
//! always streams.
//!
//! # Filesystem access is a capability, not a given
//!
//! `file://` means the local filesystem, and a Bolt client is a *remote*
//! caller. Shipping `LOAD CSV` without a gate would hand every Bolt client an
//! arbitrary-file-read primitive. [`CsvImportPolicy`] therefore defaults to
//! [`CsvImportPolicy::Denied`]: an execution path must opt in, in-process
//! embedders grant [`CsvImportPolicy::LocalFilesystem`] (their caller already
//! has the host process's filesystem access), and the Bolt server grants only
//! an explicit [`CsvImportPolicy::Directory`] when started with
//! `--allow-csv-import <DIR>`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::super::ast::{Clause, LoadCsvClause, ReturnClause, WithClause};
use super::super::result::{ResultRow, ResultSet};
use crate::datatypes::values::Value;

/// Rows per batch handed to the downstream pipeline. Large enough that
/// per-batch overhead (rebuilding the read executor, one `ResultSet`
/// allocation) is amortized, small enough that a batch of wide rows stays far
/// inside any sane memory budget.
pub const BATCH_ROWS: usize = 1_000;

/// Ceiling on the single-batch fallback taken when a downstream clause cannot
/// be batched (see [`batching_barrier`]). Such a query genuinely needs every
/// row in memory at once; this bound turns an unbounded allocation into a
/// diagnosable error.
pub const MAX_MATERIALIZED_ROWS: usize = 1_000_000;

/// Reported if `LOAD CSV` ever reaches the ordinary clause dispatcher.
///
/// It cannot: the parser accepts it only in leading position, and both engines
/// strip it before their clause loop and run [`drive`] instead. The constant
/// exists so the dispatcher's guard arm stays one line.
pub const MISDISPATCHED: &str =
    "internal error: LOAD CSV reached the clause dispatcher instead of its batch driver. \
     Please report this query.";

/// Who may read local files through `LOAD CSV`.
///
/// Default is [`Self::Denied`] so a new binding, or an execution path whose
/// author never thought about `LOAD CSV`, cannot accidentally expose the local
/// filesystem. Granting access is always an explicit act.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum CsvImportPolicy {
    /// `LOAD CSV` is rejected. The default, and what remote callers get.
    #[default]
    Denied,
    /// Any path this process can read. For in-process embedding, where the
    /// caller is the host program and already has that access anyway.
    LocalFilesystem,
    /// Only files inside this directory, after symlink resolution.
    Directory(PathBuf),
}

impl CsvImportPolicy {
    /// The error a denied `LOAD CSV` reports. Names the reason and both routes
    /// out, because "permission denied" with no route is a support ticket.
    fn denied_message() -> String {
        "LOAD CSV is not enabled for this connection. Reading local files is a capability the \
         caller must be granted: it is on by default for in-process use (the Python API, the \
         Rust library, the CLI) and off by default for remote callers, so a Bolt client cannot \
         read arbitrary server files. Start kglite-bolt-server with \
         `--allow-csv-import <DIR>` to allow imports from a specific directory, or load the \
         file client-side and pass the rows in as a parameter."
            .to_string()
    }
}

/// Resolve a `LOAD CSV FROM` operand to a readable local path under `policy`.
///
/// Accepts `file://` URLs (the conventional spelling) and bare filesystem paths
/// (what people actually type). Every other scheme is rejected by name — never
/// as a parse error, since the statement was understood perfectly well.
pub fn resolve_csv_source(raw: &str, policy: &CsvImportPolicy) -> Result<PathBuf, String> {
    if let Some(rest) = strip_scheme(raw, "http://").or_else(|| strip_scheme(raw, "https://")) {
        let _ = rest;
        return Err(format!(
            "LOAD CSV cannot read over HTTP(S): {raw}. The kglite engine is deliberately \
             network-free — it carries no HTTP client, so there is nothing to fetch with. \
             Download the file first and load it with `LOAD CSV FROM 'file:///path/to/file.csv'`, \
             or fetch and parse it in your own code (for example pandas `read_csv(url)`) and pass \
             the rows in as a parameter."
        ));
    }

    let path_part = if let Some(rest) = strip_scheme(raw, "file://") {
        // `file:///abs/path` leaves a leading `/`; `file://host/path` is a
        // remote UNC reference we deliberately do not resolve.
        if let Some(after_host) = rest.strip_prefix('/') {
            file_url_path(after_host, cfg!(windows))
        } else {
            return Err(format!(
                "LOAD CSV does not support host-qualified file URLs: {raw}. Use a local path — \
                 `file:///absolute/path.csv` (three slashes) or a plain relative path."
            ));
        }
    } else if let Some((scheme, _)) = split_scheme(raw) {
        return Err(format!(
            "LOAD CSV does not support the `{scheme}:` URL scheme: {raw}. Supported sources are \
             `file://` URLs and plain local filesystem paths."
        ));
    } else {
        raw.to_string()
    };

    if path_part.is_empty() {
        return Err("LOAD CSV FROM was given an empty path".to_string());
    }

    authorize(Path::new(&path_part), policy)
}

/// Apply `policy` to a resolved path.
///
/// Both allowing modes canonicalize first, so the existence check and any
/// containment check see the same real path — a `..` segment or a symlink
/// cannot point outside an allowed directory.
fn authorize(path: &Path, policy: &CsvImportPolicy) -> Result<PathBuf, String> {
    match policy {
        CsvImportPolicy::Denied => Err(CsvImportPolicy::denied_message()),
        CsvImportPolicy::LocalFilesystem => canonicalize_readable(path),
        CsvImportPolicy::Directory(root) => {
            let real_root = root.canonicalize().map_err(|e| {
                format!(
                    "the configured LOAD CSV import directory {} is not readable: {e}",
                    root.display()
                )
            })?;
            // A relative path is resolved against the import root rather than
            // the server's working directory, which is both more useful and
            // removes a way to escape by cwd.
            let candidate = if path.is_absolute() {
                path.to_path_buf()
            } else {
                real_root.join(path)
            };
            let real = canonicalize_readable(&candidate)?;
            if real.starts_with(&real_root) {
                Ok(real)
            } else {
                Err(format!(
                    "LOAD CSV is restricted to {} on this server, and {} resolves outside it. \
                     Move the file into the import directory.",
                    real_root.display(),
                    real.display()
                ))
            }
        }
    }
}

fn canonicalize_readable(path: &Path) -> Result<PathBuf, String> {
    path.canonicalize()
        .map_err(|e| format!("LOAD CSV cannot open {}: {e}", path.display()))
}

/// `Some(remainder)` when `raw` starts with `prefix`, case-insensitively on
/// the scheme (URL schemes are case-insensitive; `FILE://` is legal).
fn strip_scheme(raw: &str, prefix: &str) -> Option<String> {
    if raw.len() >= prefix.len() && raw[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(raw[prefix.len()..].to_string())
    } else {
        None
    }
}

/// Whether `s` starts with a Windows drive qualifier (`C:\` or `C:/`).
fn is_drive_qualified(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.next() == Some(':')
        && matches!(chars.next(), Some('/') | Some('\\'))
}

/// Convert the part of a `file://` URL after the (empty) host into a path.
///
/// The conventional spelling of a Windows path in a `file://` URL is
/// `file:///C:/data.csv`, which leaves `C:/data.csv` after the host. Restoring
/// the leading slash there yields `/C:/data.csv`, which no Windows API
/// resolves — and under a `Directory` import root it is not absolute either,
/// so `PathBuf::push` produces `C:\C:\data.csv`. The engine's own error
/// messages point users at `file:///absolute/path.csv`, so this spelling is
/// the one they are told to use.
///
/// `windows` is threaded in rather than read from `cfg!` here so both branches
/// are testable from any host. It matters that this is platform-gated: on Unix
/// `/C:/data.csv` is a perfectly legal path and must keep resolving as one.
fn file_url_path(after_host: &str, windows: bool) -> String {
    if windows && is_drive_qualified(after_host) {
        after_host.to_string()
    } else {
        format!("/{after_host}")
    }
}

/// Split a leading `scheme:` off `raw`, if it has one.
///
/// A Windows drive letter (`C:\data`) is not a scheme, so single-character
/// prefixes are excluded; RFC 3986 schemes are at least two characters and
/// start with a letter.
fn split_scheme(raw: &str) -> Option<(&str, &str)> {
    let colon = raw.find(':')?;
    let scheme = &raw[..colon];
    if scheme.len() < 2 || !scheme.starts_with(|c: char| c.is_ascii_alphabetic()) {
        return None;
    }
    if !scheme
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
    {
        return None;
    }
    Some((scheme, &raw[colon + 1..]))
}

/// The first clause in `clauses` that makes per-batch execution inequivalent
/// to whole-input execution, with the reason.
///
/// Deliberately a positive allow-list of row-local clauses: an unrecognized or
/// newly added clause reads as a barrier, so the failure mode of forgetting to
/// update this function is a slower query, never a wrong answer.
pub fn batching_barrier(clauses: &[Clause]) -> Option<String> {
    for clause in clauses {
        let reason = match clause {
            // Row-local: each output row depends only on its own input row.
            Clause::Match(_)
            | Clause::OptionalMatch(_)
            | Clause::Where(_)
            | Clause::Unwind(_)
            | Clause::Create(_)
            | Clause::Set(_)
            | Clause::Delete(_)
            | Clause::Remove(_)
            | Clause::Merge(_)
            | Clause::Foreach { .. } => continue,

            Clause::With(w) => match with_barrier(w) {
                Some(reason) => reason,
                None => continue,
            },
            Clause::Return(r) => match return_barrier(r) {
                Some(reason) => reason,
                None => continue,
            },

            Clause::OrderBy(_) => "ORDER BY sorts the whole result".to_string(),
            Clause::Skip(_) => "SKIP counts across the whole result".to_string(),
            Clause::Limit(_) => "LIMIT counts across the whole result".to_string(),
            Clause::Union(_) => "UNION combines whole result sets".to_string(),
            Clause::Call(_) => "CALL yields rows independently of the input rows".to_string(),
            Clause::CallSubquery { .. } => {
                "an uncorrelated CALL { } subquery body runs once per invocation".to_string()
            }
            Clause::LoadCsv(_) => "LOAD CSV cannot appear twice".to_string(),
            Clause::Schema(_) => "schema DDL is a standalone statement".to_string(),
            other => format!("{} is not batchable", super::clause_display_name(other)),
        };
        return Some(reason);
    }
    None
}

fn with_barrier(w: &WithClause) -> Option<String> {
    if w.distinct {
        return Some("WITH DISTINCT dedupes across the whole result".to_string());
    }
    if w.items
        .iter()
        .any(|item| super::super::ast::is_aggregate_expression(&item.expression))
    {
        return Some("WITH aggregates over the whole result".to_string());
    }
    None
}

fn return_barrier(r: &ReturnClause) -> Option<String> {
    if r.distinct {
        return Some("RETURN DISTINCT dedupes across the whole result".to_string());
    }
    if r.having.is_some() {
        return Some("HAVING filters post-aggregation".to_string());
    }
    if r.items
        .iter()
        .any(|item| super::super::ast::is_aggregate_expression(&item.expression))
    {
        return Some("RETURN aggregates over the whole result".to_string());
    }
    None
}

/// A streaming CSV reader bound to one `LOAD CSV` clause.
struct RowReader {
    reader: csv::Reader<std::fs::File>,
    /// Header names when `WITH HEADERS` was given. Rows bind as maps keyed by
    /// these; without them, rows bind as zero-indexed lists.
    headers: Option<Vec<String>>,
    variable: String,
    display_path: String,
    row_number: u64,
}

impl RowReader {
    fn open(clause: &LoadCsvClause, path: &Path) -> Result<Self, String> {
        let mut builder = csv::ReaderBuilder::new();
        builder
            // Headers are consumed explicitly below so the `WITH HEADERS`
            // decision lives in one place.
            .has_headers(false)
            // A short row is a data problem, not a parse error: the missing
            // fields bind as null rather than aborting the whole load.
            .flexible(true);
        if let Some(delimiter) = clause.field_terminator {
            builder.delimiter(delimiter);
        }
        let mut reader = builder
            .from_path(path)
            .map_err(|e| format!("LOAD CSV cannot open {}: {e}", path.display()))?;

        let headers = if clause.with_headers {
            let mut record = csv::StringRecord::new();
            let has_header_row = reader
                .read_record(&mut record)
                .map_err(|e| format!("LOAD CSV header read failed on {}: {e}", path.display()))?;
            if !has_header_row {
                return Err(format!(
                    "LOAD CSV WITH HEADERS was given an empty file: {}",
                    path.display()
                ));
            }
            Some(record.iter().map(str::to_string).collect())
        } else {
            None
        };

        Ok(RowReader {
            reader,
            headers,
            variable: clause.variable.clone(),
            display_path: path.display().to_string(),
            row_number: 0,
        })
    }

    /// Read up to `limit` rows. A short return means end of file.
    fn next_batch(&mut self, limit: usize) -> Result<Vec<ResultRow>, String> {
        let mut rows = Vec::with_capacity(limit.min(BATCH_ROWS));
        let mut record = csv::StringRecord::new();
        while rows.len() < limit {
            let more = self.reader.read_record(&mut record).map_err(|e| {
                format!(
                    "LOAD CSV failed reading {} at row {}: {e}",
                    self.display_path,
                    self.row_number + 1
                )
            })?;
            if !more {
                break;
            }
            self.row_number += 1;
            let mut row = ResultRow::new();
            row.projected
                .insert(self.variable.clone(), self.bind(&record));
            rows.push(row);
        }
        Ok(rows)
    }

    /// Convert one CSV record into the value bound to the row variable.
    ///
    /// Fields stay strings: CSV carries no types, and guessing them would
    /// silently corrupt zip codes, phone numbers, and leading-zero identifiers.
    /// Callers convert explicitly with `toInteger(row.n)` / `toFloat(row.x)`.
    fn bind(&self, record: &csv::StringRecord) -> Value {
        match &self.headers {
            Some(names) => {
                let mut map = BTreeMap::new();
                for (index, name) in names.iter().enumerate() {
                    let value = match record.get(index) {
                        // An empty field is null — otherwise every optional
                        // column needs a `= ''` guard.
                        Some("") | None => Value::Null,
                        Some(text) => Value::String(text.to_string()),
                    };
                    map.insert(name.clone(), value);
                }
                Value::Map(map)
            }
            None => Value::List(
                record
                    .iter()
                    .map(|field| {
                        if field.is_empty() {
                            Value::Null
                        } else {
                            Value::String(field.to_string())
                        }
                    })
                    .collect(),
            ),
        }
    }
}

/// Open the CSV named by `clause` and drive `run_batch` over bounded batches
/// of its rows, concatenating the results.
///
/// `source` is the already-evaluated `FROM` operand. `run_batch` runs the
/// clauses that follow `LOAD CSV` over one seed `ResultSet`; it is called once
/// per batch when the suffix is batchable, and exactly once otherwise.
pub(crate) fn drive<F>(
    clause: &LoadCsvClause,
    source: &Value,
    policy: &CsvImportPolicy,
    barrier: Option<&str>,
    mut run_batch: F,
) -> Result<ResultSet, String>
where
    F: FnMut(ResultSet) -> Result<ResultSet, String>,
{
    let raw = match source {
        Value::String(s) => s.as_str(),
        Value::Null => {
            return Err(
                "LOAD CSV FROM evaluated to null — check the parameter you passed".to_string(),
            )
        }
        other => {
            return Err(format!(
                "LOAD CSV FROM expects a string path or file:// URL, got {}",
                other.type_name()
            ))
        }
    };

    let path = resolve_csv_source(raw, policy)?;
    let mut reader = RowReader::open(clause, &path)?;

    let batch_limit = match barrier {
        None => BATCH_ROWS,
        // Not batchable: one oversized batch, bounded so the failure is a
        // message rather than an allocator death.
        Some(_) => MAX_MATERIALIZED_ROWS.saturating_add(1),
    };

    let mut merged: Option<ResultSet> = None;
    loop {
        let rows = reader.next_batch(batch_limit)?;
        if rows.is_empty() {
            break;
        }
        if let Some(reason) = barrier {
            if rows.len() > MAX_MATERIALIZED_ROWS {
                return Err(format!(
                    "LOAD CSV cannot stream this query because {reason}, so every row must be \
                     held in memory at once — and {} has more than {MAX_MATERIALIZED_ROWS} rows. \
                     Restructure the query so the clauses after LOAD CSV act on one row at a \
                     time (CREATE / MERGE / SET / DELETE ingest streams at any file size), or \
                     aggregate the file outside the query.",
                    path.display()
                ));
            }
        }
        let produced = rows.len();
        let seed = ResultSet {
            rows,
            columns: Vec::new(),
            lazy_return_items: None,
        };
        let out = run_batch(seed)?;
        merged = Some(match merged {
            None => out,
            Some(mut acc) => {
                if acc.columns.is_empty() {
                    acc.columns = out.columns;
                }
                acc.rows.extend(out.rows);
                acc
            }
        });
        if produced < batch_limit {
            break;
        }
    }

    Ok(merged.unwrap_or_else(ResultSet::new))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_sources_are_rejected_by_name_not_as_syntax() {
        let err = resolve_csv_source(
            "https://example.com/data.csv",
            &CsvImportPolicy::LocalFilesystem,
        )
        .unwrap_err();
        assert!(err.contains("network-free"), "{err}");
        assert!(err.contains("file:///"), "{err}");
    }

    #[test]
    fn unknown_schemes_name_the_supported_set() {
        let err = resolve_csv_source("s3://bucket/data.csv", &CsvImportPolicy::LocalFilesystem)
            .unwrap_err();
        assert!(err.contains("`s3:` URL scheme"), "{err}");
    }

    #[test]
    fn denied_is_the_default_policy() {
        let err = resolve_csv_source("data.csv", &CsvImportPolicy::default()).unwrap_err();
        assert!(err.contains("not enabled for this connection"), "{err}");
        assert!(err.contains("--allow-csv-import"), "{err}");
    }

    #[test]
    fn windows_drive_letters_are_not_url_schemes() {
        // Must fail on the missing file, not on a bogus `c:` scheme.
        let err =
            resolve_csv_source(r"C:\data\rows.csv", &CsvImportPolicy::LocalFilesystem).unwrap_err();
        assert!(err.contains("cannot open"), "{err}");
    }

    #[test]
    fn host_qualified_file_urls_are_refused() {
        let err = resolve_csv_source(
            "file://server/share/x.csv",
            &CsvImportPolicy::LocalFilesystem,
        )
        .unwrap_err();
        assert!(err.contains("host-qualified"), "{err}");
    }

    #[test]
    fn directory_policy_blocks_traversal_out_of_the_root() {
        let root = tempfile::tempdir().unwrap();
        let inside = root.path().join("ok.csv");
        std::fs::write(&inside, "a,b\n1,2\n").unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();

        let policy = CsvImportPolicy::Directory(root.path().to_path_buf());
        assert!(resolve_csv_source(inside.to_str().unwrap(), &policy).is_ok());
        assert!(resolve_csv_source("ok.csv", &policy).is_ok());

        let err = resolve_csv_source(outside.path().to_str().unwrap(), &policy).unwrap_err();
        assert!(err.contains("resolves outside it"), "{err}");

        // A `..` traversal to a file that really exists outside the root. The
        // previous probe pointed at a non-existent path, so it was rejected by
        // the existence check and never exercised containment at all.
        let outer = tempfile::tempdir().unwrap();
        let contained = outer.path().join("root");
        std::fs::create_dir(&contained).unwrap();
        let sibling = outer.path().join("sibling.csv");
        std::fs::write(&sibling, "a,b\n1,2\n").unwrap();
        let nested_policy = CsvImportPolicy::Directory(contained.clone());
        let escape = format!("{}/../sibling.csv", contained.display());
        let err = resolve_csv_source(&escape, &nested_policy).unwrap_err();
        assert!(err.contains("resolves outside it"), "{err}");
    }

    #[test]
    fn file_url_keeps_a_windows_drive_letter_addressable() {
        // `file:///C:/data.csv` is the spelling the engine's own error messages
        // steer users toward. Restoring the leading slash there produced
        // `/C:/data.csv`, which resolves nowhere on Windows.
        assert_eq!(file_url_path("C:/data.csv", true), "C:/data.csv");
        assert_eq!(
            file_url_path("c:\\data\\rows.csv", true),
            "c:\\data\\rows.csv"
        );
        // Unchanged on Unix, where `/C:/data.csv` is a legal path.
        assert_eq!(file_url_path("C:/data.csv", false), "/C:/data.csv");
        // Ordinary absolute paths keep their leading slash on both platforms.
        assert_eq!(file_url_path("tmp/x.csv", true), "/tmp/x.csv");
        assert_eq!(file_url_path("tmp/x.csv", false), "/tmp/x.csv");
    }

    #[test]
    fn drive_qualifier_needs_a_letter_colon_and_separator() {
        assert!(is_drive_qualified("C:/x"));
        assert!(is_drive_qualified("z:\\x"));
        // A bare `C:` is drive-relative, not drive-absolute; a scheme-like
        // prefix is not a drive.
        assert!(!is_drive_qualified("C:"));
        assert!(!is_drive_qualified("http://x"));
        assert!(!is_drive_qualified("1:/x"));
        assert!(!is_drive_qualified("/tmp/x"));
    }

    #[test]
    fn drive_lettered_file_url_reaches_the_policy_check() {
        // Whatever the platform, the resolved path must no longer be the
        // unresolvable `/C:/...`. On Unix that is still what a drive-lettered
        // URL means, so assert through the error text rather than the path.
        let err = resolve_csv_source(
            "file:///C:/nope/data.csv",
            &CsvImportPolicy::LocalFilesystem,
        )
        .unwrap_err();
        let expected = if cfg!(windows) {
            "C:"
        } else {
            "/C:/nope/data.csv"
        };
        assert!(err.contains(expected), "{err}");
    }

    #[test]
    fn ingest_shapes_stream_and_whole_result_shapes_do_not() {
        use crate::graph::languages::cypher::parser::parse_cypher;

        let batchable = parse_cypher(
            "LOAD CSV WITH HEADERS FROM 'file:///tmp/x.csv' AS row \
             CREATE (:Person {name: row.name})",
        )
        .unwrap();
        assert_eq!(batching_barrier(&batchable.clauses[1..]), None);

        let counted =
            parse_cypher("LOAD CSV FROM 'file:///tmp/x.csv' AS row RETURN count(*) AS n").unwrap();
        assert!(batching_barrier(&counted.clauses[1..])
            .unwrap()
            .contains("aggregates"));

        let ordered = parse_cypher(
            "LOAD CSV WITH HEADERS FROM 'file:///tmp/x.csv' AS row \
             RETURN row.name AS name ORDER BY name",
        )
        .unwrap();
        assert!(batching_barrier(&ordered.clauses[1..])
            .unwrap()
            .contains("ORDER BY"));
    }
}
