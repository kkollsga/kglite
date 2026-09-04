//! The `delimited` `Source`: a text file whose columns are separated by a
//! string the blueprint names, with a preamble, a line trailer, a fixed column
//! list and a legacy encoding all optional.
//!
//! Public bulk data is full of files a CSV reader cannot open: NCBI's taxonomy
//! dumps separate fields with `\t|\t` and end every line with `\t|`, KEGG ships
//! two headerless columns with `cpd:` / `path:` prefixes on the ids, Reactome
//! ships six headerless TSV columns, and a licence line above the header is
//! routine. Each of those is a `files` entry here rather than a Python
//! pre-pass.
//!
//! ## Two engines, one behaviour
//!
//! A single-byte `delimiter` goes to the `csv` crate — the same reader the
//! `csv` format uses, so quoting, embedded newlines and escapes behave exactly
//! as they do there. A multi-byte `delimiter` (`\t|\t`) goes to a line engine
//! that splits on the string and does **no quoting at all**, because no quoting
//! convention exists for such a file; `quote` together with a multi-byte
//! delimiter is refused rather than silently ignored.
//!
//! Everything else is shared: both engines read their lines through one filter
//! that strips a UTF-8 BOM, decodes the declared encoding, drops `skip_lines`
//! physical lines and any `comment_prefix` line, and removes `line_suffix`
//! once from the end — so the `\t|` trailer never becomes a phantom last
//! column. Both produce the same rectangular [`RawCsv`] the CSV reader does:
//! a short row is null-padded, fields past the header's width are dropped, and
//! null is the empty cell.
//!
//! ## Row numbers
//!
//! `row_ids` counts **data rows**, 1-based, after `skip_lines`, comment lines,
//! blank lines and the header are gone — the same thing it counts for a CSV,
//! and not the physical line number. A warning that says "row 12" means the
//! twelfth row of data. Read errors, which have no row to attribute yet, name
//! the physical line instead and say so.

use super::super::schema::FileSpec;
use super::super::table::{push_row, RawCsv};
use super::{FormatSpec, Source};
use indexmap::IndexMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;

/// Keys a `files` entry with `"format": "delimited"` reads.
pub const ACCEPTED_FILE_KEYS_DELIMITED: &[&str] = &[
    "path",
    "format",
    "delimiter",
    "quote",
    "header",
    "columns",
    "skip_lines",
    "comment_prefix",
    "line_suffix",
    "encoding",
    "prefix_strip",
];

/// The knobs above that live in `FileSpec::extra` — everything except the two
/// fields every format shares.
pub const KNOB_KEYS_DELIMITED: &[&str] = &[
    "delimiter",
    "quote",
    "header",
    "columns",
    "skip_lines",
    "comment_prefix",
    "line_suffix",
    "encoding",
    "prefix_strip",
];

pub const FORMAT: FormatSpec = FormatSpec {
    name: "delimited",
    accepted_keys: ACCEPTED_FILE_KEYS_DELIMITED,
    knob_keys: KNOB_KEYS_DELIMITED,
    validate_entry,
};

/// Every `delimited` rule is a rule about its config, so validating an entry
/// is building one and discarding it.
fn validate_entry(name: &str, file: &FileSpec) -> Result<(), String> {
    DelimitedConfig::from_spec(name, file).map(|_| ())
}

/// The byte→text rule for reading the file. No dependency decodes these: UTF-8
/// is `std`'s and latin-1 is one byte per `char`, which is what latin-1 *is*.
/// Anything else is refused by name rather than mojibaked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    Utf8,
    Latin1,
}

/// A validated `delimited` declaration. Constructing one is the whole
/// validation: every rule that can be decided without opening the file is
/// decided here, so `validate_inputs` and the registry agree by construction.
#[derive(Clone, Debug)]
pub struct DelimitedConfig {
    delimiter: String,
    quote: Option<u8>,
    /// The column names, when the file has no header row of its own. `None`
    /// means "the first surviving line names the columns" — `header` and
    /// `columns` are one decision, and `parse_columns` refuses a declaration
    /// that writes it twice, so this is the whole of it.
    columns: Option<Vec<String>>,
    skip_lines: usize,
    comment_prefix: Option<String>,
    line_suffix: Option<String>,
    encoding: Encoding,
    /// Column name → prefix removed from the start of that column's cells.
    prefix_strip: Vec<(String, String)>,
}

/// What a value is, for an error that says what was written instead.
fn json_kind(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "a list",
        serde_json::Value::Object(_) => "an object",
    }
}

type Extra = IndexMap<String, serde_json::Value>;

fn get_string(name: &str, extra: &Extra, key: &str) -> Result<Option<String>, String> {
    match extra.get(key) {
        None => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
        Some(v) => Err(format!(
            "files '{name}': '{key}' must be a string, but it is {}.",
            json_kind(v)
        )),
    }
}

fn get_bool(name: &str, extra: &Extra, key: &str) -> Result<Option<bool>, String> {
    match extra.get(key) {
        None => Ok(None),
        Some(serde_json::Value::Bool(b)) => Ok(Some(*b)),
        Some(v) => Err(format!(
            "files '{name}': '{key}' must be true or false, but it is {}.",
            json_kind(v)
        )),
    }
}

fn get_usize(name: &str, extra: &Extra, key: &str) -> Result<Option<usize>, String> {
    match extra.get(key) {
        None => Ok(None),
        Some(serde_json::Value::Number(n)) => match n.as_u64() {
            Some(n) => Ok(Some(n as usize)),
            None => Err(format!(
                "files '{name}': '{key}' must be a whole number of lines that is not negative, \
                 but it is {n}."
            )),
        },
        Some(v) => Err(format!(
            "files '{name}': '{key}' must be a whole number of lines, but it is {}.",
            json_kind(v)
        )),
    }
}

fn get_string_list(name: &str, extra: &Extra, key: &str) -> Result<Option<Vec<String>>, String> {
    let Some(v) = extra.get(key) else {
        return Ok(None);
    };
    let serde_json::Value::Array(items) = v else {
        return Err(format!(
            "files '{name}': '{key}' must be a list of column names, but it is {}.",
            json_kind(v)
        ));
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            serde_json::Value::String(s) => out.push(s.clone()),
            other => {
                return Err(format!(
                    "files '{name}': every entry of '{key}' must be a column name, but one is {}.",
                    json_kind(other)
                ))
            }
        }
    }
    Ok(Some(out))
}

fn get_string_map(
    name: &str,
    extra: &Extra,
    key: &str,
) -> Result<Option<Vec<(String, String)>>, String> {
    let Some(v) = extra.get(key) else {
        return Ok(None);
    };
    let serde_json::Value::Object(map) = v else {
        return Err(format!(
            "files '{name}': '{key}' must be an object of column name → text, but it is {}.",
            json_kind(v)
        ));
    };
    let mut out = Vec::with_capacity(map.len());
    for (column, value) in map {
        match value {
            serde_json::Value::String(s) => out.push((column.clone(), s.clone())),
            other => {
                return Err(format!(
                    "files '{name}': '{key}' entry '{column}' must be text, but it is {}.",
                    json_kind(other)
                ))
            }
        }
    }
    Ok(Some(out))
}

impl DelimitedConfig {
    /// Read and check every knob of one `files` entry.
    ///
    /// Called from `validate_inputs` so a bad declaration fails the build
    /// before any file is opened, and again by the registry, which is the one
    /// that keeps the result.
    pub fn from_spec(name: &str, file: &FileSpec) -> Result<Self, String> {
        let extra = &file.extra;
        let delimiter = get_string(name, extra, "delimiter")?.ok_or_else(|| {
            format!(
                "files '{name}': a 'delimited' input needs a 'delimiter' — the text between two \
                 fields, e.g. \"\\t\" or \"\\t|\\t\"."
            )
        })?;
        if delimiter.is_empty() {
            return Err(format!(
                "files '{name}': 'delimiter' is empty — there is no text between two fields to \
                 split on."
            ));
        }
        let quote = Self::parse_quote(name, extra, &delimiter)?;
        let header = get_bool(name, extra, "header")?.unwrap_or(true);
        let columns = Self::parse_columns(name, extra, header)?;
        let comment_prefix = Self::non_empty(name, extra, "comment_prefix")?;
        let line_suffix = Self::non_empty(name, extra, "line_suffix")?;
        let prefix_strip = Self::parse_prefix_strip(name, extra)?;
        Ok(Self {
            delimiter,
            quote,
            columns,
            skip_lines: get_usize(name, extra, "skip_lines")?.unwrap_or(0),
            comment_prefix,
            line_suffix,
            encoding: Self::parse_encoding(name, extra)?,
            prefix_strip,
        })
    }

    fn non_empty(name: &str, extra: &Extra, key: &str) -> Result<Option<String>, String> {
        match get_string(name, extra, key)? {
            Some(s) if s.is_empty() => Err(format!(
                "files '{name}': '{key}' is empty — drop the key instead, an empty one matches \
                 every line."
            )),
            other => Ok(other),
        }
    }

    /// A quote character is a single-byte concern of the `csv` crate, and the
    /// line engine has no quoting at all — so a `quote` beside a multi-byte
    /// delimiter is refused rather than accepted and ignored.
    fn parse_quote(name: &str, extra: &Extra, delimiter: &str) -> Result<Option<u8>, String> {
        let Some(quote) = get_string(name, extra, "quote")? else {
            return Ok(None);
        };
        let bytes = quote.as_bytes();
        if bytes.len() != 1 {
            return Err(format!(
                "files '{name}': 'quote' must be a single ASCII character, but it is '{quote}'."
            ));
        }
        if delimiter.len() > 1 {
            return Err(format!(
                "files '{name}': 'quote' cannot be used with the multi-character delimiter \
                 '{}' — a file split on a multi-character separator is read line by line, with \
                 no quoting. Drop 'quote'.",
                delimiter.escape_debug()
            ));
        }
        Ok(Some(bytes[0]))
    }

    /// `columns` and `header` are the same decision written two ways, so
    /// setting both is refused: silently renaming a header the file does have
    /// is exactly the failure a headerless declaration should not be able to
    /// cause.
    fn parse_columns(
        name: &str,
        extra: &Extra,
        header: bool,
    ) -> Result<Option<Vec<String>>, String> {
        let columns = get_string_list(name, extra, "columns")?;
        match (&columns, header) {
            (Some(_), true) => Err(format!(
                "files '{name}': 'columns' is set together with \"header\": true — the file's \
                 own header row would name the columns and 'columns' would be ignored. Set \
                 \"header\": false to name them here."
            )),
            (None, false) => Err(format!(
                "files '{name}': \"header\": false needs 'columns' — a file with no header row \
                 has nothing to name its columns."
            )),
            (Some(c), false) if c.is_empty() => Err(format!(
                "files '{name}': 'columns' is empty — a headerless file needs one name per \
                 column."
            )),
            (Some(c), false) => match first_duplicate(c) {
                Some(dup) => Err(format!(
                    "files '{name}': 'columns' names '{dup}' twice — a spec asking for that \
                     column would always read the first one."
                )),
                None => Ok(columns),
            },
            (None, true) => Ok(None),
        }
    }

    fn parse_encoding(name: &str, extra: &Extra) -> Result<Encoding, String> {
        let Some(raw) = get_string(name, extra, "encoding")? else {
            return Ok(Encoding::Utf8);
        };
        let normalised: String = raw
            .chars()
            .filter(|c| *c != '-' && *c != '_' && *c != ' ')
            .flat_map(char::to_lowercase)
            .collect();
        match normalised.as_str() {
            "utf8" => Ok(Encoding::Utf8),
            "latin1" | "iso88591" => Ok(Encoding::Latin1),
            _ => Err(format!(
                "files '{name}': unknown encoding '{raw}' — this build decodes 'utf-8' (the \
                 default) and 'latin-1'."
            )),
        }
    }

    fn parse_prefix_strip(name: &str, extra: &Extra) -> Result<Vec<(String, String)>, String> {
        let entries = get_string_map(name, extra, "prefix_strip")?.unwrap_or_default();
        for (column, prefix) in &entries {
            if prefix.is_empty() {
                return Err(format!(
                    "files '{name}': 'prefix_strip' entry '{column}' is empty — every cell starts \
                     with an empty prefix, so there is nothing to strip."
                ));
            }
        }
        Ok(entries)
    }

    /// The single-byte delimiter the `csv` engine needs, or `None` for the
    /// line engine.
    fn single_byte_delimiter(&self) -> Option<u8> {
        match self.delimiter.as_bytes() {
            [b] => Some(*b),
            _ => None,
        }
    }
}

fn first_duplicate(names: &[String]) -> Option<&str> {
    for (i, name) in names.iter().enumerate() {
        if names[..i].iter().any(|earlier| earlier == name) {
            return Some(name);
        }
    }
    None
}

/// A delimited text file on disk. `display` is the name the blueprint referred
/// to it by, which is what diagnostics print.
pub struct DelimitedFile {
    path: PathBuf,
    display: String,
    config: DelimitedConfig,
}

impl DelimitedFile {
    pub fn new(path: PathBuf, display: String, config: DelimitedConfig) -> Self {
        Self {
            path,
            display,
            config,
        }
    }

    /// Open the file and resolve its column names: the first surviving line
    /// when `header` is true, the declared `columns` otherwise.
    fn open(&self) -> Result<(RowReader, Vec<String>), String> {
        let file =
            File::open(&self.path).map_err(|e| format!("delimited open {}: {e}", self.display))?;
        let lines = LineFilter::new(BufReader::new(file), &self.config, &self.display);
        let engine = match self.config.single_byte_delimiter() {
            Some(delimiter) => {
                let mut builder = ::csv::ReaderBuilder::new();
                builder
                    .delimiter(delimiter)
                    .has_headers(false)
                    .flexible(true);
                if let Some(quote) = self.config.quote {
                    builder.quote(quote);
                }
                RowEngine::Csv {
                    reader: builder.from_reader(FilteredRead::new(lines)),
                    record: ::csv::StringRecord::new(),
                }
            }
            None => RowEngine::Lines {
                lines,
                delimiter: self.config.delimiter.clone(),
                current: String::new(),
            },
        };
        let mut reader = RowReader {
            engine,
            display: self.display.clone(),
        };
        let headers = match &self.config.columns {
            Some(columns) => columns.clone(),
            None if reader.advance()? => reader.fields().map(str::to_string).collect(),
            // An empty file has no header row and therefore no columns — the
            // same zero-column table the CSV reader produces for one.
            None => Vec::new(),
        };
        Ok((reader, headers))
    }

    /// Per column index, the prefix to remove from its cells. Built once per
    /// pass; a `prefix_strip` column the file does not have has no slot and is
    /// ignored.
    fn strip_slots(&self, headers: &[String]) -> Vec<Option<String>> {
        let mut slots = vec![None; headers.len()];
        for (column, prefix) in &self.config.prefix_strip {
            if let Some(i) = headers.iter().position(|h| h == column) {
                slots[i] = Some(prefix.clone());
            }
        }
        slots
    }
}

/// `cell` with its column's declared prefix removed, if it carries one. A cell
/// that does not start with the prefix is its own value — `prefix_strip` is a
/// tidy-up, not a filter.
fn strip_cell<'a>(cell: &'a str, slot: Option<&Option<String>>) -> &'a str {
    match slot {
        Some(Some(prefix)) => cell.strip_prefix(prefix.as_str()).unwrap_or(cell),
        _ => cell,
    }
}

impl Source for DelimitedFile {
    fn display_name(&self) -> &str {
        &self.display
    }

    fn size_hint(&self) -> Option<u64> {
        std::fs::metadata(&self.path).ok().map(|m| m.len())
    }

    fn can_chunk(&self) -> bool {
        true
    }

    fn read_all(&self) -> Result<RawCsv, String> {
        let (mut reader, headers) = self.open()?;
        let slots = self.strip_slots(&headers);
        let n_cols = headers.len();
        let mut rows = Vec::new();
        let mut nulls = Vec::new();
        while reader.advance()? {
            push_stripped_row(reader.fields(), n_cols, &slots, &mut rows, &mut nulls);
        }
        let row_ids: Vec<usize> = (1..=rows.len()).collect();
        Ok(RawCsv {
            headers,
            rows,
            nulls,
            row_ids,
        })
    }

    fn chunks(
        &self,
        chunk_size: usize,
    ) -> Result<Box<dyn Iterator<Item = Result<RawCsv, String>> + '_>, String> {
        let (mut reader, headers) = self.open()?;
        let slots = self.strip_slots(&headers);
        let n_cols = headers.len();
        let mut next_row_id = 1usize;
        let iter = std::iter::from_fn(move || {
            let mut rows = Vec::with_capacity(chunk_size);
            let mut nulls = Vec::with_capacity(chunk_size);
            let mut row_ids = Vec::with_capacity(chunk_size);
            for _ in 0..chunk_size {
                match reader.advance() {
                    Ok(true) => {
                        push_stripped_row(reader.fields(), n_cols, &slots, &mut rows, &mut nulls);
                        row_ids.push(next_row_id);
                        next_row_id += 1;
                    }
                    Ok(false) => break,
                    Err(e) => return Some(Err(e)),
                }
            }
            if rows.is_empty() {
                return None;
            }
            debug_assert_eq!(
                row_ids.len(),
                rows.len(),
                "every chunk row carries its source row number"
            );
            Some(Ok(RawCsv {
                headers: headers.clone(),
                rows,
                nulls,
                row_ids,
            }))
        });
        Ok(Box::new(iter))
    }

    /// One pass over the file reading only the requested columns: no table, no
    /// `String` per cell. Visits in the caller's column order, exactly as the
    /// trait default does, so the two answer identically.
    fn scan_columns(
        &self,
        columns: &[&str],
        visit: &mut dyn FnMut(usize, &str) -> bool,
    ) -> Result<(), String> {
        let (mut reader, headers) = self.open()?;
        let indices: Vec<Option<usize>> = columns
            .iter()
            .map(|name| headers.iter().position(|h| h == name))
            .collect();
        if indices.iter().all(Option::is_none) {
            return Ok(());
        }
        let slots = self.strip_slots(&headers);
        while reader.advance()? {
            for (slot, idx) in indices.iter().enumerate() {
                let Some(idx) = idx else { continue };
                let Some(cell) = reader.field(*idx) else {
                    continue;
                };
                let cell = strip_cell(cell, slots.get(*idx));
                if cell.is_empty() {
                    continue;
                }
                if !visit(slot, cell) {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

fn push_stripped_row<'a>(
    fields: impl Iterator<Item = &'a str>,
    n_cols: usize,
    slots: &[Option<String>],
    rows: &mut Vec<Vec<String>>,
    nulls: &mut Vec<Vec<bool>>,
) {
    if slots.iter().all(Option::is_none) {
        push_row(fields, n_cols, rows, nulls);
        return;
    }
    push_row(
        fields
            .enumerate()
            .map(|(i, cell)| strip_cell(cell, slots.get(i))),
        n_cols,
        rows,
        nulls,
    );
}

/// One row at a time from whichever engine reads this file.
struct RowReader {
    engine: RowEngine,
    display: String,
}

enum RowEngine {
    Csv {
        reader: ::csv::Reader<FilteredRead<BufReader<File>>>,
        record: ::csv::StringRecord,
    },
    Lines {
        lines: LineFilter<BufReader<File>>,
        delimiter: String,
        current: String,
    },
}

impl RowReader {
    /// Read the next row, or `false` at end of file. The row's cells are then
    /// readable through [`RowReader::fields`] and [`RowReader::field`].
    fn advance(&mut self) -> Result<bool, String> {
        let display = &self.display;
        match &mut self.engine {
            RowEngine::Csv { reader, record } => reader
                .read_record(record)
                .map_err(|e| csv_row_error(display, e)),
            RowEngine::Lines { lines, current, .. } => match lines.next_line()? {
                Some(line) => {
                    *current = line;
                    Ok(true)
                }
                None => Ok(false),
            },
        }
    }

    fn fields(&self) -> Fields<'_> {
        match &self.engine {
            RowEngine::Csv { record, .. } => Fields::Csv(record.iter()),
            RowEngine::Lines {
                delimiter, current, ..
            } => Fields::Lines(current.split(delimiter.as_str())),
        }
    }

    /// The cell at column `idx` of the current row, without walking the whole
    /// row into a table — what the projected type pre-pass reads.
    fn field(&self, idx: usize) -> Option<&str> {
        match &self.engine {
            RowEngine::Csv { record, .. } => record.get(idx),
            RowEngine::Lines {
                delimiter, current, ..
            } => current.split(delimiter.as_str()).nth(idx),
        }
    }
}

/// A `csv`-engine read failure, named the way the reader that produced it
/// named it. The line filter's own errors arrive here wrapped as IO errors and
/// already carry the file and the physical line, so re-prefixing them would
/// print the file's name twice in one sentence.
fn csv_row_error(display: &str, e: ::csv::Error) -> String {
    if let ::csv::ErrorKind::Io(io) = e.kind() {
        if io.kind() == std::io::ErrorKind::InvalidData {
            return io.to_string();
        }
    }
    format!("delimited row {display}: {e}")
}

enum Fields<'a> {
    Csv(::csv::StringRecordIter<'a>),
    Lines(std::str::Split<'a, &'a str>),
}

impl<'a> Iterator for Fields<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        match self {
            Fields::Csv(it) => it.next(),
            Fields::Lines(it) => it.next(),
        }
    }
}

const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

/// The shared line layer: decode, and drop or trim what is not data.
///
/// Both engines read through this, so `skip_lines`, `comment_prefix`,
/// `line_suffix`, the BOM and the encoding behave identically whichever
/// splitter runs behind it. It is line-oriented and runs *before* quoting, so
/// a `comment_prefix` or `line_suffix` that occurs inside a quoted field
/// spanning several lines is still treated as one.
struct LineFilter<R: BufRead> {
    inner: R,
    encoding: Encoding,
    skip_lines: usize,
    comment_prefix: Option<String>,
    line_suffix: Option<String>,
    display: String,
    /// Physical line number of the line last read — what a decode error names,
    /// because at that point there is no data row to name.
    physical: usize,
    buf: Vec<u8>,
}

impl<R: BufRead> LineFilter<R> {
    fn new(inner: R, config: &DelimitedConfig, display: &str) -> Self {
        Self {
            inner,
            encoding: config.encoding,
            skip_lines: config.skip_lines,
            comment_prefix: config.comment_prefix.clone(),
            line_suffix: config.line_suffix.clone(),
            display: display.to_string(),
            physical: 0,
            buf: Vec::new(),
        }
    }

    /// The next line that carries data, or `None` at end of file.
    fn next_line(&mut self) -> Result<Option<String>, String> {
        loop {
            self.buf.clear();
            let read = self
                .inner
                .read_until(b'\n', &mut self.buf)
                .map_err(|e| format!("delimited read {}: {e}", self.display))?;
            if read == 0 {
                return Ok(None);
            }
            self.physical += 1;
            let mut bytes: &[u8] = &self.buf;
            if let Some(rest) = bytes.strip_suffix(b"\n") {
                bytes = rest;
            }
            // A CRLF file read line by line leaves the CR on the last field of
            // every row; strip it here so both engines see the same text.
            if let Some(rest) = bytes.strip_suffix(b"\r") {
                bytes = rest;
            }
            if self.physical == 1 {
                if let Some(rest) = bytes.strip_prefix(UTF8_BOM) {
                    bytes = rest;
                }
            }
            // Physical lines, dropped before anything looks at them — a
            // preamble may be anything at all, including undecodable bytes.
            if self.physical <= self.skip_lines {
                continue;
            }
            if bytes.is_empty() {
                continue;
            }
            let line = self.decode(bytes)?;
            if let Some(prefix) = &self.comment_prefix {
                if line.starts_with(prefix.as_str()) {
                    continue;
                }
            }
            return Ok(Some(self.trim_suffix(line)));
        }
    }

    fn decode(&self, bytes: &[u8]) -> Result<String, String> {
        match self.encoding {
            Encoding::Utf8 => std::str::from_utf8(bytes).map(str::to_string).map_err(|e| {
                format!(
                    "delimited {} line {}: {e} — declare \"encoding\": \"latin-1\" if the \
                         file is not UTF-8.",
                    self.display, self.physical
                )
            }),
            // Latin-1 is the code points 0..=255 in order, which is what a
            // byte-to-char cast is.
            Encoding::Latin1 => Ok(bytes.iter().map(|b| *b as char).collect()),
        }
    }

    /// Remove `line_suffix` once from the end. Once, not repeatedly: a row
    /// whose last field happens to end in the trailer keeps its value.
    fn trim_suffix(&self, mut line: String) -> String {
        if let Some(suffix) = &self.line_suffix {
            if line.ends_with(suffix.as_str()) {
                line.truncate(line.len() - suffix.len());
            }
        }
        line
    }
}

/// The filtered lines as a byte stream, so the `csv` crate reads exactly what
/// the line engine splits.
struct FilteredRead<R: BufRead> {
    lines: LineFilter<R>,
    pending: Vec<u8>,
    pos: usize,
    done: bool,
}

impl<R: BufRead> FilteredRead<R> {
    fn new(lines: LineFilter<R>) -> Self {
        Self {
            lines,
            pending: Vec::new(),
            pos: 0,
            done: false,
        }
    }
}

impl<R: BufRead> Read for FilteredRead<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.pending.len() {
            if self.done {
                return Ok(0);
            }
            match self.lines.next_line() {
                Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
                Ok(None) => {
                    self.done = true;
                    return Ok(0);
                }
                Ok(Some(line)) => {
                    self.pending.clear();
                    self.pending.extend_from_slice(line.as_bytes());
                    self.pending.push(b'\n');
                    self.pos = 0;
                }
            }
        }
        let n = (self.pending.len() - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.pending[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

#[cfg(test)]
mod delimited_tests {
    use super::*;
    use std::io::Write;

    fn spec(json: &str) -> FileSpec {
        serde_json::from_str(json).expect("files entry parses")
    }

    fn config(json: &str) -> DelimitedConfig {
        DelimitedConfig::from_spec("t", &spec(json)).expect("a valid declaration")
    }

    fn config_err(json: &str) -> String {
        DelimitedConfig::from_spec("t", &spec(json)).expect_err("an invalid declaration")
    }

    fn file(content: &[u8], json: &str) -> (tempfile::NamedTempFile, DelimitedFile) {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content).unwrap();
        f.flush().unwrap();
        let src = DelimitedFile::new(
            f.path().to_path_buf(),
            "sample.txt".to_string(),
            config(json),
        );
        (f, src)
    }

    /// Everything a `RawCsv` carries, for comparing one engine's table with
    /// the other's.
    type Shape = (Vec<String>, Vec<Vec<String>>, Vec<Vec<bool>>, Vec<usize>);

    fn shape(raw: &RawCsv) -> Shape {
        (
            raw.headers.clone(),
            raw.rows.clone(),
            raw.nulls.clone(),
            raw.row_ids.clone(),
        )
    }

    /// The two engines are a performance/quoting choice, not a semantic one:
    /// the same content declared each way must produce the identical table.
    #[test]
    fn both_engines_produce_the_same_table() {
        let single = b"a\tb\tc\n1\t2\t3\n4\t\t6\n";
        let multi = "a<->b<->c\n1<->2<->3\n4<-><->6\n";
        let (_f1, csv_engine) = file(single, r#"{"path": "x", "delimiter": "\t"}"#);
        let (_f2, line_engine) = file(multi.as_bytes(), r#"{"path": "x", "delimiter": "<->"}"#);
        assert_eq!(
            shape(&csv_engine.read_all().unwrap()),
            shape(&line_engine.read_all().unwrap())
        );
        // Non-vacuity: the shared table is the expected one, not two empties.
        let raw = csv_engine.read_all().unwrap();
        assert_eq!(raw.headers, vec!["a", "b", "c"]);
        assert_eq!(raw.rows[1], vec!["4", "", "6"]);
        assert_eq!(raw.nulls[1], vec![false, true, false]);
        assert_eq!(raw.row_ids, vec![1, 2]);
    }

    /// The `.dmp` shape: `\t|\t` between fields and `\t|` closing every line.
    /// Without the suffix strip the last column is a phantom empty one.
    #[test]
    fn the_line_suffix_is_stripped_once_and_makes_no_phantom_column() {
        let content = "1\t|\troot\t|\t0\t|\n2\t|\tBacteria\t|\t1\t|\n";
        let (_f, src) = file(
            content.as_bytes(),
            r#"{"path": "x", "delimiter": "\t|\t", "line_suffix": "\t|",
                "header": false, "columns": ["id", "name", "rank"]}"#,
        );
        let raw = src.read_all().unwrap();
        assert_eq!(raw.headers, vec!["id", "name", "rank"]);
        assert_eq!(raw.rows.len(), 2);
        assert_eq!(raw.rows[0], vec!["1", "root", "0"]);
        assert_eq!(raw.rows[1], vec!["2", "Bacteria", "1"]);
    }

    /// The BugSigDB shape: a licence line above the header, skipped by count.
    /// Nothing marks it as a comment, so `skip_lines` is the only thing that
    /// can drop it.
    #[test]
    fn skip_lines_drops_a_preamble_the_header_sits_under() {
        let (_f, src) = file(
            b"BugSigDB, License: CC BY 4.0\na,b\n1,2\n3,4\n",
            r#"{"path": "x", "delimiter": ",", "skip_lines": 1}"#,
        );
        let raw = src.read_all().unwrap();
        assert_eq!(raw.headers, vec!["a", "b"]);
        assert_eq!(raw.rows, vec![vec!["1", "2"], vec!["3", "4"]]);
        assert_eq!(raw.row_ids, vec![1, 2]);
    }

    /// A comment is dropped wherever it is — above the header and between two
    /// data rows — and the row numbers count the rows that remain.
    #[test]
    fn comment_prefix_drops_a_line_anywhere_in_the_file() {
        let (_f, src) = file(
            b"# licence\na,b\n1,2\n# midway\n3,4\n",
            r##"{"path": "x", "delimiter": ",", "comment_prefix": "#"}"##,
        );
        let raw = src.read_all().unwrap();
        assert_eq!(raw.headers, vec!["a", "b"]);
        assert_eq!(raw.rows, vec![vec!["1", "2"], vec!["3", "4"]]);
        // The row numbers count data rows: the comment between them is not one.
        assert_eq!(raw.row_ids, vec![1, 2]);
    }

    /// A blank line is not a row of one empty field. The line engine would
    /// otherwise turn every blank line — and the newline closing the last row
    /// of a file that ends in two of them — into a row of nulls that no CSV
    /// of the same data produces.
    #[test]
    fn a_blank_line_is_not_a_row_in_either_engine() {
        let (_f1, csv_engine) = file(b"a,b\n1,2\n\n3,4\n\n", r#"{"path": "x", "delimiter": ","}"#);
        let raw = csv_engine.read_all().unwrap();
        assert_eq!(raw.rows, vec![vec!["1", "2"], vec!["3", "4"]]);
        assert_eq!(raw.row_ids, vec![1, 2]);

        let (_f2, line_engine) = file(
            b"a::b\n1::2\n\n3::4\n\n",
            r#"{"path": "x", "delimiter": "::"}"#,
        );
        assert_eq!(shape(&line_engine.read_all().unwrap()), shape(&raw));
    }

    /// The `csv` crate strips a BOM for us; the line engine has to do it
    /// itself, or the first column's name silently carries three bytes no
    /// `pk` will ever match.
    #[test]
    fn a_utf8_bom_is_stripped_by_both_engines() {
        let mut single = UTF8_BOM.to_vec();
        single.extend_from_slice(b"id\tname\n1\tAlice\n");
        let (_f1, csv_engine) = file(&single, r#"{"path": "x", "delimiter": "\t"}"#);
        assert_eq!(csv_engine.read_all().unwrap().headers, vec!["id", "name"]);

        let mut multi = UTF8_BOM.to_vec();
        multi.extend_from_slice("id::name\n1::Alice\n".as_bytes());
        let (_f2, line_engine) = file(&multi, r#"{"path": "x", "delimiter": "::"}"#);
        assert_eq!(line_engine.read_all().unwrap().headers, vec!["id", "name"]);
    }

    #[test]
    fn prefix_strip_removes_a_prefix_from_its_own_column_only() {
        let content = "path:map00010\tcpd:C00022\npath:map00020\tC00024\n";
        let (_f, src) = file(
            content.as_bytes(),
            r#"{"path": "x", "delimiter": "\t", "header": false,
                "columns": ["pathway", "compound"],
                "prefix_strip": {"pathway": "path:", "compound": "cpd:"}}"#,
        );
        let raw = src.read_all().unwrap();
        assert_eq!(raw.rows[0], vec!["map00010", "C00022"]);
        // A cell without the prefix is its own value, not an error.
        assert_eq!(raw.rows[1], vec!["map00020", "C00024"]);
    }

    #[test]
    fn latin1_decodes_the_high_bytes() {
        // 0xE9 is `é` in latin-1 and not valid UTF-8 on its own.
        let content = b"name\nAndr\xe9\n";
        let (_f, src) = file(
            content,
            r#"{"path": "x", "delimiter": ",", "encoding": "latin-1"}"#,
        );
        assert_eq!(src.read_all().unwrap().rows[0], vec!["André"]);

        let (_f2, utf8) = file(content, r#"{"path": "x", "delimiter": ","}"#);
        let err = utf8.read_all().err().expect("the same bytes are not UTF-8");
        assert!(err.contains("line 2"), "{err}");
        assert!(err.contains("latin-1"), "{err}");
    }

    #[test]
    fn a_short_row_is_padded_and_extra_fields_are_dropped_by_both_engines() {
        let (_f1, csv_engine) = file(b"a,b,c\n1\n1,2,3,4\n", r#"{"path": "x", "delimiter": ","}"#);
        let raw = csv_engine.read_all().unwrap();
        assert_eq!(raw.rows[0], vec!["1", "", ""]);
        assert_eq!(raw.nulls[0], vec![false, true, true]);
        assert_eq!(raw.rows[1], vec!["1", "2", "3"]);

        let (_f2, line_engine) = file(
            "a::b::c\n1\n1::2::3::4\n".as_bytes(),
            r#"{"path": "x", "delimiter": "::"}"#,
        );
        assert_eq!(shape(&line_engine.read_all().unwrap()), shape(&raw));
    }

    #[test]
    fn crlf_endings_load_through_both_engines() {
        let (_f1, csv_engine) = file(b"a,b\r\n1,2\r\n", r#"{"path": "x", "delimiter": ","}"#);
        let raw = csv_engine.read_all().unwrap();
        assert_eq!(raw.headers, vec!["a", "b"]);
        assert_eq!(raw.rows, vec![vec!["1", "2"]]);

        let (_f2, line_engine) = file(b"a::b\r\n1::2\r\n", r#"{"path": "x", "delimiter": "::"}"#);
        assert_eq!(shape(&line_engine.read_all().unwrap()), shape(&raw));
    }

    #[test]
    fn quoting_is_the_csv_engines_and_the_line_engine_has_none() {
        let (_f1, csv_engine) = file(
            b"a,b\n\"one, two\",3\n",
            r#"{"path": "x", "delimiter": ","}"#,
        );
        assert_eq!(
            csv_engine.read_all().unwrap().rows[0],
            vec!["one, two", "3"]
        );

        let (_f2, line_engine) = file(
            "a::b::c\n\"one::two\"::3\n".as_bytes(),
            r#"{"path": "x", "delimiter": "::"}"#,
        );
        assert_eq!(
            line_engine.read_all().unwrap().rows[0],
            vec!["\"one", "two\"", "3"]
        );
    }

    #[test]
    fn chunks_can_be_called_twice_and_yield_identical_batches() {
        let mut content = String::from("a<->b\n");
        for i in 0..25 {
            content.push_str(&format!("{i}<->{i}\n"));
        }
        let (_f, src) = file(content.as_bytes(), r#"{"path": "x", "delimiter": "<->"}"#);
        let batches = |src: &DelimitedFile| -> Vec<(Vec<usize>, Vec<Vec<String>>)> {
            src.chunks(10)
                .unwrap()
                .map(|c| {
                    let c = c.unwrap();
                    (c.row_ids.clone(), c.rows.clone())
                })
                .collect()
        };
        let first = batches(&src);
        assert_eq!(first.len(), 3);
        assert_eq!(first[2].0, vec![21, 22, 23, 24, 25]);
        assert_eq!(first, batches(&src));
    }

    /// The chunked pass and the whole-file pass are one table split two ways.
    #[test]
    fn chunks_and_read_all_agree() {
        let mut content = String::from("a\tb\n");
        for i in 0..17 {
            content.push_str(&format!("{i}\t{i}\n"));
        }
        let (_f, src) = file(content.as_bytes(), r#"{"path": "x", "delimiter": "\t"}"#);
        let whole = src.read_all().unwrap();
        let mut streamed_rows = Vec::new();
        let mut streamed_ids = Vec::new();
        for chunk in src.chunks(5).unwrap() {
            let chunk = chunk.unwrap();
            assert_eq!(chunk.headers, whole.headers);
            streamed_rows.extend(chunk.rows);
            streamed_ids.extend(chunk.row_ids);
        }
        assert_eq!(streamed_rows, whole.rows);
        assert_eq!(streamed_ids, whole.row_ids);
    }

    /// `scan_columns` is an optimisation of the read the trait default does;
    /// an override that saw different cells would type columns differently
    /// from the table they land in.
    #[test]
    fn scan_columns_sees_what_read_all_holds() {
        let content = "id::name::score\n1::Alice::3\n2::::4\n3::Cara::\n";
        let (_f, src) = file(content.as_bytes(), r#"{"path": "x", "delimiter": "::"}"#);
        let raw = src.read_all().unwrap();
        let wanted = ["score", "name", "absent"];
        let mut expected = Vec::new();
        for r in 0..raw.row_count() {
            for (slot, name) in wanted.iter().enumerate() {
                let Some(idx) = raw.col_index(name) else {
                    continue;
                };
                if !raw.nulls[r][idx] {
                    expected.push((slot, raw.rows[r][idx].clone()));
                }
            }
        }
        let mut seen = Vec::new();
        src.scan_columns(&wanted, &mut |slot, cell| {
            seen.push((slot, cell.to_string()));
            true
        })
        .unwrap();
        assert_eq!(seen, expected);
        // Non-vacuity: two cells in the first row, one in each of the others,
        // and the column the file does not have is never visited.
        assert_eq!(seen.len(), 4, "{seen:?}");
    }

    /// The projected scan must strip the same prefixes the table does, or a
    /// column's type is inferred from text no property ever holds.
    #[test]
    fn scan_columns_strips_the_prefixes_too() {
        let content = "cpd:1\ncpd:2\n";
        let (_f, src) = file(
            content.as_bytes(),
            r#"{"path": "x", "delimiter": "\t", "header": false, "columns": ["compound"],
                "prefix_strip": {"compound": "cpd:"}}"#,
        );
        let mut seen = Vec::new();
        src.scan_columns(&["compound"], &mut |_, cell| {
            seen.push(cell.to_string());
            true
        })
        .unwrap();
        assert_eq!(seen, vec!["1", "2"]);
    }

    #[test]
    fn scan_columns_stops_when_the_sink_says_stop() {
        let (_f, src) = file(b"a\n1\n2\n3\n", r#"{"path": "x", "delimiter": ","}"#);
        let mut visits = 0usize;
        src.scan_columns(&["a"], &mut |_, _| {
            visits += 1;
            false
        })
        .unwrap();
        assert_eq!(visits, 1);
    }

    #[test]
    fn a_file_that_is_not_there_reports_its_declared_name() {
        let src = DelimitedFile::new(
            PathBuf::from("/nonexistent/not-here.dmp"),
            "nodes.dmp".to_string(),
            config(r#"{"path": "x", "delimiter": "\t"}"#),
        );
        assert_eq!(src.size_hint(), None);
        let err = src.read_all().err().expect("a missing file fails to read");
        assert!(err.contains("delimited open nodes.dmp"), "{err}");
    }

    #[test]
    fn a_declaration_without_a_delimiter_says_what_it_is_for() {
        let err = config_err(r#"{"path": "x"}"#);
        assert!(err.contains("needs a 'delimiter'"), "{err}");
    }

    #[test]
    fn every_knob_is_checked_before_a_file_is_opened() {
        for (json, expected) in [
            (r#"{"delimiter": ""}"#, "'delimiter' is empty"),
            (r#"{"delimiter": 9}"#, "'delimiter' must be a string"),
            (r#"{"delimiter": ",", "quote": "''"}"#, "single ASCII"),
            (
                r#"{"delimiter": "\t|\t", "quote": "\"", "header": false, "columns": ["a"]}"#,
                "cannot be used with the multi-character delimiter",
            ),
            (
                r#"{"delimiter": ",", "columns": ["a"]}"#,
                "together with \"header\": true",
            ),
            (r#"{"delimiter": ",", "header": false}"#, "needs 'columns'"),
            (
                r#"{"delimiter": ",", "header": false, "columns": []}"#,
                "'columns' is empty",
            ),
            (
                r#"{"delimiter": ",", "header": false, "columns": ["a", "a"]}"#,
                "names 'a' twice",
            ),
            (
                r#"{"delimiter": ",", "header": false, "columns": "a"}"#,
                "must be a list of column names",
            ),
            (
                r#"{"delimiter": ",", "header": false, "columns": [1]}"#,
                "every entry of 'columns' must be a column name",
            ),
            (
                r#"{"delimiter": ",", "header": 1}"#,
                "must be true or false",
            ),
            (r#"{"delimiter": ",", "skip_lines": -1}"#, "not negative"),
            (
                r#"{"delimiter": ",", "skip_lines": "2"}"#,
                "must be a whole number of lines",
            ),
            (
                r#"{"delimiter": ",", "comment_prefix": ""}"#,
                "'comment_prefix' is empty",
            ),
            (
                r#"{"delimiter": ",", "line_suffix": ""}"#,
                "'line_suffix' is empty",
            ),
            (r#"{"delimiter": ",", "encoding": "utf-16"}"#, "'latin-1'"),
            (
                r#"{"delimiter": ",", "prefix_strip": ["a"]}"#,
                "must be an object of column name",
            ),
            (
                r#"{"delimiter": ",", "prefix_strip": {"a": 1}}"#,
                "entry 'a' must be text",
            ),
            (
                r#"{"delimiter": ",", "prefix_strip": {"a": ""}}"#,
                "entry 'a' is empty",
            ),
        ] {
            let err = config_err(json);
            assert!(err.contains(expected), "{json}\n  got: {err}");
            assert!(err.starts_with("files 't':"), "{json}\n  got: {err}");
        }
    }

    #[test]
    fn the_encoding_name_is_spelled_liberally() {
        for spelling in ["utf-8", "UTF8", "utf_8"] {
            let json = format!(r#"{{"delimiter": ",", "encoding": "{spelling}"}}"#);
            assert_eq!(config(&json).encoding, Encoding::Utf8, "{spelling}");
        }
        for spelling in ["latin-1", "Latin1", "ISO-8859-1"] {
            let json = format!(r#"{{"delimiter": ",", "encoding": "{spelling}"}}"#);
            assert_eq!(config(&json).encoding, Encoding::Latin1, "{spelling}");
        }
    }

    #[test]
    fn the_engine_is_chosen_by_the_delimiters_width() {
        assert_eq!(
            config(r#"{"delimiter": "\t"}"#).single_byte_delimiter(),
            Some(b'\t')
        );
        assert_eq!(
            config(r#"{"delimiter": "\t|\t", "header": false, "columns": ["a"]}"#)
                .single_byte_delimiter(),
            None
        );
        // A one-character multi-byte delimiter is still the line engine: the
        // `csv` crate takes a byte, not a `char`.
        assert_eq!(
            config(r#"{"delimiter": "→", "header": false, "columns": ["a"]}"#)
                .single_byte_delimiter(),
            None
        );
    }
}
