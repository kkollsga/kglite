//! The `xlsx` `Source`: one worksheet of an Excel workbook, read through
//! `calamine`.
//!
//! Behind the `xlsx` Cargo feature — the Python wheel enables it; the bare
//! crate does not, so a build without it refuses `"format": "xlsx"` by name
//! rather than pulling a zip reader and an XML parser into every dependency
//! tree.
//!
//! ## Why a spreadsheet is not just another table
//!
//! Published supplementary data is xlsx, and three of its habits break a
//! reader that treats a sheet as a CSV with a different envelope:
//!
//! - **A title block above the header.** `header_row` names the physical row
//!   the column names are on; everything above it is dropped unread.
//! - **Every number is a float.** Excel has one numeric type, so an id column
//!   of `260, 261, …` arrives as `260.0, 261.0, …` and a naive stringifier
//!   makes `Drug 260.0` — a node whose key matches nothing. An integral float
//!   is therefore written as an integer (see [`cell_text`]).
//! - **A wide result matrix.** One row per compound and one *column* per
//!   isolate is how a screen is published, and it is the wrong shape for a
//!   graph. `unpivot` turns those columns into rows, which is exactly the
//!   junction table the blueprint wants.
//!
//! ## Cell → text
//!
//! The sheet is typed, and the string table is not, so every cell is rendered
//! by one rule chosen to survive the loader's type inference:
//!
//! | cell | text |
//! |---|---|
//! | `Int` | `{}` — `5` |
//! | `Float`, integral and below 2^53 | integer text — `260.0` → `260` |
//! | `Float`, otherwise | `{:?}` — `0.5`, `NaN`, `inf` |
//! | `Bool` | `true` / `false` |
//! | `DateTime` (date only) | `2024-03-01` |
//! | `DateTime` (with a time) | `2024-03-01T09:30:00` |
//! | `DateTime` (a duration) | the number of days, as a float |
//! | `DateTimeIso` / `DurationIso` | the ISO text the sheet stored |
//! | `String` | as-is; an empty one is null |
//! | `Empty` | null |
//! | `Error` | null, plus one warning per column |
//!
//! The 2^53 bound is where an `f64` stops being able to tell consecutive
//! integers apart: above it, "this float is integral" no longer implies the
//! integer it prints is the one the author typed, so the float text is the
//! honest answer.
//!
//! ## Row numbers
//!
//! `row_ids` counts **data rows** below the header, 1-based — the same thing
//! the CSV and delimited readers count. Errors and warnings that can name a
//! physical location name the sheet and the cell (`sheet 'S3a' cell E7`),
//! because that is what the author clicks on. An unpivoted row keeps the
//! `row_id` of the sheet row it came from, so several output rows share one.

use super::super::schema::FileSpec;
use super::super::table::RawCsv;
use super::knobs::{first_duplicate, get_string, get_string_list, get_usize, json_kind, Extra};
use super::{FormatSpec, Source};
use crate::datatypes::values::ColumnType;
use calamine::{Data, Range, Reader, Xlsx};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Keys a `files` entry with `"format": "xlsx"` reads.
pub const ACCEPTED_FILE_KEYS_XLSX: &[&str] = &["path", "format", "sheet", "header_row", "unpivot"];

/// The knobs above that live in `FileSpec::extra`.
pub const KNOB_KEYS_XLSX: &[&str] = &["sheet", "header_row", "unpivot"];

pub const FORMAT: FormatSpec = FormatSpec {
    name: "xlsx",
    accepted_keys: ACCEPTED_FILE_KEYS_XLSX,
    knob_keys: KNOB_KEYS_XLSX,
    validate_entry,
};

/// Every `xlsx` rule that does not need the workbook open is a rule about its
/// config, so validating an entry is building one and discarding it.
fn validate_entry(name: &str, file: &FileSpec) -> Result<(), String> {
    XlsxConfig::from_spec(name, file).map(|_| ())
}

/// Which worksheet to read. Excel's own tab order is the index order, so a
/// number is a legitimate way to name the first sheet of a workbook whose tab
/// title carries a version number nobody wants in a blueprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SheetRef {
    Index(usize),
    Name(String),
}

/// Turn the columns outside `id_columns` into rows: one output row per input
/// row per such column, carrying the column's header under `name_to` and its
/// cell under `value_to`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unpivot {
    pub id_columns: Vec<String>,
    pub name_to: String,
    pub value_to: String,
}

/// Keys the `unpivot` object reads. Unlike a `files` entry's own stray keys,
/// which warn, one in here is refused: a misspelled `id_columns` would
/// silently unpivot the id columns too and produce a table of the right shape
/// and the wrong content.
const ACCEPTED_UNPIVOT_KEYS: &[&str] = &["id_columns", "name_to", "value_to"];

/// A validated `xlsx` declaration. Constructing one is the whole
/// declaration-time validation, so `validate_inputs` and the registry agree by
/// construction; the rules that need the sheet's header live in
/// [`XlsxFile::materialise`].
#[derive(Clone, Debug)]
pub struct XlsxConfig {
    sheet: SheetRef,
    /// 1-based physical row the column names are on. Rows above it are
    /// dropped without being read.
    header_row: usize,
    unpivot: Option<Unpivot>,
}

impl XlsxConfig {
    pub fn from_spec(name: &str, file: &FileSpec) -> Result<Self, String> {
        let extra = &file.extra;
        Ok(Self {
            sheet: Self::parse_sheet(name, extra)?,
            header_row: Self::parse_header_row(name, extra)?,
            unpivot: Self::parse_unpivot(name, extra)?,
        })
    }

    fn parse_sheet(name: &str, extra: &Extra) -> Result<SheetRef, String> {
        match extra.get("sheet") {
            None => Ok(SheetRef::Index(0)),
            Some(serde_json::Value::String(s)) => Ok(SheetRef::Name(s.clone())),
            Some(serde_json::Value::Number(n)) => match n.as_u64() {
                Some(i) => Ok(SheetRef::Index(i as usize)),
                None => Err(format!(
                    "files '{name}': 'sheet' must be a sheet name, or its position counting from \
                     0, but it is {n}."
                )),
            },
            Some(v) => Err(format!(
                "files '{name}': 'sheet' must be a sheet name, or its position counting from 0, \
                 but it is {}.",
                json_kind(v)
            )),
        }
    }

    fn parse_header_row(name: &str, extra: &Extra) -> Result<usize, String> {
        match get_usize(name, extra, "header_row", "rows")? {
            None => Ok(1),
            // Row numbers in a spreadsheet start at 1, and that is the number
            // the author reads off the sheet's own row gutter.
            Some(0) => Err(format!(
                "files '{name}': 'header_row' counts rows from 1, the way the spreadsheet does, \
                 so 0 is not a row. The first row is 1."
            )),
            Some(n) => Ok(n),
        }
    }

    fn parse_unpivot(name: &str, extra: &Extra) -> Result<Option<Unpivot>, String> {
        let Some(v) = extra.get("unpivot") else {
            return Ok(None);
        };
        let serde_json::Value::Object(map) = v else {
            return Err(format!(
                "files '{name}': 'unpivot' must be an object with 'id_columns', 'name_to' and \
                 'value_to', but it is {}.",
                json_kind(v)
            ));
        };
        for key in map.keys() {
            if !ACCEPTED_UNPIVOT_KEYS.contains(&key.as_str()) {
                return Err(format!(
                    "files '{name}': 'unpivot' has no key '{key}' — it reads {}.",
                    quoted_list(ACCEPTED_UNPIVOT_KEYS)
                ));
            }
        }
        // `Extra` and a serde_json object are both string-keyed maps; going
        // through one lets the shared shape errors say the same thing here.
        let inner: Extra = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let id_columns = get_string_list(name, &inner, "id_columns")?.ok_or_else(|| {
            format!(
                "files '{name}': 'unpivot' needs 'id_columns' — the columns that stay columns; \
                 every other column becomes rows."
            )
        })?;
        if id_columns.is_empty() {
            return Err(format!(
                "files '{name}': 'unpivot' has an empty 'id_columns' — every column would become \
                 rows and no row would carry what it is about."
            ));
        }
        if let Some(dup) = first_duplicate(&id_columns) {
            return Err(format!(
                "files '{name}': 'unpivot' names '{dup}' twice in 'id_columns'."
            ));
        }
        let name_to = required_name(name, &inner, "name_to")?;
        let value_to = required_name(name, &inner, "value_to")?;
        if name_to == value_to {
            return Err(format!(
                "files '{name}': 'unpivot' gives 'name_to' and 'value_to' the same name \
                 '{name_to}' — one column cannot hold both the header and the cell."
            ));
        }
        for new in [&name_to, &value_to] {
            if id_columns.iter().any(|c| c == new) {
                return Err(format!(
                    "files '{name}': 'unpivot' writes '{new}', which is already an id column — \
                     the output would carry that name twice."
                ));
            }
        }
        Ok(Some(Unpivot {
            id_columns,
            name_to,
            value_to,
        }))
    }
}

fn required_name(name: &str, inner: &Extra, key: &str) -> Result<String, String> {
    match get_string(name, inner, key)? {
        Some(s) if !s.is_empty() => Ok(s),
        Some(_) => Err(format!(
            "files '{name}': 'unpivot' has an empty '{key}' — it names an output column."
        )),
        None => Err(format!(
            "files '{name}': 'unpivot' needs '{key}' — it names an output column."
        )),
    }
}

fn quoted_list(items: &[&str]) -> String {
    items
        .iter()
        .map(|i| format!("'{i}'"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One cell's text, or `None` for a cell the string table calls null.
///
/// See the module header's table; the integral-float rule is the one with
/// user-visible consequences, and `screen.xlsx` pins it end to end.
fn cell_text(cell: &Data) -> Option<String> {
    match cell {
        Data::Empty => None,
        Data::Int(i) => Some(i.to_string()),
        Data::Float(f) => Some(float_text(*f)),
        Data::Bool(b) => Some(b.to_string()),
        Data::String(s) if s.is_empty() => None,
        Data::String(s) => Some(s.clone()),
        Data::DateTime(dt) => Some(datetime_text(dt)),
        Data::DateTimeIso(s) | Data::DurationIso(s) => Some(s.clone()),
        // Reported by the caller, which knows the column and the cell.
        Data::Error(_) => None,
    }
}

/// Largest integer an `f64` can hold without sharing its representation with
/// its neighbour. Below it an integral float prints the integer that was
/// typed; at or above it, it may not.
const F64_EXACT_INT_LIMIT: f64 = 9_007_199_254_740_992.0; // 2^53

fn float_text(f: f64) -> String {
    if f.fract() == 0.0 && f.abs() < F64_EXACT_INT_LIMIT {
        format!("{}", f as i64)
    } else {
        // f64's shortest round-tripping form, which keeps the decimal point so
        // `1.5` is not re-read as something else.
        format!("{f:?}")
    }
}

fn datetime_text(dt: &calamine::ExcelDateTime) -> String {
    // A duration is a span, not an instant; there is no date to print, so the
    // number of days is what it is. The blueprint has no duration type either
    // way, so this lands as a float property.
    if dt.is_duration() {
        return float_text(dt.as_f64());
    }
    match dt.as_datetime() {
        // A date cell and a datetime cell are the same stored type in Excel;
        // only the time part tells them apart, and printing `T00:00:00` on
        // every date would make none of them parse as the blueprint's `date`.
        Some(ndt) if ndt.time() == chrono::NaiveTime::MIN => ndt.format("%Y-%m-%d").to_string(),
        Some(ndt) => ndt.format("%Y-%m-%dT%H:%M:%S").to_string(),
        None => float_text(dt.as_f64()),
    }
}

/// What one cell says about its column's type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CellClass {
    /// Contributes nothing — a null.
    Skip,
    Int,
    Float,
    Bool,
    Date,
    /// Anything the blueprint's inference should decide for itself.
    Other,
}

fn cell_class(cell: &Data) -> CellClass {
    match cell {
        Data::Empty | Data::Error(_) => CellClass::Skip,
        Data::String(s) if s.is_empty() => CellClass::Skip,
        Data::Int(_) => CellClass::Int,
        Data::Float(f) => {
            if f.fract() == 0.0 && f.abs() < F64_EXACT_INT_LIMIT {
                CellClass::Int
            } else {
                CellClass::Float
            }
        }
        Data::Bool(_) => CellClass::Bool,
        Data::DateTime(dt) if !dt.is_duration() => CellClass::Date,
        _ => CellClass::Other,
    }
}

/// A column's type as its cells agree on it, accumulated one cell at a time.
#[derive(Default, Clone, Copy)]
struct ColumnVote {
    seen: Option<CellClass>,
    mixed: bool,
}

impl ColumnVote {
    fn cast(&mut self, class: CellClass) {
        if self.mixed || class == CellClass::Skip {
            return;
        }
        match self.seen {
            None => self.seen = Some(class),
            // Int and Float are the same column with a decimal point in it
            // somewhere; every other disagreement is a column the loader
            // should infer for itself.
            Some(CellClass::Int) if class == CellClass::Float => self.seen = Some(CellClass::Float),
            Some(CellClass::Float) if class == CellClass::Int => {}
            Some(prev) if prev == class => {}
            Some(_) => self.mixed = true,
        }
    }

    fn resolve(self) -> Option<ColumnType> {
        if self.mixed {
            return None;
        }
        match self.seen? {
            CellClass::Int => Some(ColumnType::Int64),
            CellClass::Float => Some(ColumnType::Float64),
            CellClass::Bool => Some(ColumnType::Boolean),
            CellClass::Date => Some(ColumnType::DateTime),
            CellClass::Other | CellClass::Skip => None,
        }
    }
}

/// A sheet read into the string table, with what its typed cells said about
/// their columns and anything worth telling the author about.
struct Materialised {
    table: RawCsv,
    known: HashMap<String, ColumnType>,
    warnings: Vec<String>,
}

/// One worksheet of an `.xlsx` workbook.
///
/// The sheet is read on first use and kept: `calamine` materialises a whole
/// worksheet to hand out any of it, so there is nothing to stream from, and
/// re-reading it per consumer would mean unzipping and re-parsing the workbook
/// for each phase that touches the input.
pub struct XlsxFile {
    path: PathBuf,
    display: String,
    config: XlsxConfig,
    sheet: OnceLock<Result<Materialised, String>>,
}

impl XlsxFile {
    pub fn new(path: PathBuf, display: String, config: XlsxConfig) -> Self {
        Self {
            path,
            display,
            config,
            sheet: OnceLock::new(),
        }
    }

    fn loaded(&self) -> Result<&Materialised, String> {
        self.sheet
            .get_or_init(|| self.materialise())
            .as_ref()
            .map_err(Clone::clone)
    }

    fn materialise(&self) -> Result<Materialised, String> {
        let mut workbook: Xlsx<_> = calamine::open_workbook(&self.path)
            .map_err(|e| format!("xlsx open {}: {e}", self.display))?;
        let names = workbook.sheet_names();
        let sheet_name = self.resolve_sheet(&names)?;
        let range = workbook
            .worksheet_range(&sheet_name)
            .map_err(|e| format!("xlsx {} sheet '{sheet_name}': {e}", self.display))?;
        let ctx = SheetCtx {
            display: &self.display,
            sheet: &sheet_name,
            // A range starts at the first populated cell, so a sheet with
            // blank leading rows or columns reports its own offset; every
            // physical row/column number below is relative to it.
            origin: range.start().unwrap_or((0, 0)),
        };
        let header_index = self.header_index(&ctx, &range)?;
        let headers = read_headers(&range, header_index);
        if headers.is_empty() {
            return Err(format!(
                "xlsx {} sheet '{}' row {}: the header row is empty — 'header_row' names the \
                 physical row the column names are on, counting from 1.",
                self.display, sheet_name, self.config.header_row
            ));
        }
        match &self.config.unpivot {
            Some(unpivot) => build_unpivoted(&ctx, &range, header_index, &headers, unpivot),
            None => Ok(build_wide(&ctx, &range, header_index, &headers)),
        }
    }

    fn resolve_sheet(&self, names: &[String]) -> Result<String, String> {
        let listed = || {
            if names.is_empty() {
                "it has none".to_string()
            } else {
                format!("it has {}", quoted_list_owned(names))
            }
        };
        match &self.config.sheet {
            SheetRef::Name(want) => names.iter().find(|n| *n == want).cloned().ok_or_else(|| {
                format!(
                    "xlsx {}: no sheet named '{want}' — {}.",
                    self.display,
                    listed()
                )
            }),
            SheetRef::Index(i) => names.get(*i).cloned().ok_or_else(|| {
                format!(
                    "xlsx {}: no sheet at position {i} (counting from 0) — {}.",
                    self.display,
                    listed()
                )
            }),
        }
    }

    /// The header's index inside the range, from its physical row number.
    fn header_index(&self, ctx: &SheetCtx, range: &Range<Data>) -> Result<usize, String> {
        let physical = self.config.header_row;
        let first = ctx.origin.0 as usize + 1;
        if physical < first || physical - first >= range.height() {
            return Err(format!(
                "xlsx {} sheet '{}': 'header_row' is {physical}, but the sheet's rows with \
                 content are {first}–{}.",
                self.display,
                ctx.sheet,
                first + range.height().saturating_sub(1)
            ));
        }
        Ok(physical - first)
    }
}

/// What a diagnostic needs to name a place in the workbook.
struct SheetCtx<'a> {
    display: &'a str,
    sheet: &'a str,
    /// Absolute (row, column) of the range's top-left cell, 0-based.
    origin: (u32, u32),
}

impl SheetCtx<'_> {
    /// `E7` — the reference the author sees in the spreadsheet's own name box.
    fn cell_ref(&self, row_index: usize, col_index: usize) -> String {
        let row = self.origin.0 as usize + row_index + 1;
        format!(
            "{}{row}",
            column_letters(self.origin.1 as usize + col_index)
        )
    }
}

/// Excel's bijective base-26 column names: 0 → `A`, 25 → `Z`, 26 → `AA`.
fn column_letters(mut index: usize) -> String {
    let mut out = Vec::new();
    loop {
        out.push(b'A' + (index % 26) as u8);
        if index < 26 {
            break;
        }
        index = index / 26 - 1;
    }
    out.reverse();
    String::from_utf8(out).expect("ASCII letters")
}

fn quoted_list_owned(items: &[String]) -> String {
    items
        .iter()
        .map(|i| format!("'{i}'"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The header row's names, paired with the range column each came from.
///
/// A column whose header cell is empty is dropped: nothing can reference it by
/// name, and a trailing run of them is what a spreadsheet leaves behind when
/// someone widens a formatted table.
fn read_headers(range: &Range<Data>, header_index: usize) -> Vec<(usize, String)> {
    (0..range.width())
        .filter_map(|c| {
            let cell = range.get((header_index, c))?;
            let text = cell_text(cell)?;
            let text = text.trim();
            if text.is_empty() {
                None
            } else {
                Some((c, text.to_string()))
            }
        })
        .collect()
}

/// Warnings for cells the sheet itself failed to compute, one line per column
/// naming the first one and how many followed.
#[derive(Default)]
struct ErrorTally {
    hits: Vec<(String, usize, String, String)>,
}

impl ErrorTally {
    fn record(&mut self, column: &str, cell_ref: String, error: String) {
        if let Some(hit) = self.hits.iter_mut().find(|(c, _, _, _)| c == column) {
            hit.1 += 1;
            return;
        }
        self.hits.push((column.to_string(), 1, cell_ref, error));
    }

    fn into_warnings(self, ctx: &SheetCtx) -> Vec<String> {
        self.hits
            .into_iter()
            .map(|(column, count, cell_ref, error)| {
                let more = if count > 1 {
                    format!(" ({count} cells in this column)")
                } else {
                    String::new()
                };
                format!(
                    "xlsx {} sheet '{}' column '{column}': cell {cell_ref} is {error}{more}; \
                     stored as null",
                    ctx.display, ctx.sheet
                )
            })
            .collect()
    }
}

/// The sheet as it is laid out: one output row per data row, one output column
/// per named header column.
fn build_wide(
    ctx: &SheetCtx,
    range: &Range<Data>,
    header_index: usize,
    headers: &[(usize, String)],
) -> Materialised {
    let mut rows = Vec::new();
    let mut nulls = Vec::new();
    let mut votes = vec![ColumnVote::default(); headers.len()];
    let mut tally = ErrorTally::default();

    for r in (header_index + 1)..range.height() {
        let mut row = Vec::with_capacity(headers.len());
        let mut nrow = Vec::with_capacity(headers.len());
        for (slot, (c, name)) in headers.iter().enumerate() {
            let cell = range.get((r, *c)).unwrap_or(&Data::Empty);
            note_error(&mut tally, ctx, name, r, *c, cell);
            votes[slot].cast(cell_class(cell));
            match cell_text(cell) {
                Some(text) => {
                    nrow.push(text.is_empty());
                    row.push(text);
                }
                None => {
                    row.push(String::new());
                    nrow.push(true);
                }
            }
        }
        rows.push(row);
        nulls.push(nrow);
    }

    let known = headers
        .iter()
        .zip(&votes)
        .filter_map(|((_, name), vote)| vote.resolve().map(|t| (name.clone(), t)))
        .collect();
    Materialised {
        table: RawCsv {
            headers: headers.iter().map(|(_, n)| n.clone()).collect(),
            row_ids: (1..=rows.len()).collect(),
            rows,
            nulls,
        },
        known,
        warnings: tally.into_warnings(ctx),
    }
}

/// The wide-matrix shape turned long: the `id_columns` stay columns, and every
/// other named column becomes one output row per input row carrying its header
/// and its cell.
///
/// An empty cell produces **no** row. A published screen matrix is sparse by
/// construction — the pairs that were not measured are blank — and emitting a
/// null-valued row for each would turn "not measured" into an edge.
fn build_unpivoted(
    ctx: &SheetCtx,
    range: &Range<Data>,
    header_index: usize,
    headers: &[(usize, String)],
    unpivot: &Unpivot,
) -> Result<Materialised, String> {
    let mut id_slots = Vec::with_capacity(unpivot.id_columns.len());
    for want in &unpivot.id_columns {
        let slot = headers.iter().position(|(_, n)| n == want).ok_or_else(|| {
            format!(
                "xlsx {} sheet '{}': 'unpivot' names id column '{want}', which the header row \
                 does not have — it has {}.",
                ctx.display,
                ctx.sheet,
                quoted_list_owned(&headers.iter().map(|(_, n)| n.clone()).collect::<Vec<_>>())
            )
        })?;
        id_slots.push(slot);
    }
    let value_slots: Vec<usize> = (0..headers.len())
        .filter(|s| !id_slots.contains(s))
        .collect();
    if value_slots.is_empty() {
        return Err(format!(
            "xlsx {} sheet '{}': 'unpivot' lists every column as an id column, so there is \
             nothing left to turn into rows.",
            ctx.display, ctx.sheet
        ));
    }

    let out_headers: Vec<String> = unpivot
        .id_columns
        .iter()
        .cloned()
        .chain([unpivot.name_to.clone(), unpivot.value_to.clone()])
        .collect();
    let mut rows = Vec::new();
    let mut nulls = Vec::new();
    let mut row_ids = Vec::new();
    // One vote per id column, plus one for the name column (always header
    // text) and one for the value column (every unpivoted cell, together).
    let mut votes = vec![ColumnVote::default(); out_headers.len()];
    let name_vote = id_slots.len();
    let value_vote = name_vote + 1;
    let mut tally = ErrorTally::default();

    for (data_row, r) in ((header_index + 1)..range.height()).enumerate() {
        let mut id_cells: Vec<(String, bool)> = Vec::with_capacity(id_slots.len());
        for (vote, slot) in id_slots.iter().enumerate() {
            let (c, name) = &headers[*slot];
            let cell = range.get((r, *c)).unwrap_or(&Data::Empty);
            note_error(&mut tally, ctx, name, r, *c, cell);
            votes[vote].cast(cell_class(cell));
            id_cells.push(match cell_text(cell) {
                Some(text) => (text, false),
                None => (String::new(), true),
            });
        }
        for slot in &value_slots {
            let (c, name) = &headers[*slot];
            let cell = range.get((r, *c)).unwrap_or(&Data::Empty);
            note_error(&mut tally, ctx, name, r, *c, cell);
            let Some(text) = cell_text(cell) else {
                continue;
            };
            if text.is_empty() {
                continue;
            }
            votes[value_vote].cast(cell_class(cell));
            votes[name_vote].cast(CellClass::Other);
            let mut row: Vec<String> = id_cells.iter().map(|(t, _)| t.clone()).collect();
            let mut nrow: Vec<bool> = id_cells.iter().map(|(_, n)| *n).collect();
            row.push(name.clone());
            nrow.push(false);
            row.push(text);
            nrow.push(false);
            rows.push(row);
            nulls.push(nrow);
            // The sheet row this pair came from, so several output rows share
            // the provenance of the one row an author can go and look at.
            row_ids.push(data_row + 1);
        }
    }

    let known = out_headers
        .iter()
        .zip(&votes)
        .filter_map(|(name, vote)| vote.resolve().map(|t| (name.clone(), t)))
        .collect();
    Ok(Materialised {
        table: RawCsv {
            headers: out_headers,
            rows,
            nulls,
            row_ids,
        },
        known,
        warnings: tally.into_warnings(ctx),
    })
}

fn note_error(
    tally: &mut ErrorTally,
    ctx: &SheetCtx,
    column: &str,
    row_index: usize,
    col_index: usize,
    cell: &Data,
) {
    if let Data::Error(e) = cell {
        tally.record(column, ctx.cell_ref(row_index, col_index), e.to_string());
    }
}

impl Source for XlsxFile {
    fn display_name(&self) -> &str {
        &self.display
    }

    /// `None`, which selects the buffered path. A worksheet is materialised
    /// whole by the parser before any of it can be read, so the file's size on
    /// disk says nothing about whether streaming would bound anything.
    fn size_hint(&self) -> Option<u64> {
        None
    }

    /// True. The rows are all in hand, so a chunked consumer (every junction
    /// loader) gets its chunks by slicing them.
    fn can_chunk(&self) -> bool {
        true
    }

    fn read_all(&self) -> Result<RawCsv, String> {
        let loaded = self.loaded()?;
        Ok(slice(&loaded.table, 0, loaded.table.rows.len()))
    }

    fn chunks(
        &self,
        chunk_size: usize,
    ) -> Result<Box<dyn Iterator<Item = Result<RawCsv, String>> + '_>, String> {
        let table = &self.loaded()?.table;
        let total = table.rows.len();
        let step = chunk_size.max(1);
        let mut next = 0usize;
        Ok(Box::new(std::iter::from_fn(move || {
            if next >= total {
                return None;
            }
            let end = (next + step).min(total);
            let chunk = slice(table, next, end);
            next = end;
            Some(Ok(chunk))
        })))
    }

    /// Reads straight out of the materialised table — the trait default would
    /// clone every chunk to look at a few columns.
    fn scan_columns(
        &self,
        columns: &[&str],
        visit: &mut dyn FnMut(usize, &str) -> bool,
    ) -> Result<(), String> {
        let table = &self.loaded()?.table;
        let indices: Vec<Option<usize>> = columns.iter().map(|n| table.col_index(n)).collect();
        for r in 0..table.rows.len() {
            for (slot, idx) in indices.iter().enumerate() {
                let Some(idx) = idx else { continue };
                if table.nulls[r][*idx] {
                    continue;
                }
                if !visit(slot, &table.rows[r][*idx]) {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn known_column_types(&self) -> HashMap<String, ColumnType> {
        match self.loaded() {
            Ok(loaded) => loaded.known.clone(),
            // A read failure is the read path's error to report; answering
            // "nothing known" here keeps it from being reported twice.
            Err(_) => HashMap::new(),
        }
    }

    fn read_warnings(&self) -> Vec<String> {
        match self.sheet.get() {
            Some(Ok(loaded)) => loaded.warnings.clone(),
            // Never opened, or opened and failed: no cell was read, so there
            // is nothing to say about one.
            _ => Vec::new(),
        }
    }
}

fn slice(table: &RawCsv, start: usize, end: usize) -> RawCsv {
    RawCsv {
        headers: table.headers.clone(),
        rows: table.rows[start..end].to_vec(),
        nulls: table.nulls[start..end].to_vec(),
        row_ids: table.row_ids[start..end].to_vec(),
    }
}

#[cfg(test)]
mod xlsx_tests {
    use super::*;
    use calamine::{CellErrorType, ExcelDateTime, ExcelDateTimeType};
    use std::path::Path;

    fn fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/blueprint/screen.xlsx")
    }

    fn spec(json: &str) -> FileSpec {
        serde_json::from_str(json).expect("files entry parses")
    }

    fn config(json: &str) -> XlsxConfig {
        XlsxConfig::from_spec("t", &spec(json)).expect("a valid declaration")
    }

    fn config_err(json: &str) -> String {
        XlsxConfig::from_spec("t", &spec(json)).expect_err("an invalid declaration")
    }

    fn source(json: &str) -> XlsxFile {
        XlsxFile::new(fixture(), "screen.xlsx".to_string(), config(json))
    }

    fn cell(src: &XlsxFile, row: usize, column: &str) -> (String, bool) {
        let raw = src.read_all().unwrap();
        let i = raw
            .col_index(column)
            .unwrap_or_else(|| panic!("no {column}"));
        (raw.rows[row][i].clone(), raw.nulls[row][i])
    }

    // ── cell text ────────────────────────────────────────────────────────

    #[test]
    fn an_integral_float_is_written_as_an_integer_and_a_fractional_one_is_not() {
        assert_eq!(float_text(260.0), "260");
        assert_eq!(float_text(-3.0), "-3");
        assert_eq!(float_text(0.5), "0.5");
        assert_eq!(float_text(1.5e-4), "0.00015");
    }

    /// At 2^53 an `f64` no longer holds every integer, so "integral" stops
    /// implying "the integer the author typed" and the float text is the
    /// honest answer.
    #[test]
    fn the_integer_rule_stops_at_two_to_the_fifty_third() {
        assert_eq!(float_text(9_007_199_254_740_991.0), "9007199254740991");
        assert_eq!(float_text(9_007_199_254_740_992.0), "9007199254740992.0");
        assert_eq!(float_text(-9_007_199_254_740_992.0), "-9007199254740992.0");
    }

    #[test]
    fn non_finite_floats_do_not_become_integers() {
        assert_eq!(float_text(f64::NAN), "NaN");
        assert_eq!(float_text(f64::INFINITY), "inf");
    }

    #[test]
    fn every_cell_variant_renders_under_its_own_rule() {
        assert_eq!(cell_text(&Data::Int(5)).as_deref(), Some("5"));
        assert_eq!(cell_text(&Data::Bool(true)).as_deref(), Some("true"));
        assert_eq!(cell_text(&Data::Bool(false)).as_deref(), Some("false"));
        assert_eq!(cell_text(&Data::String("x".into())).as_deref(), Some("x"));
        assert_eq!(
            cell_text(&Data::DateTimeIso("2024-03-01T09:30:00".into())).as_deref(),
            Some("2024-03-01T09:30:00")
        );
        assert_eq!(
            cell_text(&Data::DurationIso("PT1H".into())).as_deref(),
            Some("PT1H")
        );
        // Null, in three spellings.
        assert_eq!(cell_text(&Data::Empty), None);
        assert_eq!(cell_text(&Data::String(String::new())), None);
        assert_eq!(cell_text(&Data::Error(CellErrorType::Div0)), None);
    }

    /// A date and a datetime are one stored type in Excel; only the time part
    /// separates them, and a `T00:00:00` on every date would stop the
    /// blueprint's `date` keyword parsing any of them.
    #[test]
    fn a_datetime_prints_a_time_only_when_it_has_one() {
        // 45352.0 is 2024-03-01 in the 1900 epoch; +0.5 is midday.
        let date = ExcelDateTime::new(45352.0, ExcelDateTimeType::DateTime, false);
        let stamp = ExcelDateTime::new(45352.5, ExcelDateTimeType::DateTime, false);
        assert_eq!(datetime_text(&date), "2024-03-01");
        assert_eq!(datetime_text(&stamp), "2024-03-01T12:00:00");
        // A duration has no date to print at all.
        let dur = ExcelDateTime::new(1.5, ExcelDateTimeType::TimeDelta, false);
        assert_eq!(datetime_text(&dur), "1.5");
    }

    #[test]
    fn column_letters_follow_excels_bijective_base_26() {
        assert_eq!(column_letters(0), "A");
        assert_eq!(column_letters(25), "Z");
        assert_eq!(column_letters(26), "AA");
        assert_eq!(column_letters(27), "AB");
        assert_eq!(column_letters(701), "ZZ");
        assert_eq!(column_letters(702), "AAA");
    }

    // ── declaration validation ───────────────────────────────────────────

    #[test]
    fn the_defaults_are_the_first_sheet_and_the_first_row() {
        let c = config(r#"{"path": "x.xlsx", "format": "xlsx"}"#);
        assert_eq!(c.sheet, SheetRef::Index(0));
        assert_eq!(c.header_row, 1);
        assert!(c.unpivot.is_none());
    }

    #[test]
    fn a_sheet_is_named_or_positioned_and_nothing_else() {
        assert_eq!(
            config(r#"{"path": "x", "sheet": "S3a"}"#).sheet,
            SheetRef::Name("S3a".into())
        );
        assert_eq!(
            config(r#"{"path": "x", "sheet": 2}"#).sheet,
            SheetRef::Index(2)
        );
        assert!(config_err(r#"{"path": "x", "sheet": -1}"#).contains("but it is -1"));
        assert!(config_err(r#"{"path": "x", "sheet": true}"#).contains("but it is a boolean"));
    }

    #[test]
    fn header_row_counts_from_one() {
        assert_eq!(config(r#"{"path": "x", "header_row": 3}"#).header_row, 3);
        let err = config_err(r#"{"path": "x", "header_row": 0}"#);
        assert!(err.contains("counts rows from 1"), "{err}");
        assert!(config_err(r#"{"path": "x", "header_row": "3"}"#).contains("whole number of rows"));
    }

    #[test]
    fn unpivot_needs_all_three_of_its_keys() {
        assert!(config_err(r#"{"path": "x", "unpivot": {}}"#).contains("needs 'id_columns'"));
        assert!(
            config_err(r#"{"path": "x", "unpivot": {"id_columns": ["a"]}}"#)
                .contains("needs 'name_to'")
        );
        assert!(
            config_err(r#"{"path": "x", "unpivot": {"id_columns": ["a"], "name_to": "n"}}"#)
                .contains("needs 'value_to'")
        );
        assert!(config_err(r#"{"path": "x", "unpivot": ["a"]}"#).contains("must be an object"));
    }

    /// A misspelled key inside `unpivot` is refused, not warned about: a
    /// dropped `id_columns` unpivots the identifiers too and yields a table of
    /// the right shape and the wrong content.
    #[test]
    fn an_unknown_unpivot_key_is_refused() {
        let err = config_err(
            r#"{"path": "x", "unpivot": {"id_columns": ["a"], "name_to": "n",
                "value_to": "v", "id_column": ["a"]}}"#,
        );
        assert!(err.contains("'unpivot' has no key 'id_column'"), "{err}");
        assert!(err.contains("'id_columns', 'name_to', 'value_to'"), "{err}");
    }

    #[test]
    fn unpivot_refuses_output_names_that_collide() {
        let err = config_err(
            r#"{"path": "x", "unpivot": {"id_columns": ["a"], "name_to": "a", "value_to": "v"}}"#,
        );
        assert!(err.contains("already an id column"), "{err}");
        let err = config_err(
            r#"{"path": "x", "unpivot": {"id_columns": ["a"], "name_to": "v", "value_to": "v"}}"#,
        );
        assert!(err.contains("the same name 'v'"), "{err}");
    }

    #[test]
    fn unpivot_refuses_an_empty_or_repeated_id_column_list() {
        assert!(config_err(
            r#"{"path": "x", "unpivot": {"id_columns": [], "name_to": "n", "value_to": "v"}}"#
        )
        .contains("empty 'id_columns'"));
        assert!(config_err(
            r#"{"path": "x",
                "unpivot": {"id_columns": ["a", "a"], "name_to": "n", "value_to": "v"}}"#
        )
        .contains("names 'a' twice"));
    }

    // ── reading the committed workbook ───────────────────────────────────

    #[test]
    fn a_float_id_column_reads_back_as_integer_text() {
        let src = source(r#"{"path": "screen.xlsx", "sheet": "drugs"}"#);
        let raw = src.read_all().unwrap();
        assert_eq!(
            raw.headers,
            vec![
                "prestwick_ID",
                "chemical_name",
                "atc_code",
                "approved",
                "approved_on"
            ]
        );
        assert_eq!(raw.row_count(), 4);
        // The documented trap: Excel stores 260 as 260.0, and `Drug 260.0`
        // matches nothing.
        assert_eq!(cell(&src, 0, "prestwick_ID"), ("260".to_string(), false));
        assert_eq!(cell(&src, 3, "prestwick_ID"), ("263".to_string(), false));
        assert_eq!(cell(&src, 0, "approved"), ("true".to_string(), false));
        assert_eq!(cell(&src, 2, "approved"), ("false".to_string(), false));
        assert_eq!(
            cell(&src, 0, "approved_on"),
            ("1998-05-12".to_string(), false)
        );
        // The blank cell is null, not the text "None".
        assert_eq!(cell(&src, 2, "atc_code"), (String::new(), true));
        assert_eq!(raw.row_ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn the_sheets_own_types_are_reported_to_the_loader() {
        let src = source(r#"{"path": "screen.xlsx", "sheet": "drugs"}"#);
        let known = src.known_column_types();
        assert_eq!(known.get("prestwick_ID"), Some(&ColumnType::Int64));
        assert_eq!(known.get("approved"), Some(&ColumnType::Boolean));
        assert_eq!(known.get("approved_on"), Some(&ColumnType::DateTime));
        // Text columns are left to the loader's own inference, as are the
        // columns the sheet disagrees with itself about.
        assert_eq!(known.get("chemical_name"), None);
        assert_eq!(known.get("atc_code"), None);
    }

    #[test]
    fn header_row_skips_the_title_block_above_it() {
        let src = source(
            r#"{"path": "screen.xlsx", "sheet": "S3a. Adjusted p-values",
                             "header_row": 3}"#,
        );
        let raw = src.read_all().unwrap();
        assert_eq!(raw.headers.len(), 7);
        assert_eq!(raw.headers[0], "prestwick_ID");
        assert_eq!(raw.headers[4], "Akkermansia muciniphila (NT5021)");
        assert_eq!(raw.row_count(), 4);
        // Non-vacuity: reading the same sheet from row 1 sees the title
        // instead, so the skip is doing the work.
        let wrong = source(r#"{"path": "screen.xlsx", "sheet": "S3a. Adjusted p-values"}"#);
        assert_eq!(
            wrong.read_all().unwrap().headers,
            vec!["Supplementary Table S3a"]
        );
    }

    #[test]
    fn a_sheet_is_reachable_by_position_as_well_as_by_name() {
        let by_index = source(r#"{"path": "screen.xlsx", "sheet": 0}"#);
        let by_name = source(r#"{"path": "screen.xlsx", "sheet": "drugs"}"#);
        assert_eq!(
            by_index.read_all().unwrap().headers,
            by_name.read_all().unwrap().headers
        );
        assert_eq!(
            source(r#"{"path": "screen.xlsx", "sheet": 1}"#)
                .read_all()
                .unwrap()
                .headers,
            vec!["Supplementary Table S3a"]
        );
    }

    /// The workbook's third sheet carries real `#DIV/0!` cells, so the tally
    /// is pinned against a stored error, not only against a hand-built range.
    #[test]
    fn the_committed_workbooks_error_cells_are_null_and_reported_once() {
        let src = source(r#"{"path": "screen.xlsx", "sheet": "calc"}"#);
        let raw = src.read_all().unwrap();
        let ratio = raw.col_index("ratio").unwrap();
        assert_eq!(raw.rows[0][ratio], "");
        assert!(raw.nulls[0][ratio]);
        assert_eq!(raw.rows[1][ratio], "0.5");
        let warnings = src.read_warnings();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains("sheet 'calc' column 'ratio'"),
            "{warnings:?}"
        );
        assert!(warnings[0].contains("cell B2 is #DIV/0!"), "{warnings:?}");
        assert!(
            warnings[0].contains("(2 cells in this column)"),
            "{warnings:?}"
        );
    }

    /// A source nobody read has nothing to say — the warning channel is
    /// drained after the load phases, and an input the build never opened must
    /// not put a line in the report.
    #[test]
    fn an_unread_source_reports_no_warnings() {
        assert!(source(r#"{"path": "screen.xlsx", "sheet": "calc"}"#)
            .read_warnings()
            .is_empty());
    }

    #[test]
    fn a_missing_sheet_is_an_error_that_lists_the_ones_there_are() {
        let err = source(r#"{"path": "screen.xlsx", "sheet": "S3b"}"#)
            .read_all()
            .err()
            .expect("an unreadable declaration fails");
        assert!(err.contains("no sheet named 'S3b'"), "{err}");
        assert!(err.contains("'drugs'"), "{err}");
        assert!(err.contains("'S3a. Adjusted p-values'"), "{err}");

        let err = source(r#"{"path": "screen.xlsx", "sheet": 9}"#)
            .read_all()
            .err()
            .expect("an unreadable declaration fails");
        assert!(err.contains("no sheet at position 9"), "{err}");
        assert!(err.contains("'drugs'"), "{err}");
    }

    #[test]
    fn a_header_row_past_the_sheets_content_is_an_error_naming_the_range() {
        let err = source(r#"{"path": "screen.xlsx", "sheet": "drugs", "header_row": 99}"#)
            .read_all()
            .err()
            .expect("an unreadable declaration fails");
        assert!(err.contains("'header_row' is 99"), "{err}");
        assert!(err.contains("rows with content are 1–5"), "{err}");
    }

    #[test]
    fn a_workbook_that_is_not_there_fails_on_read_not_on_construction() {
        let src = XlsxFile::new(
            PathBuf::from("/nonexistent/definitely-not-here.xlsx"),
            "missing.xlsx".to_string(),
            config(r#"{"path": "missing.xlsx"}"#),
        );
        assert_eq!(src.size_hint(), None);
        let err = src.read_all().err().expect("a missing workbook fails");
        assert!(err.contains("xlsx open missing.xlsx"), "{err}");
        // The cached failure is reported identically the second time, and
        // `known_column_types` does not report it a third.
        assert_eq!(src.read_all().err().as_deref(), Some(err.as_str()));
        assert!(src.known_column_types().is_empty());
    }

    // ── unpivot ──────────────────────────────────────────────────────────

    fn unpivoted() -> XlsxFile {
        source(
            r#"{"path": "screen.xlsx", "sheet": "S3a. Adjusted p-values", "header_row": 3,
                "unpivot": {"id_columns": ["prestwick_ID", "chemical_name", "drug_class",
                                           "n_hit"],
                            "name_to": "isolate", "value_to": "adjusted_p"}}"#,
        )
    }

    #[test]
    fn unpivot_turns_the_value_columns_into_rows_and_drops_the_blanks() {
        let raw = unpivoted().read_all().unwrap();
        assert_eq!(
            raw.headers,
            vec![
                "prestwick_ID",
                "chemical_name",
                "drug_class",
                "n_hit",
                "isolate",
                "adjusted_p"
            ]
        );
        // 4 rows x 3 isolate columns, less the one blank cell.
        assert_eq!(raw.row_count(), 11);
        assert_eq!(
            raw.rows[0],
            vec![
                "260",
                "Amoxicillin",
                "antibacterial",
                "3",
                "Akkermansia muciniphila (NT5021)",
                "0.001"
            ]
        );
        // An empty cell produces no row at all — "not measured" must not
        // become an edge.
        assert!(!raw
            .rows
            .iter()
            .any(|r| r[0] == "261" && r[4].starts_with("Bacteroides")));
        assert_eq!(
            raw.rows.iter().filter(|r| r[0] == "261").count(),
            2,
            "the row with the blank cell contributes two pairs, not three"
        );
        // Provenance: every pair from one sheet row carries that row's number.
        assert_eq!(raw.row_ids[0], 1);
        assert_eq!(raw.row_ids[1], 1);
        assert_eq!(raw.row_ids[2], 1);
        assert_eq!(raw.row_ids[3], 2);
    }

    #[test]
    fn the_unpivoted_columns_carry_the_types_their_cells_agreed_on() {
        let known = unpivoted().known_column_types();
        assert_eq!(known.get("prestwick_ID"), Some(&ColumnType::Int64));
        assert_eq!(known.get("n_hit"), Some(&ColumnType::Int64));
        assert_eq!(known.get("adjusted_p"), Some(&ColumnType::Float64));
        // The header labels are text and the loader infers them, as it does
        // for any text column.
        assert_eq!(known.get("isolate"), None);
        assert_eq!(known.get("chemical_name"), None);
    }

    #[test]
    fn unpivot_refuses_an_id_column_the_header_does_not_have() {
        let err = source(
            r#"{"path": "screen.xlsx", "sheet": "S3a. Adjusted p-values", "header_row": 3,
                "unpivot": {"id_columns": ["prestwick_id"], "name_to": "isolate",
                            "value_to": "adjusted_p"}}"#,
        )
        .read_all()
        .err()
        .expect("an unreadable declaration fails");
        assert!(err.contains("id column 'prestwick_id'"), "{err}");
        assert!(err.contains("'prestwick_ID'"), "{err}");
    }

    #[test]
    fn unpivot_refuses_a_declaration_that_leaves_no_value_columns() {
        let err = source(
            r#"{"path": "screen.xlsx", "sheet": "drugs",
                "unpivot": {"id_columns": ["prestwick_ID", "chemical_name", "atc_code",
                                           "approved", "approved_on"],
                            "name_to": "k", "value_to": "v"}}"#,
        )
        .read_all()
        .err()
        .expect("an unreadable declaration fails");
        assert!(err.contains("nothing left to turn into rows"), "{err}");
    }

    // ── chunking ─────────────────────────────────────────────────────────

    /// The Python differential corpus cannot reach this format (no xlsx writer
    /// in the test environment), so chunk-invariance is pinned here instead:
    /// slicing the sheet must produce exactly the whole read, row for row.
    #[test]
    fn chunking_slices_the_same_rows_in_the_same_order() {
        for src in [
            source(r#"{"path": "screen.xlsx", "sheet": "drugs"}"#),
            unpivoted(),
        ] {
            let whole = src.read_all().unwrap();
            let mut rows = Vec::new();
            let mut nulls = Vec::new();
            let mut row_ids = Vec::new();
            let mut chunk_count = 0usize;
            for chunk in src.chunks(2).unwrap() {
                let c = chunk.unwrap();
                assert_eq!(c.headers, whole.headers);
                assert!(c.row_count() <= 2);
                chunk_count += 1;
                rows.extend(c.rows);
                nulls.extend(c.nulls);
                row_ids.extend(c.row_ids);
            }
            assert_eq!(chunk_count, whole.row_count().div_ceil(2));
            assert_eq!(rows, whole.rows);
            assert_eq!(nulls, whole.nulls);
            assert_eq!(row_ids, whole.row_ids);
        }
    }

    #[test]
    fn chunks_can_be_taken_twice_and_agree() {
        let src = source(r#"{"path": "screen.xlsx", "sheet": "drugs"}"#);
        let take = || -> Vec<Vec<Vec<String>>> {
            src.chunks(3).unwrap().map(|c| c.unwrap().rows).collect()
        };
        assert_eq!(take(), take());
        assert_eq!(take().len(), 2);
    }

    /// The optimised `scan_columns` must answer exactly what the trait's
    /// chunk-based default does — it is a read path, not a different rule.
    #[test]
    fn scan_columns_matches_the_trait_default() {
        use super::super::test_double::DefaultScanSource;
        let collect = |s: &dyn Source| {
            let mut seen = Vec::new();
            s.scan_columns(
                &["prestwick_ID", "atc_code", "missing"],
                &mut |slot, cell| {
                    seen.push((slot, cell.to_string()));
                    true
                },
            )
            .unwrap();
            seen
        };
        let fast = collect(&source(r#"{"path": "screen.xlsx", "sheet": "drugs"}"#));
        let slow = collect(&DefaultScanSource(Box::new(source(
            r#"{"path": "screen.xlsx", "sheet": "drugs"}"#,
        ))));
        assert_eq!(fast, slow);
        // Non-vacuity: 4 ids plus the 3 non-null atc codes.
        assert_eq!(fast.len(), 7);
    }

    // ── error cells ──────────────────────────────────────────────────────

    /// `Data::Error` is a cell the spreadsheet itself failed to compute. It is
    /// null in the table — there is no value — and one warning per column says
    /// so, because a silently-null `#DIV/0!` column is a missing property
    /// nobody looks for.
    #[test]
    fn an_error_cell_is_null_and_tallied_once_per_column() {
        let mut range = Range::new((0, 0), (2, 1));
        range.set_value((0, 0), Data::String("id".into()));
        range.set_value((0, 1), Data::String("ratio".into()));
        range.set_value((1, 0), Data::Int(1));
        range.set_value((1, 1), Data::Error(CellErrorType::Div0));
        range.set_value((2, 0), Data::Int(2));
        range.set_value((2, 1), Data::Error(CellErrorType::Div0));
        let ctx = SheetCtx {
            display: "screen.xlsx",
            sheet: "calc",
            origin: (0, 0),
        };
        let headers = read_headers(&range, 0);
        let out = build_wide(&ctx, &range, 0, &headers);
        assert_eq!(out.table.rows[0], vec!["1", ""]);
        assert_eq!(out.table.nulls[0], vec![false, true]);
        assert_eq!(out.warnings.len(), 1);
        let w = &out.warnings[0];
        assert!(w.contains("sheet 'calc' column 'ratio'"), "{w}");
        assert!(w.contains("cell B2 is #DIV/0!"), "{w}");
        assert!(w.contains("(2 cells in this column)"), "{w}");
        assert!(w.contains("stored as null"), "{w}");
        // An error cell says nothing about its column's type.
        assert_eq!(out.known.get("ratio"), None);
        assert_eq!(out.known.get("id"), Some(&ColumnType::Int64));
    }

    /// A header cell nobody filled in cannot be referenced by name, so its
    /// column is dropped rather than landing as an unnamed one.
    #[test]
    fn a_column_with_an_empty_header_is_dropped() {
        let mut range = Range::new((0, 0), (1, 3));
        range.set_value((0, 0), Data::String("a".into()));
        // Column 1 has no header cell at all; column 2's is spaces, which is
        // the same thing once a name is what you reference it by.
        range.set_value((0, 2), Data::String("   ".into()));
        range.set_value((0, 3), Data::String(" c ".into()));
        for (c, v) in [(0, 1), (1, 2), (2, 3), (3, 4)] {
            range.set_value((1, c), Data::Int(v));
        }
        let ctx = SheetCtx {
            display: "x.xlsx",
            sheet: "s",
            origin: (0, 0),
        };
        let out = build_wide(&ctx, &range, 0, &read_headers(&range, 0));
        // A name that survives is trimmed, so `pk: "c"` resolves.
        assert_eq!(out.table.headers, vec!["a", "c"]);
        assert_eq!(out.table.rows[0], vec!["1", "4"]);
    }

    /// A sheet whose content starts below row 1 reports its own origin, and
    /// every physical row number — `header_row` and the cell references in
    /// warnings alike — has to be read against it.
    #[test]
    fn physical_row_numbers_are_read_against_the_ranges_own_origin() {
        let ctx = SheetCtx {
            display: "x.xlsx",
            sheet: "s",
            // The range's top-left cell is C4.
            origin: (3, 2),
        };
        assert_eq!(ctx.cell_ref(0, 0), "C4");
        assert_eq!(ctx.cell_ref(2, 1), "D6");
    }
}
