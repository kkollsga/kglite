//! The in-memory frame `Source`: a `DataFrame` the caller hands the build
//! instead of a file.
//!
//! A frame is consumed exactly like a read file — same specs, filters, chunked
//! junction loading, dedupe regime and warnings — by being stringified into a
//! [`RawCsv`] up front and then flowing through the one string-keyed pipeline
//! every other format uses. It is *coerced to the blueprint's declared property
//! types through the string table*, not carried through typed; the blueprint's
//! type vocabulary is what survives, and anything outside it (time-of-day,
//! maps, nested typing) is not preserved.
//!
//! ## Canonical text — the contract
//!
//! Every cell is rendered by exactly one rule, chosen so the string table
//! round-trips back to the same type the frame had:
//!
//! | column type | text |
//! |---|---|
//! | `Int64` / `UniqueId` | `{}` — `5` |
//! | `Float64` | `{:?}` — `1.0` stays `1.0`, so it is not re-read as an int |
//! | `Boolean` | `true` / `false` |
//! | `DateTime` | ISO-8601 date, `2024-03-01` |
//! | `Timestamp` | ISO-8601 date+time, `2024-03-01T09:30:00` |
//! | `List` | compact JSON, nested values included — `["a",1]` |
//! | `Map` | compact JSON object — `{"a":1}` |
//! | `String` | the string as-is |
//!
//! Two rules carry a consequence worth stating plainly:
//!
//! - **An empty string becomes null**, exactly as it would in a CSV. The
//!   string table's null flag *is* "the cell was empty", and a frame that
//!   spells a missing value as `""` is spelling it the way a CSV does.
//! - **A `null` cell is `""` plus the null flag**, so a column of nulls is a
//!   column of nulls whatever its type.
//!
//! `Timestamp` and `Map` have no blueprint keyword, so a column of either is
//! reported as `string` by [`FrameSource::known_column_types`] and one warning
//! per column says so — the text is still faithful, but the property lands as
//! text.

use super::super::table::RawCsv;
use super::{accept_any_entry, FormatSpec, Source};
use crate::datatypes::values::{ColumnType, DataFrame, Value};
use std::collections::HashMap;

/// Keys a `files` entry with `"format": "frame"` reads. A frame carries no
/// `path`: the caller passes the rows in, keyed by this entry's name.
pub const ACCEPTED_FILE_KEYS_FRAME: &[&str] = &["format"];

pub const FORMAT: FormatSpec = FormatSpec {
    name: "frame",
    accepted_keys: ACCEPTED_FILE_KEYS_FRAME,
    knob_keys: &[],
    validate_entry: accept_any_entry,
};

/// A materialised in-memory table. Unlike a file source it holds its whole
/// input: the frame was already in memory when the caller handed it over, and
/// there is nothing behind it to stream from.
pub struct FrameSource {
    display: String,
    table: RawCsv,
    known: HashMap<String, ColumnType>,
}

impl FrameSource {
    /// Stringify `df` under the canonical rules above.
    ///
    /// Returns the source plus one warning per column whose type the blueprint
    /// vocabulary cannot hold — the caller folds those into the build report,
    /// because a `Source` has no report to write to.
    pub fn new(name: &str, df: &DataFrame) -> (Self, Vec<String>) {
        let headers = df.get_column_names();
        let n_rows = df.row_count();
        let mut known: HashMap<String, ColumnType> = HashMap::new();
        let mut warnings: Vec<String> = Vec::new();

        for header in &headers {
            let Some(ct) = df.get_column_type(header) else {
                continue;
            };
            match vocabulary_type(&ct) {
                Some(mapped) => {
                    known.insert(header.clone(), mapped);
                }
                None => {
                    known.insert(header.clone(), ColumnType::String);
                    warnings.push(format!(
                        "frame '{name}' column '{header}' has type {}, which the blueprint \
                         vocabulary cannot hold; stored as string",
                        type_label(&ct)
                    ));
                }
            }
        }

        let mut rows: Vec<Vec<String>> = Vec::with_capacity(n_rows);
        let mut nulls: Vec<Vec<bool>> = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let mut row = Vec::with_capacity(headers.len());
            let mut nrow = Vec::with_capacity(headers.len());
            for (c, _) in headers.iter().enumerate() {
                match df.get_value_by_index(r, c) {
                    Some(v) => {
                        let text = canonical_text(&v);
                        // Empty text is null, the same rule the CSV readers
                        // apply — otherwise a frame's `""` would become an
                        // empty-string property where a CSV's is a missing one.
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

        let table = RawCsv {
            headers,
            rows,
            nulls,
            row_ids: (1..=n_rows).collect(),
        };
        (
            Self {
                display: format!("frame '{name}'"),
                table,
                known,
            },
            warnings,
        )
    }

    fn slice(&self, start: usize, end: usize) -> RawCsv {
        RawCsv {
            headers: self.table.headers.clone(),
            rows: self.table.rows[start..end].to_vec(),
            nulls: self.table.nulls[start..end].to_vec(),
            row_ids: self.table.row_ids[start..end].to_vec(),
        }
    }
}

/// The `ColumnType` a blueprint can name for this stored type, or `None` when
/// the vocabulary has no keyword for it. The inverse of
/// [`super::super::typing::map_blueprint_type`]; `UniqueId` is an integer id
/// column and reads back as one.
fn vocabulary_type(ct: &ColumnType) -> Option<ColumnType> {
    match ct {
        ColumnType::Int64 | ColumnType::UniqueId => Some(ColumnType::Int64),
        ColumnType::Float64 => Some(ColumnType::Float64),
        ColumnType::Boolean => Some(ColumnType::Boolean),
        ColumnType::String => Some(ColumnType::String),
        ColumnType::DateTime => Some(ColumnType::DateTime),
        ColumnType::List => Some(ColumnType::List),
        ColumnType::Timestamp | ColumnType::Map => None,
    }
}

/// What a warning calls a column type. The blueprint keywords where one
/// exists, so an author reads back a word they can write.
fn type_label(ct: &ColumnType) -> &'static str {
    match ct {
        ColumnType::Int64 => "int",
        ColumnType::UniqueId => "unique id",
        ColumnType::Float64 => "float",
        ColumnType::Boolean => "bool",
        ColumnType::String => "string",
        ColumnType::DateTime => "date",
        ColumnType::List => "list",
        ColumnType::Timestamp => "timestamp",
        ColumnType::Map => "map",
    }
}

/// One cell's text under the table in the module header.
fn canonical_text(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Int64(i) => i.to_string(),
        Value::UniqueId(u) => u.to_string(),
        // `{:?}` is f64's shortest round-tripping form and keeps the decimal
        // point: `1.0` must not come back through inference as an int.
        Value::Float64(f) => format!("{f:?}"),
        Value::Boolean(b) => b.to_string(),
        Value::String(s) => s.clone(),
        Value::DateTime(d) => d.format("%Y-%m-%d").to_string(),
        Value::Timestamp(t) => t.format("%Y-%m-%dT%H:%M:%S").to_string(),
        // Compact JSON, nested values included — the same spelling
        // `typing::parse_list_cell` reads back.
        other => {
            serde_json::to_string(&crate::param::kglite_value_to_json(other)).unwrap_or_default()
        }
    }
}

impl Source for FrameSource {
    fn display_name(&self) -> &str {
        &self.display
    }

    /// `None`: a frame has no size on disk to compare against the streaming
    /// threshold, and nothing to stream from anyway — it is already in memory.
    /// The buffered path is what `None` selects.
    fn size_hint(&self) -> Option<u64> {
        None
    }

    /// True. The rows are all in hand, so a chunked consumer (every junction
    /// loader) gets its chunks by slicing them; what chunking cannot do for a
    /// frame is bound peak RAM.
    fn can_chunk(&self) -> bool {
        true
    }

    fn read_all(&self) -> Result<RawCsv, String> {
        Ok(self.slice(0, self.table.rows.len()))
    }

    fn chunks(
        &self,
        chunk_size: usize,
    ) -> Result<Box<dyn Iterator<Item = Result<RawCsv, String>> + '_>, String> {
        let total = self.table.rows.len();
        let step = chunk_size.max(1);
        let mut next = 0usize;
        Ok(Box::new(std::iter::from_fn(move || {
            if next >= total {
                return None;
            }
            let end = (next + step).min(total);
            let chunk = self.slice(next, end);
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
        let indices: Vec<Option<usize>> = columns
            .iter()
            .map(|name| self.table.col_index(name))
            .collect();
        for r in 0..self.table.rows.len() {
            for (slot, idx) in indices.iter().enumerate() {
                let Some(idx) = idx else { continue };
                if self.table.nulls[r][*idx] {
                    continue;
                }
                if !visit(slot, &self.table.rows[r][*idx]) {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn known_column_types(&self) -> HashMap<String, ColumnType> {
        self.known.clone()
    }
}

#[cfg(test)]
mod frame_source_tests {
    use super::*;
    use crate::datatypes::values::ColumnData;
    use crate::datatypes::PropMap;
    use chrono::{NaiveDate, NaiveDateTime};

    fn frame(columns: Vec<(&str, ColumnType, ColumnData)>) -> DataFrame {
        let mut df = DataFrame::new(Vec::new());
        for (name, ct, data) in columns {
            df.add_column(name.to_string(), ct, data).unwrap();
        }
        df
    }

    fn cells(source: &FrameSource, column: &str) -> Vec<(String, bool)> {
        let raw = source.read_all().unwrap();
        let idx = raw.col_index(column).unwrap();
        (0..raw.row_count())
            .map(|r| (raw.rows[r][idx].clone(), raw.nulls[r][idx]))
            .collect()
    }

    #[test]
    fn ints_and_floats_keep_their_shape() {
        let df = frame(vec![
            (
                "i",
                ColumnType::Int64,
                ColumnData::Int64(vec![Some(5), Some(-1), None]),
            ),
            (
                "f",
                ColumnType::Float64,
                ColumnData::Float64(vec![Some(1.0), Some(2.5), None]),
            ),
        ]);
        let (src, warnings) = FrameSource::new("t", &df);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            cells(&src, "i"),
            vec![
                ("5".into(), false),
                ("-1".into(), false),
                (String::new(), true)
            ]
        );
        // `1.0`, not `1`: a float column that printed as `1` would be read
        // back as an int by the string table's inference.
        assert_eq!(
            cells(&src, "f"),
            vec![
                ("1.0".into(), false),
                ("2.5".into(), false),
                (String::new(), true)
            ]
        );
    }

    #[test]
    fn bools_dates_and_strings_take_their_canonical_spelling() {
        let df = frame(vec![
            (
                "b",
                ColumnType::Boolean,
                ColumnData::Boolean(vec![Some(true), Some(false)]),
            ),
            (
                "d",
                ColumnType::DateTime,
                ColumnData::DateTime(vec![NaiveDate::from_ymd_opt(2024, 3, 1), None]),
            ),
            (
                "s",
                ColumnType::String,
                ColumnData::String(vec![Some("hi".into()), Some(String::new())]),
            ),
        ]);
        let (src, warnings) = FrameSource::new("t", &df);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            cells(&src, "b"),
            vec![("true".into(), false), ("false".into(), false)]
        );
        assert_eq!(
            cells(&src, "d"),
            vec![("2024-03-01".into(), false), (String::new(), true)]
        );
        // The empty string is null, exactly as it would be in a CSV.
        assert_eq!(
            cells(&src, "s"),
            vec![("hi".into(), false), (String::new(), true)]
        );
    }

    #[test]
    fn lists_and_maps_are_compact_json_including_nested_values() {
        let mut map = PropMap::default();
        map.insert("a".to_string(), Value::Int64(1));
        let df = frame(vec![
            (
                "l",
                ColumnType::List,
                ColumnData::List(vec![
                    Some(vec![
                        Value::String("a".into()),
                        Value::Int64(1),
                        Value::List(vec![Value::Boolean(true)]),
                    ]),
                    None,
                ]),
            ),
            (
                "m",
                ColumnType::Map,
                ColumnData::Map(vec![Some(map.clone()), None]),
            ),
        ]);
        let (src, warnings) = FrameSource::new("t", &df);
        assert_eq!(
            cells(&src, "l"),
            vec![(r#"["a",1,[true]]"#.into(), false), (String::new(), true)]
        );
        assert_eq!(
            cells(&src, "m"),
            vec![(r#"{"a":1}"#.into(), false), (String::new(), true)]
        );
        // The map column is the one outside the vocabulary.
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains("column 'm' has type map"),
            "{:?}",
            warnings
        );
    }

    #[test]
    fn timestamps_are_iso_text_and_warn_that_they_land_as_string() {
        let dt = NaiveDate::from_ymd_opt(2024, 3, 1)
            .unwrap()
            .and_hms_opt(9, 30, 0)
            .unwrap();
        let df = frame(vec![(
            "t",
            ColumnType::Timestamp,
            ColumnData::Timestamp(vec![Some(dt), None::<NaiveDateTime>]),
        )]);
        let (src, warnings) = FrameSource::new("events", &df);
        assert_eq!(
            cells(&src, "t"),
            vec![("2024-03-01T09:30:00".into(), false), (String::new(), true)]
        );
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings[0],
            "frame 'events' column 't' has type timestamp, which the blueprint vocabulary \
             cannot hold; stored as string"
        );
        assert_eq!(
            src.known_column_types().get("t"),
            Some(&ColumnType::String),
            "an unholdable type is reported as the string it becomes"
        );
    }

    #[test]
    fn known_column_types_names_every_column_in_the_blueprint_vocabulary() {
        let df = frame(vec![
            ("i", ColumnType::Int64, ColumnData::Int64(vec![Some(1)])),
            (
                "f",
                ColumnType::Float64,
                ColumnData::Float64(vec![Some(1.0)]),
            ),
            (
                "b",
                ColumnType::Boolean,
                ColumnData::Boolean(vec![Some(true)]),
            ),
            (
                "s",
                ColumnType::String,
                ColumnData::String(vec![Some("x".into())]),
            ),
            (
                "d",
                ColumnType::DateTime,
                ColumnData::DateTime(vec![NaiveDate::from_ymd_opt(2024, 1, 1)]),
            ),
            (
                "l",
                ColumnType::List,
                ColumnData::List(vec![Some(vec![Value::Int64(1)])]),
            ),
            (
                "u",
                ColumnType::UniqueId,
                ColumnData::UniqueId(vec![Some(7)]),
            ),
        ]);
        let (src, warnings) = FrameSource::new("t", &df);
        assert!(warnings.is_empty(), "{warnings:?}");
        let known = src.known_column_types();
        assert_eq!(known.get("i"), Some(&ColumnType::Int64));
        assert_eq!(known.get("f"), Some(&ColumnType::Float64));
        assert_eq!(known.get("b"), Some(&ColumnType::Boolean));
        assert_eq!(known.get("s"), Some(&ColumnType::String));
        assert_eq!(known.get("d"), Some(&ColumnType::DateTime));
        assert_eq!(known.get("l"), Some(&ColumnType::List));
        // A unique-id column is an integer id column, not a ninth keyword.
        assert_eq!(known.get("u"), Some(&ColumnType::Int64));
        assert_eq!(cells(&src, "u"), vec![("7".into(), false)]);
    }

    #[test]
    fn chunks_slice_the_rows_and_carry_their_row_numbers() {
        let df = frame(vec![(
            "i",
            ColumnType::Int64,
            ColumnData::Int64((1..=7).map(Some).collect()),
        )]);
        let (src, _) = FrameSource::new("t", &df);
        let chunks: Vec<RawCsv> = src.chunks(3).unwrap().map(Result::unwrap).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks.iter().map(|c| c.row_ids.clone()).collect::<Vec<_>>(),
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7]]
        );
        assert_eq!(chunks[2].rows[0][0], "7");
        // A second pass yields the same batches — the registry hands one
        // source to several phases.
        let again: Vec<Vec<usize>> = src.chunks(3).unwrap().map(|c| c.unwrap().row_ids).collect();
        assert_eq!(again, vec![vec![1, 2, 3], vec![4, 5, 6], vec![7]]);
    }

    #[test]
    fn an_empty_frame_yields_no_chunks() {
        let df = frame(vec![("i", ColumnType::Int64, ColumnData::Int64(vec![]))]);
        let (src, _) = FrameSource::new("t", &df);
        assert_eq!(src.chunks(10).unwrap().count(), 0);
        assert_eq!(src.read_all().unwrap().row_count(), 0);
    }

    /// A frame is buffered, never streamed: `size_hint` has nothing to
    /// compare against a byte threshold, and the whole table is already
    /// resident.
    #[test]
    fn a_frame_reports_no_size_and_can_be_chunked() {
        let df = frame(vec![("i", ColumnType::Int64, ColumnData::Int64(vec![]))]);
        let (src, _) = FrameSource::new("t", &df);
        assert_eq!(src.size_hint(), None);
        assert!(src.can_chunk());
        assert_eq!(src.display_name(), "frame 't'");
    }
}
