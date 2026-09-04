//! Raw string table → internal `DataFrame` conversion.
//!
//! Pandas is entirely absent. Columns are parsed to typed vectors in one
//! pass over the table. Declared blueprint types (`"string"`, `"int"`,
//! `"float"`, `"bool"`, `"date"`, `"datetime"`) win over inference; any
//! column without an explicit type falls back to light inference on
//! the first non-empty cell in each column.

use super::table::{looks_like_a_missed_list, ListMisparseTally, RawCsv};
use crate::datatypes::values::{ColumnData, ColumnType, DataFrame, Value};
use chrono::NaiveDate;
use std::collections::HashMap;

/// Build a typed `DataFrame` from raw CSV, keeping only `keep_columns`.
/// `declared_types` maps column name → blueprint type keyword; other columns
/// fall back to inference.
pub fn typed_dataframe(
    raw: &RawCsv,
    keep_columns: &[String],
    declared_types: &HashMap<String, String>,
    rename: &HashMap<String, String>,
    misparses: &mut ListMisparseTally,
) -> Result<DataFrame, String> {
    let mut df = DataFrame::new(Vec::new());
    append_typed_columns(
        &mut df,
        raw,
        keep_columns,
        declared_types,
        rename,
        misparses,
    )?;
    Ok(df)
}

/// `typed_dataframe`'s body, writing into a frame that already has columns.
/// The FK-edge loader builds its source/target id pair first — those two are
/// typed by id inference, not by the CSV's — and appends the declared
/// property columns onto it.
pub fn append_typed_columns(
    df: &mut DataFrame,
    raw: &RawCsv,
    keep_columns: &[String],
    declared_types: &HashMap<String, String>,
    rename: &HashMap<String, String>,
    misparses: &mut ListMisparseTally,
) -> Result<(), String> {
    let mut columns: Vec<(String, ColumnType)> = Vec::with_capacity(keep_columns.len());
    let mut data: Vec<ColumnData> = Vec::with_capacity(keep_columns.len());

    for name in keep_columns {
        let src_idx = raw.col_index(name).ok_or_else(|| {
            format!(
                "Column '{}' not found in CSV (available: {:?})",
                name, raw.headers
            )
        })?;
        // `declared_types` (and `col_index`) stay keyed by the CSV name;
        // only the output column carries the renamed spelling.
        let col_type = resolve_column_type(raw, src_idx, declared_types.get(name));
        let col_data = build_column_data(raw, src_idx, &col_type, name, misparses)?;
        let out_name = rename.get(name).cloned().unwrap_or_else(|| name.clone());
        columns.push((out_name, col_type));
        data.push(col_data);
    }

    for ((name, col_type), col_data) in columns.into_iter().zip(data) {
        df.add_column(name, col_type, col_data)
            .map_err(|e| format!("add_column failed: {}", e))?;
    }
    Ok(())
}

/// Map a blueprint type keyword to a KGLite `ColumnType`. Returns `None` for
/// spatial / temporal virtual types handled elsewhere.
pub fn map_blueprint_type(ty: &str) -> Option<ColumnType> {
    match ty {
        "string" | "str" => Some(ColumnType::String),
        "int" | "integer" => Some(ColumnType::Int64),
        "float" => Some(ColumnType::Float64),
        "bool" | "boolean" => Some(ColumnType::Boolean),
        "date" | "datetime" | "validFrom" | "validTo" => Some(ColumnType::DateTime),
        "list" | "array" => Some(ColumnType::List),
        _ => None,
    }
}

/// The blueprint keyword naming `ct`, or `None` where the vocabulary has no
/// word for it (`UniqueId`, `Timestamp`, `Map`).
///
/// The inverse of [`map_blueprint_type`], so a type an input already knows can
/// be handed to the typing pass through the same `declared_types` map a
/// blueprint fills. Wider than [`inferred_type_keyword`], which answers only
/// for the four types inference can produce.
pub fn blueprint_type_keyword(ct: &ColumnType) -> Option<&'static str> {
    match ct {
        ColumnType::String => Some("string"),
        ColumnType::Int64 => Some("int"),
        ColumnType::Float64 => Some("float"),
        ColumnType::Boolean => Some("bool"),
        ColumnType::DateTime => Some("date"),
        ColumnType::List => Some("list"),
        ColumnType::UniqueId | ColumnType::Timestamp | ColumnType::Map => None,
    }
}

/// Fill in, for every column `declared` does not name, the type the input
/// itself knows — the middle rung of **declared → known → inferred**.
///
/// The blueprint wins where it speaks: a spec declaring `"string"` over a
/// frame's int column gets a string property, the same as it would over a CSV.
/// Where it is silent, an already-typed input answers instead of inference,
/// which is both more faithful (a float column of whole numbers stays float)
/// and cheaper (a typed column never reaches the whole-input pre-pass).
pub fn overlay_known_types(
    declared: &mut HashMap<String, String>,
    known: &HashMap<String, ColumnType>,
) {
    for (col, ct) in known {
        if declared.contains_key(col) {
            continue;
        }
        if let Some(keyword) = blueprint_type_keyword(ct) {
            declared.insert(col.clone(), keyword.to_string());
        }
    }
}

fn resolve_column_type(raw: &RawCsv, src_idx: usize, declared: Option<&String>) -> ColumnType {
    if let Some(ty) = declared {
        if let Some(ct) = map_blueprint_type(ty) {
            return ct;
        }
    }
    infer_type(raw, src_idx)
}

/// Incremental column-type inference.
///
/// The rule is `infer_type`'s, folded one cell at a time so a chunked input
/// can be typed by a pre-pass over *every* chunk. Inferring per chunk instead
/// makes a column's type depend on the chunk size — an undeclared
/// int-then-text column came out `Int64` in the early chunks and `String` in
/// the late ones, from a knob documented as bounding memory only.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnInference {
    saw_int: bool,
    saw_float: bool,
    saw_bool: bool,
    saw_other: bool,
}

impl ColumnInference {
    /// Fold one cell in. Null/blank cells carry no evidence; once a cell has
    /// been seen that is none of the three parseable shapes the answer is
    /// `String` whatever follows, which is why the whole-table pass may stop
    /// there and the two agree regardless.
    pub fn observe(&mut self, cell: &str) {
        if self.saw_other {
            return;
        }
        let s = cell.trim();
        if s.is_empty() {
            return;
        }
        if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("false") {
            self.saw_bool = true;
        } else if s.parse::<i64>().is_ok() {
            self.saw_int = true;
        } else if s.parse::<f64>().is_ok() {
            self.saw_float = true;
        } else {
            self.saw_other = true;
        }
    }

    /// Fold in one table's column.
    pub fn observe_column(&mut self, raw: &RawCsv, src_idx: usize) {
        for (r, row) in raw.rows.iter().enumerate() {
            if self.saw_other {
                break;
            }
            if raw.nulls[r][src_idx] {
                continue;
            }
            self.observe(&row[src_idx]);
        }
    }

    /// True once no further cell can change the answer: a cell outside the
    /// three parseable shapes settles the column as `String`.
    pub fn is_settled(&self) -> bool {
        self.saw_other
    }

    pub fn resolve(&self) -> ColumnType {
        if self.saw_other {
            ColumnType::String
        } else if self.saw_float {
            ColumnType::Float64
        } else if self.saw_int {
            ColumnType::Int64
        } else if self.saw_bool {
            ColumnType::Boolean
        } else {
            ColumnType::String
        }
    }
}

/// Incremental id-column inference: `Int64` while every value seen parses as
/// a whole number, `String` from the first one that does not.
///
/// Separate from [`ColumnInference`] because an id asks a different question —
/// `"3.0"` is a valid integer id and `"true"` is not a boolean but a string id
/// — and the two must not be conflated. Incremental for the same reason:
/// deciding it per chunk makes the type depend on the chunk size.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdInference {
    saw_non_integer: bool,
}

impl IdInference {
    pub fn observe(&mut self, cell: &str) {
        if self.saw_non_integer {
            return;
        }
        let t = cell.trim();
        if t.is_empty() || t.parse::<i64>().is_ok() {
            return;
        }
        if let Ok(f) = t.parse::<f64>() {
            if f.is_finite() && f.fract() == 0.0 {
                return;
            }
        }
        self.saw_non_integer = true;
    }

    /// True once no later value can change the answer.
    pub fn is_settled(&self) -> bool {
        self.saw_non_integer
    }

    pub fn resolve(&self) -> ColumnType {
        if self.saw_non_integer {
            ColumnType::String
        } else {
            ColumnType::Int64
        }
    }
}

/// The blueprint keyword naming an inferred type, so a resolved type can be
/// handed back through the `declared_types` map the typing pass already takes.
/// `None` for a type inference never produces.
pub fn inferred_type_keyword(ct: &ColumnType) -> Option<&'static str> {
    match ct {
        ColumnType::String => Some("string"),
        ColumnType::Int64 => Some("int"),
        ColumnType::Float64 => Some("float"),
        ColumnType::Boolean => Some("bool"),
        _ => None,
    }
}

fn infer_type(raw: &RawCsv, src_idx: usize) -> ColumnType {
    let mut inference = ColumnInference::default();
    inference.observe_column(raw, src_idx);
    inference.resolve()
}

fn build_column_data(
    raw: &RawCsv,
    src_idx: usize,
    col_type: &ColumnType,
    column: &str,
    misparses: &mut ListMisparseTally,
) -> Result<ColumnData, String> {
    let n = raw.row_count();
    match col_type {
        ColumnType::Int64 => {
            let mut out: Vec<Option<i64>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                    continue;
                }
                let s = row[src_idx].trim();
                if s.is_empty() {
                    out.push(None);
                } else if let Ok(v) = s.parse::<i64>() {
                    out.push(Some(v));
                } else if let Ok(v) = s.parse::<f64>() {
                    // Pandas-style: whole-number float → int
                    if v.is_finite()
                        && v.fract() == 0.0
                        && v >= i64::MIN as f64
                        && v <= i64::MAX as f64
                    {
                        out.push(Some(v as i64));
                    } else {
                        out.push(None);
                    }
                } else {
                    out.push(None);
                }
            }
            Ok(ColumnData::Int64(out))
        }
        ColumnType::Float64 => {
            let mut out: Vec<Option<f64>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                    continue;
                }
                let s = row[src_idx].trim();
                if s.is_empty() {
                    out.push(None);
                } else {
                    out.push(s.parse::<f64>().ok());
                }
            }
            Ok(ColumnData::Float64(out))
        }
        ColumnType::Boolean => {
            let mut out: Vec<Option<bool>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                    continue;
                }
                let s = row[src_idx].trim();
                match s.to_ascii_lowercase().as_str() {
                    "true" | "1" | "t" | "yes" | "y" => out.push(Some(true)),
                    "false" | "0" | "f" | "no" | "n" => out.push(Some(false)),
                    "" => out.push(None),
                    _ => out.push(None),
                }
            }
            Ok(ColumnData::Boolean(out))
        }
        ColumnType::DateTime => {
            let mut out: Vec<Option<NaiveDate>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                    continue;
                }
                let s = row[src_idx].trim();
                out.push(parse_date_cell(s));
            }
            Ok(ColumnData::DateTime(out))
        }
        ColumnType::String => {
            let mut out: Vec<Option<String>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                } else {
                    let s = &row[src_idx];
                    if s.is_empty() {
                        out.push(None);
                    } else {
                        out.push(Some(s.clone()));
                    }
                }
            }
            Ok(ColumnData::String(out))
        }
        ColumnType::UniqueId => {
            let mut out: Vec<Option<u32>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                    continue;
                }
                let s = row[src_idx].trim();
                out.push(s.parse::<u32>().ok());
            }
            Ok(ColumnData::UniqueId(out))
        }
        ColumnType::List => {
            // CSV never *infers* a list — this arm fires only where a
            // blueprint declares `"list"` / `"array"` for the column, in a
            // node spec's `properties` or a junction edge's `property_types`.
            // The cell is parsed as a JSON array (`["a","b"]`); anything else
            // becomes a one-element list, which is the right answer for a
            // lone scalar and a plausible wrong one for `a|b` — hence the
            // tally, which the build report turns into a warning.
            let mut out: Vec<Option<Vec<Value>>> = Vec::with_capacity(n);
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][src_idx] {
                    out.push(None);
                    continue;
                }
                let s = row[src_idx].trim();
                if s.is_empty() {
                    out.push(None);
                } else {
                    let parsed = parse_list_cell(s);
                    if looks_like_a_missed_list(s) {
                        misparses.record(column, raw.row_id(r), s);
                    }
                    out.push(Some(parsed));
                }
            }
            Ok(ColumnData::List(out))
        }
        // CSV never infers or declares Timestamp / Map (`map_blueprint_type`
        // has no keyword for them, and `infer_type` never yields them), so
        // these arms are unreachable in practice. Return an all-null column of
        // the right shape to keep the match exhaustive without a panic.
        ColumnType::Timestamp => Ok(ColumnData::Timestamp(vec![None; n])),
        ColumnType::Map => Ok(ColumnData::Map(vec![None; n])),
    }
}

/// Parse a CSV cell declared as a list. A JSON array maps element-wise; any
/// other value is wrapped as a single-element list so it isn't dropped.
fn parse_list_cell(s: &str) -> Vec<Value> {
    match serde_json::from_str::<serde_json::Value>(s) {
        Ok(serde_json::Value::Array(items)) => items.iter().map(json_scalar_to_value).collect(),
        Ok(other) => vec![json_scalar_to_value(&other)],
        Err(_) => vec![Value::String(s.to_string())],
    }
}

/// Minimal JSON-scalar → `Value` mapping for list elements. Nested
/// arrays/objects are stringified — list cells are expected to hold scalars.
fn json_scalar_to_value(j: &serde_json::Value) -> Value {
    match j {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Boolean(*b),
        serde_json::Value::Number(num) => num
            .as_i64()
            .map(Value::Int64)
            .or_else(|| num.as_f64().map(Value::Float64))
            .unwrap_or(Value::Null),
        serde_json::Value::String(s) => Value::String(s.clone()),
        other => Value::String(other.to_string()),
    }
}

/// Parse a date cell. Accepts ISO dates, ISO datetimes, and epoch milliseconds.
/// The Python loader fed epoch-ms values (strings of digits) through
/// `pd.to_datetime(unit="ms")` — mirror that behaviour.
fn parse_date_cell(s: &str) -> Option<NaiveDate> {
    if s.is_empty() {
        return None;
    }
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Some(d);
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(dt.date());
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(dt.date());
    }
    // Epoch millis — e.g. "1609459200000"
    if let Ok(ms) = s.parse::<i64>() {
        if let Some(dt) = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms) {
            return Some(dt.date_naive());
        }
    }
    // Floating-point epoch ms — e.g. "1609459200000.0"
    if let Ok(ms) = s.parse::<f64>() {
        if ms.is_finite() {
            if let Some(dt) = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms as i64) {
                return Some(dt.date_naive());
            }
        }
    }
    None
}

/// The typing half's golden target: exactly what `typed_dataframe` yields for
/// each `ColumnType` arm, each inference outcome and each null shape.
///
/// Every future `RawCsv` producer (delimited, xlsx, a pandas frame) is
/// required to land on these answers, so a change here is a change to what
/// every input format means — not a test detail.
#[cfg(test)]
mod typing_tests {
    use super::*;
    use chrono::NaiveDate;

    /// Build a `RawCsv` from literals. A cell that is the empty string is
    /// null, matching what both file readers write into `nulls`.
    fn raw(headers: &[&str], rows: &[&[&str]]) -> RawCsv {
        let rows_v: Vec<Vec<String>> = rows
            .iter()
            .map(|r| r.iter().map(|s| s.to_string()).collect())
            .collect();
        let nulls = rows
            .iter()
            .map(|r| r.iter().map(|s| s.is_empty()).collect())
            .collect();
        RawCsv {
            headers: headers.iter().map(|s| s.to_string()).collect(),
            rows: rows_v,
            nulls,
            row_ids: (1..=rows.len()).collect(),
        }
    }

    fn declared(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn keep(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    /// `typed_dataframe` with no rename and a throwaway tally.
    fn typed(raw: &RawCsv, cols: &[&str], types: &[(&str, &str)]) -> DataFrame {
        let mut tally = ListMisparseTally::default();
        typed_dataframe(
            raw,
            &keep(cols),
            &declared(types),
            &HashMap::new(),
            &mut tally,
        )
        .expect("typed_dataframe")
    }

    fn date(y: i32, m: u32, d: u32) -> Value {
        Value::DateTime(NaiveDate::from_ymd_opt(y, m, d).unwrap())
    }

    #[test]
    fn every_declared_keyword_maps_to_its_column_type() {
        for (kw, want) in [
            ("string", ColumnType::String),
            ("str", ColumnType::String),
            ("int", ColumnType::Int64),
            ("integer", ColumnType::Int64),
            ("float", ColumnType::Float64),
            ("bool", ColumnType::Boolean),
            ("boolean", ColumnType::Boolean),
            ("date", ColumnType::DateTime),
            ("datetime", ColumnType::DateTime),
            ("validFrom", ColumnType::DateTime),
            ("validTo", ColumnType::DateTime),
            ("list", ColumnType::List),
            ("array", ColumnType::List),
        ] {
            assert_eq!(map_blueprint_type(kw), Some(want), "keyword '{kw}'");
        }
        // Spatial / unknown keywords are not column types: the build filters
        // them out of `declared`, so they fall through to inference.
        for kw in ["geometry", "location.lat", "location.lon", "map", "nope"] {
            assert_eq!(map_blueprint_type(kw), None, "keyword '{kw}'");
        }
    }

    #[test]
    fn declared_arms_carry_their_values() {
        let r = raw(
            &["s", "i", "f", "b", "d", "l"],
            &[
                &["x", "7", "1.5", "true", "2024-01-15", "[\"a\",\"b\"]"],
                &["y", "-2", "-0.25", "no", "2024-02-29T13:45:01", "[1,2]"],
            ],
        );
        let df = typed(
            &r,
            &["s", "i", "f", "b", "d", "l"],
            &[
                ("s", "string"),
                ("i", "int"),
                ("f", "float"),
                ("b", "bool"),
                ("d", "date"),
                ("l", "list"),
            ],
        );
        assert_eq!(df.row_count(), 2);
        assert_eq!(df.get_column_type("s"), Some(ColumnType::String));
        assert_eq!(df.get_column_type("i"), Some(ColumnType::Int64));
        assert_eq!(df.get_column_type("f"), Some(ColumnType::Float64));
        assert_eq!(df.get_column_type("b"), Some(ColumnType::Boolean));
        assert_eq!(df.get_column_type("d"), Some(ColumnType::DateTime));
        assert_eq!(df.get_column_type("l"), Some(ColumnType::List));

        assert_eq!(df.get_value(0, "s"), Some(Value::String("x".into())));
        assert_eq!(df.get_value(0, "i"), Some(Value::Int64(7)));
        assert_eq!(df.get_value(1, "i"), Some(Value::Int64(-2)));
        assert_eq!(df.get_value(0, "f"), Some(Value::Float64(1.5)));
        assert_eq!(df.get_value(1, "f"), Some(Value::Float64(-0.25)));
        assert_eq!(df.get_value(0, "b"), Some(Value::Boolean(true)));
        assert_eq!(df.get_value(1, "b"), Some(Value::Boolean(false)));
        assert_eq!(df.get_value(0, "d"), Some(date(2024, 1, 15)));
        // A datetime cell keeps only its date half — the blueprint has no
        // time-of-day column type.
        assert_eq!(df.get_value(1, "d"), Some(date(2024, 2, 29)));
        assert_eq!(
            df.get_value(0, "l"),
            Some(Value::List(vec![
                Value::String("a".into()),
                Value::String("b".into())
            ]))
        );
        assert_eq!(
            df.get_value(1, "l"),
            Some(Value::List(vec![Value::Int64(1), Value::Int64(2)]))
        );
    }

    #[test]
    fn every_boolean_spelling_the_arm_accepts() {
        let r = raw(
            &["b"],
            &[
                &["true"],
                &["TRUE"],
                &["1"],
                &["t"],
                &["yes"],
                &["Y"],
                &["false"],
                &["0"],
                &["f"],
                &["no"],
                &["N"],
                &["maybe"],
            ],
        );
        let df = typed(&r, &["b"], &[("b", "bool")]);
        let got: Vec<Option<Value>> = (0..df.row_count()).map(|i| df.get_value(i, "b")).collect();
        let t = Some(Value::Boolean(true));
        let f = Some(Value::Boolean(false));
        assert_eq!(
            got,
            vec![
                t.clone(),
                t.clone(),
                t.clone(),
                t.clone(),
                t.clone(),
                t,
                f.clone(),
                f.clone(),
                f.clone(),
                f.clone(),
                f,
                // Anything else is null, not an error and not `false`.
                None,
            ]
        );
    }

    #[test]
    fn undeclared_columns_take_the_inferred_type() {
        let r = raw(
            &["i", "f", "b", "d", "s", "mixed"],
            &[
                &["1", "1.5", "true", "2024-01-15", "abc", "1"],
                &["2", "2", "false", "2024-01-16", "def", "2.5"],
            ],
        );
        let df = typed(&r, &["i", "f", "b", "d", "s", "mixed"], &[]);
        assert_eq!(df.get_column_type("i"), Some(ColumnType::Int64));
        assert_eq!(df.get_column_type("f"), Some(ColumnType::Float64));
        assert_eq!(df.get_column_type("b"), Some(ColumnType::Boolean));
        // Inference has no date rule: an ISO-date column stays text unless the
        // blueprint declares it.
        assert_eq!(df.get_column_type("d"), Some(ColumnType::String));
        assert_eq!(df.get_column_type("s"), Some(ColumnType::String));
        // int + float in one column widens to float.
        assert_eq!(df.get_column_type("mixed"), Some(ColumnType::Float64));
        assert_eq!(df.get_value(0, "mixed"), Some(Value::Float64(1.0)));
        assert_eq!(
            df.get_value(0, "d"),
            Some(Value::String("2024-01-15".into()))
        );
    }

    #[test]
    fn inference_edge_cases() {
        // All-null and all-empty columns fall back to String.
        let r = raw(&["a", "b"], &[&["", ""], &["", ""]]);
        let df = typed(&r, &["a", "b"], &[]);
        assert_eq!(df.get_column_type("a"), Some(ColumnType::String));
        assert_eq!(df.get_value(0, "a"), None);

        // bool + int in one column: `saw_int` wins the ladder, and the bool
        // cells then parse as neither int nor float, so they become null.
        let r = raw(&["x"], &[&["true"], &["1"]]);
        let df = typed(&r, &["x"], &[]);
        assert_eq!(df.get_column_type("x"), Some(ColumnType::Int64));
        assert_eq!(df.get_value(0, "x"), None);
        assert_eq!(df.get_value(1, "x"), Some(Value::Int64(1)));

        // A single text cell anywhere in the column makes the whole column
        // text, whatever the majority looks like.
        let r = raw(&["x"], &[&["1"], &["2"], &["oops"]]);
        let df = typed(&r, &["x"], &[]);
        assert_eq!(df.get_column_type("x"), Some(ColumnType::String));
        assert_eq!(df.get_value(0, "x"), Some(Value::String("1".into())));
    }

    #[test]
    fn null_empty_and_whitespace_cells_per_arm() {
        // Column 0 is the empty (null) cell, column 1 is whitespace-only.
        let r = raw(
            &["s", "i", "f", "b", "d", "l"],
            &[
                &["", "", "", "", "", ""],
                &["   ", "   ", "   ", "   ", "   ", "   "],
            ],
        );
        let df = typed(
            &r,
            &["s", "i", "f", "b", "d", "l"],
            &[
                ("s", "string"),
                ("i", "int"),
                ("f", "float"),
                ("b", "bool"),
                ("d", "date"),
                ("l", "list"),
            ],
        );
        // Empty ≡ null in every arm.
        for c in ["s", "i", "f", "b", "d", "l"] {
            assert_eq!(df.get_value(0, c), None, "empty cell in '{c}'");
        }
        // Whitespace-only is null everywhere the arm trims — which is
        // everywhere except `String`, the one arm that does not trim.
        for c in ["i", "f", "b", "d", "l"] {
            assert_eq!(df.get_value(1, c), None, "whitespace cell in '{c}'");
        }
        assert_eq!(df.get_value(1, "s"), Some(Value::String("   ".into())));
    }

    #[test]
    fn the_string_arm_is_the_only_one_that_does_not_trim() {
        let r = raw(
            &["s", "i", "f", "b", "d"],
            &[&[" x ", " 1 ", " 1.5 ", " true ", " 2024-01-15 "]],
        );
        let df = typed(
            &r,
            &["s", "i", "f", "b", "d"],
            &[
                ("s", "string"),
                ("i", "int"),
                ("f", "float"),
                ("b", "bool"),
                ("d", "date"),
            ],
        );
        assert_eq!(df.get_value(0, "s"), Some(Value::String(" x ".into())));
        assert_eq!(df.get_value(0, "i"), Some(Value::Int64(1)));
        assert_eq!(df.get_value(0, "f"), Some(Value::Float64(1.5)));
        assert_eq!(df.get_value(0, "b"), Some(Value::Boolean(true)));
        assert_eq!(df.get_value(0, "d"), Some(date(2024, 1, 15)));
        // Inference trims too, so a padded numeric column is still numeric.
        let df = typed(&r, &["i"], &[]);
        assert_eq!(df.get_column_type("i"), Some(ColumnType::Int64));
    }

    #[test]
    fn int_declaration_takes_whole_number_floats_and_nulls_the_rest() {
        let r = raw(
            &["i"],
            &[
                &["1.0"],
                &["-3.000"],
                &["2.5"],
                &["1e3"],
                &["inf"],
                &["NaN"],
                &["abc"],
                &["9223372036854775807"],
            ],
        );
        let df = typed(&r, &["i"], &[("i", "int")]);
        let got: Vec<Option<Value>> = (0..df.row_count()).map(|i| df.get_value(i, "i")).collect();
        assert_eq!(
            got,
            vec![
                Some(Value::Int64(1)),
                Some(Value::Int64(-3)),
                // Fractional, non-finite and non-numeric all become null —
                // silently; there is no per-cell warning on this arm.
                None,
                Some(Value::Int64(1000)),
                None,
                None,
                None,
                Some(Value::Int64(i64::MAX)),
            ]
        );
    }

    #[test]
    fn date_declaration_accepts_iso_and_epoch_millis() {
        let r = raw(
            &["d"],
            &[
                &["2021-01-01"],
                &["2021-01-01 06:30:00"],
                &["2021-01-01T06:30:00"],
                // Bare digits under `date` are epoch milliseconds, so an id
                // column mis-declared as a date silently becomes 1970.
                &["1609459200000"],
                &["1609459200000.0"],
                &["7"],
                &["not-a-date"],
            ],
        );
        let df = typed(&r, &["d"], &[("d", "date")]);
        let got: Vec<Option<Value>> = (0..df.row_count()).map(|i| df.get_value(i, "d")).collect();
        assert_eq!(
            got,
            vec![
                Some(date(2021, 1, 1)),
                Some(date(2021, 1, 1)),
                Some(date(2021, 1, 1)),
                Some(date(2021, 1, 1)),
                Some(date(2021, 1, 1)),
                Some(date(1970, 1, 1)),
                None,
            ]
        );
    }

    #[test]
    fn list_cells_parse_json_and_wrap_anything_else() {
        let r = raw(
            &["l"],
            &[
                &["[\"a\",\"b\"]"],
                &["[1,2.5,true,null]"],
                &["[[1,2],{\"k\":1}]"],
                &["lone"],
                &["\"quoted\""],
                &["[]"],
            ],
        );
        let df = typed(&r, &["l"], &[("l", "list")]);
        assert_eq!(
            df.get_value(1, "l"),
            Some(Value::List(vec![
                Value::Int64(1),
                Value::Float64(2.5),
                Value::Boolean(true),
                Value::Null,
            ]))
        );
        // Nested arrays/objects are stringified, not kept as structure.
        assert_eq!(
            df.get_value(2, "l"),
            Some(Value::List(vec![
                Value::String("[1,2]".into()),
                Value::String("{\"k\":1}".into()),
            ]))
        );
        // A bare scalar is a one-element list, and so is a JSON string.
        assert_eq!(
            df.get_value(3, "l"),
            Some(Value::List(vec![Value::String("lone".into())]))
        );
        assert_eq!(
            df.get_value(4, "l"),
            Some(Value::List(vec![Value::String("quoted".into())]))
        );
        assert_eq!(df.get_value(5, "l"), Some(Value::List(vec![])));
    }

    #[test]
    fn a_separator_in_a_non_json_list_cell_is_tallied_once_per_column() {
        let r = raw(
            &["l"],
            &[&["ok"], &["a|b"], &["c;d"], &["e,f"], &["[\"g,h\"]"]],
        );
        let mut tally = ListMisparseTally::default();
        let df = typed_dataframe(
            &r,
            &keep(&["l"]),
            &declared(&[("l", "list")]),
            &HashMap::new(),
            &mut tally,
        )
        .unwrap();
        // The value is still kept whole — the tally is a warning, not a drop.
        assert_eq!(
            df.get_value(1, "l"),
            Some(Value::List(vec![Value::String("a|b".into())]))
        );
        let warnings = tally.into_warnings("node 'T'");
        assert_eq!(warnings.len(), 1, "one warning per column: {warnings:?}");
        let w = &warnings[0];
        assert!(
            w.starts_with("node 'T': column 'l' is declared list but 3 cell(s)"),
            "{w}"
        );
        // The first offending row is named by its source row id, with the
        // cell verbatim so the author can grep for it.
        assert!(w.contains("First at row 2: 'a|b'"), "{w}");
        // A JSON array containing a comma is not a misparse.
        assert!(!w.contains("g,h"), "{w}");
    }

    #[test]
    fn the_tally_names_the_source_row_not_the_chunk_row() {
        // A chunk carries the file's row ids, so a warning about a row in
        // chunk 3 names the file row, not the chunk-local index.
        let mut r = raw(&["l"], &[&["x|y"]]);
        r.row_ids = vec![5001];
        let mut tally = ListMisparseTally::default();
        typed_dataframe(
            &r,
            &keep(&["l"]),
            &declared(&[("l", "list")]),
            &HashMap::new(),
            &mut tally,
        )
        .unwrap();
        assert!(
            tally.into_warnings("j")[0].contains("First at row 5001"),
            "row provenance lost"
        );
    }

    #[test]
    fn a_long_misparse_cell_is_truncated_in_the_warning() {
        let long: String = std::iter::repeat_n('z', 200).collect::<String>() + "|tail";
        let r = raw(&["l"], &[&[long.as_str()]]);
        let mut tally = ListMisparseTally::default();
        typed_dataframe(
            &r,
            &keep(&["l"]),
            &declared(&[("l", "list")]),
            &HashMap::new(),
            &mut tally,
        )
        .unwrap();
        let w = tally.into_warnings("node 'T'").remove(0);
        assert!(w.contains(&format!("{}…", "z".repeat(80))), "{w}");
        assert!(!w.contains("tail"), "{w}");
    }

    #[test]
    fn rename_moves_the_output_name_only() {
        let r = raw(&["a", "b"], &[&["1", "x"]]);
        let mut rename = HashMap::new();
        rename.insert("a".to_string(), "renamed".to_string());
        let mut tally = ListMisparseTally::default();
        let df = typed_dataframe(
            &r,
            &keep(&["a", "b"]),
            // `declared_types` stays keyed by the *source* name.
            &declared(&[("a", "string")]),
            &rename,
            &mut tally,
        )
        .unwrap();
        assert_eq!(df.get_column_names(), vec!["renamed", "b"]);
        assert_eq!(df.get_column_type("renamed"), Some(ColumnType::String));
        assert_eq!(df.get_value(0, "renamed"), Some(Value::String("1".into())));
    }

    #[test]
    fn a_missing_keep_column_is_an_error_naming_the_headers() {
        let r = raw(&["a"], &[&["1"]]);
        let mut tally = ListMisparseTally::default();
        let err = typed_dataframe(
            &r,
            &keep(&["nope"]),
            &HashMap::new(),
            &HashMap::new(),
            &mut tally,
        )
        .unwrap_err();
        assert!(err.contains("Column 'nope' not found"), "{err}");
        assert!(err.contains("\"a\""), "{err}");
    }

    #[test]
    fn append_typed_columns_extends_an_existing_frame() {
        // The FK-edge loader builds its id pair first and appends properties.
        let r = raw(&["p"], &[&["1"], &["2"]]);
        let mut df = DataFrame::new(Vec::new());
        df.add_column(
            "src".to_string(),
            ColumnType::UniqueId,
            ColumnData::UniqueId(vec![Some(10), Some(11)]),
        )
        .unwrap();
        let mut tally = ListMisparseTally::default();
        append_typed_columns(
            &mut df,
            &r,
            &keep(&["p"]),
            &declared(&[("p", "int")]),
            &HashMap::new(),
            &mut tally,
        )
        .unwrap();
        assert_eq!(df.get_column_names(), vec!["src", "p"]);
        assert_eq!(df.get_value(1, "src"), Some(Value::UniqueId(11)));
        assert_eq!(df.get_value(1, "p"), Some(Value::Int64(2)));
    }

    #[test]
    fn the_arms_no_declaration_can_reach_still_return_full_length_columns() {
        // `map_blueprint_type` has no keyword for UniqueId / Timestamp / Map
        // and `infer_type` never yields them, so these arms are unreachable
        // through a blueprint. They must still be shape-correct.
        let r = raw(&["x"], &[&["7"], &[""], &["oops"]]);
        let mut tally = ListMisparseTally::default();
        match build_column_data(&r, 0, &ColumnType::UniqueId, "x", &mut tally).unwrap() {
            ColumnData::UniqueId(v) => assert_eq!(v, vec![Some(7), None, None]),
            other => panic!("wrong variant: {other:?}"),
        }
        match build_column_data(&r, 0, &ColumnType::Timestamp, "x", &mut tally).unwrap() {
            ColumnData::Timestamp(v) => assert_eq!(v, vec![None, None, None]),
            other => panic!("wrong variant: {other:?}"),
        }
        match build_column_data(&r, 0, &ColumnType::Map, "x", &mut tally).unwrap() {
            ColumnData::Map(v) => assert_eq!(v.len(), 3),
            other => panic!("wrong variant: {other:?}"),
        }
    }
}

#[cfg(test)]
mod incremental_inference_tests {
    use super::*;

    /// A deterministic xorshift, so the sweep is reproducible without a dep.
    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn cell(pick: u64) -> String {
        match pick % 7 {
            0 => String::new(),
            1 => " ".to_string(),
            2 => (pick % 100).to_string(),
            3 => format!("{}.5", pick % 50),
            4 => "true".to_string(),
            5 => "FALSE".to_string(),
            _ => format!("x{}", pick % 30),
        }
    }

    fn table(cells: &[String]) -> RawCsv {
        RawCsv {
            headers: vec!["c".to_string()],
            rows: cells.iter().map(|c| vec![c.clone()]).collect(),
            nulls: cells.iter().map(|c| vec![c.is_empty()]).collect(),
            row_ids: (1..=cells.len()).collect(),
        }
    }

    /// The pre-pass folds chunk by chunk; the buffered path types the whole
    /// table at once. The two must not disagree for any table or any split —
    /// that disagreement is exactly the defect this machinery removes.
    #[test]
    fn folding_chunk_by_chunk_matches_whole_table_inference() {
        let mut seed = 0x5eed_1234_u64;
        for case in 0..400 {
            let n = (rng(&mut seed) % 24) as usize + 1;
            let cells: Vec<String> = (0..n).map(|_| cell(rng(&mut seed))).collect();
            let whole = infer_type(&table(&cells), 0);

            // Random split points, plus the two degenerate ones.
            let chunk = match case % 4 {
                0 => 1,
                1 => n,
                _ => (rng(&mut seed) % n as u64) as usize + 1,
            };
            let mut folded = ColumnInference::default();
            for part in cells.chunks(chunk) {
                folded.observe_column(&table(part), 0);
            }
            assert_eq!(
                folded.resolve(),
                whole,
                "case {case}: cells {cells:?} split at {chunk}"
            );
        }
    }

    /// The resolved type travels back through the `declared_types` map as a
    /// keyword, so every type inference can produce must survive the
    /// round-trip.
    #[test]
    fn every_inferable_type_round_trips_through_its_keyword() {
        for ct in [
            ColumnType::String,
            ColumnType::Int64,
            ColumnType::Float64,
            ColumnType::Boolean,
        ] {
            let kw = inferred_type_keyword(&ct).expect("inference yields only these");
            assert_eq!(map_blueprint_type(kw), Some(ct), "{kw}");
        }
    }

    /// Every keyword `blueprint_type_keyword` hands back must be one
    /// `map_blueprint_type` reads, or a known type would be handed to the
    /// typing pass and silently ignored there.
    #[test]
    fn blueprint_type_keyword_round_trips_through_map_blueprint_type() {
        for ct in [
            ColumnType::String,
            ColumnType::Int64,
            ColumnType::Float64,
            ColumnType::Boolean,
            ColumnType::DateTime,
            ColumnType::List,
        ] {
            let kw = blueprint_type_keyword(&ct).expect("the vocabulary names these");
            assert_eq!(map_blueprint_type(kw), Some(ct), "{kw}");
        }
        for ct in [ColumnType::UniqueId, ColumnType::Timestamp, ColumnType::Map] {
            assert_eq!(blueprint_type_keyword(&ct), None, "{ct:?}");
        }
    }

    /// declared → known → inferred. The blueprint wins where it speaks; the
    /// input's own type answers where it is silent; inference is what is left.
    #[test]
    fn a_kept_column_resolves_declared_then_known_then_inferred() {
        let raw = RawCsv {
            headers: vec!["d".into(), "k".into(), "i".into()],
            rows: vec![vec!["1".into(), "1.0".into(), "7".into()]],
            nulls: vec![vec![false, false, false]],
            row_ids: vec![1],
        };
        let mut declared: HashMap<String, String> = HashMap::new();
        declared.insert("d".to_string(), "string".to_string());

        let mut known: HashMap<String, ColumnType> = HashMap::new();
        // The input knows all three; only the undeclared ones may take it.
        known.insert("d".to_string(), ColumnType::Int64);
        known.insert("k".to_string(), ColumnType::Float64);
        overlay_known_types(&mut declared, &known);

        assert_eq!(
            declared.get("d").map(String::as_str),
            Some("string"),
            "a declared column keeps the blueprint's type"
        );
        assert_eq!(
            declared.get("k").map(String::as_str),
            Some("float"),
            "an undeclared column the input knows takes the input's type"
        );
        assert!(
            !declared.contains_key("i"),
            "a column nobody knows is left to inference"
        );

        let df = typed_dataframe(
            &raw,
            &["d".to_string(), "k".to_string(), "i".to_string()],
            &declared,
            &HashMap::new(),
            &mut ListMisparseTally::default(),
        )
        .unwrap();
        assert_eq!(df.get_column_type("d"), Some(ColumnType::String));
        // Without the known type, "1.0" infers as Float64 too — the column
        // that proves the rung is `i`, which inference makes an Int64.
        assert_eq!(df.get_column_type("k"), Some(ColumnType::Float64));
        assert_eq!(df.get_column_type("i"), Some(ColumnType::Int64));
    }

    /// A type the vocabulary cannot name is not overlaid: the column falls
    /// through to inference rather than being dropped from the typing map.
    #[test]
    fn a_type_outside_the_vocabulary_is_not_overlaid() {
        let mut declared: HashMap<String, String> = HashMap::new();
        let mut known: HashMap<String, ColumnType> = HashMap::new();
        known.insert("t".to_string(), ColumnType::Timestamp);
        overlay_known_types(&mut declared, &known);
        assert!(declared.is_empty());
    }
}
