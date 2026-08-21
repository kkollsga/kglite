//! What the bulk loaders do with an input row — or a column — they cannot use
//! as given.
//!
//! `add_nodes` and `add_connections` have always *tolerated* bad input: a row
//! whose id will not key a node is dropped, a heterogeneous object column is
//! stringified, and the caller learns about it from a `UserWarning` and the
//! returned report. That is the right default for an exploratory load and the
//! wrong one for a pipeline, where a silently short load is a data-loss bug
//! discovered downstream.
//!
//! `on_invalid` lets the caller choose per call:
//!
//! - `"warn"` (default) — today's behaviour, unchanged.
//! - `"error"` — refuse the whole call *before* it mutates anything, naming
//!   how many rows are unusable, which row is the first, and what it holds.
//! - `"skip"` — do the same work as `"warn"` without the warning; the counts
//!   stay in the returned report.
//!
//! The vocabulary is a Python kwarg, not engine surface: `"warn"` means
//! "`warnings.warn`", which is a thing only this binding can do. The engine
//! keeps reporting facts; this module decides what the wheel does with them.

use crate::datatypes::values::{DataFrame, Value};
use pyo3::prelude::*;
use pyo3::Bound;

/// The `on_invalid` policy of one loader call.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OnInvalid {
    /// Emit a `UserWarning` and continue. The historical behaviour.
    #[default]
    Warn,
    /// Refuse the call. Raises before anything is written.
    Error,
    /// Continue silently; the returned report still carries the counts.
    Skip,
}

impl OnInvalid {
    /// Parse the kwarg.
    pub fn parse(value: &str) -> PyResult<Self> {
        match value {
            "warn" => Ok(Self::Warn),
            "error" => Ok(Self::Error),
            "skip" => Ok(Self::Skip),
            other => Err(crate::error_py::ArgumentError::new_err(format!(
                "on_invalid must be 'warn', 'error' or 'skip'. Got '{other}'."
            ))),
        }
    }

    /// Whether this policy still wants the loader's summary `UserWarning`.
    pub fn warns(self) -> bool {
        matches!(self, Self::Warn)
    }

    /// Whether this policy turns an unusable input into a raised error.
    pub fn raises(self) -> bool {
        matches!(self, Self::Error)
    }
}

/// Report `message` the way `policy` asks: warn, raise, or say nothing.
///
/// Used for the diagnostics that are decided *during* conversion (an object
/// column about to be stringified), where there is no report dict to fall back
/// on — the message is the only channel.
pub fn report(py: Python<'_>, policy: OnInvalid, message: String) -> PyResult<()> {
    match policy {
        OnInvalid::Skip => Ok(()),
        OnInvalid::Error => Err(crate::error_py::ArgumentError::new_err(message)),
        OnInvalid::Warn => {
            let cmsg = std::ffi::CString::new(message).unwrap_or_default();
            PyErr::warn(
                py,
                py.get_type::<pyo3::exceptions::PyUserWarning>().as_any(),
                cmsg.as_c_str(),
                1,
            )
        }
    }
}

/// The rows of a converted frame whose id column holds nothing a loader can
/// key a node with.
pub struct UnusableRows {
    /// How many rows are unusable.
    pub count: usize,
    /// Positional index of the first of them, against the converted frame.
    pub first_row: usize,
    /// Which of the scanned id columns that first row failed on.
    pub first_column: String,
}

/// Scan `columns` of `df` for rows the loaders drop.
///
/// A row is unusable when its id cell is absent or null — exactly the two
/// cases `maintain::add_nodes` counts as `skipped_null_id` / `skipped_parse_fail`
/// and `resolve_endpoints` counts as a null endpoint. A value that failed to
/// parse during conversion (`to_u32` on an out-of-range integer, say) is
/// already null by the time it lands here, so both arrive as one shape.
///
/// Returns `None` when every row is usable. Only called under
/// [`OnInvalid::Error`], so the default load pays nothing for it.
pub fn scan_unusable_rows(df: &DataFrame, columns: &[&str]) -> Option<UnusableRows> {
    let indices: Vec<(usize, &str)> = columns
        .iter()
        .filter_map(|name| df.get_column_index(name).map(|idx| (idx, *name)))
        .collect();
    if indices.is_empty() {
        return None;
    }
    let mut found: Option<UnusableRows> = None;
    for row_idx in 0..df.row_count() {
        for (col_idx, name) in &indices {
            let usable = matches!(
                df.get_value_by_index(row_idx, *col_idx),
                Some(value) if !matches!(value, Value::Null)
            );
            if usable {
                continue;
            }
            match &mut found {
                Some(seen) => seen.count += 1,
                None => {
                    found = Some(UnusableRows {
                        count: 1,
                        first_row: row_idx,
                        first_column: (*name).to_string(),
                    })
                }
            }
            // One row is unusable once, however many of its id columns are bad.
            break;
        }
    }
    found
}

/// `repr()` of the raw cell the caller supplied at `frame[column][row]`, for
/// the error message.
///
/// Positional (`.iloc`) because the converted frame is positional: the
/// conversion resets the index, so row *n* of the converted frame is row *n*
/// of the frame that was handed to it. Any failure to read it degrades to
/// `"null"` rather than replacing a data error with an access error.
pub fn raw_cell_repr(frame: Option<&Bound<'_, PyAny>>, column: &str, row: usize) -> String {
    let rendered = frame.and_then(|frame| {
        let series = frame.get_item(column).ok()?;
        let cell = series.getattr("iloc").ok()?.get_item(row).ok()?;
        cell.repr().ok().map(|r| r.to_string())
    });
    rendered.unwrap_or_else(|| "null".to_string())
}

/// The `on_invalid="error"` refusal for a loader whose input has unusable rows.
///
/// Names the count, the first offending row and what that row actually holds,
/// because "3 rows skipped" without a row number is the report the caller
/// already had and could not act on.
pub fn unusable_rows_err(
    loader: &str,
    total_rows: usize,
    bad: &UnusableRows,
    raw_value: &str,
) -> PyErr {
    crate::error_py::ArgumentError::new_err(format!(
        "{loader}: {} of {total_rows} rows cannot be loaded — row {} has {} in ID column '{}'. \
         Nothing was written (on_invalid='error'). Pass on_invalid='warn' to load the rest and \
         report the skips, or on_invalid='skip' to load the rest silently.",
        bad.count, bad.first_row, raw_value, bad.first_column
    ))
}
