//! Row-level `RawCsv` operations shared by the build phases.

use super::super::table::RawCsv;
use std::collections::HashSet;

impl RawCsv {
    pub(super) fn clone_raw(&self) -> RawCsv {
        RawCsv {
            headers: self.headers.clone(),
            rows: self.rows.clone(),
            nulls: self.nulls.clone(),
            row_ids: self.row_ids.clone(),
        }
    }
}

/// Keep only the first row per unique pk value; rows with a null pk all
/// pass through. Used for timeseries specs: one node per carrier, time
/// samples stored separately.
pub(super) fn dedupe_by_pk(raw: &RawCsv, pk_col: &str) -> RawCsv {
    let Some(idx) = raw.col_index(pk_col) else {
        return raw.clone_raw();
    };
    let mut seen: HashSet<String> = HashSet::new();
    let mut new_rows = Vec::new();
    let mut new_nulls = Vec::new();
    let mut new_row_ids = Vec::new();
    for r in 0..raw.row_count() {
        if raw.nulls[r][idx] {
            new_rows.push(raw.rows[r].clone());
            new_nulls.push(raw.nulls[r].clone());
            new_row_ids.push(raw.row_id(r));
            continue;
        }
        let key = raw.rows[r][idx].clone();
        if seen.insert(key) {
            new_rows.push(raw.rows[r].clone());
            new_nulls.push(raw.nulls[r].clone());
            new_row_ids.push(raw.row_id(r));
        }
    }
    RawCsv {
        headers: raw.headers.clone(),
        rows: new_rows,
        nulls: new_nulls,
        row_ids: new_row_ids,
    }
}

/// A row-subset of `raw`, carrying the source row numbers so a diagnostic
/// still names a row the author can find.
pub(super) fn subset_rows(raw: &RawCsv, rows: &[usize]) -> RawCsv {
    RawCsv {
        headers: raw.headers.clone(),
        rows: rows.iter().map(|&r| raw.rows[r].clone()).collect(),
        nulls: rows.iter().map(|&r| raw.nulls[r].clone()).collect(),
        row_ids: rows.iter().map(|&r| raw.row_id(r)).collect(),
    }
}
