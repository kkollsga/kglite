//! The `{col: val}` / `{col: {op: val}}` DSL that blueprints use for row
//! filtering.
//!
//! Filters operate on raw CSV strings so ordering is preserved: filter,
//! then type-coerce — same as the Python loader at `loader.py:604`.

use super::csv_loader::RawCsv;
use serde_json::Value as Json;

/// Apply all filter predicates in `filt`, retaining only rows where every
/// condition matches. Unknown columns are silently skipped (pandas parity).
pub fn apply_filter(raw: &mut RawCsv, filt: &indexmap::IndexMap<String, Json>) {
    if filt.is_empty() {
        return;
    }

    let mut keep: Vec<bool> = vec![true; raw.row_count()];

    for (col, val) in filt {
        let Some(src_idx) = raw.col_index(col) else {
            continue;
        };

        match val {
            Json::Object(ops) => {
                for (op, operand) in ops {
                    for (r, row) in raw.rows.iter().enumerate() {
                        if !keep[r] {
                            continue;
                        }
                        let cell_null = raw.nulls[r][src_idx];
                        let ok = eval_op(op, cell_null, &row[src_idx], operand);
                        if !ok {
                            keep[r] = false;
                        }
                    }
                }
            }
            _ => {
                // Simple equality
                for (r, row) in raw.rows.iter().enumerate() {
                    if !keep[r] {
                        continue;
                    }
                    let cell_null = raw.nulls[r][src_idx];
                    let ok = eq(cell_null, &row[src_idx], val);
                    if !ok {
                        keep[r] = false;
                    }
                }
            }
        }
    }

    // Compact
    let mut new_rows = Vec::with_capacity(raw.row_count());
    let mut new_nulls = Vec::with_capacity(raw.row_count());
    for (r, k) in keep.iter().enumerate() {
        if *k {
            new_rows.push(std::mem::take(&mut raw.rows[r]));
            new_nulls.push(std::mem::take(&mut raw.nulls[r]));
        }
    }
    raw.rows = new_rows;
    raw.nulls = new_nulls;
}

fn eval_op(op: &str, cell_null: bool, cell: &str, operand: &Json) -> bool {
    match op {
        "=" => eq(cell_null, cell, operand),
        "!=" => !eq(cell_null, cell, operand),
        ">" => cmp(cell_null, cell, operand)
            .map(|o| o.is_gt())
            .unwrap_or(false),
        ">=" => cmp(cell_null, cell, operand)
            .map(|o| !o.is_lt())
            .unwrap_or(false),
        "<" => cmp(cell_null, cell, operand)
            .map(|o| o.is_lt())
            .unwrap_or(false),
        "<=" => cmp(cell_null, cell, operand)
            .map(|o| !o.is_gt())
            .unwrap_or(false),
        _ => true, // unknown op — be permissive (matches loader.py which just skips)
    }
}

fn eq(cell_null: bool, cell: &str, operand: &Json) -> bool {
    match operand {
        Json::Null => cell_null,
        Json::Bool(b) => {
            if cell_null {
                return false;
            }
            let s = cell.trim().to_ascii_lowercase();
            match s.as_str() {
                "true" | "1" | "t" | "yes" | "y" => *b,
                "false" | "0" | "f" | "no" | "n" => !*b,
                _ => false,
            }
        }
        Json::Number(n) => {
            if cell_null {
                return false;
            }
            let c = cell.trim();
            if let Some(iv) = n.as_i64() {
                // Accept whole-number floats too: "260" == 260, "260.0" == 260
                if let Ok(ci) = c.parse::<i64>() {
                    return ci == iv;
                }
                if let Ok(cf) = c.parse::<f64>() {
                    return cf == iv as f64;
                }
                false
            } else if let Some(fv) = n.as_f64() {
                c.parse::<f64>().map(|cf| cf == fv).unwrap_or(false)
            } else {
                false
            }
        }
        Json::String(s) => {
            if cell_null {
                return false;
            }
            cell == s.as_str()
        }
        Json::Array(_) | Json::Object(_) => false,
    }
}

fn cmp(cell_null: bool, cell: &str, operand: &Json) -> Option<std::cmp::Ordering> {
    if cell_null {
        return None;
    }
    let c = cell.trim();
    match operand {
        Json::Number(n) => {
            if let Some(iv) = n.as_i64() {
                if let Ok(ci) = c.parse::<i64>() {
                    return Some(ci.cmp(&iv));
                }
                if let Ok(cf) = c.parse::<f64>() {
                    return cf.partial_cmp(&(iv as f64));
                }
                None
            } else if let Some(fv) = n.as_f64() {
                c.parse::<f64>().ok().and_then(|cf| cf.partial_cmp(&fv))
            } else {
                None
            }
        }
        Json::String(s) => Some(c.cmp(s.as_str())),
        _ => None,
    }
}
