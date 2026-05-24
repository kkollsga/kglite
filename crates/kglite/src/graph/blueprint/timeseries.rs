//! Blueprint timeseries preparation.
//!
//! Translates a `TimeseriesSpec` into the columnar data that the graph's
//! `timeseries_store` and `timeseries_configs` expect. Mirrors the Python
//! loader's `_prepare_timeseries` (`loader.py:737`) plus the per-node
//! grouping done by `add_nodes`'s PyO3 wrapper.

use super::csv_loader::RawCsv;
use super::schema::{TimeKey, TimeseriesSpec};
use crate::graph::features::timeseries::{
    date_from_ymd, parse_date_query, validate_resolution, NodeTimeseries, TimeseriesConfig,
};
use chrono::NaiveDate;
use std::collections::HashMap;

/// A timeseries spec, resolved against the actual CSV headers.
pub struct ResolvedTimeseries {
    /// Column names the node loader should ignore (time + channel cols — they're
    /// stored separately from node properties).
    pub excluded_columns: Vec<String>,
    /// Resolution keyword (year / month / day), either user-supplied or auto-detected.
    pub resolution: String,
    /// Channel rename plan: (csv_column, channel_name).
    pub channel_rename: Vec<(String, String)>,
    /// Channel names in their final form.
    pub channel_names: Vec<String>,
    /// Units per channel.
    pub units: HashMap<String, String>,
    /// Raw time_key (for later parsing once filtered rows are known).
    pub time_key: ResolvedTimeKey,
}

pub enum ResolvedTimeKey {
    SingleCol(String),
    /// Ordered vector of columns: [year, month, day, ...] as present.
    OrderedCols(Vec<String>),
}

/// Resolve the timeseries spec against actual CSV headers. Returns
/// `None` when the spec is missing.
pub fn resolve(spec: &TimeseriesSpec, _raw: &RawCsv) -> Result<ResolvedTimeseries, String> {
    let time_key = match &spec.time_key {
        TimeKey::Single(c) => ResolvedTimeKey::SingleCol(c.clone()),
        TimeKey::Composite(map) => {
            let order = ["year", "month", "day", "hour", "minute"];
            let mut ordered = Vec::new();
            let mut found_gap = false;
            for k in order.iter() {
                match map.get(*k) {
                    Some(col) => {
                        if found_gap {
                            return Err(format!(
                                "timeseries time_key has '{}' but is missing a higher-level component",
                                k
                            ));
                        }
                        ordered.push(col.clone());
                    }
                    None => found_gap = true,
                }
            }
            if ordered.is_empty() {
                return Err("timeseries time_key must contain at least 'year'".to_string());
            }
            ResolvedTimeKey::OrderedCols(ordered)
        }
    };

    let resolution = if let Some(r) = &spec.resolution {
        validate_resolution(r)?;
        r.clone()
    } else {
        match &time_key {
            ResolvedTimeKey::SingleCol(_) => "month".to_string(),
            ResolvedTimeKey::OrderedCols(cs) => match cs.len() {
                1 => "year".to_string(),
                2 => "month".to_string(),
                _ => "day".to_string(),
            },
        }
    };

    let mut channel_rename = Vec::new();
    let mut channel_names = Vec::new();
    for (ch_name, csv_col) in &spec.channels {
        if csv_col != ch_name {
            channel_rename.push((csv_col.clone(), ch_name.clone()));
        }
        channel_names.push(ch_name.clone());
    }

    let mut excluded = Vec::new();
    match &time_key {
        ResolvedTimeKey::SingleCol(c) => excluded.push(c.clone()),
        ResolvedTimeKey::OrderedCols(cs) => excluded.extend(cs.iter().cloned()),
    }
    for (csv_col, _) in &channel_rename {
        excluded.push(csv_col.clone());
    }
    for ch_name in &channel_names {
        // If the channel column kept its original name (no rename), exclude it too.
        if !excluded.contains(ch_name) {
            excluded.push(ch_name.clone());
        }
    }

    let units = spec
        .units
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    Ok(ResolvedTimeseries {
        excluded_columns: excluded,
        resolution,
        channel_rename,
        channel_names,
        units,
        time_key,
    })
}

/// Drop rows where any time component below `year` is zero — pandas loader
/// does this at `loader.py:634`. These are aggregate rows (e.g. month=0
/// annual totals) that would otherwise create spurious node entries.
pub fn drop_zero_time_components(raw: &mut RawCsv, spec: &TimeseriesSpec) {
    let TimeKey::Composite(map) = &spec.time_key else {
        return;
    };
    let mut zero_cols: Vec<usize> = Vec::new();
    for (label, col) in map {
        if label == "year" {
            continue;
        }
        if let Some(idx) = raw.col_index(col) {
            zero_cols.push(idx);
        }
    }
    if zero_cols.is_empty() {
        return;
    }

    let mut new_rows = Vec::with_capacity(raw.row_count());
    let mut new_nulls = Vec::with_capacity(raw.row_count());
    for r in 0..raw.row_count() {
        let drop = zero_cols.iter().any(|&idx| {
            if raw.nulls[r][idx] {
                return false;
            }
            raw.rows[r][idx].trim() == "0"
        });
        if !drop {
            new_rows.push(std::mem::take(&mut raw.rows[r]));
            new_nulls.push(std::mem::take(&mut raw.nulls[r]));
        }
    }
    raw.rows = new_rows;
    raw.nulls = new_nulls;
}

/// Build per-node timeseries from raw rows. Returns `(node_key → NodeTimeseries)`
/// where `node_key` is the raw pk value as a string — the caller resolves it
/// against `DirGraph::lookup_by_id_normalized`.
pub fn build_node_timeseries(
    raw: &RawCsv,
    pk_col: &str,
    resolved: &ResolvedTimeseries,
) -> Result<HashMap<String, NodeTimeseries>, String> {
    // Extract time keys
    let time_keys = extract_time_keys(raw, &resolved.time_key)?;

    // Extract channels. CSV holds the original column name; the stored channel
    // uses the blueprint's channel key. Missing channels are silently filled
    // with NaN — matches the Python loader's pandas-backed behaviour.
    let mut channel_cols: Vec<(String, Vec<f64>)> =
        Vec::with_capacity(resolved.channel_names.len());
    for ch_name in &resolved.channel_names {
        let csv_col = resolved
            .channel_rename
            .iter()
            .find(|(_, new)| new == ch_name)
            .map(|(orig, _)| orig.clone())
            .unwrap_or_else(|| ch_name.clone());
        let idx = raw.col_index(&csv_col).or_else(|| raw.col_index(ch_name));
        let values = match idx {
            Some(i) => raw
                .rows
                .iter()
                .enumerate()
                .map(|(r, row)| {
                    if raw.nulls[r][i] {
                        f64::NAN
                    } else {
                        row[i].trim().parse::<f64>().unwrap_or(f64::NAN)
                    }
                })
                .collect::<Vec<_>>(),
            None => vec![f64::NAN; raw.row_count()],
        };
        channel_cols.push((ch_name.clone(), values));
    }

    // Group row indices by pk value
    let pk_idx = raw
        .col_index(pk_col)
        .ok_or_else(|| format!("timeseries pk column '{}' not in CSV", pk_col))?;

    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (r, row) in raw.rows.iter().enumerate() {
        if raw.nulls[r][pk_idx] {
            continue;
        }
        let k = row[pk_idx].clone();
        groups.entry(k).or_default().push(r);
    }

    let mut out: HashMap<String, NodeTimeseries> = HashMap::new();
    for (k, mut indices) in groups {
        indices.sort_by(|&a, &b| time_keys[a].cmp(&time_keys[b]));
        let keys: Vec<NaiveDate> = indices.iter().map(|&i| time_keys[i]).collect();
        let channels: HashMap<String, Vec<f64>> = channel_cols
            .iter()
            .map(|(name, col)| (name.clone(), indices.iter().map(|&i| col[i]).collect()))
            .collect();
        out.insert(k, NodeTimeseries { keys, channels });
    }

    Ok(out)
}

fn extract_time_keys(raw: &RawCsv, time_key: &ResolvedTimeKey) -> Result<Vec<NaiveDate>, String> {
    let mut keys = Vec::with_capacity(raw.row_count());
    match time_key {
        ResolvedTimeKey::SingleCol(col) => {
            let idx = raw
                .col_index(col)
                .ok_or_else(|| format!("timeseries time column '{}' not in CSV", col))?;
            for (r, row) in raw.rows.iter().enumerate() {
                if raw.nulls[r][idx] {
                    keys.push(NaiveDate::from_ymd_opt(1970, 1, 1).unwrap());
                    continue;
                }
                let (d, _) = parse_date_query(&row[idx])?;
                keys.push(d);
            }
        }
        ResolvedTimeKey::OrderedCols(cols) => {
            let indices: Vec<usize> = cols
                .iter()
                .map(|c| {
                    raw.col_index(c)
                        .ok_or_else(|| format!("timeseries time column '{}' not in CSV", c))
                })
                .collect::<Result<_, _>>()?;
            for r in 0..raw.row_count() {
                let get = |i: usize| -> u32 {
                    if raw.nulls[r][indices[i]] {
                        0
                    } else {
                        raw.rows[r][indices[i]].trim().parse::<u32>().unwrap_or(0)
                    }
                };
                let year = if raw.nulls[r][indices[0]] {
                    0
                } else {
                    raw.rows[r][indices[0]].trim().parse::<i32>().unwrap_or(0)
                };
                let month = if indices.len() > 1 { get(1).max(1) } else { 1 };
                let day = if indices.len() > 2 { get(2).max(1) } else { 1 };
                keys.push(date_from_ymd(year, month, day)?);
            }
        }
    }
    Ok(keys)
}

/// Merge channels + units into an existing `TimeseriesConfig` on the graph.
pub fn merge_config(
    existing: Option<&TimeseriesConfig>,
    resolved: &ResolvedTimeseries,
) -> TimeseriesConfig {
    let mut channels = existing.map(|c| c.channels.clone()).unwrap_or_default();
    for ch in &resolved.channel_names {
        if !channels.contains(ch) {
            channels.push(ch.clone());
        }
    }
    let mut units = existing.map(|c| c.units.clone()).unwrap_or_default();
    for (k, v) in &resolved.units {
        units.insert(k.clone(), v.clone());
    }
    let bin_type = existing.and_then(|c| c.bin_type.clone());
    TimeseriesConfig {
        resolution: resolved.resolution.clone(),
        channels,
        units,
        bin_type,
    }
}
