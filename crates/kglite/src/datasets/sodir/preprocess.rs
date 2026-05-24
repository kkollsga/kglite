//! Sodir CSV pre-processing — derived FK columns the blueprint needs.
//!
//! The blueprint expects integer foreign keys for cross-CSV edges, but
//! some raw FactMaps tables identify referenced rows by name instead.
//! `apply` adds the missing FK columns by joining on the natural key.
//! Each step is idempotent — re-running is a no-op — and a no-op when
//! its target CSV is missing. Ported from the Python `preprocess.py`.
//!
//! Idempotency note: the `petreg_licence` children gate on the FK
//! column merely being *present*; the other three steps gate on it
//! being present *and* holding a non-empty value. This asymmetry is
//! deliberate — preserved exactly from the Python.

use std::collections::HashMap;
use std::path::Path;

use crate::datasets::sodir::error::{Result, SodirError};

/// Per-step unmapped-row counts (FK gaps). `None` = step did not run
/// (its target CSV was absent or already processed).
#[derive(Debug, Clone, Default)]
pub struct PreprocessReport {
    pub petreg_licence_pk: Option<usize>,
    pub seismic_progress_fk: Option<usize>,
    pub chrono_parent_fk: Option<usize>,
    pub announced_block_fk: Option<usize>,
}

/// Run every applicable FK-derivation step on the CSVs under `csv_dir`.
pub fn apply(csv_dir: &Path) -> Result<PreprocessReport> {
    let mut report = PreprocessReport::default();

    if csv_dir.join("petreg_licence.csv").is_file() {
        report.petreg_licence_pk = Some(add_petreg_licence_pk(csv_dir)?);
    }
    if csv_dir.join("seismic_acquisition.csv").is_file()
        && csv_dir.join("seismic_acquisition_progress.csv").is_file()
    {
        report.seismic_progress_fk = Some(add_seismic_progress_fk(csv_dir)?);
    }
    if csv_dir.join("strat_chrono.csv").is_file() {
        report.chrono_parent_fk = Some(add_chrono_parent_fk(csv_dir)?);
    }
    if csv_dir.join("block.csv").is_file() && csv_dir.join("announced_history.csv").is_file() {
        report.announced_block_fk = Some(add_announced_block_fk(csv_dir)?);
    }

    Ok(report)
}

// ─────────────────────────── the four joins ───────────────────────────

/// `petreg_licence.csv` ships a GUID PK that nothing joins cleanly.
/// Add a sequential integer `ptl_id` and propagate it to the message
/// / licensee / operator child CSVs. Children gate on `ptl_id` being
/// *present* (the GUID join can legitimately leave it all-empty).
fn add_petreg_licence_pk(csv_dir: &Path) -> Result<usize> {
    let parent_path = csv_dir.join("petreg_licence.csv");
    let (mut headers, mut rows) = read_csv(&parent_path)?;
    if !has_column(&headers, "ptl_id") {
        let ids: Vec<String> = (1..=rows.len()).map(|i| i.to_string()).collect();
        set_or_append_column(&mut headers, &mut rows, "ptl_id", ids);
        write_csv(&parent_path, &headers, &rows)?;
    }
    let guid_to_id = build_lookup(&headers, &rows, "ptlPetregLicenceID", "ptl_id");

    let mut unmapped_total = 0;
    for child in [
        "petreg_licence_message.csv",
        "petreg_licence_licensee.csv",
        "petreg_licence_operator.csv",
    ] {
        let child_path = csv_dir.join(child);
        if !child_path.is_file() {
            continue;
        }
        let (mut ch, mut cr) = read_csv(&child_path)?;
        if has_column(&ch, "ptl_id") {
            continue; // already pre-processed
        }
        let (mapped, unmapped) = map_column(&ch, &cr, "ptlPetregLicenceID", &guid_to_id);
        unmapped_total += unmapped;
        set_or_append_column(&mut ch, &mut cr, "ptl_id", mapped);
        write_csv(&child_path, &ch, &cr)?;
    }
    Ok(unmapped_total)
}

/// `seismic_acquisition_progress.csv` joins to `seismic_acquisition`
/// by name; resolve the NPDID into `seaNpdidSurvey`.
fn add_seismic_progress_fk(csv_dir: &Path) -> Result<usize> {
    let (sh, sr) = read_csv(&csv_dir.join("seismic_acquisition.csv"))?;
    let progress_path = csv_dir.join("seismic_acquisition_progress.csv");
    let (mut ph, mut pr) = read_csv(&progress_path)?;
    if column_has_value(&ph, &pr, "seaNpdidSurvey") {
        return Ok(0);
    }
    let name_to_npdid = build_lookup(&sh, &sr, "seaName", "seaNpdidSurvey");
    let (mapped, unmapped) = map_column(&ph, &pr, "seaSurveyName", &name_to_npdid);
    set_or_append_column(&mut ph, &mut pr, "seaNpdidSurvey", mapped);
    write_csv(&progress_path, &ph, &pr)?;
    Ok(unmapped)
}

/// `strat_chrono.csv` is self-referencing — each row names its parent
/// stratigraphic unit. Resolve the parent name to NPDID.
fn add_chrono_parent_fk(csv_dir: &Path) -> Result<usize> {
    let path = csv_dir.join("strat_chrono.csv");
    let (mut h, mut r) = read_csv(&path)?;
    if column_has_value(&h, &r, "strat_chrono_parent_npdid") {
        return Ok(0);
    }
    if !has_column(&h, "strat_chrono_name") || !has_column(&h, "strat_chrono_parent_name") {
        return Ok(0);
    }
    let name_to_npdid = build_lookup(&h, &r, "strat_chrono_name", "NPDID_strat_chrono");
    let (mapped, unmapped) = map_column(&h, &r, "strat_chrono_parent_name", &name_to_npdid);
    set_or_append_column(&mut h, &mut r, "strat_chrono_parent_npdid", mapped);
    write_csv(&path, &h, &r)?;
    Ok(unmapped)
}

/// `announced_history.csv` references blocks by name; resolve to
/// `blcNpdidBlock`.
fn add_announced_block_fk(csv_dir: &Path) -> Result<usize> {
    let (bh, br) = read_csv(&csv_dir.join("block.csv"))?;
    let path = csv_dir.join("announced_history.csv");
    let (mut h, mut r) = read_csv(&path)?;
    if column_has_value(&h, &r, "blcNpdidBlock") {
        return Ok(0);
    }
    let name_to_npdid = build_lookup(&bh, &br, "blcName", "blcNpdidBlock");
    let (mapped, unmapped) = map_column(&h, &r, "block", &name_to_npdid);
    set_or_append_column(&mut h, &mut r, "blcNpdidBlock", mapped);
    write_csv(&path, &h, &r)?;
    Ok(unmapped)
}

// ─────────────────────────── CSV helpers ───────────────────────────

/// Read a CSV fully into `(headers, rows)`. Rows shorter than the
/// header are padded so index access is always in bounds.
fn read_csv(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|e| SodirError::Csv(format!("open {}: {e}", path.display())))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| SodirError::Csv(format!("headers {}: {e}", path.display())))?
        .iter()
        .map(String::from)
        .collect();
    let ncols = headers.len();
    let mut rows = Vec::new();
    for rec in rdr.records() {
        let rec = rec.map_err(|e| SodirError::Csv(format!("row {}: {e}", path.display())))?;
        let mut row: Vec<String> = rec.iter().map(String::from).collect();
        if row.len() < ncols {
            row.resize(ncols, String::new());
        }
        rows.push(row);
    }
    Ok((headers, rows))
}

/// Write `headers` + `rows` back to `path`.
fn write_csv(path: &Path, headers: &[String], rows: &[Vec<String>]) -> Result<()> {
    let mut wtr = csv::WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Necessary)
        .from_path(path)
        .map_err(|e| SodirError::Csv(format!("open {}: {e}", path.display())))?;
    wtr.write_record(headers)
        .map_err(|e| SodirError::Csv(format!("header {}: {e}", path.display())))?;
    for row in rows {
        wtr.write_record(row)
            .map_err(|e| SodirError::Csv(format!("row {}: {e}", path.display())))?;
    }
    wtr.flush()?;
    Ok(())
}

fn col_index(headers: &[String], name: &str) -> Option<usize> {
    headers.iter().position(|h| h == name)
}

fn has_column(headers: &[String], name: &str) -> bool {
    headers.iter().any(|h| h == name)
}

/// True if the column is present and at least one row has a non-empty
/// value in it.
fn column_has_value(headers: &[String], rows: &[Vec<String>], col: &str) -> bool {
    match col_index(headers, col) {
        Some(i) => rows.iter().any(|r| r.get(i).is_some_and(|c| !c.is_empty())),
        None => false,
    }
}

/// Build a `key → value` lookup from two columns, skipping rows with
/// an empty key or empty value.
fn build_lookup(
    headers: &[String],
    rows: &[Vec<String>],
    key: &str,
    val: &str,
) -> HashMap<String, String> {
    let (Some(ki), Some(vi)) = (col_index(headers, key), col_index(headers, val)) else {
        return HashMap::new();
    };
    let mut map = HashMap::new();
    for r in rows {
        if let (Some(k), Some(v)) = (r.get(ki), r.get(vi)) {
            if !k.is_empty() && !v.is_empty() {
                map.insert(k.clone(), v.clone());
            }
        }
    }
    map
}

/// Map each row's `key_col` value through `lookup`. Returns the new
/// column's cells (empty string for unmapped rows) and the unmapped
/// count.
fn map_column(
    headers: &[String],
    rows: &[Vec<String>],
    key_col: &str,
    lookup: &HashMap<String, String>,
) -> (Vec<String>, usize) {
    let ki = col_index(headers, key_col);
    let mut cells = Vec::with_capacity(rows.len());
    let mut unmapped = 0;
    for r in rows {
        let key = ki.and_then(|i| r.get(i)).map(String::as_str).unwrap_or("");
        match lookup.get(key) {
            Some(v) => cells.push(v.clone()),
            None => {
                cells.push(String::new());
                unmapped += 1;
            }
        }
    }
    (cells, unmapped)
}

/// Overwrite an existing column's cells, or append a new column.
fn set_or_append_column(
    headers: &mut Vec<String>,
    rows: &mut [Vec<String>],
    col: &str,
    values: Vec<String>,
) {
    match col_index(headers, col) {
        Some(i) => {
            for (r, v) in rows.iter_mut().zip(values) {
                r[i] = v;
            }
        }
        None => {
            headers.push(col.to_string());
            for (r, v) in rows.iter_mut().zip(values) {
                r.push(v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }

    #[test]
    fn petreg_pk_added_and_propagated() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        write(
            d,
            "petreg_licence.csv",
            "ptlPetregLicenceID,name\nGUID-A,Alpha\nGUID-B,Beta\n",
        );
        write(
            d,
            "petreg_licence_message.csv",
            "ptlPetregLicenceID,text\nGUID-A,hello\nGUID-X,orphan\n",
        );

        let report = apply(d).unwrap();
        assert_eq!(report.petreg_licence_pk, Some(1)); // GUID-X unmapped

        let (ph, pr) = read_csv(&d.join("petreg_licence.csv")).unwrap();
        let pk = col_index(&ph, "ptl_id").unwrap();
        assert_eq!(pr[0][pk], "1");
        assert_eq!(pr[1][pk], "2");

        let (mh, mr) = read_csv(&d.join("petreg_licence_message.csv")).unwrap();
        let mpk = col_index(&mh, "ptl_id").unwrap();
        assert_eq!(mr[0][mpk], "1"); // GUID-A → 1
        assert_eq!(mr[1][mpk], ""); // GUID-X unmapped
    }

    #[test]
    fn seismic_progress_fk_joined() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        write(
            d,
            "seismic_acquisition.csv",
            "seaName,seaNpdidSurvey\nSURVEY-1,9001\nSURVEY-2,9002\n",
        );
        write(
            d,
            "seismic_acquisition_progress.csv",
            "seaSurveyName,pct\nSURVEY-1,50\nSURVEY-2,90\n",
        );
        let report = apply(d).unwrap();
        assert_eq!(report.seismic_progress_fk, Some(0));

        let (h, r) = read_csv(&d.join("seismic_acquisition_progress.csv")).unwrap();
        let fk = col_index(&h, "seaNpdidSurvey").unwrap();
        assert_eq!(r[0][fk], "9001");
        assert_eq!(r[1][fk], "9002");
    }

    #[test]
    fn apply_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        write(
            d,
            "petreg_licence.csv",
            "ptlPetregLicenceID,name\nGUID-A,Alpha\n",
        );
        write(
            d,
            "petreg_licence_operator.csv",
            "ptlPetregLicenceID,op\nGUID-A,Equinor\n",
        );
        write(d, "block.csv", "blcName,blcNpdidBlock\n1/2,7001\n");
        write(d, "announced_history.csv", "block,year\n1/2,2020\n");

        apply(d).unwrap();
        let snapshot: Vec<(String, String)> = ["petreg_licence.csv", "announced_history.csv"]
            .iter()
            .map(|n| (n.to_string(), std::fs::read_to_string(d.join(n)).unwrap()))
            .collect();

        // Second apply must produce byte-identical CSVs.
        apply(d).unwrap();
        for (name, before) in snapshot {
            let after = std::fs::read_to_string(d.join(&name)).unwrap();
            assert_eq!(before, after, "{name} changed on second apply");
        }
    }

    #[test]
    fn missing_target_is_a_noop() {
        let tmp = tempfile::tempdir().unwrap();
        // No CSVs at all — apply should run clean and report nothing.
        let report = apply(tmp.path()).unwrap();
        assert!(report.petreg_licence_pk.is_none());
        assert!(report.seismic_progress_fk.is_none());
        assert!(report.chrono_parent_fk.is_none());
        assert!(report.announced_block_fk.is_none());
    }
}
