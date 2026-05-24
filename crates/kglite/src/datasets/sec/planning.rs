//! Storage-mode planning — estimate the built graph's resident size
//! for a given scope, and pick the cheapest backend that fits.
//!
//! Pure formulas, ported from the Python `sec/wrapper.py`. The size
//! formula is calibrated against the loader's measured node-count
//! behaviour (see `docs/guides/sec.md` "Sizing").

/// The full SEC filer universe — used to scale a `cik_list` scope
/// down to a fraction of the whole.
const FULL_UNIVERSE: f64 = 6000.0;

/// Estimate the built graph's resident size in GB.
///
/// `cik_count` is the number of CIKs in the scope filter; `None` (or
/// `Some(0)`) means the full universe.
pub fn predict_graph_size_gb(
    years: u32,
    detailed: u32,
    cik_count: Option<usize>,
    include_subsidiaries: bool,
    include_xbrl_metrics: bool,
    include_8k_events: bool,
) -> f64 {
    let cik_fraction = match cik_count {
        None | Some(0) => 1.0,
        Some(n) => (n as f64 / FULL_UNIVERSE).min(1.0),
    };
    // Shallow filing index: ~0.1 GB per ingested year.
    let mut gb = 0.1 * years as f64 * cik_fraction;
    if detailed > 0 {
        let d = detailed as f64;
        // Form 4 + 13F + Exhibit 21 baseline.
        gb += 0.6 * d * cik_fraction;
        if include_xbrl_metrics {
            gb += 4.0 * d * cik_fraction;
        }
        if include_8k_events {
            gb += 1.0 * d * cik_fraction;
        }
        if include_subsidiaries {
            gb += 0.05 * d * cik_fraction;
        }
    }
    gb
}

/// Pick a storage backend for an estimated graph size: `memory` below
/// 4 GB, `mapped` from 4 to 16 GB, `disk` above.
pub fn pick_storage_mode(predicted_gb: f64) -> &'static str {
    if predicted_gb < 4.0 {
        "memory"
    } else if predicted_gb < 16.0 {
        "mapped"
    } else {
        "disk"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_universe_scope_uses_fraction_one() {
        // years=10, detailed=2, every payload → 1 + 1.2 + 8 + 2 + 0.1.
        let gb = predict_graph_size_gb(10, 2, None, true, true, true);
        assert!((gb - 12.3).abs() < 1e-9, "got {gb}");
    }

    #[test]
    fn cik_list_scales_size_down() {
        let full = predict_graph_size_gb(10, 2, None, true, true, true);
        let sliced = predict_graph_size_gb(10, 2, Some(600), true, true, true);
        // 600 / 6000 = 10% of the universe.
        assert!((sliced - full * 0.1).abs() < 1e-9, "got {sliced}");
    }

    #[test]
    fn detailed_zero_skips_payload_terms() {
        let gb = predict_graph_size_gb(5, 0, None, true, true, true);
        assert!((gb - 0.5).abs() < 1e-9, "got {gb}");
    }

    #[test]
    fn empty_cik_list_is_full_universe() {
        assert_eq!(
            predict_graph_size_gb(10, 2, Some(0), true, true, true),
            predict_graph_size_gb(10, 2, None, true, true, true),
        );
    }

    #[test]
    fn storage_mode_thresholds() {
        assert_eq!(pick_storage_mode(2.0), "memory");
        assert_eq!(pick_storage_mode(3.99), "memory");
        assert_eq!(pick_storage_mode(4.0), "mapped");
        assert_eq!(pick_storage_mode(15.9), "mapped");
        assert_eq!(pick_storage_mode(16.0), "disk");
        assert_eq!(pick_storage_mode(100.0), "disk");
    }
}
