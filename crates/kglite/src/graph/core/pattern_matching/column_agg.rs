//! Per-column running statistics for the column kernel: the part of a scan
//! that folds the rows a predicate mask kept into count, exact integer sum,
//! and extremes. [`column_kernel::scan`](super::column_kernel::scan) owns the
//! row walk and the mask; this module owns what is accumulated from it.
//!
//! Every statistic here is independent of the order rows are visited in, so
//! the kernel's row-order walk equals the generic scan's bucket-order walk.
//! The two that are *not* — a float sum, and the sign of a zero extreme — are
//! deliberately not computed; the driver declines instead.

use super::column_kernel::Cells;

/// Which statistics of a column are needed beyond the non-null count.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(super) struct Want {
    /// Exact `i128` sum (Int64 columns).
    pub sum: bool,
    /// Sum of absolute values (Int64 columns): bounds every partial `f64`
    /// sum, so `sum as f64` is exact when it stays within 2^53.
    pub abs_sum: bool,
    /// Extremes.
    pub min_max: bool,
}

impl Want {
    pub(super) fn union(self, other: Want) -> Want {
        Want {
            sum: self.sum | other.sum,
            abs_sum: self.abs_sum | other.abs_sum,
            min_max: self.min_max | other.min_max,
        }
    }
}

/// One column to fold, with the statistics wanted from it.
#[derive(Clone, Copy)]
pub(super) struct StatCol<'a> {
    pub cells: Cells<'a>,
    pub want: Want,
}

/// What the kernel found in one column over the matched, live rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ColStats {
    /// Matched rows whose cell is non-null.
    pub non_null: u64,
    pub sum: i128,
    pub abs_sum: u128,
    /// Valid only when `non_null > 0` on an Int64 column.
    pub int_min: i64,
    pub int_max: i64,
    /// Over the non-NaN cells; valid only when `non_nan > 0`.
    pub float_min: f64,
    pub float_max: f64,
    pub non_nan: u64,
    pub nan_seen: bool,
}

impl ColStats {
    pub(super) fn new() -> Self {
        ColStats {
            non_null: 0,
            sum: 0,
            abs_sum: 0,
            int_min: i64::MAX,
            int_max: i64::MIN,
            float_min: f64::INFINITY,
            float_max: f64::NEG_INFINITY,
            non_nan: 0,
            nan_seen: false,
        }
    }

    /// The Float64 minimum under the engine's total order (NaN highest): the
    /// smallest non-NaN cell, NaN only when every cell is NaN.
    pub(super) fn float_min_value(&self) -> Option<f64> {
        match (self.non_nan, self.nan_seen) {
            (0, false) => None,
            (0, true) => Some(f64::NAN),
            _ => Some(self.float_min),
        }
    }

    /// The Float64 maximum under the total order: NaN when any cell is NaN.
    pub(super) fn float_max_value(&self) -> Option<f64> {
        match (self.non_nan, self.nan_seen) {
            (_, true) => Some(f64::NAN),
            (0, false) => None,
            _ => Some(self.float_max),
        }
    }

    /// Fold rows `start..start + mask.len()` that `mask` kept.
    pub(super) fn fold(&mut self, col: &StatCol<'_>, start: usize, mask: &[u8]) {
        let end = start + mask.len();
        match col.cells {
            Cells::Int { data, nulls } => {
                self.fold_int(&data[start..end], &nulls[start..end], mask, col.want)
            }
            Cells::Float { data, nulls } => {
                self.fold_float(&data[start..end], &nulls[start..end], mask, col.want)
            }
        }
    }

    fn fold_int(&mut self, data: &[i64], nulls: &[u8], mask: &[u8], want: Want) {
        let keep = |m: u8, z: u8| m & (z == 0) as u8;
        self.non_null += mask
            .iter()
            .zip(nulls)
            .map(|(&m, &z)| keep(m, z) as u64)
            .sum::<u64>();
        if want.sum {
            // x = hi * 2^32 + lo with 0 <= lo < 2^32: both partial sums stay
            // far inside i64 for a 1024-row chunk, and the vectorised loop
            // needs no 128-bit lane.
            let (mut hi, mut lo) = (0i64, 0i64);
            for ((&m, &x), &z) in mask.iter().zip(data).zip(nulls) {
                let v = x & -(keep(m, z) as i64);
                hi += v >> 32;
                lo += v & 0xffff_ffff;
            }
            self.sum += ((hi as i128) << 32) + lo as i128;
        }
        if want.abs_sum {
            let (mut hi, mut lo) = (0u64, 0u64);
            for ((&m, &x), &z) in mask.iter().zip(data).zip(nulls) {
                let v = x.unsigned_abs() & (keep(m, z) as u64).wrapping_neg();
                hi += v >> 32;
                lo += v & 0xffff_ffff;
            }
            self.abs_sum += ((hi as u128) << 32) + lo as u128;
        }
        if want.min_max {
            let (mut lo, mut hi) = (i64::MAX, i64::MIN);
            for ((&m, &x), &z) in mask.iter().zip(data).zip(nulls) {
                let kept = keep(m, z) != 0;
                lo = lo.min(if kept { x } else { i64::MAX });
                hi = hi.max(if kept { x } else { i64::MIN });
            }
            self.int_min = self.int_min.min(lo);
            self.int_max = self.int_max.max(hi);
        }
    }

    fn fold_float(&mut self, data: &[f64], nulls: &[u8], mask: &[u8], want: Want) {
        let keep = |m: u8, z: u8| m & (z == 0) as u8;
        self.non_null += mask
            .iter()
            .zip(nulls)
            .map(|(&m, &z)| keep(m, z) as u64)
            .sum::<u64>();
        if want.min_max {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut non_nan, mut nan) = (0u64, 0u64);
            for ((&m, &x), &z) in mask.iter().zip(data).zip(nulls) {
                let kept = keep(m, z) != 0;
                let usable = kept & !x.is_nan();
                non_nan += usable as u64;
                nan += (kept & x.is_nan()) as u64;
                lo = lo.min(if usable { x } else { f64::INFINITY });
                hi = hi.max(if usable { x } else { f64::NEG_INFINITY });
            }
            self.float_min = self.float_min.min(lo);
            self.float_max = self.float_max.max(hi);
            self.non_nan += non_nan;
            self.nan_seen |= nan > 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::pattern_matching::column_kernel::scan;

    fn all() -> Want {
        Want {
            sum: true,
            abs_sum: true,
            min_max: true,
        }
    }

    /// Rows 0..6: row 1 tombstoned, row 4 null. Hand-computed over the live,
    /// non-null rows 0, 2, 3, 5.
    #[test]
    fn int_statistics_by_hand() {
        let data = [7i64, 1000, -3, i64::MIN, 99, 4];
        let nulls = [0u8, 0, 0, 0, 1, 0];
        let tomb = [false, true, false, false, false, false];
        let cols = [StatCol {
            cells: Cells::Int {
                data: &data,
                nulls: &nulls,
            },
            want: all(),
        }];
        let out = scan(6, &tomb, &[], &cols).unwrap();
        let s = out.stats[0];
        assert_eq!(s.non_null, 4);
        assert_eq!(s.sum, 7 - 3 + 4 + i64::MIN as i128);
        assert_eq!(s.abs_sum, 7 + 3 + 4 + (1u128 << 63));
        assert_eq!((s.int_min, s.int_max), (i64::MIN, 7));
    }

    #[test]
    fn float_extremes_put_nan_highest_and_ignore_dead_rows() {
        let data = [2.5f64, f64::NAN, -1.0, -9.0, 8.0];
        let nulls = [0u8; 5];
        let tomb = [false, false, false, true, false];
        let cols = [StatCol {
            cells: Cells::Float {
                data: &data,
                nulls: &nulls,
            },
            want: all(),
        }];
        let s = scan(5, &tomb, &[], &cols).unwrap().stats[0];
        assert_eq!(s.non_null, 4);
        assert_eq!(s.float_min_value(), Some(-1.0), "the dead -9.0 is not seen");
        assert!(s.float_max_value().unwrap().is_nan());
        // Without the NaN row the maximum is the largest cell.
        let tomb = [false, true, false, true, false];
        let s = scan(5, &tomb, &[], &cols).unwrap().stats[0];
        assert_eq!(s.float_max_value(), Some(8.0));
        // Nothing live and kept: no extremes at all.
        let s = scan(5, &[true; 5], &[], &cols).unwrap().stats[0];
        assert_eq!((s.float_min_value(), s.float_max_value()), (None, None));
    }
}
