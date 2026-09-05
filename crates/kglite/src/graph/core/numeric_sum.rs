//! Exact integer reduction without partition-dependent Int64 overflow.

/// Ordinary totals use one checked i64 addition. A partial overflow widens
/// permanently; only finalization requires the complete result to fit Int64.
/// Floating accumulation and caller admission policies are independent.
#[derive(Clone, Copy, Debug)]
pub(crate) enum IntegerSum {
    Narrow(i64),
    Wide(i128),
    Overflow,
}

impl Default for IntegerSum {
    fn default() -> Self {
        Self::Narrow(0)
    }
}

impl IntegerSum {
    pub(crate) fn add(&mut self, value: i64) {
        *self = match *self {
            Self::Narrow(total) => match total.checked_add(value) {
                Some(total) => Self::Narrow(total),
                // Two i64 inputs always have an exact i128 sum.
                None => Self::Wide(i128::from(total) + i128::from(value)),
            },
            Self::Wide(total) => total
                .checked_add(i128::from(value))
                .map_or(Self::Overflow, Self::Wide),
            Self::Overflow => Self::Overflow,
        };
    }

    pub(crate) fn merge(&mut self, other: Self) {
        match (*self, other) {
            (_, Self::Overflow) | (Self::Overflow, _) => *self = Self::Overflow,
            (_, Self::Narrow(value)) => self.add(value),
            (Self::Narrow(value), Self::Wide(total)) => {
                *self = Self::Wide(total);
                self.add(value);
            }
            (Self::Wide(left), Self::Wide(right)) => {
                *self = left.checked_add(right).map_or(Self::Overflow, Self::Wide);
            }
        }
    }

    pub(crate) fn finish(self) -> Result<i64, String> {
        let total = match self {
            Self::Narrow(total) => Some(total),
            Self::Wide(total) => i64::try_from(total).ok(),
            Self::Overflow => None,
        };
        total.ok_or_else(|| {
            "Integer overflow in sum: result is outside the 64-bit integer range".to_string()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_integer_sum_preserves_large_values_and_partition_cancellation() {
        let mut total = IntegerSum::default();
        total.add(9_007_199_254_740_993);
        assert_eq!(total.finish().unwrap(), 9_007_199_254_740_993);
        total.add(-9_007_199_254_740_992);
        assert_eq!(total.finish().unwrap(), 1);
        let mut left = IntegerSum::default();
        left.add(i64::MAX);
        left.add(1);
        assert!(left.finish().is_err());
        let mut right = IntegerSum::default();
        right.add(-i64::MAX);
        left.merge(right);
        assert_eq!(left.finish().unwrap(), 1);
    }

    #[test]
    fn exact_integer_sum_checks_final_bounds_without_saturation() {
        for bound in [i64::MIN, i64::MAX] {
            let mut total = IntegerSum::default();
            total.add(bound);
            assert_eq!(total.finish().unwrap(), bound);
            total.add(if bound < 0 { -1 } else { 1 });
            assert!(total
                .finish()
                .unwrap_err()
                .contains("Integer overflow in sum"));
        }
        let mut poisoned = IntegerSum::Wide(i128::MAX);
        poisoned.add(1);
        assert!(poisoned.finish().is_err());
        let mut total = IntegerSum::default();
        total.merge(poisoned);
        assert!(total.finish().is_err());
    }
}

#[cfg(test)]
#[test]
fn integer_sum_wide_partition_merges_remain_exact() {
    let mut positive = IntegerSum::default();
    positive.add(i64::MAX);
    positive.add(1);
    let mut negative = IntegerSum::default();
    negative.add(i64::MIN);
    negative.add(-1);
    let mut merged = positive;
    merged.merge(negative);
    assert_eq!(merged.finish().unwrap(), -1);
    let mut narrow = IntegerSum::default();
    narrow.add(-1);
    narrow.merge(positive);
    assert_eq!(narrow.finish().unwrap(), i64::MAX);
    let mut narrow = IntegerSum::default();
    narrow.add(1);
    negative.merge(narrow);
    assert_eq!(negative.finish().unwrap(), i64::MIN);
}
