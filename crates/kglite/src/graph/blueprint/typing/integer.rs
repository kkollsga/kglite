//! Declared integer cells accept exact whole decimal/scientific spellings.

/// Preserve whole-number CSV spellings without rounding through f64. Invalid,
/// fractional and out-of-range cells retain the typed loader's NULL policy.
pub(super) fn parse_exact_i64(text: &str) -> Option<i64> {
    let text = text.trim();
    if let Ok(value) = text.parse::<i64>() {
        return Some(value);
    }
    let (negative, unsigned) = if let Some(rest) = text.strip_prefix('-') {
        (true, rest)
    } else {
        (false, text.strip_prefix('+').unwrap_or(text))
    };
    let (mantissa, exponent) = if let Some(pos) = unsigned.find(['e', 'E']) {
        let exponent = &unsigned[pos + 1..];
        let digits = exponent
            .strip_prefix('+')
            .or_else(|| exponent.strip_prefix('-'))
            .unwrap_or(exponent);
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        (&unsigned[..pos], exponent.parse::<i64>().ok())
    } else {
        (unsigned, Some(0))
    };
    let digits = DecimalDigits::scan(mantissa)?;
    let Some((start, end)) = digits.nonzero_bounds else {
        return Some(0);
    };
    // An exponent outside i64 cannot be cancelled by an addressable mantissa.
    // Zero above is exact even for such an exponent; other values cannot fit.
    let scale = i128::from(exponent?) - digits.fractional as i128 + digits.trailing_zeros as i128;
    let significant = &mantissa.as_bytes()[start..end];
    let width = significant.iter().filter(|&&b| b != b'.').count();
    if scale < 0 || width as i128 + scale > 19 {
        return None;
    }
    let mut magnitude = 0_u64;
    for &digit in significant {
        if digit != b'.' {
            magnitude = magnitude
                .checked_mul(10)?
                .checked_add(u64::from(digit - b'0'))?;
        }
    }
    for _ in 0..scale {
        magnitude = magnitude.checked_mul(10)?;
    }
    let signed = if negative {
        -i128::from(magnitude)
    } else {
        i128::from(magnitude)
    };
    i64::try_from(signed).ok()
}

struct DecimalDigits {
    nonzero_bounds: Option<(usize, usize)>,
    fractional: usize,
    trailing_zeros: usize,
}

impl DecimalDigits {
    fn scan(mantissa: &str) -> Option<Self> {
        let mut result = Self {
            nonzero_bounds: None,
            fractional: 0,
            trailing_zeros: 0,
        };
        let mut saw_dot = false;
        let mut saw_digit = false;
        for (pos, digit) in mantissa.bytes().enumerate() {
            if digit == b'.' && !saw_dot {
                saw_dot = true;
                continue;
            }
            if !digit.is_ascii_digit() {
                return None;
            }
            saw_digit = true;
            result.fractional += usize::from(saw_dot);
            if digit == b'0' {
                result.trailing_zeros += 1;
            } else {
                let start = result.nonzero_bounds.map_or(pos, |(start, _)| start);
                result.nonzero_bounds = Some((start, pos + 1));
                result.trailing_zeros = 0;
            }
        }
        saw_digit.then_some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_whole_decimal_and_scientific_cells() {
        for (text, expected) in [
            ("9223372036854775807", i64::MAX),
            ("9223372036854775807.0", i64::MAX),
            ("9.223372036854775807e18", i64::MAX),
            ("-9223372036854775808.000", i64::MIN),
            ("-9.223372036854775808E+18", i64::MIN),
            ("9007199254740993.0", 9_007_199_254_740_993),
            ("9007199254740993000e-3", 9_007_199_254_740_993),
            (" +001.000e+3 ", 1000),
            (".10e1", 1),
            ("-3.", -3),
            ("-0.000e-999999999999999999999", 0),
        ] {
            assert_eq!(parse_exact_i64(text), Some(expected), "{text}");
        }
    }

    #[test]
    fn fractional_and_outside_range_cells_never_saturate() {
        for text in [
            "9223372036854775808",
            "9223372036854775808.0",
            "-9223372036854775809",
            "-9223372036854775809.0",
            "9.223372036854775808e18",
            "-9.223372036854775809e18",
            "9007199254740993.1",
            "1e-999",
            "1e999",
            "0.1",
            "NaN",
            "inf",
            "",
            "abc",
        ] {
            assert_eq!(parse_exact_i64(text), None, "{text}");
        }
    }

    #[test]
    fn typed_column_uses_exact_parser_and_preserves_nulls() {
        use super::super::{build_column_data, ColumnData, ColumnType, ListMisparseTally, RawCsv};
        let cells = [
            "9223372036854775807.0",
            "-9223372036854775808.0",
            "9223372036854775808",
            "-9223372036854775809",
            "1e3",
            "1.5",
            "",
        ];
        let raw = RawCsv {
            headers: vec!["v".into()],
            rows: cells.iter().map(|s| vec![s.to_string()]).collect(),
            nulls: cells.iter().map(|s| vec![s.is_empty()]).collect(),
            row_ids: (1..=cells.len()).collect(),
        };
        let column = build_column_data(
            &raw,
            0,
            &ColumnType::Int64,
            "v",
            &mut ListMisparseTally::default(),
        )
        .unwrap();
        let ColumnData::Int64(values) = column else {
            panic!("expected Int64 column")
        };
        assert_eq!(
            values,
            vec![
                Some(i64::MAX),
                Some(i64::MIN),
                None,
                None,
                Some(1000),
                None,
                None
            ]
        );
    }
}
