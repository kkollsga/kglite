//! Slice grammar — `SliceSpec` filters fetched and extracted entities
//! by CIK whitelist, form-type whitelist, and year range. Single source
//! of truth applied uniformly by `fetch.rs`, `extract.rs`, and the
//! Python `SEC.open()` wrapper.
//!
//! A `None` arm means "no restriction" — pure additive filtering, so
//! `SliceSpec::default()` admits everything.

use std::collections::HashSet;

/// User-supplied filters applied before any network or extract work.
#[derive(Debug, Clone, Default)]
pub struct SliceSpec {
    /// If `Some`, only CIKs in the set pass. CIKs use the integer form
    /// (no zero-padding).
    pub cik_list: Option<HashSet<u64>>,
    /// If `Some`, only form types in the set pass. Match is exact
    /// (case-sensitive — SEC's master.idx is uppercase).
    pub form_types: Option<HashSet<String>>,
    /// If `Some`, only filings with `filed_date` in the inclusive year
    /// range `[start, end]` pass. Dates are matched against the
    /// `YYYY-MM-DD` prefix.
    pub year_range: Option<(u16, u16)>,
}

impl SliceSpec {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cik_list(mut self, ciks: impl IntoIterator<Item = u64>) -> Self {
        self.cik_list = Some(ciks.into_iter().collect());
        self
    }

    pub fn with_form_types(mut self, forms: impl IntoIterator<Item = String>) -> Self {
        self.form_types = Some(forms.into_iter().collect());
        self
    }

    pub fn with_year_range(mut self, start: u16, end: u16) -> Self {
        self.year_range = Some((start, end));
        self
    }

    /// Build a `SliceSpec` from optional filter args. Empty / None args
    /// produce an unrestricted slice. Convenience wrapper for callers
    /// (CLI tools, Python bindings, JSON/YAML config loaders) that
    /// would otherwise repeat the `if let Some(...) && !empty { ... }`
    /// pattern around the builder methods. Lifted from kglite-py in 0.10.1.
    pub fn from_optional_filters(
        cik_list: Option<Vec<u64>>,
        form_types: Option<Vec<String>>,
        year_range: Option<(u16, u16)>,
    ) -> Self {
        let mut s = Self::default();
        if let Some(ciks) = cik_list {
            if !ciks.is_empty() {
                s = s.with_cik_list(ciks);
            }
        }
        if let Some(forms) = form_types {
            if !forms.is_empty() {
                s = s.with_form_types(forms);
            }
        }
        if let Some((lo, hi)) = year_range {
            s = s.with_year_range(lo, hi);
        }
        s
    }

    /// `true` if the CIK is admitted (or if no CIK filter is set).
    pub fn cik_matches(&self, cik: u64) -> bool {
        match &self.cik_list {
            Some(set) => set.contains(&cik),
            None => true,
        }
    }

    /// `true` if the form type is admitted (or if no form filter is set).
    pub fn form_matches(&self, form_type: &str) -> bool {
        match &self.form_types {
            Some(set) => set.contains(form_type),
            None => true,
        }
    }

    /// `true` if the filed date is in range (or if no range is set).
    /// Accepts dates in `YYYY-MM-DD` or `YYYYMMDD` format.
    pub fn date_matches(&self, filed_date: &str) -> bool {
        let Some((lo, hi)) = self.year_range else {
            return true;
        };
        let year_str: String = filed_date.chars().take(4).collect();
        match year_str.parse::<u16>() {
            Ok(y) => y >= lo && y <= hi,
            Err(_) => false,
        }
    }

    /// Combined gate over (cik, form_type, filed_date). True iff all
    /// applicable filters pass.
    pub fn matches(&self, cik: u64, form_type: &str, filed_date: &str) -> bool {
        self.cik_matches(cik) && self.form_matches(form_type) && self.date_matches(filed_date)
    }

    /// `true` if no filter is set — every entity passes.
    pub fn is_unrestricted(&self) -> bool {
        self.cik_list.is_none() && self.form_types.is_none() && self.year_range.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_admits_everything() {
        let s = SliceSpec::default();
        assert!(s.is_unrestricted());
        assert!(s.matches(320193, "10-K", "2024-11-01"));
        assert!(s.matches(0, "", ""));
    }

    #[test]
    fn cik_list_filters() {
        let s = SliceSpec::new().with_cik_list([320193u64, 789019u64]);
        assert!(s.cik_matches(320193));
        assert!(s.cik_matches(789019));
        assert!(!s.cik_matches(123));
        assert!(!s.is_unrestricted());
    }

    #[test]
    fn form_types_filters_case_sensitive() {
        let s = SliceSpec::new().with_form_types(["10-K".to_string(), "10-Q".to_string()]);
        assert!(s.form_matches("10-K"));
        assert!(s.form_matches("10-Q"));
        assert!(!s.form_matches("10-K/A")); // amendments are separate codes
        assert!(!s.form_matches("8-K"));
    }

    #[test]
    fn year_range_filters_inclusively() {
        let s = SliceSpec::new().with_year_range(2020, 2024);
        assert!(s.date_matches("2020-01-15"));
        assert!(s.date_matches("2024-12-31"));
        assert!(s.date_matches("2022-06-30"));
        assert!(!s.date_matches("2019-12-31"));
        assert!(!s.date_matches("2025-01-01"));
    }

    #[test]
    fn date_matches_handles_dense_format() {
        let s = SliceSpec::new().with_year_range(2024, 2024);
        assert!(s.date_matches("20240928"));
        assert!(!s.date_matches("20230928"));
    }

    #[test]
    fn date_matches_handles_malformed_input() {
        let s = SliceSpec::new().with_year_range(2024, 2024);
        // Non-numeric year prefix → reject (caller treats as parse failure)
        assert!(!s.date_matches("not-a-date"));
        assert!(!s.date_matches(""));
    }

    #[test]
    fn combined_matches_requires_all_filters_to_pass() {
        let s = SliceSpec::new()
            .with_cik_list([320193u64])
            .with_form_types(["10-K".to_string()])
            .with_year_range(2024, 2024);
        assert!(s.matches(320193, "10-K", "2024-11-01"));
        // Right CIK but wrong form
        assert!(!s.matches(320193, "8-K", "2024-11-01"));
        // Right form but wrong year
        assert!(!s.matches(320193, "10-K", "2023-11-01"));
        // Right form + year but wrong CIK
        assert!(!s.matches(789019, "10-K", "2024-11-01"));
    }
}
