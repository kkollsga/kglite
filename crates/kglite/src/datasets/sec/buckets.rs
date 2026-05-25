//! SEC form-type → per-filing-fetcher bucket mapping.
//!
//! SEC's master.idx and submissions.zip use slightly different
//! spellings (e.g. `"SCHEDULE 13D"` vs `"SC 13D"`); we accept both.
//! Per-bucket fetcher dispatch happens in the wheel's wrapper at
//! `kglite/datasets/sec/wrapper.py::_dispatch_per_filing_fetches`
//! today — future Go / JS / JVM bindings call [`resolve_fetch_buckets`]
//! here to get the same `form_type → bucket` resolution without
//! re-implementing the mapping table.
//!
//! Lifted from Python `_FORM_BUCKETS` + `_resolve_fetch_buckets`
//! in the 2026-05-25 dataset-wrapper preparation pass.

/// A per-filing fetcher bucket. Each variant maps to the SEC form
/// strings that bucket accepts (`matching_forms`) and to the
/// `_sec_internal.fetch_*_batch` call the wheel currently dispatches.
///
/// A binding wrapping SEC reads `processed/filing_index.csv`, groups
/// filings by [`SecFormBucket::from_form_string`], and calls the
/// matching batch fetcher (which today only exists in the wheel — the
/// per-batch fetchers themselves are the work tracked in
/// `consider-for-future.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SecFormBucket {
    /// Ownership forms 3, 3/A — initial insider holdings.
    Form3,
    /// Ownership forms 4, 4/A — insider transactions.
    Form4,
    /// Ownership forms 5, 5/A — annual insider statement.
    Form5,
    /// Form 144, 144/A — proposed sale by affiliates.
    Form144,
    /// 13F-HR, 13F-HR/A — institutional manager holdings (quarterly).
    Form13f,
    /// 8-K, 8-K/A — current report (material events).
    Form8k,
    /// Schedule 13D — beneficial ownership > 5%, active investor.
    Sc13d,
    /// Schedule 13G — beneficial ownership > 5%, passive investor.
    Sc13g,
    /// DEF 14A / DEFA14A / PRE 14A — proxy statement.
    Def14a,
    /// 10-K, 10-K/A — annual report; source filings for Exhibit 21.
    Form10k,
}

impl SecFormBucket {
    /// SEC form strings this bucket accepts. Includes variant
    /// spellings (e.g. `"SCHEDULE 13D"` and `"SC 13D"`) and amendment
    /// suffixes (`/A`) the SEC uses inconsistently.
    pub const fn matching_forms(self) -> &'static [&'static str] {
        match self {
            SecFormBucket::Form3 => &["3", "3/A"],
            SecFormBucket::Form4 => &["4", "4/A"],
            SecFormBucket::Form5 => &["5", "5/A"],
            SecFormBucket::Form144 => &["144", "144/A"],
            SecFormBucket::Form13f => &["13F-HR", "13F-HR/A"],
            SecFormBucket::Form8k => &["8-K", "8-K/A"],
            SecFormBucket::Sc13d => &["SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A"],
            SecFormBucket::Sc13g => &["SC 13G", "SC 13G/A", "SCHEDULE 13G", "SCHEDULE 13G/A"],
            SecFormBucket::Def14a => &["DEF 14A", "DEFA14A", "PRE 14A"],
            SecFormBucket::Form10k => &["10-K", "10-K/A"],
        }
    }

    /// Stable kebab-case identifier — used as the bucket name in
    /// `_sec_internal.fetch_*_batch` call dispatch (`"form4"`,
    /// `"form13f"`, etc.).
    pub const fn as_str(self) -> &'static str {
        match self {
            SecFormBucket::Form3 => "form3",
            SecFormBucket::Form4 => "form4",
            SecFormBucket::Form5 => "form5",
            SecFormBucket::Form144 => "form144",
            SecFormBucket::Form13f => "form13f",
            SecFormBucket::Form8k => "form8k",
            SecFormBucket::Sc13d => "sc13d",
            SecFormBucket::Sc13g => "sc13g",
            SecFormBucket::Def14a => "def14a",
            SecFormBucket::Form10k => "form10k",
        }
    }

    /// Map an SEC form string to its bucket. Returns `None` for forms
    /// that have no per-filing fetcher (e.g. minor filing types we
    /// don't model). The mapping is case-sensitive — SEC uses
    /// uppercase form codes consistently in the index files.
    pub fn from_form_string(form: &str) -> Option<Self> {
        for bucket in ALL_BUCKETS {
            if bucket.matching_forms().contains(&form) {
                return Some(*bucket);
            }
        }
        None
    }
}

/// Every defined bucket. Iteration order is stable (matches
/// declaration order in the enum).
pub const ALL_BUCKETS: &[SecFormBucket] = &[
    SecFormBucket::Form3,
    SecFormBucket::Form4,
    SecFormBucket::Form5,
    SecFormBucket::Form144,
    SecFormBucket::Form13f,
    SecFormBucket::Form8k,
    SecFormBucket::Sc13d,
    SecFormBucket::Sc13g,
    SecFormBucket::Def14a,
    SecFormBucket::Form10k,
];

/// The lean default scope when no explicit `form_types` is supplied:
/// insider ownership (Forms 3/4/5) + 8-K cover pages. Heavy payloads
/// (13F info tables, SC 13D/G, DEF 14A, Form 144, Exhibit 21, XBRL
/// company-facts) are opt-in via explicit `form_types` or wrapper-
/// level `include_*` flags.
pub const LEAN_FETCH_BUCKETS: &[SecFormBucket] = &[
    SecFormBucket::Form3,
    SecFormBucket::Form4,
    SecFormBucket::Form5,
    SecFormBucket::Form8k,
];

/// Iterate every bucket as `(bucket_name, [matching_forms])`. Lets
/// bindings materialise the full table once (at module-load time)
/// rather than calling `resolve_fetch_buckets` per filing. Iteration
/// order matches [`ALL_BUCKETS`].
pub fn all_buckets() -> Vec<(&'static str, Vec<&'static str>)> {
    ALL_BUCKETS
        .iter()
        .map(|b| (b.as_str(), b.matching_forms().to_vec()))
        .collect()
}

/// Resolve a list of SEC form strings to per-filing fetch buckets.
///
/// - `form_types = None` → returns the lean default scope
///   ([`LEAN_FETCH_BUCKETS`]) plus an empty unmatched list.
/// - `form_types = Some(list)` → maps each string to its bucket;
///   strings with no per-filing fetcher land in the `unmatched`
///   list (the binding then decides whether to warn, error, or
///   silently skip).
///
/// The returned bucket list is deduplicated and ordered by
/// [`ALL_BUCKETS`] declaration order, not by input order — that
/// makes the dispatch loop downstream deterministic.
pub fn resolve_fetch_buckets(form_types: Option<&[&str]>) -> (Vec<SecFormBucket>, Vec<String>) {
    let Some(form_types) = form_types else {
        return (LEAN_FETCH_BUCKETS.to_vec(), Vec::new());
    };
    let mut active = Vec::new();
    let mut unmatched = Vec::new();
    for form in form_types {
        match SecFormBucket::from_form_string(form) {
            Some(bucket) if !active.contains(&bucket) => active.push(bucket),
            Some(_) => {} // already in active
            None => unmatched.push((*form).to_string()),
        }
    }
    // Re-order active by ALL_BUCKETS declaration order for determinism.
    let mut ordered = Vec::with_capacity(active.len());
    for bucket in ALL_BUCKETS {
        if active.contains(bucket) {
            ordered.push(*bucket);
        }
    }
    (ordered, unmatched)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lean_default_when_form_types_none() {
        let (buckets, unmatched) = resolve_fetch_buckets(None);
        assert_eq!(buckets, LEAN_FETCH_BUCKETS);
        assert!(unmatched.is_empty());
    }

    #[test]
    fn explicit_form_types_map_to_buckets() {
        let (buckets, unmatched) = resolve_fetch_buckets(Some(&["4", "13F-HR"]));
        assert_eq!(buckets, vec![SecFormBucket::Form4, SecFormBucket::Form13f]);
        assert!(unmatched.is_empty());
    }

    #[test]
    fn variant_spellings_accepted() {
        let (buckets, _) = resolve_fetch_buckets(Some(&["SC 13D", "SCHEDULE 13D"]));
        // Same bucket; deduped.
        assert_eq!(buckets, vec![SecFormBucket::Sc13d]);
    }

    #[test]
    fn unknown_forms_land_in_unmatched() {
        let (buckets, unmatched) = resolve_fetch_buckets(Some(&["4", "XYZ-99"]));
        assert_eq!(buckets, vec![SecFormBucket::Form4]);
        assert_eq!(unmatched, vec!["XYZ-99".to_string()]);
    }

    #[test]
    fn bucket_order_is_declaration_order_not_input_order() {
        let (buckets, _) = resolve_fetch_buckets(Some(&["10-K", "4", "8-K"]));
        // ALL_BUCKETS order: Form4 (1), Form8k (5), Form10k (9).
        assert_eq!(
            buckets,
            vec![
                SecFormBucket::Form4,
                SecFormBucket::Form8k,
                SecFormBucket::Form10k,
            ]
        );
    }
}
