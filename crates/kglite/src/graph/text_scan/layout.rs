//! The record layout the scan reads, and the checked constructor that makes
//! its arrays safe to index.
//!
//! # Conventions (verified against `storage/column_store/typed_column.rs`)
//!
//! * `offsets` has **`N + 1` entries**; `offsets[0] == 0`; record `i` is
//!   `data[offsets[i]..offsets[i + 1]]` (`TypedColumn::Str`, field doc ~line 60;
//!   push at ~592-612 appends `data.len()` after each value).
//! * `nulls` has `N` entries, **`0` = present, `1` = null** (`push_pair(.., 0)`
//!   for a value, `nulls.try_push(1)` for `Value::Null` at ~619-630;
//!   `is_present` tests `== Some(0)` at ~767).
//! * A null row repeats the previous offset, so its range is empty.
//! * Updates to a row land in the column's `relocated` overlay and never touch
//!   `offsets`/`data`: a layout built from the arrays alone does not see them,
//!   so the layer that owns the column must treat relocated rows as changed.
//!
//! The column loader already enforces what the scan relies on
//! (`column_store/mod.rs` ~2269-2335: offsets[0] == 0, monotone, at most the
//! last offset, on UTF-8 boundaries; `data` length equals the last offset;
//! `nulls` length equals the row count). A column that came through it can use
//! [`BorrowedLayout::from_validated`]; anything else goes through the checked
//! [`BorrowedLayout::new`].

use std::ops::Range;

/// What the scan needs from a layout. Implementors guarantee the invariants
/// [`BorrowedLayout::new`] checks, so the defaults may index freely.
pub(crate) trait RecordLayout {
    fn offsets(&self) -> &[u64];
    fn bytes(&self) -> &[u8];
    fn nulls(&self) -> &[u8];

    /// Number of records.
    fn len(&self) -> usize {
        self.offsets().len().saturating_sub(1)
    }

    /// Byte range of record `i`; empty for an out-of-range `i`.
    fn range(&self, i: usize) -> Range<usize> {
        match (self.offsets().get(i), self.offsets().get(i + 1)) {
            (Some(&s), Some(&e)) => s as usize..e as usize,
            _ => 0..0,
        }
    }

    fn record(&self, i: usize) -> &[u8] {
        self.bytes().get(self.range(i)).unwrap_or(&[])
    }

    fn is_present(&self, i: usize) -> bool {
        self.nulls().get(i) == Some(&0)
    }

    /// The record whose range contains byte `offset`, searching `lo..hi`.
    /// Empty records contain nothing, so the answer is always a non-empty one.
    fn record_of(&self, offset: usize, lo: usize, hi: usize) -> Option<usize> {
        let hi = hi.min(self.len());
        let slice = self.offsets().get(lo..=hi)?;
        let after = slice.partition_point(|&o| o as usize <= offset);
        let record = lo + after.checked_sub(1)?;
        (record < hi && offset < self.range(record).end).then_some(record)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum LayoutError {
    OffsetsLength,
    FirstOffset,
    NotMonotone,
    PastData,
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::OffsetsLength => "offsets must have one more entry than nulls",
            Self::FirstOffset => "the first offset must be 0",
            Self::NotMonotone => "offsets must not decrease",
            Self::PastData => "the last offset is past the end of the data",
        })
    }
}

/// A layout over borrowed arrays (a mapped or heap `Str` column's).
#[derive(Debug, Clone, Copy)]
pub(crate) struct BorrowedLayout<'a> {
    offsets: &'a [u64],
    bytes: &'a [u8],
    nulls: &'a [u8],
}

impl<'a> BorrowedLayout<'a> {
    /// Check everything the scan indexes on: `offsets.len() == nulls.len() + 1`,
    /// `offsets[0] == 0`, non-decreasing, last offset at most `bytes.len()`.
    /// O(records), no access to `bytes`.
    pub(crate) fn new(
        offsets: &'a [u64],
        bytes: &'a [u8],
        nulls: &'a [u8],
    ) -> Result<Self, LayoutError> {
        if offsets.len() != nulls.len() + 1 {
            return Err(LayoutError::OffsetsLength);
        }
        if offsets[0] != 0 {
            return Err(LayoutError::FirstOffset);
        }
        if offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(LayoutError::NotMonotone);
        }
        if offsets[offsets.len() - 1] > bytes.len() as u64 {
            return Err(LayoutError::PastData);
        }
        Ok(Self {
            offsets,
            bytes,
            nulls,
        })
    }

    /// For arrays a column already validated (see the module docs): skips the
    /// O(records) pass.
    ///
    /// Precondition: exactly what [`Self::new`] checks. Violating it cannot
    /// cause undefined behaviour (every access goes through `get`), but scans
    /// would silently miss or truncate records.
    pub(crate) fn from_validated(offsets: &'a [u64], bytes: &'a [u8], nulls: &'a [u8]) -> Self {
        debug_assert!(Self::new(offsets, bytes, nulls).is_ok());
        Self {
            offsets,
            bytes,
            nulls,
        }
    }
}

impl RecordLayout for BorrowedLayout<'_> {
    fn offsets(&self) -> &[u64] {
        self.offsets
    }
    fn bytes(&self) -> &[u8] {
        self.bytes
    }
    fn nulls(&self) -> &[u8] {
        self.nulls
    }
}

/// An owned layout, built from values (tests, overlays of changed rows).
#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub(crate) struct OwnedLayout {
    offsets: Vec<u64>,
    bytes: Vec<u8>,
    nulls: Vec<u8>,
}

#[cfg(test)]
impl OwnedLayout {
    /// `None` is a null row; `Some("")` is a present, empty value.
    pub(crate) fn from_records<S: AsRef<[u8]>>(records: &[Option<S>]) -> Self {
        let mut layout = Self {
            offsets: vec![0],
            ..Self::default()
        };
        for record in records {
            if let Some(text) = record {
                layout.bytes.extend_from_slice(text.as_ref());
            }
            layout.offsets.push(layout.bytes.len() as u64);
            layout.nulls.push(u8::from(record.is_none()));
        }
        layout
    }
}

#[cfg(test)]
impl RecordLayout for OwnedLayout {
    fn offsets(&self) -> &[u64] {
        &self.offsets
    }
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    fn nulls(&self) -> &[u8] {
        &self.nulls
    }
}
