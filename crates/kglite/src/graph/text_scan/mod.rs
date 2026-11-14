//! Substring, phrase and regex scanning over *record layouts*: one contiguous
//! byte buffer holding many values back to back, delimited by a cumulative
//! offsets array (the shape of a `Str` column's `offsets`/`data`/`nulls`).
//!
//! The module owns nothing and stores nothing. It is a pure function of
//! (layout, compiled query, options) that returns hits as
//! `(record index, offset in record, length)`.
//!
//! # The one rule
//!
//! There is no separator between records, and values may contain any byte
//! including `\n`. For every present record, the hits reported inside it are
//! exactly [`Matcher::find_in_record`] run on that record alone: leftmost,
//! non-overlapping, offsets relative to the record start. Every faster path
//! (whole-slice `memmem`, a literal prefilter in front of a regex, parallel
//! chunks, scoped range scans) is an optimisation that must reproduce that
//! answer; the tests compare each against it.

// No caller exists until the layer that supplies a column's arrays is built.
#![allow(dead_code)]

pub(crate) mod layout;
pub(crate) mod matcher;
pub(crate) mod scan;
pub(crate) mod scope;

#[cfg(test)]
mod tests;
