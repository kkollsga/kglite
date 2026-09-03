//! Whole-input type inference for the chunked loaders.
//!
//! A chunked loader cannot infer a column's type from the chunk in front of
//! it: the answer would depend on where the chunk boundaries fall, and the
//! chunk size is documented as bounding peak memory, nothing else. So any kept
//! column the blueprint did not type gets its type from a pass over every
//! chunk, and the load pass then types that column identically in every chunk.
//!
//! The pass is skipped wherever it would buy nothing, because it costs a
//! second read of the input:
//!
//! - every kept column is declared → nothing to infer;
//! - the input is a single chunk → that chunk *is* the whole input, so it is
//!   typed from itself and handed straight to the loader.
//!
//! Both cases hand back the chunks already read, so only a genuinely
//! multi-chunk input with an undeclared column pays for a second pass.

use super::super::input::Source;
use super::super::table::RawCsv;
use super::super::typing::{inferred_type_keyword, ColumnInference};
use indexmap::IndexMap;
use std::collections::HashMap;

/// What the loader needs to start: the types resolved for its undeclared
/// columns, and the chunk stream to load from.
pub(super) struct Prepared<'a> {
    /// Column name → blueprint type keyword, to merge into `declared_types`.
    pub resolved: IndexMap<String, String>,
    /// The chunks to load. Either the ones already in hand (chained back
    /// together) or a fresh pass, whichever the pre-pass left valid.
    pub chunks: Box<dyn Iterator<Item = Result<RawCsv, String>> + 'a>,
    /// True when resolving the types cost a second read of the input.
    pub extra_pass: bool,
}

/// Resolve the types of every kept column `declared` does not name.
///
/// `prepare` receives each chunk exactly as the load pass will see it — it
/// must apply the same filter and return the same keep list, or the pre-pass
/// types a different table from the one that is loaded. A chunk handed back
/// in `Prepared::chunks` has already been through it, so `prepare` must be
/// idempotent (applying a row filter twice keeps the same rows).
pub(super) fn prepare_chunks<'a, F>(
    source: &'a dyn Source,
    chunk_size: usize,
    declared: &HashMap<String, String>,
    mut prepare: F,
) -> Result<Prepared<'a>, String>
where
    F: FnMut(&mut RawCsv) -> Vec<String>,
{
    let mut stream = source.chunks(chunk_size)?;
    let Some(first) = stream.next() else {
        return Ok(Prepared {
            resolved: IndexMap::new(),
            chunks: Box::new(std::iter::empty()),
            extra_pass: false,
        });
    };
    let mut first = first?;
    let keep = prepare(&mut first);

    let mut inferences: IndexMap<String, ColumnInference> = IndexMap::new();
    observe(&mut inferences, &first, &keep, declared);
    if inferences.is_empty() {
        return Ok(Prepared {
            resolved: IndexMap::new(),
            chunks: Box::new(std::iter::once(Ok(first)).chain(stream)),
            extra_pass: false,
        });
    }

    let Some(second) = stream.next() else {
        // One chunk: it is the whole input, so it types itself. Resolving here
        // rather than leaving it to the load pass still matters — the junction
        // loader types per *target group*, and a group is a subset of the
        // chunk that can infer differently from the chunk it came from.
        return Ok(Prepared {
            resolved: keywords(inferences),
            chunks: Box::new(std::iter::once(Ok(first))),
            extra_pass: false,
        });
    };

    // Genuinely multi-chunk with something to infer: fold the remaining chunks
    // and load from a fresh pass.
    drop(first);
    let mut pending = Some(second);
    while let Some(chunk) = pending.take().or_else(|| stream.next()) {
        if saturated(&inferences) {
            // Every undeclared column has seen a cell that is not one of the
            // parseable shapes; no later row can change the answer.
            break;
        }
        let mut raw = chunk?;
        let keep = prepare(&mut raw);
        observe(&mut inferences, &raw, &keep, declared);
    }

    Ok(Prepared {
        resolved: keywords(inferences),
        chunks: source.chunks(chunk_size)?,
        extra_pass: true,
    })
}

fn observe(
    inferences: &mut IndexMap<String, ColumnInference>,
    raw: &RawCsv,
    keep: &[String],
    declared: &HashMap<String, String>,
) {
    for name in keep {
        if declared.contains_key(name) {
            continue;
        }
        let Some(idx) = raw.col_index(name) else {
            continue;
        };
        inferences
            .entry(name.clone())
            .or_default()
            .observe_column(raw, idx);
    }
}

fn saturated(inferences: &IndexMap<String, ColumnInference>) -> bool {
    !inferences.is_empty() && inferences.values().all(|i| i.is_settled())
}

fn keywords(inferences: IndexMap<String, ColumnInference>) -> IndexMap<String, String> {
    inferences
        .into_iter()
        .filter_map(|(name, inference)| {
            inferred_type_keyword(&inference.resolve()).map(|kw| (name, kw.to_string()))
        })
        .collect()
}

/// One line naming the columns whose types cost an extra read, so an author
/// who cares about it knows exactly which declarations remove it.
pub(super) fn prepass_warning(where_: &str, prepared: &Prepared<'_>) -> Option<String> {
    if !prepared.extra_pass || prepared.resolved.is_empty() {
        return None;
    }
    let cols: Vec<&str> = prepared.resolved.keys().map(String::as_str).collect();
    Some(format!(
        "{where_}: {} column(s) have no declared type ({}), so the loader read the input twice \
         — once to infer them over every row, once to load. Declaring them keeps the type \
         stable and skips the extra pass.",
        cols.len(),
        cols.join(", ")
    ))
}
