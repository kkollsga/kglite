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
use super::super::typing::{inferred_type_keyword, ColumnInference, IdInference};
use crate::datatypes::values::ColumnType;
use indexmap::IndexMap;
use std::collections::HashMap;

/// What the loader needs to start: the types resolved for its undeclared
/// columns, and the chunk stream to load from.
pub(super) struct Prepared<'a> {
    /// Column name → blueprint type keyword, to merge into `declared_types`.
    pub resolved: IndexMap<String, String>,
    /// Id columns → the type an endpoint frame must give them. Separate from
    /// `resolved` because an id is typed by a different rule (`IdInference`).
    pub resolved_ids: IndexMap<String, ColumnType>,
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
    id_columns: &[String],
    row_preserving: bool,
    mut prepare: F,
) -> Result<Prepared<'a>, String>
where
    F: FnMut(&mut RawCsv) -> Vec<String>,
{
    let mut stream = source.chunks(chunk_size)?;
    let Some(first) = stream.next() else {
        return Ok(Prepared {
            resolved: IndexMap::new(),
            resolved_ids: IndexMap::new(),
            chunks: Box::new(std::iter::empty()),
            extra_pass: false,
        });
    };
    let mut first = first?;
    let keep = prepare(&mut first);

    let mut inferences: IndexMap<String, ColumnInference> = IndexMap::new();
    observe(&mut inferences, &first, &keep, declared);
    let mut ids: IndexMap<String, IdInference> = IndexMap::new();
    observe_ids(&mut ids, &first, id_columns);
    if inferences.is_empty() && ids.is_empty() {
        return Ok(Prepared {
            resolved: IndexMap::new(),
            resolved_ids: IndexMap::new(),
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
            resolved_ids: id_types(ids),
            chunks: Box::new(std::iter::once(Ok(first))),
            extra_pass: false,
        });
    };

    // Genuinely multi-chunk with something to infer: read the columns once
    // more, then load from a fresh pass.
    drop(first);
    if row_preserving {
        scan_remaining(source, &mut inferences, &mut ids)?;
    } else {
        // `prepare` drops or rewrites rows, so the cells that count are only
        // visible through it — a raw column scan would infer from rows the
        // load never sees, and could widen a type the buffered path narrows.
        let mut pending = Some(second);
        while let Some(chunk) = pending.take().or_else(|| stream.next()) {
            if settled(&inferences, &ids) {
                break;
            }
            let mut raw = chunk?;
            let keep = prepare(&mut raw);
            observe(&mut inferences, &raw, &keep, declared);
            observe_ids(&mut ids, &raw, id_columns);
        }
    }

    Ok(Prepared {
        resolved: keywords(inferences),
        resolved_ids: id_types(ids),
        chunks: source.chunks(chunk_size)?,
        extra_pass: true,
    })
}

/// Fold the whole input into `inferences` through `Source::scan_columns`,
/// which visits only these columns' cells and allocates nothing per cell —
/// the pre-pass asks a question about values, so it should not pay for a
/// table. The first chunk's evidence is already in `inferences`; re-reading
/// those rows only re-confirms it.
fn scan_remaining(
    source: &dyn Source,
    inferences: &mut IndexMap<String, ColumnInference>,
    ids: &mut IndexMap<String, IdInference>,
) -> Result<(), String> {
    // One scan feeds both rules: the value columns occupy the first slots and
    // the id columns the rest, so a column that is both is folded twice, under
    // each rule, from the same read.
    let names: Vec<String> = inferences.keys().chain(ids.keys()).cloned().collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let split = inferences.len();
    let mut values: Vec<ColumnInference> = inferences.values().copied().collect();
    let mut id_states: Vec<IdInference> = ids.values().copied().collect();
    let mut unsettled = values.iter().filter(|s| !s.is_settled()).count()
        + id_states.iter().filter(|s| !s.is_settled()).count();

    source.scan_columns(&refs, &mut |slot, cell| {
        if slot < split {
            let state = &mut values[slot];
            if !state.is_settled() {
                state.observe(cell);
                if state.is_settled() {
                    unsettled -= 1;
                }
            }
        } else {
            let state = &mut id_states[slot - split];
            if !state.is_settled() {
                state.observe(cell);
                if state.is_settled() {
                    unsettled -= 1;
                }
            }
        }
        // Nothing a later cell can say changes an all-settled answer.
        unsettled > 0
    })?;

    for (state, slot) in values.into_iter().zip(inferences.values_mut()) {
        *slot = state;
    }
    for (state, slot) in id_states.into_iter().zip(ids.values_mut()) {
        *slot = state;
    }
    Ok(())
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

fn settled(
    inferences: &IndexMap<String, ColumnInference>,
    ids: &IndexMap<String, IdInference>,
) -> bool {
    inferences.values().all(|i| i.is_settled()) && ids.values().all(|i| i.is_settled())
}

fn observe_ids(ids: &mut IndexMap<String, IdInference>, raw: &RawCsv, columns: &[String]) {
    for name in columns {
        let Some(idx) = raw.col_index(name) else {
            continue;
        };
        let state = ids.entry(name.clone()).or_default();
        for (r, row) in raw.rows.iter().enumerate() {
            if state.is_settled() {
                break;
            }
            if raw.nulls[r][idx] {
                continue;
            }
            state.observe(&row[idx]);
        }
    }
}

fn id_types(ids: IndexMap<String, IdInference>) -> IndexMap<String, ColumnType> {
    ids.into_iter().map(|(k, v)| (k, v.resolve())).collect()
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
